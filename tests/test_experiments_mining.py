"""Capture uses the same evaluation paths as mining, including window resets."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import betalens.factor.mining as mining


def test_effective_spec_context_accepts_standard_callable_wrappers():
    from functools import partial
    from betalens.experiments.snapshot import describe_spec
    context = describe_spec(SimpleNamespace(compute=partial(pow, exp=2), transform=np.negative))
    assert context["compute"]["kwargs"] == {"exp": 2}
    assert "pow" in context["compute"]["callable"]
    assert context["transform"].endswith(":negative")


@pytest.mark.parametrize("mode", ["shared", "transform", "timing", "rolling"])
def test_selected_capture_preserves_actual_window_paths(monkeypatch, mode):
    days = pd.date_range("2025-01-02", periods=8, name="datetime")
    prices = pd.DataFrame({"A": np.linspace(10., 11., len(days))}, index=days)
    data = mining.MiningData(inputs={"x": prices}, price=prices, execution_price=prices, trade_status=None, pit=None, universe=["A"])
    windows = [mining.MiningWindow("4/4/0", "2025-01-02", "2025-01-05", 4, 4),
               mining.MiningWindow("4/4/1", "2025-01-06", "2025-01-09", 4, 4)]
    factor = SimpleNamespace(name="测试", compute_kwargs={}, compute=lambda **kwargs: kwargs["x"],
                             strategy_type="timing" if mode == "timing" else "cross_sectional")
    weights = pd.DataFrame({"A": .8, "cash": .2}, index=days)
    spec = mining.MiningSpec(factor, execution_mode="rolling_fit" if mode == "rolling" else "precomputed",
        window_transform=(lambda weight, window, context: weight) if mode == "transform" else None,
        fit_window=(lambda *args: weights) if mode == "rolling" else None)
    monkeypatch.setattr(mining, "_import_spec", lambda *a: spec)
    monkeypatch.setattr(mining, "_build_weights", lambda *a, **k: weights)
    monkeypatch.setattr(mining, "_slice_data", lambda *a: data)
    monkeypatch.setattr(mining, "_signal_pairs", lambda *a: [])
    monkeypatch.setattr(mining, "_rank_ic_observations", lambda *a, **k: {})
    monkeypatch.setattr(mining, "_rank_ic_metrics", lambda *a: {})
    calls = []
    def engine(weight, **kwargs):
        calls.append(kwargs["end_date"])
        returns = pd.Series(.01, index=days)
        amount = (1 + returns).cumprod() * 100_000
        values = weights.mul(amount, axis=0)
        logs = pd.DataFrame(columns=["datetime", "code", "direction", "value", "equity_before"])
        return SimpleNamespace(nav=(amount / 100_000).rename("nav"), daily_returns=returns,
            daily_weight=weights, actual_weight=weights, weight=weights, daily_amount=amount,
            daily_position_value=values, rebalance_log=logs, synthetic_settlements=pd.DataFrame(),
            risk_events=pd.DataFrame(), initial_amount=100_000)
    import betalens.backtest
    monkeypatch.setattr(betalens.backtest, "BacktestBase", engine)
    captured = mining._capture_selected_candidate("test", "factor", {"window": 20}, windows, data,
                                                  {"span": ["2025-01-02", "2025-01-09"]})
    assert "error" not in captured
    assert mining._ARCHIVE_CAPTURE.get() is None
    if mode == "shared":
        assert len(calls) == 1 and list(captured["scopes"]) == ["total"]
        assert all(ref["semantics"] == "shared_path_slice" for ref in captured["window_refs"].values())
    else:
        assert all(captured["window_refs"][w.window_id]["scope"] == w.window_id for w in windows)
        assert all(captured["window_refs"][w.window_id]["semantics"] == "independent" for w in windows)
        assert len(calls) == (3 if mode == "timing" else 2)
    for snapshot in captured["scopes"].values():
        assert "daily_weight" in snapshot.tables
        assert not hasattr(snapshot, "_reader")


def test_capture_context_is_reset_after_an_error(monkeypatch):
    monkeypatch.setattr(mining, "_evaluate_candidate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("failed")))
    with pytest.raises(RuntimeError, match="failed"):
        mining._capture_selected_candidate("module", "factor", {}, [], None, {})
    assert mining._ARCHIVE_CAPTURE.get() is None


@pytest.mark.parametrize("fail_capture", [False, True])
def test_run_mining_opt_in_captures_only_selected_and_archives_in_memory(tmp_path, monkeypatch, fail_capture):
    import sys
    import types
    import yaml
    import betalens.backtest
    import betalens.factor.mining_audit as audit
    from betalens.experiments import ExperimentStore
    from test_experiments import bt as fixture
    backtest = fixture.__wrapped__()
    prices = backtest.cost_price[["000001.SZ"]]
    data = mining.MiningData({"x": prices}, prices, prices, None, None, ["000001.SZ"])
    windows = [mining.MiningWindow("4/4/0", "2025-01-02", "2025-01-05", 4, 4)]
    module = types.ModuleType("_archive_capture_integration")
    def make_spec(params):
        assert params["factor_id"] == "MOMENTUM"
        return mining.MiningSpec(SimpleNamespace(name="动量", compute_kwargs={"window": params["window"]},
            compute=lambda **kwargs: kwargs["x"], weight_mode="classic-long-short"), warmup_days=0)
    module.make_mining_spec = make_spec
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(mining, "_fetch_data", lambda *a, **k: data)
    monkeypatch.setattr(mining, "_windows", lambda *a: windows)
    monkeypatch.setattr(mining, "_build_weights", lambda *a, **k: backtest.weight)
    monkeypatch.setattr(mining, "_rank_ic_observations", lambda *a, **k: {})
    monkeypatch.setattr(mining, "_rank_ic_metrics", lambda *a: {})
    monkeypatch.setattr(audit, "_write_heatmap_report", lambda *a: [])
    captures = []
    def engine(*a, **k):
        if mining._ARCHIVE_CAPTURE.get() is not None:
            captures.append(1)
            if fail_capture:
                raise RuntimeError("Selected raw export failed")
        return backtest
    monkeypatch.setattr(betalens.backtest, "BacktestBase", engine)
    parameters = {"version": 1, "factor_class": "test", "factors": {"MOMENTUM": {
        "module": module.__name__, "execution_mode": "precomputed",
        "parameters": {"window": {"type": "int", "low": 20, "high": 60, "step": 20}}}},
        "evaluation": {"span": ["2025-01-01", "2025-01-09"]}, "windows": {"lengths": [4], "steps": [4]},
        "search": {"coarse": {"sampler": "random", "n_trials": 3, "seed": 1},
                   "fine": {"top_k": 1, "points_per_dimension": 1, "max_candidates": 1}},
        "selection": {"objective": {"metric": "sharpe", "aggregate": "median", "direction": "maximize"}, "top_k": 1}}
    performance = {"runtime": {"backend": "serial", "workers": 1}, "cache": {"data_enabled": False},
                   "output": {"directory": str(tmp_path / "outputs")}, "logging": {"heartbeat_seconds": 0, "task_logs": False}}
    params_path, perf_path = tmp_path / "parameters.yaml", tmp_path / "performance.yaml"
    params_path.write_text(yaml.safe_dump(parameters))
    perf_path.write_text(yaml.safe_dump(performance))
    ordinary = mining.run_mining(params_path, perf_path)
    assert not captures and not ordinary.factor_runs[0].selected_results
    result = mining.run_mining(params_path, perf_path, capture_selected=True)
    assert len(captures) == len(result.factor_runs[0].selected_candidates) == 1
    with ExperimentStore(tmp_path / "archive") as store:
        batch = store.save_mining(result)
        assert batch["status"] == ("partial" if fail_capture else "complete")
        assert store.query()["total"] >= 2
        assert store.query(has_raw=True)["total"] == (0 if fail_capture else 1)
