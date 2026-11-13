"""Library migration contracts, using isolated outputs and synthetic markets."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from .factors import FACTOR_ROOT, discover_factors, get_factor_config, load_factor_module
from .runs import DashboardRun, RunManager
from .schemas import RunRequest

sys.path.insert(0, str(FACTOR_ROOT))
from factor_runtime import execution
from factor_runtime.frames import align_daily_wides, validate_factor_output, wide_to_prequery
from betalens.factor.config import ConfigError, load_yaml_config, write_yaml_config


@pytest.fixture
def config(tmp_path):
    _, _, source = get_factor_config("alpha101", "ALPHA12")
    cfg = copy.deepcopy(source)
    cfg["run"]["output_dir"] = str(tmp_path / "run")
    cfg["run"]["include_profiling"] = False
    cfg["run"]["dump_excel"] = False
    cfg["experiment_export"] = {"enabled": False, "store_path": str(tmp_path / "experiments")}
    return cfg


def completed_result():
    from betalens.analyst import Analyst
    from factor_template import RunResult
    days = pd.bdate_range("2025-01-02", periods=8, name="datetime")
    returns = pd.Series([.01, -.02, .03, .005, -.008, .02, .01, -.003], index=days)
    amount = (1 + returns).cumprod() * 100_000
    weights = pd.DataFrame({"000001.SZ": .8, "cash": .2}, index=days)
    prices = pd.DataFrame({"000001.SZ": np.linspace(10., 11., len(days)), "cash": 1.}, index=days)
    nav = pd.concat([pd.Series([1.], index=[days[0] - pd.Timedelta(days=1)]), amount / 100_000])
    logs = pd.DataFrame(columns=["datetime", "code", "direction", "shares", "signed_shares", "value", "price", "equity_before", "fee"])
    bt = SimpleNamespace(nav=nav, daily_returns=returns, daily_amount=amount,
        weight=weights, actual_weight=weights, daily_weight=weights,
        daily_position_value=weights.mul(amount, axis=0), daily_pnl=weights.mul(amount.diff().fillna(1000), axis=0),
        position=weights.mul(amount, axis=0) / prices, cost_price=prices,
        rebalance_log=logs, synthetic_settlements=pd.DataFrame(), risk_events=pd.DataFrame(),
        symbol="测试", metric="close", initial_amount=100_000, start=days[0], end=days[-1],
        calculation_version="test")
    analyst = Analyst.from_backtest(bt)
    analyst._name_map = {"000001.SZ": "测试证券"}
    bt.trade_cycles = analyst.an.trade_cycles
    return RunResult(backtest=bt, analyst=analyst,
        factor_values=pd.DataFrame({"信号日": days, "股票代码": "000001.SZ", "因子值": 1., "分组": 0}))


class FakePipeline:
    def __init__(self, spec):
        self.spec = spec

    def run(self, start, end, **kwargs):
        return completed_result()


def test_daily_axes_missing_values_and_availability():
    first = pd.DataFrame({"B": [2., 1.]}, index=pd.to_datetime(["2025-01-03", "2025-01-02"], utc=True))
    first.attrs["availability_upper"] = {"2025-01-02": "2025-01-03T00:00:00"}
    second = pd.DataFrame({"A": [3.]}, index=pd.to_datetime(["2025-01-02"]))
    aligned = align_daily_wides({"x": first, "y": second}, universe=["A", "B", "C"])
    assert aligned["x"].shape == (2, 3)
    pd.testing.assert_index_equal(aligned["x"].index, aligned["y"].index)
    assert aligned["y"].loc["2025-01-03"].isna().all()
    assert aligned["x"].attrs == first.attrs
    long = wide_to_prequery(aligned["x"], "F", [pd.Timestamp("2025-01-02").date()])
    assert len(long) == 1 and long.iloc[0]["F"] == 1.
    assert not long.duplicated(["input_ts", "code"]).any()


def test_duplicate_days_codes_and_wrong_compute_shape_are_rejected():
    days = pd.to_datetime(["2025-01-02 09:00", "2025-01-02 15:00"])
    with pytest.raises(ValueError, match="重复"):
        align_daily_wides({"x": pd.DataFrame({"A": [1., 2.]}, index=days)})
    frame = pd.DataFrame([[1., 2.]], index=days[:1], columns=["A", "A"])
    with pytest.raises(ValueError, match="证券代码"):
        align_daily_wides({"x": frame})
    frame = pd.DataFrame({"A": [1.]}, index=days[:1])
    with pytest.raises(ValueError, match="轴"):
        validate_factor_output(frame.rename(columns={"A": "B"}), {"x": frame}, "F")


def test_fetch_keeps_business_date_and_physical_filters(monkeypatch):
    import factor_template as core
    core._ensure_runtime()
    calls = []
    class Feed:
        def __init__(self, table):
            calls.append(table)
        def query_daily_market(self, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame({"trade_date": pd.to_datetime(["2025-01-02"]), "code": ["A"],
                "metric": ["close"], "value": [10.], "available_at": pd.to_datetime(["2025-01-03"])})
        def close(self):
            calls.append("closed")
    monkeypatch.setattr(core, "Datafeed", Feed)
    panel = core.fetch_daily_wide("close", ["A"], "2025-01-02", "2025-01-02", filters={"asset_type": "stock"})
    assert panel.index[0] == pd.Timestamp("2025-01-02")
    assert panel.attrs["availability_upper"]["2025-01-02"] == "2025-01-03T00:00:00"
    assert calls[1]["filters"] == {"asset_type": "stock"} and calls[-1] == "closed"


def test_invalid_config_fails_before_pipeline_or_export(config, tmp_path):
    config["weight"].update(mode="freeplay", long_groups=None, short_groups=None)
    with pytest.raises(ConfigError, match="取数前"):
        execution.execute_factor(config, tmp_path / "factor.yaml", lambda *_: None, FakePipeline)
    assert not (tmp_path / "run").exists()


def test_scalar_zero_is_a_valid_freeplay_group(config):
    config["weight"].update(mode="freeplay", long_groups=0, short_groups=None)
    execution.validate_factor_config(config)


def test_disabled_export_is_optional_and_custom_paths_survive(config, tmp_path, monkeypatch):
    monkeypatch.setattr(execution.importlib.util, "find_spec", lambda *_: pytest.fail("optional dependency checked"))
    original = copy.deepcopy(config)
    result = execution.execute_factor(config, tmp_path / "source.yaml", lambda *_: None, FakePipeline)
    assert result.experiment_export["status"] == "disabled"
    assert config == original
    assert not (tmp_path / "experiments").exists()
    effective = load_yaml_config(tmp_path / "run/run_config.yaml")
    assert effective["run"]["output_dir"] == str(tmp_path / "run")
    assert tuple(result) == (result.backtest, result.analyst)


def test_default_outputs_are_unique_and_repo_relative(config, tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "REPO_ROOT", tmp_path)
    config["run"]["output_dir"] = "outputs/runs/manual"
    execution.execute_factor(config, tmp_path / "factor.yaml", lambda *_: None, FakePipeline, run_id="one")
    execution.execute_factor(config, tmp_path / "factor.yaml", lambda *_: None, FakePipeline, run_id="two")
    assert (tmp_path / "outputs/runs/alpha101/ALPHA12/one/run_config.yaml").exists()
    assert (tmp_path / "outputs/runs/alpha101/ALPHA12/two/run_config.yaml").exists()
    settings = execution.export_settings({})
    assert settings["store_path"] == str(tmp_path / "outputs/experiments.lmdb")


def test_export_roundtrip_idempotency_and_failure_preserve_result(config, tmp_path, monkeypatch):
    from betalens.experiments import ExperimentStore
    config["experiment_export"]["enabled"] = True
    result = execution.execute_factor(config, tmp_path / "source.yaml", lambda *_: None, FakePipeline, run_id="same")
    assert result.experiment_export["status"] == "saved"
    identity = result.experiment_export["experiment_id"]
    execution.export_run(result, config, tmp_path / "source.yaml", "same", tmp_path / "run")
    assert result.experiment_export["experiment_id"] == identity
    with ExperimentStore(tmp_path / "experiments", readonly=True) as store:
        assert store.query()["total"] == 1
        assert store.factors()[0]["key"] == "alpha101:ALPHA12:cross_sectional"
        pd.testing.assert_series_equal(store.read_table(identity, "nav"), result.backtest.nav)
        assert store.get(identity)["metadata"]["source_hash"]
    monkeypatch.setattr(ExperimentStore, "save_run", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    execution.export_run(result, config, tmp_path / "source.yaml", "other", tmp_path / "run")
    assert result.experiment_export["status"] == "failed" and "disk full" in result.experiment_export["error"]
    assert not result.backtest.nav.empty
    assert json.loads((tmp_path / "run/experiment_export.json").read_text())["status"] == "failed"


def test_exported_result_can_be_viewed_over_http(config, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from betalens.experiments.api import create_app
    from betalens.datafeed import Datafeed
    config["experiment_export"]["enabled"] = True
    result = execution.execute_factor(config, tmp_path / "source.yaml", lambda *_: None, FakePipeline)
    assert result.experiment_export["status"] == "saved"
    monkeypatch.setattr(Datafeed, "__init__", lambda *a, **k: pytest.fail("offline view queried market"))
    with TestClient(create_app(tmp_path / "experiments")) as client:
        catalog = client.get("/api/catalog").json()
        assert catalog["items"][0]["key"] == "alpha101:ALPHA12:cross_sectional"
        response = client.get(f"/api/experiments/{result.experiment_export['experiment_id']}/result")
        assert response.status_code == 200, response.text
        assert response.json()["availability"]["raw"] is True
        assert response.json()["charts"]["nav"]


def test_dashboard_exports_before_releasing_result(config, tmp_path, monkeypatch):
    from . import runs
    from betalens.analyst.presentation import archived_names
    config["experiment_export"]["enabled"] = True
    monkeypatch.setattr(runs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runs, "_CACHE_ROOT", tmp_path / "outputs/cache/dashboard")
    script = FACTOR_ROOT / "alpha101/ALPHA12/factor_ALPHA12.py"
    monkeypatch.setattr(runs, "get_factor_config", lambda *a: (script, {}, config))
    monkeypatch.setattr(runs, "load_factor_module", lambda *a: SimpleNamespace(build_spec=lambda *a: None, FactorPipeline=FakePipeline))
    manager = RunManager()
    monkeypatch.setattr(manager, "_dump_excel", lambda *a: None)
    run = DashboardRun(RunRequest(factor_class="alpha101", name="ALPHA12",
                                 parameters={"experiment_export_enabled": True}))
    try:
        with archived_names({"000001.SZ": "测试证券"}):
            manager._execute(run)
        assert run.status == "completed", run.error
        assert run.backtest is None and run.result is None
        assert run.payload["run"]["experiment_export"]["status"] == "saved"
        assert Path(run.output_dir) == tmp_path / "outputs/runs/alpha101/ALPHA12" / run.run_id
        assert run.table_path("positions").exists()
        run.cleanup()
        assert (Path(run.output_dir) / "experiment_export.json").exists()
        assert (tmp_path / "experiments").is_dir()
    finally:
        manager._executor.shutdown(wait=True)
        manager._dump_executor.shutdown(wait=True)


@pytest.mark.parametrize("value", ["false", 0, None])
def test_export_requires_boolean(value):
    with pytest.raises(ConfigError, match="布尔"):
        execution.export_settings({"experiment_export": {"enabled": value}})


@pytest.mark.parametrize("name", ["ALPHA12", "ALPHA12_timing"])
def test_dashboard_checkbox_roundtrip(name, tmp_path):
    _, _, cfg = get_factor_config("alpha101", name)
    run = DashboardRun(RunRequest(factor_class="alpha101", name=name,
        parameters={"experiment_export_enabled": True}))
    effective = RunManager._build_run_config(cfg, run, tmp_path)
    assert effective["experiment_export"]["enabled"] is True
    assert RunManager._flat_parameters(effective)["experiment_export_enabled"] is True
    run.experiment_export = {"status": "saved", "experiment_id": "test-id"}
    assert run.to_state().model_dump()["experiment_export"]["experiment_id"] == "test-id"


@pytest.mark.parametrize("enabled,capture,fail_capture", [(False, True, False), (True, False, False),
                                                         (True, True, False), (True, True, True)])
def test_mining_export_switch_identity_and_partial_results(config, tmp_path, monkeypatch,
                                                          enabled, capture, fail_capture):
    from betalens.experiments import ExperimentStore
    from betalens.experiments.snapshot import capture_run
    from betalens.factor.mining_audit import FactorMiningResult, MiningResult
    import betalens.factor.mining as mining
    params_path, perf_path = tmp_path / "parameters.yaml", tmp_path / "performance.yaml"
    write_yaml_config(params_path, {"factor_class": "DISP", "factors": {"DISP": {"module": "types"}}})
    write_yaml_config(perf_path, {"experiment_export": {"enabled": enabled,
        "store_path": str(tmp_path / "experiments"), "capture_selected": capture}})
    rows = pd.DataFrame([{"candidate_id": "winner", "params_json": '{"window":20}', "sharpe_median": 2.},
                         {"candidate_id": "other", "params_json": '{"window":40}', "sharpe_median": 1.}])
    run = FactorMiningResult("DISP", "task", tmp_path / "mining/DISP/task", "complete",
        coarse_summary=rows, selected_candidates=rows.iloc[:1],
        archive_context={"factor_class": "DISP", "strategy_type": "cross_sectional", "name": "DISP"})
    config["meta"].update({"class": "LiqDemand", "name": "DISP"})
    source = completed_result()
    if enabled:
        manual_cfg = copy.deepcopy(config)
        manual_cfg["experiment_export"]["enabled"] = True
        execution.export_run(source, manual_cfg, params_path, "manual", tmp_path / "run")
    calls = []
    def search(*args, **kwargs):
        calls.append(kwargs["capture_selected"])
        if kwargs["capture_selected"]:
            if fail_capture:
                run.capture_errors = {"winner": "capture failed"}
            else:
                run.selected_results = {"winner": {"scopes": {"total": capture_run(source)}, "config": config}}
        return MiningResult("launch", (run,))
    monkeypatch.setattr(mining, "run_mining", search)
    result = execution.run_mining_from_config(params_path, perf_path)
    assert calls == [enabled and capture]
    if not enabled:
        assert result.experiment_export["status"] == "disabled"
        assert not (tmp_path / "experiments").exists()
        return
    assert result.experiment_export["status"] == ("partial" if fail_capture else "complete")
    with ExperimentStore(tmp_path / "experiments", readonly=True) as store:
        assert len(store.factors()) == 1
        assert store.factors()[0]["key"] == "LiqDemand:DISP:cross_sectional"
        assert store.query(kind="mining")["total"] == 2
        assert store.query(kind="mining", has_raw=True)["total"] == (1 if capture and not fail_capture else 0)
        assert store.get_batch(result.experiment_export["batch_id"])["failures"] == result.experiment_export["failures"]


def test_missing_mining_dependency_reported_before_search(tmp_path, monkeypatch):
    import betalens.factor.mining as mining
    parameters = tmp_path / "parameters.yaml"
    performance = tmp_path / "performance.yaml"
    write_yaml_config(parameters, {"factors": {"missing": {"module": "_missing_factor_adapter_dependency"}}})
    write_yaml_config(performance, {})
    monkeypatch.setattr(mining, "run_mining", lambda *a, **k: pytest.fail("search started"))
    with pytest.raises(ConfigError, match="缺少依赖"):
        execution.run_mining_from_config(parameters, performance)


@pytest.mark.parametrize("relative", ["alpha101/mining/run.py", "LiqDemand/DISP/mining/run.py",
                                      "tdx/XICHOU/timing_mining/run.py"])
def test_mining_entrypoints_work_from_other_directory(relative, tmp_path):
    result = subprocess.run([sys.executable, str(FACTOR_ROOT / relative), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--parameters" in result.stdout and "--performance" in result.stdout


def test_illiq_does_not_fill_missing_close():
    module = load_factor_module(FACTOR_ROOT / "citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.py")
    close = pd.DataFrame({"A": [100., np.nan, 110., 121.]}, index=pd.bdate_range("2025-01-02", periods=4))
    result = module.compute_ILLIQ(close, close * 0 + 10., 1)
    assert result.iloc[:3].isna().all().all()
    assert result.iloc[3, 0] == pytest.approx(.01)


def test_discovery_uses_self_contained_multiple_yaml_fixture(tmp_path, monkeypatch, config):
    from . import factors
    category = tmp_path / "demo"
    directory = category / "SAMPLE"
    directory.mkdir(parents=True)
    write_yaml_config(category / "class_demo.yaml", {"source": "fixture"})
    for name, strategy in (("SAMPLE", "cross_sectional"), ("SAMPLE_timing", "timing")):
        cfg = copy.deepcopy(config)
        cfg["meta"].update({"class": "demo", "name": name, "strategy_type": strategy})
        write_yaml_config(directory / f"factor_{name}.yaml", cfg)
        (directory / f"factor_{name}.py").write_text('"""Discovery fixture; no import side effects."""\n')
    monkeypatch.setattr(factors, "FACTOR_ROOT", tmp_path)
    factors.clear_factor_cache()
    try:
        found = factors.discover_factors()
        assert {f.name for f in found} == {"SAMPLE", "SAMPLE_timing"}
        assert factors.get_factor_config("demo", "SAMPLE_timing")[0].parent.name == "SAMPLE"
    finally:
        factors.clear_factor_cache()


def test_alpha_returns_converted_once_and_all_catalog_shapes():
    from factor_template import FactorSpec
    from betalens.factor.config import factor_spec_options, run_parameters
    rng = np.random.default_rng(12)
    days = pd.bdate_range("2020-01-02", periods=550)
    columns = pd.Index([f"{i:06d}.SZ" for i in range(1, 13)])
    close = pd.DataFrame(30 * np.exp(np.cumsum(rng.normal(0, .02, (len(days), len(columns))), axis=0)), index=days, columns=columns)
    volume = pd.DataFrame(rng.lognormal(12, 1, close.shape), index=days, columns=columns)
    frames = {"close": close, "open": close * 1.005, "high": close * 1.02, "low": close * .98,
              "volume": volume, "amount": volume * close, "average_price": close * 1.001,
              "pct_change": close.pct_change(fill_method=None) * 100, "total_market_value": close * 1e8}
    catalog = discover_factors()
    assert len(catalog) == 209
    for path in sorted(FACTOR_ROOT.glob("*/*/factor_*.yaml")):
        cfg = load_yaml_config(path)
        assert execution.export_settings(cfg)["enabled"] is False
        factor_spec_options(cfg, path)
        run_parameters(cfg, path)
        mod = load_factor_module(path.with_suffix(".py"))
        spec = mod.build_spec(cfg, path)
        inputs = {key: frames[metric].copy() for key, metric in spec.inputs.items()}
        inputs.update({key: pd.DataFrame(np.tile(np.arange(len(columns)) // 3, (len(days), 1)), index=days, columns=columns)
                       for key in getattr(spec, "industry_inputs", {})})
        result = spec.compute(**inputs, **spec.compute_kwargs)
        validate_factor_output(result, inputs, spec.name)
    from alpha101_formulas import alpha1, compute_alpha
    supplied = {"close_wide": close, "returns_wide": frames["pct_change"]}
    expected = alpha1({"close": close, "returns": frames["pct_change"] / 100}, returns_threshold=-.015)
    pd.testing.assert_frame_equal(compute_alpha(1, **supplied, returns_threshold=-.015), expected)


def test_liqdemand_uses_shared_preprocessing_and_complete_months(tmp_path, monkeypatch):
    import factor_template as core
    core._ensure_runtime()
    sys.path.insert(0, str(FACTOR_ROOT / "LiqDemand"))
    import factor_template_liqdemand as liq
    days = [day.date() for day in pd.bdate_range("2025-01-01", "2025-02-28")]
    codes = [f"{i:06d}.SZ" for i in range(1, 9)]
    panel = pd.DataFrame(np.arange(len(days) * len(codes)).reshape(len(days), -1) + 10., index=pd.DatetimeIndex(days), columns=codes)
    monkeypatch.setattr(core, "get_absolute_trade_days", lambda start, end, freq: [d for d in days if str(start) <= str(d) <= str(end)])
    monkeypatch.setattr(liq, "get_absolute_trade_days", lambda start, end, freq: [d for d in days if str(start) <= str(d) <= str(end)])
    monkeypatch.setattr(core, "fetch_daily_wide", lambda *a, **k: panel)
    industry = lambda df, **kw: pd.Series([str(i % 2) for i in range(len(df))], index=pd.MultiIndex.from_frame(df[["input_ts", "code"]]))
    monkeypatch.setattr(core, "query_industry_panel", industry)
    from betalens.factor import research
    monkeypatch.setattr(research, "attach_research", lambda bt, fv, *a, **k: fv)
    captured = {}
    def engine(weights, **kwargs):
        captured.update(weights=weights, kwargs=kwargs)
        return SimpleNamespace(trade_cycles=pd.DataFrame(columns=["status"]))
    monkeypatch.setattr(core, "BacktestBase", engine)
    monkeypatch.setattr(core, "Analyst", SimpleNamespace(from_backtest=lambda *a, **k: SimpleNamespace(report=lambda **k: {})))
    monkeypatch.setattr(core, "_compute_group_nav", lambda *a: pd.DataFrame())
    spec = core.FactorSpec(name="DISP", inputs={"close": "close"}, compute=lambda close: close,
                           index_code=None, use_industry=True, weight_mode="classic-long-short")
    result = liq.LiqDemandPipeline(spec).run("2025-02-03", "2025-02-18", warmup_days=10,
        universe=codes, n_quantiles=2, output_dir=str(tmp_path), include_profiling=False, dump_excel=False, verbose=False)
    assert result.neutralize_stats is not None
    weights = captured["weights"]
    pretom = liq.get_pretom_dates("2025-02-01", "2025-02-28")
    assert weights.loc[[ts.date() not in pretom for ts in weights.index], codes].eq(0).all().all()
    assert captured["kwargs"]["table_name"] == spec.table_name
    assert captured["kwargs"]["end_date"] == "2025-02-18"
