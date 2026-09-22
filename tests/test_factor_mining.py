from __future__ import annotations

import json
import sys
import sqlite3
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import betalens.factor.mining as mining
import betalens.factor.mining_audit as mining_audit
import betalens.factor.mining_optuna as mining_optuna
from betalens.factor.mining_cache import CacheRequest, MiningCache
from betalens.factor.mining_optuna import (
    create_coarse_study,
    create_fine_grid_study,
    detect_boundary_pressure,
    expand_parameter_specs,
    generate_coarse_candidates,
    generate_fine_candidates,
    generate_perturbation_candidates,
    seed_study_with_results,
    suggest_params,
    tell_trial,
)


def test_alpha101_compact_factor_list_expands_automatic_configs(monkeypatch):
    alpha_directory = Path(__file__).parents[1] / "betalens-factor" / "alpha101"
    monkeypatch.syspath_prepend(str(alpha_directory))

    resolved = mining._resolve_alpha_configs(
        {
            "alpha101_parameter_generation": {
                "range_multiplier": 10,
                "max_dimensions": 5,
            }
        },
        ["ALPHA7", "alpha8"],
    )

    assert set(resolved) == {"ALPHA7", "ALPHA8"}
    assert resolved["ALPHA7"]["module"] == "alpha101_mining"
    assert resolved["ALPHA7"]["execution_mode"] == "precomputed"
    assert isinstance(resolved["ALPHA7"]["parameters"], dict)


def test_heatmap_uses_all_parameter_pairs_and_mean_aggregation(tmp_path):
    specs = {
        "open_sum_window": {"type": "int", "low": 1, "high": 2},
        "returns_sum_window": {"type": "int", "low": 1, "high": 2},
        "base_delay_lag": {"type": "int", "low": 1, "high": 2},
        "alpha_id": {"type": "int", "low": 8, "high": 8},
    }
    rows = []
    for length, step in ((252, 21), (504, 63)):
        for open_window in (1, 2):
            for returns_window in (1, 2):
                for delay_lag in (1, 2):
                    rows.append({
                        "factor_id": "ALPHA8",
                        "candidate_id": f"{length}-{open_window}-{returns_window}-{delay_lag}",
                        "stage": "fine",
                        "window_id": f"{length}/{step}/0",
                        "window_start": "2020-01-01",
                        "window_end": "2020-12-31",
                        "open_sum_window": open_window,
                        "returns_sum_window": returns_window,
                        "base_delay_lag": delay_lag,
                        "alpha_id": 8,
                        "sharpe": float(delay_lag),
                        "ann_ret": float(delay_lag) / 10,
                        "calmar": float(delay_lag) / 2,
                        "mdd": float(delay_lag) / 20,
                        "error": None,
                    })
    frame = pd.DataFrame(rows)
    assert mining_audit._heatmap_parameter_pairs(frame, specs) == [
        ("open_sum_window", "returns_sum_window"),
        ("open_sum_window", "base_delay_lag"),
        ("returns_sum_window", "base_delay_lag"),
    ]
    matrix = mining_audit._heatmap_matrix(
        frame.loc[frame.window_id.eq("252/21/0")],
        "open_sum_window",
        "returns_sum_window",
        "sharpe",
    )
    assert matrix.loc[1, 1] == pytest.approx(1.5)

    metadata = {
        "factor_id": "ALPHA8",
        "configuration": {
            "parameter_space": {"evaluation": {"span": ["2020-01-01", "2020-12-31"]}}
        },
        "resolved_parameter_specs": specs,
    }
    output = mining_audit._write_heatmap_report(tmp_path, metadata, frame)
    png_paths = [Path(path) for path in output if str(path).endswith(".png")]
    assert len(png_paths) == 6
    assert all("_total_" in path.name for path in png_paths)
    assert not any(name in path.name for path in png_paths for name in ("train", "test", "valid"))
    report = json.loads((tmp_path / "热力图报告.json").read_text(encoding="utf-8"))
    assert len(report) == 6
    assert {row["aggregation"] for row in report} == {"mean"}
    assert {(row["window_length"], row["window_step"]) for row in report} == {(252, 21), (504, 63)}


def test_optuna_search_supports_log_and_composite_categories():
    specs = {
        "window": {"type": "int", "low": 10, "high": 80, "step": 1, "scale": "log"},
        "pair": {"type": "categorical", "choices": [[9, 4], [7, 4]]},
    }
    study = create_coarse_study({"sampler": "random", "seed": 7})
    trial = study.ask()
    params = suggest_params(trial, specs)
    tell_trial(study, trial, 1.0)
    assert 10 <= params["window"] <= 80
    assert params["pair"] in [[9, 4], [7, 4]]

    plan = generate_fine_candidates(
        specs,
        [{"window": 20, "pair": [9, 4]}],
        {"points_per_dimension": 3, "max_candidates": 6},
        coarse_candidates=[
            {"window": 10, "pair": [9, 4]},
            {"window": 20, "pair": [7, 4]},
            {"window": 40, "pair": [9, 4]},
            {"window": 80, "pair": [7, 4]},
        ],
    )
    candidates = plan.candidates
    assert len(candidates) <= 6
    assert all(10 <= row["window"] <= 40 for row in candidates)
    assert plan.local_bounds["window"]["local_low"] == 10
    assert plan.local_bounds["window"]["local_high"] == 40

    grid = create_fine_grid_study({"window": [20, 40], "pair": [[9, 4], [7, 4]]})
    categorical_specs = {
        "window": {"type": "categorical", "choices": [20, 40]},
        "pair": specs["pair"],
    }
    seen = []
    for _ in range(4):
        trial = grid.ask()
        seen.append(suggest_params(trial, categorical_specs))
        tell_trial(grid, trial, 0.0)
    assert len({str(row) for row in seen}) == 4


def test_fine_grid_applies_candidate_limit_before_materializing_cartesian_product():
    specs = {
        f"parameter_{index}": {"type": "categorical", "choices": list(range(5))}
        for index in range(30)
    }
    anchor = {name: 2 for name in specs}

    plan = generate_fine_candidates(
        specs,
        [anchor],
        {"points_per_dimension": 5, "max_candidates": 17},
    )

    assert len(plan.candidates) == 17
    assert all(set(candidate) == set(specs) for candidate in plan.candidates)


def test_qmc_wide_search_boundary_expansion_and_perturbations_are_bounded():
    specs = {
        "window": {"type": "int", "low": 1, "high": 100, "step": 1, "scale": "log"},
        "weight": {"type": "float", "low": 0.0, "high": 1.0, "scale": "linear"},
    }
    candidates = generate_coarse_candidates(
        specs,
        {"sampler": "qmc", "qmc_type": "sobol", "scramble": True, "seed": 7, "n_trials": 16},
    )
    windows = {row["window"] for row in candidates}
    assert min(windows) <= 3
    assert max(windows) >= 60

    pressure = detect_boundary_pressure(
        specs,
        [{"window": 80}, {"window": 90}, {"window": 100}],
        tolerance=0.1,
        winner_ratio=2 / 3,
    )
    assert pressure["window"]["sides"] == ["high"]
    expanded = expand_parameter_specs(
        specs,
        pressure,
        multiplier=3,
        limits={"window": {"low": 1, "high": 250}},
    )
    assert expanded["window"]["high"] == 250
    assert specs["window"]["high"] == 100

    plan = generate_perturbation_candidates(
        expanded,
        [{"candidate_id": "WINNER", "window": 100, "weight": 0.5}],
        perturbations_per_candidate=8,
        radius_ratio=0.1,
        seed=11,
    )
    assert len(plan.candidates) == 8
    assert all(1 <= row["window"] <= 250 for row in plan.candidates)
    assert all(0 <= row["weight"] <= 1 for row in plan.candidates)
    assert all(row != {"window": 100, "weight": 0.5} for row in plan.candidates)
    assert {value["parent_candidate_id"] for value in plan.metadata.values()} == {"WINNER"}

    tpe = create_coarse_study({"sampler": "tpe", "seed": 9, "n_startup_trials": 1})
    added = seed_study_with_results(
        tpe,
        specs,
        [{"window": 10, "weight": 0.2}, {"window": 80, "weight": 0.8}],
        [1.0, 2.0],
    )
    assert added == 2
    assert len(tpe.trials) == 2
    assert [trial.value for trial in tpe.trials] == [1.0, 2.0]


def test_tpe_with_all_invalid_previous_results_does_not_raise_keyerror():
    module = types.ModuleType("_mining_invalid_history_test")

    class Factor:
        name = "INVALID_HISTORY"
        compute_kwargs = {}
        weight_mode = "classic-long-short"

        @staticmethod
        def compute(**kwargs):
            return kwargs["x"]

    module.make_mining_spec = lambda params: mining.MiningSpec(Factor(), warmup_days=0)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    try:
        invalid = pd.DataFrame([{
            "factor_id": "INVALID_HISTORY",
            "candidate_id": "C",
            "window_id": "1",
            "error": "ValueError: empty nav",
            "window": 3,
        }])
        assert mining._summary(invalid, {"objective": {"metric": "sharpe", "aggregate": "median"}}).empty
    finally:
        monkeypatch.undo()


def test_alpha101_auto_space_uses_multiplier_dimensions_and_type_limits(monkeypatch):
    alpha_dir = Path("betalens-factor/alpha101").resolve()
    monkeypatch.syspath_prepend(str(alpha_dir))
    sys.modules.pop("alpha101_parameters", None)
    from alpha101_parameters import mining_parameter_specs

    alpha3 = mining_parameter_specs(3)
    window = alpha3["rank_open_rank_volume_correlation_window"]
    assert window == {"type": "int", "low": 1, "high": 100, "scale": "log", "step": 1}

    capped = mining_parameter_specs(
        3,
        range_multiplier=20,
        max_dimensions=1,
        type_limits={"window": {"low": 2, "high": 50}},
    )
    assert capped["rank_open_rank_volume_correlation_window"]["low"] == 2
    assert capped["rank_open_rank_volume_correlation_window"]["high"] == 50

    alpha1 = mining_parameter_specs(1, max_dimensions=1)
    assert alpha1["returns_threshold"]["type"] == "float"
    assert alpha1["returns_threshold"]["low"] < 0 < alpha1["returns_threshold"]["high"]


def test_cache_open_or_build_and_slice(tmp_path):
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=index)
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return {
            "inputs": {"x": values},
            "price": values,
            "execution_price": values,
            "trade_status": pd.DataFrame(1, index=index, columns=["A"]),
            "industry_by_scheme": {},
            "pit": {day.date(): {"A"} for day in index},
            "universe": ["A"],
            "metadata": {"version": "test"},
        }

    request = CacheRequest(tmp_path, "signature")
    first = MiningCache.open_or_build(request, builder)
    second = MiningCache.open_or_build(request, builder)
    assert calls == 1
    pd.testing.assert_frame_equal(
        first.load("x", "2024-01-02", "2024-01-03"),
        values.loc["2024-01-02":"2024-01-03"],
        check_freq=False,
    )
    assert second.universe == ["A"]
    assert (tmp_path / "input_manifest.json").exists()
    assert (tmp_path / "datasets").is_dir()
    assert not (tmp_path / "READY.json").exists()


def _synthetic_data() -> mining.MiningData:
    index = pd.date_range("2024-01-01", periods=15, freq="D")
    return mining.MiningData(
        inputs={"x": pd.DataFrame({"A": range(15)}, index=index)},
        price=pd.DataFrame({"A": range(100, 115)}, index=index),
        execution_price=pd.DataFrame({"A": range(100, 115)}, index=index),
        trade_status=None,
        pit=None,
        universe=["A"],
    )


def test_execution_modes_compute_once_and_isolate_windows(monkeypatch):
    data = _synthetic_data()
    windows = [
        mining.MiningWindow("4/4/0", "2024-01-05", "2024-01-08", 4, 4),
        mining.MiningWindow("4/4/1", "2024-01-09", "2024-01-12", 4, 4),
    ]
    calls = {"compute": 0, "nav": 0}

    class Factor:
        name = "fake"
        compute_kwargs = {}
        weight_mode = "classic-long-short"

        def compute(self, **kwargs):
            calls["compute"] += 1
            return kwargs["x"]

    precomputed = types.ModuleType("_mining_precomputed_test")
    precomputed.make_mining_spec = lambda params: mining.MiningSpec(Factor())
    monkeypatch.setitem(sys.modules, precomputed.__name__, precomputed)
    index = data.price.index
    monkeypatch.setattr(
        mining,
        "_build_weights",
        lambda *args, **kwargs: pd.DataFrame({"A": 1.0}, index=index),
    )

    def vector_nav(weights, price):
        calls["nav"] += 1
        return pd.Series(range(1, len(price) + 1), index=price.index, dtype=float)

    monkeypatch.setattr(mining, "_vector_nav", vector_nav)
    evaluation = {
        "span": ["2024-01-01", "2024-01-15"],
        "engine": "vector",
        "rebal_freq": "D",
    }
    rows = mining._evaluate_candidate(
        precomputed.__name__, "FAKE", {"p": 1}, "coarse", windows, data, evaluation
    )
    assert calls == {"compute": 1, "nav": 1}
    assert all(row["error"] is None for row in rows)

    seen = []
    factory_calls = []
    rolling = types.ModuleType("_mining_rolling_test")

    def fit_window(window_data, params, window, context):
        dates = pd.DatetimeIndex(window_data.inputs["x"].index).normalize()
        assert dates.max() <= pd.Timestamp(window.end)
        assert dates.min() >= pd.Timestamp(window.start) - pd.Timedelta(days=2)
        seen.append(window.window_id)
        return pd.DataFrame({"A": 1.0}, index=dates)

    def rolling_factory(params):
        value = mining.MiningSpec(
            Factor(), execution_mode="rolling_fit", fit_window=fit_window, warmup_days=2
        )
        factory_calls.append(value)
        return value

    rolling.make_mining_spec = rolling_factory
    monkeypatch.setitem(sys.modules, rolling.__name__, rolling)
    calls["nav"] = 0
    rows = mining._evaluate_candidate(
        rolling.__name__, "ROLL", {"p": 1}, "coarse", windows, data, evaluation
    )
    assert seen == ["4/4/0", "4/4/1"]
    assert len(factory_calls) == 1 + len(windows)
    assert len({id(value) for value in factory_calls}) == len(factory_calls)
    assert calls["nav"] == 2
    assert all(row["error"] is None for row in rows)


def test_precomputed_window_transform_reuses_factor_and_retests_each_window(monkeypatch):
    data = _synthetic_data()
    windows = [
        mining.MiningWindow("4/4/0", "2024-01-05", "2024-01-08", 4, 4),
        mining.MiningWindow("4/4/1", "2024-01-09", "2024-01-12", 4, 4),
    ]
    calls = {"compute": 0, "transform": 0, "nav": 0}

    class Factor:
        name = "transform"
        compute_kwargs = {}
        weight_mode = "classic-long-short"

        def compute(self, **kwargs):
            calls["compute"] += 1
            return kwargs["x"]

    def transform(weights, window, context):
        calls["transform"] += 1
        return weights

    module = types.ModuleType("_mining_transform_test")
    module.make_mining_spec = lambda params: mining.MiningSpec(
        Factor(), window_transform=transform,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        mining,
        "_build_weights",
        lambda *args, **kwargs: pd.DataFrame({"A": 1.0}, index=data.price.index),
    )

    def vector_nav(weights, price):
        calls["nav"] += 1
        return pd.Series(range(1, len(price) + 1), index=price.index, dtype=float)

    monkeypatch.setattr(mining, "_vector_nav", vector_nav)
    rows = mining._evaluate_candidate(
        module.__name__,
        "TRANSFORM",
        {"p": 1},
        "coarse",
        windows,
        data,
        {"span": ["2024-01-01", "2024-01-15"], "engine": "vector", "rebal_freq": "D"},
    )
    assert calls == {"compute": 1, "transform": 2, "nav": 2}
    assert all(row["error"] is None for row in rows)


def test_legacy_mining_section_has_migration_error(tmp_path):
    config = tmp_path / "factor.yaml"
    config.write_text("mining:\n  grid: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parameter_space.yaml"):
        mining._load_yaml(config)


def test_configuration_validation_rejects_unknown_fields_and_windows():
    with pytest.raises(ValueError, match="unknown fields"):
        mining.validate_parameter_specs(
            {"window": {"type": "int", "low": 10, "high": 20, "typo": 1}}
        )
    with pytest.raises(ValueError, match="must not exceed"):
        mining._validate_window_config({"lengths": [20], "steps": [21]})


def test_summary_supports_configured_metric_aggregation():
    frame = pd.DataFrame(
        [
            {"factor_id": "F", "candidate_id": "C", "window_id": "1", "params_json": "{}", "ann_ret": 0.1, "sharpe": 1.0, "mdd": 0.2, "turnover": 0.3, "rank_ic": 0.1, "error": None},
            {"factor_id": "F", "candidate_id": "C", "window_id": "2", "params_json": "{}", "ann_ret": 0.3, "sharpe": 2.0, "mdd": 0.1, "turnover": 0.2, "rank_ic": 0.2, "error": None},
        ]
    )
    result = mining._summary(
        frame,
        {"objective": {"metric": "ann_ret", "aggregate": "median"}, "top_k": 1},
    )
    assert result.loc[0, "objective"] == pytest.approx(0.2)
    assert result.loc[0, "selection_status"] == "selected"


def test_timing_metric_name_collision_restores_parameters_for_tpe_seed():
    """A timing metric must not replace its same-named search parameter."""
    frame = pd.DataFrame([
        {
            "factor_id": "TIMING",
            "candidate_id": "C",
            "window_id": "1",
            "params_json": json.dumps({"max_gross_exposure": 0.8, "factor_id": "TIMING"}),
            # This is the observed metric, not the candidate parameter.
            "max_gross_exposure": 0.0,
            "sharpe": 1.0,
            "error": None,
        },
        {
            "factor_id": "TIMING",
            "candidate_id": "C",
            "window_id": "2",
            "params_json": json.dumps({"max_gross_exposure": 0.8, "factor_id": "TIMING"}),
            "max_gross_exposure": 0.6,
            "sharpe": 1.5,
            "error": None,
        },
    ])
    summary = mining._summary(
        frame,
        {"objective": {"metric": "sharpe", "aggregate": "median"}},
    )
    specs = {"max_gross_exposure": {"type": "float", "low": 0.25, "high": 1.0}}
    params = mining._top_parameter_rows(summary, specs, 1)
    assert params == [{"max_gross_exposure": 0.8}]
    assert summary.loc[0, "max_gross_exposure_median"] == pytest.approx(0.3)
    filtered = mining._summary(
        frame,
        {
            "objective": {"metric": "sharpe", "aggregate": "median"},
            "filters": [{"metric": "max_gross_exposure", "op": ">=", "value": 0.5}],
        },
    )
    assert filtered.loc[0, "selection_status"] == "filtered"
    study = create_coarse_study({"sampler": "tpe", "seed": 3, "n_startup_trials": 1})
    assert seed_study_with_results(study, specs, params, [1.25]) == 1


def test_rank_ic_observations_are_deduplicated_across_overlapping_windows():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    factor = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 1.0, 3.0, 2.0],
            [2.0, 4.0, 1.0, 3.0],
            [1.0, 3.0, 4.0, 2.0],
            [3.0, 2.0, 1.0, 4.0],
        ],
        index=dates,
        columns=list("ABCD"),
    )
    price = pd.DataFrame(
        [
            [100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0],
            [101.0, 102.0, 103.0, 104.0],
            [102.0, 101.0, 106.0, 103.0],
            [104.0, 105.0, 103.0, 108.0],
        ],
        index=dates,
        columns=list("ABCD"),
    )
    first = [(dates[0], dates[1]), (dates[1], dates[2]), (dates[2], dates[3])]
    second = [(dates[1], dates[2]), (dates[2], dates[3]), (dates[3], dates[4])]

    observations = mining._rank_ic_observations(factor, price, [first, second])

    assert len(observations) == 3
    assert mining._rank_ic_metrics(observations, first) == mining._rank_ic(factor, price, first)
    assert np.isfinite(mining._rank_ic_metrics(observations, first)["ic_ir"])
    audit_rows = mining._rank_ic_audit_rows(
        observations,
        factor_id="F",
        candidate_id="C",
        stage="coarse",
        params_json="{}",
    )
    assert len(audit_rows) == 3
    assert {"signal_date", "rebalance_date", "forward_end_date", "cross_section_n"}.issubset(audit_rows[0])


def test_effective_workers_are_capped_by_physical_cores(monkeypatch):
    data = _synthetic_data()
    monkeypatch.setattr(mining, "_physical_cpu_count", lambda: 3)
    monkeypatch.setattr(mining, "_system_memory", lambda: (64 * 1024**3, 64 * 1024**3))

    assert mining._effective_workers(16, data, 0.7) == 3


def test_exact_engine_uses_preloaded_frames():
    index = pd.date_range("2024-01-02", periods=12, freq="B") + pd.Timedelta(hours=15)
    price = pd.DataFrame(
        {
            "A": [10 + value * 0.1 for value in range(12)],
            "B": [20 - value * 0.1 for value in range(12)],
        },
        index=index,
    )
    status = pd.DataFrame(1, index=index.normalize(), columns=["A", "B"])
    rebalance = pd.DatetimeIndex([index[1].normalize(), index[6].normalize()]) + pd.Timedelta(minutes=10)
    weights = pd.DataFrame(
        {"A": [0.5, 0.5], "B": [-0.5, -0.5], "cash": [1.0, 1.0]},
        index=rebalance,
    )
    data = mining.MiningData(
        inputs={},
        price=price,
        execution_price=price,
        trade_status=status,
        pit=None,
        universe=["A", "B"],
    )
    factor = SimpleNamespace(
        name="EXACT_SMOKE",
        backtest_metric="close",
        table_name="daily_market",
    )
    nav = mining._exact_nav(weights, data, mining.MiningSpec(factor), 1_000_000, 24)
    assert not nav.empty
    assert nav.notna().all()


def test_run_logging_is_live_and_persisted(tmp_path, monkeypatch, capsys):
    module = types.ModuleType("_mining_logging_test")

    class Factor:
        name = "LOG_TEST"
        compute_kwargs = {}
        weight_mode = "classic-long-short"

        @staticmethod
        def compute(**kwargs):
            return kwargs["x"]

    module.make_mining_spec = lambda params: mining.MiningSpec(Factor(), warmup_days=0)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    data = _synthetic_data()
    windows = [
        mining.MiningWindow("4/4/0", "2024-01-05", "2024-01-08", 4, 4),
        mining.MiningWindow("4/4/1", "2024-01-09", "2024-01-12", 4, 4),
    ]
    monkeypatch.setattr(mining, "_windows", lambda *args, **kwargs: windows)
    monkeypatch.setattr(mining, "_fetch_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        mining,
        "_build_weights",
        lambda *args, **kwargs: pd.DataFrame({"A": 1.0}, index=data.price.index),
    )
    monkeypatch.setattr(
        mining,
        "_vector_nav",
        lambda weights, price: pd.Series(
            np.linspace(1.0, 1.2, len(price)), index=price.index
        ),
    )
    parameter_path = tmp_path / "parameter_space.yaml"
    performance_path = tmp_path / "performance.yaml"
    output_dir = tmp_path / "output"
    parameter_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "factor_class": "test",
                "factors": {
                    "LOG_TEST": {
                        "module": module.__name__,
                        "execution_mode": "precomputed",
                        "parameters": {
                            "window": {"type": "int", "low": 3, "high": 3, "step": 1}
                        },
                    }
                },
                "evaluation": {
                    "span": ["2024-01-01", "2024-01-15"],
                    "engine": "vector",
                    "rebal_freq": "D",
                },
                "windows": {"lengths": [4], "steps": [4]},
                "search": {
                    "coarse": {"sampler": "random", "n_trials": 1, "seed": 1},
                    "fine": {"top_k": 1, "points_per_dimension": 1, "max_candidates": 1},
                },
                "selection": {
                    "objective": {"metric": "sharpe", "aggregate": "median", "direction": "maximize"},
                    "top_k": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    performance_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"backend": "serial", "workers": 1},
                "cache": {"data_enabled": False},
                "output": {
                    "directory": str(output_dir),
                },
                "logging": {"level": "INFO", "task_logs": True, "heartbeat_seconds": 0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = mining.run_mining(parameter_path, performance_path)
    terminal = capsys.readouterr().out
    assert len(result.factor_runs) == 1
    factor_run = result.factor_runs[0]
    assert factor_run.run_dir.parent == output_dir / "LOG_TEST"
    for marker in (
        "开始参数挖掘",
        "开始处理因子",
        "开始宽范围搜索",
        "开始评价第1/",
        "宽范围粗搜进度",
        "细搜筛选完成",
        "因子挖掘完成",
    ):
        assert marker in terminal
    assert "信息 [主进程 PID=" in terminal
    assert "窗口方案=4日窗口、每4日滑动、第1个窗口" in terminal
    metadata = yaml.safe_load((factor_run.run_dir / "metadata.yaml").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["launch_id"] == result.launch_id
    audit = factor_run.run_dir / "audit" / "运行日志.log"
    assert audit.exists()
    audit_text = audit.read_text(encoding="utf-8")
    assert "开始参数挖掘" in audit_text
    assert "完成第1/" in audit_text
    assert "因子挖掘完成" in audit_text
    assert (factor_run.run_dir / "audit" / "挖掘审计.xlsx").exists()
    result_db = factor_run.run_dir / "cache" / "results.sqlite3"
    assert result_db.exists()
    with sqlite3.connect(result_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM window_results").fetchone()[0] == 4
        assert connection.execute("SELECT COUNT(*) FROM rank_ic_series").fetchone()[0] > 0
        assert connection.execute("SELECT COUNT(*) FROM candidate_summary").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM winners").fetchone()[0] == 1
    workbook = pd.ExcelFile(factor_run.run_dir / "audit" / "挖掘审计.xlsx")
    assert {
        "运行概览", "运行配置", "参数空间", "搜索进度", "窗口表现",
        "IC分析", "候选汇总", "赢家参数", "赢家汇总", "错误",
    }.issubset(workbook.sheet_names)
    ic_audit = pd.read_excel(factor_run.run_dir / "audit" / "挖掘审计.xlsx", sheet_name="IC分析")
    assert {"记录类型", "IC IR", "Rank IC", "信号日", "收益截止日"}.issubset(ic_audit.columns)
    assert {"窗口IC IR", "逐期Rank IC"}.issubset(set(ic_audit["记录类型"]))
    assert not (output_dir / "run_manifest.json").exists()
    assert not any(
        getattr(handler, "_betalens_mining_handler", False)
        for handler in mining._LOGGER.handlers
    )
    second = mining.run_mining(parameter_path, performance_path).factor_runs[0]
    assert second.run_dir != factor_run.run_dir
    assert second.run_dir.parent == factor_run.run_dir.parent


def test_multistage_search_runs_qmc_tpe_expansion_grid_and_stability(tmp_path, monkeypatch):
    module = types.ModuleType("_mining_multistage_test")

    class Factor:
        name = "MULTISTAGE"
        compute_kwargs = {}
        weight_mode = "classic-long-short"

        @staticmethod
        def compute(**kwargs):
            return kwargs["x"]

    module.make_mining_spec = lambda params: mining.MiningSpec(Factor(), warmup_days=0)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    data = _synthetic_data()
    windows = [mining.MiningWindow("4/4/0", "2024-01-05", "2024-01-08", 4, 4)]
    monkeypatch.setattr(mining, "_windows", lambda *args, **kwargs: windows)
    monkeypatch.setattr(mining, "_fetch_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        mining,
        "_build_weights",
        lambda *args, **kwargs: pd.DataFrame({"A": 1.0}, index=data.price.index),
    )
    monkeypatch.setattr(
        mining,
        "_vector_nav",
        lambda weights, price: pd.Series(np.linspace(1.0, 1.2, len(price)), index=price.index),
    )
    monkeypatch.setattr(
        mining_optuna,
        "detect_boundary_pressure",
        lambda *args, **kwargs: {
            "window": {"sides": ["high"], "low_ratio": 0.0, "high_ratio": 1.0, "winner_count": 1}
        },
    )

    parameter_path = tmp_path / "parameter_space.yaml"
    performance_path = tmp_path / "performance.yaml"
    output_dir = tmp_path / "output"
    parameter_path.write_text(
        yaml.safe_dump({
            "version": 1,
            "factor_class": "test",
            "factors": {
                "MULTISTAGE": {
                    "module": module.__name__,
                    "execution_mode": "precomputed",
                    "parameters": {"window": {"type": "int", "low": 2, "high": 4, "step": 1}},
                    "parameter_limits": {"window": {"low": 1, "high": 8}},
                }
            },
            "evaluation": {"span": ["2024-01-01", "2024-01-15"], "engine": "vector", "rebal_freq": "D"},
            "windows": {"lengths": [4], "steps": [4]},
            "search": {
                "coarse": {"sampler": "qmc", "qmc_type": "sobol", "n_trials": 4, "seed": 1},
                "refine": {"enabled": True, "sampler": "tpe", "n_trials": 4, "batch_size": 2, "n_startup_trials": 1, "seed": 2},
                "expansion": {"enabled": True, "sampler": "qmc", "n_trials": 2, "max_rounds": 1, "range_multiplier": 2, "seed": 3},
                "fine": {"top_k": 1, "points_per_dimension": 2, "max_candidates": 2},
                "stability": {"enabled": True, "top_k": 1, "perturbations_per_candidate": 2, "radius_ratio": 0.5, "required_pass_ratio": 0.5, "minimum_valid_ratio": 0.5, "seed": 4},
            },
            "selection": {"objective": {"metric": "sharpe", "aggregate": "median", "direction": "maximize"}, "top_k": 1},
        }, sort_keys=False),
        encoding="utf-8",
    )
    performance_path.write_text(
        yaml.safe_dump({
            "runtime": {"backend": "serial", "workers": 1},
            "cache": {"data_enabled": False},
            "output": {"directory": str(output_dir)},
            "logging": {"level": "WARNING", "task_logs": False, "heartbeat_seconds": 0},
        }, sort_keys=False),
        encoding="utf-8",
    )

    factor_run = mining.run_mining(parameter_path, performance_path).factor_runs[0]
    assert {"coarse", "refine", "expansion_1", "fine", "stability"}.issubset(factor_run.stage_summaries)
    assert factor_run.selected_candidates.iloc[0]["stability_status"] == "stable"
    metadata = yaml.safe_load((factor_run.run_dir / "metadata.yaml").read_text(encoding="utf-8"))
    assert "expansion_1" in metadata["result_counts"]["stage_candidates"]
    workbook = pd.ExcelFile(factor_run.run_dir / "audit" / "挖掘审计.xlsx")
    assert "稳定性验证" in workbook.sheet_names


def test_multi_factor_launch_creates_isolated_task_directories(tmp_path, monkeypatch):
    parameter_path = tmp_path / "parameter_space.yaml"
    performance_path = tmp_path / "performance.yaml"
    output_dir = tmp_path / "output"
    parameter_path.write_text(
        yaml.safe_dump({
            "version": 1,
            "factor_class": "test",
            "factors": {
                "F1": {"module": "unused", "execution_mode": "precomputed", "parameters": {}},
                "F2": {"module": "unused", "execution_mode": "precomputed", "parameters": {}},
            },
            "evaluation": {"span": ["2024-01-01", "2024-01-15"]},
            "windows": {"lengths": [4], "steps": [4]},
        }, sort_keys=False),
        encoding="utf-8",
    )
    performance_path.write_text(
        yaml.safe_dump({
            "runtime": {"backend": "serial", "workers": 1},
            "cache": {"data_enabled": False},
            "output": {"directory": str(output_dir)},
            "logging": {"level": "INFO", "task_logs": False, "heartbeat_seconds": 0},
        }, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mining,
        "_windows",
        lambda *args, **kwargs: [mining.MiningWindow("4/4/0", "2024-01-01", "2024-01-04", 4, 4)],
    )

    def fake_run_factor(task, factor_id, *args, **kwargs):
        return mining.FactorMiningResult(
            factor_id=factor_id,
            run_id=task.run_id,
            run_dir=task.run_dir,
            status="complete",
        )

    monkeypatch.setattr(mining, "_run_factor", fake_run_factor)
    result = mining.run_mining(parameter_path, performance_path)
    assert [run.factor_id for run in result.factor_runs] == ["F1", "F2"]
    assert len({run.run_dir for run in result.factor_runs}) == 2
    for run in result.factor_runs:
        metadata = yaml.safe_load((run.run_dir / "metadata.yaml").read_text(encoding="utf-8"))
        assert metadata["launch_id"] == result.launch_id
        assert run.run_dir.parent == output_dir / run.factor_id


def test_legacy_performance_options_are_rejected(tmp_path):
    parameter_path = tmp_path / "parameter_space.yaml"
    performance_path = tmp_path / "performance.yaml"
    parameter_path.write_text(
        yaml.safe_dump({
            "version": 1,
            "factors": {"F": {"module": "unused", "execution_mode": "precomputed", "parameters": {}}},
            "evaluation": {"span": ["2024-01-01", "2024-01-15"]},
            "windows": {"lengths": [4], "steps": [4]},
        }),
        encoding="utf-8",
    )
    performance_path.write_text(
        yaml.safe_dump({"cache": {"enabled": True}, "output": {"directory": str(tmp_path)}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cache.enabled"):
        mining.run_mining(parameter_path, performance_path)


def test_failed_run_updates_manifest_and_audit_log(tmp_path):
    parameter_path = tmp_path / "parameter_space.yaml"
    performance_path = tmp_path / "performance.yaml"
    output_dir = tmp_path / "output"
    parameter_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "factor_class": "test",
                "factors": {
                    "BROKEN": {
                        "module": "_missing_mining_module",
                        "execution_mode": "precomputed",
                        "parameters": {"window": {"type": "int", "low": 3, "high": 3}},
                    }
                },
                "evaluation": {"span": ["2024-01-01", "2024-01-15"]},
                "windows": {"lengths": [4], "steps": [4]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    performance_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"backend": "serial", "workers": 1},
                "cache": {"data_enabled": False},
                "output": {"directory": str(output_dir)},
                "logging": {"level": "INFO", "task_logs": True, "heartbeat_seconds": 0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModuleNotFoundError):
        mining.run_mining(parameter_path, performance_path)
    run_dirs = list((output_dir / "BROKEN").iterdir())
    assert len(run_dirs) == 1
    metadata = yaml.safe_load((run_dirs[0] / "metadata.yaml").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert "ModuleNotFoundError" in metadata["error"]
    assert "Traceback" in metadata["traceback"]
    audit = (run_dirs[0] / "audit" / "运行日志.log").read_text(encoding="utf-8")
    assert "参数挖掘失败" in audit
    assert "ModuleNotFoundError" in audit
    assert (run_dirs[0] / "audit" / "挖掘审计.xlsx").exists()
    assert not any(
        getattr(handler, "_betalens_mining_handler", False)
        for handler in mining._LOGGER.handlers
    )
