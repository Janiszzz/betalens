from __future__ import annotations

from types import SimpleNamespace
import logging
import pickle
import sys

import numpy as np
import pandas as pd

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
ALPHA_ROOT = REPO_ROOT / "betalens-factor" / "alpha101"
for path in (REPO_ROOT, REPO_ROOT / "betalens-factor", ALPHA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha101_formulas import (  # noqa: E402
    compute_alpha,
    default_compute_kwargs,
    get_definition,
    required_history_bars_for_alpha,
    resolve_compute_kwargs,
)
from alpha101_parameters import (  # noqa: E402
    candidate_values,
    default_search_space,
    formula_param_candidates,
    formula_param_gid,
    grid_candidate_count,
)
from betalens.factor import mining  # noqa: E402
from betalens.factor import mining_optuna  # noqa: E402


def _panel() -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2024-01-01", periods=8)
    columns = ["A", "B"]
    base = pd.DataFrame(np.arange(16, dtype=float).reshape(8, 2) + 10, index=index, columns=columns)
    return {"close_wide": base, "open_wide": base - 0.2, "high_wide": base + 0.4, "low_wide": base - 0.4}


def test_formula_defaults_and_overrides_are_validated() -> None:
    inputs = _panel()
    inputs["returns_wide"] = inputs["close_wide"].pct_change(fill_method=None).fillna(0) * 100
    baseline = compute_alpha(1, **inputs)
    defaults = compute_alpha(1, **inputs, **default_compute_kwargs(1))
    pd.testing.assert_frame_equal(baseline, defaults)

    changed = dict(default_compute_kwargs(1))
    changed["returns_stddev_window"] = 40
    assert required_history_bars_for_alpha(1, changed) > required_history_bars_for_alpha(1)
    with __import__("pytest").raises(KeyError):
        resolve_compute_kwargs(1, {"unknown": 1})
    with __import__("pytest").raises(TypeError, match="formula_params is no longer supported"):
        compute_alpha(1, formula_params={}, **inputs)


def test_candidate_generation_is_bounded_and_deterministic() -> None:
    search_space = default_search_space(65)
    first = formula_param_candidates(65, search_space, max_grid_candidates=256)
    second = formula_param_candidates(65, search_space, max_grid_candidates=256)
    assert first == second
    assert len(first) == grid_candidate_count(search_space) == 27
    assert formula_param_gid(65, first[0]) != formula_param_gid(65, first[1])
    window_spec = get_definition(65).parameters["amount_average_window"]
    values = candidate_values(window_spec)
    assert len(values) == 3
    assert values == sorted(values)
    assert values[-1] >= values[0] * 4
    with __import__("pytest").raises(ValueError, match="exceeding max_grid_candidates"):
        formula_param_candidates(65, search_space, max_grid_candidates=8)


def test_rank_ic_alignment_and_one_way_turnover() -> None:
    signals = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-07"])
    rebalances = pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-06", "2024-01-08"])
    columns = ["A", "B", "C"]
    factor = pd.DataFrame([[1.0, 2.0, 3.0]] * 4, index=signals, columns=columns)
    price = pd.DataFrame(
        [
            [100.0, 100.0, 100.0],
            [101.0, 102.0, 103.0],
            [102.01, 104.04, 106.09],
            [103.0301, 106.1208, 109.2727],
        ],
        index=rebalances,
        columns=columns,
    )

    metrics = mining.robust_rank_ic_metrics(
        factor,
        price,
        list(zip(signals, rebalances, strict=True)),
    )

    assert metrics["robust_rank_ic"] == 1.0
    assert metrics["ic_coverage"] == 1.0
    weights = pd.DataFrame({"A": [1.0, 0.5], "B": [-1.0, -0.5], "cash": [0.0, 0.0]})
    assert mining.mean_one_way_turnover(weights) == 0.75


def test_pareto_selection_and_constraint_fallback() -> None:
    config = SimpleNamespace(ic_coverage_min=0.8, max_drawdown_max=0.35, turnover_max=1.0)
    frame = pd.DataFrame([
        {"trial_number": 0, "robust_rank_ic": 0.10, "sharpe": 3.0, "mdd": 0.1, "turnover": 0.2, "ic_coverage": 0.9},
        {"trial_number": 1, "robust_rank_ic": 0.50, "sharpe": 1.0, "mdd": 0.1, "turnover": 0.2, "ic_coverage": 0.9},
        {"trial_number": 2, "robust_rank_ic": 0.40, "sharpe": 2.0, "mdd": 0.1, "turnover": 0.2, "ic_coverage": 0.9},
    ])

    selected, reason = mining_optuna._selection_table(frame, config)

    assert reason == "feasible_pareto"
    assert int(selected.iloc[0]["trial_number"]) == 1

    frame["ic_coverage"] = 0.4
    frame.loc[2, "ic_coverage"] = 0.7
    selected, reason = mining_optuna._selection_table(frame, config)
    assert reason == "minimum_constraint_violation"
    assert int(selected.iloc[0]["trial_number"]) == 2


def test_sqlite_storage_lock_and_stale_trial_recovery(tmp_path) -> None:
    optuna = __import__("pytest").importorskip("optuna")
    logger = logging.getLogger("test-alpha101-optuna")
    lock_path = tmp_path / ".study.lock"
    owner = mining_optuna._acquire_lock(lock_path, "hash")
    with __import__("pytest").raises(RuntimeError, match="active study coordinator"):
        mining_optuna._acquire_lock(lock_path, "hash")

    storage, database_path, _url = mining_optuna._sqlite_storage(optuna, tmp_path, None, logger)
    config = SimpleNamespace(
        sampler="grid",
        grid={"x": [1]},
        random_seed=1,
        config_hash="hash",
        n_trials=1,
        paper_params={"x": 1},
    )
    study = mining_optuna._make_study(optuna, config, storage, "stale", logger)
    trial = study.ask()
    trial.suggest_categorical("x", [1])
    remaining = mining_optuna._recover_and_queue(
        optuna,
        study,
        config,
        logger,
        tmp_path / "events.jsonl",
    )

    assert remaining == 1
    assert any(item.state == optuna.trial.TrialState.FAIL for item in study.trials)
    assert any(item.state == optuna.trial.TrialState.WAITING for item in study.trials)
    interrupted = study.ask()
    interrupted.suggest_categorical("x", [1])
    marked = mining_optuna._mark_running_trials_failed(
        optuna,
        storage,
        logger,
        tmp_path / "events.jsonl",
    )
    assert marked == 1
    assert not study.get_trials(states=(optuna.trial.TrialState.RUNNING,))
    assert "INTERRUPTED_TRIAL" in (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    with storage.engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
        assert connection.exec_driver_sql("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.exec_driver_sql("PRAGMA busy_timeout").fetchone()[0] == 30000
    mining_optuna._release_lock(lock_path, owner)
    storage.remove_session()
    storage.engine.dispose()
    assert not lock_path.exists()


def _optuna_test_config(tmp_path):
    return SimpleNamespace(
        output_dir=tmp_path / "ALPHA1",
        cache_dir=tmp_path / "cache",
        grid={"alpha_id": [1], "x": [1, 2], "n_quantiles": [10]},
        train=("2020-01-01", "2020-01-05"),
        test=("2020-01-01", "2020-01-05"),
        valid=("2020-02-01", "2020-02-05"),
        paired_schemes=[(2, 1, 1)],
        rolling_span=("2020-01-01", "2020-01-10"),
        max_windows_per_scheme=None,
        factor_module="test_factor",
        spec_factory="make_mining_spec",
        gid_factory=None,
        weight_hook=None,
        valid_report_hook=None,
        warmup_days_factory=None,
        engine="vector",
        workers=1,
        trial_batch_size=None,
        n_quantiles_param="n_quantiles",
        initial_amount=1e6,
        time_tolerance=24,
        max_memory_ratio=0.5,
        max_memory_bytes=None,
        max_warmup_days=100,
        rebuild_cache=False,
        rebal_freq="D",
        universe=None,
        sampler="grid",
        paper_params={"alpha_id": 1, "x": 1, "n_quantiles": 10},
        n_trials=96,
        max_grid_candidates=256,
        random_seed=1,
        ic_coverage_min=0.8,
        max_drawdown_max=0.35,
        turnover_max=1.0,
        config_hash="integration-grid",
        log_level="WARNING",
        storage_url=None,
        config_path="walkforward.yaml",
        report_top_n=3,
    )


def test_optuna_grid_walk_forward_writes_auditable_outputs(monkeypatch, tmp_path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("optuna")
    config = _optuna_test_config(tmp_path)
    monkeypatch.setattr(mining_optuna.core, "_resolve_gid", lambda *args: f"gid_{args[-1]['x']}")
    monkeypatch.setattr(mining_optuna.core, "_warmup_days", lambda *args: 30)
    monkeypatch.setattr(
        mining_optuna.core,
        "build_cache_for_config",
        lambda *args, **kwargs: SimpleNamespace(data=tmp_path / "cache.pkl", pit=tmp_path / "pit.pkl"),
    )
    monkeypatch.setattr(mining_optuna.core, "_effective_workers_for_memory", lambda *args: 1)
    monkeypatch.setattr(
        mining_optuna.core,
        "gen_rolling_train_test_windows",
        lambda *args, **kwargs: [
            ("2020-01-01", "2020-01-03", "2020-01-04", "2020-01-05"),
            ("2020-01-02", "2020-01-04", "2020-01-05", "2020-01-06"),
        ],
    )

    def fake_run_tasks(_config, tasks, _cache_paths, *, executor=None):
        rows = []
        for task in tasks:
            x = float(task["params"]["x"])
            rows.append({
                **task["params"], "trial_number": task.get("trial_number", -1), "gid": task["gid"],
                "phase": task["phase"], "scheme": task["scheme"], "win_start": task["win_start"],
                "win_end": task["win_end"], "robust_rank_ic": x, "sharpe": 1.0 + x,
                "mdd": 0.1, "turnover": 0.2, "ic_coverage": 1.0, "error": "",
            })
        return pd.DataFrame(rows)

    monkeypatch.setattr(mining_optuna.core, "run_tasks", fake_run_tasks)
    result = mining_optuna.run_optuna_walk_forward(config)

    assert not (config.output_dir / ".study.lock").exists()
    for relative in (
        "study.sqlite3", "study.snapshot.sqlite3", "run.log", "audit/events.jsonl",
        "audit/trials.partial.csv", "audit/pareto_front.partial.csv", "audit/oos_parameter_path.partial.csv",
        "trials.csv", "pareto_front.csv", "cv_results.csv", "oos_parameter_path.csv", "final_candidates.yaml",
    ):
        assert (config.output_dir / relative).exists(), relative
    assert len(result["train_results"]) == 4
    assert set(result["oos_parameter_path"]["params_json"]) == {'{"alpha_id":1,"n_quantiles":10,"x":2}'}


def test_database_lock_retry_and_keyboard_interrupt_cleanup(monkeypatch, tmp_path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("optuna")
    attempts = []
    sleeps = []

    def locked_then_ready():
        attempts.append(1)
        if len(attempts) <= 3:
            raise RuntimeError("database is locked")
        return "ready"

    monkeypatch.setattr(mining_optuna.time, "sleep", sleeps.append)
    assert mining_optuna._db_retry(locked_then_ready, logging.getLogger("retry"), "test") == "ready"
    assert sleeps == [1.0, 2.0, 4.0]

    config = _optuna_test_config(tmp_path)
    monkeypatch.setattr(mining_optuna.core, "_resolve_gid", lambda *args: f"gid_{args[-1]['x']}")
    monkeypatch.setattr(
        mining_optuna.core,
        "build_cache_for_config",
        lambda *args, **kwargs: SimpleNamespace(data=tmp_path / "cache.pkl", pit=tmp_path / "pit.pkl"),
    )
    monkeypatch.setattr(mining_optuna.core, "_effective_workers_for_memory", lambda *args: 1)
    monkeypatch.setattr(
        mining_optuna.core,
        "gen_rolling_train_test_windows",
        lambda *args, **kwargs: [("2020-01-01", "2020-01-03", "2020-01-04", "2020-01-05")],
    )
    monkeypatch.setattr(
        mining_optuna,
        "_run_study",
        lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(SystemExit) as stopped:
        mining_optuna.run_optuna_walk_forward(config)

    assert stopped.value.code == 130
    assert not (config.output_dir / ".study.lock").exists()
    status = __import__("json").loads((config.output_dir / "audit" / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "interrupted"
    assert status["exit_reason"] == "keyboard_interrupt"
    assert (config.output_dir / "study.snapshot.sqlite3").exists()
    assert (config.output_dir / "audit" / "trials.partial.csv").exists()
    assert (config.output_dir / "audit" / "pareto_front.partial.csv").exists()
    assert (config.output_dir / "audit" / "oos_parameter_path.partial.csv").exists()


def test_paired_rolling_windows_are_contiguous(monkeypatch) -> None:
    days = list(pd.bdate_range("2024-01-01", periods=12).date)
    monkeypatch.setattr("betalens.datafeed.get_absolute_trade_days", lambda *args, **kwargs: days)

    pairs = mining.gen_rolling_train_test_windows(
        "2024-01-01", "2024-01-31", train_len=4, test_len=2, step=3
    )

    assert pairs[0] == ("2024-01-01", "2024-01-04", "2024-01-05", "2024-01-08")
    assert pairs[1][0] == "2024-01-04"
    assert pairs[1][2] == "2024-01-10"


def test_align_daily_wides_uses_latest_timestamp_per_day() -> None:
    days = pd.bdate_range("2024-01-01", periods=2)
    open_index = pd.DatetimeIndex([days[0] + pd.Timedelta(hours=9, minutes=30), days[1] + pd.Timedelta(hours=9, minutes=30)])
    close_index = pd.DatetimeIndex([days[0] + pd.Timedelta(hours=15), days[1] + pd.Timedelta(hours=15)])
    opens = pd.DataFrame([[1.0], [2.0]], index=open_index, columns=["A"])
    closes = pd.DataFrame([[1.1], [2.1]], index=close_index, columns=["A"])

    aligned = mining.align_daily_wides({"open": opens, "close": closes})

    expected = close_index
    pd.testing.assert_index_equal(aligned["open"].index, expected)
    pd.testing.assert_index_equal(aligned["close"].index, expected)
    assert aligned["open"].iloc[:, 0].tolist() == [1.0, 2.0]


def test_core_cache_masks_market_and_industry_inputs(monkeypatch, tmp_path) -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    columns = ["A", "B"]
    market = pd.DataFrame(np.arange(8, dtype=float).reshape(4, 2), index=index, columns=columns)
    industry = pd.DataFrame([["x", "y"]] * 4, index=index, columns=columns)
    pit_days = [ts.date() for ts in index]
    spec = SimpleNamespace(
        inputs={"close_wide": "收盘价(元)"},
        industry_inputs={"sector_wide": "申万一级行业"},
        index_code="000906.SH",
        mask_inputs_by_pit=True,
        table_name="daily_market",
        industry_scheme="申万一级行业",
        use_industry=True,
        backtest_metric="收盘价(元)",
    )
    config = SimpleNamespace(
        cache_dir=tmp_path,
        output_dir=tmp_path,
        rebuild_cache=True,
        max_warmup_days=2,
        universe=None,
        factor_module="alpha101_mining",
        spec_factory="make_mining_spec",
    )

    monkeypatch.setattr(
        "betalens.datafeed.get_absolute_trade_days",
        lambda *args, **kwargs: pit_days,
    )
    monkeypatch.setattr(mining, "_load_spec", lambda *args, **kwargs: spec)
    monkeypatch.setattr(
        mining,
        "build_pit_universe",
        lambda days, index_code: {day: {"A"} for day in days},
    )
    monkeypatch.setattr(mining, "fetch_daily_wide", lambda *args, **kwargs: market.copy())
    monkeypatch.setattr(mining, "fetch_industry_wide", lambda *args, **kwargs: industry.copy())

    paths = mining.build_cache_for_config(config, [("2024-01-01", "2024-01-04")], {})
    payload = pickle.loads(paths.data.read_bytes())
    assert payload["inputs"]["close_wide"]["B"].isna().all()
    assert payload["inputs"]["sector_wide"]["B"].isna().all()
    assert payload["price"]["B"].notna().all()
    assert "申万一级行业" in payload["industry_by_scheme"]

    config.rebuild_cache = False
    monkeypatch.setattr(mining, "fetch_daily_wide", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    mining.build_cache_for_config(config, [("2024-01-01", "2024-01-04")], {})
