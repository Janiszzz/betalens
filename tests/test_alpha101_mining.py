from __future__ import annotations

from types import SimpleNamespace
from concurrent.futures import Future, ProcessPoolExecutor
import logging
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
from betalens.factor import mining_cache  # noqa: E402


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


def test_pareto_selection_excludes_pruned_stage_metrics() -> None:
    config = SimpleNamespace(ic_coverage_min=0.8, max_drawdown_max=0.35, turnover_max=1.0)
    frame = pd.DataFrame([
        {
            "trial_number": 0, "state": "PRUNED", "robust_rank_ic": 0.99,
            "sharpe": 9.0, "mdd": 0.01, "turnover": 0.01, "ic_coverage": 1.0,
        },
        {
            "trial_number": 1, "state": "COMPLETE", "robust_rank_ic": 0.10,
            "sharpe": 1.0, "mdd": 0.10, "turnover": 0.20, "ic_coverage": 0.9,
        },
    ])

    selected, reason = mining_optuna._selection_table(frame, config)

    assert reason == "feasible_pareto"
    assert selected["trial_number"].tolist() == [1]


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
        assert connection.exec_driver_sql("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.exec_driver_sql("PRAGMA synchronous").fetchone()[0] == 1
        assert connection.exec_driver_sql("PRAGMA busy_timeout").fetchone()[0] == 30000
    mining_optuna._release_lock(lock_path, owner)
    storage.remove_session()
    storage.engine.dispose()
    assert not lock_path.exists()


def test_pruning_restart_requeues_only_survivors_without_duplicates(tmp_path) -> None:
    optuna = __import__("pytest").importorskip("optuna")
    logger = logging.getLogger("test-alpha101-pruning-restart")
    storage, _database_path, _url = mining_optuna._sqlite_storage(optuna, tmp_path, None, logger)
    config = SimpleNamespace(
        sampler="grid", grid={"x": [1, 2, 3, 4]}, random_seed=1,
        config_hash="pruning-restart", search_hash="pruning-restart", n_trials=4,
        paper_params={"x": 1}, pruning_enabled=True, pruning_stages=(0.25, 0.5, 1.0),
        pruning_reduction_factor=2, pruning_min_full_candidates=2,
        pruning_keep_paper=True,
    )
    study = mining_optuna._make_study(optuna, config, storage, "pruning-restart", logger)
    trials = []
    for x in range(1, 5):
        study.enqueue_trial({"x": x})
        trial = study.ask()
        trial.suggest_categorical("x", [1, 2, 3, 4])
        trial.set_user_attr("stage_results", [{"stage_index": 0, "stage_fraction": 0.25}])
        trials.append(trial)

    for trial in trials[2:]:
        trial.set_user_attr("pruning_reason", "successive_halving")
        mining_optuna._study_tell(study, trial, state=optuna.trial.TrialState.PRUNED)
    survivor_keys = [mining_optuna._parameter_key({"x": 1}), mining_optuna._parameter_key({"x": 2})]
    study.set_user_attr("active_candidate_keys", survivor_keys)
    study.set_user_attr("current_stage_index", 1)

    first_remaining = mining_optuna._recover_and_queue(
        optuna, study, config, logger, tmp_path / "events.jsonl"
    )
    second_remaining = mining_optuna._recover_and_queue(
        optuna, study, config, logger, tmp_path / "events.jsonl"
    )

    waiting = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.WAITING,))
    pruned = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,))
    stale = [
        trial for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.FAIL,))
        if trial.user_attrs.get("failure_reason") == "stale_after_restart"
    ]
    assert first_remaining == second_remaining == 2
    assert {
        mining_optuna._parameter_key(
            trial.params or dict(trial.system_attrs.get("fixed_params") or {})
        )
        for trial in waiting
    } == set(survivor_keys)
    assert len(waiting) == 2
    assert {trial.params["x"] for trial in pruned} == {3, 4}
    assert {trial.params["x"] for trial in stale} == {1, 2}
    assert all(trial.user_attrs["stage_results"][0]["stage_index"] == 0 for trial in stale)
    assert {trial.user_attrs["recovered_from_trial"] for trial in waiting} == {0, 1}
    assert all(trial.user_attrs["stage_results"][0]["stage_index"] == 0 for trial in waiting)
    storage.remove_session()
    storage.engine.dispose()


def test_study_name_changes_with_data_search_hash() -> None:
    first = mining_optuna._study_name(
        "a" * 64, "train", "paired/252/63/21", "2020-01-01", "2020-12-31"
    )
    second = mining_optuna._study_name(
        "b" * 64, "train", "paired/252/63/21", "2020-01-01", "2020-12-31"
    )
    assert first != second


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
        pruning_enabled=False,
        pruning_stages=(1.0,),
        pruning_reduction_factor=3,
        pruning_min_full_candidates=3,
        pruning_keep_paper=True,
        config_hash="integration-grid",
        log_level="WARNING",
        storage_url=None,
        config_path="walkforward.yaml",
        report_top_n=3,
        runtime_root=tmp_path / "runtime",
        min_workers=1,
        max_workers=1,
        max_active_windows=2,
        max_inflight_batches=1,
        grid_batch_max_candidates=8,
        grid_batch_target_seconds=10,
        grid_prefetch_batches=2,
        storage_queue_max_batches=24,
        audit_queue_max_events=1000,
        console_trial_events=False,
        scheduler_schema_version=2,
        resource_check_seconds=1,
        resource_wait_minutes=0,
        memory_low_watermark_ratio=0.0,
        memory_resume_watermark_ratio=0.0,
        audit_mirror_seconds=60,
        partial_refresh_seconds=1,
        partial_refresh_trials=1,
        snapshot_seconds=3600,
        blas_threads=1,
        search_hash="integration-grid",
        runtime_hash="runtime-test",
    )


def test_optuna_grid_walk_forward_writes_auditable_outputs(monkeypatch, tmp_path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("optuna")
    config = _optuna_test_config(tmp_path)
    config.scheduler_schema_version = 3
    config.max_persist_overlap_batches = 1
    config.sqlite_batch_transactions = True
    config.scheduler_wait_seconds = 0.01
    config.audit_incremental = True
    monkeypatch.setattr(mining_optuna.core, "_resolve_gid", lambda *args: f"gid_{args[-1]['x']}")
    monkeypatch.setattr(mining_optuna.core, "_warmup_days", lambda *args: 30)
    fake_manifest = tmp_path / "manifest.json"
    fake_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        mining_optuna.core,
        "build_cache_for_config",
        lambda *args, **kwargs: SimpleNamespace(data=fake_manifest, pit=fake_manifest),
    )
    monkeypatch.setattr(mining_optuna, "_wait_for_worker_capacity", lambda *args: 1)
    monkeypatch.setattr(
        mining_optuna.core,
        "gen_rolling_train_test_windows",
        lambda *args, **kwargs: [
            ("2020-01-01", "2020-01-03", "2020-01-04", "2020-01-05"),
            ("2020-01-02", "2020-01-04", "2020-01-05", "2020-01-06"),
        ],
    )

    submitted = []

    class ImmediateExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, _fn, task):
            if isinstance(task, list):
                future = Future()
                values = []
                for item in task:
                    submitted.append(item.copy())
                    x = float(item["params"]["x"])
                    values.append({
                        **item["params"], "trial_number": item.get("trial_number", -1),
                        "gid": item["gid"], "phase": item["phase"], "scheme": item["scheme"],
                        "win_start": item["win_start"], "win_end": item["win_end"],
                        "robust_rank_ic": x, "sharpe": 1.0 + x, "mdd": 0.1,
                        "turnover": 0.2, "ic_coverage": 1.0, "error": "",
                    })
                future.set_result(values)
                return future
            submitted.append(task.copy())
            x = float(task["params"]["x"])
            future = Future()
            future.set_result({
                **task["params"], "trial_number": task.get("trial_number", -1), "gid": task["gid"],
                "phase": task["phase"], "scheme": task["scheme"], "win_start": task["win_start"],
                "win_end": task["win_end"], "robust_rank_ic": x, "sharpe": 1.0 + x,
                "mdd": 0.1, "turnover": 0.2, "ic_coverage": 1.0, "error": "",
            })
            return future

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(mining_optuna, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(mining_optuna.core, "run_tasks", lambda *_args, **_kwargs: pd.DataFrame([{
        "robust_rank_ic": 1.0, "sharpe": 2.0, "mdd": 0.1, "turnover": 0.2,
        "ic_coverage": 1.0, "error": "",
    }]))
    result = mining_optuna.run_optuna_walk_forward(config)

    assert not (config.output_dir / ".study.lock").exists()
    for relative in (
        "study.snapshot.sqlite3", "run.log", "audit/events.jsonl",
        "audit/trials.partial.csv", "audit/pareto_front.partial.csv", "audit/oos_parameter_path.partial.csv",
        "trials.csv", "pareto_front.csv", "cv_results.csv", "oos_parameter_path.csv", "final_candidates.yaml",
    ):
        assert (config.output_dir / relative).exists(), relative
    assert len(result["train_results"]) == 4
    assert set(result["oos_parameter_path"]["params_json"]) == {'{"alpha_id":1,"n_quantiles":10,"x":2}'}
    train_window_ids = [task["window_id"] for task in submitted if task["phase"] == "train"]
    assert len(set(train_window_ids)) == 2


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
    fake_manifest = tmp_path / "manifest.json"
    fake_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        mining_optuna.core,
        "build_cache_for_config",
        lambda *args, **kwargs: SimpleNamespace(data=fake_manifest, pit=fake_manifest),
    )
    monkeypatch.setattr(mining_optuna, "_wait_for_worker_capacity", lambda *args: 1)
    monkeypatch.setattr(
        mining_optuna.core,
        "gen_rolling_train_test_windows",
        lambda *args, **kwargs: [("2020-01-01", "2020-01-03", "2020-01-04", "2020-01-05")],
    )
    monkeypatch.setattr(
        mining_optuna,
        "_run_multiwindow_scheduler",
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


def test_grid_batch_sqlite_transaction_updates_all_trials_once(tmp_path) -> None:
    pytest = __import__("pytest")
    optuna = pytest.importorskip("optuna")
    storage, _database, _url = mining_optuna._sqlite_storage(
        optuna, tmp_path / "runtime", None, logging.getLogger("batch-sqlite"),
    )
    study = optuna.create_study(
        study_name="batch",
        storage=storage,
        sampler=optuna.samplers.GridSampler({"x": [1, 2]}),
        directions=("maximize", "maximize"),
        load_if_exists=True,
    )
    trials = [
        mining_optuna._grid_trial(optuna, study, {"x": [1, 2]}, {"x": value}, index)
        for index, value in enumerate((1, 2))
    ]
    config = SimpleNamespace(
        sampler="grid", sqlite_batch_transactions=True,
        ic_coverage_min=0.8, max_drawdown_max=0.35, turnover_max=1.0,
    )
    records = []
    for trial in trials:
        task = {
            "params": dict(trial.params), "gid": "gid", "phase": "train",
            "scheme": "paired/1/1/1", "win_start": "2020-01-01",
            "win_end": "2020-01-02", "train_start": "2020-01-01",
            "train_end": "2020-01-02", "test_start": "2020-01-03",
            "test_end": "2020-01-04", "trial_number": trial.number,
            "study_name": "batch", "window_id": "w", "candidate_order": trial.number,
            "stage_index": 0, "stage_fraction": 1.0,
        }
        records.append({
            "trial": trial, "trial_number": trial.number, "params": dict(trial.params),
            "candidate_order": trial.number, "task": task,
            "result": {
                "robust_rank_ic": float(trial.params["x"]), "sharpe": 1.0,
                "mdd": 0.1, "turnover": 0.2, "ic_coverage": 1.0, "error": "",
            },
        })
    commits = []
    from sqlalchemy import event
    engine = mining_optuna._rdb_storage(storage).engine
    mutations, durable = mining_optuna._prepare_result_mutations(
        optuna, config, study, records, last_stage=True,
    )
    def on_commit(*_args):
        commits.append(True)
    event.listen(engine, "commit", on_commit)
    mining_optuna._sqlite_apply_trial_mutations(study._storage, mutations)
    event.remove(engine, "commit", on_commit)
    assert len(durable) == 2
    assert len(commits) == 1
    assert [trial.state.name for trial in study.trials] == ["COMPLETE", "COMPLETE"]
    assert [list(trial.values) for trial in study.trials] == [[1.0, 1.0], [2.0, 1.0]]


def test_incremental_audit_index_refreshes_without_database_scan(tmp_path) -> None:
    logger = logging.getLogger("incremental-audit")
    initial = pd.DataFrame([{
        "study_name": "s", "trial_number": 0, "state": "RUNNING",
        "values": "[]", "params_json": '{"x":1}', "candidate_id": "c0",
        "candidate_order": 0, "stage_results": "[]", "pruning_reason": "",
        "x": 1,
    }])
    requests, thread, errors = mining_optuna._start_audit_worker(
        logger, tmp_path / "audit", 100, initial,
    )
    row = dict(initial.iloc[0])
    row.update({"state": "COMPLETE", "values": "[1.0, 2.0]", "stage_index": 0})
    mining_optuna._audit_trial_rows(requests, [row])
    mining_optuna._audit_refresh_incremental(requests, [], [], {"status": "running"})
    for _ in range(50):
        if (tmp_path / "audit" / "trials.partial.csv").exists():
            break
        __import__("time").sleep(0.01)
    mining_optuna._stop_audit_worker(requests, thread)
    assert not errors
    result = pd.read_csv(tmp_path / "audit" / "trials.partial.csv")
    assert result.loc[0, "state"] == "COMPLETE"
    assert result.loc[0, "values"] == "[1.0, 2.0]"


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
    monkeypatch.setattr(
        mining,
        "fetch_trade_status_wide",
        lambda universe, dates: pd.DataFrame(1, index=pd.DatetimeIndex(dates), columns=universe, dtype=np.int8),
    )

    paths = mining.build_cache_for_config(config, [("2024-01-01", "2024-01-04")], {})
    cache = __import__("betalens.factor.mining_cache", fromlist=["open_manifest"])
    payload = cache.open_manifest(paths.data)
    close = cache.frame(payload["inputs"]["close_wide"])
    sector = cache.frame(payload["inputs"]["sector_wide"])
    price = cache.frame(payload["price"])
    assert close["B"].isna().all()
    assert sector["B"].isna().all()
    assert price["B"].notna().all()
    assert "申万一级行业" in payload["industry_by_scheme"]

    config.rebuild_cache = False
    monkeypatch.setattr(mining, "fetch_daily_wide", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    mining.build_cache_for_config(config, [("2024-01-01", "2024-01-04")], {})


def test_memmap_cache_roundtrip_and_window_slice(tmp_path) -> None:
    index = pd.date_range("2024-01-01", periods=20, freq="D")
    columns = ["A", "B", "C"]
    numeric = pd.DataFrame(np.arange(60, dtype=float).reshape(20, 3), index=index, columns=columns)
    industry = pd.DataFrame([["x", "y", None]] * 20, index=index, columns=columns)
    manifest = mining_cache.publish(
        tmp_path,
        "roundtrip",
        inputs={"close": numeric},
        price=numeric,
        execution_price=numeric,
        trade_status=pd.DataFrame(1, index=index, columns=columns, dtype=np.int8),
        industry_by_scheme={"industry": industry},
        pit={day.date(): {"A", "B"} for day in index},
        universe=columns,
        metadata={},
    )

    opened = mining_cache.open_manifest(manifest)
    assert isinstance(opened["price"]["values"], np.memmap)
    assert opened["price"]["dates"] is not opened["inputs"]["close"]["dates"]
    assert opened["price"]["dates"].filename == opened["inputs"]["close"]["dates"].filename
    assert opened["price"]["columns"] == opened["inputs"]["close"]["columns"]
    sliced = mining_cache.frame(opened["inputs"]["close"], "2024-01-05", "2024-01-09", ["B"])
    pd.testing.assert_frame_equal(
        sliced, numeric.loc["2024-01-05":"2024-01-09", ["B"]], check_freq=False
    )
    decoded = mining_cache.frame(opened["industry_by_scheme"]["industry"])
    assert decoded.iloc[0, :2].tolist() == ["x", "y"]
    assert pd.isna(decoded.iloc[0, 2])


def test_memmap_cache_maps_in_spawned_worker(tmp_path) -> None:
    index = pd.bdate_range("2024-01-01", periods=80)
    columns = ["A", "B", "C"]
    numeric = pd.DataFrame(
        np.arange(240, dtype=float).reshape(80, 3) + 10.0,
        index=index,
        columns=columns,
    )
    manifest = mining_cache.publish(
        tmp_path,
        "process-smoke",
        inputs={"close": numeric},
        price=numeric,
        execution_price=numeric,
        trade_status=pd.DataFrame(1, index=index, columns=columns, dtype=np.int8),
        industry_by_scheme={},
        pit={day.date(): set(columns) for day in index},
        universe=columns,
        metadata={},
    )

    with ProcessPoolExecutor(max_workers=1) as executor:
        private_bytes = executor.submit(
            mining._worker_private_bytes_probe,
            str(manifest),
            str(manifest),
        ).result(timeout=30)

    assert private_bytes > 0


def test_worker_slot_lock_refuses_second_coordinator(tmp_path) -> None:
    config = SimpleNamespace(
        runtime_root=tmp_path,
        resource_check_seconds=1,
        resource_wait_minutes=0,
    )
    owner = mining_optuna._acquire_worker_slots(config, 12, logging.getLogger("slot"))
    try:
        with __import__("pytest").raises(RuntimeError, match="worker slots remained occupied"):
            mining_optuna._acquire_worker_slots(config, 10, logging.getLogger("slot"))
    finally:
        mining_optuna._release_worker_slots(owner)
    assert not (tmp_path / ".worker-slots.lock").exists()


def test_resource_capacity_honors_physical_and_commit_reserves(monkeypatch, tmp_path) -> None:
    config = SimpleNamespace(workers=12, min_workers=10, max_workers=12)
    cache_paths = SimpleNamespace(data=tmp_path / "manifest.json")
    gib = 1024**3
    monkeypatch.setattr(mining_optuna.core, "_estimate_worker_memory_bytes", lambda *args: gib)
    monkeypatch.setattr(
        mining_optuna.core,
        "_system_resource_snapshot",
        lambda: {
            "physical_total": 64 * gib,
            "physical_available": 32 * gib,
            "commit_total": 80 * gib,
            "commit_available": 26 * gib,
        },
    )

    capacity, diagnostics = mining_optuna._resource_capacity(config, cache_paths)

    assert capacity == 11
    assert diagnostics["minimum_workers"] == 10


def test_resource_capacity_uses_measured_worker_private_bytes(monkeypatch, tmp_path) -> None:
    config = SimpleNamespace(workers=12, min_workers=10, max_workers=12)
    cache_paths = SimpleNamespace(data=tmp_path / "manifest.json")
    gib = 1024**3
    monkeypatch.setattr(mining_optuna.core, "_estimate_worker_memory_bytes", lambda *args: gib)
    monkeypatch.setattr(
        mining_optuna.core,
        "_system_resource_snapshot",
        lambda: {
            "physical_total": 64 * gib,
            "physical_available": 32 * gib,
            "commit_total": 80 * gib,
            "commit_available": 50 * gib,
        },
    )

    capacity, diagnostics = mining_optuna._resource_capacity(
        config, cache_paths, measured_private_bytes=2 * gib
    )

    assert capacity == 9
    assert diagnostics["per_worker"] == int(2.5 * gib)


def test_multiwindow_scheduler_fills_twelve_distinct_windows(monkeypatch, tmp_path) -> None:
    optuna = __import__("pytest").importorskip("optuna")
    logger = logging.getLogger("multiwindow")
    storage, _database_path, _url = mining_optuna._sqlite_storage(optuna, tmp_path, None, logger)
    config = SimpleNamespace(
        sampler="grid", grid={"x": [1]}, random_seed=1, config_hash="multiwindow",
        search_hash="multiwindow", n_trials=1, paper_params={"x": 1},
        factor_module="test_factor", gid_factory=None, spec_factory="make_mining_spec",
        weight_hook=None, warmup_days_factory=None, engine="vector", rebal_freq="D",
        n_quantiles_param="n_quantiles", initial_amount=1e6, time_tolerance=24,
        ic_coverage_min=0.8, max_drawdown_max=0.35, turnover_max=1.0,
        workers=12, max_active_windows=12, max_inflight_batches=12,
        grid_batch_max_candidates=8, grid_batch_target_seconds=10,
        grid_prefetch_batches=24, storage_queue_max_batches=24, audit_queue_max_events=1000,
        partial_refresh_seconds=9999, partial_refresh_trials=9999, snapshot_seconds=9999,
        memory_low_watermark_ratio=0.0, memory_resume_watermark_ratio=0.0,
    )
    monkeypatch.setattr(mining_optuna.core, "_resolve_gid", lambda *args: "gid")
    monkeypatch.setattr(mining_optuna.core, "_warmup_days", lambda *args: 1)
    submitted: list[Future] = []
    submitted_tasks: list[dict] = []

    class ControlledExecutor:
        def submit(self, _fn, task):
            future = Future()
            submitted.append(future)
            copied = [item.copy() for item in task] if isinstance(task, list) else task.copy()
            submitted_tasks.append(copied)
            if len(submitted) == 12:
                for current, current_task in zip(submitted, submitted_tasks, strict=True):
                    if not current.done():
                        current.set_result([{
                            **item, "robust_rank_ic": 1.0, "sharpe": 2.0,
                            "mdd": 0.1, "turnover": 0.2,
                            "ic_coverage": 1.0, "error": "",
                        } for item in current_task])
            elif not isinstance(task, list) and task["phase"] == "test":
                future.set_result({
                    **task, "robust_rank_ic": 1.0, "sharpe": 2.0,
                    "mdd": 0.1, "turnover": 0.2,
                    "ic_coverage": 1.0, "error": "",
                })
            return future

    windows = [
        {
            "train_start": f"2024-01-{index + 1:02d}", "train_end": f"2024-01-{index + 1:02d}",
            "test_start": f"2024-02-{index + 1:02d}", "test_end": f"2024-02-{index + 1:02d}",
            "scheme": "paired/1/1/1",
        }
        for index in range(12)
    ]
    train, test, final_ordered = mining_optuna._run_multiwindow_scheduler(
        optuna=optuna, config=config, storage=storage, cache_paths=SimpleNamespace(),
        executor=ControlledExecutor(), logger=logger, audit_dir=tmp_path / "audit",
        windows=windows, pareto_rows=[], candidate_rows=[], oos_rows=[], status={},
        database_path=None, snapshot_path=tmp_path / "snapshot.sqlite3", mirror_callback=lambda: None,
    )

    flat_tasks = [item for value in submitted_tasks for item in (value if isinstance(value, list) else [value])]
    first_train = [task for task in flat_tasks if task["phase"] == "train"][:12]
    assert len({task["window_id"] for task in first_train}) == 12
    train_positions = {
        task["window_id"]: index
        for index, task in enumerate(flat_tasks)
        if task["phase"] == "train"
    }
    test_positions = {
        task["window_id"]: index
        for index, task in enumerate(flat_tasks)
        if task["phase"] == "test"
    }
    assert set(test_positions) == set(train_positions)
    assert all(test_positions[window_id] > train_positions[window_id] for window_id in test_positions)
    for study_name in {frame["study_name"].iloc[0] for frame in train}:
        study = optuna.load_study(study_name=study_name, storage=storage)
        checkpoint = study.user_attrs["betalens"]
        assert checkpoint["locked_trial"] == 0
        assert checkpoint["locked_params"] == {"x": 1}
        assert checkpoint["locked_at"]
    assert len(train) == len(test) == 12
    assert final_ordered is not None
    storage.remove_session()
    storage.engine.dispose()


def test_successive_halving_keeps_paper_and_marks_pruned_trials(monkeypatch, tmp_path) -> None:
    optuna = __import__("pytest").importorskip("optuna")
    logger = logging.getLogger("successive-halving")
    storage, _database_path, _url = mining_optuna._sqlite_storage(optuna, tmp_path, None, logger)
    config = SimpleNamespace(
        sampler="grid", grid={"x": [1, 2, 3, 4, 5, 6]}, random_seed=1,
        config_hash="halving", search_hash="halving", n_trials=6, paper_params={"x": 1},
        factor_module="test_factor", gid_factory=None, spec_factory="make_mining_spec",
        weight_hook=None, warmup_days_factory=None, engine="vector", rebal_freq="D",
        n_quantiles_param="n_quantiles", initial_amount=1e6, time_tolerance=24,
        ic_coverage_min=0.8, max_drawdown_max=0.35, turnover_max=1.0,
        workers=1, max_active_windows=1, max_inflight_batches=1,
        grid_batch_max_candidates=8, grid_batch_target_seconds=10,
        grid_prefetch_batches=1, storage_queue_max_batches=24, audit_queue_max_events=1000,
        partial_refresh_seconds=9999, partial_refresh_trials=9999, snapshot_seconds=9999,
        memory_low_watermark_ratio=0.0, memory_resume_watermark_ratio=0.0,
        pruning_enabled=True, pruning_stages=(0.25, 0.5, 1.0),
        pruning_reduction_factor=2, pruning_min_full_candidates=2,
        pruning_keep_paper=True,
    )
    monkeypatch.setattr(mining_optuna.core, "_resolve_gid", lambda *args: f"gid_{args[-1]['x']}")
    monkeypatch.setattr(mining_optuna.core, "_warmup_days", lambda *args: 1)
    submitted_tasks: list[dict] = []

    class ImmediateExecutor:
        def submit(self, _fn, task):
            if isinstance(task, list):
                submitted_tasks.extend(item.copy() for item in task)
                future = Future()
                future.set_result([{
                    **item, "robust_rank_ic": float(item["params"]["x"]),
                    "sharpe": float(item["params"]["x"]), "mdd": 0.1,
                    "turnover": 0.2, "ic_coverage": 1.0, "error": "",
                } for item in task])
                return future
            submitted_tasks.append(task.copy())
            x = float(task["params"]["x"])
            future = Future()
            future.set_result({
                **task, "robust_rank_ic": x, "sharpe": x,
                "mdd": 0.1, "turnover": 0.2, "ic_coverage": 1.0, "error": "",
            })
            return future

    train, test, final_ordered = mining_optuna._run_multiwindow_scheduler(
        optuna=optuna, config=config, storage=storage, cache_paths=SimpleNamespace(),
        executor=ImmediateExecutor(), logger=logger, audit_dir=tmp_path / "audit",
        windows=[{
            "train_start": "2020-01-01", "train_end": "2020-12-31",
            "test_start": "2021-01-01", "test_end": "2021-03-31",
            "scheme": "paired/252/63/21",
        }],
        pareto_rows=[], candidate_rows=[], oos_rows=[], status={},
        database_path=None, snapshot_path=tmp_path / "snapshot.sqlite3",
        mirror_callback=lambda: None,
    )

    studies = optuna.study.get_all_study_summaries(storage=storage)
    study = optuna.load_study(study_name=studies[0].study_name, storage=storage)
    states = [trial.state.name for trial in study.trials]
    assert states.count("COMPLETE") == 2
    assert states.count("PRUNED") == 4
    assert {trial.params["x"] for trial in study.trials if trial.state.name == "COMPLETE"} == {1, 6}
    assert all(
        len(trial.user_attrs["betalens"]["stage_results"]) == 3
        for trial in study.trials if trial.state.name == "COMPLETE"
    )
    assert all(
        trial.user_attrs["betalens"]["pruning_reason"] == "successive_halving"
        for trial in study.trials if trial.state.name == "PRUNED"
    )
    assert {task["stage_fraction"] for task in submitted_tasks if task["phase"] == "train"} == {0.25, 0.5, 1.0}
    assert len([task for task in submitted_tasks if task["phase"] == "train"]) == 11
    assert len(train) == len(test) == 1
    assert final_ordered is not None
    assert int(final_ordered.iloc[0]["x"]) == 6
    assert "PRUNED" in (tmp_path / "audit" / "events.jsonl").read_text(encoding="utf-8")
    storage.remove_session()
    storage.engine.dispose()


def test_cached_trade_days_drive_signal_to_rebalance_mapping(monkeypatch) -> None:
    days = pd.bdate_range("2024-01-01", periods=10)

    def refuse_database(*_args, **_kwargs):
        raise AssertionError("worker signal dates must not query the database")

    monkeypatch.setattr("betalens.datafeed.get_absolute_trade_days", refuse_database)
    pairs = mining._task_signal_rebalance_pairs(
        "2024-01-03", "2024-01-12", "2024-01-01", "W", days
    )
    assert pairs == [
        (pd.Timestamp("2024-01-04").date(), pd.Timestamp("2024-01-05").date()),
        (pd.Timestamp("2024-01-11").date(), pd.Timestamp("2024-01-12").date()),
    ]

    weights = pd.DataFrame(
        {"A": [1.0, 0.5], "cash": [0.0, 0.5]},
        index=pd.DatetimeIndex(["2024-01-04", "2024-01-11"]),
    )
    executed = mining._weights_on_rebalance_days(weights, pairs)
    pd.testing.assert_index_equal(
        executed.index,
        pd.DatetimeIndex(["2024-01-05 00:10", "2024-01-12 00:10"]),
    )
