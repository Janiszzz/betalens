"""Causal, auditable Optuna orchestration for Betalens factor mining."""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from betalens.factor import mining as core


_DB_RETRY_DELAYS = (1.0, 2.0, 4.0)


def _replace_with_retry(source: Path, target: Path) -> None:
    for delay in (0.0, *_DB_RETRY_DELAYS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if delay == _DB_RETRY_DELAYS[-1]:
                raise
            if delay:
                time.sleep(delay)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    _replace_with_retry(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    _replace_with_retry(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any], *, sync: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        if sync:
            os.fsync(stream.fileno())


def _configure_logger(output_dir: Path, level_name: str) -> logging.Logger:
    logger = logging.getLogger(f"betalens.mining.{hash(output_dir.resolve())}")
    logger.handlers.clear()
    logger.propagate = False
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    terminal = logging.StreamHandler(sys.stdout)
    terminal.setFormatter(formatter)
    terminal.setLevel(level)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(terminal)
    logger.addHandler(file_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _pid_is_active(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            int(pid),
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_lock(path: Path, config_hash: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "started_at": _now(), "config_hash": config_hash}
    while True:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            return payload
        except FileExistsError:
            try:
                holder = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                holder = {"pid": -1, "error": "unreadable lock"}
            if _pid_is_active(int(holder.get("pid", -1))):
                raise RuntimeError(f"active study coordinator refused: lock={path.resolve()} holder={holder}")
            path.unlink(missing_ok=True)


def _release_lock(path: Path, owner: Mapping[str, Any]) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if int(current.get("pid", -1)) == int(owner.get("pid", -2)):
        path.unlink(missing_ok=True)


def _db_retry(action: Callable[[], Any], logger: logging.Logger, label: str) -> Any:
    for attempt, delay in enumerate((*_DB_RETRY_DELAYS, None), 1):
        try:
            return action()
        except Exception as exc:
            if "locked" not in str(exc).lower() or delay is None:
                raise
            logger.warning("DB RETRY action=%s attempt=%d delay=%.0fs error=%s", label, attempt, delay, exc)
            time.sleep(delay)


def _mark_running_trials_failed(
    optuna: Any,
    storage: Any,
    logger: logging.Logger,
    event_path: Path,
) -> int:
    """Close coordinator-owned RUNNING trials before an interrupted exit."""
    if storage is None:
        return 0
    marked = 0
    summaries = _db_retry(
        lambda: optuna.study.get_all_study_summaries(storage=storage),
        logger,
        "list-running",
    )
    for summary in summaries:
        study = _db_retry(
            lambda name=summary.study_name: optuna.load_study(study_name=name, storage=storage),
            logger,
            "load-running-study",
        )
        running = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
        for frozen in running:
            _db_retry(
                lambda trial_id=frozen._trial_id, current=study: current._storage.set_trial_user_attr(
                    trial_id, "failure_reason", "interrupted"
                ),
                logger,
                "mark-interrupted",
            )
            _db_retry(
                lambda number=frozen.number, current=study: _study_tell(
                    current, number, state=optuna.trial.TrialState.FAIL
                ),
                logger,
                "fail-interrupted",
            )
            _append_jsonl(event_path, {
                "time": _now(),
                "event": "INTERRUPTED_TRIAL",
                "study": study.study_name,
                "trial": frozen.number,
                "params": frozen.params,
            })
            marked += 1
    return marked


def _study_tell(study: Any, *args: Any, **kwargs: Any) -> Any:
    """Use ask/tell with GridSampler, whose exhaustion hook calls Study.stop()."""
    previous = bool(getattr(study._thread_local, "in_optimize_loop", False))
    study._thread_local.in_optimize_loop = True
    try:
        return study.tell(*args, **kwargs)
    finally:
        study._thread_local.in_optimize_loop = previous
        study._stop_flag = False


def _sqlite_storage(optuna: Any, output_dir: Path, storage_url: str | None, logger: logging.Logger):
    database_path: Path | None = None
    if storage_url:
        url = str(storage_url)
        if url.startswith("sqlite:///"):
            raw = url.removeprefix("sqlite:///")
            database_path = Path(raw)
            if not database_path.is_absolute():
                database_path = (output_dir / database_path).resolve()
            url = f"sqlite:///{database_path.as_posix()}"
    else:
        database_path = (output_dir / "study.sqlite3").resolve()
        url = f"sqlite:///{database_path.as_posix()}"

    if database_path is not None:
        database_path.parent.mkdir(parents=True, exist_ok=True)

        def configure_sqlite() -> None:
            with closing(sqlite3.connect(database_path, timeout=30)) as connection:
                with connection:
                    connection.execute("PRAGMA journal_mode=DELETE")
                    connection.execute("PRAGMA busy_timeout=30000")
                    connection.execute("PRAGMA synchronous=FULL")

        _db_retry(configure_sqlite, logger, "configure-sqlite")
        storage = _db_retry(
            lambda: optuna.storages.RDBStorage(
                url=url,
                engine_kwargs={
                    "connect_args": {"timeout": 30},
                    "pool_size": 1,
                    # Optuna's schema/version bootstrap checks out two
                    # connections; the live coordinator pool is tightened
                    # to zero overflow immediately after initialization.
                    "max_overflow": 1,
                },
            ),
            logger,
            "open-storage",
        )
        storage.engine.pool._max_overflow = 0
    else:
        storage = _db_retry(
            lambda: optuna.storages.RDBStorage(
                url=url,
                engine_kwargs={"pool_size": 1, "max_overflow": 1},
            ),
            logger,
            "open-storage",
        )
        storage.engine.pool._max_overflow = 0
    return storage, database_path, url


def _snapshot_database(database_path: Path | None, snapshot_path: Path) -> str | None:
    if database_path is None or not database_path.exists():
        return None
    temporary = snapshot_path.with_name(snapshot_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with closing(sqlite3.connect(database_path, timeout=30)) as source:
        with closing(sqlite3.connect(temporary)) as target:
            source.backup(target)
    _replace_with_retry(temporary, snapshot_path)
    return str(snapshot_path.resolve())


def _parameter_key(params: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(params), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _grid_candidates(search_space: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    names = list(search_space)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(search_space[name] for name in names))
    ]


def _constraint_values(result: Mapping[str, Any], config: Any) -> tuple[float, float, float]:
    coverage = float(result.get("ic_coverage", 0.0) or 0.0)
    drawdown = float(result.get("mdd", float("inf")))
    turnover = float(result.get("turnover", float("inf")))
    return (
        float(config.ic_coverage_min) - coverage,
        drawdown - float(config.max_drawdown_max),
        turnover - float(config.turnover_max),
    )


def _normalized_violation(result: Mapping[str, Any], config: Any) -> float:
    values = _constraint_values(result, config)
    return (
        max(0.0, values[0]) / float(config.ic_coverage_min)
        + max(0.0, values[1]) / float(config.max_drawdown_max)
        + max(0.0, values[2]) / float(config.turnover_max)
    )


def _trial_result_frame(study: Any) -> pd.DataFrame:
    rows = []
    for trial in study.trials:
        result = dict(trial.user_attrs.get("result") or {})
        row = {
            "study_name": study.study_name,
            "trial_number": trial.number,
            "state": trial.state.name,
            "values": json.dumps(list(trial.values or ())),
            "params_json": _parameter_key(trial.params),
            **trial.params,
            **result,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _all_trials_frame(optuna: Any, storage: Any) -> pd.DataFrame:
    frames = []
    for summary in optuna.study.get_all_study_summaries(storage=storage):
        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        frames.append(_trial_result_frame(study))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pareto_front(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    values = frame[["robust_rank_ic", "sharpe"]].to_numpy(dtype=float)
    keep = np.ones(len(frame), dtype=bool)
    for index, value in enumerate(values):
        dominates = np.all(values >= value, axis=1) & np.any(values > value, axis=1)
        dominates[index] = False
        if dominates.any():
            keep[index] = False
    return frame.iloc[np.flatnonzero(keep)].copy()


def _selection_table(frame: pd.DataFrame, config: Any) -> tuple[pd.DataFrame, str]:
    if frame.empty:
        raise RuntimeError("study produced no trials")
    valid = frame.copy()
    if "error" in valid:
        valid = valid[valid["error"].isna() | (valid["error"].astype(str) == "")]
    for column in ("robust_rank_ic", "sharpe", "mdd", "turnover", "ic_coverage"):
        valid[column] = pd.to_numeric(valid.get(column), errors="coerce")
    valid = valid[np.isfinite(valid["robust_rank_ic"]) & np.isfinite(valid["sharpe"])]
    if valid.empty:
        raise RuntimeError("study produced no valid multi-objective trials")
    valid["constraint_violation"] = valid.apply(lambda row: _normalized_violation(row, config), axis=1)
    valid["feasible"] = valid["constraint_violation"] <= 1e-12
    if valid["feasible"].any():
        pool = valid[valid["feasible"]].copy()
        reason = "feasible_pareto"
    else:
        minimum = float(valid["constraint_violation"].min())
        pool = valid[np.isclose(valid["constraint_violation"], minimum)].copy()
        reason = "minimum_constraint_violation"
    frontier = _pareto_front(pool)
    threshold = float(frontier["robust_rank_ic"].quantile(0.80))
    frontier["ic_top20"] = frontier["robust_rank_ic"] >= threshold
    ordered = frontier.sort_values(
        ["ic_top20", "sharpe", "mdd", "turnover", "trial_number"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["lock_rank"] = np.arange(1, len(ordered) + 1)
    ordered["selection_reason"] = reason
    return ordered, reason


def _study_name(config_hash: str, phase: str, scheme: str, start: str, end: str) -> str:
    digest = hashlib.sha1(f"{phase}|{scheme}|{start}|{end}".encode("utf-8")).hexdigest()[:12]
    return f"wf_{config_hash[:12]}_{phase}_{digest}"


def _make_study(optuna: Any, config: Any, storage: Any, name: str, logger: logging.Logger):
    sampler_name = str(config.sampler).lower()
    if sampler_name == "grid":
        sampler = optuna.samplers.GridSampler(dict(config.grid), seed=int(config.random_seed))
    elif sampler_name in {"motpe", "tpe"}:
        sampler = optuna.samplers.TPESampler(
            seed=int(config.random_seed),
            multivariate=True,
            constraints_func=lambda trial: trial.user_attrs.get("constraints", (0.0, 0.0, 0.0)),
        )
    else:
        raise ValueError(f"sampler must be grid or motpe, got {config.sampler!r}")
    study = _db_retry(
        lambda: optuna.create_study(
            study_name=name,
            storage=storage,
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
            directions=("maximize", "maximize"),
            load_if_exists=True,
        ),
        logger,
        "create-study",
    )
    _db_retry(
        lambda: study.set_user_attr("objectives", ["robust_rank_ic", "sharpe"]),
        logger,
        "set-study-objectives",
    )
    _db_retry(
        lambda: study.set_user_attr("config_hash", config.config_hash),
        logger,
        "set-study-config-hash",
    )
    return study


def _recover_and_queue(optuna: Any, study: Any, config: Any, logger: logging.Logger, event_path: Path) -> int:
    stale_params = []
    for frozen in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        _db_retry(
            lambda trial_id=frozen._trial_id: study._storage.set_trial_user_attr(
                trial_id, "failure_reason", "stale_after_restart"
            ),
            logger,
            "mark-stale",
        )
        _db_retry(lambda number=frozen.number: _study_tell(study, number, state=optuna.trial.TrialState.FAIL), logger, "fail-stale")
        stale_params.append(dict(frozen.params))
        _append_jsonl(event_path, {
            "time": _now(), "event": "STALE_AFTER_RESTART", "study": study.study_name,
            "trial": frozen.number, "params": frozen.params,
        })
    for params in stale_params:
        _db_retry(lambda values=params: study.enqueue_trial(values), logger, "enqueue-stale")

    terminal = [
        trial for trial in study.trials
        if trial.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL}
        and trial.user_attrs.get("failure_reason") != "stale_after_restart"
    ]
    if str(config.sampler).lower() == "grid":
        candidates = _grid_candidates(config.grid)
        terminal_keys = {_parameter_key(trial.params) for trial in terminal}
        waiting_keys = {
            _parameter_key(trial.params)
            for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.WAITING,))
        }
        missing = [params for params in candidates if _parameter_key(params) not in terminal_keys]
        for params in missing:
            key = _parameter_key(params)
            if key not in waiting_keys:
                _db_retry(lambda values=params: study.enqueue_trial(values), logger, "enqueue-grid")
                waiting_keys.add(key)
        return len(missing)

    if not study.trials:
        defaults = dict(config.paper_params or {name: values[0] for name, values in config.grid.items()})
        _db_retry(lambda: study.enqueue_trial(defaults), logger, "enqueue-paper-default")
    return max(0, int(config.n_trials) - len(terminal))


def _task_dict(config: Any, params: Mapping[str, Any], gid: str, **fields: Any) -> dict[str, Any]:
    task = {
        "params": dict(params),
        "gid": gid,
        "factor_module": config.factor_module,
        "spec_factory": config.spec_factory,
        "weight_hook": config.weight_hook,
        "warmup_days": core._warmup_days(config, params),
        "engine": config.engine,
        "rebal_freq": config.rebal_freq,
        "n_quantiles_param": config.n_quantiles_param,
        "initial_amount": config.initial_amount,
        "time_tolerance": config.time_tolerance,
        **fields,
    }
    return task


def _log_trial(logger: logging.Logger, event: str, result: Mapping[str, Any]) -> None:
    params = dict(result.get("params") or {})
    logger.info(
        "%s alpha=%s phase=%s scheme=%s train=%s~%s test=%s~%s trial=%s gid=%s "
        "params=%s IC=%s Sharpe=%s MDD=%s turnover=%s coverage=%s violation=%s feasible=%s "
        "elapsed=%ss error=%s log=%s",
        event,
        result.get("alpha_id", params.get("alpha_id")), result.get("phase"), result.get("scheme"),
        result.get("train_start", result.get("win_start")), result.get("train_end", result.get("win_end")),
        result.get("test_start", ""), result.get("test_end", ""), result.get("trial_number", ""),
        result.get("gid", ""), _parameter_key(params),
        result.get("robust_rank_ic", ""), result.get("sharpe", ""), result.get("mdd", ""),
        result.get("turnover", ""), result.get("ic_coverage", ""), result.get("constraint_violation", ""),
        result.get("feasible", ""), result.get("elapsed_seconds", ""), result.get("error", ""),
        result.get("task_log_path", ""),
    )


def _refresh_audit(
    optuna: Any,
    storage: Any,
    audit_dir: Path,
    pareto_rows: list[dict[str, Any]],
    status: Mapping[str, Any],
) -> None:
    _atomic_csv(audit_dir / "trials.partial.csv", _all_trials_frame(optuna, storage))
    _atomic_csv(audit_dir / "pareto_front.partial.csv", pd.DataFrame(pareto_rows))
    _atomic_json(audit_dir / "status.json", status)


def _ask_params(study: Any, search_space: Mapping[str, Sequence[Any]], logger: logging.Logger):
    def ask():
        trial = study.ask()
        params = {name: trial.suggest_categorical(name, list(values)) for name, values in search_space.items()}
        return trial, params

    return _db_retry(ask, logger, "ask")


def _run_study(
    *,
    optuna: Any,
    config: Any,
    storage: Any,
    cache_paths: Any,
    executor: ProcessPoolExecutor | None,
    logger: logging.Logger,
    audit_dir: Path,
    snapshot_path: Path,
    database_path: Path | None,
    pareto_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    status: dict[str, Any],
    phase: str,
    scheme: str,
    train_start: str,
    train_end: str,
    test_start: str = "",
    test_end: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    name = _study_name(config.config_hash, phase, scheme, train_start, train_end)
    study = _make_study(optuna, config, storage, name, logger)
    remaining = _recover_and_queue(optuna, study, config, logger, audit_dir / "events.jsonl")
    batch_size = max(1, int(config.trial_batch_size or config.workers))
    status.update({"study": name, "phase": phase, "scheme": scheme, "window": [train_start, train_end]})

    while remaining > 0:
        batch_count = min(batch_size, remaining)
        pending = []
        tasks = []
        for _ in range(batch_count):
            trial, params = _ask_params(study, config.grid, logger)
            gid = core._resolve_gid(config.factor_module, config.gid_factory, params)
            trial_dir = audit_dir / "task_logs" / str(trial.number)
            safe_window = f"{train_start}_{train_end}".replace(":", "-")
            task = _task_dict(
                config, params, gid,
                phase=phase, scheme=scheme, win_start=train_start, win_end=train_end,
                train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end,
                trial_number=trial.number, study_name=name,
                task_log_path=str(trial_dir / f"{phase}_{safe_window}.log"),
            )
            pending.append((trial, params, task))
            tasks.append(task)
            event = {
                "time": _now(), "event": "START", "run": config.config_hash,
                "study": name, "trial": trial.number, "gid": gid, "params": params,
                "phase": phase, "scheme": scheme, "train": [train_start, train_end],
                "test": [test_start, test_end],
            }
            _append_jsonl(audit_dir / "events.jsonl", event)
            if str(config.sampler).lower() != "grid":
                candidate_rows.append({
                    "study_name": name,
                    "trial_number": trial.number,
                    "asked_at": event["time"],
                    "gid": gid,
                    "params_json": _parameter_key(params),
                    **params,
                })
                _atomic_csv(audit_dir / "candidate_manifest.csv", pd.DataFrame(candidate_rows))
            _log_trial(logger, "START", {**task, "params": params})

        try:
            result_frame = core.run_tasks(config, tasks, cache_paths, executor=executor)
        except BaseException:
            for trial, _params, _task in pending:
                try:
                    _db_retry(
                        lambda current=trial: current.set_user_attr("failure_reason", "interrupted"),
                        logger,
                        "set-interrupted",
                    )
                    _db_retry(lambda current=trial: _study_tell(study, current, state=optuna.trial.TrialState.FAIL), logger, "tell-interrupted")
                except Exception:
                    logger.exception("failed to mark interrupted trial=%s", trial.number)
            raise

        by_number = {
            int(row["trial_number"]): row.to_dict()
            for _index, row in result_frame.iterrows()
        }
        for trial, params, task in pending:
            result = by_number.get(trial.number, {**task, "error": "worker returned no result"})
            result["params"] = dict(params)
            result["constraint_violation"] = _normalized_violation(result, config)
            result["feasible"] = result["constraint_violation"] <= 1e-12
            constraints = _constraint_values(result, config)
            _db_retry(
                lambda current=trial, value=_jsonable(result): current.set_user_attr("result", value),
                logger,
                "set-result",
            )
            _db_retry(
                lambda current=trial, value=list(constraints): current.set_user_attr("constraints", value),
                logger,
                "set-constraints",
            )
            error = result.get("error")
            values = (result.get("robust_rank_ic"), result.get("sharpe"))
            valid = not error and all(value is not None and np.isfinite(float(value)) for value in values)
            if valid:
                _db_retry(lambda current=trial, score=values: _study_tell(study, current, values=score), logger, "tell-complete")
            else:
                _db_retry(
                    lambda current=trial, reason=str(error or "invalid objective"): current.set_user_attr(
                        "failure_reason", reason
                    ),
                    logger,
                    "set-failure-reason",
                )
                _db_retry(lambda current=trial: _study_tell(study, current, state=optuna.trial.TrialState.FAIL), logger, "tell-fail")
                _append_jsonl(audit_dir / "errors.jsonl", {
                    "time": _now(), "event": "TRIAL_ERROR", "study": name, "trial": trial.number,
                    "params": params, "error": error or "invalid objective",
                    "traceback_path": result.get("task_log_path"),
                })
            _append_jsonl(audit_dir / "events.jsonl", {
                "time": _now(), "event": "END", "run": config.config_hash,
                "study": name, "trial": trial.number, "gid": result.get("gid"),
                "params": params, "window": [train_start, train_end], "phase": phase,
                "metrics": {key: result.get(key) for key in ("robust_rank_ic", "sharpe", "mdd", "turnover", "ic_coverage")},
                "constraints": constraints, "constraint_violation": result["constraint_violation"],
                "feasible": result["feasible"], "status": "COMPLETE" if valid else "FAIL",
                "error": error, "traceback_path": result.get("task_log_path"),
            })
            _log_trial(logger, "END", result)
            logger.debug(
                "TASK trial=%s worker_pid=%s wide=%sx%s weights=%sx%s engine=%s elapsed=%s log=%s",
                trial.number, result.get("worker_pid"), result.get("wide_rows"), result.get("wide_columns"),
                result.get("weight_rows"), result.get("weight_columns"), result.get("engine", task.get("engine")),
                result.get("elapsed_seconds"), result.get("task_log_path"),
            )

        remaining -= batch_count
        current = _trial_result_frame(study)
        try:
            ordered, reason = _selection_table(current, config)
            window_front = ordered.assign(study_name=name, phase=phase).to_dict("records")
            pareto_rows[:] = [row for row in pareto_rows if row.get("study_name") != name] + window_front
            summary_columns = [
                column for column in (
                    "trial_number", "robust_rank_ic", "sharpe", "mdd", "turnover",
                    "ic_coverage", "constraint_violation", "feasible", "ic_top20", "params_json",
                ) if column in ordered
            ]
            logger.info(
                "PARETO study=%s reason=%s candidates=%d\n%s",
                name, reason, len(ordered), ordered.loc[:, summary_columns].head(10).to_string(index=False),
            )
        except RuntimeError:
            pass
        terminal_count = sum(
            trial.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL}
            for trial in study.trials
        )
        status.update({"completed_trials": terminal_count, "remaining_trials": remaining})
        status["last_snapshot"] = _db_retry(
            lambda: _snapshot_database(database_path, snapshot_path),
            logger,
            "snapshot-batch",
        )
        _db_retry(
            lambda: _refresh_audit(optuna, storage, audit_dir, pareto_rows, status),
            logger,
            "refresh-audit",
        )
        _append_jsonl(audit_dir / "events.jsonl", {"time": _now(), "event": "BATCH_FLUSH", "study": name}, sync=True)

    frame = _trial_result_frame(study)
    ordered, reason = _selection_table(frame, config)
    logger.info(
        "LOCK study=%s trial=%s reason=%s params=%s",
        name, int(ordered.iloc[0]["trial_number"]), reason, ordered.iloc[0]["params_json"],
    )
    return frame, ordered


def _oos_task(
    config: Any,
    params: Mapping[str, Any],
    *,
    phase: str,
    scheme: str,
    start: str,
    end: str,
    trial_number: int,
    audit_dir: Path,
    engine: str | None = None,
) -> dict[str, Any]:
    gid = core._resolve_gid(config.factor_module, config.gid_factory, params)
    task = _task_dict(
        config, params, gid,
        phase=phase, scheme=scheme, win_start=start, win_end=end,
        trial_number=trial_number,
        task_log_path=str(audit_dir / "task_logs" / str(trial_number) / f"{phase}_{start}_{end}.log"),
    )
    if engine:
        task["engine"] = engine
    return task


def _cv_results(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return trials.copy()
    frame = trials[trials["state"] == "COMPLETE"].copy()
    if frame.empty:
        return frame
    frame["params"] = frame["params_json"]
    parsed_params = frame["params_json"].map(json.loads)
    parameter_names = sorted({name for params in parsed_params for name in params})
    for name in parameter_names:
        frame[f"param_{name}"] = parsed_params.map(lambda params, key=name: params.get(key))
    frame["mean_test_robust_rank_ic"] = pd.to_numeric(frame["robust_rank_ic"], errors="coerce")
    frame["mean_test_sharpe"] = pd.to_numeric(frame["sharpe"], errors="coerce")
    frame["std_test_robust_rank_ic"] = 0.0
    frame["std_test_sharpe"] = 0.0
    elapsed = frame["elapsed_seconds"] if "elapsed_seconds" in frame else pd.Series(0.0, index=frame.index)
    frame["mean_fit_time"] = pd.to_numeric(elapsed, errors="coerce").fillna(0.0)
    frame["std_fit_time"] = 0.0
    frame["rank_test_robust_rank_ic"] = frame["mean_test_robust_rank_ic"].rank(method="min", ascending=False).astype("Int64")
    frame["rank_test_sharpe"] = frame["mean_test_sharpe"].rank(method="min", ascending=False).astype("Int64")
    return frame


def run_optuna_walk_forward(config: Any) -> dict[str, pd.DataFrame]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna mining requires `pip install betalens[mining]`") from exc

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logger(output_dir, config.log_level)
    if not config.config_hash:
        config.config_hash = hashlib.sha256(
            json.dumps(_jsonable({"grid": config.grid, "span": config.rolling_span, "valid": config.valid}), sort_keys=True).encode()
        ).hexdigest()
    lock_path = output_dir / ".study.lock"
    try:
        owner = _acquire_lock(lock_path, config.config_hash)
    except Exception:
        logger.exception("LOCK_REFUSED path=%s", lock_path.resolve())
        _close_logger(logger)
        raise
    database_path: Path | None = None
    storage: Any | None = None
    executor: ProcessPoolExecutor | None = None
    status = {
        "alpha": next(iter(config.grid.get("alpha_id", [None]))), "pid": os.getpid(),
        "started_at": owner["started_at"], "config_hash": config.config_hash,
        "cache_dir": str(Path(config.cache_dir or output_dir / "_cache").resolve()), "status": "running",
    }
    pareto_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []
    snapshot_path = output_dir / "study.snapshot.sqlite3"
    try:
        storage, database_path, storage_url = _sqlite_storage(optuna, output_dir, config.storage_url, logger)
        status.update({"storage": storage_url, "database": str(database_path.resolve()) if database_path else storage_url})
        grid_count = int(np.prod([len(values) for values in config.grid.values()]))
        if str(config.sampler).lower() == "grid" and grid_count > int(config.max_grid_candidates):
            raise ValueError(f"grid has {grid_count} candidates, exceeding max_grid_candidates={config.max_grid_candidates}")
        candidates = _grid_candidates(config.grid)
        manifest = pd.DataFrame([
            {"candidate_index": index, "gid": core._resolve_gid(config.factor_module, config.gid_factory, params),
             "params_json": _parameter_key(params), **params}
            for index, params in enumerate(candidates)
        ])
        if str(config.sampler).lower() == "grid":
            _atomic_csv(audit_dir / "candidate_manifest.csv", manifest)
        elif (audit_dir / "candidate_manifest.csv").exists():
            candidate_rows = pd.read_csv(audit_dir / "candidate_manifest.csv").to_dict("records")

        rolling_span = config.rolling_span or (config.train[0], config.test[1])
        pairs = []
        for train_len, test_len, step in config.paired_schemes or ():
            scheme = f"paired/{train_len}/{test_len}/{step}"
            generated = core.gen_rolling_train_test_windows(*rolling_span, train_len, test_len, step)
            if config.max_windows_per_scheme is not None:
                generated = generated[: int(config.max_windows_per_scheme)]
            pairs.extend([(*window, scheme) for window in generated])
        if not pairs:
            raise ValueError("Optuna walk-forward requires at least one paired TRAIN/TEST window")

        expected_cache = core._cache_paths(config.cache_dir or output_dir / "_cache")
        cache_mtime_before = expected_cache.data.stat().st_mtime_ns if expected_cache.data.exists() else None
        cache_paths = core.build_cache_for_config(config, [rolling_span, config.valid], config.paper_params or candidates[0])
        cache_hit = (
            cache_mtime_before is not None
            and cache_paths.data.exists()
            and cache_paths.data.stat().st_mtime_ns == cache_mtime_before
            and not config.rebuild_cache
        )
        effective_workers = core._effective_workers_for_memory(config, cache_paths)
        executor = ProcessPoolExecutor(
            max_workers=max(1, effective_workers),
            initializer=core._init_worker,
            initargs=(str(cache_paths.data), str(cache_paths.pit)),
        )
        logger.info(
            "STARTUP config=%s config_hash=%s range=%s valid=%s cache=%s cache_hit=%s memory_budget=%s workers=%d sampler=%s objectives=%s "
            "constraints=coverage>=%.2f,mdd<=%.2f,turnover<=%.2f dimensions=%d candidates_or_budget=%d "
            "db=%s audit=%s log=%s snapshot=%s",
            getattr(config, "config_path", None), config.config_hash, rolling_span, config.valid,
            cache_paths.data.resolve(), cache_hit, core._format_bytes(core._memory_budget_bytes(config)), effective_workers,
            config.sampler, ["robust_rank_ic", "sharpe"], config.ic_coverage_min,
            config.max_drawdown_max, config.turnover_max, sum(len(values) > 1 for values in config.grid.values()),
            grid_count if str(config.sampler).lower() == "grid" else config.n_trials,
            status["database"], audit_dir.resolve(), (output_dir / "run.log").resolve(), snapshot_path.resolve(),
        )
        _atomic_json(audit_dir / "status.json", status)
        train_frames, test_frames = [], []
        previous_params: dict[str, Any] | None = None
        final_ordered: pd.DataFrame | None = None
        final_train_end: pd.Timestamp | None = None
        for train_start, train_end, test_start, test_end, scheme in pairs:
            train_frame, ordered = _run_study(
                optuna=optuna, config=config, storage=storage, cache_paths=cache_paths, executor=executor,
                logger=logger, audit_dir=audit_dir, snapshot_path=snapshot_path, database_path=database_path,
                pareto_rows=pareto_rows, candidate_rows=candidate_rows, status=status, phase="train", scheme=scheme,
                train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end,
            )
            train_frames.append(train_frame)
            if final_train_end is None or pd.Timestamp(train_end) > final_train_end:
                final_ordered = ordered
                final_train_end = pd.Timestamp(train_end)
            locked = ordered.iloc[0]
            params = json.loads(str(locked["params_json"]))
            trial_number = int(locked["trial_number"])
            test_task = _oos_task(
                config, params, phase="test", scheme=scheme, start=test_start, end=test_end,
                trial_number=trial_number, audit_dir=audit_dir,
            )
            _log_trial(logger, "START", {**test_task, "params": params, "train_start": train_start, "train_end": train_end})
            test_result = core.run_tasks(config, [test_task], cache_paths, executor=executor).iloc[0].to_dict()
            changed = previous_params is not None and _parameter_key(previous_params) != _parameter_key(params)
            oos = {
                "scheme": scheme, "train_start": train_start, "train_end": train_end,
                "test_start": test_start, "test_end": test_end, "trial_number": trial_number,
                "gid": test_result.get("gid"), "params_json": _parameter_key(params),
                "parameter_changed": changed, **{key: test_result.get(key) for key in (
                    "robust_rank_ic", "mean_rank_ic", "sharpe", "mdd", "turnover", "ic_coverage", "error", "task_log_path"
                )},
            }
            oos_rows.append(oos)
            test_frames.append(pd.DataFrame([test_result]))
            previous_params = params
            _atomic_csv(audit_dir / "oos_parameter_path.partial.csv", pd.DataFrame(oos_rows))
            _append_jsonl(audit_dir / "events.jsonl", {"time": _now(), "event": "TEST_END", **oos}, sync=True)
            logger.info("TEST END locked_trial=%d changed=%s params=%s metrics=%s", trial_number, changed, _parameter_key(params), _parameter_key(oos))
            status.update({
                "last_test": [test_start, test_end],
                "last_snapshot": _db_retry(
                    lambda: _snapshot_database(database_path, snapshot_path),
                    logger,
                    "snapshot-test",
                ),
            })
            _atomic_json(audit_dir / "status.json", status)

        if final_ordered is None:
            raise RuntimeError("no final calibration candidates")
        final_rows = final_ordered.head(max(1, min(3, int(config.report_top_n)))).copy()
        final_candidates = []
        valid_tasks = []
        for rank, row in final_rows.iterrows():
            params = json.loads(str(row["params_json"]))
            final_candidates.append({
                "rank": rank + 1, "trial_number": int(row["trial_number"]),
                "selection_reason": row["selection_reason"], "params": params,
            })
            valid_tasks.append(_oos_task(
                config, params, phase="valid", scheme="final_calibration", start=config.valid[0], end=config.valid[1],
                trial_number=int(row["trial_number"]), audit_dir=audit_dir, engine="exact",
            ))
        _atomic_text(output_dir / "final_candidates.yaml", yaml.safe_dump({"candidates": final_candidates}, allow_unicode=True, sort_keys=False))
        valid_df = core.run_tasks(config, valid_tasks, cache_paths, executor=executor)
        if config.valid_report_hook:
            for rank, candidate in enumerate(final_candidates, 1):
                try:
                    core._call_module_function(
                        config.factor_module, config.valid_report_hook, candidate["params"], rank,
                        str(output_dir), config.valid[0], config.valid[1],
                    )
                except Exception as exc:
                    logger.exception("VALID report failed rank=%d error=%s", rank, exc)
                    _append_jsonl(audit_dir / "errors.jsonl", {
                        "time": _now(), "event": "VALID_REPORT_ERROR", "rank": rank,
                        "error": f"{type(exc).__name__}: {exc}",
                    }, sync=True)

        trials = _db_retry(
            lambda: _all_trials_frame(optuna, storage),
            logger,
            "read-final-trials",
        )
        pareto = pd.DataFrame(pareto_rows)
        oos_frame = pd.DataFrame(oos_rows)
        _atomic_csv(output_dir / "trials.csv", trials)
        _atomic_csv(output_dir / "pareto_front.csv", pareto)
        _atomic_csv(output_dir / "cv_results.csv", _cv_results(trials))
        _atomic_csv(output_dir / "oos_parameter_path.csv", oos_frame)
        _atomic_csv(audit_dir / "trials.partial.csv", trials)
        _atomic_csv(audit_dir / "pareto_front.partial.csv", pareto)
        _atomic_csv(audit_dir / "oos_parameter_path.partial.csv", oos_frame)
        status.update({
            "status": "completed",
            "finished_at": _now(),
            "exit_reason": "completed",
            "last_snapshot": _db_retry(
                lambda: _snapshot_database(database_path, snapshot_path),
                logger,
                "snapshot-complete",
            ),
        })
        _atomic_json(audit_dir / "status.json", status)
        logger.info(
            "COMPLETE db=%s snapshot=%s log=%s events=%s trials=%s pareto=%s cv=%s oos=%s candidates=%s",
            status["database"], snapshot_path.resolve(), (output_dir / "run.log").resolve(),
            (audit_dir / "events.jsonl").resolve(), (output_dir / "trials.csv").resolve(),
            (output_dir / "pareto_front.csv").resolve(), (output_dir / "cv_results.csv").resolve(),
            (output_dir / "oos_parameter_path.csv").resolve(), (output_dir / "final_candidates.yaml").resolve(),
        )
        return {
            "train_results": pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame(),
            "test_results": pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame(),
            "valid_results": valid_df, "trials": trials, "pareto_front": pareto,
            "oos_parameter_path": oos_frame,
        }
    except KeyboardInterrupt:
        status.update({"status": "interrupted", "finished_at": _now(), "exit_reason": "keyboard_interrupt"})
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            finally:
                executor = None
        marked = 0
        try:
            marked = _mark_running_trials_failed(optuna, storage, logger, audit_dir / "events.jsonl")
        except Exception:
            logger.exception("failed to mark RUNNING trials after interrupt")
        status["interrupted_trials"] = marked
        if storage is not None:
            try:
                _db_retry(
                    lambda: _refresh_audit(optuna, storage, audit_dir, pareto_rows, status),
                    logger,
                    "refresh-interrupted-audit",
                )
            except Exception:
                logger.exception("failed to refresh partial audit after interrupt")
        try:
            _atomic_csv(audit_dir / "oos_parameter_path.partial.csv", pd.DataFrame(oos_rows))
        except Exception:
            logger.exception("failed to refresh OOS audit after interrupt")
        try:
            status["last_snapshot"] = _db_retry(
                lambda: _snapshot_database(database_path, snapshot_path),
                logger,
                "snapshot-interrupted",
            )
        except Exception:
            logger.exception("failed to snapshot database after interrupt")
        try:
            _atomic_json(audit_dir / "status.json", status)
            _append_jsonl(
                audit_dir / "events.jsonl",
                {"time": _now(), "event": "INTERRUPTED", "exit_code": 130},
                sync=True,
            )
        except Exception:
            logger.exception("failed to write interrupt status")
        logger.error("INTERRUPTED exit_code=130 db=%s lock=%s", database_path, lock_path.resolve())
        raise SystemExit(130)
    except Exception as exc:
        status.update({"status": "failed", "finished_at": _now(), "exit_reason": f"{type(exc).__name__}: {exc}"})
        try:
            status["last_snapshot"] = _db_retry(
                lambda: _snapshot_database(database_path, snapshot_path),
                logger,
                "snapshot-failed",
            )
        except Exception:
            logger.exception("failed to snapshot database after coordinator error")
        try:
            _atomic_json(audit_dir / "status.json", status)
            _append_jsonl(audit_dir / "errors.jsonl", {
                "time": _now(), "event": "COORDINATOR_ERROR", "error": status["exit_reason"],
                "database": str(database_path) if database_path else config.storage_url,
                "lock": str(lock_path.resolve()), "holder": owner,
            }, sync=True)
        except Exception:
            logger.exception("failed to write coordinator error audit")
        logger.exception("FAILED %s", exc)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if storage is not None:
            storage.remove_session()
            storage.engine.dispose()
        _release_lock(lock_path, owner)
        _close_logger(logger)


__all__ = ["run_optuna_walk_forward"]
