"""Causal, auditable Optuna orchestration for Betalens factor mining."""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
import queue
import shutil
import sqlite3
import socket
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from betalens.factor import mining as core


_DB_RETRY_DELAYS = (1.0, 2.0, 4.0)
_JSONL_HANDLES: dict[Path, Any] = {}
_JSONL_LAST_SYNC: dict[Path, float] = {}
_SNAPSHOT_LOCK = threading.Lock()
_CPU_TIMES_LAST: tuple[int, int] | None = None
_PROCESS_CPU_LAST: tuple[float, float] | None = None
_IO_METRICS: dict[str, float | int] = {
    "last_sqlite_txn_seconds": 0.0,
    "sqlite_txn_count": 0,
}
_AUDIT_METRICS: dict[str, float | int] = {
    "last_refresh_seconds": 0.0,
    "refresh_count": 0,
}


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


def _write_batch_journal(audit_dir: Path, batch_id: str, payload: Mapping[str, Any]) -> Path:
    directory = audit_dir / "inflight_batches"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{batch_id}.json"
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, ensure_ascii=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    _replace_with_retry(temporary, target)
    return target


def _commit_batch_journal(audit_dir: Path, batch_id: str) -> None:
    source = audit_dir / "inflight_batches" / f"{batch_id}.json"
    if not source.exists():
        return
    target_dir = audit_dir / "inflight_batches" / "committed"
    target_dir.mkdir(parents=True, exist_ok=True)
    _replace_with_retry(source, target_dir / source.name)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    _replace_with_retry(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any], *, sync: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    stream = _JSONL_HANDLES.get(resolved)
    if stream is None or stream.closed:
        stream = resolved.open("a", encoding="utf-8", buffering=1)
        _JSONL_HANDLES[resolved] = stream
        _JSONL_LAST_SYNC[resolved] = time.monotonic()
    stream.write(json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()
    now = time.monotonic()
    if sync or now - _JSONL_LAST_SYNC.get(resolved, 0.0) >= 5.0:
        os.fsync(stream.fileno())
        _JSONL_LAST_SYNC[resolved] = now


def _close_jsonl_handles() -> None:
    for path, stream in list(_JSONL_HANDLES.items()):
        try:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        finally:
            _JSONL_HANDLES.pop(path, None)
            _JSONL_LAST_SYNC.pop(path, None)


def _configure_logger(
    output_dir: Path,
    level_name: str,
    console_trial_events: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(f"betalens.mining.{hash(output_dir.resolve())}")
    logger.handlers.clear()
    logger.propagate = False
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    terminal = logging.StreamHandler(sys.stdout)
    terminal.addFilter(
        lambda record: bool(console_trial_events)
        or not bool(getattr(record, "trial_detail", False))
    )
    terminal.setFormatter(formatter)
    terminal.setLevel(level)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(terminal)
    logger.addHandler(file_handler)
    return logger


def _start_io_worker(max_batches: int):
    requests: queue.PriorityQueue[Any] = queue.PriorityQueue(
        maxsize=max(1, int(max_batches))
    )
    errors: list[str] = []
    sequence = itertools.count()
    requests._betalens_sequence = sequence

    def run() -> None:
        while True:
            _priority, _sequence, item = requests.get()
            if item is None:
                return
            action, args, kwargs, result = item
            try:
                for delay in (*_DB_RETRY_DELAYS, None):
                    try:
                        result.set_result(action(*args, **kwargs))
                        break
                    except Exception as exc:
                        if "locked" not in str(exc).lower() or delay is None:
                            raise
                        time.sleep(delay)
            except BaseException as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                result.set_exception(exc)

    thread = threading.Thread(target=run, name="betalens-optuna-io", daemon=True)
    thread.start()
    return requests, thread, errors


def _io_submit(
    requests: queue.PriorityQueue[Any],
    action: Callable[..., Any],
    *args: Any,
    _priority: int = 10,
    _sequence: Any = None,
    **kwargs: Any,
) -> Future:
    result: Future = Future()
    if _sequence is None:
        _sequence = getattr(requests, "_betalens_sequence", None)
    if _sequence is None:
        raise RuntimeError("Optuna I/O queue sequence is required")
    requests.put((int(_priority), next(_sequence), (action, args, kwargs, result)))
    return result


def _io_call(
    requests: queue.PriorityQueue[Any],
    action: Callable[..., Any],
    *args: Any,
    _sequence: Any = None,
    **kwargs: Any,
) -> Any:
    return _io_submit(
        requests, action, *args, _sequence=_sequence, **kwargs
    ).result()


def _stop_io_worker(
    requests: queue.PriorityQueue[Any] | None,
    thread: threading.Thread | None,
    sequence: Any = None,
) -> None:
    if requests is None or thread is None:
        return
    if sequence is None:
        sequence = getattr(requests, "_betalens_sequence", None)
    requests.put((100, next(sequence), None))
    thread.join(timeout=30)


def _io_once(action: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    requests, thread, errors = _start_io_worker(2)
    try:
        future = _io_submit(requests, action, *args, **kwargs)
        value = future.result()
        if errors:
            raise RuntimeError(errors[-1])
        return value
    finally:
        _stop_io_worker(requests, thread)


def _audit_index_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("study_name", "")),
        int(row.get("trial_number", -1)),
        int(row.get("stage_index", -1)),
    )


def _audit_index_from_frame(frame: pd.DataFrame | None) -> dict[tuple[str, int, int], dict[str, Any]]:
    index: dict[tuple[str, int, int], dict[str, Any]] = {}
    if frame is None or frame.empty:
        return index
    for row in frame.to_dict("records"):
        stages = row.get("stage_results")
        stage_index = -1
        if isinstance(stages, str) and stages:
            try:
                parsed = json.loads(stages)
                if parsed:
                    stage_index = max(int(item.get("stage_index", -1)) for item in parsed)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        row["stage_index"] = stage_index
        index[_audit_index_key(row)] = dict(row)
    return index


def _start_audit_worker(
    logger: logging.Logger,
    audit_dir: Path,
    max_events: int,
    initial_trials: pd.DataFrame | None = None,
):
    requests: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(max_events)))
    errors: list[str] = []
    trial_index = _audit_index_from_frame(initial_trials)

    def run() -> None:
        while True:
            item = requests.get()
            if item is None:
                return
            try:
                kind = item[0]
                if kind == "event":
                    _append_jsonl(audit_dir / item[1], item[2], sync=bool(item[3]))
                elif kind == "trial":
                    logger.info(item[1], extra={"trial_detail": True})
                elif kind == "json":
                    _atomic_json(audit_dir / item[1], item[2])
                elif kind == "csv":
                    _atomic_csv(audit_dir / item[1], item[2])
                elif kind == "trial_rows":
                    for row in item[1]:
                        trial_index[_audit_index_key(row)] = dict(row)
                elif kind == "refresh_incremental":
                    refresh_started = time.perf_counter()
                    trials = pd.DataFrame(trial_index.values())
                    if not trials.empty:
                        trials = trials.sort_values(
                            ["study_name", "trial_number", "stage_index"],
                            kind="mergesort",
                        ).drop_duplicates(
                            ["study_name", "trial_number"], keep="last"
                        ).drop(columns=["stage_index"], errors="ignore")
                    _atomic_csv(audit_dir / "trials.partial.csv", trials)
                    _atomic_csv(audit_dir / "pareto_front.partial.csv", item[1])
                    _atomic_csv(audit_dir / "oos_parameter_path.partial.csv", item[2])
                    _atomic_json(audit_dir / "status.json", item[3])
                    _AUDIT_METRICS["last_refresh_seconds"] = max(
                        time.perf_counter() - refresh_started, 0.0
                    )
                    _AUDIT_METRICS["refresh_count"] = int(
                        _AUDIT_METRICS["refresh_count"]
                    ) + 1
            except BaseException as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

    thread = threading.Thread(target=run, name="betalens-mining-audit", daemon=True)
    thread.start()
    return requests, thread, errors


def _audit_event(requests: queue.Queue[Any], filename: str, payload: Mapping[str, Any], *, sync: bool = False) -> None:
    requests.put(("event", filename, dict(payload), sync))


def _audit_trial(requests: queue.Queue[Any], event: str, value: Mapping[str, Any]) -> None:
    params = dict(value.get("params") or {})
    requests.put((
        "trial",
        f"{event} study={value.get('study_name', '')} trial={value.get('trial_number', '')} "
        f"candidate={value.get('candidate_id', '')} params={_parameter_key(params)} "
        f"IC={value.get('robust_rank_ic', '')} Sharpe={value.get('sharpe', '')} "
        f"MDD={value.get('mdd', '')} turnover={value.get('turnover', '')} "
        f"coverage={value.get('ic_coverage', '')} error={value.get('error', '')}",
    ))


def _audit_json(requests: queue.Queue[Any], filename: str, payload: Mapping[str, Any]) -> None:
    requests.put(("json", filename, dict(payload)))


def _audit_csv(requests: queue.Queue[Any], filename: str, frame: pd.DataFrame) -> None:
    requests.put(("csv", filename, frame.copy()))


def _audit_trial_rows(requests: queue.Queue[Any], rows: Sequence[Mapping[str, Any]]) -> None:
    if rows:
        requests.put(("trial_rows", [dict(row) for row in rows]))


def _audit_refresh_incremental(
    requests: queue.Queue[Any],
    pareto_rows: Sequence[Mapping[str, Any]],
    oos_rows: Sequence[Mapping[str, Any]],
    status: Mapping[str, Any],
) -> None:
    requests.put((
        "refresh_incremental",
        pd.DataFrame([dict(row) for row in pareto_rows]),
        pd.DataFrame([dict(row) for row in oos_rows]),
        dict(status),
    ))


def _stop_audit_worker(requests: queue.Queue[Any] | None, thread: threading.Thread | None) -> None:
    if requests is None or thread is None:
        return
    requests.put(None)
    thread.join(timeout=30)


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _process_started_at(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(int(pid)).create_time())
    except Exception:
        return None


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


def _lock_owner_active(owner: Mapping[str, Any]) -> bool:
    if str(owner.get("host") or socket.gethostname()) != socket.gethostname():
        return True
    pid = int(owner.get("pid", -1))
    if not _pid_is_active(pid):
        return False
    recorded = owner.get("process_started_at")
    actual = _process_started_at(pid)
    return recorded is None or actual is None or abs(float(recorded) - actual) < 1.0


def _acquire_lock(path: Path, config_hash: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "process_started_at": _process_started_at(os.getpid()),
        "started_at": _now(),
        "config_hash": config_hash,
        "owner_token": uuid.uuid4().hex,
    }
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
            if _lock_owner_active(holder):
                raise RuntimeError(f"active study coordinator refused: lock={path.resolve()} holder={holder}")
            path.unlink(missing_ok=True)


def _release_lock(path: Path, owner: Mapping[str, Any]) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if current.get("owner_token") == owner.get("owner_token"):
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
    local_runtime = database_path is not None

    if database_path is not None:
        database_path.parent.mkdir(parents=True, exist_ok=True)

        def configure_sqlite() -> None:
            with closing(sqlite3.connect(database_path, timeout=30)) as connection:
                with connection:
                    connection.execute(f"PRAGMA journal_mode={'WAL' if local_runtime else 'DELETE'}")
                    connection.execute("PRAGMA busy_timeout=30000")
                    connection.execute(f"PRAGMA synchronous={'NORMAL' if local_runtime else 'FULL'}")

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
        from sqlalchemy import event

        synchronous = "NORMAL" if local_runtime else "FULL"

        def configure_connection(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute(f"PRAGMA synchronous={synchronous}")
            finally:
                cursor.close()

        event.listen(storage.engine, "connect", configure_connection)
        storage.engine.dispose()
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


def _default_runtime_root() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "Betalens" / "mining"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "betalens" / "mining"


def _runtime_dir(config: Any, output_dir: Path) -> Path:
    root = Path(config.runtime_root).expanduser() if getattr(config, "runtime_root", None) else _default_runtime_root()
    alpha = next(iter(config.grid.get("alpha_id", [output_dir.name])), output_dir.name)
    search_hash = str(getattr(config, "search_hash", "") or config.config_hash)
    return (root / f"ALPHA{alpha}" / search_hash[:16]).resolve()


def _copy_database(source: Path, target: Path) -> None:
    if not source.exists() or target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    with closing(sqlite3.connect(source, timeout=30)) as source_connection:
        with closing(sqlite3.connect(temporary)) as target_connection:
            source_connection.backup(target_connection)
    os.replace(temporary, target)


def _atomic_copy(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
    shutil.copy2(source, temporary)
    _replace_with_retry(temporary, target)


def _start_mirror_worker(runtime_dir: Path, output_dir: Path):
    requests: queue.Queue[Any] = queue.Queue(maxsize=1)
    errors: list[str] = []

    def run() -> None:
        while True:
            item = requests.get()
            if item is None:
                return
            relatives, completed = item
            try:
                for relative in relatives:
                    try:
                        _atomic_copy(runtime_dir / relative, output_dir / relative)
                    except Exception as exc:
                        errors.append(f"{relative}: {type(exc).__name__}: {exc}")
            finally:
                completed.set()

    thread = threading.Thread(target=run, name="betalens-audit-mirror", daemon=True)
    thread.start()
    return requests, thread, errors


def _request_mirror(
    requests: queue.Queue[Any],
    relatives: Sequence[Path | str],
    *,
    block: bool = False,
) -> threading.Event | None:
    completed = threading.Event()
    item = ([Path(relative) for relative in relatives], completed)
    try:
        if block:
            requests.put(item, timeout=30)
        else:
            requests.put_nowait(item)
    except queue.Full:
        if block:
            raise RuntimeError("audit mirror queue remained busy")
        return None
    return completed


def _start_snapshot_worker(database_path: Path | None, snapshot_path: Path):
    requests: queue.Queue[Any] = queue.Queue(maxsize=1)
    errors: list[str] = []

    def run() -> None:
        while True:
            item = requests.get()
            if item is None:
                return
            try:
                _snapshot_database(database_path, snapshot_path)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

    thread = threading.Thread(target=run, name="betalens-sqlite-snapshot", daemon=True)
    thread.start()
    return requests, thread, errors


def _request_snapshot(requests: queue.Queue[Any] | None) -> None:
    if requests is None:
        return
    try:
        requests.put_nowait(True)
    except queue.Full:
        pass


def _snapshot_database(database_path: Path | None, snapshot_path: Path) -> str | None:
    if database_path is None or not database_path.exists():
        return None
    with _SNAPSHOT_LOCK:
        temporary = snapshot_path.with_name(
            snapshot_path.name + f".tmp-{os.getpid()}-{threading.get_ident()}"
        )
        temporary.unlink(missing_ok=True)
        with closing(sqlite3.connect(database_path, timeout=30)) as source:
            with closing(sqlite3.connect(temporary)) as target:
                source.backup(target)
        _replace_with_retry(temporary, snapshot_path)
    return str(snapshot_path.resolve())


def _parameter_key(params: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(params), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _grid_candidates(
    search_space: Mapping[str, Sequence[Any]],
    paper_params: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    names = list(search_space)
    candidates = [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(search_space[name] for name in names))
    ]
    if paper_params:
        paper = {name: paper_params[name] for name in names}
        paper_key = _parameter_key(paper)
        candidates = [paper] + [item for item in candidates if _parameter_key(item) != paper_key]
    return candidates


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


def _pruning_stages(config: Any) -> tuple[float, ...]:
    if not bool(getattr(config, "pruning_enabled", False)):
        return (1.0,)
    if str(config.sampler).lower() != "grid":
        raise ValueError("successive-halving pruning currently requires sampler=grid")
    stages = tuple(float(value) for value in getattr(config, "pruning_stages", (1.0,)))
    if not stages or any(not np.isfinite(value) or value <= 0.0 or value > 1.0 for value in stages):
        raise ValueError("pruning.stages must contain finite fractions in (0, 1]")
    if tuple(sorted(set(stages))) != stages or not np.isclose(stages[-1], 1.0):
        raise ValueError("pruning.stages must be strictly increasing and end at 1.0")
    if int(getattr(config, "pruning_reduction_factor", 3)) < 2:
        raise ValueError("pruning.reduction_factor must be >= 2")
    if int(getattr(config, "pruning_min_full_candidates", 3)) < 1:
        raise ValueError("pruning.min_full_candidates must be >= 1")
    return stages


def _pruning_stage_end(start: str, end: str, fraction: float) -> str:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if float(fraction) >= 1.0:
        return end_ts.strftime("%Y-%m-%d")
    span_days = max(0, int((end_ts - start_ts).days))
    offset = max(0, min(span_days, int(math.floor(span_days * float(fraction)))))
    return (start_ts + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")


def _rung_survivors(
    records: Sequence[Mapping[str, Any]],
    config: Any,
    *,
    paper_params: Mapping[str, Any] | None = None,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Rank one causal prefix rung and retain a deterministic survivor set."""
    if not records:
        return [], []
    rows = []
    by_trial: dict[int, Mapping[str, Any]] = {}
    for record in records:
        result = dict(record["result"])
        trial_number = int(record["trial_number"])
        candidate_order = int(record.get("candidate_order", trial_number))
        by_trial[trial_number] = record
        rows.append({
            "trial_number": trial_number,
            "candidate_order": candidate_order,
            "params_json": _parameter_key(record["params"]),
            "constraint_violation": _normalized_violation(result, config),
            "robust_rank_ic": pd.to_numeric(result.get("robust_rank_ic"), errors="coerce"),
            "sharpe": pd.to_numeric(result.get("sharpe"), errors="coerce"),
            "mdd": pd.to_numeric(result.get("mdd"), errors="coerce"),
            "turnover": pd.to_numeric(result.get("turnover"), errors="coerce"),
        })
    frame = pd.DataFrame(rows)
    finite = np.isfinite(frame["robust_rank_ic"]) & np.isfinite(frame["sharpe"])
    frame = frame[finite].copy()
    if frame.empty:
        return [], list(records)
    frame["feasible_rank"] = (frame["constraint_violation"] > 1e-12).astype(int)
    frame = frame.sort_values(
        ["feasible_rank", "constraint_violation", "robust_rank_ic", "sharpe", "mdd", "turnover", "candidate_order"],
        ascending=[True, True, False, False, True, True, True],
        kind="mergesort",
    )
    reduction = int(getattr(config, "pruning_reduction_factor", 3))
    minimum = int(getattr(config, "pruning_min_full_candidates", 3))
    keep_count = min(len(frame), max(minimum, int(math.ceil(len(frame) / reduction))))
    survivor_numbers = frame.head(keep_count)["trial_number"].astype(int).tolist()
    if bool(getattr(config, "pruning_keep_paper", True)) and paper_params:
        paper_key = _parameter_key(paper_params)
        paper_rows = frame[frame["params_json"] == paper_key]
        if not paper_rows.empty:
            paper_number = int(paper_rows.iloc[0]["trial_number"])
            if paper_number not in survivor_numbers:
                survivor_numbers[-1] = paper_number
    survivor_set = set(survivor_numbers)
    survivors = [by_trial[number] for number in survivor_numbers]
    pruned = [record for record in records if int(record["trial_number"]) not in survivor_set]
    return survivors, pruned


def _trial_result_frame(study: Any) -> pd.DataFrame:
    rows = []
    for trial in study.trials:
        payload = dict(trial.user_attrs.get("betalens") or {})
        result = dict(payload.get("result") or trial.user_attrs.get("result") or {})
        row = {
            "study_name": study.study_name,
            "trial_number": trial.number,
            "state": trial.state.name,
            "values": json.dumps(list(trial.values or ())),
            "params_json": _parameter_key(trial.params),
            "candidate_id": payload.get("candidate_id") or trial.user_attrs.get("candidate_id") or _candidate_id(trial.params),
            "candidate_order": int(payload.get("candidate_order", trial.number)),
            "stage_results": json.dumps(payload.get("stage_results") or trial.user_attrs.get("stage_results") or [], ensure_ascii=False),
            "pruning_reason": payload.get("pruning_reason") or trial.user_attrs.get("pruning_reason", ""),
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
    if "candidate_order" not in valid:
        valid["candidate_order"] = pd.to_numeric(
            valid.get("trial_number"), errors="coerce"
        )
    if "state" in valid:
        valid = valid[valid["state"].astype(str) == "COMPLETE"]
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
        ["ic_top20", "sharpe", "mdd", "turnover", "candidate_order"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["lock_rank"] = np.arange(1, len(ordered) + 1)
    ordered["selection_reason"] = reason
    return ordered, reason


def _study_name(search_hash: str, phase: str, scheme: str, start: str, end: str) -> str:
    digest = hashlib.sha1(f"{phase}|{scheme}|{start}|{end}".encode("utf-8")).hexdigest()[:12]
    return f"wf_{search_hash[:12]}_{phase}_{digest}"


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
    previous_hash = study.user_attrs.get("config_hash")
    if previous_hash and previous_hash != config.config_hash:
        _db_retry(
            lambda: study.set_user_attr("claimed_from_config_hash", previous_hash),
            logger,
            "set-claimed-config-hash",
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
    _db_retry(
        lambda: study.set_user_attr("search_hash", str(getattr(config, "search_hash", config.config_hash))),
        logger,
        "set-study-search-hash",
    )
    _db_retry(
        lambda: study.set_user_attr("pruning", {
            "enabled": bool(getattr(config, "pruning_enabled", False)),
            "mode": "causal_successive_halving",
            "stages": list(_pruning_stages(config)),
            "reduction_factor": int(getattr(config, "pruning_reduction_factor", 3)),
            "min_full_candidates": int(getattr(config, "pruning_min_full_candidates", 3)),
            "keep_paper_candidate": bool(getattr(config, "pruning_keep_paper", True)),
        }),
        logger,
        "set-study-pruning",
    )
    return study


def _legacy_study_index(optuna: Any, storage: Any, config: Any, logger: logging.Logger) -> dict[tuple[str, str, str], str]:
    """Index studies created before search/runtime hashes were separated."""
    index: dict[tuple[str, str, str], str] = {}
    summaries = _db_retry(
        lambda: optuna.study.get_all_study_summaries(storage=storage), logger, "index-existing-studies"
    )
    allowed = {name: {_parameter_key({"value": value}) for value in values} for name, values in config.grid.items()}
    for summary in summaries:
        study = _db_retry(
            lambda name=summary.study_name: optuna.load_study(study_name=name, storage=storage),
            logger,
            "load-existing-study",
        )
        previous_search_hash = study.user_attrs.get("search_hash")
        current_search_hash = str(getattr(config, "search_hash", config.config_hash))
        if previous_search_hash and previous_search_hash != current_search_hash:
            continue
        trials = study.get_trials(deepcopy=False)
        sample = next((trial for trial in trials if trial.user_attrs.get("result")), None)
        if sample is None:
            continue
        result = dict(sample.user_attrs["result"])
        scheme = str(result.get("scheme", ""))
        start = str(result.get("train_start") or result.get("win_start") or "")
        end = str(result.get("train_end") or result.get("win_end") or "")
        if not scheme or not start or not end:
            continue
        key = (scheme, start, end)
        for trial in trials:
            if not trial.params:
                continue
            if set(trial.params) != set(config.grid):
                raise RuntimeError(
                    f"existing study parameter names do not match current search space: "
                    f"study={study.study_name} expected={sorted(config.grid)} actual={sorted(trial.params)}"
                )
            for name, value in trial.params.items():
                if _parameter_key({"value": value}) not in allowed[name]:
                    raise RuntimeError(
                        f"existing study parameter value is outside current search space: "
                        f"study={study.study_name} parameter={name} value={value!r}"
                    )
        existing = index.get(key)
        if existing and existing != study.study_name:
            raise RuntimeError(f"multiple existing studies claim the same TRAIN window: {key}")
        index[key] = study.study_name
    if index:
        logger.info("RESUME INDEX claimed_windows=%d studies_scanned=%d", len(index), len(summaries))
    return index


def _recover_and_queue(
    optuna: Any,
    study: Any,
    config: Any,
    logger: logging.Logger,
    event_path: Path | None,
) -> int:
    active_candidate_keys = set(study.user_attrs.get("active_candidate_keys") or ())
    stale_trials: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for frozen in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        key = _parameter_key(frozen.params)
        if active_candidate_keys and key not in active_candidate_keys:
            _db_retry(
                lambda trial_id=frozen._trial_id: study._storage.set_trial_user_attr(
                    trial_id, "pruning_reason", "successive_halving_recovered"
                ),
                logger,
                "mark-recovered-pruned",
            )
            _db_retry(
                lambda number=frozen.number: _study_tell(
                    study, number, state=optuna.trial.TrialState.PRUNED
                ),
                logger,
                "prune-recovered",
            )
            if event_path is not None:
                _append_jsonl(event_path, {
                    "time": _now(), "event": "PRUNED_AFTER_RESTART", "study": study.study_name,
                    "trial": frozen.number, "params": frozen.params,
                })
            continue
        _db_retry(
            lambda trial_id=frozen._trial_id: study._storage.set_trial_user_attr(
                trial_id, "failure_reason", "stale_after_restart"
            ),
            logger,
            "mark-stale",
        )
        _db_retry(lambda number=frozen.number: _study_tell(study, number, state=optuna.trial.TrialState.FAIL), logger, "fail-stale")
        recovered_attrs = {
            "recovered_from_trial": int(frozen.number),
            "stage_results": list(frozen.user_attrs.get("stage_results") or ()),
        }
        if frozen.user_attrs.get("candidate_id"):
            recovered_attrs["candidate_id"] = frozen.user_attrs["candidate_id"]
        stale_trials.append((dict(frozen.params), recovered_attrs))
        if event_path is not None:
            _append_jsonl(event_path, {
                "time": _now(), "event": "STALE_AFTER_RESTART", "study": study.study_name,
                "trial": frozen.number, "params": frozen.params,
            })
    queued_keys = {
        _parameter_key(trial.params or dict(trial.system_attrs.get("fixed_params") or {}))
        for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.WAITING,))
    }
    for params, user_attrs in stale_trials:
        key = _parameter_key(params)
        if key in queued_keys:
            continue
        _db_retry(
            lambda values=params, attrs=user_attrs: study.enqueue_trial(
                values, user_attrs=attrs
            ),
            logger,
            "enqueue-stale",
        )
        queued_keys.add(key)

    terminal = [
        trial for trial in study.trials
        if trial.state in {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.PRUNED,
        }
        and trial.user_attrs.get("failure_reason") != "stale_after_restart"
    ]
    if str(config.sampler).lower() == "grid":
        candidates = _grid_candidates(config.grid)
        if active_candidate_keys:
            candidates = [
                params for params in candidates if _parameter_key(params) in active_candidate_keys
            ]
        terminal_keys = {_parameter_key(trial.params) for trial in terminal}
        waiting_keys = {
            _parameter_key(
                trial.params or dict(trial.system_attrs.get("fixed_params") or {})
            )
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
        "stage=%s fraction=%s params=%s IC=%s Sharpe=%s MDD=%s turnover=%s coverage=%s violation=%s feasible=%s "
        "elapsed=%ss error=%s log=%s",
        event,
        result.get("alpha_id", params.get("alpha_id")), result.get("phase"), result.get("scheme"),
        result.get("train_start", result.get("win_start")), result.get("train_end", result.get("win_end")),
        result.get("test_start", ""), result.get("test_end", ""), result.get("trial_number", ""),
        result.get("gid", ""), result.get("stage_index", ""), result.get("stage_fraction", ""),
        _parameter_key(params),
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
        sampler = study.sampler
        if sampler.__class__.__name__ == "GridSampler":
            attrs = study._storage.get_trial_system_attrs(trial._trial_id)
            if "search_space" not in attrs:
                study._storage.set_trial_system_attr(
                    trial._trial_id, "search_space", dict(getattr(sampler, "_search_space", search_space))
                )
            if "grid_id" not in attrs:
                names = list(getattr(sampler, "_param_names", tuple(search_space)))
                target = tuple(params[name] for name in names)
                grids = list(getattr(sampler, "_all_grids", ()))
                grid_id = next(
                    (index for index, values in enumerate(grids) if tuple(values) == target),
                    None,
                )
                if grid_id is None:
                    raise RuntimeError(f"GridSampler cannot map fixed params to a grid id: {params}")
                study._storage.set_trial_system_attr(trial._trial_id, "grid_id", int(grid_id))
        return trial, params

    return _db_retry(ask, logger, "ask")


def _grid_trial(
    optuna: Any,
    study: Any,
    search_space: Mapping[str, Sequence[Any]],
    params: Mapping[str, Any],
    candidate_order: int,
) -> Any:
    """Create a fixed RUNNING Grid trial without invoking GridSampler.ask()."""
    distributions = {
        name: optuna.distributions.CategoricalDistribution(list(search_space[name]))
        for name in search_space
    }
    candidate_id = _candidate_id(params)
    template = optuna.trial.create_trial(
        state=optuna.trial.TrialState.RUNNING,
        params=dict(params),
        distributions=distributions,
        user_attrs={
            "betalens": {
                "candidate_id": candidate_id,
                "candidate_order": int(candidate_order),
                "stage_results": [],
            }
        },
    )
    study.add_trial(template)
    frozen = study.get_trials(deepcopy=False)[-1]
    return optuna.trial.Trial(study, frozen._trial_id)


def _register_grid_records(
    optuna: Any,
    study: Any,
    search_space: Mapping[str, Sequence[Any]],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    registered = [dict(record) for record in records]
    current_trials = study.get_trials(deepcopy=False)
    reusable: dict[str, Any] = {}
    for frozen in current_trials:
        if frozen.state.is_finished():
            continue
        payload = dict(frozen.user_attrs.get("betalens") or {})
        candidate_id = payload.get("candidate_id") or _candidate_id(frozen.params)
        reusable[str(candidate_id)] = frozen
    for record in registered:
        if record.get("trial") is not None:
            continue
        frozen = reusable.get(_candidate_id(record["params"]))
        if frozen is not None:
            record["trial"] = optuna.trial.Trial(study, frozen._trial_id)
            record["trial_number"] = int(frozen.number)
    missing = [record for record in registered if record.get("trial") is None]
    if missing:
        distributions = {
            name: optuna.distributions.CategoricalDistribution(list(search_space[name]))
            for name in search_space
        }
        templates = [
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.RUNNING,
                params=dict(record["params"]),
                distributions=distributions,
                user_attrs={
                    "betalens": {
                        "candidate_id": _candidate_id(record["params"]),
                        "candidate_order": int(record["candidate_order"]),
                        "stage_results": [],
                        **dict(record.get("recovery_attrs") or {}),
                    }
                },
            )
            for record in missing
        ]
        study.add_trials(templates)
        by_candidate: dict[str, Any] = {}
        for frozen in study.get_trials(deepcopy=False):
            payload = dict(frozen.user_attrs.get("betalens") or {})
            candidate_id = payload.get("candidate_id") or _candidate_id(frozen.params)
            by_candidate[str(candidate_id)] = frozen
        for record in missing:
            candidate_id = _candidate_id(record["params"])
            frozen = by_candidate[candidate_id]
            record["trial"] = optuna.trial.Trial(study, frozen._trial_id)
            record["trial_number"] = int(frozen.number)
    for record in registered:
        if "trial_number" not in record:
            record["trial_number"] = int(record["trial"].number)
    return registered


def _mark_grid_running_stale(
    optuna: Any,
    study: Any,
    event_path: Path | None,
) -> int:
    marked = 0
    for frozen in study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
    ):
        payload = dict(frozen.user_attrs.get("betalens") or {})
        payload["failure_reason"] = "stale_after_restart"
        payload["recovered_from_trial"] = int(frozen.number)
        trial = optuna.trial.Trial(study, frozen._trial_id)
        trial.set_user_attr("betalens", payload)
        _study_tell(study, trial, state=optuna.trial.TrialState.FAIL)
        marked += 1
        if event_path is not None:
            _append_jsonl(event_path, {
                "time": _now(), "event": "STALE_AFTER_RESTART",
                "study": study.study_name, "trial": frozen.number,
                "params": frozen.params,
            })
    return marked


def _reconcile_running_last_stage(
    optuna: Any,
    config: Any,
    study: Any,
    stage_index: int,
    last_stage_index: int,
) -> int:
    """Finish trials whose final payload committed before their tell transaction."""
    if int(stage_index) != int(last_stage_index):
        return 0
    reconciled = 0
    running = study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
    )
    for frozen in running:
        payload = dict(frozen.user_attrs.get("betalens") or {})
        stage = next(
            (
                row for row in payload.get("stage_results", ())
                if int(row.get("stage_index", -1)) == int(stage_index)
            ),
            None,
        )
        if stage is None:
            continue
        value = dict(payload.get("result") or stage.get("result") or {})
        objectives = (value.get("robust_rank_ic"), value.get("sharpe"))
        valid = not value.get("error") and all(
            item is not None and np.isfinite(float(item)) for item in objectives
        )
        trial = optuna.trial.Trial(study, frozen._trial_id)
        if valid:
            _study_tell(study, trial, values=objectives)
        else:
            payload["failure_reason"] = str(value.get("error") or "invalid objective")
            trial.set_user_attr("betalens", payload)
            _study_tell(study, trial, state=optuna.trial.TrialState.FAIL)
        reconciled += 1
    return reconciled


def _rdb_storage(storage: Any) -> Any:
    current = storage
    while not hasattr(current, "engine") and hasattr(current, "_backend"):
        current = current._backend
    return current


def _validate_sqlite_batch_adapter(storage: Any) -> None:
    """Fail fast if the installed Optuna RDB schema cannot support bulk writes."""
    backend = _rdb_storage(storage)
    if not hasattr(backend, "engine"):
        raise RuntimeError("SQLite batch persistence requires an Optuna RDBStorage backend")
    from optuna.storages._rdb import models

    required = {
        models.TrialModel: {"trial_id", "state", "datetime_complete"},
        models.TrialUserAttributeModel: {"trial_id", "key", "value_json"},
        models.TrialValueModel: {"trial_id", "objective", "value", "value_type"},
        models.StudyUserAttributeModel: {"study_id", "key", "value_json"},
    }
    for model, columns in required.items():
        available = {column.name for column in model.__table__.columns}
        missing = columns - available
        if missing:
            raise RuntimeError(
                f"Optuna SQLite batch adapter schema mismatch for {model.__name__}: {sorted(missing)}"
            )


def _sqlite_upsert(connection: Any, model: Any, keys: Mapping[str, Any], values: Mapping[str, Any]) -> None:
    """Upsert one Optuna RDB row inside the caller-owned transaction."""
    from sqlalchemy import and_, insert, select, update

    table = model.__table__
    predicates = [table.c[name] == value for name, value in keys.items()]
    found = connection.execute(select(table.c[list(keys)[0]]).where(and_(*predicates))).first()
    if found is None:
        connection.execute(insert(table).values(**dict(keys), **dict(values)))
    else:
        connection.execute(update(table).where(and_(*predicates)).values(**dict(values)))


def _sqlite_trial_payloads(
    study: Any,
    trial_ids: Sequence[int] | None = None,
) -> dict[int, tuple[Any, dict[str, Any]]]:
    if trial_ids is None:
        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)
    else:
        trials = [
            study._storage.get_trial(int(trial_id))
            for trial_id in trial_ids
        ]
    return {
        int(trial._trial_id): (trial, dict(trial.user_attrs.get("betalens") or {}))
        for trial in trials
    }


def _audit_row_from_payload(
    *, trial: Any, study_name: str, payload: Mapping[str, Any], value: Mapping[str, Any], state: str,
    stage_index: int,
) -> dict[str, Any]:
    params = dict(value.get("params") or trial.params)
    result = dict(payload.get("result") or value)
    return {
        "study_name": str(study_name),
        "trial_number": int(trial.number),
        "state": state,
        "values": json.dumps(list(trial.values or ()), ensure_ascii=False),
        "params_json": _parameter_key(params),
        "candidate_id": payload.get("candidate_id") or _candidate_id(params),
        "candidate_order": int(payload.get("candidate_order", trial.number)),
        "stage_results": json.dumps(payload.get("stage_results") or [], ensure_ascii=False),
        "pruning_reason": payload.get("pruning_reason", ""),
        "stage_index": int(stage_index),
        **params,
        **result,
    }


def _sqlite_apply_trial_mutations(
    storage: Any,
    mutations: Sequence[Mapping[str, Any]],
    *,
    study: Any | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> None:
    if not mutations and checkpoint is None:
        return
    from optuna.storages._rdb import models

    trial_model = models.TrialModel
    attr_model = models.TrialUserAttributeModel
    value_model = models.TrialValueModel
    checkpoint_payload = None
    if study is not None and checkpoint is not None:
        checkpoint_payload = dict(study.user_attrs.get("betalens") or {})
        checkpoint_payload.update(_jsonable(checkpoint))
    with _rdb_storage(storage).engine.begin() as connection:
        for mutation in mutations:
            trial_id = int(mutation["trial_id"])
            _sqlite_upsert(
                connection,
                attr_model,
                {"trial_id": trial_id, "key": "betalens"},
                {"value_json": json.dumps(_jsonable(mutation["payload"]), ensure_ascii=False)},
            )
            if mutation.get("constraints") is not None:
                _sqlite_upsert(
                    connection,
                    attr_model,
                    {"trial_id": trial_id, "key": "constraints"},
                    {"value_json": json.dumps(_jsonable(mutation["constraints"]), ensure_ascii=False)},
                )
            state = mutation.get("state")
            if state is None:
                continue
            now = datetime.now(timezone.utc)
            connection.execute(
                trial_model.__table__.update()
                .where(trial_model.__table__.c.trial_id == trial_id)
                .values(state=str(state), datetime_complete=now)
            )
            if state == "COMPLETE":
                for objective, value in enumerate(mutation["objectives"]):
                    _sqlite_upsert(
                        connection,
                        value_model,
                        {"trial_id": trial_id, "objective": int(objective)},
                        {"value": float(value), "value_type": "FINITE"},
                    )
        if study is not None and checkpoint_payload is not None:
            _sqlite_upsert(
                connection,
                models.StudyUserAttributeModel,
                {"study_id": int(study._study_id), "key": "betalens"},
                {"value_json": json.dumps(checkpoint_payload, ensure_ascii=False)},
            )


def _prepare_result_mutations(
    optuna: Any,
    config: Any,
    study: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    last_stage: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads = _sqlite_trial_payloads(
        study, [int(record["trial"]._trial_id) for record in records]
    )
    mutations: list[dict[str, Any]] = []
    durable: list[dict[str, Any]] = []
    for record in records:
        trial = record["trial"]
        task = dict(record["task"])
        result = dict(record["result"])
        params = dict(record["params"])
        candidate_id = _candidate_id(params)
        value = {**task, **result, "params": params, "candidate_id": candidate_id}
        value["constraint_violation"] = _normalized_violation(value, config)
        value["feasible"] = value["constraint_violation"] <= 1e-12
        objectives = (value.get("robust_rank_ic"), value.get("sharpe"))
        valid = not value.get("error") and all(
            item is not None and np.isfinite(float(item)) for item in objectives
        )
        frozen, previous = payloads[int(trial._trial_id)]
        existing_stage = next(
            (row for row in previous.get("stage_results", ())
             if int(row.get("stage_index", -1)) == int(task["stage_index"])),
            None,
        )
        if frozen.state.is_finished() and existing_stage is not None:
            existing = dict(existing_stage.get("result") or value)
            durable.append({
                **dict(record),
                "result": existing,
                "valid": frozen.state.name == "COMPLETE" and not existing.get("error"),
                "audit_row": _audit_row_from_payload(
                    trial=frozen, study_name=str(existing.get("study_name", "")), payload=previous, value=existing,
                    state=frozen.state.name, stage_index=int(task["stage_index"]),
                ),
            })
            continue
        stages = [
            row for row in previous.get("stage_results", [])
            if int(row.get("stage_index", -1)) != int(task["stage_index"])
        ]
        stages.append({
            "stage_index": int(task["stage_index"]),
            "stage_fraction": float(task["stage_fraction"]),
            "stage_end": str(task["win_end"]),
            "result": _jsonable(value),
        })
        stages.sort(key=lambda row: int(row["stage_index"]))
        payload = dict(previous)
        payload.update({
            "candidate_id": candidate_id,
            "candidate_order": int(record["candidate_order"]),
            "stage_results": stages,
        })
        if last_stage or not valid:
            payload["result"] = _jsonable(value)
        state = None
        if last_stage or not valid:
            state = "COMPLETE" if valid else "FAIL"
        if state == "FAIL":
            payload["failure_reason"] = str(value.get("error") or "invalid objective")
        mutation = {
            "trial_id": int(trial._trial_id), "payload": payload,
            "constraints": None if str(config.sampler).lower() == "grid" else list(_constraint_values(value, config)),
            "state": state, "objectives": objectives,
        }
        mutations.append(mutation)
        audit_row = _audit_row_from_payload(
            trial=frozen, study_name=str(task.get("study_name", "")), payload=payload, value=value,
            state=state or "RUNNING", stage_index=int(task["stage_index"]),
        )
        if state == "COMPLETE":
            audit_row["values"] = json.dumps(list(objectives), ensure_ascii=False)
        durable.append({
            **dict(record), "result": value, "valid": valid,
            "audit_row": audit_row,
        })
    return mutations, durable


def _persist_result_batch_sqlite(
    optuna: Any,
    config: Any,
    study: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    last_stage: bool,
) -> list[dict[str, Any]]:
    mutations, durable = _prepare_result_mutations(
        optuna, config, study, records, last_stage=last_stage,
    )
    started = time.perf_counter()
    _sqlite_apply_trial_mutations(study._storage, mutations)
    _IO_METRICS["last_sqlite_txn_seconds"] = max(time.perf_counter() - started, 0.0)
    _IO_METRICS["sqlite_txn_count"] = int(_IO_METRICS["sqlite_txn_count"]) + 1
    return durable


def _persist_pruned_batch_sqlite(
    optuna: Any,
    config: Any,
    study: Any,
    records: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    del optuna, config
    payloads = _sqlite_trial_payloads(
        study, [int(record["trial"]._trial_id) for record in records]
    )
    mutations: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for record in records:
        frozen, previous = payloads[int(record["trial"]._trial_id)]
        payload = dict(previous)
        payload["pruning_reason"] = "successive_halving"
        if "result" not in payload:
            payload["result"] = _jsonable(record["result"])
        mutations.append({
            "trial_id": int(record["trial"]._trial_id), "payload": payload,
            "constraints": None, "state": "PRUNED", "objectives": (),
        })
        audit_rows.append(_audit_row_from_payload(
            trial=frozen, study_name=str(record["task"].get("study_name", "")),
            payload=payload, value=record["result"],
            state="PRUNED", stage_index=int(record["task"]["stage_index"]),
        ))
    started = time.perf_counter()
    _sqlite_apply_trial_mutations(
        study._storage, mutations, study=study, checkpoint=checkpoint,
    )
    _IO_METRICS["last_sqlite_txn_seconds"] = max(time.perf_counter() - started, 0.0)
    _IO_METRICS["sqlite_txn_count"] = int(_IO_METRICS["sqlite_txn_count"]) + 1
    return audit_rows


def _persist_study_checkpoint_sqlite(study: Any, values: Mapping[str, Any]) -> None:
    from optuna.storages._rdb import models

    payload = dict(study.user_attrs.get("betalens") or {})
    payload.update(_jsonable(values))
    with _rdb_storage(study._storage).engine.begin() as connection:
        _sqlite_upsert(
            connection,
            models.StudyUserAttributeModel,
            {"study_id": int(study._study_id), "key": "betalens"},
            {"value_json": json.dumps(payload, ensure_ascii=False)},
        )


def _persist_rung_transition_sqlite(
    optuna: Any,
    config: Any,
    study: Any,
    pruned: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return _persist_pruned_batch_sqlite(
        optuna, config, study, pruned, checkpoint=checkpoint,
    )


def _reconcile_pending_batch(
    optuna: Any,
    config: Any,
    study: Any,
    pending_dir: Path,
) -> int:
    """Replay fsynced Grid batch journals that were not ACKed before restart."""
    if str(config.sampler).lower() != "grid" or not pending_dir.exists():
        return 0
    all_trials = list(study.get_trials(deepcopy=False))
    by_candidate: dict[str, list[Any]] = {}
    for trial in all_trials:
        payload = dict(trial.user_attrs.get("betalens") or {})
        candidate_id = str(payload.get("candidate_id") or _candidate_id(trial.params))
        by_candidate.setdefault(candidate_id, []).append(trial)
    replayed = 0
    for path in sorted(pending_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("study_name", "")) != study.study_name:
            continue
        batch_stage_index = int(payload.get("stage_index", -1))
        records = []
        for item in payload.get("records", []):
            params = dict(item.get("params") or {})
            candidate_id = str(item.get("candidate_id") or _candidate_id(params))
            candidates = by_candidate.get(candidate_id, [])
            frozen = next(
                (
                    trial for trial in candidates
                    if trial.state == optuna.trial.TrialState.RUNNING
                ),
                None,
            )
            if frozen is None:
                frozen = next(
                    (
                        trial for trial in candidates
                        if any(
                            int(stage.get("stage_index", -1)) == batch_stage_index
                            for stage in (trial.user_attrs.get("betalens") or {}).get("stage_results", ())
                        )
                    ),
                    None,
                )
            if frozen is None and candidates:
                frozen = max(candidates, key=lambda trial: int(trial.number))
            if frozen is None:
                continue
            trial_number = int(frozen.number)
            records.append({
                "trial": optuna.trial.Trial(study, frozen._trial_id),
                "trial_number": trial_number,
                "params": params,
                "candidate_order": int(item.get("candidate_order", trial_number)),
                "task": dict(item.get("task") or {}),
                "result": dict(item.get("result") or {}),
            })
        if not records:
            continue
        _persist_result_batch_sqlite(
            optuna, config, study, records,
            last_stage=bool(payload.get("last_stage", False)),
        )
        _commit_batch_journal(pending_dir.parent, str(payload.get("batch_id") or path.stem))
        replayed += len(records)
    return replayed


def _persist_result_batch(
    optuna: Any,
    config: Any,
    study: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    last_stage: bool,
) -> list[dict[str, Any]]:
    if (
        str(config.sampler).lower() == "grid"
        and bool(getattr(config, "sqlite_batch_transactions", False))
        and hasattr(_rdb_storage(study._storage), "engine")
    ):
        return _persist_result_batch_sqlite(
            optuna, config, study, records, last_stage=last_stage,
        )
    durable = []
    for record in records:
        trial = record["trial"]
        task = dict(record["task"])
        result = dict(record["result"])
        params = dict(record["params"])
        candidate_id = _candidate_id(params)
        value = {**task, **result, "params": params, "candidate_id": candidate_id}
        value["constraint_violation"] = _normalized_violation(value, config)
        value["feasible"] = value["constraint_violation"] <= 1e-12
        objectives = (value.get("robust_rank_ic"), value.get("sharpe"))
        valid = not value.get("error") and all(
            item is not None and np.isfinite(float(item)) for item in objectives
        )
        frozen = study._storage.get_trial(trial._trial_id)
        payload = dict(frozen.user_attrs.get("betalens") or {})
        existing_stage = next(
            (
                row for row in payload.get("stage_results", ())
                if int(row.get("stage_index", -1)) == int(task["stage_index"])
            ),
            None,
        )
        if frozen.state.is_finished() and existing_stage is not None:
            existing = dict(existing_stage.get("result") or value)
            existing_valid = (
                frozen.state == optuna.trial.TrialState.COMPLETE
                and not existing.get("error")
            )
            durable.append({
                **dict(record), "result": existing, "valid": existing_valid,
            })
            continue
        stages = [
            row for row in payload.get("stage_results", [])
            if int(row.get("stage_index", -1)) != int(task["stage_index"])
        ]
        stages.append({
            "stage_index": int(task["stage_index"]),
            "stage_fraction": float(task["stage_fraction"]),
            "stage_end": str(task["win_end"]),
            "result": _jsonable(value),
        })
        stages.sort(key=lambda row: int(row["stage_index"]))
        payload.update({
            "candidate_id": candidate_id,
            "candidate_order": int(record["candidate_order"]),
            "stage_results": stages,
        })
        if last_stage or not valid:
            payload["result"] = _jsonable(value)
        trial.set_user_attr("betalens", payload)
        if str(config.sampler).lower() != "grid":
            trial.set_user_attr("constraints", list(_constraint_values(value, config)))
        if last_stage or not valid:
            if valid:
                _study_tell(study, trial, values=objectives)
            else:
                payload["failure_reason"] = str(value.get("error") or "invalid objective")
                trial.set_user_attr("betalens", payload)
                _study_tell(study, trial, state=optuna.trial.TrialState.FAIL)
        durable.append({**dict(record), "result": value, "valid": valid})
    return durable


def _persist_pruned_batch(optuna: Any, study: Any, records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        trial = record["trial"]
        frozen = study._storage.get_trial(trial._trial_id)
        payload = dict(frozen.user_attrs.get("betalens") or {})
        payload["pruning_reason"] = "successive_halving"
        if "result" not in payload:
            payload["result"] = _jsonable(record["result"])
        trial.set_user_attr("betalens", payload)
        _study_tell(study, trial, state=optuna.trial.TrialState.PRUNED)


def _persist_window_checkpoint(study: Any, values: Mapping[str, Any]) -> None:
    payload = dict(study.user_attrs.get("betalens") or {})
    payload.update(_jsonable(values))
    study.set_user_attr("betalens", payload)


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
    name = _study_name(
        str(getattr(config, "search_hash", config.config_hash)),
        phase,
        scheme,
        train_start,
        train_end,
    )
    study = _make_study(optuna, config, storage, name, logger)
    remaining = _recover_and_queue(optuna, study, config, logger, audit_dir / "events.jsonl")
    batch_size = max(
        1,
        min(
            int(getattr(config, "grid_batch_max_candidates", 8)),
            int(getattr(config, "workers", 1)),
        ),
    )
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


def _candidate_id(params: Mapping[str, Any]) -> str:
    return hashlib.sha256(_parameter_key(params).encode("ascii")).hexdigest()[:20]


def _available_ratio() -> float:
    total, available = core._system_memory_snapshot()
    if not total or available is None:
        return 1.0
    return max(0.0, min(1.0, float(available) / float(total)))


def _cpu_percent() -> float | None:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=None))
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class FILETIME(ctypes.Structure):
                _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

            idle = FILETIME()
            kernel = FILETIME()
            user = FILETIME()
            if not ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            ):
                return None
            to_int = lambda value: (int(value.high) << 32) | int(value.low)
            current = (to_int(kernel) + to_int(user), to_int(idle))
            global _CPU_TIMES_LAST
            previous = _CPU_TIMES_LAST
            _CPU_TIMES_LAST = current
            if previous is None:
                return None
            total_delta = current[0] - previous[0]
            idle_delta = current[1] - previous[1]
            return 100.0 * max(0, total_delta - idle_delta) / total_delta if total_delta > 0 else None
        except Exception:
            return None
    try:
        return min(100.0, 100.0 * float(os.getloadavg()[0]) / max(1, os.cpu_count() or 1))
    except (AttributeError, OSError):
        return None


def _process_cpu_percent() -> float | None:
    global _PROCESS_CPU_LAST
    try:
        process_time = time.process_time()
        wall_time = time.monotonic()
        previous = _PROCESS_CPU_LAST
        _PROCESS_CPU_LAST = (process_time, wall_time)
        if previous is None or wall_time <= previous[1]:
            return None
        return max(0.0, 100.0 * (process_time - previous[0]) / (wall_time - previous[1]))
    except Exception:
        return None


def _resource_capacity(
    config: Any,
    cache_paths: Any,
    measured_private_bytes: int = 0,
) -> tuple[int, dict[str, Any]]:
    requested = max(1, min(int(getattr(config, "max_workers", config.workers)), int(config.workers)))
    minimum = max(1, min(int(getattr(config, "min_workers", requested)), requested))
    estimated = int(core._estimate_worker_memory_bytes(config, cache_paths))
    per_worker = int(max(estimated, int(measured_private_bytes)) * 1.25)
    resources = core._system_resource_snapshot()
    physical_available = resources["physical_available"]
    commit_available = resources["commit_available"]
    physical_reserve = 8 * 1024**3
    physical_capacity = requested
    if physical_available is not None:
        physical_capacity = max(0, int(max(0, physical_available - physical_reserve) // per_worker))
    commit_capacity = requested
    if commit_available is not None:
        commit_reserve = int((resources["commit_total"] or 0) * 0.15)
        commit_capacity = max(0, int(max(0, commit_available - commit_reserve) // per_worker))
    capacity = min(requested, physical_capacity, commit_capacity)
    return capacity, {
        **resources,
        "requested_workers": requested,
        "minimum_workers": minimum,
        "per_worker": per_worker,
        "measured_private_bytes": int(measured_private_bytes),
        "physical_reserve": physical_reserve,
        "capacity": capacity,
    }


def _wait_for_worker_capacity(config: Any, cache_paths: Any, logger: logging.Logger) -> int:
    measured_private_bytes = 0
    try:
        with ProcessPoolExecutor(max_workers=1) as probe_executor:
            measured_private_bytes = int(
                probe_executor.submit(
                    core._worker_private_bytes_probe,
                    str(cache_paths.data),
                    str(cache_paths.pit),
                ).result(timeout=120)
            )
    except Exception as exc:
        logger.warning("RESOURCE PROBE failed; using static estimate error=%s", exc)
    wait_seconds = max(1, int(getattr(config, "resource_check_seconds", 15)))
    deadline = time.monotonic() + max(0, int(getattr(config, "resource_wait_minutes", 30))) * 60
    while True:
        capacity, diagnostics = _resource_capacity(
            config, cache_paths, measured_private_bytes
        )
        requested = int(diagnostics["requested_workers"])
        minimum = int(diagnostics["minimum_workers"])
        if capacity >= minimum:
            return min(requested, max(minimum, capacity))
        if time.monotonic() >= deadline:
            raise MemoryError(
                "resources did not reach the hard minimum within the configured wait: "
                f"workers={minimum} capacity={capacity} "
                f"physical_available={core._format_bytes(diagnostics['physical_available'])} "
                f"commit_available={core._format_bytes(diagnostics['commit_available'])} "
                f"per_worker={core._format_bytes(diagnostics['per_worker'])} "
                f"cache={cache_paths.data}"
            )
        logger.warning(
            "RESOURCE WAIT workers=%d capacity=%d physical_available=%s commit_available=%s "
            "per_worker=%s measured_private=%s cache=%s retry=%ss",
            minimum,
            capacity,
            core._format_bytes(diagnostics["physical_available"]),
            core._format_bytes(diagnostics["commit_available"]),
            core._format_bytes(diagnostics["per_worker"]),
            core._format_bytes(diagnostics["measured_private_bytes"]),
            cache_paths.data,
            wait_seconds,
        )
        time.sleep(wait_seconds)


def _worker_slot_path(config: Any) -> Path:
    root = Path(config.runtime_root).expanduser() if getattr(config, "runtime_root", None) else _default_runtime_root()
    return root / ".worker-slots.lock"


def _acquire_worker_slots(config: Any, workers: int, logger: logging.Logger) -> dict[str, Any]:
    path = _worker_slot_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    wait_seconds = max(1, int(getattr(config, "resource_check_seconds", 15)))
    deadline = time.monotonic() + max(0, int(getattr(config, "resource_wait_minutes", 30))) * 60
    owner = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "process_started_at": _process_started_at(os.getpid()),
        "workers": int(workers),
        "started_at": _now(),
        "owner_token": uuid.uuid4().hex,
    }
    while True:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(owner, stream)
                stream.flush()
                os.fsync(stream.fileno())
            owner["path"] = str(path)
            return owner
        except FileExistsError:
            try:
                holder = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                holder = {"pid": -1, "error": "unreadable"}
            if not _lock_owner_active(holder):
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError(f"worker slots remained occupied for the configured wait: {holder}")
            logger.warning("WORKER SLOT WAIT holder=%s retry=%ss", holder, wait_seconds)
            time.sleep(wait_seconds)


def _release_worker_slots(owner: Mapping[str, Any] | None) -> None:
    if not owner:
        return
    path = Path(str(owner["path"]))
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if current.get("owner_token") == owner.get("owner_token"):
        path.unlink(missing_ok=True)


def _heartbeat_lock(path: Path, owner: Mapping[str, Any]) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if current.get("owner_token") != owner.get("owner_token"):
        raise RuntimeError(f"coordinator lock ownership changed while running: {path}")
    current["heartbeat_at"] = _now()
    _atomic_json(path, current)


def _heartbeat_worker_slots(owner: Mapping[str, Any] | None) -> None:
    if not owner:
        return
    path = Path(str(owner["path"]))
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if current.get("owner_token") != owner.get("owner_token"):
        raise RuntimeError(f"worker slot ownership changed while running: {path}")
    current["heartbeat_at"] = _now()
    _atomic_json(path, current)


def _task_result(
    optuna: Any,
    config: Any,
    study: Any,
    trial: Any,
    params: Mapping[str, Any],
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    logger: logging.Logger,
    audit_dir: Path,
) -> dict[str, Any]:
    value = {**dict(task), **dict(result), "params": dict(params)}
    value["candidate_id"] = _candidate_id(params)
    value["constraint_violation"] = _normalized_violation(value, config)
    value["feasible"] = value["constraint_violation"] <= 1e-12
    constraints = _constraint_values(value, config)
    _db_retry(
        lambda: trial.set_user_attr("candidate_id", value["candidate_id"]),
        logger,
        "set-candidate-id",
    )
    _db_retry(lambda: trial.set_user_attr("result", _jsonable(value)), logger, "set-result")
    _db_retry(lambda: trial.set_user_attr("constraints", list(constraints)), logger, "set-constraints")
    error = value.get("error")
    objectives = (value.get("robust_rank_ic"), value.get("sharpe"))
    valid = not error and all(item is not None and np.isfinite(float(item)) for item in objectives)
    if valid:
        _db_retry(lambda: _study_tell(study, trial, values=objectives), logger, "tell-complete")
    else:
        reason = str(error or "invalid objective")
        _db_retry(lambda: trial.set_user_attr("failure_reason", reason), logger, "set-failure-reason")
        _db_retry(lambda: _study_tell(study, trial, state=optuna.trial.TrialState.FAIL), logger, "tell-fail")
        _append_jsonl(audit_dir / "errors.jsonl", {
            "time": _now(), "event": "TRIAL_ERROR", "study": study.study_name,
            "trial": trial.number, "candidate_id": value["candidate_id"],
            "params": params, "error": reason, "traceback_path": value.get("task_log_path"),
        })
    _append_jsonl(audit_dir / "events.jsonl", {
        "time": _now(), "event": "END", "run": config.config_hash,
        "study": study.study_name, "trial": trial.number,
        "candidate_id": value["candidate_id"], "gid": value.get("gid"),
        "params": params, "window": [task["win_start"], task["win_end"]], "phase": "train",
        "metrics": {key: value.get(key) for key in ("robust_rank_ic", "sharpe", "mdd", "turnover", "ic_coverage")},
        "constraints": constraints, "constraint_violation": value["constraint_violation"],
        "feasible": value["feasible"], "status": "COMPLETE" if valid else "FAIL",
        "error": error, "traceback_path": value.get("task_log_path"),
    })
    _log_trial(logger, "END", value)
    return value


def _prune_trial(
    optuna: Any,
    config: Any,
    study: Any,
    record: Mapping[str, Any],
    logger: logging.Logger,
    audit_dir: Path,
) -> None:
    trial = record["trial"]
    params = dict(record["params"])
    result = {**dict(record["task"]), **dict(record["result"]), "params": params}
    result["candidate_id"] = _candidate_id(params)
    result["constraint_violation"] = _normalized_violation(result, config)
    result["feasible"] = result["constraint_violation"] <= 1e-12
    result["pruned"] = True
    result["pruning_reason"] = "successive_halving"
    constraints = _constraint_values(result, config)
    _db_retry(lambda: trial.set_user_attr("candidate_id", result["candidate_id"]), logger, "set-candidate-id")
    _db_retry(lambda: trial.set_user_attr("result", _jsonable(result)), logger, "set-pruned-result")
    _db_retry(lambda: trial.set_user_attr("constraints", list(constraints)), logger, "set-pruned-constraints")
    _db_retry(lambda: trial.set_user_attr("pruning_reason", "successive_halving"), logger, "set-pruning-reason")
    _db_retry(lambda: _study_tell(study, trial, state=optuna.trial.TrialState.PRUNED), logger, "tell-pruned")
    event = {
        "time": _now(), "event": "PRUNED", "run": config.config_hash,
        "study": study.study_name, "trial": trial.number,
        "candidate_id": result["candidate_id"], "params": params,
        "stage_index": result.get("stage_index"), "stage_fraction": result.get("stage_fraction"),
        "stage_end": result.get("win_end"),
        "metrics": {key: result.get(key) for key in ("robust_rank_ic", "sharpe", "mdd", "turnover", "ic_coverage")},
        "constraint_violation": result["constraint_violation"],
        "reason": "successive_halving",
    }
    _append_jsonl(audit_dir / "events.jsonl", event)
    logger.info(
        "PRUNED study=%s trial=%d stage=%s fraction=%s params=%s IC=%s Sharpe=%s violation=%s",
        study.study_name, trial.number, result.get("stage_index"), result.get("stage_fraction"),
        _parameter_key(params), result.get("robust_rank_ic"), result.get("sharpe"),
        result["constraint_violation"],
    )


def _persist_stage_result(
    trial: Any,
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    logger: logging.Logger,
) -> None:
    payload = {
        "stage_index": int(task["stage_index"]),
        "stage_fraction": float(task["stage_fraction"]),
        "stage_end": str(task["win_end"]),
        "metrics": {
            key: result.get(key)
            for key in ("robust_rank_ic", "sharpe", "mdd", "turnover", "ic_coverage")
        },
        "elapsed_seconds": result.get("elapsed_seconds"),
        "worker_pid": result.get("worker_pid"),
    }
    previous = list(trial.user_attrs.get("stage_results") or ())
    previous = [row for row in previous if int(row.get("stage_index", -1)) != payload["stage_index"]]
    previous.append(payload)
    previous.sort(key=lambda row: int(row["stage_index"]))
    _db_retry(lambda: trial.set_user_attr("stage_results", _jsonable(previous)), logger, "set-stage-results")


def _run_multiwindow_scheduler_legacy(
    *,
    optuna: Any,
    config: Any,
    storage: Any,
    cache_paths: Any,
    executor: ProcessPoolExecutor,
    logger: logging.Logger,
    audit_dir: Path,
    windows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    oos_rows: list[dict[str, Any]],
    status: dict[str, Any],
    database_path: Path | None,
    snapshot_path: Path,
    mirror_callback: Callable[[], None],
    legacy_index: Mapping[tuple[str, str, str], str] | None = None,
    heartbeat_callback: Callable[[], None] | None = None,
    snapshot_callback: Callable[[], None] | None = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame | None]:
    stages = _pruning_stages(config)
    trials_by_study: dict[str, pd.DataFrame] = {}
    completed_oos = {
        (str(row.get("scheme")), str(row.get("train_start")), str(row.get("train_end")),
         str(row.get("test_start")), str(row.get("test_end")))
        for row in oos_rows
    }
    waiting = deque()
    for window in windows:
        window.update({
            "window_id": hashlib.sha1(
                f"{window['scheme']}|{window['train_start']}|{window['train_end']}".encode("utf-8")
            ).hexdigest()[:16],
            "study": None,
            "study_name": None,
            "remaining": None,
            "inflight": 0,
            "stage_index": 0,
            "stage_queue": deque(),
            "rung_records": [],
            "state": "WAITING_TRAIN",
            "ordered": None,
            "locked_params": None,
            "locked_trial": None,
        })
        waiting.append(window)

    max_active = max(1, int(getattr(config, "max_active_windows", config.workers)))
    max_inflight = max(1, min(int(getattr(config, "max_inflight_batches", config.workers)), int(config.workers)))
    test_cap = max(1, int(math.floor(max_inflight * 0.20)))
    per_study = 1
    active: deque[dict[str, Any]] = deque()
    tests_waiting: deque[dict[str, Any]] = deque()
    futures: dict[Any, dict[str, Any]] = {}
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    final_ordered: pd.DataFrame | None = None
    final_train_end: pd.Timestamp | None = None
    completed_since_flush = 0
    last_flush = time.monotonic()
    last_mirror = time.monotonic()
    last_snapshot = time.monotonic()
    last_heartbeat = time.monotonic()
    scheduler_started = time.monotonic()
    completed_train_tasks = 0
    db_write_seconds = 0.0
    audit_write_seconds = 0.0
    paused = False

    def activate() -> None:
        while waiting and len(active) < max_active:
            window = waiting.popleft()
            key = (window["scheme"], window["train_start"], window["train_end"])
            name = (legacy_index or {}).get(key) or _study_name(
                str(getattr(config, "search_hash", config.config_hash)),
                "train", window["scheme"], window["train_start"], window["train_end"]
            )
            study = _make_study(optuna, config, storage, name, logger)
            remaining = _recover_and_queue(optuna, study, config, logger, audit_dir / "events.jsonl")
            stage_index = int(study.user_attrs.get("current_stage_index", 0) or 0)
            if stage_index >= len(stages):
                raise RuntimeError(f"invalid persisted pruning stage: study={name} stage={stage_index}")
            window.update({
                "study": study, "study_name": name, "remaining": remaining,
                "stage_index": stage_index,
            })
            trials_by_study[name] = _trial_result_frame(study)
            window["state"] = "RUNNING_TRAIN"
            active.append(window)

    def lock_finished(window: dict[str, Any]) -> None:
        nonlocal final_ordered, final_train_end
        frame = _trial_result_frame(window["study"])
        ordered, reason = _selection_table(frame, config)
        trials_by_study[window["study_name"]] = frame
        train_frames.append(frame)
        window_front = ordered.assign(
            study_name=window["study_name"], phase="train", window_id=window["window_id"]
        ).to_dict("records")
        pareto_rows[:] = [
            row for row in pareto_rows if row.get("study_name") != window["study_name"]
        ] + window_front
        locked = ordered.iloc[0]
        window["ordered"] = ordered
        window["locked_params"] = json.loads(str(locked["params_json"]))
        window["locked_trial"] = int(locked["trial_number"])
        locked_at = _now()
        _db_retry(
            lambda: window["study"].set_user_attr("locked_params", window["locked_params"]),
            logger,
            "persist-locked-params",
        )
        _db_retry(
            lambda: window["study"].set_user_attr("locked_trial", window["locked_trial"]),
            logger,
            "persist-locked-trial",
        )
        _db_retry(
            lambda: window["study"].set_user_attr("locked_at", locked_at),
            logger,
            "persist-locked-at",
        )
        _db_retry(
            lambda: window["study"].set_user_attr("selection_reason", reason),
            logger,
            "persist-selection-reason",
        )
        window["state"] = "LOCKED"
        logger.info(
            "LOCK window=%s study=%s trial=%d reason=%s params=%s",
            window["window_id"], window["study_name"], window["locked_trial"], reason,
            locked["params_json"],
        )
        train_end = pd.Timestamp(window["train_end"])
        if final_train_end is None or train_end > final_train_end:
            final_ordered = ordered
            final_train_end = train_end
        key = (
            window["scheme"], window["train_start"], window["train_end"],
            window["test_start"], window["test_end"],
        )
        if key in completed_oos:
            window["state"] = "COMPLETE"
        else:
            tests_waiting.append(window)

    def submit_train(window: dict[str, Any]) -> bool:
        if window["remaining"] <= 0 or window["inflight"] >= per_study:
            return False
        if window["stage_queue"]:
            record = window["stage_queue"].popleft()
            trial, params = record["trial"], dict(record["params"])
        else:
            trial, params = _ask_params(window["study"], config.grid, logger)
        gid = core._resolve_gid(config.factor_module, config.gid_factory, params)
        candidate_id = _candidate_id(params)
        stage_index = int(window["stage_index"])
        stage_fraction = float(stages[stage_index])
        stage_end = _pruning_stage_end(window["train_start"], window["train_end"], stage_fraction)
        task = _task_dict(
            config, params, gid,
            phase="train", scheme=window["scheme"],
            win_start=window["train_start"], win_end=stage_end,
            train_start=window["train_start"], train_end=window["train_end"],
            test_start=window["test_start"], test_end=window["test_end"],
            trial_number=trial.number, study_name=window["study_name"],
            window_id=window["window_id"], candidate_id=candidate_id,
            stage_index=stage_index, stage_fraction=stage_fraction,
            task_log_path=str(
                audit_dir / "task_logs" / window["study_name"] / str(trial.number)
                / f"train_stage_{stage_index}.log"
            ),
        )
        future = executor.submit(core._run_one_task, task)
        futures[future] = {
            "kind": "train", "window": window, "trial": trial, "params": params, "task": task,
        }
        window["inflight"] += 1
        event = {
            "time": _now(), "event": "START", "run": config.config_hash,
            "study": window["study_name"], "trial": trial.number,
            "window_id": window["window_id"], "candidate_id": candidate_id,
            "gid": gid, "params": params, "phase": "train", "scheme": window["scheme"],
            "train": [window["train_start"], window["train_end"]],
            "test": [window["test_start"], window["test_end"]],
            "stage_index": stage_index, "stage_fraction": stage_fraction, "stage_end": stage_end,
        }
        _append_jsonl(audit_dir / "events.jsonl", event)
        if str(config.sampler).lower() != "grid":
            candidate_rows.append({
                "study_name": window["study_name"], "trial_number": trial.number,
                "window_id": window["window_id"], "candidate_id": candidate_id,
                "asked_at": event["time"], "gid": gid, "params_json": _parameter_key(params), **params,
            })
        _log_trial(logger, "START", {**task, "params": params})
        return True

    def submit_test(window: dict[str, Any]) -> None:
        task = _oos_task(
            config, window["locked_params"], phase="test", scheme=window["scheme"],
            start=window["test_start"], end=window["test_end"],
            trial_number=window["locked_trial"], audit_dir=audit_dir,
        )
        task.update({
            "window_id": window["window_id"], "train_start": window["train_start"],
            "train_end": window["train_end"], "test_start": window["test_start"],
            "test_end": window["test_end"],
            "task_log_path": str(audit_dir / "task_logs" / window["study_name"] / "test.log"),
        })
        future = executor.submit(core._run_one_task, task)
        futures[future] = {"kind": "test", "window": window, "task": task}
        window["state"] = "RUNNING_TEST"
        _log_trial(logger, "START", {**task, "params": window["locked_params"]})

    activate()
    for window in list(active):
        if window["remaining"] == 0:
            lock_finished(window)
            active.remove(window)
    activate()

    while active or waiting or tests_waiting or futures:
        now = time.monotonic()
        ratio = _available_ratio()
        low = float(getattr(config, "memory_low_watermark_ratio", 0.15))
        high = float(getattr(config, "memory_resume_watermark_ratio", 0.20))
        if not paused and ratio < low:
            paused = True
            logger.warning("RESOURCE PAUSE available_ratio=%.1f%% active=%d", ratio * 100, len(futures))
        elif paused and ratio >= high:
            paused = False
            logger.info("RESOURCE RESUME available_ratio=%.1f%%", ratio * 100)

        if not paused:
            active_tests = sum(meta["kind"] == "test" for meta in futures.values())
            while tests_waiting and len(futures) < max_inflight and active_tests < test_cap:
                submit_test(tests_waiting.popleft())
                active_tests += 1

            made_progress = True
            while len(futures) < max_inflight and made_progress:
                made_progress = False
                for _ in range(len(active)):
                    window = active[0]
                    active.rotate(-1)
                    if submit_train(window):
                        made_progress = True
                        if len(futures) >= max_inflight:
                            break
            if str(config.sampler).lower() == "grid" and len(futures) < max_inflight:
                # When fewer windows remain than worker slots, let Grid studies
                # use spare capacity without changing the one-inflight MOTPE rule.
                for window in list(active):
                    while (
                        len(futures) < max_inflight
                        and window["remaining"] - window["inflight"] > 0
                    ):
                        original_limit = per_study
                        per_study = max_inflight
                        try:
                            if not submit_train(window):
                                break
                        finally:
                            per_study = original_limit

        if not futures:
            if paused:
                time.sleep(max(1, int(getattr(config, "resource_check_seconds", 15))))
                continue
            break

        finished, _pending = wait(tuple(futures), timeout=1.0, return_when=FIRST_COMPLETED)
        for future in finished:
            meta = futures.pop(future)
            window = meta["window"]
            try:
                result = future.result()
            except BaseException as exc:
                result = {**meta["task"], "error": f"{type(exc).__name__}: {exc}"}
            if meta["kind"] == "train":
                window["inflight"] -= 1
                db_started = time.perf_counter()
                last_stage = int(window["stage_index"]) == len(stages) - 1
                error = result.get("error")
                objectives = (result.get("robust_rank_ic"), result.get("sharpe"))
                valid = not error and all(
                    item is not None and np.isfinite(float(item)) for item in objectives
                )
                if valid:
                    _persist_stage_result(meta["trial"], meta["task"], result, logger)
                if last_stage or not valid:
                    _task_result(
                        optuna, config, window["study"], meta["trial"], meta["params"],
                        meta["task"], result, logger, audit_dir,
                    )
                    trials_by_study[window["study_name"]] = _trial_result_frame(window["study"])
                else:
                    window["rung_records"].append({**meta, "result": dict(result)})
                    _append_jsonl(audit_dir / "events.jsonl", {
                        "time": _now(), "event": "STAGE_END", "run": config.config_hash,
                        "study": window["study_name"], "trial": meta["trial"].number,
                        "candidate_id": _candidate_id(meta["params"]), "params": meta["params"],
                        "stage_index": meta["task"]["stage_index"],
                        "stage_fraction": meta["task"]["stage_fraction"],
                        "stage_end": meta["task"]["win_end"],
                        "metrics": {key: result.get(key) for key in (
                            "robust_rank_ic", "sharpe", "mdd", "turnover", "ic_coverage"
                        )},
                    })
                    _log_trial(logger, "STAGE END", {**meta["task"], **dict(result), "params": meta["params"]})
                db_write_seconds += time.perf_counter() - db_started
                completed_train_tasks += 1
                window["remaining"] -= 1
                completed_since_flush += 1
                if window["remaining"] == 0 and window["inflight"] == 0:
                    if not last_stage:
                        survivors, pruned = _rung_survivors(
                            window["rung_records"], config, paper_params=config.paper_params
                        )
                        if not survivors:
                            raise RuntimeError(
                                f"pruning stage produced no valid survivors: study={window['study_name']} "
                                f"stage={window['stage_index']}"
                            )
                        for record in pruned:
                            _prune_trial(optuna, config, window["study"], record, logger, audit_dir)
                        next_stage = int(window["stage_index"]) + 1
                        survivor_keys = [_parameter_key(record["params"]) for record in survivors]
                        _db_retry(
                            lambda: window["study"].set_user_attr("active_candidate_keys", survivor_keys),
                            logger,
                            "persist-active-candidates",
                        )
                        _db_retry(
                            lambda: window["study"].set_user_attr("current_stage_index", next_stage),
                            logger,
                            "persist-current-stage",
                        )
                        logger.info(
                            "RUNG study=%s stage=%d fraction=%.2f evaluated=%d survivors=%d pruned=%d next_fraction=%.2f",
                            window["study_name"], window["stage_index"], stages[window["stage_index"]],
                            len(window["rung_records"]), len(survivors), len(pruned), stages[next_stage],
                        )
                        _append_jsonl(audit_dir / "events.jsonl", {
                            "time": _now(), "event": "RUNG_END", "study": window["study_name"],
                            "stage_index": window["stage_index"],
                            "stage_fraction": stages[window["stage_index"]],
                            "evaluated": len(window["rung_records"]),
                            "survivors": len(survivors), "pruned": len(pruned),
                            "survivor_candidate_ids": [_candidate_id(record["params"]) for record in survivors],
                        })
                        window["stage_index"] = next_stage
                        window["stage_queue"] = deque(survivors)
                        window["rung_records"] = []
                        window["remaining"] = len(survivors)
                        trials_by_study[window["study_name"]] = _trial_result_frame(window["study"])
                    else:
                        current_frame = _trial_result_frame(window["study"])
                        trials_by_study[window["study_name"]] = current_frame
                        lock_finished(window)
                        try:
                            active.remove(window)
                        except ValueError:
                            pass
                        activate()
            else:
                params = window["locked_params"]
                value = {**meta["task"], **dict(result), "params": params}
                logger.debug(
                    "TASK trial=%s worker_pid=%s wide=%sx%s weights=%sx%s engine=%s "
                    "elapsed=%s log=%s",
                    window["locked_trial"], value.get("worker_pid"), value.get("wide_rows"),
                    value.get("wide_columns"), value.get("weight_rows"),
                    value.get("weight_columns"), value.get("engine", meta["task"].get("engine")),
                    value.get("elapsed_seconds"), value.get("task_log_path"),
                )
                changed = bool(oos_rows) and oos_rows[-1].get("params_json") != _parameter_key(params)
                oos = {
                    "window_id": window["window_id"], "scheme": window["scheme"],
                    "train_start": window["train_start"], "train_end": window["train_end"],
                    "test_start": window["test_start"], "test_end": window["test_end"],
                    "trial_number": window["locked_trial"], "gid": value.get("gid"),
                    "candidate_id": _candidate_id(params), "params_json": _parameter_key(params),
                    "parameter_changed": changed,
                    **{key: value.get(key) for key in (
                        "robust_rank_ic", "mean_rank_ic", "sharpe", "mdd", "turnover",
                        "ic_coverage", "error", "task_log_path", "worker_pid",
                    )},
                }
                oos_rows.append(oos)
                test_frames.append(pd.DataFrame([value]))
                window["state"] = "COMPLETE"
                _append_jsonl(audit_dir / "events.jsonl", {"time": _now(), "event": "TEST_END", **oos})
                logger.info(
                    "TEST END window=%s locked_trial=%d params=%s metrics=%s",
                    window["window_id"], window["locked_trial"], _parameter_key(params), _parameter_key(oos),
                )

        now = time.monotonic()
        refresh_seconds = max(1, int(getattr(config, "partial_refresh_seconds", 60)))
        refresh_trials = max(1, int(getattr(config, "partial_refresh_trials", 250)))
        if completed_since_flush >= refresh_trials or now - last_flush >= refresh_seconds:
            audit_started = time.perf_counter()
            partial_trials = pd.concat(list(trials_by_study.values()), ignore_index=True) if trials_by_study else pd.DataFrame()
            _atomic_csv(audit_dir / "trials.partial.csv", partial_trials)
            _atomic_csv(audit_dir / "pareto_front.partial.csv", pd.DataFrame(pareto_rows))
            _atomic_csv(audit_dir / "oos_parameter_path.partial.csv", pd.DataFrame(oos_rows))
            if str(config.sampler).lower() != "grid":
                _atomic_csv(audit_dir / "candidate_manifest.csv", pd.DataFrame(candidate_rows))
            state_counts = {}
            if trials_by_study:
                states = pd.concat(list(trials_by_study.values()), ignore_index=True).get("state")
                if states is not None:
                    state_counts = {str(key): int(value) for key, value in states.value_counts().items()}
            status.update({
                "completed_trials": int(state_counts.get("COMPLETE", 0)),
                "pruned_trials": int(state_counts.get("PRUNED", 0)),
                "failed_trials": int(state_counts.get("FAIL", 0)),
                "completed_stage_tasks": int(completed_train_tasks),
                "active_workers": len(futures),
                "active_windows": len({meta["window"]["window_id"] for meta in futures.values()}),
                "waiting_windows": len(waiting),
                "active_window_ids": sorted({
                    meta["window"]["window_id"] for meta in futures.values()
                }),
                "active_trials": sorted(
                    int(meta["trial"].number)
                    for meta in futures.values()
                    if meta["kind"] == "train"
                ),
            })
            _atomic_json(audit_dir / "status.json", status)
            _append_jsonl(audit_dir / "events.jsonl", {"time": _now(), "event": "AUDIT_FLUSH"}, sync=True)
            mirror_callback()
            last_mirror = now
            completed_since_flush = 0
            last_flush = now
            audit_write_seconds += time.perf_counter() - audit_started
        if now - last_mirror >= max(1, int(getattr(config, "audit_mirror_seconds", 60))):
            mirror_callback()
            last_mirror = now
        if database_path and now - last_snapshot >= max(1, int(getattr(config, "snapshot_seconds", 300))):
            if snapshot_callback:
                snapshot_callback()
                status["last_snapshot_requested_at"] = _now()
            last_snapshot = now
        if now - last_heartbeat >= 10:
            if heartbeat_callback:
                heartbeat_callback()
            resources = core._system_resource_snapshot()
            elapsed = max(time.monotonic() - scheduler_started, 1e-9)
            cpu_percent = _cpu_percent()
            logger.info(
                "HEARTBEAT workers=%d/%d active_windows=%d train_windows=%d tests_waiting=%d "
                "waiting_windows=%d queue_depth=%d memory_available=%.1f%% "
                "commit_available=%s stage_task_per_second=%.4f complete=%d pruned=%d cpu=%.1f%% "
                "db_write_seconds=%.3f audit_seconds=%.3f paused=%s",
                len(futures), max_inflight,
                len({meta["window"]["window_id"] for meta in futures.values()}),
                len(active), len(tests_waiting), len(waiting),
                len(active) + len(tests_waiting) + len(waiting),
                ratio * 100, core._format_bytes(resources["commit_available"]),
                completed_train_tasks / elapsed,
                int(sum((frame["state"] == "COMPLETE").sum() for frame in trials_by_study.values() if "state" in frame)),
                int(sum((frame["state"] == "PRUNED").sum() for frame in trials_by_study.values() if "state" in frame)),
                float("nan") if cpu_percent is None else cpu_percent,
                db_write_seconds, audit_write_seconds, paused,
            )
            last_heartbeat = now

    return train_frames, test_frames, final_ordered


def _candidate_records_from_study(
    optuna: Any,
    study: Any,
    candidates: Sequence[Mapping[str, Any]],
    stage_index: int,
    candidate_orders: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {_parameter_key(trial.params): trial for trial in study.get_trials(deepcopy=False)}
    pending = []
    completed = []
    for local_order, params in enumerate(candidates):
        key = _parameter_key(params)
        frozen = by_key.get(key)
        payload = dict(frozen.user_attrs.get("betalens") or {}) if frozen is not None else {}
        stage_rows = list(payload.get("stage_results") or ())
        matching = next(
            (row for row in stage_rows if int(row.get("stage_index", -1)) == int(stage_index)),
            None,
        )
        if (
            frozen is not None
            and frozen.state == optuna.trial.TrialState.FAIL
            and payload.get("failure_reason") == "stale_after_restart"
        ):
            matching = None
        candidate_order = int(
            payload.get(
                "candidate_order",
                (candidate_orders or {}).get(key, local_order),
            )
        )
        reusable = frozen is not None and frozen.state in {
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.COMPLETE,
        } and (
            not frozen.state.is_finished() or matching is not None
        )
        record = {
            "params": dict(params),
            "candidate_order": candidate_order,
            "trial": (
                optuna.trial.Trial(study, frozen._trial_id) if reusable else None
            ),
            "trial_number": int(frozen.number) if reusable else None,
            "recovery_attrs": (
                {"stage_results": stage_rows, "candidate_id": payload.get("candidate_id")}
                if frozen is not None and not reusable else {}
            ),
        }
        if matching is None:
            pending.append(record)
        else:
            record["result"] = dict(matching.get("result") or {})
            completed.append(record)
    return pending, completed


def _prepare_window_study(
    optuna: Any,
    config: Any,
    storage: Any,
    name: str,
    logger: logging.Logger,
    grid_candidates: Sequence[Mapping[str, Any]],
    last_stage_index: int,
    event_path: Path,
    pending_dir: Path | None = None,
) -> dict[str, Any]:
    """Load one window entirely on the storage owner thread."""
    study = _make_study(optuna, config, storage, name, logger)
    checkpoint = dict(study.user_attrs.get("betalens") or {})
    stage_index = int(checkpoint.get("current_stage_index", 0))
    reconciled = _reconcile_running_last_stage(
        optuna, config, study, stage_index, last_stage_index
    )
    if pending_dir is not None:
        reconciled += _reconcile_pending_batch(optuna, config, study, pending_dir)
    if str(config.sampler).lower() == "grid":
        orders = {
            _parameter_key(params): index
            for index, params in enumerate(grid_candidates)
        }
        candidate_keys = set(checkpoint.get("active_candidate_keys") or ())
        _mark_grid_running_stale(optuna, study, event_path)
        candidates = list(grid_candidates)
        if candidate_keys:
            candidates = [
                params for params in candidates
                if _parameter_key(params) in candidate_keys
            ]
        pending, completed = _candidate_records_from_study(
            optuna, study, candidates, stage_index, orders
        )
        remaining = 0
    else:
        remaining = _recover_and_queue(
            optuna, study, config, logger, event_path
        )
        candidates, pending, completed = [], [], []
    return {
        "study": study,
        "study_name": name,
        "checkpoint": checkpoint,
        "stage_index": stage_index,
        "candidates": candidates,
        "pending": pending,
        "records": completed,
        "remaining_motpe": remaining,
        "reconciled": reconciled,
    }


def _ask_motpe_record(
    study: Any,
    search_space: Mapping[str, Sequence[Any]],
    logger: logging.Logger,
) -> dict[str, Any]:
    trial, params = _ask_params(study, search_space, logger)
    return {
        "trial": trial,
        "trial_number": int(trial.number),
        "params": params,
        "candidate_order": int(trial.number),
    }


def _persist_rung_transition(
    optuna: Any,
    study: Any,
    pruned: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
) -> None:
    _persist_pruned_batch(optuna, study, pruned)
    _persist_window_checkpoint(study, checkpoint)


def _lock_window_study(study: Any, config: Any) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    frame = _trial_result_frame(study)
    ordered, reason = _selection_table(frame, config)
    locked = ordered.iloc[0]
    _persist_window_checkpoint(study, {
        "locked_params": json.loads(str(locked["params_json"])),
        "locked_trial": int(locked["trial_number"]),
        "locked_at": _now(),
        "selection_reason": reason,
    })
    return frame, ordered, reason


def _adaptive_batch_size(config: Any, ewma_seconds: float | None) -> int:
    maximum = max(1, int(getattr(config, "grid_batch_max_candidates", 8)))
    target = max(0.1, float(getattr(config, "grid_batch_target_seconds", 10.0)))
    if ewma_seconds is None or ewma_seconds <= 0:
        return maximum
    return max(1, min(maximum, int(math.floor(target / ewma_seconds))))


def _run_multiwindow_scheduler(
    *,
    optuna: Any,
    config: Any,
    storage: Any,
    cache_paths: Any,
    executor: ProcessPoolExecutor,
    logger: logging.Logger,
    audit_dir: Path,
    windows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    oos_rows: list[dict[str, Any]],
    status: dict[str, Any],
    database_path: Path | None,
    snapshot_path: Path,
    mirror_callback: Callable[[], None],
    legacy_index: Mapping[tuple[str, str, str], str] | None = None,
    heartbeat_callback: Callable[[], None] | None = None,
    snapshot_callback: Callable[[], None] | None = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame | None]:
    del cache_paths, legacy_index, snapshot_path
    sampler_name = str(config.sampler).lower()
    stages = _pruning_stages(config)
    grid_candidates = (
        _grid_candidates(config.grid, config.paper_params)
        if sampler_name == "grid" else []
    )
    max_active = max(1, int(getattr(config, "max_active_windows", config.workers)))
    max_inflight = max(
        1,
        min(
            int(getattr(config, "max_inflight_batches", config.workers)),
            int(config.workers),
        ),
    )
    prefetch_limit = max(1, int(getattr(config, "grid_prefetch_batches", 24)))
    max_persist_overlap = max(0, min(1, int(getattr(config, "max_persist_overlap_batches", 1))))
    test_cap = max(1, int(math.floor(max_inflight * 0.20)))
    io_requests, io_thread, io_errors = _start_io_worker(
        int(getattr(config, "storage_queue_max_batches", 24))
    )
    initial_trials_future = _io_submit(
        io_requests, _all_trials_frame, optuna, storage, _priority=30,
    )
    initial_trials = initial_trials_future.result()
    audit_requests, audit_thread, audit_errors = _start_audit_worker(
        logger, audit_dir, int(getattr(config, "audit_queue_max_events", 20_000)),
        initial_trials if bool(getattr(config, "audit_incremental", True)) else None,
    )
    waiting = deque(dict(window) for window in windows)
    active: deque[dict[str, Any]] = deque()
    tests_waiting: deque[dict[str, Any]] = deque()
    init_futures: dict[Future, dict[str, Any]] = {}
    registration_futures: dict[Future, dict[str, Any]] = {}
    compute_futures: dict[Future, dict[str, Any]] = {}
    persist_futures: dict[Future, dict[str, Any]] = {}
    transition_futures: dict[Future, dict[str, Any]] = {}
    lock_futures: dict[Future, dict[str, Any]] = {}
    refresh_future: Future | None = None
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    final_ordered: pd.DataFrame | None = None
    final_train_end: pd.Timestamp | None = None
    completed_oos = {
        (
            row.get("scheme"), row.get("train_start"), row.get("train_end"),
            row.get("test_start"), row.get("test_end"),
        )
        for row in oos_rows
    }
    ewma: dict[tuple[Any, ...], float] = {}
    completed_stage_tasks = 0
    completed_since_flush = 0
    sqlite_ack_total = 0.0
    sqlite_ack_count = 0
    compute_busy_seconds = 0.0
    persist_wait_seconds = 0.0
    batch_count = 0
    worker_pids_seen: set[int] = set()
    scheduler_started = time.monotonic()
    last_flush = last_mirror = last_snapshot = last_heartbeat = time.monotonic()
    last_resource_check = 0.0
    ratio = _available_ratio()
    paused = False

    def activate() -> None:
        while waiting and len(active) + len(init_futures) < max_active:
            window = waiting.popleft()
            window["window_id"] = hashlib.sha1(
                f"{window['scheme']}|{window['train_start']}|{window['train_end']}".encode("utf-8")
            ).hexdigest()[:16]
            name = _study_name(
                str(config.search_hash), "train", window["scheme"],
                window["train_start"], window["train_end"],
            )
            future = _io_submit(
                io_requests, _prepare_window_study, optuna, config, storage, name,
                logger, grid_candidates, len(stages) - 1,
                audit_dir / "events.jsonl",
                audit_dir / "inflight_batches",
                _priority=5,
            )
            init_futures[future] = window

    def build_task(window: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        params = dict(record["params"])
        stage_index = int(window["stage_index"])
        stage_fraction = float(stages[stage_index])
        stage_end = _pruning_stage_end(
            window["train_start"], window["train_end"], stage_fraction
        )
        trial_number = int(record["trial_number"])
        return _task_dict(
            config, params,
            core._resolve_gid(config.factor_module, config.gid_factory, params),
            phase="train", scheme=window["scheme"],
            win_start=window["train_start"], win_end=stage_end,
            train_start=window["train_start"], train_end=window["train_end"],
            test_start=window["test_start"], test_end=window["test_end"],
            trial_number=trial_number, study_name=window["study_name"],
            window_id=window["window_id"], candidate_id=_candidate_id(params),
            candidate_order=int(record["candidate_order"]), stage_index=stage_index,
            stage_fraction=stage_fraction,
            task_log_path=str(
                audit_dir / "task_logs" / window["study_name"] / str(trial_number)
                / f"train_stage_{stage_index}.log"
            ),
        )

    def can_finish(window: dict[str, Any]) -> bool:
        no_pending = (
            not window["pending"] if sampler_name == "grid"
            else int(window["remaining_motpe"]) <= 0
        )
        return bool(
            window["durable_gate"] and no_pending
            and window.get("registration_future") is None
            and window.get("prefetched") is None
            and not window.get("compute_inflight")
            and not window.get("persist_inflight")
            and not window.get("persist_backlog")
            and not window.get("transition_inflight")
            and not window.get("lock_inflight")
        )

    def begin_finish_rung(window: dict[str, Any]) -> None:
        if not can_finish(window):
            return
        if int(window["stage_index"]) == len(stages) - 1:
            future = _io_submit(
                io_requests, _lock_window_study, window["study"], config
                , _priority=0
            )
            lock_futures[future] = {"window": window}
            window["lock_inflight"] = True
            if window in active:
                active.remove(window)
            activate()
            return
        survivors, pruned = _rung_survivors(
            window["records"], config, paper_params=config.paper_params
        )
        if not survivors:
            raise RuntimeError(
                f"pruning stage produced no valid survivors: {window['study_name']}"
            )
        next_stage = int(window["stage_index"]) + 1
        survivor_keys = [_parameter_key(record["params"]) for record in survivors]
        transition_action: Callable[..., Any] = _persist_rung_transition
        transition_args: tuple[Any, ...] = (
            optuna, window["study"], pruned,
            {
                "active_candidate_keys": survivor_keys,
                "current_stage_index": next_stage,
            },
        )
        if (
            sampler_name == "grid"
            and bool(getattr(config, "sqlite_batch_transactions", False))
        ):
            transition_action = _persist_rung_transition_sqlite
            transition_args = (
                optuna, config, window["study"], pruned,
                {
                    "active_candidate_keys": survivor_keys,
                    "current_stage_index": next_stage,
                },
            )
        future = _io_submit(
            io_requests, transition_action, *transition_args,
            _priority=0,
        )
        transition_futures[future] = {
            "window": window, "survivors": survivors, "pruned": pruned,
            "next_stage": next_stage, "evaluated": len(window["records"]),
        }
        window["transition_inflight"] = True

    def request_prefetch(window: dict[str, Any]) -> bool:
        prefetched_count = len(registration_futures) + sum(
            item.get("prefetched") is not None for item in active
        )
        if prefetched_count >= prefetch_limit:
            return False
        if window.get("registration_future") is not None or window.get("prefetched") is not None:
            return False
        if window.get("transition_inflight") or window.get("lock_inflight"):
            return False
        if sampler_name == "grid":
            if not window["pending"]:
                return False
            key = (
                next(iter(config.grid.get("alpha_id", [None]))),
                window["scheme"].split("/")[1],
                float(stages[int(window["stage_index"])]),
            )
            count = min(
                len(window["pending"]),
                _adaptive_batch_size(config, ewma.get(key)),
            )
            records = [window["pending"].popleft() for _ in range(count)]
            future = _io_submit(
                io_requests, _register_grid_records, optuna, window["study"],
                config.grid, records, _priority=20,
            )
        else:
            if not window["durable_gate"] or int(window["remaining_motpe"]) <= 0:
                return False
            window["remaining_motpe"] -= 1
            future = _io_submit(
                io_requests, _ask_motpe_record, window["study"], config.grid, logger,
                _priority=0,
            )
        window["registration_future"] = future
        registration_futures[future] = {"window": window}
        return True

    def submit_prefetched(window: dict[str, Any]) -> bool:
        overlap_ready = bool(
            sampler_name == "grid"
            and max_persist_overlap > 0
            and window.get("persist_inflight")
            and window.get("overlap_available")
        )
        if (not window["durable_gate"] and not overlap_ready) or window.get("prefetched") is None:
            return False
        if window.get("compute_inflight"):
            return False
        records = window.pop("prefetched")
        tasks = [build_task(window, record) for record in records]
        batch_id = hashlib.sha1(
            f"{window['window_id']}|{window['stage_index']}|{tasks[0]['candidate_order']}".encode("utf-8")
        ).hexdigest()[:16]
        for record, task in zip(records, tasks, strict=True):
            task["batch_id"] = batch_id
            record["task"] = task
            _audit_event(audit_requests, "events.jsonl", {
                "time": _now(), "event": "START", "batch_id": batch_id,
                "study": window["study_name"], "trial": record["trial_number"],
                "candidate_id": task["candidate_id"],
                "candidate_order": record["candidate_order"],
                "params": record["params"], "stage_index": window["stage_index"],
                "stage_fraction": task["stage_fraction"],
                "window_id": window["window_id"],
            })
            _audit_trial(audit_requests, "START", task)
        if sampler_name == "grid":
            future = executor.submit(core._run_task_batch, tasks)
        else:
            future = executor.submit(core._run_one_task, tasks[0])
        compute_futures[future] = {
            "kind": "train", "window": window, "records": records,
            "tasks": tasks, "batch_id": batch_id, "compute_started": time.perf_counter(),
        }
        window["compute_inflight"] = True
        window["durable_gate"] = False
        if overlap_ready:
            window["overlap_available"] = False
        logger.info(
            "BATCH START batch=%s window=%s stage=%d count=%d trials=%d~%d",
            batch_id, window["window_id"], window["stage_index"], len(records),
            records[0]["trial_number"], records[-1]["trial_number"],
        )
        return True

    def submit_test(window: dict[str, Any]) -> None:
        task = _oos_task(
            config, window["locked_params"], phase="test", scheme=window["scheme"],
            start=window["test_start"], end=window["test_end"],
            trial_number=window["locked_trial"], audit_dir=audit_dir, engine="exact",
        )
        task.update({
            "window_id": window["window_id"], "train_start": window["train_start"],
            "train_end": window["train_end"], "test_start": window["test_start"],
            "test_end": window["test_end"],
            "task_log_path": str(
                audit_dir / "task_logs" / window["study_name"] / "test.log"
            ),
        })
        future = executor.submit(core._run_one_task, task)
        compute_futures[future] = {"kind": "test", "window": window, "task": task}

    def enqueue_persist(window: dict[str, Any], meta: dict[str, Any]) -> None:
        future = _io_submit(
            io_requests, _persist_result_batch, optuna, config,
            window["study"], meta["records"], last_stage=meta["last_stage"],
            _priority=0,
        )
        persist_futures[future] = {**meta, "window": window, "persist_started": time.perf_counter()}
        window["persist_inflight"] = True

    def queue_audit_snapshot(trials: pd.DataFrame | None = None) -> None:
        status.update({
            "completed_stage_tasks": completed_stage_tasks,
            "active_workers": len(compute_futures),
            "active_windows": len({
                meta["window"]["window_id"] for meta in compute_futures.values()
            }),
            "ready_queue_depth": sum(
                item.get("prefetched") is not None and item["durable_gate"]
                for item in active
            ),
            "registration_queue_depth": len(registration_futures),
            "persistence_queue_depth": len(persist_futures),
            "pending_batches": sum(bool(item.get("persist_backlog")) for item in active),
            "overlap_windows": sum(bool(item.get("overlap_available")) for item in active),
            "audit_queue_depth": audit_requests.qsize(),
            "sqlite_txn_seconds": float(_IO_METRICS["last_sqlite_txn_seconds"]),
            "sqlite_txn_count": int(_IO_METRICS["sqlite_txn_count"]),
            "sqlite_ack_seconds": sqlite_ack_total / max(sqlite_ack_count, 1),
            "compute_busy_ratio": min(
                1.0, compute_busy_seconds / max(max_inflight * (time.monotonic() - scheduler_started), 1e-9)
            ),
            "persist_wait_ratio": min(
                1.0, persist_wait_seconds / max(time.monotonic() - scheduler_started, 1e-9)
            ),
            "audit_refresh_seconds": float(_AUDIT_METRICS["last_refresh_seconds"]),
            "audit_refresh_count": int(_AUDIT_METRICS["refresh_count"]),
        })
        if trials is not None:
            _audit_trial_rows(audit_requests, trials.to_dict("records"))
        _audit_refresh_incremental(audit_requests, pareto_rows, oos_rows, status)

    activate()
    try:
        while (
            waiting or active or tests_waiting or init_futures
            or registration_futures or compute_futures or persist_futures
            or transition_futures or lock_futures
        ):
            now = time.monotonic()
            resource_period = max(1.0, float(getattr(config, "resource_check_seconds", 15)))
            if now - last_resource_check >= resource_period:
                ratio = _available_ratio()
                last_resource_check = now
            low = float(getattr(config, "memory_low_watermark_ratio", 0.15))
            high = float(getattr(config, "memory_resume_watermark_ratio", 0.20))
            if not paused and ratio < low:
                paused = True
                logger.warning("RESOURCE PAUSE available_ratio=%.1f%%", ratio * 100)
            elif paused and ratio >= high:
                paused = False
                logger.info("RESOURCE RESUME available_ratio=%.1f%%", ratio * 100)

            for future in [item for item in init_futures if item.done()]:
                window = init_futures.pop(future)
                prepared = future.result()
                window.update(prepared)
                window.update({
                    "pending": deque(prepared["pending"]),
                    "records": list(prepared["records"]),
                    "durable_gate": True, "registration_future": None,
                    "prefetched": None, "compute_inflight": False,
                    "persist_inflight": False, "transition_inflight": False,
                    "persist_backlog": None, "overlap_available": False,
                    "journaled_batches": {},
                    "lock_inflight": False, "locked_params": None,
                    "locked_trial": None,
                })
                active.append(window)
                if prepared["reconciled"]:
                    logger.info(
                        "RECOVERY reconciled=%d study=%s",
                        prepared["reconciled"], window["study_name"],
                    )
                for record in window["records"]:
                    _audit_event(audit_requests, "events.jsonl", {
                        "time": _now(), "event": "RECOVERED_STAGE",
                        "study": window["study_name"],
                        "trial": record["trial_number"],
                        "candidate_id": _candidate_id(record["params"]),
                        "stage_index": window["stage_index"],
                    })
                if can_finish(window):
                    begin_finish_rung(window)
            activate()

            for future in [item for item in registration_futures if item.done()]:
                window = registration_futures.pop(future)["window"]
                window["registration_future"] = None
                value = future.result()
                records = value if sampler_name == "grid" else [value]
                window["prefetched"] = records

            for future in [item for item in transition_futures if item.done()]:
                meta = transition_futures.pop(future)
                transition_rows = future.result() or []
                _audit_trial_rows(audit_requests, transition_rows)
                window = meta["window"]
                old_stage = int(window["stage_index"])
                window["stage_index"] = int(meta["next_stage"])
                window["records"] = []
                window["pending"] = deque({
                    "trial": record["trial"],
                    "trial_number": int(record["trial_number"]),
                    "params": dict(record["params"]),
                    "candidate_order": int(record["candidate_order"]),
                } for record in meta["survivors"])
                window["transition_inflight"] = False
                logger.info(
                    "RUNG study=%s stage=%d evaluated=%d survivors=%d pruned=%d next_fraction=%.2f",
                    window["study_name"], old_stage, meta["evaluated"],
                    len(meta["survivors"]), len(meta["pruned"]),
                    stages[window["stage_index"]],
                )
                _audit_event(audit_requests, "events.jsonl", {
                    "time": _now(), "event": "RUNG_END",
                    "study": window["study_name"], "stage_index": old_stage,
                    "survivors": len(meta["survivors"]),
                    "pruned": len(meta["pruned"]),
                })
                for record in meta["pruned"]:
                    _audit_event(audit_requests, "events.jsonl", {
                        "time": _now(), "event": "PRUNED",
                        "study": window["study_name"],
                        "trial": int(record["trial_number"]),
                        "candidate_id": _candidate_id(record["params"]),
                        "candidate_order": int(record["candidate_order"]),
                        "params": record["params"], "stage_index": old_stage,
                        "reason": "successive_halving",
                    })

            for future in [item for item in lock_futures if item.done()]:
                nonlocal_values = future.result()
                frame, ordered, reason = nonlocal_values
                window = lock_futures.pop(future)["window"]
                train_frames.append(frame)
                pareto_rows[:] = [
                    row for row in pareto_rows
                    if row.get("study_name") != window["study_name"]
                ] + ordered.assign(
                    study_name=window["study_name"], phase="train",
                    window_id=window["window_id"],
                ).to_dict("records")
                locked = ordered.iloc[0]
                window["locked_params"] = json.loads(str(locked["params_json"]))
                window["locked_trial"] = int(locked["trial_number"])
                window["lock_inflight"] = False
                logger.info(
                    "LOCK window=%s study=%s trial=%d reason=%s params=%s",
                    window["window_id"], window["study_name"],
                    window["locked_trial"], reason, locked["params_json"],
                )
                train_end = pd.Timestamp(window["train_end"])
                if final_train_end is None or train_end > final_train_end:
                    final_ordered, final_train_end = ordered, train_end
                key = (
                    window["scheme"], window["train_start"], window["train_end"],
                    window["test_start"], window["test_end"],
                )
                if key not in completed_oos:
                    tests_waiting.append(window)

            for future in [item for item in persist_futures if item.done()]:
                meta = persist_futures.pop(future)
                durable = future.result()
                window = meta["window"]
                window["records"].extend(durable)
                window["persist_inflight"] = False
                window["durable_gate"] = not bool(window.get("persist_backlog"))
                window["overlap_available"] = False
                _commit_batch_journal(audit_dir, meta["batch_id"])
                completed_stage_tasks += len(durable)
                completed_since_flush += len(durable)
                batch_count += 1
                ack_seconds = max(time.perf_counter() - meta["persist_started"], 0.0)
                sqlite_ack_total += ack_seconds
                sqlite_ack_count += 1
                persist_wait_seconds += ack_seconds
                errors = sum(bool(record["result"].get("error")) for record in durable)
                for record in durable:
                    pid = record["result"].get("worker_pid")
                    if pid:
                        worker_pids_seen.add(int(pid))
                    _audit_trial(audit_requests, "END", record["result"])
                _audit_trial_rows(
                    audit_requests,
                    [record["audit_row"] for record in durable if record.get("audit_row")],
                )
                _audit_event(audit_requests, "events.jsonl", {
                    "time": _now(), "event": "BATCH_COMMIT",
                    "batch_id": meta["batch_id"], "study": window["study_name"],
                    "count": len(durable), "errors": errors,
                    "sqlite_ack_seconds": ack_seconds,
                })
                logger.info(
                    "BATCH END batch=%s window=%s count=%d errors=%d compute=%.3fs sqlite_ack=%.3fs",
                    meta["batch_id"], window["window_id"], len(durable), errors,
                    meta["compute_seconds"], ack_seconds,
                )
                backlog = window.pop("persist_backlog", None)
                if backlog is not None:
                    window["durable_gate"] = False
                    enqueue_persist(window, backlog)
                if can_finish(window):
                    begin_finish_rung(window)

            if not paused:
                for _ in range(len(active)):
                    window = active[0]
                    active.rotate(-1)
                    request_prefetch(window)
                active_tests = sum(
                    meta["kind"] == "test" for meta in compute_futures.values()
                )
                while (
                    tests_waiting and len(compute_futures) < max_inflight
                    and active_tests < test_cap
                ):
                    submit_test(tests_waiting.popleft())
                    active_tests += 1
                made_progress = True
                while len(compute_futures) < max_inflight and made_progress:
                    made_progress = False
                    for _ in range(len(active)):
                        window = active[0]
                        active.rotate(-1)
                        if submit_prefetched(window):
                            made_progress = True
                            request_prefetch(window)
                            if len(compute_futures) >= max_inflight:
                                break

            tracked_futures = tuple({
                *init_futures, *registration_futures, *compute_futures,
                *persist_futures, *transition_futures, *lock_futures,
            })
            if tracked_futures:
                finished, _ = wait(
                    tracked_futures,
                    timeout=max(0.05, float(getattr(config, "scheduler_wait_seconds", 0.25))),
                    return_when=FIRST_COMPLETED,
                )
            else:
                finished = set()
                time.sleep(max(0.05, float(getattr(config, "scheduler_wait_seconds", 0.25))))
            for future in [item for item in finished if item in compute_futures]:
                meta = compute_futures.pop(future)
                window = meta["window"]
                if meta["kind"] == "test":
                    result = future.result()
                    params = window["locked_params"]
                    oos = {
                        "window_id": window["window_id"], "scheme": window["scheme"],
                        "train_start": window["train_start"],
                        "train_end": window["train_end"],
                        "test_start": window["test_start"],
                        "test_end": window["test_end"],
                        "trial_number": window["locked_trial"],
                        "candidate_id": _candidate_id(params),
                        "params_json": _parameter_key(params),
                        **{key: result.get(key) for key in (
                            "robust_rank_ic", "mean_rank_ic", "sharpe", "mdd",
                            "turnover", "ic_coverage", "error", "task_log_path",
                            "worker_pid",
                        )},
                    }
                    oos_rows.append(oos)
                    test_frames.append(pd.DataFrame([{
                        **meta["task"], **result, "params": params,
                    }]))
                    _audit_event(
                        audit_requests, "events.jsonl",
                        {"time": _now(), "event": "TEST_END", **oos},
                    )
                    logger.info(
                        "TEST END window=%s trial=%d metrics=%s",
                        window["window_id"], window["locked_trial"],
                        _parameter_key(oos),
                    )
                    continue
                results = future.result()
                if sampler_name != "grid":
                    results = [results]
                records = meta["records"]
                for record, task, result in zip(
                    records, meta["tasks"], results, strict=True
                ):
                    record["task"], record["result"] = task, result
                key = (
                    next(iter(config.grid.get("alpha_id", [None]))),
                    window["scheme"].split("/")[1],
                    float(stages[int(window["stage_index"])]),
                )
                batch_elapsed = max(
                    float(result.get("batch_elapsed_seconds", 0.0) or 0.0)
                    for result in results
                )
                compute_busy_seconds += max(
                    batch_elapsed or time.perf_counter() - meta["compute_started"], 0.0
                )
                sample = (
                    batch_elapsed / max(len(results), 1)
                    if batch_elapsed > 0 else sum(
                        float(result.get("elapsed_seconds", 0.0))
                        for result in results
                    ) / max(len(results), 1)
                )
                ewma[key] = sample if key not in ewma else 0.25 * sample + 0.75 * ewma[key]
                window["compute_inflight"] = False
                last_stage = int(window["stage_index"]) == len(stages) - 1
                journal_records = [
                    {
                        "trial_number": int(record["trial_number"]),
                        "candidate_id": _candidate_id(record["params"]),
                        "candidate_order": int(record["candidate_order"]),
                        "params": dict(record["params"]),
                        "task": dict(record["task"]),
                        "result": dict(record["result"]),
                    }
                    for record in records
                ]
                _write_batch_journal(audit_dir, meta["batch_id"], {
                    "batch_id": meta["batch_id"],
                    "study_name": window["study_name"],
                    "window_id": window["window_id"],
                    "stage_index": int(window["stage_index"]),
                    "last_stage": last_stage,
                    "records": journal_records,
                })
                persist_meta = {
                    "batch_id": meta["batch_id"],
                    "records": records,
                    "last_stage": last_stage,
                    "compute_seconds": max(
                        time.perf_counter() - meta["compute_started"], 0.0
                    ),
                }
                if window.get("persist_inflight"):
                    window["persist_backlog"] = persist_meta
                else:
                    enqueue_persist(window, persist_meta)
                    if sampler_name == "grid" and max_persist_overlap > 0:
                        window["overlap_available"] = True

            refresh_due = (
                completed_since_flush >= int(getattr(config, "partial_refresh_trials", 250))
                or now - last_flush >= int(getattr(config, "partial_refresh_seconds", 60))
            )
            if refresh_due:
                queue_audit_snapshot()
                completed_since_flush = 0
                last_flush = now
            if now - last_mirror >= int(getattr(config, "audit_mirror_seconds", 60)):
                mirror_callback()
                last_mirror = now
            if database_path and now - last_snapshot >= int(getattr(config, "snapshot_seconds", 300)):
                if snapshot_callback:
                    snapshot_callback()
                last_snapshot = now
            if now - last_heartbeat >= 10:
                if heartbeat_callback:
                    heartbeat_callback()
                elapsed = max(now - scheduler_started, 1e-9)
                active_train = [
                    meta for meta in compute_futures.values()
                    if meta["kind"] == "train"
                ]
                logger.info(
                    "HEARTBEAT batches=%d/%d active_windows=%d registration=%d prefetched=%d "
                    "persist=%d audit=%d candidate_per_second=%.4f cpu=%s memory_available=%.1f%% "
                    "worker_pids=%s sqlite_ack_avg=%.4fs sqlite_txn_seconds=%.4fs sqlite_txn_count=%d "
                    "compute_busy=%.1f%% persist_wait=%.1f%% audit_refresh=%.4fs "
                    "scheduler_cpu=%s paused=%s "
                    "pending=%d overlap=%d",
                    len(compute_futures), max_inflight,
                    len({meta["window"]["window_id"] for meta in active_train}),
                    len(registration_futures),
                    sum(item.get("prefetched") is not None for item in active),
                    len(persist_futures), audit_requests.qsize(),
                    completed_stage_tasks / elapsed,
                    _cpu_percent(), ratio * 100, sorted(worker_pids_seen),
                    sqlite_ack_total / max(sqlite_ack_count, 1),
                    float(_IO_METRICS["last_sqlite_txn_seconds"]),
                    int(_IO_METRICS["sqlite_txn_count"]),
                    min(100.0, 100.0 * compute_busy_seconds / max(max_inflight * elapsed, 1e-9)),
                    min(100.0, 100.0 * persist_wait_seconds / max(elapsed, 1e-9)),
                    float(_AUDIT_METRICS["last_refresh_seconds"]),
                    _process_cpu_percent(), paused,
                    sum(bool(item.get("persist_backlog")) for item in active),
                    sum(bool(item.get("overlap_available")) for item in active),
                )
                last_heartbeat = now
            if io_errors:
                raise RuntimeError(f"Optuna I/O worker failed: {io_errors[-1]}")
            if audit_errors:
                raise RuntimeError(f"audit worker failed: {audit_errors[-1]}")
    finally:
        if refresh_future is not None:
            try:
                queue_audit_snapshot(refresh_future.result(timeout=30))
            except Exception:
                pass
        _stop_io_worker(io_requests, io_thread)
        _stop_audit_worker(audit_requests, audit_thread)
    return train_frames, test_frames, final_ordered


def run_optuna_walk_forward(config: Any) -> dict[str, pd.DataFrame]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna mining requires `pip install betalens[mining]`") from exc

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not config.config_hash:
        config.config_hash = hashlib.sha256(
            json.dumps(_jsonable({
                "scheduler_schema_version": int(
                    getattr(config, "scheduler_schema_version", 3)
                ),
                "grid": config.grid,
                "span": config.rolling_span,
                "valid": config.valid,
            }), sort_keys=True).encode()
        ).hexdigest()
    config.search_hash = str(getattr(config, "search_hash", "") or config.config_hash)
    runtime_dir = _runtime_dir(config, output_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = runtime_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logger(
        runtime_dir,
        config.log_level,
        bool(getattr(config, "console_trial_events", False)),
    )
    legacy_locks = [output_dir / ".study.lock"]
    if output_dir.parent != output_dir:
        legacy_locks.append(output_dir.parent / ".study.lock")
    for legacy_lock in legacy_locks:
        if legacy_lock.exists() and legacy_lock.resolve() != (runtime_dir / ".study.lock").resolve():
            try:
                holder = json.loads(legacy_lock.read_text(encoding="utf-8"))
            except Exception:
                holder = {"pid": -1, "error": "unreadable lock"}
            if _lock_owner_active(holder):
                logger.error("LEGACY LOCK ACTIVE path=%s holder=%s", legacy_lock, holder)
                _close_logger(logger)
                raise RuntimeError(f"existing OneDrive coordinator is still active: {holder}")
    lock_path = runtime_dir / ".study.lock"
    try:
        owner = _acquire_lock(lock_path, config.config_hash)
    except Exception:
        logger.exception("LOCK_REFUSED path=%s", lock_path.resolve())
        _close_logger(logger)
        raise
    database_path: Path | None = None
    storage: Any | None = None
    executor: ProcessPoolExecutor | None = None
    slot_owner: dict[str, Any] | None = None
    snapshot_requests: queue.Queue[Any] | None = None
    snapshot_thread: threading.Thread | None = None
    snapshot_errors: list[str] = []
    status = {
        "alpha": next(iter(config.grid.get("alpha_id", [None]))), "pid": os.getpid(),
        "started_at": owner["started_at"], "config_hash": config.config_hash,
        "search_hash": config.search_hash,
        "runtime_hash": str(getattr(config, "runtime_hash", "")),
        "runtime_dir": str(runtime_dir),
        "output_dir": str(output_dir),
        "cache_dir": str((runtime_dir / "cache").resolve()), "status": "running",
    }
    pareto_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []
    snapshot_path = runtime_dir / "study.snapshot.sqlite3"
    mirror_requests, mirror_thread, mirror_errors = _start_mirror_worker(
        runtime_dir, output_dir
    )
    mirrored_signatures: dict[str, tuple[int, int]] = {}

    def mirror() -> None:
        relatives = [
            "run.log", "study.snapshot.sqlite3", "audit/events.jsonl", "audit/errors.jsonl",
            "audit/trials.partial.csv", "audit/pareto_front.partial.csv",
            "audit/oos_parameter_path.partial.csv", "audit/status.json", "audit/candidate_manifest.csv",
        ]
        changed = []
        for relative in relatives:
            source = runtime_dir / relative
            if not source.exists():
                continue
            signature = (int(source.stat().st_mtime_ns), int(source.stat().st_size))
            if mirrored_signatures.get(relative) != signature:
                mirrored_signatures[relative] = signature
                changed.append(relative)
        _request_mirror(mirror_requests, changed)

    try:
        storage, database_path, storage_url = _sqlite_storage(optuna, runtime_dir, config.storage_url, logger)
        if (
            str(config.sampler).lower() == "grid"
            and bool(getattr(config, "sqlite_batch_transactions", False))
        ):
            _validate_sqlite_batch_adapter(storage)
        snapshot_requests, snapshot_thread, snapshot_errors = _start_snapshot_worker(
            database_path, snapshot_path
        )
        status.update({"storage": storage_url, "database": str(database_path.resolve()) if database_path else storage_url})
        grid_count = int(np.prod([len(values) for values in config.grid.values()]))
        if str(config.sampler).lower() == "grid" and grid_count > int(config.max_grid_candidates):
            raise ValueError(f"grid has {grid_count} candidates, exceeding max_grid_candidates={config.max_grid_candidates}")
        candidates = _grid_candidates(config.grid)
        manifest = pd.DataFrame([
            {"candidate_order": index, "candidate_id": _candidate_id(params),
             "gid": core._resolve_gid(config.factor_module, config.gid_factory, params),
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

        config.cache_dir = runtime_dir / "cache"
        expected_cache = core._cache_paths(config.cache_dir)
        cache_mtime_before = expected_cache.data.stat().st_mtime_ns if expected_cache.data.exists() else None
        cache_paths = core.build_cache_for_config(config, [rolling_span, config.valid], config.paper_params or candidates[0])
        cache_hit = (
            cache_mtime_before is not None
            and cache_paths.data.exists()
            and cache_paths.data.stat().st_mtime_ns == cache_mtime_before
            and not config.rebuild_cache
        )
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = str(max(1, int(getattr(config, "blas_threads", 1))))
        effective_workers = _wait_for_worker_capacity(config, cache_paths, logger)
        config.workers = effective_workers
        slot_owner = _acquire_worker_slots(config, effective_workers, logger)
        executor = ProcessPoolExecutor(
            max_workers=max(1, effective_workers),
            initializer=core._init_worker,
            initargs=(str(cache_paths.data), str(cache_paths.pit)),
        )
        logger.info(
            "STARTUP config=%s config_hash=%s range=%s valid=%s cache=%s cache_hit=%s memory_budget=%s workers=%d sampler=%s objectives=%s "
            "constraints=coverage>=%.2f,mdd<=%.2f,turnover<=%.2f dimensions=%d candidates_or_budget=%d "
            "pruning_stages=%s reduction_factor=%d min_full_candidates=%d keep_paper=%s "
            "db=%s audit=%s log=%s snapshot=%s",
            getattr(config, "config_path", None), config.config_hash, rolling_span, config.valid,
            cache_paths.data.resolve(), cache_hit, core._format_bytes(core._memory_budget_bytes(config)), effective_workers,
            config.sampler, ["robust_rank_ic", "sharpe"], config.ic_coverage_min,
            config.max_drawdown_max, config.turnover_max, sum(len(values) > 1 for values in config.grid.values()),
            grid_count if str(config.sampler).lower() == "grid" else config.n_trials,
            list(_pruning_stages(config)), int(getattr(config, "pruning_reduction_factor", 3)),
            int(getattr(config, "pruning_min_full_candidates", 3)),
            bool(getattr(config, "pruning_keep_paper", True)),
            status["database"], audit_dir.resolve(), (runtime_dir / "run.log").resolve(), snapshot_path.resolve(),
        )
        _atomic_json(audit_dir / "status.json", status)
        oos_partial = audit_dir / "oos_parameter_path.partial.csv"
        if oos_partial.exists():
            oos_rows = pd.read_csv(oos_partial).to_dict("records")
        elif (output_dir / "audit" / "oos_parameter_path.partial.csv").exists():
            oos_rows = pd.read_csv(output_dir / "audit" / "oos_parameter_path.partial.csv").to_dict("records")
        windows = [
            {
                "train_start": train_start, "train_end": train_end,
                "test_start": test_start, "test_end": test_end, "scheme": scheme,
            }
            for train_start, train_end, test_start, test_end, scheme in pairs
        ]
        train_frames, test_frames, final_ordered = _run_multiwindow_scheduler(
            optuna=optuna, config=config, storage=storage, cache_paths=cache_paths,
            executor=executor, logger=logger, audit_dir=audit_dir, windows=windows,
            pareto_rows=pareto_rows, candidate_rows=candidate_rows, oos_rows=oos_rows,
            status=status, database_path=database_path, snapshot_path=snapshot_path,
            mirror_callback=mirror, legacy_index=None,
            heartbeat_callback=lambda: (
                _heartbeat_lock(lock_path, owner), _heartbeat_worker_slots(slot_owner)
            ),
            snapshot_callback=lambda: _request_snapshot(snapshot_requests),
        )

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
        _atomic_text(runtime_dir / "final_candidates.yaml", yaml.safe_dump({"candidates": final_candidates}, allow_unicode=True, sort_keys=False))
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

        trials = _io_once(_all_trials_frame, optuna, storage)
        pareto = pd.DataFrame(pareto_rows)
        oos_frame = pd.DataFrame(oos_rows)
        if not oos_frame.empty:
            oos_frame = oos_frame.sort_values(
                ["test_start", "test_end", "scheme", "window_id"], kind="mergesort"
            ).reset_index(drop=True)
            oos_frame["parameter_changed"] = (
                oos_frame["params_json"].ne(oos_frame["params_json"].shift(1))
            )
            oos_frame.loc[0, "parameter_changed"] = False
        _atomic_csv(runtime_dir / "trials.csv", trials)
        _atomic_csv(runtime_dir / "pareto_front.csv", pareto)
        _atomic_csv(runtime_dir / "cv_results.csv", _cv_results(trials))
        _atomic_csv(runtime_dir / "oos_parameter_path.csv", oos_frame)
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
        final_mirror_files = (
            "trials.csv", "pareto_front.csv", "cv_results.csv", "oos_parameter_path.csv",
            "final_candidates.yaml", "run.log", "study.snapshot.sqlite3",
            "audit/events.jsonl", "audit/errors.jsonl", "audit/trials.partial.csv",
            "audit/pareto_front.partial.csv", "audit/oos_parameter_path.partial.csv",
            "audit/status.json", "audit/candidate_manifest.csv",
        )
        if snapshot_errors:
            for error in snapshot_errors:
                _append_jsonl(audit_dir / "errors.jsonl", {
                    "time": _now(), "event": "SNAPSHOT_WORKER_ERROR", "error": error,
                })
            logger.warning("snapshot worker reported %d error(s); final snapshot succeeded", len(snapshot_errors))
        mirror_errors.clear()
        final_mirror_done = _request_mirror(mirror_requests, final_mirror_files, block=True)
        if final_mirror_done is None or not final_mirror_done.wait(timeout=120):
            raise RuntimeError("final audit mirror did not finish within 120 seconds")
        if mirror_errors:
            raise RuntimeError(f"final audit mirror failed: {mirror_errors[-1]}")
        logger.info(
            "COMPLETE db=%s snapshot=%s log=%s events=%s trials=%s pareto=%s cv=%s oos=%s candidates=%s",
            status["database"], snapshot_path.resolve(), (runtime_dir / "run.log").resolve(),
            (audit_dir / "events.jsonl").resolve(), (runtime_dir / "trials.csv").resolve(),
            (runtime_dir / "pareto_front.csv").resolve(), (runtime_dir / "cv_results.csv").resolve(),
            (runtime_dir / "oos_parameter_path.csv").resolve(), (runtime_dir / "final_candidates.yaml").resolve(),
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
            executor.shutdown(wait=False, cancel_futures=True)
        marked = 0
        try:
            marked = _io_once(
                _mark_running_trials_failed,
                optuna,
                storage,
                logger,
                audit_dir / "events.jsonl",
            )
        except Exception:
            logger.exception("failed to mark RUNNING trials after interrupt")
        status["interrupted_trials"] = marked
        if storage is not None:
            try:
                _io_once(
                    _refresh_audit, optuna, storage, audit_dir, pareto_rows, status,
                )
            except Exception:
                logger.exception("failed to refresh partial audit after interrupt")
        try:
            _atomic_csv(audit_dir / "oos_parameter_path.partial.csv", pd.DataFrame(oos_rows))
        except Exception:
            logger.exception("failed to refresh OOS audit after interrupt")
        try:
            status["last_snapshot"] = _snapshot_database(database_path, snapshot_path)
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
        for relative in ("run.log", "study.snapshot.sqlite3", "audit/events.jsonl", "audit/errors.jsonl", "audit/trials.partial.csv", "audit/pareto_front.partial.csv", "audit/oos_parameter_path.partial.csv", "audit/status.json"):
            _atomic_copy(runtime_dir / relative, output_dir / relative)
        raise SystemExit(130)
    except Exception as exc:
        status.update({"status": "failed", "finished_at": _now(), "exit_reason": f"{type(exc).__name__}: {exc}"})
        try:
            status["last_snapshot"] = _snapshot_database(database_path, snapshot_path)
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
            executor = None
        if storage is not None:
            storage.remove_session()
            storage.engine.dispose()
        if snapshot_requests is not None and snapshot_thread is not None:
            try:
                snapshot_requests.put(None, timeout=30)
            except Exception:
                pass
            snapshot_thread.join(timeout=60)
            if snapshot_thread.is_alive():
                logger.error("snapshot worker did not stop; coordinator lock will remain for safety")
        try:
            mirror_requests.put(None, timeout=30)
            mirror_thread.join(timeout=60)
        except Exception:
            pass
        _close_jsonl_handles()
        if snapshot_thread is None or not snapshot_thread.is_alive():
            _release_worker_slots(slot_owner)
            _release_lock(lock_path, owner)
        _close_logger(logger)


__all__ = ["run_optuna_walk_forward"]
