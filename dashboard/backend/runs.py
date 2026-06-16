from __future__ import annotations

import contextlib
import dataclasses
import io
import queue
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from betalens.analyst import Analyst

from .factors import get_factor_config, load_factor_module
from .schemas import RunRequest, RunState, RunStatus
from .serialization import serialize_result


class LogBuffer(io.TextIOBase):
    def __init__(self, run: "DashboardRun"):
        self.run = run

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if s:
            self.run.append_log(s)
        return len(s)

    def flush(self) -> None:
        return None


class DashboardRun:
    def __init__(self, request: RunRequest):
        self.run_id = uuid.uuid4().hex
        self.factor_class = request.factor_class
        self.name = request.name
        self.parameters = dict(request.parameters)
        self.compute_kwargs = dict(request.compute_kwargs)
        self.status: RunStatus = "queued"
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.error: str | None = None
        self.result: Any = None
        self.backtest: Any = None
        self.analyst: Analyst | None = None
        self.factor_dir: str = ""
        self._log = ""
        self._events: "queue.Queue[str | None]" = queue.Queue()
        self._lock = threading.Lock()

    def append_log(self, text: str) -> None:
        with self._lock:
            self._log += text
        self._events.put(text)

    def close_log(self) -> None:
        self._events.put(None)

    def iter_log_events(self):
        with self._lock:
            existing = self._log
        if existing:
            yield existing
        while True:
            item = self._events.get()
            if item is None:
                break
            yield item

    @property
    def log(self) -> str:
        with self._lock:
            return self._log

    def mark_started(self) -> None:
        self.status = "running"
        self.started_at = datetime.now(timezone.utc)
        self.append_log(f"[dashboard] run {self.run_id} started\n")

    def mark_completed(self) -> None:
        self.status = "completed"
        self.finished_at = datetime.now(timezone.utc)
        self.append_log(f"[dashboard] run {self.run_id} completed\n")
        self.close_log()

    def mark_failed(self, exc: BaseException) -> None:
        self.status = "failed"
        self.finished_at = datetime.now(timezone.utc)
        self.error = str(exc)
        self.append_log("\n[dashboard] run failed\n")
        self.append_log(traceback.format_exc())
        self.close_log()

    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or datetime.now(timezone.utc)
        return round((end - self.started_at).total_seconds(), 3)

    def to_state(self) -> RunState:
        return RunState(
            run_id=self.run_id,
            status=self.status,
            factor_class=self.factor_class,
            name=self.name,
            started_at=self.started_at.isoformat() if self.started_at else None,
            finished_at=self.finished_at.isoformat() if self.finished_at else None,
            elapsed_seconds=self.elapsed_seconds(),
            error=self.error,
            log_size=len(self.log),
        )


class RunManager:
    def __init__(self):
        self._runs: dict[str, DashboardRun] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dashboard-run")

    def create(self, request: RunRequest) -> DashboardRun:
        run = DashboardRun(request)
        with self._lock:
            self._runs[run.run_id] = run
        self._executor.submit(self._execute, run)
        return run

    def get(self, run_id: str) -> DashboardRun:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        return run

    def _execute(self, run: DashboardRun) -> None:
        run.mark_started()
        try:
            script, _spec_data, _factor_cfg = get_factor_config(run.factor_class, run.name)
            run.factor_dir = str(script.parent)
            mod = load_factor_module(script)
            factor_spec = getattr(mod, "spec")

            spec_updates = {}
            for key in (
                "direction",
                "index_code",
                "use_industry",
                "use_mktcap",
                "industry_scheme",
                "backtest_metric",
            ):
                if key in run.parameters:
                    spec_updates[key] = run.parameters[key]
            if run.compute_kwargs:
                merged_kwargs = dict(getattr(factor_spec, "compute_kwargs", {}) or {})
                merged_kwargs.update(run.compute_kwargs)
                spec_updates["compute_kwargs"] = merged_kwargs
            if spec_updates:
                factor_spec = dataclasses.replace(factor_spec, **spec_updates)

            class_dir = script.parent.parent
            import sys
            for p in (class_dir.parent, class_dir):
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            from factor_template import FactorPipeline

            start_date = run.parameters.get("start_date")
            end_date = run.parameters.get("end_date")
            if not start_date or not end_date:
                raise ValueError("start_date and end_date are required")

            output_dir = Path(run.factor_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            kwargs = {
                "rebal_freq": run.parameters.get("rebal_freq", "D"),
                "n_quantiles": int(run.parameters.get("n_quantiles", 20)),
                "initial_amount": float(run.parameters.get("initial_amount", 100000)),
                "output_dir": str(output_dir),
                "include_profiling": bool(run.parameters.get("include_profiling", True)),
            }

            log_writer = LogBuffer(run)
            with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
                result = FactorPipeline(factor_spec).run(str(start_date), str(end_date), **kwargs)
            run.result = result
            run.backtest = result.backtest
            run.analyst = result.analyst or Analyst.from_backtest(result.backtest, name=run.name)
            run.mark_completed()
        except Exception as exc:
            run.mark_failed(exc)

    def serialize_result(self, run_id: str) -> dict[str, Any]:
        run = self.get(run_id)
        return serialize_result(run)


manager = RunManager()
