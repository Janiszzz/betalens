from __future__ import annotations

import contextlib
import dataclasses
import io
import shutil
import tempfile
import threading
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from betalens.analyst import Analyst

from .factors import get_factor_config, load_factor_module
from .schemas import RunRequest, RunState, RunStatus
from .serialization import (
    build_downloads,
    build_result_payload,
    build_table,
    read_table_page,
    write_table_parquet,
)


MAX_RUNS = 20  # 内存里最多保留最近 N 次回测,超出按 LRU 淘汰
TABLE_KINDS = ("trades", "positions")
_CACHE_ROOT = Path(tempfile.gettempdir()) / "betalens_dashboard_runs"


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
        # 巨表落 parquet 后释放 bt,这里缓存可 JSON 化的结果与各表元数据/路径
        self.payload: dict[str, Any] | None = None
        self.table_meta: dict[str, dict[str, Any]] = {}
        self.cache_dir: Path = _CACHE_ROOT / self.run_id
        self._log_parts: list[str] = []
        self._log_len = 0
        self._lock = threading.Lock()

    def append_log(self, text: str) -> None:
        with self._lock:
            self._log_parts.append(text)
            self._log_len += len(text)

    @property
    def log(self) -> str:
        with self._lock:
            return "".join(self._log_parts)

    def table_path(self, kind: str) -> Path:
        return self.cache_dir / f"{kind}.parquet"

    def cleanup(self) -> None:
        # LRU 淘汰时调用,删掉该 run 的 parquet 临时目录
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def mark_started(self) -> None:
        self.status = "running"
        self.started_at = datetime.now(timezone.utc)
        self.append_log(f"[dashboard] run {self.run_id} started\n")

    def mark_completed(self) -> None:
        self.status = "completed"
        self.finished_at = datetime.now(timezone.utc)
        self.append_log(f"[dashboard] run {self.run_id} completed\n")

    def mark_failed(self, exc: BaseException) -> None:
        self.status = "failed"
        self.finished_at = datetime.now(timezone.utc)
        self.error = str(exc)
        self.append_log("\n[dashboard] run failed\n")
        self.append_log(traceback.format_exc())

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
            log_size=self._log_len,
        )


class RunManager:
    def __init__(self):
        self._runs: "OrderedDict[str, DashboardRun]" = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dashboard-run")
        # dump_to_excel 较慢,放到独立后台线程,不阻塞回测线程出结果
        self._dump_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dashboard-dump")

    def create(self, request: RunRequest) -> DashboardRun:
        run = DashboardRun(request)
        with self._lock:
            self._runs[run.run_id] = run
            self._runs.move_to_end(run.run_id)
            evicted = []
            while len(self._runs) > MAX_RUNS:
                _, old = self._runs.popitem(last=False)
                evicted.append(old)
        for old in evicted:
            old.cleanup()
        self._executor.submit(self._execute, run)
        return run

    def get(self, run_id: str) -> DashboardRun:
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                self._runs.move_to_end(run_id)
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
            # FactorPipeline 取自因子模块本身（已 import 各自类模板的 re-export），
            # 与因子类解耦：tdx 走 factor_template_tdx，alpha101 走 factor_template_alpha101。
            FactorPipeline = getattr(mod, "FactorPipeline")

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
                # dump 由后端异步落盘,避免阻塞出结果
                "dump_excel": False,
            }

            log_writer = LogBuffer(run)
            with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
                result = FactorPipeline(factor_spec).run(str(start_date), str(end_date), **kwargs)
            run.result = result
            run.backtest = result.backtest
            run.analyst = result.analyst or Analyst.from_backtest(result.backtest, name=run.name)

            # 巨表落 parquet → 缓存可 JSON 化 payload → 释放 bt(省内存)→ 后台异步写 dump
            self._persist_tables(run)
            bt = run.backtest
            factor_values = getattr(result, "factor_values", None)
            name = run.name
            out_dir = output_dir
            run.mark_completed()
            run.payload = build_result_payload(run, run.table_meta, factor_values)
            run.backtest = None
            run.result = None
            run.analyst = None
            self._dump_executor.submit(self._dump_excel, bt, out_dir, name, factor_values)
        except Exception as exc:
            run.mark_failed(exc)

    def _persist_tables(self, run: DashboardRun) -> None:
        for kind in TABLE_KINDS:
            rows = build_table(run.backtest, kind)
            run.table_meta[kind] = write_table_parquet(rows, run.table_path(kind))

    @staticmethod
    def _dump_excel(
        bt: Any,
        output_dir: Path,
        name: str,
        factor_values: Any = None,
    ) -> None:
        try:
            dump_path = f"{output_dir}/{name}_dump.xlsx"
            bt.dump_to_excel(dump_path)
            if factor_values is not None:
                import pandas as pd

                with pd.ExcelWriter(
                    dump_path,
                    engine="openpyxl",
                    mode="a",
                    if_sheet_exists="replace",
                ) as writer:
                    factor_values.to_excel(writer, sheet_name="factor_values", index=False)
        except Exception:
            pass  # dump 仅供下载,失败不影响已展示的结果

    def serialize_result(self, run_id: str) -> dict[str, Any]:
        run = self.get(run_id)
        if run.payload is None:
            raise ValueError(f"run is {run.status}")
        payload = dict(run.payload)
        # downloads 实时探测磁盘(dump 是异步写的,可能稍后才出现)
        payload["downloads"] = build_downloads(Path(run.factor_dir), run.name)
        return payload

    def table_page(
        self,
        run_id: str,
        kind: str,
        page: int,
        size: int,
        query: str | None = None,
        filters: dict[str, str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        run = self.get(run_id)
        if run.payload is None:
            raise ValueError(f"run is {run.status}")
        return read_table_page(
            run.table_path(kind),
            page=page,
            size=size,
            query=query,
            filters=filters,
            date_from=date_from,
            date_to=date_to,
        )


manager = RunManager()
