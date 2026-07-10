from __future__ import annotations

import contextlib
import copy
import inspect
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

from betalens.factor.config import run_parameters, section, write_yaml_config

from .factors import get_factor_config, load_factor_module
from .schemas import RunRequest, RunState, RunStatus


MAX_RUNS = 20  # 内存里最多保留最近 N 次回测,超出按 LRU 淘汰
TABLE_KINDS = ("trades", "positions")
_CACHE_ROOT = Path(tempfile.gettempdir()) / "betalens_dashboard_runs"


def _normalize_group_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        raw = [part for part in text.replace("，", ",").replace("；", ",").replace(";", ",").split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = [value]

    groups: list[Any] = []
    for item in raw:
        if item is None or item == "":
            continue
        parsed = int(item) if str(item).strip().lstrip("-").isdigit() else item
        groups.append(parsed)
    return groups if groups else None


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
        self.output_dir: str = ""
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

    def factor_values_path(self) -> Path:
        return self.cache_dir / "factor_values.parquet"

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

    def clear(self) -> dict[str, int]:
        with self._lock:
            runs = list(self._runs.values())
            self._runs.clear()
        cleared = 0
        for run in runs:
            if run.status in ("queued", "running"):
                run.status = "failed"
                run.finished_at = datetime.now(timezone.utc)
                run.error = "run cleared"
                run.append_log("\n[dashboard] run cleared\n")
            run.cleanup()
            cleared += 1

        # 丢弃排队任务。正在执行的线程无法安全强杀，但它已不再阻塞新的 executor。
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dashboard-run")
        return {"cleared": cleared}

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
            run.append_log("[dashboard] resolving factor config\n")
            script, _spec_data, factor_cfg = get_factor_config(run.factor_class, run.name)
            run.factor_dir = str(script.parent)
            output_dir = Path(run.factor_dir) / "outputs" / "runs" / run.run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            run.output_dir = str(output_dir)
            run_config = self._build_run_config(factor_cfg, run, output_dir)
            run_config_path = write_yaml_config(output_dir / "run_config.yaml", run_config)
            run.parameters = self._flat_parameters(run_config)
            run.compute_kwargs = dict(section(run_config, "factor_spec").get("compute_kwargs") or {})
            run.append_log(f"[dashboard] wrote run YAML: {run_config_path}\n")
            run.append_log(f"[dashboard] loading factor module: {script}\n")
            mod = load_factor_module(script)
            if hasattr(mod, "build_spec"):
                factor_spec = mod.build_spec(run_config, run_config_path)
            else:
                factor_spec = getattr(mod, "spec")
            run.append_log("[dashboard] factor module loaded\n")

            class_dir = script.parent.parent
            import sys
            for p in (class_dir.parent, class_dir):
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            # FactorPipeline 取自因子模块本身（已 import 各自类模板的 re-export），
            # 与因子类解耦：tdx 走 factor_template_tdx，alpha101 走 factor_template_alpha101。
            FactorPipeline = getattr(mod, "FactorPipeline")
            run.append_log(f"[dashboard] pipeline: {FactorPipeline.__name__}\n")

            start_date = run.parameters.get("start_date")
            end_date = run.parameters.get("end_date")
            if not start_date or not end_date:
                raise ValueError("start_date and end_date are required")

            kwargs = run_parameters(run_config, run_config_path)
            start_date = kwargs.pop("start_date")
            end_date = kwargs.pop("end_date")
            run_signature = inspect.signature(FactorPipeline.run)
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in run_signature.parameters.values()):
                kwargs = {key: value for key, value in kwargs.items() if key in run_signature.parameters}
            run.append_log(
                "[dashboard] starting pipeline "
                f"{start_date} -> {end_date}, kwargs={kwargs}, compute_kwargs={run.compute_kwargs}\n"
            )

            log_writer = LogBuffer(run)
            with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
                result = FactorPipeline(factor_spec).run(str(start_date), str(end_date), **kwargs)
            run.append_log("[dashboard] pipeline returned, serializing result\n")
            run.result = result
            run.backtest = result.backtest
            factor_values = getattr(result, "factor_values", None)
            run.append_log("[dashboard] resolving analyst\n")
            if result.analyst is not None:
                run.analyst = result.analyst
            else:
                from betalens.analyst import Analyst

                run.analyst = Analyst.from_backtest(
                    result.backtest,
                    name=run.name,
                    benchmark_code=run.parameters.get("benchmark_code"),
                    benchmark_metric=run.parameters.get("backtest_metric", "收盘价(元)"),
                    factor_values=factor_values,
                )
            run.append_log("[dashboard] analyst ready\n")

            # 巨表落 parquet → 缓存可 JSON 化 payload → 释放 bt(省内存)→ 后台异步写 dump
            run.append_log("[dashboard] persisting tables\n")
            self._persist_tables(run)
            run.append_log("[dashboard] tables persisted\n")
            bt = run.backtest
            pit_validation = getattr(result, "pit_validation", None)
            neutralize_stats = getattr(result, "neutralize_stats", None)
            from .serialization import build_result_payload, write_factor_values_parquet

            run.append_log("[dashboard] writing factor_values parquet\n")
            write_factor_values_parquet(factor_values, run.factor_values_path())
            name = run.name
            out_dir = output_dir
            run.append_log("[dashboard] building result payload\n")
            run.payload = build_result_payload(
                run,
                run.table_meta,
                factor_values,
                pit_validation=pit_validation,
                neutralize_stats=neutralize_stats,
            )
            run.append_log("[dashboard] marking run completed\n")
            run.mark_completed()
            run.payload["run"] = run.to_state().model_dump()
            run.backtest = None
            run.result = None
            run.analyst = None
            self._dump_executor.submit(
                self._dump_excel,
                bt,
                out_dir,
                name,
                factor_values,
                pit_validation,
                neutralize_stats,
            )
        except Exception as exc:
            run.mark_failed(exc)

    @staticmethod
    def _build_run_config(factor_cfg: dict[str, Any], run: DashboardRun, output_dir: Path) -> dict[str, Any]:
        config = copy.deepcopy(factor_cfg)
        meta = section(config, "meta")
        factor_spec = section(config, "factor_spec")
        weight = section(config, "weight")
        run_section = section(config, "run")
        strategy_type = str(meta.get("strategy_type") or "cross_sectional").strip()

        for key in (
            "direction",
            "index_code",
            "use_industry",
            "use_mktcap",
            "industry_scheme",
            "backtest_metric",
        ):
            if key in run.parameters:
                factor_spec[key] = run.parameters[key]
        if run.compute_kwargs:
            factor_spec["compute_kwargs"] = dict(run.compute_kwargs)

        weight_mapping = {
            "weight_mode": "mode",
            "long_groups": "long_groups",
            "short_groups": "short_groups",
            "group_weights": "group_weights",
            "intra_group_allocation": "intra_group_allocation",
        }
        for source, target in weight_mapping.items():
            if source in run.parameters:
                value = run.parameters[source]
                if source in ("long_groups", "short_groups"):
                    value = _normalize_group_list(value)
                weight[target] = value
        if weight.get("mode") == "freeplay":
            long_groups = _normalize_group_list(weight.get("long_groups"))
            short_groups = _normalize_group_list(weight.get("short_groups"))
            weight["long_groups"] = long_groups
            weight["short_groups"] = short_groups
            if strategy_type != "timing" and not long_groups and not short_groups:
                raise ValueError("freeplay 模式必须至少设置 long_groups 或 short_groups")

        for key in (
            "start_date",
            "end_date",
            "rebal_freq",
            "n_quantiles",
            "initial_amount",
            "benchmark_code",
            "include_profiling",
            "warmup_days",
            "pretom_only",
            "pretom_lo",
            "pretom_hi",
            "universe",
        ):
            if key in run.parameters:
                run_section[key] = run.parameters[key]
        run_section["dump_excel"] = False
        run_section["output_dir"] = str(output_dir.resolve())
        return config

    @staticmethod
    def _flat_parameters(config: dict[str, Any]) -> dict[str, Any]:
        factor_spec = section(config, "factor_spec")
        weight = section(config, "weight")
        run_section = section(config, "run")
        flat = dict(run_section)
        flat.update(
            {
                "direction": factor_spec.get("direction"),
                "index_code": factor_spec.get("index_code"),
                "use_industry": factor_spec.get("use_industry"),
                "use_mktcap": factor_spec.get("use_mktcap"),
                "industry_scheme": factor_spec.get("industry_scheme"),
                "backtest_metric": factor_spec.get("backtest_metric"),
                "weight_mode": weight.get("mode"),
                "long_groups": weight.get("long_groups"),
                "short_groups": weight.get("short_groups"),
                "group_weights": weight.get("group_weights", {}),
                "intra_group_allocation": weight.get("intra_group_allocation", {}),
            }
        )
        return flat

    def _persist_tables(self, run: DashboardRun) -> None:
        from .serialization import build_table, write_table_parquet

        for kind in TABLE_KINDS:
            run.append_log(f"[dashboard] building {kind} table\n")
            rows = build_table(run.backtest, kind)
            run.append_log(f"[dashboard] writing {kind} table rows={len(rows)}\n")
            run.table_meta[kind] = write_table_parquet(rows, run.table_path(kind))

    @staticmethod
    def _dump_excel(
        bt: Any,
        output_dir: Path,
        name: str,
        factor_values: Any = None,
        pit_validation: Any = None,
        neutralize_stats: Any = None,
    ) -> None:
        try:
            dump_path = f"{output_dir}/{name}_dump.xlsx"
            bt.dump_to_excel(dump_path)
            if (
                factor_values is not None
                or (pit_validation is not None and not pit_validation.empty)
                or (neutralize_stats is not None and not neutralize_stats.empty)
            ):
                import pandas as pd

                with pd.ExcelWriter(
                    dump_path,
                    engine="openpyxl",
                    mode="a",
                    if_sheet_exists="replace",
                ) as writer:
                    if factor_values is not None:
                        factor_values.to_excel(writer, sheet_name="factor_values", index=False)
                    if pit_validation is not None and not pit_validation.empty:
                        pit_validation.to_excel(writer, sheet_name="pit_validation")
                    if neutralize_stats is not None and not neutralize_stats.empty:
                        neutralize_stats.to_excel(writer, sheet_name="neutralize_stats")
        except Exception:
            pass  # dump 仅供下载,失败不影响已展示的结果

    def serialize_result(self, run_id: str) -> dict[str, Any]:
        from .serialization import build_downloads

        run = self.get(run_id)
        if run.payload is None:
            raise ValueError(f"run is {run.status}")
        payload = dict(run.payload)
        # downloads 实时探测磁盘(dump 是异步写的,可能稍后才出现)
        payload["downloads"] = build_downloads(Path(run.output_dir or run.factor_dir), run.name)
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
        from .serialization import read_table_page

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

    def factor_profile(
        self,
        run_id: str,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        from .serialization import read_factor_profile

        run = self.get(run_id)
        if run.payload is None:
            raise ValueError(f"run is {run.status}")
        return read_factor_profile(run.factor_values_path(), date_from=date_from, date_to=date_to)


manager = RunManager()
