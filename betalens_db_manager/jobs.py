"""Import job orchestration and persistent logging."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .constants import DB_COLUMNS, INSERT_ONLY
from .db import DatabaseClient
from .import_adapters import infer_adapter, load_import_batches
from .importers import DatabaseWriter
from .records import ImportRecordStore
from .utils import file_sha256
from .validators import validate_import_frame


class ImportCancelled(RuntimeError):
    """Signal that the current file transaction must be rolled back."""


class ImportJobRunner:
    def __init__(self, client: DatabaseClient | None = None, store: ImportRecordStore | None = None):
        self.client = client or DatabaseClient()
        self.store = store or ImportRecordStore()

    def preview(
        self,
        path: str | Path,
        import_type: str | None = None,
        options: dict[str, Any] | None = None,
        *,
        table: str | None = None,
        mode: str = INSERT_ONLY,
        conflict_sample_limit: int = 20,
        inspect_database: bool = True,
    ) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()
        adapter = import_type or infer_adapter(source)
        target = table or self._default_target(adapter)
        source_sha256 = file_sha256(source)
        preview_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []
        stats = {"rows": 0, "source_rows": 0, "rejected_rows": 0, "codes": set(), "metrics": set(), "date_min": None, "date_max": None}
        changes = {"existing": 0, "conflict_count": 0, "new_rows_estimate": 0, "conflicts": []}
        writer = DatabaseWriter(self.client)
        for batch in load_import_batches(adapter, source, table=target, options=options):
            self._accumulate_batch_stats(stats, batch)
            if len(preview_rows) < 100:
                preview_rows.extend(
                    batch.frame.head(100 - len(preview_rows)).where(batch.frame.notnull(), None).to_dict("records")
                )
            if len(rejected_rows) < 100 and not batch.rejected.empty:
                rejected_rows.extend(batch.rejected.head(100 - len(rejected_rows)).to_dict("records"))
            if not batch.frame.empty:
                report = validate_import_frame(batch.frame)
                errors.extend(report.errors)
                warnings.extend(report.warnings)
            if inspect_database and not batch.frame.empty:
                dry_run = writer.dry_run(target, batch.frame, conflict_sample_limit)
                changes["existing"] += dry_run.get("existing", 0)
                changes["conflict_count"] += dry_run.get("conflict_count", 0)
                changes["new_rows_estimate"] += dry_run.get("new_rows_estimate", 0)
                remaining = conflict_sample_limit - len(changes["conflicts"])
                changes["conflicts"].extend(dry_run.get("conflicts", [])[: max(0, remaining)])
        if stats["rejected_rows"]:
            errors.append(f"源文件包含 rejected rows: {stats['rejected_rows']}")
        summary = self._finalize_stats(stats)
        token = self.preview_token(source_sha256, target, adapter, mode, options or {})
        return {
            "import_type": adapter,
            "target_table": target,
            "mode": mode,
            "source_file": str(source),
            "source_sha256": source_sha256,
            "preview_token": token,
            "summary": summary,
            "validation": {"ok": not errors, "errors": list(dict.fromkeys(errors)), "warnings": list(dict.fromkeys(warnings)), "stats": summary},
            "columns": list(DB_COLUMNS),
            "preview": preview_rows,
            "rejected_preview": rejected_rows,
            "changes": changes,
        }

    def run(
        self,
        path: str | Path,
        table: str,
        import_type: str | None = None,
        mode: str = INSERT_ONLY,
        options: dict[str, Any] | None = None,
        allow_unsafe_metrics: bool = False,
        allow_nan_values: bool = False,
        progress: Callable[[str], None] | None = None,
        on_rejected: str = "fail",
        expected_sha256: str | None = None,
        preview_token: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        path = Path(path).expanduser().resolve()
        import_type = import_type or infer_adapter(path)
        if on_rejected not in {"fail", "skip"}:
            raise ValueError("on_rejected 必须是 fail 或 skip")
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        log_path = self.store.job_log_path(job_id)
        rejected_path = log_path.with_suffix(".rejected.csv")
        logger = self._build_logger(job_id, log_path, progress)
        record: dict[str, Any] = {
            "job_id": job_id,
            "started_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "status": "running",
            "source_kind": "file",
            "source_file": str(path),
            "import_type": import_type,
            "target_table": table,
            "mode": mode,
            "options": options or {},
            "log_path": str(log_path),
        }
        state: dict[str, Any] = {
            "source_rows": 0,
            "valid_rows": 0,
            "rejected_rows": 0,
            "codes": set(),
            "metrics": set(),
            "date_min": None,
            "date_max": None,
            "rejected_header": False,
            "warnings": [],
        }
        try:
            if cancel_check and cancel_check():
                raise ImportCancelled("导入任务已取消")
            logger.info("hashing source file")
            try:
                source_sha256 = file_sha256(path)
            except OSError as exc:
                raise ValueError(f"检查后无法读取文件: {exc}") from exc
            record["file_sha256"] = source_sha256
            if expected_sha256 and source_sha256.lower() != expected_sha256.lower():
                raise ValueError(f"源文件 SHA256 不匹配: expected={expected_sha256}, actual={source_sha256}")
            actual_token = self.preview_token(source_sha256, table, import_type, mode, options or {})
            record["preview_token"] = actual_token
            if preview_token and preview_token != actual_token:
                raise ValueError("preview_token 已失效；文件、目标、模式或 options 已变化")

            logger.info("loading, validating and writing source batches")
            def valid_frames():
                for batch_number, batch in enumerate(
                    load_import_batches(import_type, path, table=table, options=options, logger=logger),
                    start=1,
                ):
                    if cancel_check and cancel_check():
                        raise ImportCancelled("导入任务已取消；当前文件已回滚")
                    self._accumulate_batch_stats(state, batch)
                    if not batch.rejected.empty:
                        batch.rejected.to_csv(
                            rejected_path,
                            mode="a" if state["rejected_header"] else "w",
                            header=not state["rejected_header"],
                            index=False,
                            encoding="utf-8-sig",
                        )
                        state["rejected_header"] = True
                        if on_rejected == "fail":
                            sample = batch.rejected[["_source_row", "_errors"]].head(10).to_dict("records")
                            raise ValueError(f"源文件包含 rejected rows: {sample}")
                    if batch.frame.empty:
                        continue
                    validation = validate_import_frame(
                        batch.frame,
                        allow_unsafe_metrics=allow_unsafe_metrics,
                        allow_nan_values=False,
                    )
                    state["warnings"].extend(validation.warnings)
                    if not validation.ok:
                        raise ValueError("; ".join(validation.errors))
                    logger.info("validated batch %s: %s rows", batch_number, len(batch.frame))
                    yield batch.frame

                if cancel_check and cancel_check():
                    raise ImportCancelled("导入任务已取消；当前文件已回滚")

            writer = DatabaseWriter(self.client)
            result = writer.write_batches(
                table,
                valid_frames(),
                mode=mode,
                progress=lambda event: logger.info(
                    "wrote batch %(batch)s: %(rows)s rows; total=%(total_rows)s", event
                ),
            )
            summary = self._finalize_stats(state)
            record.update(summary)
            record["rejected_path"] = str(rejected_path) if state["rejected_rows"] else None
            record["conflict_count"] = result.get("conflict_count", 0)
            record["conflict_samples"] = result.get("conflicts", [])
            record["existing_rows"] = result.get("existing", 0)
            record.update(
                {
                    "status": "completed",
                    "finished_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
                    "total_rows": result["total"],
                    "inserted_rows": result["inserted"],
                    "updated_rows": result.get("updated", 0),
                    "skipped_rows": result["skipped"],
                    "warnings": list(dict.fromkeys(state["warnings"])),
                }
            )
            if allow_nan_values:
                record["warnings"].append(
                    "allow_nan_values 已弃用；非有限值不会写入，只能通过 on_rejected='skip' 跳过"
                )
            logger.info("job completed")
        except Exception as exc:
            logger.exception("job failed")
            record.update(self._finalize_stats(state))
            if state["rejected_rows"]:
                record["rejected_path"] = str(rejected_path)
            record.update(
                {
                    "status": "canceled" if isinstance(exc, ImportCancelled) else "failed",
                    "finished_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
                    "error": str(exc),
                }
            )
        finally:
            self.store.append(record)
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
        return record

    @staticmethod
    def _default_target(import_type: str) -> str:
        return {
            "industry": "industry",
            "index_universe": "index_universe",
            "trade_status": "trade_status",
        }.get(import_type, "daily_market")

    @staticmethod
    def preview_token(
        source_sha256: str,
        table: str,
        import_type: str,
        mode: str,
        options: dict[str, Any],
    ) -> str:
        payload = json.dumps(
            {
                "source_sha256": source_sha256,
                "table": table,
                "import_type": import_type,
                "mode": mode,
                "options": options,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _accumulate_batch_stats(stats: dict[str, Any], batch) -> None:
        stats["source_rows"] = int(stats.get("source_rows", 0)) + int(batch.source_rows)
        stats["rows"] = int(stats.get("rows", stats.get("valid_rows", 0))) + len(batch.frame)
        stats["valid_rows"] = int(stats.get("valid_rows", 0)) + len(batch.frame)
        stats["rejected_rows"] = int(stats.get("rejected_rows", 0)) + len(batch.rejected)
        stats.setdefault("codes", set()).update(batch.frame["code"].dropna().astype(str).unique())
        stats.setdefault("metrics", set()).update(batch.frame["metric"].dropna().astype(str).unique())
        if not batch.frame.empty:
            minimum = batch.frame["datetime"].min()
            maximum = batch.frame["datetime"].max()
            stats["date_min"] = minimum if stats.get("date_min") is None else min(stats["date_min"], minimum)
            stats["date_max"] = maximum if stats.get("date_max") is None else max(stats["date_max"], maximum)

    @staticmethod
    def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
        return {
            "rows": int(stats.get("valid_rows", stats.get("rows", 0))),
            "source_rows": int(stats.get("source_rows", 0)),
            "rejected_rows": int(stats.get("rejected_rows", 0)),
            "codes": len(stats.get("codes", set())),
            "metrics": len(stats.get("metrics", set())),
            "date_min": stats.get("date_min"),
            "date_max": stats.get("date_max"),
        }

    def _build_logger(self, job_id: str, log_path: Path, progress: Callable[[str], None] | None) -> logging.Logger:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"betalens_db_manager.{job_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        if progress is not None:
            logger.addHandler(_CallbackHandler(progress, fmt))
        return logger


class _CallbackHandler(logging.Handler):
    def __init__(self, callback: Callable[[str], None], formatter: logging.Formatter):
        super().__init__()
        self.callback = callback
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        self.callback(self.format(record))


__all__ = ["ImportCancelled", "ImportJobRunner"]
