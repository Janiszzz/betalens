"""Import job orchestration and persistent logging."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .constants import INSERT_ONLY
from .db import DatabaseClient
from .importers import DatabaseWriter, infer_import_type, load_import_frame
from .records import ImportRecordStore
from .utils import dataframe_summary, file_sha256
from .validators import validate_import_frame


class ImportJobRunner:
    def __init__(self, client: DatabaseClient | None = None, store: ImportRecordStore | None = None):
        self.client = client or DatabaseClient()
        self.store = store or ImportRecordStore()

    def preview(self, path: str | Path, import_type: str | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]:
        import_type = import_type or infer_import_type(path)
        df = load_import_frame(import_type, path, options=options)
        report = validate_import_frame(df)
        return {
            "import_type": import_type,
            "summary": dataframe_summary(df),
            "validation": report.__dict__,
            "columns": list(df.columns),
            "preview": df.head(100).where(df.notnull(), None).to_dict("records"),
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
    ) -> dict[str, Any]:
        path = Path(path)
        import_type = import_type or infer_import_type(path)
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        log_path = self.store.job_log_path(job_id)
        logger = self._build_logger(job_id, log_path, progress)
        record: dict[str, Any] = {
            "job_id": job_id,
            "started_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "status": "running",
            "source_kind": "file",
            "source_file": str(path),
            "file_sha256": file_sha256(path),
            "import_type": import_type,
            "target_table": table,
            "mode": mode,
            "log_path": str(log_path),
        }
        try:
            logger.info("loading source file")
            df = load_import_frame(import_type, path, options=options, logger=logger)
            record.update(dataframe_summary(df))
            logger.info("validating import frame")
            report = validate_import_frame(df, allow_unsafe_metrics=allow_unsafe_metrics, allow_nan_values=allow_nan_values)
            if not report.ok:
                raise ValueError("; ".join(report.errors))
            writer = DatabaseWriter(self.client)
            logger.info("running dry-run conflict check")
            dry_run = writer.dry_run(table, df)
            record["conflict_count"] = dry_run.get("conflict_count", len(dry_run.get("conflicts", [])))
            record["conflict_samples"] = dry_run.get("conflicts", [])
            record["existing_rows"] = dry_run.get("existing", 0)
            logger.info("writing data")
            result = writer.write(table, df, mode=mode)
            record.update(
                {
                    "status": "completed",
                    "finished_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
                    "total_rows": result["total"],
                    "inserted_rows": result["inserted"],
                    "skipped_rows": result["skipped"],
                    "warnings": report.warnings,
                }
            )
            logger.info("job completed")
        except Exception as exc:
            logger.exception("job failed")
            record.update(
                {
                    "status": "failed",
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
