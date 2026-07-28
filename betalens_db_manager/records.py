"""Compatibility wrapper for database-manager job persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import IMPORT_RECORDS_FILE, JOB_LOG_DIR, MANAGER_LOG_ROOT
from .job_store import JobStore


class ImportRecordStore:
    """Retain the old append/read API on top of the shared SQLite JobStore."""

    def __init__(
        self,
        records_file: Path = IMPORT_RECORDS_FILE,
        job_log_dir: Path = JOB_LOG_DIR,
        *,
        job_store: JobStore | None = None,
    ):
        self.records_file = Path(records_file)
        self.job_log_dir = Path(job_log_dir)
        if job_store is None:
            default_records = Path(IMPORT_RECORDS_FILE)
            sqlite_path = (
                MANAGER_LOG_ROOT / "jobs.sqlite3"
                if self.records_file == default_records
                else self.records_file.with_suffix(".sqlite3")
            )
            job_store = JobStore(sqlite_path, self.job_log_dir)
        self.job_store = job_store
        self.job_log_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_json_lines()

    def job_log_path(self, job_id: str) -> Path:
        return self.job_store.job_log_path(job_id)

    def append(self, record: dict[str, Any]) -> None:
        self.job_store.append_record(record)

    def read_all(self) -> list[dict[str, Any]]:
        return self.job_store.read_legacy_records()

    def _migrate_json_lines(self) -> None:
        """Import pre-redesign JSONL records once; upserts make this idempotent."""

        if not self.records_file.exists():
            return
        import json

        with self.records_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {
                        "job_id": f"corrupt-jsonl-{line_number}",
                        "status": "corrupt",
                        "raw": raw,
                    }
                if isinstance(payload, dict):
                    self.job_store.append_record(payload)

