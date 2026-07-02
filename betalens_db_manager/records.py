"""Persistent local import records."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import IMPORT_RECORDS_FILE, JOB_LOG_DIR
from .utils import ensure_parent, to_json_line


class ImportRecordStore:
    def __init__(self, records_file: Path = IMPORT_RECORDS_FILE, job_log_dir: Path = JOB_LOG_DIR):
        self.records_file = records_file
        self.job_log_dir = job_log_dir
        ensure_parent(self.records_file)
        self.job_log_dir.mkdir(parents=True, exist_ok=True)

    def job_log_path(self, job_id: str) -> Path:
        return self.job_log_dir / f"{job_id}.log"

    def append(self, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("recorded_at", datetime.now().isoformat(sep=" ", timespec="seconds"))
        ensure_parent(self.records_file)
        with self.records_file.open("a", encoding="utf-8") as fh:
            fh.write(to_json_line(payload))

    def read_all(self) -> list[dict[str, Any]]:
        if not self.records_file.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.records_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"status": "corrupt", "raw": line})
        return rows

