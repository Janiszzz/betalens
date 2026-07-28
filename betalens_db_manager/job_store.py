"""Local SQLite persistence for database-manager runs and checkpoints."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .constants import JOB_LOG_DIR, MANAGER_LOG_ROOT
from .utils import json_default


DEFAULT_JOB_DATABASE = MANAGER_LOG_ROOT / "jobs.sqlite3"


def _now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=json_default, separators=(",", ":"))


def _loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


class JobStore:
    """Persist schema/import jobs without adding management tables to PostgreSQL."""

    def __init__(self, path: str | Path = DEFAULT_JOB_DATABASE, job_log_dir: str | Path = JOB_LOG_DIR):
        self.path = Path(path)
        self.job_log_dir = Path(job_log_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.job_log_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS job_run (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    target_database TEXT,
                    schema_version INTEGER,
                    progress REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    log_path TEXT,
                    report_path TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS job_run_started_idx
                    ON job_run(started_at DESC, created_at DESC);
                CREATE INDEX IF NOT EXISTS job_run_kind_status_idx
                    ON job_run(kind, status);

                CREATE TABLE IF NOT EXISTS job_item (
                    item_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES job_run(job_id) ON DELETE CASCADE,
                    item_key TEXT,
                    status TEXT NOT NULL,
                    source_path TEXT,
                    source_hash TEXT,
                    target TEXT,
                    adapter TEXT,
                    mode TEXT,
                    position INTEGER,
                    total_rows INTEGER,
                    inserted_rows INTEGER,
                    updated_rows INTEGER,
                    skipped_rows INTEGER,
                    rejected_rows INTEGER,
                    started_at TEXT,
                    finished_at TEXT,
                    error TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE UNIQUE INDEX IF NOT EXISTS job_item_key_idx
                    ON job_item(job_id, item_key) WHERE item_key IS NOT NULL;

                CREATE TABLE IF NOT EXISTS checkpoint (
                    token TEXT PRIMARY KEY,
                    target_database TEXT,
                    schema_version INTEGER,
                    item_key TEXT,
                    source_hash TEXT,
                    status TEXT,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS checkpoint_resume_idx
                    ON checkpoint(target_database, schema_version, item_key, source_hash, status);
                """
            )

    def job_log_path(self, job_id: str) -> Path:
        return self.job_log_dir / f"{job_id}.log"

    def create_job(
        self,
        kind: str,
        *,
        job_id: str | None = None,
        status: str = "planned",
        target_database: str | None = None,
        schema_version: int | None = None,
        log_path: str | Path | None = None,
        report_path: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        identifier = job_id or f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        created_at = _now()
        actual_log_path = Path(log_path) if log_path else self.job_log_path(identifier)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO job_run (
                    job_id, kind, status, target_database, schema_version,
                    created_at, log_path, report_path, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    str(kind),
                    str(status),
                    target_database,
                    schema_version,
                    created_at,
                    str(actual_log_path),
                    str(report_path) if report_path else None,
                    _json(dict(metadata or {})),
                ),
            )
        return self.get_job(identifier) or {"job_id": identifier}

    def update_job(self, job_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {
            "kind",
            "status",
            "target_database",
            "schema_version",
            "progress",
            "started_at",
            "finished_at",
            "log_path",
            "report_path",
            "error",
            "cancel_requested",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in changes.items():
            if key in {"metadata", "result"}:
                assignments.append(f"{key}_json = ?")
                values.append(_json(value or {}))
            elif key in allowed:
                assignments.append(f"{key} = ?")
                values.append(int(bool(value)) if key == "cancel_requested" else value)
        if not assignments:
            return self.get_job(job_id) or {}
        values.append(job_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE job_run SET {', '.join(assignments)} WHERE job_id = ?",
                values,
            )
            if cursor.rowcount == 0:
                raise KeyError(f"未知任务: {job_id}")
        return self.get_job(job_id) or {}

    def start_job(self, job_id: str) -> dict[str, Any]:
        return self.update_job(job_id, status="running", started_at=_now(), progress=0.0)

    def finish_job(
        self,
        job_id: str,
        status: str,
        *,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        progress = 1.0 if status in {"completed", "completed_with_errors"} else 0.0
        return self.update_job(
            job_id,
            status=status,
            finished_at=_now(),
            progress=progress,
            result=dict(result or {}),
            error=error,
        )

    def request_cancel(self, job_id: str) -> None:
        self.update_job(job_id, cancel_requested=True)

    def is_cancel_requested(self, job_id: str) -> bool:
        row = self.get_job(job_id)
        return bool(row and row.get("cancel_requested"))

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM job_run WHERE job_id = ?", (job_id,)).fetchone()
        return self._job_row(row) if row else None

    def list_jobs(
        self,
        *,
        limit: int = 100,
        kind: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if kind:
            clauses.append("kind = ?")
            values.append(kind)
        if status:
            clauses.append("status = ?")
            values.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.append(max(1, min(int(limit), 5000)))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM job_run{where} ORDER BY COALESCE(started_at, created_at) DESC LIMIT ?",
                values,
            ).fetchall()
        return [self._job_row(row) for row in rows]

    @staticmethod
    def _job_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["cancel_requested"] = bool(result["cancel_requested"])
        result["metadata"] = _loads(result.pop("metadata_json"), {})
        payload = _loads(result.pop("result_json"), {})
        result["result"] = payload
        return result

    def create_item(
        self,
        job_id: str,
        *,
        item_id: str | None = None,
        item_key: str | None = None,
        status: str = "pending",
        **fields: Any,
    ) -> dict[str, Any]:
        identifier = item_id or f"{job_id}:{uuid.uuid4().hex[:8]}"
        columns = ["item_id", "job_id", "item_key", "status"]
        values: list[Any] = [identifier, job_id, item_key, status]
        allowed = {
            "source_path", "source_hash", "target", "adapter", "mode", "position",
            "total_rows", "inserted_rows", "updated_rows", "skipped_rows", "rejected_rows",
            "started_at", "finished_at", "error",
        }
        for key, value in fields.items():
            if key in allowed:
                columns.append(key)
                values.append(value)
        columns.append("payload_json")
        values.append(_json(fields.get("payload", {})))
        placeholders = ", ".join("?" for _ in columns)
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO job_item ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
        return self.get_item(identifier) or {"item_id": identifier}

    def update_item(self, item_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {
            "status", "source_path", "source_hash", "target", "adapter", "mode", "position",
            "total_rows", "inserted_rows", "updated_rows", "skipped_rows", "rejected_rows",
            "started_at", "finished_at", "error",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in changes.items():
            if key == "payload":
                assignments.append("payload_json = ?")
                values.append(_json(value or {}))
            elif key in allowed:
                assignments.append(f"{key} = ?")
                values.append(value)
        if assignments:
            values.append(item_id)
            with self._connect() as connection:
                cursor = connection.execute(
                    f"UPDATE job_item SET {', '.join(assignments)} WHERE item_id = ?",
                    values,
                )
                if cursor.rowcount == 0:
                    raise KeyError(f"未知任务项: {item_id}")
        return self.get_item(item_id) or {}

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM job_item WHERE item_id = ?", (item_id,)).fetchone()
        return self._item_row(row) if row else None

    def list_items(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM job_item WHERE job_id = ? ORDER BY position, rowid",
                (job_id,),
            ).fetchall()
        return [self._item_row(row) for row in rows]

    @staticmethod
    def _item_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = _loads(result.pop("payload_json"), {})
        return result

    def save(self, token: str, payload: Mapping[str, Any]) -> None:
        """Save a ManifestRunner-compatible checkpoint payload."""

        data = dict(payload)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO checkpoint (
                    token, target_database, schema_version, item_key,
                    source_hash, status, payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(token) DO UPDATE SET
                    target_database=excluded.target_database,
                    schema_version=excluded.schema_version,
                    item_key=excluded.item_key,
                    source_hash=excluded.source_hash,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    token,
                    data.get("target_database"),
                    data.get("schema_version"),
                    data.get("item_id") or data.get("item_key"),
                    data.get("source_hash") or data.get("sha256"),
                    data.get("status"),
                    _json(data),
                    _now(),
                ),
            )

    def load(self, token: str) -> dict[str, Any] | None:
        """Load a ManifestRunner-compatible checkpoint payload."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM checkpoint WHERE token = ?",
                (token,),
            ).fetchone()
        return _loads(row["payload_json"], {}) if row else None

    def append_record(self, record: Mapping[str, Any]) -> None:
        """Ingest a legacy ImportRecordStore record into the run table."""

        payload = dict(record)
        job_id = str(payload.get("job_id") or f"legacy_{uuid.uuid4().hex}")
        started_at = payload.get("started_at") or payload.get("recorded_at") or _now()
        kind = str(payload.get("kind") or ("import" if payload.get("source_file") else "legacy"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO job_run (
                    job_id, kind, status, target_database, schema_version,
                    progress, created_at, started_at, finished_at, log_path,
                    report_path, error, metadata_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    finished_at=excluded.finished_at,
                    log_path=excluded.log_path,
                    report_path=excluded.report_path,
                    error=excluded.error,
                    result_json=excluded.result_json
                """,
                (
                    job_id,
                    kind,
                    str(payload.get("status", "unknown")),
                    payload.get("target_database"),
                    payload.get("schema_version"),
                    1.0 if payload.get("status") == "completed" else 0.0,
                    payload.get("recorded_at") or started_at,
                    started_at,
                    payload.get("finished_at"),
                    payload.get("log_path"),
                    payload.get("report_path"),
                    payload.get("error"),
                    _json(payload),
                ),
            )

    def read_legacy_records(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT result_json, job_id, status, created_at FROM job_run ORDER BY created_at, rowid"
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            payload = _loads(row["result_json"], {})
            if not payload:
                payload = {
                    "job_id": row["job_id"],
                    "status": row["status"],
                    "recorded_at": row["created_at"],
                }
            records.append(payload)
        return records

