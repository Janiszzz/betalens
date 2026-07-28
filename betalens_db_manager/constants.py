"""Shared constants for the Betalens database manager."""

from __future__ import annotations

from pathlib import Path


from .registry import LOGICAL_TABLES, WRITABLE_TABLES


# Backwards-compatible name used by the existing CLI/GUI.  It now describes
# logical datasets, not physical PostgreSQL tables.
ALLOWED_TABLES = LOGICAL_TABLES
ALLOWED_WRITE_TABLES = WRITABLE_TABLES

DEFAULT_LIMIT = 500
MAX_PREVIEW_ROWS = 100
DEFAULT_STATEMENT_TIMEOUT_MS = 15000

REQUIRED_DB_COLUMNS = ("datetime", "code", "name", "metric", "value")
OPTIONAL_DB_COLUMNS = ("remark",)
DB_COLUMNS = REQUIRED_DB_COLUMNS + OPTIONAL_DB_COLUMNS

MANAGER_LOG_ROOT = Path("logs") / "database-manager"
IMPORT_RECORDS_FILE = MANAGER_LOG_ROOT / "import_records.jsonl"
JOB_LOG_DIR = MANAGER_LOG_ROOT / "jobs"

INSERT_ONLY = "insert_only"
UPSERT = "upsert"
IMPORT_MODES = (INSERT_ONLY, UPSERT)

IMPORT_TYPES = (
    "auto",
    "standard_long",
    "wind_wide",
    "wind_long",
    "ede",
    "industry",
    "index_universe",
    "trade_status",
)

