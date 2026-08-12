"""Local database management tools for Betalens.

This package is intentionally separate from :mod:`betalens.datafeed`.
The datafeed package is the research/runtime data access layer; this package
owns schema inspection, imports, local import records, and the desktop GUI.
"""

from .constants import ALLOWED_TABLES, ALLOWED_WRITE_TABLES, DEFAULT_LIMIT
from .db import DatabaseClient, QueryRequest
from .importers import DatabaseWriter, DeleteRequest, load_trade_calendar
from .import_adapters import (
    ADAPTERS,
    AdapterRegistry,
    ImportBatch,
    IndexSnapshotBatch,
    IndustryBatch,
    MarketBatch,
    ObservationBatch,
    TradeStatusBatch,
    TradeCalendarBatch,
)
from .import_manifest import ManifestEntry, ManifestPlan, ManifestRunner
from .job_store import JobStore
from .jobs import ImportJobRunner
from .manager import DatabaseManager
from .profiles import ConnectionProfile, ConnectionResolver, ProfileStore, ResolvedConnection
from .registry import DATASETS, DatasetSpec, get_dataset
from .records import ImportRecordStore
from .schema import SchemaManager

__all__ = [
    "ALLOWED_TABLES",
    "ALLOWED_WRITE_TABLES",
    "ADAPTERS",
    "AdapterRegistry",
    "ConnectionProfile",
    "ConnectionResolver",
    "DATASETS",
    "DEFAULT_LIMIT",
    "DatabaseClient",
    "DatabaseManager",
    "DatabaseWriter",
    "DeleteRequest",
    "DatasetSpec",
    "ImportJobRunner",
    "ImportBatch",
    "ImportRecordStore",
    "IndexSnapshotBatch",
    "IndustryBatch",
    "JobStore",
    "ManifestEntry",
    "ManifestPlan",
    "ManifestRunner",
    "MarketBatch",
    "ObservationBatch",
    "ProfileStore",
    "QueryRequest",
    "SchemaManager",
    "ResolvedConnection",
    "TradeStatusBatch",
    "TradeCalendarBatch",
    "get_dataset",
    "load_trade_calendar",
]
