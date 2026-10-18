"""Current table/column metadata for read-only inspection; no migration versions."""

from __future__ import annotations

ALL_BASE_TABLES = (
    "dataset_coverage",
    "entity_dim",
    "entity_name_history",
    "metric_dim",
    "metric_alias",
    "industry_scheme_dim",
    "industry_dim",
    "market_daily_fact",
    "observation_fact",
    "industry_membership",
    "index_snapshot",
    "index_constituent",
    "trade_status_event",
    "trade_calendar_day",
)

ALL_COMPATIBILITY_VIEWS = (
    "daily_market",
    "daily_index",
    "daily_fund",
    "daily_bond",
    "fundamentals",
    "macro",
    "factors",
    "industry",
    "index_universe",
    "trade_status",
    "trade_calendar",
)

TABLE_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "dataset_coverage": (
        ("logical_dataset", "character varying"),
        ("min_available_at", "timestamp without time zone"),
        ("max_available_at", "timestamp without time zone"),
        ("row_count", "bigint"),
        ("updated_at", "timestamp with time zone"),
        ("metadata", "jsonb"),
    ),
    "entity_dim": (
        ("entity_id", "bigint"),
        ("code", "character varying"),
        ("entity_type", "character varying"),
        ("current_name", "character varying"),
        ("first_trade_date", "date"),
        ("delist_date", "date"),
        ("created_at", "timestamp with time zone"),
        ("updated_at", "timestamp with time zone"),
    ),
    "entity_name_history": (
        ("entity_id", "bigint"),
        ("valid_from", "timestamp without time zone"),
        ("valid_to", "timestamp without time zone"),
        ("name", "character varying"),
    ),
    "metric_dim": (
        ("metric_id", "integer"),
        ("logical_dataset", "character varying"),
        ("metric_name", "character varying"),
        ("storage_kind", "character varying"),
        ("storage_column", "character varying"),
        ("availability_time", "time without time zone"),
        ("unit", "character varying"),
        ("description", "text"),
    ),
    "metric_alias": (
        ("logical_dataset", "character varying"),
        ("alias", "character varying"),
        ("metric_id", "integer"),
    ),
    "industry_scheme_dim": (
        ("scheme_id", "integer"),
        ("scheme_name", "character varying"),
        ("description", "text"),
    ),
    "industry_dim": (
        ("industry_id", "bigint"),
        ("scheme_id", "integer"),
        ("industry_code", "character varying"),
        ("industry_name", "character varying"),
    ),
    "market_daily_fact": (
        ("entity_id", "bigint"),
        ("trade_date", "date"),
        ("open", "double precision"),
        ("high", "double precision"),
        ("low", "double precision"),
        ("close", "double precision"),
        ("prev_close", "double precision"),
        ("volume", "double precision"),
        ("amount", "double precision"),
        ("turnover_rate", "double precision"),
        ("remark", "jsonb"),
        ("updated_at", "timestamp with time zone"),
    ),
    "observation_fact": (
        ("available_at", "timestamp without time zone"),
        ("entity_id", "bigint"),
        ("metric_id", "integer"),
        ("period_end", "date"),
        ("value", "double precision"),
        ("remark", "jsonb"),
        ("updated_at", "timestamp with time zone"),
    ),
    "industry_membership": (
        ("entity_id", "bigint"),
        ("industry_id", "bigint"),
        ("valid_from", "timestamp without time zone"),
        ("valid_to", "timestamp without time zone"),
        ("remark", "jsonb"),
    ),
    "index_snapshot": (
        ("snapshot_id", "bigint"),
        ("index_entity_id", "bigint"),
        ("effective_at", "timestamp without time zone"),
        ("index_name_snapshot", "character varying"),
        ("remark", "jsonb"),
    ),
    "index_constituent": (
        ("snapshot_id", "bigint"),
        ("constituent_entity_id", "bigint"),
        ("ordinal", "integer"),
        ("weight", "double precision"),
        ("remark", "jsonb"),
    ),
    "trade_status_event": (
        ("entity_id", "bigint"),
        ("event_date", "date"),
        ("status", "smallint"),
        ("status_text", "character varying"),
        ("remark", "jsonb"),
    ),
    "trade_calendar_day": (
        ("exchange", "character varying"),
        ("trade_date", "date"),
    ),
}

VIEW_COLUMNS = (
    ("datetime", "timestamp without time zone"),
    ("code", "character varying"),
    ("name", "character varying"),
    ("metric", "character varying"),
    ("value", "double precision"),
    ("remark", "jsonb"),
)

BASE_TABLES = ALL_BASE_TABLES
COMPATIBILITY_VIEWS = ALL_COMPATIBILITY_VIEWS
