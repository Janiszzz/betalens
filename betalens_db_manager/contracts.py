"""Versioned database contracts used by schema installation and verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping


LEGACY_MIGRATION_VERSION = 6
COMPATIBILITY_VIEW_VERSION = 7
FINALIZE_VERSION = 8
LIFECYCLE_AUDIT_VERSION = 9
LATEST_SCHEMA_VERSION = LIFECYCLE_AUDIT_VERSION

ALL_BASE_TABLES = (
    "schema_migration",
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
)

TABLE_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "schema_migration": (
        ("version", "integer"),
        ("name", "text"),
        ("checksum", "character"),
        ("applied_at", "timestamp with time zone"),
        ("execution_ms", "integer"),
    ),
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
}

REQUIRED_INDEXES: dict[str, tuple[str, ...]] = {
    "entity_dim": ("entity_dim_pkey", "entity_dim_code_key", "idx_entity_dim_type_code"),
    "entity_name_history": ("entity_name_history_pkey", "idx_entity_name_history_asof"),
    "metric_dim": ("metric_dim_pkey", "uq_metric_dim_dataset_name", "idx_metric_dim_storage"),
    "metric_alias": ("metric_alias_pkey", "idx_metric_alias_metric"),
    "industry_dim": (
        "industry_dim_pkey",
        "industry_dim_scheme_id_industry_code_key",
        "idx_industry_dim_scheme_name",
    ),
    "market_daily_fact": (
        "market_daily_fact_pkey",
        "idx_market_daily_fact_trade_date_entity",
    ),
    "observation_fact": (
        "observation_fact_pkey",
        "idx_observation_metric_time_entity",
    ),
    "industry_membership": (
        "industry_membership_pkey",
        "idx_industry_membership_entity_asof",
        "idx_industry_membership_industry_asof",
    ),
    "index_snapshot": ("index_snapshot_pkey", "uq_index_snapshot_entity_effective"),
    "index_constituent": ("index_constituent_pkey", "idx_index_constituent_entity"),
    "trade_status_event": ("trade_status_event_pkey", "idx_trade_status_event_date_entity"),
}

REQUIRED_CONSTRAINTS: dict[str, tuple[str, ...]] = {
    "schema_migration": ("schema_migration_pkey",),
    "dataset_coverage": ("dataset_coverage_pkey",),
    "entity_dim": ("entity_dim_pkey", "entity_dim_code_key"),
    "entity_name_history": ("entity_name_history_pkey",),
    "metric_dim": (
        "metric_dim_pkey",
        "uq_metric_dim_dataset_name",
        "uq_metric_dim_id_dataset",
    ),
    "metric_alias": ("metric_alias_pkey", "fk_metric_alias_metric_dataset"),
    "industry_scheme_dim": (
        "industry_scheme_dim_pkey",
        "industry_scheme_dim_scheme_name_key",
    ),
    "industry_dim": ("industry_dim_pkey", "industry_dim_scheme_id_industry_code_key"),
    "market_daily_fact": ("market_daily_fact_pkey",),
    "observation_fact": ("observation_fact_pkey",),
    "industry_membership": ("industry_membership_pkey",),
    "index_snapshot": ("index_snapshot_pkey", "uq_index_snapshot_entity_effective"),
    "index_constituent": ("index_constituent_pkey",),
    "trade_status_event": ("trade_status_event_pkey",),
}

INDEX_DEFINITION_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "idx_entity_dim_type_code": ("entity_type", "code"),
    "idx_entity_name_history_asof": ("entity_id", "valid_from DESC", "valid_to"),
    "idx_metric_dim_storage": ("logical_dataset", "storage_kind", "storage_column"),
    "idx_metric_alias_metric": ("metric_id",),
    "idx_industry_dim_scheme_name": ("scheme_id", "industry_name"),
    "idx_market_daily_fact_trade_date_entity": ("trade_date", "entity_id"),
    "idx_observation_metric_time_entity": ("metric_id", "available_at", "entity_id"),
    "idx_industry_membership_entity_asof": ("entity_id", "valid_from DESC", "valid_to", "industry_id"),
    "idx_industry_membership_industry_asof": ("industry_id", "valid_from", "valid_to", "entity_id"),
    "idx_index_constituent_entity": ("constituent_entity_id", "snapshot_id"),
    "idx_trade_status_event_date_entity": ("event_date", "entity_id"),
}

VIEW_COLUMNS = (
    ("datetime", "timestamp without time zone"),
    ("code", "character varying"),
    ("name", "character varying"),
    ("metric", "character varying"),
    ("value", "double precision"),
    ("remark", "jsonb"),
)

NOT_NULL_COLUMNS: dict[str, tuple[str, ...]] = {
    "schema_migration": ("version", "name", "checksum", "applied_at", "execution_ms"),
    "dataset_coverage": ("logical_dataset", "row_count", "updated_at", "metadata"),
    "entity_dim": (
        "entity_id", "code", "entity_type", "current_name", "created_at", "updated_at"
    ),
    "entity_name_history": ("entity_id", "valid_from", "name"),
    "metric_dim": (
        "metric_id", "logical_dataset", "metric_name", "storage_kind", "availability_time"
    ),
    "metric_alias": ("logical_dataset", "alias", "metric_id"),
    "industry_scheme_dim": ("scheme_id", "scheme_name"),
    "industry_dim": ("industry_id", "scheme_id", "industry_code", "industry_name"),
    "market_daily_fact": ("entity_id", "trade_date", "updated_at"),
    "observation_fact": ("available_at", "entity_id", "metric_id", "updated_at"),
    "industry_membership": ("entity_id", "industry_id", "valid_from"),
    "index_snapshot": ("snapshot_id", "index_entity_id", "effective_at", "index_name_snapshot"),
    "index_constituent": ("snapshot_id", "constituent_entity_id"),
    "trade_status_event": ("entity_id", "event_date", "status", "status_text"),
}

IDENTITY_COLUMNS: dict[str, tuple[str, ...]] = {
    "entity_dim": ("entity_id",),
    "metric_dim": ("metric_id",),
    "industry_scheme_dim": ("scheme_id",),
    "industry_dim": ("industry_id",),
    "index_snapshot": ("snapshot_id",),
}

DEFAULT_DEFINITION_FRAGMENTS: dict[tuple[str, str], str] = {
    ("schema_migration", "applied_at"): "clock_timestamp()",
    ("schema_migration", "execution_ms"): "0",
    ("dataset_coverage", "row_count"): "0",
    ("dataset_coverage", "updated_at"): "clock_timestamp()",
    ("dataset_coverage", "metadata"): "'{}'::jsonb",
    ("entity_dim", "entity_type"): "'unknown'::character varying",
    ("entity_dim", "current_name"): "''::character varying",
    ("entity_dim", "created_at"): "clock_timestamp()",
    ("entity_dim", "updated_at"): "clock_timestamp()",
    ("metric_dim", "storage_kind"): "'observation'::character varying",
    ("metric_dim", "availability_time"): "'15:00:01'::time without time zone",
    ("market_daily_fact", "updated_at"): "clock_timestamp()",
    ("observation_fact", "updated_at"): "clock_timestamp()",
    ("index_snapshot", "index_name_snapshot"): "''::character varying",
    ("trade_status_event", "status_text"): "''::character varying",
}

REQUIRED_CONSTRAINT_TYPES: dict[str, tuple[str, ...]] = {
    "schema_migration": ("p", "c"),
    "dataset_coverage": ("p", "c"),
    "entity_dim": ("p", "u", "c"),
    "entity_name_history": ("p", "f", "c"),
    "metric_dim": ("p", "u", "c"),
    "metric_alias": ("p", "f"),
    "industry_scheme_dim": ("p", "u"),
    "industry_dim": ("p", "u", "f"),
    "market_daily_fact": ("p", "f"),
    "observation_fact": ("p", "f"),
    "industry_membership": ("p", "f", "c"),
    "index_snapshot": ("p", "u", "f"),
    "index_constituent": ("p", "f", "c"),
    "trade_status_event": ("p", "f", "c"),
}

CONSTRAINT_DEFINITION_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "market_daily_fact_pkey": ("PRIMARY KEY", "entity_id", "trade_date"),
    "observation_fact_pkey": ("PRIMARY KEY", "entity_id", "metric_id", "available_at"),
    "uq_metric_dim_dataset_name": ("UNIQUE", "logical_dataset", "metric_name"),
    "fk_metric_alias_metric_dataset": (
        "FOREIGN KEY", "metric_id", "logical_dataset", "metric_dim", "ON DELETE CASCADE"
    ),
    "industry_membership_pkey": ("PRIMARY KEY", "entity_id", "industry_id", "valid_from"),
    "uq_index_snapshot_entity_effective": ("UNIQUE", "index_entity_id", "effective_at"),
    "index_constituent_pkey": ("PRIMARY KEY", "snapshot_id", "constituent_entity_id"),
    "trade_status_event_pkey": ("PRIMARY KEY", "entity_id", "event_date"),
}

COMMENTED_TABLES = ALL_BASE_TABLES

REQUIRED_FUNCTIONS = (
    "betalens.entity_name_at(bigint,timestamp without time zone)",
)

LIFECYCLE_FUNCTIONS = REQUIRED_FUNCTIONS + (
    "betalens.assert_legacy_equivalence()",
)

VIEW_DEFINITION_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "daily_market": ("betalens.market_daily_fact", "betalens.observation_fact", "daily_market", "entity_name_at"),
    "daily_index": ("betalens.market_daily_fact", "betalens.observation_fact", "daily_index", "entity_name_at"),
    "daily_fund": ("betalens.market_daily_fact", "betalens.observation_fact", "daily_fund", "entity_name_at"),
    "daily_bond": ("betalens.market_daily_fact", "betalens.observation_fact", "daily_bond", "entity_name_at"),
    "fundamentals": ("betalens.observation_fact", "fundamentals", "entity_name_at"),
    "macro": ("betalens.observation_fact", "macro", "entity_name_at"),
    "factors": ("betalens.observation_fact", "factors", "entity_name_at"),
    "industry": ("betalens.industry_membership", "betalens.industry_dim", "betalens.industry_scheme_dim"),
    "index_universe": ("betalens.index_snapshot", "betalens.index_constituent", "jsonb_agg"),
    "trade_status": ("first_trade_date", "betalens.trade_status_event", "entity_name_at"),
}

VIEW_DEFINITION_HASHES: dict[str, str] = {
    name: hashlib.sha256("\x1f".join(fragments).encode("utf-8")).hexdigest()
    for name, fragments in VIEW_DEFINITION_FRAGMENTS.items()
}

CORE_METRIC_SEEDS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "daily_market": (
        ("开盘价(元)", "open", "09:30:01"),
        ("最高价(元)", "high", "15:00:01"),
        ("最低价(元)", "low", "15:00:01"),
        ("收盘价(元)", "close", "15:00:01"),
        ("前收盘价", "prev_close", "09:30:01"),
        ("成交量(股)", "volume", "15:00:01"),
        ("成交金额(元)", "amount", "15:00:01"),
        ("换手率(%)", "turnover_rate", "15:00:01"),
    ),
    "daily_index": (
        ("开盘价", "open", "09:30:01"),
        ("最高价", "high", "15:00:01"),
        ("最低价", "low", "15:00:01"),
        ("收盘价", "close", "15:00:01"),
        ("前收盘价", "prev_close", "09:30:01"),
        ("成交量", "volume", "15:00:01"),
        ("成交额", "amount", "15:00:01"),
        ("换手率", "turnover_rate", "15:00:01"),
    ),
    "daily_fund": (
        ("开盘价(元)", "open", "09:30:01"),
        ("最高价(元)", "high", "15:00:01"),
        ("最低价(元)", "low", "15:00:01"),
        ("收盘价(元)", "close", "15:00:01"),
        ("前收盘价", "prev_close", "09:30:01"),
        ("成交量(份)", "volume", "15:00:01"),
        ("成交额(元)", "amount", "15:00:01"),
        ("换手率(%)", "turnover_rate", "15:00:01"),
    ),
    "daily_bond": (
        ("开盘价(元)", "open", "09:30:01"),
        ("最高价(元)", "high", "15:00:01"),
        ("最低价(元)", "low", "15:00:01"),
        ("收盘价(元)", "close", "15:00:01"),
        ("前收盘价", "prev_close", "09:30:01"),
        ("成交量(手)", "volume", "15:00:01"),
        ("成交额(元)", "amount", "15:00:01"),
        ("换手率(%)", "turnover_rate", "15:00:01"),
    ),
}

REQUIRED_ALIASES: dict[str, Mapping[str, str]] = {
    "daily_market": {
        "开盘价": "开盘价(元)", "最高价": "最高价(元)", "最低价": "最低价(元)",
        "收盘价": "收盘价(元)", "前收盘价(元)": "前收盘价", "成交量": "成交量(股)",
        "成交额": "成交金额(元)", "成交额(元)": "成交金额(元)",
        "成交金额": "成交金额(元)", "换手率": "换手率(%)",
    },
    "daily_index": {
        "开盘价(元)": "开盘价", "最高价(元)": "最高价", "最低价(元)": "最低价",
        "收盘价(元)": "收盘价", "前收盘价(元)": "前收盘价",
        "成交金额(元)": "成交额", "成交额(元)": "成交额", "换手率(%)": "换手率",
    },
    "daily_fund": {
        "开盘价": "开盘价(元)", "最高价": "最高价(元)", "最低价": "最低价(元)",
        "收盘价": "收盘价(元)", "前收盘价(元)": "前收盘价",
        "成交金额(元)": "成交额(元)", "成交额": "成交额(元)",
        "成交量": "成交量(份)", "换手率": "换手率(%)",
    },
    "daily_bond": {
        "开盘价": "开盘价(元)", "最高价": "最高价(元)", "最低价": "最低价(元)",
        "收盘价": "收盘价(元)", "前收盘价(元)": "前收盘价",
        "成交金额(元)": "成交额(元)", "成交额": "成交额(元)",
        "成交量": "成交量(手)", "换手率": "换手率(%)",
    },
}


@dataclass(frozen=True)
class SchemaContract:
    version: int
    tables: tuple[str, ...]
    views: tuple[str, ...] = ()
    functions: tuple[str, ...] = ()
    columns: Mapping[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    not_null_columns: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    identity_columns: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    defaults: Mapping[tuple[str, str], str] = field(default_factory=dict)
    indexes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    index_definitions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    constraints: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    constraint_types: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    constraint_definitions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    view_columns: tuple[tuple[str, str], ...] = ()
    view_definition_hashes: Mapping[str, str] = field(default_factory=dict)
    seeds: Mapping[str, Any] = field(default_factory=dict)
    partition_policy: Mapping[str, Any] = field(default_factory=dict)
    verify_dataset_coverage: bool = False
    verify_comments: bool = False
    verify_view_definitions: bool = False


_TABLES_BY_VERSION = {
    1: ("schema_migration",),
    2: ("schema_migration", "entity_dim", "entity_name_history", "metric_dim", "metric_alias"),
    3: ("schema_migration", "entity_dim", "entity_name_history", "metric_dim", "metric_alias", "market_daily_fact"),
    4: ("schema_migration", "entity_dim", "entity_name_history", "metric_dim", "metric_alias", "market_daily_fact", "observation_fact"),
    5: ALL_BASE_TABLES,
}

SCHEMA_CONTRACTS: dict[int, SchemaContract] = {}
for _version in range(1, LATEST_SCHEMA_VERSION + 1):
    _tables = _TABLES_BY_VERSION.get(_version, ALL_BASE_TABLES)
    _views = ALL_COMPATIBILITY_VIEWS if _version >= COMPATIBILITY_VIEW_VERSION else ()
    SCHEMA_CONTRACTS[_version] = SchemaContract(
        version=_version,
        tables=_tables,
        views=_views,
        functions=(
            LIFECYCLE_FUNCTIONS if _version >= LIFECYCLE_AUDIT_VERSION
            else REQUIRED_FUNCTIONS if _version >= COMPATIBILITY_VIEW_VERSION
            else ()
        ),
        columns={name: TABLE_COLUMNS[name] for name in _tables},
        not_null_columns={name: NOT_NULL_COLUMNS.get(name, ()) for name in _tables},
        identity_columns={name: IDENTITY_COLUMNS.get(name, ()) for name in _tables},
        defaults={
            key: value for key, value in DEFAULT_DEFINITION_FRAGMENTS.items()
            if key[0] in _tables
        },
        indexes={name: REQUIRED_INDEXES.get(name, ()) for name in _tables},
        index_definitions=dict(INDEX_DEFINITION_FRAGMENTS),
        constraints={name: REQUIRED_CONSTRAINTS.get(name, ()) for name in _tables},
        constraint_types={name: REQUIRED_CONSTRAINT_TYPES.get(name, ()) for name in _tables},
        constraint_definitions=dict(CONSTRAINT_DEFINITION_FRAGMENTS),
        view_columns=VIEW_COLUMNS if _views else (),
        view_definition_hashes={
            name: VIEW_DEFINITION_HASHES[name] for name in _views
        } if _version >= LIFECYCLE_AUDIT_VERSION else {},
        seeds={
            "core_metrics": CORE_METRIC_SEEDS,
            "metric_aliases": REQUIRED_ALIASES,
            "coverage_datasets": _views,
        } if "metric_dim" in _tables else {},
        partition_policy={
            "table": "observation_fact",
            "key": "available_at",
            "interval": "year",
            "required": ("current_year", "next_year", "data_coverage_years", "requested_years"),
            "inherits_parent_indexes": True,
        } if "observation_fact" in _tables else {},
        verify_dataset_coverage=_version >= FINALIZE_VERSION,
        verify_comments=_version >= FINALIZE_VERSION,
        verify_view_definitions=_version >= LIFECYCLE_AUDIT_VERSION,
    )


def get_schema_contract(version: int) -> SchemaContract:
    try:
        return SCHEMA_CONTRACTS[int(version)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"没有 schema version {version} 的 contract") from exc


# Compatibility exports used by existing callers and documentation.
BASE_TABLES = ALL_BASE_TABLES
COMPATIBILITY_VIEWS = ALL_COMPATIBILITY_VIEWS


__all__ = [
    "BASE_TABLES",
    "COMMENTED_TABLES",
    "COMPATIBILITY_VIEWS",
    "COMPATIBILITY_VIEW_VERSION",
    "CONSTRAINT_DEFINITION_FRAGMENTS",
    "CORE_METRIC_SEEDS",
    "DEFAULT_DEFINITION_FRAGMENTS",
    "FINALIZE_VERSION",
    "IDENTITY_COLUMNS",
    "INDEX_DEFINITION_FRAGMENTS",
    "LATEST_SCHEMA_VERSION",
    "LEGACY_MIGRATION_VERSION",
    "LIFECYCLE_FUNCTIONS",
    "LIFECYCLE_AUDIT_VERSION",
    "NOT_NULL_COLUMNS",
    "REQUIRED_ALIASES",
    "REQUIRED_CONSTRAINTS",
    "REQUIRED_CONSTRAINT_TYPES",
    "REQUIRED_FUNCTIONS",
    "REQUIRED_INDEXES",
    "SCHEMA_CONTRACTS",
    "SchemaContract",
    "TABLE_COLUMNS",
    "VIEW_COLUMNS",
    "VIEW_DEFINITION_FRAGMENTS",
    "VIEW_DEFINITION_HASHES",
    "get_schema_contract",
]
