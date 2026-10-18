"""Inspect installed tables and maintain observation partitions.

The historical schema migration executor and version tracking are retired.
"""
from __future__ import annotations

import os
import re
from datetime import date
from typing import Any, Iterable

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from betalens.datafeed.config import get_database_config
from .constants import ALLOWED_TABLES
from .contracts import BASE_TABLES, COMPATIBILITY_VIEWS, TABLE_COLUMNS, VIEW_COLUMNS
from .registry import get_dataset
from .utils import clean_database_config

PHYSICAL_SCHEMA = "betalens"
ADVISORY_LOCK_NAME = "betalens_observation_partitions"


class MigrationDisabledError(RuntimeError):
    """A retired schema lifecycle entry point was called."""


def migration_disabled(*args, **kwargs):
    raise MigrationDisabledError(
        "schema migration/bootstrap/verify 已停用；不会自动创建或升级表。"
        "请用 inspect_schema()/CLI inspect 检查已有结构，"
        "用 plan_manifest()/run_manifest() 或 CLI import 导入文件。"
    )


load_migrations = migration_disabled


def validate_database_name(dbname: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9_]+", dbname))


class SchemaManager:
    """Connection helpers, read-only inspection and partition maintenance."""

    bootstrap_local = staticmethod(migration_disabled)
    ensure_schema = staticmethod(migration_disabled)
    plan_migration = staticmethod(migration_disabled)
    verify_schema = staticmethod(migration_disabled)
    verify_schema_precommit = staticmethod(migration_disabled)

    def __init__(self, db_config: dict[str, Any] | None = None):
        config = dict(get_database_config())
        env_mapping = {
            "dbname": ("BETALENS_DB_NAME", "BETALENS_DBNAME", "PGDATABASE"),
            "user": ("BETALENS_DB_USER", "PGUSER"),
            "password": ("BETALENS_DB_PASSWORD", "PGPASSWORD"),
            "host": ("BETALENS_DB_HOST", "PGHOST"),
            "port": ("BETALENS_DB_PORT", "PGPORT"),
        }
        for key, candidates in env_mapping.items():
            for env_name in candidates:
                if os.environ.get(env_name) is not None:
                    config[key] = os.environ[env_name]
                    break
        if db_config:
            config.update({key: value for key, value in db_config.items() if value is not None})
        self.db_config = clean_database_config(config)

    def validate_table(self, table_name: str) -> str:
        if table_name not in ALLOWED_TABLES and table_name not in COMPATIBILITY_VIEWS:
            raise ValueError(f"非法逻辑表名: {table_name}")
        return table_name

    def connect(self, dbname: str | None = None):
        cfg = dict(self.db_config)
        if dbname is not None:
            cfg["dbname"] = dbname
        return psycopg2.connect(**cfg)

    def database_exists(self, dbname: str | None = None) -> bool:
        target = dbname or self.db_config["dbname"]
        conn = self.connect("postgres")
        try:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target,))
                return cur.fetchone() is not None
        finally:
            conn.close()

    def create_database(self, dbname: str | None = None) -> bool:
        target = dbname or self.db_config["dbname"]
        if not validate_database_name(target):
            raise ValueError(f"非法数据库名: {target}")
        conn = self.connect("postgres")
        try:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target,))
                if cur.fetchone() is not None:
                    return False
                try:
                    cur.execute(
                        sql.SQL("CREATE DATABASE {} ENCODING 'UTF8' TEMPLATE template0").format(
                            sql.Identifier(target)
                        )
                    )
                except psycopg2.errors.DuplicateDatabase:
                    # Another caller may have won the race after the
                    # existence check.  The target state is still satisfied.
                    return False
                return True
        finally:
            conn.close()

    @staticmethod
    def _normalize_partition_years(years: Iterable[int] | None = None) -> set[int]:
        current_year = date.today().year
        requested = {current_year, current_year + 1}
        if years is not None:
            requested.update(int(year) for year in years)
        invalid = sorted(year for year in requested if year < 1900 or year > 9998)
        if invalid:
            raise ValueError(f"不支持的 observation 分区年份: {invalid}")
        return requested

    def ensure_observation_partitions(
        self,
        years: Iterable[int] | None = None,
        *,
        connection=None,
    ) -> list[str]:
        """Create observation partitions for explicit years.

        The current and next calendar year are always included.  Importers call
        this method before loading historical years; they do not construct DDL.
        """

        requested = self._normalize_partition_years(years)

        owns_connection = connection is None
        conn = connection or self.connect()
        created: list[str] = []
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (ADVISORY_LOCK_NAME,))
                cur.execute("SELECT to_regclass('betalens.observation_fact')")
                if cur.fetchone()[0] is None:
                    raise RuntimeError("observation_fact 尚未创建；请先准备目标表结构或从备份恢复")
                for year in sorted(requested):
                    name = f"observation_fact_{year}"
                    qualified = f"{PHYSICAL_SCHEMA}.{name}"
                    cur.execute("SELECT to_regclass(%s)", (qualified,))
                    existed = cur.fetchone()[0] is not None
                    statement = sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.{}
                        PARTITION OF {}.observation_fact
                        FOR VALUES FROM ({}) TO ({})
                        """
                    ).format(
                        sql.Identifier(PHYSICAL_SCHEMA),
                        sql.Identifier(name),
                        sql.Identifier(PHYSICAL_SCHEMA),
                        sql.Literal(date(year, 1, 1)),
                        sql.Literal(date(year + 1, 1, 1)),
                    )
                    cur.execute(statement)
                    if not existed:
                        created.append(name)
            if owns_connection:
                conn.commit()
        except Exception:
            if owns_connection:
                conn.rollback()
            raise
        finally:
            if owns_connection:
                conn.close()
        return created

    def table_exists(self, cur, table_name: str) -> bool:
        table_name = self.validate_table(table_name)
        cur.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
        row = cur.fetchone()
        value = row.get("to_regclass") if isinstance(row, dict) else row[0]
        return value is not None

    def inspect_schema(self, tables: Iterable[str] | None = None) -> dict[str, Any]:
        """Read required relations/columns and execute empty view queries.

        This does not create objects or certify indexes, constraints, metric
        seeds or historical data. No migration files or records are consulted.
        """
        datasets = list(COMPATIBILITY_VIEWS if tables is None else tables)
        specs = [get_dataset(name) for name in datasets]
        required = set(BASE_TABLES) if tables is None else {"dataset_coverage"}
        for spec in specs:
            required.update(spec.physical_tables)
        # Non-calendar views read names through entity_name_at().
        if any(spec.storage != "trade_calendar" for spec in specs):
            required.add("entity_name_history")
        result = {
            "ok": False, "database": self.db_config.get("dbname"),
            "database_exists": False, "schema_mode": "manual",
            "migration_tracking": False, "tables": {}, "views": {},
            "checks": ["relations", "column_types", "view_queries"],
            "errors": [], "warnings": [],
        }
        conn = None
        try:
            conn = self.connect()
            conn.set_session(readonly=True, autocommit=True)
            result["database_exists"] = True
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = '15s'")
                for schema, relations, kinds, bucket in (
                    (PHYSICAL_SCHEMA, sorted(required), {"r", "p"}, "tables"),
                    ("public", datasets, {"v"}, "views"),
                ):
                    for name in relations:
                        cur.execute(
                            "SELECT c.relkind FROM pg_catalog.pg_class c "
                            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
                            "WHERE n.nspname=%s AND c.relname=%s", (schema, name),
                        )
                        row = cur.fetchone()
                        if row is None or row[0] not in kinds:
                            result["errors"].append(f"缺少预期表或视图: {schema}.{name}")
                            result[bucket][name] = False
                            continue
                        cur.execute(
                            "SELECT column_name, data_type FROM information_schema.columns "
                            "WHERE table_schema=%s AND table_name=%s", (schema, name),
                        )
                        actual = dict(cur.fetchall())
                        expected = TABLE_COLUMNS[name] if bucket == "tables" else VIEW_COLUMNS
                        invalid = [column for column, kind in expected if actual.get(column) != kind]
                        result[bucket][name] = not invalid
                        if invalid:
                            result["errors"].append(f"字段缺失或类型不匹配: {schema}.{name}: {invalid}")
                        if bucket == "views":
                            cur.execute(sql.SQL("SELECT * FROM {}.{} LIMIT 0").format(
                                sql.Identifier(schema), sql.Identifier(name)))
        except Exception as exc:
            result["errors"].append(str(exc))
        finally:
            if conn is not None:
                conn.close()
        result["ok"] = result["database_exists"] and not result["errors"]
        result["status"] = "completed" if result["ok"] else "failed"
        return result
