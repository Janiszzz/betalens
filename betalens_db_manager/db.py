"""PostgreSQL access helpers for database management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2 import sql

from betalens.datafeed.config import get_database_config

from .constants import ALLOWED_TABLES, DEFAULT_LIMIT, DEFAULT_STATEMENT_TIMEOUT_MS
from .utils import clean_database_config


@dataclass(frozen=True)
class QueryRequest:
    table: str
    code: str | None = None
    metric: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    limit: int = DEFAULT_LIMIT


class DatabaseClient:
    """Small scoped PostgreSQL client for the local manager.

    The class opens short-lived connections for operations. It is deliberately
    separate from ``betalens.datafeed.Datafeed`` so management code does not
    widen the runtime research API.
    """

    def __init__(self, db_config: dict[str, Any] | None = None, statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS):
        self.db_config = clean_database_config(db_config or get_database_config())
        self.statement_timeout_ms = int(statement_timeout_ms)

    def connect(self):
        cfg = self.db_config
        conn = psycopg2.connect(
            dbname=cfg["dbname"],
            user=cfg["user"],
            password=cfg["password"],
            host=cfg["host"],
            port=cfg["port"],
            connect_timeout=5,
        )
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = %s", (self.statement_timeout_ms,))
        return conn

    def test_connection(self) -> dict[str, Any]:
        with self.connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT version(), current_database(), current_user")
                row = cur.fetchone()
                return dict(row) if row else {}

    def validate_table(self, table: str) -> str:
        if table not in ALLOWED_TABLES:
            raise ValueError(f"非法表名: {table}")
        return table

    def table_overview(self) -> list[dict[str, Any]]:
        query = """
        SELECT c.relname AS table_name,
               c.reltuples::bigint AS estimated_rows,
               pg_total_relation_size(c.oid) AS total_bytes,
               pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
               obj_description(c.oid, 'pg_class') AS table_comment
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relkind = 'r'
          AND c.relname = ANY(%s)
        ORDER BY c.relname
        """
        with self.connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (list(ALLOWED_TABLES),))
                rows = [dict(r) for r in cur.fetchall()]
                for row in rows:
                    table = row["table_name"]
                    row["date_range"] = self._date_range(cur, table)
                    row["warnings"] = self._light_warnings(cur, table)
                return rows

    def table_schema(self, table: str) -> dict[str, Any]:
        table = self.validate_table(table)
        with self.connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT column_name, data_type, is_nullable,
                           col_description((%s)::regclass, ordinal_position) AS comment
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (f"public.{table}", table),
                )
                columns = [dict(r) for r in cur.fetchall()]
                cur.execute(
                    """
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname = 'public' AND tablename = %s
                    ORDER BY indexname
                    """,
                    (table,),
                )
                indexes = [dict(r) for r in cur.fetchall()]
                cur.execute(
                    """
                    SELECT conname, contype, pg_get_constraintdef(c.oid) AS definition
                    FROM pg_constraint c
                    WHERE c.conrelid = %s::regclass
                    ORDER BY conname
                    """,
                    (f"public.{table}",),
                )
                constraints = [dict(r) for r in cur.fetchall()]
        return {"columns": columns, "indexes": indexes, "constraints": constraints}

    def query_table(self, request: QueryRequest) -> pd.DataFrame:
        table = self.validate_table(request.table)
        limit = max(1, min(int(request.limit or DEFAULT_LIMIT), 5000))

        if table == "daily_market" and not any([request.code, request.metric, request.start_date, request.end_date]):
            raise ValueError("daily_market 体量很大，至少提供 code、metric 或日期条件之一")

        conditions: list[sql.SQL] = []
        params: list[Any] = []
        if request.code:
            conditions.append(sql.SQL("code = %s"))
            params.append(request.code.strip())
        if request.metric:
            conditions.append(sql.SQL("metric = %s"))
            params.append(request.metric.strip())
        if request.start_date:
            conditions.append(sql.SQL("datetime >= %s::timestamp"))
            params.append(request.start_date)
        if request.end_date:
            conditions.append(sql.SQL("datetime <= %s::timestamp"))
            params.append(request.end_date)

        query = sql.SQL("SELECT datetime, code, name, metric, value, remark FROM {}").format(sql.Identifier(table))
        if conditions:
            query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)
        query += sql.SQL(" ORDER BY datetime DESC LIMIT %s")
        params.append(limit)

        with self.connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                return pd.DataFrame(cur.fetchall())

    def distinct_values(self, table: str, column: str, limit: int = 100) -> list[Any]:
        table = self.validate_table(table)
        if column not in {"code", "metric"}:
            raise ValueError(f"不支持的列: {column}")
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT DISTINCT {} FROM {} ORDER BY {} LIMIT %s").format(
                        sql.Identifier(column),
                        sql.Identifier(table),
                        sql.Identifier(column),
                    ),
                    (int(limit),),
                )
                return [r[0] for r in cur.fetchall()]

    def _date_range(self, cur, table: str) -> dict[str, Any]:
        try:
            cur.execute(sql.SQL("SELECT MIN(datetime) AS min_dt, MAX(datetime) AS max_dt FROM {}").format(sql.Identifier(table)))
            row = cur.fetchone()
            return dict(row) if row else {"min_dt": None, "max_dt": None}
        except Exception as exc:
            return {"error": str(exc)}

    def _light_warnings(self, cur, table: str) -> list[str]:
        warnings: list[str] = []
        try:
            cur.execute(
                sql.SQL("SELECT 1 FROM {} WHERE metric LIKE 'Unnamed:%' LIMIT 1").format(sql.Identifier(table))
            )
            if cur.fetchone():
                warnings.append("存在 Unnamed:* 指标")
        except Exception:
            pass
        try:
            cur.execute(
                sql.SQL("SELECT 1 FROM {} WHERE value = 'NaN'::numeric LIMIT 1").format(sql.Identifier(table))
            )
            if cur.fetchone():
                warnings.append("存在 numeric NaN")
        except Exception:
            pass
        return warnings
