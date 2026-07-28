"""PostgreSQL access helpers for database management."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2 import sql

from betalens.datafeed.config import get_database_config

from .constants import ALLOWED_TABLES, DEFAULT_LIMIT, DEFAULT_STATEMENT_TIMEOUT_MS
from .registry import DATASETS, DatasetSpec, get_dataset
from .utils import clean_database_config


_LONG_COLUMNS = ("datetime", "code", "name", "metric", "value", "remark")
_CORE_VALUE_SQL = """
CASE md.storage_column
    WHEN 'open' THEN f.open
    WHEN 'high' THEN f.high
    WHEN 'low' THEN f.low
    WHEN 'close' THEN f.close
    WHEN 'prev_close' THEN f.prev_close
    WHEN 'volume' THEN f.volume
    WHEN 'amount' THEN f.amount
    WHEN 'turnover_rate' THEN f.turnover_rate
END
"""


@dataclass(frozen=True)
class QueryRequest:
    table: str
    code: str | None = None
    metric: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    limit: int = DEFAULT_LIMIT
    page_token: str | None = None


class DatabaseClient:
    """Short-lived connections for inspecting logical Betalens datasets."""

    def __init__(
        self,
        db_config: dict[str, Any] | None = None,
        statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
    ):
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

    def validate_table(self, table: str, *, writable: bool = False) -> str:
        get_dataset(table, writable=writable)
        return table

    @staticmethod
    def make_page_token(row: dict[str, Any] | pd.Series) -> str:
        """Build an opaque keyset token from the last row of a result page."""

        dt = pd.Timestamp(row["datetime"])
        payload = {
            "datetime": dt.isoformat(),
            "code": str(row["code"]),
            "metric": str(row["metric"]),
        }
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def parse_page_token(token: str) -> tuple[str, str, str]:
        try:
            raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
            payload = json.loads(raw.decode("utf-8"))
            return str(payload["datetime"]), str(payload["code"]), str(payload["metric"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("无效的 page_token") from exc

    def _has_new_schema(self, cur) -> bool:
        cur.execute("SELECT to_regclass('betalens.entity_dim') IS NOT NULL")
        row = cur.fetchone()
        if isinstance(row, dict):
            return bool(next(iter(row.values())))
        return bool(row and row[0])

    def table_overview(self, *, include_checks: bool = False) -> list[dict[str, Any]]:
        with self.connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if not self._has_new_schema(cur):
                    return self._legacy_table_overview(cur)
                cur.execute(
                    """
                    SELECT d.logical_dataset AS table_name,
                           c.row_count AS estimated_rows,
                           c.min_available_at AS min_dt,
                           c.max_available_at AS max_dt,
                           c.updated_at AS coverage_updated_at
                    FROM unnest(%s::text[]) WITH ORDINALITY AS d(logical_dataset, ord)
                    LEFT JOIN betalens.dataset_coverage c
                      ON c.logical_dataset = d.logical_dataset
                    ORDER BY d.ord
                    """,
                    (list(ALLOWED_TABLES),),
                )
                coverage = [dict(row) for row in cur.fetchall()]
                checks = self._coverage_checks(cur) if include_checks else {}
                size_by_table = self._physical_sizes(cur)
                rows: list[dict[str, Any]] = []
                for row in coverage:
                    spec = DATASETS[row["table_name"]]
                    min_dt = row.pop("min_dt")
                    max_dt = row.pop("max_dt")
                    warnings = [] if row.get("estimated_rows") is not None else ["尚无数据覆盖记录"]
                    checked = checks.get(row["table_name"], {})
                    checked_rows = checked.get("checked_rows")
                    coverage_stale = (
                        checked_rows is not None
                        and row.get("estimated_rows") is not None
                        and int(checked_rows) != int(row["estimated_rows"])
                    )
                    if coverage_stale:
                        warnings.append(
                            f"coverage 记录已陈旧：存储 {row['estimated_rows']}，实查 {checked_rows}"
                        )
                    if checked.get("error"):
                        warnings.append(f"只读覆盖核验失败: {checked['error']}")
                    if row["table_name"] == "trade_status" and max_dt is not None:
                        if pd.Timestamp(max_dt) <= pd.Timestamp("2015-06-01 23:59:59"):
                            warnings.append(
                                "trade_status 覆盖仅截至 2015-06-01，之后交易状态不可视为完整"
                            )
                    total_bytes = sum(size_by_table.get(name, 0) for name in set(spec.physical_tables))
                    row.update(
                        {
                            "total_bytes": total_bytes,
                            "total_size": self._pretty_bytes(total_bytes),
                            "table_comment": f"逻辑数据集；存储类型: {spec.storage}",
                            "date_range": {"min_dt": min_dt, "max_dt": max_dt},
                            "warnings": warnings,
                            "physical_tables": list(spec.physical_tables),
                            "checked_rows": checked_rows,
                            "checked_min_dt": checked.get("checked_min_dt"),
                            "checked_max_dt": checked.get("checked_max_dt"),
                            "coverage_stale": coverage_stale,
                        }
                    )
                    rows.append(row)
                return rows

    def coverage_checks(self) -> list[dict[str, Any]]:
        """Compute read-only logical row/date coverage for all datasets."""

        with self.connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if not self._has_new_schema(cur):
                    return []
                return list(self._coverage_checks(cur).values())

    def _coverage_checks(self, cur) -> dict[str, dict[str, Any]]:
        checks: dict[str, dict[str, Any]] = {}
        for position, (name, spec) in enumerate(DATASETS.items(), start=1):
            source_sql, params = self._logical_source(spec)
            savepoint = f"coverage_{position}"
            cur.execute(sql.SQL("SAVEPOINT {}").format(sql.Identifier(savepoint)))
            try:
                cur.execute(
                    "WITH logical_rows AS (" + source_sql + ") "
                    "SELECT count(*)::bigint AS checked_rows, "
                    "min(datetime) AS checked_min_dt, max(datetime) AS checked_max_dt "
                    "FROM logical_rows",
                    params,
                )
                row = dict(cur.fetchone())
                row["table_name"] = name
                checks[name] = row
                cur.execute(sql.SQL("RELEASE SAVEPOINT {}").format(sql.Identifier(savepoint)))
            except Exception as exc:
                cur.execute(sql.SQL("ROLLBACK TO SAVEPOINT {}").format(sql.Identifier(savepoint)))
                cur.execute(sql.SQL("RELEASE SAVEPOINT {}").format(sql.Identifier(savepoint)))
                checks[name] = {"table_name": name, "error": str(exc)}
        return checks

    def _physical_sizes(self, cur) -> dict[str, int]:
        names = sorted({table for spec in DATASETS.values() for table in spec.physical_tables})
        cur.execute(
            """
            WITH RECURSIVE roots AS (
                SELECT c.oid, c.relname
                FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE n.nspname='betalens' AND c.relname=ANY(%s)
            ), relation_tree(root_oid, oid) AS (
                SELECT oid, oid FROM roots
                UNION ALL
                SELECT tree.root_oid, child.inhrelid
                FROM relation_tree tree
                JOIN pg_inherits child ON child.inhparent=tree.oid
            )
            SELECT roots.relname AS table_name,
                   sum(pg_total_relation_size(relation_tree.oid))::bigint AS total_bytes
            FROM roots JOIN relation_tree ON relation_tree.root_oid=roots.oid
            GROUP BY roots.oid, roots.relname
            """,
            (names,),
        )
        return {row["table_name"]: int(row["total_bytes"] or 0) for row in cur.fetchall()}

    @staticmethod
    def _pretty_bytes(value: int) -> str:
        size = float(value)
        for unit in ("bytes", "kB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.0f} {unit}" if unit == "bytes" else f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def _legacy_table_overview(self, cur) -> list[dict[str, Any]]:
        cur.execute(
            """
            SELECT c.relname AS table_name,
                   c.reltuples::bigint AS estimated_rows,
                   pg_total_relation_size(c.oid) AS total_bytes,
                   pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
                   obj_description(c.oid, 'pg_class') AS table_comment
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relkind IN ('r', 'p', 'v')
              AND c.relname = ANY(%s)
            ORDER BY c.relname
            """,
            (list(ALLOWED_TABLES),),
        )
        rows = [dict(row) for row in cur.fetchall()]
        for row in rows:
            table = row["table_name"]
            row["date_range"] = self._legacy_date_range(cur, table)
            row["warnings"] = self._legacy_warnings(cur, table)
        return rows

    def table_schema(self, table: str) -> dict[str, Any]:
        table = self.validate_table(table)
        spec = DATASETS[table]
        with self.connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if not self._has_new_schema(cur):
                    return self._legacy_table_schema(cur, table)
                cur.execute(
                    """
                    SELECT table_name, column_name, data_type, is_nullable,
                           col_description((table_schema || '.' || table_name)::regclass,
                                           ordinal_position) AS comment
                    FROM information_schema.columns
                    WHERE table_schema = 'betalens' AND table_name = ANY(%s)
                    ORDER BY table_name, ordinal_position
                    """,
                    (list(spec.physical_tables),),
                )
                physical_columns = [dict(row) for row in cur.fetchall()]
                cur.execute(
                    """
                    SELECT tablename AS table_name, indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname = 'betalens' AND tablename = ANY(%s)
                    ORDER BY tablename, indexname
                    """,
                    (list(spec.physical_tables),),
                )
                indexes = [dict(row) for row in cur.fetchall()]
                cur.execute(
                    """
                    SELECT rel.relname AS table_name, con.conname, con.contype,
                           pg_get_constraintdef(con.oid) AS definition
                    FROM pg_constraint con
                    JOIN pg_class rel ON rel.oid = con.conrelid
                    JOIN pg_namespace n ON n.oid = rel.relnamespace
                    WHERE n.nspname = 'betalens' AND rel.relname = ANY(%s)
                    ORDER BY rel.relname, con.conname
                    """,
                    (list(spec.physical_tables),),
                )
                constraints = [dict(row) for row in cur.fetchall()]
        logical_columns = [
            {"column_name": name, "data_type": data_type, "is_nullable": "YES", "comment": "兼容长表字段"}
            for name, data_type in zip(
                _LONG_COLUMNS,
                ("timestamp without time zone", "character varying", "character varying", "character varying", "double precision", "jsonb"),
            )
        ]
        return {
            "logical_table": table,
            "storage": spec.storage,
            "columns": logical_columns,
            "physical_tables": list(spec.physical_tables),
            "physical_columns": physical_columns,
            "indexes": indexes,
            "constraints": constraints,
        }

    def _legacy_table_schema(self, cur, table: str) -> dict[str, Any]:
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
        columns = [dict(row) for row in cur.fetchall()]
        cur.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname='public' AND tablename=%s ORDER BY indexname",
            (table,),
        )
        indexes = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT conname, contype, pg_get_constraintdef(c.oid) AS definition
            FROM pg_constraint c WHERE c.conrelid = %s::regclass ORDER BY conname
            """,
            (f"public.{table}",),
        )
        return {"columns": columns, "indexes": indexes, "constraints": [dict(row) for row in cur.fetchall()]}

    def query_table(self, request: QueryRequest) -> pd.DataFrame:
        table = self.validate_table(request.table)
        limit = max(1, min(int(request.limit or DEFAULT_LIMIT), 5000))
        if table == "daily_market" and not any(
            [request.code, request.metric, request.start_date, request.end_date, request.page_token]
        ):
            raise ValueError("daily_market 体量很大，至少提供 code、metric、日期条件或 page_token 之一")

        with self.connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if not self._has_new_schema(cur):
                    return self._query_legacy(cur, request, limit)
                metric = self._resolve_metric(cur, table, request.metric) if request.metric else None
                source_sql, source_params = self._logical_source(DATASETS[table])
                conditions: list[str] = []
                params: list[Any] = list(source_params)
                if request.code:
                    conditions.append("code = %s")
                    params.append(request.code.strip())
                if metric:
                    conditions.append("metric = %s")
                    params.append(metric)
                if request.start_date:
                    conditions.append("datetime >= %s::timestamp")
                    params.append(request.start_date)
                if request.end_date:
                    if self._is_date_only(request.end_date):
                        conditions.append("datetime < %s::date + interval '1 day'")
                    else:
                        conditions.append("datetime <= %s::timestamp")
                    params.append(request.end_date)
                if request.page_token:
                    token_dt, token_code, token_metric = self.parse_page_token(request.page_token)
                    conditions.append(
                        "(datetime < %s::timestamp OR "
                        "(datetime = %s::timestamp AND code > %s) OR "
                        "(datetime = %s::timestamp AND code = %s AND metric > %s))"
                    )
                    params.extend([token_dt, token_dt, token_code, token_dt, token_code, token_metric])
                where_sql = " WHERE " + " AND ".join(conditions) if conditions else ""
                query = (
                    "WITH logical_rows AS (" + source_sql + ") "
                    "SELECT datetime, code, name, metric, value, remark FROM logical_rows"
                    + where_sql
                    + " ORDER BY datetime DESC, code, metric LIMIT %s"
                )
                params.append(limit)
                cur.execute(query, params)
                return pd.DataFrame(cur.fetchall(), columns=_LONG_COLUMNS)

    def execute_readonly_sql(self, query: str, *, limit: int = 5000) -> pd.DataFrame:
        """Execute one bounded, read-only SQL query for the desktop explorer.

        This intentionally is not a general ``run_query`` compatibility API.
        It is a GUI-facing read path guarded both syntactically and by a
        PostgreSQL read-only transaction.
        """

        statement = str(query or "").strip()
        if not statement:
            raise ValueError("请输入 SQL 查询")
        if statement.endswith(";"):
            statement = statement[:-1].rstrip()
        if ";" in statement:
            raise ValueError("一次只能执行一条 SQL 查询")
        first = re.match(r"^\s*([a-z]+)", statement, flags=re.IGNORECASE)
        if first is None or first.group(1).lower() not in {"select", "with", "explain"}:
            raise ValueError("仅允许 SELECT、WITH 或 EXPLAIN 只读查询")
        forbidden = re.compile(
            r"\b(insert|update|delete|merge|copy|create|alter|drop|truncate|grant|revoke|call|do)\b",
            flags=re.IGNORECASE,
        )
        if forbidden.search(statement):
            raise ValueError("SQL 查询不能包含写入或 DDL 关键字")
        row_limit = max(1, min(int(limit), 5000))
        with self.connect() as conn:
            conn.rollback()
            conn.set_session(readonly=True, autocommit=False)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(statement)
                if cur.description is None:
                    raise ValueError("SQL 没有返回结果集")
                rows = cur.fetchmany(row_limit)
                columns = [item.name for item in cur.description]
        return pd.DataFrame(rows, columns=columns)

    def diagnose_data(self, table: str, *, sample_limit: int = 10) -> list[dict[str, Any]]:
        """Return read-only, actionable integrity checks for one logical dataset."""

        table = self.validate_table(table)
        limit = max(1, min(int(sample_limit), 100))
        with self.connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if not self._has_new_schema(cur):
                    return self._diagnose_legacy_table(cur, table, limit)
                checks: list[dict[str, Any]] = []
                checks.extend(
                    self._diagnose_query(
                        cur,
                        "实体代码为空或名称为空",
                        """
                        SELECT code, entity_type, current_name
                        FROM betalens.entity_dim
                        WHERE btrim(code) = '' OR btrim(current_name) = ''
                        """,
                        [],
                        limit,
                    )
                )
                spec = DATASETS[table]
                if spec.storage in {"market", "observation"}:
                    checks.extend(self._diagnose_market_observation(cur, spec, limit))
                elif spec.storage == "industry":
                    checks.extend(
                        self._diagnose_query(
                            cur,
                            "行业归属 remark 不是 JSON object",
                            """
                            SELECT e.code, membership.valid_from AS datetime,
                                   scheme.scheme_name AS metric, membership.remark
                            FROM betalens.industry_membership membership
                            JOIN betalens.entity_dim e ON e.entity_id = membership.entity_id
                            JOIN betalens.industry_dim industry ON industry.industry_id = membership.industry_id
                            JOIN betalens.industry_scheme_dim scheme ON scheme.scheme_id = industry.scheme_id
                            WHERE membership.remark IS NOT NULL
                              AND jsonb_typeof(membership.remark) <> 'object'
                            """,
                            [],
                            limit,
                        )
                    )
                elif spec.storage == "index_universe":
                    checks.extend(
                        self._diagnose_query(
                            cur,
                            "指数快照 remark 不是 JSON object",
                            """
                            SELECT entity.code, snapshot.effective_at AS datetime,
                                   'universe'::text AS metric, snapshot.remark
                            FROM betalens.index_snapshot snapshot
                            JOIN betalens.entity_dim entity ON entity.entity_id = snapshot.index_entity_id
                            WHERE snapshot.remark IS NOT NULL
                              AND jsonb_typeof(snapshot.remark) <> 'object'
                            """,
                            [],
                            limit,
                        )
                    )
                else:
                    checks.extend(
                        self._diagnose_query(
                            cur,
                            "交易状态 remark 不是 JSON object",
                            """
                            SELECT entity.code, event.event_date AS datetime,
                                   '交易状态'::text AS metric, event.remark
                            FROM betalens.trade_status_event event
                            JOIN betalens.entity_dim entity ON entity.entity_id = event.entity_id
                            WHERE event.remark IS NOT NULL
                              AND jsonb_typeof(event.remark) <> 'object'
                            """,
                            [],
                            limit,
                        )
                    )
        coverage = next(
            (row for row in self.table_overview(include_checks=True) if row["table_name"] == table),
            None,
        )
        if coverage and coverage.get("coverage_stale"):
            checks.append(
                {
                    "issue": "覆盖范围记录与实查不一致",
                    "count": 1,
                    "sample": [
                        {
                            "stored_rows": coverage.get("estimated_rows"),
                            "checked_rows": coverage.get("checked_rows"),
                            "stored_max": coverage.get("date_range", {}).get("max_dt"),
                            "checked_max": coverage.get("checked_max_dt"),
                        }
                    ],
                }
            )
        if not checks:
            checks.append({"issue": "未发现常见脏数据", "count": 0, "sample": []})
        return checks

    def _diagnose_market_observation(
        self,
        cur,
        spec: DatasetSpec,
        limit: int,
    ) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        if spec.storage == "market":
            checks.extend(
                self._diagnose_query(
                    cur,
                    "行情数值包含 NaN 或无穷大",
                    """
                    SELECT entity.code, fact.trade_date AS datetime, value.column_name AS metric,
                           value.metric_value::text AS value
                    FROM betalens.market_daily_fact fact
                    JOIN betalens.entity_dim entity ON entity.entity_id = fact.entity_id
                    CROSS JOIN LATERAL (VALUES
                        ('open', fact.open), ('high', fact.high), ('low', fact.low),
                        ('close', fact.close), ('prev_close', fact.prev_close),
                        ('volume', fact.volume), ('amount', fact.amount),
                        ('turnover_rate', fact.turnover_rate)
                    ) AS value(column_name, metric_value)
                    WHERE entity.entity_type = %s
                      AND value.metric_value::text IN ('NaN', 'Infinity', '-Infinity')
                    """,
                    [spec.entity_type],
                    limit,
                )
            )
            checks.extend(
                self._diagnose_query(
                    cur,
                    "行情 remark 不是 JSON object",
                    """
                    SELECT entity.code, fact.trade_date AS datetime,
                           'market_daily_fact'::text AS metric, fact.remark
                    FROM betalens.market_daily_fact fact
                    JOIN betalens.entity_dim entity ON entity.entity_id = fact.entity_id
                    WHERE entity.entity_type = %s
                      AND fact.remark IS NOT NULL
                      AND jsonb_typeof(fact.remark) <> 'object'
                    """,
                    [spec.entity_type],
                    limit,
                )
            )
        checks.extend(
            self._diagnose_query(
                cur,
                "观测值包含 NaN 或无穷大",
                """
                SELECT entity.code, observation.available_at AS datetime,
                       metric.metric_name AS metric, observation.value::text AS value
                FROM betalens.observation_fact observation
                JOIN betalens.entity_dim entity ON entity.entity_id = observation.entity_id
                JOIN betalens.metric_dim metric ON metric.metric_id = observation.metric_id
                WHERE metric.logical_dataset = %s AND entity.entity_type = %s
                  AND observation.value::text IN ('NaN', 'Infinity', '-Infinity')
                """,
                [spec.name, spec.entity_type],
                limit,
            )
        )
        checks.extend(
            self._diagnose_query(
                cur,
                "观测值 remark 不是 JSON object",
                """
                SELECT entity.code, observation.available_at AS datetime,
                       metric.metric_name AS metric, observation.remark
                FROM betalens.observation_fact observation
                JOIN betalens.entity_dim entity ON entity.entity_id = observation.entity_id
                JOIN betalens.metric_dim metric ON metric.metric_id = observation.metric_id
                WHERE metric.logical_dataset = %s AND entity.entity_type = %s
                  AND observation.remark IS NOT NULL
                  AND jsonb_typeof(observation.remark) <> 'object'
                """,
                [spec.name, spec.entity_type],
                limit,
            )
        )
        return checks

    @staticmethod
    def _diagnose_query(
        cur,
        issue: str,
        query: str,
        params: list[Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        cur.execute("SELECT count(*) AS count FROM (" + query + ") AS bad", params)
        count = int(cur.fetchone()["count"])
        if count == 0:
            return []
        cur.execute(query + " LIMIT %s", [*params, limit])
        return [{"issue": issue, "count": count, "sample": [dict(row) for row in cur.fetchall()]}]

    def _diagnose_legacy_table(self, cur, table: str, limit: int) -> list[dict[str, Any]]:
        query = sql.SQL(
            """
            SELECT code, datetime, metric, value, remark
            FROM public.{}
            WHERE code IS NULL OR btrim(code) = ''
               OR metric IS NULL OR btrim(metric) = ''
               OR value::text IN ('NaN', 'Infinity', '-Infinity')
               OR (remark IS NOT NULL AND jsonb_typeof(remark) <> 'object')
            """
        ).format(sql.Identifier(table))
        cur.execute(sql.SQL("SELECT count(*) AS count FROM ({}) AS bad").format(query))
        count = int(cur.fetchone()["count"])
        if count == 0:
            return [{"issue": "未发现常见脏数据", "count": 0, "sample": []}]
        cur.execute(query + sql.SQL(" LIMIT %s"), (limit,))
        return [{"issue": "旧长表存在空键、非有限值或非法 remark", "count": count, "sample": [dict(row) for row in cur.fetchall()]}]

    @staticmethod
    def _is_date_only(value: str) -> bool:
        text = str(value).strip()
        return len(text) == 10 and text[4:5] == "-" and text[7:8] == "-"

    def _resolve_metric(self, cur, table: str, metric: str) -> str:
        cur.execute(
            """
            SELECT md.metric_name
            FROM betalens.metric_alias ma
            JOIN betalens.metric_dim md ON md.metric_id = ma.metric_id
            WHERE ma.logical_dataset = %s AND ma.alias = %s
            UNION ALL
            SELECT metric_name FROM betalens.metric_dim
            WHERE logical_dataset = %s AND metric_name = %s
            LIMIT 1
            """,
            (table, metric.strip(), table, metric.strip()),
        )
        row = cur.fetchone()
        return row["metric_name"] if row else metric.strip()

    def _logical_source(self, spec: DatasetSpec) -> tuple[str, list[Any]]:
        if spec.storage in {"market", "observation"}:
            observation_sql = """
                SELECT o.available_at AS datetime, e.code, e.current_name AS name,
                       md.metric_name AS metric, o.value, o.remark
                FROM betalens.observation_fact o
                JOIN betalens.entity_dim e ON e.entity_id = o.entity_id
                JOIN betalens.metric_dim md ON md.metric_id = o.metric_id
                WHERE md.logical_dataset = %s
                  AND e.entity_type = %s
            """
            if spec.storage == "observation":
                return observation_sql, [spec.name, spec.entity_type]
            core_sql = f"""
                SELECT (f.trade_date + md.availability_time)::timestamp AS datetime,
                       e.code, e.current_name AS name, md.metric_name AS metric,
                       {_CORE_VALUE_SQL}::double precision AS value,
                       NULLIF(f.remark -> md.metric_name, 'null'::jsonb) AS remark
                FROM betalens.market_daily_fact f
                JOIN betalens.entity_dim e ON e.entity_id = f.entity_id
                JOIN betalens.metric_dim md
                  ON md.logical_dataset = %s AND md.storage_kind = 'core'
                WHERE e.entity_type = %s AND ({_CORE_VALUE_SQL}) IS NOT NULL
            """
            return core_sql + " UNION ALL " + observation_sql, [spec.name, spec.entity_type, spec.name, spec.entity_type]
        if spec.storage == "industry":
            return """
                SELECT im.valid_from AS datetime, e.code, e.current_name AS name,
                       s.scheme_name AS metric,
                       NULLIF(regexp_replace(d.industry_code, '[^0-9]', '', 'g'), '')::double precision AS value,
                       jsonb_build_object(
                           'ind_name', d.industry_name, 'ind_code', d.industry_code,
                           'scheme', s.scheme_name) || COALESCE(im.remark, '{}'::jsonb) AS remark
                FROM betalens.industry_membership im
                JOIN betalens.entity_dim e ON e.entity_id = im.entity_id
                JOIN betalens.industry_dim d ON d.industry_id = im.industry_id
                JOIN betalens.industry_scheme_dim s ON s.scheme_id = d.scheme_id
            """, []
        if spec.storage == "index_universe":
            return """
                SELECT s.effective_at AS datetime, idx.code,
                       COALESCE(NULLIF(s.index_name_snapshot, ''), idx.current_name) AS name,
                       'universe'::varchar AS metric,
                       COUNT(c.constituent_entity_id)::double precision AS value,
                       jsonb_build_object(
                           'index_code', idx.code,
                           'index_name', COALESCE(NULLIF(s.index_name_snapshot, ''), idx.current_name),
                           'constituents', COALESCE(
                               jsonb_agg(member.code ORDER BY c.ordinal NULLS LAST, member.code)
                                   FILTER (WHERE member.code IS NOT NULL),
                               '[]'::jsonb)) || COALESCE(s.remark, '{}'::jsonb) AS remark
                FROM betalens.index_snapshot s
                JOIN betalens.entity_dim idx ON idx.entity_id = s.index_entity_id
                LEFT JOIN betalens.index_constituent c ON c.snapshot_id = s.snapshot_id
                LEFT JOIN betalens.entity_dim member ON member.entity_id = c.constituent_entity_id
                GROUP BY s.snapshot_id, s.effective_at, idx.code, idx.current_name,
                         s.index_name_snapshot, s.remark
            """, []
        if spec.storage == "trade_status":
            return """
                SELECT (e.first_trade_date + time '15:00:01')::timestamp AS datetime,
                       e.code, e.current_name AS name, '交易状态'::varchar AS metric,
                       1::double precision AS value,
                       jsonb_build_object('status', '交易', 'first_normal', true) AS remark
                FROM betalens.entity_dim e
                WHERE e.entity_type = 'stock' AND e.first_trade_date IS NOT NULL
                UNION ALL
                SELECT (t.event_date + time '15:00:01')::timestamp AS datetime,
                       e.code, e.current_name AS name, '交易状态'::varchar AS metric,
                       t.status::double precision AS value,
                       jsonb_build_object('status', t.status_text)
                           || COALESCE(t.remark, '{}'::jsonb) AS remark
                FROM betalens.trade_status_event t
                JOIN betalens.entity_dim e ON e.entity_id = t.entity_id
            """, []
        raise AssertionError(f"未实现的存储路由: {spec.storage}")

    def _query_legacy(self, cur, request: QueryRequest, limit: int) -> pd.DataFrame:
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
            if self._is_date_only(request.end_date):
                conditions.append(sql.SQL("datetime < %s::date + interval '1 day'"))
            else:
                conditions.append(sql.SQL("datetime <= %s::timestamp"))
            params.append(request.end_date)
        if request.page_token:
            token_dt, token_code, token_metric = self.parse_page_token(request.page_token)
            conditions.append(
                sql.SQL(
                    "(datetime < %s::timestamp OR (datetime = %s::timestamp AND code > %s) "
                    "OR (datetime = %s::timestamp AND code = %s AND metric > %s))"
                )
            )
            params.extend([token_dt, token_dt, token_code, token_dt, token_code, token_metric])
        query = sql.SQL("SELECT datetime, code, name, metric, value, remark FROM public.{}").format(
            sql.Identifier(request.table)
        )
        if conditions:
            query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)
        query += sql.SQL(" ORDER BY datetime DESC, code, metric LIMIT %s")
        params.append(limit)
        cur.execute(query, params)
        return pd.DataFrame(cur.fetchall(), columns=_LONG_COLUMNS)

    def distinct_values(self, table: str, column: str, limit: int = 100) -> list[Any]:
        table = self.validate_table(table)
        if column not in {"code", "metric"}:
            raise ValueError(f"不支持的列: {column}")
        limit = max(1, min(int(limit), 5000))
        spec = DATASETS[table]
        with self.connect() as conn:
            with conn.cursor() as cur:
                if not self._has_new_schema(cur):
                    cur.execute(
                        sql.SQL("SELECT DISTINCT {} FROM public.{} ORDER BY {} LIMIT %s").format(
                            sql.Identifier(column), sql.Identifier(table), sql.Identifier(column)
                        ),
                        (limit,),
                    )
                    return [row[0] for row in cur.fetchall()]
                query, params = self._distinct_query(spec, column)
                cur.execute(query + " LIMIT %s", [*params, limit])
                return [row[0] for row in cur.fetchall()]

    def _distinct_query(self, spec: DatasetSpec, column: str) -> tuple[str, list[Any]]:
        if column == "code":
            entity_type = "index" if spec.storage == "index_universe" else spec.entity_type
            return "SELECT code FROM betalens.entity_dim WHERE entity_type=%s ORDER BY code", [entity_type]
        if spec.storage in {"market", "observation"}:
            return (
                "SELECT metric_name FROM betalens.metric_dim WHERE logical_dataset=%s ORDER BY metric_name",
                [spec.name],
            )
        if spec.storage == "industry":
            return "SELECT scheme_name FROM betalens.industry_scheme_dim ORDER BY scheme_name", []
        if spec.storage == "index_universe":
            return "SELECT 'universe'::text", []
        return "SELECT '交易状态'::text", []

    def _legacy_date_range(self, cur, table: str) -> dict[str, Any]:
        try:
            cur.execute(
                sql.SQL("SELECT MIN(datetime) AS min_dt, MAX(datetime) AS max_dt FROM public.{}").format(
                    sql.Identifier(table)
                )
            )
            row = cur.fetchone()
            return dict(row) if row else {"min_dt": None, "max_dt": None}
        except Exception as exc:
            return {"error": str(exc)}

    def _legacy_warnings(self, cur, table: str) -> list[str]:
        warnings: list[str] = []
        try:
            cur.execute(
                sql.SQL("SELECT 1 FROM public.{} WHERE metric LIKE 'Unnamed:%' LIMIT 1").format(
                    sql.Identifier(table)
                )
            )
            if cur.fetchone():
                warnings.append("存在 Unnamed:* 指标")
        except Exception:
            pass
        try:
            cur.execute(
                sql.SQL("SELECT 1 FROM public.{} WHERE value = 'NaN'::numeric LIMIT 1").format(
                    sql.Identifier(table)
                )
            )
            if cur.fetchone():
                warnings.append("存在 numeric NaN")
        except Exception:
            pass
        return warnings
