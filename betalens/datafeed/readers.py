"""Two small SELECT-only transports. Validate physical table/field names and bind predicate values."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
import json
import re

import pandas as pd
import psycopg2.extras
from psycopg2.extensions import TRANSACTION_STATUS_IDLE
import requests

from .contracts import DatafeedError, SchemaError, timestamp
from .pool import get_read_pool


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z_][a-z0-9_]*", value):
        raise ValueError("非法数据库标识符")
    return value


def logical_type(value):
    value = value.lower().replace("nullable(", "").replace("lowcardinality(", "")
    if "timestamp" in value or "datetime" in value:
        return "timestamp"
    if value.startswith("date"):
        return "date"
    if "numeric" in value or "decimal" in value or "float" in value or "double" in value:
        return "numeric"
    if "int" in value or value == "boolean":
        return "integer"
    if "json" in value:
        return "json"
    if any(part in value for part in ("text", "char", "string")):
        return "text"
    return value


def require_columns(reader, table, expected):
    actual = reader.columns(table)
    if not actual:
        raise SchemaError(f"缺少订阅表: {table}")
    for name, kind in expected.items():
        if name not in actual:
            raise SchemaError(f"{table} 缺少字段 {name}")
        if kind and actual[name] != kind:
            raise SchemaError(f"{table}.{name} 类型应为 {kind}，实际为 {actual[name]}")


def select_sql(reader, table, columns, filters=(), order=(), limit=None):
    """Internal SELECT helper, deliberately limited to bound scalar predicates."""
    names = reader.columns(table)
    require_columns(reader, table, {c: None for c in columns or names})
    quote = '"' if reader.backend == "postgres" else '`'
    fields = ",".join(quote + identifier(c) + quote for c in (columns or names))
    relation = quote + reader.namespace + quote + "." + quote + identifier(table) + quote
    sql = f"SELECT {fields} FROM {relation}" + (" FINAL" if reader.backend == "clickhouse" else "")
    params, clauses = {}, []
    for column, operator, value in filters:
        if column not in names or operator not in {"=", "!=", "<", "<=", ">", ">=", "IN", "IS NOT NULL"}:
            raise ValueError("无效的内部查询条件")
        field = quote + identifier(column) + quote
        if operator == "IS NOT NULL":
            clauses.append(field + " IS NOT NULL")
            continue
        values = list(value) if operator == "IN" else [value]
        if not values:
            clauses.append("1=0")
            continue
        placeholders = []
        for item in values:
            key = "p" + str(len(params))
            if reader.backend == "postgres":
                placeholders.append("%(" + key + ")s")
                params[key] = item.to_pydatetime() if isinstance(item, pd.Timestamp) else item
                continue
            typ = "String"
            if isinstance(item, (pd.Timestamp, datetime)):
                typ = "DateTime64(6,'UTC')"
                item = timestamp(item, stored=True).strftime("%Y-%m-%d %H:%M:%S.%f")
            elif isinstance(item, date):
                typ, item = "Date32", item.isoformat()
            elif isinstance(item, int):
                typ = "Int64"
            placeholders.append("{" + key + ":" + typ + "}")
            params[key] = str(item)
        bound = "(" + ",".join(placeholders) + ")" if operator == "IN" else placeholders[0]
        clauses.append(field + " " + operator + " " + bound)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if order:
        for column, direction in order:
            if column not in names or direction not in {"ASC", "DESC"}:
                raise ValueError("无效排序")
        sql += " ORDER BY " + ",".join(quote + c + quote + " " + d for c, d in order)
    if limit is not None:
        if isinstance(limit, bool) or int(limit) != limit or limit < 0:
            raise ValueError("limit 必须是非负整数")
        sql += " LIMIT " + str(int(limit))
    return sql, params


class PostgresReader:
    backend = "postgres"

    def __init__(self, config=None, *, cursor=None, schema="betalens"):
        self.config = dict(config or {})
        self.namespace = identifier(self.config.get("schema", schema))
        self._columns = {}
        self._pool = None
        self._depth = 0
        self._owned = cursor is None
        self.conn = getattr(cursor, "connection", None)
        self.cursor = cursor
        if cursor is not None:
            return
        self._pool = get_read_pool(self.config,
            min_connections=self.config.get("pool_min_connections", 1),
            max_connections=self.config.get("pool_max_connections", 10),
            statement_timeout_ms=self.config.get("statement_timeout_ms", 120000))
        self.conn = self._pool.acquire()
        try:
            self.conn.set_session(readonly=True, autocommit=False, isolation_level="REPEATABLE READ")
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            self.cursor._reader = self
        except Exception:
            self._pool.release(self.conn)
            raise

    @contextmanager
    def snapshot(self):
        previous = None
        if not self._owned and not self._depth:
            if self.conn.get_transaction_status() == TRANSACTION_STATUS_IDLE:
                previous = (self.conn.readonly, self.conn.autocommit, self.conn.isolation_level)
                self.conn.set_session(readonly=True, autocommit=False, isolation_level="REPEATABLE READ")
            else:
                self.cursor.execute("SHOW transaction_read_only")
                readonly = next(iter(self.cursor.fetchone().values())) if isinstance(self.cursor, psycopg2.extras.RealDictCursor) else self.cursor.fetchone()[0]
                self.cursor.execute("SHOW transaction_isolation")
                isolation = next(iter(self.cursor.fetchone().values())) if isinstance(self.cursor, psycopg2.extras.RealDictCursor) else self.cursor.fetchone()[0]
                if readonly != "on" or isolation not in {"repeatable read", "serializable"}:
                    raise DatafeedError("外部游标已有事务，需使用只读 REPEATABLE READ 事务")
        self._depth += 1
        try:
            yield self
        finally:
            self._depth -= 1
            if (self._owned or previous is not None) and not self._depth and self.conn is not None:
                self.conn.rollback()
                if previous is not None:
                    self.conn.set_session(readonly=previous[0] if previous[0] is not None else "DEFAULT",
                                          autocommit=previous[1],
                                          isolation_level=previous[2] if previous[2] is not None else "DEFAULT")

    def execute(self, sql, params=None):
        if not sql.lstrip().upper().startswith("SELECT "):
            raise ValueError("订阅连接只执行 SELECT")
        self.cursor.execute(sql, params or None)
        rows = self.cursor.fetchall()
        names = [c[0] for c in self.cursor.description]
        return [dict(row) if isinstance(row, dict) else dict(zip(names, row)) for row in rows]

    def columns(self, table):
        identifier(table)
        if table not in self._columns:
            rows = self.execute("SELECT column_name, data_type FROM information_schema.columns "
                                "WHERE table_schema=%(schema)s AND table_name=%(table)s ORDER BY ordinal_position",
                                {"schema": self.namespace, "table": table})
            self._columns[table] = {r["column_name"]: logical_type(r["data_type"]) for r in rows}
        return self._columns[table]

    def rows(self, table, columns=None, filters=(), order=(), limit=None):
        return self.execute(*select_sql(self, table, columns, filters, order, limit))

    def close(self):
        if self._owned and self.conn is not None:
            try:
                self.cursor.close()
            finally:
                self._pool.release(self.conn)
                self.conn = None


class ClickHouseReader:
    backend = "clickhouse"

    def __init__(self, config):
        self.config = dict(config)
        self.namespace = identifier(config["database"])
        self._columns = {}
        self.session = requests.Session()
        self.session.auth = (config["user"], config.get("password", ""))

    @contextmanager
    def snapshot(self):
        # Reference queries capture the publication event cut BEFORE manifests.
        # CH has no cross-request transaction: never pretend otherwise.
        yield self

    def execute(self, sql, params=None):
        if not sql.lstrip().upper().startswith("SELECT "):
            raise ValueError("订阅连接只执行 SELECT")
        query_params = {"database": self.namespace, "readonly": "1",
                        "max_execution_time": str(max(1, self.config.get("statement_timeout_ms", 120000) // 1000)),
                        "output_format_json_quote_64bit_integers": "1"}
        query_params.update({"param_" + key: value for key, value in (params or {}).items()})
        try:
            response = self.session.post(self.config["url"], params=query_params,
                data=(sql + " FORMAT JSONEachRow").encode(),
                timeout=(self.config.get("connect_timeout", 5), self.config.get("statement_timeout_ms", 120000) / 1000 + 5))
            response.raise_for_status()
            return [json.loads(line, parse_float=Decimal) for line in response.text.splitlines() if line.strip()]
        except (requests.RequestException, ValueError):
            # HTTP errors can contain credentials, source rows and query params.
            raise DatafeedError("ClickHouse 只读查询失败或超时") from None

    def columns(self, table):
        identifier(table)
        if table not in self._columns:
            rows = self.execute("SELECT name,type FROM system.columns WHERE database={db:String} "
                                "AND table={table:String} ORDER BY position",
                                {"db": self.namespace, "table": table})
            self._columns[table] = {r["name"]: logical_type(r["type"]) for r in rows}
        return self._columns[table]

    def rows(self, table, columns=None, filters=(), order=(), limit=None):
        rows = self.execute(*select_sql(self, table, columns, filters, order, limit))
        return typed_rows(self, table, rows)

    def close(self):
        self.session.close()


def get_reader(value):
    """Accept the public facade, a reader, or the legacy PG cursor argument."""
    if hasattr(value, "_reader"):
        return value._reader
    if isinstance(value, (PostgresReader, ClickHouseReader)) or hasattr(value, "rows"):
        return value
    return PostgresReader(cursor=value)


def typed_rows(reader, table, rows):
    types = reader.columns(table)
    for row in rows:
        for key, value in row.items():
            if value is None:
                continue
            kind = types[key]
            if kind == "timestamp":
                row[key] = timestamp(value, stored=True).to_pydatetime()
            elif kind == "date" and not isinstance(value, date):
                row[key] = date.fromisoformat(value)
            elif kind == "integer":
                row[key] = int(value)
            elif kind == "numeric":
                row[key] = Decimal(str(value))
    return rows
