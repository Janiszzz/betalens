"""Two small SELECT-only transports. Validate physical table/field names and bind predicate values."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal, localcontext
from numbers import Real
import json
import re

import pandas as pd
import psycopg2.extras
import requests

from .contracts import DatafeedError, SchemaError, timestamp, calendar_date, integer
from .pool import get_read_pool


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z_][a-z0-9_]*", value):
        raise ValueError("非法数据库标识符")
    return value


def scalar_type(value):
    value = re.sub(r"\s+", " ", value.strip().lower())
    for _ in range(4):
        wrapper = re.fullmatch(r"(?:nullable|lowcardinality)\((.*)\)", value)
        if not wrapper:
            return value
        value = wrapper[1].strip()
    return value


def decimal_shape(value):
    found = re.fullmatch(r"(?:numeric|decimal)\(\s*(\d+)\s*,\s*(\d+)\s*\)", scalar_type(value))
    return (int(found[1]), int(found[2])) if found else None


def logical_type(value):
    value = scalar_type(value)
    if re.fullmatch(r"timestamp(?:\(\d+\))?(?: with(?:out)? time zone)?|timestamptz|datetime(?:64\(\d+(?:,\s*'[^']+')?\)|\('[^']+'\))?", value):
        return "timestamp"
    if value in {"date", "date32"}:
        return "date"
    if value in {"numeric", "real", "double precision", "float32", "float64"} or decimal_shape(value):
        return "numeric"
    if value in {"smallint", "integer", "bigint"} or re.fullmatch(r"u?int(?:8|16|32|64|128|256)", value):
        return "integer"
    if value in {"bool", "boolean"}:
        return "boolean"
    if value in {"json", "jsonb"}:
        return "json"
    if value in {"text", "string", "character", "character varying"} or re.fullmatch(r"(?:varchar|char|character varying|fixedstring)\(\d+\)", value):
        return "text"
    return "unsupported"


def filter_value(value, kind):
    if kind == "timestamp":
        return timestamp(value).to_pydatetime()
    if kind == "date":
        return calendar_date(value)
    if kind == "integer":
        return integer(value, "整数查询条件")
    if kind == "boolean" and isinstance(value, bool):
        return value
    if kind == "text" and isinstance(value, str):
        return value
    if kind == "numeric" and isinstance(value, (Real, Decimal)) and not isinstance(value, bool):
        number = Decimal(str(value))
        if number.is_finite():
            return number
    raise ValueError(f"查询条件不符合字段类型 {kind}")


def _check_decimal(value, shape):
    if not isinstance(value, Decimal) or shape is None:
        return
    precision, scale = shape
    with localcontext() as context:
        context.prec = max(precision + scale + 2, len(value.as_tuple().digits) + abs(value.as_tuple().exponent) + 2)
        if abs(value) >= Decimal(10) ** (precision - scale) or value != value.quantize(Decimal(1).scaleb(-scale)):
            raise ValueError("数值查询条件超出数据库 Decimal 精度，拒绝截断")


def require_columns(reader, table, expected):
    actual = reader.columns(table)
    if not actual:
        raise SchemaError(f"缺少订阅表: {table}")
    for name, kind in expected.items():
        if name not in actual:
            raise SchemaError(f"{table} 缺少字段 {name}")
        if actual[name] == "unsupported":
            raise SchemaError(f"{table}.{name} 不支持数组、复合或未知字段类型")
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
            item = filter_value(item, names[column])
            shape = getattr(reader, "_numeric_shapes", {}).get(table, {}).get(column)
            _check_decimal(item, shape)
            key = "p" + str(len(params))
            if reader.backend == "postgres":
                placeholders.append("%(" + key + ")s")
                params[key] = item.to_pydatetime() if isinstance(item, pd.Timestamp) else item
                continue
            typ = "String"
            if isinstance(item, (pd.Timestamp, datetime)):
                typ = "DateTime64(6,'UTC')"
                item = timestamp(item).strftime("%Y-%m-%d %H:%M:%S.%f")
            elif isinstance(item, date):
                typ, item = "Date32", item.isoformat()
            elif isinstance(item, bool):
                typ, item = "Bool", int(item)
            elif isinstance(item, int):
                typ = "Int64"
            elif isinstance(item, Decimal):
                typ = f"Decimal({shape[0]},{shape[1]})" if shape else "Float64"
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
        sql += " LIMIT " + str(integer(limit, "limit", minimum=0))
    return sql, params


class PostgresReader:
    backend = "postgres"

    def __init__(self, config):
        self.config = dict(config)
        self.namespace = identifier(config["schema"])
        self._columns = {}
        self._numeric_shapes = {}
        self._depth = 0
        self._pool = get_read_pool(config,
            min_connections=config["pool_min_connections"],
            max_connections=config["pool_max_connections"],
            statement_timeout_ms=config["statement_timeout_ms"])
        self.conn = self._pool.acquire()
        try:
            self.conn.set_session(readonly=True, autocommit=False, isolation_level="REPEATABLE READ")
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        except Exception:
            self._pool.release(self.conn)
            raise

    @contextmanager
    def snapshot(self):
        self._depth += 1
        try:
            yield self
        finally:
            self._depth -= 1
            if not self._depth and self.conn is not None:
                self.conn.rollback()

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
            rows = self.execute("SELECT column_name, data_type, numeric_precision, numeric_scale FROM information_schema.columns "
                                "WHERE table_schema=%(schema)s AND table_name=%(table)s ORDER BY ordinal_position",
                                {"schema": self.namespace, "table": table})
            self._columns[table] = {r["column_name"]: logical_type(r["data_type"]) for r in rows}
            self._numeric_shapes[table] = {r["column_name"]: (r["numeric_precision"], r["numeric_scale"])
                for r in rows if r["data_type"] == "numeric" and r["numeric_precision"] is not None}
        return self._columns[table]

    def rows(self, table, columns=None, filters=(), order=(), limit=None):
        return self.execute(*select_sql(self, table, columns, filters, order, limit))

    def close(self):
        if self.conn is not None:
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
        self._numeric_shapes = {}
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
            self._numeric_shapes[table] = {r["name"]: decimal_shape(r["type"]) for r in rows if decimal_shape(r["type"])}
        return self._columns[table]

    def rows(self, table, columns=None, filters=(), order=(), limit=None):
        rows = self.execute(*select_sql(self, table, columns, filters, order, limit))
        return typed_rows(self, table, rows)

    def close(self):
        self.session.close()


def get_reader(value):
    """Resolve a structured subscription or its internal reader; never a raw cursor."""
    if hasattr(value, "_reader"):
        return value._reader
    if isinstance(value, (PostgresReader, ClickHouseReader)) or hasattr(value, "rows"):
        return value
    raise TypeError("查询必须传入 Datafeed，不再接受原生数据库游标")


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
