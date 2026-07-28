#%%By Janis 250226
"""
数据库查询工具模块（函数式）
功能：
- 重构query_nearest_after和query_nearest_before
- 解耦数据库查询逻辑
- 提供灵活的查询参数构建
- 支持时间点匹配和数据提取
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Any, Union
import itertools
import re
from datetime import datetime, timedelta
import logging

from psycopg2 import sql as psql

from .registry import CoreMetric, get_core_metric, get_dataset


_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NORMALIZED_CONNECTIONS: set[tuple[Any, ...]] = set()
_METRIC_ROUTE_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _connection_cache_key(cursor) -> tuple[Any, ...]:
    connection = getattr(cursor, "connection", None)
    if connection is None:
        return (id(cursor),)
    try:
        params = connection.get_dsn_parameters()
        return (
            params.get("host"),
            params.get("port"),
            params.get("dbname"),
            params.get("user"),
            connection.get_backend_pid(),
        )
    except Exception:
        return (id(connection),)


def _normalized_schema_available(cursor) -> bool:
    """Return true when the db-manager normalized schema is installed."""
    connection_key = _connection_cache_key(cursor)
    if connection_key in _NORMALIZED_CONNECTIONS:
        return True
    cursor.execute(
        "SELECT to_regclass('betalens.entity_dim') IS NOT NULL "
        "AND to_regclass('betalens.market_daily_fact') IS NOT NULL AS ready"
    )
    row = cursor.fetchone()
    ready = bool(row.get("ready") if isinstance(row, dict) else row[0])
    if ready:
        _NORMALIZED_CONNECTIONS.add(connection_key)
    return ready


def _time_bounds(
    start_date: Optional[str],
    end_date: Optional[str],
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, bool]:
    start = pd.Timestamp(start_date) if start_date else None
    if not end_date:
        return start, None, False
    end_text = str(end_date).strip()
    end = pd.Timestamp(end_text)
    end_exclusive = bool(_DATE_ONLY_RE.fullmatch(end_text))
    if end_exclusive:
        end += pd.Timedelta(days=1)
    return start, end, end_exclusive


def _trade_date_bounds(
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    end_exclusive: bool,
    available_time,
):
    """Convert fixed intraday availability bounds into indexable dates."""

    def naive(value: pd.Timestamp) -> pd.Timestamp:
        return value.tz_localize(None) if value.tzinfo is not None else value

    lower = None
    if start is not None:
        start = naive(start)
        lower = start.date()
        if datetime.combine(lower, available_time) < start.to_pydatetime():
            lower += timedelta(days=1)
    upper = None
    if end is not None:
        end = naive(end)
        upper = end.date()
        available_at = datetime.combine(upper, available_time)
        if available_at > end.to_pydatetime() or (
            end_exclusive and available_at == end.to_pydatetime()
        ):
            upper -= timedelta(days=1)
    return lower, upper


def _resolve_metric(cursor, dataset: str, metric: str):
    cache_key = (*_connection_cache_key(cursor), dataset, metric)
    cached = _METRIC_ROUTE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cursor.execute(
        """
        SELECT m.metric_id, m.metric_name, m.storage_kind,
               m.storage_column, m.availability_time
        FROM betalens.metric_alias a
        JOIN betalens.metric_dim m ON m.metric_id = a.metric_id
        WHERE a.logical_dataset = %s AND a.alias = %s
        UNION ALL
        SELECT m.metric_id, m.metric_name, m.storage_kind,
               m.storage_column, m.availability_time
        FROM betalens.metric_dim m
        WHERE m.logical_dataset = %s AND m.metric_name = %s
          AND NOT EXISTS (
              SELECT 1 FROM betalens.metric_alias a
              WHERE a.logical_dataset = %s AND a.alias = %s
          )
        LIMIT 1
        """,
        (dataset, metric, dataset, metric, dataset, metric),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    if isinstance(row, dict):
        resolved = dict(row)
    else:
        keys = (
            "metric_id",
            "metric_name",
            "storage_kind",
            "storage_column",
            "availability_time",
        )
        resolved = dict(zip(keys, row))
    _METRIC_ROUTE_CACHE[cache_key] = resolved
    return resolved


def _row_value(row, key: str, position: int):
    return row.get(key) if isinstance(row, dict) else row[position]


def _resolved_core_metric(row) -> CoreMetric | None:
    if not row or _row_value(row, "storage_kind", 2) != "core":
        return None
    column = _row_value(row, "storage_column", 3)
    available_time = _row_value(row, "availability_time", 4)
    if not column or available_time is None:
        return None
    return CoreMetric(
        canonical_name=_row_value(row, "metric_name", 1),
        column=column,
        available_time=available_time,
    )


def _query_compatibility_view(
    cursor,
    table_name: str,
    codes: Optional[List[str]],
    start_date: Optional[str],
    end_date: Optional[str],
    metric: Optional[str],
    limit: Optional[int],
) -> pd.DataFrame:
    conditions: list[psql.SQL] = []
    params: list[Any] = []
    start, end, end_exclusive = _time_bounds(start_date, end_date)
    if codes:
        conditions.append(psql.SQL("code = ANY(%s::text[])"))
        params.append(list(codes))
    if start is not None:
        conditions.append(psql.SQL("datetime >= %s"))
        params.append(start.to_pydatetime())
    if end is not None:
        operator = psql.SQL("<" if end_exclusive else "<=")
        conditions.append(psql.SQL("datetime {} %s").format(operator))
        params.append(end.to_pydatetime())
    if metric is not None:
        conditions.append(psql.SQL("metric = %s"))
        params.append(metric)
    query = psql.SQL(
        "SELECT datetime, code, name, metric, value, remark FROM public.{}"
    ).format(psql.Identifier(table_name))
    if conditions:
        query += psql.SQL(" WHERE ") + psql.SQL(" AND ").join(conditions)
    query += psql.SQL(" ORDER BY datetime DESC, code, metric")
    if limit is not None:
        query += psql.SQL(" LIMIT %s")
        params.append(max(0, int(limit)))
    cursor.execute(query, params)
    return pd.DataFrame(cursor.fetchall())


def _empty_nearest(
    codes,
    anchors,
    metric: str,
    ranges: bool = False,
    direction: str = "before",
) -> pd.DataFrame:
    records = []
    for code in codes:
        for anchor in anchors:
            input_ts = (
                anchor[0] if ranges and direction == "after"
                else anchor[1] if ranges
                else anchor
            )
            records.append(
                {
                    "code": code,
                    "input_ts": pd.Timestamp(input_ts),
                    "datetime": pd.NaT,
                    "diff_hours": np.nan,
                    metric: np.nan,
                    "name": None,
                }
            )
    return pd.DataFrame(
        records,
        columns=["code", "input_ts", "datetime", "diff_hours", metric, "name"],
    )


def _query_time_range_normalized(
    cursor,
    table_name: str,
    codes: Optional[List[str]],
    start_date: Optional[str],
    end_date: Optional[str],
    metric: Optional[str],
    limit: Optional[int],
) -> pd.DataFrame:
    spec = get_dataset(table_name)
    if spec is None:
        raise ValueError(f"不支持的逻辑数据集: {table_name}")
    if metric is None or spec.kind not in {"market", "observation"}:
        # Multi-metric and specialized PIT reads retain the six-column view
        # contract. Hot market/observation reads below bypass the views.
        return _query_compatibility_view(
            cursor, table_name, codes, start_date, end_date, metric, limit
        )

    resolved = _resolve_metric(cursor, table_name, metric)
    core = _resolved_core_metric(resolved) or get_core_metric(table_name, metric)
    start, end, end_exclusive = _time_bounds(start_date, end_date)
    conditions: list[psql.SQL] = []
    params: list[Any] = []
    if core is not None and spec.kind == "market":
        trade_start, trade_end = _trade_date_bounds(
            start, end, end_exclusive, core.available_time
        )
        conditions.extend(
            [
                psql.SQL("e.entity_type = %s"),
                psql.SQL("f.{} IS NOT NULL").format(psql.Identifier(core.column)),
            ]
        )
        params.append(spec.entity_type)
        if codes:
            conditions.append(psql.SQL("e.code = ANY(%s::text[])"))
            params.append(list(codes))
        if trade_start is not None:
            conditions.append(psql.SQL("f.trade_date >= %s"))
            params.append(trade_start)
        if trade_end is not None:
            conditions.append(psql.SQL("f.trade_date <= %s"))
            params.append(trade_end)
        fact_query = psql.SQL(
            """
            SELECT f.trade_date + %s::time AS datetime,
                   e.code, COALESCE(historical_name.name, e.current_name) AS name,
                   %s AS metric,
                   f.{}::double precision AS value,
                   NULLIF(f.remark->%s, 'null'::jsonb) AS remark
            FROM betalens.market_daily_fact f
            JOIN betalens.entity_dim e ON e.entity_id = f.entity_id
            LEFT JOIN LATERAL (
                SELECT h.name
                FROM betalens.entity_name_history h
                WHERE h.entity_id = e.entity_id
                  AND h.valid_from <= f.trade_date + %s::time
                  AND (h.valid_to IS NULL OR h.valid_to > f.trade_date + %s::time)
                ORDER BY h.valid_from DESC
                LIMIT 1
            ) historical_name ON TRUE
            WHERE {}
            """
        ).format(psql.Identifier(core.column), psql.SQL(" AND ").join(conditions))
        fact_params = [
            core.available_time,
            metric,
            core.canonical_name,
            core.available_time,
            core.available_time,
            *params,
        ]
        metric_id = _row_value(resolved, "metric_id", 0) if resolved else None
        if metric_id is not None:
            observation_conditions = [psql.SQL("o.metric_id = %s")]
            observation_params: list[Any] = [metric_id]
            if spec.entity_type:
                observation_conditions.append(psql.SQL("e.entity_type = %s"))
                observation_params.append(spec.entity_type)
            if codes:
                observation_conditions.append(psql.SQL("e.code = ANY(%s::text[])"))
                observation_params.append(list(codes))
            if start is not None:
                observation_conditions.append(psql.SQL("o.available_at >= %s"))
                observation_params.append(start.to_pydatetime())
            if end is not None:
                observation_conditions.append(
                    psql.SQL("o.available_at {} %s").format(
                        psql.SQL("<" if end_exclusive else "<=")
                    )
                )
                observation_params.append(end.to_pydatetime())
            observation_query = psql.SQL(
                """
                SELECT o.available_at AS datetime, e.code,
                       COALESCE(historical_name.name, e.current_name) AS name,
                       %s AS metric,
                       o.value::double precision AS value, o.remark
                FROM betalens.observation_fact o
                JOIN betalens.entity_dim e ON e.entity_id = o.entity_id
                LEFT JOIN LATERAL (
                    SELECT h.name
                    FROM betalens.entity_name_history h
                    WHERE h.entity_id = e.entity_id
                      AND h.valid_from <= o.available_at
                      AND (h.valid_to IS NULL OR h.valid_to > o.available_at)
                    ORDER BY h.valid_from DESC
                    LIMIT 1
                ) historical_name ON TRUE
                WHERE {}
                """
            ).format(psql.SQL(" AND ").join(observation_conditions))
            query = psql.SQL(
                "SELECT * FROM (({}) UNION ALL ({})) AS logical_rows "
                "ORDER BY datetime DESC, code, metric"
            ).format(fact_query, observation_query)
            query_params = [*fact_params, metric, *observation_params]
        else:
            query = fact_query + psql.SQL(" ORDER BY datetime DESC, code, metric")
            query_params = fact_params
    else:
        resolved = resolved or _resolve_metric(cursor, table_name, metric)
        if not resolved:
            return pd.DataFrame(columns=["datetime", "code", "name", "metric", "value", "remark"])
        if _row_value(resolved, "storage_kind", 2) != "observation":
            return pd.DataFrame(columns=["datetime", "code", "name", "metric", "value", "remark"])
        metric_id = _row_value(resolved, "metric_id", 0)
        conditions.append(psql.SQL("o.metric_id = %s"))
        params.append(metric_id)
        if spec.entity_type:
            conditions.append(psql.SQL("e.entity_type = %s"))
            params.append(spec.entity_type)
        if codes:
            conditions.append(psql.SQL("e.code = ANY(%s::text[])"))
            params.append(list(codes))
        if start is not None:
            conditions.append(psql.SQL("o.available_at >= %s"))
            params.append(start.to_pydatetime())
        if end is not None:
            conditions.append(psql.SQL("o.available_at {} %s").format(psql.SQL("<" if end_exclusive else "<=")))
            params.append(end.to_pydatetime())
        query = psql.SQL(
            """
            SELECT o.available_at AS datetime, e.code,
                   COALESCE(historical_name.name, e.current_name) AS name,
                   %s AS metric, o.value::double precision AS value, o.remark
            FROM betalens.observation_fact o
            JOIN betalens.entity_dim e ON e.entity_id = o.entity_id
            LEFT JOIN LATERAL (
                SELECT h.name
                FROM betalens.entity_name_history h
                WHERE h.entity_id = e.entity_id
                  AND h.valid_from <= o.available_at
                  AND (h.valid_to IS NULL OR h.valid_to > o.available_at)
                ORDER BY h.valid_from DESC
                LIMIT 1
            ) historical_name ON TRUE
            WHERE {}
            ORDER BY o.available_at DESC, e.code, metric
            """
        ).format(psql.SQL(" AND ").join(conditions))
        query_params = [metric, *params]
    if limit is not None:
        query += psql.SQL(" LIMIT %s")
        query_params.append(max(0, int(limit)))
    cursor.execute(query, query_params)
    return pd.DataFrame(cursor.fetchall())


def _nearest_input_cte(ranges: bool) -> psql.SQL:
    if ranges:
        return psql.SQL(
            """
            input_ranges AS (
                SELECT start_ts, end_ts, range_ord
                FROM unnest(%s::timestamp[], %s::timestamp[]) WITH ORDINALITY
                     AS r(start_ts, end_ts, range_ord)
            ),
            input_data AS (
                SELECT c.code, r.start_ts, r.end_ts,
                       (c.code_ord - 1) * cardinality(%s::timestamp[]) + r.range_ord AS input_ord
                FROM unnest(%s::text[]) WITH ORDINALITY AS c(code, code_ord)
                CROSS JOIN input_ranges r
            )
            """
        )
    return psql.SQL(
        """
        input_data AS (
            SELECT c.code, t.input_ts,
                   (c.code_ord - 1) * cardinality(%s::timestamp[]) + t.time_ord AS input_ord
            FROM unnest(%s::text[]) WITH ORDINALITY AS c(code, code_ord)
            CROSS JOIN unnest(%s::timestamp[]) WITH ORDINALITY AS t(input_ts, time_ord)
        )
        """
    )


def _query_nearest_normalized(
    cursor,
    table_name: str,
    codes: List[str],
    anchors,
    metric: str,
    direction: str,
    time_tolerance: Optional[float],
    ranges: bool = False,
) -> pd.DataFrame:
    spec = get_dataset(table_name)
    if spec is None:
        raise ValueError(f"不支持的逻辑数据集: {table_name}")
    resolved_metric = _resolve_metric(cursor, table_name, metric)
    core = _resolved_core_metric(resolved_metric) or get_core_metric(table_name, metric)
    if direction not in {"after", "before"}:
        raise ValueError(f"无效的 direction: {direction}")

    if ranges:
        starts = [pd.Timestamp(value[0]).to_pydatetime() for value in anchors]
        ends = [pd.Timestamp(value[1]).to_pydatetime() for value in anchors]
        cte_params: list[Any] = [starts, ends, starts, list(codes)]
        input_ts = "r.start_ts" if direction == "after" else "r.end_ts"
        range_condition = (
            "x.available_at > r.start_ts AND x.available_at < r.end_ts"
            if direction == "after"
            else "x.available_at >= r.start_ts AND x.available_at <= r.end_ts"
        )
    else:
        datetimes = [pd.Timestamp(value).to_pydatetime() for value in anchors]
        cte_params = [datetimes, list(codes), datetimes]
        input_ts = "r.input_ts"
        range_condition = (
            "x.available_at > r.input_ts" if direction == "after" else "x.available_at <= r.input_ts"
        )
    order = "ASC" if direction == "after" else "DESC"
    diff_expr = (
        f"x.available_at - {input_ts}" if direction == "after" else f"{input_ts} - x.available_at"
    )
    output_diff_expr = (
        f"hit.available_at - {input_ts}"
        if direction == "after"
        else f"{input_ts} - hit.available_at"
    )
    tolerance_sql = ""
    tolerance_params: list[Any] = []
    if time_tolerance is not None:
        tolerance_sql = (
            f"AND x.available_at <= {input_ts} + %s * INTERVAL '1 hour'"
            if direction == "after"
            else f"AND x.available_at >= {input_ts} - %s * INTERVAL '1 hour'"
        )
        tolerance_params.append(float(time_tolerance))

    input_cte = _nearest_input_cte(ranges)
    observation_time_condition = psql.SQL(range_condition.replace("x.", "o."))
    if core is not None and spec.kind == "market":
        if ranges:
            fact_date_condition = psql.SQL(
                "AND f.trade_date >= r.start_ts::date "
                "AND f.trade_date <= r.end_ts::date"
            )
        elif direction == "after":
            fact_date_condition = psql.SQL("AND f.trade_date >= r.input_ts::date")
        else:
            fact_date_condition = psql.SQL("AND f.trade_date <= r.input_ts::date")
        source_sql = psql.SQL(
            """
            SELECT f.trade_date + %s::time AS available_at,
                   f.{}::double precision AS value
            FROM betalens.market_daily_fact f
            WHERE f.entity_id = r.entity_id AND f.{} IS NOT NULL {}
            """
        ).format(
            psql.Identifier(core.column),
            psql.Identifier(core.column),
            fact_date_condition,
        )
        source_params: list[Any] = [core.available_time]
        metric_id = (
            _row_value(resolved_metric, "metric_id", 0)
            if resolved_metric is not None
            else None
        )
        if metric_id is not None:
            source_sql += psql.SQL(
                """
                UNION ALL
                SELECT o.available_at, o.value::double precision AS value
                FROM betalens.observation_fact o
                WHERE o.entity_id = r.entity_id AND o.metric_id = %s
                  AND {}
                """
            ).format(observation_time_condition)
            source_params.append(metric_id)
        resolved_extra = psql.SQL("AND e.entity_type = %s")
        resolved_params: list[Any] = [spec.entity_type]
    else:
        resolved_metric = resolved_metric or _resolve_metric(cursor, table_name, metric)
        if not resolved_metric:
            return _empty_nearest(
                codes, anchors, metric, ranges=ranges, direction=direction
            )
        if _row_value(resolved_metric, "storage_kind", 2) != "observation":
            return _empty_nearest(
                codes, anchors, metric, ranges=ranges, direction=direction
            )
        metric_id = _row_value(resolved_metric, "metric_id", 0)
        source_sql = psql.SQL(
            """
            SELECT o.available_at, o.value::double precision AS value
            FROM betalens.observation_fact o
            WHERE o.entity_id = r.entity_id AND o.metric_id = %s
              AND {}
            """
        ).format(observation_time_condition)
        source_params = [metric_id]
        resolved_extra = psql.SQL("AND (%s IS NULL OR e.entity_type = %s)")
        resolved_params = [spec.entity_type, spec.entity_type]

    range_condition_sql = psql.SQL(range_condition)
    tolerance_fragment = psql.SQL(tolerance_sql)
    query = psql.SQL(
        f"""
        WITH {{input_cte}},
        resolved AS (
            SELECT i.*, e.entity_id, e.current_name
            FROM input_data i
            LEFT JOIN betalens.entity_dim e ON e.code = i.code {{resolved_extra}}
        )
        SELECT r.code, {input_ts} AS input_ts, hit.available_at AS datetime,
               EXTRACT(EPOCH FROM ({output_diff_expr}))/3600.0 AS diff_hours,
               hit.value,
               CASE WHEN hit.available_at IS NULL THEN NULL
                    ELSE COALESCE(historical_name.name, r.current_name) END AS name
        FROM resolved r
        LEFT JOIN LATERAL (
            SELECT x.available_at, x.value
            FROM ({{source_sql}}) x
            WHERE {{range_condition}} {{tolerance}}
            ORDER BY x.available_at {order}
            LIMIT 1
        ) hit ON TRUE
        LEFT JOIN LATERAL (
            SELECT h.name
            FROM betalens.entity_name_history h
            WHERE h.entity_id = r.entity_id
              AND h.valid_from <= hit.available_at
              AND (h.valid_to IS NULL OR h.valid_to > hit.available_at)
            ORDER BY h.valid_from DESC
            LIMIT 1
        ) historical_name ON hit.available_at IS NOT NULL
        ORDER BY r.input_ord
        """
    ).format(
        input_cte=input_cte,
        resolved_extra=resolved_extra,
        source_sql=source_sql,
        range_condition=range_condition_sql,
        tolerance=tolerance_fragment,
    )
    params = [*cte_params, *resolved_params, *source_params, *tolerance_params]
    cursor.execute(query, params)
    df = pd.DataFrame(cursor.fetchall())
    if "value" in df.columns:
        df.rename(columns={"value": metric}, inplace=True)
    return df


def _query_trade_status_normalized(
    cursor,
    codes: Optional[List[str]],
    dates: List[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    date_days = sorted({pd.Timestamp(value).normalize() for value in dates})
    cursor.execute(
        "SELECT max_available_at FROM betalens.dataset_coverage "
        "WHERE logical_dataset = 'trade_status'"
    )
    coverage = cursor.fetchone()
    coverage_value = None
    if coverage:
        coverage_value = coverage.get("max_available_at") if isinstance(coverage, dict) else coverage[0]
    if coverage_value is None or date_days[-1].date() > pd.Timestamp(coverage_value).date():
        logger.warning(
            "trade_status 数据覆盖不足：请求截至 %s，已知覆盖截至 %s；"
            "覆盖外日期只按上市状态和已知异常事件还原",
            date_days[-1].date(),
            None if coverage_value is None else pd.Timestamp(coverage_value).date(),
        )
    if codes:
        entity_source = """
            SELECT c.code, e.entity_id, e.current_name, e.first_trade_date, e.delist_date
            FROM unnest(%s::text[]) WITH ORDINALITY AS c(code, code_ord)
            LEFT JOIN betalens.entity_dim e ON e.code = c.code AND e.entity_type = 'stock'
        """
        params: list[Any] = [list(codes), [value.to_pydatetime() for value in date_days]]
    else:
        entity_source = """
            SELECT e.code, e.entity_id, e.current_name, e.first_trade_date, e.delist_date
            FROM betalens.entity_dim e WHERE e.entity_type = 'stock'
        """
        params = [[value.to_pydatetime() for value in date_days]]
    query = f"""
        WITH entities AS ({entity_source}),
        requested_dates AS (
            SELECT value::date AS status_date
            FROM unnest(%s::timestamp[]) AS d(value)
        )
        SELECT e.code, d.status_date::timestamp AS datetime,
               CASE
                   WHEN e.entity_id IS NULL OR e.first_trade_date IS NULL
                        OR d.status_date < e.first_trade_date
                        OR (e.delist_date IS NOT NULL AND d.status_date > e.delist_date) THEN -1
                   WHEN s.entity_id IS NOT NULL THEN s.status
                   ELSE 1
               END AS value,
               CASE
                   WHEN e.entity_id IS NULL OR e.first_trade_date IS NULL
                        OR d.status_date < e.first_trade_date
                        OR (e.delist_date IS NOT NULL AND d.status_date > e.delist_date) THEN '无法交易'
                   WHEN s.entity_id IS NOT NULL THEN COALESCE(s.status_text, '异常')
                   ELSE '交易'
               END AS status_text,
               COALESCE(historical_name.name, e.current_name) AS name
        FROM entities e CROSS JOIN requested_dates d
        LEFT JOIN LATERAL (
            SELECT h.name
            FROM betalens.entity_name_history h
            WHERE h.entity_id = e.entity_id
              AND h.valid_from <= d.status_date::timestamp
              AND (h.valid_to IS NULL OR h.valid_to > d.status_date::timestamp)
            ORDER BY h.valid_from DESC
            LIMIT 1
        ) historical_name ON e.entity_id IS NOT NULL
        LEFT JOIN betalens.trade_status_event s
          ON s.entity_id = e.entity_id AND s.event_date = d.status_date
        ORDER BY d.status_date, e.code
    """
    cursor.execute(query, params)
    return pd.DataFrame(
        cursor.fetchall(),
        columns=["code", "datetime", "value", "status_text", "name"],
    )


def _get_default_logger():
    """获取默认logger"""
    logger = logging.getLogger('TimeSeriesQueryEngine')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def build_query(
    table_name: str,
    conditions: Optional[List[str]] = None,
    params: Optional[List] = None,
    select_columns: str = '*',
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[str, List]:
    """
    构建SQL查询

    Args:
        table_name: 数据库表名
        conditions: 条件列表
        params: 参数列表
        select_columns: 要选择的列
        order_by: ORDER BY 子句（如 "datetime DESC"）
        limit: 最大返回行数

    Returns:
        (SQL语句, 参数列表)
    """
    query = f"SELECT {select_columns} FROM {table_name}"

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    if order_by:
        query += f" ORDER BY {order_by}"

    if limit is not None:
        query += f" LIMIT {int(limit)}"

    if params is None:
        params = []

    return query, params


def generate_input_pairs(
    codes: List[str],
    datetimes: List[str]
) -> List[Tuple[str, str]]:
    """
    生成(code, datetime)笛卡尔积
    
    Args:
        codes: 代码列表
        datetimes: 时间戳列表
        
    Returns:
        (code, datetime)元组列表
    """
    return list(itertools.product(codes, datetimes))


def generate_input_range_pairs(
    codes: List[str],
    ranges: List[Tuple[str, str]]
) -> List[Tuple[str, str, str]]:
    """
    生成 (code, start_ts, end_ts) 笛卡尔积

    Args:
        codes: 代码列表
        ranges: (start_ts, end_ts) 区间列表

    Returns:
        (code, start_ts, end_ts) 元组列表
    """
    return [(code, start, end) for code, (start, end) in itertools.product(codes, ranges)]


def build_nearest_in_range_query(
    table_name: str,
    input_tuples: List[Tuple[str, str, str]],
    metric: str,
    direction: str = 'after',  # 'after' or 'before'
    time_tolerance: Optional[float] = None
) -> Tuple[str, List]:
    """
    构建区间内最近时点匹配查询

    在每个 (code, start_ts, end_ts) 区间内，按方向查找距锚点最近的数据：
    - direction='after'：锚点为 start_ts，区间过滤 t.datetime > start AND t.datetime < end
    - direction='before'：锚点为 end_ts，区间过滤 t.datetime <= end AND t.datetime >= start

    Args:
        table_name: 表名
        input_tuples: (code, start_ts, end_ts) 元组列表
        metric: 指标名
        direction: 查询方向，'after' 或 'before'
        time_tolerance: 锚点容差（小时），与区间共同生效（取交集）

    Returns:
        (SQL语句, 参数列表)
    """
    value_placeholders = ', '.join(
        ['(%s, %s::TIMESTAMP, %s::TIMESTAMP)'] * len(input_tuples)
    )

    if direction == 'after':
        # 锚点 = start_ts
        anchor_select = 'i.start_ts AS input_ts'
        range_condition = 'AND t.datetime > i.start_ts AND t.datetime < i.end_ts'
        time_diff_expr = 't.datetime - i.start_ts'
        order_by = 'ASC'
    elif direction == 'before':
        # 锚点 = end_ts
        anchor_select = 'i.end_ts AS input_ts'
        range_condition = 'AND t.datetime <= i.end_ts AND t.datetime >= i.start_ts'
        time_diff_expr = 'i.end_ts - t.datetime'
        order_by = 'DESC'
    else:
        raise ValueError(f"无效的direction: {direction}，应为'after'或'before'")

    tolerance_condition = ""
    if time_tolerance is not None:
        tolerance_condition = f"AND ({time_diff_expr}) <= %s * INTERVAL '1 hour'"

    sql = f"""
    WITH input_data (code, start_ts, end_ts) AS (
        VALUES {value_placeholders}
    ),
    candidate_data AS (
        SELECT
            i.code,
            {anchor_select},
            t.datetime AS datetime,
            EXTRACT(EPOCH FROM ({time_diff_expr}))/3600 AS diff_hours,
            t.value,
            t.name,
            ROW_NUMBER() OVER (
                PARTITION BY i.code, i.start_ts, i.end_ts
                ORDER BY t.datetime {order_by}
            ) AS rn
        FROM input_data i
        LEFT JOIN {table_name} t
            ON i.code = t.code
            {range_condition}
            AND t.metric = %s
            {tolerance_condition}
    )
    SELECT
        code,
        input_ts,
        datetime,
        diff_hours,
        value,
        name
    FROM candidate_data
    WHERE rn = 1
    """

    params_list = []
    for code, start_ts, end_ts in input_tuples:
        params_list.extend([code, start_ts, end_ts])
    params_list.append(metric)

    if time_tolerance is not None:
        params_list.append(time_tolerance)

    return sql, params_list


def build_nearest_query(
    table_name: str,
    input_tuples: List[Tuple[str, str]],
    metric: str,
    direction: str = 'after',  # 'after' or 'before'
    time_tolerance: Optional[float] = None
) -> Tuple[str, List]:
    """
    构建最近时点匹配查询
    
    Args:
        table_name: 表名
        input_tuples: (code, datetime)元组列表
        metric: 指标名
        direction: 查询方向，'after'（之后）或'before'（之前）
        time_tolerance: 时间容差（小时）
        
    Returns:
        (SQL语句, 参数列表)
    """
    # 生成输入数据占位符
    value_placeholders = ', '.join(['(%s, %s::TIMESTAMP)'] * len(input_tuples))
    
    # 根据方向设置比较运算符和排序
    if direction == 'after':
        comparison_op = '>'
        order_by = 'ASC'
        time_diff_expr = 't.datetime - i.input_ts'
    elif direction == 'before':
        comparison_op = '<='
        order_by = 'DESC'
        time_diff_expr = 'i.input_ts - t.datetime'
    else:
        raise ValueError(f"无效的direction: {direction}，应为'after'或'before'")
    
    # 时间容差条件
    tolerance_condition = ""
    if time_tolerance is not None:
        tolerance_condition = f"AND ({time_diff_expr}) <= %s * INTERVAL '1 hour'"
    
    # 构建SQL
    sql = f"""
    WITH input_data (code, input_ts) AS (
        VALUES {value_placeholders}
    ),
    candidate_data AS (
        SELECT
            i.code,
            i.input_ts,
            t.datetime AS datetime,
            EXTRACT(EPOCH FROM ({time_diff_expr}))/3600 AS diff_hours,
            t.value,
            t.name,
            ROW_NUMBER() OVER (
                PARTITION BY i.code, i.input_ts 
                ORDER BY t.datetime {order_by}
            ) AS rn
        FROM input_data i
        LEFT JOIN {table_name} t
            ON i.code = t.code
            AND t.datetime {comparison_op} i.input_ts
            AND t.metric = %s
            {tolerance_condition}
    )
    SELECT 
        code,
        input_ts,
        datetime,
        diff_hours,
        value,
        name
    FROM candidate_data
    WHERE rn = 1
    """
    
    # 构造参数列表
    params_list = []
    for code, dt in input_tuples:
        params_list.extend([code, dt])
    params_list.append(metric)
    
    if time_tolerance is not None:
        params_list.append(time_tolerance)
    
    return sql, params_list


def query_nearest_after(
    cursor,
    table_name: str,
    codes: List[str],
    datetimes: List[str],
    metric: str,
    time_tolerance: Optional[float] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    查询每个时点之后最近的有效值
    
    用途：主要用于回测时提取价格
    时间结构：最新特征 <= 提数时点 < 调仓时点
    
    Args:
        cursor: 数据库游标
        table_name: 表名
        codes: 代码列表
        datetimes: 时间戳列表，格式'YYYY-MM-DD HH:MM:SS'
        metric: 查询的指标名称
        time_tolerance: 允许的最大时间间隔（单位：小时）
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        DataFrame，包含列：
            - code: 代码
            - input_ts: 输入时间戳（提数时点）
            - datetime: 匹配到的数据时间戳
            - diff_hours: 时间差（小时）
            - value: 数据值
            - name: 名称
    """
    if logger is None:
        logger = _get_default_logger()
    
    # 参数验证
    if not codes:
        raise ValueError("codes不能为空")
    if not datetimes:
        raise ValueError("datetimes不能为空")
    if not metric:
        raise ValueError("metric不能为空")

    dataset = get_dataset(table_name)
    if (
        dataset is not None
        and dataset.kind in {"market", "observation"}
        and _normalized_schema_available(cursor)
    ):
        return _query_nearest_normalized(
            cursor, table_name, codes, datetimes, metric,
            direction="after", time_tolerance=time_tolerance,
        )
    
    # 生成输入对
    input_tuples = generate_input_pairs(codes, datetimes)
    
    # 构建查询
    sql, params = build_nearest_query(
        table_name=table_name,
        input_tuples=input_tuples,
        metric=metric,
        direction='after',
        time_tolerance=time_tolerance
    )
    
    # 执行查询
    logger.info(f"执行query_nearest_after: {len(codes)}个代码 × {len(datetimes)}个时点 = {len(input_tuples)}个查询")
    
    cursor.execute(sql, params)
    df = pd.DataFrame(cursor.fetchall())
    
    # 重命名value列为实际指标名
    if not df.empty and 'value' in df.columns:
        df.rename(columns={'value': metric}, inplace=True)
    
    logger.info(f"查询完成，返回 {len(df)} 条记录")
    
    return df


def query_nearest_before(
    cursor,
    table_name: str,
    codes: List[str],
    datetimes: List[str],
    metric: str,
    time_tolerance: Optional[float] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    查询每个时点之前最近的有效值
    
    用途：主要用于回测时提取历史价格特征
    时间结构：调仓时点 <= 提数时点 < 最新特征时点
    
    Args:
        cursor: 数据库游标
        table_name: 表名
        codes: 代码列表
        datetimes: 时间戳列表，格式'YYYY-MM-DD HH:MM:SS'
        metric: 查询的指标名称
        time_tolerance: 允许的最大时间间隔（单位：小时）
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        DataFrame，包含列：
            - code: 代码
            - input_ts: 输入时间戳（提数时点）
            - datetime: 匹配到的数据时间戳
            - diff_hours: 时间差（小时）
            - value: 数据值
            - name: 名称
    """
    if logger is None:
        logger = _get_default_logger()
    
    # 参数验证
    if not codes:
        raise ValueError("codes不能为空")
    if not datetimes:
        raise ValueError("datetimes不能为空")
    if not metric:
        raise ValueError("metric不能为空")

    dataset = get_dataset(table_name)
    if (
        dataset is not None
        and dataset.kind in {"market", "observation"}
        and _normalized_schema_available(cursor)
    ):
        return _query_nearest_normalized(
            cursor, table_name, codes, datetimes, metric,
            direction="before", time_tolerance=time_tolerance,
        )
    
    # 生成输入对
    input_tuples = generate_input_pairs(codes, datetimes)
    
    # 构建查询
    sql, params = build_nearest_query(
        table_name=table_name,
        input_tuples=input_tuples,
        metric=metric,
        direction='before',
        time_tolerance=time_tolerance
    )
    
    # 执行查询
    logger.info(f"执行query_nearest_before: {len(codes)}个代码 × {len(datetimes)}个时点 = {len(input_tuples)}个查询")
    
    cursor.execute(sql, params)
    df = pd.DataFrame(cursor.fetchall())
    
    # 重命名value列为实际指标名
    if not df.empty and 'value' in df.columns:
        df.rename(columns={'value': metric}, inplace=True)
    
    logger.info(f"查询完成，返回 {len(df)} 条记录")

    return df


def query_nearest_in_range_after(
    cursor,
    table_name: str,
    codes: List[str],
    ranges: List[Tuple[str, str]],
    metric: str,
    time_tolerance: Optional[float] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    在每个 (start, end) 区间内查询距 start 最近的有效值（向后查）

    时间结构：start <= t.datetime - epsilon, t.datetime < end，锚点 = start

    Args:
        cursor: 数据库游标
        table_name: 表名
        codes: 代码列表
        ranges: (start, end) 区间列表，时间格式 'YYYY-MM-DD HH:MM:SS'
        metric: 指标名
        time_tolerance: 锚点容差（小时），与区间共同生效
        logger: 日志记录器

    Returns:
        DataFrame: code, input_ts(=start), datetime, diff_hours, value, name
    """
    if logger is None:
        logger = _get_default_logger()

    if not codes:
        raise ValueError("codes不能为空")
    if not ranges:
        raise ValueError("ranges不能为空")
    if not metric:
        raise ValueError("metric不能为空")

    dataset = get_dataset(table_name)
    if (
        dataset is not None
        and dataset.kind in {"market", "observation"}
        and _normalized_schema_available(cursor)
    ):
        return _query_nearest_normalized(
            cursor, table_name, codes, ranges, metric,
            direction="after", time_tolerance=time_tolerance, ranges=True,
        )

    input_tuples = generate_input_range_pairs(codes, ranges)

    sql, params = build_nearest_in_range_query(
        table_name=table_name,
        input_tuples=input_tuples,
        metric=metric,
        direction='after',
        time_tolerance=time_tolerance
    )

    logger.info(
        f"执行query_nearest_in_range_after: {len(codes)}个代码 × "
        f"{len(ranges)}个区间 = {len(input_tuples)}个查询"
    )

    cursor.execute(sql, params)
    df = pd.DataFrame(cursor.fetchall())

    if not df.empty and 'value' in df.columns:
        df.rename(columns={'value': metric}, inplace=True)

    logger.info(f"查询完成，返回 {len(df)} 条记录")

    return df


def query_nearest_in_range_before(
    cursor,
    table_name: str,
    codes: List[str],
    ranges: List[Tuple[str, str]],
    metric: str,
    time_tolerance: Optional[float] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    在每个 (start, end) 区间内查询距 end 最近的有效值（向前查）

    时间结构：start <= t.datetime <= end，锚点 = end

    Args:
        cursor: 数据库游标
        table_name: 表名
        codes: 代码列表
        ranges: (start, end) 区间列表，时间格式 'YYYY-MM-DD HH:MM:SS'
        metric: 指标名
        time_tolerance: 锚点容差（小时），与区间共同生效
        logger: 日志记录器

    Returns:
        DataFrame: code, input_ts(=end), datetime, diff_hours, value, name
    """
    if logger is None:
        logger = _get_default_logger()

    if not codes:
        raise ValueError("codes不能为空")
    if not ranges:
        raise ValueError("ranges不能为空")
    if not metric:
        raise ValueError("metric不能为空")

    dataset = get_dataset(table_name)
    if (
        dataset is not None
        and dataset.kind in {"market", "observation"}
        and _normalized_schema_available(cursor)
    ):
        return _query_nearest_normalized(
            cursor, table_name, codes, ranges, metric,
            direction="before", time_tolerance=time_tolerance, ranges=True,
        )

    input_tuples = generate_input_range_pairs(codes, ranges)

    sql, params = build_nearest_in_range_query(
        table_name=table_name,
        input_tuples=input_tuples,
        metric=metric,
        direction='before',
        time_tolerance=time_tolerance
    )

    logger.info(
        f"执行query_nearest_in_range_before: {len(codes)}个代码 × "
        f"{len(ranges)}个区间 = {len(input_tuples)}个查询"
    )

    cursor.execute(sql, params)
    df = pd.DataFrame(cursor.fetchall())

    if not df.empty and 'value' in df.columns:
        df.rename(columns={'value': metric}, inplace=True)

    logger.info(f"查询完成，返回 {len(df)} 条记录")

    return df


def query_time_range(
    cursor,
    table_name: str,
    codes: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    metric: Optional[str] = None,
    limit: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    查询指定时间范围的数据

    Args:
        cursor: 数据库游标
        table_name: 表名
        codes: 代码列表，None表示所有代码
        start_date: 开始日期
        end_date: 结束日期
        metric: 指标名称
        limit: 最大返回行数，None表示不限制（按 datetime DESC 返回最新的 N 行）
        logger: 日志记录器，如果为None则使用默认logger

    Returns:
        DataFrame
    """
    if logger is None:
        logger = _get_default_logger()

    if get_dataset(table_name) and _normalized_schema_available(cursor):
        return _query_time_range_normalized(
            cursor, table_name, codes, start_date, end_date, metric, limit
        )

    conditions = []
    params = []

    if start_date:
        conditions.append("datetime >= %s::TIMESTAMP")
        params.append(start_date)

    if end_date:
        end_text = str(end_date).strip()
        if _DATE_ONLY_RE.fullmatch(end_text):
            conditions.append("datetime < %s::TIMESTAMP")
            params.append(str(pd.Timestamp(end_text) + pd.Timedelta(days=1)))
        else:
            conditions.append("datetime <= %s::TIMESTAMP")
            params.append(end_date)

    if codes:
        placeholders = ','.join(['%s'] * len(codes))
        conditions.append(f"code IN ({placeholders})")
        params.extend(codes)

    if metric:
        conditions.append("metric = %s")
        params.append(metric)

    sql, params = build_query(
        table_name, conditions, params,
        order_by="datetime DESC, code, metric",
        limit=limit,
    )

    logger.info(f"执行时间范围查询: {sql}")

    cursor.execute(sql, params)
    df = pd.DataFrame(cursor.fetchall())

    logger.info(f"查询完成，返回 {len(df)} 条记录")

    return df


def get_available_dates(
    cursor,
    table_name: str,
    code: str,
    metric: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> List[datetime]:
    """
    获取指定代码和指标的可用日期列表
    
    Args:
        cursor: 数据库游标
        table_name: 表名
        code: 代码
        metric: 指标
        start_date: 开始日期
        end_date: 结束日期
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        日期列表
    """
    if logger is None:
        logger = _get_default_logger()

    if get_dataset(table_name) and _normalized_schema_available(cursor):
        data = _query_time_range_normalized(
            cursor, table_name, [code], start_date, end_date, metric, None
        )
        if data.empty or "datetime" not in data:
            return []
        return sorted(pd.to_datetime(data["datetime"]).dropna().unique().tolist())
    
    conditions = []
    params = []
    
    conditions.append("code = %s")
    params.append(code)
    
    conditions.append("metric = %s")
    params.append(metric)
    
    if start_date:
        conditions.append("datetime >= %s::TIMESTAMP")
        params.append(start_date)
    
    if end_date:
        end_text = str(end_date).strip()
        if _DATE_ONLY_RE.fullmatch(end_text):
            conditions.append("datetime < %s::TIMESTAMP")
            params.append(str(pd.Timestamp(end_text) + pd.Timedelta(days=1)))
        else:
            conditions.append("datetime <= %s::TIMESTAMP")
            params.append(end_date)
    
    sql, params = build_query(table_name, conditions, params, select_columns='DISTINCT datetime')
    sql += " ORDER BY datetime"
    
    cursor.execute(sql, params)
    results = cursor.fetchall()
    
    dates = [row['datetime'] for row in results]
    
    logger.info(f"获取到 {len(dates)} 个可用日期")
    
    return dates


def get_latest_date(
    cursor,
    table_name: str,
    code: Optional[str] = None,
    metric: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[datetime]:
    """
    获取最新的数据日期
    
    Args:
        cursor: 数据库游标
        table_name: 表名
        code: 代码，None表示所有代码
        metric: 指标，None表示所有指标
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        最新日期
    """
    if logger is None:
        logger = _get_default_logger()

    if get_dataset(table_name) and _normalized_schema_available(cursor):
        data = _query_time_range_normalized(
            cursor, table_name, [code] if code else None, None, None, metric, 1
        )
        if data.empty or "datetime" not in data:
            return None
        return pd.Timestamp(data.iloc[0]["datetime"]).to_pydatetime()
    
    conditions = []
    params = []
    
    if code:
        conditions.append("code = %s")
        params.append(code)
    
    if metric:
        conditions.append("metric = %s")
        params.append(metric)
    
    sql, params = build_query(table_name, conditions, params, select_columns='MAX(datetime) as max_date')
    
    cursor.execute(sql, params)
    result = cursor.fetchone()
    
    if result and result['max_date']:
        return result['max_date']
    
    return None


def query_trade_status(
    cursor,
    table_name: str,
    codes: Optional[List[str]],
    dates: List[str],
    metric: str = '交易状态',
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    查询个券交易状态（适配稀疏存储）

    表中仅存异常状态(value=0)与首次正常交易日锚点(value=1, remark.first_normal=true)。
    本函数在 Python 端把稀疏记录解析为每个 (code, date) 的完整状态：
        -1 = 无法交易（首次正常交易日之前，视为未上市/未交易）
         0 = 异常（停牌等，status_text 给出文本）
         1 = 正常交易

    Args:
        cursor: 数据库游标（RealDictCursor）
        table_name: 表名（trade_status）
        codes: 代码列表；None 表示全市场（取所有有锚点的代码）
        dates: 日期列表，格式 'YYYY-MM-DD' 或带时间，按自然日匹配
        metric: 指标名，默认 '交易状态'
        logger: 日志记录器

    Returns:
        DataFrame，列：code, datetime(=输入日), value(-1/0/1), status_text, name
    """
    if logger is None:
        logger = _get_default_logger()
    if not dates:
        raise ValueError("dates不能为空")

    if table_name == "trade_status" and _normalized_schema_available(cursor):
        return _query_trade_status_normalized(cursor, codes, dates, logger)

    # 输入日期归一到自然日
    date_days = sorted({pd.Timestamp(d).normalize() for d in dates})
    day_start = date_days[0]
    day_end = date_days[-1] + pd.Timedelta(days=1)

    # 1) 查首次正常交易日锚点（每个 code 一条）
    anchor_conds = ["metric = %s", "value = 1"]
    anchor_params: List = [metric]
    if codes:
        anchor_conds.append(f"code IN ({','.join(['%s'] * len(codes))})")
        anchor_params.extend(codes)
    anchor_sql = (
        f"SELECT code, name, MIN(datetime) AS first_normal "
        f"FROM {table_name} WHERE {' AND '.join(anchor_conds)} GROUP BY code, name"
    )
    cursor.execute(anchor_sql, anchor_params)
    anchor_rows = cursor.fetchall()
    first_normal = {r['code']: pd.Timestamp(r['first_normal']).normalize() for r in anchor_rows}
    name_map = {r['code']: r['name'] for r in anchor_rows}

    # 2) 查请求日期范围内的异常记录
    abn_conds = ["metric = %s", "value = 0",
                 "datetime >= %s::TIMESTAMP", "datetime < %s::TIMESTAMP"]
    abn_params: List = [metric, str(day_start), str(day_end)]
    if codes:
        abn_conds.append(f"code IN ({','.join(['%s'] * len(codes))})")
        abn_params.extend(codes)
    abn_sql = (
        f"SELECT code, name, datetime, remark "
        f"FROM {table_name} WHERE {' AND '.join(abn_conds)}"
    )
    cursor.execute(abn_sql, abn_params)
    abn_rows = cursor.fetchall()
    # (code, day) -> status_text
    abnormal: Dict[Tuple[str, pd.Timestamp], str] = {}
    for r in abn_rows:
        day = pd.Timestamp(r['datetime']).normalize()
        remark = r.get('remark') or {}
        status_text = remark.get('status') if isinstance(remark, dict) else None
        abnormal[(r['code'], day)] = status_text or '异常'
        name_map.setdefault(r['code'], r['name'])

    # 3) 目标代码集合：显式 codes 或全部有锚点的 code
    target_codes = codes if codes else list(first_normal.keys())

    # 4) 解析每个 (code, day) 的状态
    records = []
    for code in target_codes:
        fn = first_normal.get(code)
        for day in date_days:
            if (code, day) in abnormal:
                value, text = 0, abnormal[(code, day)]
            elif fn is None or day < fn:
                value, text = -1, '无法交易'
            else:
                value, text = 1, '交易'
            records.append({
                'code': code,
                'datetime': day,
                'value': value,
                'status_text': text,
                'name': name_map.get(code),
            })

    df = pd.DataFrame(
        records, columns=['code', 'datetime', 'value', 'status_text', 'name']
    )
    logger.info(f"query_trade_status: {len(target_codes)}个代码 × {len(date_days)}个日期 = {len(df)} 条状态")
    return df


# DataFrame辅助函数（保持为独立函数）
def pivot_to_wide(
    df: pd.DataFrame,
    index_cols: List[str],
    pivot_col: str,
    value_col: str
) -> pd.DataFrame:
    """
    将长格式数据转换为宽格式
    
    Args:
        df: 长格式DataFrame
        index_cols: 索引列
        pivot_col: 用于pivot的列（将变为新列名）
        value_col: 值列
        
    Returns:
        宽格式DataFrame
    """
    return df.pivot_table(
        index=index_cols,
        columns=pivot_col,
        values=value_col,
        aggfunc='first'
    ).reset_index()


def align_to_dates(
    df: pd.DataFrame,
    target_dates: List[datetime],
    date_column: str = 'datetime',
    method: str = 'ffill'
) -> pd.DataFrame:
    """
    将数据对齐到目标日期序列
    
    Args:
        df: 输入DataFrame
        target_dates: 目标日期列表
        date_column: 日期列名
        method: 填充方法，'ffill'或'bfill'
        
    Returns:
        对齐后的DataFrame
    """
    # 创建目标日期的DataFrame
    target_df = pd.DataFrame({date_column: target_dates})
    
    # 合并
    result = pd.merge(
        target_df,
        df,
        on=date_column,
        how='left'
    )
    
    # 填充
    if method == 'ffill':
        result = result.ffill()
    elif method == 'bfill':
        result = result.bfill()
    
    return result


def calculate_returns(
    df: pd.DataFrame,
    price_column: str,
    periods: List[int] = [1],
    group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    计算收益率
    
    Args:
        df: 包含价格数据的DataFrame
        price_column: 价格列名
        periods: 计算周期列表
        group_by: 分组列（如code）
        
    Returns:
        添加了收益率列的DataFrame
    """
    df = df.copy()
    
    if group_by:
        grouped = df.groupby(group_by)
    else:
        grouped = df
    
    for period in periods:
        return_col = f'return_{period}d'
        df[return_col] = grouped[price_column].pct_change(periods=period)
    
    return df
