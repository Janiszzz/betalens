"""Business time-series queries over the source-neutral wide tables.

Legacy public SQL builders and DataFrame utilities below are retained pending
migration approval. Structured queries never use legacy physical relations.
"""
from __future__ import annotations

import itertools
import logging
import re
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Tuple, Any, Union
import numpy as np
import pandas as pd

from .contracts import (RANGE_COLUMNS, STATUS_COLUMNS, SchemaError, PITUnavailableError,
    VersionConflictError, bounds, frame, local_time, proven, timestamp)
from .readers import get_reader, identifier, require_columns, PostgresReader, ClickHouseReader

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

PROVEN_COLUMNS = {"available_at": "timestamp", "availability_basis": "text",
                  "data_quality_status": "text"}


def _column_unit(reader, table, column):
    """Read native units without guessing a unit from a metric alias."""
    if not reader.columns("meta_column"):
        return
    require_columns(reader, "meta_column", {"table_name": "text", "column_name": "text", "unit": "text"})
    cache = getattr(reader, "_unit_contract", {})
    if table not in cache:
        rows = reader.rows("meta_column", ["column_name", "unit"], [("table_name", "=", table)])
        cache[table] = {r["column_name"]: r["unit"] for r in rows}
        reader._unit_contract = cache
    return cache[table].get(column)


def _query_fields(reader, table, metric, filters):
    """Validate actual column names and explicit dimension equality filters."""
    identifier(table)
    if table in {"stock_industry_membership_versions", "trade_calendar_versions",
                 "stock_suspension_intervals_versions", "index_membership"}:
        raise NotImplementedError("参考域与指数成分必须使用专用接口核验完整范围及发布证据")
    if metric is None:
        raise ValueError("metric 必须显式指定数据库数值字段")
    identifier(metric)
    if metric in {"code", "input_ts", "datetime", "diff_hours", "name"}:
        raise ValueError("metric 与固定返回列名冲突")
    if filters is not None and not isinstance(filters, dict):
        raise ValueError("filters 必须是数据库字段到单个值的字典")
    filters = dict(filters or {})
    for column, value in filters.items():
        identifier(column)
        if value is None or not pd.api.types.is_scalar(value) or pd.isna(value):
            raise ValueError("filters 仅接受非空标量相等条件")
    if table == "fundamental_periodic" and any(not filters.get(key) for key in ("statement_type", "report_type")):
        raise ValueError("fundamental_periodic 必须在 filters 中指定 statement_type 和 report_type")
    expected = {"security_code": "text", "event_date": "date", **PROVEN_COLUMNS}
    if table == "fundamental_periodic":
        expected.update(period_end="date", statement_type="text", report_type="text", version_id="text")
    elif table in {"market_daily_raw", "market_daily_post_adjusted"}:
        expected.update(asset_type="text", trade_date="date")
    require_columns(reader, table, {metric: None, **{key: None for key in filters}})
    require_columns(reader, table, expected)
    if reader.columns(table)[metric] not in {"numeric", "integer"}:
        raise SchemaError(f"{table}.{metric} 必须是数值字段")
    expected[metric] = reader.columns(table)[metric]
    return expected, [(key, "=", value) for key, value in filters.items()]


def _codes(codes, *, required=False):
    if codes is None and not required:
        return None
    if isinstance(codes, str) or codes is None:
        raise ValueError("codes 必须是证券代码列表")
    result = list(codes)
    if any(not isinstance(code, str) or not code.strip() for code in result):
        raise ValueError("证券代码必须是非空字符串")
    return result


def _name_rows(reader, codes):
    if not reader.columns("security_name_history"):
        return {}
    fields = {"security_code": "text", "name": "text", "start_date": "date",
              "end_date": "date", **PROVEN_COLUMNS}
    require_columns(reader, "security_name_history", fields)
    grouped = {}
    for row in reader.rows("security_name_history", list(fields), [("security_code", "IN", codes)]):
        grouped.setdefault(row["security_code"], []).append(row)
    return grouped


def historical_name(rows, code, at):
    day = local_time(at).date()
    candidates = [r for r in (rows.get(code, []) if rows else []) if proven(r, at)
                  and r["start_date"] is not None and r["start_date"] <= day
                  and (r["end_date"] is None or day <= r["end_date"])]
    if not candidates:
        return None
    latest = max(timestamp(r["available_at"], stored=True) for r in candidates)
    names = {r["name"] for r in candidates if timestamp(r["available_at"], stored=True) == latest}
    if len(names) != 1:
        raise VersionConflictError(f"历史名称冲突: {code}")
    return names.pop()


def observations(reader, table, metric, codes=None, lower=None, upper=None,
                 *, filters=None, end_exclusive=False, nearest=None):
    expected, predicates = _query_fields(reader, table, metric, filters)
    unit = _column_unit(reader, table, metric)
    if codes is not None:
        predicates.append(("security_code", "IN", list(dict.fromkeys(codes))))
    if lower is not None:
        predicates.append(("available_at", ">=", lower))
    if upper is not None:
        predicates.append(("available_at", "<" if end_exclusive else "<=", upper))
    if nearest is None:
        rows = reader.rows(table, list(expected), predicates)
    else:
        from .matching import nearest_candidates
        anchors, direction, ranges = nearest
        rows = nearest_candidates(reader, table, expected, predicates, codes, anchors, direction, ranges=ranges)
    rows = [r for r in rows if proven(r)]
    names = _name_rows(reader, list({r["security_code"] for r in rows})) if rows else []
    records = []
    for row in rows:
        at = timestamp(row["available_at"], stored=True)
        records.append({"code": row["security_code"], "available_at": at,
                        "value": float(row[metric]) if row[metric] is not None else np.nan,
                        "_value_key": Decimal(str(row[metric])) if row[metric] is not None else None,
                        "name": historical_name(names, row["security_code"], at),
                        "period_end": row.get("period_end"),
                        "remark": {"event_date": row["event_date"].isoformat(),
                                   "period_end": row["period_end"].isoformat() if row.get("period_end") else None,
                                   "version_id": row.get("version_id"), "table": table, "unit": unit,
                                   "asset_type": row.get("asset_type"),
                                   "statement_type": row.get("statement_type"), "report_type": row.get("report_type")}})
    # An old-period revision is a new observation, not silently a replacement
    # for the entire time series. Simultaneous releases prefer the newest period.
    selected = {}
    for record in records:
        key = (record["code"], record["available_at"])
        old = selected.get(key)
        if old is not None and old["period_end"] != record["period_end"]:
            if (record["period_end"] or datetime.min.date()) > (old["period_end"] or datetime.min.date()):
                selected[key] = record
            continue
        if old is not None and (old["remark"] != record["remark"] or old["_value_key"] != record["_value_key"]):
            raise VersionConflictError(f"同一可得时间存在冲突值: {table}/{metric}/{key[0]}")
        selected[key] = record
    return sorted(selected.values(), key=lambda r: (r["available_at"], r["code"]))


def query_time_range(cursor, table_name, codes=None, start_date=None, end_date=None,
                     metric=None, limit=None, logger=None, *, as_of=None, filters=None):
    reader = get_reader(cursor)
    codes = _codes(codes)
    lower, upper, exclusive = bounds(start_date, end_date)
    if as_of is not None:
        cutoff = timestamp(as_of)
        if upper is None or cutoff < upper:
            upper, exclusive = cutoff, False
    if limit is not None and (isinstance(limit, bool) or int(limit) != limit or limit < 0):
        raise ValueError("limit 必须是非负整数")
    records = []
    with reader.snapshot():
        rows = observations(reader, table_name, metric, codes, lower, upper,
                            filters=filters, end_exclusive=exclusive)
        records.extend({"datetime": local_time(r["available_at"]), "code": r["code"],
                        "name": r["name"], "metric": metric, "value": r["value"], "remark": r["remark"]} for r in rows)
    result = frame(records, RANGE_COLUMNS, numeric=("value",), times=("datetime",))
    result = result.sort_values(["datetime", "code", "metric"], ascending=[False, True, True], kind="stable")
    return result.iloc[:limit].reset_index(drop=True)


def match_nearest(rows, codes, anchors, metric, direction, time_tolerance=None, *, ranges=False):
    """Vectorized as-of matching; duplicate inputs retain their original rows."""
    if direction not in {"before", "after"}:
        raise ValueError("无效匹配方向")
    tolerance = None
    if time_tolerance is not None:
        if not np.isfinite(float(time_tolerance)) or float(time_tolerance) < 0:
            raise ValueError("time_tolerance 必须是非负小时数")
        tolerance = pd.Timedelta(hours=float(time_tolerance))
    inputs = []
    for code, anchor in itertools.product(codes, anchors):
        start, end = (timestamp(anchor[0]), timestamp(anchor[1])) if ranges else (timestamp(anchor), timestamp(anchor))
        if start > end:
            raise ValueError("区间开始不能晚于结束")
        inputs.append({"code": code, "input_ts": start if direction == "after" else end,
                       "start": start, "end": end, "ordinal": len(inputs)})
    columns = ["code", "input_ts", "datetime", "diff_hours", metric, "name"]
    if not inputs:
        return frame([], columns, numeric=("diff_hours", metric), times=("input_ts", "datetime"))
    requested = pd.DataFrame(inputs)
    if rows:
        source = pd.DataFrame(rows)[["code", "available_at", "value", "name"]]
        result = pd.merge_asof(requested.sort_values("input_ts"), source.sort_values("available_at"),
            left_on="input_ts", right_on="available_at", by="code",
            direction="backward" if direction == "before" else "forward",
            allow_exact_matches=direction == "before", tolerance=tolerance).sort_values("ordinal")
        if ranges:
            valid = ((result.available_at > result.start) & (result.available_at < result.end)) if direction == "after" else (
                (result.available_at >= result.start) & (result.available_at <= result.end))
            result.loc[~valid, "available_at"] = pd.NaT
            result.loc[~valid, ["value", "name"]] = [np.nan, None]
    else:
        result = requested.assign(available_at=pd.NaT, value=np.nan, name=None)
    records = []
    for r in result.to_dict("records"):
        hit = r["available_at"]
        diff = abs((r["input_ts"] - hit).total_seconds()) / 3600 if not pd.isna(hit) else np.nan
        records.append({"code": r["code"], "input_ts": local_time(r["input_ts"]),
                        "datetime": local_time(hit), "diff_hours": diff, metric: r["value"], "name": r["name"]})
    return frame(records, columns, numeric=("diff_hours", metric), times=("input_ts", "datetime"))


def _nearest(cursor, table_name, codes, anchors, metric, direction, time_tolerance,
             *, ranges=False, filters=None, engine=None):
    codes = _codes(codes, required=True)
    if metric is None:
        raise ValueError("metric 必须显式指定数据库数值字段")
    identifier(metric)
    if isinstance(anchors, (str, datetime)):
        raise ValueError("查询时点/区间必须是列表")
    anchors = list(anchors)
    if ranges and any(not isinstance(a, (list, tuple)) or len(a) != 2 for a in anchors):
        raise ValueError("ranges 的每个元素必须是 (start, end)")
    # Validate direction, ranges and tolerance even when no source rows exist.
    empty = match_nearest([], codes, anchors, metric, direction, time_tolerance, ranges=ranges)
    reader = get_reader(cursor)
    if not codes or not anchors:
        with reader.snapshot():
            _query_fields(reader, table_name, metric, filters)
        return empty
    times = [timestamp(x) for a in anchors for x in a] if ranges else [timestamp(a) for a in anchors]
    lower = min(times) if direction == "after" or ranges else None
    upper = max(times) if direction == "before" or ranges else None
    if time_tolerance is not None and not ranges:
        delta = pd.Timedelta(hours=float(time_tolerance))
        lower = min(times) - delta if direction == "before" else lower
        upper = max(times) + delta if direction == "after" else upper
    if engine is None:
        engine = "python"
        native = isinstance(reader, (PostgresReader, ClickHouseReader))
        count = len(set(codes))
        # Only use the SQL wins demonstrated by the reproducible benchmark.
        # Other query shapes keep the simpler bulk Python implementation.
        if native and direction == "before" and not ranges:
            if reader.backend == "postgres" and count >= 200:
                engine = "sql"
            elif reader.backend == "clickhouse" and count >= 5000 and len(set(times)) == 1:
                engine = "sql"
    if engine not in {"python", "sql"}:
        raise ValueError("无效的内部匹配实现")
    with reader.snapshot():
        rows = observations(reader, table_name, metric, codes, lower, upper,
                            filters=filters,
                            nearest=(anchors, direction, ranges) if engine == "sql" else None)
    return match_nearest(rows, codes, anchors, metric, direction, time_tolerance, ranges=ranges)


def query_nearest_before(cursor, table_name, codes, datetimes, metric, time_tolerance=None, logger=None, *, filters=None):
    return _nearest(cursor, table_name, codes, datetimes, metric, "before", time_tolerance, filters=filters)


def query_nearest_after(cursor, table_name, codes, datetimes, metric, time_tolerance=None, logger=None, *, filters=None):
    return _nearest(cursor, table_name, codes, datetimes, metric, "after", time_tolerance, filters=filters)


def query_nearest_in_range_before(cursor, table_name, codes, ranges, metric, time_tolerance=None, logger=None, *, filters=None):
    return _nearest(cursor, table_name, codes, ranges, metric, "before", time_tolerance, ranges=True, filters=filters)


def query_nearest_in_range_after(cursor, table_name, codes, ranges, metric, time_tolerance=None, logger=None, *, filters=None):
    return _nearest(cursor, table_name, codes, ranges, metric, "after", time_tolerance, ranges=True, filters=filters)


def get_available_dates(cursor, table_name, code, metric, start_date=None, end_date=None, logger=None, *, as_of=None, filters=None):
    result = query_time_range(cursor, table_name, [code], start_date, end_date, metric,
                              as_of=as_of, filters=filters)
    return sorted(value.to_pydatetime() for value in result.datetime.drop_duplicates())


def get_latest_date(cursor, table_name, code=None, metric=None, logger=None, *, as_of=None, filters=None):
    result = query_time_range(cursor, table_name, [code] if code else None, metric=metric,
                              limit=1, as_of=as_of, filters=filters)
    return None if result.empty else result.datetime.iloc[0].to_pydatetime()


def query_trade_status(cursor, table_name, codes, dates, metric="交易状态", logger=None):
    from .reference import load_reference, known_scopes, suspension_status

    if table_name != "stock_suspension_intervals_versions" or metric != "交易状态":
        raise NotImplementedError("交易状态查询仅支持 stock_suspension_intervals_versions/交易状态")
    codes = _codes(codes)
    if isinstance(dates, str):
        raise ValueError("dates 必须是列表")
    dates = list(dates)
    reader = get_reader(cursor)
    records = []
    with reader.snapshot():
        scopes = load_reference(reader, "suspension", codes=codes) if dates and codes != [] else []
        if codes is None:
            require_columns(reader, "security_dim", {"security_code": "text", "asset_type": "text"})
            securities = reader.rows("security_dim", ["security_code"], [("asset_type", "=", "stock")])
            codes = sorted({r["security_code"] for r in securities} | {s["selector"]["security_code"] for s in scopes})
        names = _name_rows(reader, codes) if codes and dates else []
        for date in dates:
            at = timestamp(date)
            day = local_time(at).date().isoformat()
            selected = {s["selector"]["security_code"]: s for s in known_scopes(scopes, at)
                        if s["selector"].get("trade_date") == day}
            for code in codes:
                scope = selected.get(code)
                value, reason = suspension_status(scope, at) if scope else (None, "缺少历史可知的完整停牌范围")
                records.append({"code": code, "datetime": local_time(at), "value": value,
                                "status_text": reason, "name": historical_name(names, code, at)})
    return frame(records, STATUS_COLUMNS, integers=("value",), times=("datetime",))

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
