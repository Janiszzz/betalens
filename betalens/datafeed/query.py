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
from datetime import datetime, timedelta
import logging


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

    conditions = []
    params = []

    if start_date:
        conditions.append("datetime >= %s::TIMESTAMP")
        params.append(start_date)

    if end_date:
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
        order_by="datetime DESC",
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
