#%%By Janis 260603
"""
指数历史股票池查询模块（函数式）

设计要点
--------
指数成分股池是 point-in-time 的"状态"数据：某指数从某调整日起拥有一组成分股，
直到下次调整。每个生效日存为一行，约定：
    - code   : 指数代码（如 '000906.SH'）
    - name   : 指数名称（如 '中证800'）
    - metric : 固定为 'universe'，标识成分股池
    - value  : 成分股数量（便于校验）
    - remark : JSONB，约定 {"index_code", "index_name", "constituents": [...]}
    - datetime: 该股票池的生效时点（最早可知日）

查询语义 = 取 datetime <= 查询日 的最近一条，与 query.query_nearest_before 同构，
天然避免前视偏差。由于成分股列表存在 remark（JSONB），而 query_nearest_before 只
返回 value/name，故本模块用它先"定位"最近生效日，再补一条小查询取出 remark。

主要接口
--------
    get_index_universe      : 返回某指数某日生效的成分股代码列表
    get_index_universe_date : 返回某指数某日实际生效的快照日期（便于排查）
"""

import logging
from datetime import date as Date
from typing import Iterable, Optional, List

import pandas as pd

from .query import query_nearest_before
from .query import _normalized_schema_available


DEFAULT_TABLE = 'index_universe'
DEFAULT_METRIC = 'universe'


def _normalize_query_dates(dates: Iterable) -> list[pd.Timestamp]:
    out = []
    seen = set()
    for value in dates:
        day = pd.Timestamp(value).normalize()
        if day in seen:
            continue
        seen.add(day)
        out.append(day)
    return out


def get_index_universe_panel(
    cursor,
    index_code: str,
    dates: Iterable,
    table_name: str = DEFAULT_TABLE,
    metric: str = DEFAULT_METRIC,
    logger: Optional[logging.Logger] = None,
) -> dict[Date, set[str]]:
    """Return point-in-time constituents for many dates in one call."""
    if logger is None:
        logger = _get_default_logger()
    if not index_code:
        raise ValueError("index_code不能为空")
    query_dates = _normalize_query_dates(dates)
    if not query_dates:
        return {}

    panel = {day.date(): set() for day in query_dates}
    if (
        table_name == DEFAULT_TABLE
        and metric == DEFAULT_METRIC
        and _normalized_schema_available(cursor)
    ):
        cursor.execute(
            """
            WITH requested AS (
                SELECT query_date, input_ord
                FROM unnest(%s::timestamp[]) WITH ORDINALITY
                     AS d(query_date, input_ord)
            )
            SELECT r.query_date, member.code
            FROM requested r
            LEFT JOIN LATERAL (
                SELECT s.snapshot_id
                FROM betalens.index_snapshot s
                JOIN betalens.entity_dim idx ON idx.entity_id = s.index_entity_id
                WHERE idx.code = %s AND s.effective_at <= r.query_date
                ORDER BY s.effective_at DESC
                LIMIT 1
            ) latest ON TRUE
            LEFT JOIN betalens.index_constituent c
                   ON c.snapshot_id = latest.snapshot_id
            LEFT JOIN betalens.entity_dim member
                   ON member.entity_id = c.constituent_entity_id
            ORDER BY r.input_ord, COALESCE(c.ordinal, 2147483647), member.code
            """,
            ([day.to_pydatetime() for day in query_dates], index_code),
        )
        for row in cursor.fetchall():
            query_date = row.get("query_date") if isinstance(row, dict) else row[0]
            code = row.get("code") if isinstance(row, dict) else row[1]
            if code:
                panel[pd.Timestamp(query_date).date()].add(str(code))
        return panel

    for day in query_dates:
        panel[day.date()] = set(
            get_index_universe(
                cursor,
                index_code=index_code,
                date=day.strftime("%Y-%m-%d"),
                table_name=table_name,
                metric=metric,
                logger=logger,
            )
        )
    return panel


def _get_default_logger():
    logger = logging.getLogger('IndexUniverseQuery')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
    return logger


def get_index_universe_date(
    cursor,
    index_code: str,
    date: str,
    table_name: str = DEFAULT_TABLE,
    metric: str = DEFAULT_METRIC,
    logger: Optional[logging.Logger] = None,
):
    """
    返回某指数在某日实际生效的快照日期（point-in-time，取 datetime<=date 的最近一条）。

    复用 query.query_nearest_before 定位最近生效日。

    Args:
        cursor: 数据库游标（建议 RealDictCursor）
        index_code: 指数代码，如 '000906.SH'
        date: 查询日期，'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'
        table_name: 表名，默认 'index_universe'
        metric: 指标名，默认 'universe'
        logger: 日志器

    Returns:
        生效快照的 datetime（pandas.Timestamp）；该日前无可用股票池则返回 None
    """
    if logger is None:
        logger = _get_default_logger()
    if not index_code:
        raise ValueError("index_code不能为空")
    if not date:
        raise ValueError("date不能为空")

    if (
        table_name == DEFAULT_TABLE
        and metric == DEFAULT_METRIC
        and _normalized_schema_available(cursor)
    ):
        cursor.execute(
            """
            SELECT s.effective_at
            FROM betalens.index_snapshot s
            JOIN betalens.entity_dim e ON e.entity_id = s.index_entity_id
            WHERE e.code = %s AND s.effective_at <= %s::timestamp
            ORDER BY s.effective_at DESC
            LIMIT 1
            """,
            (index_code, date),
        )
        row = cursor.fetchone()
        if not row:
            return None
        value = row.get("effective_at") if isinstance(row, dict) else row[0]
        return pd.Timestamp(value)

    df = query_nearest_before(
        cursor,
        table_name=table_name,
        codes=[index_code],
        datetimes=[date],
        metric=metric,
        logger=logger,
    )

    if df.empty or 'datetime' not in df.columns:
        return None
    eff_dt = df.iloc[0]['datetime']
    # query_nearest_before 用 LEFT JOIN，无匹配时 datetime 为 NaT/None
    if eff_dt is None or (hasattr(eff_dt, '__class__') and str(eff_dt) == 'NaT'):
        return None
    if pd.isna(eff_dt):
        return None
    return eff_dt


def get_index_universe(
    cursor,
    index_code: str,
    date: str,
    table_name: str = DEFAULT_TABLE,
    metric: str = DEFAULT_METRIC,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    返回 index_code 在 date 当日生效的成分股代码列表（point-in-time）。

    步骤：用 query.query_nearest_before 找到 <=date 的最近生效快照日，再取该行
    remark 中的 constituents 列表。该日前无可用股票池则返回空列表。

    Args:
        cursor: 数据库游标（建议 RealDictCursor）
        index_code: 指数代码，如 '000906.SH'
        date: 查询日期，'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'
        table_name: 表名，默认 'index_universe'
        metric: 指标名，默认 'universe'
        logger: 日志器

    Returns:
        成分股代码列表（如 ['000001.SZ', ...]）；无可用股票池则返回 []
    """
    if logger is None:
        logger = _get_default_logger()
    if not index_code:
        raise ValueError("index_code不能为空")
    if not date:
        raise ValueError("date不能为空")

    if (
        table_name == DEFAULT_TABLE
        and metric == DEFAULT_METRIC
        and _normalized_schema_available(cursor)
    ):
        cursor.execute(
            """
            SELECT member.code
            FROM betalens.index_snapshot s
            JOIN betalens.entity_dim idx ON idx.entity_id = s.index_entity_id
            JOIN betalens.index_constituent c ON c.snapshot_id = s.snapshot_id
            JOIN betalens.entity_dim member ON member.entity_id = c.constituent_entity_id
            WHERE idx.code = %s AND s.effective_at = (
                SELECT MAX(s2.effective_at)
                FROM betalens.index_snapshot s2
                WHERE s2.index_entity_id = s.index_entity_id
                  AND s2.effective_at <= %s::timestamp
            )
            ORDER BY COALESCE(c.ordinal, 2147483647), member.code
            """,
            (index_code, date),
        )
        rows = cursor.fetchall()
        return [row.get("code") if isinstance(row, dict) else row[0] for row in rows]

    eff_dt = get_index_universe_date(
        cursor, index_code, date,
        table_name=table_name, metric=metric, logger=logger)

    if eff_dt is None:
        logger.info(f"{index_code} @ {date}: 无可用股票池")
        return []

    # 按精确 datetime 取该行 remark（query_nearest_before 不返回 remark）
    cursor.execute(
        f"SELECT remark FROM {table_name} "
        f"WHERE code = %s AND metric = %s AND datetime = %s",
        (index_code, metric, eff_dt)
    )
    row = cursor.fetchone()
    if row is None:
        logger.warning(f"{index_code} @ {eff_dt}: 定位到生效日但取 remark 为空")
        return []

    # 兼容 RealDictCursor(dict) 与普通 cursor(tuple)
    remark = row['remark'] if isinstance(row, dict) else row[0]
    if not isinstance(remark, dict):
        logger.warning(f"{index_code} @ {eff_dt}: remark 非预期结构")
        return []

    constituents = list(remark.get('constituents', []))
    logger.info(f"{index_code} @ {date}: 生效日 {eff_dt}, 成分股 {len(constituents)} 只")
    return constituents
