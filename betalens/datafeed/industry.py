#%%By Janis 260602
"""
行业归属查询模块（函数式）

设计要点
--------
行业归属是 point-in-time 的"状态"数据：某公司从某日起属于某行业，直到下次变更。
因此不另造存储模型，而是复用现有长格式时序表，约定：
    - metric : 分类体系名，如 '申万一级行业' / '申万二级行业' / '中信一级行业'
    - value  : 行业代码的数值部分（如 801780），便于数值索引与分组
    - remark : JSONB，存行业名等文本，约定 {"ind_name", "ind_code", "scheme", "level"}
    - datetime: 该归属关系的生效时点（最早可知日）

查询语义 = 取 datetime <= 查询日 的最近一条，与 query.query_nearest_before 同构，
天然避免前视偏差。本模块额外把 remark(JSONB) 解析出来返回行业名。

主要接口
--------
    query_industry        : 正查——某公司在某日所属行业
    get_industry_members  : 反查——某日某行业的成分股
    build_industry_records: 入库辅助——把 (code,name,生效日,行业) 整理成长格式
"""

import itertools
import logging
from typing import Optional, List, Tuple, Union

import pandas as pd


DEFAULT_TABLE = 'industry'


def _get_default_logger():
    logger = logging.getLogger('IndustryQuery')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
    return logger


def _explode_remark(df: pd.DataFrame) -> pd.DataFrame:
    """把 remark(JSONB->dict) 展开成 ind_name / ind_code / scheme 列"""
    if df.empty or 'remark' not in df.columns:
        for c in ('ind_name', 'ind_code', 'scheme'):
            df[c] = None
        return df

    def _get(r, k):
        return r.get(k) if isinstance(r, dict) else None

    df['ind_name'] = df['remark'].apply(lambda r: _get(r, 'ind_name'))
    df['ind_code'] = df['remark'].apply(lambda r: _get(r, 'ind_code'))
    df['scheme'] = df['remark'].apply(lambda r: _get(r, 'scheme'))
    return df


def _scheme_clause(scheme: str, exact: bool, col: str = 't.metric') -> Tuple[str, str]:
    """生成 metric 匹配子句与参数。

    版本无关查询：scheme 不带版本后缀（如 '申万一级行业'）时用前缀匹配，
    覆盖 '申万一级行业（旧版/2014/2021）' 等全部版本；配合 ORDER BY datetime DESC，
    取 datetime<=查询日 的最近一条 → 自动落到查询日生效的那个版本，无需硬编码版本边界。

    带版本后缀（如 '申万一级行业（2021）'）时前缀匹配退化为精确，只命中该版本。
    exact=True 则强制精确匹配（旧行为）。

    Args:
        scheme: 分类体系名
        exact: True 强制精确匹配
        col: metric 列引用（带表别名前缀，如 't.metric' 或 'metric'）

    Returns:
        (SQL 片段, 参数值)；SQL 片段形如 '{col} = %s' 或 '{col} LIKE %s ESCAPE ...'
    """
    if exact:
        return f'{col} = %s', scheme
    # 转义 LIKE 通配符（中文 metric 名一般不含，但稳妥起见）
    esc = scheme.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    return f"{col} LIKE %s ESCAPE '\\'", esc + '%'


def query_industry(
    cursor,
    codes: List[str],
    dates: Union[str, List[str]],
    scheme: str = '申万一级行业',
    table_name: str = DEFAULT_TABLE,
    exact: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    正查：每个 (code, date) 在该日所属的行业（point-in-time，取 datetime<=date 的最近一条）

    Args:
        cursor: 数据库游标（建议 RealDictCursor）
        codes: 证券代码列表
        dates: 查询日期，单个或列表，格式 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'
        scheme: 分类体系（即 metric）。不带版本后缀（如 '申万一级行业'）时自动匹配全部
                版本，最近一条天然落到查询日生效的版本；带后缀（如 '申万一级行业（2021）'）
                则只查该版本。
        table_name: 表名，默认 'industry'
        exact: 强制精确匹配 metric（关闭版本自动选择），默认 False
        logger: 日志器

    Returns:
        DataFrame: code | query_date | effective_dt | sec_name |
                   industry_value | ind_name | ind_code | scheme
        无归属记录的 (code,date) 行业字段为 NaN/None
    """
    if logger is None:
        logger = _get_default_logger()
    if not codes:
        raise ValueError("codes不能为空")
    if isinstance(dates, str):
        dates = [dates]
    if not dates:
        raise ValueError("dates不能为空")

    pairs = list(itertools.product(codes, dates))
    value_ph = ', '.join(['(%s, %s::TIMESTAMP)'] * len(pairs))
    metric_clause, metric_param = _scheme_clause(scheme, exact, col='t.metric')

    sql = f"""
    WITH input_data (code, q_date) AS (
        VALUES {value_ph}
    ),
    cand AS (
        SELECT
            i.code,
            i.q_date,
            t.datetime AS effective_dt,
            t.name     AS sec_name,
            t.value    AS industry_value,
            t.remark   AS remark,
            ROW_NUMBER() OVER (
                PARTITION BY i.code, i.q_date
                ORDER BY t.datetime DESC
            ) AS rn
        FROM input_data i
        LEFT JOIN {table_name} t
            ON i.code = t.code
            AND {metric_clause}
            AND t.datetime <= i.q_date
    )
    SELECT code, q_date AS query_date, effective_dt, sec_name,
           industry_value, remark
    FROM cand
    WHERE rn = 1
    """

    params: List = []
    for code, dt in pairs:
        params.extend([code, dt])
    params.append(metric_param)

    logger.info(f"query_industry: {len(codes)}代码 × {len(dates)}日期, scheme={scheme}")
    cursor.execute(sql, params)
    df = pd.DataFrame(cursor.fetchall())
    df = _explode_remark(df)
    logger.info(f"返回 {len(df)} 条")
    return df


def get_industry_members(
    cursor,
    industry: Union[str, int, float],
    date: str,
    scheme: str = '申万一级行业',
    table_name: str = DEFAULT_TABLE,
    by: str = 'name',
    exact: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    反查：某日某行业的成分股（每只股票取 datetime<=date 的最近归属，再筛目标行业）

    Args:
        cursor: 数据库游标
        industry: 目标行业，可为行业名(str，匹配 remark->>'ind_name')
                  或行业代码数值(int/float，匹配 value)
        date: 查询日期
        scheme: 分类体系（metric）。不带版本后缀时自动匹配全部版本（最近一条天然落到
                查询日生效的版本）；带后缀只查该版本。
        table_name: 表名
        by: 'name' 用行业名匹配，'value' 用行业代码数值匹配；
            industry 类型也会自动推断
        exact: 强制精确匹配 metric（关闭版本自动选择），默认 False
        logger: 日志器

    Returns:
        DataFrame: code | sec_name | industry_value | ind_name | ind_code | scheme
    """
    if logger is None:
        logger = _get_default_logger()

    use_value = (by == 'value') or isinstance(industry, (int, float))

    if use_value:
        match_cond = "WHERE rn = 1 AND industry_value = %s"
        match_param: Union[str, int, float] = industry
    else:
        match_cond = "WHERE rn = 1 AND (remark->>'ind_name') = %s"
        match_param = str(industry)

    metric_clause, metric_param = _scheme_clause(scheme, exact, col='metric')

    sql = f"""
    WITH latest AS (
        SELECT
            code,
            name AS sec_name,
            value AS industry_value,
            remark,
            datetime,
            ROW_NUMBER() OVER (
                PARTITION BY code ORDER BY datetime DESC
            ) AS rn
        FROM {table_name}
        WHERE {metric_clause} AND datetime <= %s::TIMESTAMP
    )
    SELECT code, sec_name, industry_value, remark
    FROM latest
    {match_cond}
    ORDER BY code
    """

    params = [metric_param, date, match_param]
    logger.info(f"get_industry_members: {scheme}={industry} @ {date}")
    cursor.execute(sql, params)
    df = pd.DataFrame(cursor.fetchall())
    df = _explode_remark(df)
    logger.info(f"成分股 {len(df)} 只")
    return df


def build_industry_records(
    df: pd.DataFrame,
    scheme: str = '申万一级行业',
    code_col: str = 'code',
    name_col: str = 'name',
    date_col: str = 'effective_dt',
    ind_name_col: str = 'ind_name',
    ind_code_col: Optional[str] = 'ind_code',
) -> pd.DataFrame:
    """
    入库辅助：把行业归属明细整理成可直接 incremental_insert 的长格式

    输入每行 = 一条归属事件 (证券, 生效日, 行业)。输出列：
        datetime, code, name, metric(=scheme), value(=行业代码数值), remark(dict)

    Args:
        df: 明细 DataFrame
        scheme: 分类体系，写入 metric
        code_col/name_col/date_col: 证券代码/名称/生效日 列名
        ind_name_col: 行业名列名
        ind_code_col: 行业代码列名（如 '801780.SI'）；为 None 则不填 value

    Returns:
        长格式 DataFrame（datetime, code, name, metric, value, remark）
    """
    import re

    out = pd.DataFrame()
    out['datetime'] = pd.to_datetime(df[date_col])
    out['code'] = df[code_col].astype(str)
    out['name'] = df[name_col].astype(str)
    out['metric'] = scheme

    def _to_num(c):
        if c is None or (isinstance(c, float) and pd.isna(c)):
            return None
        m = re.search(r'\d+', str(c))
        return int(m.group()) if m else None

    if ind_code_col and ind_code_col in df.columns:
        out['value'] = df[ind_code_col].apply(_to_num)
        ind_codes = df[ind_code_col]
    else:
        out['value'] = None
        ind_codes = pd.Series([None] * len(df), index=df.index)

    out['remark'] = [
        {
            'ind_name': (None if pd.isna(n) else str(n)),
            'ind_code': (None if pd.isna(c) else str(c)),
            'scheme': scheme,
        }
        for n, c in zip(df[ind_name_col], ind_codes)
    ]
    return out
