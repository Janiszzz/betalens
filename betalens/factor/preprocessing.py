#%%
"""
因子预处理模块

在 pre_query_characteristic_data() 之后、single_characteristic() 之前调用。
提供截面级别的去极值、标准化、中性化，以及一键预处理流水线。
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from datafeed.validation import fix_null_values, FillStrategy


# ─────────────────────────────────────────────
# 单截面处理函数（接收 Series，返回 Series）
# ─────────────────────────────────────────────

def winsorize_factor(
    factor_series: pd.Series,
    method: str = 'mad',
    n: float = 3.0,
) -> pd.Series:
    """
    截面去极值（单截面，index=code）。

    Args:
        factor_series: 单截面因子值，index=code
        method: 'mad'（中位数绝对偏差，推荐）| 'percentile'（百分位截尾）| 'std'（均值±n倍标准差）
        n: 阈值倍数（percentile 方法时为单侧截尾百分比，如 n=1 截 [1%, 99%]）

    Returns:
        去极值后的 Series，index 不变

    Example:
        >>> s = pd.Series({'A': 100, 'B': 2, 'C': 3, 'D': 1})
        >>> winsorize_factor(s, method='mad', n=3)
    """
    s = factor_series.copy()
    valid = s.dropna()
    if len(valid) == 0:
        return s

    if method == 'mad':
        med = np.median(valid)
        mad = np.median(np.abs(valid - med))
        lower = med - n * mad
        upper = med + n * mad
    elif method == 'percentile':
        lower = np.percentile(valid, n)
        upper = np.percentile(valid, 100 - n)
    elif method == 'std':
        lower = valid.mean() - n * valid.std()
        upper = valid.mean() + n * valid.std()
    else:
        raise ValueError(f"winsorize method 须为 'mad' / 'percentile' / 'std'，收到: {method}")

    return s.clip(lower=lower, upper=upper)


def standardize_factor(
    factor_series: pd.Series,
    method: str = 'zscore',
) -> pd.Series:
    """
    截面标准化（单截面，index=code）。

    Args:
        factor_series: 单截面因子值
        method: 'zscore'（(x-mean)/std）| 'rank'（rank/N，结果在(0,1)）| 'minmax'（缩放到[0,1]）

    Returns:
        标准化后的 Series

    Example:
        >>> standardize_factor(s, method='zscore')
    """
    s = factor_series.copy()
    valid = s.dropna()

    if method == 'zscore':
        std = valid.std()
        if std == 0:
            return s * 0.0
        return (s - valid.mean()) / std

    elif method == 'rank':
        n = len(valid)
        if n == 0:
            return s
        ranked = s.rank(method='average')
        return ranked / n

    elif method == 'minmax':
        vmin, vmax = valid.min(), valid.max()
        if vmax == vmin:
            return s * 0.0
        return (s - vmin) / (vmax - vmin)

    else:
        raise ValueError(f"standardize method 须为 'zscore' / 'rank' / 'minmax'，收到: {method}")


def neutralize_factor(
    factor_series: pd.Series,
    industry_labels: pd.Series = None,
    log_market_cap: pd.Series = None,
) -> pd.Series:
    """
    OLS 残差中性化（单截面）。

    对行业哑变量 + log(市值) 做截面 OLS，返回残差。
    参考 robust.py 中 neu() 的实现模式。

    Args:
        factor_series: 因子值 Series，index=code（已标准化）
        industry_labels: 行业标签 Series，index=code（传 None 则跳过）
        log_market_cap: log市值 Series，index=code（传 None 则跳过）

    Returns:
        残差 Series，index=code；无法计算的 code 填 NaN

    Example:
        >>> neutralize_factor(s, industry_labels=ind, log_market_cap=lmc)
    """
    if industry_labels is None and log_market_cap is None:
        return factor_series

    controls = pd.DataFrame(index=factor_series.index)

    if industry_labels is not None:
        dummies = pd.get_dummies(industry_labels, prefix='ind', drop_first=True)
        controls = pd.concat([controls, dummies], axis=1)

    if log_market_cap is not None:
        controls['log_mktcap'] = log_market_cap

    aligned = pd.concat([factor_series.rename('y'), controls], axis=1).dropna()
    if len(aligned) < controls.shape[1] + 5:
        return factor_series

    y = aligned['y']
    X = sm.add_constant(aligned.drop(columns='y').astype(float))
    model = sm.OLS(y, X).fit()

    result = pd.Series(np.nan, index=factor_series.index, name=factor_series.name)
    result.loc[y.index] = model.resid.values
    return result


# ─────────────────────────────────────────────
# 因子对因子中性化
# ─────────────────────────────────────────────

def neutralize_factor_by_factor(
    factor_b_data: pd.DataFrame,
    factor_a_data: pd.DataFrame,
    metric_b: str,
    metric_a: str,
) -> pd.DataFrame:
    """
    用因子A对因子B做截面OLS中性化，返回残差作为"剔除A影响后的B"。

    使用场景：
        - 检验规模因子(SIZE)对盈利因子(ROE)的解释程度，取残差得到"纯ROE"
        - 对双因子做正交化，使两因子线性无关

    Args:
        factor_b_data: 被解释因子的 pre_query_characteristic_data() 输出
                       （列含 input_ts, code, {metric_b}）
        factor_a_data: 解释因子的输出（列含 input_ts, code, {metric_a}）
        metric_b: 被解释因子列名（因子B）
        metric_a: 解释因子列名（因子A）

    Returns:
        同 factor_b_data 格式的 DataFrame，{metric_b} 列替换为残差值
        （无法匹配 A 的截面行被删除）

    Example:
        >>> # 用 SIZE 对 ROE 中性化，得到"剔除市值影响的ROE"
        >>> roe_pure = neutralize_factor_by_factor(roe_data, size_data, 'ROE', 'SIZE')
        >>> labeled = single_characteristic(roe_pure, 'ROE', quantiles={'ROE': 10})
    """
    groups = []

    # 以 factor_b 的截面日期为基准
    for ts, grp_b in factor_b_data.groupby('input_ts'):
        if ts not in factor_a_data['input_ts'].values:
            continue

        grp_a = factor_a_data[factor_a_data['input_ts'] == ts]
        b_series = grp_b.set_index('code')[metric_b].dropna()
        a_series = grp_a.set_index('code')[metric_a].dropna()

        common = b_series.index.intersection(a_series.index)
        if len(common) < 10:
            continue

        y = b_series.loc[common]
        X = sm.add_constant(a_series.loc[common].astype(float))
        try:
            model = sm.OLS(y.astype(float), X).fit()
        except Exception:
            continue

        resid = pd.Series(model.resid, index=common, name=metric_b)

        sub = grp_b.set_index('code').copy()
        sub[metric_b] = resid
        sub = sub.dropna(subset=[metric_b]).reset_index()
        groups.append(sub)

    if not groups:
        return factor_b_data.iloc[0:0].copy()
    return pd.concat(groups, ignore_index=True)


# ─────────────────────────────────────────────
# 行业中性化 — 选股池过滤 & 权重约束
# ─────────────────────────────────────────────

def filter_pool_by_industry(
    labeled_pool: pd.DataFrame,
    industry_map: pd.DataFrame,
    include_industries: list,
) -> pd.DataFrame:
    """
    将打标签的选股池限制在指定行业范围内。

    用途：在单个或多个行业的股票池中按因子选股，完全隔离其他行业。

    Args:
        labeled_pool: single_characteristic() 的输出，
                      MultiIndex(input_ts, code)
        industry_map: 行业映射表，列须含 input_ts, code, industry
                      （可复用 pre_queried_data 中的行业列，或单独查询）
        include_industries: 保留的行业列表，如 ['银行', '非银金融']
                            传 None 或 [] 则不过滤（返回原表）

    Returns:
        过滤后的 labeled_pool，MultiIndex 结构不变

    Example:
        >>> # 只在金融行业选股
        >>> filtered = filter_pool_by_industry(labeled_pool, ind_map, ['银行', '非银金融'])
        >>> weights = get_single_factor_weight(filtered, params)
    """
    if not include_industries:
        return labeled_pool

    # 构建 (input_ts, code) → industry 的索引
    ind_idx = industry_map.set_index(['input_ts', 'code'])['industry']
    mask = ind_idx.reindex(labeled_pool.index).isin(include_industries)
    return labeled_pool.loc[mask.fillna(False)]


def apply_industry_weight_constraint(
    weights: pd.DataFrame,
    industry_map: pd.DataFrame,
    method: str = 'equal',
    target_weights: Optional[dict] = None,
) -> pd.DataFrame:
    """
    对已生成的权重矩阵施加行业权重约束（行业中性化后处理）。

    三种模式：
      - 'equal'    : 全行业等权，多头各行业权重之和相等，空头同理
      - 'market'   : 按市场基准（target_weights 传入各行业目标比例之和=1）
      - 'original' : 不调整（直接返回，供流程统一调用）

    Args:
        weights: get_single_factor_weight() 的输出，
                 index=input_ts，columns=code，值为权重（多头>0，空头<0）
        industry_map: 行业映射表，列含 input_ts, code, industry
        method: 'equal' | 'market' | 'original'
        target_weights: method='market' 时使用，dict {industry: float}，
                        多头行业目标比例，如 {'银行': 0.3, '地产': 0.2, ...}
                        比例之和须 ≤ 1，其余行业平分剩余权重

    Returns:
        调整后的权重 DataFrame，long侧仍归一到1，short侧归一到-1

    Example:
        >>> # 按等权行业约束
        >>> w_neutral = apply_industry_weight_constraint(weights, ind_map, method='equal')

        >>> # 按自定义行业目标比例
        >>> w_custom = apply_industry_weight_constraint(
        ...     weights, ind_map, method='market',
        ...     target_weights={'银行': 0.4, '非银': 0.3, '地产': 0.3}
        ... )
    """
    if method == 'original':
        return weights

    # 行业映射：(input_ts, code) → industry
    ind_idx = industry_map.set_index(['input_ts', 'code'])['industry']

    result = weights.copy()

    for ts in result.index:
        row = result.loc[ts]
        long_codes = row[row > 0].index
        short_codes = row[row < 0].index

        # 取该期行业标签
        ts_key = pd.Timestamp(ts)
        ind_ts = ind_idx.xs(ts_key, level='input_ts') if ts_key in ind_idx.index.get_level_values('input_ts') else pd.Series(dtype=str)

        result.loc[ts] = _rescale_side_by_industry(
            row, long_codes, ind_ts, method, target_weights, sign=1
        ) + _rescale_side_by_industry(
            row, short_codes, ind_ts, method, target_weights, sign=-1
        )

    return result


def _rescale_side_by_industry(
    row: pd.Series,
    codes: pd.Index,
    ind_ts: pd.Series,
    method: str,
    target_weights: Optional[dict],
    sign: int,
) -> pd.Series:
    """
    对多头或空头一侧按行业重新分配权重。
    sign=1 → 多头（归一到 +1），sign=-1 → 空头（归一到 -1）。
    """
    out = pd.Series(0.0, index=row.index)
    if len(codes) == 0:
        return out

    # 构建 code → industry（缺失行业归入 '__other__'）
    ind_for_codes = ind_ts.reindex(codes).fillna('__other__')
    industries = ind_for_codes.unique().tolist()
    n_ind = len(industries)

    # 确定各行业目标比例（多头侧；空头侧使用同比例结构）
    if method == 'equal':
        ind_target = {ind: 1.0 / n_ind for ind in industries}
    elif method == 'market' and target_weights:
        specified = {ind: target_weights[ind] for ind in industries if ind in target_weights}
        remaining = 1.0 - sum(specified.values())
        unspecified = [ind for ind in industries if ind not in specified]
        share = remaining / len(unspecified) if unspecified else 0.0
        ind_target = {ind: specified.get(ind, share) for ind in industries}
        # 归一化
        total = sum(ind_target.values())
        ind_target = {k: v / total for k, v in ind_target.items()}
    else:
        # fallback：保持原始比例
        orig = row[codes].abs()
        total = orig.sum()
        if total > 0:
            out[codes] = (row[codes] / total) * sign
        return out

    # 在每个行业内按等权分配该行业的目标比例
    for ind, ind_share in ind_target.items():
        ind_codes = ind_for_codes[ind_for_codes == ind].index
        ind_codes_valid = ind_codes.intersection(codes)
        if len(ind_codes_valid) == 0:
            continue
        per_stock = ind_share / len(ind_codes_valid)
        out[ind_codes_valid] = per_stock * sign

    return out


# ─────────────────────────────────────────────
# 一键预处理流水线（处理整个 pre_queried_data）
# ─────────────────────────────────────────────

def preprocess_factor(
    pre_queried_data: pd.DataFrame,
    metric: str,
    winsorize_method: str = 'mad',
    winsorize_n: float = 3.0,
    standardize_method: str = 'zscore',
    industry_col: str = None,
    log_mktcap_col: str = None,
) -> pd.DataFrame:
    """
    逐截面（按 input_ts）依次执行：
      fix_null_values(drop) → winsorize_factor() → standardize_factor() → neutralize_factor()

    Args:
        pre_queried_data: pre_query_characteristic_data() 的输出，
                          列含 input_ts, code, {metric}, datetime, diff_hours
        metric: 因子列名
        winsorize_method: 'mad' | 'percentile' | 'std'
        winsorize_n: 去极值阈值
        standardize_method: 'zscore' | 'rank' | 'minmax'
        industry_col: pre_queried_data 中的行业列名（None 则跳过中性化该项）
        log_mktcap_col: pre_queried_data 中的 log市值列名（None 则跳过市值中性化）

    Returns:
        同 pre_queried_data 格式的 DataFrame，{metric} 列已替换为处理后的值

    Example:
        >>> cleaned = preprocess_factor(raw_data, 'ROE', industry_col='industry')
        >>> labeled = single_characteristic(cleaned, 'ROE', quantiles={'ROE': 10})
    """
    # 1. 删除因子列空值行
    data = fix_null_values(pre_queried_data, strategy=FillStrategy.DROP, columns=[metric])

    groups = []
    for ts, group in data.groupby('input_ts'):
        sub = group.copy()
        series = sub.set_index('code')[metric]

        # 2. 去极值
        series = winsorize_factor(series, method=winsorize_method, n=winsorize_n)

        # 3. 标准化
        series = standardize_factor(series, method=standardize_method)

        # 4. 中性化（可选）
        industry = sub.set_index('code')[industry_col] if industry_col and industry_col in sub.columns else None
        mktcap = sub.set_index('code')[log_mktcap_col] if log_mktcap_col and log_mktcap_col in sub.columns else None
        if industry is not None or mktcap is not None:
            series = neutralize_factor(series, industry_labels=industry, log_market_cap=mktcap)

        # 写回
        sub = sub.set_index('code')
        sub[metric] = series
        sub = sub.reset_index()
        groups.append(sub)

    if not groups:
        return data.iloc[0:0]
    return pd.concat(groups, ignore_index=True)
