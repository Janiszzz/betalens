#%%
"""
因子值分析（Profiling）模块

只看因子值本身的体检，不依赖未来收益（与 stats.py 的有效性检验严格分工）。
在 pre_query_characteristic_data() 之后、preprocess_factor() 之前/之后均可调用，
用于判断因子是否需要去极值、分布是否漂移、选股是否拥挤、多因子是否冗余。

数据流定位：
    pre_query → [profiling 体检] → preprocessing 清洗 → single_characteristic → stats 有效性

三大块：
    1. 分布与值域：describe_distribution / coverage_stats / detect_outliers
    2. 时变与稳定：factor_autocorrelation / factor_turnover / selection_overlap / distribution_stability
    3. 多因子交叉：cross_correlation / correlation_timeseries / selection_coincidence / factor_clustering

设计要点：
    - 长表/宽表双支持，公开函数自动判别（_to_wide / _to_long）
    - 截面优先：相关/自相关均"逐期算→时序平均"，避免把面板拉平导致的规模偏误
    - 可选依赖优雅降级：scipy / scikit-learn 缺失时回退
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from scipy.cluster import hierarchy as _scipy_hier
    from scipy.spatial.distance import squareform as _squareform
    _HAS_CLUSTER = True
except ImportError:
    _HAS_CLUSTER = False


# ─────────────────────────────────────────────
# 输入归一化工具：长表 ↔ 宽表
# ─────────────────────────────────────────────

def _to_wide(factor_data, metric=None):
    """
    归一化为宽表：index=input_ts，columns=code，值为因子值。

    自动判别输入：
      - 已是宽表（无 input_ts/code 列，index 为时间）：原样返回
      - 长表（含 input_ts, code 列）：用 metric 列 pivot
    """
    if isinstance(factor_data, pd.DataFrame) and \
       'input_ts' in factor_data.columns and 'code' in factor_data.columns:
        if metric is None:
            # 推断 metric：排除已知辅助列
            aux = {'input_ts', 'code', 'datetime', 'diff_hours', 'name'}
            cands = [c for c in factor_data.columns if c not in aux]
            if len(cands) != 1:
                raise ValueError(f"无法自动推断 metric，请显式指定。候选列: {cands}")
            metric = cands[0]
        wide = factor_data.pivot_table(index='input_ts', columns='code', values=metric)
        wide.index = pd.to_datetime(wide.index)
        return wide.sort_index()
    # 已是宽表
    wide = factor_data.copy()
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def _to_long(factor_data, metric='factor'):
    """归一化为长表：列 input_ts, code, {metric}。"""
    if isinstance(factor_data, pd.DataFrame) and \
       'input_ts' in factor_data.columns and 'code' in factor_data.columns:
        return factor_data.copy()
    wide = factor_data.copy()
    wide.index = pd.to_datetime(wide.index)
    long = wide.stack().reset_index()
    long.columns = ['input_ts', 'code', metric]
    return long


# ─────────────────────────────────────────────
# 1. 分布与值域
# ─────────────────────────────────────────────

def describe_distribution(factor_data, metric=None, by_period=True) -> pd.DataFrame:
    """
    因子值分布与值域统计（横截面体检）。

    Args:
        factor_data: 长表（input_ts/code/metric）或宽表（index=ts, col=code）
        metric: 长表的因子列名（宽表可不传）
        by_period: True 逐截面统计后给出跨期均值与全样本两行视角；
                   False 仅返回全样本汇总一行

    Returns:
        DataFrame，index 含 '全样本' 及（by_period 时）'逐期均值'，
        columns=[count, mean, std, min, 1%, 25%, 50%, 75%, 99%, max,
                 skew, kurt, 缺失率]

    Example:
        >>> describe_distribution(pre_queried_data, 'ROE')
    """
    wide = _to_wide(factor_data, metric)

    def _row(s: pd.Series) -> dict:
        v = s.dropna()
        n = len(v)
        return {
            'count': n,
            'mean': v.mean() if n else np.nan,
            'std': v.std() if n else np.nan,
            'min': v.min() if n else np.nan,
            '1%': v.quantile(0.01) if n else np.nan,
            '25%': v.quantile(0.25) if n else np.nan,
            '50%': v.quantile(0.50) if n else np.nan,
            '75%': v.quantile(0.75) if n else np.nan,
            '99%': v.quantile(0.99) if n else np.nan,
            'max': v.max() if n else np.nan,
            'skew': v.skew() if n > 2 else np.nan,
            'kurt': v.kurt() if n > 3 else np.nan,
            '缺失率': s.isna().mean(),
        }

    # 全样本：所有 (期,股票) 拉平（含 NaN 以正确计算缺失率）
    flat = wide.to_numpy().ravel()
    flat = pd.Series(flat)
    rows = {'全样本': _row(flat)}

    if by_period:
        per = wide.apply(lambda r: pd.Series(_row(r)), axis=1)
        rows['逐期均值'] = per.mean().to_dict()

    return pd.DataFrame(rows).T


def coverage_stats(factor_data, metric=None) -> pd.DataFrame:
    """
    因子覆盖度时变统计。

    应对股票池随时间扩张：每期有效（非缺失）股票数与覆盖率，
    用于发现某些时段因子大面积缺失（如早年财报字段未披露）。

    Args:
        factor_data: 长表或宽表
        metric: 长表因子列名

    Returns:
        DataFrame，index=input_ts，columns=[有效数, 总数, 覆盖率]

    Example:
        >>> cov = coverage_stats(pre_queried_data, 'ROE')
        >>> cov['覆盖率'].plot()
    """
    wide = _to_wide(factor_data, metric)
    valid = wide.notna().sum(axis=1)
    total = wide.shape[1]
    return pd.DataFrame({
        '有效数': valid,
        '总数': total,
        '覆盖率': valid / total if total else np.nan,
    })


def detect_outliers(factor_data, metric=None, method='mad', n=3.0) -> pd.DataFrame:
    """
    逐期极值占比检测，提示是否需要 winsorize。

    Args:
        factor_data: 长表或宽表
        metric: 长表因子列名
        method: 'mad'（中位数绝对偏差）| 'std'（均值±n倍标准差）
        n: 阈值倍数

    Returns:
        DataFrame，index=input_ts，columns=[下界, 上界, 极值数, 极值占比]
        末行 'Total' 为全样本汇总

    Example:
        >>> detect_outliers(pre_queried_data, 'ROE', method='mad', n=3)
    """
    wide = _to_wide(factor_data, metric)

    def _bounds(v: pd.Series):
        if method == 'mad':
            med = v.median()
            mad = (v - med).abs().median()
            return med - n * mad, med + n * mad
        elif method == 'std':
            mu, sd = v.mean(), v.std()
            return mu - n * sd, mu + n * sd
        raise ValueError(f"method 须为 'mad' / 'std'，收到: {method}")

    records = {}
    for ts, row in wide.iterrows():
        v = row.dropna()
        if len(v) == 0:
            continue
        lo, hi = _bounds(v)
        n_out = ((v < lo) | (v > hi)).sum()
        records[ts] = {'下界': lo, '上界': hi,
                       '极值数': int(n_out), '极值占比': n_out / len(v)}

    result = pd.DataFrame(records).T
    flat = wide.stack().dropna()
    if len(flat):
        lo, hi = _bounds(flat)
        n_out = ((flat < lo) | (flat > hi)).sum()
        result.loc['Total'] = {'下界': lo, '上界': hi,
                               '极值数': int(n_out), '极值占比': n_out / len(flat)}
    return result


# ─────────────────────────────────────────────
# 2. 时变特征与稳定性
# ─────────────────────────────────────────────

def factor_autocorrelation(factor_data, metric=None, lags=None, method='spearman') -> pd.DataFrame:
    """
    因子记忆性：截面 rank 自相关（corr(值_t, 值_{t-lag}) 逐期算后时序平均）。

    高自相关 → 因子缓变、换手低；低自相关 → 因子快变、换手高、交易成本敏感。

    Args:
        factor_data: 长表或宽表
        metric: 长表因子列名
        lags: 滞后期列表，默认 [1, 3, 6, 12]
        method: 'spearman'（rank，推荐）| 'pearson'

    Returns:
        DataFrame，index=lag，columns=[自相关均值, 自相关std, 有效期数]

    Example:
        >>> factor_autocorrelation(pre_queried_data, 'ROE', lags=[1,3,6])
    """
    if lags is None:
        lags = [1, 3, 6, 12]
    wide = _to_wide(factor_data, metric)
    n_periods = len(wide.index)

    rows = []
    for lag in lags:
        corrs = []
        for i in range(lag, n_periods):
            cur = wide.iloc[i]
            prev = wide.iloc[i - lag]
            pair = pd.DataFrame({'cur': cur, 'prev': prev}).dropna()
            if len(pair) < 5:
                continue
            if method == 'spearman':
                c = pair['cur'].rank().corr(pair['prev'].rank())
            else:
                c = pair['cur'].corr(pair['prev'])
            if not np.isnan(c):
                corrs.append(c)
        s = pd.Series(corrs)
        rows.append({'lag': lag, '自相关均值': s.mean() if len(s) else np.nan,
                     '自相关std': s.std() if len(s) else np.nan, '有效期数': len(s)})
    return pd.DataFrame(rows).set_index('lag')


def factor_turnover(factor_data, metric=None, quantile=0.2, side='top') -> pd.Series:
    """
    头部组成分换手率：相邻期 top/bottom 分位组的成分变动比例。

    turnover_t = 1 - |持仓_t ∩ 持仓_{t-1}| / |持仓_t|

    Args:
        factor_data: 长表或宽表
        metric: 长表因子列名
        quantile: 分位阈值（0.2 表示 top/bottom 20%）
        side: 'top'（因子值最高组）| 'bottom'（最低组）

    Returns:
        Series，index=input_ts，name='turnover'（首期为 NaN）

    Example:
        >>> to = factor_turnover(pre_queried_data, 'ROE', quantile=0.2)
        >>> to.mean()  # 平均换手率
    """
    wide = _to_wide(factor_data, metric)

    def _members(row: pd.Series) -> set:
        v = row.dropna()
        if len(v) < 5:
            return set()
        if side == 'top':
            thr = v.quantile(1 - quantile)
            return set(v[v >= thr].index)
        else:
            thr = v.quantile(quantile)
            return set(v[v <= thr].index)

    dates = wide.index
    out = {dates[0]: np.nan} if len(dates) else {}
    prev = _members(wide.iloc[0]) if len(dates) else set()
    for i in range(1, len(dates)):
        cur = _members(wide.iloc[i])
        if len(cur) == 0:
            out[dates[i]] = np.nan
        else:
            out[dates[i]] = 1 - len(cur & prev) / len(cur)
        prev = cur
    return pd.Series(out, name='turnover')


def selection_overlap(factor_data, metric=None, quantile=0.2, side='top') -> pd.DataFrame:
    """
    相邻期选股重合度（Jaccard）：|A∩B| / |A∪B|。

    与 factor_turnover 互补：turnover 看流出，Jaccard 看整体相似度。

    Args:
        factor_data: 长表或宽表
        metric: 长表因子列名
        quantile: 分位阈值
        side: 'top' | 'bottom'

    Returns:
        DataFrame，index=input_ts，columns=[Jaccard, 持仓数]（首期 NaN）

    Example:
        >>> ov = selection_overlap(pre_queried_data, 'ROE')
        >>> ov['Jaccard'].mean()
    """
    wide = _to_wide(factor_data, metric)

    def _members(row: pd.Series) -> set:
        v = row.dropna()
        if len(v) < 5:
            return set()
        thr = v.quantile(1 - quantile) if side == 'top' else v.quantile(quantile)
        return set(v[v >= thr].index) if side == 'top' else set(v[v <= thr].index)

    dates = wide.index
    records = {}
    prev = _members(wide.iloc[0]) if len(dates) else set()
    if len(dates):
        records[dates[0]] = {'Jaccard': np.nan, '持仓数': len(prev)}
    for i in range(1, len(dates)):
        cur = _members(wide.iloc[i])
        union = cur | prev
        jac = len(cur & prev) / len(union) if union else np.nan
        records[dates[i]] = {'Jaccard': jac, '持仓数': len(cur)}
        prev = cur
    return pd.DataFrame(records).T


def distribution_stability(factor_data, metric=None) -> pd.DataFrame:
    """
    分布漂移监测：逐期均值/标准差/偏度/峰度时序，判断因子分布是否随时间漂移。

    Args:
        factor_data: 长表或宽表
        metric: 长表因子列名

    Returns:
        DataFrame，index=input_ts，columns=[mean, std, skew, kurt, 有效数]

    Example:
        >>> ds = distribution_stability(pre_queried_data, 'ROE')
        >>> ds[['mean', 'std']].plot()
    """
    wide = _to_wide(factor_data, metric)
    records = {}
    for ts, row in wide.iterrows():
        v = row.dropna()
        records[ts] = {
            'mean': v.mean() if len(v) else np.nan,
            'std': v.std() if len(v) else np.nan,
            'skew': v.skew() if len(v) > 2 else np.nan,
            'kurt': v.kurt() if len(v) > 3 else np.nan,
            '有效数': len(v),
        }
    return pd.DataFrame(records).T


# ─────────────────────────────────────────────
# 3. 多因子交叉分析
# ─────────────────────────────────────────────

def _align_factor_dict(factor_dict: dict) -> dict:
    """把 {因子名: 长表/宽表} 统一转成 {因子名: 宽表}。"""
    return {name: _to_wide(data) for name, data in factor_dict.items()}


def cross_correlation(factor_dict: dict, method='spearman') -> pd.DataFrame:
    """
    多因子平均截面相关矩阵（逐期算 corr 再时序平均，避免规模偏误）。

    Args:
        factor_dict: {因子名: 长表/宽表}
        method: 'spearman'（rank，推荐）| 'pearson'

    Returns:
        DataFrame，N×N 对称相关矩阵，index/columns=因子名

    Example:
        >>> cross_correlation({'ROE': roe, 'PE': pe, 'SIZE': size})
    """
    wides = _align_factor_dict(factor_dict)
    names = list(wides.keys())
    n = len(names)

    # 公共截面日期
    common_dates = None
    for w in wides.values():
        common_dates = w.index if common_dates is None else common_dates.intersection(w.index)

    mat = pd.DataFrame(np.eye(n), index=names, columns=names)
    for a in range(n):
        for b in range(a + 1, n):
            wa, wb = wides[names[a]], wides[names[b]]
            corrs = []
            for ts in common_dates:
                pair = pd.DataFrame({'a': wa.loc[ts], 'b': wb.loc[ts]}).dropna()
                if len(pair) < 5:
                    continue
                if method == 'spearman':
                    c = pair['a'].rank().corr(pair['b'].rank())
                else:
                    c = pair['a'].corr(pair['b'])
                if not np.isnan(c):
                    corrs.append(c)
            avg = float(np.mean(corrs)) if corrs else np.nan
            mat.iloc[a, b] = mat.iloc[b, a] = avg
    return mat


def correlation_timeseries(factor_dict: dict, pair: tuple, method='spearman') -> pd.Series:
    """
    指定因子对的截面相关系数时序，观察相关性是否稳定（突变=轮动/风格切换信号）。

    Args:
        factor_dict: {因子名: 长表/宽表}
        pair: 因子名二元组，如 ('ROE', 'SIZE')
        method: 'spearman' | 'pearson'

    Returns:
        Series，index=input_ts，name='corr(A,B)'

    Example:
        >>> correlation_timeseries({'ROE': roe, 'SIZE': size}, ('ROE', 'SIZE'))
    """
    wides = _align_factor_dict(factor_dict)
    wa, wb = wides[pair[0]], wides[pair[1]]
    common = wa.index.intersection(wb.index)
    out = {}
    for ts in common:
        p = pd.DataFrame({'a': wa.loc[ts], 'b': wb.loc[ts]}).dropna()
        if len(p) < 5:
            continue
        if method == 'spearman':
            out[ts] = p['a'].rank().corr(p['b'].rank())
        else:
            out[ts] = p['a'].corr(p['b'])
    return pd.Series(out, name=f'corr({pair[0]},{pair[1]})')


def selection_coincidence(factor_dict: dict, quantile=0.2, side='top') -> pd.DataFrame:
    """
    两两因子选股重合度矩阵：各因子 top 组的平均 Jaccard 相似度（逐期算后平均）。

    高重合 → 两因子选出的股票高度雷同，组合层面冗余（即便值相关性中等）。

    Args:
        factor_dict: {因子名: 长表/宽表}
        quantile: 分位阈值
        side: 'top' | 'bottom'

    Returns:
        DataFrame，N×N 对称重合度矩阵（对角线为 1）

    Example:
        >>> selection_coincidence({'ROE': roe, 'PE': pe}, quantile=0.2)
    """
    wides = _align_factor_dict(factor_dict)
    names = list(wides.keys())
    n = len(names)

    common_dates = None
    for w in wides.values():
        common_dates = w.index if common_dates is None else common_dates.intersection(w.index)

    def _members(row: pd.Series) -> set:
        v = row.dropna()
        if len(v) < 5:
            return set()
        thr = v.quantile(1 - quantile) if side == 'top' else v.quantile(quantile)
        return set(v[v >= thr].index) if side == 'top' else set(v[v <= thr].index)

    mat = pd.DataFrame(np.eye(n), index=names, columns=names)
    for a in range(n):
        for b in range(a + 1, n):
            wa, wb = wides[names[a]], wides[names[b]]
            jacs = []
            for ts in common_dates:
                ma, mb = _members(wa.loc[ts]), _members(wb.loc[ts])
                union = ma | mb
                if union:
                    jacs.append(len(ma & mb) / len(union))
            avg = float(np.mean(jacs)) if jacs else np.nan
            mat.iloc[a, b] = mat.iloc[b, a] = avg
    return mat


def factor_clustering(corr_matrix: pd.DataFrame, threshold=0.6) -> dict:
    """
    基于相关矩阵的层次聚类，提示冗余因子组。

    距离 = 1 - |corr|，相关性高的因子聚为一类，超阈值即视为冗余候选。
    无 scipy 时退化为简单的并查集分组（按 |corr| ≥ threshold 连边）。

    Args:
        corr_matrix: cross_correlation 的输出（N×N）
        threshold: 归为同组的相关性阈值（|corr| ≥ threshold）

    Returns:
        dict: {
            'clusters': [[因子名,...], ...],  # 每个子列表为一个冗余组
            'n_clusters': int,
            'method': 'hierarchical' | 'union_find',
        }

    Example:
        >>> cm = cross_correlation(factor_dict)
        >>> factor_clustering(cm, threshold=0.6)
    """
    names = list(corr_matrix.index)
    n = len(names)
    abs_corr = corr_matrix.abs().fillna(0).values

    if _HAS_CLUSTER and n >= 2:
        dist = 1 - abs_corr
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2  # 强制对称
        condensed = _squareform(dist, checks=False)
        Z = _scipy_hier.linkage(condensed, method='average')
        labels = _scipy_hier.fcluster(Z, t=1 - threshold, criterion='distance')
        method = 'hierarchical'
    else:
        # 并查集回退
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if abs_corr[i, j] >= threshold:
                    parent[find(i)] = find(j)
        labels = np.array([find(i) for i in range(n)])
        method = 'union_find'

    clusters = {}
    for name, lab in zip(names, labels):
        clusters.setdefault(int(lab), []).append(name)
    cluster_list = list(clusters.values())

    return {
        'clusters': cluster_list,
        'n_clusters': len(cluster_list),
        'method': method,
    }


# PLACEHOLDER_PLOTS


