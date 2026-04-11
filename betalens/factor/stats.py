#%%
"""
因子统计检验模块

提供 IC/ICIR 分析、Fama-MacBeth 截面回归、分组收益统计。
在 preprocess_factor() + single_characteristic() 之后调用。
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ─────────────────────────────────────────────
# IC / ICIR
# ─────────────────────────────────────────────

def calc_ic(
    factor_data: pd.DataFrame,
    return_data: pd.DataFrame,
    method: str = 'spearman',
) -> pd.Series:
    """
    逐截面计算 IC（Information Coefficient）。

    Args:
        factor_data: 宽表，index=input_ts，columns=code，值为因子值
        return_data: 宽表，index=input_ts，columns=code，值为持仓期收益率
        method: 'spearman'（Rank IC，推荐）| 'pearson'（普通 IC）

    Returns:
        Series，index=input_ts，name='IC'

    Example:
        >>> ic = calc_ic(factor_wide, return_wide)
        >>> print(ic.mean(), ic.std())
    """
    common_dates = factor_data.index.intersection(return_data.index)
    ic_values = {}

    for ts in common_dates:
        f = factor_data.loc[ts].dropna()
        r = return_data.loc[ts].dropna()
        common_codes = f.index.intersection(r.index)
        if len(common_codes) < 5:
            ic_values[ts] = np.nan
            continue

        f_c = f.loc[common_codes]
        r_c = r.loc[common_codes]

        if method == 'spearman':
            ic_values[ts] = f_c.rank().corr(r_c.rank())
        else:
            ic_values[ts] = f_c.corr(r_c)

    return pd.Series(ic_values, name='IC')


def calc_icir(
    ic_series: pd.Series,
    window: int = None,
):
    """
    计算 ICIR = mean(IC) / std(IC)。

    Args:
        ic_series: calc_ic() 的输出
        window: None 返回全样本 float；整数返回滚动 Series

    Returns:
        float（全样本）或 Series（滚动）

    Example:
        >>> icir = calc_icir(ic)           # 全样本
        >>> rolling_icir = calc_icir(ic, window=12)  # 滚动12期
    """
    if window is None:
        std = ic_series.std()
        return ic_series.mean() / std if std != 0 else np.nan
    else:
        return ic_series.rolling(window).mean() / ic_series.rolling(window).std()


def summarize_ic(ic_series: pd.Series) -> dict:
    """
    IC 统计摘要。

    Returns:
        dict: {
            'IC均值': float, 'IC_std': float, 'ICIR': float,
            '胜率(IC>0)': float, 't统计量': float, 'p值': float
        }

    Example:
        >>> summary = summarize_ic(ic)
        >>> pd.Series(summary)
    """
    valid = ic_series.dropna()
    n = len(valid)
    if n == 0:
        return {k: np.nan for k in ['IC均值', 'IC_std', 'ICIR', '胜率(IC>0)', 't统计量', 'p值']}

    ic_mean = valid.mean()
    ic_std = valid.std()
    icir = ic_mean / ic_std if ic_std != 0 else np.nan
    win_rate = (valid > 0).mean()

    if ic_std > 0:
        t_stat = ic_mean / (ic_std / np.sqrt(n))
        if _HAS_SCIPY:
            p_val = float(2 * (1 - _scipy_stats.t.cdf(abs(t_stat), df=n - 1)))
        else:
            # 正态近似（n 较大时可用）
            p_val = float(2 * (1 - _normal_cdf(abs(t_stat))))
    else:
        t_stat = np.nan
        p_val = np.nan

    return {
        'IC均值': ic_mean,
        'IC_std': ic_std,
        'ICIR': icir,
        '胜率(IC>0)': win_rate,
        't统计量': t_stat,
        'p值': p_val,
    }


def _normal_cdf(x):
    """标准正态 CDF，用于无 scipy 时近似 p 值。"""
    return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))


# ─────────────────────────────────────────────
# Fama-MacBeth 截面回归
# ─────────────────────────────────────────────

def fama_macbeth(
    factor_data_dict: dict,
    return_data: pd.DataFrame,
    industry_dummies: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Fama-MacBeth 两步法截面回归。

    第一步：每个截面期 t 做 OLS：
        R_i = α_t + Σ(λ_k,t * F_k,i) + ε_i
    第二步：对 λ_k 时间序列做 t 检验（Newey-West 可选）。

    Args:
        factor_data_dict: {因子名: 宽表 DataFrame (index=date, columns=code)}
        return_data: 宽表 (index=date, columns=code)，持仓期超额收益
        industry_dummies: 可选，宽表 (index=date, columns=行业哑变量名)
                          （用于控制行业效应，不纳入 lambda 报告）

    Returns:
        DataFrame，index=factor_name，columns=['lambda_mean','lambda_std','t_stat','p_value','n_periods']

    Example:
        >>> fm = fama_macbeth({'ROE': roe_wide, 'PE': pe_wide}, return_wide)
        >>> print(fm[['lambda_mean', 't_stat']])
    """
    factor_names = list(factor_data_dict.keys())
    lambda_records = {name: [] for name in factor_names}

    common_dates = return_data.index
    for f_df in factor_data_dict.values():
        common_dates = common_dates.intersection(f_df.index)

    for ts in common_dates:
        r = return_data.loc[ts].dropna()

        # 构建因子矩阵
        factor_cols = {}
        for name in factor_names:
            col = factor_data_dict[name].loc[ts].dropna()
            factor_cols[name] = col

        # 取所有因子与收益的公共 code
        valid_codes = r.index
        for col in factor_cols.values():
            valid_codes = valid_codes.intersection(col.index)

        if len(valid_codes) < len(factor_names) + 5:
            continue

        y_t = r.loc[valid_codes]
        X_t = pd.DataFrame({name: factor_cols[name].loc[valid_codes] for name in factor_names})

        # 加入行业哑变量（控制变量，不报告 lambda）
        if industry_dummies is not None and ts in industry_dummies.index:
            ind_t = industry_dummies.loc[ts].reindex(valid_codes).fillna(0)
            X_t = pd.concat([X_t, ind_t], axis=1)

        X_t = sm.add_constant(X_t.astype(float))
        try:
            model = sm.OLS(y_t.astype(float), X_t).fit()
        except Exception:
            continue

        for name in factor_names:
            if name in model.params.index:
                lambda_records[name].append(model.params[name])

    # 第二步：t 检验
    rows = []
    for name in factor_names:
        lambdas = pd.Series(lambda_records[name])
        n = len(lambdas)
        if n == 0:
            rows.append({'factor': name, 'lambda_mean': np.nan, 'lambda_std': np.nan,
                         't_stat': np.nan, 'p_value': np.nan, 'n_periods': 0})
            continue

        lmean = lambdas.mean()
        lstd = lambdas.std()
        if lstd > 0:
            t_stat = lmean / (lstd / np.sqrt(n))
            if _HAS_SCIPY:
                p_val = float(2 * (1 - _scipy_stats.t.cdf(abs(t_stat), df=n - 1)))
            else:
                p_val = float(2 * (1 - _normal_cdf(abs(t_stat))))
        else:
            t_stat = np.nan
            p_val = np.nan

        rows.append({
            'factor': name,
            'lambda_mean': lmean,
            'lambda_std': lstd,
            't_stat': t_stat,
            'p_value': p_val,
            'n_periods': n,
        })

    return pd.DataFrame(rows).set_index('factor')


# ─────────────────────────────────────────────
# 分组收益统计
# ─────────────────────────────────────────────

def group_return_summary(
    labeled_pool: pd.DataFrame,
    return_data: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    计算各分组在持仓期内的等权平均收益。

    Args:
        labeled_pool: single_characteristic() 输出，
                      MultiIndex(input_ts, code)，含 {metric}_label 列
        return_data: 宽表 (index=input_ts, columns=code)，持仓期收益率
        metric: 因子名（用于找 label 列：{metric}_label）

    Returns:
        DataFrame，index=input_ts，columns=['G1'...'GN', 'long_short']
        long_short = G_max - G_min（自动判断因子方向：若 G_max > G_min 则正向）

    Example:
        >>> gr = group_return_summary(labeled_pool, return_wide, 'ROE')
        >>> gr.cumsum().plot()
    """
    label_col = f"{metric}_label"
    n_groups = int(labeled_pool[label_col].max())

    dates = labeled_pool.index.get_level_values('input_ts').unique()
    common_dates = dates.intersection(return_data.index)

    records = {}
    for ts in common_dates:
        labels = labeled_pool.loc[ts][label_col]
        returns = return_data.loc[ts].dropna()
        row = {}
        for g in range(1, n_groups + 1):
            codes = labels[labels == g].index
            common = codes.intersection(returns.index)
            row[f'G{g}'] = returns.loc[common].mean() if len(common) > 0 else np.nan
        records[ts] = row

    result = pd.DataFrame(records).T.sort_index()

    # long_short：G_max - G1（自动判断方向）
    g1_mean = result['G1'].mean()
    gn_mean = result[f'G{n_groups}'].mean()
    if gn_mean >= g1_mean:
        result['long_short'] = result[f'G{n_groups}'] - result['G1']
    else:
        result['long_short'] = result['G1'] - result[f'G{n_groups}']

    return result
