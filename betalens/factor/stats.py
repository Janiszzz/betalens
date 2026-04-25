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
    """ 
    # long_short：G_max - G1（自动判断方向）
    g1_mean = result['G1'].mean()
    gn_mean = result[f'G{n_groups}'].mean()
    if gn_mean >= g1_mean:
        result['long_short'] = result[f'G{n_groups}'] - result['G1']
    else:
        result['long_short'] = result['G1'] - result[f'G{n_groups}']

    """

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 择时因子评价
# ═════════════════════════════════════════════════════════════════════════════

def calc_timing_ic(
    factor: pd.Series,
    returns: pd.Series,
    periods: list = None,
    method: str = 'spearman',
    rolling_window: int = 60,
) -> dict:
    """
    择时因子 IC 计算：滚动窗口内 factor 与 N 日前瞻收益的相关系数。

    Args:
        factor: 因子时间序列，index=datetime
        returns: 收益率时间序列，index=datetime
        periods: 预测周期列表，如 [5, 10, 20]
        method: 'spearman' | 'pearson'
        rolling_window: 滚动窗口大小

    Returns:
        {period: {'ic_series': Series, 'stats': summarize_ic 结果}}

    Example:
        >>> result = calc_timing_ic(factor, returns, periods=[5, 10, 20])
        >>> result[5]['stats']['IC均值']
    """
    if periods is None:
        periods = [5, 10, 20]

    factor = factor.dropna()
    returns = returns.dropna()
    common_idx = factor.index.intersection(returns.index)
    factor = factor.loc[common_idx]
    returns = returns.loc[common_idx]

    results = {}
    for period in periods:
        fwd_ret = returns.rolling(period).sum().shift(-period)
        aligned = pd.DataFrame({'factor': factor, 'fwd_ret': fwd_ret}).dropna()

        if len(aligned) < rolling_window:
            results[period] = {
                'ic_series': pd.Series(dtype=float),
                'stats': summarize_ic(pd.Series(dtype=float)),
            }
            continue

        if method == 'spearman':
            ic_series = aligned['factor'].rolling(rolling_window).corr(
                aligned['fwd_ret']
            )
            # spearman 需要用 rank
            ic_list = []
            for i in range(rolling_window - 1, len(aligned)):
                window_data = aligned.iloc[i - rolling_window + 1:i + 1]
                corr = window_data['factor'].rank().corr(window_data['fwd_ret'].rank())
                ic_list.append((aligned.index[i], corr))
            ic_series = pd.Series(
                dict(ic_list), name=f'IC_{period}d'
            )
        else:
            ic_series = aligned['factor'].rolling(rolling_window).corr(
                aligned['fwd_ret']
            )
            ic_series.name = f'IC_{period}d'

        ic_series = ic_series.dropna()
        results[period] = {
            'ic_series': ic_series,
            'stats': summarize_ic(ic_series),
        }

    return results


def generate_timing_signals(
    factor: pd.Series,
    sigma: float = 1.0,
    ma_window: int = 250,
    extreme_quantile: float = 0.1,
) -> dict:
    """
    生成 7 种择时信号。

    Args:
        factor: 标准化后的因子序列
        sigma: 阈值法的 σ 倍数
        ma_window: 均线法窗口
        extreme_quantile: 极值法分位数

    Returns:
        {方法名: signal_series}，signal ∈ {1, 0, -1}

    Example:
        >>> signals = generate_timing_signals(factor, sigma=1.0)
        >>> signals['阈值法'].value_counts()
    """
    f = factor.dropna()
    mean, std = f.mean(), f.std()
    ma = f.rolling(ma_window, min_periods=1).mean()
    f_diff = f.diff()
    ma_diff = ma.diff()
    ma250_diff = f.rolling(250, min_periods=1).mean().diff()

    upper = mean + sigma * std
    lower = mean - sigma * std
    q_high = f.quantile(1 - extreme_quantile)
    q_low = f.quantile(extreme_quantile)

    def _to_signal(long_mask, short_mask):
        s = pd.Series(0, index=f.index)
        s[long_mask] = 1
        s[short_mask] = -1
        return s

    signals = {
        '阈值法': _to_signal(f > upper, f < lower),
        '均线法': _to_signal(f > ma, f < ma),
        '极值法': _to_signal(f > q_high, f < q_low),
        'zero': _to_signal(f > 0, f < 0),
        'diff_zero': _to_signal(f_diff > 0, f_diff < 0),
        'MA250_diff_zero': _to_signal(ma250_diff > 0, ma250_diff < 0),
        'ma_diff_zero': _to_signal(ma_diff > 0, ma_diff < 0),
    }
    return signals


def test_timing_signal(
    signal: pd.Series,
    returns: pd.Series,
    period: int = 5,
) -> dict:
    """
    单个择时信号的绩效检验。

    Returns:
        dict 含多头/空头样本数、胜率、均收益、盈亏比、综合胜率、t 统计量、p 值

    Example:
        >>> stats = test_timing_signal(signal, returns, period=5)
        >>> stats['综合胜率']
    """
    fwd_ret = returns.rolling(period).sum().shift(-period)
    aligned = pd.DataFrame({
        'signal': signal, 'fwd_ret': fwd_ret
    }).dropna()

    long_rets = aligned.loc[aligned['signal'] == 1, 'fwd_ret']
    short_rets = aligned.loc[aligned['signal'] == -1, 'fwd_ret']

    def _stats(rets, side):
        n = len(rets)
        if n == 0:
            return {f'{side}样本数': 0, f'{side}胜率': 0.0,
                    f'{side}均收益': 0.0, f'{side}盈亏比': 0.0}
        wins = (rets > 0).sum() if side == '多头' else (rets < 0).sum()
        win_rate = wins / n
        avg_ret = rets.mean()
        avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0.0
        avg_loss = abs(rets[rets <= 0].mean()) if (rets <= 0).any() else 1e-9
        if side == '空头':
            avg_win = abs(rets[rets < 0].mean()) if (rets < 0).any() else 0.0
            avg_loss = rets[rets >= 0].mean() if (rets >= 0).any() else 1e-9
        pnl_ratio = avg_win / avg_loss if avg_loss != 0 else 0.0
        return {f'{side}样本数': n, f'{side}胜率': win_rate,
                f'{side}均收益': avg_ret, f'{side}盈亏比': pnl_ratio}

    long_stats = _stats(long_rets, '多头')
    short_stats = _stats(short_rets, '空头')

    total_n = long_stats['多头样本数'] + short_stats['空头样本数']
    if total_n > 0:
        long_wins = long_stats['多头样本数'] * long_stats['多头胜率']
        short_wins = short_stats['空头样本数'] * short_stats['空头胜率']
        combined_win_rate = (long_wins + short_wins) / total_n
    else:
        combined_win_rate = 0.0

    # 多空收益差异 t 检验
    spread = long_rets.mean() - short_rets.mean() if len(long_rets) > 0 and len(short_rets) > 0 else 0.0
    if len(long_rets) > 1 and len(short_rets) > 1 and _HAS_SCIPY:
        t_stat, p_val = _scipy_stats.ttest_ind(long_rets, short_rets, equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan

    result = {**long_stats, **short_stats}
    result['综合胜率'] = combined_win_rate
    result['多空收益价差'] = spread
    result['T统计量'] = t_stat
    result['P值'] = p_val
    result['是否显著'] = '✓ 显著' if (not np.isnan(p_val) and p_val < 0.05) else '✗ 不显著'
    return result


def test_all_timing_signals(
    signals_dict: dict,
    returns: pd.Series,
    period: int = 5,
    is_ratio: float = 0.7,
) -> pd.DataFrame:
    """
    批量检验所有信号方法，含样本内/外分割。

    Returns:
        DataFrame，每行一个信号方法，列匹配 timing_report.xlsx sheet3

    Example:
        >>> df = test_all_timing_signals(signals, returns, period=5)
        >>> df[['综合胜率', '是否显著']]
    """
    n = len(returns)
    split_idx = int(n * is_ratio)
    split_date = returns.index[split_idx] if split_idx < n else returns.index[-1]

    rows = []
    for name, signal in signals_dict.items():
        full_stats = test_timing_signal(signal, returns, period)

        # IS / OOS 分割
        is_signal = signal.loc[signal.index <= split_date]
        oos_signal = signal.loc[signal.index > split_date]
        is_returns = returns.loc[returns.index <= split_date]
        oos_returns = returns.loc[returns.index > split_date]

        is_stats = test_timing_signal(is_signal, is_returns, period)
        oos_stats = test_timing_signal(oos_signal, oos_returns, period)

        row = {'检验方法': name, **full_stats, '切分日期': str(split_date.date())}
        row['IS多头胜率'] = is_stats['多头胜率']
        row['IS多头盈亏比'] = is_stats['多头盈亏比']
        row['OOS多头胜率'] = oos_stats['多头胜率']
        row['OOS多头盈亏比'] = oos_stats['多头盈亏比']
        rows.append(row)

    return pd.DataFrame(rows)


def timing_regression(
    factor: pd.Series,
    returns: pd.Series,
    period: int = 5,
) -> dict:
    """
    择时因子回归分析：fwd_return ~ α + β * factor。

    Returns:
        dict: Alpha, Beta, t 值, p 值, R², 调整R², F统计量, 样本量

    Example:
        >>> reg = timing_regression(factor, returns, period=5)
        >>> reg['Beta'], reg['Beta-P值']
    """
    fwd_ret = returns.rolling(period).sum().shift(-period)
    aligned = pd.DataFrame({'factor': factor, 'fwd_ret': fwd_ret}).dropna()

    if len(aligned) < 10:
        return {k: np.nan for k in [
            'Alpha', 'Beta', 'Alpha-T值', 'Beta-T值',
            'Alpha-P值', 'Beta-P值', 'R²', '调整R²',
            'F统计量', 'F-P值', '样本量', 'β是否显著'
        ]}

    X = sm.add_constant(aligned['factor'].astype(float))
    y = aligned['fwd_ret'].astype(float)
    model = sm.OLS(y, X).fit()

    beta_p = model.pvalues.iloc[1] if len(model.pvalues) > 1 else np.nan

    return {
        'Alpha': model.params.iloc[0],
        'Beta': model.params.iloc[1] if len(model.params) > 1 else np.nan,
        'Alpha-T值': model.tvalues.iloc[0],
        'Beta-T值': model.tvalues.iloc[1] if len(model.tvalues) > 1 else np.nan,
        'Alpha-P值': model.pvalues.iloc[0],
        'Beta-P值': beta_p,
        'R²': model.rsquared,
        '调整R²': model.rsquared_adj,
        'F统计量': model.fvalue,
        'F-P值': model.f_pvalue,
        '样本量': int(model.nobs),
        'β是否显著': '✓ 显著' if beta_p < 0.05 else '✗ 不显著',
    }


def timing_robustness(
    factor: pd.Series,
    returns: pd.Series,
    period: int = 5,
    is_ratio: float = 0.7,
    method: str = 'spearman',
) -> dict:
    """
    稳健性检验：样本内/外 IC 对比和 IC 衰减。

    Returns:
        dict: 样本分割日期, IS/OOS IC 均值和 ICIR, IC 衰减幅度, 是否稳健

    Example:
        >>> rob = timing_robustness(factor, returns, period=5)
        >>> rob['是否稳健']
    """
    fwd_ret = returns.rolling(period).sum().shift(-period)
    aligned = pd.DataFrame({'factor': factor, 'fwd_ret': fwd_ret}).dropna()

    n = len(aligned)
    split_idx = int(n * is_ratio)
    split_date = aligned.index[split_idx] if split_idx < n else aligned.index[-1]

    is_data = aligned.iloc[:split_idx]
    oos_data = aligned.iloc[split_idx:]

    def _ic_stats(data):
        if len(data) < 5:
            return {'IC均值': np.nan, 'ICIR': np.nan}
        if method == 'spearman':
            corr = data['factor'].rank().corr(data['fwd_ret'].rank())
        else:
            corr = data['factor'].corr(data['fwd_ret'])
        # 用滚动方式得到 IC 序列
        ic_list = []
        window = min(60, max(10, len(data) // 3))
        for i in range(window - 1, len(data)):
            sub = data.iloc[i - window + 1:i + 1]
            if method == 'spearman':
                c = sub['factor'].rank().corr(sub['fwd_ret'].rank())
            else:
                c = sub['factor'].corr(sub['fwd_ret'])
            ic_list.append(c)
        ic_s = pd.Series(ic_list)
        ic_mean = ic_s.mean()
        ic_std = ic_s.std()
        icir = ic_mean / ic_std if ic_std != 0 else np.nan
        return {'IC均值': ic_mean, 'ICIR': icir}

    is_stats = _ic_stats(is_data)
    oos_stats = _ic_stats(oos_data)

    # IC 衰减幅度
    is_ic = is_stats['IC均值']
    oos_ic = oos_stats['IC均值']
    if is_ic != 0 and not np.isnan(is_ic) and not np.isnan(oos_ic):
        decay = abs(oos_ic - is_ic) / abs(is_ic)
    else:
        decay = np.nan

    # 稳健判定：OOS IC 同号且衰减 < 50%
    is_robust = (
        not np.isnan(is_ic) and not np.isnan(oos_ic)
        and np.sign(is_ic) == np.sign(oos_ic)
        and (np.isnan(decay) or decay < 0.5)
    )

    return {
        '样本分割日期': str(split_date.date()) if hasattr(split_date, 'date') else str(split_date),
        '样本内IC均值': is_stats['IC均值'],
        '样本内ICIR': is_stats['ICIR'],
        '样本外IC均值': oos_stats['IC均值'],
        '样本外ICIR': oos_stats['ICIR'],
        'IC衰减幅度': decay,
        '是否稳健': '✓ 稳健' if is_robust else '✗ 不稳健',
    }


def backtest_timing_signal(
    signal: pd.Series,
    returns: pd.Series,
    benchmark_returns: pd.Series = None,
) -> dict:
    """
    择时信号回测绩效。

    signal 在 t 时刻的值决定 t+1 的仓位：1=多头, -1=空头, 0=空仓。

    Returns:
        dict: 总收益, 年化收益, 波动率, Sharpe, 最大回撤, Calmar, 日胜率, 交易次数等

    Example:
        >>> perf = backtest_timing_signal(signal, returns)
        >>> perf['Sharpe(策略)']
    """
    common = signal.index.intersection(returns.index)
    signal = signal.loc[common]
    returns = returns.loc[common]

    # 仓位：signal 延迟一期（t 时刻信号决定 t+1 仓位）
    position = signal.shift(1).fillna(0)
    strategy_ret = position * returns
    nav = (1 + strategy_ret).cumprod()

    # 策略指标
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1 if len(nav) > 0 else 0.0
    n_days = len(nav)
    ann_ret = strategy_ret.mean() * 252
    ann_vol = strategy_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0.0
    peak = nav.expanding().max()
    drawdown = (peak - nav) / peak
    max_dd = drawdown.max()
    calmar = ann_ret / max_dd if max_dd != 0 else 0.0
    daily_win_rate = (strategy_ret > 0).sum() / len(strategy_ret) if len(strategy_ret) > 0 else 0.0

    # 交易次数
    pos_changes = position.diff().fillna(0)
    trades = (pos_changes != 0).sum()
    # 交易胜率
    trade_entries = pos_changes[pos_changes != 0].index
    trade_returns = []
    for i in range(len(trade_entries) - 1):
        seg = strategy_ret.loc[trade_entries[i]:trade_entries[i + 1]]
        trade_returns.append(seg.sum())
    if trade_returns:
        trade_win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
        pnl_ratio_val = (
            np.mean([r for r in trade_returns if r > 0]) /
            abs(np.mean([r for r in trade_returns if r <= 0]))
            if any(r <= 0 for r in trade_returns) and any(r > 0 for r in trade_returns)
            else 0.0
        )
    else:
        trade_win_rate = 0.0
        pnl_ratio_val = 0.0

    holding_ratio = (position != 0).sum() / len(position) if len(position) > 0 else 0.0

    result = {
        '总收益率(策略)': total_ret,
        '年化收益(策略)': ann_ret,
        '年化波动(策略)': ann_vol,
        'Sharpe(策略)': sharpe,
        '最大回撤(策略)': -max_dd,
        'Calmar(策略)': calmar,
        '日胜率(策略)': daily_win_rate,
        '交易次数': int(trades),
        '交易胜率': trade_win_rate,
        '持有时间占比': holding_ratio,
        '赔率': pnl_ratio_val,
    }

    # 基准
    if benchmark_returns is not None:
        bm = benchmark_returns.loc[common]
        bm_nav = (1 + bm).cumprod()
        bm_total = bm_nav.iloc[-1] / bm_nav.iloc[0] - 1
        bm_ann = bm.mean() * 252
        bm_vol = bm.std() * np.sqrt(252)
        bm_sharpe = bm_ann / bm_vol if bm_vol != 0 else 0.0
        bm_peak = bm_nav.expanding().max()
        bm_dd = ((bm_peak - bm_nav) / bm_peak).max()
        excess_ret = strategy_ret - bm
        excess_ann = excess_ret.mean() * 252
        excess_vol = excess_ret.std() * np.sqrt(252)
        excess_sharpe = excess_ann / excess_vol if excess_vol != 0 else 0.0
        excess_win = (excess_ret > 0).sum() / len(excess_ret) if len(excess_ret) > 0 else 0.0

        result.update({
            '超额胜率': excess_win,
            '总收益率(基准)': bm_total,
            '年化收益(基准)': bm_ann,
            'Sharpe(基准)': bm_sharpe,
            '最大回撤(基准)': -bm_dd,
            '超额年化收益': excess_ann,
            '超额Sharpe': excess_sharpe,
        })

    return result


def compute_timing_score(
    ic_stats: dict,
    signal_results: pd.DataFrame,
    regression_stats: dict,
    robustness_stats: dict,
) -> dict:
    """
    择时因子综合评分（4维度，0~1）。

    - IC (30%): |ICIR| 和 IC>0 占比
    - 信号 (25%): 最佳综合胜率
    - 回归 (20%): R² 和 β 显著性
    - 稳健性 (25%): IS/OOS IC 一致性

    Returns:
        {'IC': float, '信号': float, '回归': float, '稳健性': float,
         '综合评分': float, '评级': str}

    Example:
        >>> score = compute_timing_score(ic_stats, signal_df, reg, rob)
        >>> score['评级']
    """
    # IC 维度
    icir = abs(ic_stats.get('ICIR', 0))
    ic_win = ic_stats.get('胜率(IC>0)', 0.5)
    if icir >= 2:
        ic_score = 0.9
    elif icir >= 1:
        ic_score = 0.7
    elif icir >= 0.5:
        ic_score = 0.5
    else:
        ic_score = max(0.1, icir / 2)
    ic_score = ic_score * 0.7 + min(ic_win, 1.0) * 0.3

    # 信号维度
    if len(signal_results) > 0 and '综合胜率' in signal_results.columns:
        best_wr = signal_results['综合胜率'].max()
        sig_score = min(1.0, max(0, (best_wr - 0.3) / 0.4))
    else:
        sig_score = 0.0

    # 回归维度
    r2 = regression_stats.get('R²', 0)
    beta_p = regression_stats.get('Beta-P值', 1.0)
    r2_score = min(1.0, r2 * 10)  # R² 通常很小，放大
    sig_bonus = 0.5 if (not np.isnan(beta_p) and beta_p < 0.05) else 0.0
    reg_score = r2_score * 0.5 + sig_bonus

    # 稳健性维度
    is_ic = robustness_stats.get('样本内IC均值', 0)
    oos_ic = robustness_stats.get('样本外IC均值', 0)
    decay = robustness_stats.get('IC衰减幅度', 1.0)
    if np.isnan(is_ic) or np.isnan(oos_ic):
        rob_score = 0.1
    else:
        same_sign = 0.5 if np.sign(is_ic) == np.sign(oos_ic) else 0.0
        decay_score = max(0, 0.5 - min(decay, 1.0) * 0.5) if not np.isnan(decay) else 0.0
        rob_score = same_sign + decay_score

    composite = ic_score * 0.30 + sig_score * 0.25 + reg_score * 0.20 + rob_score * 0.25

    if composite >= 0.8:
        grade = 'A'
    elif composite >= 0.6:
        grade = 'B'
    elif composite >= 0.4:
        grade = 'C'
    elif composite >= 0.2:
        grade = 'D'
    else:
        grade = 'F'

    return {
        'IC': round(ic_score, 2),
        '信号': round(sig_score, 2),
        '回归': round(reg_score, 2),
        '稳健性': round(rob_score, 2),
        '综合评分': round(composite, 2),
        '评级': grade,
    }


def run_timing_evaluation(
    factor: pd.Series,
    returns: pd.Series,
    periods: list = None,
    method: str = 'spearman',
    rolling_window: int = 60,
    sigma: float = 1.0,
    ma_window: int = 250,
    is_ratio: float = 0.7,
    factor_name: str = '',
) -> dict:
    """
    一键运行择时因子评价，返回全部结果。

    Args:
        factor: 因子时间序列
        returns: 收益率时间序列
        periods: 预测周期列表
        method: IC 计算方法
        rolling_window: 滚动 IC 窗口
        sigma: 信号阈值
        ma_window: 均线窗口
        is_ratio: 样本内比例
        factor_name: 因子名称

    Returns:
        dict: 包含 ic_results, signals, signal_tests, regression,
              robustness, backtest, score, factor_std 等全部结果

    Example:
        >>> results = run_timing_evaluation(factor, returns, periods=[5,10,20])
        >>> results['score']['评级']
    """
    if periods is None:
        periods = [5, 10, 20]

    # 标准化因子
    f_mean, f_std = factor.mean(), factor.std()
    factor_std = (factor - f_mean) / f_std if f_std != 0 else factor * 0

    # IC 计算
    ic_results = calc_timing_ic(factor_std, returns, periods, method, rolling_window)

    # 信号生成
    signals = generate_timing_signals(factor_std, sigma, ma_window)

    # 取主预测周期（第一个）做信号检验
    main_period = periods[0]
    signal_tests = test_all_timing_signals(signals, returns, main_period, is_ratio)

    # 回归分析
    regression = timing_regression(factor_std, returns, main_period)

    # 稳健性检验
    robustness = timing_robustness(factor_std, returns, main_period, is_ratio, method)

    # 回测（用最佳信号方法）
    best_method = '阈值法'
    if len(signal_tests) > 0:
        best_idx = signal_tests['综合胜率'].idxmax()
        best_method = signal_tests.loc[best_idx, '检验方法']
    backtest = backtest_timing_signal(signals.get(best_method, signals['阈值法']), returns)

    # 综合评分
    main_ic_stats = ic_results[main_period]['stats'] if main_period in ic_results else {}
    score = compute_timing_score(main_ic_stats, signal_tests, regression, robustness)

    return {
        'factor_name': factor_name,
        'factor_std': factor_std,
        'periods': periods,
        'method': method,
        'main_period': main_period,
        'ic_results': ic_results,
        'signals': signals,
        'signal_tests': signal_tests,
        'regression': regression,
        'robustness': robustness,
        'backtest': backtest,
        'score': score,
    }


def export_timing_report(results: dict) -> bytes:
    """
    将择时评价结果导出为 Excel（6 sheet），返回 bytes。

    Example:
        >>> excel_bytes = export_timing_report(results)
        >>> with open('report.xlsx', 'wb') as f: f.write(excel_bytes)
    """
    import io as _io
    buf = _io.BytesIO()

    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        name = results.get('factor_name', '')
        main_period = results['main_period']
        method = results['method'].upper()
        score = results['score']
        ic_stats = results['ic_results'].get(main_period, {}).get('stats', {})
        regression = results['regression']
        robustness = results['robustness']
        backtest = results['backtest']
        signal_tests = results['signal_tests']

        # Sheet 1: 综合概览
        best_wr = {}
        if len(signal_tests) > 0:
            for _, row in signal_tests.iterrows():
                best_wr[row['检验方法']] = row['综合胜率']
        overview = pd.DataFrame([{
            '信号名称': name,
            '预测周期': f'{main_period}期',
            'IC方法': method,
            'IC均值': f"{ic_stats.get('IC均值', 0):.4f}",
            'ICIR': f"{ic_stats.get('ICIR', 0):.4f}",
            'IC正值占比': f"{ic_stats.get('胜率(IC>0)', 0):.2%}",
            '胜率(阈值法)': f"{best_wr.get('阈值法', 0):.2%}",
            '胜率(均线法)': f"{best_wr.get('均线法', 0):.2%}",
            '胜率(极值法)': f"{best_wr.get('极值法', 0):.2%}",
            'β系数': f"{regression.get('Beta', 0):.4f}",
            'β显著性': regression.get('β是否显著', ''),
            'R²': f"{regression.get('R²', 0):.4f}",
            '样本外稳健': robustness.get('是否稳健', ''),
            'IC衰减幅度': f"{robustness.get('IC衰减幅度', 0):.2%}" if not np.isnan(robustness.get('IC衰减幅度', np.nan)) else 'N/A',
            '综合评分': f"{score['综合评分']:.4f}",
            '评级': score['评级'],
        }])
        overview.to_excel(writer, sheet_name='综合概览', index=False)

        # Sheet 2: IC相关性分析
        ic_rows = []
        for period, data in results['ic_results'].items():
            s = data['stats']
            ic_rows.append({
                '信号名称': name,
                '预测周期': f'{period}日',
                'IC方法': method,
                'IC均值': f"{s.get('IC均值', 0):.4f}",
                'IC标准差': f"{s.get('IC_std', 0):.4f}",
                'ICIR': f"{s.get('ICIR', 0):.4f}",
                'IC>0占比': f"{s.get('胜率(IC>0)', 0):.2%}",
                'T统计量': f"{s.get('t统计量', 0):.4f}",
                'P值': f"{s.get('p值', 0):.4f}",
                '是否显著': '✓ 显著' if s.get('p值', 1) < 0.05 else '✗ 不显著',
            })
        pd.DataFrame(ic_rows).to_excel(writer, sheet_name='IC相关性分析', index=False)

        # Sheet 3: 信号检验结果
        if len(signal_tests) > 0:
            export_cols = signal_tests.copy()
            export_cols.insert(0, '信号名称', name)
            export_cols.to_excel(writer, sheet_name='信号检验结果', index=False)

        # Sheet 4: 回归分析
        reg_df = pd.DataFrame([{'信号名称': name, **regression}])
        reg_df.to_excel(writer, sheet_name='回归分析', index=False)

        # Sheet 5: 稳健性检验
        rob_df = pd.DataFrame([{'信号名称': name, **robustness}])
        rob_df.to_excel(writer, sheet_name='稳健性检验', index=False)

        # Sheet 6: 回测绩效汇总
        bt_df = pd.DataFrame([{'信号名称': name, **backtest}])
        bt_df.to_excel(writer, sheet_name='回测绩效汇总', index=False)

    buf.seek(0)
    return buf.read()


# ═════════════════════════════════════════════════════════════════════════════
# 择时因子图表
# ═════════════════════════════════════════════════════════════════════════════

import io as _io
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['simhei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


def _fig_to_bytes(fig) -> bytes:
    buf = _io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_factor_timeseries(
    factor_std: pd.Series,
    signal: pd.Series = None,
    sigma: float = 1.0,
    title: str = '因子值（预处理后）',
) -> bytes:
    """
    因子值时序图：折线 + ±σ 虚线 + 信号区域着色。

    Example:
        >>> img = plot_factor_timeseries(factor_std, signal)
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(factor_std.index, factor_std.values, color='#4FC3F7', linewidth=1, label=factor_std.name or '因子')
    ax.axhline(sigma, color='#FFCDD2', linestyle='--', alpha=0.7, label=f'±{sigma}σ 阈值')
    ax.axhline(-sigma, color='#FFCDD2', linestyle='--', alpha=0.7)
    ax.axhline(0, color='gray', linewidth=0.5)

    if signal is not None:
        common = factor_std.index.intersection(signal.index)
        for i in range(1, len(common)):
            if signal.loc[common[i]] == 1:
                ax.axvspan(common[i - 1], common[i], alpha=0.15, color='green')
            elif signal.loc[common[i]] == -1:
                ax.axvspan(common[i - 1], common[i], alpha=0.15, color='red')

    ax.set_title(title, fontsize=13)
    ax.set_ylabel('标准化因子值')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    return _fig_to_bytes(fig)


def plot_rolling_ic(
    ic_series: pd.Series,
    ic_mean: float = None,
    period: int = 5,
    title: str = None,
) -> bytes:
    """
    滚动 IC 时序图。

    Example:
        >>> img = plot_rolling_ic(ic_series, ic_mean=0.15, period=5)
    """
    if title is None:
        title = f'滚动 IC（预测 {period}日）'
    if ic_mean is None:
        ic_mean = ic_series.mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ic_series.index, ic_series.values, color='#FFA726', linewidth=1)
    ax.axhline(ic_mean, color='#4FC3F7', linestyle='--', linewidth=1,
               label=f'均值={ic_mean:.3f}')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.fill_between(ic_series.index, 0, ic_series.values,
                    where=ic_series.values > 0, alpha=0.1, color='green')
    ax.fill_between(ic_series.index, 0, ic_series.values,
                    where=ic_series.values < 0, alpha=0.1, color='red')
    ax.set_title(title, fontsize=13)
    ax.set_ylabel('IC')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _fig_to_bytes(fig)


def plot_signal_avg_return(
    signal_tests: pd.DataFrame,
    method_name: str = '阈值法',
    title: str = None,
) -> bytes:
    """
    各信号平均收益率柱状图（多头绿/空头红）。

    Example:
        >>> img = plot_signal_avg_return(signal_tests, method_name='阈值法')
    """
    if title is None:
        title = f'各信号平均收益率（{method_name}）'

    row = signal_tests[signal_tests['检验方法'] == method_name]
    if len(row) == 0:
        row = signal_tests.iloc[:1]

    row = row.iloc[0]
    labels = ['多头信号', '空头信号']
    values = [row['多头均收益'], row['空头均收益']]
    colors = ['#66BB6A', '#EF5350']

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'+{val:.3%}' if val >= 0 else f'{val:.3%}',
                ha='center', va='bottom', fontsize=10)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel('平均收益率 (%)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    return _fig_to_bytes(fig)


def plot_win_rate_comparison(
    signal_tests: pd.DataFrame,
    title: str = '三种方法胜率对比',
) -> bytes:
    """
    各方法综合胜率柱状图 + 50% 基线。

    Example:
        >>> img = plot_win_rate_comparison(signal_tests)
    """
    methods = signal_tests['检验方法'].tolist()
    rates = signal_tests['综合胜率'].tolist()
    colors = ['#66BB6A' if r >= 0.5 else '#EF5350' for r in rates]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(range(len(methods)), [r * 100 for r in rates], color=colors, width=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
    ax.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.7, label='50% 基准线')
    ax.set_title(title, fontsize=13)
    ax.set_ylabel('综合胜率 (%)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    return _fig_to_bytes(fig)


def plot_ic_by_period(
    ic_results: dict,
    title: str = '各预测周期 IC & ICIR',
) -> bytes:
    """
    分组柱状图，双 Y 轴（IC 均值 + ICIR）。

    Args:
        ic_results: calc_timing_ic 的返回值

    Example:
        >>> img = plot_ic_by_period(ic_results)
    """
    periods = sorted(ic_results.keys())
    ic_means = [ic_results[p]['stats'].get('IC均值', 0) for p in periods]
    icirs = [ic_results[p]['stats'].get('ICIR', 0) for p in periods]
    labels = [f'{p}日' for p in periods]

    fig, ax1 = plt.subplots(figsize=(4, 4))
    x = np.arange(len(periods))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, ic_means, width, color='#42A5F5', label='IC 均值')
    ax1.set_ylabel('IC 均值', color='#42A5F5')
    ax1.tick_params(axis='y', labelcolor='#42A5F5')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, icirs, width, color='#FFA726', label='ICIR')
    ax2.set_ylabel('ICIR', color='#FFA726')
    ax2.tick_params(axis='y', labelcolor='#FFA726')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title(title, fontsize=13)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    return _fig_to_bytes(fig)


def plot_return_distribution(
    returns: pd.Series,
    signal: pd.Series,
    period: int = 5,
    title: str = '按信号分组的收益分布',
) -> bytes:
    """
    三色直方图（中性灰/多头绿/空头红）。

    Example:
        >>> img = plot_return_distribution(returns, signal)
    """
    fwd_ret = returns.rolling(period).sum().shift(-period)
    aligned = pd.DataFrame({'signal': signal, 'fwd_ret': fwd_ret}).dropna()
    fwd_pct = aligned['fwd_ret'] * 100

    neutral = fwd_pct[aligned['signal'] == 0]
    long_ = fwd_pct[aligned['signal'] == 1]
    short_ = fwd_pct[aligned['signal'] == -1]

    fig, ax = plt.subplots(figsize=(4, 4))
    bins = np.linspace(fwd_pct.min(), fwd_pct.max(), 40)
    if len(neutral) > 0:
        ax.hist(neutral, bins=bins, alpha=0.5, color='gray', label='中性')
    if len(long_) > 0:
        ax.hist(long_, bins=bins, alpha=0.5, color='#66BB6A', label='多头')
    if len(short_) > 0:
        ax.hist(short_, bins=bins, alpha=0.5, color='#EF5350', label='空头')
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('未来收益率 (%)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _fig_to_bytes(fig)


def plot_factor_vs_return(
    factor: pd.Series,
    returns: pd.Series,
    period: int = 5,
    title: str = None,
) -> bytes:
    """
    因子值 vs 未来收益率散点图 + OLS 拟合线。

    Example:
        >>> img = plot_factor_vs_return(factor, returns, period=5)
    """
    if title is None:
        title = f'因子值 vs 未来收益率（{period}日）'

    fwd_ret = returns.rolling(period).sum().shift(-period)
    aligned = pd.DataFrame({'factor': factor, 'fwd_ret': fwd_ret}).dropna()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(aligned['factor'], aligned['fwd_ret'] * 100,
               alpha=0.3, s=10, color='#42A5F5')

    # OLS fit
    if len(aligned) > 2:
        X = sm.add_constant(aligned['factor'].astype(float))
        model = sm.OLS(aligned['fwd_ret'].astype(float) * 100, X).fit()
        x_range = np.linspace(aligned['factor'].min(), aligned['factor'].max(), 100)
        y_hat = model.params.iloc[0] + model.params.iloc[1] * x_range
        ax.plot(x_range, y_hat, color='#EF5350', linewidth=2, label='OLS 拟合')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('因子值')
    ax.set_ylabel('未来收益率 (%)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _fig_to_bytes(fig)


def plot_composite_score(
    scores: dict,
    title: str = None,
) -> bytes:
    """
    综合评分横向柱状图。

    Args:
        scores: compute_timing_score 的返回值

    Example:
        >>> img = plot_composite_score(score_dict)
    """
    grade = scores.get('评级', '')
    composite = scores.get('综合评分', 0)
    if title is None:
        title = f'综合评分：{composite:.2f}  （等级 {grade}）'

    dims = ['IC', '信号', '回归', '稳健性']
    vals = [scores.get(d, 0) for d in dims]

    fig, ax = plt.subplots(figsize=(4, 4))
    colors = []
    for v in vals:
        if v >= 0.6:
            colors.append('#66BB6A')
        elif v >= 0.4:
            colors.append('#FFA726')
        else:
            colors.append('#EF5350')

    bars = ax.barh(dims, vals, color=colors, height=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=10)

    ax.axvline(0.6, color='green', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('评分 [0 - 1]')
    ax.set_title(title, fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    return _fig_to_bytes(fig)


def plot_group_cumulative_returns(
    group_returns: pd.DataFrame,
    title: str = '分组累积收益率',
) -> bytes:
    """
    截面因子分组累积收益折线图。

    Args:
        group_returns: group_return_summary 的输出

    Example:
        >>> img = plot_group_cumulative_returns(group_df)
    """
    cum = group_returns.cumsum()
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(cum.columns)))
    for i, col in enumerate(cum.columns):
        ls = '--' if col == 'long_short' else '-'
        lw = 2 if col == 'long_short' else 1
        ax.plot(cum.index, cum[col], label=col, linestyle=ls, linewidth=lw, color=colors[i])

    ax.set_title(title, fontsize=13)
    ax.set_ylabel('累积收益')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    return _fig_to_bytes(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 截面因子评价补充函数
# ═════════════════════════════════════════════════════════════════════════════

def monotonicity_test(group_returns: pd.DataFrame) -> dict:
    """
    分组收益单调性检验。

    检查各组平均收益是否单调递增/递减。

    Args:
        group_returns: group_return_summary 的输出

    Returns:
        dict: 各组均值, 是否单调, 方向, Spearman 相关系数

    Example:
        >>> mono = monotonicity_test(group_df)
        >>> mono['是否单调']
    """
    group_cols = [c for c in group_returns.columns if c.startswith('G')]
    means = group_returns[group_cols].mean()
    ranks = np.arange(1, len(group_cols) + 1)

    if _HAS_SCIPY:
        corr, p_val = _scipy_stats.spearmanr(ranks, means.values)
    else:
        corr = np.corrcoef(ranks, means.values)[0, 1]
        p_val = np.nan

    is_monotone = abs(corr) > 0.8
    direction = '递增' if corr > 0 else '递减'

    return {
        '各组均值': means.to_dict(),
        '是否单调': '✓ 单调' if is_monotone else '✗ 非单调',
        '方向': direction if is_monotone else '无明确方向',
        'Spearman相关系数': corr,
        'P值': p_val,
    }


def run_cross_section_evaluation(
    factor_data: pd.DataFrame,
    return_data: pd.DataFrame,
    periods: list = None,
    method: str = 'spearman',
    n_groups: int = 5,
    factor_name: str = '',
) -> dict:
    """
    一键运行截面因子评价。

    Args:
        factor_data: 宽表，index=datetime，columns=code
        return_data: 宽表，index=datetime，columns=code
        periods: 预测周期列表（用不同持仓期的 return_data 逐个计算）
        method: IC 方法
        n_groups: 分组数
        factor_name: 因子名称

    Returns:
        dict: ic_series, ic_stats, fm_results, group_returns, mono_test, score

    Example:
        >>> results = run_cross_section_evaluation(factor_wide, return_wide)
        >>> results['ic_stats']
    """
    if periods is None:
        periods = [5]

    # IC 计算（复用 calc_ic）
    ic_all = {}
    for period in periods:
        # 对 return_data 做 N 期前瞻累计
        fwd_ret = return_data.rolling(period).sum().shift(-period)
        ic_series = calc_ic(factor_data, fwd_ret, method=method)
        ic_stats = summarize_ic(ic_series)
        ic_all[period] = {'ic_series': ic_series, 'stats': ic_stats}

    # Fama-MacBeth 回归
    main_period = periods[0]
    fwd_ret_main = return_data.rolling(main_period).sum().shift(-main_period)
    fm_results = fama_macbeth({factor_name or 'factor': factor_data}, fwd_ret_main)

    # 分组收益（简化版，不依赖 labeled_pool）
    group_returns_dict = {}
    common_dates = factor_data.index.intersection(fwd_ret_main.index)
    records = {}
    for ts in common_dates:
        f = factor_data.loc[ts].dropna()
        r = fwd_ret_main.loc[ts].dropna()
        common_codes = f.index.intersection(r.index)
        if len(common_codes) < n_groups * 2:
            continue
        f_c = f.loc[common_codes]
        r_c = r.loc[common_codes]
        labels = pd.qcut(f_c.rank(method='first'), n_groups, labels=False) + 1
        row = {}
        for g in range(1, n_groups + 1):
            codes_g = labels[labels == g].index
            row[f'G{g}'] = r_c.loc[codes_g].mean()
        records[ts] = row
    group_returns = pd.DataFrame(records).T.sort_index()
    if len(group_returns.columns) >= 2:
        g1_mean = group_returns['G1'].mean()
        gn_mean = group_returns[f'G{n_groups}'].mean()
        if gn_mean >= g1_mean:
            group_returns['long_short'] = group_returns[f'G{n_groups}'] - group_returns['G1']
        else:
            group_returns['long_short'] = group_returns['G1'] - group_returns[f'G{n_groups}']

    # 单调性检验
    mono = monotonicity_test(group_returns) if len(group_returns) > 0 else {}

    return {
        'factor_name': factor_name,
        'periods': periods,
        'method': method,
        'n_groups': n_groups,
        'ic_all': ic_all,
        'fm_results': fm_results,
        'group_returns': group_returns,
        'mono_test': mono,
    }
