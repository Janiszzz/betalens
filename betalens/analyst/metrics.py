"""
指标库（纯函数）

所有函数无副作用，输入 nav / returns / weight / position / rebalance_log，
输出标量或 Series，便于单测与组合。日期索引须为 DatetimeIndex。

约定：
- nav: 净值序列（pd.Series, index=日期）
- returns: 日收益率（pd.Series），缺省由 nav.pct_change() 得到
- weight: 调仓权重（pd.DataFrame, index=调仓日, columns=code[+cash]）
- daily_position_value: 日频持仓金额（pd.DataFrame, index=日, columns=code+cash）
- rebalance_log: 调仓长表（列 datetime/code/target_weight/actual_weight/.../value）
"""
import numpy as np
import pandas as pd


# ── 基础工具 ────────────────────────────────────────────────────────────────

def _to_returns(nav: pd.Series) -> pd.Series:
    return nav.sort_index().pct_change().dropna()


def _drawdown_series(nav: pd.Series) -> pd.Series:
    """回撤序列（正数表示回撤幅度）"""
    nav = nav.sort_index()
    peak = nav.expanding(min_periods=1).max()
    return (peak - nav) / peak


# ── 收益类（几何年化，修正原 mean×252 口径）─────────────────────────────────

def total_return(nav: pd.Series) -> float:
    nav = nav.sort_index()
    return nav.iloc[-1] / nav.iloc[0] - 1


def annualized_return(nav: pd.Series, annualizer: int = 252) -> float:
    """几何年化收益率"""
    nav = nav.sort_index()
    n = len(nav)
    if n < 2:
        return np.nan
    return (nav.iloc[-1] / nav.iloc[0]) ** (annualizer / n) - 1


def annualized_volatility(returns: pd.Series, annualizer: int = 252) -> float:
    return returns.std() * np.sqrt(annualizer)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, annualizer: int = 252) -> float:
    excess = returns - rf / annualizer
    if excess.std() == 0:
        return np.nan
    return excess.mean() / excess.std() * np.sqrt(annualizer)


def max_drawdown(nav: pd.Series) -> float:
    return _drawdown_series(nav).max()


def calmar_ratio(nav: pd.Series, annualizer: int = 252) -> float:
    mdd = max_drawdown(nav)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annualized_return(nav, annualizer) / mdd


# ── 回撤类（溃疡 / 痛苦 / Martin / 最长回撤期）──────────────────────────────

def ulcer_index(nav: pd.Series) -> float:
    """溃疡指数：回撤的均方根，惩罚深而久的回撤"""
    dd = _drawdown_series(nav) * 100  # 百分数
    return np.sqrt((dd ** 2).mean())


def martin_ratio(nav: pd.Series, rf: float = 0.0, annualizer: int = 252) -> float:
    """Martin 比率（UPI）= 年化超额收益 / 溃疡指数"""
    ui = ulcer_index(nav)
    if ui == 0 or np.isnan(ui):
        return np.nan
    return (annualized_return(nav, annualizer) - rf) / (ui / 100)


def pain_index(nav: pd.Series) -> float:
    """痛苦指数：平均回撤深度"""
    return _drawdown_series(nav).mean()


def pain_ratio(nav: pd.Series, rf: float = 0.0, annualizer: int = 252) -> float:
    pi = pain_index(nav)
    if pi == 0 or np.isnan(pi):
        return np.nan
    return (annualized_return(nav, annualizer) - rf) / pi


def max_drawdown_duration(nav: pd.Series) -> int:
    """最长回撤持续期（距前高的最长天数，按数据点计）"""
    nav = nav.sort_index()
    peak = nav.expanding(min_periods=1).max()
    under = nav < peak
    longest = cur = 0
    for flag in under:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    return int(longest)


# ── 风险分布类（Sortino / VaR / CVaR / 偏度峰度）────────────────────────────

def downside_deviation(returns: pd.Series, mar: float = 0.0, annualizer: int = 252) -> float:
    """下行偏差（年化），mar 为最低可接受日收益"""
    downside = returns[returns < mar] - mar
    if len(downside) == 0:
        return 0.0
    return np.sqrt((downside ** 2).mean()) * np.sqrt(annualizer)


def sortino_ratio(returns: pd.Series, rf: float = 0.0, annualizer: int = 252) -> float:
    dd = downside_deviation(returns, 0.0, annualizer)
    if dd == 0 or np.isnan(dd):
        return np.nan
    ann_ret = returns.mean() * annualizer - rf
    return ann_ret / dd


def value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """历史法 VaR，返回正数表示潜在损失幅度"""
    if len(returns) == 0:
        return np.nan
    return -np.percentile(returns, level * 100)


def conditional_var(returns: pd.Series, level: float = 0.05) -> float:
    """历史法 CVaR（期望损失）"""
    if len(returns) == 0:
        return np.nan
    var = np.percentile(returns, level * 100)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return -var
    return -tail.mean()


def skewness(returns: pd.Series) -> float:
    return returns.skew()


def kurtosis(returns: pd.Series) -> float:
    return returns.kurtosis()


# ── 滚动类（修正原 rolling_win_rate 错误实现）──────────────────────────────

def rolling_win_rate(returns: pd.Series, window: int = 30) -> pd.Series:
    """滚动胜率：窗口内日收益>0 的占比"""
    return (returns > 0).rolling(window).mean()


def rolling_sharpe(returns: pd.Series, window: int = 60, rf: float = 0.0,
                   annualizer: int = 252) -> pd.Series:
    excess = returns - rf / annualizer
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std()
    return (mean / std) * np.sqrt(annualizer)


def rolling_max_drawdown(nav: pd.Series, window: int = 60) -> pd.Series:
    return nav.sort_index().rolling(window).apply(
        lambda x: ((np.maximum.accumulate(x) - x) / np.maximum.accumulate(x)).max(),
        raw=True,
    )


# ── 交易 / 持仓类（换手 / 最频繁持仓 / 权重堆积 / 持仓数）────────────────────

def turnover(weight: pd.DataFrame, annualizer: int = 252,
             include_cash: bool = False) -> dict:
    """
    换手率。逐期单边换手 = 0.5 * Σ|w_t - w_{t-1}|。

    Returns:
        dict: 含 per_period(Series)、avg_oneway、avg_twoway、annualized
    """
    w = weight.sort_index().fillna(0.0)
    if not include_cash and 'cash' in w.columns:
        w = w.drop(columns='cash')
    diff = w.diff().abs().sum(axis=1)
    oneway = diff * 0.5
    oneway.iloc[0] = w.iloc[0].abs().sum() * 0.5 if len(w) else np.nan
    n = len(w)
    avg_oneway = oneway.mean()
    # 年化：按调仓频率推算每年调仓次数
    periods_per_year = annualizer / max((_avg_period_days(w.index)), 1)
    return {
        'per_period': oneway,
        'avg_oneway': avg_oneway,
        'avg_twoway': avg_oneway * 2,
        'annualized': avg_oneway * periods_per_year,
    }


def _avg_period_days(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    deltas = pd.Series(index).diff().dropna().dt.days
    return deltas.mean() or 1.0


def top_holdings(weight: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """
    最频繁持仓：按出现频率（权重非零的期数占比）+ 平均权重排序。

    Returns:
        DataFrame: index=code, 列 freq(出现频率)、avg_weight(平均权重)、
                   max_weight(最大权重)
    """
    w = weight.sort_index().fillna(0.0)
    if 'cash' in w.columns:
        w = w.drop(columns='cash')
    freq = (w != 0).mean()
    avg_w = w.replace(0, np.nan).mean()
    max_w = w.max()
    out = pd.DataFrame({'freq': freq, 'avg_weight': avg_w, 'max_weight': max_w})
    out = out.sort_values(['freq', 'avg_weight'], ascending=False)
    return out.head(top)


def weight_hhi(weight: pd.DataFrame) -> pd.Series:
    """赫芬达尔指数（权重堆积/集中度），逐期 Σw²，越高越集中"""
    w = weight.sort_index().fillna(0.0)
    if 'cash' in w.columns:
        w = w.drop(columns='cash')
    return (w ** 2).sum(axis=1)


def top_n_concentration(weight: pd.DataFrame, n: int = 5) -> pd.Series:
    """逐期前 N 大持仓权重之和"""
    w = weight.sort_index().fillna(0.0)
    if 'cash' in w.columns:
        w = w.drop(columns='cash')
    return w.apply(lambda row: row.nlargest(n).sum(), axis=1)


def avg_holdings_count(weight: pd.DataFrame) -> dict:
    """持仓标的个数（逐期 + 平均）"""
    w = weight.sort_index().fillna(0.0)
    if 'cash' in w.columns:
        w = w.drop(columns='cash')
    count = (w != 0).sum(axis=1)
    return {'per_period': count, 'avg': count.mean()}


def holding_period(weight: pd.DataFrame) -> float:
    """平均持仓寿命（标的从建仓到清仓的平均持有期数）"""
    w = weight.sort_index().fillna(0.0)
    if 'cash' in w.columns:
        w = w.drop(columns='cash')
    held = (w != 0)
    spans = []
    for code in held.columns:
        col = held[code].values
        run = 0
        for flag in col:
            if flag:
                run += 1
            elif run > 0:
                spans.append(run)
                run = 0
        if run > 0:
            spans.append(run)
    return float(np.mean(spans)) if spans else np.nan


# ── 归因类（收益贡献分解 / 逐笔盈亏）────────────────────────────────────────

def return_contribution(daily_pnl: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """
    收益贡献分解：各标的累计损益及占比。

    Args:
        daily_pnl: 日频损益表（index=日, columns=code+cash）
    Returns:
        DataFrame: index=code, 列 pnl(累计损益)、contribution(占总损益比例)
    """
    total_by_code = daily_pnl.sum()
    grand = total_by_code.sum()
    contrib = total_by_code / grand if grand != 0 else total_by_code * np.nan
    out = pd.DataFrame({'pnl': total_by_code, 'contribution': contrib})
    out = out.reindex(out['pnl'].abs().sort_values(ascending=False).index)
    return out.head(top)


def trade_pnl(rebalance_log: pd.DataFrame) -> pd.DataFrame:
    """
    逐标的盈亏配对统计（基于调仓记录的成交金额变化近似）。

    对每个 code 按调仓日排序，相邻成交金额的差视为已实现/浮动盈亏的代理，
    汇总成交次数、胜率、平均盈亏。

    Returns:
        DataFrame: index=code, 列 trades(成交次数)、win_rate(盈利次数占比)、
                   avg_value(平均成交金额)、total_value(累计成交金额)
    """
    if rebalance_log is None or rebalance_log.empty:
        return pd.DataFrame()
    rows = []
    for code, grp in rebalance_log.groupby('code'):
        grp = grp.sort_values('datetime')
        diffs = grp['value'].diff().dropna()
        trades = len(grp)
        win_rate = (diffs > 0).mean() if len(diffs) else np.nan
        rows.append({
            'code': code,
            'trades': trades,
            'win_rate': win_rate,
            'avg_value': grp['value'].mean(),
            'total_value': grp['value'].sum(),
        })
    out = pd.DataFrame(rows).set_index('code')
    return out.sort_values('total_value', ascending=False)


# ── 基准相对类 ──────────────────────────────────────────────────────────────

def beta(returns: pd.Series, bench_returns: pd.Series) -> float:
    df = pd.concat([returns, bench_returns], axis=1, join='inner').dropna()
    if len(df) < 2 or df.iloc[:, 1].var() == 0:
        return np.nan
    cov = df.cov().iloc[0, 1]
    return cov / df.iloc[:, 1].var()


def alpha(returns: pd.Series, bench_returns: pd.Series, rf: float = 0.0,
          annualizer: int = 252) -> float:
    b = beta(returns, bench_returns)
    if np.isnan(b):
        return np.nan
    ann_p = returns.mean() * annualizer
    ann_b = bench_returns.mean() * annualizer
    return (ann_p - rf) - b * (ann_b - rf)


def tracking_error(returns: pd.Series, bench_returns: pd.Series,
                   annualizer: int = 252) -> float:
    diff = (returns - bench_returns).dropna()
    return diff.std() * np.sqrt(annualizer)


def information_ratio(returns: pd.Series, bench_returns: pd.Series,
                      annualizer: int = 252) -> float:
    te = tracking_error(returns, bench_returns, annualizer)
    if te == 0 or np.isnan(te):
        return np.nan
    diff = (returns - bench_returns).dropna()
    return diff.mean() * annualizer / te


def win_rate_vs_benchmark(returns: pd.Series, bench_returns: pd.Series) -> float:
    diff = (returns - bench_returns).dropna()
    return (diff > 0).mean() if len(diff) else np.nan


# ── 月度收益矩阵 ────────────────────────────────────────────────────────────

def monthly_returns_table(nav: pd.Series) -> pd.DataFrame:
    """月度收益矩阵：index=年, columns=月(1-12)+全年"""
    nav = nav.sort_index()
    monthly = nav.resample('ME').last().pct_change().dropna()
    if monthly.empty:
        return pd.DataFrame()
    df = monthly.to_frame('ret')
    df['year'] = df.index.year
    df['month'] = df.index.month
    table = df.pivot_table(index='year', columns='month', values='ret')
    yearly = nav.resample('YE').last().pct_change().dropna()
    table['全年'] = [yearly[yearly.index.year == y].iloc[0]
                     if (yearly.index.year == y).any() else np.nan
                     for y in table.index]
    return table
