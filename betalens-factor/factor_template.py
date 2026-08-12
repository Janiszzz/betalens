#%%
"""
因子复现通用模板（合并 alpha101 + tdx 两类）

本模板是 betalens-factor 下所有因子类的唯一公共依赖。各类因子目录
（alpha101 / tdx / ...）下的 factor_<NAME>.py 只定义算子与 FactorSpec，
取数 / 分组 / 权重 / 回测 / 评价主干全部复用本文件的 FactorPipeline。

使用方式（最小例）：
    from factor_template import FactorSpec, FactorPipeline

    def compute_my_factor(close_wide, window=20):
        return close_wide.pct_change(window)

    spec = FactorSpec(
        name="MYFACTOR",
        inputs={"close_wide": "收盘价(元)"},
        compute=compute_my_factor,
        direction="positive",          # 高分组做多
        compute_kwargs={"window": 20},
        index_code="000906.SH",         # 指数成分股池（PIT 防前视）
    )

    if __name__ == "__main__":
        FactorPipeline(spec).run("2024-01-01", "2025-12-31")

算子约定:
    - 入参：spec.inputs 中声明的每个 key 对应一个宽表 DataFrame
      (index=datetime, columns=code)；外加 compute_kwargs 透传的参数。
    - 出参：同形状宽表，框架自动 stack 为长表喂给 single_characteristic。
    - 若算子需要额外宽表，通过 extra_inputs 提供。

技术指标口径（TDX 类）：betalens 无现成封装，全部用 pandas ewm/rolling 自实现：
    TDX SMA(X,N,M) → X.ewm(alpha=M/N, adjust=False).mean()
    TDX EMA(X,N)   → X.ewm(span=N, adjust=False).mean()
    TDX REF(X,n)   → X.shift(n);  LLV/HHV → rolling(n).min()/.max()

返回值：run() 返回 RunResult（含 backtest/analyst/profiling/neutralize_stats），
支持 `bt, analyst = pipeline.run(...)` 解包（向后兼容）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Callable, Any


DB_TABLE = "daily_market"


def _ensure_runtime():
    """Load heavy runtime dependencies only when a pipeline actually runs."""
    global pd, np
    global Datafeed, get_absolute_trade_days, get_index_universe
    global single_characteristic, get_single_factor_weight
    global winsorize_factor, standardize_factor, neutralize_factor, query_industry_panel
    global fix_null_values, FillStrategy, BacktestBase, Analyst

    if "pd" in globals():
        return

    print("    import pandas", flush=True)
    import pandas as _pd
    print("    import numpy", flush=True)
    import numpy as _np
    print("    import betalens.datafeed", flush=True)
    from betalens.datafeed import (
        Datafeed as _Datafeed,
        get_absolute_trade_days as _get_absolute_trade_days,
        get_index_universe as _get_index_universe,
    )
    print("    import betalens.factor.factor", flush=True)
    from betalens.factor.factor import (
        single_characteristic as _single_characteristic,
        get_single_factor_weight as _get_single_factor_weight,
    )
    print("    import betalens.factor.preprocessing", flush=True)
    from betalens.factor.preprocessing import (
        winsorize_factor as _winsorize_factor,
        standardize_factor as _standardize_factor,
        neutralize_factor as _neutralize_factor,
        query_industry_panel as _query_industry_panel,
    )
    print("    import datafeed.validation", flush=True)
    from datafeed.validation import fix_null_values as _fix_null_values, FillStrategy as _FillStrategy
    print("    import betalens.backtest", flush=True)
    from betalens.backtest import BacktestBase as _BacktestBase
    print("    import betalens.analyst", flush=True)
    from betalens.analyst import Analyst as _Analyst

    pd = _pd
    np = _np
    Datafeed = _Datafeed
    get_absolute_trade_days = _get_absolute_trade_days
    get_index_universe = _get_index_universe
    single_characteristic = _single_characteristic
    get_single_factor_weight = _get_single_factor_weight
    winsorize_factor = _winsorize_factor
    standardize_factor = _standardize_factor
    neutralize_factor = _neutralize_factor
    query_industry_panel = _query_industry_panel
    fix_null_values = _fix_null_values
    FillStrategy = _FillStrategy
    BacktestBase = _BacktestBase
    Analyst = _Analyst


def _ensure_profiling_runtime():
    global describe_distribution, coverage_stats, detect_outliers
    global distribution_stability
    global factor_profile_payload, plt

    if "factor_profile_payload" in globals():
        return

    from betalens.factor.profiling import (
        describe_distribution as _describe_distribution,
        coverage_stats as _coverage_stats,
        detect_outliers as _detect_outliers,
        distribution_stability as _distribution_stability,
        factor_profile_payload as _factor_profile_payload,
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    describe_distribution = _describe_distribution
    coverage_stats = _coverage_stats
    detect_outliers = _detect_outliers
    distribution_stability = _distribution_stability
    factor_profile_payload = _factor_profile_payload
    plt = _plt


# ============================================================
# 通用工具函数
# ============================================================

def fetch_daily_wide(metric, universe=None, start_date=None, end_date=None,
                     table_name=DB_TABLE):
    data = Datafeed(table_name)
    try:
        df = data.query_time_range(codes=universe, start_date=start_date,
                                   end_date=end_date, metric=metric)
    finally:
        data.close()
    if df.empty:
        return pd.DataFrame()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.pivot_table(index='datetime', columns='code', values='value').sort_index()


def align_daily_wides(wides):
    """Align daily metrics by trade date at that day's latest availability time."""
    import pandas as pd

    nonempty = [wide for wide in wides.values() if wide is not None and not wide.empty]
    if not nonempty:
        return dict(wides)

    latest_by_day = {}
    for wide in nonempty:
        for ts in pd.DatetimeIndex(wide.index):
            stamp = pd.Timestamp(ts)
            day = stamp.normalize()
            if day not in latest_by_day or stamp > latest_by_day[day]:
                latest_by_day[day] = stamp
    days = pd.DatetimeIndex(sorted(latest_by_day))
    canonical_index = pd.DatetimeIndex([latest_by_day[day] for day in days])

    aligned = {}
    for name, wide in wides.items():
        if wide is None or wide.empty:
            aligned[name] = wide
            continue
        frame = wide.copy()
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        frame = frame.loc[~frame.index.duplicated(keep="last")].reindex(days)
        frame.index = canonical_index
        aligned[name] = frame
    return aligned


def wide_to_prequery(wide_df, metric_name, signal_dates):
    """宽表 → betalens 长表（仅保留 signal_dates 当日截面）。

    输出列与 pre_query_characteristic_data 对齐：input_ts/code/{metric}/datetime/
    diff_hours，可直接喂给 preprocess / single_characteristic。
    """
    date_set = set(signal_dates)
    mask = wide_df.index.map(lambda ts: ts.date() in date_set)
    wide_df = wide_df.loc[mask]
    long = wide_df.stack().reset_index()
    long.columns = ['input_ts', 'code', metric_name]
    long['input_ts'] = pd.to_datetime(long['input_ts'])
    long['datetime'] = long['input_ts']
    long['diff_hours'] = 0.0
    return long


def build_pit_universe(signal_dates, index_code, table_name="index_universe"):
    """构建 {信号日: [成分股代码]} 的 point-in-time 成分股映射（防前视）。"""
    data = Datafeed(table_name)
    try:
        pit = data.get_index_universe_panel(index_code, signal_dates)
    finally:
        data.close()
    return pit


def mask_wide_by_pit_universe(wide_df, pit_universe):
    """Mask every row to constituents effective on that row's calendar date."""
    import pandas as pd

    if wide_df is None or wide_df.empty or not pit_universe:
        return wide_df
    mask = pd.DataFrame(False, index=wide_df.index, columns=wide_df.columns)
    columns = set(map(str, wide_df.columns))
    for ts in wide_df.index:
        members = pit_universe.get(pd.Timestamp(ts).date(), set())
        keep = list(columns.intersection(map(str, members)))
        if keep:
            mask.loc[ts, keep] = True
    return wide_df.where(mask)


def fetch_industry_wide(scheme, universe, dates, reference_index, chunk_size=30):
    """Fetch a PIT industry label panel and align it to market-wide timestamps."""
    if not universe or not dates or reference_index is None:
        return pd.DataFrame()
    pieces = []
    data = Datafeed("industry")
    try:
        day_list = list(dict.fromkeys(pd.Timestamp(day).date() for day in dates))
        for offset in range(0, len(day_list), int(chunk_size)):
            chunk = day_list[offset:offset + int(chunk_size)]
            frame = data.query_industry(codes=list(universe), dates=chunk, scheme=scheme)
            if frame is not None and not frame.empty:
                pieces.append(frame[["query_date", "code", "ind_name"]])
    finally:
        data.close()
    if not pieces:
        return pd.DataFrame(index=reference_index, columns=universe, dtype=object)
    labels = pd.concat(pieces, ignore_index=True)
    labels["query_date"] = pd.to_datetime(labels["query_date"]).dt.normalize()
    pivot = labels.pivot_table(
        index="query_date", columns="code", values="ind_name", aggfunc="last"
    )
    normalized = pd.DatetimeIndex(reference_index).normalize()
    out = pivot.reindex(index=normalized, columns=universe)
    out.index = reference_index
    return out


def filter_long_by_pit_universe(long_df, pit_universe):
    """按 point-in-time 成分股逐期过滤长表。

    某信号日成分股为空（指数无快照）时严格剔除该期，避免在无实时股票池
    约束的情况下误选全市场股票。
    """
    if long_df.empty or not pit_universe:
        return long_df

    def _keep(row):
        members = pit_universe.get(row['input_ts'].date())
        if not members:
            return False
        return row['code'] in members

    mask = long_df.apply(_keep, axis=1)
    return long_df.loc[mask].reset_index(drop=True)


def infer_warmup_days(compute_kwargs, minimum=0):
    """根据常见窗口参数自动推断取数预热天数。

    rolling/ewm/delta 类因子通常需要回测起点前的历史数据。这里把
    window/lookback/period/span/lag/n 等整数参数视为交易日窗口，并按约
    2 倍日历天加缓冲换算，保证年化窗口在回测首日附近已有完整历史。
    """
    candidates = []
    for key, value in (compute_kwargs or {}).items():
        key_l = str(key).lower()
        if not any(token in key_l for token in ("window", "lookback", "period", "span", "lag")) and key_l not in {"n"}:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and np.isfinite(value) and value > 1:
            candidates.append(int(value))
    if not candidates:
        return int(minimum or 0)
    return max(int(minimum or 0), int(max(candidates) * 2 + 30))


def validate_weights_in_pit_universe(weights, pit_universe):
    """校验每期非零权重股票是否都属于该期 PIT 股票池。"""
    import pandas as pd

    if not pit_universe or weights is None or weights.empty:
        return pd.DataFrame()
    rows = []
    for ts, row in weights.iterrows():
        signal_date = pd.Timestamp(ts).date()
        members = pit_universe.get(signal_date, set())
        selected = {
            str(code)
            for code, weight in row.items()
            if str(code) != "cash" and pd.notna(weight) and abs(float(weight)) > 0
        }
        outside = sorted(selected - {str(code) for code in members})
        rows.append({
            "input_ts": pd.Timestamp(ts),
            "pit_size": len(members),
            "selected_count": len(selected),
            "outside_count": len(outside),
            "outside_codes": ",".join(outside[:20]),
            "passed": len(outside) == 0 and len(members) > 0,
        })
    return pd.DataFrame(rows).set_index("input_ts").sort_index()


def _labeled_to_factor_values(labeled, name):
    label_col = f"{name}_label"
    factor_values = labeled.reset_index()[['input_ts', 'code', name, label_col]].copy()
    factor_values = factor_values.rename(columns={
        'input_ts': '信号日', 'code': '股票代码',
        name: '因子值', label_col: '分组',
    })
    return factor_values.sort_values(
        ['信号日', '分组', '因子值'], ascending=[True, False, False]
    ).reset_index(drop=True)


def _expand_weights_to_factor_universe(weights, factor_values):
    """保留全量因子股票代码，让回测收益矩阵覆盖所有分组。

    未选分组的权重仍为 0，因此不会进入策略持仓；但 BacktestBase 会据此
    查询这些股票的价格，供独立的分组净值图使用。
    """
    import pandas as pd

    if weights is None or factor_values is None or factor_values.empty:
        return weights

    code_col = "股票代码" if "股票代码" in factor_values.columns else "code"
    if code_col not in factor_values.columns:
        return weights

    codes = pd.Index(
        factor_values[code_col].dropna().astype(str).drop_duplicates(),
        name=getattr(weights.columns, "name", None),
    )
    expanded = weights.copy()
    expanded.columns = expanded.columns.astype(str)
    return expanded.reindex(columns=codes, fill_value=0.0)


def _factor_values_for_group_nav(factor_values, n_quantiles):
    """把内部 0 基分组标签转换为 group_nav 使用的 1 基标签。"""
    import pandas as pd

    if factor_values is None or factor_values.empty:
        return factor_values
    group_col = "分组" if "分组" in factor_values.columns else "group"
    if group_col not in factor_values.columns:
        return factor_values

    groups = pd.to_numeric(factor_values[group_col], errors="coerce")
    valid = groups.dropna()
    if valid.empty:
        return factor_values
    if valid.min() >= 0 and valid.max() < int(n_quantiles) and (valid == 0).any():
        out = factor_values.copy()
        out[group_col] = groups + 1
        return out
    return factor_values


def grouped_factor_statistics(labeled, name):
    """基于 single_characteristic 的全量分组矩阵生成统计表。"""
    label_col = f"{name}_label"
    df = labeled.reset_index()[["input_ts", "code", name, label_col]].copy()
    df = df.rename(columns={"input_ts": "信号日", "code": "股票代码", name: "因子值", label_col: "分组"})
    by_date_group = (
        df.groupby(["信号日", "分组"])["因子值"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            min="min",
            q25=lambda s: s.quantile(0.25),
            median="median",
            q75=lambda s: s.quantile(0.75),
            max="max",
        )
        .reset_index()
    )
    summary = (
        df.groupby("分组")["因子值"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            min="min",
            q25=lambda s: s.quantile(0.25),
            median="median",
            q75=lambda s: s.quantile(0.75),
            max="max",
        )
        .reset_index()
    )
    return df, by_date_group, summary


def append_grouped_profiling_excel(output_dir, name, labeled):
    """把全量分组矩阵与分组统计写入 profiling Excel。"""
    excel_path = f"{output_dir}/{name}_profiling.xlsx"
    values, by_date_group, summary = grouped_factor_statistics(labeled, name)
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        by_date_group.to_excel(writer, sheet_name="group_stats_by_date", index=False)
        summary.to_excel(writer, sheet_name="group_stats_summary", index=False)
        values.to_excel(writer, sheet_name="group_factor_values", index=False)
    return {
        "group_stats_by_date": by_date_group,
        "group_stats_summary": summary,
        "group_factor_values": values,
    }


# ============================================================
# 静态图辅助函数
# ============================================================

def _match_trade_pairs(rebalance_log):
    """兼容旧模板调用，实际口径由 analyst.metrics 统一维护。"""
    from betalens.analyst.metrics import match_trade_pairs

    return match_trade_pairs(rebalance_log)


def _compute_group_nav(bt, factor_values, n_quantiles: int):
    """从已有回测的 cost_ret 直接计算各分组等权净值，无需额外查库。

    原理：bt.cost_ret 已记录每个调仓区间内各标的的价格变化率；
    factor_values 给出每个信号日各分组的成员；两者结合即可得到各组
    等权持仓期收益，cumprod 后得到净值曲线（从 1.0 出发）。

    Args:
        bt:            已完成回测的 BacktestBase 实例（含 cost_ret）
        factor_values: _labeled_to_factor_values 输出（信号日/股票代码/分组）
        n_quantiles:   分组数

    Returns:
        DataFrame: index=调仓日, columns=[G1..Gn]，净值从 1.0 出发
    """
    from betalens.analyst.metrics import group_nav

    return group_nav(
        getattr(bt, 'cost_ret', None),
        _factor_values_for_group_nav(factor_values, n_quantiles),
        n_quantiles,
    )


# ============================================================
# 因子声明 + 运行结果容器
# ============================================================

@dataclass
class FactorSpec:
    """声明一个因子的全部信息。

    name:           因子名（输出文件前缀、长表列名）
    inputs:         {算子参数名: 数据库 metric}，框架按此抓取每个宽表
    compute:        算子函数；签名 = inputs 中所有 key + compute_kwargs
    direction:      "positive"→高分组做多 (long=[n_q-1]) | "negative"→低分组做多 (long=[0])
    compute_kwargs: 透传给 compute 的额外关键字参数（如 window=20）
    table_name:     Datafeed 数据表名
    use_industry / use_mktcap:
                    是否做行业 / 市值中性化。True 时管线自动 point-in-time
                    查 industry 表 / 查"市值"宽表取 log。
    industry_scheme: 行业中性化分类体系，如 '申万一级行业'。
    index_code:     指数代码（如 '000906.SH'=中证800）。给定后逐期用 PIT 成分股
                    过滤面板（防前视）；None 则用传入的静态 universe。
    long_groups / short_groups: 显式覆盖 direction 给出的分组列表
    weight_mode:    get_single_factor_weight 的 mode 参数
    backtest_metric: BacktestBase 的成交价 metric
    """
    name: str
    inputs: dict[str, str]
    compute: Callable[..., pd.DataFrame]
    industry_inputs: dict[str, str] = field(default_factory=dict)
    required_history_bars: int = 0
    mask_inputs_by_pit: bool = False
    direction: str = "positive"
    compute_kwargs: dict[str, Any] = field(default_factory=dict)
    table_name: str = DB_TABLE
    use_industry: bool = False
    use_mktcap: bool = False
    industry_scheme: str = "申万一级行业"
    index_code: str | None = None
    long_groups: list | None = None
    short_groups: list | None = None
    weight_mode: str = "freeplay"
    group_weights: dict[str, Any] = field(default_factory=dict)
    intra_group_allocation: dict[str, Any] = field(default_factory=dict)
    backtest_metric: str = "收盘价(元)"
    strategy_type: str = "cross_section"  # "cross_section" | "timing"


@dataclass
class RunResult:
    """FactorPipeline.run() 的统一结果容器。

    支持 `bt, analyst = pipeline.run(...)` 解包（向后兼容旧调用方）；
    新代码用 result.profiling / result.neutralize_stats 取增量产物。
    """
    backtest: Any = None
    analyst: Any = None
    profiling: dict | None = None
    neutralize_stats: pd.DataFrame | None = None
    factor_values: pd.DataFrame | None = None
    pit_validation: pd.DataFrame | None = None
    chart_data: dict | None = None

    def __iter__(self):
        return iter((self.backtest, self.analyst))


# ============================================================
# 运行管线
# ============================================================

class FactorPipeline:
    def __init__(self, spec: FactorSpec):
        self.spec = spec

    def _resolve_groups(self, n_q: int) -> tuple[list, list]:
        sp = self.spec
        if sp.weight_mode == "classic-long-short":
            return [n_q - 1], [0]
        long_groups = sp.long_groups or []
        short_groups = sp.short_groups or []
        if not long_groups and not short_groups:
            raise ValueError("freeplay 模式必须至少设置 long_groups 或 short_groups")
        return long_groups or [], short_groups or []

    def _preprocess_with_stats(self, prequery, metric, industry_scheme,
                               mktcap_col, verbose):
        """逐截面 winsorize→standardize→neutralize，同时收集中性化诊断。

        等价于 betalens.preprocess_factor，但额外返回逐期诊断 DataFrame
        （preprocess_factor 仅 print 不返回，故此处内联以便 dashboard 展示）。

        Returns: (processed_df, neu_stats_df)
            neu_stats_df 列：input_ts/n_obs/n_industry_dummies/r2/skipped
        """
        data = fix_null_values(prequery, strategy=FillStrategy.DROP, columns=[metric])

        ind_panel = None
        if industry_scheme:
            ind_panel = query_industry_panel(
                data, scheme=industry_scheme, industry_table='industry',
                verbose=False)

        groups, neu_stats = [], []
        for ts, group in data.groupby('input_ts'):
            sub = group.copy()
            series = sub.set_index('code')[metric]
            series = winsorize_factor(series, method='mad', n=3.0)
            series = standardize_factor(series, method='zscore')
            before = series.copy()

            industry = None
            if ind_panel is not None and \
               pd.Timestamp(ts) in ind_panel.index.get_level_values('input_ts'):
                industry = ind_panel.xs(pd.Timestamp(ts), level='input_ts').reindex(series.index)
            mktcap = sub.set_index('code')[mktcap_col] \
                if mktcap_col and mktcap_col in sub.columns else None

            if industry is not None or mktcap is not None:
                series, st = neutralize_factor(
                    series, industry_labels=industry,
                    log_market_cap=mktcap, return_stats=True)
                st['input_ts'] = pd.Timestamp(ts)
                if mktcap is not None:
                    pre_pair = pd.concat([before.rename('factor'), mktcap.rename('log_mktcap')], axis=1).dropna()
                    post_pair = pd.concat([series.rename('factor'), mktcap.rename('log_mktcap')], axis=1).dropna()
                    st['mktcap_corr_before'] = pre_pair['factor'].corr(pre_pair['log_mktcap']) if len(pre_pair) > 2 else np.nan
                    st['mktcap_corr_after'] = post_pair['factor'].corr(post_pair['log_mktcap']) if len(post_pair) > 2 else np.nan
                if industry is not None:
                    pre_ind = pd.concat([before.rename('factor'), industry.rename('industry')], axis=1).dropna()
                    post_ind = pd.concat([series.rename('factor'), industry.rename('industry')], axis=1).dropna()
                    st['industry_abs_mean_before'] = pre_ind.groupby('industry')['factor'].mean().abs().mean() if len(pre_ind) else np.nan
                    st['industry_abs_mean_after'] = post_ind.groupby('industry')['factor'].mean().abs().mean() if len(post_ind) else np.nan
                neu_stats.append(st)

            sub = sub.set_index('code')
            sub[metric] = series
            sub = sub.reset_index()
            groups.append(sub)

        processed = pd.concat(groups, ignore_index=True) if groups else data.iloc[0:0]
        neu_df = pd.DataFrame(neu_stats) if neu_stats else None
        if neu_df is not None:
            neu_df = neu_df.set_index('input_ts').sort_index()
            if verbose:
                done = neu_df[~neu_df['skipped']]
                print(f"  中性化: 总{len(neu_df)}期 成功{len(done)} 跳过{len(neu_df)-len(done)}"
                      f" 平均R2={done['r2'].mean():.4f}" if len(done) else
                      f"  中性化: 总{len(neu_df)}期 全部跳过")
        return processed, neu_df

    def _run_profiling(self, factor_wide, name, output_dir, verbose):
        """因子值体检：分布函数/集中度/p值阈值/时变稳定性 + PNG。"""
        _ensure_profiling_runtime()
        profile = factor_profile_payload(factor_wide)
        profile_autocorrelation = pd.DataFrame(profile['autocorrelation']).rename(columns={
            'mean': '自相关均值',
            'std': '自相关std',
            'periods': '有效期数',
        })
        if not profile_autocorrelation.empty:
            profile_autocorrelation = profile_autocorrelation.set_index('lag')
        profile_turnover = pd.Series(
            {
                pd.Timestamp(row['date']): row['turnover']
                for row in profile['turnover']
            },
            name='turnover',
            dtype=float,
        )
        results = {
            'distribution': describe_distribution(factor_wide),
            'profile_summary': pd.DataFrame([profile['summary']]),
            'profile_histogram': pd.DataFrame(profile['histogram']),
            'profile_ecdf': pd.DataFrame(profile['ecdf']),
            'profile_quantiles': pd.DataFrame(profile['quantiles']),
            'profile_p_tests': pd.DataFrame(profile['tests']),
            'profile_timeseries': pd.DataFrame(profile['timeseries']),
            'coverage': coverage_stats(factor_wide),
            'outliers': detect_outliers(factor_wide),
            'autocorrelation': profile_autocorrelation,
            'turnover': profile_turnover,
            'stability': distribution_stability(factor_wide),
        }

        excel_path = f"{output_dir}/{name}_profiling.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            for sheet_name, df in results.items():
                d = df.to_frame() if isinstance(df, pd.Series) else df
                d.to_excel(writer, sheet_name=sheet_name[:31])

        fig, axes = plt.subplots(4, 2, figsize=(15, 16))
        fig.suptitle(f'{name} Factor Profiling', fontsize=14)

        ax = axes[0, 0]
        hist = pd.DataFrame(profile['histogram'])
        if not hist.empty:
            ax.bar(hist['mid'], hist['count'], width=(hist['right'] - hist['left']).replace(0, np.nan), color='#2d66a8')
        ax.set_title('Factor value distribution')
        ax.set_xlabel('factor value')
        ax.set_ylabel('count')

        ax = axes[0, 1]
        ecdf = pd.DataFrame(profile['ecdf'])
        if not ecdf.empty:
            ax.plot(ecdf['value'], ecdf['probability'], color='#6a9f42')
        ax.set_title('Empirical CDF')
        ax.set_xlabel('factor value')
        ax.set_ylabel('F(x)')
        ax.set_ylim(0, 1.02)

        ax = axes[1, 0]
        cov = results['coverage']
        ax.plot(cov.index, cov['覆盖率'])
        ax.set_title('Coverage'); ax.set_ylim(0, 1.05)

        ax = axes[1, 1]
        out = results['outliers']
        out_ts = out.drop('Total') if 'Total' in out.index else out
        ax.bar(range(len(out_ts)), out_ts['极值占比'].values, width=1)
        ax.set_title('Outlier ratio')

        ax = axes[2, 0]
        ac = results['autocorrelation']
        ax.bar(ac.index.astype(str), ac['自相关均值'])
        ax.set_title('Rank autocorr'); ax.set_xlabel('lag')

        ax = axes[2, 1]
        to = results['turnover']
        ax.plot(to.index, to.values)
        ax.set_title('Top 20% turnover')

        ax = axes[3, 0]
        stab = results['stability']
        ax.plot(stab.index, stab['mean'], label='mean')
        ax2 = ax.twinx()
        ax2.plot(stab.index, stab['std'], color='orange', label='std')
        ax.set_title('Distribution drift (mean/std)')

        ax = axes[3, 1]
        ax.plot(stab.index, stab['skew'], label='skew')
        ax.plot(stab.index, stab['kurt'], label='kurt')
        ax.legend(); ax.set_title('Skew / Kurt')

        plt.tight_layout()
        png_path = f"{output_dir}/{name}_profiling.png"
        fig.savefig(png_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        if verbose:
            print(f"  Profiling: {excel_path} + {png_path}")
        return results

    def run(self, start_date: str, end_date: str, *,
            rebal_freq: str = "D",
            universe: list | None = None,
            n_quantiles: int = 20,
            initial_amount: float = 1e8,
            benchmark_code: str | None = None,
            output_dir: str = ".",
            extra_inputs: dict[str, pd.DataFrame] | None = None,
            include_profiling: bool = True,
            dump_excel: bool = True,
            warmup_days: int | None = None,
            verbose: bool = True) -> RunResult:
        """运行完整管线: 取数 → 算子 → [profiling] → 中性化 → 分组 → 权重 → 回测 → 报告

        返回 RunResult（可解包为 bt, analyst 向后兼容）。
        股票池：index_code 给定时逐期 PIT 成分股过滤（防前视）；否则用静态 universe。
        中性化：use_industry / use_mktcap 控制，诊断收入 RunResult.neutralize_stats。
        dump_excel=False 时跳过 dump_to_excel（调用方可自行异步落盘，避免阻塞）。
        """
        print("  加载运行依赖", flush=True)
        _ensure_runtime()
        print("  运行依赖已加载", flush=True)
        sp = self.spec
        inferred_days = infer_warmup_days(
            sp.compute_kwargs,
            minimum=max(30, int(sp.required_history_bars) * 2 + 30),
        )
        warmup = int(warmup_days) if warmup_days is not None else inferred_days
        fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=warmup)).strftime("%Y-%m-%d")

        rebalance_dates = get_absolute_trade_days(start_date, end_date, rebal_freq)
        all_trade_days = get_absolute_trade_days(fetch_start, end_date, "D")
        if verbose:
            print(f"取数起始(自动预热): {fetch_start}  调仓日数量: {len(rebalance_dates)}")

        # 0. 信号日 = 调仓日前一交易日
        td = sorted(all_trade_days)
        td_idx = {d: i for i, d in enumerate(td)}
        signal_dates = []
        for rd in rebalance_dates:
            i = td_idx.get(rd)
            if i is not None and i > 0:
                signal_dates.append(td[i - 1])

        # 1. 股票池：时变成分股（PIT）或静态 universe
        pit_universe = None
        if sp.index_code:
            if verbose:
                print(f"  构建 PIT 股票池: {sp.index_code}, 信号日 {len(signal_dates)} 期", flush=True)
            pit_dates = all_trade_days if sp.mask_inputs_by_pit else signal_dates
            pit_universe = build_pit_universe(pit_dates, sp.index_code)
            universe = sorted({c for codes in pit_universe.values() for c in codes})
            if verbose:
                empty_days = sum(1 for codes in pit_universe.values() if not codes)
                print(f"  {sp.index_code} 成分股并集: {len(universe)} 只 (逐期 PIT 过滤, 空快照 {empty_days} 期)", flush=True)
        elif universe is None:
            raise ValueError("未设 index_code 时必须传入静态 universe")

        # 2. 批量抓宽表
        wides = {}
        for arg_name, metric in sp.inputs.items():
            if verbose:
                print(f"  开始取数: {arg_name} ({metric}) {fetch_start} -> {end_date}, universe={len(universe or [])}", flush=True)
            w = fetch_daily_wide(metric, universe=universe,
                                 start_date=fetch_start, end_date=end_date,
                                 table_name=sp.table_name)
            if verbose:
                print(f"  完成取数: {arg_name} ({metric}) {w.shape}", flush=True)
            wides[arg_name] = w
        wides = align_daily_wides(wides)
        if sp.mask_inputs_by_pit and pit_universe is not None:
            wides = {
                name: mask_wide_by_pit_universe(wide, pit_universe)
                for name, wide in wides.items()
            }
        if sp.industry_inputs:
            reference_index = next(
                (wide.index for wide in wides.values() if wide is not None and not wide.empty),
                pd.DatetimeIndex([]),
            )
            for arg_name, scheme in sp.industry_inputs.items():
                industry_wide = fetch_industry_wide(
                    scheme,
                    universe=universe,
                    dates=all_trade_days,
                    reference_index=reference_index,
                )
                if sp.mask_inputs_by_pit and pit_universe is not None:
                    industry_wide = mask_wide_by_pit_universe(industry_wide, pit_universe)
                wides[arg_name] = industry_wide
        if extra_inputs:
            wides.update(extra_inputs)

        # 3. 调用算子
        if verbose:
            print(f"  开始计算因子: {sp.name}", flush=True)
        factor_wide = sp.compute(**wides, **sp.compute_kwargs)
        if verbose:
            print(f"  完成计算因子: {factor_wide.shape}", flush=True)

        # 4. 宽 → 长（仅信号日）
        if verbose:
            print(f"  因子宽表转长表: signal_dates={len(signal_dates)}", flush=True)
        prequery = wide_to_prequery(factor_wide, sp.name, signal_dates)
        if verbose:
            print(f"  长表行数: {len(prequery)}", flush=True)

        # 4b. 时变成分股逐期过滤
        if pit_universe is not None:
            n0 = len(prequery)
            prequery = filter_long_by_pit_universe(prequery, pit_universe)
            if verbose:
                print(f"  成分股过滤: {n0} → {len(prequery)} 行")

        # 4c. 中性化（去极值→标准化→行业/市值中性化）+ 诊断收集
        neu_stats = None
        if sp.use_industry or sp.use_mktcap:
            mktcap_col = None
            if sp.use_mktcap:
                mktcap_wide = fetch_daily_wide("A股流通市值(元)", universe=universe,
                                               start_date=fetch_start,
                                               end_date=end_date,
                                               table_name=sp.table_name)
                if not mktcap_wide.empty:
                    log_mktcap = np.log(mktcap_wide.replace(0, np.nan))
                    lm_long = wide_to_prequery(log_mktcap, "log_mktcap", signal_dates)
                    prequery = prequery.merge(
                        lm_long[['input_ts', 'code', 'log_mktcap']],
                        on=['input_ts', 'code'], how='left')
                    mktcap_col = 'log_mktcap'
            prequery, neu_stats = self._preprocess_with_stats(
                prequery, sp.name,
                industry_scheme=sp.industry_scheme if sp.use_industry else None,
                mktcap_col=mktcap_col, verbose=verbose)

        # 5. 分组
        if verbose:
            print(f"  开始分组: n_quantiles={n_quantiles}", flush=True)
        labeled = single_characteristic(prequery, sp.name, {sp.name: n_quantiles})
        if verbose:
            print(f"  完成分组: {len(labeled)} 行", flush=True)
        factor_values = _labeled_to_factor_values(labeled, sp.name)

        # 5a. Profiling 体检：基于 single_characteristic 返回的全量 n 分组矩阵
        #     （不是最终持仓股票子集），便于检查所有可选股票的分组分布。
        profiling = None
        if include_profiling:
            if verbose:
                print("  开始 Profiling: 全量分组矩阵", flush=True)
            grouped_wide = labeled[sp.name].unstack('code')
            profiling = self._run_profiling(grouped_wide, sp.name, output_dir, verbose)
            profiling.update(append_grouped_profiling_excel(output_dir, sp.name, labeled))
            if verbose:
                print("  完成 Profiling", flush=True)

        # 6. 权重
        long_groups, short_groups = self._resolve_groups(n_quantiles)
        if verbose:
            print(f"  开始生成权重: long={long_groups}, short={short_groups}", flush=True)
        weights = get_single_factor_weight(labeled, {
            'factor_key': sp.name, 'mode': sp.weight_mode,
            'long': long_groups, 'short': short_groups,
            'group_weights': sp.group_weights,
            'intra_group_allocation': sp.intra_group_allocation,
        })
        weights = _expand_weights_to_factor_universe(weights, factor_values)
        if verbose:
            print(f"  完成生成权重: {weights.shape}", flush=True)
        pit_validation = validate_weights_in_pit_universe(weights, pit_universe)
        if verbose and pit_universe is not None and not pit_validation.empty:
            bad = int((~pit_validation['passed']).sum())
            print(f"  PIT 权重校验: {len(pit_validation)-bad}/{len(pit_validation)} 期通过")
            if bad:
                print("  [WARN] 存在调仓股票不在当期 PIT 股票池内，请检查 pit_validation")
        if "cash" not in weights.columns:
            weights = weights.copy()
            weights["cash"] = 1.0 - weights.sum(axis=1)
        weights.index = weights.index + pd.Timedelta(minutes=10)

        # 7. 回测
        if verbose:
            print("  开始回测", flush=True)
        bt = BacktestBase(weights, metric=sp.backtest_metric, symbol=sp.name,
                          amount=initial_amount, time_tolerance=24 * 11)
        if verbose:
            print("  完成回测", flush=True)

        # 8. 绩效评价：Analyst 门面一键出全指标分组表 + Excel
        if verbose:
            print("  开始生成报告", flush=True)
        analyst = Analyst.from_backtest(
            bt,
            name=sp.name,
            benchmark_code=benchmark_code,
            benchmark_metric=sp.backtest_metric,
            factor_values=factor_values,
        )
        summary = analyst.report(
            to_excel=f"{output_dir}/{sp.name}_report.xlsx",
        )
        if verbose:
            print(f"  {sp.name} 指标项数: {len(summary)}  报告: {sp.name}_report.xlsx")

        # 9. 静态图输出（替代交互 HTML）
        if verbose:
            print("  开始生成静态图", flush=True)
        import betalens.analyst.plotting as _P

        trade_pairs = _match_trade_pairs(bt.rebalance_log)
        group_nav = pd.DataFrame()

        if sp.strategy_type == 'cross_section':
            if verbose:
                print(f"  跑各分组净值回测 (n_quantiles={n_quantiles})", flush=True)
            group_nav = _compute_group_nav(bt, factor_values, n_quantiles)
            if not group_nav.empty:
                img = _P.plot_group_nav(
                    group_nav,
                    title=f'{sp.name} {n_quantiles}分组净值曲线',
                    n_quantiles=n_quantiles,
                )
                with open(f'{output_dir}/{sp.name}_group_nav.png', 'wb') as _f:
                    _f.write(img)
                if verbose:
                    print(f"  已保存: {sp.name}_group_nav.png", flush=True)

        elif sp.strategy_type == 'timing':
            img = _P.plot_timing_nav_with_trades(
                bt.nav,
                trade_pairs,
                title=f'{sp.name} 净值曲线（含买卖点）',
            )
            with open(f'{output_dir}/{sp.name}_timing_nav.png', 'wb') as _f:
                _f.write(img)
            if verbose:
                print(f"  已保存: {sp.name}_timing_nav.png", flush=True)

        if not trade_pairs.empty:
            img_annual = _P.plot_annual_trade_performance(
                trade_pairs,
                title=f'{sp.name} 分年度交易表现',
            )
            with open(f'{output_dir}/{sp.name}_annual.png', 'wb') as _f:
                _f.write(img_annual)
            if verbose:
                print(f"  已保存: {sp.name}_annual.png", flush=True)

        if verbose:
            print("  静态图生成完毕", flush=True)

        if dump_excel:
            dump_path = f'{output_dir}/{sp.name}_dump.xlsx'
            bt.dump_to_excel(dump_path)
            # 追加因子值 sheet：dump_to_excel 是 betalens 通用回测方法（只 dump bt
            # 自身属性），因子值/分组是因子层产物，故在此用 append 模式补写。
            with pd.ExcelWriter(dump_path, engine='openpyxl', mode='a',
                                if_sheet_exists='replace') as writer:
                factor_values.to_excel(writer, sheet_name='factor_values', index=False)

        return RunResult(backtest=bt, analyst=analyst,
                         profiling=profiling, neutralize_stats=neu_stats,
                         factor_values=factor_values,
                         pit_validation=pit_validation,
                         chart_data={
                             'group_nav': group_nav,
                             'trade_pairs': trade_pairs,
                         })
