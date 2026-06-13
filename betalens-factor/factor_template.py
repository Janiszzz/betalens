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
from typing import Callable, Any

import pandas as pd
import numpy as np

from betalens.datafeed import (
    Datafeed, get_absolute_trade_days, get_index_universe,
)
from betalens.factor.factor import single_characteristic, get_single_factor_weight
from betalens.factor.preprocessing import (
    winsorize_factor, standardize_factor, neutralize_factor,
    query_industry_panel,
)
from datafeed.validation import fix_null_values, FillStrategy
from betalens.factor.profiling import (
    describe_distribution, coverage_stats, detect_outliers,
    factor_autocorrelation, factor_turnover, distribution_stability,
)
from betalens.backtest import BacktestBase
from betalens.analyst import Analyst

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DB_TABLE = "daily_market"


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
    pit = {}
    try:
        for d in signal_dates:
            date_str = pd.Timestamp(d).strftime('%Y-%m-%d')
            codes = get_index_universe(data.cursor, index_code, date_str)
            pit[d] = set(codes)
    finally:
        data.close()
    return pit


def filter_long_by_pit_universe(long_df, pit_universe):
    """按 point-in-time 成分股逐期过滤长表。

    某信号日成分股为空（指数无快照）时该期不过滤，避免误删全部样本。
    """
    if not pit_universe:
        return long_df

    def _keep(row):
        members = pit_universe.get(row['input_ts'].date())
        if not members:
            return True
        return row['code'] in members

    mask = long_df.apply(_keep, axis=1)
    return long_df.loc[mask].reset_index(drop=True)


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
    backtest_metric: str = "收盘价(元)"


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
        if sp.long_groups is not None or sp.short_groups is not None:
            return sp.long_groups or [], sp.short_groups or []
        if sp.direction == "positive":
            return [n_q - 1], []
        if sp.direction == "negative":
            return [0], []
        raise ValueError(f"未知 direction: {sp.direction}")

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
        """因子值体检：分布/时变/稳定性指标 + 6 子图 PNG。"""
        results = {
            'distribution': describe_distribution(factor_wide),
            'coverage': coverage_stats(factor_wide),
            'outliers': detect_outliers(factor_wide),
            'autocorrelation': factor_autocorrelation(factor_wide),
            'turnover': factor_turnover(factor_wide),
            'stability': distribution_stability(factor_wide),
        }

        excel_path = f"{output_dir}/{name}_profiling.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            for sheet_name, df in results.items():
                d = df.to_frame() if isinstance(df, pd.Series) else df
                d.to_excel(writer, sheet_name=sheet_name)

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'{name} Factor Profiling', fontsize=14)

        ax = axes[0, 0]
        cov = results['coverage']
        ax.plot(cov.index, cov['覆盖率'])
        ax.set_title('Coverage'); ax.set_ylim(0, 1.05)

        ax = axes[0, 1]
        out = results['outliers']
        out_ts = out.drop('Total') if 'Total' in out.index else out
        ax.bar(range(len(out_ts)), out_ts['极值占比'].values, width=1)
        ax.set_title('Outlier ratio')

        ax = axes[1, 0]
        ac = results['autocorrelation']
        ax.bar(ac.index.astype(str), ac['自相关均值'])
        ax.set_title('Rank autocorr'); ax.set_xlabel('lag')

        ax = axes[1, 1]
        to = results['turnover']
        ax.plot(to.index, to.values)
        ax.set_title('Top 20% turnover')

        ax = axes[2, 0]
        stab = results['stability']
        ax.plot(stab.index, stab['mean'], label='mean')
        ax2 = ax.twinx()
        ax2.plot(stab.index, stab['std'], color='orange', label='std')
        ax.set_title('Distribution drift (mean/std)')

        ax = axes[2, 1]
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
            output_dir: str = ".",
            extra_inputs: dict[str, pd.DataFrame] | None = None,
            include_profiling: bool = True,
            verbose: bool = True) -> RunResult:
        """运行完整管线: 取数 → 算子 → [profiling] → 中性化 → 分组 → 权重 → 回测 → 报告

        返回 RunResult（可解包为 bt, analyst 向后兼容）。
        股票池：index_code 给定时逐期 PIT 成分股过滤（防前视）；否则用静态 universe。
        中性化：use_industry / use_mktcap 控制，诊断收入 RunResult.neutralize_stats。
        """
        sp = self.spec

        rebalance_dates = get_absolute_trade_days(start_date, end_date,
                                                  rebal_freq, use_pmc=False)
        all_trade_days = get_absolute_trade_days(start_date, end_date,
                                                 "D", use_pmc=False)
        if verbose:
            print(f"调仓日数量: {len(rebalance_dates)}")

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
            pit_universe = build_pit_universe(signal_dates, sp.index_code)
            universe = sorted({c for codes in pit_universe.values() for c in codes})
            if verbose:
                print(f"  {sp.index_code} 成分股并集: {len(universe)} 只 (逐期 PIT 过滤)")
        elif universe is None:
            raise ValueError("未设 index_code 时必须传入静态 universe")

        # 2. 批量抓宽表
        wides = {}
        for arg_name, metric in sp.inputs.items():
            w = fetch_daily_wide(metric, universe=universe,
                                 start_date=start_date, end_date=end_date,
                                 table_name=sp.table_name)
            if verbose:
                print(f"  {arg_name} ({metric}): {w.shape}")
            wides[arg_name] = w
        if extra_inputs:
            wides.update(extra_inputs)

        # 3. 调用算子
        factor_wide = sp.compute(**wides, **sp.compute_kwargs)

        # 3a. Profiling 体检（中性化之前，反映原始因子分布）
        profiling = None
        if include_profiling:
            profiling = self._run_profiling(factor_wide, sp.name, output_dir, verbose)

        # 4. 宽 → 长（仅信号日）
        prequery = wide_to_prequery(factor_wide, sp.name, signal_dates)

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
                mktcap_wide = fetch_daily_wide("市值", universe=universe,
                                               start_date=start_date,
                                               end_date=end_date,
                                               table_name="fundamentals")
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
        labeled = single_characteristic(prequery, sp.name, {sp.name: n_quantiles})

        # 6. 权重
        long_groups, short_groups = self._resolve_groups(n_quantiles)
        weights = get_single_factor_weight(labeled, {
            'factor_key': sp.name, 'mode': sp.weight_mode,
            'long': long_groups, 'short': short_groups,
        })
        weights.index = weights.index + pd.Timedelta(minutes=10)

        # 7. 回测
        bt = BacktestBase(weights, metric=sp.backtest_metric, symbol=sp.name,
                          amount=initial_amount, time_tolerance=24 * 11)

        # 8. 绩效评价：Analyst 门面一键出全指标分组表 + Excel + 交互 HTML
        analyst = Analyst.from_backtest(bt, name=sp.name)
        summary = analyst.report(
            to_excel=f"{output_dir}/{sp.name}_report.xlsx",
            to_html=f"{output_dir}/{sp.name}_report.html",
        )
        if verbose:
            print(f"  {sp.name} 指标项数: {len(summary)}  报告: {sp.name}_report.xlsx / .html")

        bt.dump_to_excel(f'{output_dir}/{sp.name}_dump.xlsx')
        return RunResult(backtest=bt, analyst=analyst,
                         profiling=profiling, neutralize_stats=neu_stats)
