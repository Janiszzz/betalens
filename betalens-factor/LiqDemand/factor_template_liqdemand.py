#%%
"""
LiqDemand 类因子专用模板 —— 复现 Nathan & Suominen (2026)
《The Liquidity-Demand Component of the Factor Zoo》(SSRN 6909918)

本类聚焦论文的两件可在 A 股落地的核心构造：

  1. dispensability 因子（公式 4）：δ = − z_m( P / max_{过去252日} P )，
     即「距 52 周高点的距离」。离高点越远 → 越"可处置"(dispensable) → 月末越易被抛售。
     仅需后复权收盘价。

  2. PreTOM 窗口择时（论文核心创新）：月末前为筹现金的抛压集中在
     PreTOM = [τ−9, τ−4]（τ=当月最后交易日，倒数第 9~第 4 个交易日，共 6 天）。
     本类把日频多空组合改成「仅在 PreTOM 窗口持仓、其余空仓」的日历叠加。

—— 输入 / 输出 / 管线约定 ——
  输入：单一后复权收盘价宽表（index=datetime, columns=code）。
  输出：同形状因子宽表（截面 zscore 由通用中性化流程统一处理，本类不强制中性化）。
  管线：复用通用 factor_template.FactorPipeline 主干（取数/算子/中性化/分组/权重/回测），
        仅在两处扩展 —— 见 LiqDemandPipeline：
          (a) warmup_days：取数起始提前，保证 rolling(252) 在回测首日已就绪（解决预热期）；
          (b) pretom_only：权重表生成后，把非 PreTOM 交易日的整行权重清零（空仓）。

—— 复用（直接 re-export）——
    FactorSpec / RunResult ← 通用 factor_template

—— 本类独有 API ——
    get_pretom_dates(start, end, lo=9, hi=4)  → 每月 [τ−lo, τ−hi] 交易日集合
    LiqDemandPipeline(spec).run(..., warmup_days=400, pretom_only=True)

使用示例（最小例）：
    from factor_template_liqdemand import FactorSpec, LiqDemandPipeline

    def compute_disp(close_wide, window=252):
        ratio = close_wide / close_wide.rolling(window, min_periods=120).max()
        return (-ratio).replace([np.inf, -np.inf], np.nan)

    spec = FactorSpec(name="DISP", inputs={"close_wide": "收盘价(元)"},
                      compute=compute_disp, direction="negative",
                      compute_kwargs={"window": 252}, index_code="000906.SH")
    LiqDemandPipeline(spec).run("2024-01-01", "2025-12-31",
                                warmup_days=400, pretom_only=True)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 通用核心在 betalens-factor/ 根；保证可被 import（脚本独立运行 / dashboard 加载皆可）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from factor_template import (  # noqa: E402  re-export 通用主干
    FactorSpec, FactorPipeline, RunResult,
    fetch_daily_wide, wide_to_prequery, build_pit_universe,
    filter_long_by_pit_universe, infer_warmup_days,
    validate_weights_in_pit_universe, _labeled_to_factor_values,
    append_grouped_profiling_excel,
)
from betalens.datafeed import get_absolute_trade_days  # noqa: E402
from betalens.factor.factor import (  # noqa: E402
    single_characteristic, get_single_factor_weight,
)
from betalens.backtest import BacktestBase  # noqa: E402
from betalens.analyst import Analyst  # noqa: E402

__all__ = [
    "FactorSpec", "RunResult", "FactorPipeline",
    "LiqDemandPipeline", "get_pretom_dates", "clean_inf",
]


def clean_inf(x):
    """把 ±inf 置为 NaN（算子末尾统一调用）。"""
    return x.replace([np.inf, -np.inf], np.nan)


def get_pretom_dates(start_date: str, end_date: str,
                     lo: int = 9, hi: int = 4) -> set[date]:
    """每月 PreTOM 窗口 [τ−lo, τ−hi] 的交易日集合（论文 §2.2，默认倒数第 9~第 4，共 6 天）。

    τ 为当月最后一个交易日。按 (年,月) 分组后，取该月交易日序列的
    `month_days[-lo : -(hi-1) or None]`：
        -lo        = 倒数第 lo 个（含）
        -(hi-1)    = 倒数第 hi 个的下一个（不含），hi=4 时为 -3
        `or None`  = 当 hi==1 时 -(0)==0 会取空，需退化为 None 取到末尾
    返回 set[date]，供权重表掩码使用（窗口内保留、窗口外清零）。
    """
    all_days = get_absolute_trade_days(start_date, end_date, "D", use_pmc=False)
    by_month: dict[tuple[int, int], list[date]] = defaultdict(list)
    for d in sorted(all_days):
        by_month[(d.year, d.month)].append(d)

    pretom: set[date] = set()
    end_slice = -(hi - 1) if hi > 1 else None
    for days in by_month.values():
        if len(days) >= lo:                      # 该月交易日不足 lo 个则跳过（防越界）
            pretom.update(days[-lo:end_slice])
    return pretom


class LiqDemandPipeline(FactorPipeline):
    """LiqDemand 类管线：在通用 FactorPipeline 主干上扩展 warmup 预热 + PreTOM 择时。

    主干（取数→算子→profiling→中性化→分组→权重→回测→报告）逐字沿用通用实现，
    仅插入两处本类逻辑，便于与 alpha101/tdx 口径对齐：
      1. 取数区间向前扩 warmup_days 天，使 rolling(window) 在回测首日已就绪；
      2. pretom_only=True 时，把非 PreTOM 交易日的权重整行清零（空仓）。
    """

    def run(self, start_date: str, end_date: str, *,
            rebal_freq: str = "D",
            warmup_days: int = 400,
            pretom_only: bool = True,
            pretom_lo: int = 9,
            pretom_hi: int = 4,
            universe: list | None = None,
            n_quantiles: int = 20,
            initial_amount: float = 1e8,
            output_dir: str = ".",
            extra_inputs: dict[str, pd.DataFrame] | None = None,
            include_profiling: bool = True,
            dump_excel: bool = True,
            verbose: bool = True) -> RunResult:
        sp = self.spec

        # 取数起始向前扩：显式 warmup_days 与自动窗口推断取较大值。
        effective_warmup = max(int(warmup_days), infer_warmup_days(sp.compute_kwargs))
        fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=effective_warmup)).strftime("%Y-%m-%d")

        # 调仓日 / 信号日：始终落在 [start_date, end_date]（不含预热期）
        rebalance_dates = get_absolute_trade_days(start_date, end_date, rebal_freq, use_pmc=False)
        all_trade_days = get_absolute_trade_days(fetch_start, end_date, "D", use_pmc=False)
        if verbose:
            print(f"取数起始(含预热): {fetch_start}  warmup_days={effective_warmup}  调仓日数量: {len(rebalance_dates)}")

        # 0. 信号日 = 调仓日前一交易日（与通用主干一致，防前视）
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

        # 2. 批量抓宽表（用 fetch_start 取到预热期，rolling 才完整）
        wides = {}
        for arg_name, metric in sp.inputs.items():
            w = fetch_daily_wide(metric, universe=universe,
                                 start_date=fetch_start, end_date=end_date,
                                 table_name=sp.table_name)
            if verbose:
                print(f"  {arg_name} ({metric}): {w.shape}")
            wides[arg_name] = w
        if extra_inputs:
            wides.update(extra_inputs)

        # 3. 调用算子（在含预热期的完整宽表上算，rolling(252) 在 start_date 已就绪）
        factor_wide = sp.compute(**wides, **sp.compute_kwargs)

        # 4. 宽 → 长（仅 signal_dates，预热期自然被排除）
        prequery = wide_to_prequery(factor_wide, sp.name, signal_dates)

        # 4b. 时变成分股逐期过滤
        if pit_universe is not None:
            n0 = len(prequery)
            prequery = filter_long_by_pit_universe(prequery, pit_universe)
            if verbose:
                print(f"  成分股过滤: {n0} → {len(prequery)} 行")

        # 4c. 中性化（去极值→标准化→行业/市值中性化）+ 诊断；
        #     未开 use_industry/use_mktcap 时仅由分组前的 single_characteristic 处理，
        #     截面 zscore 在 _preprocess_with_stats 内做（与通用主干一致）。
        neu_stats = None
        if sp.use_industry or sp.use_mktcap:
            mktcap_col = None
            if sp.use_mktcap:
                mktcap_wide = fetch_daily_wide("A股流通市值(元)", universe=universe,
                                               start_date=fetch_start, end_date=end_date,
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
        labeled = single_characteristic(prequery, sp.name, {sp.name: n_quantiles})
        factor_values = _labeled_to_factor_values(labeled, sp.name)

        # 5a. Profiling：基于 single_characteristic 的全量 n 分组矩阵。
        profiling = None
        if include_profiling:
            grouped_wide = labeled[sp.name].unstack('code')
            profiling = self._run_profiling(grouped_wide, sp.name, output_dir, verbose)
            profiling.update(append_grouped_profiling_excel(output_dir, sp.name, labeled))

        # 6. 权重（日频多空）
        long_groups, short_groups = self._resolve_groups(n_quantiles)
        weights = get_single_factor_weight(labeled, {
            'factor_key': sp.name, 'mode': sp.weight_mode,
            'long': long_groups, 'short': short_groups,
        })
        pit_validation = validate_weights_in_pit_universe(weights, pit_universe)
        if verbose and pit_universe is not None and not pit_validation.empty:
            bad = int((~pit_validation['passed']).sum())
            print(f"  PIT 权重校验: {len(pit_validation)-bad}/{len(pit_validation)} 期通过")
            if bad:
                print("  [WARN] 存在调仓股票不在当期 PIT 股票池内，请检查 pit_validation")
        weights.index = weights.index + pd.Timedelta(minutes=10)

        # 6b. PreTOM 择时掩码：非窗口交易日整行清零（空仓）。论文核心日历叠加。
        if pretom_only:
            pretom = get_pretom_dates(start_date, end_date, lo=pretom_lo, hi=pretom_hi)
            keep = np.array([ts.date() in pretom for ts in weights.index])
            n_keep = int(keep.sum())
            weights = weights.loc[keep]            # 仅保留 PreTOM 调仓日，其余日不持仓
            if verbose:
                print(f"  PreTOM zeshi: {keep.size} rebal-days -> keep {n_keep} "
                      f"([tau-{pretom_lo}, tau-{pretom_hi}], ~{n_keep / max(keep.size,1):.0%})")
            if weights.empty:
                raise ValueError("PreTOM 窗口内无调仓日，请检查区间或窗口参数")

        # 7. 回测
        bt = BacktestBase(weights, metric=sp.backtest_metric, symbol=sp.name,
                          amount=initial_amount, time_tolerance=24 * 11)

        # 8. 绩效评价
        analyst = Analyst.from_backtest(bt, name=sp.name)
        summary = analyst.report(
            to_excel=f"{output_dir}/{sp.name}_report.xlsx",
            to_html=f"{output_dir}/{sp.name}_report.html",
        )
        if verbose:
            print(f"  {sp.name} 指标项数: {len(summary)}  报告: {sp.name}_report.xlsx / .html")

        if dump_excel:
            dump_path = f'{output_dir}/{sp.name}_dump.xlsx'
            bt.dump_to_excel(dump_path)
            with pd.ExcelWriter(dump_path, engine='openpyxl', mode='a',
                                if_sheet_exists='replace') as writer:
                factor_values.to_excel(writer, sheet_name='factor_values', index=False)

        return RunResult(backtest=bt, analyst=analyst,
                         profiling=profiling, neutralize_stats=neu_stats,
                         factor_values=factor_values,
                         pit_validation=pit_validation)
