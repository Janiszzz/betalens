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

    spec = FactorSpec(name="DISP", inputs={"close_wide": "close"},
                      compute=compute_disp, direction="negative",
                      compute_kwargs={"window": 252}, index_code="000906.SH")
    LiqDemandPipeline(spec).run("2024-01-01", "2025-12-31",
                                warmup_days=400, pretom_only=True)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# 通用核心在 betalens-factor/ 根；保证可被 import（脚本独立运行 / dashboard 加载皆可）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from factor_template import FactorSpec, FactorPipeline, RunResult, infer_warmup_days
from betalens.datafeed import get_absolute_trade_days

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
    all_days = get_absolute_trade_days(start_date, end_date, "D")
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

    主干（取数→算子→profiling→中性化→分组→权重→回测→报告）复用通用实现，
    仅插入两处本类逻辑，便于与 alpha101/tdx 口径对齐：
      1. 取数区间向前扩 warmup_days 天，使 rolling(window) 在回测首日已就绪；
      2. pretom_only=True 时，把非 PreTOM 交易日的权重整行清零（空仓）。
    """

    def run(self, start_date: str, end_date: str, *,
            rebal_freq: str = "D", warmup_days: int | None = 400,
            pretom_only: bool = True, pretom_lo: int = 9, pretom_hi: int = 4,
            **kwargs) -> RunResult:
        if not 1 <= int(pretom_hi) <= int(pretom_lo):
            raise ValueError("PreTOM 要求 1 <= pretom_hi <= pretom_lo")
        warmup = max(400 if warmup_days is None else int(warmup_days),
                     infer_warmup_days(self.spec.compute_kwargs))

        def transform(weights):
            if not pretom_only:
                return weights
            # Complete boundary months are needed even for partial-month runs.
            first = pd.Timestamp(start_date).replace(day=1).strftime("%Y-%m-%d")
            last = (pd.Timestamp(end_date) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
            dates = get_pretom_dates(first, last, lo=pretom_lo, hi=pretom_hi)
            result = weights.copy()
            stocks = [code for code in result if code != "cash"]
            keep = [ts.date() in dates for ts in result.index]
            result.loc[np.logical_not(keep), stocks] = 0.0
            result["cash"] = 1.0 - result[stocks].sum(axis=1)
            return result

        kwargs.setdefault("report_html", True)
        return super().run(start_date, end_date, rebal_freq=rebal_freq,
                           warmup_days=warmup, weight_transform=transform, **kwargs)
