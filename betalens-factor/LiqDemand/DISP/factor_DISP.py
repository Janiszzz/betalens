#%%
"""DISP —— dispensability（可处置性）因子，复现 SSRN 6909918 公式 (4)

公式: δ_{i,m} = − z_m( P_{i,m} / max_{s∈过去252日} P_{i,s} )
      本脚本算 raw = −( P / rolling(252).max(P) )，截面 zscore 由分组前流程统一处理。

逻辑: P / 52周最高收盘价 ∈ (0,1]，贴近 1=接近高点。取负后，
      离高点越远（越"可处置"）因子值越高。论文发现这类股票在月末筹现金窗口
      被优先抛售而跑输 → 方向 negative（做空高分组=可处置股，做多低分组）。

输入: 后复权收盘价（rolling(252).max 需后复权，否则除权除息污染历史最高价）。

回测: 通过 LiqDemandPipeline 叠加 PreTOM 择时 —— 仅在每月 [τ−9, τ−4] 六个
      交易日持仓，其余空仓（论文核心日历创新）。warmup_days 提前取数预热 rolling(252)。

本脚本设置：
    股票池   中证800（000906.SH），逐信号日 point-in-time 取成分股，时变
    中性化   不做（论文用横截面 zscore，由分组前 single_characteristic 处理）
"""
import sys
import logging
from pathlib import Path

import numpy as np  # noqa: F401  供 compute 内使用

# 压制 point-in-time 成分股查询的逐日 INFO 日志
logging.getLogger("IndexUniverseQuery").setLevel(logging.WARNING)

_CLASS_DIR = Path(__file__).resolve().parent.parent   # LiqDemand/
sys.path.insert(0, str(_CLASS_DIR))
from factor_template_liqdemand import (  # noqa: E402
    FactorSpec, FactorPipeline, LiqDemandPipeline, clean_inf,
)


def compute_disp(close_wide, window=252):
    """δ_raw = −( P / 过去 window 日最高收盘价 )。

    min_periods=120：至少半年历史才出值，预热期不足的早期截面自然为 NaN。
    """
    ratio = close_wide / close_wide.rolling(window, min_periods=120).max()
    return clean_inf(-ratio)


spec = FactorSpec(
    name="DISP",
    inputs={"close_wide": "收盘价(元)"},
    compute=compute_disp,
    direction="positive",            # 可处置（高因子值）股做空，论文称月末跑输
    compute_kwargs={"window": 252},  # 52 周 ≈ 252 交易日
    index_code="000906.SH",          # 中证800，时变成分股
    backtest_metric="收盘价(元)",
)


if __name__ == "__main__":
    out = str(Path(__file__).resolve().parent)
    # PreTOM 择时版（论文核心）：仅月末前 [τ−9, τ−4] 六交易日持仓
    LiqDemandPipeline(spec).run(
        "2024-01-01", "2025-12-31",
        warmup_days=400,      # 预热 rolling(252)：取数提前约 1.1 年
        pretom_only=False,     # 改 False 即退化为普通日频多空（对照组）
        n_quantiles=20,
        output_dir=out,
    )
