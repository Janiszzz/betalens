#%%
"""
RSI_SLOW  —  tdx 类因子

公式: RSI_SLOW = SMA(MAX(CLOSE-REF(CLOSE,1),0),12,1) / SMA(|CLOSE-REF(CLOSE,1)|,12,1) * 100
逻辑: 12 日慢速 RSI；高值=中期上涨动量强 → 高分组做多（positive）。与 RSI_FAST 同公式族，仅 window 不同。
来源: 通达信(TDX)指标公式：吸筹能量 + RSI 快慢线
方向: 正向（高分组做多）

本脚本由 factor-forge/scaffold.py 生成；只定义算子与 FactorSpec，
取数 / 分组 / 权重 / 回测全部复用 factor_template.FactorPipeline。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CLASS_DIR = Path(__file__).resolve().parent.parent   # tdx/
sys.path.insert(0, str(_CLASS_DIR))                   # tdx/（类模板所在）
from factor_template_tdx import FactorSpec, FactorPipeline, SMA, REF, clean_inf


def compute_rsi_slow(close_wide, window=12):
    """RSI_SLOW = SMA(MAX(CLOSE-REF(CLOSE,1),0),12,1) / SMA(|CLOSE-REF(CLOSE,1)|,12,1) * 100"""
    diff = close_wide - REF(close_wide, 1)
    up = SMA(diff.clip(lower=0), window, 1)
    ab = SMA(diff.abs(), window, 1)
    return clean_inf(up / ab * 100)


spec = FactorSpec(
    name="RSI_SLOW",
    inputs={"close_wide": "收盘价(元)"},
    compute=compute_rsi_slow,
    direction="positive",
    compute_kwargs={"window": 12},
    use_industry=True,
    index_code="000906.SH",
)


if __name__ == "__main__":
    FactorPipeline(spec).run(
        start_date="2024-01-01",
        end_date="2025-12-31",
        rebal_freq="D",
        n_quantiles=20,
        output_dir=str(Path(__file__).resolve().parent),
    )
