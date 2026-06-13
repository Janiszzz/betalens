#%%
"""
ILLIQ_v2 — 用 factor_template 重写的等价版本，供与原 factor_ILLIQ.py 对账。

公式: ILLIQ = mean(|r| / amount, N)   (N=20)
来源: 中信建投高频和行为金融学因子周报 — 流动性类
因子方向: 正向（高分组做多）
"""
import numpy as np
import pandas as pd
from factor_template import FactorSpec, FactorPipeline


def compute_ILLIQ(close_wide, amount_wide, window=20):
    ret = close_wide.pct_change().abs()
    illiq_daily = (ret / amount_wide).replace([np.inf, -np.inf], np.nan)
    return illiq_daily.rolling(window, min_periods=10).mean()


spec = FactorSpec(
    name="ILLIQ_v2",
    inputs={
        "close_wide": "收盘价(元)",
        "amount_wide": "成交金额(元)",
    },
    compute=compute_ILLIQ,
    direction="positive",
    compute_kwargs={"window": 20},
)


if __name__ == "__main__":
    FactorPipeline(spec).run(
        start_date="2024-01-01",
        end_date="2025-12-31",
        rebal_freq="D",
        n_quantiles=20,
    )
