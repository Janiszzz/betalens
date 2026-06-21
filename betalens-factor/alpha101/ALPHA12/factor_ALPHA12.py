#%%
"""Alpha#12 = sign(delta(volume, 1)) * (-1 * delta(close, 1))

来源: WorldQuant 101 Formulaic Alphas (Kakushadze, 2016), Appendix A.1
逻辑: 量增价跌 / 量缩价涨 → 因子值高（看多）；方向 positive。

本脚本设置：
    股票池   中证800（000906.SH），逐信号日 point-in-time 取成分股，时变
    中性化   申万一级行业中性化（preprocess_factor 自动查 industry 表）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import logging

# 压制 point-in-time 成分股查询的逐日 INFO 日志（每个信号日一条，过于啰嗦）
logging.getLogger("IndexUniverseQuery").setLevel(logging.WARNING)

_CLASS_DIR = Path(__file__).resolve().parent.parent   # alpha101/
sys.path.insert(0, str(_CLASS_DIR))                   # alpha101/（类模板所在）
from factor_template_alpha101 import (
    FactorSpec, FactorPipeline, sign, delta, clean_inf,
)


def compute_alpha12(close_wide, volume_wide):
    return clean_inf(sign(delta(volume_wide, 1)) * (-1 * delta(close_wide, 1)))


spec = FactorSpec(
    name="ALPHA12",
    inputs={"close_wide": "收盘价(元)", "volume_wide": "成交量(股)"},
    compute=compute_alpha12,
    direction="positive",
    index_code="000906.SH",          # 中证800，时变成分股
    use_industry=True,               # 申万一级行业中性化
    industry_scheme="申万一级行业",
    backtest_metric="开盘价(元)",     # alpha101 类沿用开盘价撮合
)


if __name__ == "__main__":
    FactorPipeline(spec).run(
        "2024-01-01", "2025-12-31",
        n_quantiles=20,
        output_dir=str(Path(__file__).resolve().parent),
    )
