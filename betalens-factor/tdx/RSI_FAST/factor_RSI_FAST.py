#%%
"""
RSI_FAST  —  tdx 类因子

公式: RSI_FAST = SMA(MAX(CLOSE-REF(CLOSE,1),0),3,1) / SMA(|CLOSE-REF(CLOSE,1)|,3,1) * 100
逻辑: 3 日快速 RSI；高值=近期上涨动量强 → 高分组做多（positive，按用户选定动量方向）。
来源: 通达信(TDX)指标公式：吸筹能量 + RSI 快慢线
方向: 正向（高分组做多）

本脚本由 factor-forge/scaffold.py 生成；只定义算子与 FactorSpec，
取数 / 分组 / 权重 / 回测全部复用 factor_template.FactorPipeline。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CLASS_DIR = Path(__file__).resolve().parent.parent   # tdx/
sys.path.insert(0, str(_CLASS_DIR.parent))            # betalens-factor/
from factor_template import FactorSpec, FactorPipeline

_SPEC_FILE = _CLASS_DIR / "spec_tdx.json"

def _load_defaults():
    cfg = json.loads(_SPEC_FILE.read_text(encoding="utf-8"))
    d = cfg["defaults"]
    factor_cfg = next(f for f in cfg["factors"] if f["name"] == "RSI_FAST")
    return d, factor_cfg


def compute_rsi_fast(close_wide, window=3):
    """RSI_FAST = SMA(MAX(CLOSE-REF(CLOSE,1),0),3,1) / SMA(|CLOSE-REF(CLOSE,1)|,3,1) * 100"""
    diff = close_wide - close_wide.shift(1)
    up = diff.clip(lower=0).ewm(alpha=1.0/window, adjust=False).mean()
    ab = diff.abs().ewm(alpha=1.0/window, adjust=False).mean()
    return (up / ab * 100).replace([np.inf, -np.inf], np.nan)


_defaults, _factor_cfg = _load_defaults()

spec = FactorSpec(
    name="RSI_FAST",
    inputs=_factor_cfg["inputs"],
    compute=compute_rsi_fast,
    direction=_defaults["direction"],
    compute_kwargs=_factor_cfg.get("compute_kwargs", {}),
    use_industry=_defaults["use_industry"],
    index_code=_defaults["index_code"],
)


if __name__ == "__main__":
    FactorPipeline(spec).run(
        start_date=_defaults["start_date"],
        end_date=_defaults["end_date"],
        rebal_freq=_defaults["rebal_freq"],
        n_quantiles=_defaults["n_quantiles"],
        output_dir=str(Path(__file__).resolve().parent),
    )
