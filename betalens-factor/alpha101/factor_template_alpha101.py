#%%
"""
alpha101 类因子专用模板

在通用 factor_template 之上，补齐 **WorldQuant 101 公式化 alpha** 常用算子。
这是 alpha101 类因子的唯一公共依赖：同类下的 factor_<NAME>.py 只
`from factor_template_alpha101 import FactorSpec, FactorPipeline, delta, sign, ...`，
取数 / 分组 / 权重 / 回测 / 评价主干全部复用通用 factor_template.FactorPipeline。

—— 复用（直接 re-export，口径与其它类保持一致）——
    FactorSpec / FactorPipeline / RunResult  ← 通用 factor_template

—— 本类独有 API：WorldQuant 表达式算子 ——
全部作用于 index=datetime、columns=code 的宽表 DataFrame。时序算子按列
（个股时间轴）滚动；截面算子（rank）按行（同一日截面）计算：

    delta(X, n)          → X.diff(n)                  # 时序差分
    delay(X, n)          → X.shift(n)                 # 时序滞后
    sign(X)              → np.sign(X)                 # 符号
    rank(X)              → X.rank(axis=1, pct=True)   # 截面百分位排名
    ts_rank(X, n)        → n 周期内当前值的时序百分位排名
    ts_min/ts_max(X, n)  → X.rolling(n).min()/.max()
    ts_sum(X, n)         → X.rolling(n).sum()
    correlation(X, Y, n) → X.rolling(n).corr(Y)       # 滚动相关
    covariance(X, Y, n)  → X.rolling(n).cov(Y)
    stddev(X, n)         → X.rolling(n).std()
    clean_inf(X)         → X.replace([inf,-inf], nan) # 算子末尾统一清理

使用示例（最小例）：
    from factor_template_alpha101 import FactorSpec, FactorPipeline, sign, delta, clean_inf

    def compute_alpha12(close_wide, volume_wide):
        return clean_inf(sign(delta(volume_wide, 1)) * (-1 * delta(close_wide, 1)))
"""
from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path
from typing import Any, Callable

# 通用核心在 betalens-factor/ 根；保证可被 import（脚本独立运行 / dashboard 加载皆可）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DB_TABLE = "daily_market"


@dataclass
class FactorSpec:
    name: str
    inputs: dict[str, str]
    compute: Callable[..., Any]
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


class FactorPipeline:
    def __init__(self, spec: FactorSpec):
        self.spec = spec

    def run(self, *args, **kwargs):
        print("  加载通用 FactorPipeline", flush=True)
        from factor_template import FactorPipeline as CorePipeline, FactorSpec as CoreFactorSpec

        print("  通用 FactorPipeline 已加载", flush=True)
        core_spec = CoreFactorSpec(**self.spec.__dict__)
        return CorePipeline(core_spec).run(*args, **kwargs)


RunResult = Any

__all__ = [
    "FactorSpec", "FactorPipeline", "RunResult",
    "delta", "delay", "sign", "rank", "ts_rank",
    "ts_min", "ts_max", "ts_sum", "correlation", "covariance",
    "stddev", "clean_inf",
]


def delta(x, n=1):
    """X 相对 n 周期前的时序差分。"""
    return x.diff(n)


def delay(x, n=1):
    """X 的 n 周期时序滞后值。"""
    return x.shift(n)


def sign(x):
    """逐元素符号（-1/0/1）。"""
    import numpy as np

    return np.sign(x)


def rank(x):
    """同一日截面（按行）百分位排名，范围 (0,1]。"""
    return x.rank(axis=1, pct=True)


def ts_rank(x, n):
    """n 周期窗口内当前值的时序百分位排名。"""
    import pandas as pd

    return x.rolling(n).apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=True)


def ts_min(x, n):
    """n 周期内时序最小值。"""
    return x.rolling(n).min()


def ts_max(x, n):
    """n 周期内时序最大值。"""
    return x.rolling(n).max()


def ts_sum(x, n):
    """n 周期内时序求和。"""
    return x.rolling(n).sum()


def correlation(x, y, n):
    """X 与 Y 的 n 周期滚动相关系数（逐列）。"""
    return x.rolling(n).corr(y)


def covariance(x, y, n):
    """X 与 Y 的 n 周期滚动协方差（逐列）。"""
    return x.rolling(n).cov(y)


def stddev(x, n):
    """n 周期滚动标准差。"""
    return x.rolling(n).std()


def clean_inf(x):
    """把 ±inf 置为 NaN（算子末尾统一调用，防止除零污染）。"""
    return x.replace([float("inf"), float("-inf")], float("nan"))
