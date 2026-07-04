#%%
"""
tdx 类因子专用模板

在通用 factor_template 之上，补齐 **通达信(TDX) 技术指标口径** 的算子封装。
这是 tdx 类因子的唯一公共依赖：同类下的 factor_<NAME>.py 只
`from factor_template_tdx import FactorSpec, FactorPipeline, SMA, REF, ...`，
取数 / 分组 / 权重 / 回测 / 评价主干全部复用通用 factor_template.FactorPipeline。

—— 复用（直接 re-export，口径与其它类保持一致）——
    FactorSpec / FactorPipeline / RunResult  ← 通用 factor_template

—— 本类独有 API：TDX 技术指标算子 ——
betalens 无现成封装，统一用 pandas ewm/rolling 自实现（入出参均为
index=datetime、columns=code 的宽表 DataFrame）：

    SMA(X, N, M) → X.ewm(alpha=M/N, adjust=False).mean()   # TDX 加权移动平均
    EMA(X, N)    → X.ewm(span=N, adjust=False).mean()       # 指数移动平均
    MA(X, N)     → X.rolling(N).mean()                       # 简单移动平均
    REF(X, n)    → X.shift(n)                                # n 周期前的值
    LLV(X, N)    → X.rolling(N).min()                        # N 周期内最低
    HHV(X, N)    → X.rolling(N).max()                        # N 周期内最高
    clean_inf(X) → X.replace([inf,-inf], nan)               # 算子末尾统一清理

使用示例（最小例）：
    from factor_template_tdx import FactorSpec, FactorPipeline, SMA, REF, clean_inf

    def compute_rsi(close_wide, window=3):
        diff = close_wide - REF(close_wide, 1)
        up = SMA(diff.clip(lower=0), window, 1)
        ab = SMA(diff.abs(), window, 1)
        return clean_inf(up / ab * 100)
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
    "SMA", "EMA", "MA", "REF", "LLV", "HHV", "clean_inf",
]


def SMA(x, n, m=1):
    """TDX SMA(X,N,M)：加权移动平均，权重 alpha=M/N。"""
    return x.ewm(alpha=m / n, adjust=False).mean()


def EMA(x, n):
    """TDX EMA(X,N)：指数移动平均。"""
    return x.ewm(span=n, adjust=False).mean()


def MA(x, n):
    """TDX MA(X,N)：N 周期简单移动平均。"""
    return x.rolling(n).mean()


def REF(x, n=1):
    """TDX REF(X,n)：n 周期前的值。"""
    return x.shift(n)


def LLV(x, n):
    """TDX LLV(X,N)：N 周期内最低值。"""
    return x.rolling(n).min()


def HHV(x, n):
    """TDX HHV(X,N)：N 周期内最高值。"""
    return x.rolling(n).max()


def clean_inf(x):
    """把 ±inf 置为 NaN（算子末尾统一调用，防止除零污染）。"""
    return x.replace([float("inf"), float("-inf")], float("nan"))
