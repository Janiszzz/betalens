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
    strategy_type: str = "cross_section"
    industry_inputs: dict[str, str] = field(default_factory=dict)
    required_history_bars: int = 0
    mask_inputs_by_pit: bool = False
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
    "FactorSpec", "FactorPipeline", "TimingFactorPipeline", "RunResult",
    "window", "delta", "delay", "sign", "rank", "ts_rank",
    "ts_min", "ts_max", "ts_sum", "correlation", "covariance",
    "ts_mean", "stddev", "product", "ts_argmax", "ts_argmin",
    "decay_linear", "scale", "signed_power", "indneutralize",
    "where", "elementwise_min", "elementwise_max", "clean_inf",
]


def window(value):
    """Convert a paper window to the plan's nearest-integer convention."""
    import math

    return max(1, int(math.floor(float(value) + 0.5)))


def delta(x, n=1):
    """X 相对 n 周期前的时序差分。"""
    return x.diff(window(n))


def delay(x, n=1):
    """X 的 n 周期时序滞后值。"""
    return x.shift(window(n))


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

    n = window(n)
    return x.rolling(n, min_periods=n).apply(
        lambda values: pd.Series(values).rank(method="average", pct=True).iloc[-1],
        raw=True,
    )


def ts_min(x, n):
    """n 周期内时序最小值。"""
    n = window(n)
    return x.rolling(n, min_periods=n).min()


def ts_max(x, n):
    """n 周期内时序最大值。"""
    n = window(n)
    return x.rolling(n, min_periods=n).max()


def ts_sum(x, n):
    """n 周期内时序求和。"""
    n = window(n)
    return x.rolling(n, min_periods=n).sum()


def ts_mean(x, n):
    n = window(n)
    return x.rolling(n, min_periods=n).mean()


def correlation(x, y, n):
    """X 与 Y 的 n 周期滚动相关系数（逐列）。"""
    n = window(n)
    return x.rolling(n, min_periods=n).corr(y)


def covariance(x, y, n):
    """X 与 Y 的 n 周期滚动协方差（逐列）。"""
    n = window(n)
    return x.rolling(n, min_periods=n).cov(y)


def stddev(x, n):
    """n 周期滚动标准差。"""
    n = window(n)
    return x.rolling(n, min_periods=n).std()


def product(x, n):
    import numpy as np

    n = window(n)
    return x.rolling(n, min_periods=n).apply(np.prod, raw=True)


def ts_argmax(x, n):
    import numpy as np

    n = window(n)
    return x.rolling(n, min_periods=n).apply(lambda values: float(np.argmax(values) + 1), raw=True)


def ts_argmin(x, n):
    import numpy as np

    n = window(n)
    return x.rolling(n, min_periods=n).apply(lambda values: float(np.argmin(values) + 1), raw=True)


def decay_linear(x, n):
    import numpy as np

    n = window(n)
    weights = np.arange(1.0, n + 1.0)
    weights /= weights.sum()
    return x.rolling(n, min_periods=n).apply(lambda values: float(np.dot(values, weights)), raw=True)


def scale(x, a=1.0):
    denominator = x.abs().sum(axis=1).replace(0.0, float("nan"))
    return x.div(denominator, axis=0) * float(a)


def signed_power(x, exponent):
    import numpy as np

    return np.sign(x) * np.power(np.abs(x), exponent)


def indneutralize(x, groups):
    import pandas as pd

    aligned_groups = groups.reindex(index=x.index, columns=x.columns)
    out = pd.DataFrame(float("nan"), index=x.index, columns=x.columns, dtype=float)
    for ts in x.index:
        values = x.loc[ts]
        labels = aligned_groups.loc[ts]
        valid = values.notna() & labels.notna()
        if not valid.any():
            continue
        centered = values[valid] - values[valid].groupby(labels[valid]).transform("mean")
        out.loc[ts, centered.index] = centered
    return out


def where(condition, when_true, when_false):
    import pandas as pd

    if not hasattr(when_true, "where"):
        when_true = pd.DataFrame(when_true, index=condition.index, columns=condition.columns)
    return when_true.where(condition, when_false)


def elementwise_min(x, y):
    import numpy as np
    import pandas as pd

    return pd.DataFrame(np.minimum(x, y), index=x.index, columns=x.columns)


def elementwise_max(x, y):
    import numpy as np
    import pandas as pd

    return pd.DataFrame(np.maximum(x, y), index=x.index, columns=x.columns)


def clean_inf(x):
    """把 ±inf 置为 NaN（算子末尾统一调用，防止除零污染）。"""
    return x.replace([float("inf"), float("-inf")], float("nan"))


def _timing_codes(params, universe):
    import re

    stock_code = str((params or {}).get("stock_code") or "").strip()
    if stock_code:
        return [stock_code]
    if isinstance(universe, str):
        values = [part for part in re.split(r"[,\s，;；]+", universe.strip()) if part]
    else:
        values = [str(value).strip() for value in (universe or []) if str(value).strip()]
    if len(values) != 1:
        raise ValueError("timing factor requires compute_kwargs.stock_code or one run.universe code")
    return values


def _with_timing_targets(pit_universe, target_codes):
    """Keep PIT constituents as formula context while always admitting targets."""
    targets = {str(code) for code in target_codes}
    return {
        day: {str(code) for code in members}.union(targets)
        for day, members in pit_universe.items()
    }


class TimingFactorPipeline:
    """Shared full-cross-section-to-single-stock Alpha101 timing pipeline."""

    def __init__(self, spec: FactorSpec):
        self.spec = spec

    def run(
        self,
        start_date: str,
        end_date: str,
        *,
        rebal_freq: str = "D",
        universe: list | None = None,
        n_quantiles: int = 10,
        initial_amount: float = 1e8,
        benchmark_code: str | None = None,
        output_dir: str = ".",
        include_profiling: bool = False,
        dump_excel: bool = True,
        warmup_days: int | None = None,
        verbose: bool = True,
    ):
        del n_quantiles, include_profiling
        import pandas as pd

        from betalens.analyst import Analyst
        from betalens.backtest import BacktestBase
        from betalens.datafeed import get_absolute_trade_days
        from betalens.factor.signal import (
            build_signal_weights,
            infer_signal_warmup,
            resolve_timing_start_date,
        )
        from factor_template import (
            RunResult as CoreRunResult,
            _ensure_runtime,
            align_daily_wides,
            build_pit_universe,
            fetch_daily_wide,
            fetch_industry_wide,
            mask_wide_by_pit_universe,
        )

        _ensure_runtime()

        sp = self.spec
        params = dict(sp.compute_kwargs or {})
        target_codes = _timing_codes(params, universe)
        if not sp.index_code:
            raise ValueError("Alpha101 timing requires factor_spec.index_code for cross-sectional computation")

        requested_start_date = str(start_date)
        start_date = resolve_timing_start_date(
            requested_start_date,
            end_date,
            target_codes=target_codes,
            metrics=[*sp.inputs.values(), sp.backtest_metric],
            table_name=sp.table_name,
        )

        signal_bars = infer_signal_warmup(params, minimum=30)
        required_bars = max(1, int(sp.required_history_bars)) + int(signal_bars)
        warmup = int(warmup_days) if warmup_days is not None else required_bars * 2 + 30
        fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=warmup)).strftime("%Y-%m-%d")
        trade_days = sorted(get_absolute_trade_days(fetch_start, end_date, "D", use_pmc=False))
        rebalance_dates = get_absolute_trade_days(start_date, end_date, rebal_freq, use_pmc=False)
        day_position = {day: i for i, day in enumerate(trade_days)}
        signal_dates = [trade_days[day_position[day] - 1] for day in rebalance_dates if day_position.get(day, 0) > 0]
        if not signal_dates:
            raise ValueError(f"no signal dates for {start_date}~{end_date}, rebal_freq={rebal_freq}")

        context_pit_universe = build_pit_universe(trade_days, sp.index_code)
        formula_pit_universe = _with_timing_targets(context_pit_universe, target_codes)
        calculation_universe = sorted({code for codes in formula_pit_universe.values() for code in codes})
        if not calculation_universe:
            raise ValueError(f"empty PIT universe: {sp.index_code}")
        if verbose:
            print(
                f"{sp.name}: target={target_codes} context={sp.index_code} "
                f"universe={len(calculation_universe)} warmup_days={warmup}",
                flush=True,
            )

        wides = {}
        for arg_name, metric in sp.inputs.items():
            wide = fetch_daily_wide(
                metric,
                universe=calculation_universe,
                start_date=fetch_start,
                end_date=end_date,
                table_name=sp.table_name,
            )
            if wide.empty:
                raise ValueError(f"empty input data: {arg_name} ({metric})")
            wides[arg_name] = wide

        wides = {
            name: mask_wide_by_pit_universe(wide, formula_pit_universe)
            for name, wide in align_daily_wides(wides).items()
        }

        reference_index = next(iter(wides.values())).index
        for arg_name, scheme in sp.industry_inputs.items():
            labels = fetch_industry_wide(
                scheme,
                universe=calculation_universe,
                dates=trade_days,
                reference_index=reference_index,
            )
            wides[arg_name] = mask_wide_by_pit_universe(labels, formula_pit_universe)

        factor_wide = sp.compute(**wides, **params)
        if factor_wide is None or factor_wide.empty:
            raise ValueError(f"empty {sp.name} factor values")
        signal_result = build_signal_weights(
            factor_wide=factor_wide,
            signal_dates=signal_dates,
            codes=target_codes,
            params=params,
            direction=sp.direction,
        )
        weights = signal_result.weights
        factor_values = signal_result.factor_values
        if weights.empty:
            raise ValueError("empty timing weights")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        weights.to_csv(out_dir / f"{sp.name}_weights.csv", encoding="utf-8-sig")
        factor_values.to_csv(out_dir / f"{sp.name}_factor_values.csv", index=False, encoding="utf-8-sig")
        for code, events in signal_result.events.items():
            safe_code = str(code).replace(".", "_")
            events.to_csv(out_dir / f"{sp.name}_{safe_code}_trigger_events.csv", index=False, encoding="utf-8-sig")

        bt = BacktestBase(
            weights,
            symbol=sp.name,
            amount=initial_amount,
            metric=sp.backtest_metric,
            table_name=sp.table_name,
            time_tolerance=24 * 11,
            verbose=verbose,
        )
        bt.requested_start_date = requested_start_date
        bt.effective_start_date = start_date
        analyst = Analyst.from_backtest(
            bt,
            name=sp.name,
            benchmark_code=benchmark_code,
            benchmark_metric=sp.backtest_metric,
            factor_values=factor_values,
        )
        analyst.report(
            to_excel=str(out_dir / f"{sp.name}_report.xlsx"),
            to_html=str(out_dir / f"{sp.name}_report.html"),
        )
        if dump_excel:
            dump_path = out_dir / f"{sp.name}_dump.xlsx"
            bt.dump_to_excel(str(dump_path))
            with pd.ExcelWriter(dump_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                factor_values.to_excel(writer, sheet_name="factor_values", index=False)

        return CoreRunResult(
            backtest=bt,
            analyst=analyst,
            profiling=None,
            neutralize_stats=None,
            factor_values=factor_values,
            pit_validation=None,
        )
