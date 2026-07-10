from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


BASE_FACTOR_VALUE_COLUMNS = ["信号日", "股票代码", "因子值", "分组", "目标仓位", "是否触发"]


@dataclass
class SignalWeightResult:
    weights: pd.DataFrame
    factor_values: pd.DataFrame
    events: dict[str, pd.DataFrame] = field(default_factory=dict)


def _signal_config(params: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not params:
        return {}
    config = params.get("signal_weight")
    return config if isinstance(config, Mapping) else {}


def _setting(
    params: Mapping[str, Any] | None,
    key: str,
    default: Any = None,
    aliases: Sequence[str] = (),
) -> Any:
    names = (key, *aliases)
    config = _signal_config(params)
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    if params:
        for name in names:
            if name in params and params[name] is not None:
                return params[name]
    return default


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_codes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,\s，;；]+", value.strip())
        return [part for part in parts if part]
    return [text for text in (_optional_text(item) for item in value) if text]


def _resolve_codes(
    factor_wide: pd.DataFrame,
    codes: Sequence[str] | None = None,
    params: Mapping[str, Any] | None = None,
    universe: Sequence[str] | None = None,
) -> list[str]:
    requested = _normalize_codes(codes)
    if requested:
        return requested
    stock_code = _setting(params, "stock_code")
    requested = _normalize_codes(stock_code)
    if requested:
        return requested
    requested = _normalize_codes(universe)
    if requested:
        return [code for code in requested if code in factor_wide.columns]
    return [str(code) for code in factor_wide.columns]


def _daily_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").copy()
    out.index = pd.to_datetime(out.index).normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _signal_index(signal_dates: Sequence[Any]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex([pd.Timestamp(date).normalize() for date in signal_dates]).sort_values()
    return index[~index.duplicated(keep="last")]


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _max_weight(params: Mapping[str, Any] | None, default: float = 1.0) -> float:
    value = _finite_float(_setting(params, "max_weight", default), default)
    return min(max(value, 0.0), 1.0)


def resolve_operator(direction: str = "positive", operator: str | None = None) -> str:
    op = str(operator or "auto").strip().lower()
    aliases = {
        "auto": "gt" if str(direction).lower() == "positive" else "lt",
        "gt": "gt",
        ">": "gt",
        "above": "gt",
        "greater": "gt",
        "lt": "lt",
        "<": "lt",
        "below": "lt",
        "less": "lt",
    }
    if op not in aliases:
        raise ValueError(f"operator supports auto/gt/lt only: {operator}")
    return aliases[op]


def resolve_side(
    params: Mapping[str, Any] | None = None,
    *,
    side: str | None = None,
    direction: str = "positive",
) -> tuple[str, float]:
    raw = _setting(params, "side", side, aliases=("signal_side", "position_side"))
    text = str(raw or "auto").strip().lower()
    aliases = {
        "long": ("long", 1.0),
        "buy": ("long", 1.0),
        "+1": ("long", 1.0),
        "1": ("long", 1.0),
        "short": ("short", -1.0),
        "sell": ("short", -1.0),
        "-1": ("short", -1.0),
        "auto": ("long", 1.0) if str(direction).lower() == "positive" else ("short", -1.0),
    }
    if text not in aliases:
        raise ValueError(f"side supports auto/long/short only: {raw}")
    return aliases[text]


def _event_active(values: pd.Series | pd.DataFrame, threshold: float, operator: str):
    active = values > threshold if operator == "gt" else values < threshold
    return active.fillna(False).astype(bool)


def _target_from_active(active: pd.Series, side_sign: float, max_weight: float) -> pd.Series:
    return active.astype(float) * float(side_sign) * float(max_weight)


def _order_factor_values(df: pd.DataFrame) -> pd.DataFrame:
    extras = [col for col in df.columns if col not in BASE_FACTOR_VALUE_COLUMNS]
    return df[[col for col in BASE_FACTOR_VALUE_COLUMNS if col in df.columns] + extras]


def _scale_stock_weights(stock_weights: pd.DataFrame) -> pd.DataFrame:
    stock = stock_weights.apply(lambda col: pd.to_numeric(col, errors="coerce")).fillna(0.0)
    pos = stock.clip(lower=0.0)
    neg = stock.clip(upper=0.0)

    pos_sum = pos.sum(axis=1)
    pos_scale = pd.Series(1.0, index=stock.index)
    pos_scale.loc[pos_sum > 1.0] = 1.0 / pos_sum.loc[pos_sum > 1.0]

    neg_abs_sum = (-neg).sum(axis=1)
    neg_scale = pd.Series(1.0, index=stock.index)
    neg_scale.loc[neg_abs_sum > 1.0] = 1.0 / neg_abs_sum.loc[neg_abs_sum > 1.0]
    return pos.mul(pos_scale, axis=0) + neg.mul(neg_scale, axis=0)


def cash_from_weights(stock_weights: pd.DataFrame, *, scale: bool = True) -> pd.DataFrame:
    """Add a cash column using net exposure: full short -1 implies cash 2."""
    if stock_weights is None or stock_weights.empty:
        return pd.DataFrame()
    stock = stock_weights.drop(columns=["cash"], errors="ignore").copy()
    if scale:
        stock = _scale_stock_weights(stock)
    out = stock.copy()
    out["cash"] = 1.0 - out.sum(axis=1)
    return out.fillna(0.0)


def _with_execution_time(weights: pd.DataFrame, execution_delay: pd.Timedelta | None) -> pd.DataFrame:
    if execution_delay is None:
        return weights
    out = weights.copy()
    out.index = pd.DatetimeIndex(out.index) + execution_delay
    return out


def _factor_value_frame(
    *,
    signal_index: pd.DatetimeIndex,
    code: str,
    factor: pd.Series,
    active: pd.Series,
    target_weight: pd.Series,
    extras: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    data: dict[str, Any] = {
        "信号日": signal_index,
        "股票代码": code,
        "因子值": factor.to_numpy(dtype=float, copy=False),
        "分组": active.astype(int).to_numpy(copy=False),
        "目标仓位": target_weight.to_numpy(dtype=float, copy=False),
        "是否触发": active.to_numpy(copy=False),
    }
    if extras:
        for key, series in extras.items():
            data[key] = series.to_numpy(copy=False)
    return _order_factor_values(pd.DataFrame(data))


def threshold_weight(
    *,
    factor_wide: pd.DataFrame,
    signal_dates: Sequence[Any],
    codes: Sequence[str] | None = None,
    params: Mapping[str, Any] | None = None,
    threshold: float | None = None,
    operator: str | None = None,
    side: str | None = None,
    direction: str = "positive",
    max_weight: float | None = None,
    execution_delay: pd.Timedelta | None = pd.Timedelta(minutes=10),
) -> SignalWeightResult:
    signal_idx = _signal_index(signal_dates)
    if signal_idx.empty:
        raise ValueError("no signal dates for timing weights")
    code_list = _resolve_codes(factor_wide, codes, params)
    if not code_list:
        raise ValueError("no codes for timing weights")

    threshold_value = _finite_float(
        threshold if threshold is not None else _setting(params, "threshold", 0.0, aliases=("trigger_threshold",)),
        0.0,
    )
    op = resolve_operator(direction, operator or _setting(params, "operator", None, aliases=("trigger_operator",)))
    _, side_sign = resolve_side(params, side=side, direction=direction)
    weight_cap = _max_weight(params, 1.0) if max_weight is None else min(max(float(max_weight), 0.0), 1.0)

    stocks = pd.DataFrame(0.0, index=signal_idx, columns=code_list)
    frames = []
    for code in code_list:
        if code not in factor_wide.columns:
            continue
        factor = _daily_series(factor_wide[code]).reindex(signal_idx)
        active = _event_active(factor, threshold_value, op)
        target = _target_from_active(active, side_sign, weight_cap)
        stocks[code] = target.fillna(0.0).astype(float)
        frames.append(
            _factor_value_frame(
                signal_index=signal_idx,
                code=str(code),
                factor=factor,
                active=active,
                target_weight=target,
                extras={"历史阈值": pd.Series(threshold_value, index=signal_idx)},
            )
        )

    weights = _with_execution_time(cash_from_weights(stocks), execution_delay)
    factor_values = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BASE_FACTOR_VALUE_COLUMNS)
    return SignalWeightResult(weights=weights, factor_values=factor_values, events={})


def rolling_z_weight(
    *,
    factor_wide: pd.DataFrame,
    signal_dates: Sequence[Any],
    codes: Sequence[str] | None = None,
    params: Mapping[str, Any] | None = None,
    window: int | None = None,
    sigma: float | None = None,
    operator: str | None = None,
    side: str | None = None,
    direction: str = "positive",
    max_weight: float | None = None,
    execution_delay: pd.Timedelta | None = pd.Timedelta(minutes=10),
) -> SignalWeightResult:
    signal_idx = _signal_index(signal_dates)
    if signal_idx.empty:
        raise ValueError("no signal dates for timing weights")
    code_list = _resolve_codes(factor_wide, codes, params)
    if not code_list:
        raise ValueError("no codes for timing weights")

    win = int(window if window is not None else _setting(params, "window", 120, aliases=("threshold_window",)))
    if win <= 1:
        raise ValueError("rolling_z window must be greater than 1")
    sig = _finite_float(sigma if sigma is not None else _setting(params, "sigma", 1.0, aliases=("threshold_sigma",)), 1.0)
    op = resolve_operator(direction, operator or _setting(params, "operator", None, aliases=("trigger_operator",)))
    _, side_sign = resolve_side(params, side=side, direction=direction)
    weight_cap = _max_weight(params, 1.0) if max_weight is None else min(max(float(max_weight), 0.0), 1.0)

    stocks = pd.DataFrame(0.0, index=signal_idx, columns=code_list)
    frames = []
    for code in code_list:
        if code not in factor_wide.columns:
            continue
        factor = _daily_series(factor_wide[code]).reindex(signal_idx)
        history = factor.shift(1)
        rolling = history.rolling(window=win, min_periods=win)
        rolling_mean = rolling.mean()
        rolling_std = rolling.std()
        threshold = rolling_mean + sig * rolling_std
        active = _event_active(factor, threshold, op)
        target = _target_from_active(active, side_sign, weight_cap)
        stocks[code] = target.fillna(0.0).astype(float)
        frames.append(
            _factor_value_frame(
                signal_index=signal_idx,
                code=str(code),
                factor=factor,
                active=active,
                target_weight=target,
                extras={
                    "滚动均值": rolling_mean,
                    "滚动标准差": rolling_std,
                    "历史阈值": threshold,
                },
            )
        )

    weights = _with_execution_time(cash_from_weights(stocks), execution_delay)
    factor_values = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BASE_FACTOR_VALUE_COLUMNS)
    return SignalWeightResult(weights=weights, factor_values=factor_values, events={})


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or abs(float(denominator)) < 1e-12:
        return np.nan
    return float(numerator) / float(denominator)


def _slope(values: Sequence[float]) -> float:
    y = pd.Series(values, dtype="float64").dropna().to_numpy()
    if len(y) < 2:
        return np.nan
    x = np.arange(len(y), dtype="float64")
    return float(np.polyfit(x, y, 1)[0])


def _trend_label(slope: float, net_change: float, eps: float = 1e-10) -> str:
    if pd.isna(slope):
        return "single"
    if slope > eps:
        return "up"
    if slope < -eps:
        return "down"
    if abs(float(net_change)) <= eps:
        return "flat"
    return "mixed"


def _event_ids(factor: pd.Series, threshold: float, operator: str) -> pd.Series:
    active = _event_active(factor, threshold, operator)
    return (active & ~active.shift(1, fill_value=False)).cumsum().where(active)


def event_table_for_code(
    factor: pd.Series,
    high: pd.Series | None,
    threshold: float,
    operator: str,
) -> pd.DataFrame:
    factor = _daily_series(factor)
    high = _daily_series(high).reindex(factor.index) if high is not None else pd.Series(index=factor.index, dtype=float)
    event_ids = _event_ids(factor, threshold, operator)
    rows = []

    for event_id in event_ids.dropna().unique():
        dates = event_ids[event_ids == event_id].index
        values = factor.loc[dates].dropna().astype(float)
        if values.empty:
            continue
        start_date = pd.Timestamp(dates[0])
        end_date = pd.Timestamp(dates[-1])
        start_value = float(values.iloc[0])
        end_value = float(values.iloc[-1])
        max_date = pd.Timestamp(values.idxmax())
        min_date = pd.Timestamp(values.idxmin())
        max_value = float(values.loc[max_date])
        min_value = float(values.loc[min_date])
        net_change = end_value - start_value
        slope = _slope(values)
        trigger_high = high.loc[dates].max()
        post_dates = factor.index[factor.index > end_date]
        post_high = high.reindex(post_dates)
        hit = post_high[post_high > trigger_high].dropna().head(1)
        if hit.empty:
            new_high_date = pd.NaT
            wait_days = np.nan
        else:
            new_high_date = pd.Timestamp(hit.index[0])
            wait_days = int(post_dates.get_loc(new_high_date) + 1)
        rows.append(
            {
                "event_id": int(event_id),
                "start_date": start_date,
                "end_date": end_date,
                "duration": int(len(dates)),
                "factor_sum": float(values.sum()),
                "start_value": start_value,
                "end_value": end_value,
                "max_value": max_value,
                "max_date": max_date,
                "min_value": min_value,
                "min_date": min_date,
                "amplitude": max_value - min_value,
                "relative_amplitude": _safe_div(max_value - min_value, start_value),
                "net_change": net_change,
                "change_rate": _safe_div(net_change, start_value),
                "daily_slope": slope,
                "trend": _trend_label(slope, net_change),
                "up_days": int((values.diff() > 0).sum()),
                "down_days": int((values.diff() < 0).sum()),
                "up_day_ratio": _safe_div(float((values.diff() > 0).sum()), max(len(values) - 1, 1)),
                "mean_value": float(values.mean()),
                "median_value": float(values.median()),
                "trigger_high": float(trigger_high) if pd.notna(trigger_high) else np.nan,
                "new_high_date": new_high_date,
                "wait_days_to_new_high": wait_days,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "event_id",
                "start_date",
                "end_date",
                "duration",
                "factor_sum",
                "trigger_high",
                "new_high_date",
                "wait_days_to_new_high",
            ]
        )
    return pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)


def _history_before(events: pd.DataFrame, event_start: pd.Timestamp, window: int) -> pd.DataFrame:
    previous = events[events["end_date"] < event_start].sort_values("end_date")
    return previous.tail(int(window))


def _wait_history_before(events: pd.DataFrame, event_start: pd.Timestamp, window: int) -> pd.DataFrame:
    previous = events[
        (events["end_date"] < event_start)
        & events["new_high_date"].notna()
        & (events["new_high_date"] < event_start)
        & events["wait_days_to_new_high"].notna()
    ].sort_values("end_date")
    return previous.tail(int(window))


def _dynamic_event_weight_for_code(
    factor: pd.Series,
    events: pd.DataFrame,
    params: Mapping[str, Any] | None,
) -> pd.Series:
    factor = _daily_series(factor)
    target = pd.Series(0.0, index=factor.index)
    if events.empty:
        return target

    history_window = int(_setting(params, "history_window", 10))
    duration_q = min(max(_finite_float(_setting(params, "duration_quantile", 0.65), 0.65), 0.0), 1.0)
    exit_q = min(max(_finite_float(_setting(params, "exit_wait_quantile", 0.75), 0.75), 0.0), 1.0)
    min_history_events = int(_setting(params, "min_history_events", 3))
    default_exit_wait_days = int(_setting(params, "default_exit_wait_days", 5))
    max_weight = _max_weight(params, 1.0)

    for event in events.itertuples(index=False):
        event_start = pd.Timestamp(event.start_date)
        event_end = pd.Timestamp(event.end_date)
        history = _history_before(events, event_start, history_window)
        if len(history) < min_history_events:
            continue

        factor_sum_level = pd.to_numeric(history["factor_sum"], errors="coerce").dropna().mean()
        if not np.isfinite(factor_sum_level) or abs(float(factor_sum_level)) <= 1e-12:
            continue

        open_day = int(math.ceil(history["duration"].quantile(duration_q)))
        open_day = max(open_day, 1)
        wait_history = _wait_history_before(events, event_start, history_window)
        if wait_history.empty:
            exit_wait_days = default_exit_wait_days
        else:
            exit_wait_days = int(math.ceil(wait_history["wait_days_to_new_high"].quantile(exit_q)))
        exit_wait_days = max(exit_wait_days, 0)

        event_values = factor.loc[(factor.index >= event_start) & (factor.index <= event_end)].dropna()
        cumulative = 0.0
        last_weight = 0.0
        for day_number, (date, value) in enumerate(event_values.items(), start=1):
            cumulative += float(value)
            if day_number < open_day:
                continue
            last_weight = min(max_weight, max(0.0, cumulative / float(factor_sum_level)))
            target.loc[date] = max(target.loc[date], last_weight)

        if last_weight <= 0 or event_end not in target.index:
            continue
        end_pos = target.index.get_loc(event_end)
        hold_dates = target.index[end_pos + 1 : end_pos + 1 + exit_wait_days]
        if len(hold_dates):
            target.loc[hold_dates] = np.maximum(target.loc[hold_dates], last_weight)

    return target.clip(lower=0.0, upper=max_weight)


def event_history_weight(
    *,
    factor_wide: pd.DataFrame,
    signal_dates: Sequence[Any],
    high_wide: pd.DataFrame | None = None,
    codes: Sequence[str] | None = None,
    params: Mapping[str, Any] | None = None,
    threshold: float | None = None,
    operator: str | None = None,
    side: str | None = None,
    direction: str = "positive",
    execution_delay: pd.Timedelta | None = pd.Timedelta(minutes=10),
) -> SignalWeightResult:
    signal_idx = _signal_index(signal_dates)
    if signal_idx.empty:
        raise ValueError("no signal dates for timing weights")
    code_list = _resolve_codes(factor_wide, codes, params)
    if not code_list:
        raise ValueError("no codes for timing weights")

    threshold_value = _finite_float(
        threshold if threshold is not None else _setting(params, "threshold", 0.1, aliases=("trigger_threshold",)),
        0.1,
    )
    op = resolve_operator(direction, operator or _setting(params, "operator", None, aliases=("trigger_operator",)))
    _, side_sign = resolve_side(params, side=side, direction=direction)

    stocks = pd.DataFrame(0.0, index=signal_idx, columns=code_list)
    frames = []
    events_by_code: dict[str, pd.DataFrame] = {}
    for code in code_list:
        if code not in factor_wide.columns:
            events_by_code[str(code)] = pd.DataFrame()
            continue
        high = high_wide[code] if high_wide is not None and code in high_wide.columns else None
        factor_full = _daily_series(factor_wide[code])
        events = event_table_for_code(factor_full, high, threshold_value, op)
        events_by_code[str(code)] = events
        magnitude = _dynamic_event_weight_for_code(factor_full, events, params)
        signed_target = magnitude * side_sign
        target = signed_target.reindex(signal_idx).fillna(0.0).astype(float)
        stocks[code] = target

        factor = factor_full.reindex(signal_idx)
        active = _event_active(factor, threshold_value, op)
        event_ids = _event_ids(factor_full, threshold_value, op).reindex(signal_idx)
        frames.append(
            _factor_value_frame(
                signal_index=signal_idx,
                code=str(code),
                factor=factor,
                active=active,
                target_weight=target,
                extras={
                    "历史阈值": pd.Series(threshold_value, index=signal_idx),
                    "event_id": event_ids,
                },
            )
        )

    weights = _with_execution_time(cash_from_weights(stocks), execution_delay)
    factor_values = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BASE_FACTOR_VALUE_COLUMNS)
    return SignalWeightResult(weights=weights, factor_values=factor_values, events=events_by_code)


def _resolve_method(params: Mapping[str, Any] | None, method: str | None = None) -> str:
    raw = method or _setting(params, "method", None, aliases=("signal_method", "signal_weight_method"))
    text = str(raw or "").strip().lower().replace("-", "_")
    aliases = {
        "static": "threshold",
        "static_threshold": "threshold",
        "threshold": "threshold",
        "rolling": "rolling_z",
        "rolling_z": "rolling_z",
        "rolling_z_threshold": "rolling_z",
        "zscore": "rolling_z",
        "event": "event_history",
        "event_history": "event_history",
    }
    if text in aliases:
        return aliases[text]
    if text:
        raise ValueError(f"unknown signal_weight method: {raw}")

    if _setting(params, "window", None, aliases=("threshold_window",)) is not None:
        return "rolling_z"
    if (
        _setting(params, "history_window", None) is not None
        or _setting(params, "duration_quantile", None) is not None
        or _setting(params, "exit_wait_quantile", None) is not None
    ):
        return "event_history"
    if _setting(params, "threshold", None, aliases=("trigger_threshold",)) is not None:
        return "threshold"
    raise ValueError("signal_weight.method is required when no legacy timing parameters are present")


def build_signal_weights(
    *,
    factor_wide: pd.DataFrame,
    signal_dates: Sequence[Any],
    codes: Sequence[str] | None = None,
    params: Mapping[str, Any] | None = None,
    high_wide: pd.DataFrame | None = None,
    direction: str = "positive",
    method: str | None = None,
    side: str | None = None,
    execution_delay: pd.Timedelta | None = pd.Timedelta(minutes=10),
) -> SignalWeightResult:
    resolved_method = _resolve_method(params, method)
    if resolved_method == "threshold":
        return threshold_weight(
            factor_wide=factor_wide,
            signal_dates=signal_dates,
            codes=codes,
            params=params,
            side=side,
            direction=direction,
            execution_delay=execution_delay,
        )
    if resolved_method == "rolling_z":
        return rolling_z_weight(
            factor_wide=factor_wide,
            signal_dates=signal_dates,
            codes=codes,
            params=params,
            side=side,
            direction=direction,
            execution_delay=execution_delay,
        )
    if resolved_method == "event_history":
        return event_history_weight(
            factor_wide=factor_wide,
            high_wide=high_wide,
            signal_dates=signal_dates,
            codes=codes,
            params=params,
            side=side,
            direction=direction,
            execution_delay=execution_delay,
        )
    raise ValueError(f"unsupported signal_weight method: {resolved_method}")


def infer_signal_warmup(params: Mapping[str, Any] | None, minimum: int = 30) -> int:
    candidates = [int(minimum)]
    window = _setting(params, "window", None, aliases=("threshold_window",))
    if isinstance(window, (int, float)) and np.isfinite(window) and window > 1:
        candidates.append(int(window))
    history_window = _setting(params, "history_window", None)
    if isinstance(history_window, (int, float)) and np.isfinite(history_window) and history_window > 0:
        candidates.append(int(history_window) * 30 + 60)
    return int(max(candidates))


def standard_timing_weight_hook(weights: pd.DataFrame, task: Mapping[str, Any]) -> pd.DataFrame:
    del weights
    context = task.get("context")
    if not context:
        raise ValueError("standard timing weight hook requires mining task context")

    spec = context["spec"]
    params: dict[str, Any] = {}
    params.update(dict(getattr(spec, "compute_kwargs", {}) or {}))
    params.update(dict(task.get("params", {}) or {}))

    input_wides = context.get("input_wides") or {}
    high_wide = input_wides.get("high_wide")
    factor_wide = context["factor_wide"]
    universe = context.get("universe")
    codes = _resolve_codes(factor_wide, params=params, universe=universe)

    result = build_signal_weights(
        factor_wide=factor_wide,
        high_wide=high_wide,
        signal_dates=context["signal_dates"],
        codes=codes,
        params=params,
        direction=getattr(spec, "direction", "positive"),
    )
    return result.weights


__all__ = [
    "BASE_FACTOR_VALUE_COLUMNS",
    "SignalWeightResult",
    "build_signal_weights",
    "cash_from_weights",
    "event_history_weight",
    "event_table_for_code",
    "infer_signal_warmup",
    "resolve_operator",
    "resolve_side",
    "rolling_z_weight",
    "standard_timing_weight_hook",
    "threshold_weight",
]
