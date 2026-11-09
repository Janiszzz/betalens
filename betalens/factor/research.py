"""Research schedules and ideal group returns independent of strategy fills."""
from __future__ import annotations

import numpy as np
import pandas as pd


def local_time(value):
    ts = pd.Timestamp(value)
    return ts.tz_convert("Asia/Shanghai").tz_localize(None) if ts.tzinfo else ts


def align_daily_wides(wides):
    days = pd.DatetimeIndex(sorted({local_time(t).normalize() for w in wides.values()
                                   if w is not None for t in w.index}))
    output = {}
    for name, wide in wides.items():
        if wide is None:
            output[name] = wide
            continue
        frame = wide.copy()
        frame.index = pd.DatetimeIndex([local_time(t).normalize() for t in frame.index])
        output[name] = frame.groupby(level=0).last().reindex(days)
        output[name].attrs = dict(wide.attrs)
    return output


def build_rebalance_schedule(signal_dates, rebalance_dates, *, wides=None):
    if len(signal_dates) != len(rebalance_dates):
        raise ValueError("信号日与调仓日必须逐项对应")
    availability = []
    for wide in (wides or {}).values():
        if wide is None:
            continue
        availability.extend((pd.Timestamp(d).normalize(), local_time(t))
                            for d, t in wide.attrs.get("availability_upper", {}).items())
    upper_by_day = {}
    for day, at in sorted(availability):
        upper_by_day[day] = max(upper_by_day.get(day, at), at)
    availability_days = pd.DatetimeIndex(sorted(upper_by_day))
    prefix, latest = [], None
    for day in availability_days:
        latest = max(latest, upper_by_day[day]) if latest is not None else upper_by_day[day]
        prefix.append(latest)
    rows = []
    for signal, rebalance in zip(signal_dates, rebalance_dates):
        day = local_time(signal).normalize()
        target = local_time(rebalance).normalize() + pd.Timedelta(minutes=10)
        position = availability_days.searchsorted(day, side='right') - 1
        signal_ts = prefix[position] if position >= 0 else day + pd.Timedelta(hours=15)
        rows.append(dict(signal_date=day, signal_ts=signal_ts, target_ts=target,
                         valid=signal_ts <= target,
                         reason="" if signal_ts <= target else "signal_available_after_target"))
    result = pd.DataFrame(rows, columns=["signal_date", "signal_ts", "target_ts", "valid", "reason"])
    if result.empty:
        result["forward_end_ts"] = pd.Series(dtype="datetime64[ns]")
        return result
    result = result.sort_values("target_ts").reset_index(drop=True)
    if result.signal_date.duplicated().any() or result.target_ts.duplicated().any():
        raise ValueError("研究调仓映射存在重复信号日或目标时点")
    result["forward_end_ts"] = result.target_ts.shift(-1)
    return result


def weights_on_schedule(weights, schedule):
    if weights is None or weights.empty or schedule.empty:
        return weights.iloc[:0].copy() if weights is not None else pd.DataFrame()
    mapping = schedule[schedule.valid].set_index("signal_date").target_ts
    days = pd.DatetimeIndex([local_time(t).normalize() for t in weights.index])
    keep = days.isin(mapping.index)
    result = weights.loc[keep].copy()
    result.index = pd.DatetimeIndex([mapping.loc[d] for d in days[keep]])
    return result.sort_index()


def annotate_factor_values(values, schedule):
    if values is None or values.empty:
        return values
    result = values.drop(columns=[c for c in schedule.columns if c != 'signal_date'], errors='ignore').copy()
    key = next(key for key in ("信号日", "signal_date", "input_ts", "date") if key in result)
    result["signal_date"] = pd.to_datetime(result[key]).map(lambda t: local_time(t).normalize())
    return result.merge(schedule, on="signal_date", how="left", validate="many_to_one")


def _execution_series(prices, code, metric, price_time_map=None):
    if code not in prices:
        return pd.Series(dtype=float)
    clock = {"open": "09:30", "close": "15:00", **(price_time_map or {})}.get(metric)
    if clock is None:
        raise ValueError("请配置研究价格字段的 price_time_map: " + metric)
    offset = pd.Timedelta(clock + ":00" if len(clock) == 5 else clock)
    values = pd.to_numeric(prices[code], errors="coerce").copy()
    index = pd.DatetimeIndex(values.index)
    if index.tz is not None:
        index = index.tz_convert('Asia/Shanghai').tz_localize(None)
    values.index = index.normalize() + offset
    return values.sort_index().dropna()


def _forward_observations(prices, schedule, metric, price_time_map):
    schedule = schedule.reset_index(drop=True)
    targets = pd.DatetimeIndex(schedule.target_ts)
    ends = pd.DatetimeIndex(schedule.forward_end_ts)
    requested = schedule.valid.to_numpy(dtype=bool) & ends.notna()
    for code in prices:
        if code == "cash":
            continue
        series = _execution_series(prices, code, metric, price_time_map)
        frame = pd.DataFrame({'signal_date': schedule.signal_date, 'code': str(code),
            'start_ts': pd.NaT, 'end_ts': pd.NaT, 'return': np.nan, 'valid': False,
            'reason': np.where(schedule.valid, 'missing_next_target', schedule.reason)})
        start_locs, end_locs = series.index.searchsorted(targets), series.index.searchsorted(ends)
        frame.loc[requested, 'reason'] = 'missing_price'
        covered = requested & (start_locs < len(series)) & (end_locs < len(series))
        positions = np.flatnonzero(covered)
        start_locs, end_locs = start_locs[covered], end_locs[covered]
        chronological = (series.index[start_locs] < ends[covered]) & (end_locs > start_locs)
        frame.loc[positions, 'reason'] = 'superseded_or_nonpositive_interval'
        positions, start_locs, end_locs = positions[chronological], start_locs[chronological], end_locs[chronological]
        start_values, end_values = series.to_numpy()[start_locs], series.to_numpy()[end_locs]
        positive = np.isfinite(start_values) & np.isfinite(end_values) & (start_values > 0) & (end_values > 0)
        frame.loc[positions, 'reason'] = 'invalid_price'
        valid_positions = positions[positive]
        frame.loc[valid_positions, 'start_ts'] = series.index[start_locs[positive]]
        frame.loc[valid_positions, 'end_ts'] = series.index[end_locs[positive]]
        frame.loc[valid_positions, 'return'] = end_values[positive] / start_values[positive] - 1
        frame.loc[valid_positions, 'valid'] = True
        frame.loc[valid_positions, 'reason'] = ''
        yield str(code), frame


def build_forward_returns(prices, schedule, *, metric="close", price_time_map=None):
    frames = [frame for _, frame in _forward_observations(prices, schedule, metric, price_time_map)]
    columns = ["signal_date", "code", "start_ts", "end_ts", "return", "valid", "reason"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def forward_return_matrix(prices, schedule, *, metric="close", price_time_map=None):
    """Compact period-by-code returns for mining; share matching and validity rules."""
    return pd.DataFrame({code: frame['return'] for code, frame in
        _forward_observations(prices, schedule, metric, price_time_map)}, index=pd.RangeIndex(len(schedule)))


def ideal_group_nav(close_prices, execution_prices, factor_values, schedule, n_quantiles,
                    *, metric="close", price_time_map=None):
    """Equal entry weights drift within each original research period.

    A missing required constituent price makes the group observation missing;
    constituents are never reweighted to hide missing data.
    """
    if n_quantiles < 1 or factor_values is None or factor_values.empty or schedule.empty or close_prices.empty:
        return pd.DataFrame(), pd.DataFrame()
    values = factor_values.copy()
    for original, canonical in (("信号日", "signal_date"), ("股票代码", "code"), ("分组", "group")):
        if original in values:
            values[canonical] = values[original]
    values["signal_date"] = pd.to_datetime(values.signal_date).map(lambda t: local_time(t).normalize())
    values["group"] = pd.to_numeric(values["group"], errors="coerce")
    if values["group"].min() == 0:
        values["group"] += 1
    days = pd.DatetimeIndex(close_prices.index).normalize()
    close = close_prices.copy()
    close.index = days + pd.Timedelta(hours=15)
    execution_series = {code: _execution_series(execution_prices, code, metric, price_time_map)
                        for code in values.code.astype(str).unique()}
    navs, audit = {}, []
    valid_schedule = schedule[schedule.valid].reset_index(drop=True)
    for group in range(1, n_quantiles + 1):
        curve = pd.Series(np.nan, index=close.index, dtype=float)
        capital = 1.0
        for i, row in valid_schedule.iterrows():
            target = row.target_ts
            end = valid_schedule.iloc[i + 1].target_ts if i + 1 < len(valid_schedule) else close.index[-1] + pd.Timedelta(microseconds=1)
            codes = values.loc[(values.signal_date == row.signal_date) & (values["group"] == group), "code"].astype(str).drop_duplicates().tolist()
            entries = {}
            for code in codes:
                available = execution_series[code]
                available = available.loc[(available.index >= target) & (available.index < end)]
                entries[code] = None if available.empty else (available.index[0], float(available.iloc[0]))
            window = close.index[(close.index >= target) & (close.index < end)]
            for ts in window:
                ratios = []
                for code in codes:
                    entry = entries[code]
                    if entry is None:
                        ratios.append(np.nan)
                    elif ts < entry[0]:
                        ratios.append(1.0)
                    else:
                        mark = close.at[ts, code] if code in close else np.nan
                        ratios.append(mark / entry[1])
                valid = sum(pd.notna(ratios))
                curve.loc[ts] = capital * (float(np.mean(ratios)) if codes else 1.0)
                audit.append(dict(datetime=ts, group=group, expected=len(codes), valid=valid,
                                  coverage=valid / len(codes) if codes else 1.0))
            if i + 1 < len(valid_schedule):
                exit_ratios = []
                for code in codes:
                    entry = entries[code]
                    exit_series = execution_series[code]
                    exit_series = exit_series.loc[exit_series.index >= end]
                    exit_ratios.append(exit_series.iloc[0] / entry[1] if entry is not None and len(exit_series) else np.nan)
                capital *= float(np.mean(exit_ratios)) if codes else 1.0
        navs[f"G{group}"] = curve
    result = pd.DataFrame(navs)
    result.attrs["return_model"] = "ideal_equal_entry_weights_with_drift"
    return result, pd.DataFrame(audit)


def attach_research(bt, factor_values, schedule, n_quantiles=0, *, metric="close", price_time_map=None):
    from betalens.datafeed import Datafeed
    from betalens.datafeed.daily import daily_wide
    code_col = "股票代码" if "股票代码" in factor_values else "code"
    codes = factor_values[code_col].dropna().astype(str).unique().tolist()
    db = Datafeed("market_daily_post_adjusted")
    try:
        raw = db.query_daily_market(codes, schedule.target_ts.min().date(), bt.end.date(), metrics=list(dict.fromkeys([metric, "close"])))
    finally:
        db.close()
    execution, close = daily_wide(raw, metric), daily_wide(raw, "close")
    bt.research_schedule = schedule.copy()
    bt.research_forward_returns = build_forward_returns(execution, schedule, metric=metric, price_time_map=price_time_map)
    bt.factor_values = annotate_factor_values(factor_values, schedule)
    bt.research_group_nav, bt.research_group_coverage = ideal_group_nav(close, execution, factor_values, schedule, n_quantiles, metric=metric, price_time_map=price_time_map)
    return bt.factor_values
