"""Daily panel contracts shared by factor scripts (no market queries)."""
from __future__ import annotations

import pandas as pd


def daily_panel(frame, name="panel"):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name}: 必须是 DataFrame")
    if len(frame.index) and pd.api.types.is_numeric_dtype(frame.index):
        raise ValueError(f"{name}: 日期索引不能是数字")
    result = frame.copy()
    days = pd.DatetimeIndex(frame.index)
    if days.tz is not None:
        days = days.tz_convert("Asia/Shanghai").tz_localize(None)
    result.index = days.normalize()
    if result.index.hasnans or result.index.has_duplicates:
        raise ValueError(f"{name}: 交易日为空或重复")
    if result.columns.has_duplicates or any(not isinstance(c, str) for c in result.columns):
        raise ValueError(f"{name}: 证券代码必须是唯一字符串")
    return result.sort_index().sort_index(axis=1)


def align_daily_wides(wides, *, universe=None):
    """Align both axes, retaining missing observations and availability attrs."""
    panels = {name: daily_panel(frame, name) for name, frame in wides.items()}
    days = pd.DatetimeIndex(sorted({day for frame in panels.values() for day in frame.index}))
    columns = sorted(set(universe) if universe is not None else
                     {code for frame in panels.values() for code in frame.columns})
    result = {}
    for name, frame in panels.items():
        result[name] = frame.reindex(index=days, columns=columns)
        result[name].attrs = dict(frame.attrs)
    return result


def validate_factor_output(frame, inputs, name):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name}: compute 必须返回 DataFrame")
    reference = next(iter(inputs.values()), None)
    if reference is None or reference.empty:
        raise ValueError(f"{name}: 无可用输入面板")
    if not frame.index.equals(reference.index) or not frame.columns.equals(reference.columns):
        raise ValueError(f"{name}: compute 输出日期/证券轴与输入不一致")
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"{name}: compute 输出存在重复键")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError(f"{name}: compute 输出必须是数值面板")
    return frame


def wide_to_prequery(frame, metric, signal_dates):
    panel = daily_panel(frame, metric)
    dates = {pd.Timestamp(day).date() for day in signal_dates}
    panel = panel.loc[[day.date() in dates for day in panel.index]]
    result = panel.stack().rename(metric).rename_axis(["input_ts", "code"]).reset_index()
    if result.duplicated(["input_ts", "code"]).any():
        raise ValueError(f"{metric}: 因子长表存在重复 (input_ts, code)")
    result["datetime"] = result["input_ts"]
    result["diff_hours"] = 0.0
    return result
