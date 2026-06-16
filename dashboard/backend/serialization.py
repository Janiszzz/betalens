from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from betalens.analyst import metrics as M
from betalens.analyst.naming import get_name_map, label


PERCENT_METRICS = {
    "策略收益",
    "策略年化收益",
    "超额收益",
    "基准收益",
    "最大回撤",
    "索提诺比率",
    "日均超额收益",
    "超额收益最大回撤",
    "日胜率",
    "策略波动率",
    "基准波动率",
}


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _json_records(df: pd.DataFrame | None, max_rows: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(pd.notnull(out), None)
    return [{str(k): _clean_scalar(v) for k, v in row.items()} for row in out.to_dict("records")]


def _series_points(series: pd.Series | None, name: str) -> list[dict[str, Any]]:
    if series is None or series.empty:
        return []
    s = series.sort_index().replace([np.inf, -np.inf], np.nan).dropna()
    return [{"date": pd.Timestamp(idx).strftime("%Y-%m-%d"), name: _clean_scalar(val)} for idx, val in s.items()]


def _wide_long_records(df: pd.DataFrame | None, value_name: str) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    wide = df.copy().sort_index().replace([np.inf, -np.inf], np.nan)
    records: list[dict[str, Any]] = []
    for dt, row in wide.iterrows():
        date = pd.Timestamp(dt).strftime("%Y-%m-%d")
        for code, value in row.dropna().items():
            records.append({"date": date, "code": str(code), value_name: _clean_scalar(value)})
    return records


def _drawdown_interval(nav: pd.Series) -> str | None:
    if nav is None or nav.empty:
        return None
    nav = nav.sort_index()
    peak = nav.cummax()
    dd = (peak - nav) / peak
    if dd.empty:
        return None
    trough = dd.idxmax()
    start = nav.loc[:trough].idxmax()
    return f"{pd.Timestamp(start).strftime('%Y/%m/%d')},{pd.Timestamp(trough).strftime('%Y/%m/%d')}"


def _profit_loss_counts(daily_pnl_total: pd.Series | None) -> tuple[int | None, int | None]:
    if daily_pnl_total is None or daily_pnl_total.empty:
        return None, None
    pnl = daily_pnl_total.dropna()
    return int((pnl > 0).sum()), int((pnl < 0).sum())


def _profit_loss_ratio(daily_pnl_total: pd.Series | None) -> float | None:
    if daily_pnl_total is None or daily_pnl_total.empty:
        return None
    pnl = daily_pnl_total.dropna()
    gains = pnl[pnl > 0]
    losses = pnl[pnl < 0].abs()
    if gains.empty or losses.empty or losses.mean() == 0:
        return None
    return float(gains.mean() / losses.mean())


def build_metrics(analyst: Any, bt: Any) -> list[dict[str, Any]]:
    summary = analyst.an.summary() if analyst is not None else {}
    nav = getattr(bt, "nav", None)
    returns = nav.pct_change().dropna() if nav is not None and len(nav) else None
    daily_pnl_total = getattr(bt, "daily_pnl_total", None)
    wins, losses = _profit_loss_counts(daily_pnl_total)
    values = {
        "策略收益": summary.get("累计收益"),
        "策略年化收益": summary.get("年化收益"),
        "超额收益": summary.get("超额收益"),
        "基准收益": summary.get("基准收益"),
        "阿尔法": summary.get("Alpha"),
        "贝塔": summary.get("Beta"),
        "夏普比率": summary.get("夏普比率"),
        "胜率": (returns > 0).mean() if returns is not None and len(returns) else None,
        "盈亏比": _profit_loss_ratio(daily_pnl_total),
        "最大回撤": summary.get("最大回撤"),
        "索提诺比率": summary.get("索提诺比率"),
        "日均超额收益": summary.get("日均超额收益"),
        "超额收益最大回撤": summary.get("超额收益最大回撤"),
        "超额收益夏普比率": summary.get("超额收益夏普比率"),
        "日胜率": (returns > 0).mean() if returns is not None and len(returns) else None,
        "盈利次数": wins,
        "亏损次数": losses,
        "信息比率": summary.get("信息比率"),
        "策略波动率": summary.get("年化波动率"),
        "基准波动率": summary.get("基准波动率"),
        "最大回撤区间": _drawdown_interval(nav),
    }
    return [
        {
            "label": key,
            "value": _clean_scalar(value),
            "format": "percent" if key in PERCENT_METRICS else "number",
        }
        for key, value in values.items()
    ]


def build_chart_data(bt: Any) -> dict[str, Any]:
    nav = getattr(bt, "nav", None)
    daily_pnl_total = getattr(bt, "daily_pnl_total", None)
    daily_position_value = getattr(bt, "daily_position_value", None)
    daily_amount = getattr(bt, "daily_amount", None)
    drawdown = M._drawdown_series(nav) if nav is not None and len(nav) else None
    return {
        "nav": _series_points(nav, "nav"),
        "drawdown": _series_points(drawdown, "drawdown"),
        "dailyPnl": _series_points(daily_pnl_total, "pnl"),
        "dailyAmount": _series_points(daily_amount, "amount"),
        "positionValue": _wide_long_records(daily_position_value, "value"),
    }


def build_trade_table(bt: Any) -> list[dict[str, Any]]:
    trade = getattr(bt, "rebalance_log", None)
    if trade is None:
        return []
    df = trade.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return _json_records(df)


def build_position_table(bt: Any) -> list[dict[str, Any]]:
    position = getattr(bt, "position", None)
    position_value = getattr(bt, "daily_position_value", None)
    daily_pnl = getattr(bt, "daily_pnl", None)
    cost_price = getattr(bt, "cost_price", None)
    daily_amount = getattr(bt, "daily_amount", None)
    if position is None and position_value is None and daily_pnl is None:
        return []

    frames = [x for x in (position, position_value, daily_pnl) if x is not None and not x.empty]
    codes = sorted({str(c) for frame in frames for c in frame.columns})
    name_map = get_name_map([c for c in codes if c != "cash"])

    dates = sorted({pd.Timestamp(idx) for frame in frames for idx in frame.index})
    records: list[dict[str, Any]] = []
    for dt in dates:
        date_key = dt.strftime("%Y-%m-%d")
        total = None
        if daily_amount is not None and dt in daily_amount.index:
            total = float(daily_amount.loc[dt]) if pd.notna(daily_amount.loc[dt]) else None
        pnl_total = None
        if daily_pnl is not None and dt in daily_pnl.index:
            pnl_total = float(daily_pnl.loc[dt].sum())
        for code in codes:
            qty = _lookup(position, dt, code)
            value = _lookup(position_value, dt, code)
            pnl = _lookup(daily_pnl, dt, code)
            price = _lookup(cost_price, dt, code)
            records.append(
                {
                    "date": date_key,
                    "品种": label(code, name_map),
                    "代码": code,
                    "多空": "现金" if code == "cash" else ("多" if (qty or 0) >= 0 else "空"),
                    "数量": qty,
                    "可用数量": qty,
                    "收盘价/结算价": price,
                    "市值/价值": value,
                    "盈亏/逐笔浮盈": pnl,
                    "开仓均价": price,
                    "持仓均价(期货)": None,
                    "保证金": None,
                    "当日盈亏": pnl,
                    "今手数": None,
                    "仓位占比": (value / total) if total not in (None, 0) and value is not None else None,
                    "盈亏占比": (pnl / pnl_total) if pnl_total not in (None, 0) and pnl is not None else None,
                }
            )
    return records


def _lookup(df: pd.DataFrame | None, dt: pd.Timestamp, code: str) -> float | None:
    if df is None or df.empty or dt not in df.index or code not in df.columns:
        return None
    value = df.at[dt, code]
    return _clean_scalar(value)


def build_downloads(factor_dir: Path, name: str) -> dict[str, dict[str, Any]]:
    candidates = {
        "dump": factor_dir / f"{name}_dump.xlsx",
        "report": factor_dir / f"{name}_report.xlsx",
        "html": factor_dir / f"{name}_report.html",
        "profiling": factor_dir / f"{name}_profiling.xlsx",
    }
    return {
        kind: {"path": str(path), "exists": path.exists()}
        for kind, path in candidates.items()
    }


def serialize_result(run: Any) -> dict[str, Any]:
    if run.result is None or run.backtest is None or run.analyst is None:
        raise ValueError("Run has no completed result")
    factor_dir = Path(run.factor_dir)
    return {
        "run": run.to_state().model_dump(),
        "factor": {
            "class": run.factor_class,
            "name": run.name,
            "parameters": run.parameters,
            "compute_kwargs": run.compute_kwargs,
        },
        "metrics": build_metrics(run.analyst, run.backtest),
        "charts": build_chart_data(run.backtest),
        "trades": build_trade_table(run.backtest),
        "positions": build_position_table(run.backtest),
        "downloads": build_downloads(factor_dir, run.name),
    }
