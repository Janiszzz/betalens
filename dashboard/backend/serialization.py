from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PERCENT_METRICS = {
    "策略收益",
    "策略年化收益",
    "策略波动率",
    "最大回撤",
    "日胜率",
    "IC胜率",
    "超额收益",
    "超额年化收益",
    "超额波动率",
    "超额最大回撤",
    "相对基准胜率",
    "基准收益",
    "基准年化收益",
    "基准波动率",
    "阿尔法",
    "跟踪误差",
    "交易胜率",
    "平均单笔收益",
    "平均盈利",
    "平均亏损",
    "最大单笔盈利",
    "最大单笔亏损",
    "平均仓位",
    "最大仓位",
    "开仓占比",
    "空仓占比",
    "累计收益",
    "年化收益",
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


def _name_map_for_codes(codes: list[str] | set[str] | tuple[str, ...]) -> dict[str, str]:
    lookup_codes = [
        str(code)
        for code in dict.fromkeys(codes)
        if str(code) not in ("cash", "现金", "其他")
    ]
    if not lookup_codes:
        return {}
    try:
        from betalens.analyst.naming import get_name_map

        return get_name_map(lookup_codes)
    except Exception:
        return {}


def _label_code(code: str, name_map: dict[str, str] | None = None) -> str:
    code = str(code)
    if code in ("cash", "现金"):
        return "现金"
    if code == "其他":
        return "其他"
    name_map = name_map if name_map is not None else _name_map_for_codes([code])
    name = name_map.get(code)
    return f"{name}({code})" if name else code


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


def _wide_long_records(
    df: pd.DataFrame | None, value_name: str, top_n: int | None = None
) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    wide = df.copy().sort_index().replace([np.inf, -np.inf], np.nan)
    # 图表只画前 top_n 个品种,宽股票池下据此裁剪传输量
    if top_n is not None and wide.shape[1] > top_n:
        ranking = wide.abs().max().sort_values(ascending=False)
        wide = wide[list(ranking.head(top_n).index)]
    records: list[dict[str, Any]] = []
    for dt, row in wide.iterrows():
        date = pd.Timestamp(dt).strftime("%Y-%m-%d")
        for code, value in row.dropna().items():
            records.append({"date": date, "code": str(code), value_name: _clean_scalar(value)})
    return records


def _position_weight_records(
    daily_position_value: pd.DataFrame | None,
    top: int = 10,
    max_codes: int = 25,
) -> list[dict[str, Any]]:
    if daily_position_value is None or daily_position_value.empty:
        return []
    dpv = daily_position_value.copy().sort_index().replace([np.inf, -np.inf], np.nan)
    weights = dpv.div(dpv.sum(axis=1), axis=0).fillna(0.0)
    stock_cols = [c for c in weights.columns if str(c) != "cash"]
    stock_w = weights[stock_cols] if stock_cols else pd.DataFrame(index=weights.index)
    selected: set[Any] = set()
    for _, row in stock_w.iterrows():
        non_zero = row[row > 0]
        if len(non_zero):
            selected.update(non_zero.nlargest(top).index)

    selected_cols = list(selected)
    if len(selected_cols) > max_codes:
        peak = stock_w[selected_cols].max().sort_values(ascending=False)
        selected_cols = list(peak.index[:max_codes])

    order = (
        stock_w[selected_cols].sum().sort_values(ascending=False).index.tolist()
        if selected_cols
        else []
    )
    other_cols = [c for c in stock_cols if c not in set(order)]

    plot_df = pd.DataFrame(index=weights.index)
    for col in order:
        plot_df[str(col)] = weights[col]
    if other_cols:
        plot_df["其他"] = weights[other_cols].sum(axis=1)
    if "cash" in weights.columns:
        plot_df["现金"] = weights["cash"]

    name_map = _name_map_for_codes([str(code) for code in order])
    records: list[dict[str, Any]] = []
    for dt, row in plot_df.iterrows():
        date = pd.Timestamp(dt).strftime("%Y-%m-%d")
        for code, weight in row.items():
            if pd.isna(weight) or float(weight) <= 0:
                continue
            records.append(
                {
                    "date": date,
                    "code": str(code),
                    "name": _label_code(str(code), name_map),
                    "weight": _clean_scalar(weight),
                }
            )
    return records


def _normalize_factor_values(factor_values: pd.DataFrame | None) -> pd.DataFrame:
    if factor_values is None or factor_values.empty:
        return pd.DataFrame(columns=["signal_date", "date_key", "code", "factor_value", "group"])

    df = factor_values.copy()
    rename_map = {
        "信号日": "signal_date",
        "input_ts": "signal_date",
        "date": "signal_date",
        "datetime": "signal_date",
        "股票代码": "code",
        "code": "code",
        "因子值": "factor_value",
        "factor_value": "factor_value",
        "分组": "group",
        "group": "group",
    }
    df = df.rename(columns={col: rename_map.get(str(col), str(col)) for col in df.columns})
    required = {"signal_date", "code", "factor_value"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["signal_date", "date_key", "code", "factor_value", "group"])

    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    df = df.dropna(subset=["signal_date", "code"])
    df["date_key"] = df["signal_date"].dt.strftime("%Y-%m-%d")
    df["code"] = df["code"].astype(str)
    if "group" not in df.columns:
        df["group"] = None
    return df[["signal_date", "date_key", "code", "factor_value", "group"]]


def _filter_factor_dates(
    factor_df: pd.DataFrame,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    if factor_df.empty:
        return factor_df
    out = factor_df.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out = out.dropna(subset=["signal_date"])
    if date_from:
        out = out[out["signal_date"] >= pd.Timestamp(date_from)]
    if date_to:
        out = out[out["signal_date"] <= pd.Timestamp(date_to) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)]
    return out


def build_factor_profile_payload(
    factor_values: pd.DataFrame | None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    factor_df = _filter_factor_dates(_normalize_factor_values(factor_values), date_from, date_to)
    if factor_df.empty:
        return {
            "available": False,
            "dateFrom": date_from,
            "dateTo": date_to,
            "summary": {"count": 0},
            "histogram": [],
            "ecdf": [],
            "quantiles": [],
            "tests": [],
            "timeseries": [],
            "autocorrelation": [],
            "turnover": [],
        }

    long = factor_df.rename(columns={"signal_date": "input_ts", "factor_value": "factor"})
    from betalens.factor.profiling import factor_profile_payload

    payload = factor_profile_payload(long[["input_ts", "code", "factor"]], metric="factor")
    dates = factor_df["signal_date"].sort_values()
    payload.update(
        {
            "available": True,
            "dateFrom": pd.Timestamp(dates.iloc[0]).strftime("%Y-%m-%d"),
            "dateTo": pd.Timestamp(dates.iloc[-1]).strftime("%Y-%m-%d"),
        }
    )
    return payload


def write_factor_values_parquet(factor_values: pd.DataFrame | None, path: Path) -> dict[str, Any]:
    factor_df = _normalize_factor_values(factor_values)
    meta = _table_meta(_json_records(factor_df.head(0)))
    if factor_df.empty:
        return {"total": 0, "columns": list(factor_df.columns)}
    path.parent.mkdir(parents=True, exist_ok=True)
    factor_df.to_parquet(path, index=False)
    return {"total": len(factor_df), "columns": list(factor_df.columns)}


def read_factor_profile(
    path: Path | None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    if path is None or not path.exists():
        return build_factor_profile_payload(None, date_from, date_to)
    return build_factor_profile_payload(pd.read_parquet(path), date_from, date_to)


def _factor_lookup_for_date(factor_df: pd.DataFrame, dt: pd.Timestamp) -> dict[str, dict[str, Any]]:
    if factor_df.empty:
        return {}

    date_key = dt.strftime("%Y-%m-%d")
    day_df = factor_df[factor_df["date_key"] == date_key]
    if day_df.empty:
        prior = factor_df[factor_df["signal_date"] <= dt]
        if prior.empty:
            return {}
        latest = prior["signal_date"].max()
        day_df = prior[prior["signal_date"] == latest]

    return {
        str(row["code"]): {
            "signalDate": pd.Timestamp(row["signal_date"]).strftime("%Y-%m-%d"),
            "factorValue": _clean_scalar(row["factor_value"]),
            "group": _clean_scalar(row.get("group")),
        }
        for _, row in day_df.iterrows()
    }


def _rebalance_holding_records(
    bt: Any,
    factor_values: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    weight = getattr(bt, "actual_weight", None)
    weight_source = "actual_weight"
    if weight is None or weight.empty:
        weight = getattr(bt, "weight", None)
        weight_source = "weight"
    if weight is None or weight.empty:
        return []

    w = weight.copy().sort_index().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    stock_cols = [c for c in w.columns if str(c) != "cash"]
    name_map = _name_map_for_codes([str(code) for code in stock_cols])
    factor_df = _normalize_factor_values(factor_values)

    records: list[dict[str, Any]] = []
    for dt, row in w.iterrows():
        ts = pd.Timestamp(dt)
        factor_lookup = _factor_lookup_for_date(factor_df, ts)
        held = row[stock_cols]
        held = held[held != 0]
        held = held.reindex(held.abs().sort_values(ascending=False).index)
        for rank, (code, weight_value) in enumerate(held.items(), 1):
            code_str = str(code)
            factor = factor_lookup.get(code_str, {})
            weight_float = float(weight_value)
            records.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "rank": rank,
                    "code": code_str,
                    "name": _label_code(code_str, name_map),
                    "side": "long" if weight_float > 0 else "short",
                    "weight": _clean_scalar(weight_float),
                    "factorValue": factor.get("factorValue"),
                    "group": factor.get("group"),
                    "signalDate": factor.get("signalDate"),
                    "weightSource": weight_source,
                }
            )
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


def build_metrics(analyst: Any, bt: Any) -> list[dict[str, Any]]:
    del bt
    summary = analyst.an.summary() if analyst is not None else {}
    values = [
        ("raw", "策略收益", summary.get("累计收益")),
        ("raw", "策略年化收益", summary.get("年化收益")),
        ("raw", "策略波动率", summary.get("年化波动率")),
        ("raw", "最大回撤", summary.get("最大回撤")),
        ("raw", "夏普比率", summary.get("夏普比率")),
        ("raw", "索提诺比率", summary.get("索提诺比率")),
        ("raw", "卡玛比率", summary.get("卡玛比率")),
        ("raw", "日胜率", summary.get("日胜率")),
        ("raw", "盈利次数", summary.get("盈利次数")),
        ("raw", "亏损次数", summary.get("亏损次数")),
        ("raw", "盈亏比", summary.get("盈亏比")),
        ("raw", "IC", summary.get("IC均值")),
        ("raw", "ICIR", summary.get("ICIR")),
        ("raw", "IC胜率", summary.get("IC胜率")),
        ("raw", "最大回撤区间", summary.get("最大回撤区间")),
        ("excess", "基准收益", summary.get("基准收益")),
        ("excess", "基准年化收益", summary.get("基准年化收益")),
        ("excess", "基准波动率", summary.get("基准波动率")),
        ("excess", "超额收益", summary.get("超额收益")),
        ("excess", "超额年化收益", summary.get("超额年化收益")),
        ("excess", "超额波动率", summary.get("超额波动率")),
        ("excess", "超额最大回撤", summary.get("超额最大回撤")),
        ("excess", "超额夏普比率", summary.get("超额夏普比率")),
        ("excess", "超额卡玛比率", summary.get("超额卡玛比率")),
        ("excess", "贝塔", summary.get("Beta")),
        ("excess", "阿尔法", summary.get("Alpha")),
        ("excess", "跟踪误差", summary.get("跟踪误差")),
        ("excess", "信息比率", summary.get("信息比率")),
        ("excess", "相对基准胜率", summary.get("相对基准胜率")),
    ]
    return [
        {
            "label": key,
            "value": _clean_scalar(value),
            "format": "percent" if key in PERCENT_METRICS else "number",
            "group": group,
        }
        for group, key, value in values
    ]


def _empty_timing_payload() -> dict[str, Any]:
    return {
        "metrics": [],
        "charts": {
            "navPrice": [],
            "tradeMarkers": [],
            "position": [],
            "drawdown": [],
            "dailyPnl": [],
            "tradeReturns": [],
            "predictionScatter": [],
            "openForwardReturns": [],
        },
        "tables": {
            "tradeSegments": [],
            "prediction": [],
        },
    }


def _metric(group: str, label: str, value: Any) -> dict[str, Any]:
    return {
        "label": label,
        "value": _clean_scalar(value),
        "format": "percent" if label in PERCENT_METRICS else "number",
        "group": group,
    }


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _numeric_series(series: Any) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except Exception:
            return pd.Series(dtype=float)
    if series.empty:
        return pd.Series(dtype=float)
    out = series.copy()
    idx = pd.to_datetime(out.index, errors="coerce")
    mask = ~pd.isna(idx)
    if not mask.any():
        return pd.Series(dtype=float)
    out = pd.to_numeric(out.loc[mask], errors="coerce")
    out.index = pd.DatetimeIndex(idx[mask]).normalize()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if out.empty:
        return pd.Series(dtype=float)
    return out.groupby(level=0).last().sort_index().astype(float)


def _numeric_frame(df: Any) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    idx = pd.to_datetime(out.index, errors="coerce")
    mask = ~pd.isna(idx)
    if not mask.any():
        return pd.DataFrame()
    out = out.loc[mask]
    out.index = pd.DatetimeIndex(idx[mask]).normalize()
    out = out.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.groupby(level=0).last().sort_index()


def _timing_weight_frames(bt: Any) -> tuple[pd.DataFrame, pd.Series]:
    weight = getattr(bt, "actual_weight", None)
    if weight is None or getattr(weight, "empty", True):
        weight = getattr(bt, "weight", None)

    w = _numeric_frame(weight)
    if w.empty:
        position_value = _numeric_frame(getattr(bt, "daily_position_value", None))
        if not position_value.empty:
            daily_amount = _numeric_series(getattr(bt, "daily_amount", None))
            if daily_amount.empty:
                daily_amount = position_value.sum(axis=1)
            denom = daily_amount.reindex(position_value.index).replace(0, np.nan)
            w = position_value.div(denom, axis=0)

    if w.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    stock_cols = [col for col in w.columns if str(col) not in ("cash", "现金")]
    stock = w[stock_cols].fillna(0.0) if stock_cols else pd.DataFrame(index=w.index)
    if "cash" in w.columns:
        cash = pd.to_numeric(w["cash"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elif "现金" in w.columns:
        cash = pd.to_numeric(w["现金"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        cash = (1.0 - stock.abs().sum(axis=1)).clip(lower=0.0, upper=1.0)
    cash.index = stock.index if len(stock.index) == len(cash.index) else cash.index
    return stock, cash.sort_index().astype(float)


def _timing_primary_code(stock_weight: pd.DataFrame) -> str | None:
    if stock_weight is None or stock_weight.empty or not len(stock_weight.columns):
        return None
    peak = stock_weight.abs().max(axis=0).sort_values(ascending=False)
    if peak.empty or not np.isfinite(float(peak.iloc[0])):
        return str(stock_weight.columns[0])
    return str(peak.index[0])


def _timing_price_series(bt: Any, primary_code: str | None) -> pd.Series:
    prices = _numeric_frame(getattr(bt, "daily_price", None))
    if prices.empty:
        prices = _numeric_frame(getattr(bt, "cost_price", None))
    if prices.empty:
        return pd.Series(dtype=float)
    if primary_code and primary_code in prices.columns:
        return _numeric_series(prices[primary_code])
    stock_cols = [col for col in prices.columns if str(col) not in ("cash", "现金")]
    if not stock_cols:
        return pd.Series(dtype=float)
    return _numeric_series(prices[stock_cols[0]])


def _timing_nav_price_records(
    nav: pd.Series,
    price: pd.Series,
    position: pd.Series,
) -> list[dict[str, Any]]:
    if nav.empty and price.empty and position.empty:
        return []
    index = nav.index.union(price.index).union(position.index).sort_values()
    records: list[dict[str, Any]] = []
    for dt in index:
        records.append(
            {
                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "nav": _clean_scalar(nav.get(dt)),
                "price": _clean_scalar(price.get(dt)),
                "position": _clean_scalar(position.get(dt)),
            }
        )
    return records


def _timing_trade_marker_records(
    rebalance_log: Any,
    price: pd.Series,
    primary_code: str | None,
) -> list[dict[str, Any]]:
    """Return buy/sell events whose y values sit on the displayed price curve."""
    if (
        rebalance_log is None
        or not isinstance(rebalance_log, pd.DataFrame)
        or rebalance_log.empty
        or price.empty
    ):
        return []
    required = {"datetime", "code", "direction"}
    if not required.issubset(rebalance_log.columns):
        return []

    records: list[dict[str, Any]] = []
    for _, row in rebalance_log.sort_values("datetime").iterrows():
        code = str(row["code"])
        direction = str(row["direction"]).lower()
        if primary_code and code != primary_code:
            continue
        if direction not in {"buy", "sell"}:
            continue
        dt = pd.to_datetime(row["datetime"], errors="coerce")
        if pd.isna(dt):
            continue
        day = pd.Timestamp(dt).normalize()
        curve_price = _finite_float(price.get(day))
        if curve_price is None:
            following = price.loc[price.index >= day]
            curve_price = _finite_float(following.iloc[0]) if not following.empty else None
        if curve_price is None:
            continue
        records.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "code": code,
                "side": direction,
                "price": curve_price,
                "tradePrice": _clean_scalar(row.get("price")),
            }
        )
    return records


def _timing_position_records(position: pd.Series, cash: pd.Series) -> list[dict[str, Any]]:
    if position.empty and cash.empty:
        return []
    index = position.index.union(cash.index).sort_values()
    records: list[dict[str, Any]] = []
    for dt in index:
        records.append(
            {
                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "position": _clean_scalar(position.get(dt)),
                "cash": _clean_scalar(cash.get(dt)),
            }
        )
    return records


def _drawdown_from_nav(nav: pd.Series) -> pd.Series:
    if nav.empty:
        return pd.Series(dtype=float)
    peak = nav.cummax().replace(0, np.nan)
    return ((peak - nav) / peak).replace([np.inf, -np.inf], np.nan).dropna()


def _timing_return_series(nav: pd.Series) -> pd.Series:
    if nav.empty:
        return pd.Series(dtype=float)
    return nav.sort_index().pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _timing_trade_segments(
    stock_weight: pd.DataFrame,
    position: pd.Series,
    returns: pd.Series,
    daily_pnl: pd.Series,
    epsilon: float = 1e-8,
) -> list[dict[str, Any]]:
    if position.empty:
        return []
    pos = position.sort_index().fillna(0.0)
    net = stock_weight.sum(axis=1).reindex(pos.index).fillna(0.0) if not stock_weight.empty else pos
    active = pos.abs() > epsilon
    dates = list(pos.index)
    records: list[dict[str, Any]] = []
    start_i: int | None = None
    start_side = 0

    def side_of(value: float) -> int:
        if value > epsilon:
            return 1
        if value < -epsilon:
            return -1
        return 0

    def close_segment(end_i: int) -> None:
        nonlocal start_i, start_side
        if start_i is None or end_i < start_i:
            start_i = None
            start_side = 0
            return
        seg_index = pd.DatetimeIndex(dates[start_i : end_i + 1])
        seg_returns = returns.reindex(seg_index).fillna(0.0)
        trade_return = (1.0 + seg_returns).prod() - 1.0 if len(seg_returns) else np.nan
        pnl_value = daily_pnl.reindex(seg_index).sum() if not daily_pnl.empty else np.nan
        seg_pos = pos.reindex(seg_index).fillna(0.0)
        records.append(
            {
                "tradeNo": len(records) + 1,
                "startDate": pd.Timestamp(seg_index[0]).strftime("%Y-%m-%d"),
                "endDate": pd.Timestamp(seg_index[-1]).strftime("%Y-%m-%d"),
                "holdingDays": int(len(seg_index)),
                "side": "long" if start_side >= 0 else "short",
                "avgPosition": _clean_scalar(seg_pos.mean()),
                "maxPosition": _clean_scalar(seg_pos.max()),
                "return": _clean_scalar(trade_return),
                "pnl": _clean_scalar(pnl_value),
                "isWin": bool(pd.notna(trade_return) and trade_return > 0),
            }
        )
        start_i = None
        start_side = 0

    for i, dt in enumerate(dates):
        is_active = bool(active.iloc[i])
        side = side_of(float(net.loc[dt]))
        if is_active and side == 0:
            side = start_side or 1
        if is_active and start_i is None:
            start_i = i
            start_side = side
            continue
        if not is_active and start_i is not None:
            close_segment(i - 1)
            continue
        if is_active and start_i is not None and side != 0 and start_side != 0 and side != start_side:
            close_segment(i - 1)
            start_i = i
            start_side = side

    if start_i is not None:
        close_segment(len(dates) - 1)
    return records


def _timing_open_forward_returns(nav: pd.Series, position: pd.Series) -> list[dict[str, Any]]:
    if nav.empty or position.empty:
        return []
    pos = position.reindex(nav.index).ffill().fillna(0.0)
    active = pos.abs() > 1e-8
    opens = active & ~active.shift(1, fill_value=False)
    open_locs = [nav.index.get_loc(dt) for dt in nav.index[opens]]
    records: list[dict[str, Any]] = []
    for horizon in range(1, 21):
        values = []
        for loc in open_locs:
            end = loc + horizon
            if end >= len(nav) or nav.iloc[loc] == 0:
                continue
            values.append(nav.iloc[end] / nav.iloc[loc] - 1.0)
        records.append(
            {
                "horizon": horizon,
                "avgReturn": _clean_scalar(float(np.mean(values)) if values else None),
                "sampleCount": int(len(values)),
            }
        )
    return records


def _timing_trade_metrics(segments: list[dict[str, Any]]) -> dict[str, Any]:
    returns = pd.Series([row.get("return") for row in segments], dtype="float64").dropna()
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    avg_loss = losses.mean() if len(losses) else np.nan
    odds = wins.mean() / abs(avg_loss) if len(wins) and len(losses) and avg_loss != 0 else np.nan
    holding_days = pd.Series([row.get("holdingDays") for row in segments], dtype="float64").dropna()
    return {
        "trade_count": int(len(segments)),
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "odds": odds,
        "avg_trade_return": returns.mean() if len(returns) else np.nan,
        "avg_win": wins.mean() if len(wins) else np.nan,
        "avg_loss": avg_loss,
        "max_win": returns.max() if len(returns) else np.nan,
        "max_loss": returns.min() if len(returns) else np.nan,
        "avg_holding_days": holding_days.mean() if len(holding_days) else np.nan,
    }


def _timing_performance_metrics(nav: pd.Series, returns: pd.Series) -> dict[str, Any]:
    if nav.empty:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "calmar": np.nan,
            "daily_win_rate": np.nan,
        }
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0 if nav.iloc[0] else np.nan
    periods = max(len(nav), 1)
    annualized_return = (1.0 + total_return) ** (252 / periods) - 1.0 if pd.notna(total_return) and total_return > -1 else np.nan
    dd = _drawdown_from_nav(nav)
    max_drawdown = dd.max() if len(dd) else np.nan
    ret = returns.dropna()
    ret_for_stats = ret.iloc[1:] if len(ret) > 1 else ret
    sharpe = (
        ret_for_stats.mean() / ret_for_stats.std() * math.sqrt(252)
        if len(ret_for_stats) > 1 and ret_for_stats.std() != 0
        else np.nan
    )
    calmar = annualized_return / max_drawdown if pd.notna(max_drawdown) and max_drawdown != 0 else np.nan
    daily_win_rate = float((ret_for_stats > 0).mean()) if len(ret_for_stats) else np.nan
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "calmar": calmar,
        "daily_win_rate": daily_win_rate,
    }


def _timing_factor_series(
    factor_values: pd.DataFrame | None,
    primary_code: str | None = None,
) -> pd.Series:
    factor_df = _normalize_factor_values(factor_values)
    if factor_df.empty:
        return pd.Series(dtype=float)
    if primary_code and primary_code in set(factor_df["code"].astype(str)):
        factor_df = factor_df[factor_df["code"].astype(str) == primary_code]
    elif factor_df["code"].nunique() == 1:
        pass
    else:
        return pd.Series(dtype=float)

    factor_df = factor_df.copy()
    factor_df["factor_value"] = pd.to_numeric(factor_df["factor_value"], errors="coerce")
    factor_df = factor_df.dropna(subset=["signal_date", "factor_value"])
    if factor_df.empty:
        return pd.Series(dtype=float)
    factor_df["date"] = factor_df["signal_date"].dt.normalize()
    series = factor_df.groupby("date")["factor_value"].mean().sort_index()
    series.index = pd.DatetimeIndex(series.index)
    return series.astype(float)


def _forward_returns_from_nav(nav: pd.Series, horizon: int) -> pd.Series:
    if nav.empty or horizon <= 0:
        return pd.Series(dtype=float)
    daily = nav.copy()
    daily.index = pd.DatetimeIndex(daily.index).normalize()
    daily = daily.groupby(level=0).last().sort_index()
    return (daily.shift(-horizon) / daily - 1.0).replace([np.inf, -np.inf], np.nan).dropna()


def _rolling_rank_ic(aligned: pd.DataFrame, window: int) -> pd.Series:
    if len(aligned) < window:
        return pd.Series(dtype=float)
    rows: list[tuple[pd.Timestamp, float]] = []
    for i in range(window - 1, len(aligned)):
        sub = aligned.iloc[i - window + 1 : i + 1]
        corr = sub["factor"].rank().corr(sub["fwdReturn"].rank())
        if pd.notna(corr):
            rows.append((aligned.index[i], float(corr)))
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(dict(rows)).sort_index()


def _ols_prediction_stats(aligned: pd.DataFrame) -> dict[str, Any]:
    if len(aligned) < 3:
        return {"beta": np.nan, "beta_p": np.nan, "r2": np.nan}
    x = aligned["factor"].astype(float)
    y = aligned["fwdReturn"].astype(float)
    x_centered = x - x.mean()
    denom = float((x_centered ** 2).sum())
    if denom <= 0:
        return {"beta": np.nan, "beta_p": np.nan, "r2": np.nan}
    beta = float((x_centered * (y - y.mean())).sum() / denom)
    alpha = float(y.mean() - beta * x.mean())
    fitted = alpha + beta * x
    resid = y - fitted
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    beta_p = np.nan
    if len(aligned) > 3:
        try:
            from scipy import stats as scipy_stats

            sigma2 = ss_res / (len(aligned) - 2)
            se_beta = math.sqrt(sigma2 / denom) if denom > 0 else np.nan
            if se_beta and np.isfinite(se_beta) and se_beta > 0:
                t_stat = beta / se_beta
                beta_p = float(2 * scipy_stats.t.sf(abs(t_stat), len(aligned) - 2))
        except Exception:
            beta_p = np.nan
    return {"beta": beta, "beta_p": beta_p, "r2": r2}


def _timing_prediction_payload(
    factor_values: pd.DataFrame | None,
    nav: pd.Series,
    primary_code: str | None,
    main_horizon: int = 5,
) -> dict[str, Any]:
    factor = _timing_factor_series(factor_values, primary_code)
    metric_values = {
        "ic": np.nan,
        "icir": np.nan,
        "ic_win_rate": np.nan,
        "beta": np.nan,
        "beta_p": np.nan,
        "r2": np.nan,
    }
    charts = {"predictionScatter": [], "prediction": []}
    if factor.empty or nav.empty:
        return {"metrics": metric_values, "charts": charts}

    prediction_rows: list[dict[str, Any]] = []
    main_aligned = pd.DataFrame()
    for horizon in (1, 3, 5, 10, 20):
        fwd = _forward_returns_from_nav(nav, horizon)
        aligned = pd.DataFrame({"factor": factor, "fwdReturn": fwd}).dropna()
        if horizon == main_horizon:
            main_aligned = aligned
        ic = aligned["factor"].rank().corr(aligned["fwdReturn"].rank()) if len(aligned) >= 3 else np.nan
        prediction_rows.append(
            {
                "horizon": horizon,
                "sampleCount": int(len(aligned)),
                "avgForwardReturn": _clean_scalar(aligned["fwdReturn"].mean() if len(aligned) else np.nan),
                "IC": _clean_scalar(ic),
            }
        )

    if not main_aligned.empty and len(main_aligned) >= 3:
        ic = main_aligned["factor"].rank().corr(main_aligned["fwdReturn"].rank())
        window = min(60, max(5, len(main_aligned) // 3))
        rolling_ic = _rolling_rank_ic(main_aligned, window)
        icir = rolling_ic.mean() / rolling_ic.std() if len(rolling_ic) > 1 and rolling_ic.std() != 0 else np.nan
        ols = _ols_prediction_stats(main_aligned)
        metric_values.update(
            {
                "ic": ic,
                "icir": icir,
                "ic_win_rate": float((rolling_ic > 0).mean()) if len(rolling_ic) else np.nan,
                "beta": ols["beta"],
                "beta_p": ols["beta_p"],
                "r2": ols["r2"],
            }
        )
        charts["predictionScatter"] = [
            {
                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "factor": _clean_scalar(row["factor"]),
                "fwdReturn": _clean_scalar(row["fwdReturn"]),
            }
            for dt, row in main_aligned.iterrows()
        ]

    charts["prediction"] = prediction_rows
    return {"metrics": metric_values, "charts": charts}


def build_timing_payload(bt: Any, factor_values: pd.DataFrame | None = None) -> dict[str, Any]:
    payload = _empty_timing_payload()
    if bt is None:
        return payload

    nav = _numeric_series(getattr(bt, "nav", None))
    returns = _timing_return_series(nav)
    daily_pnl = _numeric_series(getattr(bt, "daily_pnl_total", None))
    stock_weight, cash = _timing_weight_frames(bt)
    position = stock_weight.abs().sum(axis=1).sort_index() if not stock_weight.empty else pd.Series(dtype=float)
    primary_code = _timing_primary_code(stock_weight)
    price = _timing_price_series(bt, primary_code)
    drawdown = _drawdown_from_nav(nav)
    segments = _timing_trade_segments(stock_weight, position, returns, daily_pnl)
    trade = _timing_trade_metrics(segments)
    perf = _timing_performance_metrics(nav, returns)
    active = position.abs() > 1e-8 if not position.empty else pd.Series(dtype=bool)
    position_change_count = int((position.diff().abs() > 1e-8).sum()) if not position.empty else 0
    prediction = _timing_prediction_payload(factor_values, nav, primary_code)

    payload["metrics"] = [
        _metric("trade", "交易次数", trade["trade_count"]),
        _metric("trade", "交易胜率", trade["win_rate"]),
        _metric("trade", "赔率", trade["odds"]),
        _metric("trade", "平均单笔收益", trade["avg_trade_return"]),
        _metric("trade", "平均盈利", trade["avg_win"]),
        _metric("trade", "平均亏损", trade["avg_loss"]),
        _metric("trade", "最大单笔盈利", trade["max_win"]),
        _metric("trade", "最大单笔亏损", trade["max_loss"]),
        _metric("position", "平均仓位", position.mean() if not position.empty else np.nan),
        _metric("position", "最大仓位", position.max() if not position.empty else np.nan),
        _metric("position", "开仓占比", float(active.mean()) if len(active) else np.nan),
        _metric("position", "空仓占比", float((~active).mean()) if len(active) else np.nan),
        _metric("position", "平均持仓天数", trade["avg_holding_days"]),
        _metric("position", "仓位变化次数", position_change_count),
        _metric("return", "累计收益", perf["total_return"]),
        _metric("return", "年化收益", perf["annualized_return"]),
        _metric("return", "最大回撤", perf["max_drawdown"]),
        _metric("return", "Sharpe", perf["sharpe"]),
        _metric("return", "Calmar", perf["calmar"]),
        _metric("return", "日胜率", perf["daily_win_rate"]),
        _metric("prediction", "主预测周期 IC", prediction["metrics"]["ic"]),
        _metric("prediction", "ICIR", prediction["metrics"]["icir"]),
        _metric("prediction", "IC胜率", prediction["metrics"]["ic_win_rate"]),
        _metric("prediction", "Beta", prediction["metrics"]["beta"]),
        _metric("prediction", "Beta-P 值", prediction["metrics"]["beta_p"]),
        _metric("prediction", "R²", prediction["metrics"]["r2"]),
    ]
    payload["charts"] = {
        "navPrice": _timing_nav_price_records(nav, price, position),
        "tradeMarkers": _timing_trade_marker_records(
            getattr(bt, "rebalance_log", None), price, primary_code
        ),
        "position": _timing_position_records(position, cash),
        "drawdown": _series_points(drawdown, "drawdown"),
        "dailyPnl": _series_points(daily_pnl, "pnl"),
        "tradeReturns": [
            {
                "tradeNo": row["tradeNo"],
                "startDate": row["startDate"],
                "endDate": row["endDate"],
                "return": row["return"],
            }
            for row in segments
        ],
        "predictionScatter": prediction["charts"]["predictionScatter"],
        "openForwardReturns": _timing_open_forward_returns(nav, position),
    }
    payload["tables"] = {
        "tradeSegments": segments,
        "prediction": prediction["charts"]["prediction"],
    }
    return payload


def build_chart_data(bt: Any, factor_values: pd.DataFrame | None = None) -> dict[str, Any]:
    nav = getattr(bt, "nav", None)
    daily_pnl_total = getattr(bt, "daily_pnl_total", None)
    daily_position_value = getattr(bt, "daily_position_value", None)
    daily_amount = getattr(bt, "daily_amount", None)
    from betalens.analyst import metrics as M

    drawdown = M._drawdown_series(nav) if nav is not None and len(nav) else None
    return {
        "nav": _series_points(nav, "nav"),
        "drawdown": _series_points(drawdown, "drawdown"),
        "dailyPnl": _series_points(daily_pnl_total, "pnl"),
        "dailyAmount": _series_points(daily_amount, "amount"),
        "positionWeight": _position_weight_records(daily_position_value),
        "rebalanceHoldings": _rebalance_holding_records(bt, factor_values),
    }


def _nav_value_for_trade(nav: pd.Series | None, trade_date: Any) -> float | None:
    if nav is None or nav.empty:
        return None
    series = nav.sort_index()
    date = pd.Timestamp(trade_date)
    if date in series.index:
        return _clean_scalar(series.asof(date))
    following = series.loc[series.index >= date]
    value = following.iloc[0] if not following.empty else series.iloc[-1]
    return _clean_scalar(value)


def build_generated_chart_data(
    bt: Any,
    factor_values: pd.DataFrame | None = None,
    n_quantiles: Any = None,
    precomputed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """复用脚本静态图口径，生成供 dashboard 渲染的结构化数据。"""
    from betalens.analyst import metrics as M

    try:
        quantiles = int(n_quantiles)
    except (TypeError, ValueError):
        normalized = _normalize_factor_values(factor_values)
        groups = pd.to_numeric(normalized.get("group"), errors="coerce")
        quantiles = int(groups.max()) if groups is not None and groups.notna().any() else 0

    precomputed = precomputed or {}
    group_frame = precomputed.get("group_nav")
    if group_frame is None:
        group_frame = M.group_nav(getattr(bt, "cost_ret", None), factor_values, quantiles)
    group_records: list[dict[str, Any]] = []
    if not group_frame.empty:
        for date, row in group_frame.sort_index().iterrows():
            for group, value in row.items():
                if pd.isna(value):
                    continue
                group_records.append(
                    {
                        "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                        "group": str(group),
                        "nav": _clean_scalar(value),
                        "cumulativeReturn": _clean_scalar(float(value) - 1.0),
                    }
                )

    nav = getattr(bt, "nav", None)
    trade_pairs = precomputed.get("trade_pairs")
    if trade_pairs is None:
        trade_pairs = M.match_trade_pairs(getattr(bt, "rebalance_log", None))
    trade_records = []
    for _, row in trade_pairs.iterrows():
        trade_records.append(
            {
                "code": str(row["code"]),
                "buyDate": pd.Timestamp(row["buy_date"]).strftime("%Y-%m-%d"),
                "sellDate": pd.Timestamp(row["sell_date"]).strftime("%Y-%m-%d"),
                "buyPrice": _clean_scalar(row["buy_price"]),
                "sellPrice": _clean_scalar(row["sell_price"]),
                "return": _clean_scalar(row["return"]),
                "buyNav": _nav_value_for_trade(nav, row["buy_date"]),
                "sellNav": _nav_value_for_trade(nav, row["sell_date"]),
            }
        )

    annual = M.annual_trade_performance(trade_pairs)
    annual_records = [
        {
            "year": str(int(row["year"])),
            "avgReturn": _clean_scalar(row["avg_return"]),
            "winRate": _clean_scalar(row["win_rate"]),
            "tradeCount": int(row["n_trades"]),
        }
        for _, row in annual.iterrows()
    ]
    return {
        "groupNav": group_records,
        "tradePairs": trade_records,
        "annualTrade": annual_records,
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
    name_map = _name_map_for_codes(codes)
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
            if qty is not None and float(qty) == 0:
                continue
            value = _lookup(position_value, dt, code)
            pnl = _lookup(daily_pnl, dt, code)
            price = _lookup(cost_price, dt, code)
            records.append(
                {
                    "date": date_key,
                    "品种": _label_code(code, name_map),
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
        "config": factor_dir / "run_config.yaml",
        "dump": factor_dir / f"{name}_dump.xlsx",
        "report": factor_dir / f"{name}_report.xlsx",
        "html": factor_dir / f"{name}_report.html",
        "profiling": factor_dir / f"{name}_profiling.xlsx",
        "profiling_png": factor_dir / f"{name}_profiling.png",
    }
    return {
        kind: {"path": str(path), "exists": path.exists()}
        for kind, path in candidates.items()
    }


def build_result_payload(
    run: Any,
    table_metas: dict[str, dict[str, Any]],
    factor_values: pd.DataFrame | None = None,
    pit_validation: pd.DataFrame | None = None,
    neutralize_stats: pd.DataFrame | None = None,
    chart_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建可 JSON 化的结果（指标+图表+表元数据）。巨表明细不在内,走 /table 分页。
    不含 downloads —— 那个按需实时探测磁盘,因为 dump 是异步落盘的。"""
    profiling = build_factor_profile_payload(factor_values)
    return {
        "run": run.to_state().model_dump(),
        "factor": {
            "class": run.factor_class,
            "name": run.name,
            "parameters": run.parameters,
            "compute_kwargs": run.compute_kwargs,
        },
        "metrics": build_metrics(run.analyst, run.backtest),
        "timing": build_timing_payload(run.backtest, factor_values),
        "charts": {
            **build_chart_data(run.backtest, factor_values),
            **build_generated_chart_data(
                run.backtest,
                factor_values,
                run.parameters.get("n_quantiles"),
                chart_data,
            ),
            "profiling": profiling,
        },
        "tables": table_metas,
        "diagnostics": {
            "pitValidation": _json_records(
                pit_validation.reset_index() if pit_validation is not None and not pit_validation.empty else pit_validation
            ),
            "neutralizeStats": _json_records(
                neutralize_stats.reset_index() if neutralize_stats is not None and not neutralize_stats.empty else neutralize_stats
            ),
        },
    }


def _table_meta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return {"total": len(rows), "columns": columns}


def build_table(bt: Any, kind: str) -> list[dict[str, Any]]:
    if kind == "trades":
        return build_trade_table(bt)
    if kind == "positions":
        return build_position_table(bt)
    raise KeyError(kind)


def write_table_parquet(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """把巨表落成 parquet,返回 {total, columns} 元数据。空表不落盘。"""
    meta = _table_meta(rows)
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows, columns=meta["columns"])
        df.to_parquet(path, index=False)
    return meta


def read_table_page(
    path: Path | None,
    page: int = 1,
    size: int = 50,
    query: str | None = None,
    filters: dict[str, str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """从 parquet 读取分页数据。

    pyarrow 目前不能在任意文本搜索后直接只读目标页；这里保留 DataFrame
    过滤，但避免先全量转成 Python records，降低大表接口的额外内存和 CPU。
    """
    if path is None or not path.exists():
        return {"rows": [], "total": 0, "page": max(1, page), "size": max(1, size), "pages": 0}
    df = pd.read_parquet(path)
    if "date" in df.columns and (date_from or date_to):
        dates = pd.to_datetime(df["date"], errors="coerce")
        if date_from:
            df = df[dates >= pd.Timestamp(date_from)]
            dates = dates.loc[df.index]
        if date_to:
            df = df[dates <= pd.Timestamp(date_to)]
    if filters:
        for col, val in filters.items():
            if col not in df.columns:
                df = df.iloc[0:0]
                break
            df = df[df[col].astype(str).eq(str(val))]
    if query:
        needle = query.lower()
        haystack = df.astype(str).agg(" ".join, axis=1).str.lower()
        df = df[haystack.str.contains(needle, regex=False, na=False)]

    total = len(df)
    page = max(1, page)
    size = max(1, size)
    start = (page - 1) * size
    page_df = df.iloc[start : start + size].replace([np.inf, -np.inf], np.nan)
    page_df = page_df.where(pd.notnull(page_df), None)
    rows = [
        {str(k): _clean_scalar(v) for k, v in row.items()}
        for row in page_df.to_dict("records")
    ]
    return {
        "rows": rows,
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size if total else 0,
    }


def paginate_table(
    rows: list[dict[str, Any]],
    page: int = 1,
    size: int = 50,
    query: str | None = None,
    filters: dict[str, str] | None = None,
) -> dict[str, Any]:
    filtered = rows
    if filters:
        for col, val in filters.items():
            filtered = [r for r in filtered if str(r.get(col, "")) == val]
    if query:
        needle = query.lower()
        filtered = [r for r in filtered if needle in " ".join(str(v) for v in r.values()).lower()]
    total = len(filtered)
    page = max(1, page)
    size = max(1, size)
    start = (page - 1) * size
    return {
        "rows": filtered[start : start + size],
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size if total else 0,
    }
