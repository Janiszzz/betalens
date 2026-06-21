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
    name_map = get_name_map([str(c) for c in stock_cols])

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

    records: list[dict[str, Any]] = []
    for dt, row in plot_df.iterrows():
        date = pd.Timestamp(dt).strftime("%Y-%m-%d")
        for code, weight in row.items():
            if pd.isna(weight) or float(weight) <= 0:
                continue
            display_name = str(code) if code in ("其他", "现金") else label(str(code), name_map)
            records.append(
                {
                    "date": date,
                    "code": str(code),
                    "name": display_name,
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
    name_map = get_name_map([str(c) for c in stock_cols])
    factor_df = _normalize_factor_values(factor_values)

    records: list[dict[str, Any]] = []
    for dt, row in w.iterrows():
        ts = pd.Timestamp(dt)
        factor_lookup = _factor_lookup_for_date(factor_df, ts)
        held = row[stock_cols]
        held = held[held > 0].sort_values(ascending=False)
        for rank, (code, weight_value) in enumerate(held.items(), 1):
            code_str = str(code)
            factor = factor_lookup.get(code_str, {})
            records.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "rank": rank,
                    "code": code_str,
                    "name": label(code_str, name_map),
                    "weight": _clean_scalar(weight_value),
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


def build_chart_data(bt: Any, factor_values: pd.DataFrame | None = None) -> dict[str, Any]:
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
        "positionWeight": _position_weight_records(daily_position_value),
        "rebalanceHoldings": _rebalance_holding_records(bt, factor_values),
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
            if qty is not None and float(qty) == 0:
                continue
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


def build_result_payload(
    run: Any,
    table_metas: dict[str, dict[str, Any]],
    factor_values: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """构建可 JSON 化的结果（指标+图表+表元数据）。巨表明细不在内,走 /table 分页。
    不含 downloads —— 那个按需实时探测磁盘,因为 dump 是异步落盘的。"""
    return {
        "run": run.to_state().model_dump(),
        "factor": {
            "class": run.factor_class,
            "name": run.name,
            "parameters": run.parameters,
            "compute_kwargs": run.compute_kwargs,
        },
        "metrics": build_metrics(run.analyst, run.backtest),
        "charts": build_chart_data(run.backtest, factor_values),
        "tables": table_metas,
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
