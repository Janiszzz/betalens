from __future__ import annotations

import math
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from betalens.datafeed import Datafeed
from betalens.eventstudy.eventstudy import EventStudy

from .factors import FACTOR_ROOT, REPO_ROOT


EVENT_ROOT = FACTOR_ROOT / "eventstudy"
EVENT_OUTPUT_ROOT = Path(tempfile.gettempdir()) / "betalens_dashboard_eventstudy"
EVENT_PARAMS_FILE = EVENT_ROOT / "eventstudy_params.json"
EVENT_FALLBACK_PARAMS: dict[str, Any] = {
    "event_file": "1.春节假期.xlsx",
    "code": "000906.SH",
    "benchmark_code": "",
    "metric": "收盘价",
    "table_name": "daily_market",
    "mode": "flexible",
    "window_before": 20,
    "window_after": 20,
    "holding_start_offset": 0,
    "market_close_hour": 15,
    "holding_days": "1,2,3,4,5",
    "holding_months": "1,3,6,9,12",
    "save_results": False
}


def load_eventstudy_params() -> dict[str, Any]:
    params = dict(EVENT_FALLBACK_PARAMS)
    if not EVENT_PARAMS_FILE.exists():
        return params
    try:
        loaded = json.loads(EVENT_PARAMS_FILE.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            params.update(loaded)
    except Exception:
        pass
    return params


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _records(df: pd.DataFrame | None, index_name: str = "day") -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.copy().reset_index()
    if out.columns[0] == "index":
        out = out.rename(columns={"index": index_name})
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(pd.notnull(out), None)
    return [{str(k): _clean_scalar(v) for k, v in row.items()} for row in out.to_dict("records")]


def _safe_event_path(file_id: str) -> Path:
    candidate = (EVENT_ROOT / file_id).resolve()
    root = EVENT_ROOT.resolve()
    if root not in candidate.parents and candidate != root:
        raise FileNotFoundError("invalid event file")
    if candidate.suffix.lower() not in {".xlsx", ".xls", ".csv"}:
        raise FileNotFoundError("unsupported event file")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"event file not found: {file_id}")
    return candidate


def _read_event_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    if "date" not in df.columns:
        raise ValueError(f"{path.name} 缺少 date 列")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    if "event" not in out.columns:
        out["event"] = 1
    out["event"] = pd.to_numeric(out["event"], errors="coerce").fillna(0).astype(int)
    return out.sort_values("date")


def _event_series(path: Path) -> pd.Series:
    df = _read_event_frame(path)
    events = df.set_index("date")["event"].sort_index()
    return events[events == 1]


def discover_event_files() -> dict[str, Any]:
    defaults = load_eventstudy_params()
    if not EVENT_ROOT.exists():
        return {"defaults": defaults, "files": []}

    files = sorted(
        [
            p
            for p in EVENT_ROOT.iterdir()
            if p.is_file() and p.suffix.lower() in {".xlsx", ".xls", ".csv"}
        ],
        key=lambda p: p.name,
    )
    result: list[dict[str, Any]] = []
    for path in files:
        try:
            df = _read_event_frame(path)
            events = df[df["event"] == 1]
            sample_cols = [c for c in ("date", "event", "remark") if c in df.columns]
            result.append(
                {
                    "id": path.name,
                    "name": path.stem,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "eventCount": int(len(events)),
                    "dateFrom": events["date"].min().strftime("%Y-%m-%d") if len(events) else "",
                    "dateTo": events["date"].max().strftime("%Y-%m-%d") if len(events) else "",
                    "columns": [str(c) for c in df.columns],
                    "sample": [
                        {
                            str(k): (
                                v.strftime("%Y-%m-%d %H:%M:%S")
                                if isinstance(v, pd.Timestamp)
                                else _clean_scalar(v)
                            )
                            for k, v in row.items()
                        }
                        for row in df[sample_cols].head(5).to_dict("records")
                    ],
                }
            )
        except Exception as exc:
            result.append(
                {
                    "id": path.name,
                    "name": path.stem,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "eventCount": 0,
                    "dateFrom": "",
                    "dateTo": "",
                    "columns": [],
                    "sample": [],
                    "error": str(exc),
                }
            )
    return {"defaults": defaults, "files": result}


def _parse_codes(value: Any) -> str | list[str]:
    if isinstance(value, list):
        codes = [str(v).strip() for v in value if str(v).strip()]
    else:
        text = str(value or "")
        codes = [part.strip() for part in text.replace("\n", ",").replace(";", ",").split(",") if part.strip()]
    if not codes:
        raise ValueError("至少需要一个标的代码")
    return codes[0] if len(codes) == 1 else codes


def _parse_int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value or "").replace("，", ",").split(",")
    result = []
    for item in raw:
        text = str(item).strip()
        if text:
            result.append(int(text))
    return result


def _build_holding_periods(params: dict[str, Any]) -> dict[str, list[int]] | None:
    if str(params.get("mode", "flexible")) != "fixed":
        return None
    days = _parse_int_list(params.get("holding_days", "1,2,3,4,5"))
    months = _parse_int_list(params.get("holding_months", "1,3,6,9,12"))
    return {"days": days, "months": months}


def _param_value(params: dict[str, Any], snake_name: str, camel_name: str, fallback: Any) -> Any:
    value = params.get(snake_name)
    if value not in (None, ""):
        return value
    value = params.get(camel_name)
    if value not in (None, ""):
        return value
    return fallback


def _key_metric(rows: list[dict[str, Any]], day: int) -> dict[str, Any] | None:
    exact = next((row for row in rows if row.get("day") == day), None)
    if exact:
        return exact
    return rows[-1] if rows else None


def _returns_matrix_records(df: pd.DataFrame | None, max_events: int = 30) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    limited = df.iloc[:, :max_events]
    rows = []
    for day, series in limited.iterrows():
        for event_idx, value in series.items():
            rows.append(
                {
                    "day": _clean_scalar(day),
                    "event": str(event_idx),
                    "return": _clean_scalar(value),
                }
            )
    return rows


def _cumulative_matrix_records(df: pd.DataFrame | None, max_events: int = 30) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    limited = df.iloc[:, :max_events]
    rows = []
    for day, series in limited.iterrows():
        for event_idx, value in series.items():
            rows.append(
                {
                    "day": _clean_scalar(day),
                    "event": str(event_idx),
                    "cumulativeReturn": _clean_scalar(value),
                }
            )
    return rows


def _event_rows(path: Path) -> list[dict[str, Any]]:
    df = _read_event_frame(path)
    columns = [c for c in ("date", "event", "remark") if c in df.columns]
    out = df[df["event"] == 1][columns].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out = out.where(pd.notnull(out), None)
    return [{str(k): _clean_scalar(v) for k, v in row.items()} for row in out.to_dict("records")]


def run_event_study(params: dict[str, Any]) -> dict[str, Any]:
    defaults = load_eventstudy_params()
    merged = {**defaults, **{k: v for k, v in params.items() if v not in (None, "")}}

    file_id = str(merged.get("event_file") or merged.get("eventFile") or "")
    path = _safe_event_path(file_id)
    events = _event_series(path)
    if events.empty:
        raise ValueError("事件文件中没有 event=1 的记录")

    code = _parse_codes(merged.get("code"))
    benchmark_code = str(merged.get("benchmark_code") or merged.get("benchmarkCode") or "").strip() or None
    metric = str(merged.get("metric") or EVENT_FALLBACK_PARAMS["metric"])
    table_name = str(merged.get("table_name") or merged.get("tableName") or EVENT_FALLBACK_PARAMS["table_name"])
    mode = str(merged.get("mode") or EVENT_FALLBACK_PARAMS["mode"])
    window_before = int(_param_value(merged, "window_before", "windowBefore", EVENT_FALLBACK_PARAMS["window_before"]))
    window_after = int(_param_value(merged, "window_after", "windowAfter", EVENT_FALLBACK_PARAMS["window_after"]))
    holding_start_offset = int(_param_value(merged, "holding_start_offset", "holdingStartOffset", 0))
    market_close_hour = int(_param_value(merged, "market_close_hour", "marketCloseHour", EVENT_FALLBACK_PARAMS["market_close_hour"]))

    datafeed = Datafeed(table_name)
    try:
        study = EventStudy(datafeed)
        raw = study.analyze(
            events=events,
            code=code,
            benchmark_code=benchmark_code,
            window_before=window_before,
            window_after=window_after,
            metric=metric,
            mode=mode,
            holding_periods=_build_holding_periods(merged),
            holding_start_offset=holding_start_offset,
            market_close_hour=market_close_hour,
        )
    finally:
        datafeed.close()

    if "error" in raw:
        raise ValueError(str(raw["error"]))

    daily = _records(raw.get("daily_stats"), "day")
    cumulative = _records(raw.get("cumulative_stats"), "day")
    day0 = _key_metric(daily, 0)
    final = _key_metric(cumulative, window_after)

    result = {
        "eventFile": {
            "id": path.name,
            "name": path.stem,
            "path": str(path.relative_to(REPO_ROOT)),
        },
        "parameters": {
            "code": code,
            "benchmarkCode": benchmark_code,
            "metric": metric,
            "tableName": table_name,
            "mode": mode,
            "windowBefore": window_before,
            "windowAfter": window_after,
            "holdingStartOffset": holding_start_offset,
            "marketCloseHour": market_close_hour,
        },
        "summary": {
            "eventCount": int(raw.get("event_count", 0)),
            "validCodes": raw.get("valid_codes", [code] if isinstance(code, str) else code),
            "day0Mean": day0.get("mean") if day0 else None,
            "day0TStat": day0.get("t_stat") if day0 else None,
            "day0PositiveProb": day0.get("positive_prob") if day0 else None,
            "finalDay": final.get("day") if final else None,
            "finalMean": final.get("mean") if final else None,
            "finalTStat": final.get("t_stat") if final else None,
            "finalPositiveProb": final.get("positive_prob") if final else None,
        },
        "charts": {
            "dailyStats": daily,
            "cumulativeStats": cumulative,
            "returnsMatrix": _returns_matrix_records(raw.get("returns_matrix")),
            "cumulativeReturnsMatrix": _cumulative_matrix_records(raw.get("cumulative_returns_matrix")),
        },
        "tables": {
            "dailyStats": daily,
            "cumulativeStats": cumulative,
            "events": _event_rows(path),
        },
    }
    return result
