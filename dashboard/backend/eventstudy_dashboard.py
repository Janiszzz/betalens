from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from betalens.analyst.naming import get_name_map
from betalens.datafeed import Datafeed
from betalens.eventstudy.eventstudy import EventStudy
from betalens.factor.config import load_yaml_config, section

from .factors import FACTOR_ROOT, REPO_ROOT


EVENT_ROOT = FACTOR_ROOT / "tools" / "eventstudy"
EVENT_OUTPUT_ROOT = Path(tempfile.gettempdir()) / "betalens_dashboard_eventstudy"
EVENT_PARAMS_FILE = EVENT_ROOT / "eventstudy.yaml"
MAX_COMPARISON_EVENTS = 30


def load_eventstudy_params() -> dict[str, Any]:
    loaded = load_yaml_config(EVENT_PARAMS_FILE, required_sections=("eventstudy",))
    return section(loaded, "eventstudy", context=str(EVENT_PARAMS_FILE))


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
    items = value if isinstance(value, list) else [value]
    codes = [
        code.strip()
        for item in items
        for code in re.split(r"[,，;；\r\n]+", str(item or ""))
        if code.strip()
    ]
    codes = list(dict.fromkeys(codes))
    if not codes:
        raise ValueError("至少需要一个标的代码")
    return codes[0] if len(codes) == 1 else codes


def _asset_payload(codes: Any) -> list[dict[str, str | None]]:
    if isinstance(codes, str):
        normalized_codes = [codes]
    else:
        normalized_codes = [str(code) for code in (codes or []) if str(code).strip()]
    normalized_codes = list(dict.fromkeys(code.strip() for code in normalized_codes if code.strip()))

    try:
        name_map = get_name_map(normalized_codes)
    except Exception:
        name_map = {}

    assets: list[dict[str, str | None]] = []
    for code in normalized_codes:
        raw_name = name_map.get(code)
        name = str(raw_name).strip() if raw_name is not None and pd.notna(raw_name) else None
        if not name:
            name = None
        assets.append(
            {
                "code": code,
                "name": name,
                "label": f"{code} {name}" if name else code,
            }
        )
    return assets


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
    if str(params["mode"]) != "fixed":
        return None
    days = _parse_int_list(params["holding_days"])
    months = _parse_int_list(params["holding_months"])
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


def _event_date_for_index(event_dates: Any, event_idx: Any) -> Any:
    if event_dates is None:
        return None
    try:
        idx = int(event_idx)
    except (TypeError, ValueError):
        return None
    try:
        return _clean_scalar(event_dates[idx])
    except (IndexError, KeyError, TypeError):
        return None


def _returns_matrix_records(
    df: pd.DataFrame | None,
    event_dates: Any = None,
    max_events: int = 30
) -> list[dict[str, Any]]:
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
                    "eventDate": _event_date_for_index(event_dates, event_idx),
                    "return": _clean_scalar(value),
                }
            )
    return rows


def _cumulative_matrix_records(
    df: pd.DataFrame | None,
    event_dates: Any = None,
    max_events: int = 30
) -> list[dict[str, Any]]:
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
                    "eventDate": _event_date_for_index(event_dates, event_idx),
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


def _comparison_payload(raw: dict[str, Any], window_after: int) -> dict[str, Any] | None:
    comparison = raw.get("comparison")
    if not comparison:
        return None

    events = comparison.get("events", [])
    displayed_events = events[:MAX_COMPARISON_EVENTS]
    displayed_ids = {int(event["event_id"]) for event in displayed_events}
    event_by_id = {
        int(event["event_id"]): _clean_scalar(event.get("event_date"))
        for event in events
    }
    summary_by_code: list[dict[str, Any]] = []
    daily_by_code: list[dict[str, Any]] = []
    cumulative_by_code: list[dict[str, Any]] = []
    event_cumulative_by_code: list[dict[str, Any]] = []

    for code, item in comparison.get("by_code", {}).items():
        daily = _records(item.get("daily_stats"), "day")
        cumulative = _records(item.get("cumulative_stats"), "day")
        day0 = _key_metric(daily, 0)
        final = _key_metric(cumulative, window_after)
        summary_by_code.append(
            {
                "code": str(code),
                "eventCount": int(item.get("event_count", 0)),
                "coverage": _clean_scalar(item.get("coverage")),
                "day0Mean": day0.get("mean") if day0 else None,
                "day0TStat": day0.get("t_stat") if day0 else None,
                "day0PositiveProb": day0.get("positive_prob") if day0 else None,
                "finalDay": final.get("day") if final else None,
                "finalMean": final.get("mean") if final else None,
                "finalTStat": final.get("t_stat") if final else None,
                "finalPositiveProb": final.get("positive_prob") if final else None,
            }
        )
        daily_by_code.extend({"code": str(code), **row} for row in daily)
        cumulative_by_code.extend({"code": str(code), **row} for row in cumulative)

        cumulative_matrix = item.get("cumulative_returns_matrix")
        if cumulative_matrix is None or cumulative_matrix.empty:
            continue
        for day, series in cumulative_matrix.iterrows():
            for event_id, value in series.items():
                normalized_event_id = int(event_id)
                if normalized_event_id not in displayed_ids:
                    continue
                event_cumulative_by_code.append(
                    {
                        "code": str(code),
                        "eventId": normalized_event_id,
                        "eventDate": event_by_id.get(normalized_event_id),
                        "day": _clean_scalar(day),
                        "cumulativeReturn": _clean_scalar(value),
                    }
                )

    return {
        "mode": "compare",
        "events": [
            {
                "eventId": int(event["event_id"]),
                "eventDate": _clean_scalar(event.get("event_date")),
            }
            for event in displayed_events
        ],
        "validCodes": [str(code) for code in comparison.get("valid_codes", [])],
        "skippedCodes": [
            {str(key): _clean_scalar(value) for key, value in item.items()}
            for item in comparison.get("skipped_codes", [])
        ],
        "totalEventCount": len(events),
        "displayedEventCount": len(displayed_events),
        "truncated": len(events) > len(displayed_events),
        "summaryByCode": summary_by_code,
        "dailyByCode": daily_by_code,
        "cumulativeByCode": cumulative_by_code,
        "eventCumulativeByCode": event_cumulative_by_code,
    }


def run_event_study(params: dict[str, Any]) -> dict[str, Any]:
    defaults = load_eventstudy_params()
    merged = {**defaults, **{k: v for k, v in params.items() if v not in (None, "")}}

    file_id = str(merged.get("event_file") or merged.get("eventFile") or "")
    path = _safe_event_path(file_id)
    events = _event_series(path)
    if events.empty:
        raise ValueError("事件文件中没有 event=1 的记录")

    # A missing request field keeps the YAML default for backward compatibility,
    # while an explicitly blank field must not silently run a different target.
    code_value = defaults.get("code") if params.get("code") is None else params.get("code")
    code = _parse_codes(code_value)
    benchmark_code = str(merged.get("benchmark_code") or merged.get("benchmarkCode") or "").strip() or None
    metric = str(merged.get("metric"))
    table_name = str(merged.get("table_name") or merged.get("tableName"))
    mode = str(merged.get("mode"))
    multi_asset_mode = str(
        _param_value(merged, "multi_asset_mode", "multiAssetMode", "aggregate")
    )
    window_before = int(_param_value(merged, "window_before", "windowBefore", merged["window_before"]))
    window_after = int(_param_value(merged, "window_after", "windowAfter", merged["window_after"]))
    holding_start_offset = int(_param_value(merged, "holding_start_offset", "holdingStartOffset", merged["holding_start_offset"]))
    market_close_hour = int(_param_value(merged, "market_close_hour", "marketCloseHour", merged["market_close_hour"]))

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
            multi_asset_mode=multi_asset_mode,
        )
    finally:
        datafeed.close()

    if "error" in raw:
        raise ValueError(str(raw["error"]))

    valid_codes = raw.get("valid_codes", [code] if isinstance(code, str) else code)
    assets = _asset_payload(valid_codes)
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
        "assets": assets,
        "parameters": {
            "code": code,
            "benchmarkCode": benchmark_code,
            "metric": metric,
            "tableName": table_name,
            "mode": mode,
            "multiAssetMode": multi_asset_mode,
            "windowBefore": window_before,
            "windowAfter": window_after,
            "holdingStartOffset": holding_start_offset,
            "marketCloseHour": market_close_hour,
        },
        "summary": {
            "eventCount": int(raw.get("event_count", 0)),
            "validCodes": valid_codes,
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
            "returnsMatrix": _returns_matrix_records(raw.get("returns_matrix"), raw.get("event_dates")),
            "cumulativeReturnsMatrix": _cumulative_matrix_records(
                raw.get("cumulative_returns_matrix"),
                raw.get("event_dates"),
            ),
        },
        "tables": {
            "dailyStats": daily,
            "cumulativeStats": cumulative,
            "events": _event_rows(path),
        },
    }
    comparison = _comparison_payload(raw, window_after)
    if comparison is not None:
        result["comparison"] = comparison
    return result
