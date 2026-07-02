"""Small utility helpers used by the database manager."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd


DATABASE_CONFIG_KEYS = ("dbname", "user", "password", "host", "port")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_database_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return only psycopg2 connection keys from a datafeed config section."""
    return {key: config[key] for key in DATABASE_CONFIG_KEYS if key in config}


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        if pd.isna(value):
            return None
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        try:
            if value.is_nan():
                return None
        except Exception:
            pass
        return float(value)
    try:
        if not isinstance(value, (list, tuple, dict, set)) and pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def to_json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, default=json_default) + "\n"


def dataframe_preview(df: pd.DataFrame, rows: int = 100) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.head(rows).copy()
    out = out.where(pd.notnull(out), None)
    return out.to_dict("records")


def dataframe_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "rows": 0,
            "codes": 0,
            "metrics": 0,
            "date_min": None,
            "date_max": None,
        }
    result: dict[str, Any] = {"rows": int(len(df))}
    if "code" in df.columns:
        result["codes"] = int(df["code"].nunique(dropna=True))
    if "metric" in df.columns:
        result["metrics"] = int(df["metric"].nunique(dropna=True))
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        result["date_min"] = None if dt.dropna().empty else dt.min()
        result["date_max"] = None if dt.dropna().empty else dt.max()
    return result
