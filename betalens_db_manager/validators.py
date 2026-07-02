"""Validation helpers for import-ready database frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .constants import REQUIRED_DB_COLUMNS


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def validate_import_frame(df: pd.DataFrame, allow_unsafe_metrics: bool = False, allow_nan_values: bool = False) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    if df is None or df.empty:
        errors.append("没有可导入的数据")
        return ValidationReport(False, errors, warnings, {"rows": 0})

    missing = [c for c in REQUIRED_DB_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"缺少必填列: {missing}")

    if "metric" in df.columns:
        unnamed = df["metric"].astype(str).str.startswith("Unnamed:", na=False)
        if unnamed.any():
            msg = f"发现 Unnamed:* 指标 {int(unnamed.sum())} 行"
            if allow_unsafe_metrics:
                warnings.append(msg)
            else:
                errors.append(msg)

    if "value" in df.columns:
        values = pd.to_numeric(df["value"], errors="coerce")
        nan_mask = values.isna() | np.isinf(values.to_numpy(dtype=float, na_value=np.nan))
        if nan_mask.any():
            msg = f"发现 NaN/Inf value {int(nan_mask.sum())} 行"
            if allow_nan_values:
                warnings.append(msg)
            else:
                errors.append(msg)

    if "datetime" in df.columns:
        parsed = pd.to_datetime(df["datetime"], errors="coerce")
        bad_dt = parsed.isna()
        if bad_dt.any():
            errors.append(f"发现无法解析的 datetime {int(bad_dt.sum())} 行")

    if {"datetime", "code", "metric"}.issubset(df.columns):
        duplicates = df.duplicated(subset=["datetime", "code", "metric"], keep=False)
        if duplicates.any():
            warnings.append(f"待导入文件内部有重复键 {int(duplicates.sum())} 行")

    stats = {
        "rows": int(len(df)),
        "codes": int(df["code"].nunique(dropna=True)) if "code" in df.columns else 0,
        "metrics": int(df["metric"].nunique(dropna=True)) if "metric" in df.columns else 0,
    }
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        stats["date_min"] = None if dt.dropna().empty else dt.min()
        stats["date_max"] = None if dt.dropna().empty else dt.max()
    return ValidationReport(not errors, errors, warnings, stats)

