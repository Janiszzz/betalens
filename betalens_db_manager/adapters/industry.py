"""Industry-membership source normalization."""

from __future__ import annotations

import re

import pandas as pd


def build_industry_records(
    df: pd.DataFrame,
    scheme: str = "申万一级行业",
    code_col: str = "code",
    name_col: str = "name",
    date_col: str = "effective_dt",
    ind_name_col: str = "ind_name",
    ind_code_col: str | None = "ind_code",
) -> pd.DataFrame:
    """Convert industry events to the standard six-column import frame."""

    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(df[date_col])
    out["code"] = df[code_col].astype(str)
    out["name"] = df[name_col].astype(str)
    out["metric"] = scheme

    def numeric_code(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        match = re.search(r"\d+", str(value))
        return int(match.group()) if match else None

    if ind_code_col and ind_code_col in df.columns:
        out["value"] = df[ind_code_col].apply(numeric_code)
        industry_codes = df[ind_code_col]
    else:
        out["value"] = None
        industry_codes = pd.Series([None] * len(df), index=df.index)

    out["remark"] = [
        {
            "ind_name": None if pd.isna(name) else str(name),
            "ind_code": None if pd.isna(code) else str(code),
            "scheme": scheme,
        }
        for name, code in zip(df[ind_name_col], industry_codes)
    ]
    return out[["datetime", "code", "name", "metric", "value", "remark"]]
