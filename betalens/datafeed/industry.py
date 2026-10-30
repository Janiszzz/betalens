"""Historical industry membership from complete, published reference scopes."""
from __future__ import annotations

import re
import pandas as pd

from .contracts import INDUSTRY_COLUMNS, PITUnavailableError, VersionConflictError, frame, local_time, timestamp
from .query import _codes, _name_rows, historical_name
from .readers import get_reader
from .reference import load_reference, known_scopes

DEFAULT_TABLE = "stock_industry_membership_versions"


def _scheme(scheme):
    text = scheme.replace("（", "(").replace("）", ")")
    found = re.fullmatch(r"(申万|中信)(一|二|三)级行业(?:\((2014|2021)\))?", text)
    if not found:
        raise NotImplementedError("未登记行业体系: " + scheme)
    family, level, version = found.groups()
    codes = ["CITICS", "CITIC"] if family == "中信" else (["SW" + version] if version else ["SW2014", "SW2021"])
    return codes, {"一": 1, "二": 2, "三": 3}[level]


def _scheme_label(code, level):
    if not code:
        return None
    label = {1: "一", 2: "二", 3: "三"}[level] + "级行业"
    return "申万" + label + "（" + code[2:] + "）" if code.startswith("SW") else "中信" + label


def _industry_frame(reader, scopes, codes, dates, scheme, exact):
    schemes, level = _scheme(scheme)
    if exact and len(schemes) > 1 and schemes[0].startswith("SW"):
        raise ValueError("exact=True 时请明确行业版本，例如 申万一级行业（2021）")
    names = _name_rows(reader, codes) if codes and dates else {}
    snapshots = {}
    for date in dates:
        at = timestamp(date)
        if at in snapshots:
            continue
        by_code = {}
        for scope in known_scopes(scopes, at):
            selector = scope["selector"]
            if selector.get("scheme_code") in schemes and selector.get("level") == level:
                by_code.setdefault(selector["security_code"], []).extend(scope["rows"])
        snapshots[at] = by_code
    records = []
    for code in codes:
        for date in dates:
            at = timestamp(date)
            day = local_time(at).date().isoformat()
            candidates = [r for r in snapshots[at].get(code, [])
                          if r.get("valid_from") and r["valid_from"] <= day
                          and (r.get("valid_to") is None or day < r["valid_to"])
                          and r.get("boundary_basis") in {"source_effective_interval", "source_in_date_open_end", "verified_left_closed_right_open"}
                          and r.get("membership_status", "mapped") == "mapped"]
            if len(candidates) > 1:
                raise VersionConflictError(f"行业区间或分类版本重叠: {code}/{date}；请指定版本")
            hit = candidates[0] if candidates else {}
            ind_code = hit.get("industry_code")
            digits = re.sub(r"\D", "", str(ind_code or ""))
            remark = {"ind_name": hit.get("industry_name"), "ind_code": ind_code,
                      "scheme": _scheme_label(hit.get("scheme_code"), level),
                      "scheme_code": hit.get("scheme_code")}
            records.append({"code": code, "query_date": local_time(at),
                "effective_dt": pd.Timestamp(hit["valid_from"]) if hit else pd.NaT,
                "sec_name": historical_name(names, code, at), "industry_value": float(digits) if digits else None,
                "remark": remark, **remark})
    return frame(records, INDUSTRY_COLUMNS, numeric=("industry_value",), times=("query_date", "effective_dt"))


def query_industry(cursor, codes, dates, scheme="申万一级行业", table_name=DEFAULT_TABLE,
                   exact=False, logger=None):
    if table_name != DEFAULT_TABLE:
        raise NotImplementedError("行业查询仅支持 stock_industry_membership_versions")
    codes = _codes(codes, required=True)
    dates = [dates] if isinstance(dates, str) else list(dates)
    reader = get_reader(cursor)
    with reader.snapshot():
        scopes = load_reference(reader, "industry", codes=codes) if codes and dates else []
        return _industry_frame(reader, scopes, codes, dates, scheme, exact)


def get_industry_members(cursor, industry, date, scheme="申万一级行业", table_name=DEFAULT_TABLE,
                         by="name", exact=False, logger=None):
    if table_name != DEFAULT_TABLE:
        raise NotImplementedError("行业查询仅支持 stock_industry_membership_versions")
    if by not in {"name", "value", "code"}:
        raise ValueError("by 必须是 name/value/code")
    reader = get_reader(cursor)
    with reader.snapshot():
        scopes = load_reference(reader, "industry")
        codes = sorted({s["selector"]["security_code"] for s in scopes})
        result = _industry_frame(reader, scopes, codes, [date], scheme, exact)
    if not codes or result.ind_code.isna().any():
        raise PITUnavailableError("行业反查存在未知归属，不能解释为完整成分集合")
    column = "industry_value" if by == "value" or isinstance(industry, (int, float)) else "ind_code" if by == "code" else "ind_name"
    columns = ["code", "sec_name", "industry_value", "remark", "ind_name", "ind_code", "scheme"]
    return result.loc[result[column] == industry, columns].reset_index(drop=True)
