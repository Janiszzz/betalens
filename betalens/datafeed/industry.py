"""Historical industry membership from complete, published reference scopes."""
from __future__ import annotations

import re
import pandas as pd

from .contracts import INDUSTRY_COLUMNS, PITUnavailableError, VersionConflictError, frame, local_time, timestamp
from .query import _codes, _name_rows, historical_name
from .readers import get_reader
from .reference import load_reference, known_scopes

DEFAULT_TABLE = "stock_industry_membership_versions"


def _parameters(scheme_code, level):
    from .contracts import integer
    level = integer(level, "level", minimum=1)
    if level not in {1, 2, 3}:
        raise ValueError("level 必须是 1、2 或 3")
    if scheme_code is not None and not re.fullmatch(r"SW[0-9]{4}|CITICS", scheme_code):
        raise ValueError("scheme_code 必须是精确版本代码，如 SW2014/SW2021/CITICS；省略时只选择申万")
    return level


def _active(row, day):
    return (row.get("valid_from") is not None and row["valid_from"] <= day
            and (row.get("valid_to") is None or day < row["valid_to"]))


def _version_scopes(scopes, at, scheme_code):
    result = [scope for scope in known_scopes(scopes, at)
              if scope["selector"].get("scheme_code") == scheme_code or
              (scheme_code is None and re.fullmatch(r"SW[0-9]{4}", scope["selector"].get("scheme_code", "")))]
    for scope in result:
        selector = scope["selector"]
        for row in scope["rows"]:
            if any(row.get(key) != selector.get(key) for key in ("security_code", "scheme_code", "level")):
                raise VersionConflictError("行业记录与完整范围选择器不一致")
            if row.get("classification_version") is not None and row["classification_version"] != row["scheme_code"]:
                raise VersionConflictError("行业年份版本与 scheme_code 不一致")
    return [scope for scope in result if all(row.get("classification_version") for row in scope["rows"])]


def _select_scheme(scopes, at, scheme_code):
    if scheme_code is not None:
        return scheme_code
    day = local_time(at).date().isoformat()
    candidates = set()
    for scope in scopes:
        scheme = scope["selector"].get("scheme_code", "")
        if not re.fullmatch(r"SW[0-9]{4}", scheme):
            continue
        if scope.get("evidence", {}).get("classification_version_basis") != "versioned_dictionary_code_match":
            continue
        if any(_active(row, day) and row.get("membership_status") == "mapped"
               and row.get("boundary_basis") in {"source_effective_interval", "source_in_date_open_end", "verified_left_closed_right_open"}
               for row in scope["rows"]):
            candidates.add(scheme)
    if not candidates:
        raise PITUnavailableError("查询时点缺少已发布、历史可知且有效的申万版本证据")
    if len(candidates) != 1:
        raise VersionConflictError("查询时点存在多个有效申万版本，请提供更完整的版本边界证据或显式指定 scheme_code")
    return candidates.pop()


def _scheme_label(code, level):
    if not code:
        return None
    label = {1: "一", 2: "二", 3: "三"}[level] + "级行业"
    return "申万" + label + "（" + code[2:] + "）" if code.startswith("SW") else "中信" + label


def _industry_frame(reader, scopes, codes, dates, scheme_code, level):
    names = _name_rows(reader, codes) if codes and dates else {}
    snapshots = {}
    for date in dates:
        at = timestamp(date)
        if at in snapshots:
            continue
        by_code = {}
        known = _version_scopes(scopes, at, scheme_code)
        selected = _select_scheme(known, at, scheme_code)
        for scope in known:
            selector = scope["selector"]
            if selector.get("scheme_code") == selected and selector.get("level") == level:
                by_code.setdefault(selector["security_code"], []).extend(scope["rows"])
        snapshots[at] = by_code
    records = []
    for code in codes:
        for date in dates:
            at = timestamp(date)
            day = local_time(at).date().isoformat()
            active = [r for r in snapshots[at].get(code, []) if not r.get("valid_from") or _active(r, day)]
            unknown = any(not r.get("valid_from") or not r.get("industry_code")
                          or r.get("membership_status") != "mapped"
                          or r.get("boundary_basis") not in {"source_effective_interval", "source_in_date_open_end", "verified_left_closed_right_open"}
                          for r in active)
            candidates = [] if unknown else active
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


def query_industry(source, codes, dates, scheme_code=None, table_name=DEFAULT_TABLE, *, level=1):
    if table_name != DEFAULT_TABLE:
        raise NotImplementedError("行业查询仅支持 stock_industry_membership_versions")
    level = _parameters(scheme_code, level)
    codes = _codes(codes, required=True)
    dates = [dates] if isinstance(dates, str) else list(dates)
    dates = [timestamp(value) for value in dates]
    if not codes or not dates:
        return frame([], INDUSTRY_COLUMNS, numeric=("industry_value",), times=("query_date", "effective_dt"))
    reader = get_reader(source)
    with reader.snapshot():
        # Auto selection sees the same version evidence across stocks/levels.
        scopes = load_reference(reader, "industry", codes=None if scheme_code is None else codes) if codes and dates else []
        return _industry_frame(reader, scopes, codes, dates, scheme_code, level)


def get_industry_members(source, industry, date, scheme_code=None, table_name=DEFAULT_TABLE,
                         *, by="name", level=1):
    if table_name != DEFAULT_TABLE:
        raise NotImplementedError("行业查询仅支持 stock_industry_membership_versions")
    level = _parameters(scheme_code, level)
    if by not in {"name", "value", "code"}:
        raise ValueError("by 必须是 name/value/code")
    reader = get_reader(source)
    with reader.snapshot():
        scopes = load_reference(reader, "industry")
        codes = sorted({s["selector"]["security_code"] for s in scopes})
        result = _industry_frame(reader, scopes, codes, [date], scheme_code, level)
    if not codes or result.ind_code.isna().any():
        raise PITUnavailableError("行业反查存在未知归属，不能解释为完整成分集合")
    column = "industry_value" if by == "value" else "ind_code" if by == "code" else "ind_name"
    columns = ["code", "sec_name", "industry_value", "remark", "ind_name", "ind_code", "scheme"]
    return result.loc[result[column] == industry, columns].reset_index(drop=True)
