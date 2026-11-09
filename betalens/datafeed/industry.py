"""Historical industry membership from complete, published reference scopes."""
from __future__ import annotations

import re
import pandas as pd

from .contracts import INDUSTRY_COLUMNS, PITUnavailableError, VersionConflictError, frame, local_time, timestamp, available, pit_policy as resolve_policy
from .query import _codes, _name_rows, historical_name
from .readers import get_reader, require_columns
from .reference import load_reference, known_scopes

DEFAULT_TABLE = "security_industry_membership"
VERSION_TABLE = "stock_industry_membership_versions"


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
                "sec_name": historical_name(names, code, at, "strict"), "industry_value": float(digits) if digits else None,
                "remark": remark, **remark})
    return frame(records, INDUSTRY_COLUMNS, numeric=("industry_value",), times=("query_date", "effective_dt"))


def query_industry(source, codes, dates, scheme_code=None, table_name=DEFAULT_TABLE, *, level=1, pit_policy=None):
    policy = resolve_policy(source, pit_policy)
    if table_name == DEFAULT_TABLE:
        if policy == "strict":
            raise PITUnavailableError("strict 行业查询须显式使用 stock_industry_membership_versions")
        return research_industry(source, codes, dates, scheme_code, level)
    if table_name != VERSION_TABLE:
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
                         *, by="name", level=1, pit_policy=None):
    policy = resolve_policy(source, pit_policy)
    if table_name == DEFAULT_TABLE:
        if policy == "strict":
            raise PITUnavailableError("strict 行业查询须显式使用 stock_industry_membership_versions")
        return research_members(source, industry, date, scheme_code, by, level)
    if table_name != VERSION_TABLE:
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


def _research_rows(reader, codes):
    membership_fields = {"security_code": "text", "scheme_code": "text",
        "industry_code": "text", "valid_from": "date", "valid_to": "date",
        "available_at": "timestamp", "availability_basis": "text", "data_quality_status": "text"}
    dimension_fields = {"scheme_code": "text", "industry_code": "text", "industry_name": "text", "level": "text"}
    scheme_fields = {"scheme_code": "text", "event_date": "date", "available_at": "timestamp",
        "availability_basis": "text", "data_quality_status": "text"}
    require_columns(reader, DEFAULT_TABLE, membership_fields)
    require_columns(reader, "industry_dim", dimension_fields)
    require_columns(reader, "industry_scheme_dim", scheme_fields)
    dimensions = {}
    for row in reader.rows("industry_dim", list(dimension_fields)):
        key = (row["scheme_code"], row["industry_code"])
        value = (row["industry_name"], row["level"])
        if key in dimensions and dimensions[key] != value:
            raise VersionConflictError("行业字典冲突: " + str(key))
        dimensions[key] = value
    schemes = reader.rows("industry_scheme_dim", list(scheme_fields))
    filters = [] if codes is None else [("security_code", "IN", list(dict.fromkeys(codes)))]
    memberships = reader.rows(DEFAULT_TABLE, list(membership_fields), filters=filters)
    return schemes, dimensions, memberships


def _research_scheme(schemes, at, requested):
    day = local_time(at).date()
    eligible = [r for r in schemes if available(r, at) and r.get("event_date") is not None
                and r["event_date"] <= day and (r["scheme_code"] == requested if requested else
                re.fullmatch(r"SW[0-9]{4}", r["scheme_code"]))]
    if not eligible:
        raise PITUnavailableError("查询时点缺少已可得且生效的行业方案: " + str(requested or "申万"))
    latest = max(r["event_date"] for r in eligible)
    selected = {r["scheme_code"] for r in eligible if r["event_date"] == latest}
    if len(selected) != 1:
        raise VersionConflictError("行业方案生效时间冲突")
    return selected.pop()


def research_industry(source, codes, dates, scheme_code=None, level=1):
    level = _parameters(scheme_code, level)
    codes = _codes(codes, required=True)
    dates = [timestamp(d) for d in ([dates] if isinstance(dates, str) else dates)]
    if not codes or not dates:
        return frame([], INDUSTRY_COLUMNS, numeric=("industry_value",), times=("query_date", "effective_dt"))
    reader = get_reader(source)
    with reader.snapshot():
        schemes, dimensions, memberships = _research_rows(reader, codes)
        names = _name_rows(reader, codes)
    records = []
    for at in dates:
        selected = _research_scheme(schemes, at, scheme_code)
        day = local_time(at).date()
        hits = {}
        for row in memberships:
            if row["scheme_code"] != selected or not available(row, at):
                continue
            if row["valid_from"] is None or row["valid_from"] > day or (row["valid_to"] is not None and day >= row["valid_to"]):
                continue
            dimension = dimensions.get((selected, row["industry_code"]))
            if dimension is None:
                raise VersionConflictError("行业归属缺少对应字典: " + row["industry_code"])
            if dimension[1] != "L" + str(level):
                continue
            code = row["security_code"]
            if code in hits and hits[code] != row:
                raise VersionConflictError("公司行业归属重叠: " + code)
            hits[code] = row
        for code in codes:
            hit = hits.get(code)
            ind_code = hit["industry_code"] if hit else None
            name = dimensions[(selected, ind_code)][0] if hit else None
            label = _scheme_label(selected, level)
            remark = {"scheme_code": selected, "scheme": label, "ind_code": ind_code,
                "ind_name": name, "pit_policy": "research",
                "availability_basis": hit.get("availability_basis") if hit else None}
            digits = re.sub(r"\D", "", ind_code or "")
            records.append({"code": code, "query_date": local_time(at),
                "effective_dt": pd.Timestamp(hit["valid_from"]) if hit else pd.NaT,
                "sec_name": historical_name(names, code, at),
                "industry_value": float(digits) if digits else None, "remark": remark,
                "ind_name": name, "ind_code": ind_code, "scheme": label})
    # Keep the same code-major order as the strict implementation, including duplicate inputs.
    result = frame(records, INDUSTRY_COLUMNS, numeric=("industry_value",), times=("query_date", "effective_dt"))
    order = [i for n in range(len(codes)) for i in range(n, len(records), len(codes))]
    return result.iloc[order].reset_index(drop=True)


def research_members(source, industry, date, scheme_code, by, level):
    if by not in {"name", "value", "code"}:
        raise ValueError("by 必须是 name/value/code")
    reader = get_reader(source)
    with reader.snapshot():
        schemes, _, rows = _research_rows(reader, None)
        at = timestamp(date)
        selected = _research_scheme(schemes, at, scheme_code)
        day = local_time(at).date()
        codes = sorted({r["security_code"] for r in rows if r["scheme_code"] == selected
                        and available(r, at) and r["valid_from"] is not None and r["valid_from"] <= day
                        and (r["valid_to"] is None or day < r["valid_to"])})
        result = research_industry(source, codes, [date], selected, level)
    if not codes or result.ind_code.isna().any():
        raise PITUnavailableError("行业反查存在未知归属，不能解释为完整成分集合")
    column = "industry_value" if by == "value" else "ind_code" if by == "code" else "ind_name"
    return result.loc[result[column] == industry, ["code", "sec_name", "industry_value", "remark", "ind_name", "ind_code", "scheme"]].reset_index(drop=True)
