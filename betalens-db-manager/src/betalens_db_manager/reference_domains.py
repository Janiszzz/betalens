"""Pure normalization and trusted-scope reads for the three reference domains.

Presence of a row is deliberately not evidence of completeness.  Every answer
is tied to a published scope (including a complete, empty suspension query).
This module neither contacts a source nor writes a database.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .reference_contracts import VERSION
from .util import digest, dumps

SHANGHAI = ZoneInfo("Asia/Shanghai")
UTC = timezone.utc
MODULES = ("calendar", "industry", "suspension")


def _date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    value = str(value)
    try:
        return datetime.strptime(value, "%Y%m%d").date() if re.fullmatch(r"\d{8}", value) else date.fromisoformat(value)
    except ValueError:
        return None


def _days(request):
    first, last = _date(request.get("start_date")), _date(request.get("end_date"))
    if first is None or last is None or first > last:
        raise ValueError("start_date/end_date 必须是有效且有序的日期")
    if (last-first).days > 36600:
        raise ValueError("一次完整性检查的日期范围不能超过 100 年")
    return [first+timedelta(days=i) for i in range((last-first).days+1)]


def scope_id(module: str, selector: dict) -> str:
    return digest({"module": module, "selector": selector})


def _scope(module, selector, rows, complete, reasons=(), evidence=None):
    return {"scope_id": scope_id(module, selector), "module": module, "selector": selector,
            "rows": rows, "complete": bool(complete), "reasons": list(dict.fromkeys(reasons)),
            "evidence": evidence or {}, "dependencies": []}


def _codes(parameters):
    value = parameters.get("ts_code")
    if not value:
        return None
    return {str(v).strip() for v in (value if isinstance(value, (list, tuple)) else str(value).split(","))}


def _covers(bundle, *, code=None, day=None, exchange=None):
    p = bundle.get("parameters") or {}
    if code is not None and _codes(p) is not None and code not in _codes(p):
        return False
    if exchange is not None and (p.get("exchange") or "SSE") != exchange:
        return False
    if day is not None:
        exact = p.get("trade_date")
        if exact:
            return _date(exact) == day
        first, last = _date(p.get("start_date")), _date(p.get("end_date"))
        if first is None or last is None or not first <= day <= last:
            return False
    return True


def _proof(bundles):
    return [{"endpoint": b.get("endpoint"), "parameters": b.get("parameters", {}),
             "archive_sha": b.get("archive_sha"), "downloaded_at": b.get("downloaded_at"),
             "complete": bool(b.get("complete")), "reason": b.get("reason"),
             "row_count": len(b.get("rows", []))} for b in bundles]


def _source(bundle, row, row_no, event_date=None):
    return {"event_date": event_date, "available_at": None,
            "downloaded_at": bundle.get("downloaded_at"),
            "availability_basis": "source_publication_time_unknown",
            "source_name": bundle.get("source_name", "tushare"),
            "source_id": bundle.get("source_id", "tushare"),
            "source_endpoint": bundle.get("endpoint"), "source_row_no": row_no,
            "source_record_hash": digest(row), "mapping_version": VERSION,
            "data_quality_status": "accepted"}


def _distinct(rows):
    """Duplicate pages do not multiply events; differing records are retained."""
    result, seen = [], set()
    for row in rows:
        key = (row.get("source_endpoint"), row.get("source_record_hash"))
        if key not in seen:
            result.append(row)
            seen.add(key)
    return result


def _calendar_scope(exchange, day, bundles):
    reference = "SSE" if exchange == "BSE" else exchange
    candidates = [b for b in bundles if b.get("endpoint") == "trade_cal"
                  and _covers(b, day=day, exchange=reference)
                  and not (b.get("parameters") or {}).get("is_open")]
    rows = []
    for b in candidates:
        for n, r in enumerate(b.get("rows", []), 1):
            if _date(r.get("cal_date")) != day or (r.get("exchange") or "SSE") != reference:
                continue
            opened = str(r.get("is_open"))
            value = int(opened) if opened in ("0", "1") else None
            row = {"exchange": exchange, "trade_date": day.isoformat(), "is_open": value,
                   "previous_trade_date": (_date(r.get("pretrade_date")) or None),
                   "reference_exchange": reference if exchange == "BSE" else None,
                   "calendar_basis": "tushare_documented_reference_calendar" if exchange == "BSE" else "source_exchange_calendar",
                   **_source(b, r, n, day.isoformat())}
            if value is None or (r.get("pretrade_date") and row["previous_trade_date"] is None) or (row["previous_trade_date"] and row["previous_trade_date"] >= day):
                row["data_quality_status"] = "unknown"
            if isinstance(row["previous_trade_date"], date):
                row["previous_trade_date"] = row["previous_trade_date"].isoformat()
            rows.append(row)
    rows = _distinct(rows)
    complete = bool(candidates) and all(b.get("complete") for b in candidates) and bool(rows)
    reasons = [] if complete else ["calendar_date_missing" if candidates and not rows else "query_incomplete" if candidates else "not_requested"]
    if rows and any(r["is_open"] is None for r in rows):
        reasons.append("invalid_is_open")
    return _scope("calendar", {"exchange": exchange, "trade_date": day.isoformat()}, rows, complete,
                  reasons, {"queries": _proof(candidates), "reference_exchange": reference,
                            "calendar_basis": rows[0]["calendar_basis"] if rows else None})


def _history_complete(candidates):
    """Tushare's default is_new=Y is not a complete historical snapshot."""
    complete = [b for b in candidates if b.get("complete")]
    # An explicitly recorded source capability may prove an unfiltered response.
    if any(b.get("history_complete") is True for b in complete):
        return True
    flags = {str((b.get("parameters") or {}).get("is_new", "Y")).upper() for b in complete}
    return {"Y", "N"} <= flags


def _industry_scope(code, scheme, level, bundles):
    endpoint = "ci_index_member" if scheme in ("CITICS", "CI", "CITIC") else "index_member_all"
    # A stock-scoped query can establish a full snapshot.  An industry-filtered
    # query cannot prove that the stock has no other historical memberships.
    candidates = [b for b in bundles if b.get("endpoint") == endpoint and _covers(b, code=code)
                  and not any((b.get("parameters") or {}).get(k) for k in ("l1_code", "l2_code", "l3_code"))]
    dictionaries = {}
    dictionary_bundles = []
    for b in bundles:
        if b.get("endpoint") != "index_classify" or not b.get("complete"):
            continue
        if (b.get("parameters") or {}).get("src") == scheme:
            dictionary_bundles.append(b)
        for r in b.get("rows", []):
            requested_src = (b.get("parameters") or {}).get("src")
            # Some source contracts identify only the SW family in the output;
            # the explicit version request is still required and archived.
            src = requested_src if requested_src in ("SW2014", "SW2021") and r.get("src") in (None, "", "SW", requested_src) else r.get("src")
            dict_level = r.get("level") or (b.get("parameters") or {}).get("level")
            if src == scheme and str(dict_level or "").upper() in (str(level), f"L{level}"):
                dictionaries[r.get("index_code")] = r
    rows = []
    for b in candidates:
        for n, r in enumerate(b.get("rows", []), 1):
            if r.get("ts_code") != code:
                continue
            industry_code, industry_name = r.get(f"l{level}_code"), r.get(f"l{level}_name")
            # The membership endpoint has no SW version argument.  If both
            # versions are requested, a matching dictionary is the evidence.
            version_proven = endpoint == "ci_index_member" or industry_code in dictionaries
            first, last = _date(r.get("in_date")), _date(r.get("out_date"))
            valid = bool(industry_code and industry_name and first and version_proven)
            invalid_end = bool(r.get("out_date") and last is None)
            if last is not None and first is not None and last <= first:
                valid = False
            if invalid_end:
                valid = False
            status = "mapped" if valid else "unknown"
            row = {"security_code": code, "scheme_code": scheme, "classification_version": scheme,
                   "level": level, "industry_code": industry_code,
                   "industry_name": industry_name,
                   "valid_from": first.isoformat() if first else None,
                   "valid_to": last.isoformat() if last else None,
                   "boundary_basis": "source_in_date_out_date_inclusivity_unproven" if last else "source_in_date_open_end" if str(r.get("is_new", "")).upper() == "Y" else "historical_end_unknown",
                   "membership_status": status, "raw_record": dumps(r),
                   **_source(b, r, n, first.isoformat() if first else None)}
            if status == "unknown":
                row["data_quality_status"] = "unknown"
            rows.append(row)
    rows = _distinct(rows)
    complete = _history_complete(candidates) and all(b.get("complete") for b in candidates)
    reasons = [] if complete else ["history_query_incomplete" if candidates else "not_requested"]
    if not rows:
        reasons.append("source_no_record")
    return _scope("industry", {"security_code": code, "scheme_code": scheme, "level": level},
                  rows, complete, reasons, {"queries": _proof(candidates), "whole_history": True,
                                            "dictionary_queries": _proof(dictionary_bundles),
                                            "classification_version_basis": "source_endpoint" if endpoint == "ci_index_member" else "versioned_dictionary_code_match",
                                            "source_interval_end_inclusivity": "unknown"})


def _lifecycle(code, bundles):
    candidates = [b for b in bundles if b.get("endpoint") == "stock_basic" and _covers(b, code=code)]
    rows = [r for b in candidates if b.get("complete") for r in b.get("rows", []) if r.get("ts_code") == code]
    variants = {(str(r.get("list_date") or ""), str(r.get("delist_date") or ""), str(r.get("list_status") or "")) for r in rows}
    if len(variants) != 1:
        return {"complete": False, "reason": "lifecycle_conflict" if variants else "lifecycle_unknown", "queries": _proof(candidates)}
    row = rows[0]
    listed, delisted = _date(row.get("list_date")), _date(row.get("delist_date"))
    valid = listed is not None and (not row.get("delist_date") or delisted is not None)
    if str(row.get("list_status")) == "D" and delisted is None:
        valid = False
    return {"complete": valid, "list_date": listed.isoformat() if listed else None,
            "delist_date": delisted.isoformat() if delisted else None,
            "reason": None if valid else "lifecycle_unknown", "queries": _proof(candidates)}


def _parse_intervals(day, timing):
    # Only complete clock ranges are accepted.  Open ranges and a lone R event
    # cannot establish a resumption instant.  Do not invent session boundaries.
    text = str(timing).strip()
    pieces = re.split(r"\s*[,;，；、]\s*", text)
    parsed = []
    for piece in pieces:
        match = re.fullmatch(r"(\d{1,2}:\d{2}(?::\d{2})?)\s*[-~～至]\s*(\d{1,2}:\d{2}(?::\d{2})?)", piece)
        if not match:
            return [], "unparseable_timing"
        stamps = []
        for value in match.groups():
            try:
                clock = time.fromisoformat(value if len(value.split(":")[0]) == 2 else "0"+value)
                stamps.append(datetime.combine(day, clock, SHANGHAI).astimezone(UTC))
            except ValueError:
                return [], "invalid_clock"
        if stamps[1] <= stamps[0]:
            return [], "non_increasing_interval"
        parsed.append((stamps[0].isoformat(), stamps[1].isoformat(), "second" if piece.count(":") > 2 else "minute"))
    return parsed, None


def _suspension_scope(code, day, bundles):
    candidates = [b for b in bundles if b.get("endpoint") == "suspend_d" and _covers(b, code=code, day=day)]
    complete_bundles = [b for b in candidates if b.get("complete")]
    filters = {str((b.get("parameters") or {}).get("suspend_type") or "").upper() for b in complete_bundles}
    query_complete = bool(complete_bundles) and ("" in filters or {"S", "R"} <= filters) and all(b.get("complete") for b in candidates)
    exchange = "BSE" if code.endswith(".BJ") else "SSE" if code.endswith(".SH") else "SZSE" if code.endswith(".SZ") else None
    calendar = _calendar_scope(exchange, day, bundles) if exchange else None
    lifecycle = _lifecycle(code, bundles)
    rows = []
    for b in candidates:
        for n, r in enumerate(b.get("rows", []), 1):
            if r.get("ts_code") != code or _date(r.get("trade_date")) != day:
                continue
            event_type, raw = str(r.get("suspend_type") or "").upper(), r.get("suspend_timing")
            base = {"security_code": code, "trade_date": day.isoformat(), "suspend_type": event_type,
                    "raw_timing": raw, "start_at": None, "end_at": None, "is_full_day": 0,
                    "time_precision": "unknown", "parse_status": "unknown", "parse_reason": None,
                    "raw_record": dumps(r), **_source(b, r, n, day.isoformat())}
            if event_type == "S" and (raw is None or str(raw).strip() == ""):
                first = datetime.combine(day, time.min, SHANGHAI).astimezone(UTC)
                rows.append({**base, "start_at": first.isoformat(), "end_at": (first+timedelta(days=1)).isoformat(),
                             "is_full_day": 1, "time_precision": "day", "parse_status": "parsed",
                             "parse_reason": "source_full_day_suspension"})
            elif event_type == "S":
                intervals, error = _parse_intervals(day, raw)
                if error:
                    rows.append({**base, "parse_reason": error, "data_quality_status": "unknown"})
                else:
                    for first, last, precision in intervals:
                        rows.append({**base, "start_at": first, "end_at": last, "time_precision": precision,
                                     "parse_status": "parsed", "parse_reason": "source_range_normalized_left_closed_right_open"})
            else:
                rows.append({**base, "parse_reason": "resumption_time_unknown" if event_type == "R" else "unknown_event_type",
                             "data_quality_status": "unknown"})
    # Distinguish distinct intervals split from a single raw event.
    unique = {digest({k: r[k] for k in ("source_record_hash", "start_at", "end_at")}): r for r in rows}
    rows = list(unique.values())
    reasons = [] if query_complete else ["query_incomplete" if candidates else "not_requested"]
    if not rows and query_complete:
        reasons.append("source_no_record")
    evidence = {"queries": _proof(candidates), "event_query_complete": query_complete,
                "calendar": {"complete": calendar["complete"], "scope_id": calendar["scope_id"],
                             "rows": calendar["rows"], "evidence": calendar["evidence"]} if calendar else None,
                "lifecycle": lifecycle, "timezone": "Asia/Shanghai", "interval_convention": "left_closed_right_open",
                "source_intraday_endpoint_inclusivity": "unknown",
                "dependency_basis": "self_contained_archived_source_queries"}
    return _scope("suspension", {"security_code": code, "trade_date": day.isoformat()}, rows,
                  query_complete, reasons, evidence)


def build_scopes(request: dict, bundles: list[dict]) -> list[dict]:
    """Build deterministic request scopes, including missing and empty responses."""
    days = _days(request)
    modules = request.get("modules") or list(MODULES)
    if any(m not in MODULES for m in modules):
        raise ValueError("不支持的业务模块")
    codes = list(dict.fromkeys(request.get("security_codes") or []))
    if any(not isinstance(code, str) or not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", code) for code in codes):
        raise ValueError("股票代码必须包含六位代码及 SH/SZ/BJ 交易所后缀")
    exchanges = list(dict.fromkeys(request.get("exchanges") or ["SSE", "SZSE", "BSE"]))
    if any(e not in ("SSE", "SZSE", "BSE") for e in exchanges):
        raise ValueError("仅支持 SSE/SZSE/BSE 交易日历")
    schemes = list(dict.fromkeys(request.get("schemes") or ["SW2021", "CITICS"]))
    if any(s not in ("SW2014", "SW2021", "CITICS", "CI", "CITIC") for s in schemes):
        raise ValueError("不支持的行业分类体系")
    levels = list(dict.fromkeys(request.get("levels") or [1, 2, 3]))
    if any(type(level) is not int or level not in (1, 2, 3) for level in levels):
        raise ValueError("行业层级必须是 1、2 或 3")
    scopes = []
    if "calendar" in modules:
        scopes.extend(_calendar_scope(e, day, bundles) for e in exchanges for day in days)
    if "industry" in modules:
        scopes.extend(_industry_scope(code, scheme, level, bundles) for code in codes for scheme in schemes for level in levels)
    if "suspension" in modules:
        scopes.extend(_suspension_scope(code, day, bundles) for code in codes for day in days)
    return scopes


def _trusted_heads(snapshot):
    revisions = {r["revision_id"]: r for r in snapshot.get("revisions", [])}
    events = snapshot.get("events", [])
    revoked = {e["revision_id"] for e in events if e.get("event_type") in ("revoke", "revoked")}
    published = {e["revision_id"]: e for e in sorted(events, key=lambda e: int(e.get("event_seq", 0))) if e.get("event_type") in ("publish", "published")}
    invalid = set(snapshot.get("invalid_revision_ids", []))
    invalid.update(v["revision_id"] for v in snapshot.get("verification", []) if v.get("verified") is False)
    published = {k: v for k, v in published.items() if k not in revoked and k not in invalid and k in revisions}
    by_scope = defaultdict(list)
    rows = defaultdict(list)
    for row in snapshot.get("rows", []):
        rows[(row.get("revision_id"), row.get("scope_id"))].append(row)
    for scope in snapshot.get("scopes", []):
        revision_id = scope.get("revision_id")
        if revision_id not in published:
            continue
        event = published[revision_id]
        event_scopes = event.get("scope_ids")
        if event_scopes is not None and scope.get("scope_id") not in event_scopes:
            continue
        revision = {**revisions[revision_id], "published_at": event.get("published_at") or event.get("created_at")}
        rank = (int(revision.get("ingestion_seq", scope.get("ingestion_seq", 0))), int(event.get("event_seq", 0)), revision_id)
        by_scope[scope["scope_id"]].append((rank, scope, revision, rows.get((revision_id, scope["scope_id"]), scope.get("rows", []))))
    return {key: max(values, key=lambda item: item[0]) for key, values in by_scope.items()}, published, revoked


def _result(module, selector, head=None, *, status="unknown", reason="not_requested", data=None, **extra):
    result = {"module": module, **selector, "status": status, "reason": reason, "data": data, **extra}
    if head:
        _, scope, revision, _ = head
        for field in ("revision_id", "import_task_id", "attempt_id", "ingestion_seq", "ingested_at", "published_at", "source_name", "source_id"):
            if revision.get(field) is not None:
                result[field] = revision[field]
            elif scope.get(field) is not None:
                result[field] = scope[field]
        result["scope_id"] = scope["scope_id"]
        result["evidence"] = scope.get("evidence", {})
    return result


def _dependency_reason(scope, heads, published, revoked):
    for dep in scope.get("dependencies", []):
        if isinstance(dep, str):
            if dep in revoked:
                return "dependency_revoked"
            if dep not in published:
                return "dependency_unpublished"
            continue
        revision = dep.get("revision_id")
        if revision in revoked:
            return "dependency_revoked"
        if revision and revision not in published:
            return "dependency_unpublished"
        if dep.get("scope_id"):
            latest = heads.get(dep["scope_id"])
            if not latest:
                return "dependency_unpublished"
            if revision and latest[1].get("revision_id") != revision:
                return "dependency_changed"
    return None


def _unpublished_reason(sid, snapshot):
    scopes = [s for s in snapshot.get("scopes", []) if s.get("scope_id") == sid]
    if not scopes:
        return "not_requested"
    revs = {r["revision_id"]: r for r in snapshot.get("revisions", [])}
    revoked = {e.get("revision_id") for e in snapshot.get("events", []) if e.get("event_type") in ("revoke", "revoked")}
    latest = max(scopes, key=lambda s: int(revs.get(s.get("revision_id"), {}).get("ingestion_seq", s.get("ingestion_seq", 0))))
    invalid = set(snapshot.get("invalid_revision_ids", []))
    invalid.update(v["revision_id"] for v in snapshot.get("verification", []) if v.get("verified") is False)
    if latest.get("revision_id") in invalid:
        return "target_verification_failed"
    if latest.get("revision_id") in revoked:
        return "revoked"
    state = revs.get(latest.get("revision_id"), {}).get("status", "unpublished")
    return state if state in ("failed", "pending_confirmation", "interrupted") else "unpublished"


def _calendar_answer(selector, head):
    _, scope, _, rows = head
    if not scope.get("complete"):
        return _result("calendar", selector, head, reason=(scope.get("reasons") or ["query_incomplete"])[0], data=rows)
    meanings = {(r.get("is_open"), r.get("previous_trade_date"), r.get("reference_exchange"), r.get("calendar_basis")) for r in rows}
    if len(meanings) > 1:
        return _result("calendar", selector, head, status="conflict", reason="conflicting_calendar_records", data=rows)
    if not rows or rows[0].get("is_open") not in (0, 1):
        return _result("calendar", selector, head, reason="source_no_record" if not rows else "invalid_is_open", data=rows)
    if rows[0].get("data_quality_status") == "unknown":
        return _result("calendar", selector, head, reason="calendar_field_meaning_unknown", data=rows)
    return _result("calendar", selector, head, status="complete", reason="open" if rows[0]["is_open"] else "closed", data=rows[0])


def _industry_answer(selector, day, head):
    _, scope, _, rows = head
    sel = {**selector, "trade_date": day.isoformat()}
    if not scope.get("complete"):
        return _result("industry", sel, head, reason="history_query_incomplete", data=rows)
    if not rows:
        return _result("industry", sel, head, reason="source_no_record", data=[])
    active, uncertain = [], []
    for row in rows:
        first, last = _date(row.get("valid_from")), _date(row.get("valid_to"))
        if last is not None and day > last:
            continue
        if first is not None and day < first:
            continue
        if last == day and "unproven" in str(row.get("boundary_basis", "")):
            uncertain.append(row)
            continue
        if last == day:
            continue  # internal representation is always [first,last)
        if first is None or row.get("membership_status") != "mapped" or row.get("boundary_basis") == "historical_end_unknown":
            uncertain.append(row)
        else:
            active.append(row)
    active = list({digest({k: r.get(k) for k in ("industry_code", "industry_name", "valid_from", "valid_to")}): r for r in active}.values())
    if len(active) > 1:
        return _result("industry", sel, head, status="conflict", reason="overlapping_memberships", data=active+uncertain)
    if uncertain:
        return _result("industry", sel, head, reason="industry_boundary_or_meaning_unknown", data=active+uncertain)
    if not active:
        return _result("industry", sel, head, reason="industry_history_gap", data=[])
    return _result("industry", sel, head, status="complete", reason="membership_known", data=active[0])


def _check_instant(value, day):
    if value is None:
        return None
    text = str(value)
    try:
        if "T" in text or " " in text:
            instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if instant.tzinfo is None:
                instant = instant.replace(tzinfo=SHANGHAI)
            if instant.astimezone(SHANGHAI).date() != day:
                return False
        else:
            instant = datetime.combine(day, time.fromisoformat(text), SHANGHAI)
        return instant.astimezone(UTC)
    except ValueError as exc:
        raise ValueError("日内检查时刻必须为有效时间或 ISO 时间戳") from exc


def _suspension_answer(selector, day, instant, head):
    _, scope, _, rows = head
    sel = {**selector, "check_at": instant.isoformat() if instant else None}
    if not scope.get("complete") or not scope.get("evidence", {}).get("event_query_complete"):
        return _result("suspension", sel, head, reason="query_incomplete", data=rows)
    evidence = scope.get("evidence", {})
    calendar, lifecycle = evidence.get("calendar") or {}, evidence.get("lifecycle") or {}
    cal_rows = calendar.get("rows", [])
    cal_values = {r.get("is_open") for r in cal_rows}
    calendar_known = bool(calendar.get("complete")) and len(cal_values) == 1 and next(iter(cal_values), None) in (0, 1)
    if calendar_known and cal_values == {0}:
        return _result("suspension", sel, head, status="conflict" if rows else "complete",
                       reason="event_on_closed_day" if rows else "not_open", data=rows)
    if lifecycle.get("complete"):
        listed, delisted = _date(lifecycle.get("list_date")), _date(lifecycle.get("delist_date"))
        # Listing dates are effective dates; exact delisting boundary is not
        # guessed to mean a tradable final day.
        if (listed and day < listed) or (delisted and day > delisted):
            return _result("suspension", sel, head, status="conflict" if rows else "complete",
                           reason="event_outside_listing" if rows else "not_listed", data=rows)
        if delisted == day:
            return _result("suspension", sel, head, reason="delisting_boundary_unknown", data=rows)
    if not rows:
        if not calendar_known:
            return _result("suspension", sel, head, reason="calendar_unknown", data=[])
        if not lifecycle.get("complete"):
            return _result("suspension", sel, head, reason=lifecycle.get("reason") or "lifecycle_unknown", data=[])
        return _result("suspension", sel, head, status="complete", reason="no_reported_suspension", data=[],
                       is_suspended=False, negative_evidence=True)
    unknown = [r for r in rows if r.get("parse_status") != "parsed" or r.get("suspend_type") != "S"]
    if unknown:
        return _result("suspension", sel, head, reason="suspension_time_unknown", data=rows)
    intervals = []
    for row in rows:
        try:
            first = datetime.fromisoformat(str(row["start_at"]).replace("Z", "+00:00"))
            last = datetime.fromisoformat(str(row["end_at"]).replace("Z", "+00:00"))
            if first.tzinfo is None or last.tzinfo is None or first >= last:
                raise ValueError("invalid interval")
            intervals.append((first, last, row))
        except (TypeError, KeyError, ValueError):
            return _result("suspension", sel, head, reason="suspension_time_unknown", data=rows)
    intervals.sort(key=lambda v: v[0])
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1] and (current[0], current[1]) != (previous[0], previous[1]):
            return _result("suspension", sel, head, status="conflict", reason="overlapping_suspension_intervals", data=rows)
    if instant is None:
        whole = any(r.get("is_full_day") == 1 for r in rows)
        return _result("suspension", sel, head, status="complete", reason="full_day_suspension" if whole else "intraday_suspension_intervals",
                       data=rows, is_suspended=True if whole else None)
    # Source docs do not specify endpoint inclusivity.  Keep normalized [a,b)
    # storage, while refusing an unjustified claim at an exact intraday edge.
    if any(not row.get("is_full_day") and instant in (first, last) for first, last, row in intervals):
        return _result("suspension", sel, head, reason="source_time_boundary_unknown", data=rows)
    inside = any(first <= instant < last for first, last, _ in intervals)
    if not inside and (not calendar_known or not lifecycle.get("complete")):
        return _result("suspension", sel, head, reason="calendar_unknown" if not calendar_known else "lifecycle_unknown", data=rows)
    return _result("suspension", sel, head, status="complete", reason="suspended" if inside else "outside_reported_intervals",
                   data=rows, is_suspended=inside)


def check_integrity(request: dict, snapshot: dict) -> dict:
    """Read only published scope heads, then evaluate business meaning.

    A newer unpublished row never hides an older trusted snapshot.  Revoke
    events remove publications from consideration without deleting history.
    """
    days = _days(request)
    selectors = build_scopes(request, [])
    heads, published, revoked = _trusted_heads(snapshot)
    results = []
    for wanted in selectors:
        module, selector, sid = wanted["module"], wanted["selector"], wanted["scope_id"]
        head = heads.get(sid)
        reason = _dependency_reason(head[1], heads, published, revoked) if head else _unpublished_reason(sid, snapshot)
        if not reason and module == "suspension":
            calendar = head[1].get("evidence", {}).get("calendar") or {}
            calendar_head = heads.get(calendar.get("scope_id"))
            if calendar_head:
                # A copied source proof does not depend on another module's
                # deployment, but a subsequently trusted contradictory calendar
                # cannot silently leave a derived "no event" interpretation.
                fields = ("exchange", "trade_date", "is_open", "previous_trade_date", "reference_exchange", "calendar_basis")
                semantic = lambda values: {digest({field: row.get(field) for field in fields}) for row in values}
                if not calendar_head[1].get("complete") or semantic(calendar_head[3]) != semantic(calendar.get("rows", [])):
                    reason = "dependency_changed"
        effective_days = days if module == "industry" else [_date(selector["trade_date"])]
        for day in effective_days:
            checks = request.get("check_times") or [None] if module == "suspension" else [None]
            for check in checks:
                instant = _check_instant(check, day) if module == "suspension" else None
                if instant is False:
                    continue
                item_selector = {**selector, "trade_date": day.isoformat()} if module == "industry" else selector
                if reason:
                    results.append(_result(module, item_selector, head, reason=reason,
                                           **({"check_at": instant.isoformat() if instant else None} if module == "suspension" else {})))
                elif module == "calendar":
                    results.append(_calendar_answer(selector, head))
                elif module == "industry":
                    results.append(_industry_answer(selector, day, head))
                else:
                    results.append(_suspension_answer(selector, day, instant, head))
                results[-1].setdefault("scope_id", sid)
    invalid = set(snapshot.get("invalid_revision_ids", []))
    invalid.update(v["revision_id"] for v in snapshot.get("verification", []) if v.get("verified") is False)
    affected = defaultdict(set)
    for scope in snapshot.get("scopes", []):
        if scope.get("revision_id") in invalid:
            affected[scope["scope_id"]].add(scope["revision_id"])
    for result in results:
        # Trustworthy older data can still be delivered, with an explicit
        # warning that a later attempted published revision failed readback.
        sid = result.get("scope_id")
        if sid in affected:
            result["warnings"] = [{"reason": "target_verification_failed", "revision_id": revision}
                                  for revision in sorted(affected[sid])]
    counts = Counter(r["status"] for r in results)
    response = {"rows": results, "complete": bool(results) and counts["complete"] == len(results),
                "summary": {"total": len(results), "complete": counts["complete"], "unknown": counts["unknown"],
                            "conflict": counts["conflict"], "reasons": dict(Counter(r["reason"] for r in results))}}
    if request.get("audit"):
        wanted_ids = {s["scope_id"] for s in selectors}
        response["versions"] = [{**s, "trusted": s.get("revision_id") in published,
                                  "revoked": s.get("revision_id") in revoked,
                                  "is_current": bool(heads.get(s["scope_id"]) and heads[s["scope_id"]][1].get("revision_id") == s.get("revision_id"))}
                                 for s in snapshot.get("scopes", []) if s.get("scope_id") in wanted_ids]
    return response
