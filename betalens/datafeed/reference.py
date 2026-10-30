"""Read target-local publication manifests without importing the manager.

The hash contract matches reference-domains-v1. Publication and historical
availability are separate gates. Complete scope availability is required even
for zero-row snapshots, so a deletion cannot reveal a partial older snapshot.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .contracts import (DatafeedError, PITUnavailableError, VersionConflictError,
                        digest, document, proven, timestamp, local_time)
from .readers import require_columns

TABLES = {"industry": "stock_industry_membership_versions",
          "calendar": "trade_calendar_versions",
          "suspension": "stock_suspension_intervals_versions"}
BUSINESS_COLUMNS = {
    "calendar": {"exchange": "text", "trade_date": "date", "is_open": "integer"},
    "industry": {"security_code": "text", "scheme_code": "text", "level": "integer",
                 "industry_code": "text", "industry_name": "text", "valid_from": "date",
                 "valid_to": "date", "boundary_basis": "text"},
    "suspension": {"security_code": "text", "trade_date": "date", "suspend_type": "text",
                   "parse_status": "text", "start_at": "timestamp", "end_at": "timestamp"},
}


def _normal(row, types):
    result = dict(row)
    for key, kind in types.items():
        value = result.get(key)
        if value is None:
            continue
        if kind == "timestamp":
            result[key] = timestamp(value, stored=True).isoformat(timespec="microseconds")
        elif kind == "date":
            result[key] = str(value)[:10]
        elif kind == "integer":
            result[key] = int(value)
    return result


def _documents(reader, table, filters=()):
    require_columns(reader, table, {"payload": "text"})
    types = reader.columns(table)
    result = []
    for raw in reader.rows(table, filters=filters):
        row = _normal(raw, types)
        payload = document(row["payload"])
        for field in types:
            if field != "payload" and row[field] != payload.get(field):
                raise VersionConflictError(f"发布元数据与载荷不一致: {table}.{field}")
        result.append(payload)
    return result


def load_reference(reader, module, *, codes=None, exchange=None):
    """Capture events first, verify immutable manifests, then return scopes.

    Unpublished/revoked rows never become candidates. A malformed published
    manifest raises rather than hiding a broken integrity contract.
    """
    table = TABLES[module]
    require_columns(reader, table, BUSINESS_COLUMNS[module])
    require_columns(reader, table, {"revision_id": "text", "scope_id": "text",
                                    "record_id": "text", "ingestion_seq": "integer",
                                    "available_at": "timestamp", "availability_basis": "text",
                                    "data_quality_status": "text"})
    events = _documents(reader, "reference_publication_event")
    revoked = {e["revision_id"] for e in events if e["event_type"] == "revoke"}
    published = {e["revision_id"]: e for e in events if e["event_type"] == "publish" and e["revision_id"] not in revoked}
    revisions = _documents(reader, "reference_revision", [("module", "=", module)])
    scopes = _documents(reader, "reference_scope", [("module", "=", module)])
    wanted = [s for s in scopes if s["revision_id"] in published
              and (codes is None or s["selector"].get("security_code") in codes)
              and (exchange is None or s["selector"].get("exchange") == exchange)]
    revision_ids = {s["revision_id"] for s in wanted}
    if not revision_ids:
        return []
    raw_rows = reader.rows(table, filters=[("revision_id", "IN", sorted(revision_ids))])
    normalized = [_normal(r, reader.columns(table)) for r in raw_rows]
    grouped = defaultdict(list)
    for row in normalized:
        grouped[(row["revision_id"], row["scope_id"])].append(row)
    heads = {}
    for revision in revisions:
        rid = revision["revision_id"]
        if rid not in revision_ids:
            continue
        group = sorted((s for s in scopes if s["revision_id"] == rid), key=lambda s: s["scope_id"])
        manifest = {k: v for k, v in revision.items() if k != "manifest_hash"}
        event = published[rid]
        if (event.get("manifest_hash") != revision["manifest_hash"]
                or event.get("scope_ids") != revision["scope_ids"]
                or [s["scope_id"] for s in group] != revision["scope_ids"]
                or digest({"revision": manifest, "scopes": group}) != revision["manifest_hash"]):
            raise VersionConflictError("可信发布清单核验失败: " + rid)
        total = 0
        for scope in group:
            records = sorted(grouped[(rid, scope["scope_id"])], key=lambda r: (r["scope_id"], r["record_id"]))
            total += len(records)
            if (len(records) != scope["row_count"] or len({r["record_id"] for r in records}) != len(records)
                    or digest(records) != scope["content_hash"]
                    or digest(sorted(r["record_id"] for r in records)) != scope["key_hash"]
                    or digest(scope["selector"]) != scope["selector_hash"]):
                raise VersionConflictError("可信范围内容核验失败: " + scope["scope_id"])
        if total != revision["row_count"] or total != sum(r["revision_id"] == rid for r in normalized):
            raise VersionConflictError("可信修订记录数不匹配: " + rid)
        heads[rid] = revision
    if revision_ids - heads.keys():
        raise VersionConflictError("可信发布缺少修订头")
    return [{**s, "rows": grouped[(s["revision_id"], s["scope_id"])],
             "ingestion_seq": heads[s["revision_id"]]["ingestion_seq"]} for s in wanted]


def known_scopes(scopes, as_of):
    result = {}
    for scope in scopes:
        availability = scope.get("evidence", {}).get("availability", {})
        if not scope.get("complete") or not proven(availability, as_of):
            continue
        if not all(proven(row, as_of) for row in scope["rows"]):
            continue
        old = result.get(scope["scope_id"])
        if old and old["ingestion_seq"] == scope["ingestion_seq"] and old["revision_id"] != scope["revision_id"]:
            raise VersionConflictError("同一范围的版本顺序冲突")
        if old is None or scope["ingestion_seq"] > old["ingestion_seq"]:
            result[scope["scope_id"]] = scope
    return list(result.values())


def calendar_rows(scopes, as_of):
    days = {}
    for scope in known_scopes(scopes, as_of):
        if len(scope["rows"]) != 1:
            raise PITUnavailableError("日历完整范围必须有一条自然日记录")
        row = scope["rows"][0]
        day = row["trade_date"]
        if row.get("is_open") not in (0, 1):
            raise PITUnavailableError("日历开市状态未知: " + day)
        if day in days and days[day] != row:
            raise VersionConflictError("日历范围冲突: " + day)
        days[day] = row
    return days


def calendar_days(reader, begin, end, exchange="SSE", as_of=None):
    exchange = exchange.upper()
    begin, end = local_time(timestamp(begin)).normalize(), local_time(timestamp(end)).normalize()
    if begin > end:
        raise ValueError("开始日期不能晚于结束日期")
    scopes = load_reference(reader, "calendar", exchange=exchange)
    cutoff = timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    rows = calendar_rows(scopes, cutoff)
    requested = pd.date_range(begin, end, freq="D")
    missing = [day.date().isoformat() for day in requested if day.date().isoformat() not in rows]
    if missing:
        raise PITUnavailableError(f"{exchange} 日历缺少历史可知的完整范围: {missing[:3]}")
    return [day.date() for day in requested if rows[day.date().isoformat()]["is_open"] == 1]


def suspension_status(scope, at):
    """0/1/-1/None: subscription status, not a promise of all trade rules."""
    evidence = scope.get("evidence", {})
    rows = scope["rows"]
    lifecycle = evidence.get("lifecycle", {})
    day = local_time(at).date().isoformat()
    if lifecycle.get("complete") and proven(lifecycle, at):
        if lifecycle.get("list_date") and day < lifecycle["list_date"]:
            return (-1, "未上市") if not rows else (None, "上市范围与事件冲突")
        if lifecycle.get("delist_date") and day >= lifecycle["delist_date"]:
            return None, "退市边界或交易范围未知"
    else:
        return None, "上市范围历史可知性不足"
    calendar = evidence.get("calendar", {})
    cal_rows = calendar.get("rows", [])
    if (not calendar.get("complete") or len(cal_rows) != 1
            or not proven(cal_rows[0], at) or cal_rows[0].get("is_open") != 1):
        return None, "开市证据不足或休市"
    if not evidence.get("event_query_complete"):
        return None, "停牌查询范围不完整"
    intervals = []
    for row in rows:
        if row.get("parse_status") != "parsed" or row.get("suspend_type") != "S" or not row.get("start_at") or not row.get("end_at"):
            return None, "停复牌时段未知"
        start, end = timestamp(row["start_at"], stored=True), timestamp(row["end_at"], stored=True)
        if start >= end:
            return None, "停牌区间无效"
        intervals.append((start, end))
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        return None, "停牌区间冲突"
    if any(start <= at < end for start, end in intervals):
        return 0, "停牌"
    return 1, "已核验上市和开市范围，未报告该时点停牌"
