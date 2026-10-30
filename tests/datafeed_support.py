"""Deterministic business fixtures shared by unit and real-backend tests."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import date
import json

from betalens.datafeed.contracts import digest, timestamp
from betalens.datafeed.reference import _normal

BASIS = "source_version_publication_time"
MARKET_TYPES = {"security_code": "text", "asset_type": "text", "trade_date": "date",
    "event_date": "date", "open": "numeric", "close": "numeric", "high": "numeric",
    "low": "numeric", "previous_close": "numeric", "volume": "numeric", "amount": "numeric",
    "turnover_rate": "numeric", "available_at": "timestamp", "availability_basis": "text", "data_quality_status": "text"}
FIN_TYPES = {"security_code": "text", "event_date": "date", "period_end": "date",
    "statement_type": "text", "report_type": "text", "version_id": "text", "revenue": "numeric",
    "available_at": "timestamp", "availability_basis": "text", "data_quality_status": "text"}
NAME_TYPES = {"security_code": "text", "name": "text", "start_date": "date", "end_date": "date",
    "available_at": "timestamp", "availability_basis": "text", "data_quality_status": "text"}
REF_TYPES = {"revision_id": "text", "scope_id": "text", "record_id": "text",
    "ingestion_seq": "integer", "ingested_at": "timestamp", "available_at": "timestamp",
    "availability_basis": "text", "data_quality_status": "text"}
REF_BUSINESS = {
    "calendar": {"exchange": "text", "trade_date": "date", "is_open": "integer"},
    "industry": {"security_code": "text", "scheme_code": "text", "level": "integer",
                 "industry_code": "text", "industry_name": "text", "valid_from": "date", "valid_to": "date", "boundary_basis": "text"},
    "suspension": {"security_code": "text", "trade_date": "date", "suspend_type": "text",
                   "parse_status": "text", "start_at": "timestamp", "end_at": "timestamp"},
}
CONTROL_TYPES = {
    "reference_revision": {"revision_id": "text", "module": "text", "ingestion_seq": "integer", "manifest_hash": "text", "payload": "text"},
    "reference_scope": {"revision_id": "text", "scope_id": "text", "module": "text", "complete": "integer",
                        "row_count": "integer", "content_hash": "text", "key_hash": "text", "selector_hash": "text", "payload": "text"},
    "reference_publication_event": {"event_id": "text", "event_seq": "integer", "revision_id": "text",
                                    "event_type": "text", "created_at": "timestamp", "payload": "text"},
}


def quote(code="A", at="2024-01-02 15:00", value=10, **extra):
    day = timestamp(at).tz_convert("Asia/Shanghai").date()
    row = {k: None for k in MARKET_TYPES}
    row.update(security_code=code, asset_type="stock", trade_date=day, event_date=day,
               open=value, high=value, low=value, close=value, previous_close=value,
               volume=1000, amount=10000, turnover_rate=1,
               available_at=timestamp(at).to_pydatetime(), availability_basis=BASIS, data_quality_status="accepted")
    row.update(extra)
    return row


class MemoryReader:
    backend = "postgres"
    namespace = "test"

    def __init__(self, tables=None, types=None):
        self.tables = deepcopy(tables or {})
        self.types = deepcopy(types or {})
        self.calls = []

    def columns(self, table):
        return self.types.get(table, {})

    @contextmanager
    def snapshot(self):
        yield self

    def rows(self, table, columns=None, filters=(), order=(), limit=None):
        self.calls.append(table)
        rows = deepcopy(self.tables.get(table, []))
        for field, op, value in filters:
            rows = [r for r in rows if matches(r.get(field), op, value)]
        for field, direction in reversed(order):
            rows.sort(key=lambda r: r[field], reverse=direction == "DESC")
        return [{k: r.get(k) for k in columns or self.types[table]} for r in rows[:limit]]


def matches(left, op, right):
    if op == "IN":
        return left in right
    if op == "IS NOT NULL":
        return left is not None
    if left is None:
        return False
    if op == "=": return left == right
    if op == "!=": return left != right
    if op == ">=": return left >= right
    if op == "<=": return left <= right
    if op == ">": return left > right
    if op == "<": return left < right
    raise AssertionError(op)


def market_reader(rows=None):
    rows = rows if rows is not None else [quote(), quote(at="2024-01-03 15:00", value=20)]
    return MemoryReader({"market_daily_post_adjusted": rows, "market_daily_raw": [dict(r, close=r["close"] / 2) for r in rows]},
                        {"market_daily_post_adjusted": MARKET_TYPES, "market_daily_raw": MARKET_TYPES})


def reference_reader(module, revisions):
    from betalens.datafeed.reference import TABLES
    types = {**CONTROL_TYPES, TABLES[module]: {**REF_TYPES, **REF_BUSINESS[module]}}
    tables = {table: [] for table in types}
    reader = MemoryReader(tables, types)
    for seq, (rid, scopes, event_type) in enumerate(revisions, 1):
        append_revision(reader, module, rid, seq, scopes, event_type)
    return reader


def append_revision(reader, module, rid, seq, scopes, event_type="publish"):
    from betalens.datafeed.reference import TABLES
    table = TABLES[module]
    docs = []
    for source in scopes:
        scope = deepcopy(source)
        rows = []
        for i, original in enumerate(scope.pop("rows")):
            record = {k: None for k in reader.types[table]}
            record.update(original, revision_id=rid, scope_id=scope["scope_id"], record_id=str(i), ingestion_seq=seq,
                          ingested_at=timestamp("2026-01-01").to_pydatetime())
            record.setdefault("data_quality_status", "accepted")
            rows.append(_normal(record, reader.types[table]))
        reader.tables[table].extend(rows)
        scope.update(revision_id=rid, module=module, row_count=len(rows),
                     content_hash=digest(rows), key_hash=digest(sorted(r["record_id"] for r in rows)),
                     selector_hash=digest(scope["selector"]))
        docs.append(scope)
    docs.sort(key=lambda s: s["scope_id"])
    rev = {"revision_id": rid, "module": module, "ingestion_seq": seq,
           "scope_ids": [s["scope_id"] for s in docs], "row_count": sum(s["row_count"] for s in docs)}
    rev["manifest_hash"] = digest({"revision": rev, "scopes": docs})
    reader.tables["reference_revision"].append({k: rev[k] for k in reader.types["reference_revision"] if k != "payload"} | {"payload": json.dumps(rev)})
    for scope in docs:
        reader.tables["reference_scope"].append({k: scope[k] for k in reader.types["reference_scope"] if k != "payload"} | {"payload": json.dumps(scope)})
    if event_type is None:
        return
    for kind in (["publish", "revoke"] if event_type == "revoke" else [event_type]):
        event = {"event_id": rid + kind, "event_seq": seq, "revision_id": rid, "event_type": kind,
                 "created_at": timestamp("2026-01-01").isoformat(timespec="microseconds"),
                 "manifest_hash": rev["manifest_hash"], "scope_ids": rev["scope_ids"]}
        reader.tables["reference_publication_event"].append({k: event[k] for k in reader.types["reference_publication_event"] if k != "payload"} | {"payload": json.dumps(event)})


def evidence(at="2024-01-01", **extra):
    return {"available_at": timestamp(at).isoformat(timespec="microseconds"),
            "availability_basis": BASIS, "data_quality_status": "accepted", **extra}


def calendar_scope(day="2024-01-02", opened=1, available="2024-01-01"):
    return {"scope_id": "SSE:" + day, "selector": {"exchange": "SSE", "trade_date": day},
            "complete": True, "evidence": {"availability": evidence(available)},
            "rows": [{"exchange": "SSE", "trade_date": day, "is_open": opened, **evidence(available)}]}
