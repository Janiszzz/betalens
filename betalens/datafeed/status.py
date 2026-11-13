"""Read-only trading risks; unknown evidence never implies an executable order."""
from __future__ import annotations

import re
import pandas as pd

from .contracts import (STATUS_COLUMNS, PITUnavailableError, VersionConflictError,
    available, frame, local_time, timestamp, pit_policy as resolve_policy)
from .query import _codes, _name_rows, historical_name, _strict_trade_status
from .readers import get_reader, require_columns

RISK_COLUMNS = ["code", "datetime", "is_listed", "is_st", "is_suspended", "name", "remark"]


def _event_suspended(rows, at, policy):
    intervals = []
    full_day = set()
    for row in rows:
        if not available(row, at, policy):
            continue
        timing = row.get("suspend_timing")
        if not timing:
            if row.get("suspend_type") not in {"S", "R"}:
                return None, "无法识别停复牌事件"
            full_day.add(int(row["suspend_type"] == "S"))
            continue
        if row.get("suspend_type") != "S":
            return None, "停复牌时段无法明确解析"
        for period in timing.split(","):
            match = re.fullmatch(r"([0-2]?[0-9]:[0-5][0-9])-([0-2]?[0-9]:[0-5][0-9])", period)
            if not match:
                return None, "停复牌时段无法明确解析"
            day = str(local_time(at).date())
            try:
                start = timestamp(day + " " + match[1].zfill(5))
                end = timestamp(day + " " + match[2].zfill(5))
            except ValueError:
                return None, "停复牌时段无效"
            if start >= end:
                return None, "停牌区间无效"
            intervals.append((start, end))
    if len(full_day) > 1 or full_day and intervals:
        return None, "停复牌事件冲突"
    if full_day:
        return full_day.pop(), "源端整日停复牌事件"
    if intervals:
        return int(any(start <= at < end for start, end in intervals)), "源端日内停牌区间"
    return None, "缺少已可得停复牌事件"


def _risk_inputs(reader, table_name, codes, dates):
    if table_name != "trade_status_daily":
        raise NotImplementedError("研究风险查询仅支持 trade_status_daily")
    require_columns(reader, table_name, {"security_code": "text", "asset_type": "text",
        "trade_date": "date", "is_tradable": "integer", "status_source": "text",
        "available_at": "timestamp", "availability_basis": "text", "data_quality_status": "text"})
    require_columns(reader, "security_dim", {"security_code": "text", "asset_type": "text",
        "list_date": "date", "delist_date": "date"})
    securities = reader.rows("security_dim", ["security_code", "asset_type", "list_date", "delist_date"], filters=[("asset_type", "=", "stock")])
    lifecycle = {}
    for row in securities:
        code = row["security_code"]
        value = (row.get("list_date"), row.get("delist_date"))
        if code in lifecycle and lifecycle[code] != value:
            raise VersionConflictError("证券上市范围冲突: " + code)
        lifecycle[code] = value
    codes = sorted(lifecycle) if codes is None else codes
    filters = [("security_code", "IN", list(dict.fromkeys(codes))),
               ("trade_date", "IN", sorted({local_time(at).date() for at in dates}))]
    statuses = {}
    for row in reader.rows(table_name, ["security_code", "trade_date", "is_tradable", "status_source", "available_at", "availability_basis", "data_quality_status"], filters=[*filters, ("asset_type", "=", "stock")]):
        statuses.setdefault((row["security_code"], row["trade_date"]), []).append(row)
    events = {}
    if reader.columns("suspension_event"):
        require_columns(reader, "suspension_event", {"security_code": "text", "trade_date": "date",
            "suspend_type": "text", "suspend_timing": "text", "available_at": "timestamp",
            "availability_basis": "text", "data_quality_status": "text"})
        for row in reader.rows("suspension_event", ["security_code", "trade_date", "suspend_type", "suspend_timing", "available_at", "availability_basis", "data_quality_status"], filters=filters):
            events.setdefault((row["security_code"], row["trade_date"]), []).append(row)
    quotes = {}
    calendar = {}
    quote_columns = {'security_code': 'text', 'asset_type': 'text', 'trade_date': 'date',
                     'volume': 'numeric', 'available_at': 'timestamp', 'availability_basis': 'text',
                     'data_quality_status': 'text'}
    if reader.columns('market_daily_raw') and reader.columns('trade_calendar'):
        require_columns(reader, 'market_daily_raw', quote_columns)
        require_columns(reader, 'trade_calendar', {'exchange': 'text', 'trade_date': 'date', 'is_open': 'integer'})
        for row in reader.rows('market_daily_raw', list(quote_columns), filters=[*filters, ('asset_type', '=', 'stock')]):
            quotes.setdefault((row['security_code'], row['trade_date']), []).append(row)
        for row in reader.rows('trade_calendar', ['exchange', 'trade_date', 'is_open'],
                               filters=[('trade_date', 'IN', sorted({local_time(at).date() for at in dates}))]):
            calendar.setdefault((row['exchange'], row['trade_date']), set()).add(row['is_open'])
    return codes, lifecycle, statuses, events, quotes, calendar


def query_trade_risks(source, table_name="trade_status_daily", codes=None, dates=None, *, pit_policy=None):
    policy = resolve_policy(source, pit_policy)
    codes = _codes(codes)
    if dates is None or isinstance(dates, str):
        raise ValueError("dates 必须是列表")
    dates = [timestamp(d) for d in dates]
    if not dates or codes == []:
        return frame([], RISK_COLUMNS, integers=("is_listed", "is_st", "is_suspended"), times=("datetime",))
    if table_name == "stock_suspension_intervals_versions":
        status = _strict_trade_status(source, table_name, codes, dates, pit_policy=policy)
        records = []
        for row in status.to_dict("records"):
            value = row["value"]
            known = not pd.isna(value)
            records.append({"code": row["code"], "datetime": row["datetime"], "name": row["name"],
                "is_listed": 0 if known and value == -1 else 1 if known else None,
                "is_st": int(bool(re.match(r"^(?:S\*?ST|\*?ST)", row["name"]))) if row["name"] else None,
                "is_suspended": int(value == 0) if known and value != -1 else None,
                "remark": {"pit_policy": policy, "reason": row["status_text"], "st_basis": "historical_name_assumed"}})
        return frame(records, RISK_COLUMNS, integers=("is_listed", "is_st", "is_suspended"), times=("datetime",))
    if policy == "strict":
        raise PITUnavailableError("strict 风险查询须显式使用 stock_suspension_intervals_versions")
    reader = get_reader(source)
    with reader.snapshot():
        codes, lifecycle, statuses, events, quotes, calendar = _risk_inputs(reader, table_name, codes, dates)
        from .reference import load_reference, suspension_status
        from .suspension import SuspensionScopes
        reference_tables = ['stock_suspension_intervals_versions', 'reference_revision', 'reference_scope', 'reference_publication_event']
        days = [local_time(at).date().isoformat() for at in dates]
        scopes = load_reference(reader, 'suspension', codes=codes, start=min(days), end=max(days)) if all(reader.columns(t) for t in reference_tables) else []
        suspension_index = SuspensionScopes(scopes)
        names = _name_rows(reader, codes)
    records = []
    for at in dates:
        day = local_time(at).date()
        for code in codes:
            listed, delisted = lifecycle.get(code, (None, None))
            is_listed = int(listed <= day and (delisted is None or day < delisted)) if listed else None
            rows = events.get((code, day), [])
            suspended, reason = _event_suspended(rows, at, policy)
            known_events = any(available(r, at, policy) for r in rows)
            daily = [r for r in statuses.get((code, day), []) if available(r, at, policy)]
            values = {r["is_tradable"] for r in daily if r["is_tradable"] in (0, 1)}
            if len(values) > 1:
                raise VersionConflictError("交易状态冲突: " + code)
            scope = suspension_index.at(code, at, policy)
            if scope:
                value, reason = suspension_status(scope, at, policy)
                suspended = 1 - value if value in (0, 1) else None
                life = scope['evidence'].get('lifecycle') or {}
                if life.get('complete') and life.get('list_date'):
                    is_listed = int(life['list_date'] <= day.isoformat() and
                                    (not life.get('delist_date') or day.isoformat() < life['delist_date']))
            elif not known_events and values:
                suspended = 1 - values.pop()
                reason = "源端日频状态；不承诺订单可成交"
            elif not known_events and not daily:
                observations = [r for r in quotes.get((code, day), []) if available(r, at, policy)]
                exchange = {'SH': 'SSE', 'SZ': 'SZSE', 'BJ': 'BSE'}.get(code.rsplit('.', 1)[-1])
                if calendar.get((exchange, day)) == {1} and any(r.get('volume') is not None and r['volume'] > 0 for r in observations):
                    suspended, reason = 0, '当日成交推定；未核验完整日内停牌范围'
            name = historical_name(names, code, at, policy)
            st = int(bool(re.match(r"^(?:S\*?ST|\*?ST)", name))) if name else None
            records.append({"code": code, "datetime": local_time(at), "name": name,
                "is_listed": is_listed, "is_st": st, "is_suspended": suspended,
                "remark": {"pit_policy": policy, "reason": reason, "lifecycle_basis": "source_history_assumed",
                    "st_basis": "historical_name_assumed" if name else "unknown",
                    "before_listing": bool(listed and day < listed),
                    "status_sources": sorted({r["status_source"] for r in daily if r.get("status_source")})}})
    return frame(records, RISK_COLUMNS, integers=("is_listed", "is_st", "is_suspended"), times=("datetime",))


def query_trade_status(source, table_name="trade_status_daily", codes=None, dates=None, metric="交易状态", *, pit_policy=None):
    policy = resolve_policy(source, pit_policy)
    if metric != "交易状态":
        raise NotImplementedError("状态接口仅支持 交易状态")
    if table_name == "stock_suspension_intervals_versions":
        return _strict_trade_status(source, table_name, codes, dates, pit_policy=policy)
    risks = query_trade_risks(source, table_name, codes, dates, pit_policy=policy)
    records = []
    for row in risks.to_dict("records"):
        listed, suspended = row["is_listed"], row["is_suspended"]
        value = -1 if not pd.isna(listed) and listed == 0 and row["remark"].get("before_listing") else None
        if not pd.isna(listed) and listed == 1 and not pd.isna(suspended):
            value = 0 if suspended else 1
        records.append({"code": row["code"], "datetime": row["datetime"], "value": value,
            "status_text": "未上市或已退出上市范围" if value == -1 else row["remark"]["reason"], "name": row["name"]})
    return frame(records, STATUS_COLUMNS, integers=("value",), times=("datetime",))
