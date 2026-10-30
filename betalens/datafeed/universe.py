"""Index snapshots. Row presence/weight sums are not completeness evidence."""
from __future__ import annotations

from collections import defaultdict

from .contracts import PITUnavailableError, VersionConflictError, local_time, proven, timestamp
from .readers import get_reader, require_columns
from .query import PROVEN_COLUMNS

DEFAULT_TABLE = "index_membership"
DEFAULT_METRIC = "universe"


def _snapshots(reader, index_code):
    if not index_code:
        raise ValueError("index_code 不能为空")
    table = "index_membership"
    require_columns(reader, table, {"index_code": "text", "security_code": "text", "trade_date": "date", **PROVEN_COLUMNS})
    # Optional additive contract for a future manager revision; no DDL here.
    proof = {"snapshot_complete": "integer", "snapshot_member_count": "integer"}
    if not proof.keys() <= reader.columns(table).keys():
        raise PITUnavailableError("index_membership 缺少完整快照证明(snapshot_complete/snapshot_member_count)；不能推定成分完整")
    require_columns(reader, table, proof)
    groups = defaultdict(list)
    for row in reader.rows(table, filters=[("index_code", "=", index_code)]):
        groups[row["trade_date"]].append(row)
    return groups


def _snapshot_at(groups, at):
    day = local_time(at).date()
    candidates = [d for d, rows in groups.items() if d <= day and all(proven(row, at) for row in rows)]
    if not candidates:
        raise PITUnavailableError("查询时点没有历史可知的指数完整快照")
    selected = max(candidates)
    rows = groups[selected]
    codes = {r["security_code"] for r in rows}
    counts = {r["snapshot_member_count"] for r in rows}
    if any(r["snapshot_complete"] != 1 for r in rows) or counts != {len(codes)} or not all(codes):
        raise PITUnavailableError("指数快照范围不完整")
    if len(codes) != len(rows):
        raise VersionConflictError("指数快照成分重复或版本冲突")
    return selected, sorted(codes)


def get_index_universe_panel(cursor, index_code, dates, table_name=DEFAULT_TABLE,
                             metric=DEFAULT_METRIC, logger=None):
    if table_name != DEFAULT_TABLE or metric != DEFAULT_METRIC:
        raise NotImplementedError("指数成分查询仅支持 index_membership/universe")
    if isinstance(dates, str):
        raise ValueError("dates 必须是列表")
    dates = list(dates)
    if not dates:
        return {}
    reader = get_reader(cursor)
    with reader.snapshot():
        groups = _snapshots(reader, index_code)
    return {local_time(timestamp(d)).date(): set(_snapshot_at(groups, timestamp(d))[1]) for d in dates}


def get_index_universe_date(cursor, index_code, date, table_name=DEFAULT_TABLE,
                            metric=DEFAULT_METRIC, logger=None):
    if table_name != DEFAULT_TABLE or metric != DEFAULT_METRIC:
        raise NotImplementedError("指数成分查询仅支持 index_membership/universe")
    reader = get_reader(cursor)
    with reader.snapshot():
        groups = _snapshots(reader, index_code)
    return timestamp(_snapshot_at(groups, timestamp(date))[0]).tz_convert("Asia/Shanghai").tz_localize(None).to_pydatetime()


def get_index_universe(cursor, index_code, date, table_name=DEFAULT_TABLE,
                        metric=DEFAULT_METRIC, logger=None):
    panel = get_index_universe_panel(cursor, index_code, [date], table_name, metric, logger)
    return sorted(panel[local_time(timestamp(date)).date()])
