"""Select one whole, evidenced NAV publication without mixing revisions."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from .canonical import canonical
from .registry import SYSTEM
from .util import digest, parse_date


VERSION = "nav-publication-v1"


def _business(row: dict) -> dict:
    result = {}
    for field in canonical("fund_nav_daily").fields:
        name = field["name"]
        if name in SYSTEM or field.get("system"):
            continue
        value = row.get(name)
        if value is None or value == "":
            value = None
        elif field["type"] == "date":
            value = parse_date(value)
        elif field["type"] == "numeric":
            value = Decimal(str(value))
            if not value.is_finite():
                raise ValueError("净值不是有限数值：" + name)
        else:
            value = str(value)
        result[name] = value
    if not result.get("security_code") or not result.get("nav_date"):
        raise ValueError("净值证券代码或净值日期为空")
    return result


def _conflict(key, reason, rows):
    return {
        "dataset": "fund_nav_daily", "table": "fund_nav_daily",
        "business_key": {"security_code": key[0], "nav_date": key[1]},
        "reason": reason, "blocking": True, "code": "nav_publication_conflict",
        "source_rows": len(rows), "sample_rows": [dict(row) for row in rows[:20]],
    }


class NavPublicationConflict(ValueError):
    """Current publication cannot be chosen safely from the available dates."""

    def __init__(self, reason, before, incoming):
        super().__init__(reason)
        key = (incoming.get("security_code"), parse_date(incoming["nav_date"]))
        self.conflict = _conflict(key, reason, [before, incoming])


def nav_publication_decision(before: dict, incoming: dict) -> str:
    """Return replace/skip for the atomic daily snapshot, or block ambiguity.

    This policy is independent of a generic fill-missing/refresh mode.  A later
    publication is a complete new snapshot, not a bag of optional columns.
    """
    previous, candidate = _business(before), _business(incoming)
    old_day, new_day = previous.pop("announcement_date", None), candidate.pop("announcement_date", None)
    same_values = previous == candidate
    if old_day is not None and new_day is not None:
        if new_day > old_day:
            return "replace"
        if new_day < old_day:
            return "skip"
        if same_values:
            return "skip"
        raise NavPublicationConflict("同一公告日期的净值与目标库业务值不同，必须解决版本冲突后重新审计", before, incoming)
    if old_day == new_day:  # both unknown
        if same_values:
            return "skip"
        raise NavPublicationConflict("新旧净值均缺少公告日期且业务值不同，无法判断版本先后", before, incoming)
    if not same_values:
        raise NavPublicationConflict("新旧净值存在缺少公告日期的不同业务值，无法判断版本先后", before, incoming)
    # Adding an actual announcement date to identical values is safe.  An
    # undated repeat cannot erase already known publication evidence.
    return "replace" if new_day is not None else "skip"


def resolve_nav_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Choose the latest announcement per security/NAV date, keeping its NULLs.

    ``fund_nav_daily`` uses ``announcement_date``; the source/detail field
    ``ann_date`` must already have been mapped by the caller.  System fields
    are evidence and never turn otherwise identical publications into a value
    conflict.  One undated value can be retained, but competing values with
    unknown announcement order cannot be silently chosen.
    """
    groups, chosen, conflicts = defaultdict(list), [], []
    for row in rows:
        try:
            business = _business(row)
            key = business["security_code"], business["nav_date"]
            groups[key].append((dict(row), business))
        except (ValueError, TypeError, ArithmeticError) as exc:
            conflicts.append(_conflict((row.get("security_code"), row.get("nav_date")), str(exc), [row]))
    for key, records in sorted(groups.items()):
        by_announcement = defaultdict(list)
        for row, business in records:
            # Announcement is a publication version; all other business fields
            # are compared to detect incompatible versions without a date.
            values = {name: value for name, value in business.items() if name != "announcement_date"}
            by_announcement[business.get("announcement_date")].append((row, digest(values)))
        ambiguous = next((day for day, candidates in by_announcement.items() if len({h for _, h in candidates}) > 1), False)
        if ambiguous is not False:
            label = "缺少公告日期，无法确定不同净值版本先后" if ambiguous is None else "同一公告日期存在不同净值业务值"
            conflicts.append(_conflict(key, label, [row for row, _ in records]))
            continue
        undated = by_announcement.get(None)
        if undated and len({h for candidates in by_announcement.values() for _, h in candidates}) > 1:
            conflicts.append(_conflict(key, "存在缺少公告日期的不同净值版本，无法确定应采用的当前值", [row for row, _ in records]))
            continue
        dated = [day for day in by_announcement if day is not None]
        selected = by_announcement[max(dated) if dated else None]
        # Deduplicate equivalent publications with deterministic evidence choice.
        # The selected record is copied whole; no older non-NULL value fills it.
        chosen.append(dict(max((row for row, _ in selected), key=lambda row: (str(row.get("downloaded_at") or ""), digest(row)))))
    return chosen, conflicts
