"""Pure, versioned business rules shared by mapping and archive replay.

These functions do not fetch data or guess missing source dates.  Inputs may be
normalized source rows or the corresponding raw values; evidence and transport
metadata are deliberately excluded from business version identifiers.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal, localcontext

from .catalog import dataset
from .util import digest, parse_date


VERSION = "business-semantics-v2"
FINANCIAL_ENDPOINTS = frozenset({"income", "balancesheet", "cashflow", "fina_indicator"})
RESEARCH_COLUMNS = {
    "income": "research_and_development_expense",
    "fina_indicator": "research_and_development_investment",
}


def daily_average_price(asset_type: str, amount, volume) -> Decimal | None:
    """Compute a security price only from already normalized amount/volume.

    Aggregated index turnover is not a measure of index point prices.
    """
    if asset_type == "index" or amount is None or volume is None:
        return None
    amount, volume = Decimal(str(amount)), Decimal(str(volume))
    if not amount.is_finite() or not volume.is_finite():
        raise ValueError("均价输入必须是有限数值")
    if volume <= 0:
        return None
    with localcontext() as context:
        context.prec = 80
        return amount / volume


def normalized_business_content(endpoint: str, source: dict) -> dict:
    """Canonical source values, independent of field order and response format.

    Source normalization has already applied numeric unit multipliers.  They
    must not be applied again here.  All registered fields, including optional
    and textual revision descriptors, take part in a financial version.
    """
    content = {}
    for field in dataset(endpoint.removesuffix("_vip")).fields:
        value = source.get(field["name"])
        if value is None or value == "":
            value = None
        elif field["type"] == "date":
            value = parse_date(value)
        elif field["type"] == "numeric":
            value = Decimal(str(value))
            if not value.is_finite():
                raise ValueError("非有限业务数值：" + field["name"])
        else:
            value = str(value)
        content[field["name"]] = value
    return content


def financial_record(endpoint: str, source: dict, field_map: dict) -> dict:
    """Produce the complete financial payload before calculating its version.

    Ordinary and VIP endpoints use one statement identity.  A revision to
    ann_date, f_ann_date, end_type, flags, or any source measure changes the
    version; fetching identical data under another task does not.
    """
    statement = endpoint.removesuffix("_vip")
    if statement not in FINANCIAL_ENDPOINTS:
        raise ValueError("不是已登记财报接口：" + endpoint)
    content = normalized_business_content(statement, source)
    partial = {destination[1]: content.get(field) for field, destination in field_map.items()}
    partial.update(
        announcement_date=content.get("ann_date"),
        actual_announcement_date=content.get("f_ann_date"),
        end_type=content.get("end_type"),
        report_type=content.get("report_type") or "unspecified",
        statement_type=statement,
    )
    # rd_exp is retained only for old persisted data.  Its two source meanings
    # belong in different columns and must never be reintroduced by new rows.
    partial.pop("rd_exp", None)
    if statement in RESEARCH_COLUMNS:
        partial[RESEARCH_COLUMNS[statement]] = content.get("rd_exp")
    partial["version_id"] = digest({"statement_type": statement, "content": content})
    return partial


def corporate_action_partial(endpoint: str, source: dict, *, old_name: str | None = None) -> dict | None:
    """Project actual events while keeping proposals in their detail table.

    Dividend report periods and name interval end dates are never event dates.
    Tushare cash_div is after-tax cash per share; stk_bo_rate is the share
    dividend ratio, distinct from total stock distributions and capitalization.
    """
    if endpoint == "dividend":
        if source.get("div_proc") not in {"实施", "已实施"} or not source.get("ex_date"):
            return None
        effective = parse_date(source["ex_date"])
        return {
            "security_code": source.get("ts_code"), "event_type": endpoint,
            "event_date": effective, "effective_date": effective,
            "cash_dividend": source.get("cash_div"), "share_dividend": source.get("stk_bo_rate"),
        }
    if endpoint == "namechange":
        if not source.get("start_date"):
            return None
        effective = parse_date(source["start_date"])
        return {
            "security_code": source.get("ts_code"), "event_type": endpoint,
            "event_date": effective, "effective_date": effective,
            "new_name": source.get("name"), "old_name": old_name,
        }
    raise ValueError("不是已登记名称或分红事件接口：" + endpoint)


def derive_previous_names(source_rows: list[dict], *, complete_history: bool = False) -> dict[tuple[str, date], str]:
    """Return only immediately adjacent, unambiguous historical name links.

    The caller must establish a complete source response before setting the
    flag.  Merely receiving a page or a date-filtered subset is not proof.
    First records, gaps, overlaps, competing revisions and unknown interval
    ends intentionally produce no previous-name value.
    """
    if not complete_history:
        return {}
    by_security = defaultdict(dict)
    for row in source_rows:
        code, start, name = row.get("ts_code"), row.get("start_date"), row.get("name")
        if not code or not start or not name:
            continue
        start = parse_date(start)
        end = parse_date(row["end_date"]) if row.get("end_date") else None
        signature = (str(name), end)
        by_security[str(code)].setdefault(start, set()).add(signature)
    output = {}
    for code, starts in by_security.items():
        intervals = sorted(starts.items())
        for index, (start, candidates) in enumerate(intervals):
            if index == 0 or len(candidates) != 1:
                continue
            _, current_end = next(iter(candidates))
            if current_end is not None and current_end < start:
                continue
            previous_start, previous_candidates = intervals[index - 1]
            if len(previous_candidates) != 1:
                continue
            previous_name, previous_end = next(iter(previous_candidates))
            if previous_end is None or previous_end < previous_start or previous_end + timedelta(days=1) != start:
                continue
            # A longer earlier interval would make the apparent predecessor
            # ambiguous even if its own next interval is adjacent.
            if any(end is None or end >= start for _, alternatives in intervals[:index - 1] for _, end in alternatives):
                continue
            output[(code, start)] = previous_name
    return output
