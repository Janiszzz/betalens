from datetime import date
from decimal import Decimal

import pytest

from betalens_db_manager.nav_semantics import resolve_nav_rows


def row(announcement="20261020", **updates):
    result = dict(security_code="510300.SH", nav_date="20261016", announcement_date=announcement,
                  unit_nav="1.20", accum_nav="2.30", accum_div=None, net_asset="1000",
                  total_netasset="2000", adj_nav="2.31", event_date="2026-10-16",
                  downloaded_at="2026-10-21T00:00:00Z", import_task_id="first", source_row_no=1,
                  source_name="tushare", source_record_hash="first", mapping_version="old")
    result.update(updates)
    return result


def test_latest_publication_is_selected_as_one_whole_record_without_fill():
    old = row(net_asset="1000", unit_nav="1.20")
    latest = row("20261021", net_asset=None, unit_nav="1.21", source_record_hash="latest", import_task_id="latest")
    chosen, conflicts = resolve_nav_rows([latest, old])
    assert not conflicts and chosen == [latest]
    assert chosen[0]["net_asset"] is None
    assert chosen[0]["source_record_hash"] == "latest"
    assert old["net_asset"] == "1000"  # caller inputs are not mutated


def test_same_announcement_with_different_business_values_blocks_only_that_key():
    good = row(security_code="510050.SH")
    chosen, conflicts = resolve_nav_rows([row(), row(unit_nav="1.21"), good])
    assert chosen == [good] and len(conflicts) == 1
    assert conflicts[0]["blocking"] and "同一公告日期" in conflicts[0]["reason"]
    assert conflicts[0]["business_key"] == {"security_code": "510300.SH", "nav_date": date(2026, 10, 16)}


def test_system_metadata_differences_are_not_business_conflicts():
    older = row()
    newer = row(announcement=date(2026, 10, 20), unit_nav=Decimal("1.200"),
                event_date=None, available_at="2026-10-20T16:00:00Z", availability_basis="ann_date_inferred",
                downloaded_at="2026-10-22T00:00:00Z", ingested_at="2026-10-22T01:00:00Z",
                import_task_id="different", source_id="other", source_name="other", source_endpoint="new",
                source_row_no=99, source_record_hash="different", mapping_version="new", schema_version="new",
                data_quality_status="accepted")
    chosen, conflicts = resolve_nav_rows([older, newer])
    assert not conflicts and chosen == [newer]
    assert resolve_nav_rows([newer, older]) == (chosen, conflicts)


@pytest.mark.parametrize("other", [row(None, unit_nav="1.21"), row("20261021", unit_nav="1.21")])
def test_missing_announcement_prevents_choosing_between_different_values(other):
    chosen, conflicts = resolve_nav_rows([row(None), other])
    assert not chosen and len(conflicts) == 1 and conflicts[0]["blocking"]
    assert "公告日期" in conflicts[0]["reason"]


def test_undated_identical_values_can_be_deduplicated_or_use_dated_evidence():
    undated = row(None)
    dated = row()
    assert resolve_nav_rows([undated, dict(undated)]) == ([undated], [])
    assert resolve_nav_rows([undated, dated]) == ([dated], [])


def test_conflicting_historical_publication_is_not_hidden_by_a_newer_version():
    chosen, conflicts = resolve_nav_rows([row(), row(unit_nav="1.21"), row("20261022", unit_nav="1.22")])
    assert not chosen and conflicts[0]["blocking"]


def test_null_and_zero_are_different_business_values_and_zero_is_retained():
    zero = row(unit_nav="0")
    assert resolve_nav_rows([zero]) == ([zero], [])
    chosen, conflicts = resolve_nav_rows([zero, row(unit_nav=None)])
    assert not chosen and conflicts[0]["blocking"]


def test_distinct_nav_dates_do_not_conflict_and_invalid_values_are_rejected():
    first, second = row(), row(nav_date="20261019", unit_nav="1.21")
    chosen, conflicts = resolve_nav_rows([first, second, row(unit_nav="NaN"), row(nav_date=None)])
    assert chosen == [first, second] and len(conflicts) == 2
    assert all(c["blocking"] for c in conflicts)


def test_empty_input_and_normalized_numeric_equivalence():
    assert resolve_nav_rows([]) == ([], [])
    chosen, conflicts = resolve_nav_rows([row(unit_nav="1.20"), row(unit_nav=Decimal("1.2000"))])
    assert len(chosen) == 1 and not conflicts
