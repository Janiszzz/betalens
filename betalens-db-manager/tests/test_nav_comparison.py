from datetime import date
from decimal import Decimal

import pytest

from betalens_db_manager.canonical import canonical
from betalens_db_manager.comparison import compare, row_hash, typed
from betalens_db_manager.commit_protocol import remaining_changes
from betalens_db_manager.nav_semantics import NavPublicationConflict


def row(announcement="2026-10-20", **updates):
    result = dict(security_code="510300.SH", nav_date="2026-10-16", announcement_date=announcement,
                  unit_nav="1.20", accum_nav="2.30", accum_div=None, net_asset="1000",
                  total_netasset="2000", adj_nav="2.31", event_date="2026-10-16",
                  available_at="2026-10-20T16:00:00Z", availability_basis="announcement_inferred",
                  downloaded_at="2026-10-21T00:00:00Z", import_task_id="first", source_row_no=1,
                  source_name="tushare", source_record_hash="first", mapping_version="old")
    result.update(updates)
    return result


@pytest.mark.parametrize("mode", ["fill_missing", "refresh_source"])
def test_newer_nav_publication_replaces_whole_row_including_nulls(mode):
    spec = canonical("fund_nav_daily")
    before = row(source_name="old_provider")
    incoming = row("2026-10-21", unit_nav="1.21", net_asset=None, total_netasset=None,
                   source_name="new_provider", import_task_id="next", source_record_hash="next")
    result = compare(spec, [incoming], [before], mode)
    change = result["changes"][0]
    assert result["counts"]["updated"] == 1
    assert change["after"] == typed(spec, incoming)
    assert change["after"]["net_asset"] is None and change["after"]["total_netasset"] is None
    assert change["after"]["announcement_date"] == date(2026, 10, 21)
    assert change["after"]["source_name"] == "new_provider"
    assert change["before_hash"] == row_hash(spec, before)
    assert change["after_hash"] == row_hash(spec, incoming)
    assert remaining_changes(spec, result, [before]) == result["changes"]
    assert remaining_changes(spec, result, [incoming]) == []


@pytest.mark.parametrize("mode", ["fill_missing", "refresh_source"])
def test_earlier_publication_is_skipped_even_when_it_has_values_missing_today(mode):
    spec = canonical("fund_nav_daily")
    current = row("2026-10-21", net_asset=None)
    old = row("2026-10-20", unit_nav="1.19", net_asset="999")
    result = compare(spec, [old], [current], mode)
    assert not result["changes"] and result["counts"]["skipped"] == 1
    assert result["checks"][0]["before_hash"] == row_hash(spec, current)
    assert not remaining_changes(spec, result, [current])


@pytest.mark.parametrize("mode", ["fill_missing", "refresh_source"])
def test_same_announcement_with_changed_value_blocks_instead_of_filling_or_refreshing(mode):
    spec = canonical("fund_nav_daily")
    with pytest.raises(NavPublicationConflict, match="同一公告日期") as raised:
        compare(spec, [row(unit_nav="1.21")], [row()], mode)
    assert raised.value.conflict["blocking"]
    assert raised.value.conflict["table"] == "fund_nav_daily"


def test_unknown_publication_never_overwrites_known_values_or_announcement():
    spec = canonical("fund_nav_daily")
    result = compare(spec, [row(None)], [row()], "refresh_source")
    assert result["counts"]["skipped"] == 1 and not result["changes"]
    with pytest.raises(NavPublicationConflict, match="缺少公告日期"):
        compare(spec, [row(None, unit_nav="1.21")], [row()], "refresh_source")


def test_unknown_previous_version_accepts_only_identical_values_with_better_date_evidence():
    spec = canonical("fund_nav_daily")
    result = compare(spec, [row()], [row(None)], "fill_missing")
    assert result["changes"][0]["after"]["announcement_date"] == date(2026, 10, 20)
    with pytest.raises(NavPublicationConflict, match="缺少公告日期"):
        compare(spec, [row(unit_nav="1.21")], [row(None)], "refresh_source")


def test_identical_publication_does_not_refresh_only_transport_fields():
    spec = canonical("fund_nav_daily")
    result = compare(spec, [row(import_task_id="new", source_record_hash="new")], [row()], "refresh_source")
    assert result["counts"]["skipped"] == 1 and not result["changes"]


def test_other_tables_keep_existing_fill_missing_and_null_preservation_semantics():
    spec = canonical("market_daily_raw")
    previous = dict(asset_type="stock", security_code="000001.SZ", trade_date="2026-10-16", close="12", amount="1000")
    incoming = {**previous, "close": "13", "amount": None}
    filled = compare(spec, [incoming], [previous], "fill_missing")
    assert filled["counts"]["skipped"] == 1 and not filled["changes"]
    refreshed = compare(spec, [incoming], [previous], "refresh_source")
    assert refreshed["changes"][0]["after"]["close"] == Decimal("13")
    assert refreshed["changes"][0]["after"]["amount"] == Decimal("1000")
