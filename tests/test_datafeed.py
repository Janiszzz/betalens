from contextlib import contextmanager
from copy import deepcopy
from datetime import date
import json

import numpy as np
import pandas as pd
import pytest

from betalens.datafeed import Datafeed, PITUnavailableError, SchemaError, VersionConflictError
from betalens.datafeed.config import load_subscription_config
from betalens.datafeed.contracts import timestamp
from betalens.datafeed import query
from betalens.datafeed.industry import query_industry
from betalens.datafeed.reference import load_reference, known_scopes, calendar_days
from betalens.datafeed.readers import select_sql, ClickHouseReader
from betalens.datafeed.universe import get_index_universe_panel
from datafeed_support import (MemoryReader, market_reader, quote, reference_reader, calendar_scope,
    evidence, BASIS, MARKET_TYPES, FIN_TYPES, NAME_TYPES)


def test_nearest_preserves_duplicates_order_missing_and_output_types():
    reader = market_reader()
    result = query.query_nearest_before(reader, "market_daily_post_adjusted", ["A", "X", "A"],
                                       ["2024-01-03 15:00", "2024-01-01"], "close")
    assert result.code.tolist() == ["A", "A", "X", "X", "A", "A"]
    assert result["close"].iloc[[0, 4]].tolist() == [20, 20]
    assert result["close"].iloc[[1, 2, 3, 5]].isna().all()
    assert result.diff_hours.iloc[0] == 0
    assert list(result.columns) == ["code", "input_ts", "datetime", "diff_hours", "close", "name"]
    assert str(result.datetime.dtype) == "datetime64[ns]"
    assert result["close"].dtype == np.dtype("float64")
    assert reader.calls.count("market_daily_post_adjusted") == 1


@pytest.mark.parametrize("function,expected", [(query.query_nearest_before, 10), (query.query_nearest_after, 20)])
def test_exact_match_direction_and_timezone(function, expected):
    result = function(market_reader(), "market_daily_post_adjusted", ["A"], ["2024-01-02T07:00:00Z"], "close")
    assert result["close"].iloc[0] == expected
    assert result.input_ts.iloc[0] == pd.Timestamp("2024-01-02 15:00")


def test_ranges_have_original_endpoint_rules_and_tolerance():
    interval = [("2024-01-02 15:00", "2024-01-03 15:00")]
    before = query.query_nearest_in_range_before(market_reader(), "market_daily_post_adjusted", ["A"], interval, "close")
    after = query.query_nearest_in_range_after(market_reader(), "market_daily_post_adjusted", ["A"], interval, "close")
    assert before["close"].iloc[0] == 20
    assert after["close"].isna().all()
    assert query.query_nearest_before(market_reader(), "market_daily_post_adjusted", ["A"], ["2024-01-03 14:00"], "close", 1)["close"].isna().all()
    with pytest.raises(ValueError):
        query.query_nearest_before(market_reader(), "market_daily_post_adjusted", ["A"], [], "close", -1)


def test_range_date_end_asof_limit_and_empty_contract():
    reader = market_reader()
    result = query.query_time_range(reader, "market_daily_post_adjusted", ["A"], "2024-01-02", "2024-01-03", "close", 1)
    assert result.value.tolist() == [20]
    asof = query.query_time_range(reader, "market_daily_post_adjusted", ["A"], end_date="2024-01-03", metric="close", as_of="2024-01-03 14:59")
    assert asof.value.tolist() == [10]
    empty = query.query_time_range(reader, "market_daily_post_adjusted", [], metric="close")
    assert empty.empty
    assert empty.dtypes.to_dict() == result.dtypes.to_dict()
    assert query.get_available_dates(reader, "market_daily_post_adjusted", "A", "close")[0] == pd.Timestamp("2024-01-02 15:00")


@pytest.mark.parametrize("basis", [None, "unknown", "next_calendar_day_assumed", "source_publication_time_unknown", "downloaded_at", "source_publication_time"])
def test_inferred_or_version_unproven_data_never_becomes_visible(basis):
    reader = market_reader([quote(availability_basis=basis)])
    result = query.query_nearest_before(reader, "market_daily_post_adjusted", ["A"], ["2025-01-01"], "close")
    assert result["close"].isna().all()


def test_null_revision_does_not_resurrect_old_non_null_value():
    reader = market_reader()
    reader.tables["market_daily_post_adjusted"][1]["close"] = None
    result = query.query_nearest_before(reader, "market_daily_post_adjusted", ["A"], ["2024-01-04"], "close")
    assert result["close"].isna().all()
    assert result.datetime.iloc[0] == pd.Timestamp("2024-01-03 15:00")


def test_conflicting_values_and_required_schema_drift_are_not_silenced():
    reader = market_reader([quote(), quote(value=99)])
    with pytest.raises(VersionConflictError):
        query.query_time_range(reader, "market_daily_post_adjusted", metric="close")
    reader = market_reader()
    reader.types["market_daily_post_adjusted"].pop("available_at")
    with pytest.raises(SchemaError, match="available_at"):
        query.query_time_range(reader, "market_daily_post_adjusted", metric="close")


def test_financial_versions_selected_by_public_availability_and_statement():
    rows = []
    for version, at, value in [("z", "2024-04-30 18:00", 10), ("a", "2024-05-30 18:00", 12)]:
        rows.append({"security_code": "A", "event_date": date(2024, 3, 31), "period_end": date(2024, 3, 31),
                     "statement_type": "income", "report_type": "1", "version_id": version, "revenue": value,
                     **evidence(at)})
        rows[-1]["available_at"] = timestamp(at)
    reader = MemoryReader({"fundamental_periodic": rows}, {"fundamental_periodic": FIN_TYPES})
    result = query.query_nearest_before(reader, "fundamental_periodic", ["A"], ["2024-04-30 17:00", "2024-05-01", "2024-06-01"], "revenue", filters={"statement_type": "income", "report_type": "1"})
    assert np.isnan(result["revenue"].iloc[0])
    assert result["revenue"].iloc[1:].tolist() == [10, 12]
    with pytest.raises(ValueError):
        query.query_nearest_before(reader, "fundamental_periodic", [], [], "股息率(报告期)")


def test_historical_names_never_fall_back_to_current_names():
    reader = market_reader()
    reader.types["security_name_history"] = NAME_TYPES
    reader.tables["security_name_history"] = [{"security_code": "A", "name": "later",
        "start_date": date(2020, 1, 1), "end_date": None, **evidence("2025-01-01")}]
    result = query.query_time_range(reader, "market_daily_post_adjusted", metric="close")
    assert result.name.isna().all()


def test_raw_post_and_asset_type_are_explicit():
    reader = market_reader()
    post = query.query_time_range(reader, "market_daily_post_adjusted", metric="close")
    raw = query.query_time_range(reader, "market_daily_raw", metric="close")
    assert post.value.tolist() == [2 * v for v in raw.value]
    reader.tables["market_daily_raw"][0]["asset_type"] = "index"
    assert len(query.query_time_range(reader, "market_daily_raw", metric="close", filters={"asset_type": "index"})) == 1


@pytest.mark.parametrize("event", [None, "revoke"])
def test_unpublished_and_revoked_new_scope_cannot_hide_older_snapshot(event):
    reader = reference_reader("calendar", [("old", [calendar_scope()], "publish"),
                                            ("new", [calendar_scope(opened=0)], event)])
    scopes = load_reference(reader, "calendar")
    assert calendar_days(reader, "2024-01-02", "2024-01-02", as_of="2024-01-03") == [date(2024, 1, 2)]
    assert known_scopes(scopes, "2024-01-03")[0]["revision_id"] == "old"


def test_later_scope_does_not_leak_into_earlier_asof_and_unknown_scope_cannot_be_used():
    reader = reference_reader("calendar", [("old", [calendar_scope()], "publish"),
        ("new", [calendar_scope(opened=0, available="2024-02-01")], "publish")])
    assert calendar_days(reader, "2024-01-02", "2024-01-02", as_of="2024-01-03") == [date(2024, 1, 2)]
    assert calendar_days(reader, "2024-01-02", "2024-01-02", as_of="2024-02-01") == []
    scope = calendar_scope()
    scope["evidence"] = {}
    unknown = reference_reader("calendar", [("unknown", [scope], "publish")])
    with pytest.raises(PITUnavailableError):
        calendar_days(unknown, "2024-01-02", "2024-01-02", as_of="2024-02-01")


def test_reference_integrity_and_calendar_gaps():
    reader = reference_reader("calendar", [("r", [calendar_scope()], "publish")])
    with pytest.raises(PITUnavailableError):
        calendar_days(reader, "2024-01-01", "2024-01-02", as_of="2024-01-03")
    reader.tables["trade_calendar_versions"][0]["is_open"] = 0
    with pytest.raises(VersionConflictError):
        load_reference(reader, "calendar")


def test_industry_replaces_whole_scope_and_does_not_fill_old_values():
    scope = {"scope_id": "A:SW2021:1", "selector": {"security_code": "A", "scheme_code": "SW2021", "level": 1},
             "complete": True, "evidence": {"availability": evidence()},
             "rows": [{"security_code": "A", "scheme_code": "SW2021", "level": 1,
                       "industry_code": "01", "industry_name": "old", "valid_from": "2020-01-01", "valid_to": None,
                       "boundary_basis": "source_effective_interval", **evidence()}]}
    newer = deepcopy(scope)
    newer["rows"][0]["industry_name"] = None
    reader = reference_reader("industry", [("old", [scope], "publish"), ("new", [newer], "publish")])
    result = query_industry(reader, ["A", "unknown"], ["2024-01-02"], "申万一级行业（2021）")
    assert result.ind_name.isna().all()
    assert result.ind_code.iloc[0] == "01"
    assert result.ind_code.iloc[1] is None


def suspension_scope(rows=None):
    return {"scope_id": "A:2024-01-02", "selector": {"security_code": "A", "trade_date": "2024-01-02"},
            "complete": True, "rows": rows or [], "evidence": {
                "availability": evidence(), "event_query_complete": True,
                "lifecycle": {"complete": True, "list_date": "2020-01-01", **evidence()},
                "calendar": {"complete": True, "rows": [{"is_open": 1, **evidence()}]}}}


def test_zero_event_proof_and_unknown_state_stay_distinct():
    reader = reference_reader("suspension", [("r", [suspension_scope()], "publish")])
    result = query.query_trade_status(reader, "stock_suspension_intervals_versions", ["A", "X"], ["2024-01-02 10:00"])
    assert result.value.iloc[0] == 1
    assert pd.isna(result.value.iloc[1])
    assert str(result.value.dtype) == "Int8"
    scope = suspension_scope()
    scope["evidence"]["lifecycle"].pop("available_at")
    unknown = reference_reader("suspension", [("r", [scope], "publish")])
    assert query.query_trade_status(unknown, "stock_suspension_intervals_versions", ["A"], ["2024-01-02 10:00"]).value.isna().all()


def test_suspension_interval_is_left_closed_right_open():
    row = {"security_code": "A", "trade_date": "2024-01-02", "suspend_type": "S", "parse_status": "parsed",
           "start_at": timestamp("2024-01-02 10:00").isoformat(), "end_at": timestamp("2024-01-02 11:00").isoformat(), **evidence()}
    reader = reference_reader("suspension", [("r", [suspension_scope([row])], "publish")])
    result = query.query_trade_status(reader, "stock_suspension_intervals_versions", ["A"], ["2024-01-02 10:00", "2024-01-02 11:00"])
    assert result.value.tolist() == [0, 1]


def test_index_needs_explicit_complete_snapshot_and_does_not_fill_partial():
    types = {"index_code": "text", "security_code": "text", "trade_date": "date", **query.PROVEN_COLUMNS}
    reader = MemoryReader({"index_membership": []}, {"index_membership": types})
    with pytest.raises(PITUnavailableError, match="完整快照"):
        get_index_universe_panel(reader, "I", ["2024-01-02"])
    reader.types["index_membership"].update(snapshot_complete="integer", snapshot_member_count="integer")
    reader.tables["index_membership"] = [{"index_code": "I", "security_code": c, "trade_date": date(2024, 1, 1),
        "snapshot_complete": 1, "snapshot_member_count": 2, **evidence()} for c in ("A", "B")]
    assert get_index_universe_panel(reader, "I", ["2024-01-02"]) == {date(2024, 1, 2): {"A", "B"}}
    reader.tables["index_membership"].pop()
    with pytest.raises(PITUnavailableError):
        get_index_universe_panel(reader, "I", ["2024-01-02"])


def test_config_explicit_invalid_file_never_falls_back_and_overrides_are_isolated(tmp_path, monkeypatch):
    for key in list(__import__('os').environ):
        if key.startswith("BETALENS_"):
            monkeypatch.delenv(key)
    path = tmp_path / "subscription.json"
    path.write_text(json.dumps({"postgres": {"host": "db", "dbname": "research", "user": "reader"},
                                "clickhouse": {"url": "http://db:8123", "database": "research", "user": "reader"}}))
    monkeypatch.setenv("BETALENS_DB_NAME", "from_env")
    assert load_subscription_config(config_file=path)["dbname"] == "from_env"
    assert load_subscription_config({"dbname": "explicit"}, config_file=path)["dbname"] == "explicit"
    assert load_subscription_config(config_file=path, backend="clickhouse")["database"] == "research"
    path.write_text("not json")
    with pytest.raises(ValueError, match="无法读取"):
        load_subscription_config({"host": "db", "dbname": "x", "user": "r"}, config_file=path)


def test_sql_values_are_bound_and_extra_columns_do_not_change_projection():
    reader = market_reader()
    reader.backend = "clickhouse"
    reader.types["market_daily_raw"]["extra_column"] = "text"
    sql, params = select_sql(reader, "market_daily_raw", ["close"], [("security_code", "IN", ["x'; DROP TABLE t;--"])])
    assert "DROP TABLE" not in sql and "FINAL" in sql
    assert "extra_column" not in sql
    assert list(params.values()) == ["x'; DROP TABLE t;--"]
    with pytest.raises(ValueError):
        select_sql(reader, "market_daily_raw", ["close"], [("close", "OR 1=1", 0)])


def test_clickhouse_transport_is_readonly_and_sanitizes_errors(monkeypatch):
    import requests
    from unittest.mock import Mock
    reader = ClickHouseReader({"url": "http://localhost:8123", "database": "test", "user": "reader", "password": "secret"})
    request = Mock(side_effect=requests.ConnectionError("secret"))
    monkeypatch.setattr(reader.session, "post", request)
    with pytest.raises(RuntimeError, match="只读查询失败") as error:
        reader.execute("SELECT 1")
    assert "secret" not in str(error.value)
    assert request.call_args.kwargs["params"]["readonly"] == "1"
    with pytest.raises(ValueError):
        reader.execute("DELETE FROM x")
    reader.close()


def test_calendar_half_year_is_not_a_rolling_two_quarter_group():
    from betalens.datafeed.core import _period_days
    days = ["2024-03-29", "2024-06-28", "2024-09-30", "2024-12-31"]
    assert _period_days(days, "H") == [date(2024, 6, 28), date(2024, 12, 31)]
    assert _period_days(days, "S") == _period_days(days, "H")


def test_native_unit_metadata_is_returned_without_alias_assumptions():
    reader = market_reader()
    reader.types["meta_column"] = {"table_name": "text", "column_name": "text", "unit": "text"}
    reader.tables["meta_column"] = [{"table_name": "market_daily_post_adjusted", "column_name": "close", "unit": "万元"}]
    result = query.query_time_range(reader, "market_daily_post_adjusted", metric="close")
    assert result.value.tolist() == [20, 10]
    assert result.remark.iloc[0]["unit"] == "万元"


def test_empty_optional_unit_catalog_does_not_invent_units():
    reader = market_reader()
    reader.types["meta_column"] = {"table_name": "text", "column_name": "text", "unit": "text"}
    reader.tables["meta_column"] = []
    assert query.query_time_range(reader, "market_daily_post_adjusted", metric="close").value.tolist() == [20, 10]


def test_conflict_detection_precedes_float64_rounding():
    from decimal import Decimal
    reader = market_reader([quote(value=Decimal("10000000000000000")), quote(value=Decimal("10000000000000001"))])
    with pytest.raises(VersionConflictError):
        query.query_time_range(reader, "market_daily_post_adjusted", metric="close")


def test_same_value_different_financial_versions_is_still_a_conflict():
    row = {"security_code": "A", "event_date": date(2024, 3, 31), "period_end": date(2024, 3, 31),
           "statement_type": "income", "report_type": "1", "version_id": "v1", "revenue": 10,
           **evidence("2024-05-01")}
    row["available_at"] = timestamp("2024-05-01")
    reader = MemoryReader({"fundamental_periodic": [row, dict(row, version_id="v2")]}, {"fundamental_periodic": FIN_TYPES})
    with pytest.raises(VersionConflictError):
        query.query_nearest_before(reader, "fundamental_periodic", ["A"], ["2024-06-01"], "revenue", filters={"statement_type": "income", "report_type": "1"})


def test_empty_and_bad_ranges_are_not_accidental_full_scans():
    reader = market_reader()
    assert query.query_nearest_before(reader, "market_daily_post_adjusted", [], [], "close").empty
    assert not reader.calls
    with pytest.raises(ValueError, match="ranges"):
        query.query_nearest_in_range_before(reader, "market_daily_post_adjusted", ["A"], [("2024-01-01",)], "close")


def test_industry_reverse_query_does_not_turn_unknown_into_empty_membership():
    from betalens.datafeed.industry import get_industry_members
    reader = reference_reader("industry", [])
    with pytest.raises(PITUnavailableError):
        get_industry_members(reader, "银行", "2024-01-02")


def test_direct_tables_fields_and_explicit_filters_need_no_registry():
    reader = market_reader()
    rows = reader.tables.pop("market_daily_post_adjusted")
    types = reader.types.pop("market_daily_post_adjusted")
    reader.tables["new_observations"] = [dict(r, custom_value=r["close"] * 3) for r in rows]
    reader.types["new_observations"] = dict(types, custom_value="numeric")
    result = query.query_time_range(reader, "new_observations", metric="custom_value",
                                    filters={"asset_type": "stock"})
    assert result.value.tolist() == [60, 30]
    assert result.metric.tolist() == ["custom_value", "custom_value"]
    assert result.remark.iloc[0]["table"] == "new_observations"
    assert result.remark.iloc[0]["unit"] is None
    assert query.query_time_range(reader, "new_observations", metric="custom_value",
                                  filters={"asset_type": "index"}).empty


def test_logical_dataset_and_metric_aliases_are_not_resolved():
    reader = market_reader()
    with pytest.raises(SchemaError, match="daily_market"):
        query.query_nearest_before(reader, "daily_market", [], [], "close")
    with pytest.raises(ValueError, match="标识符"):
        query.query_time_range(reader, "market_daily_post_adjusted", metric="收盘价(元)")
    with pytest.raises(SchemaError, match="prev_close"):
        query.query_time_range(reader, "market_daily_post_adjusted", metric="prev_close")
    with pytest.raises(ValueError, match="显式指定"):
        query.query_time_range(reader, "market_daily_post_adjusted")
    with pytest.raises(SchemaError, match="数值字段"):
        query.query_time_range(reader, "market_daily_post_adjusted", metric="asset_type")


def test_direct_filters_validate_names_and_keep_financial_scope_explicit():
    reader = market_reader()
    with pytest.raises(ValueError, match="标识符"):
        query.query_time_range(reader, "market_daily_post_adjusted", metric="close",
                               filters={"asset_type OR 1=1": "stock"})
    with pytest.raises(SchemaError, match="missing_column"):
        query.query_time_range(reader, "market_daily_post_adjusted", metric="close",
                               filters={"missing_column": "stock"})
    with pytest.raises(ValueError, match="非空标量"):
        query.query_time_range(reader, "market_daily_post_adjusted", metric="close",
                               filters={"asset_type": ["stock", "index"]})
    financial = MemoryReader({"fundamental_periodic": []}, {"fundamental_periodic": FIN_TYPES})
    with pytest.raises(ValueError, match="statement_type 和 report_type"):
        query.query_time_range(financial, "fundamental_periodic", metric="revenue")


def test_calendar_uses_sse_literally_without_shse_fallback():
    reader = reference_reader("calendar", [("r", [calendar_scope()], "publish")])
    assert calendar_days(reader, "2024-01-02", "2024-01-02", exchange="SSE") == [date(2024, 1, 2)]
    with pytest.raises(PITUnavailableError, match="SHSE"):
        calendar_days(reader, "2024-01-02", "2024-01-02", exchange="SHSE")


def test_direct_field_queries_cannot_bypass_reference_publication_checks():
    reader = reference_reader("calendar", [("r", [calendar_scope()], "revoke")])
    with pytest.raises(NotImplementedError, match="发布证据"):
        query.query_time_range(reader, "trade_calendar_versions", metric="is_open")
