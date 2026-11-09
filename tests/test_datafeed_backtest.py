"""Execution bars retain raw/adjusted pairs and source-version provenance."""
from types import SimpleNamespace

import pandas as pd
import pytest

from betalens.datafeed.backtest import query_backtest_bars
from betalens.datafeed.contracts import SchemaError, VersionConflictError
from datafeed_support import MemoryReader, MARKET_TYPES, quote


def source(rows=None, *, policy="strict"):
    fields = {**MARKET_TYPES, "raw_open": "numeric", "raw_close": "numeric", "adjustment_factor": "numeric"}
    default = quote(at="2024-01-02 16:00", value=20, raw_open=10, raw_close=10, adjustment_factor=2)
    reader = MemoryReader({"market_daily_post_adjusted": rows or [default], "stock_profile": [{"security_code": "A", "market": "创业板"}]},
                          {"market_daily_post_adjusted": fields, "stock_profile": {"security_code": "text", "market": "text"}})
    return SimpleNamespace(_reader=reader, pit_policy=policy)


def test_paired_bar_retains_business_day_and_available_time():
    result = query_backtest_bars(source(), "market_daily_post_adjusted", ["A"], "2024-01-02", "2024-01-02")
    assert len(result) == 1
    row = result.iloc[0]
    assert row.trade_date == pd.Timestamp("2024-01-02")
    assert row.available_at == pd.Timestamp("2024-01-02 16:00")
    assert row.open == row.raw_open * row.adjustment_factor
    assert row.market == "创业板"


def test_missing_raw_price_and_inconsistent_factor_are_rejected():
    data = source()
    del data._reader.types["market_daily_post_adjusted"]["raw_close"]
    with pytest.raises(SchemaError):
        query_backtest_bars(data, "market_daily_post_adjusted", ["A"], "2024-01-02")
    data = source()
    data._reader.tables["market_daily_post_adjusted"][0]["raw_close"] = 11
    with pytest.raises(ValueError, match="不一致"):
        query_backtest_bars(data, "market_daily_post_adjusted", ["A"], "2024-01-02")


def test_policy_does_not_promote_assumed_source_to_strict():
    row = quote(at="2024-01-02 16:00", value=20, raw_open=10, raw_close=10, adjustment_factor=2,
                availability_basis="next_calendar_day_assumed")
    strict = query_backtest_bars(source([row]), "market_daily_post_adjusted", ["A"], "2024-01-02")
    research = query_backtest_bars(source([row], policy="research"), "market_daily_post_adjusted", ["A"], "2024-01-02")
    assert strict.empty
    assert len(research) == 1
    assert research.iloc[0].availability_basis == "next_calendar_day_assumed"


def test_conflicting_business_day_versions_are_not_arbitrarily_picked():
    first = quote(at="2024-01-02 16:00", value=20, raw_open=10, raw_close=10, adjustment_factor=2)
    second = {**first, "close": 22, "raw_close": 11}
    with pytest.raises(VersionConflictError):
        query_backtest_bars(source([first, second]), "market_daily_post_adjusted", ["A"], "2024-01-02")
