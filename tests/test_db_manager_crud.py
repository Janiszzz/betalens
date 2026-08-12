from __future__ import annotations

import pandas as pd
import pytest

from betalens_db_manager import (
    ALLOWED_TABLES,
    DATASETS,
    DatabaseClient,
    DatabaseWriter,
    DeleteRequest,
    QueryRequest,
)


def _frame(values=(10.0,)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": ["2026-07-01 15:00:01"] * len(values),
            "code": ["000001.SZ"] * len(values),
            "name": ["平安银行"] * len(values),
            "metric": ["收盘价(元)"] * len(values),
            "value": list(values),
            "remark": [{"source": "test"}] * len(values),
        }
    )


def test_registry_exposes_eleven_logical_datasets_and_physical_routes():
    assert len(ALLOWED_TABLES) == 11
    assert tuple(DATASETS) == ALLOWED_TABLES
    assert DATASETS["daily_market"].storage == "market"
    assert "market_daily_fact" in DATASETS["daily_market"].physical_tables
    assert DATASETS["fundamentals"].storage == "observation"
    assert DATASETS["industry"].storage == "industry"
    assert DATASETS["trade_calendar"].physical_tables == ("trade_calendar_day",)


def test_query_request_page_token_roundtrip():
    row = {
        "datetime": pd.Timestamp("2026-07-01 15:00:01"),
        "code": "000001.SZ",
        "metric": "收盘价(元)",
    }

    token = DatabaseClient.make_page_token(row)

    assert DatabaseClient.parse_page_token(token) == (
        "2026-07-01T15:00:01",
        "000001.SZ",
        "收盘价(元)",
    )
    assert QueryRequest("daily_market", page_token=token).page_token == token


def test_prepare_frame_deduplicates_identical_rows_and_rejects_conflicts():
    writer = object.__new__(DatabaseWriter)

    assert len(writer._prepare_frame(_frame((10.0, 10.0)))) == 1
    with pytest.raises(ValueError, match="同键记录存在不同值"):
        writer._prepare_frame(_frame((10.0, 11.0)))

    conflicting_names = pd.concat([_frame(), _frame()], ignore_index=True)
    conflicting_names.loc[1, "metric"] = "最高价(元)"
    conflicting_names.loc[1, "name"] = "不同名称"
    with pytest.raises(ValueError, match="同一实体时点存在不同名称"):
        writer._prepare_frame(conflicting_names)


def test_delete_request_requires_a_scope_and_normalizes_codes():
    with pytest.raises(ValueError, match="禁止无条件删除"):
        DeleteRequest("daily_market").validate()

    request = DeleteRequest(
        "daily_market",
        code="000001.SZ",
        codes=("000001.SZ", "600000.SH"),
    )
    request.validate()
    assert request.normalized_codes() == ("000001.SZ", "600000.SH")
    assert DeleteRequest("daily_market", codes="000001.SZ").normalized_codes() == ("000001.SZ",)


def test_new_schema_market_source_reads_wide_and_observation_fallback():
    client = object.__new__(DatabaseClient)

    query, params = client._logical_source(DATASETS["daily_market"])

    assert "betalens.market_daily_fact" in query
    assert "betalens.observation_fact" in query
    assert "UNION ALL" in query
    assert params == ["daily_market", "stock", "daily_market", "stock"]


def test_trade_calendar_source_and_writer_normalize_exchange_dates():
    client = object.__new__(DatabaseClient)
    source, params = client._logical_source(DATASETS["trade_calendar"])
    assert "betalens.trade_calendar_day" in source
    assert params == []

    writer = object.__new__(DatabaseWriter)
    frame = pd.DataFrame(
        {
            "datetime": ["2024-01-02 12:34:00", "2024-01-02"],
            "code": [" shse ", "SHSE"],
            "name": ["ignored", "SHSE"],
            "metric": ["交易日", "交易日"],
            "value": [1.0, 1.0],
            "remark": [None, None],
        }
    )
    normalized = writer._prepare_frame(frame, table="trade_calendar")
    assert normalized[["datetime", "code", "name"]].to_dict("records") == [
        {"datetime": pd.Timestamp("2024-01-02"), "code": "SHSE", "name": "SHSE"}
    ]


def test_writer_streams_copy_in_batches():
    class Cursor:
        def __init__(self):
            self.copy_payloads = []

        def execute(self, _query, _params=None):
            return None

        def copy_expert(self, _query, stream):
            self.copy_payloads.append(stream.read())

    writer = object.__new__(DatabaseWriter)
    cursor = Cursor()
    frame = pd.concat([_frame((10.0,)), _frame((10.0,)), _frame((10.0,))], ignore_index=True)
    frame.loc[:, "datetime"] = pd.to_datetime(frame["datetime"])

    writer._copy_stage(cursor, frame, batch_size=2)

    assert len(cursor.copy_payloads) == 2
    assert sum(payload.count("000001.SZ") for payload in cursor.copy_payloads) == 3
