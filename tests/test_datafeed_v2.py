from __future__ import annotations

import logging
from datetime import date, datetime, time

import pandas as pd
import pytest
from psycopg2 import sql as psql

from betalens.datafeed import industry, pool, query, universe
from betalens.datafeed.registry import DATASETS as READ_DATASETS
from betalens_db_manager.registry import DATASETS as MANAGER_DATASETS


@pytest.fixture(autouse=True)
def _clear_datafeed_route_caches():
    query._METRIC_ROUTE_CACHE.clear()
    query._NORMALIZED_CONNECTIONS.clear()


class RecordingCursor:
    def __init__(self, *, fetchone=(), fetchall=()):
        self.calls = []
        self._fetchone = iter(fetchone)
        self._fetchall = iter(fetchall)

    def execute(self, statement, params=None):
        self.calls.append((statement, params))

    def fetchone(self):
        return next(self._fetchone, None)

    def fetchall(self):
        return next(self._fetchall, [])


def _sql_text(statement) -> str:
    """Render psycopg2.sql objects without requiring a live connection."""
    if isinstance(statement, psql.SQL):
        return statement._wrapped
    if isinstance(statement, psql.Identifier):
        return ".".join(f'"{part.replace(chr(34), chr(34) * 2)}"' for part in statement._wrapped)
    if isinstance(statement, psql.Composed):
        return "".join(_sql_text(part) for part in statement._wrapped)
    return str(statement)


def test_date_only_end_bound_is_next_day_exclusive():
    start, end, exclusive = query._time_bounds("2024-01-02", "2024-01-31")

    assert start == pd.Timestamp("2024-01-02")
    assert end == pd.Timestamp("2024-02-01")
    assert exclusive is True

    _, timed_end, timed_exclusive = query._time_bounds(
        None, "2024-01-31 15:00:01"
    )
    assert timed_end == pd.Timestamp("2024-01-31 15:00:01")
    assert timed_exclusive is False


@pytest.mark.parametrize(
    ("start", "end", "exclusive", "available", "expected"),
    [
        (
            pd.Timestamp("2024-01-02 15:00:01"),
            None,
            False,
            time(15, 0, 1),
            (date(2024, 1, 2), None),
        ),
        (
            pd.Timestamp("2024-01-02 15:00:02"),
            None,
            False,
            time(15, 0, 1),
            (date(2024, 1, 3), None),
        ),
        (
            None,
            pd.Timestamp("2024-01-31 14:59:59"),
            False,
            time(15, 0, 1),
            (None, date(2024, 1, 30)),
        ),
        (
            None,
            pd.Timestamp("2024-01-31 15:00:01"),
            False,
            time(15, 0, 1),
            (None, date(2024, 1, 31)),
        ),
        (
            None,
            pd.Timestamp("2024-02-01 00:00:00"),
            True,
            time(15, 0, 1),
            (None, date(2024, 1, 31)),
        ),
    ],
)
def test_fixed_market_availability_converts_to_indexable_trade_dates(
    start, end, exclusive, available, expected
):
    assert query._trade_date_bounds(start, end, exclusive, available) == expected


def test_metric_resolver_union_has_the_full_projection_on_both_arms():
    cursor = RecordingCursor(fetchone=[None])

    query._resolve_metric(cursor, "fundamentals", "ROE")

    statement = _sql_text(cursor.calls[0][0])
    arms = statement.split("UNION ALL")
    assert len(arms) == 2
    for arm in arms:
        assert "m.storage_kind" in arm
        assert "m.storage_column" in arm
        assert "m.availability_time" in arm
    assert cursor.calls[0][1] == (
        "fundamentals",
        "ROE",
        "fundamentals",
        "ROE",
        "fundamentals",
        "ROE",
    )


def test_read_and_write_dataset_registries_agree_on_routing_contract():
    assert set(READ_DATASETS) == set(MANAGER_DATASETS)
    for name, read_spec in READ_DATASETS.items():
        manager_spec = MANAGER_DATASETS[name]
        assert read_spec.kind == manager_spec.storage
        assert read_spec.entity_type == manager_spec.entity_type


def test_market_time_range_core_route_parameter_order_and_date_bound():
    resolved = {
        "metric_id": 4,
        "metric_name": "收盘价(元)",
        "storage_kind": "core",
        "storage_column": "close",
        "availability_time": time(15, 0, 1),
    }
    cursor = RecordingCursor(fetchone=[resolved], fetchall=[[]])

    result = query._query_time_range_normalized(
        cursor,
        "daily_market",
        ["000001.SZ"],
        "2024-01-02",
        "2024-01-31",
        "收盘价(元)",
        25,
    )

    statement, params = cursor.calls[-1]
    statement = _sql_text(statement)
    assert statement.count("%s") == len(params)
    assert "UNION ALL" in statement
    compact = " ".join(statement.split())
    assert "f.trade_date >= %s" in compact
    assert "f.trade_date <= %s" in compact
    assert "o.available_at < %s" in compact
    assert "ELSE f.remark" not in compact
    assert params == [
        time(15, 0, 1),
        "收盘价(元)",
        "收盘价(元)",
        time(15, 0, 1),
        time(15, 0, 1),
        "stock",
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 1, 31),
        "收盘价(元)",
        4,
        "stock",
        ["000001.SZ"],
        datetime(2024, 1, 2),
        datetime(2024, 2, 1),
        25,
    ]
    assert result.empty


def test_observation_time_range_resolves_alias_then_binds_metric_id():
    resolved = {
        "metric_id": 17,
        "metric_name": "净资产收益率",
        "storage_kind": "observation",
        "storage_column": None,
        "availability_time": time(15, 0, 1),
    }
    cursor = RecordingCursor(fetchone=[resolved], fetchall=[[]])

    query._query_time_range_normalized(
        cursor,
        "fundamentals",
        ["000001.SZ"],
        "2023-01-01",
        "2023-12-31",
        "ROE",
        None,
    )

    statement, params = cursor.calls[-1]
    statement = _sql_text(statement)
    assert statement.count("%s") == len(params)
    assert "o.metric_id = %s" in statement
    assert params == [
        "ROE",
        17,
        "stock",
        ["000001.SZ"],
        datetime(2023, 1, 1),
        datetime(2024, 1, 1),
    ]


def test_nearest_market_uses_unnest_lateral_and_preserves_input_order():
    resolved = {
        "metric_id": 4,
        "metric_name": "收盘价(元)",
        "storage_kind": "core",
        "storage_column": "close",
        "availability_time": time(15, 0, 1),
    }
    row = {
        "code": "000001.SZ",
        "input_ts": datetime(2024, 1, 2, 10),
        "datetime": datetime(2024, 1, 2, 15, 0, 1),
        "diff_hours": 5.0003,
        "value": 10.5,
        "name": "平安银行",
    }
    cursor = RecordingCursor(fetchone=[resolved], fetchall=[[row]])

    result = query._query_nearest_normalized(
        cursor,
        "daily_market",
        ["000001.SZ", "000002.SZ"],
        ["2024-01-02 10:00:00", "2024-01-03 10:00:00"],
        "收盘价(元)",
        direction="after",
        time_tolerance=24,
    )

    statement, params = cursor.calls[-1]
    statement = " ".join(_sql_text(statement).split())
    assert statement.count("%s") == len(params)
    assert "unnest(%s::text[]) WITH ORDINALITY" in statement
    assert "CROSS JOIN unnest(%s::timestamp[]) WITH ORDINALITY" in statement
    assert "LEFT JOIN LATERAL" in statement
    assert "x.available_at > r.input_ts" in statement
    assert "EXTRACT(EPOCH FROM (hit.available_at - r.input_ts))" in statement
    assert "ORDER BY x.available_at ASC" in statement
    assert "ORDER BY r.input_ord" in statement
    datetimes = [datetime(2024, 1, 2, 10), datetime(2024, 1, 3, 10)]
    assert params == [
        datetimes,
        ["000001.SZ", "000002.SZ"],
        datetimes,
        "stock",
        time(15, 0, 1),
        4,
        24.0,
    ]
    assert "收盘价(元)" in result.columns
    assert "value" not in result.columns


def test_nearest_observation_range_before_parameter_order_and_bounds():
    resolved = {
        "metric_id": 23,
        "metric_name": "ROE",
        "storage_kind": "observation",
        "storage_column": None,
        "availability_time": time(15, 0, 1),
    }
    cursor = RecordingCursor(fetchone=[resolved], fetchall=[[]])

    query._query_nearest_normalized(
        cursor,
        "fundamentals",
        ["000001.SZ"],
        [("2023-01-01", "2023-12-31")],
        "ROE",
        direction="before",
        time_tolerance=48,
        ranges=True,
    )

    statement, params = cursor.calls[-1]
    statement = " ".join(_sql_text(statement).split())
    assert statement.count("%s") == len(params)
    assert "x.available_at >= r.start_ts" in statement
    assert "x.available_at <= r.end_ts" in statement
    assert "EXTRACT(EPOCH FROM (r.end_ts - hit.available_at))" in statement
    assert "ORDER BY x.available_at DESC" in statement
    starts = [datetime(2023, 1, 1)]
    ends = [datetime(2023, 12, 31)]
    assert params == [
        starts,
        ends,
        starts,
        ["000001.SZ"],
        "stock",
        "stock",
        23,
        48.0,
    ]


def test_trade_status_uses_event_date_and_warns_beyond_coverage(caplog):
    cursor = RecordingCursor(
        fetchone=[{"max_available_at": datetime(2020, 12, 31)}],
        fetchall=[[]],
    )
    logger = logging.getLogger("test.datafeed.trade_status")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        result = query._query_trade_status_normalized(
            cursor,
            ["000001.SZ"],
            ["2021-01-04", "2021-01-04 15:00:00"],
            logger,
        )

    statement, params = cursor.calls[-1]
    statement = " ".join(_sql_text(statement).split())
    assert "s.event_date = d.status_date" in statement
    assert params == [["000001.SZ"], [datetime(2021, 1, 4)]]
    assert list(result.columns) == ["code", "datetime", "value", "status_text", "name"]
    assert "覆盖不足" in caplog.text


def test_industry_normalized_uses_arrays_and_exclusive_valid_to():
    cursor = RecordingCursor(fetchall=[[]])
    logger = logging.getLogger("test.datafeed.industry")

    industry._query_industry_normalized(
        cursor,
        ["000001.SZ", "000002.SZ"],
        ["2024-01-31"],
        "申万一级行业",
        exact=False,
        logger=logger,
    )

    statement, params = cursor.calls[-1]
    statement = " ".join(_sql_text(statement).split())
    assert "LEFT JOIN LATERAL" in statement
    assert "im.valid_from <= r.query_date" in statement
    assert "im.valid_to IS NULL OR im.valid_to > r.query_date" in statement
    dates = [datetime(2024, 1, 31)]
    assert params == (
        dates,
        ["000001.SZ", "000002.SZ"],
        dates,
        "申万一级行业%",
    )


def test_industry_normalized_prefix_escaping_matches_legacy_route():
    cursor = RecordingCursor(fetchall=[[]])
    logger = logging.getLogger("test.datafeed.industry.escape")
    scheme = r"申万\一级_%"

    industry._query_industry_normalized(
        cursor,
        ["000001.SZ"],
        ["2024-01-31"],
        scheme,
        exact=False,
        logger=logger,
    )

    _, legacy_pattern = industry._scheme_clause(scheme, exact=False)
    assert cursor.calls[-1][1][-1] == legacy_pattern


def test_index_universe_normalized_uses_latest_pit_snapshot(monkeypatch):
    cursor = RecordingCursor(fetchall=[[{"code": "000001.SZ"}, {"code": "000002.SZ"}]])
    monkeypatch.setattr(universe, "_normalized_schema_available", lambda _: True)

    result = universe.get_index_universe(cursor, "000300.SH", "2024-01-31")

    statement, params = cursor.calls[-1]
    statement = " ".join(_sql_text(statement).split())
    assert "MAX(s2.effective_at)" in statement
    assert "s2.effective_at <= %s::timestamp" in statement
    assert "ORDER BY COALESCE(c.ordinal, 2147483647), member.code" in statement
    assert params == ("000300.SH", "2024-01-31")
    assert result == ["000001.SZ", "000002.SZ"]


def test_index_universe_panel_normalized_batches_dates(monkeypatch):
    cursor = RecordingCursor(
        fetchall=[[
            {"query_date": datetime(2024, 1, 31), "code": "000001.SZ"},
            {"query_date": datetime(2024, 1, 31), "code": "000002.SZ"},
            {"query_date": datetime(2024, 2, 1), "code": "000002.SZ"},
        ]]
    )
    monkeypatch.setattr(universe, "_normalized_schema_available", lambda _: True)

    result = universe.get_index_universe_panel(
        cursor,
        "000300.SH",
        ["2024-01-31", "2024-02-01", "2024-02-01 15:00:01"],
    )

    statement, params = cursor.calls[-1]
    statement = " ".join(_sql_text(statement).split())
    assert "unnest(%s::timestamp[]) WITH ORDINALITY" in statement
    assert "LEFT JOIN LATERAL" in statement
    assert len(cursor.calls) == 1
    assert params == (
        [datetime(2024, 1, 31), datetime(2024, 2, 1)],
        "000300.SH",
    )
    assert result == {
        date(2024, 1, 31): {"000001.SZ", "000002.SZ"},
        date(2024, 2, 1): {"000002.SZ"},
    }


def test_index_universe_panel_legacy_falls_back_per_unique_date(monkeypatch):
    cursor = RecordingCursor()
    calls = []
    monkeypatch.setattr(universe, "_normalized_schema_available", lambda _: False)

    def fake_get_index_universe(_cursor, index_code, date, **kwargs):
        calls.append((index_code, date, kwargs["table_name"], kwargs["metric"]))
        return [f"member-{date}"]

    monkeypatch.setattr(universe, "get_index_universe", fake_get_index_universe)

    result = universe.get_index_universe_panel(
        cursor,
        "000300.SH",
        ["2024-01-31", "2024-01-31 15:00:01", "2024-02-01"],
        table_name="legacy_universe",
        metric="members",
    )

    assert calls == [
        ("000300.SH", "2024-01-31", "legacy_universe", "members"),
        ("000300.SH", "2024-02-01", "legacy_universe", "members"),
    ]
    assert result[date(2024, 1, 31)] == {"member-2024-01-31"}
    assert result[date(2024, 2, 1)] == {"member-2024-02-01"}


@pytest.mark.parametrize(
    ("index_code", "query_date", "message"),
    [("", "2024-01-31", "index_code不能为空"), ("000300.SH", "", "date不能为空")],
)
def test_index_universe_keeps_legacy_input_validation(
    monkeypatch, index_code, query_date, message
):
    cursor = RecordingCursor(fetchall=[[]])
    monkeypatch.setattr(universe, "_normalized_schema_available", lambda _: True)

    with pytest.raises(ValueError, match=message):
        universe.get_index_universe(cursor, index_code, query_date)


class _SessionCursor:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def execute(self, statement, params=None):
        self.calls.append((statement, params))


class _Connection:
    closed = 0

    def __init__(self):
        self.rollbacks = 0
        self.sessions = []
        self.session_cursor = _SessionCursor()

    def rollback(self):
        self.rollbacks += 1

    def set_session(self, **kwargs):
        self.sessions.append(kwargs)

    def cursor(self):
        return self.session_cursor


class _RawPool:
    def __init__(self, *args, **kwargs):
        self.connection = _Connection()
        self.returned = []

    def getconn(self):
        return self.connection

    def putconn(self, connection, close=False):
        self.returned.append((connection, close))

    def closeall(self):
        return None


def test_read_pool_sets_session_readonly_and_releases_connection(monkeypatch):
    monkeypatch.setattr(pool, "ThreadedConnectionPool", _RawPool)
    monkeypatch.setattr(
        pool,
        "get_database_config",
        lambda: {
            "dbname": "unit_test",
            "user": "postgres",
            "password": "",
            "host": "localhost",
            "port": "5432",
        },
    )
    read_pool = pool.ReadOnlyConnectionPool(statement_timeout_ms=4567)

    connection = read_pool.acquire()
    read_pool.release(connection)

    assert connection.sessions == [{"readonly": True, "autocommit": True}]
    assert connection.session_cursor.calls == [
        (
            "SELECT set_config('statement_timeout', %s, false)",
            ("4567",),
        )
    ]
    assert connection.rollbacks == 2
    assert read_pool._pool.returned == [(connection, False)]
