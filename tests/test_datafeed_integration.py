"""Opt-in real PG/CH tests. Only unique datafeed_test_* namespaces are written.

Set BETALENS_DATAFEED_TEST_CONFIG to a subscription JSON with both connections,
or BETALENS_DATAFEED_TEST_ENV to an existing manager env (requires dotenv).
Credentials are never copied to a repository file or printed.
"""
from contextlib import contextmanager
from datetime import date
import json
import os
from pathlib import Path
import uuid

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import pytest
import requests

from betalens.datafeed import Datafeed, get_absolute_trade_days, trade_days_offset, PITUnavailableError
from betalens.datafeed.contracts import json_value
from betalens.datafeed.pool import close_all_pools
from datafeed_support import market_reader, reference_reader, calendar_scope, quote, NAME_TYPES, FIN_TYPES, evidence


def live_config():
    path = os.environ.get("BETALENS_DATAFEED_TEST_CONFIG")
    if path:
        return json.loads(Path(path).read_text())
    path = os.environ.get("BETALENS_DATAFEED_TEST_ENV")
    if not path:
        pytest.skip("live DB tests require an explicit test connection configuration")
    from dotenv import dotenv_values
    env = dotenv_values(path, interpolate=False)
    return {"postgres": {"host": env.get("PGHOST", "127.0.0.1"), "port": env.get("PGPORT", 5432),
                          "dbname": env.get("PGDATABASE", "betalens"), "user": env.get("PGUSER", "betalens"),
                          "password": env.get("PGPASSWORD", "")},
            "clickhouse": {"url": env.get("CLICKHOUSE_URL", "http://127.0.0.1:8123"),
                           "database": env.get("CLICKHOUSE_DB", "betalens"), "user": env.get("CLICKHOUSE_USER", "default"),
                           "password": env.get("CLICKHOUSE_PASSWORD", "")}}


def ch_command(session, config, sql, data=""):
    response = session.post(config["url"], params={"date_time_input_format": "best_effort"},
                            data=(sql + "\n" + data).encode(), timeout=60)
    if not response.ok:
        # Do not include URL, authorization or row values in failures.
        raise RuntimeError("ClickHouse test SQL failed: " + response.text[:500])
    return response.text


def keys(table):
    if table.startswith("market_daily_"): return ["asset_type", "security_code", "trade_date"]
    if table == "fundamental_periodic": return ["security_code", "period_end", "statement_type", "report_type", "version_id"]
    if table == "reference_revision": return ["revision_id"]
    if table == "reference_scope": return ["revision_id", "scope_id"]
    if table == "reference_publication_event": return ["event_id"]
    if table.endswith("_versions"): return ["revision_id", "scope_id", "record_id"]
    if table == "security_name_history": return ["security_code", "start_date"]
    return ["security_code"]


@contextmanager
def isolated_databases(tables, types):
    config = live_config()
    namespace = "datafeed_test_" + uuid.uuid4().hex[:12]
    assert namespace.startswith("datafeed_test_") and namespace.isidentifier()
    pg_config = dict(config["postgres"], schema=namespace)
    ch_config = dict(config["clickhouse"], database=namespace)
    conn = psycopg2.connect(**{k: v for k, v in pg_config.items() if k in {"host", "port", "dbname", "user", "password"}}, connect_timeout=5)
    conn.autocommit = True
    session = requests.Session()
    session.auth = (ch_config["user"], ch_config.get("password", ""))
    pg_created = ch_created = False
    try:
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA "{namespace}"')
        pg_created = True
        ch_command(session, ch_config, f"CREATE DATABASE `{namespace}`")
        ch_created = True
        for table, fields in types.items():
            pk = keys(table)
            pg_types = {"text": "text", "integer": "bigint", "numeric": "numeric(38,12)", "timestamp": "timestamptz", "date": "date"}
            ch_types = {"text": "String", "integer": "Int64", "numeric": "Decimal(38,12)", "timestamp": "DateTime64(6,'UTC')", "date": "Date32"}
            pg_fields = ",".join('"' + k + '" ' + pg_types[v] for k, v in fields.items())
            ch_fields = ",".join('`' + k + '` ' + (ch_types[v] if k in pk else "Nullable(" + ch_types[v] + ")") for k, v in fields.items())
            with conn.cursor() as cur:
                cur.execute(f'CREATE TABLE "{namespace}"."{table}" ({pg_fields},PRIMARY KEY ({",".join(pk)}))')
                if tables[table]:
                    execute_values(cur, f'INSERT INTO "{namespace}"."{table}" ({",".join(fields)}) VALUES %s',
                                   [[int(r[k]) if fields[k] == "integer" and r.get(k) is not None else r.get(k)
                                     for k in fields] for r in tables[table]], page_size=5000)
            ch_command(session, ch_config, f"CREATE TABLE `{namespace}`.`{table}` ({ch_fields}) ENGINE=ReplacingMergeTree ORDER BY ({','.join(pk)})")
            if tables[table]:
                payload = "\n".join(json.dumps({k: r.get(k) for k in fields}, default=json_value, ensure_ascii=False) for r in tables[table])
                ch_command(session, ch_config, f"INSERT INTO `{namespace}`.`{table}` FORMAT JSONEachRow", payload)
        yield {"postgres": pg_config, "clickhouse": ch_config}, conn, session
    finally:
        close_all_pools()
        if pg_created:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA "{namespace}" CASCADE')
        conn.close()
        if ch_created:
            ch_command(session, ch_config, f"DROP DATABASE `{namespace}` SYNC")
        session.close()


@pytest.fixture(scope="module")
def live():
    memory = market_reader([quote(), quote(at="2024-01-03 15:00", value=20), quote(at="2024-01-04 15:00", value=99, availability_basis="next_calendar_day_assumed")])
    cal = reference_reader("calendar", [("calendar", [calendar_scope(day=d, opened=opened) for d, opened in
                [("2024-01-01", 0), ("2024-01-02", 1), ("2024-01-03", 1), ("2024-01-04", 1)]], "publish")])
    memory.tables.update(cal.tables)
    memory.types.update(cal.types)
    memory.types["security_name_history"] = NAME_TYPES
    memory.tables["security_name_history"] = [{"security_code": "A", "name": "Historical", "start_date": date(2020, 1, 1), "end_date": None, **evidence()}]
    memory.types["security_dim"] = {"security_code": "text", "name": "text", "asset_type": "text"}
    memory.tables["security_dim"] = [{"security_code": "A", "name": "Current", "asset_type": "stock"}]
    memory.types["fundamental_periodic"] = FIN_TYPES
    memory.tables["fundamental_periodic"] = [{"security_code": "A", "event_date": date(2023, 12, 31),
        "period_end": date(2023, 12, 31), "statement_type": "income", "report_type": "1", "version_id": version,
        "revenue": value, **evidence(at)} for version, at, value in [("z", "2024-04-01", 100), ("a", "2024-06-01", 200)]]
    with isolated_databases(memory.tables, memory.types) as live:
        yield live


def test_live_pg_ch_frames_and_precision_match(live):
    configs, _, _ = live
    results = {}
    for backend, config in configs.items():
        with Datafeed("market_daily_post_adjusted", config, backend=backend) as data:
            results[backend] = [data.query_time_range(["A"], "2024-01-01", "2024-01-04", "close"),
                data.query_nearest_before({"codes": ["A", "missing", "A"], "datetimes": ["2024-01-03 15:00", "2024-01-01"], "metric": "close"}),
                data.query_nearest_after({"codes": ["A"], "datetimes": ["2024-01-02 15:00"], "metric": "close"}),
                data.query_time_range([], metric="close"),
                data.query_names(["A"]), data.query_names(["A"], as_of="2024-01-03")]
    for pg, ch in zip(results["postgres"], results["clickhouse"]):
        pd.testing.assert_frame_equal(pg, ch, check_exact=True)
    assert results["postgres"][0].value.tolist() == [20, 10]
    assert results["postgres"][4].name.tolist() == ["Current"]
    assert results["postgres"][5].name.tolist() == ["Historical"]


def test_live_calendar_manifests_and_local_offset(live):
    configs, _, _ = live
    for backend, config in configs.items():
        assert get_absolute_trade_days("2024-01-01", "2024-01-04", "D", backend=backend, db_config=config, as_of="2024-01-04") == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
        assert trade_days_offset("2024-01-02 10:00", 1, backend=backend, db_config=config, as_of="2024-01-04") == pd.Timestamp("2024-01-03 10:00")
        with pytest.raises(PITUnavailableError):
            get_absolute_trade_days("2024-01-01", "2024-01-05", "D", backend=backend, db_config=config)


def test_live_financial_revision_and_raw_adjustment(live):
    configs, _, _ = live
    results = {}
    for backend, config in configs.items():
        with Datafeed("fundamental_periodic", config, backend=backend) as data:
            results[backend] = data.query_nearest_before({"codes": ["A"], "datetimes": ["2024-05-01", "2024-07-01"], "metric": "revenue", "filters": {"statement_type": "income", "report_type": "1"}})
        with Datafeed("market_daily_raw", config, backend=backend) as data:
            assert data.query_time_range(metric="close").value.tolist() == [10, 5]
    pd.testing.assert_frame_equal(results["postgres"], results["clickhouse"])
    assert results["postgres"]["revenue"].tolist() == [100, 200]


def test_live_pg_connection_recovery_readonly_and_legacy_cursor(live):
    configs, _, _ = live
    config = configs["postgres"]
    from betalens.datafeed.query import query_nearest_before
    with Datafeed("market_daily_post_adjusted", config) as data:
        with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
            data._reader.cursor.execute(f'CREATE TABLE "{config["schema"]}".forbidden (x integer)')
        data._reader.conn.rollback()
        with pytest.warns(DeprecationWarning):
            cur = data.cursor
        got = query_nearest_before(cur, "market_daily_post_adjusted", ["A"], ["2024-01-03 15:00"], "close")
        assert got["close"].iloc[0] == 20
    with Datafeed("market_daily_post_adjusted", config) as again:
        assert again.get_latest_date(metric="close") == pd.Timestamp("2024-01-03 15:00")


@pytest.mark.parametrize("direction,ranges", [("before", False), ("after", False), ("before", True), ("after", True)])
def test_live_sql_candidate_reduction_matches_python(live, direction, ranges):
    from betalens.datafeed.query import _nearest
    configs, _, _ = live
    anchors = [("2024-01-01", "2024-01-03 15:00"), ("2024-01-02 15:00", "2024-01-03 15:00")] if ranges else ["2024-01-02 15:00", "2024-01-04"]
    for backend, config in configs.items():
        with Datafeed("market_daily_post_adjusted", config, backend=backend) as data:
            expected = _nearest(data, "market_daily_post_adjusted", ["A", "missing", "A"], anchors, "close", direction, 48, ranges=ranges, engine="python")
            actual = _nearest(data, "market_daily_post_adjusted", ["A", "missing", "A"], anchors, "close", direction, 48, ranges=ranges, engine="sql")
            pd.testing.assert_frame_equal(actual, expected)


def test_live_selected_default_path_and_missing_candidates(live):
    from betalens.datafeed.query import _nearest
    configs, _, _ = live
    codes = ["A"] + [f"missing{i}" for i in range(199)]
    for backend, config in configs.items():
        with Datafeed("market_daily_post_adjusted", config, backend=backend) as data:
            expected = _nearest(data, "market_daily_post_adjusted", codes, ["2024-01-03 15:00"], "close", "before", None, engine="python")
            actual = data.query_nearest_before({"codes": codes, "datetimes": ["2024-01-03 15:00"], "metric": "close"})
            pd.testing.assert_frame_equal(actual, expected)


def test_live_borrowed_pg_cursor_does_not_commit_or_change_caller_settings(live):
    from betalens.datafeed.readers import PostgresReader
    from betalens.datafeed.query import query_nearest_before
    configs, _, _ = live
    config = configs["postgres"]
    with psycopg2.connect(**{k: v for k, v in config.items() if k in {"host", "port", "dbname", "user", "password"}}) as conn:
        conn.autocommit = True
        before = (conn.readonly, conn.autocommit, conn.isolation_level)
        with conn.cursor() as cur:
            reader = PostgresReader(cursor=cur, schema=config["schema"])
            result = query_nearest_before(reader, "market_daily_post_adjusted", ["A"], ["2024-01-03 15:00"], "close")
            assert result["close"].iloc[0] == 20
            assert before == (conn.readonly, conn.autocommit, conn.isolation_level)


def test_live_clickhouse_duplicate_parts_are_deduplicated(live):
    configs, _, session = live
    config = configs["clickhouse"]
    relation = '`' + config['database'] + '`.`market_daily_post_adjusted`'
    ch_command(session, config, f"INSERT INTO {relation} SELECT * FROM {relation} FINAL WHERE security_code='A' AND trade_date='2024-01-02'")
    with Datafeed("market_daily_post_adjusted", config, backend="clickhouse") as data:
        assert data.query_time_range(metric="close").value.tolist() == [20, 10]


@pytest.mark.parametrize("table,metric,filters", [
    ("market_daily_post_adjusted", "close", {"asset_type": "stock", "security_code": "missing"}),
    ("fundamental_periodic", "revenue", {"statement_type": "income", "report_type": "1"}),
    ("fundamental_periodic", "revenue", {"statement_type": "income", "report_type": "2"}),
])
def test_live_explicit_field_filters_match_sql_python_and_backends(live, table, metric, filters):
    from betalens.datafeed.query import _nearest
    configs, _, _ = live
    results = []
    for backend, config in configs.items():
        with Datafeed(table, config, backend=backend) as data:
            args = (data, table, ["A", "missing"], ["2024-07-01"], metric, "before", None)
            expected = _nearest(*args, filters=filters, engine="python")
            actual = _nearest(*args, filters=filters, engine="sql")
            pd.testing.assert_frame_equal(actual, expected)
            assert data.get_latest_date(metric=metric, filters=filters) == (
                pd.Timestamp("2024-06-01") if filters.get("report_type") == "1" else None)
            results.append(actual)
    pd.testing.assert_frame_equal(*results)


def test_live_calendar_rejects_shse_as_an_sse_alias(live):
    configs, _, _ = live
    for backend, config in configs.items():
        with pytest.raises(PITUnavailableError, match="SHSE"):
            get_absolute_trade_days("2024-01-02", "2024-01-02", "D", exchange="SHSE",
                                    backend=backend, db_config=config)
        with pytest.raises(IndexError):
            trade_days_offset("2024-01-02", 1, exchange="SHSE", backend=backend, db_config=config)
