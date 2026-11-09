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

from betalens.datafeed import Datafeed, get_absolute_trade_days, trade_periods_offset, PITUnavailableError
from betalens.datafeed.contracts import json_value
from betalens.datafeed.pool import close_all_pools
from datafeed_support import market_reader, reference_reader, calendar_scope, industry_scope, quote, NAME_TYPES, FIN_TYPES, evidence


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
    if table == "industry_scheme_dim": return ["scheme_code"]
    if table == "industry_dim": return ["scheme_code", "industry_code"]
    if table == "security_industry_membership": return ["scheme_code", "security_code", "industry_code", "valid_from"]
    if table == "trade_calendar": return ["exchange", "trade_date"]
    if table == "trade_status_daily": return ["asset_type", "security_code", "trade_date"]
    if table == "index_membership": return ["index_code", "security_code", "trade_date"]
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
    scopes = [industry_scope('SW2014', level, '2020-01-01', '2024-01-03', '2024-01-01') for level in (1, 2, 3)]
    scopes += [industry_scope('SW2021', level) for level in (1, 2, 3)]
    industries = reference_reader('industry', [('industry', scopes, 'publish')])
    memory.types.update(industries.types)
    for table, rows in industries.tables.items():
        memory.tables.setdefault(table, []).extend(rows)
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
        with Datafeed("market_daily_post_adjusted", config, backend=backend, pit_policy="strict") as data:
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
        assert get_absolute_trade_days("2024-01-01", "2024-01-04", "D", backend=backend, db_config=config, as_of="2024-01-04", table_name="trade_calendar_versions", pit_policy="strict") == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
        assert trade_periods_offset("2024-01-02 10:00", 1, backend=backend, db_config=config, as_of="2024-01-04", table_name="trade_calendar_versions", pit_policy="strict") == pd.Timestamp("2024-01-03 10:00")
        with pytest.raises(PITUnavailableError):
            get_absolute_trade_days("2024-01-01", "2024-01-05", "D", backend=backend, db_config=config, table_name="trade_calendar_versions", pit_policy="strict")


def test_live_financial_revision_and_raw_adjustment(live):
    configs, _, _ = live
    results = {}
    for backend, config in configs.items():
        with Datafeed("fundamental_periodic", config, backend=backend, pit_policy="strict") as data:
            results[backend] = data.query_nearest_before({"codes": ["A"], "datetimes": ["2024-05-01", "2024-07-01"], "metric": "revenue", "filters": {"statement_type": "income", "report_type": "1"}})
        with Datafeed("market_daily_raw", config, backend=backend, pit_policy="strict") as data:
            assert data.query_time_range(metric="close").value.tolist() == [10, 5]
    pd.testing.assert_frame_equal(results["postgres"], results["clickhouse"])
    assert results["postgres"]["revenue"].tolist() == [100, 200]


def test_live_pg_connection_recovery_readonly_and_subscription(live):
    configs, _, _ = live
    config = configs["postgres"]
    from betalens.datafeed.query import query_nearest_before
    with Datafeed("market_daily_post_adjusted", config, pit_policy="strict") as data:
        with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
            data._reader.cursor.execute(f'CREATE TABLE "{config["schema"]}".forbidden (x integer)')
        data._reader.conn.rollback()
        got = query_nearest_before(data, "market_daily_post_adjusted", ["A"], ["2024-01-03 15:00"], "close")
        assert got["close"].iloc[0] == 20
    with Datafeed("market_daily_post_adjusted", config, pit_policy="strict") as again:
        assert again.get_latest_date(metric="close") == pd.Timestamp("2024-01-03 15:00")


@pytest.mark.parametrize("direction,ranges", [("before", False), ("after", False), ("before", True), ("after", True)])
def test_live_sql_candidate_reduction_matches_python(live, direction, ranges):
    from betalens.datafeed.query import _nearest
    configs, _, _ = live
    anchors = [("2024-01-01", "2024-01-03 15:00"), ("2024-01-02 15:00", "2024-01-03 15:00")] if ranges else ["2024-01-02 15:00", "2024-01-04"]
    for backend, config in configs.items():
        with Datafeed("market_daily_post_adjusted", config, backend=backend, pit_policy="strict") as data:
            expected = _nearest(data, "market_daily_post_adjusted", ["A", "missing", "A"], anchors, "close", direction, 48, ranges=ranges, engine="python")
            actual = _nearest(data, "market_daily_post_adjusted", ["A", "missing", "A"], anchors, "close", direction, 48, ranges=ranges, engine="sql")
            pd.testing.assert_frame_equal(actual, expected)


def test_live_selected_default_path_and_missing_candidates(live):
    from betalens.datafeed.query import _nearest
    configs, _, _ = live
    codes = ["A"] + [f"missing{i}" for i in range(199)]
    for backend, config in configs.items():
        with Datafeed("market_daily_post_adjusted", config, backend=backend, pit_policy="strict") as data:
            expected = _nearest(data, "market_daily_post_adjusted", codes, ["2024-01-03 15:00"], "close", "before", None, engine="python")
            actual = data.query_nearest_before({"codes": codes, "datetimes": ["2024-01-03 15:00"], "metric": "close"})
            pd.testing.assert_frame_equal(actual, expected)


def test_live_raw_cursor_is_rejected(live):
    from betalens.datafeed.query import query_nearest_before
    configs, _, _ = live
    with Datafeed("market_daily_post_adjusted", configs["postgres"], pit_policy="strict") as data:
        with pytest.raises(TypeError, match="游标"):
            query_nearest_before(data._reader.cursor, "market_daily_post_adjusted", ["A"], ["2024-01-03"], "close")
        assert not hasattr(data, "cursor") and not hasattr(data, "run_query")


def test_live_clickhouse_duplicate_parts_are_deduplicated(live):
    configs, _, session = live
    config = configs["clickhouse"]
    relation = '`' + config['database'] + '`.`market_daily_post_adjusted`'
    ch_command(session, config, f"INSERT INTO {relation} SELECT * FROM {relation} FINAL WHERE security_code='A' AND trade_date='2024-01-02'")
    with Datafeed("market_daily_post_adjusted", config, backend="clickhouse", pit_policy="strict") as data:
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
        with Datafeed(table, config, backend=backend, pit_policy="strict") as data:
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
        with pytest.raises(ValueError, match="exchange"):
            get_absolute_trade_days("2024-01-02", "2024-01-02", "D", exchange="SHSE",
                                    backend=backend, db_config=config, table_name="trade_calendar_versions", pit_policy="strict")
        with pytest.raises(ValueError, match="exchange"):
            trade_periods_offset("2024-01-02", 1, exchange="SHSE", backend=backend, db_config=config, table_name="trade_calendar_versions", pit_policy="strict")


def test_live_shanghai_filters_numeric_parameters_and_industry_versions(live):
    from datetime import datetime
    from decimal import Decimal
    configs, _, _ = live
    frames = {}
    for backend, config in configs.items():
        frames[backend] = []
        with Datafeed('market_daily_post_adjusted', config, backend=backend, pit_policy="strict") as data:
            for at in (datetime(2024, 1, 2, 15), '2024-01-02T07:00:00Z', '2024-01-02T15:00:00+08:00'):
                result = data.query_time_range(metric='close', filters={'available_at': at, 'close': Decimal('10')})
                assert result.value.tolist() == [10]
                frames[backend].append(result)
        with Datafeed('stock_industry_membership_versions', config, backend=backend, pit_policy="strict") as data:
            for level in (1, 2, 3):
                result = data.query_industry(['A'], ['2024-01-02', '2024-01-04'], level=level)
                assert result.ind_name.tolist() == [f'SW2014-L{level}', f'SW2021-L{level}']
                frames[backend].append(result)
    for pg, ch in zip(frames['postgres'], frames['clickhouse']):
        pd.testing.assert_frame_equal(pg, ch, check_exact=True)


def test_live_research_industry_calendar_risks_and_index_contracts():
    from test_datafeed_research import industry_reader, status_reader
    from datafeed_support import evidence
    memory = industry_reader()
    status = status_reader()
    memory.tables.update(status.tables)
    memory.types.update(status.types)
    memory.types["trade_calendar"] = {"exchange": "text", "trade_date": "date", "is_open": "integer"}
    memory.tables["trade_calendar"] = [{"exchange": "SSE", "trade_date": date(2024,1,d), "is_open": 1} for d in [2,3]]
    memory.types["index_membership"] = {"index_code": "text", "security_code": "text", "trade_date": "date",
        "snapshot_complete": "integer", "snapshot_member_count": "integer", "available_at": "timestamp",
        "availability_basis": "text", "data_quality_status": "text"}
    memory.tables["index_membership"] = [{"index_code":"I","security_code":"A","trade_date":date(2024,1,2),
        "snapshot_complete":1,"snapshot_member_count":1,**evidence("2024-01-03",availability_basis="next_calendar_day_assumed")}]
    outputs = []
    with isolated_databases(memory.tables,memory.types) as (configs,conn,session):
        for table in ['industry_dim','industry_scheme_dim','security_industry_membership','trade_calendar','trade_status_daily','suspension_event','index_membership','security_dim']:
            with conn.cursor() as cur:
                cur.execute(f'ALTER TABLE "{configs["postgres"]["schema"]}"."{table}" ADD COLUMN unrelated text[]')
            ch_command(session,configs['clickhouse'],f'ALTER TABLE `{configs["clickhouse"]["database"]}`.`{table}` ADD COLUMN unrelated Array(String)')
        for backend,config in configs.items():
            frames=[]
            with Datafeed("security_industry_membership",config,backend=backend) as data:
                for level in [1,2,3]:
                    frames.append(data.query_industry(["A","X","A"],["2021-07-31","2021-08-01"],level=level))
            with Datafeed("trade_status_daily",config,backend=backend) as data:
                frames.append(data.query_trade_risks(["A","B"],["2024-01-02 09:30","2024-01-02 10:00"]))
                frames.append(data.query_trade_status({"codes":["A","B"],"dates":["2024-01-02 09:30"]}))
            with Datafeed("market_daily_post_adjusted",config,backend=backend) as data:
                frames.append(data.query_nearest_before({"codes":["A"],"datetimes":["2024-01-03"],"metric":"close"}))
                from betalens.datafeed.query import _nearest
                for direction in ["before","after"]:
                    args=(data,"market_daily_post_adjusted",["A","X","A"],["2024-01-02","2024-01-03"],"close",direction,None)
                    pd.testing.assert_frame_equal(_nearest(*args,engine="sql"),_nearest(*args,engine="python"))
            with Datafeed("index_membership",config,backend=backend) as data:
                assert data.get_index_universe("I","2024-01-03") == ["A"]
            assert get_absolute_trade_days("2024-01-02","2024-01-03","D",db_config=config,backend=backend,as_of="2020-01-01") == [date(2024,1,2),date(2024,1,3)]
            assert trade_periods_offset("2024-01-02 10:00",1,db_config=config,backend=backend) == pd.Timestamp("2024-01-03 10:00")
            outputs.append(frames)
    for pg,ch in zip(*outputs):
        pd.testing.assert_frame_equal(pg,ch)
