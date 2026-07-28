from __future__ import annotations

import os

import pandas as pd
import pytest

from betalens.datafeed import Datafeed
from betalens.datafeed.config import get_database_config
from betalens_db_manager import (
    DatabaseClient,
    DatabaseWriter,
    DeleteRequest,
    QueryRequest,
    SchemaManager,
)
from betalens_db_manager.__main__ import _run_manifest


TEST_DATABASE_ENV = "BETALENS_TEST_DB_NAME"


@pytest.fixture(scope="module")
def db_config():
    dbname = os.environ.get(TEST_DATABASE_ENV)
    if not dbname:
        pytest.skip(f"set {TEST_DATABASE_ENV} to run PostgreSQL integration tests")
    config = dict(get_database_config())
    config["dbname"] = dbname
    return config


@pytest.fixture(scope="module")
def services(db_config):
    client = DatabaseClient(db_config)
    writer = DatabaseWriter(client)
    assert SchemaManager(db_config).verify_schema()["ok"]
    return client, writer


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["datetime", "code", "name", "metric", "value", "remark"],
    )


def _walk_plan(node):
    yield node
    for child in node.get("Plans", []):
        yield from _walk_plan(child)


def test_market_copy_upsert_read_and_conditional_delete(services, db_config):
    client, writer = services
    code = "TST0001.SZ"
    writer.delete(DeleteRequest("daily_market", code=code))

    frame = _frame(
        [
            ("2024-01-02 15:00:01", code, "测试股票", "收盘价(元)", 10.0, {"source": "test"}),
            ("2024-01-02 15:00:01", code, "测试股票", "成交额(元)", 1000.0, {"source": "test"}),
            ("2024-01-03 12:00:00", code, "测试股票", "收盘价(元)", 11.0, {"source": "test"}),
            ("2024-01-04 15:00:01", code, "测试股票新名", "收盘价(元)", 12.0, {"source": "test"}),
        ]
    )

    assert writer.write("daily_market", frame, mode="insert_only") == {
        "inserted": 4,
        "updated": 0,
        "skipped": 0,
        "total": 4,
    }
    assert writer.write("daily_market", frame, mode="insert_only") == {
        "inserted": 0,
        "updated": 0,
        "skipped": 4,
        "total": 4,
    }

    changed = frame.iloc[[0]].copy()
    changed.loc[:, "value"] = 10.5
    assert writer.write("daily_market", changed, mode="upsert") == {
        "inserted": 0,
        "updated": 1,
        "skipped": 0,
        "total": 1,
    }

    with Datafeed("daily_market", db_config=db_config) as feed:
        result = feed.query_time_range(
            codes=[code],
            start_date="2024-01-01",
            end_date="2024-01-31",
            metric="收盘价(元)",
        )
        executed_sql = feed._cursor.query.decode("utf-8")
        feed._cursor.execute("SET enable_seqscan = off")
        feed._cursor.execute("EXPLAIN (FORMAT JSON) " + executed_sql)
        market_plan = str(feed._cursor.fetchone())
    assert "market_daily_fact_pkey" in market_plan
    assert result[["datetime", "name", "value"]].to_dict("records") == [
        {"datetime": pd.Timestamp("2024-01-04 15:00:01"), "name": "测试股票新名", "value": 12.0},
        {"datetime": pd.Timestamp("2024-01-03 12:00:00"), "name": "测试股票", "value": 11.0},
        {"datetime": pd.Timestamp("2024-01-02 15:00:01"), "name": "测试股票", "value": 10.5},
    ]

    manager_rows = client.query_table(
        QueryRequest("daily_market", code=code, metric="成交金额(元)", limit=10)
    )
    assert manager_rows["value"].tolist() == [1000.0]

    deleted = writer.delete(
        DeleteRequest(
            "daily_market",
            code=code,
            metric="收盘价(元)",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
    )
    assert deleted["deleted"] == 3
    remaining = client.query_table(QueryRequest("daily_market", code=code, limit=10))
    assert remaining["metric"].tolist() == ["成交金额(元)"]
    writer.delete(DeleteRequest("daily_market", code=code))


def test_observation_partition_and_pit_datasets(services, db_config):
    client, writer = services
    stock_code = "TST0002.SZ"
    index_code = "TSTINDEX.SH"
    for table, code in (
        ("fundamentals", stock_code),
        ("industry", stock_code),
        ("trade_status", stock_code),
        ("index_universe", index_code),
    ):
        writer.delete(DeleteRequest(table, code=code))

    fundamental = _frame(
        [
            (
                "2023-04-30 15:00:01",
                stock_code,
                "测试股票二",
                "测试ROE",
                12.3,
                {"period_end": "2022-12-31"},
            )
        ]
    )
    assert writer.write("fundamentals", fundamental)["inserted"] == 1

    industry = _frame(
        [
            (
                "2024-01-01",
                stock_code,
                "测试股票二",
                "申万一级行业",
                801780.0,
                {"ind_name": "测试行业", "ind_code": "801780.SI"},
            )
        ]
    )
    assert writer.write("industry", industry)["inserted"] == 1

    universe = _frame(
        [
            (
                "2024-01-01",
                index_code,
                "测试指数",
                "universe",
                1.0,
                {"constituents": [stock_code]},
            )
        ]
    )
    assert writer.write("index_universe", universe)["inserted"] == 1

    statuses = _frame(
        [
            ("2024-01-01", stock_code, "测试股票二", "交易状态", 1.0, {"status": "交易"}),
            ("2024-01-03", stock_code, "测试股票二", "交易状态", 0.0, {"status": "停牌"}),
        ]
    )
    assert writer.write("trade_status", statuses)["inserted"] == 2

    with Datafeed("fundamentals", db_config=db_config) as feed:
        nearest = feed.query_nearest_before(
            {
                "codes": [stock_code],
                "datetimes": ["2023-05-01"],
                "metric": "测试ROE",
            }
        )
        executed_sql = feed._cursor.query.decode("utf-8")
        feed._cursor.execute("SET enable_seqscan = off")
        feed._cursor.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + executed_sql)
        observation_plan = feed._cursor.fetchone()["QUERY PLAN"][0]["Plan"]
    assert nearest.iloc[0]["测试ROE"] == 12.3
    observation_nodes = list(_walk_plan(observation_plan))
    assert any(
        node.get("Relation Name") == "observation_fact_2023"
        and node.get("Actual Loops", 0) > 0
        for node in observation_nodes
    )
    assert all(
        node.get("Actual Loops", 0) == 0
        for node in observation_nodes
        if node.get("Relation Name", "").startswith("observation_fact_2024")
    )

    with Datafeed("industry", db_config=db_config) as feed:
        membership = feed.query_industry([stock_code], "2024-02-01", "申万一级行业")
    assert membership.iloc[0]["ind_name"] == "测试行业"

    with Datafeed("index_universe", db_config=db_config) as feed:
        assert feed.get_index_universe(index_code, "2024-02-01") == [stock_code]

    with Datafeed("trade_status", db_config=db_config) as feed:
        status = feed.query_trade_status(
            {"codes": [stock_code], "dates": ["2023-12-31", "2024-01-02", "2024-01-03"]}
        )
    assert status["value"].tolist() == [-1, 1, 0]

    with client.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('betalens.observation_fact_2023')")
        assert cur.fetchone()[0] is not None

    for table, code in (
        ("fundamentals", stock_code),
        ("industry", stock_code),
        ("trade_status", stock_code),
        ("index_universe", index_code),
    ):
        writer.delete(DeleteRequest(table, code=code))


def test_bootstrap_manifest_runs_multiple_file_imports(services, db_config, tmp_path):
    client, writer = services
    code = "TSTMANIFEST.SZ"
    writer.delete(DeleteRequest("daily_market", code=code))
    writer.delete(DeleteRequest("fundamentals", code=code))

    market_path = tmp_path / "market.csv"
    fundamental_path = tmp_path / "fundamental.csv"
    _frame(
        [
            (
                "2024-02-01 15:00:01",
                code,
                "清单测试股票",
                "收盘价(元)",
                8.8,
                None,
            )
        ]
    ).to_csv(market_path, index=False)
    _frame(
        [
            (
                "2024-02-01 15:00:01",
                code,
                "清单测试股票",
                "清单测试指标",
                6.6,
                None,
            )
        ]
    ).to_csv(fundamental_path, index=False)
    manifest_path = tmp_path / "imports.yaml"
    manifest_path.write_text(
        """
imports:
  - path: market.csv
    table: daily_market
    import_type: wind_long
  - path: fundamental.csv
    table: fundamentals
    import_type: wind_long
""".lstrip(),
        encoding="utf-8",
    )

    result = _run_manifest(manifest_path, SchemaManager(db_config))

    assert result["status"] == "completed"
    assert len(result["imports"]) == 2
    assert all(row["status"] == "completed" for row in result["imports"])
    assert len(client.query_table(QueryRequest("daily_market", code=code))) == 1
    assert len(client.query_table(QueryRequest("fundamentals", code=code))) == 1

    writer.delete(DeleteRequest("daily_market", code=code))
    writer.delete(DeleteRequest("fundamentals", code=code))
