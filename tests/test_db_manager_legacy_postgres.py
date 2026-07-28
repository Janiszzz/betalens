from __future__ import annotations

import os
from datetime import date

import pytest
from psycopg2.extras import Json

from betalens.datafeed import Datafeed
from betalens.datafeed.config import get_database_config
from betalens_db_manager import SchemaManager


LEGACY_DATABASE_ENV = "BETALENS_TEST_LEGACY_DB_NAME"


@pytest.fixture(scope="module")
def legacy_config():
    dbname = os.environ.get(LEGACY_DATABASE_ENV)
    if not dbname:
        pytest.skip(f"set {LEGACY_DATABASE_ENV} to run legacy migration tests")
    config = dict(get_database_config())
    config["dbname"] = dbname
    manager = SchemaManager(config)
    if not manager.database_exists():
        manager.create_database()
    with manager.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('betalens.schema_migration')")
        if cur.fetchone()[0] is not None:
            pytest.fail(f"legacy test database {dbname} was already bootstrapped")
        for table in ("industry", "index_universe", "trade_status"):
            cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
            if cur.fetchone()[0] is not None:
                pytest.fail(f"legacy test database already contains public.{table}")
    return config


def _create_legacy_tables(config):
    manager = SchemaManager(config)
    with manager.connect() as conn, conn.cursor() as cur:
        for table in ("industry", "index_universe", "trade_status"):
            cur.execute(
                f"""
                CREATE TABLE public.{table} (
                    datetime TIMESTAMP NOT NULL,
                    code VARCHAR(32) NOT NULL,
                    name VARCHAR(200),
                    metric VARCHAR(160) NOT NULL,
                    value DOUBLE PRECISION,
                    remark JSONB,
                    PRIMARY KEY (datetime, code, metric)
                )
                """
            )
        cur.executemany(
            """
            INSERT INTO public.industry
                (datetime, code, name, metric, value, remark)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    "2024-01-01",
                    "LGCY0001.SZ",
                    "旧股票一",
                    "申万一级行业",
                    801010.0,
                    Json({"ind_name": "农业", "ind_code": "801010.SI", "scheme": "申万一级行业"}),
                ),
                (
                    "2024-06-01",
                    "LGCY0001.SZ",
                    "旧股票一新名",
                    "申万一级行业",
                    801020.0,
                    Json({"ind_name": "采掘", "ind_code": "801020.SI", "scheme": "申万一级行业"}),
                ),
            ],
        )
        cur.execute(
            """
            INSERT INTO public.index_universe
                (datetime, code, name, metric, value, remark)
            VALUES (%s, %s, %s, 'universe', 2, %s)
            """,
            (
                "2024-01-01",
                "LGCYINDEX.SH",
                "旧测试指数",
                Json({"constituents": ["LGCY0001.SZ", "LGCY0002.SZ"]}),
            ),
        )
        cur.executemany(
            """
            INSERT INTO public.trade_status
                (datetime, code, name, metric, value, remark)
            VALUES (%s, %s, %s, '交易状态', %s, %s)
            """,
            [
                ("2024-01-01", "LGCY0001.SZ", "旧股票一", 1.0, Json({"status": "交易"})),
                ("2024-01-01", "LGCY0002.SZ", "旧股票二", 1.0, Json({"status": "交易"})),
                ("2024-01-03", "LGCY0001.SZ", "旧股票一", 0.0, Json({"status": "停牌"})),
            ],
        )


def test_three_legacy_tables_migrate_with_pit_contracts(legacy_config, tmp_path):
    _create_legacy_tables(legacy_config)
    manager = SchemaManager(legacy_config)

    report = manager.bootstrap_local(report_path=tmp_path / "legacy-bootstrap.json")

    assert report["status"] == "completed"
    assert set(report["legacy_tables_preserved"]) == {
        "industry",
        "index_universe",
        "trade_status",
    }
    assert set(report["created_partitions"]) == {
        f"observation_fact_{date.today().year}",
        f"observation_fact_{date.today().year + 1}",
    }
    assert report["verification"]["ok"]
    with manager.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM betalens.industry_membership")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT count(*) FROM betalens.index_snapshot")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM betalens.index_constituent")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT count(*) FROM betalens.trade_status_event")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM betalens.entity_dim WHERE first_trade_date IS NOT NULL")
        assert cur.fetchone()[0] == 2
        cur.execute("SELECT value FROM public.industry ORDER BY datetime DESC LIMIT 1")
        assert cur.fetchone()[0] == 801020.0
        for table in ("industry", "index_universe", "trade_status"):
            cur.execute("SELECT to_regclass(%s), to_regclass(%s)", (f"public.{table}", f"betalens_legacy.{table}"))
            public_relation, legacy_relation = cur.fetchone()
            assert public_relation is not None
            assert legacy_relation is not None

    with Datafeed("industry", db_config=legacy_config) as feed:
        membership = feed.query_industry(["LGCY0001.SZ"], "2024-07-01", "申万一级行业")
    assert membership.iloc[0]["ind_name"] == "采掘"
    assert membership.iloc[0]["sec_name"] == "旧股票一新名"

    with Datafeed("index_universe", db_config=legacy_config) as feed:
        assert feed.get_index_universe("LGCYINDEX.SH", "2024-07-01") == [
            "LGCY0001.SZ",
            "LGCY0002.SZ",
        ]

    with Datafeed("trade_status", db_config=legacy_config) as feed:
        statuses = feed.query_trade_status(
            {"codes": ["LGCY0001.SZ"], "dates": ["2023-12-31", "2024-01-02", "2024-01-03"]}
        )
    assert statuses["value"].tolist() == [-1, 1, 0]

    second = manager.bootstrap_local(report_path=tmp_path / "legacy-bootstrap-second.json")
    assert second["applied_migrations"] == []
    assert second["created_partitions"] == []
    assert second["verification"]["ok"]
