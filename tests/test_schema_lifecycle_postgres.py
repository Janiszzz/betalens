from __future__ import annotations

import os

import pytest
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import Json

from betalens.datafeed.config import get_database_config
from betalens_db_manager import SchemaManager


TEST_DATABASE_ENV = "BETALENS_TEST_SCHEMA_LIFECYCLE_DB_NAME"


@pytest.fixture()
def lifecycle_config():
    dbname = os.environ.get(TEST_DATABASE_ENV)
    if not dbname:
        pytest.skip(f"set {TEST_DATABASE_ENV} to run schema lifecycle tests")
    config = dict(get_database_config())
    config["dbname"] = dbname
    manager = SchemaManager(config)
    if manager.database_exists():
        pytest.fail(f"schema lifecycle test database already exists: {dbname}")
    manager.create_database()
    try:
        yield config
    finally:
        admin = manager.connect("postgres")
        try:
            admin.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with admin.cursor() as cur:
                cur.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname=%s AND pid <> pg_backend_pid()",
                    (dbname,),
                )
                cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(dbname)))
        finally:
            admin.close()


def test_lifecycle_audit_warns_on_same_count_wrong_value_and_allows_cutover(
    lifecycle_config,
    tmp_path,
):
    manager = SchemaManager(lifecycle_config)
    with manager.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE public.industry (
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
        cur.execute(
            """
            INSERT INTO public.industry
                (datetime, code, name, metric, value, remark)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                "2024-01-01",
                "AUDIT0001.SZ",
                "审计股票",
                "申万一级行业",
                801010.0,
                Json({"ind_name": "农业", "ind_code": "801010.SI", "scheme": "申万一级行业"}),
            ),
        )

    stage_one = manager.bootstrap_local(
        create_compat_views=False,
        report_path=tmp_path / "stage-one.json",
    )
    assert stage_one["schema_version"] == 6

    with manager.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE betalens.industry_dim
            SET industry_name = '错误行业'
            WHERE industry_code = '801010.SI'
            """
        )

    completed = manager.bootstrap_local(report_path=tmp_path / "completed-cutover.json")
    assert completed["schema_version"] == 9
    assert completed["precommit_verification"]["ok"]
    assert completed["verification"]["ok"]

    with manager.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT max(version) FROM betalens.schema_migration")
        assert cur.fetchone()[0] == 9
        cur.execute(
            "SELECT c.relkind FROM pg_class c WHERE c.oid=to_regclass('public.industry')"
        )
        assert cur.fetchone()[0] == "v"
        cur.execute("SELECT to_regclass('betalens_legacy.industry')")
        assert cur.fetchone()[0] is not None
