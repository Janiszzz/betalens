from __future__ import annotations

import hashlib
from datetime import date

import pytest

from betalens_db_manager.contracts import (
    LATEST_SCHEMA_VERSION,
    get_schema_contract,
)
from betalens_db_manager.schema import (
    Migration,
    MigrationChecksumError,
    SchemaDowngradeError,
    SchemaManager,
    canonicalize_migration_bytes,
    load_migrations,
    migration_checksum_variants,
)


def test_migration_checksum_is_lf_canonical_and_accepts_historical_line_endings():
    lf = b"SELECT 1;\nSELECT 2;\n"
    crlf = lf.replace(b"\n", b"\r\n")

    lf_checksum, lf_accepted = migration_checksum_variants(lf)
    crlf_checksum, crlf_accepted = migration_checksum_variants(crlf)

    assert canonicalize_migration_bytes(crlf) == lf
    assert lf_checksum == crlf_checksum == hashlib.sha256(lf).hexdigest()
    assert hashlib.sha256(lf).hexdigest() in crlf_accepted
    assert hashlib.sha256(crlf).hexdigest() in lf_accepted

    migration = Migration(
        version=1,
        name="bootstrap",
        resource_name="0001_bootstrap.sql",
        checksum=lf_checksum,
        accepted_checksums=lf_accepted,
        sql_text=lf.decode("utf-8"),
    )
    manager = object.__new__(SchemaManager)
    manager._validate_applied_checksums(
        (migration,),
        {1: {"name": "bootstrap", "checksum": hashlib.sha256(crlf).hexdigest()}},
    )
    with pytest.raises(MigrationChecksumError):
        manager._validate_applied_checksums(
            (migration,),
            {1: {"name": "bootstrap", "checksum": "0" * 64}},
        )


def test_packaged_migrations_and_versioned_contract_are_contiguous():
    migrations = load_migrations()

    assert [migration.version for migration in migrations] == list(
        range(1, LATEST_SCHEMA_VERSION + 1)
    )
    assert all("\r" not in migration.sql_text for migration in migrations)
    assert len(get_schema_contract(5).tables) == 14
    assert get_schema_contract(5).views == ()
    assert len(get_schema_contract(7).views) == 10
    assert "betalens.assert_legacy_equivalence()" in get_schema_contract(9).functions
    assert len(get_schema_contract(10).tables) == 15
    assert len(get_schema_contract(10).views) == 11
    assert "trade_calendar_day" in get_schema_contract(10).tables
    assert "trade_calendar" in get_schema_contract(10).views


def test_lifecycle_audit_is_advisory_and_accepts_the_prior_deployed_checksum():
    migration = next(item for item in load_migrations() if item.version == 9)

    assert "RAISE WARNING" in migration.sql_text
    assert (
        "bf7b4de327d4a20111b1bf49474ac729a5ef25e82215ae0b2ffc4bc5bdc7a44d"
        in migration.accepted_checksums
    )


def test_schema_target_rejects_downgrades():
    applied = {version: {} for version in range(1, 10)}

    with pytest.raises(SchemaDowngradeError, match="当前版本 9，请求版本 6"):
        SchemaManager._reject_downgrade(applied, 6)
    SchemaManager._reject_downgrade(applied, 9)


def test_partition_contract_checks_every_explicit_and_existing_year():
    current = date.today().year
    years = {1999, current, current + 1}
    rows = [
        {
            "partition_name": f"observation_fact_{year}",
            "partition_schema": "betalens",
            "partition_bound": (
                f"FOR VALUES FROM ('{year}-01-01 00:00:00') "
                f"TO ('{year + 1}-01-01 00:00:00')"
            ),
        }
        for year in years
    ]

    errors, required = SchemaManager._partition_contract_errors(rows, {1999})

    assert errors == []
    assert required == years
    rows[0]["partition_bound"] = "FOR VALUES FROM ('1900-01-01') TO ('1901-01-01')"
    errors, _ = SchemaManager._partition_contract_errors(rows, {1999})
    assert any("边界不正确" in error for error in errors)


def test_early_failure_report_is_persisted(tmp_path):
    manager = object.__new__(SchemaManager)
    manager.db_config = {"dbname": "isolated_test"}
    output = tmp_path / "planning.json"

    report = manager.write_failure_report("missing migration", stage="planning", report_path=output)

    assert report["status"] == "failed"
    assert report["stage"] == "planning"
    assert output.exists()
    assert output.with_suffix(".log").exists()
