"""Versioned schema management for the local Betalens database.

Database objects are owned by :mod:`betalens_db_manager`.  Runtime readers in
``betalens.datafeed`` never execute DDL and use the physical schema created by
this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

import psycopg2
import psycopg2.extras
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from betalens.datafeed.config import get_database_config

from .constants import ALLOWED_TABLES, MANAGER_LOG_ROOT
from .contracts import (
    BASE_TABLES,
    COMPATIBILITY_VIEWS,
    COMPATIBILITY_VIEW_VERSION,
    CONSTRAINT_DEFINITION_FRAGMENTS,
    CORE_METRIC_SEEDS,
    DEFAULT_DEFINITION_FRAGMENTS,
    FINALIZE_VERSION,
    IDENTITY_COLUMNS,
    INDEX_DEFINITION_FRAGMENTS,
    LATEST_SCHEMA_VERSION,
    LEGACY_MIGRATION_VERSION,
    NOT_NULL_COLUMNS,
    REQUIRED_ALIASES,
    REQUIRED_CONSTRAINTS,
    REQUIRED_CONSTRAINT_TYPES,
    REQUIRED_INDEXES,
    TABLE_COLUMNS,
    VIEW_COLUMNS,
    VIEW_DEFINITION_FRAGMENTS,
    VIEW_DEFINITION_HASHES,
    get_schema_contract,
)
from .utils import clean_database_config, json_default


PHYSICAL_SCHEMA = "betalens"
LEGACY_SCHEMA = "betalens_legacy"
MIGRATION_PACKAGE = "betalens_db_manager.migrations"
MIGRATION_FILE_RE = re.compile(r"^(?P<version>\d{4})_(?P<name>[a-z0-9_]+)\.sql$")
ADVISORY_LOCK_NAME = "betalens_schema_migration_v2"

# Version 0009 originally rolled back schema installation whenever a retained
# legacy table was not perfectly equivalent to its normalized representation.
# Its revised form keeps that audit as a warning.  Databases that completed the
# former version remain valid and must not be rejected solely for this intended
# behavior correction.
_SUPERSEDED_MIGRATION_CHECKSUMS: dict[int, frozenset[str]] = {
    9: frozenset(("bf7b4de327d4a20111b1bf49474ac729a5ef25e82215ae0b2ffc4bc5bdc7a44d",)),
}


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    resource_name: str
    checksum: str
    accepted_checksums: frozenset[str]
    sql_text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "resource": self.resource_name,
            "checksum": self.checksum,
        }


class MigrationChecksumError(RuntimeError):
    """Raised when an already applied migration was edited in place."""


class SchemaDowngradeError(RuntimeError):
    """Raised when a caller requests a target older than the installed schema."""


def validate_database_name(dbname: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9_]+", dbname))


def canonicalize_migration_bytes(raw: bytes) -> bytes:
    """Return the platform-independent UTF-8/LF migration representation."""

    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def migration_checksum_variants(raw: bytes) -> tuple[str, frozenset[str]]:
    """Return the canonical checksum plus accepted historical LF/CRLF hashes."""

    canonical = canonicalize_migration_bytes(raw)
    canonical_checksum = hashlib.sha256(canonical).hexdigest()
    crlf_checksum = hashlib.sha256(canonical.replace(b"\n", b"\r\n")).hexdigest()
    raw_checksum = hashlib.sha256(raw).hexdigest()
    return canonical_checksum, frozenset((canonical_checksum, crlf_checksum, raw_checksum))


def _read_migration_bytes(resource_name: str) -> bytes:
    # ``files`` was added in Python 3.9; keep source and wheel installs working
    # on the package's declared Python 3.8 floor.
    if hasattr(resources, "files"):
        return resources.files(MIGRATION_PACKAGE).joinpath(resource_name).read_bytes()
    return resources.read_binary(MIGRATION_PACKAGE, resource_name)


def load_migrations() -> tuple[Migration, ...]:
    if hasattr(resources, "files"):
        names = [item.name for item in resources.files(MIGRATION_PACKAGE).iterdir()]
    else:
        names = list(resources.contents(MIGRATION_PACKAGE))
    migrations: list[Migration] = []
    for resource_name in sorted(names):
        match = MIGRATION_FILE_RE.fullmatch(resource_name)
        if not match:
            continue
        raw = _read_migration_bytes(resource_name)
        canonical = canonicalize_migration_bytes(raw)
        checksum, historical_checksums = migration_checksum_variants(raw)
        version = int(match.group("version"))
        accepted_checksums = (
            historical_checksums if version <= FINALIZE_VERSION else frozenset((checksum,))
        )
        accepted_checksums = accepted_checksums | _SUPERSEDED_MIGRATION_CHECKSUMS.get(
            version, frozenset()
        )
        migrations.append(
            Migration(
                version=version,
                name=match.group("name"),
                resource_name=resource_name,
                checksum=checksum,
                accepted_checksums=accepted_checksums,
                sql_text=canonical.decode("utf-8"),
            )
        )
    if not migrations:
        raise RuntimeError("未找到 betalens_db_manager migration SQL")
    versions = [item.version for item in migrations]
    if versions != list(range(1, len(versions) + 1)):
        raise RuntimeError(f"migration 版本必须从 1 连续递增，当前为: {versions}")
    if versions[-1] != LATEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"migration 最新版本 {versions[-1]} 与 contract {LATEST_SCHEMA_VERSION} 不一致"
        )
    return tuple(migrations)


class SchemaManager:
    """Create, migrate, and strictly verify the local PostgreSQL schema."""

    def __init__(self, db_config: dict[str, Any] | None = None):
        config = dict(get_database_config())
        env_mapping = {
            "dbname": ("BETALENS_DB_NAME", "BETALENS_DBNAME", "PGDATABASE"),
            "user": ("BETALENS_DB_USER", "PGUSER"),
            "password": ("BETALENS_DB_PASSWORD", "PGPASSWORD"),
            "host": ("BETALENS_DB_HOST", "PGHOST"),
            "port": ("BETALENS_DB_PORT", "PGPORT"),
        }
        for key, candidates in env_mapping.items():
            for env_name in candidates:
                if os.environ.get(env_name) is not None:
                    config[key] = os.environ[env_name]
                    break
        if db_config:
            config.update({key: value for key, value in db_config.items() if value is not None})
        self.db_config = clean_database_config(config)

    def validate_table(self, table_name: str) -> str:
        if table_name not in ALLOWED_TABLES and table_name not in COMPATIBILITY_VIEWS:
            raise ValueError(f"非法逻辑表名: {table_name}")
        return table_name

    def connect(self, dbname: str | None = None):
        cfg = dict(self.db_config)
        if dbname is not None:
            cfg["dbname"] = dbname
        return psycopg2.connect(**cfg)

    def database_exists(self, dbname: str | None = None) -> bool:
        target = dbname or self.db_config["dbname"]
        conn = self.connect("postgres")
        try:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target,))
                return cur.fetchone() is not None
        finally:
            conn.close()

    def create_database(self, dbname: str | None = None) -> bool:
        target = dbname or self.db_config["dbname"]
        if not validate_database_name(target):
            raise ValueError(f"非法数据库名: {target}")
        conn = self.connect("postgres")
        try:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target,))
                if cur.fetchone() is not None:
                    return False
                try:
                    cur.execute(
                        sql.SQL("CREATE DATABASE {} ENCODING 'UTF8' TEMPLATE template0").format(
                            sql.Identifier(target)
                        )
                    )
                except psycopg2.errors.DuplicateDatabase:
                    # Another bootstrap may have won the race after the
                    # existence check.  The target state is still satisfied.
                    return False
                return True
        finally:
            conn.close()

    def _read_applied_migrations(self, conn) -> dict[int, dict[str, Any]]:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT to_regclass('betalens.schema_migration') AS relation")
            if cur.fetchone()["relation"] is None:
                return {}
            cur.execute(
                """
                SELECT version, name, checksum, applied_at, execution_ms
                FROM betalens.schema_migration
                ORDER BY version
                """
            )
            return {int(row["version"]): dict(row) for row in cur.fetchall()}

    def _validate_applied_checksums(
        self,
        migrations: Iterable[Migration],
        applied: dict[int, dict[str, Any]],
    ) -> None:
        known = {migration.version: migration for migration in migrations}
        for version, row in applied.items():
            migration = known.get(version)
            if migration is None:
                raise MigrationChecksumError(f"数据库包含未知 migration 版本: {version}")
            applied_checksum = str(row["checksum"]).strip()
            if row["name"] != migration.name or applied_checksum not in migration.accepted_checksums:
                raise MigrationChecksumError(
                    f"migration {version:04d}_{migration.name} checksum 不一致；"
                    "已执行 migration 不可原地修改"
                )

    @staticmethod
    def _reject_downgrade(applied: dict[int, dict[str, Any]], target_version: int) -> None:
        current_version = max(applied, default=0)
        if current_version > target_version:
            raise SchemaDowngradeError(
                f"不支持 schema 降级: 当前版本 {current_version}，请求版本 {target_version}"
            )

    def plan_migration(self, target_version: int | None = None) -> dict[str, Any]:
        migrations = load_migrations()
        latest = migrations[-1].version
        target = latest if target_version is None else int(target_version)
        if target < 0 or target > latest:
            raise ValueError(f"target_version 必须位于 0..{latest}")
        exists = self.database_exists()
        applied: dict[int, dict[str, Any]] = {}
        if exists:
            with self.connect() as conn:
                applied = self._read_applied_migrations(conn)
                self._validate_applied_checksums(migrations, applied)
                self._reject_downgrade(applied, target)
        pending = [
            migration.as_dict()
            for migration in migrations
            if migration.version <= target and migration.version not in applied
        ]
        return {
            "database": self.db_config.get("dbname"),
            "database_exists": exists,
            "current_version": max(applied, default=0),
            "target_version": target,
            "latest_version": latest,
            "applied": [dict(applied[key]) for key in sorted(applied)],
            "pending": pending,
        }

    def _apply_migrations(
        self,
        conn,
        target_version: int,
    ) -> tuple[list[dict[str, Any]], list[str], set[int]]:
        migrations = load_migrations()
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (ADVISORY_LOCK_NAME,))
        applied = self._read_applied_migrations(conn)
        self._validate_applied_checksums(migrations, applied)
        self._reject_downgrade(applied, target_version)
        created_partitions: list[str] = []
        legacy_years = self._legacy_observation_years(conn) if target_version >= 4 else set()
        required_partition_years = self._normalize_partition_years(legacy_years)
        if target_version >= 4 and 4 in applied:
            created_partitions.extend(
                self.ensure_observation_partitions(
                    legacy_years,
                    connection=conn,
                )
            )
        completed: list[dict[str, Any]] = []
        for migration in migrations:
            if migration.version > target_version or migration.version in applied:
                continue
            started = time.perf_counter()
            with conn.cursor() as cur:
                cur.execute(migration.sql_text)
                execution_ms = max(0, int((time.perf_counter() - started) * 1000))
                cur.execute(
                    """
                    INSERT INTO betalens.schema_migration
                        (version, name, checksum, execution_ms)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (migration.version, migration.name, migration.checksum, execution_ms),
                )
            item = migration.as_dict()
            item["execution_ms"] = execution_ms
            completed.append(item)
            if migration.version == 4:
                # Legacy rows are copied by migration 6.  Create every source
                # year now so the copy cannot fall into a missing partition.
                created_partitions.extend(
                    self.ensure_observation_partitions(
                        legacy_years,
                        connection=conn,
                    )
                )
        return completed, created_partitions, required_partition_years

    def _legacy_observation_years(self, conn) -> set[int]:
        years: set[int] = set()
        legacy_tables = (
            "daily_market",
            "daily_index",
            "daily_fund",
            "daily_bond",
            "fundamentals",
            "macro",
            "factors",
        )
        with conn.cursor() as cur:
            for table_name in legacy_tables:
                cur.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
                relation = cur.fetchone()[0]
                if relation is None:
                    continue
                cur.execute("SELECT relkind FROM pg_class WHERE oid = %s::regclass", (relation,))
                row = cur.fetchone()
                if row is None or row[0] not in ("r", "p"):
                    continue
                cur.execute(
                    sql.SQL(
                        "SELECT min(datetime), max(datetime) FROM public.{}"
                    ).format(sql.Identifier(table_name))
                )
                minimum, maximum = cur.fetchone()
                if minimum is not None and maximum is not None:
                    years.update(range(int(minimum.year), int(maximum.year) + 1))
        return years

    @staticmethod
    def _normalize_partition_years(years: Iterable[int] | None = None) -> set[int]:
        current_year = date.today().year
        requested = {current_year, current_year + 1}
        if years is not None:
            requested.update(int(year) for year in years)
        invalid = sorted(year for year in requested if year < 1900 or year > 9998)
        if invalid:
            raise ValueError(f"不支持的 observation 分区年份: {invalid}")
        return requested

    @classmethod
    def _partition_contract_errors(
        cls,
        partition_rows: Iterable[dict[str, Any]],
        expected_years: Iterable[int] | None = None,
    ) -> tuple[list[str], set[int]]:
        rows = list(partition_rows)
        required_years = cls._normalize_partition_years(expected_years)
        errors: list[str] = []
        partition_names: set[str] = set()
        for row in rows:
            partition_name = row["partition_name"]
            partition_names.add(partition_name)
            match = re.fullmatch(r"observation_fact_(\d{4})", partition_name)
            if row.get("partition_schema") != PHYSICAL_SCHEMA or match is None:
                errors.append(
                    f"非标准 observation 分区: "
                    f"{row.get('partition_schema')}.{partition_name}"
                )
                continue
            year = int(match.group(1))
            bound = row.get("partition_bound") or ""
            if f"{year}-01-01" not in bound or f"{year + 1}-01-01" not in bound:
                errors.append(f"observation 分区 {year} 边界不正确")
            parent_index_count = row.get("parent_index_count")
            inherited_index_count = row.get("inherited_index_count")
            if (
                parent_index_count is not None
                and inherited_index_count is not None
                and int(inherited_index_count) < int(parent_index_count)
            ):
                errors.append(
                    f"observation 分区 {year} 缺少继承索引: "
                    f"{inherited_index_count}/{parent_index_count}"
                )
        for year in sorted(required_years):
            if f"observation_fact_{year}" not in partition_names:
                errors.append(f"缺少 observation 年度分区: {year}")
        return errors, required_years

    def ensure_observation_partitions(
        self,
        years: Iterable[int] | None = None,
        *,
        connection=None,
    ) -> list[str]:
        """Create observation partitions for explicit years.

        The current and next calendar year are always included.  Importers call
        this method before loading historical years; they do not construct DDL.
        """

        requested = self._normalize_partition_years(years)

        owns_connection = connection is None
        conn = connection or self.connect()
        created: list[str] = []
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (ADVISORY_LOCK_NAME,))
                cur.execute("SELECT to_regclass('betalens.observation_fact')")
                if cur.fetchone()[0] is None:
                    raise RuntimeError("observation_fact 尚未创建；请先运行 schema migration")
                for year in sorted(requested):
                    name = f"observation_fact_{year}"
                    qualified = f"{PHYSICAL_SCHEMA}.{name}"
                    cur.execute("SELECT to_regclass(%s)", (qualified,))
                    existed = cur.fetchone()[0] is not None
                    statement = sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.{}
                        PARTITION OF {}.observation_fact
                        FOR VALUES FROM ({}) TO ({})
                        """
                    ).format(
                        sql.Identifier(PHYSICAL_SCHEMA),
                        sql.Identifier(name),
                        sql.Identifier(PHYSICAL_SCHEMA),
                        sql.Literal(date(year, 1, 1)),
                        sql.Literal(date(year + 1, 1, 1)),
                    )
                    cur.execute(statement)
                    if not existed:
                        created.append(name)
            if owns_connection:
                conn.commit()
        except Exception:
            if owns_connection:
                conn.rollback()
            raise
        finally:
            if owns_connection:
                conn.close()
        return created

    def write_failure_report(
        self,
        error: Exception | str,
        *,
        stage: str,
        report_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Persist a diagnostic report for failures before bootstrap starts."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = Path(report_path) if report_path else MANAGER_LOG_ROOT / f"bootstrap_{timestamp}.json"
        log_path = output_path.with_suffix(".log")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "failed",
            "lifecycle_status": "failed_before_commit",
            "database": self.db_config.get("dbname"),
            "stage": stage,
            "started_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "finished_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "error": str(error),
            "report_path": str(output_path),
            "log_path": str(log_path),
        }
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=json_default),
            encoding="utf-8",
        )
        log_path.write_text(f"{stage}: {error}\n", encoding="utf-8")
        return report

    def _bootstrap(
        self,
        *,
        create_database_if_missing: bool,
        migrate_legacy: bool,
        create_compat_views: bool,
        verify: bool,
        observation_years: Iterable[int] | None,
        report_path: str | Path | None,
    ) -> dict[str, Any]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = Path(report_path) if report_path else MANAGER_LOG_ROOT / f"bootstrap_{timestamp}.json"
        log_path = output_path.with_suffix(".log")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"betalens_db_manager.bootstrap.{timestamp}.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)

        report: dict[str, Any] = {
            "status": "running",
            "database": self.db_config.get("dbname"),
            "started_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "created_database": False,
            "target_version": None,
            "applied_migrations": [],
            "created_partitions": [],
            "expected_partition_years": [],
            "precommit_verification": None,
            "objects": {
                "base_tables": list(BASE_TABLES),
                "compatibility_views": list(COMPATIBILITY_VIEWS) if create_compat_views else [],
            },
            "legacy_tables_preserved": [],
            "warnings": [],
            "report_path": str(output_path),
            "log_path": str(log_path),
        }
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=json_default),
            encoding="utf-8",
        )
        committed = False
        try:
            migrations = load_migrations()
            if not migrate_legacy and create_compat_views:
                raise ValueError(
                    "create_compat_views=True 需要 migrate_legacy=True；"
                    "否则无法在不覆盖 public 旧表的前提下切换视图"
                )
            if not migrate_legacy:
                target_version = LEGACY_MIGRATION_VERSION - 1
            elif not create_compat_views:
                target_version = COMPATIBILITY_VIEW_VERSION - 1
            else:
                target_version = migrations[-1].version
            report["target_version"] = target_version
            logger.info("checking target database")
            if not self.database_exists():
                if not create_database_if_missing:
                    raise RuntimeError("数据库不存在；请显式允许 create_database_if_missing")
                report["created_database"] = self.create_database()
                logger.info("created target database")

            logger.info("applying migrations through version %s", target_version)
            conn = self.connect()
            try:
                applied_migrations, created_partitions, required_partition_years = self._apply_migrations(
                    conn, target_version
                )
                report["applied_migrations"] = applied_migrations
                report["warnings"].extend(
                    notice.strip()
                    for notice in conn.notices
                    if "legacy equivalence audit" in notice.lower()
                )
                if target_version >= 4:
                    required_partition_years.update(self._normalize_partition_years(observation_years))
                    created_partitions.extend(
                        self.ensure_observation_partitions(
                            observation_years,
                            connection=conn,
                        )
                    )
                report["created_partitions"] = sorted(set(created_partitions))
                report["expected_partition_years"] = sorted(required_partition_years)
                if verify:
                    report["precommit_verification"] = self.verify_schema_precommit(
                        conn,
                        require_compat_views=create_compat_views and target_version >= COMPATIBILITY_VIEW_VERSION,
                        expected_version=target_version,
                        expected_partition_years=required_partition_years,
                    )
                    if not report["precommit_verification"]["ok"]:
                        raise RuntimeError(
                            "提交前 schema 校验失败: "
                            + "; ".join(report["precommit_verification"]["errors"])
                        )
                conn.commit()
                committed = True
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

            conn = self.connect()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT c.relname
                        FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE n.nspname = %s AND c.relkind IN ('r', 'p')
                        ORDER BY c.relname
                        """,
                        (LEGACY_SCHEMA,),
                    )
                    report["legacy_tables_preserved"] = [row["relname"] for row in cur.fetchall()]
                    cur.execute(
                        """
                        SELECT row_count FROM betalens.dataset_coverage
                        WHERE logical_dataset = 'daily_market'
                        """
                    )
                    daily_coverage = cur.fetchone()
                    if daily_coverage is not None and int(daily_coverage["row_count"]) == 0:
                        report["warnings"].append(
                            "daily_market schema 已创建，但没有历史行情数据；请通过 manifest 或导入任务装载"
                        )
            finally:
                conn.close()

            if verify:
                logger.info("verifying schema")
                report["verification"] = self.verify_schema(
                    require_compat_views=(
                        create_compat_views and target_version >= COMPATIBILITY_VIEW_VERSION
                    ),
                    expected_version=target_version,
                    expected_partition_years=report["expected_partition_years"],
                )
                if not report["verification"]["ok"]:
                    raise RuntimeError("schema 校验失败: " + "; ".join(report["verification"]["errors"]))
            report["status"] = "completed"
            report["schema_version"] = target_version
            report["finished_at"] = datetime.now().isoformat(sep=" ", timespec="seconds")
            logger.info("bootstrap completed")
            return report
        except Exception as exc:
            report["status"] = (
                "committed_verification_failed" if committed else "failed_before_commit"
            )
            report["error"] = str(exc)
            report["finished_at"] = datetime.now().isoformat(sep=" ", timespec="seconds")
            logger.exception("bootstrap failed")
            raise RuntimeError(f"数据库初始化失败；诊断报告: {output_path}: {exc}") from exc
        finally:
            output_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=json_default),
                encoding="utf-8",
            )
            handler.close()
            logger.removeHandler(handler)

    def bootstrap_local(
        self: "SchemaManager | None" = None,
        *,
        create_database_if_missing: bool = True,
        migrate_legacy: bool = True,
        create_compat_views: bool = True,
        verify: bool = True,
        observation_years: Iterable[int] | None = None,
        report_path: str | Path | None = None,
        db_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create or upgrade a reproducible local database in one call.

        This intentionally supports both ``SchemaManager.bootstrap_local(...)``
        and ``SchemaManager(config).bootstrap_local(...)``.  ``db_config`` is
        accepted only for the class-style call.
        """

        manager = self if isinstance(self, SchemaManager) else SchemaManager(db_config)
        if isinstance(self, SchemaManager) and db_config is not None:
            raise ValueError("实例调用 bootstrap_local 时请在构造 SchemaManager 时传 db_config")
        return manager._bootstrap(
            create_database_if_missing=create_database_if_missing,
            migrate_legacy=migrate_legacy,
            create_compat_views=create_compat_views,
            verify=verify,
            observation_years=observation_years,
            report_path=report_path,
        )

    def ensure_schema(
        self,
        tables: list[str] | None = None,
        force: bool = False,
        create_database_if_missing: bool = False,
        create_indexes: bool = True,
        create_comments: bool = True,
    ) -> dict[str, Any]:
        """Compatibility wrapper around the atomic versioned bootstrap.

        Individual table creation is no longer supported because it can leave a
        schema whose views and routing metadata disagree with its fact tables.
        """

        del create_indexes, create_comments
        if force:
            raise ValueError("force DROP CASCADE 已停用；请使用版本化 migration")
        if tables is not None:
            unknown = set(tables) - set(COMPATIBILITY_VIEWS)
            if unknown:
                raise ValueError(f"非法逻辑表名: {sorted(unknown)}")
        return self.bootstrap_local(create_database_if_missing=create_database_if_missing)

    def table_exists(self, cur, table_name: str) -> bool:
        table_name = self.validate_table(table_name)
        cur.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
        row = cur.fetchone()
        value = row.get("to_regclass") if isinstance(row, dict) else row[0]
        return value is not None

    @contextmanager
    def _connection_scope(self, connection=None):
        if connection is not None:
            yield connection
            return
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    def _verify_schema(
        self,
        *,
        require_compat_views: bool = True,
        expected_version: int | None = None,
        expected_partition_years: Iterable[int] | None = None,
        connection=None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "database_exists": False,
            "schema": PHYSICAL_SCHEMA,
            "tables": {},
            "views": {},
            "functions": {},
            "partitions": [],
            "expected_partition_years": [],
            "migrations": [],
            "seeds": {},
            "errors": [],
            "warnings": [],
        }
        try:
            if connection is None and not self.database_exists():
                result["errors"].append("目标数据库不存在")
                return result
            migrations = load_migrations()
            with self._connection_scope(connection) as conn:
                applied = self._read_applied_migrations(conn)
                self._validate_applied_checksums(migrations, applied)
                result["database_exists"] = True
                result["migrations"] = [dict(applied[key]) for key in sorted(applied)]
                target = expected_version if expected_version is not None else migrations[-1].version
                contract = get_schema_contract(target)
                self._reject_downgrade(applied, target)
                missing_versions = [version for version in range(1, target + 1) if version not in applied]
                if missing_versions:
                    result["errors"].append(f"缺少 migration 版本: {missing_versions}")

                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT c.relname, c.relkind
                        FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE n.nspname = %s AND c.relname = ANY(%s)
                        """,
                        (PHYSICAL_SCHEMA, list(contract.tables)),
                    )
                    relations = {row["relname"]: row["relkind"] for row in cur.fetchall()}
                    cur.execute(
                        """
                        SELECT table_name, column_name, data_type, is_nullable,
                               column_default, is_identity, identity_generation
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = ANY(%s)
                        ORDER BY table_name, ordinal_position
                        """,
                        (PHYSICAL_SCHEMA, list(contract.tables)),
                    )
                    columns: dict[str, list[dict[str, Any]]] = {}
                    for row in cur.fetchall():
                        columns.setdefault(row["table_name"], []).append(dict(row))
                    cur.execute(
                        """
                        SELECT table_relation.relname AS tablename,
                               index_relation.relname AS indexname,
                               pg_get_indexdef(index_info.indexrelid) AS indexdef,
                               index_info.indisvalid,
                               index_info.indisready
                        FROM pg_index index_info
                        JOIN pg_class index_relation ON index_relation.oid = index_info.indexrelid
                        JOIN pg_class table_relation ON table_relation.oid = index_info.indrelid
                        JOIN pg_namespace n ON n.oid = table_relation.relnamespace
                        WHERE n.nspname = %s AND table_relation.relname = ANY(%s)
                        """,
                        (PHYSICAL_SCHEMA, list(contract.tables)),
                    )
                    indexes: dict[str, dict[str, dict[str, Any]]] = {}
                    for row in cur.fetchall():
                        indexes.setdefault(row["tablename"], {})[row["indexname"]] = dict(row)
                    cur.execute(
                        """
                        SELECT c.relname AS table_name, constraint_info.conname,
                               constraint_info.contype, constraint_info.convalidated,
                               pg_get_constraintdef(constraint_info.oid) AS definition
                        FROM pg_constraint constraint_info
                        JOIN pg_class c ON c.oid = constraint_info.conrelid
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE n.nspname = %s AND c.relname = ANY(%s)
                        """,
                        (PHYSICAL_SCHEMA, list(contract.tables)),
                    )
                    constraints: dict[str, dict[str, dict[str, Any]]] = {}
                    for row in cur.fetchall():
                        constraints.setdefault(row["table_name"], {})[row["conname"]] = dict(row)

                    for table_name in contract.tables:
                        expected_kind = "p" if table_name == "observation_fact" else "r"
                        info = {
                            "exists": table_name in relations,
                            "relkind": relations.get(table_name),
                            "columns": columns.get(table_name, []),
                            "indexes": sorted(indexes.get(table_name, {})),
                            "constraints": sorted(constraints.get(table_name, {})),
                        }
                        result["tables"][table_name] = info
                        if not info["exists"]:
                            result["errors"].append(f"缺少基础表 {PHYSICAL_SCHEMA}.{table_name}")
                            continue
                        if info["relkind"] != expected_kind:
                            result["errors"].append(
                                f"{PHYSICAL_SCHEMA}.{table_name} relkind 应为 {expected_kind}，"
                                f"实际为 {info['relkind']}"
                            )
                        actual_columns = [(row["column_name"], row["data_type"]) for row in info["columns"]]
                        if actual_columns != list(TABLE_COLUMNS[table_name]):
                            result["errors"].append(f"{PHYSICAL_SCHEMA}.{table_name} 列定义不符合 manifest")
                        column_map = {row["column_name"]: row for row in info["columns"]}
                        actual_not_null = {
                            column_name
                            for column_name, column in column_map.items()
                            if column["is_nullable"] == "NO"
                        }
                        expected_not_null = set(NOT_NULL_COLUMNS[table_name])
                        if actual_not_null != expected_not_null:
                            result["errors"].append(
                                f"{PHYSICAL_SCHEMA}.{table_name} nullable 定义不符合 manifest"
                            )
                        for identity_column in IDENTITY_COLUMNS.get(table_name, ()):
                            column = column_map.get(identity_column, {})
                            if column.get("is_identity") != "YES" or column.get("identity_generation") != "BY DEFAULT":
                                result["errors"].append(
                                    f"{PHYSICAL_SCHEMA}.{table_name}.{identity_column} identity 定义不正确"
                                )
                        for (default_table, default_column), fragment in DEFAULT_DEFINITION_FRAGMENTS.items():
                            if default_table != table_name:
                                continue
                            definition = (column_map.get(default_column) or {}).get("column_default") or ""
                            if fragment not in definition:
                                result["errors"].append(
                                    f"{PHYSICAL_SCHEMA}.{table_name}.{default_column} default 定义不正确"
                                )
                        missing_indexes = set(REQUIRED_INDEXES.get(table_name, ())) - set(info["indexes"])
                        if missing_indexes:
                            result["errors"].append(
                                f"{PHYSICAL_SCHEMA}.{table_name} 缺少索引: {sorted(missing_indexes)}"
                            )
                        missing_constraints = set(REQUIRED_CONSTRAINTS.get(table_name, ())) - set(
                            info["constraints"]
                        )
                        if missing_constraints:
                            result["errors"].append(
                                f"{PHYSICAL_SCHEMA}.{table_name} 缺少约束: {sorted(missing_constraints)}"
                            )
                        invalid_constraints = [
                            name
                            for name, constraint in constraints.get(table_name, {}).items()
                            if not constraint["convalidated"]
                        ]
                        if invalid_constraints:
                            result["errors"].append(
                                f"{PHYSICAL_SCHEMA}.{table_name} 存在未验证约束: {invalid_constraints}"
                            )
                        actual_constraint_types = {
                            constraint["contype"] for constraint in constraints.get(table_name, {}).values()
                        }
                        missing_constraint_types = set(REQUIRED_CONSTRAINT_TYPES[table_name]) - actual_constraint_types
                        if missing_constraint_types:
                            result["errors"].append(
                                f"{PHYSICAL_SCHEMA}.{table_name} 缺少约束类型: "
                                f"{sorted(missing_constraint_types)}"
                            )

                    all_index_definitions = {
                        name: index_info["indexdef"]
                        for table_indexes in indexes.values()
                        for name, index_info in table_indexes.items()
                    }
                    invalid_indexes = [
                        name
                        for table_indexes in indexes.values()
                        for name, index_info in table_indexes.items()
                        if not index_info["indisvalid"] or not index_info["indisready"]
                    ]
                    if invalid_indexes:
                        result["errors"].append(f"存在无效或未就绪索引: {sorted(invalid_indexes)}")
                    for index_name, fragments in INDEX_DEFINITION_FRAGMENTS.items():
                        definition = all_index_definitions.get(index_name, "")
                        if definition and not all(fragment in definition for fragment in fragments):
                            result["errors"].append(f"索引 {index_name} 键顺序不符合 manifest")
                    all_constraints = {
                        name: constraint
                        for table_constraints in constraints.values()
                        for name, constraint in table_constraints.items()
                    }
                    for constraint_name, fragments in CONSTRAINT_DEFINITION_FRAGMENTS.items():
                        constraint = all_constraints.get(constraint_name)
                        definition = "" if constraint is None else constraint["definition"]
                        if definition and not all(fragment in definition for fragment in fragments):
                            result["errors"].append(
                                f"约束 {constraint_name} 定义不符合 manifest"
                            )

                    if "observation_fact" in contract.tables:
                        cur.execute(
                            """
                            SELECT child.relname AS partition_name,
                                   child_ns.nspname AS partition_schema,
                                   pg_get_expr(child.relpartbound, child.oid) AS partition_bound,
                                   (
                                       SELECT count(*)
                                       FROM pg_index parent_index
                                       WHERE parent_index.indrelid = parent.oid
                                   ) AS parent_index_count,
                                   (
                                       SELECT count(*)
                                       FROM pg_index parent_index
                                       JOIN pg_inherits index_inheritance
                                         ON index_inheritance.inhparent = parent_index.indexrelid
                                       JOIN pg_index child_index
                                         ON child_index.indexrelid = index_inheritance.inhrelid
                                        AND child_index.indrelid = child.oid
                                       WHERE parent_index.indrelid = parent.oid
                                         AND child_index.indisvalid
                                         AND child_index.indisready
                                   ) AS inherited_index_count
                            FROM pg_inherits inheritance
                            JOIN pg_class child ON child.oid = inheritance.inhrelid
                            JOIN pg_namespace child_ns ON child_ns.oid = child.relnamespace
                            JOIN pg_class parent ON parent.oid = inheritance.inhparent
                            JOIN pg_namespace parent_ns ON parent_ns.oid = parent.relnamespace
                            WHERE parent_ns.nspname = %s AND parent.relname = 'observation_fact'
                            ORDER BY child.relname
                            """,
                            (PHYSICAL_SCHEMA,),
                        )
                        partition_rows = [dict(row) for row in cur.fetchall()]
                        result["partitions"] = partition_rows
                        partition_errors, required_years = self._partition_contract_errors(
                            partition_rows, expected_partition_years
                        )
                        result["expected_partition_years"] = sorted(required_years)
                        result["errors"].extend(partition_errors)

                    required_views = contract.views if require_compat_views else ()
                    if required_views:
                        cur.execute(
                            """
                            SELECT table_name, is_updatable
                            FROM information_schema.views
                            WHERE table_schema = 'public' AND table_name = ANY(%s)
                            """,
                            (list(required_views),),
                        )
                        view_rows = {row["table_name"]: row["is_updatable"] for row in cur.fetchall()}
                        cur.execute(
                            """
                            SELECT table_name, column_name, data_type
                            FROM information_schema.columns
                            WHERE table_schema = 'public' AND table_name = ANY(%s)
                            ORDER BY table_name, ordinal_position
                            """,
                            (list(required_views),),
                        )
                        view_columns: dict[str, list[tuple[str, str]]] = {}
                        for row in cur.fetchall():
                            view_columns.setdefault(row["table_name"], []).append(
                                (row["column_name"], row["data_type"])
                            )
                        for view_name in required_views:
                            info = {
                                "exists": view_name in view_rows,
                                "read_only": view_rows.get(view_name) == "NO",
                                "columns": view_columns.get(view_name, []),
                            }
                            result["views"][view_name] = info
                            if not info["exists"]:
                                result["errors"].append(f"缺少兼容视图 public.{view_name}")
                            elif not info["read_only"]:
                                result["errors"].append(f"兼容视图 public.{view_name} 必须只读")
                            if info["exists"] and info["columns"] != list(VIEW_COLUMNS):
                                result["errors"].append(f"兼容视图 public.{view_name} 六列契约不符合 manifest")

                        if contract.verify_view_definitions:
                            cur.execute(
                                """
                                SELECT table_name, view_definition
                                FROM information_schema.views
                                WHERE table_schema = 'public' AND table_name = ANY(%s)
                                """,
                                (list(required_views),),
                            )
                            view_definitions = {
                                row["table_name"]: " ".join((row["view_definition"] or "").lower().split())
                                for row in cur.fetchall()
                            }
                            for view_name, fragments in VIEW_DEFINITION_FRAGMENTS.items():
                                definition = view_definitions.get(view_name, "")
                                present_fragments = [
                                    fragment for fragment in fragments
                                    if fragment.lower() in definition
                                ]
                                missing_fragments = [
                                    fragment for fragment in fragments
                                    if fragment.lower() not in definition
                                ]
                                observed_hash = hashlib.sha256(
                                    "\x1f".join(present_fragments).encode("utf-8")
                                ).hexdigest()
                                result["views"].setdefault(view_name, {}).update(
                                    {
                                        "definition_hash": observed_hash,
                                        "expected_definition_hash": VIEW_DEFINITION_HASHES[view_name],
                                    }
                                )
                                if missing_fragments:
                                    result["errors"].append(
                                        f"兼容视图 public.{view_name} 定义缺少: {missing_fragments}"
                                    )
                                elif observed_hash != VIEW_DEFINITION_HASHES[view_name]:
                                    result["errors"].append(
                                        f"兼容视图 public.{view_name} 定义 hash 不符合 contract"
                                    )

                    for function_signature in contract.functions:
                        cur.execute("SELECT to_regprocedure(%s)::text AS signature", (function_signature,))
                        row = cur.fetchone()
                        present = row is not None and row["signature"] is not None
                        result["functions"][function_signature] = {"exists": present}
                        if not present:
                            result["errors"].append(f"缺少函数 {function_signature}")

                    if "metric_dim" in contract.tables:
                        cur.execute(
                            """
                            SELECT logical_dataset, metric_name, storage_column,
                                   availability_time::text AS availability_time
                            FROM betalens.metric_dim
                            WHERE storage_kind = 'core'
                            """
                        )
                        actual_core = {
                            (
                                row["logical_dataset"],
                                row["metric_name"],
                                row["storage_column"],
                                str(row["availability_time"])[:8],
                            )
                            for row in cur.fetchall()
                        }
                        expected_core = {
                            (dataset, metric_name, storage_column, availability_time)
                            for dataset, seeds in CORE_METRIC_SEEDS.items()
                            for metric_name, storage_column, availability_time in seeds
                        }
                        missing_core = sorted(expected_core - actual_core)
                        cur.execute("SELECT count(*) AS count FROM betalens.metric_dim WHERE storage_kind = 'core'")
                        core_metric_count = int(cur.fetchone()["count"])
                        cur.execute("SELECT count(*) AS count FROM betalens.metric_alias")
                        alias_count = int(cur.fetchone()["count"])
                        cur.execute(
                            """
                            SELECT alias.logical_dataset, alias.alias, canonical.metric_name
                            FROM betalens.metric_alias alias
                            JOIN betalens.metric_dim canonical ON canonical.metric_id = alias.metric_id
                            """
                        )
                        actual_aliases = {
                            (row["logical_dataset"], row["alias"], row["metric_name"])
                            for row in cur.fetchall()
                        }
                        expected_aliases = {
                            (dataset, metric_name, metric_name)
                            for dataset, seeds in CORE_METRIC_SEEDS.items()
                            for metric_name, _storage_column, _availability_time in seeds
                        }
                        expected_aliases.update(
                            (dataset, alias, canonical)
                            for dataset, aliases in REQUIRED_ALIASES.items()
                            for alias, canonical in aliases.items()
                        )
                        missing_aliases = sorted(expected_aliases - actual_aliases)
                        amount_alias = next(
                            (
                                canonical
                                for dataset, alias, canonical in actual_aliases
                                if dataset == "daily_market" and alias == "成交额(元)"
                            ),
                            None,
                        )
                        result["seeds"].update(
                            {
                                "core_metric_count": core_metric_count,
                                "metric_alias_count": alias_count,
                                "amount_alias": amount_alias,
                                "missing_core_metrics": missing_core,
                                "missing_metric_aliases": missing_aliases,
                            }
                        )
                        if missing_core:
                            result["errors"].append(f"核心 metric seed 缺失: {missing_core}")
                        if missing_aliases:
                            result["errors"].append(f"metric alias seed 缺失: {missing_aliases}")

                    if "dataset_coverage" in contract.tables:
                        cur.execute("SELECT logical_dataset FROM betalens.dataset_coverage")
                        coverage_names = {row["logical_dataset"] for row in cur.fetchall()}
                        required_coverage = set(contract.views) if contract.verify_dataset_coverage else set()
                        missing_coverage = sorted(required_coverage - coverage_names)
                        result["seeds"]["dataset_coverage_count"] = len(coverage_names)
                        result["seeds"]["missing_dataset_coverage"] = missing_coverage
                        if missing_coverage:
                            result["errors"].append(f"dataset_coverage seed 缺失: {missing_coverage}")

                    if contract.verify_comments:
                        cur.execute(
                            """
                            SELECT c.relname, obj_description(c.oid, 'pg_class') AS comment
                            FROM pg_class c
                            JOIN pg_namespace n ON n.oid = c.relnamespace
                            WHERE n.nspname = %s AND c.relname = ANY(%s)
                            """,
                            (PHYSICAL_SCHEMA, list(contract.tables)),
                        )
                        comments = {row["relname"]: row["comment"] for row in cur.fetchall()}
                        missing_comments = [name for name in contract.tables if not comments.get(name)]
                        if missing_comments:
                            result["errors"].append(
                                f"schema comments 未完整安装: {missing_comments}"
                            )
        except MigrationChecksumError as exc:
            result["errors"].append(str(exc))
        except Exception as exc:
            result["errors"].append(str(exc))
        result["ok"] = result["database_exists"] and not result["errors"]
        return result

    def verify_schema(
        self,
        *,
        require_compat_views: bool = True,
        expected_version: int | None = None,
        expected_partition_years: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        """Verify a committed schema using the contract for ``expected_version``."""

        return self._verify_schema(
            require_compat_views=require_compat_views,
            expected_version=expected_version,
            expected_partition_years=expected_partition_years,
        )

    def verify_schema_precommit(
        self,
        connection,
        *,
        require_compat_views: bool = True,
        expected_version: int | None = None,
        expected_partition_years: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        """Verify uncommitted DDL using the migration transaction connection."""

        return self._verify_schema(
            require_compat_views=require_compat_views,
            expected_version=expected_version,
            expected_partition_years=expected_partition_years,
            connection=connection,
        )


__all__ = [
    "BASE_TABLES",
    "COMPATIBILITY_VIEWS",
    "LEGACY_SCHEMA",
    "Migration",
    "MigrationChecksumError",
    "PHYSICAL_SCHEMA",
    "SchemaDowngradeError",
    "SchemaManager",
    "canonicalize_migration_bytes",
    "load_migrations",
    "migration_checksum_variants",
    "validate_database_name",
]
