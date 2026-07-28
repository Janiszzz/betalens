"""Unified database-manager service facade.

The facade is deliberately Qt-free.  CLI, BAT and the desktop application all
call these methods, so schema lifecycle, manifest preflight, import reporting
and verification cannot silently diverge between entry points.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .db import DatabaseClient, QueryRequest
from .importers import DatabaseWriter
from .job_store import JobStore
from .jobs import ImportJobRunner
from .profiles import ConnectionProfile, ConnectionResolver, ProfileStore, ResolvedConnection
from .records import ImportRecordStore
from .schema import SchemaManager
from .registry import get_dataset
from .utils import json_default


MANIFEST_TIMEOUT_MS = 30 * 60 * 1000
ProgressCallback = Callable[[dict[str, Any]], None]


def _serialize(value: Any) -> Any:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _serialize(value.as_dict())
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_serialize(payload), ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class DatabaseManager:
    """Own the complete local database lifecycle and import workflow."""

    def __init__(
        self,
        db_config: Mapping[str, Any] | None = None,
        *,
        profile: str | ConnectionProfile | None = None,
        profile_store: ProfileStore | None = None,
        resolver: ConnectionResolver | None = None,
        job_store: JobStore | None = None,
        import_statement_timeout_ms: int = MANIFEST_TIMEOUT_MS,
    ):
        self.profile_store = profile_store or ProfileStore()
        self.resolver = resolver or ConnectionResolver(self.profile_store)
        self._profile_selector = profile
        self._resolved: ResolvedConnection = self.resolver.resolve(db_config, profile=profile)
        self.import_statement_timeout_ms = int(import_statement_timeout_ms)
        self.job_store = job_store or JobStore()
        self._rebuild_services()

    def _rebuild_services(self) -> None:
        self.db_config = dict(self._resolved.config)
        self.schema = SchemaManager(self.db_config)
        self.client = DatabaseClient(self.db_config)
        self.import_client = DatabaseClient(
            self.db_config,
            statement_timeout_ms=self.import_statement_timeout_ms,
        )
        self.writer = DatabaseWriter(self.import_client)
        self.record_store = ImportRecordStore(job_store=self.job_store)
        self.import_runner = ImportJobRunner(client=self.import_client, store=self.record_store)

    @property
    def effective_config(self) -> dict[str, str]:
        """Return the runtime config, including the session-only password."""

        return dict(self._resolved.config)

    @property
    def connection_info(self) -> dict[str, Any]:
        return self._resolved.as_dict()

    @property
    def profile_name(self) -> str | None:
        return self._resolved.profile_name

    def reconfigure(
        self,
        overrides: Mapping[str, Any] | None = None,
        *,
        profile: str | ConnectionProfile | None = None,
    ) -> dict[str, Any]:
        self._profile_selector = profile if profile is not None else self._profile_selector
        self._resolved = self.resolver.resolve(
            overrides,
            profile=self._profile_selector,
        )
        self._rebuild_services()
        return self.connection_info

    def save_profile(self, profile: ConnectionProfile, *, make_active: bool = True) -> ConnectionProfile:
        return self.profile_store.save(profile, make_active=make_active)

    def delete_profile(self, name: str) -> bool:
        return self.profile_store.delete(name)

    def list_profiles(self) -> list[ConnectionProfile]:
        return self.profile_store.list()

    def probe_connection(self) -> dict[str, Any]:
        """Probe PostgreSQL without raising for an offline GUI startup."""

        result: dict[str, Any] = {
            "status": "unreachable",
            "database": self.db_config.get("dbname"),
            "config": self._resolved.display_config(),
            "sources": dict(self._resolved.sources),
            "error": None,
        }
        try:
            if not self.schema.database_exists():
                result["status"] = "database_missing"
                return result
            result["connection"] = self.client.test_connection()
            result["status"] = "online"
        except Exception as exc:  # psycopg2 is optional at import time
            result["error"] = str(exc)
        return result

    # SchemaManager-compatible delegates used by existing controllers.
    def database_exists(self, dbname: str | None = None) -> bool:
        return self.schema.database_exists(dbname)

    def create_database(self, dbname: str | None = None) -> bool:
        return self.schema.create_database(dbname)

    def connect(self, dbname: str | None = None):
        return self.schema.connect(dbname)

    def plan_schema(
        self,
        *,
        target_version: int | None = None,
        observation_years: Iterable[int] | None = None,
        create_compat_views: bool = True,
    ) -> dict[str, Any]:
        plan = self.schema.plan_migration(target_version=target_version)
        plan["observation_years"] = sorted({int(year) for year in (observation_years or ())})
        plan["create_compat_views"] = bool(create_compat_views)
        plan["connection"] = self.connection_info
        return plan

    def plan(
        self,
        *,
        target_version: int | None = None,
        observation_years: Iterable[int] | None = None,
        manifest: str | Path | Mapping[str, Any] | None = None,
        create_compat_views: bool = True,
    ) -> dict[str, Any]:
        """Return a read-only combined schema/manifest plan."""

        schema_plan = self.plan_schema(
            target_version=target_version,
            observation_years=observation_years,
            create_compat_views=create_compat_views,
        )
        result = dict(schema_plan)
        result["schema"] = schema_plan
        if manifest is not None:
            result["manifest"] = self.plan_manifest(manifest)
        return result

    # Explicit aliases used by older callers and the GUI controller.
    plan_migration = plan_schema
    plan_schema_lifecycle = plan_schema

    def _report_path(self, requested: str | Path | None, prefix: str) -> Path:
        if requested is not None:
            return Path(requested).expanduser().resolve()
        from .constants import MANAGER_LOG_ROOT

        return MANAGER_LOG_ROOT / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.json"

    def _begin_job(
        self,
        kind: str,
        report_path: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        job = self.job_store.create_job(
            kind,
            status="running",
            target_database=self.db_config.get("dbname"),
            report_path=report_path,
            metadata=metadata,
        )
        self.job_store.start_job(job["job_id"])
        return self.job_store.get_job(job["job_id"]) or job

    @staticmethod
    def _schema_report_path(report_path: Path) -> Path:
        return report_path.with_name(report_path.stem + ".schema.json")

    def bootstrap(
        self,
        *,
        create_database_if_missing: bool = True,
        migrate_legacy: bool = True,
        create_compat_views: bool = True,
        verify: bool = True,
        observation_years: Iterable[int] | None = None,
        report_path: str | Path | None = None,
        manifest: str | Path | Mapping[str, Any] | None = None,
        resume: bool = True,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Bootstrap schema, optionally import a manifest, then verify again."""

        output_path = self._report_path(report_path, "run")
        job = self._begin_job("bootstrap", output_path)
        run_id = job["job_id"]
        report: dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "database": self.db_config.get("dbname"),
            "connection": self.connection_info,
            "report_path": str(output_path),
            "started_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "schema": None,
            "manifest": None,
            "verification": None,
            "warnings": [],
        }
        # Persist a report before touching PostgreSQL.  This covers malformed
        # migration resources, unreachable servers and invalid manifests.
        _write_json(output_path, report)
        try:
            manifest_plan = None
            if manifest is not None:
                manifest_plan = (
                    dict(manifest)
                    if isinstance(manifest, Mapping)
                    else self.plan_manifest(manifest)
                )
                report["manifest_plan"] = manifest_plan
                _write_json(output_path, report)

            if progress:
                progress({"phase": "schema", "message": "开始建库/升级", "run_id": run_id})
            schema_result = self.schema.bootstrap_local(
                create_database_if_missing=create_database_if_missing,
                migrate_legacy=migrate_legacy,
                create_compat_views=create_compat_views,
                verify=verify,
                observation_years=observation_years,
                report_path=self._schema_report_path(output_path),
            )
            report["schema"] = schema_result
            report["warnings"].extend(schema_result.get("warnings", []))
            _write_json(output_path, report)

            if manifest_plan is not None:
                if progress:
                    progress({"phase": "manifest", "message": "开始批量导入", "run_id": run_id})
                manifest_result = self._execute_manifest(
                    manifest_plan,
                    resume=resume,
                    progress=progress,
                    cancel=lambda: self.job_store.is_cancel_requested(run_id),
                )
                report["manifest"] = manifest_result
                self._persist_manifest_items(run_id, manifest_result)
                _write_json(output_path, report)

            if verify:
                if progress:
                    progress({"phase": "verify", "message": "执行最终核验", "run_id": run_id})
                report["verification"] = self.schema.verify_schema(
                    require_compat_views=create_compat_views,
                )
                if not report["verification"].get("ok"):
                    report["status"] = "committed_verification_failed"
                    report["error"] = "提交后 schema 核验失败"
                elif report.get("manifest") and report["manifest"].get("status") != "completed":
                    report["status"] = report["manifest"].get("status", "completed_with_errors")
                else:
                    report["status"] = "completed"
            else:
                report["status"] = (
                    report["manifest"].get("status", "completed_with_errors")
                    if report.get("manifest") and report["manifest"].get("status") != "completed"
                    else "completed"
                )
        except Exception as exc:
            report["status"] = self._failure_status(report)
            report["error"] = str(exc)
            # SchemaManager writes its own diagnostic report; include it when
            # available without replacing the original exception.
            schema_path = self._schema_report_path(output_path)
            if schema_path.exists():
                try:
                    report["schema"] = json.loads(schema_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
        finally:
            report["finished_at"] = datetime.now().isoformat(sep=" ", timespec="seconds")
            _write_json(output_path, report)
            self.job_store.finish_job(
                run_id,
                report["status"],
                result=report,
                error=report.get("error"),
            )
        return report

    @staticmethod
    def _failure_status(report: Mapping[str, Any]) -> str:
        schema = report.get("schema")
        if isinstance(schema, Mapping) and schema.get("status") in {
            "failed_before_commit", "committed_verification_failed"
        }:
            return str(schema["status"])
        return "failed_before_commit" if not schema else "failed"

    def bootstrap_local(self, **kwargs: Any) -> dict[str, Any]:
        """Compatibility name for callers migrating from ``SchemaManager``."""

        return self.bootstrap(**kwargs)

    def ensure_dataset(self, table: str, *, report_path: str | Path | None = None) -> dict[str, Any]:
        """Install the contract dependencies required by one logical dataset.

        PostgreSQL fact tables depend on shared dimensions, constraints and
        compatibility views.  The safe operation is therefore an idempotent
        contract bootstrap, not a brittle bare ``CREATE TABLE`` for one name.
        """

        spec = get_dataset(table, writable=False)
        report = self.bootstrap(
            create_database_if_missing=True,
            migrate_legacy=True,
            create_compat_views=True,
            verify=True,
            report_path=report_path,
        )
        report["requested_dataset"] = table
        report["physical_tables"] = list(spec.physical_tables)
        return report

    def _manifest_runner(self):
        try:
            from .import_manifest import ManifestRunner
        except ImportError as exc:
            raise RuntimeError("Manifest 导入模块未安装完整") from exc
        return ManifestRunner(
            job_runner=self.import_runner,
            checkpoint_store=self.job_store,
            target_database=str(self.db_config.get("dbname", "")),
        )

    def plan_manifest(self, path: str | Path) -> dict[str, Any]:
        plan = self._manifest_runner().preflight(path)
        return _serialize(plan)

    load_manifest = plan_manifest
    preflight_manifest = plan_manifest

    def _execute_manifest(
        self,
        plan: Any,
        *,
        resume: bool = True,
        progress: ProgressCallback | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        runner = self._manifest_runner()

        def callback(event: Any) -> None:
            if progress is None:
                return
            if isinstance(event, Mapping):
                progress(dict(event))
            else:
                progress({"phase": "manifest", "message": str(event)})

        result = runner.run(plan, resume=resume, progress=callback, cancel=cancel)
        return _serialize(result)

    def run_manifest(
        self,
        path_or_plan: str | Path | Mapping[str, Any],
        *,
        resume: bool = True,
        progress: ProgressCallback | None = None,
        report_path: str | Path | None = None,
    ) -> dict[str, Any]:
        output_path = self._report_path(report_path, "manifest")
        job = self._begin_job("manifest", output_path)
        run_id = job["job_id"]
        report: dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "database": self.db_config.get("dbname"),
            "report_path": str(output_path),
            "started_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        }
        _write_json(output_path, report)
        try:
            plan = path_or_plan
            if isinstance(path_or_plan, (str, Path)):
                plan = self.plan_manifest(path_or_plan)
            report["plan"] = _serialize(plan)
            report["result"] = self._execute_manifest(
                plan,
                resume=resume,
                progress=progress,
                cancel=lambda: self.job_store.is_cancel_requested(run_id),
            )
            self._persist_manifest_items(run_id, report["result"])
            report["status"] = report["result"].get("status", "completed")
            if report["status"] not in {"completed", "completed_with_errors", "canceled"}:
                report["status"] = "failed"
        except Exception as exc:
            report["status"] = "failed_before_commit"
            report["error"] = str(exc)
        finally:
            report["finished_at"] = datetime.now().isoformat(sep=" ", timespec="seconds")
            _write_json(output_path, report)
            self.job_store.finish_job(
                run_id,
                report["status"],
                result=report,
                error=report.get("error"),
            )
        return report

    def _persist_manifest_items(self, run_id: str, result: Mapping[str, Any]) -> None:
        for position, item in enumerate(result.get("items", ()), start=1):
            record = item.get("record") or {}
            item_key = str(item.get("item_id") or item.get("id") or f"item-{position:04d}")
            identifier = f"{run_id}:{item_key}"
            fields = {
                "source_path": item.get("path") or record.get("source_file"),
                "source_hash": item.get("sha256") or record.get("file_sha256"),
                "target": item.get("target") or item.get("table") or record.get("target_table"),
                "adapter": item.get("adapter") or item.get("import_type") or record.get("import_type"),
                "mode": item.get("mode") or record.get("mode"),
                "position": item.get("position", position),
                "total_rows": record.get("total_rows"),
                "inserted_rows": record.get("inserted_rows"),
                "updated_rows": record.get("updated_rows"),
                "skipped_rows": record.get("skipped_rows"),
                "rejected_rows": record.get("rejected_rows"),
                "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"),
                "error": record.get("error"),
                "payload": dict(item),
            }
            try:
                self.job_store.create_item(
                    run_id,
                    item_id=identifier,
                    item_key=item_key,
                    status=str(item.get("status", "unknown")),
                    **fields,
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "failed to persist manifest item %s", item_key
                )

    def import_manifest(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.run_manifest(*args, **kwargs)

    def preflight_import(
        self,
        path: str | Path,
        *,
        table: str,
        adapter: str | None = None,
        import_type: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        adapter_name = adapter or import_type
        preview = self.import_runner.preview(
            path,
            import_type=adapter_name,
            options=dict(options or {}),
            table=table,
        )
        return _serialize(preview)

    def run_import(
        self,
        path: str | Path,
        *,
        table: str,
        adapter: str | None = None,
        import_type: str | None = None,
        mode: str = "insert_only",
        options: Mapping[str, Any] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        return self.import_runner.run(
            path,
            table=table,
            import_type=adapter or import_type,
            mode=mode,
            options=dict(options or {}),
            progress=progress,
        )

    def verify(
        self,
        *,
        deep: bool = False,
        require_compat_views: bool = True,
        report_path: str | Path | None = None,
    ) -> dict[str, Any]:
        output_path = self._report_path(report_path, "verify")
        job = self._begin_job("verify", output_path)
        report: dict[str, Any] = {
            "run_id": job["job_id"],
            "status": "running",
            "database": self.db_config.get("dbname"),
            "deep": bool(deep),
            "report_path": str(output_path),
        }
        _write_json(output_path, report)
        try:
            report["verification"] = self.schema.verify_schema(
                require_compat_views=require_compat_views
            )
            if deep and report["verification"].get("database_exists"):
                coverage = self.import_client.table_overview(include_checks=True)
                report["coverage"] = coverage
                stale = [row["table_name"] for row in coverage if row.get("coverage_stale")]
                check_errors = [
                    {"table": row["table_name"], "warnings": row.get("warnings", [])}
                    for row in coverage
                    if any("只读覆盖核验失败" in warning for warning in row.get("warnings", []))
                ]
                if stale:
                    report["verification"]["errors"].append(
                        f"dataset_coverage 与只读实查不一致: {stale}"
                    )
                if check_errors:
                    report["verification"]["errors"].append(
                        f"覆盖实查失败: {check_errors}"
                    )
                report["verification"]["ok"] = not report["verification"]["errors"]
            report["status"] = "completed" if report["verification"].get("ok") else "failed"
        except Exception as exc:
            report["status"] = "failed_before_commit"
            report["error"] = str(exc)
        finally:
            report["finished_at"] = datetime.now().isoformat(sep=" ", timespec="seconds")
            _write_json(output_path, report)
            self.job_store.finish_job(job["job_id"], report["status"], result=report, error=report.get("error"))
        return report

    def verify_schema(self, **kwargs: Any) -> dict[str, Any]:
        """Direct compatibility delegate without creating a local verify job."""

        return self.schema.verify_schema(**kwargs)

    def dashboard_snapshot(self) -> dict[str, Any]:
        probe = self.probe_connection()
        snapshot: dict[str, Any] = {
            "connection": probe,
            "schema": None,
            "coverage": [],
            "warnings": [],
        }
        if probe["status"] != "online":
            snapshot["warnings"].append("目标数据库尚未可用，当前为离线状态")
            return snapshot
        try:
            snapshot["schema"] = self.schema.verify_schema(require_compat_views=True)
            snapshot["coverage"] = self.client.table_overview()
            snapshot["warnings"].extend(snapshot["schema"].get("warnings", []))
        except Exception as exc:
            snapshot["warnings"].append(str(exc))
        return snapshot

    def table_schema(self, table: str) -> dict[str, Any]:
        return self.client.table_schema(table)

    def query(self, request: QueryRequest):
        return self.client.query_table(request)

    def execute_readonly_sql(self, query: str, *, limit: int = 5000):
        return self.client.execute_readonly_sql(query, limit=limit)

    def diagnose_data(self, table: str, *, sample_limit: int = 10):
        return self.client.diagnose_data(table, sample_limit=sample_limit)

    def list_jobs(self, *, limit: int = 100, kind: str | None = None, status: str | None = None):
        return self.job_store.list_jobs(limit=limit, kind=kind, status=status)

    @staticmethod
    def read_log(path: str | Path) -> str:
        return Path(path).expanduser().read_text(encoding="utf-8")

    def cancel(self, job_id: str) -> None:
        self.job_store.request_cancel(job_id)


__all__ = ["DatabaseManager", "MANIFEST_TIMEOUT_MS"]
