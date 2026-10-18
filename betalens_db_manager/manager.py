"""Unified database-manager service facade.

The facade is deliberately Qt-free.  CLI, BAT and the desktop application all
call these methods for read-only structure inspection, manifest preflight
and file imports. Schema migrations and bootstrap are retired.
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
from .schema import SchemaManager, migration_disabled
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
    """Inspect existing datasets and manage file imports."""

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
        self, *, target_version: int | None = None,
        observation_years: Iterable[int] | None = None,
        create_compat_views: bool = True,
    ) -> dict[str, Any]:
        """Inspect the installed structure without planning any DDL."""
        if target_version is not None or observation_years is not None or not create_compat_views:
            migration_disabled()
        result = self.inspect_schema()
        result["connection"] = self.connection_info
        return result

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
    plan_migration = staticmethod(migration_disabled)
    plan_schema_lifecycle = staticmethod(migration_disabled)

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

    def bootstrap(self, **kwargs):
        """Retired; use run_manifest() to import into installed tables."""
        migration_disabled()

    def bootstrap_local(self, **kwargs):
        migration_disabled()

    def ensure_dataset(self, table: str, *, report_path: str | Path | None = None) -> dict[str, Any]:
        """Check the selected dataset without creating any database objects."""
        spec = get_dataset(table, writable=False)
        report = self.schema.inspect_schema([table])
        report["requested_dataset"] = table
        report["physical_tables"] = list(spec.physical_tables)
        if report_path is not None:
            _write_json(Path(report_path).expanduser().resolve(), report)
        if not report["ok"]:
            raise RuntimeError("; ".join(report["errors"]))
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

    def verify(self, **kwargs):
        """Retired migration verification; use inspect_schema()."""
        migration_disabled()

    def verify_schema(self, **kwargs):
        migration_disabled()

    def inspect_schema(self) -> dict[str, Any]:
        return self.schema.inspect_schema()

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
            snapshot["schema"] = self.inspect_schema()
            snapshot["coverage"] = self.client.table_overview()
            snapshot["warnings"].extend(snapshot["schema"].get("warnings", []))
            snapshot["warnings"].extend(snapshot["schema"].get("errors", []))
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
