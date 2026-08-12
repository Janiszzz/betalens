"""Small Qt-free controller for the beginner-oriented desktop GUI."""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .constants import IMPORT_MODES, INSERT_ONLY
from .db import DatabaseClient, QueryRequest
from .import_adapters import ADAPTERS
from .jobs import ImportJobRunner
from .manager import DatabaseManager
from .registry import DATASETS, get_dataset
from .utils import json_default


SUPPORTED_IMPORT_SUFFIXES = (".csv", ".csv.gz", ".xls", ".xlsx", ".parquet", ".pq")
ProgressCallback = Callable[[dict[str, Any]], None]


class BusyOperationError(RuntimeError):
    """Raised when a second database-changing GUI operation starts."""


class StalePlanError(RuntimeError):
    """Raised when a file changed after the user checked it."""


class OperationRegistry:
    def __init__(self) -> None:
        self._active: set[str] = set()
        self._lock = threading.Lock()

    @contextmanager
    def claim(self, name: str) -> Iterator[None]:
        with self._lock:
            if self._active:
                raise BusyOperationError("已有数据库操作正在执行，请等待完成")
            self._active.add(name)
        try:
            yield
        finally:
            with self._lock:
                self._active.discard(name)

    def active(self) -> bool:
        with self._lock:
            return bool(self._active)


@dataclass(frozen=True)
class ConnectionDraft:
    dbname: str
    user: str
    host: str = "localhost"
    port: str = "5432"
    password: str | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ConnectionDraft":
        return cls(
            dbname=str(config.get("dbname") or "").strip(),
            user=str(config.get("user") or "").strip(),
            host=str(config.get("host") or "localhost").strip(),
            port=str(config.get("port") or "5432").strip(),
            password=None if config.get("password") is None else str(config.get("password")),
        )

    def as_config(self) -> dict[str, str]:
        values = {
            "dbname": self.dbname.strip(),
            "user": self.user.strip(),
            "host": self.host.strip(),
            "port": self.port.strip(),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"连接信息不完整: {', '.join(missing)}")
        try:
            port = int(values["port"])
        except ValueError as exc:
            raise ValueError("端口必须是整数") from exc
        if not 1 <= port <= 65535:
            raise ValueError("端口必须位于 1 到 65535")
        values["port"] = str(port)
        if self.password is not None:
            values["password"] = self.password
        return values


@dataclass(frozen=True)
class FileImportItem:
    path: Path
    source_sha256: str | None
    preview_token: str | None
    summary: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)
    rejected_preview: tuple[Mapping[str, Any], ...] = ()
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.error is None and bool(self.validation.get("ok"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file": self.path.name,
            "status": "可导入" if self.ready else "需处理",
            "rows": self.summary.get("rows", 0),
            "rejected_rows": self.summary.get("rejected_rows", 0),
            "codes": self.summary.get("codes", 0),
            "metrics": self.summary.get("metrics", 0),
            "error": self.error or "; ".join(self.validation.get("errors", [])),
        }


@dataclass(frozen=True)
class FileImportPlan:
    table: str
    adapter: str
    mode: str
    options: Mapping[str, Any]
    items: tuple[FileImportItem, ...]
    fingerprint: str
    source_label: str

    @property
    def ready_items(self) -> tuple[FileImportItem, ...]:
        return tuple(item for item in self.items if item.ready)


class GuiController:
    """One-purpose actions for four simple pages, independent of PySide6."""

    def __init__(
        self,
        manager: DatabaseManager | None = None,
        *,
        client: DatabaseClient | None = None,
        runner: ImportJobRunner | None = None,
        manager_factory: Callable[[Mapping[str, Any]], DatabaseManager] = DatabaseManager,
    ) -> None:
        self.manager_factory = manager_factory
        self.manager = manager or DatabaseManager()
        self.client = client or self.manager.client
        self.runner = runner or self.manager.import_runner
        self.connection_state = "offline"
        self.connection_details: dict[str, Any] = {}
        self.operations = OperationRegistry()

    @property
    def connection_draft(self) -> ConnectionDraft:
        return ConnectionDraft.from_config(self.manager.effective_config)

    def connect(self, draft: ConnectionDraft | None = None) -> dict[str, Any]:
        if draft is not None:
            self.manager = self.manager_factory(draft.as_config())
            self.client = self.manager.client
            self.runner = self.manager.import_runner
        result = self.manager.probe_connection()
        self.connection_state = str(result.get("status", "unreachable"))
        self.connection_details = dict(result)
        return result

    def is_online(self) -> bool:
        return self.connection_state == "online"

    def table_catalog(self) -> list[dict[str, Any]]:
        overview: dict[str, dict[str, Any]] = {}
        if self.is_online():
            try:
                overview = {row["table_name"]: row for row in self.client.table_overview()}
            except Exception:
                overview = {}
        rows: list[dict[str, Any]] = []
        for name, spec in DATASETS.items():
            current = overview.get(name, {})
            date_range = current.get("date_range") or {}
            if not self.is_online():
                state = "未连接"
            elif current:
                state = "已建立"
            else:
                # A reachable database can still be an empty or pre-contract
                # database.  Do not claim that the selected dataset exists.
                state = "尚未建立"
            rows.append(
                {
                    "table": name,
                    "storage": spec.storage,
                    "physical_tables": ", ".join(spec.physical_tables),
                    "state": state,
                    "rows": current.get("estimated_rows"),
                    "date_min": date_range.get("min_dt"),
                    "date_max": date_range.get("max_dt"),
                    "warning": "; ".join(current.get("warnings") or []),
                }
            )
        return rows

    def table_metadata(self, table: str) -> dict[str, Any]:
        spec = get_dataset(table)
        if spec.storage == "trade_calendar":
            columns = [
                {"column_name": "exchange", "data_type": "varchar"},
                {"column_name": "trade_date", "data_type": "date"},
            ]
        else:
            columns = [
                {"column_name": name, "data_type": data_type}
                for name, data_type in (
                    ("datetime", "timestamp"),
                    ("code", "varchar"),
                    ("name", "varchar"),
                    ("metric", "varchar"),
                    ("value", "double precision"),
                    ("remark", "jsonb object"),
                )
            ]
        contract = {
            "logical_table": table,
            "storage": spec.storage,
            "physical_tables": list(spec.physical_tables),
            "columns": columns,
        }
        if not self.is_online():
            return contract
        try:
            current = self.client.table_schema(table)
        except Exception as exc:
            contract["warning"] = str(exc)
            return contract
        return {**contract, **current}

    def create_selected_table(self, table: str) -> dict[str, Any]:
        get_dataset(table)
        with self.operations.claim("create-table"):
            report = self.manager.ensure_dataset(table)
        connection = self.manager.probe_connection()
        self.connection_state = str(connection.get("status", "unreachable"))
        self.connection_details = dict(connection)
        return report

    @staticmethod
    def discover_files(paths: Sequence[str | Path]) -> list[Path]:
        files: set[Path] = set()
        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()
            if path.is_file():
                candidates: Iterable[Path] = (path,)
            elif path.is_dir():
                candidates = (item for item in path.rglob("*") if item.is_file())
            else:
                continue
            for item in candidates:
                name = item.name.lower()
                if name.endswith(SUPPORTED_IMPORT_SUFFIXES):
                    files.add(item.resolve())
        return sorted(files, key=lambda item: str(item).casefold())

    @staticmethod
    def _plan_fingerprint(
        table: str,
        adapter: str,
        mode: str,
        options: Mapping[str, Any],
        items: Iterable[FileImportItem],
    ) -> str:
        payload = {
            "table": table,
            "adapter": adapter,
            "mode": mode,
            "options": dict(options),
            "items": [
                {"path": str(item.path), "sha256": item.source_sha256,
                 "token": item.preview_token}
                for item in items
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default).encode("utf-8")
        ).hexdigest()

    def preflight_import(
        self,
        paths: Sequence[str | Path],
        *,
        table: str,
        adapter: str,
        mode: str = INSERT_ONLY,
        options: Mapping[str, Any] | None = None,
        progress: ProgressCallback | None = None,
    ) -> FileImportPlan:
        if not self.is_online():
            raise RuntimeError("请先连接数据库")
        get_dataset(table, writable=True)
        if mode not in IMPORT_MODES:
            raise ValueError(f"未知写入模式: {mode}")
        normalized_options = dict(options or {})
        ADAPTERS.validate(adapter, table, normalized_options, strict_options=True)
        files = self.discover_files(paths)
        if not files:
            raise ValueError("未找到可导入的 CSV、Excel 或 Parquet 文件")
        items: list[FileImportItem] = []
        with self.operations.claim("check-files"):
            for position, path in enumerate(files, start=1):
                if progress is not None:
                    progress({"current": position, "total": len(files), "message": f"检查 {path.name}"})
                try:
                    preview = self.runner.preview(
                        path,
                        import_type=adapter,
                        options=normalized_options,
                        table=table,
                        mode=mode,
                        inspect_database=False,
                    )
                    items.append(
                        FileImportItem(
                            path=path,
                            source_sha256=preview.get("source_sha256"),
                            preview_token=preview.get("preview_token"),
                            summary=preview.get("summary") or {},
                            validation=preview.get("validation") or {},
                            rejected_preview=tuple(
                                dict(row)
                                for row in preview.get("rejected_preview", [])
                                if isinstance(row, Mapping)
                            ),
                        )
                    )
                except Exception as exc:
                    items.append(FileImportItem(path=path, source_sha256=None, preview_token=None, error=str(exc)))
        return FileImportPlan(
            table=table,
            adapter=adapter,
            mode=mode,
            options=normalized_options,
            items=tuple(items),
            fingerprint=self._plan_fingerprint(table, adapter, mode, normalized_options, items),
            source_label="; ".join(str(Path(value).expanduser()) for value in paths),
        )

    def run_import_plan(
        self,
        plan: FileImportPlan,
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        current = self._plan_fingerprint(
            plan.table, plan.adapter, plan.mode, plan.options, plan.items
        )
        if current != plan.fingerprint:
            raise StalePlanError("文件预检结果已失效，请重新检查文件")
        records: list[dict[str, Any]] = []
        with self.operations.claim("import-files"):
            for position, item in enumerate(plan.items, start=1):
                if not item.ready:
                    records.append(
                        {"path": str(item.path), "status": "skipped", "error": item.error or "文件未通过检查"}
                    )
                    continue
                if progress is not None:
                    progress({"current": position, "total": len(plan.items), "message": f"导入 {item.path.name}"})
                try:
                    record = self.runner.run(
                        item.path,
                        table=plan.table,
                        import_type=plan.adapter,
                        mode=plan.mode,
                        options=dict(plan.options),
                        expected_sha256=item.source_sha256,
                        preview_token=item.preview_token,
                        on_rejected="fail",
                    )
                    records.append(record)
                except Exception as exc:
                    # Each file is independently atomic in ImportJobRunner.
                    # Keep working through a folder so one bad spreadsheet
                    # does not hide successful imports from the user.
                    records.append(
                        {
                            "path": str(item.path),
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
        failures = [record for record in records if record.get("status") not in {"completed", "skipped"}]
        skipped = [record for record in records if record.get("status") == "skipped"]
        return {
            "status": "completed" if not failures and not skipped else "completed_with_errors",
            "table": plan.table,
            "adapter": plan.adapter,
            "total_files": len(records),
            "completed_files": sum(record.get("status") == "completed" for record in records),
            "failed_files": len(failures),
            "skipped_files": len(skipped),
            "items": records,
        }

    def query(self, request: QueryRequest):
        if not self.is_online():
            raise RuntimeError("请先连接数据库")
        return self.client.query_table(request)

    def execute_sql(self, statement: str, *, limit: int = 5000):
        if not self.is_online():
            raise RuntimeError("请先连接数据库")
        return self.client.execute_readonly_sql(statement, limit=limit)

    def diagnose_dirty_data(self, table: str, *, sample_limit: int = 10) -> list[dict[str, Any]]:
        if not self.is_online():
            raise RuntimeError("请先连接数据库")
        return self.client.diagnose_data(table, sample_limit=sample_limit)


__all__ = [
    "BusyOperationError",
    "ConnectionDraft",
    "FileImportItem",
    "FileImportPlan",
    "GuiController",
    "OperationRegistry",
    "SUPPORTED_IMPORT_SUFFIXES",
    "StalePlanError",
]
