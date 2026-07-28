"""Versioned, validated and resumable manifest orchestration."""

from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import pandas as pd
import yaml

from .constants import INSERT_ONLY, IMPORT_MODES
from .contracts import LATEST_SCHEMA_VERSION
from .import_adapters import ADAPTERS, infer_adapter, load_import_batches
from .job_store import JobStore
from .jobs import ImportJobRunner
from .utils import file_sha256, json_default
from .validators import validate_import_frame


ProgressCallback = Callable[[dict[str, Any]], None]
CancelCallback = Callable[[], bool]
_GLOB_CHARS = frozenset("*?[")
_ENTRY_KEYS = frozenset(
    (
        "id", "path", "target", "table", "adapter", "import_type", "mode",
        "options", "allow_unsafe_metrics", "on_rejected", "sha256",
    )
)


class CheckpointStore(Protocol):
    def load(self, token: str) -> dict[str, Any] | None: ...

    def save(self, token: str, payload: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class ManifestEntry:
    item_id: str
    group_id: str
    position: int
    path: Path
    target: str
    adapter: str
    mode: str = INSERT_ONLY
    options: dict[str, Any] = field(default_factory=dict)
    allow_unsafe_metrics: bool = False
    on_rejected: str = "fail"
    sha256: str = ""
    preview_token: str = ""

    @property
    def table(self) -> str:
        return self.target

    @property
    def import_type(self) -> str:
        return self.adapter

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "item_id": self.item_id,
            "group_id": self.group_id,
            "position": self.position,
            "path": str(self.path),
            "target": self.target,
            "table": self.target,
            "adapter": self.adapter,
            "import_type": self.adapter,
            "mode": self.mode,
            "options": dict(self.options),
            "allow_unsafe_metrics": self.allow_unsafe_metrics,
            "on_rejected": self.on_rejected,
            "sha256": self.sha256,
            "preview_token": self.preview_token,
        }


@dataclass(frozen=True)
class ManifestPlan:
    version: int
    path: Path
    token: str
    entries: tuple[ManifestEntry, ...]
    target_database: str
    schema_version: int
    on_error: str = "continue"
    warnings: tuple[str, ...] = ()
    previews: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "path": str(self.path),
            "token": self.token,
            "target_database": self.target_database,
            "schema_version": self.schema_version,
            "on_error": self.on_error,
            "entries": [entry.as_dict() for entry in self.entries],
            "warnings": list(self.warnings),
            "previews": list(self.previews),
        }


class ManifestRunner:
    """Preflight every source, then commit one transaction per source file."""

    ROOT_KEYS = frozenset(("version", "defaults", "imports", "on_error"))

    def __init__(
        self,
        job_runner: ImportJobRunner | None = None,
        checkpoint_store: CheckpointStore | None = None,
        *,
        target_database: str | None = None,
        schema_version: int = LATEST_SCHEMA_VERSION,
    ) -> None:
        self.job_runner = job_runner or ImportJobRunner()
        inherited_store = getattr(getattr(self.job_runner, "store", None), "job_store", None)
        self.checkpoint_store = checkpoint_store or inherited_store or JobStore()
        client_config = getattr(getattr(self.job_runner, "client", None), "db_config", {})
        self.target_database = str(target_database or client_config.get("dbname") or "")
        self.schema_version = int(schema_version)

    @staticmethod
    def _normalize_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
        result = dict(options or {})
        if "chunk_rows" in result:
            if "chunk_size" in result and int(result["chunk_size"]) != int(result["chunk_rows"]):
                raise ValueError("options.chunk_rows 与 chunk_size 不能冲突")
            result["chunk_size"] = int(result.pop("chunk_rows"))
        return result

    @staticmethod
    def _expand_path(manifest_path: Path, value: Any) -> list[Path]:
        source_text = str(value).strip()
        if not source_text:
            raise ValueError("manifest path 不能为空")
        source = Path(source_text).expanduser()
        if not source.is_absolute():
            source = manifest_path.parent / source
        pattern = str(source)
        is_glob = any(character in pattern for character in _GLOB_CHARS)
        if is_glob:
            matches = [Path(item).resolve() for item in glob.glob(pattern, recursive=True)]
            files = sorted({item for item in matches if item.is_file()}, key=lambda item: str(item).casefold())
            if not files:
                raise FileNotFoundError(f"glob 未匹配任何文件: {source_text}")
            return files
        resolved = source.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"导入文件不存在: {resolved}")
        return [resolved]

    @staticmethod
    def _effective_value(raw: Mapping[str, Any], defaults: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in raw and raw[key] is not None:
                return raw[key]
        for key in keys:
            if key in defaults and defaults[key] is not None:
                return defaults[key]
        return None

    @staticmethod
    def _expected_hash(value: Any, source: Path, sources: list[Path]) -> str | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            candidates = (
                str(source), source.as_posix(), source.name,
                str(source.resolve()), source.resolve().as_posix(),
            )
            for candidate in candidates:
                if candidate in value:
                    return str(value[candidate])
            raise ValueError(f"sha256 映射缺少文件: {source}")
        if len(sources) != 1:
            raise ValueError("glob 匹配多个文件时 sha256 必须是按路径/文件名索引的映射")
        return str(value)

    def preflight(self, path: str | Path) -> ManifestPlan:
        manifest_path = Path(path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"导入清单不存在: {manifest_path}")
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        legacy = isinstance(payload, list)
        if legacy:
            root: dict[str, Any] = {"imports": payload}
        elif isinstance(payload, dict):
            root = dict(payload)
        else:
            raise ValueError("导入清单必须是列表或包含 imports 列表的对象")
        unknown_root = sorted(set(root) - self.ROOT_KEYS)
        if unknown_root:
            raise ValueError(f"导入清单包含未知字段: {unknown_root}")

        version = int(root.get("version", 0 if legacy or "version" not in root else -1))
        if version not in {0, 1}:
            raise ValueError(f"不支持的 manifest version: {version}; 当前仅支持 version: 1")
        defaults = root.get("defaults") or {}
        if not isinstance(defaults, dict):
            raise ValueError("manifest.defaults 必须是对象")
        unknown_defaults = sorted(set(defaults) - _ENTRY_KEYS)
        if unknown_defaults:
            raise ValueError(f"manifest.defaults 包含未知字段: {unknown_defaults}")
        imports = root.get("imports")
        if not isinstance(imports, list) or not imports:
            raise ValueError("导入清单 imports 必须是非空列表")
        on_error = str(root.get("on_error", "continue"))
        if on_error not in {"stop", "continue"}:
            raise ValueError("manifest.on_error 必须是 stop 或 continue")

        warnings: list[str] = []
        if version == 0:
            warnings.append("旧版 manifest 已兼容读取；建议补 version: 1 并改用 target/adapter")
        entries: list[ManifestEntry] = []
        group_ids: set[str] = set()
        source_owners: dict[Path, str] = {}
        previews: list[dict[str, Any]] = []

        for source_position, raw_value in enumerate(imports, start=1):
            if not isinstance(raw_value, dict):
                raise ValueError(f"导入清单第 {source_position} 项必须是对象")
            raw = dict(raw_value)
            unknown = sorted(set(raw) - _ENTRY_KEYS)
            if unknown:
                raise ValueError(f"导入清单第 {source_position} 项包含未知字段: {unknown}")
            if not self._effective_value(raw, defaults, "path"):
                raise ValueError(f"导入清单第 {source_position} 项必须包含 path")
            group_id_value = self._effective_value(raw, defaults, "id")
            if version == 1 and not group_id_value:
                raise ValueError(f"导入清单第 {source_position} 项必须包含 id")
            group_id = str(group_id_value or f"item-{source_position:04d}")
            if group_id in group_ids:
                raise ValueError(f"导入清单 item id 重复: {group_id}")
            group_ids.add(group_id)

            target_value = self._effective_value(raw, defaults, "target", "table")
            adapter_value = self._effective_value(raw, defaults, "adapter", "import_type")
            if not target_value:
                raise ValueError(f"导入清单第 {source_position} 项必须包含 target（旧名 table）")
            target = str(target_value)
            sources = self._expand_path(
                manifest_path,
                self._effective_value(raw, defaults, "path"),
            )
            if not adapter_value:
                suggestion = infer_adapter(sources[0])
                raise ValueError(
                    f"导入清单第 {source_position} 项必须显式包含 adapter（建议: {suggestion}）；"
                    "文件名只能用于建议，不能决定实际导入类型"
                )
            adapter = str(adapter_value)
            mode = str(self._effective_value(raw, defaults, "mode") or INSERT_ONLY)
            if mode not in IMPORT_MODES:
                raise ValueError(f"导入清单第 {source_position} 项 mode 非法: {mode}")
            default_options = defaults.get("options") or {}
            item_options = raw.get("options") or {}
            if not isinstance(default_options, dict) or not isinstance(item_options, dict):
                raise ValueError(f"导入清单第 {source_position} 项 options 必须是对象")
            options = self._normalize_options({**default_options, **item_options})
            ADAPTERS.validate(adapter, target, options, strict_options=True)
            on_rejected = str(self._effective_value(raw, defaults, "on_rejected") or "fail")
            if on_rejected not in {"fail", "skip"}:
                raise ValueError(f"导入清单第 {source_position} 项 on_rejected 必须是 fail 或 skip")
            allow_unsafe = bool(self._effective_value(raw, defaults, "allow_unsafe_metrics") or False)
            hash_spec = self._effective_value(raw, defaults, "sha256")

            for file_position, source in enumerate(sources, start=1):
                previous = source_owners.get(source)
                if previous:
                    raise ValueError(f"同一源文件被重复安排: {source}; items={previous},{group_id}")
                source_owners[source] = group_id
                digest = file_sha256(source)
                expected = self._expected_hash(hash_spec, source, sources)
                if expected and expected.lower() != digest.lower():
                    raise ValueError(
                        f"SHA256 不匹配: {source}; expected={expected}, actual={digest}"
                    )
                item_id = group_id if len(sources) == 1 else f"{group_id}:{file_position:04d}"
                preview_token = ImportJobRunner.preview_token(digest, target, adapter, mode, options)
                entry = ManifestEntry(
                    item_id=item_id,
                    group_id=group_id,
                    position=len(entries) + 1,
                    path=source,
                    target=target,
                    adapter=adapter,
                    mode=mode,
                    options=dict(options),
                    allow_unsafe_metrics=allow_unsafe,
                    on_rejected=on_rejected,
                    sha256=digest,
                    preview_token=preview_token,
                )
                preview = self._preview_source(entry)
                if preview["rejected_rows"] and on_rejected == "fail":
                    raise ValueError(
                        f"预检发现 rejected rows: {source}; count={preview['rejected_rows']}; "
                        f"sample={preview['rejected_preview'][:5]}"
                    )
                if source.suffix.lower() in {".xls", ".xlsx"} and source.stat().st_size >= 100 * 1024 * 1024:
                    warnings.append(f"大型 Excel 将整表读取，建议转 CSV/Parquet: {source}")
                entries.append(entry)
                previews.append(preview)

        token_payload = {
            "manifest": str(manifest_path),
            "version": 1,
            "target_database": self.target_database,
            "schema_version": self.schema_version,
            "on_error": on_error,
            "entries": [entry.as_dict() for entry in entries],
        }
        token = hashlib.sha256(
            json.dumps(
                token_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=json_default,
            ).encode("utf-8")
        ).hexdigest()
        return ManifestPlan(
            version=1,
            path=manifest_path,
            token=token,
            entries=tuple(entries),
            target_database=self.target_database,
            schema_version=self.schema_version,
            on_error=on_error,
            warnings=tuple(warnings),
            previews=tuple(previews),
        )

    def _preview_source(self, entry: ManifestEntry) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "source_rows": 0,
            "valid_rows": 0,
            "rejected_rows": 0,
            "codes": set(),
            "metrics": set(),
            "date_min": None,
            "date_max": None,
        }
        rejected_preview: list[dict[str, Any]] = []
        row_preview: list[dict[str, Any]] = []
        for batch in load_import_batches(
            entry.adapter,
            entry.path,
            table=entry.target,
            options=entry.options,
            strict_options=True,
        ):
            stats["source_rows"] += int(batch.source_rows)
            stats["valid_rows"] += len(batch.frame)
            stats["rejected_rows"] += len(batch.rejected)
            stats["codes"].update(batch.frame["code"].dropna().astype(str).unique())
            stats["metrics"].update(batch.frame["metric"].dropna().astype(str).unique())
            if len(row_preview) < 20:
                row_preview.extend(
                    batch.frame.head(20 - len(row_preview)).where(batch.frame.notna(), None).to_dict("records")
                )
            if len(rejected_preview) < 20 and not batch.rejected.empty:
                columns = [
                    column for column in
                    ("source_file", "source_row", "field", "raw_value", "reason")
                    if column in batch.rejected.columns
                ]
                rejected_preview.extend(
                    batch.rejected[columns].head(20 - len(rejected_preview)).to_dict("records")
                )
            if not batch.frame.empty:
                validation = validate_import_frame(
                    batch.frame,
                    allow_unsafe_metrics=entry.allow_unsafe_metrics,
                    allow_nan_values=False,
                )
                if not validation.ok:
                    raise ValueError(
                        f"文件预检失败: {entry.path}: {'; '.join(validation.errors)}"
                    )
                minimum = pd.Timestamp(batch.frame["datetime"].min())
                maximum = pd.Timestamp(batch.frame["datetime"].max())
                stats["date_min"] = minimum if stats["date_min"] is None else min(stats["date_min"], minimum)
                stats["date_max"] = maximum if stats["date_max"] is None else max(stats["date_max"], maximum)
        return {
            "item_id": entry.item_id,
            "path": str(entry.path),
            "source_rows": int(stats["source_rows"]),
            "valid_rows": int(stats["valid_rows"]),
            "rejected_rows": int(stats["rejected_rows"]),
            "codes": len(stats["codes"]),
            "metrics": len(stats["metrics"]),
            "date_min": stats["date_min"],
            "date_max": stats["date_max"],
            "preview": row_preview,
            "rejected_preview": rejected_preview,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> ManifestPlan:
        entries = tuple(
            ManifestEntry(
                item_id=str(item.get("item_id") or item.get("id")),
                group_id=str(item.get("group_id") or item.get("item_id") or item.get("id")),
                position=int(item["position"]),
                path=Path(item["path"]),
                target=str(item.get("target") or item.get("table")),
                adapter=str(item.get("adapter") or item.get("import_type")),
                mode=str(item.get("mode", INSERT_ONLY)),
                options=dict(item.get("options") or {}),
                allow_unsafe_metrics=bool(item.get("allow_unsafe_metrics", False)),
                on_rejected=str(item.get("on_rejected", "fail")),
                sha256=str(item.get("sha256", "")),
                preview_token=str(item.get("preview_token", "")),
            )
            for item in payload.get("entries", ())
        )
        return ManifestPlan(
            version=int(payload.get("version", 1)),
            path=Path(payload["path"]),
            token=str(payload["token"]),
            entries=entries,
            target_database=str(payload.get("target_database", "")),
            schema_version=int(payload.get("schema_version", LATEST_SCHEMA_VERSION)),
            on_error=str(payload.get("on_error", "continue")),
            warnings=tuple(payload.get("warnings", ())),
            previews=tuple(payload.get("previews", ())),
        )

    def run(
        self,
        path_or_plan: str | Path | ManifestPlan | Mapping[str, Any],
        *,
        resume: bool = True,
        on_error: str | None = None,
        progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> dict[str, Any]:
        if isinstance(path_or_plan, ManifestPlan):
            plan = path_or_plan
        elif isinstance(path_or_plan, Mapping):
            plan = self.from_dict(path_or_plan)
        else:
            plan = self.preflight(path_or_plan)
        if plan.target_database != self.target_database or plan.schema_version != self.schema_version:
            raise ValueError("manifest plan 的目标数据库或 schema 版本已变化，请重新预检")
        policy = on_error or plan.on_error
        if policy not in {"stop", "continue"}:
            raise ValueError("on_error 必须是 stop 或 continue")
        checkpoint = self.checkpoint_store.load(plan.token) if resume else None
        checkpoint_items = dict((checkpoint or {}).get("items", {}))
        completed = {
            key for key, value in checkpoint_items.items()
            if value.get("status") == "completed" and value.get("sha256")
        }
        result: dict[str, Any] = {
            "status": "running",
            "path": str(plan.path),
            "token": plan.token,
            "target_database": plan.target_database,
            "schema_version": plan.schema_version,
            "on_error": policy,
            "total": len(plan.entries),
            "completed": 0,
            "failed": 0,
            "resumed": 0,
            "canceled": 0,
            "items": [],
        }
        self._emit(progress, {"phase": "manifest_start", "token": plan.token, "total": len(plan.entries)})
        stopped = False
        for entry in plan.entries:
            if stopped:
                result["items"].append({**entry.as_dict(), "status": "unattempted"})
                continue
            if cancel and cancel():
                result["items"].append({**entry.as_dict(), "status": "canceled"})
                result["canceled"] += 1
                stopped = True
                continue
            current_hash = file_sha256(entry.path)
            current_token = ImportJobRunner.preview_token(
                current_hash, entry.target, entry.adapter, entry.mode, entry.options
            )
            if current_hash.lower() != entry.sha256.lower() or current_token != entry.preview_token:
                raise ValueError(f"预检后文件或目标发生变化，请重新预检: {entry.path}")
            checkpoint_item = checkpoint_items.get(entry.item_id, {})
            if (
                resume
                and entry.item_id in completed
                and checkpoint_item.get("sha256", "").lower() == entry.sha256.lower()
            ):
                item = {
                    **entry.as_dict(),
                    "status": "resumed",
                    "record": checkpoint_item.get("record"),
                }
                result["items"].append(item)
                result["completed"] += 1
                result["resumed"] += 1
                self._emit(progress, {"phase": "item_resumed", "item_id": entry.item_id, "position": entry.position})
                continue
            self._emit(progress, {"phase": "item_start", "item_id": entry.item_id, "position": entry.position})
            record = self.job_runner.run(
                entry.path,
                table=entry.target,
                import_type=entry.adapter,
                mode=entry.mode,
                options=entry.options,
                allow_unsafe_metrics=entry.allow_unsafe_metrics,
                on_rejected=entry.on_rejected,
                expected_sha256=entry.sha256,
                preview_token=entry.preview_token,
                cancel_check=cancel,
            )
            status = str(record.get("status", "failed"))
            result["items"].append({**entry.as_dict(), "status": status, "record": record})
            checkpoint_items[entry.item_id] = {
                "status": status,
                "sha256": entry.sha256,
                "record": record,
            }
            self.checkpoint_store.save(
                plan.token,
                {
                    "status": "running",
                    "target_database": plan.target_database,
                    "schema_version": plan.schema_version,
                    "manifest": str(plan.path),
                    "items": checkpoint_items,
                },
            )
            if status == "completed":
                result["completed"] += 1
                self._emit(progress, {"phase": "item_complete", "item_id": entry.item_id, "position": entry.position})
            elif status == "canceled":
                result["canceled"] += 1
                stopped = True
            else:
                result["failed"] += 1
                self._emit(
                    progress,
                    {"phase": "item_failed", "item_id": entry.item_id, "position": entry.position, "error": record.get("error")},
                )
                if policy == "stop":
                    stopped = True

        if result["canceled"]:
            result["status"] = "canceled"
        elif result["failed"]:
            result["status"] = "completed_with_errors" if policy == "continue" else "failed"
        else:
            result["status"] = "completed"
        result["imports"] = list(result["items"])
        self.checkpoint_store.save(
            plan.token,
            {
                "status": result["status"],
                "target_database": plan.target_database,
                "schema_version": plan.schema_version,
                "manifest": str(plan.path),
                "items": checkpoint_items,
                "result": result,
            },
        )
        self._emit(progress, {"phase": "manifest_complete", "status": result["status"], "token": plan.token})
        return result

    @staticmethod
    def _emit(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
        if callback is not None:
            callback(event)


__all__ = [
    "CancelCallback",
    "CheckpointStore",
    "ManifestEntry",
    "ManifestPlan",
    "ManifestRunner",
    "ProgressCallback",
]

