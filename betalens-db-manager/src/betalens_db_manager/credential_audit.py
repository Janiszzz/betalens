"""Read-only inventory of credentials in legacy manager configuration and tasks.

This deliberately does not open archives or repair persisted records. An audit
finding is a location and a migration action, never a copy or hash of a value.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

import yaml

from .security import auth_reference, contains_known_secret, sensitive_key
from .util import digest


_SAFE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,79}\Z")
_INLINE_CREDENTIAL = re.compile(
    r"(?i)(?:[?&]|\b)(?:api[_-]?key|access[_-]?key|password|passwd|secret|"
    r"token|authorization|credential)\s*[:=]"
)
_URL_USERINFO = re.compile(r"(?i)\bhttps?://[^\s/@]+:[^\s/@]+@")
_BEARER = re.compile(r"(?i)\bBearer\s+\S+")
_TASK_COLUMNS = {
    "jobs": ("plan", "preview", "error"),
    "slices": ("request", "result"),
    "tasks": ("payload", "error"),
    "schedules": ("request",),
    "event_log": ("value",),
    "ledger": ("value",),
}
_PARAMETER_CONTAINERS = {"parameters", "params", "headers", "query", "body"}
_MAX_CONFIG_BYTES = 8 * 1024 * 1024


class _BudgetExceeded(Exception):pass


def _safe_name(value: object, known: tuple[str, ...]) -> str:
    """Keep untrusted map keys and filenames from becoming an output channel."""
    name = str(value)
    return name if _SAFE_NAME.fullmatch(name) and not contains_known_secret(name, secrets=known) else "<field>"


def _pointer(base: str, part: object, known: tuple[str, ...]) -> str:
    label = str(part) if type(part) is int else _safe_name(part, known)
    return base + "/" + label.replace("~", "~0").replace("/", "~1")


def _is_confirmation_token(item: Mapping, name: object, value: object) -> bool:
    if name != "token" or not isinstance(value, str):
        return False
    try:
        return value == digest({key: nested for key, nested in item.items() if key != "token"})
    except (TypeError, ValueError):
        return False


def _inline_credential(value: str) -> bool:
    return bool(_INLINE_CREDENTIAL.search(value) or _URL_USERINFO.search(value) or _BEARER.search(value))


class _Inventory:
    def __init__(self, settings, max_values=2000000, max_seconds=60):
        self.settings = settings
        self.known = tuple(value for value in (
            settings.token, settings.pg_password, settings.ch_password
        ) if isinstance(value, str) and value)
        self.auth_names: dict[str, set[str]] = {}
        self.confirmation_tokens=set()
        self.findings: dict[tuple[str, str, str], dict] = {}
        self.warnings: list[dict] = []
        if max_values<=0 or max_seconds<=0:raise ValueError('扫描预算必须为正数')
        self.max_values=max_values;self.deadline=time.monotonic()+max_seconds;self.scanned_values=0

    def _name(self, value: object) -> str:
        return _safe_name(value, self.known)

    def _warning(self, file: str, kind: str):
        # Parser and SQLite exception messages can include the offending value.
        self.warnings.append({"file": file, "kind": kind})

    def _finding(self, scope: str, file: str, field: str, reason: str):
        key = (scope, file, field)
        finding = self.findings.setdefault(key, {
            "scope": scope, "file": file, "field": field, "reasons": [],
            "action": "move_to_private_env_and_use_auth_reference" if scope == "config"
                      else "review_historical_record_without_rewriting_archive_or_hash",
        })
        if reason not in finding["reasons"]:
            finding["reasons"].append(reason)

    def _collect_auth(self, document: object):
        if isinstance(document, Mapping):
            source_id = document.get("source_id")
            if isinstance(source_id, str):
                try:
                    reference = auth_reference(document.get("auth"))
                except ValueError:
                    reference = None
                if reference:
                    self.auth_names.setdefault(source_id, set()).add(reference["name"].split(".")[-1])
            for nested in document.values():
                self._collect_auth(nested)
        elif isinstance(document, (list, tuple)):
            for nested in document:
                self._collect_auth(nested)

    def _scan(self, item: object, scope: str, file: str, field: str,
              source_id: str | None = None, in_parameters: bool = False,
              protected_column: bool = False):
        self.scanned_values+=1
        if self.scanned_values>self.max_values or time.monotonic()>self.deadline:
            raise _BudgetExceeded()
        if isinstance(item, Mapping):
            local_source = item.get("source_id")
            if isinstance(local_source, str):
                source_id = local_source
            reserved = self.auth_names.get(source_id or "", set())
            fields = item.get("fields")
            protected_indices = {
                index for index, name in enumerate(fields)
                if isinstance(name, str) and (sensitive_key(name) or
                    (in_parameters and name in reserved))
            } if isinstance(fields, list) else set()
            for name, nested in item.items():
                child = _pointer(field, name, self.known)
                if name == "auth" and isinstance(nested, Mapping):
                    try:
                        if auth_reference(nested) is not None:
                            continue  # A reference contains names, not a credential.
                    except ValueError:
                        pass
                sensitive = sensitive_key(name)
                if isinstance(name, str) and name.endswith("_configured") and isinstance(nested, bool):
                    sensitive = False
                if _is_confirmation_token(item, name, nested):
                    sensitive = False
                if name=='token' and isinstance(nested,str) and nested in self.confirmation_tokens:
                    sensitive=False
                if sensitive and nested not in (None, "", {}, []):
                    self._finding(scope, file, child, "sensitive_field")
                if in_parameters and name in reserved and nested not in (None, "", {}, []):
                    self._finding(scope, file, child, "declared_auth_field")
                if name == "items" and protected_indices and isinstance(nested, list):
                    for row_index, row in enumerate(nested):
                        if isinstance(row, (list, tuple)):
                            for column_index, value in enumerate(row):
                                self._scan(value, scope, file,
                                           _pointer(_pointer(child, row_index, self.known), column_index, self.known),
                                           source_id, in_parameters, column_index in protected_indices)
                        else:
                            self._scan(row, scope, file, _pointer(child, row_index, self.known),
                                       source_id, in_parameters)
                    continue
                self._scan(nested, scope, file, child, source_id,
                           in_parameters or name in _PARAMETER_CONTAINERS)
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                self._scan(nested, scope, file, _pointer(field, index, self.known),
                           source_id, in_parameters, protected_column)
        else:
            if protected_column and item not in (None, ""):
                self._finding(scope, file, field, "sensitive_column")
            if isinstance(item, str):
                if contains_known_secret(item, secrets=self.known):
                    self._finding(scope, file, field, "resolved_secret_value")
                elif _inline_credential(item):
                    self._finding(scope, file, field, "embedded_credential_marker")

    def _yaml(self, path: Path, file: str):
        if path.is_symlink():
            self._warning(file, "symlink_skipped")
            return None
        try:
            if path.stat().st_size > _MAX_CONFIG_BYTES:
                self._warning(file, "config_too_large")
                return None
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            self._collect_auth(document)
            return document
        except (OSError, UnicodeError, yaml.YAMLError, ValueError):
            self._warning(file, "config_unreadable")
            return None

    def _config_documents(self):
        root = self.settings.config_dir
        paths = [root / "sources.yaml"]
        for directory in (root / "sources", root / "mappings"):
            if directory.is_dir() and not directory.is_symlink():
                paths.extend(sorted(directory.glob("*.yaml")))
                paths.extend(sorted(directory.glob("*.yml")))
        documents = []
        for path in paths:
            if not path.exists() and not path.is_symlink():
                continue
            parent = path.parent.name
            file = (self._name(parent) + "/" if path.parent != root else "") + self._name(path.stem) + path.suffix
            document = self._yaml(path, file)
            if document is not None:
                documents.append((file, document))
        return documents

    def _state(self):
        path = self.settings.state_dir / "tasks.sqlite3"
        if not path.exists() and not path.is_symlink():
            return
        if path.is_symlink():
            self._warning("tasks.sqlite3", "symlink_skipped")
            return
        try:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
                conn.execute("PRAGMA query_only=ON")
                existing = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if 'jobs' in existing:
                    columns={row[1] for row in conn.execute('PRAGMA table_info("jobs")')}
                    if 'preview' in columns:
                        for (raw,) in conn.execute('SELECT preview FROM jobs WHERE preview IS NOT NULL'):
                            try:
                                preview=json.loads(raw)
                                if isinstance(preview,Mapping) and _is_confirmation_token(preview,'token',preview.get('token')):
                                    self.confirmation_tokens.add(preview['token'])
                            except (TypeError,ValueError):pass
                if "ledger" in existing:
                    for (raw,) in conn.execute("SELECT value FROM ledger WHERE table_name='source_definition'"):
                        try:
                            self._collect_auth(json.loads(raw))
                        except (TypeError, ValueError):
                            pass
                for table, columns in _TASK_COLUMNS.items():
                    if table not in existing:
                        continue
                    available = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
                    selected = [column for column in columns if column in available]
                    if not selected:
                        continue
                    projection = ",".join('"' + column + '"' for column in selected)
                    for row in conn.execute(f'SELECT rowid,{projection} FROM "{table}"'):
                        rowid = row[0]
                        for index, column in enumerate(selected, 1):
                            raw = row[index]
                            if raw in (None, ""):
                                continue
                            if isinstance(raw, str):
                                try:
                                    value = json.loads(raw)
                                except ValueError:
                                    value = raw
                            else:
                                value = raw
                            self._scan(value, "state", "tasks.sqlite3",
                                       f"/{table}/{rowid}/{column}")
        except (OSError, sqlite3.Error, ValueError):
            self._warning("tasks.sqlite3", "state_unreadable")

    def run(self) -> dict:
        documents = self._config_documents()
        # SQLite source definitions may name auth fields absent from current YAML.
        try:
            for file, document in documents:
                self._scan(document, "config", file, "")
            self._state()
        except _BudgetExceeded:
            self._warning('tasks.sqlite3','scan_budget_exceeded')
        findings = sorted(self.findings.values(), key=lambda row: (row["scope"], row["file"], row["field"]))
        for row in findings:
            row["reasons"].sort()
        return {
            "status": "incomplete" if self.warnings else "findings" if findings else "clean",
            "count": len(findings), "findings": findings, "warnings": self.warnings,"scanned_values":self.scanned_values,
            "migration_preview": [
                "Move live credentials to the private environment file and replace source authentication with auth.env references.",
                "Review affected historical task and audit records separately; preserve original archives and confirmed hashes.",
            ] if findings else [],
        }


def audit_credentials(settings, *, max_values=2000000, max_seconds=60) -> dict:
    """Inventory only the configured manager tree and its SQLite task database."""
    return _Inventory(settings,max_values,max_seconds).run()
