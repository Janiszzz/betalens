"""Connection profile resolution without persisting database passwords."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from betalens.datafeed.config import DEFAULT_CONFIG, get_config


PROFILE_FIELDS = ("host", "port", "dbname", "user")
CONNECTION_FIELDS = PROFILE_FIELDS + ("password",)
ENVIRONMENT_NAMES = {
    "host": ("BETALENS_DB_HOST", "PGHOST"),
    "port": ("BETALENS_DB_PORT", "PGPORT"),
    "dbname": ("BETALENS_DB_NAME", "BETALENS_DBNAME", "PGDATABASE"),
    "user": ("BETALENS_DB_USER", "PGUSER"),
    "password": ("BETALENS_DB_PASSWORD", "PGPASSWORD"),
}


def default_profile_path() -> Path:
    root = Path(os.environ.get("APPDATA") or Path.home() / ".config")
    return root / "betalens" / "database_profiles.json"


@dataclass(frozen=True)
class ConnectionProfile:
    """A named, persistable connection profile.

    Passwords deliberately are not part of this type. They can only enter an
    effective connection through environment variables or runtime overrides.
    """

    name: str
    host: str = "localhost"
    port: str = "5432"
    dbname: str = "datafeed"
    user: str = "postgres"

    @classmethod
    def from_mapping(cls, name: str, values: Mapping[str, Any]) -> "ConnectionProfile":
        if "password" in values:
            values = {key: value for key, value in values.items() if key != "password"}
        profile = cls(
            name=str(name).strip(),
            host=str(values.get("host", "localhost")).strip(),
            port=str(values.get("port", "5432")).strip(),
            dbname=str(values.get("dbname", values.get("database", "datafeed"))).strip(),
            user=str(values.get("user", "postgres")).strip(),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if not self.name:
            raise ValueError("profile 名称不能为空")
        for field_name in PROFILE_FIELDS:
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"profile 字段不能为空: {field_name}")
        try:
            port = int(self.port)
        except (TypeError, ValueError) as exc:
            raise ValueError("PostgreSQL port 必须是整数") from exc
        if not 1 <= port <= 65535:
            raise ValueError("PostgreSQL port 必须位于 1..65535")

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, **{key: str(getattr(self, key)) for key in PROFILE_FIELDS}}


class ProfileStore:
    """JSON persistence for non-secret connection settings."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path is not None else default_profile_path()

    def _read_document(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "active": None, "profiles": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取数据库 profile: {self.path}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("profiles", []), list):
            raise ValueError(f"数据库 profile 文件格式无效: {self.path}")
        return payload

    def list(self) -> list[ConnectionProfile]:
        document = self._read_document()
        profiles: list[ConnectionProfile] = []
        for item in document.get("profiles", []):
            if not isinstance(item, dict) or not item.get("name"):
                continue
            profiles.append(ConnectionProfile.from_mapping(str(item["name"]), item))
        return sorted(profiles, key=lambda item: item.name.casefold())

    def active_name(self) -> str | None:
        value = self._read_document().get("active")
        return str(value) if value else None

    def get(self, name: str | None = None) -> ConnectionProfile | None:
        selected = name or self.active_name()
        if not selected:
            return None
        return next((profile for profile in self.list() if profile.name == selected), None)

    def save(self, profile: ConnectionProfile, *, make_active: bool = True) -> ConnectionProfile:
        profile.validate()
        document = self._read_document()
        profiles = {
            item.name: item
            for item in self.list()
        }
        profiles[profile.name] = profile
        document = {
            "version": 1,
            "active": profile.name if make_active else document.get("active"),
            "profiles": [profiles[name].as_dict() for name in sorted(profiles, key=str.casefold)],
        }
        self._write_document(document)
        return profile

    def delete(self, name: str) -> bool:
        document = self._read_document()
        profiles = [profile for profile in self.list() if profile.name != name]
        if len(profiles) == len(document.get("profiles", [])):
            return False
        active = document.get("active")
        self._write_document(
            {
                "version": 1,
                "active": None if active == name else active,
                "profiles": [profile.as_dict() for profile in profiles],
            }
        )
        return True

    def _write_document(self, payload: Mapping[str, Any]) -> None:
        # Defensive filtering guarantees a password can never reach disk even
        # if a caller supplied a hand-built mapping to a future API.
        sanitized_profiles = []
        for item in payload.get("profiles", []):
            if not isinstance(item, Mapping):
                continue
            sanitized_profiles.append({key: item[key] for key in ("name", *PROFILE_FIELDS) if key in item})
        document = {
            "version": 1,
            "active": payload.get("active"),
            "profiles": sanitized_profiles,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


@dataclass(frozen=True)
class ResolvedConnection:
    config: dict[str, str]
    sources: dict[str, str] = field(default_factory=dict)
    profile_name: str | None = None

    def display_config(self) -> dict[str, str]:
        result = dict(self.config)
        if "password" in result:
            result["password"] = "********" if result["password"] else ""
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "config": self.display_config(),
            "sources": dict(self.sources),
        }


class ConnectionResolver:
    """Resolve defaults, local config, named profile, environment and CLI values."""

    def __init__(self, profile_store: ProfileStore | None = None):
        self.profile_store = profile_store or ProfileStore()

    def resolve(
        self,
        overrides: Mapping[str, Any] | None = None,
        *,
        profile: str | ConnectionProfile | None = None,
    ) -> ResolvedConnection:
        defaults = dict(DEFAULT_CONFIG.get("database", {}))
        config = {key: str(defaults.get(key, "")) for key in CONNECTION_FIELDS}
        sources = {key: "example_default" for key in CONNECTION_FIELDS}

        manager = get_config()
        local = manager.get_section("database")
        local_source = str(manager.config_file) if getattr(manager, "_loaded", False) else "example_default"
        for key in CONNECTION_FIELDS:
            if key in local and local[key] is not None:
                config[key] = str(local[key])
                sources[key] = local_source

        selected: ConnectionProfile | None
        if isinstance(profile, ConnectionProfile):
            selected = profile
        else:
            selected = self.profile_store.get(profile)
        if selected is not None:
            for key in PROFILE_FIELDS:
                config[key] = str(getattr(selected, key))
                sources[key] = f"profile:{selected.name}"

        for key, names in ENVIRONMENT_NAMES.items():
            for name in names:
                if name in os.environ:
                    config[key] = os.environ[name]
                    sources[key] = f"environment:{name}"
                    break

        for key, value in (overrides or {}).items():
            normalized = "dbname" if key == "database" else key
            if normalized in CONNECTION_FIELDS and value is not None:
                config[normalized] = str(value)
                sources[normalized] = "runtime_override"

        missing = [key for key in ("host", "port", "dbname", "user") if not config.get(key)]
        if missing:
            raise ValueError(f"数据库配置缺少字段: {missing}")
        try:
            port = int(config["port"])
        except (TypeError, ValueError) as exc:
            raise ValueError("PostgreSQL port 必须是整数") from exc
        if not 1 <= port <= 65535:
            raise ValueError("PostgreSQL port 必须位于 1..65535")

        return ResolvedConnection(
            config=config,
            sources=sources,
            profile_name=selected.name if selected else None,
        )

