"""Read-only subscription configuration. No legacy defaults or mutable state."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from .contracts import integer

_COMMON = {"connect_timeout", "statement_timeout_ms"}
_CONNECTION = {
    "postgres": {"host", "port", "dbname", "schema", "user", "password",
                 "pool_min_connections", "pool_max_connections"} | _COMMON,
    "clickhouse": {"url", "database", "user", "password"} | _COMMON,
}
_ENV = {
    "postgres": {"host": "BETALENS_PG_HOST", "port": "BETALENS_PG_PORT",
                 "dbname": "BETALENS_PG_DATABASE", "schema": "BETALENS_PG_SCHEMA",
                 "user": "BETALENS_PG_USER", "password": "BETALENS_PG_PASSWORD"},
    "clickhouse": {"url": "BETALENS_CH_URL", "database": "BETALENS_CH_DATABASE",
                   "user": "BETALENS_CH_USER", "password": "BETALENS_CH_PASSWORD"},
}


def _keys(value, allowed, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是对象")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"{name} 包含不支持的配置键: {sorted(unknown)}")


def _positive_integer(value, name):
    # Environment variables are strings; fractional/bool values are never coerced.
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    return integer(value, name, minimum=1)


def load_subscription_config(db_config=None, *, config_file=None, backend=None):
    """Explicit overrides > environment > one modern subscription JSON file."""
    if any(key in os.environ for key in ("BETALENS_DB_HOST", "BETALENS_DB_PORT", "BETALENS_DB_NAME",
                                        "BETALENS_DB_USER", "BETALENS_DB_PASSWORD", "BETALENS_DB_SCHEMA")):
        raise ValueError("旧 BETALENS_DB_* 配置已移除，请使用 BETALENS_PG_* 或新版订阅 JSON")
    explicit = config_file if config_file is not None else os.environ.get("BETALENS_CONFIG")
    if explicit is not None and not str(explicit):
        raise ValueError("config_file 不能为空")
    root = Path(os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    path = Path(explicit).expanduser() if explicit is not None else root / "betalens" / "config.json"
    config = {}
    if explicit is not None or path.exists():
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"无法读取订阅配置: {path}") from exc
    _keys(config, {"backend", "postgres", "clickhouse"}, "订阅配置")
    for target in ("postgres", "clickhouse"):
        if target in config:
            _keys(config[target], _CONNECTION[target], target)
    overrides = {} if db_config is None else db_config
    if not isinstance(overrides, dict):
        raise ValueError("db_config 必须是对象")
    target = backend if backend is not None else overrides.get("backend", os.environ.get("BETALENS_BACKEND", config.get("backend", "postgres")))
    if target not in _CONNECTION:
        raise ValueError("backend 必须是 postgres 或 clickhouse")
    _keys(overrides, _CONNECTION[target] | {"backend"}, "db_config")
    connection = dict(config.get(target, {}))
    connection.update({key: os.environ[env] for key, env in _ENV[target].items() if env in os.environ})
    connection.update({key: value for key, value in overrides.items() if key != "backend"})
    required = ("host", "dbname", "user") if target == "postgres" else ("url", "database", "user")
    if any(not isinstance(connection.get(key), str) or not connection[key].strip() for key in required):
        raise ValueError(f"{target} 订阅配置必须指定非空连接项: {', '.join(required)}")
    defaults = {"connect_timeout": 5, "statement_timeout_ms": 120000}
    if target == "postgres":
        connection.setdefault("schema", "betalens")
        defaults.update(port=5432, pool_min_connections=1, pool_max_connections=10)
    else:
        url = urlsplit(connection["url"])
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("ClickHouse URL 必须是无凭据、无查询参数的 HTTP(S) 地址")
    for key, default in defaults.items():
        connection[key] = _positive_integer(connection.get(key, default), key)
    if target == "postgres" and connection["pool_max_connections"] < connection["pool_min_connections"]:
        raise ValueError("pool_max_connections 不能小于 pool_min_connections")
    return {**connection, "backend": target}
