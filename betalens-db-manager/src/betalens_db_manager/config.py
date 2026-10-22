from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values
from .private import config_directory, state_directory, atomic_private

from .util import digest, identifier


@dataclass(frozen=True)
class Settings:
    state_dir: Path = field(default_factory=lambda: state_directory())
    config_dir: Path = field(default_factory=config_directory)
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_database: str = "betalens"
    pg_user: str = "betalens"
    pg_password: str = field(default="", repr=False)
    pg_schema: str = "betalens"
    ch_url: str = "http://127.0.0.1:8123"
    ch_database: str = "betalens"
    ch_user: str = "default"
    ch_password: str = field(default="", repr=False)
    token: str = field(default="", repr=False)
    requests_per_minute: int = 40
    timeout: int = 60
    archive_dir: Path | None = None
    env_file: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        identifier(self.pg_schema)
        identifier(self.ch_database)
        from urllib.parse import urlsplit
        url=urlsplit(self.ch_url)
        if url.scheme not in {'http','https'} or not url.netloc or url.username or url.password or url.query:
            raise ValueError('ClickHouse 地址须为不含凭证或查询参数的 HTTP(S) 地址')
        from .retrieval import request_rate
        request_rate(self.requests_per_minute)
        if not 1 <= self.pg_port <= 65535:
            raise ValueError("无效的限频或端口配置")

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        path = env_file or os.environ.get("BETALENS_DB_MANAGER_ENV") or os.environ.get("BETALENS_INFRA_ENV") or config_directory()/"credentials.env"
        private = {k:v for k,v in dotenv_values(path,interpolate=False).items() if v is not None} if Path(path).exists() else {}
        environment = {**os.environ, **private}
        defaults = cls()
        mapping = {
            "pg_host": "PGHOST", "pg_port": "PGPORT", "pg_database": "PGDATABASE", "pg_user": "PGUSER",
            "pg_password": "PGPASSWORD", "pg_schema": "PGSCHEMA", "ch_url": "CLICKHOUSE_URL",
            "ch_database": "CLICKHOUSE_DB", "ch_user": "CLICKHOUSE_USER", "ch_password": "CLICKHOUSE_PASSWORD",
            "token": "TUSHARE_TOKEN", "state_dir": "BETALENS_INFRA_STATE", "config_dir": "BETALENS_DB_MANAGER_CONFIG", "requests_per_minute": "TUSHARE_RPM",
        }
        values = {key: environment.get(name, getattr(defaults, key)) for key, name in mapping.items()}
        values["state_dir"] = Path(environment.get("BETALENS_DB_MANAGER_STATE", values["state_dir"])).expanduser().resolve()
        values["config_dir"] = Path(values["config_dir"]).expanduser().resolve()
        for key in ("pg_port", "requests_per_minute"):
            values[key] = int(values[key])
        values["env_file"] = Path(path).expanduser().resolve() if path else None
        if environment.get('BETALENS_INFRA_ARCHIVE'): values['archive_dir']=Path(environment['BETALENS_INFRA_ARCHIVE']).expanduser().resolve()
        return cls(**values)

    def pg(self, database: str | None = None) -> dict:
        return dict(host=self.pg_host, port=self.pg_port, dbname=database or self.pg_database,
                    user=self.pg_user, password=self.pg_password, connect_timeout=5)

    @property
    def fingerprint(self) -> str:
        return digest([self.pg_host, self.pg_port, self.pg_database, self.pg_schema, self.ch_url, self.ch_database])

    def public(self) -> dict:
        return {"postgres": {"host": self.pg_host, "port": self.pg_port, "database": self.pg_database,
                             "user": self.pg_user, "schema": self.pg_schema},
                "clickhouse": {"url": self.ch_url, "database": self.ch_database, "user": self.ch_user},
                "config_dir": str(self.config_dir), "env_file":str(self.env_file or self.config_dir/"credentials.env"), "state_dir": str(self.state_dir), "archive_dir":str(self.archive_dir or self.state_dir/'archive'), "postgres_password_configured":bool(self.pg_password), "clickhouse_password_configured":bool(self.ch_password), "token_configured": bool(self.token),
                "wind_configured": False,
                "wind_credentials_configured":bool(os.environ.get('WIND_API_KEY') or os.environ.get('WIND_USER')),
                "requests_per_minute": self.requests_per_minute, "fingerprint": self.fingerprint}

    def write_private_env(self, values: dict[str, str]) -> Path:
        """Write secrets outside the repository with strict permissions."""
        path = self.env_file or self.config_dir/'credentials.env'
        current={k:v for k,v in dotenv_values(path,interpolate=False).items() if v is not None} if path.exists() else {}
        import re
        for key,value in values.items():
            if not re.fullmatch(r'[A-Z][A-Z0-9_]*',key):raise ValueError('环境变量名称无效')
            if value in (None,''):continue
            value=str(value)
            if any(c in value for c in ('\n','\r','\x00')):raise ValueError('配置值不能包含换行或空字符')
            current[key]=value
        def quote(value):
            return "'"+value.replace('\\','\\\\').replace("'","\\'")+"'"
        return atomic_private(path,''.join(k+'='+quote(v)+'\n' for k,v in sorted(current.items())),credential=True)
