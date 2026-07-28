"""Thread-safe, read-only PostgreSQL connection pooling for Datafeed."""

from __future__ import annotations

import atexit
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from psycopg2.extensions import connection as Connection
from psycopg2.pool import ThreadedConnectionPool

from .config import get_database_config


_CONNECTION_KEYS = ("dbname", "user", "password", "host", "port")
_POOLS: dict[tuple[tuple[str, str], ...], "ReadOnlyConnectionPool"] = {}
_POOLS_LOCK = threading.RLock()


def _clean_config(config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(get_database_config())
    if config:
        raw.update(config)
    cleaned = {key: raw[key] for key in _CONNECTION_KEYS if key in raw}
    cleaned["connect_timeout"] = int(raw.get("connect_timeout", 5))
    return cleaned


class ReadOnlyConnectionPool:
    """A small pool whose checked-out sessions cannot modify the database."""

    def __init__(
        self,
        db_config: dict[str, Any] | None = None,
        min_connections: int = 1,
        max_connections: int = 10,
        statement_timeout_ms: int = 120_000,
    ) -> None:
        self.db_config = _clean_config(db_config)
        self.statement_timeout_ms = int(statement_timeout_ms)
        self._pool = ThreadedConnectionPool(
            max(1, int(min_connections)),
            max(int(min_connections), int(max_connections)),
            **self.db_config,
        )

    def acquire(self) -> Connection:
        conn = self._pool.getconn()
        try:
            if conn.closed:
                self._pool.putconn(conn, close=True)
                conn = self._pool.getconn()
            conn.rollback()
            conn.set_session(readonly=True, autocommit=True)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT set_config('statement_timeout', %s, false)",
                    (str(self.statement_timeout_ms),),
                )
            return conn
        except Exception:
            self._pool.putconn(conn, close=True)
            raise

    def release(self, conn: Connection | None) -> None:
        if conn is None:
            return
        try:
            if not conn.closed:
                conn.rollback()
        finally:
            self._pool.putconn(conn, close=bool(conn.closed))

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)

    def closeall(self) -> None:
        self._pool.closeall()


def get_read_pool(
    db_config: dict[str, Any] | None = None,
    min_connections: int = 1,
    max_connections: int = 10,
    statement_timeout_ms: int = 120_000,
) -> ReadOnlyConnectionPool:
    cleaned = _clean_config(db_config)
    key = tuple(
        sorted((name, str(value)) for name, value in cleaned.items())
        + [
            ("pool_min_connections", str(max(1, int(min_connections)))),
            ("pool_max_connections", str(max(int(min_connections), int(max_connections)))),
            ("statement_timeout_ms", str(int(statement_timeout_ms))),
        ]
    )
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = ReadOnlyConnectionPool(
                cleaned,
                min_connections=min_connections,
                max_connections=max_connections,
                statement_timeout_ms=statement_timeout_ms,
            )
            _POOLS[key] = pool
        return pool


def close_all_pools() -> None:
    with _POOLS_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        pool.closeall()


atexit.register(close_all_pools)
