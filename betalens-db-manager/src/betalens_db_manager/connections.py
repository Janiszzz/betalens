"""Bounded, process-local PG pools and thread-local HTTP sessions."""
import atexit
import os
import queue
import threading
from contextlib import contextmanager

import psycopg2
import requests

from .resource_budget import BUDGET, shared_slot

_guard = threading.Lock()
_pools = {}
_http = threading.local()


class PgPool:
    def __init__(self, parameters):
        self.parameters = parameters
        self.idle = queue.LifoQueue(BUDGET.pg_connections)
        self.slots = threading.BoundedSemaphore(BUDGET.pg_connections)

    @contextmanager
    def connection(self):
        with self.slots:
            try:conn = self.idle.get_nowait()
            except queue.Empty:conn = None
            if conn is None or conn.closed:conn = psycopg2.connect(**self.parameters)
            try:
                yield conn
            finally:
                # Roll back errors and clear read-only, timezone, temp tables,
                # advisory locks and any caller session settings before reuse.
                try:conn.reset()
                except psycopg2.Error:conn.close()
                if not conn.closed:self.idle.put_nowait(conn)

    def close(self):
        while True:
            try:self.idle.get_nowait().close()
            except queue.Empty:return


@contextmanager
def pg_connection(settings, database=None):
    parameters = settings.pg(database)
    key = (os.getpid(), tuple(sorted(parameters.items())))
    with _guard:pool = _pools.setdefault(key, PgPool(parameters))
    with shared_slot(settings.state_dir, 'pg-connection', BUDGET.pg_connections), pool.connection() as conn:
        yield conn


def http_session():
    if getattr(_http, 'pid', None) != os.getpid():
        _http.pid = os.getpid()
        _http.session = requests.Session()
        # No implicit retries: mutation reconciliation belongs to the writer.
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
        _http.session.mount('http://', adapter)
        _http.session.mount('https://', adapter)
    return _http.session


@atexit.register
def close_pools():
    for pool in list(_pools.values()):pool.close()
    _pools.clear()
