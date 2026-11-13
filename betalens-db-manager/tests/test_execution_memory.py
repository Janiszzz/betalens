"""Large audit payloads retain canonical hashes without loading them to enqueue."""
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
from fastapi.testclient import TestClient

from betalens_db_manager.api import create_app
from betalens_db_manager.util import digest, dumps, encode
from test_manager import manager


def legacy_dumps(value):
    def safe(item):
        if isinstance(item, int) and not isinstance(item, bool) and abs(item) > 9007199254740991:
            return str(item)
        if isinstance(item, dict):
            return {key: safe(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(value) for value in item]
        return item
    return json.dumps(safe(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), default=encode, allow_nan=False)


@pytest.mark.parametrize('value', [
    {'scope': {'rows': [], 'complete': True}, 'count': 9007199254740991},
    {'negative': -9007199254740992, 'nested': [{'positive': 9007199254740992}, False, None]},
    ({'at': datetime(2026, 11, 10, tzinfo=timezone.utc), 'date': date(2026, 11, 10)},
     Decimal('123.4500'), Decimal('-0.000'), Path('/tmp/audit.json')),
    {'text': 'x' * 65535 + '中文😀' * 40000, 'nested': [1, {'precise': 10 ** 25}]},
])
def test_canonical_json_and_receipt_hashes_remain_byte_identical(value):
    original = legacy_dumps(value)
    assert dumps(value) == original
    assert digest(value) == hashlib.sha256(original.encode()).hexdigest()
    assert legacy_dumps(value) == original  # Serialization never mutates the input.


@pytest.mark.parametrize('value', [float('nan'), float('inf'), {'nested': [-float('inf')]}])
def test_nonfinite_values_still_reject(value):
    with pytest.raises(ValueError):
        dumps(value)
    with pytest.raises(ValueError):
        digest(value)


def planned(service, reference=False):
    request = {'start_date': '2026-10-20', 'end_date': '2026-10-21',
               'targets': ['postgres', 'clickhouse']}
    if reference:
        request.update(modules=['calendar'], exchanges=['SSE'])
    else:
        request.update(tables=['market_daily_raw'], source_id='tushare', endpoints=['daily'],
                       asset_types=['stock'], codes=['000001.SZ'])
    return service.plan(request)


def forbid_large_payload_reads(state, monkeypatch):
    original = state.connect
    @contextmanager
    def connect():
        with original() as connection:
            def authorize(action, table, column, *_):
                if action == sqlite3.SQLITE_READ and (
                    table in {'slices', 'job_requests', 'request_payloads'} or
                    (table == 'jobs' and column == 'preview')
                ):
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            connection.set_authorizer(authorize)
            yield connection
    monkeypatch.setattr(state, 'connect', connect)


@pytest.mark.parametrize('reference', [False, True])
def test_start_preserves_validation_without_reading_archive_graph_or_preview(tmp_path, monkeypatch, reference):
    service = manager(tmp_path)
    job = planned(service, reference)
    service.state.update_job(job['id'], status='interrupted', cancel=1,
                             preview={'changes': [{'payload': 'large audit preview'}]})
    app = create_app(service)
    forbid_large_payload_reads(service.state, monkeypatch)
    with TestClient(app) as client:
        response = client.post('/api/tasks/' + job['id'] + '/start', json={})
    assert response.status_code == 200, response.text
    header = service.state.job_header(job['id'])
    assert header['status'] == 'queued' and header['cancel'] == 0
    with service.state.connect() as connection:
        task = connection.execute('SELECT operation,payload FROM tasks WHERE job=?', (job['id'],)).fetchone()
    assert task['operation'] == 'pipeline_run' and json.loads(task['payload'])['background'] is True


def test_start_still_rejects_changed_contract_before_enqueuing(tmp_path):
    service = manager(tmp_path)
    job = planned(service)
    plan = service.state.plan(job['id'])
    plan['canonical_contracts']['market_daily_raw'] = 'changed-contract'
    service.state.replace_plan(job['id'], plan)
    with TestClient(create_app(service)) as client:
        response = client.post('/api/tasks/' + job['id'] + '/start', json={})
    assert response.status_code == 409 and '定义已变化' in response.text
    assert service.state.job(job['id'])['tasks'] == []


def test_start_still_rejects_changed_target_before_enqueuing(tmp_path):
    service = manager(tmp_path)
    job = planned(service)
    with service.state.connect() as connection:
        connection.execute("UPDATE jobs SET target='other-target' WHERE id=?", (job['id'],))
    with TestClient(create_app(service)) as client:
        response = client.post('/api/tasks/' + job['id'] + '/start', json={})
    assert response.status_code == 409 and '其他数据库配置' in response.text
    assert service.state.job(job['id'])['tasks'] == []
