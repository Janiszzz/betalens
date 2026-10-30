import json

from fastapi.testclient import TestClient

from betalens_db_manager.api import create_app
from betalens_db_manager.util import now
from test_manager import manager


def seed_logs(state, job, start, count):
    with state.connect() as conn:
        conn.executemany(
            'INSERT INTO event_log(job,created,event,value) VALUES(?,?,?,?)',
            [(job, now(), 'request_started', json.dumps({'endpoint': f'log_{n}'}))
             for n in range(start, start + count)],
        )


def test_latest_logs_are_bounded_chronological_and_task_scoped(tmp_path):
    state = manager(tmp_path).state
    seed_logs(state, 'first', 1, 1500)
    seed_logs(state, 'second', 1501, 20)
    tail = state.logs('first', after=1, limit=500, latest=True)
    assert [r['id'] for r in tail] == list(range(1001, 1501))
    assert tail[-1]['endpoint'] == 'log_1500'
    assert [r['id'] for r in state.logs('first', 10, 3)] == [11, 12, 13]
    assert len(state.logs('first', limit=100000, latest=True)) == 1000
    assert state.logs('first', limit=0, latest=True)[0]['id'] == 1500
    assert state.logs('empty', latest=True) == []
    with state.connect() as conn:
        plan = conn.execute(
            'EXPLAIN QUERY PLAN SELECT * FROM event_log WHERE job=? ORDER BY id DESC LIMIT ?',
            ('first', 500),
        ).fetchall()
    assert any('event_log_job_id' in row['detail'] for row in plan)


def test_http_latest_logs_skip_backlog_and_keep_legacy_cursor(tmp_path):
    service = manager(tmp_path)
    seed_logs(service.state, 'first', 1, 1500)
    seed_logs(service.state, 'second', 1501, 10)
    with TestClient(create_app(service)) as client:
        response = client.get('/api/tasks/first/logs?latest=true&limit=500&after=1')
        assert response.status_code == 200
        assert [r['id'] for r in response.json()] == list(range(1001, 1501))
        assert [r['id'] for r in client.get('/api/tasks/first/logs?after=10&limit=3').json()] == [11, 12, 13]
        assert client.get('/api/tasks/missing/logs?latest=true').json() == []


def test_live_latest_starts_at_tail_and_skips_new_backlog(tmp_path,monkeypatch):
    service = manager(tmp_path)
    job = service.state.create_job({'request': {'tables': [], 'targets': []}}, service.settings.fingerprint)['id']
    seed_logs(service.state, job, 1, 1500)
    with TestClient(create_app(service)) as client:
        original=service.state.job
        monkeypatch.setattr(service.state,'job',lambda *_:(_ for _ in ()).throw(AssertionError('日志流不得加载完整任务快照')))
        with client.websocket_connect('/api/live/' + job + '?latest=true&limit=3') as socket:
            snapshot = socket.receive_json()
            assert snapshot['type'] == 'logs'
            assert [r['id'] for r in snapshot['records']] == [1498, 1499, 1500]
            assert socket.receive_json()['type'] == 'heartbeat'
            assert socket.receive_json()['type'] == 'task'
            seed_logs(service.state, job, 1501, 100)
            rows = [socket.receive_json()['record'] for _ in range(3)]
            assert [r['id'] for r in rows] == [1598, 1599, 1600]
        monkeypatch.setattr(service.state,'job',original)
        with client.websocket_connect('/api/live/' + job + '?after=5') as socket:
            assert socket.receive_json()['record']['id'] == 6
