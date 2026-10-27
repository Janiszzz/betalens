"""Publication must be a target-local credential, never a task status flag."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
import threading

import pytest

from betalens_db_manager.reference_contracts import MODULE_TABLE, TABLES
from betalens_db_manager.reference_storage import ReferenceStore, METADATA, _normal, publication_heads


class MemoryStore(ReferenceStore):
    def __init__(self, target='postgres'):
        service = SimpleNamespace(settings=SimpleNamespace(pg_schema='betalens'), pg=object(), ch=object())
        super().__init__(service, target)
        self.tables = {}
        self.lock = threading.RLock()
        self.fail_data_once = False

    @contextmanager
    def _write(self, operation):
        with self.lock:
            yield None

    @contextmanager
    def _read(self):
        with self.lock:
            yield None

    def _rows(self, spec, conn=None, revision_id=None):
        return deepcopy([row for row in self.tables.get(spec.name, [])
                         if revision_id is None or row.get('revision_id') == revision_id])

    def _insert(self, spec, rows, conn=None):
        if spec.name in TABLES and self.fail_data_once:
            self.fail_data_once = False
            raise RuntimeError('injected exit before data insert')
        self.tables.setdefault(spec.name, []).extend(_normal(spec, row) for row in rows)

    def _assert_structure(self, conn=None, module=None):
        if getattr(self, 'structure_drift', False):
            raise ValueError('参考版本物理结构漂移，已阻止操作')


def revision(rid='r1', module='calendar', task='task1'):
    return {'revision_id': rid, 'import_task_id': task, 'attempt_id': 'attempt-' + rid,
            'module': module, 'mapping_version': 'mapping-v1', 'source_id': 'mock',
            'source_name': 'mock', 'downloaded_at': '2026-10-25T01:00:00+00:00',
            'request': {'start': '2026-10-20', 'end': '2026-10-22'}}


def calendar_scope(day='2026-10-20', exchange='SSE', is_open=1):
    scope_id = exchange + ':' + day
    return {'scope_id': scope_id, 'module': 'calendar',
            'selector': {'exchange': exchange, 'trade_date': day},
            'rows': [{'record_id': 'calendar', 'exchange': exchange, 'trade_date': day,
                      'is_open': is_open, 'source_endpoint': 'trade_cal'}],
            'complete': True, 'reasons': [],
            'evidence': {'full_response': True}, 'dependencies': []}


def test_append_is_invisible_until_verified_published_and_revocation_restores_previous():
    store = MemoryStore()
    scope = calendar_scope()
    first = store.stage(revision(), [scope])
    assert first['ingestion_seq'] == 1
    assert store.snapshot()['rows'] == []
    assert len(store.snapshot(audit=True)['rows']) == 1
    assert store.verify('r1')['verified']
    event = store.publish('r1', {scope['scope_id']: None})
    assert event['event_seq'] == 1
    assert store.snapshot()['rows'][0]['is_open'] == 1
    second = store.stage(revision('r2'), [calendar_scope(is_open=0)])
    assert second['ingestion_seq'] == 2
    assert store.snapshot()['rows'][0]['revision_id'] == 'r1'
    store.publish('r2', {scope['scope_id']: 'r1'})
    assert store.snapshot()['rows'][0]['is_open'] == 0
    inverse = store.revoke('r2', 'rollback-r2')
    assert inverse['event_seq'] == 3
    assert store.heads([scope['scope_id']]) == {scope['scope_id']: 'r1'}
    assert store.snapshot()['rows'][0]['is_open'] == 1
    assert len(store.snapshot(audit=True)['rows']) == 2
    assert len(store.snapshot(audit=True)['events']) == 3


def test_zero_rows_still_requires_and_preserves_explicit_scope_evidence():
    store = MemoryStore()
    scope = {'scope_id': 'stock:000001.SZ:2026-10-20', 'module': 'suspension',
             'selector': {'security_code': '000001.SZ', 'trade_date': '2026-10-20'},
             'rows': [], 'complete': True, 'evidence': {'query_complete': True, 'raw_rows': 0},
             'reasons': [], 'dependencies': [{'kind': 'source_calendar_evidence', 'evidence_id': 'calendar1'}]}
    store.stage(revision(module='suspension'), [scope])
    assert store.verify('r1')['scope_results'][0]['actual_rows'] == 0
    assert store.snapshot()['scopes'] == []
    store.publish('r1', {scope['scope_id']: None})
    assert store.snapshot()['scopes'][0]['evidence']['raw_rows'] == 0
    assert store.snapshot()['rows'] == []
    with pytest.raises(ValueError, match='覆盖范围'):
        store.stage(revision('empty'), [])


def test_same_request_idempotent_but_changed_input_cannot_reuse_revision():
    store = MemoryStore()
    scope = calendar_scope()
    a = store.stage(revision(), [scope])
    b = store.stage(revision(), [scope])
    assert a == b
    assert len(store.snapshot(audit=True)['rows']) == 1
    e1 = store.publish('r1', {scope['scope_id']: None})
    e2 = store.publish('r1', {scope['scope_id']: None})
    assert e1['event_id'] == e2['event_id']
    assert len(store.snapshot(audit=True)['events']) == 1
    with pytest.raises(ValueError, match='不同输入'):
        store.stage(revision(), [calendar_scope(is_open=0)])
    store.revoke('r1', 'rb')
    with pytest.raises(ValueError, match='已撤销'):
        store.publish('r1', {scope['scope_id']: None})


def test_stale_audit_head_blocks_publication_without_discarding_staged_history():
    store = MemoryStore()
    scope = calendar_scope()
    for rid in ('r1', 'r2'):
        store.stage(revision(rid), [scope])
    store.publish('r1', {scope['scope_id']: None})
    with pytest.raises(ValueError, match='版本头已变化'):
        store.publish('r2', {scope['scope_id']: None})
    assert store.snapshot()['rows'][0]['revision_id'] == 'r1'
    assert len(store.snapshot(audit=True)['rows']) == 2
    store.publish('r2', {scope['scope_id']: 'r1'})
    assert store.snapshot()['rows'][0]['revision_id'] == 'r2'


def test_crash_after_manifest_before_rows_is_never_published_and_retry_can_reconcile():
    store = MemoryStore('clickhouse')
    store.fail_data_once = True
    scope = calendar_scope()
    with pytest.raises(RuntimeError, match='injected exit'):
        store.stage(revision(), [scope])
    assert store.snapshot()['rows'] == []
    assert not store.verify('r1')['verified']
    with pytest.raises(ValueError, match='核验失败'):
        store.publish('r1', {scope['scope_id']: None})
    # A precise uncertain-write retry is harmless. Orchestration recovery after
    # an interrupted task deliberately chooses a new linked revision instead.
    store.stage(revision(), [scope])
    assert store.verify('r1')['verified']
    assert len(store.snapshot(audit=True)['rows']) == 1


def test_target_and_module_publications_are_independent():
    pg, ch = MemoryStore(), MemoryStore('clickhouse')
    scope = calendar_scope()
    for store in (pg, ch):
        store.stage(revision(), [scope])
    pg.publish('r1', {scope['scope_id']: None})
    assert pg.heads([scope['scope_id']])[scope['scope_id']] == 'r1'
    assert ch.heads([scope['scope_id']])[scope['scope_id']] is None
    unfinished = {'scope_id': 'sw:000001.SZ:1', 'module': 'industry',
                  'selector': {'security_code': '000001.SZ', 'scheme_code': 'SW2021', 'level': 1},
                  'rows': [], 'complete': False, 'reasons': ['source_request_failed']}
    pg.stage(revision('industry-failed', 'industry'), [unfinished])
    assert len(pg.snapshot()['scopes']) == 1
    assert pg.snapshot()['scopes'][0]['module'] == 'calendar'


def test_new_complete_snapshot_does_not_mix_old_intervals_or_fill_nulls():
    store = MemoryStore()
    def industry(code, name, start):
        return {'scope_id': 'industry:stock:sw:1', 'module': 'industry',
                'selector': {'security_code': '000001.SZ', 'scheme_code': 'SW2021', 'level': 1},
                'rows': [{'security_code': '000001.SZ', 'scheme_code': 'SW2021', 'level': 1,
                          'industry_code': code, 'industry_name': name, 'valid_from': start,
                          'source_endpoint': 'index_member_all'}], 'complete': True}
    old, new = industry('old', '旧分类', '2004-01-01'), industry('new', None, '2005-01-01')
    for rid, scope in (('r1', old), ('r2', new)):
        store.stage(revision(rid, 'industry'), [scope])
        store.publish(rid, {scope['scope_id']: None if rid == 'r1' else 'r1'})
    assert [(r['industry_code'], r['industry_name'], r['valid_from']) for r in store.snapshot()['rows']] == [('new', None, '2005-01-01')]
    store.revoke('r2', 'rollback')
    assert store.snapshot()['rows'][0]['industry_code'] == 'old'


def test_bad_values_duplicate_ids_and_metadata_tampering_are_rejected():
    store = MemoryStore()
    scope = calendar_scope()
    bad = deepcopy(scope); bad['rows'].append(deepcopy(bad['rows'][0]))
    with pytest.raises(ValueError, match='重复'):
        store.stage(revision(), [bad])
    bad = deepcopy(scope); bad['rows'][0]['intruder'] = 'drop table'
    with pytest.raises(ValueError, match='未登记字段'):
        store.stage(revision(), [bad])
    bad = deepcopy(scope); bad['rows'][0]['is_open'] = Decimal('1.5')
    with pytest.raises(ValueError, match='整数'):
        store.stage(revision(), [bad])
    bad_revision = revision(); bad_revision['downloaded_at'] = '2026-10-25T01:00:00'
    with pytest.raises(ValueError, match='时区'):
        store.stage(bad_revision, [scope])
    store.stage(revision(), [scope])
    row = store.tables[MODULE_TABLE['calendar']][0]
    row['is_open'] = 0
    assert not store.verify('r1')['verified']
    row['is_open'] = 1
    store.tables['reference_revision'][0]['module'] = 'industry'
    with pytest.raises(ValueError, match='载荷不一致'):
        store.snapshot(audit=True)


def test_clock_rollback_does_not_change_monotonic_target_sequence(monkeypatch):
    from betalens_db_manager import reference_storage
    store = MemoryStore()
    monkeypatch.setattr(reference_storage, 'now', lambda: '2030-01-01T00:00:00+00:00')
    first = store.stage(revision(), [calendar_scope()])
    monkeypatch.setattr(reference_storage, 'now', lambda: '2020-01-01T00:00:00+00:00')
    second = store.stage(revision('r2'), [calendar_scope()])
    assert second['ingested_at'] < first['ingested_at']
    assert second['ingestion_seq'] > first['ingestion_seq']


def test_legacy_untrusted_task_status_never_affects_visibility():
    store = MemoryStore()
    store.stage(dict(revision(), task_status='completed'), [calendar_scope()])
    assert store.heads(['SSE:2026-10-20']) == {'SSE:2026-10-20': None}
    assert not store.snapshot()['rows']


def test_physical_drift_blocks_stage_and_publish_and_readback_drift_never_becomes_trusted():
    store = MemoryStore()
    store.structure_drift = True
    with pytest.raises(ValueError, match='结构漂移'):
        store.stage(revision(), [calendar_scope()])
    store.structure_drift = False
    store.stage(revision(), [calendar_scope()])
    store.structure_drift = True
    with pytest.raises(ValueError, match='结构漂移'):
        store.publish('r1', {'SSE:2026-10-20': None})
    store.structure_drift = False
    store.publish('r1', {'SSE:2026-10-20': None})
    store.tables[MODULE_TABLE['calendar']][0]['is_open'] = 0
    with pytest.raises(ValueError, match='已发布版本的目标库核验失败'):
        store.publish('r1', {'SSE:2026-10-20': None})
    result = store.snapshot()
    assert result['invalid_revision_ids'] == ['r1']
    assert result['rows'] == []
    assert result['heads'] == {}
    assert store.snapshot(audit=True)['scopes'][0]['revision_id'] == 'r1'


def test_dependency_must_be_published_target_local_and_current_and_scope_id_cannot_alias():
    store = MemoryStore()
    scope = calendar_scope()
    store.stage(revision(), [scope])
    empty = {'scope_id': 'suspend:000001.SZ:2026-10-20', 'module': 'suspension',
             'selector': {'security_code': '000001.SZ', 'trade_date': '2026-10-20'},
             'rows': [], 'complete': True,
             'dependencies': [{'revision_id': 'r1', 'scope_id': scope['scope_id']}]}
    store.stage(revision('r2', 'suspension'), [empty])
    with pytest.raises(ValueError, match='依赖版本'):
        store.publish('r2', {empty['scope_id']: None})
    store.publish('r1', {scope['scope_id']: None})
    store.publish('r2', {empty['scope_id']: None})
    store.revoke('r1', 'rollback-calendar')
    changed = calendar_scope(exchange='SZSE')
    changed['scope_id'] = scope['scope_id']
    with pytest.raises(ValueError, match='不同的业务范围'):
        store.stage(revision('r3'), [changed])


# Import only the disposable-database fixture, not the old business test cases.
from test_manager import live_manager  # noqa: E402,F401


def test_isolated_pg_ch_append_publish_empty_scope_and_revoke(live_manager):
    service = live_manager
    service.pg.create_database()
    scope = calendar_scope()
    stores = [ReferenceStore(service, target) for target in ('postgres', 'clickhouse')]
    for store in stores:
        store.install()
        store.stage(revision(), [scope])
        assert not store.snapshot()['rows']
        assert store.verify('r1')['verified']
    stores[0].publish('r1', {scope['scope_id']: None})
    assert stores[0].snapshot()['rows']
    assert not stores[1].snapshot()['rows']
    stores[1].publish('r1', {scope['scope_id']: None})
    for store in stores:
        store.stage(revision('r2'), [calendar_scope(is_open=0)])
        store.publish('r2', {scope['scope_id']: 'r1'})
        assert store.snapshot()['rows'][0]['is_open'] == 0
        assert len(store.snapshot(audit=True)['rows']) == 2
        store.revoke('r2', 'rollback-r2')
        assert store.snapshot()['rows'][0]['is_open'] == 1
        empty = {'scope_id': 'suspend:000001.SZ:2026-10-20', 'module': 'suspension',
                 'selector': {'security_code': '000001.SZ', 'trade_date': '2026-10-20'},
                 'rows': [], 'complete': True, 'evidence': {'query_complete': True}}
        store.stage(revision('r3', 'suspension'), [empty])
        assert store.verify('r3')['verified']
        store.publish('r3', {empty['scope_id']: None})
        assert store.heads([empty['scope_id']])[empty['scope_id']] == 'r3'
