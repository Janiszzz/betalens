from copy import deepcopy
from datetime import date

import pytest

from betalens_db_manager.reference_domains import build_scopes as daily_scopes, check_integrity
from betalens_db_manager.suspension_ranges import FORMAT, build_scopes, coverage_heads, day_scope, referenced_pool
from betalens_db_manager.util import dumps, digest
from test_reference_domains import request, calendar, lifecycle, suspended, bundle, snapshot, combine
from test_reference_storage import MemoryStore, revision


def ranged_snapshot(req, bundles, rid='range', seq=1, publish=True):
    scopes, pool = build_scopes(req, bundles)
    value = snapshot(scopes, rid, seq, publish)
    value['revisions'][0]['evidence_pool'] = pool
    return value


def range_event(timing=None):
    value = suspended(timing)
    value['parameters'] = {'ts_code': '000001.SZ', 'start_date': '20261019', 'end_date': '20261021'}
    return value


def test_compaction_preserves_every_daily_answer_and_raw_event():
    req = request(['suspension'], check_times=['09:30', '09:45', '10:00', '11:00'])
    source = [range_event('09:30-10:00'), calendar(), lifecycle()]
    dense = check_integrity(req, snapshot(daily_scopes(req, source)))
    compact = ranged_snapshot(req, source)
    actual = check_integrity(req, compact)
    assert [(r['reason'], r.get('is_suspended')) for r in actual['rows']] == [
        (r['reason'], r.get('is_suspended')) for r in dense['rows']]
    assert len(compact['scopes']) == 1 and len(compact['rows']) == 1
    assert compact['rows'][0]['raw_timing'] == '09:30-10:00'
    assert compact['scopes'][0]['selector']['end_date'] == '2026-10-22'


def test_sparse_empty_coverage_never_bridges_missing_queries():
    req = request(['suspension'])
    sources = [bundle('suspend_d', [], ts_code='000001.SZ', trade_date=day)
               for day in ['20261019', '20261021']]
    compact = ranged_snapshot(req, [*sources, calendar(), lifecycle()])
    result = check_integrity(req, compact)
    assert [r['reason'] for r in result['rows']] == ['no_reported_suspension', 'query_incomplete', 'no_reported_suspension']
    assert not result['complete']


def test_old_daily_and_new_overlapping_ranges_choose_latest_per_day_and_revoke():
    req = request(['suspension'])
    source = [range_event(), calendar(), lifecycle()]
    old = snapshot(daily_scopes(req, source), 'old', 1)
    newer_req = {**req, 'start_date': '2026-10-20', 'end_date': '2026-10-20'}
    newer = ranged_snapshot(newer_req, [bundle('suspend_d', [], ts_code='000001.SZ', trade_date='20261020'),
                                      calendar(), lifecycle()], 'new', 2)
    joined = combine(old, newer)
    assert all(r['reason'] == 'no_reported_suspension' for r in check_integrity(req, joined)['rows'])
    joined['events'].append({'revision_id': 'new', 'event_type': 'revoke'})
    assert check_integrity(req, joined)['rows'][1]['reason'] == 'full_day_suspension'
    pending = ranged_snapshot(newer_req, [], 'pending', 3, publish=False)
    assert check_integrity(req, combine(joined, pending))['rows'][1]['reason'] == 'full_day_suspension'


def test_range_manifest_verifies_and_stale_overlap_audit_blocks_publish():
    req = request(['suspension'])
    scopes, pool = build_scopes(req, [range_event(), calendar(), lifecycle()])
    store = MemoryStore()
    rev = {**revision('range', 'suspension'), 'evidence_pool': pool, 'expected_coverage_heads': {}}
    store.stage(rev, scopes)
    assert store.verify('range')['verified']
    store.publish('range', {s['scope_id']: None for s in scopes})
    middle_req = {**req, 'start_date': '2026-10-20', 'end_date': '2026-10-20'}
    middle, evidence = build_scopes(middle_req, [suspended(), calendar(), lifecycle()])
    r2 = {**revision('middle', 'suspension'), 'evidence_pool': evidence, 'expected_coverage_heads': {}}
    store.stage(r2, middle)
    with pytest.raises(ValueError, match='重叠停牌范围版本头'):
        store.publish('middle', {s['scope_id']: None for s in middle})
    r3 = {**revision('middle-fixed', 'suspension'), 'evidence_pool': evidence,
          'expected_coverage_heads': {scopes[0]['scope_id']: 'range'}}
    store.stage(r3, middle)
    store.publish('middle-fixed', {s['scope_id']: None for s in middle})
    assert len(store.snapshot(audit=True)['events']) == 2
    assert store.stage(r3, middle)['revision_id'] == 'middle-fixed'
    assert {r['revision_id'] for r in store.snapshot()['rows']} == {'middle-fixed'}


def test_shared_evidence_and_event_bounds_are_verified_before_stage():
    req = request(['suspension'])
    scopes, pool = build_scopes(req, [range_event(), calendar(), lifecycle()])
    rev = {**revision('range', 'suspension'), 'evidence_pool': deepcopy(pool)}
    rev['evidence_pool'][next(iter(pool))] = {'tampered': True}
    with pytest.raises(ValueError, match='哈希'):
        MemoryStore().stage(rev, scopes)
    scopes[0]['rows'][0]['security_code'] = '000002.SZ'
    with pytest.raises(ValueError, match='超出覆盖范围'):
        MemoryStore().stage({**revision('range', 'suspension'), 'evidence_pool': pool}, scopes)


def test_bulk_empty_proofs_share_calendar_and_queries_across_securities():
    codes = ['000001.SZ', '000002.SZ']
    req = request(['suspension'], security_codes=codes)
    stocks = bundle('stock_basic', [{'ts_code': c, 'list_date': '19910403', 'list_status': 'L'} for c in codes])
    sources = [bundle('suspend_d', [], start_date='20261019', end_date='20261021'), calendar(), stocks]
    scopes, pool = build_scopes(req, sources)
    assert len(scopes) == 2 and all(s['format'] == FORMAT for s in scopes)
    assert scopes[0]['evidence']['calendar_ref'] == scopes[1]['evidence']['calendar_ref']
    assert scopes[0]['evidence']['queries_ref'] == scopes[1]['evidence']['queries_ref']
    assert len(dumps({'scopes': scopes, 'pool': pool})) < len(dumps(daily_scopes(req, sources)))


def test_new_plans_stop_dense_derivation_but_preserve_legacy_plan_behavior(tmp_path, monkeypatch):
    from test_manager import manager
    from betalens_db_manager.canonical import registry_context, canonical
    from betalens_db_manager.pipeline_options import light_normalization
    service = manager(tmp_path)
    monkeypatch.setattr(service.pg, 'query', lambda *args, **kwargs: {'rows': [], 'next': None})
    monkeypatch.setattr(service.ch, 'query', lambda *args, **kwargs: {'rows': [], 'next': None})
    req = {'tables': ['trade_status_daily'], 'source_id': 'tushare', 'endpoints': ['suspend_d'],
           'asset_types': ['stock'], 'codes': ['000001.SZ'], 'start_date': '2026-10-20', 'end_date': '2026-10-20'}
    job = service.plan(req)
    job['plan']['request']['datasets'] = ['daily', 'suspend_d']
    assert job['plan']['status_storage'] == 'sparse-v2'
    assert not any(r['endpoint'] == 'daily' for r in job['plan']['requests'])
    stamp = '2026-10-21T00:00:00+00:00'
    with registry_context(service.registry), light_normalization():
        raw = service._canonical_record(canonical('market_daily_raw'),
            {'asset_type': 'stock', 'security_code': '000001.SZ', 'trade_date': date(2026, 10, 20), 'volume': 100},
            'daily', job['id'], stamp, 1, {}, date(2026, 10, 20))
        tables, _ = service._assemble_canonical(job, {'market_daily_raw': [raw]}, {}, [], [date(2026, 10, 20)], stamp)
        assert not tables.get('trade_status_daily')
        legacy = deepcopy(job);legacy['plan'].pop('status_storage')
        tables, _ = service._assemble_canonical(legacy, {'market_daily_raw': [raw]}, {}, [], [date(2026, 10, 20)], stamp)
        assert tables['trade_status_daily'][0]['status_code'] == 'QUOTE_PRESENT'


def test_workflow_audits_and_publishes_compact_evidence_with_overlap_protection(tmp_path, monkeypatch):
    from test_manager import manager
    from betalens_db_manager.reference_service import ReferenceWorkflow
    service=manager(tmp_path)
    workflow=ReferenceWorkflow(service)
    stores={target:MemoryStore(target) for target in ('postgres','clickhouse')}
    monkeypatch.setattr(workflow,'store',lambda target:stores[target])
    sources=[range_event(),calendar(),lifecycle()]
    for source in sources:source['reason']=''
    monkeypatch.setattr(workflow,'bundles',lambda item:sources)
    req=request(['suspension'])
    item=workflow.plan(req)
    assert item['plan']['suspension_storage']==FORMAT
    service.state.update_job(item['id'],status='downloaded')
    preview=workflow.build_cache(item['id'])
    assert preview['scope_count']==1 and not preview['blocking']
    result=workflow.publish_cache(item['id'],preview['token'])
    assert result['status']=='completed'
    for store in stores.values():
        assert len(store.snapshot()['scopes'])==1
        assert store.snapshot()['revisions'][0]['evidence_pool']
        assert check_integrity(req,store.snapshot())['rows'][1]['is_suspended']
    successor=workflow.plan(req)
    service.state.update_job(successor['id'],status='downloaded')
    same=workflow.build_cache(successor['id'])
    assert all(change['operation']=='no_business_change' for change in same['changes'])
    assert all(change['before_revisions'] for change in same['changes'])
    assert workflow.publish_cache(successor['id'],same['token'])['status']=='completed'


def test_incomplete_compact_query_preserves_permission_reason_in_daily_checks(tmp_path,monkeypatch):
    from test_manager import manager
    from betalens_db_manager.reference_service import ReferenceWorkflow
    workflow=ReferenceWorkflow(manager(tmp_path))
    stores={target:MemoryStore(target) for target in ('postgres','clickhouse')}
    monkeypatch.setattr(workflow,'store',lambda target:stores[target])
    sources=[range_event(),calendar(),lifecycle()]
    for source in sources:source['reason']=''
    sources[0].update(complete=False,reason='源接口权限不足')
    monkeypatch.setattr(workflow,'bundles',lambda item:sources)
    req=request(['suspension'])
    item=workflow.plan(req)
    workflow.s.state.update_job(item['id'],status='downloaded')
    preview=workflow.build_cache(item['id'])
    assert preview['blocking'] and preview['rejected']==1
    checked=workflow.check(req)
    for result in checked['results'].values():
        assert all(row['reason']=='source_permission_denied' for row in result['rows'])
        assert not result['complete']
