"""Managed-reference acceptance, including real isolated target publication."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import hashlib

import pytest
from fastapi.testclient import TestClient

from betalens_db_manager.api import create_app
from betalens_db_manager.manager import DatabaseManagerService
from betalens_db_manager.config import Settings
from betalens_db_manager.reference_service import ReferenceWorkflow, normalize_request
from betalens_db_manager.reference_api import schema_preview
from betalens_db_manager.reference_contracts import TABLES
from betalens_db_manager.util import dumps, now, atomic_json
from test_manager import live_manager


def request(**values):
    return {'modules':['calendar'],'targets':['postgres','clickhouse'],
            'start_date':'2026-10-12','end_date':'2026-10-14','exchanges':['SSE'],**values}


class SampleSource:
    """Contract-complete source with explicit pagination and archived raw bytes."""
    def __init__(self):self.calls=[];self.fail=set();self.calendar_value=1
    def fetch(self,endpoint,params,fields,archive):
        from betalens_db_manager.util import parse_date
        from datetime import timedelta
        self.calls.append((endpoint,deepcopy(params)))
        if endpoint in self.fail:raise RuntimeError('injected endpoint unavailable')
        rows=[]
        if endpoint=='trade_cal':
            day=parse_date(params['start_date'])
            while day<=parse_date(params['end_date']):
                rows.append({'exchange':params['exchange'],'cal_date':day.strftime('%Y%m%d'),
                             'is_open':self.calendar_value,'pretrade_date':(day-timedelta(days=1)).strftime('%Y%m%d')})
                day+=timedelta(days=1)
        elif endpoint=='stock_basic' and params['list_status']=='L':
            rows=[{'ts_code':params['ts_code'],'name':'样本股票','list_status':'L','list_date':'19910403','exchange':'SZSE'}]
        elif endpoint in {'index_member_all','ci_index_member'} and params['is_new']=='Y':
            rows=[{'ts_code':params['ts_code'],'name':'样本股票','is_new':'Y','in_date':'20100101',
                   'l1_code':'801010.SI','l1_name':'农林牧渔','l2_code':'801011.SI','l2_name':'种植业',
                   'l3_code':'850111.SI','l3_name':'粮食种植'}]
        elif endpoint=='index_classify':
            level=int(params['level'][1:]);code={1:'801010.SI',2:'801011.SI',3:'850111.SI'}[level]
            rows=[{'index_code':code,'industry_name':'样本行业','level':params['level'],'src':params['src']}]
        offset=int(params.get('offset',0));size=int(params.get('limit',100000))
        rows=rows[offset:offset+size]
        data={'fields':list(reversed(fields)),'items':[[r.get(f) for f in reversed(fields)] for r in rows]}
        raw=dumps({'code':0,'data':data}).encode();archive.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(archive,'wb') as handle:handle.write(raw)
        atomic_json(archive.with_suffix('.meta.json'),{'fetched_at':now()})
        return data,hashlib.sha256(raw).hexdigest()


def prepared(service,**params):
    workflow=ReferenceWorkflow(service);item=workflow.plan(request(**params))
    service.download(item['id']);preview=service.preview(item['id'])
    assert not preview['blocking'],preview
    return item['id'],preview


@pytest.mark.parametrize('change',[
    {'targets':['postgres; DROP TABLE x']},{'modules':['market']},
    {'security_codes':['000001.SZ OR 1=1']},{'token':'secret'},
    {'check_times':['99:00']},{'levels':[True]},{'start_date':'2026-10-15'},
])
def test_request_scope_rejects_injection_or_ambiguous_values(change):
    with pytest.raises(ValueError):normalize_request(request(**change))


def test_persistent_plan_uses_all_calendar_dates_and_explicit_bse_reference(tmp_path):
    service=DatabaseManagerService(Settings(config_dir=tmp_path/'config',state_dir=tmp_path/'state'))
    workflow=ReferenceWorkflow(service)
    job=workflow.plan(request(exchanges=['SSE','BSE'],modules=['calendar','industry','suspension'],security_codes=['000001.SZ']))
    roots=job['plan']['requests']
    assert all('is_open' not in r['parameters'] for r in roots if r['endpoint']=='trade_cal')
    assert not any(r['parameters'].get('exchange')=='BSE' for r in roots)
    assert {r['parameters']['is_new'] for r in roots if r['endpoint']=='index_member_all'}=={'Y','N'}
    assert len([r for r in roots if r['endpoint']=='suspend_d'])==3
    workflow.s.state.update_job(job['id'],status='interrupted')
    new=workflow.retry(job['id'])
    assert new['plan']['parent_task_id']==job['id'] and new['status']=='planned'
    assert new['id']!=job['id']
    with pytest.raises(ValueError,match='中断'):workflow.download(job['id'])


def test_worker_hard_exit_records_reference_attempt_without_granting_trust(tmp_path):
    from betalens_db_manager.worker import interrupt_attempts
    service=DatabaseManagerService(Settings(config_dir=tmp_path/'config',state_dir=tmp_path/'state'))
    job=ReferenceWorkflow(service).plan(request())
    for kind in ('reference_attempt','execution_attempt'):
        service.state.ledger_put(kind,kind,{'task_id':job['id'],'attempt_id':kind,'status':'running'})
    interrupt_attempts(service,job['id'],'forced child exit')
    for kind in ('reference_attempt','execution_attempt'):
        record=service.state.ledger_get(kind,kind)
        assert record['status']=='interrupted' and record['finished_at'] and record['error']=='forced child exit'
    assert not service.state.ledger_rows('reference_task_result',job['id'])


def test_archived_baseline_requires_matching_query_evidence_and_fresh_confirmation(tmp_path,monkeypatch):
    service=DatabaseManagerService(Settings(config_dir=tmp_path/'config',state_dir=tmp_path/'state'))
    service.source=SampleSource();workflow=ReferenceWorkflow(service)
    source=workflow.plan(request(targets=['postgres']))
    service.download(source['id'])
    assert all('value' not in row for row in workflow.detail(source['id'])['logs'])
    # A completed old task is still not a publication credential.
    service.state.update_job(source['id'],status='completed')
    calls=len(service.source.calls)
    baseline=workflow.baseline(request(targets=['postgres']),[source['id']])
    assert len(service.source.calls)==calls
    assert all(r['status']=='archive_verified' for r in baseline['plan']['baseline_report'])
    assert baseline['status']=='downloaded' and not baseline['preview']
    assert not service.state.ledger_rows('reference_task_result',baseline['id'])
    assert all(source['id'] not in p['archive'] for p in baseline['slices'])
    before=workflow.bundles(workflow.job(source['id']))
    after=workflow.bundles(workflow.job(baseline['id']))
    assert [b['downloaded_at'] for b in before]==[b['downloaded_at'] for b in after]
    partial=workflow.baseline(request(targets=['postgres'],end_date='2026-10-15'),[source['id']])
    assert all(r['status']=='needs_reacquisition' for r in partial['plan']['baseline_report'])
    assert not partial['slices']


def test_reference_routes_preserve_preview_confirm_boundary(tmp_path,monkeypatch):
    service=DatabaseManagerService(Settings(config_dir=tmp_path/'config',state_dir=tmp_path/'state'))
    calls=[]
    class Store:
        def snapshot(self,audit=False):return {'rows':[],'scopes':[],'revisions':[],'events':[],'heads':{}}
        def install(self):calls.append('install');return {'status':'installed'}
    monkeypatch.setattr(ReferenceWorkflow,'store',lambda *args:Store())
    with TestClient(create_app(service)) as client:
        check=client.post('/api/integrity/check',json=request())
        assert check.status_code==200 and not check.json()['results']['postgres']['complete']
        preview=client.post('/api/reference/schema/preview',json={}).json()
        assert not calls and len(preview['statements'])==12 and not preview['destructive']
        assert client.post('/api/reference/schema/apply',json={'token':'bad'}).status_code==409
        assert not calls
        applied=client.post('/api/reference/schema/apply',json={'token':preview['token']})
        assert applied.status_code==200 and len(calls)==2
        job=client.post('/api/reference/tasks',json=request()).json()
        assert client.get('/api/reference/tasks/'+job['id']).status_code==200
        assert client.post('/api/reference/tasks/'+job['id']+'/commit',json={'token':'fake'}).status_code==409
        assert client.get('/integrity').status_code==200


def test_failed_source_and_corrupt_archive_are_isolated_to_their_scopes(tmp_path,monkeypatch):
    service=DatabaseManagerService(Settings(config_dir=tmp_path/'config',state_dir=tmp_path/'state'))
    source=SampleSource();source.fail={'ci_index_member'};service.source=source
    workflow=ReferenceWorkflow(service)
    item=workflow.plan(request(modules=['calendar','industry'],schemes=['CITICS'],security_codes=['000001.SZ']))
    service.download(item['id'])
    bundles=workflow.bundles(workflow.job(item['id']))
    assert all(b['complete'] for b in bundles if b['endpoint']=='trade_cal')
    assert all(not b['complete'] for b in bundles if b['endpoint']=='ci_index_member')
    class Store:
        def heads(self,ids):return {i:None for i in ids}
        def snapshot(self,audit=False):return {'rows':[],'scopes':[],'revisions':[],'events':[]}
    monkeypatch.setattr(ReferenceWorkflow,'store',lambda *args:Store())
    preview=workflow.preview(item['id'])
    assert not preview['blocking'] and preview['rejected']==3
    report=workflow.check(request(modules=['calendar','industry'],schemes=['CITICS'],security_codes=['000001.SZ']))
    assert all(r['reason']=='request_failed' for r in report['results']['postgres']['rows'] if r['module']=='industry')
    part=next(p for p in workflow.job(item['id'])['slices'] if p['dataset']=='trade_cal')
    Path(part['archive']).write_bytes(b'corrupt gzip')
    broken=workflow.bundles(workflow.job(item['id']))
    assert all(not b['complete'] for b in broken)


def test_isolated_two_target_gap_append_audit_readback_and_revoke(live_manager):
    service=live_manager;service.source=SampleSource();workflow=ReferenceWorkflow(service)
    service.pg.create_database()
    for target in ('postgres','clickhouse'):workflow.store(target).install()
    for day in ('2026-10-12','2026-10-14'):
        job,preview=prepared(service,start_date=day,end_date=day)
        service.commit(job,preview['token'])
    checked=workflow.check(request())
    for target in ('postgres','clickhouse'):
        assert [r['trade_date'] for r in checked['results'][target]['rows'] if r['status']!='complete']==['2026-10-13']
    job,preview=prepared(service,start_date='2026-10-13',end_date='2026-10-13')
    for target in ('postgres','clickhouse'):
        assert len(workflow.store(target).snapshot()['rows'])==2 # not written by audit
    result=service.commit(job,preview['token'])
    assert result['status']=='completed',result
    assert all(r['complete'] for r in workflow.check(request())['results'].values())
    # Exact repeat never creates a second publication or extra row.
    service.commit(job,preview['token'])
    for target in ('postgres','clickhouse'):
        snapshot=workflow.store(target).snapshot(audit=True)
        assert len(snapshot['rows'])==3 and len(snapshot['events'])==3
    rolled=service.rollback(job)
    assert rolled['status']=='rolled_back'
    for target,value in workflow.check(request())['results'].items():
        assert not value['complete']
        assert [r['trade_date'] for r in value['rows'] if r['status']!='complete']==['2026-10-13']
        assert len(workflow.store(target).snapshot(audit=True)['rows'])==3


def test_isolated_modules_partial_fail_stale_confirmation_and_interrupted_stage(live_manager,monkeypatch):
    service=live_manager;service.source=SampleSource();workflow=ReferenceWorkflow(service)
    service.pg.create_database()
    for target in ('postgres','clickhouse'):workflow.store(target).install()
    job,preview=prepared(service,modules=['calendar','industry','suspension'],security_codes=['000001.SZ'],
                         schemes=['SW2021'],start_date='2026-10-13',end_date='2026-10-13')
    original=ReferenceWorkflow.store
    class FailOne:
        def __init__(self,store):self.store=store
        def __getattr__(self,key):return getattr(self.store,key)
        def publish(self,revision_id,heads):
            manifests=self.store.snapshot(audit=True)['revisions']
            module=next(r['module'] for r in manifests if r['revision_id']==revision_id)
            if module=='industry':raise RuntimeError('injected publication failure')
            return self.store.publish(revision_id,heads)
    monkeypatch.setattr(ReferenceWorkflow,'store',lambda self,target:FailOne(original(self,target)) if target=='clickhouse' else original(self,target))
    result=service.commit(job,preview['token'])
    assert result['status']=='partial'
    rows=result['result']['results']
    assert any(r['target']=='clickhouse' and r['module']=='industry' and r['status']=='failed' for r in rows)
    assert any(r['target']=='clickhouse' and r['module']=='calendar' and r['status']=='published' for r in rows)
    assert all(r['status']=='published' for r in rows if r['target']=='postgres')
    monkeypatch.setattr(ReferenceWorkflow,'store',original)
    stale,old=prepared(service,targets=['postgres'],start_date='2026-10-13',end_date='2026-10-13')
    newer,new=prepared(service,targets=['postgres'],start_date='2026-10-13',end_date='2026-10-13')
    service.commit(newer,new['token'])
    stale_result=service.commit(stale,old['token'])
    assert stale_result['status']=='partial'
    assert '重新审计' in stale_result['result']['results'][0]['reason']
    # A real persisted stage plus simulated hard exit still cannot become a
    # trusted head. No status-only recovery promotes this orphan.
    crashed,cached=prepared(service,targets=['postgres'],start_date='2026-10-14',end_date='2026-10-14')
    class ExitAfterStage(FailOne):
        def publish(self,*args):raise KeyboardInterrupt('forced exit before publish')
    monkeypatch.setattr(ReferenceWorkflow,'store',lambda self,target:ExitAfterStage(original(self,target)))
    with pytest.raises(KeyboardInterrupt):service.commit(crashed,cached['token'])
    monkeypatch.setattr(ReferenceWorkflow,'store',original)
    service.state.update_job(crashed,status='interrupted')
    snapshot=workflow.store('postgres').snapshot(audit=True)
    orphan=next(r for r in snapshot['revisions'] if r['import_task_id']==crashed)
    assert orphan['revision_id'] not in set(snapshot['heads'].values())
    service.state.update_job(crashed,status='completed')
    assert orphan['revision_id'] not in set(workflow.store('postgres').snapshot()['heads'].values())
    retry=workflow.retry(crashed)
    assert retry['id']!=crashed and retry['plan']['parent_task_id']==crashed
