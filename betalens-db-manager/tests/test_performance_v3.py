"""Performance invariants preserve exact integrity and deployment bounds."""
import threading
from pathlib import Path
from types import SimpleNamespace
import pytest

from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.pipeline_options import options
from betalens_db_manager.resource_budget import BUDGET


def downloader(slices=()):
    obj=TaskDownloader.__new__(TaskDownloader)
    obj.item={'slices':list(slices)}
    obj._page_guard=threading.RLock();obj._page_index=None;obj._page_seeded=set()
    # Fetching the whole growing task is prohibited on this hot path.
    obj.s=SimpleNamespace(state=SimpleNamespace(job=lambda *_:pytest.fail('whole-task scan')))
    obj.job='test'
    return obj


def request(offset,endpoint='prices',dependency=False):
    return {'source_id':'local','endpoint':endpoint,'dependency_only':dependency,
            'parameters':{'code':'A','offset':offset,'limit':1}}


def data(*ids):return {'fields':['id'],'items':[[i] for i in ids]}


def test_long_page_chain_never_reopens_fresh_archives(monkeypatch):
    import betalens_db_manager.downloader as module
    monkeypatch.setattr(module,'read_archive',lambda *_:pytest.fail('fresh archive reread'))
    d=downloader()
    try:
        for i in range(200):assert not d.repeated(request(i),{},data(i))
        assert d.repeated(request(200),{},data(99))
        assert d.repeated(request(201),{},data(300,300))
        assert not d.repeated(request(0),{},data(0))  # Same-page retry is not an earlier page.
        assert not d.repeated(request(202,'other'),{},data(99))
        assert not d.repeated(request(202,dependency=True),{},data(99))
        path=Path(d._page_index.directory.name)
    finally:d.close()
    assert not path.exists()


def test_resume_indexes_each_checkpoint_once_including_later_cached_pages(monkeypatch):
    import betalens_db_manager.downloader as module
    reads=[]
    parts=[{'status':'paged','request':request(i),'archive':str(i),'sha':str(i)} for i in range(1,100)]
    def read(path,sha):reads.append(sha);return data(int(sha))
    monkeypatch.setattr(module,'read_archive',read)
    d=downloader(parts)
    try:
        assert not d.repeated(request(0),{},data(0))
        assert d.repeated(request(100),{},data(55))
        assert not d.repeated(request(100),{},data(100))
        assert len(reads)==99
    finally:d.close()


def test_resume_does_not_accept_corrupt_archive(monkeypatch):
    import betalens_db_manager.downloader as module
    def corrupt(*_):raise ValueError('原始归档哈希不匹配')
    monkeypatch.setattr(module,'read_archive',corrupt)
    d=downloader([{'status':'paged','request':request(0),'archive':'0','sha':'0'}])
    try:
        with pytest.raises(ValueError,match='哈希'):d.repeated(request(1),{},data(1))
    finally:d.close()


def test_plan_storage_deduplicates_and_roundtrips_without_mutating_frozen_plan(tmp_path):
    from copy import deepcopy
    from betalens_db_manager.state import State
    from betalens_db_manager.util import digest
    state=State(tmp_path)
    plan={'pipeline_version':2,'revision':'manager-v1','request':{'targets':['postgres']},
          'mapping_snapshots':{'large':{'rules':['fixed']*1000}},
          'requests':[request(i) for i in range(100)]}
    first=state.create_job(plan,'test');second=state.create_job({**plan,'pipeline_parent':first['id']},'test')
    assert first['plan']==plan and second['plan']['pipeline_parent']==first['id']
    assert digest(state.plan(first['id']))==digest(plan)
    assert state.job_header(first['id'])['request']==plan['request']
    with state.connect() as conn:
        assert conn.execute('SELECT count(*) FROM plan_snapshots').fetchone()[0]==1
        assert conn.execute('SELECT count(*) FROM request_payloads').fetchone()[0]==100
        assert conn.execute('SELECT max(length(plan)) FROM jobs').fetchone()[0]<1000
        conn.execute("UPDATE request_payloads SET value='{}' WHERE hash=(SELECT hash FROM request_payloads LIMIT 1)")
    with pytest.raises(ValueError,match='请求清单校验失败'):state.plan(first['id'])


def test_light_audit_projects_rules_without_loading_any_journal(tmp_path,monkeypatch):
    from datetime import date
    from decimal import Decimal
    from betalens_db_manager.pipeline_validation import ValidationRunner
    from betalens_db_manager.canonical import canonical
    from betalens_db_manager.comparison import row_key
    import betalens_db_manager.pipeline_validation as module
    from test_manager import manager
    s=manager(tmp_path);runner=ValidationRunner(s);spec=canonical('market_daily_raw')
    row={'asset_type':'stock','security_code':'000001.SZ','trade_date':date(2026,10,20),'high':Decimal(8),'low':Decimal(9),'volume':Decimal(-1)}
    def live(db,actual,keys,fields):
        assert {'high','low','volume'}<=set(fields)
        assert 'source_record_hash' not in fields and 'ingested_at' not in fields
        return {row_key(spec,row):row}
    monkeypatch.setattr(module,'existing',live)
    monkeypatch.setattr(module,'load_journal',lambda *_:pytest.fail('L1 loaded journal'))
    monkeypatch.setattr(runner.store,'receipt',lambda *_:pytest.fail('L1 loaded receipt'))
    result=runner._target_check('job','child',spec.name,0,spec,[row],['values'],{},None,'postgres')
    assert result['checks'][0]['failed']==1
    assert set(result['samples'][0]['problems'])>={'high_below_low','negative_volume'}


def test_parquet_projection_filters_still_verify_entire_file(tmp_path):
    from datetime import date
    from betalens_db_manager.pipeline_cache import write_rows,read_rows
    from betalens_db_manager.canonical import canonical
    import pyarrow.dataset as ds
    spec=canonical('market_daily_raw')
    rec=write_rows(tmp_path/'rows.parquet',spec,[{'security_code':'A','trade_date':date(2026,10,20)}, {'security_code':'B','trade_date':date(2026,10,21)}])
    rows=read_rows(rec,spec,columns=['security_code'],filters=ds.field('trade_date')==date(2026,10,21))
    assert rows==[{'security_code':'B'}]
    Path(rec['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='校验失败'):read_rows(rec,spec,columns=['security_code'])


def test_one_download_pool_overlaps_groups_without_reordering_them(tmp_path,monkeypatch):
    from test_manager import manager
    s=manager(tmp_path)
    d=TaskDownloader(s,{'id':'fixture','plan':{},'slices':[]})
    barrier=threading.Barrier(2);seen=[]
    monkeypatch.setattr(d,'prepared',lambda r:(r,None,None,None))
    def fetch(root):
        seen.append(root['endpoint']);barrier.wait(timeout=3)
        return []
    monkeypatch.setattr(d,'_retrieve',fetch)
    groups={'first':[request(0,'slow')],'second':[request(0,'fast')]}
    try:
        assert [g for g,_,errors in d.run_groups(groups,2) if not errors]==['first','second']
        pool=d._executor
        assert d.run_concurrent([],2)==[] and d._executor is pool
        assert set(seen)=={'slow','fast'}
    finally:d.close()


def test_pg_pool_clears_session_and_recovers_after_failed_query(monkeypatch):
    from betalens_db_manager.connections import PgPool
    class Conn:
        closed=False
        def __init__(self):self.resets=0
        def reset(self):self.resets+=1
        def close(self):self.closed=True
    created=[]
    def connect(**_):created.append(Conn());return created[-1]
    monkeypatch.setattr('betalens_db_manager.connections.psycopg2.connect',connect)
    pool=PgPool({})
    try:
        with pytest.raises(RuntimeError),pool.connection() as first:raise RuntimeError('query failed')
        with pool.connection() as second:assert second is first and second.resets==1
        assert len(created)==1 and first.resets==2
    finally:pool.close()


def test_shared_slots_bound_separate_threads_and_release_after_error(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from betalens_db_manager.resource_budget import shared_slot
    import time
    guard=threading.Lock();active=0;peak=0
    def work(i):
        nonlocal active,peak
        try:
            with shared_slot(tmp_path,'test',2):
                with guard:active+=1;peak=max(peak,active)
                time.sleep(.02)
                with guard:active-=1
                if i%2:raise RuntimeError('injected')
        except RuntimeError:pass
    with ThreadPoolExecutor(6) as pool:list(pool.map(work,range(12)))
    assert peak==2 and active==0


def test_v2_default_and_historical_write_routes_return_archive_error(tmp_path):
    from test_manager import manager
    from test_pipeline_v2 import prepare
    from fastapi.testclient import TestClient
    from betalens_db_manager.api import create_app
    from betalens_db_manager.execution_policy import LegacyTaskError
    s=manager(tmp_path);req=prepare(s,'http://127.0.0.1:1');req.pop('pipeline_version')
    plan=s.compile(req);assert plan['pipeline_version']==2
    old={**plan};old.pop('pipeline_version');old['request']={**old['request'],'pipeline_version':1}
    archived=s.state.create_job(old,s.settings.fingerprint)
    with TestClient(create_app(s)) as client:
        assert client.post('/api/tasks',json={**req,'pipeline_version':1}).status_code==422
        assert client.get('/api/tasks/'+archived['id']).status_code==200
        for operation in ('start','resume','cancel','rollback','preview','replay','replan','pipeline/commit','validations'):
            result=client.post('/api/tasks/'+archived['id']+'/'+operation,json={})
            assert result.status_code==410,(operation,result.text)
            assert result.json()['code']=='legacy_task_read_only'
    with pytest.raises(LegacyTaskError):s.state.enqueue(archived['id'],'pipeline_run')
    assert s.state.job(archived['id'])['status']=='planned'


def test_maintenance_blocks_admission_but_preserves_history(tmp_path):
    from test_manager import manager
    from test_pipeline_v2 import prepare
    from betalens_db_manager.execution_policy import MaintenanceError
    from betalens_db_manager.util import atomic_json
    s=manager(tmp_path);job=s.plan(prepare(s,'http://127.0.0.1:1'))
    s.state.enqueue(job['id'],'pipeline_run')
    atomic_json(s.settings.state_dir/'maintenance.json',{'enabled':True})
    assert s.state.claim() is None
    with pytest.raises(MaintenanceError):s.state.enqueue(job['id'],'pipeline_validate')
    assert s.state.job(job['id'])['tasks'][0]['status']=='queued'


def test_resource_defaults_cannot_exceed_shared_budget():
    value=options({})
    assert value['resource_profile']=='throughput_90'
    assert value['download_workers']==12 and value['normalization_workers']==3
    assert value['memory_budget_bytes']==32*1024**3
    assert options({'memory_budget_bytes':32*1024**3})['memory_budget_bytes']==32*1024**3
    from betalens_db_manager.resource_budget import BUDGET
    assert BUDGET.ch_read_memory_bytes==BUDGET.manager_high_bytes==32*1024**3
    for override in [{'normalization_workers':4},{'download_workers':17},{'memory_budget_bytes':33*1024**3}]:
        with pytest.raises(ValueError):options(override)
