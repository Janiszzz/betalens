from dataclasses import replace
from datetime import date,datetime,timezone
from decimal import Decimal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading,time,json
import pytest
from test_manager import manager,table_definition,live_manager
from betalens_db_manager.pipeline_options import options,light_normalization
from betalens_db_manager.pipeline_store import PipelineStore
from betalens_db_manager.pipeline_cache import write_rows,read_rows,save_journal,load_journal
from betalens_db_manager.canonical import canonical,registry_context
from betalens_db_manager.comparison import compare,row_hash
from betalens_db_manager.pipeline_plan import params_for
from betalens_db_manager.api import create_app


def test_frozen_options_and_large_requests(tmp_path):
    assert options({})['download_workers']==8
    for field,value in [('download_workers',True),('normalization_workers',0),('post_check_rules',['unknown'])]:
        with pytest.raises(ValueError):options({field:value})
    request={'start_date':'2005-01-01','end_date':'2026-10-26','codes':['000001.SZ']}
    assert len(list(params_for('daily',request,{})))==1
    assert len(list(params_for('trade_cal',request,{'exchange':'SSE'})))==2
    suspension=list(params_for('suspend_d',request,{}))
    assert suspension[0]['end_date']=='20050131' and suspension[1]['start_date']=='20050201'
    s=manager(tmp_path)
    planned=s.compile({'pipeline_version':2,'tables':['market_daily_post_adjusted'],'source_id':'tushare','endpoints':['daily','adj_factor'],'asset_types':['stock'],**request})
    assert planned['pipeline_version']==2
    assert not any(r['endpoint']=='daily_basic' for r in planned['requests'])
    assert sum(r['endpoint']=='daily' for r in planned['requests'])==1
    assert planned['request']['requests_per_minute']==450
    assert planned['retrieval_policies']['tushare:stock_basic']['endpoint_rpm']==50


def test_stage_leases_are_exclusive_and_explicitly_recoverable(tmp_path):
    s=manager(tmp_path);store=PipelineStore(s.state);store.put('parent','one','normalize',{'a':1})
    with ThreadPoolExecutor(4) as pool:tokens=list(pool.map(lambda _:store.claim('parent','one'),range(4)))
    assert sum(t is not None for t in tokens)==1
    store.recover('parent');token=store.claim('parent','one');assert token
    store.finish('parent','one',{'rows':4},token=token)
    assert store.claim('parent','one') is None
    other=PipelineStore(s.state);assert other.get('parent','one')['result']=={'rows':4}


def test_parquet_and_journal_preserve_precision_timestamps_and_nulls(tmp_path):
    spec=canonical('market_daily_raw')
    row={'asset_type':'stock','security_code':'000001.SZ','trade_date':date(2026,10,20),'close':Decimal('12.123456789012'),'available_at':datetime(2026,10,20,tzinfo=timezone.utc)}
    rec=write_rows(tmp_path/'rows.parquet',spec,[row]);actual=read_rows(rec,spec)[0]
    assert actual['close']==row['close'] and actual['available_at']==row['available_at']
    comp=compare(spec,[row],[],'fill_missing');manifest=save_journal(tmp_path/'journal',spec,comp)
    recovered=load_journal(manifest,spec)
    assert recovered['changes'][0]['before'] is None
    assert row_hash(spec,recovered['changes'][0]['after'])==comp['changes'][0]['after_hash']
    Path(rec['path']).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='校验失败'):read_rows(rec,spec)


def test_default_light_normalization_defers_domain_checks(tmp_path):
    from test_manager import cached_part
    s=manager(tmp_path)
    job=s.plan({'tables':['market_daily_raw'],'source_id':'tushare','endpoints':['daily'],'asset_types':['stock'],'codes':['000001.SZ'],'start_date':'2026-10-20','end_date':'2026-10-20'})
    contract=s.sources.endpoint('tushare','daily');fields=[f['name'] for f in contract['fields']]
    values={'ts_code':'000001.SZ','trade_date':'20261020','open':'12','high':'9','low':'10','close':'11','vol':'-2','amount':'1'}
    cached_part(s,job,'daily',{'fields':fields,'items':[[values.get(f) for f in fields]]})
    with light_normalization():rows,rejects=s._build_canonical(s.state.job(job['id']))
    assert rows['market_daily_raw'] and not rejects
    strict,bad=s._build_canonical(s.state.job(job['id']));assert bad


@pytest.fixture
def pipeline_http():
    from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data={'fields':['ticker','day','price'],'items':[['A','20261020','12.123456780123'],['B','20261021','10.0']]}
            body=json.dumps(data).encode();self.send_response(200);self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield 'http://127.0.0.1:'+str(server.server_port)
    server.shutdown();thread.join();server.server_close()


def prepare(s,url):
    definition=table_definition('pipeline_quotes');s.registry.save(definition)
    s.sources.save({'source_id':'pipeline_http','provider':'http','base_url':url,'endpoints':{'prices':{'path':'/prices','strategy':'range','keys':['ticker','day'],'date_field':'day','date_format':'%Y%m%d','fields':[{'name':'ticker','type':'text'},{'name':'day','type':'date'},{'name':'price','type':'numeric'}]}}})
    binding={'source_id':'pipeline_http','endpoint':'prices','fields':[{'source_field':'ticker','column':'code'},{'source_field':'day','column':'business_date','date_format':'%Y%m%d'},{'source_field':'price','column':'price'}]}
    s.mappings.save({'format_version':1,'table':'pipeline_quotes','bindings':[binding]},None)
    return {'pipeline_version':2,'tables':['pipeline_quotes'],'source_id':'pipeline_http','start_date':'2026-10-20','end_date':'2026-10-21','commit_batch_rows':1,'min_free_bytes':0}


def test_new_api_start_is_explicit_and_background_without_browser(tmp_path,pipeline_http):
    from fastapi.testclient import TestClient
    s=manager(tmp_path);request=prepare(s,pipeline_http)
    with TestClient(create_app(s)) as client:
        job=client.post('/api/tasks',json=request).json()
        assert job['plan']['pipeline_version']==2
        assert not job['tasks']
        started=client.post('/api/tasks/'+job['id']+'/start',json={});assert started.status_code==200,started.text
        task=s.state.claim(allow_ui=False);assert task['operation']=='pipeline_run' and task['payload']['background'] is True
        assert client.get('/api/tasks/'+job['id']+'/pipeline').status_code==200
    assert s.state.job(job['id'])['cancel']==0


def test_real_pipeline_two_targets_validation_restore_and_no_redownload(live_manager,pipeline_http,monkeypatch):
    from betalens_db_manager.pipeline import Pipeline
    from betalens_db_manager.pipeline_validation import ValidationRunner
    from betalens_db_manager.source_registry import HTTPAdapter
    s=live_manager;request=prepare(s,pipeline_http)
    with registry_context(s.registry):
        plan=s.schema_plan();s.initialize(plan['token'])
        operation=s.schema.plan('pipeline_quotes');assert s.schema.apply(operation['token'])['status']=='completed'
    job=s.plan(request);result=Pipeline(s).run(job['id'])
    assert result['status']=='completed',result.get('error')
    assert len(PipelineStore(s.state).receipts(job['id']))==4
    assert result['audit_sync_status']=='synced'
    with registry_context(s.registry):
        for db in (s.pg,s.ch):
            data=db.query('pipeline_quotes')['rows'];assert len(data)==2
            assert data[0]['price']==Decimal('12.12345678')
    report=ValidationRunner(s).run(job['id'],{'rules':['readback']});assert report['status']=='passed',report
    monkeypatch.setattr(HTTPAdapter,'_fetch',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('回退不得重新下载')))
    rolled=Pipeline(s).rollback(job['id']);assert rolled['status']=='rolled_back'
    with registry_context(s.registry):
        for db in (s.pg,s.ch):assert db.query('pipeline_quotes')['rows']==[]


def test_all_31_policies_and_shared_adaptive_feedback(tmp_path,monkeypatch):
    from betalens_db_manager.catalog import load_catalog
    from betalens_db_manager.retrieval import tushare_policy
    from betalens_db_manager.adaptive_rate import cool,effective,scope
    s=manager(tmp_path,token='test-subscription');policies={n:tushare_policy(n,points=5000) for n in load_catalog()}
    assert len(policies)==31 and all(p['row_limit']>0 and p['points']==5000 for p in policies.values())
    assert policies['daily']['row_limit']==6000 and policies['daily']['endpoint_rpm']==500
    assert policies['fund_daily']['endpoint_rpm']==50 and policies['etf_basic']['endpoint_rpm']==50
    assert policies['income_vip']['row_limit']==6000
    clock=[1000.0];monkeypatch.setattr('betalens_db_manager.adaptive_rate.time.time',lambda:clock[0])
    cool(s.source,'daily',60,450);assert effective(s.source,'daily',450)==225
    assert effective(s.source,'adj_factor',450)==450
    clock[0]+=299;assert effective(s.source,'daily',450)==225
    clock[0]+=1;assert effective(s.source,'daily',450)==270
    cool(s.source,'daily',60,450,account=True);assert effective(s.source,'',450)==225
    other=manager(tmp_path/'other',token='another-subscription');assert scope(other.source)!=scope(s.source)


def test_postchecks_detect_business_errors_and_missing_causality():
    from betalens_db_manager.pipeline_validation import values,relations,ValidationRunner
    row={'high':Decimal(8),'low':Decimal(9),'close':Decimal(12),'volume':Decimal(-1)}
    assert set(values(row))>={'high_below_low','close_above_high','negative_volume'}
    checked,unknown,bad=relations({'raw_close':Decimal(10),'adjustment_factor':Decimal(2),'close':Decimal(21)})
    assert checked==['post_adjustment'] and bad==['post_adjustment_mismatch']
    # A midnight Shanghai publication is the previous UTC day, with no causal violation.
    _,_,bad=relations({'event_date':date(2026,10,20),'available_at':datetime(2026,10,19,16,tzinfo=timezone.utc),'availability_basis':'source_publication_timestamp'})
    assert 'available_before_event' not in bad
    checked,unknown,bad=ValidationRunner._cross_relations({'security_code':'000001.SZ','asset_type':'stock','trade_date':date(2026,10,20),'close':Decimal(1)},{'calendar':{},'lifecycle':{},'balance':{}})
    assert set(unknown)=={'calendar_missing','lifecycle_missing'} and not checked and not bad


def test_real_pipeline_recovers_lost_receipt_and_preserves_later_changes(live_manager,pipeline_http,monkeypatch):
    from betalens_db_manager.pipeline import Pipeline
    from betalens_db_manager.source_registry import HTTPAdapter
    from betalens_db_manager.pipeline_writer import pg_apply
    from betalens_db_manager.publishing import existing
    s=live_manager;request=prepare(s,pipeline_http)
    with registry_context(s.registry):
        s.initialize(s.schema_plan()['token']);s.schema.apply(s.schema.plan('pipeline_quotes')['token'])
    original=PipelineStore.acknowledge_many;fail=[True]
    def lose_ack(store,job,target,entries):
        if target=='postgres' and any(result.get('verified') for _,result,_ in entries) and fail[0]:fail[0]=False;raise RuntimeError('simulated process exit after commit')
        return original(store,job,target,entries)
    monkeypatch.setattr(PipelineStore,'acknowledge_many',lose_ack)
    job=s.plan(request);assert Pipeline(s).run(job['id'])['status']=='partial'
    with registry_context(s.registry):assert len(s.ch.query('pipeline_quotes')['rows'])==2
    monkeypatch.setattr(HTTPAdapter,'_fetch',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('成功归档不得重下')))
    assert Pipeline(s).run(job['id'])['status']=='completed'
    first=s.audit(job['id'],limit=2);second=s.audit(job['id'],after=first['next_after'],limit=2)
    assert len(first['changes'])==2 and len(second['changes'])==2
    assert not set(r['id'] for r in first['changes'])&set(r['id'] for r in second['changes'])
    with registry_context(s.registry):
        spec=s.registry.spec('pipeline_quotes');old=s.pg.query('pipeline_quotes')['rows'];new=[{**old[0],'price':Decimal(99),'import_task_id':'later-task'}]
        comparison=compare(spec,new,old,'refresh_source');pg_apply(s.pg,spec,comparison)
        with pytest.raises(ValueError,match='目标数据已变化'):Pipeline(s).rollback(job['id'],['postgres'])
        assert existing(s.pg,spec,new)[next(iter(existing(s.pg,spec,new)))]['price']==Decimal(99)


def test_real_overlap_fill_missing_peer_mismatch_and_stale_report(live_manager,pipeline_http):
    from betalens_db_manager.pipeline import Pipeline
    from betalens_db_manager.pipeline_validation import ValidationRunner
    from betalens_db_manager.pipeline_writer import pg_apply
    s=live_manager;request=prepare(s,pipeline_http)
    with registry_context(s.registry):
        s.initialize(s.schema_plan()['token']);s.schema.apply(s.schema.plan('pipeline_quotes')['token'])
    first=s.plan(request);assert Pipeline(s).run(first['id'])['status']=='completed'
    report=ValidationRunner(s).run(first['id'],{'rules':['readback']});assert report['status']=='passed'
    repeated=s.plan(request);assert Pipeline(s).run(repeated['id'])['status']=='completed'
    with registry_context(s.registry):
        spec=s.registry.spec('pipeline_quotes');old=s.pg.query('pipeline_quotes')['rows'];new=[{**old[0],'price':Decimal(99),'import_task_id':'later-task'}]
        pg_apply(s.pg,spec,compare(spec,new,old,'refresh_source'))
    overlapping=s.plan(request);result=Pipeline(s).run(overlapping['id'])
    assert result['status']=='partial' and '双库业务值不同' in result['error']
    with registry_context(s.registry):assert s.pg.query('pipeline_quotes')['rows'][0]['price']==Decimal(99)
    # A subsequent managed rollback invalidates old reports instead of silently leaving a green result.
    Pipeline(s).rollback(overlapping['id'])
    Pipeline(s).rollback(first['id'],['clickhouse'])
    assert s.state.ledger_get('pipeline_validation',report['id'])['status']=='stale'


def test_real_manual_commit_uses_typed_cache_and_reference_background_flow(live_manager,pipeline_http,monkeypatch):
    from betalens_db_manager.pipeline import Pipeline
    from betalens_db_manager.reference_service import ReferenceWorkflow
    from betalens_db_manager.source_registry import HTTPAdapter
    from test_reference_workflow import SampleSource,request as reference_request
    from betalens_db_manager.pipeline_validation import ValidationRunner
    s=live_manager;request=prepare(s,pipeline_http);request['auto_commit']=False
    with registry_context(s.registry):
        s.initialize(s.schema_plan()['token']);s.schema.apply(s.schema.plan('pipeline_quotes')['token'])
    job=s.plan(request);assert Pipeline(s).run(job['id'])['status']=='ready_to_commit'
    with registry_context(s.registry):assert s.pg.query('pipeline_quotes')['rows']==[]
    monkeypatch.setattr(HTTPAdapter,'_fetch',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('手动提交应复用归档')))
    assert Pipeline(s).commit(job['id'])['status']=='completed'
    s.source=SampleSource();workflow=ReferenceWorkflow(s)
    for target in ('postgres','clickhouse'):workflow.store(target).install()
    reference=workflow.plan(reference_request(pipeline_version=2))
    assert Pipeline(s).run(reference['id'])['status']=='completed'
    for target in ('postgres','clickhouse'):assert workflow.store(target).snapshot()['rows']
    report=ValidationRunner(s).run(reference['id'],{'rules':['readback']});assert report['status']=='passed',report
    assert Pipeline(s).rollback(reference['id'])['status']=='rolled_back'
    assert s.state.ledger_get('pipeline_validation',report['id'])['status']=='stale'


def test_adaptive_http_429_reduces_shared_budget_and_respects_cooldown(tmp_path,monkeypatch):
    import betalens_db_manager.source as module
    from test_retrieval import Clock
    from betalens_db_manager.retrieval import tushare_policy
    from betalens_db_manager.adaptive_rate import effective
    clock=Clock();monkeypatch.setattr(module.time,'time',clock.time);monkeypatch.setattr(module.time,'monotonic',clock.time);monkeypatch.setattr(module.time,'sleep',clock.sleep)
    s=manager(tmp_path,token='test-subscription');s.source.retrieval_policies={'tushare:daily':tushare_policy('daily',points=5000)};s.source.requests_per_minute=450
    calls=[]
    class Reply:
        status_code=200;headers={};content=b'{"code":0,"data":{"fields":["ts_code"],"items":[]}}'
    def post(*args,**kw):
        calls.append(clock.value);response=Reply()
        if len(calls)==1:response.status_code=429;response.headers={'Retry-After':'2'}
        return response
    monkeypatch.setattr(module.requests.Session,'post',post)
    s.source.fetch('daily',{},['ts_code'],tmp_path/'raw.json.gz')
    assert calls==pytest.approx([1000,1060])
    assert effective(s.source,'daily',450)==225 and effective(s.source,'adj_factor',450)==450


def test_real_clickhouse_inverse_reuses_unaffected_partition(live_manager,monkeypatch):
    from betalens_db_manager.pipeline_writer import ch_apply
    from betalens_db_manager.comparison import typed
    from betalens_db_manager.util import now
    s=live_manager;spec=canonical('market_daily_raw');s.ch.install({spec.name:spec})
    rows=[]
    for day in (date(2005,1,5),date(2026,10,20)):
        row=s._canonical_record(spec,{'asset_type':'stock','security_code':'000001.SZ','trade_date':day,'close':Decimal(10)},'daily','isolated-native',now(),1,{'date':str(day)},day)
        rows.append(typed(spec,row))
    original=compare(spec,rows,[],'fill_missing');ch_apply(s.ch,s.state,'native','write',spec,original)
    change=original['changes'][0]
    inverse={'changes':[{'row_key':change['row_key'],'before':change['after'],'after':None,'before_hash':change['after_hash'],'after_hash':None}]}
    statements=[];execute=s.ch.execute
    def observed(query,*a,**kw):
        statements.append(query);return execute(query,*a,**kw)
    monkeypatch.setattr(s.ch,'execute',observed)
    assert ch_apply(s.ch,s.state,'native','inverse',spec,inverse,inverse=True)['verified']
    assert any('ATTACH PARTITION' in q and '202610' in q for q in statements)
    assert s.ch.query(spec.name)['rows'][0]['trade_date']==date(2026,10,20)
    assert ch_apply(s.ch,s.state,'native','inverse',spec,inverse,inverse=True)['recovered']


def test_full_market_finance_seed_does_not_drop_vip_bindings(tmp_path):
    s=manager(tmp_path)
    plan=s.compile({'pipeline_version':2,'tables':['fundamental_periodic'],'source_id':'tushare','asset_types':['stock'],'start_date':'2005-01-01','end_date':'2026-10-27'})
    actual={r['endpoint'] for r in plan['requests']}
    assert {'income_vip','balancesheet_vip','cashflow_vip','fina_indicator_vip'}<=actual
    assert not actual&{'income','balancesheet','cashflow','fina_indicator'}
    assert all(r['parameters']['limit']==6000 for r in plan['requests'])


def test_real_failed_download_refreshes_child_checkpoint_on_retry(live_manager,pipeline_http,monkeypatch):
    from betalens_db_manager.pipeline import Pipeline
    from betalens_db_manager.source_registry import HTTPAdapter
    s=live_manager;request=prepare(s,pipeline_http)
    with registry_context(s.registry):
        s.initialize(s.schema_plan()['token']);s.schema.apply(s.schema.plan('pipeline_quotes')['token'])
    fetch=HTTPAdapter._fetch;failed=[False]
    def temporarily_unavailable(*args,**kwargs):
        if not failed[0]:failed[0]=True;raise RuntimeError('temporary transport failure')
        return fetch(*args,**kwargs)
    monkeypatch.setattr(HTTPAdapter,'_fetch',temporarily_unavailable)
    job=s.plan(request);assert Pipeline(s).run(job['id'])['status']=='partial'
    assert PipelineStore(s.state).units(job['id'],'normalize')[0]['status']=='failed'
    assert Pipeline(s).run(job['id'])['status']=='completed'
    with registry_context(s.registry):
        assert len(s.pg.query('pipeline_quotes')['rows'])==2 and len(s.ch.query('pipeline_quotes')['rows'])==2
