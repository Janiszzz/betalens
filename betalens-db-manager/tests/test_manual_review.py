"""Human review preserves original evidence, frozen intent and rollback ownership."""
from copy import deepcopy
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient

from test_manager import manager,live_manager
from test_pipeline_v2 import prepare
from betalens_db_manager.api import create_app
from betalens_db_manager.canonical import registry_context
from betalens_db_manager.pipeline import Pipeline
from betalens_db_manager.pipeline_cache import write_rows
from betalens_db_manager.pipeline_store import PipelineStore
from betalens_db_manager.review_service import ReviewService
from betalens_db_manager.util import digest,dumps,now,atomic_json


def seed(s,bucket='history_batch_0',cached=True):
    request=prepare(s,'http://127.0.0.1:1')
    parent=s.plan(request);plan=deepcopy(parent['plan']);plan['pipeline_parent']=parent['id'];plan['pipeline_bucket']=bucket
    child=s.state.create_job(plan,s.settings.fingerprint);store=PipelineStore(s.state)
    tables={}
    if cached:
        with registry_context(s.registry):
            spec=s.registry.spec('pipeline_quotes')
            tables={'pipeline_quotes':[write_rows(s.settings.state_dir/'fixture'/child['id']/'accepted.parquet',spec,[{'code':'A','business_date':date(2026,10,20),'price':Decimal('12.34567890')}])]}
    store.put(parent['id'],bucket,'normalize',{'roots':[]})
    store.finish(parent['id'],bucket,{'child_id':child['id'],'tables':tables,'rejected':1,'reject_sample':[{'reason':'源数据业务主键为空','source_row':2,'raw':{'code':None,'business_date':'20261021'}}]},error='转换拒收，等待重试')
    s.state.ledger_put('pipeline_child',parent['id']+':'+bucket,{'task_id':parent['id'],'child_id':child['id'],'bucket':bucket})
    s.state.update_job(parent['id'],status='partial')
    return parent,child


def preview(s,parent,action='release',**extra):
    r=ReviewService(s);detail=r.detail(parent['id'],'history_batch_0')
    return r.preview(parent['id'],'history_batch_0',{'action':action,'note':'已人工核对来源与记录','version':detail['version'],**extra})


def perform(s):
    task=s.state.claim(allow_ui=False);assert task and task['operation']=='pipeline_review'
    try:
        result=ReviewService(s).execute(task['payload']['decision_id']);s.state.finish(task['id']);return result
    except BaseException as exc:
        s.state.finish(task['id'],str(exc));raise


def initialize(s):
    with registry_context(s.registry):
        s.initialize(s.schema_plan()['token']);s.schema.apply(s.schema.plan('pipeline_quotes')['token'])


def offline_targets(s,monkeypatch):
    import betalens_db_manager.review_service as module
    import betalens_db_manager.schema_contract as contracts
    monkeypatch.setattr(module,'existing',lambda *a: {})
    monkeypatch.setattr(s.schema,'inspect',lambda *a: {})
    monkeypatch.setattr(contracts,'compare_structure',lambda *a: {'ready':True})


def seed_missing_archive(s):
    import gzip,hashlib
    from betalens_db_manager.downloader import TaskDownloader
    parent,child=seed(s,cached=False);plan=deepcopy(child['plan'])
    root=plan['requests'][0];other={**root,'parameters':{**root['parameters'],'start_date':'20261021','end_date':'20261021'}}
    plan['requests']=[root,other]
    with s.state.connect() as c:c.execute('UPDATE jobs SET plan=? WHERE id=?',(dumps(plan),child['id']))
    child=s.state.job(child['id']);request=TaskDownloader(s,child).prepared(root)[0]
    data={'fields':['ticker','day','price'],'items':[['A','20261020','12.34567890']]}
    raw=dumps({'code':0,'data':data}).encode();path=s.settings.state_dir/'fixture'/child['id']/'source.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wb') as handle:handle.write(raw)
    sha=hashlib.sha256(raw).hexdigest();atomic_json(path.with_suffix('.meta.json'),{'fetched_at':now(),'source_id':'pipeline_http','endpoint':'prices','sha256':sha})
    s.state.slice(child['id'],TaskDownloader.key(request),'prices',request,status='downloaded',archive=str(path),sha=sha,rows=1,result={'children':[],'complete':True})
    unit=PipelineStore(s.state).get(parent['id'],'history_batch_0')
    PipelineStore(s.state).finish(parent['id'],'history_batch_0',{**unit['result'],'reject_sample':[{'code':'missing_request','reason':'固定请求缺少归档分片','blocking':True}]},error='来源覆盖不完整')
    return parent,s.state.job(child['id'])


def test_review_list_details_and_reject_export_are_read_only(tmp_path,monkeypatch):
    s=manager(tmp_path);parent,child=seed(s);original=digest(s.state.job(parent['id'])['plan'])
    monkeypatch.setattr(s.pg,'connect',lambda *a,**k:(_ for _ in ()).throw(AssertionError('拒绝和导出不访问业务库')))
    monkeypatch.setattr(s.ch,'execute',lambda *a,**k:(_ for _ in ()).throw(AssertionError('拒绝和导出不访问业务库')))
    with TestClient(create_app(s)) as client:
        base='/api/tasks/'+parent['id']+'/reviews'
        rows=client.get(base).json();assert rows['total']==1 and rows['counts']['pending']==1
        detail=client.get(base+'/history_batch_0').json();assert detail['cached_rows']==1 and detail['samples'][0]['source_row']==2
        bad=client.post(base+'/history_batch_0/reject',json={'version':'stale','note':'不采用'});assert bad.status_code==409
        rejected=client.post(base+'/history_batch_0/reject',json={'version':detail['version'],'note':'不采用缺失关键字段的来源记录'})
        assert rejected.status_code==200,rejected.text
        exported=client.get(rejected.json()['export_url']);assert exported.status_code==200
        assert 'attachment' in exported.headers['content-disposition']
        assert exported.json()['decisions'][0]['note']=='不采用缺失关键字段的来源记录'
        assert client.get(base).json()['counts']['rejected']==1
        assert client.post(base+'/history_batch_0/reopen',json={}).status_code==200
        assert client.get(base).json()['counts']['pending']==1
    assert digest(s.state.job(parent['id'])['plan'])==original
    assert PipelineStore(s.state).receipts(parent['id'])==[]
    assert s.state.job(parent['id'])['tasks']==[]


def test_release_requires_note_and_valid_cache_and_stable_snapshot(tmp_path,monkeypatch):
    import betalens_db_manager.review_service as module
    s=manager(tmp_path);parent,child=seed(s);r=ReviewService(s);detail=r.detail(parent['id'],'history_batch_0')
    offline_targets(s,monkeypatch)
    with pytest.raises(ValueError,match='说明'):r.preview(parent['id'],'history_batch_0',{'action':'release','version':detail['version'],'note':''})
    value=preview(s,parent)
    queued=r.confirm(parent['id'],'history_batch_0',value['token'])
    assert r.confirm(parent['id'],'history_batch_0',value['token'])==queued
    assert s.state.job(parent['id'])['tasks']==[]
    assert s.state.job(queued['operation_job'])['plan']['pipeline_parent']==parent['id']
    assert r.counts(parent['id'])['queued']==1
    other=manager(tmp_path/'empty');empty,_=seed(other,cached=False);offline_targets(other,monkeypatch)
    with pytest.raises(ValueError,match='没有满足'):preview(other,empty)


def test_repair_checks_scope_keys_types_and_system_fields(tmp_path,monkeypatch):
    import betalens_db_manager.review_service as module
    s=manager(tmp_path);parent,child=seed(s);offline_targets(s,monkeypatch)
    invalid=[({'other_table':[]},'修补表'),({'pipeline_quotes':[{'code':'B','price':'1'}]},'主键'),
             ({'pipeline_quotes':[{'code':'B','business_date':'2001-01-01','price':'1'}]},'日期范围'),
             ({'pipeline_quotes':[{'code':'B','business_date':'2026-10-21','price':'NaN'}]},'类型'),
             ({'pipeline_quotes':[{'code':'B','business_date':'2026-10-21','import_task_id':'forged'}]},'审计字段')]
    for rows,reason in invalid:
        with pytest.raises(ValueError,match=reason):preview(s,parent,'repair',rows=rows)
    value=preview(s,parent,'repair',rows={'pipeline_quotes':[{'code':'B','business_date':'2026-10-21','price':'9.12345678'}]})
    assert sum(r['rows'] for r in value['rows'] if r['target']=='postgres')==2
    assert s.state.job(parent['id'])['status']=='partial'


def test_active_task_and_changed_preview_reject_approval(tmp_path,monkeypatch):
    import betalens_db_manager.review_service as module
    s=manager(tmp_path);parent,_=seed(s);offline_targets(s,monkeypatch)
    value=preview(s,parent);store=PipelineStore(s.state);unit=store.get(parent['id'],'history_batch_0')
    store.finish(parent['id'],'history_batch_0',{**unit['result'],'cache_key':'new'},error='new reject')
    with pytest.raises(ValueError,match='已变化'):ReviewService(s).confirm(parent['id'],'history_batch_0',value['token'])
    s.state.enqueue(parent['id'],'pipeline_run',{'background':True})
    with pytest.raises(ValueError,match='仍在执行'):preview(s,parent)


def test_api_preserves_confirmation_token_without_exposing_source_secret(tmp_path,monkeypatch):
    s=manager(tmp_path,token='source-secret-must-remain-private');parent,_=seed(s);offline_targets(s,monkeypatch)
    with TestClient(create_app(s)) as client:
        base='/api/tasks/'+parent['id']+'/reviews/history_batch_0'
        detail=client.get(base).json()
        response=client.post(base+'/preview',json={'action':'release','note':'人工核对通过','version':detail['version']})
        assert response.status_code==200,response.text
        value=response.json();assert len(value['token'])==64 and value['token']==digest({k:v for k,v in value.items() if k!='token'})
        assert 'source-secret-must-remain-private' not in response.text
        assert client.post(base+'/confirm',json={'token':value['token']}).status_code==200


def test_review_blocks_obey_original_byte_and_row_budget(tmp_path):
    import pyarrow as pa
    from betalens_db_manager.pipeline_cache import schema
    from betalens_db_manager.review_service import bounded_blocks
    s=manager(tmp_path);seed(s)
    with registry_context(s.registry):
        spec=s.registry.spec('pipeline_quotes');rows=[{'code':str(i)+'x'*200,'business_date':date(2026,10,20),'price':Decimal(1)} for i in range(10)]
        blocks=list(bounded_blocks(spec,rows,10,1024))
        assert len(blocks)>1 and sum(map(len,blocks))==10
        assert all(pa.Table.from_pylist(b,schema=schema(spec)).nbytes<=1024 for b in blocks)
        with pytest.raises(ValueError,match='单行'):list(bounded_blocks(spec,[{**rows[0],'code':'x'*5000}],10,1024))


def test_manual_coverage_context_is_scoped_and_preserves_other_checks(tmp_path):
    from betalens_db_manager.response_validation import validate_archives,manual_coverage_review
    s=manager(tmp_path);parent,child=seed_missing_archive(s)
    assert {e['code'] for e in validate_archives(s,child)}=={'missing_request'}
    with manual_coverage_review(child['id']):assert validate_archives(s,child)==[]
    assert {e['code'] for e in validate_archives(s,child)}=={'missing_request'}
    Path(child['slices'][0]['archive']).write_bytes(b'corrupt')
    with manual_coverage_review(child['id']):assert validate_archives(s,child)


def test_real_explicit_coverage_acceptance_releases_valid_rows_and_keeps_evidence(live_manager):
    from betalens_db_manager.response_validation import validate_archives
    s=live_manager;parent,child=seed_missing_archive(s);initialize(s);r=ReviewService(s)
    detail=r.detail(parent['id'],'history_batch_0');assert detail['coverage_acceptance_available'] and detail['can_release']
    with pytest.raises(ValueError,match='没有满足'):preview(s,parent)
    value=preview(s,parent,accept_coverage=True);assert value['coverage_accepted'] is True
    decision=r.confirm(parent['id'],'history_batch_0',value['token']);result=perform(s)
    assert result['status']=='released' and result['coverage_accepted'] is True
    assert {e['code'] for e in validate_archives(s,child)}=={'missing_request'}
    with registry_context(s.registry):
        for db in (s.pg,s.ch):
            rows=db.query('pipeline_quotes')['rows'];assert len(rows)==1 and rows[0]['data_quality_status']=='manual_release_with_exclusions'
    assert Pipeline(s).rollback(parent['id'])['status']=='rolled_back'


def test_real_release_repair_idempotence_and_original_task_rollback(live_manager):
    s=live_manager;parent,child=seed(s);initialize(s);frozen=digest(s.state.job(parent['id'])['plan'])
    value=preview(s,parent,'repair',rows={'pipeline_quotes':[{'code':'B','business_date':'2026-10-21','price':'9.12345678'}]})
    r=ReviewService(s);decision=r.confirm(parent['id'],'history_batch_0',value['token']);assert perform(s)['status']=='repaired'
    assert r.execute(decision['id'])['status']=='repaired'
    with registry_context(s.registry):
        for db in (s.pg,s.ch):
            rows=db.query('pipeline_quotes')['rows'];assert len(rows)==2
            assert {str(row['price']) for row in rows}=={'12.34567890','9.12345678'} or {Decimal(row['price']) for row in rows}=={Decimal('12.34567890'),Decimal('9.12345678')}
            assert all(row['ingested_at'] is not None for row in rows)
    assert len(PipelineStore(s.state).receipts(parent['id']))==4
    assert digest(s.state.job(parent['id'])['plan'])==frozen
    assert PipelineStore(s.state).get(parent['id'],'history_batch_0')['status']=='failed'
    rolled=Pipeline(s).rollback(parent['id']);assert rolled['status']=='rolled_back'
    with registry_context(s.registry):
        for db in (s.pg,s.ch):assert db.query('pipeline_quotes')['rows']==[]


def test_real_preview_rejects_concurrent_target_change(live_manager):
    s=live_manager;parent,_=seed(s);initialize(s);value=preview(s,parent)
    with s.pg.connect() as conn,conn.cursor() as cur:
        from psycopg2 import sql
        from betalens_db_manager.storage import relation
        cur.execute(sql.SQL('INSERT INTO {} (code,business_date,price) VALUES(%s,%s,%s)').format(relation(s.settings,'pipeline_quotes')),('A','2026-10-20','99'))
    r=ReviewService(s);r.confirm(parent['id'],'history_batch_0',value['token'])
    with pytest.raises(Exception):perform(s)
    assert r.counts(parent['id'])['failed']==1
    with registry_context(s.registry):assert s.pg.query('pipeline_quotes')['rows'][0]['price']==Decimal('99')


def test_real_half_failure_retries_same_receipts_without_duplicate_writes(live_manager,monkeypatch):
    import betalens_db_manager.pipeline_writer as module
    s=live_manager;parent,_=seed(s);initialize(s);r=ReviewService(s)
    value=preview(s,parent);r.confirm(parent['id'],'history_batch_0',value['token'])
    original=module.ch_apply
    monkeypatch.setattr(module,'ch_apply',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('isolated CH failure')))
    with pytest.raises(RuntimeError,match='CH failure'):perform(s)
    with registry_context(s.registry):assert len(s.pg.query('pipeline_quotes')['rows'])==1
    monkeypatch.setattr(module,'ch_apply',original)
    r.retry(parent['id'],'history_batch_0');assert perform(s)['status']=='released'
    with registry_context(s.registry):
        assert len(s.pg.query('pipeline_quotes')['rows'])==len(s.ch.query('pipeline_quotes')['rows'])==1
    assert len(PipelineStore(s.state).receipts(parent['id']))==2
