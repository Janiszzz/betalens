"""Concurrency must preserve database order, partial results and frozen revisions."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
from decimal import Decimal
from threading import Barrier
from types import SimpleNamespace
import pytest

from test_manager import manager,live_manager
from test_manual_review import seed,initialize,preview,perform,offline_targets
from betalens_db_manager.canonical import registry_context
from betalens_db_manager.comparison import compare
from betalens_db_manager.pipeline import Pipeline
from betalens_db_manager.pipeline_cache import save_journal
from betalens_db_manager.pipeline_store import PipelineStore
from betalens_db_manager.target_lanes import target_lanes,lane_values


def test_independent_lanes_keep_context_join_failure_and_return_requested_order():
    scope=ContextVar('test_lane_scope');scope.set('frozen-task');barrier=Barrier(2);finished=[]
    def work(target):
        assert scope.get()=='frozen-task';barrier.wait(timeout=5)
        if target=='postgres':raise ValueError('PG unavailable')
        finished.append(target);return 'CH completed'
    results=target_lanes(['postgres','clickhouse'],work)
    assert [r.target for r in results]==['postgres','clickhouse']
    assert finished==['clickhouse'] and results[1].value=='CH completed'
    with pytest.raises(ValueError,match='PG unavailable'):lane_values(results)


def test_metadata_collections_use_bounded_queries_not_one_connection_per_record(tmp_path,monkeypatch):
    s=manager(tmp_path);store=PipelineStore(s.state)
    for i in range(25):
        store.put('job',str(i),'normalize',{'roots':[i]})
        store.prepare('job',str(i),'clickhouse',{'table':'prices','path':str(i)})
    original=s.state.connect;calls=[]
    @contextmanager
    def counted():
        calls.append(1)
        with original() as c:yield c
    monkeypatch.setattr(s.state,'connect',counted)
    units=store.units('job');receipts=store.receipts('job')
    assert len(units)==len(receipts)==25 and len(calls)<=2
    assert [r['batch'] for r in receipts]==list(map(str,range(25)))
    assert all(isinstance(r['payload'],dict) for r in units)


def test_sync_batches_preserve_newer_revision_and_retry_failed_packets(tmp_path):
    s=manager(tmp_path);store=PipelineStore(s.state);calls=[];advance=[True];fail=[False]
    for i in range(257):store.prepare('job',str(i),'clickhouse',{'table':'prices','path':str(i)})
    def write(_,rows):
        calls.append(rows)
        if fail[0]:raise RuntimeError('audit offline')
        if advance[0]:
            advance[0]=False;store.acknowledge('job','0','clickhouse',{'applied':1})
    fake=SimpleNamespace(ch=SimpleNamespace(ddl=lambda _: 'ddl',execute=lambda _:None,control_write=write),safe_error=str)
    store.sync(fake,'job','clickhouse')
    assert [len(r) for r in calls]==[128,128,1]
    first=next(r for r in store.sync_status('job') if r['batch']=='0')
    assert first['revision']==2 and first['acknowledged']==1
    fail[0]=True
    with pytest.raises(RuntimeError,match='audit offline'):store.sync(fake,'job','clickhouse')
    assert next(r for r in store.sync_status('job') if r['batch']=='0')['acknowledged']==1
    fail[0]=False;store.sync(fake,'job','clickhouse')
    assert all(r['revision']==r['acknowledged'] and r['error'] is None for r in store.sync_status('job'))
    before=len(calls);store.sync(fake,'job','clickhouse');assert len(calls)==before


def test_manual_preview_reads_both_targets_concurrently_without_writes(tmp_path,monkeypatch):
    import betalens_db_manager.review_service as module
    s=manager(tmp_path);parent,_=seed(s);offline_targets(s,monkeypatch);barrier=Barrier(2)
    def read(*_):barrier.wait(timeout=5);return {}
    monkeypatch.setattr(module,'existing',read)
    result=preview(s,parent)
    assert {r['target'] for r in result['rows']}=={'postgres','clickhouse'}
    assert PipelineStore(s.state).receipts(parent['id'])==[]


def seeded_overlapping_receipts(s):
    import betalens_db_manager.pipeline_writer as writer
    parent,_=seed(s);initialize(s);store=PipelineStore(s.state)
    with registry_context(s.registry):
        spec=s.registry.spec('pipeline_quotes');previous=[]
        for number,price in enumerate((10,20)):
            incoming=[{'code':'A','business_date':date(2026,10,20),'price':Decimal(price)}]
            comparison=compare(spec,incoming,previous,'refresh_source')
            for target in ('postgres','clickhouse'):
                batch='overlap-'+str(number);manifest=save_journal(s.settings.state_dir/target/batch,spec,comparison,incoming)
                store.prepare(parent['id'],batch,target,manifest)
                result=writer.pg_apply(s.pg,spec,comparison) if target=='postgres' else writer.ch_apply(s.ch,s.state,parent['id'],batch,spec,comparison)
                store.acknowledge(parent['id'],batch,target,result)
            previous=incoming
    return parent,store


def test_real_rollback_parallel_targets_keep_reverse_order(live_manager,monkeypatch):
    import betalens_db_manager.pipeline_writer as writer
    s=live_manager;parent,store=seeded_overlapping_receipts(s);barrier=Barrier(2);seen={'postgres':[],'clickhouse':[]}
    pg,ch=writer.pg_apply,writer.ch_apply
    def track(target,comparison):
        if not seen[target]:barrier.wait(timeout=10)
        seen[target].append(comparison['changes'][0]['before']['price'])
    def pg_apply(db,spec,comparison):track('postgres',comparison);return pg(db,spec,comparison)
    def ch_apply(db,state,job,batch,spec,comparison,**kw):track('clickhouse',comparison);return ch(db,state,job,batch,spec,comparison,**kw)
    monkeypatch.setattr(writer,'pg_apply',pg_apply);monkeypatch.setattr(writer,'ch_apply',ch_apply)
    assert Pipeline(s).rollback(parent['id'])['status']=='rolled_back'
    assert seen=={'postgres':[Decimal(20),Decimal(10)],'clickhouse':[Decimal(20),Decimal(10)]}
    with registry_context(s.registry):assert s.pg.query('pipeline_quotes')['rows']==s.ch.query('pipeline_quotes')['rows']==[]
    assert len([r for r in store.receipts(parent['id']) if r['batch'].startswith('rollback:') and r['status']=='completed'])==4


@pytest.mark.parametrize('failed_target',['postgres','clickhouse'])
def test_real_rollback_one_target_failure_does_not_block_peer_and_retry_completes(live_manager,monkeypatch,failed_target):
    import betalens_db_manager.pipeline_writer as writer
    s=live_manager;parent,store=seeded_overlapping_receipts(s);name='pg_apply' if failed_target=='postgres' else 'ch_apply';original=getattr(writer,name)
    monkeypatch.setattr(writer,name,lambda *a,**k:(_ for _ in ()).throw(RuntimeError('isolated target failure')))
    with pytest.raises(RuntimeError,match='isolated target failure'):Pipeline(s).rollback(parent['id'])
    assert s.state.job(parent['id'])['status']=='partially_rolled_back'
    peer='clickhouse' if failed_target=='postgres' else 'postgres'
    assert len([r for r in store.receipts(parent['id']) if r['target']==peer and r['batch'].startswith('rollback:') and r['status']=='completed'])==2
    monkeypatch.setattr(writer,name,original)
    assert Pipeline(s).rollback(parent['id'],[failed_target])['status']=='rolled_back'
    with registry_context(s.registry):assert s.pg.query('pipeline_quotes')['rows']==s.ch.query('pipeline_quotes')['rows']==[]


def test_real_review_write_and_validation_readback_use_independent_lanes(live_manager,monkeypatch):
    import betalens_db_manager.pipeline_writer as writer
    import betalens_db_manager.pipeline_validation as validation
    from betalens_db_manager.review_service import ReviewService
    s=live_manager;parent,_=seed(s);initialize(s);p=preview(s,parent)
    ReviewService(s).confirm(parent['id'],'history_batch_0',p['token']);barrier=Barrier(2)
    pg,ch=writer.pg_apply,writer.ch_apply
    def pg_apply(*a,**k):barrier.wait(timeout=10);return pg(*a,**k)
    def ch_apply(*a,**k):barrier.wait(timeout=10);return ch(*a,**k)
    monkeypatch.setattr(writer,'pg_apply',pg_apply);monkeypatch.setattr(writer,'ch_apply',ch_apply)
    assert perform(s)['status']=='released'
    store=PipelineStore(s.state);unit=store.units(parent['id'],'normalize')[0];store.finish(parent['id'],unit['id'],unit['result'])
    barrier=Barrier(2);original=validation.existing
    def read(db,*a):
        barrier.wait(timeout=10)
        if db is s.pg:raise RuntimeError('PG read unavailable')
        return original(db,*a)
    monkeypatch.setattr(validation,'existing',read)
    report=validation.ValidationRunner(s).run(parent['id'],{'rules':['values']})
    assert report['status']=='failed'
    assert any(c['target']=='postgres' and 'PG read unavailable' in c.get('error','') for c in report['checks'])
    assert any(c['target']=='clickhouse' and c.get('rows')==1 for c in report['checks'])
