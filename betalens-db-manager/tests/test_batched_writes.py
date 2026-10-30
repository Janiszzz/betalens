"""Set-based lookup, coalescing, and recovery must preserve original evidence."""
from copy import deepcopy
from dataclasses import replace
from datetime import date,datetime,timezone
from decimal import Decimal
from pathlib import Path
from threading import Event
from time import monotonic
import json
import pytest
import pyarrow as pa

from test_manager import manager,live_manager
from test_pipeline_v2 import prepare
from test_manual_review import initialize
from betalens_db_manager.canonical import CanonicalTable,registry_context
from betalens_db_manager.clickhouse import ClickHouse
from betalens_db_manager.clickhouse_keys import partition_filter
from betalens_db_manager.comparison import compare,typed
from betalens_db_manager.commit_protocol import remaining_changes,NeedsReconciliation
from betalens_db_manager.config import Settings
from betalens_db_manager.pipeline import Pipeline
from betalens_db_manager.pipeline_batching import WriteLane
from betalens_db_manager.pipeline_cache import write_rows,load_journal
from betalens_db_manager.pipeline_store import PipelineStore
from betalens_db_manager.pipeline_writer import ch_insert,ch_apply,pg_apply


def key_spec():
    return CanonicalTable('key_probe','查重测试','隔离测试','每键一行',(
        {'name':'code','type':'text','nullable':False},
        {'name':'trade_date','type':'date','nullable':False},
        {'name':'event_date','type':'date','nullable':False},
        {'name':'amount','type':'numeric','precision':28,'scale':8,'nullable':True},
    ),('code','trade_date'))


def test_external_key_upload_is_bounded_typed_and_readonly(monkeypatch):
    db=ClickHouse(Settings());spec=key_spec();calls=[]
    def post(url,**kw):
        calls.append(kw);query=kw['params']['query']
        data=kw['files']['_betalens_keys'][1]
        keys=pa.ipc.open_stream(data).read_all().to_pylist()
        assert len(keys)==50000 and keys[0]['code']=="quoted'!code"
        assert keys[0]['trade_date']==date(2026,10,20)
        assert 'data' not in kw and ' OR ' not in query
        output='{"partition":"202610"}' if 'SELECT DISTINCT _partition_id' in query else ''
        return type('Response',(),{'status_code':200,'text':output})()
    monkeypatch.setattr('requests.post',post)
    rows=[{'code':str(i),'trade_date':date(2026,10,20),'event_date':date(2026,10,20)} for i in range(50000)]
    rows[0]['code']="quoted'!code";rows.append(rows[0])
    assert db._find_existing(spec,rows)=={}
    assert len(calls)==2 and all(c['params'].get('query_id') is None for c in calls)
    query=calls[-1]['params']['query']
    assert 'INNER JOIN _betalens_keys' in query and "_partition_id IN ('202610')" in query
    assert "quoted'!code" not in query
    with pytest.raises(ValueError,match='仅用于只读'):
        db.execute('INSERT INTO target VALUES (1)',external_tables={'keys':{}})


def test_only_immutable_partition_keys_allow_direct_pruning():
    db=ClickHouse(Settings());spec=key_spec();rows=[{'event_date':date(1990,1,1)}]
    assert partition_filter(db,spec,rows)==''
    assert '199001' in partition_filter(db,replace(spec,keys=('code','event_date')),rows)
    assert partition_filter(db,replace(spec,keys=('code','event_date')),[{}])==''


def test_lane_flushes_small_batches_on_close_and_preserves_fifo():
    seen=[]
    lane=WriteLane(lambda items:seen.append([c for c,_ in items]) or [c for c,_ in items],max_wait=60,row_limit=50000)
    futures=[lane.submit(str(i),{'tables':{'quotes':[{'rows':1}]}}) for i in range(8)]
    lane.close();lane.join()
    assert [f.result() for f in futures]==list(map(str,range(8)))
    assert seen==[list(map(str,range(8)))]


def test_lane_flushes_on_deadline_and_propagates_failure():
    ran=Event()
    def fail(items):ran.set();raise RuntimeError('write failed')
    lane=WriteLane(fail,max_wait=.03,row_limit=50000)
    future=lane.submit('a',{'tables':{'quotes':[{'rows':1}]}})
    try:
        assert ran.wait(2)
        with pytest.raises(RuntimeError,match='write failed'):future.result(timeout=2)
    finally:lane.close();lane.join()


def test_bulk_receipts_are_atomic_on_manifest_conflict(tmp_path):
    s=manager(tmp_path);store=PipelineStore(s.state)
    store.prepare('job','old','clickhouse',{'table':'x','path':'original'})
    with pytest.raises(ValueError,match='不可覆盖'):
        store.prepare_many('job','clickhouse',[('new',{'table':'x'}),('old',{'table':'x','path':'changed'})])
    assert store.receipt('job','new','clickhouse') is None
    assert store.receipt('job','old','clickhouse')['manifest']['path']=='original'


def cohorts(s,rows,limit=50000):
    request=prepare(s,'http://127.0.0.1:1');request.update(commit_batch_rows=limit,mode='refresh_source')
    initialize(s);parent=s.plan(request);items=[]
    with registry_context(s.registry):
        spec=s.registry.spec('pipeline_quotes')
        for i,row in enumerate(rows):
            plan=deepcopy(parent['plan']);plan['pipeline_parent']=parent['id'];plan['pipeline_bucket']=str(i)
            child=s.state.create_job(plan,s.settings.fingerprint)
            record=write_rows(s.settings.state_dir/'cohorts'/child['id']/'rows.parquet',spec,[row])
            items.append((child['id'],{'tables':{'pipeline_quotes':[record]},'rejected':0}))
    return parent,items


def test_real_small_cohorts_merge_but_keep_each_receipt_and_rollback(live_manager,monkeypatch):
    s=live_manager;assert s.settings.ch_database.startswith('manager_test_')
    rows=[{'code':str(i),'business_date':date(2026,10,20),'price':Decimal(i+1)} for i in range(6)]
    parent,items=cohorts(s,rows);counts={'postgres':0,'clickhouse':0};audit=[]
    import betalens_db_manager.pipeline_writer as writer
    for target,name in [('postgres','pg_apply'),('clickhouse','ch_apply')]:
        original=getattr(writer,name)
        def wrapped(*args,_target=target,_original=original,**kwargs):
            counts[_target]+=1;return _original(*args,**kwargs)
        monkeypatch.setattr(writer,name,wrapped)
    original=ClickHouse.control_write
    def control(db,name,values):
        if name=='import_batch_receipt':audit.append(len(values))
        return original(db,name,values)
    monkeypatch.setattr(ClickHouse,'control_write',control)
    p=Pipeline(s)
    for target in counts:assert len(p._write_many(parent['id'],items,target))==6
    assert counts=={'postgres':1,'clickhouse':1} and audit==[6,6]
    receipts=p.store.receipts(parent['id']);assert len(receipts)==12
    with registry_context(s.registry):
        spec=s.registry.spec('pipeline_quotes')
        for r in receipts:
            assert r['status']=='completed' and r['result']['applied']==1
            assert len(load_journal(r['manifest'],spec)['changes'])==1
        assert len(s.ch.query('pipeline_quotes')['rows'])==len(s.pg.query('pipeline_quotes')['rows'])==6
        assert p.rollback(parent['id'])['status']=='rolled_back'
        assert not s.ch.query('pipeline_quotes')['rows'] and not s.pg.query('pipeline_quotes')['rows']


def test_real_overlap_remains_ordered_and_inverse_restores_each_version(live_manager):
    s=live_manager
    rows=[{'code':'A','business_date':date(2026,10,20),'price':Decimal(i)} for i in (10,20,30)]
    parent,items=cohorts(s,rows);p=Pipeline(s)
    for target in ('postgres','clickhouse'):p._write_many(parent['id'],items,target)
    with registry_context(s.registry):
        spec=s.registry.spec('pipeline_quotes')
        for target in ('postgres','clickhouse'):
            own=[r for r in p.store.receipts(parent['id']) if r['target']==target]
            changes=[load_journal(r['manifest'],spec)['changes'][0] for r in own]
            assert [c['before']['price'] if c['before'] else None for c in changes]==[None,Decimal(10),Decimal(20)]
        assert s.ch.query('pipeline_quotes')['rows'][0]['price']==Decimal(30)
        assert p.rollback(parent['id'])['status']=='rolled_back'
        assert s.ch.query('pipeline_quotes')['rows']==s.pg.query('pipeline_quotes')['rows']==[]


@pytest.mark.parametrize('target',['postgres','clickhouse'])
def test_real_combined_commit_recovers_lost_ack_and_protects_later_edit(live_manager,monkeypatch,target):
    s=live_manager;parent,items=cohorts(s,[{'code':str(i),'business_date':date(2026,10,20),'price':Decimal(i+1)} for i in range(3)])
    p=Pipeline(s);original=PipelineStore.acknowledge_many;once=[True]
    def lose_ack(store,job,t,entries):
        if t==target and once[0] and any(r.get('verified') for _,r,_ in entries):once[0]=False;raise RuntimeError('lost combined acknowledgement')
        return original(store,job,t,entries)
    monkeypatch.setattr(PipelineStore,'acknowledge_many',lose_ack)
    with pytest.raises(RuntimeError,match='lost combined'):p._write_many(parent['id'],items,target)
    p._write_many(parent['id'],items,target)
    own=p.store.receipts(parent['id']);assert len(own)==3 and all(r['status']=='completed' for r in own)
    with registry_context(s.registry):
        spec=s.registry.spec('pipeline_quotes');db=s.pg if target=='postgres' else s.ch
        current=db.query('pipeline_quotes')['rows'];assert len(current)==3
        changed={**current[0],'price':Decimal(99),'import_task_id':'later'}
        comparison=compare(spec,[changed],current,'refresh_source')
        if target=='postgres':pg_apply(db,spec,comparison)
        else:ch_apply(db,s.state,'later','edit',spec,comparison)
        with pytest.raises(NeedsReconciliation,match='目标数据已变化'):p.rollback(parent['id'],[target])
        assert db.query('pipeline_quotes')['rows'][0]['price']==Decimal(99)


def test_real_lookup_partition_discovery_and_50000_key_benchmark(live_manager):
    s=live_manager;db=s.ch;spec=key_spec();db.execute(db.ddl(spec))
    day=date(2026,10,20)
    rows=[{'code':str(i),'trade_date':day,'event_date':day,'amount':Decimal('12.12345678')} for i in range(50000)]
    # A matching key in an unexpected physical month must still be found.
    rows[0]['event_date']=date(2025,1,1)
    with db.write_window('test:seed'):ch_insert(db,spec,db.table(spec.name),rows)
    incoming=[{**r,'event_date':day} for r in rows]
    started=monotonic();actual=db._find_existing(spec,incoming);fresh_seconds=monotonic()-started
    assert len(actual)==50000 and actual[('0',day)]['event_date']==date(2025,1,1)
    assert actual[('1',day)]['amount']==Decimal('12.12345678')
    # Same data/keys, original 500-key OR algorithm, isolated fixture only.
    started=monotonic();legacy={}
    for offset in range(0,len(incoming),500):
        conditions=['('+ ' AND '.join('`'+key+'`='+db._literal(row[key]) for key in spec.keys)+')' for row in incoming[offset:offset+500]]
        for row in db._json_rows(db.execute(f'SELECT * FROM {db.table(spec.name)} FINAL WHERE '+ ' OR '.join(conditions)+' FORMAT JSONEachRow')):
            value=typed(spec,row);legacy[tuple(value[k] for k in spec.keys)]=value
    old_seconds=monotonic()-started
    assert actual==legacy
    report={'rows':50000,'legacy_lookup_requests':100,'new_lookup_requests':2,'legacy_seconds':round(old_seconds,3),'new_seconds':round(fresh_seconds,3)}
    Path('/tmp/betalens-batched-key-benchmark.json').write_text(json.dumps(report,indent=2))
    # Key-only partition discovery must not make a malformed old value look absent.
    frozen=compare(spec,[incoming[0]],[],'fill_missing')
    with pytest.raises(NeedsReconciliation):remaining_changes(spec,frozen,[actual[('0',day)]])


@pytest.mark.parametrize('target',['postgres','clickhouse'])
def test_real_group_respects_row_limit_and_precommit_conflict(live_manager,monkeypatch,target):
    import betalens_db_manager.pipeline_writer as writer
    s=live_manager
    rows=[{'code':str(i),'business_date':date(2026,10,20),'price':Decimal(i+1)} for i in range(5)]
    parent,items=cohorts(s,rows,limit=2);p=Pipeline(s);name='pg_apply' if target=='postgres' else 'ch_apply'
    original=getattr(writer,name);seen=[];inject=[True]
    def apply(*args,**kwargs):
        db=args[0];spec=args[1] if target=='postgres' else args[4];intent=args[2] if target=='postgres' else args[5]
        seen.append(len(intent['checks']))
        if inject[0]:
            inject[0]=False
            external={**intent['changes'][0]['after'],'price':Decimal(99),'import_task_id':'concurrent'}
            changed=compare(spec,[external],[],'refresh_source')
            if target=='postgres':original(db,spec,changed)
            else:original(db,s.state,'concurrent','write',spec,changed)
        return original(*args,**kwargs)
    monkeypatch.setattr(writer,name,apply)
    with pytest.raises(NeedsReconciliation):p._write_many(parent['id'],items,target)
    original_manifests=[r['manifest'] for r in p.store.receipts(parent['id'])]
    with pytest.raises(NeedsReconciliation):p._write_many(parent['id'],items,target)
    assert [r['manifest'] for r in p.store.receipts(parent['id'])]==original_manifests
    assert seen==[2,2]
    with registry_context(s.registry):
        db=s.pg if target=='postgres' else s.ch
        data=db.query('pipeline_quotes')['rows']
        assert len(data)==1 and data[0]['price']==Decimal(99)


def test_real_readback_detects_changed_target_and_keeps_receipt_recoverable(live_manager,monkeypatch):
    import betalens_db_manager.pipeline_writer as writer
    s=live_manager;parent,items=cohorts(s,[{'code':'A','business_date':date(2026,10,20),'price':Decimal(1)}])
    original=writer._ch_check;once=[True]
    def check(db,spec,stage,**kw):
        if once[0]:
            once[0]=False
            damaged={**kw['rows'][0],'price':Decimal(99)}
            ch_insert(db,spec,db.table(spec.name),[damaged])
        return original(db,spec,stage,**kw)
    monkeypatch.setattr(writer,'_ch_check',check)
    p=Pipeline(s)
    with pytest.raises(ValueError,match='集合核验失败'):p._write_many(parent['id'],items,'clickhouse')
    receipt=p.store.receipts(parent['id'])[0]
    assert receipt['status']=='failed' and Path(receipt['manifest']['path']).exists()
    with pytest.raises(NeedsReconciliation):p._write_many(parent['id'],items,'clickhouse')
