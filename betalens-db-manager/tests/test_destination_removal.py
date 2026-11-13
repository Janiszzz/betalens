"""Cached mixed cohorts obey the current destination scope on recovery."""
from types import SimpleNamespace
import json
import pytest

from betalens_db_manager.canonical import CanonicalTable, registry_context
from betalens_db_manager.comparison import row_key
from betalens_db_manager.config import Settings
from betalens_db_manager.pipeline_cache import write_rows
from betalens_db_manager.pipeline_commit import write_cohorts
from betalens_db_manager.pipeline_store import PipelineStore
from betalens_db_manager.state import State
from betalens_db_manager.destination_scope import remove_destination


@pytest.mark.parametrize('fail_write',[False,True])
def test_removed_destination_never_reads_or_prepares_cached_rows_but_other_tables_write(tmp_path,monkeypatch,fail_write):
    spec=CanonicalTable('other','其他表','test','one row',
        ({'name':'code','type':'text','nullable':False},),('code',))
    state=State(tmp_path/'state');store=PipelineStore(state)
    settings=Settings(state_dir=state.root,config_dir=tmp_path/'config')
    registry=SimpleNamespace(catalog=lambda:{'other':spec},spec=lambda name:{'other':spec}[name])
    request={'tables':['other'],'mode':'refresh_source','commit_batch_rows':100,
             'commit_batch_bytes':1000000,'commit_batch_max_wait_seconds':1}
    def metadata_only_job(*args,**kwargs):
        assert kwargs.get('metadata_only') is True
        return {'plan':{'request':request}}
    service=SimpleNamespace(settings=settings,state=state,registry=registry,pg=object(),
        schema=SimpleNamespace(inspect=lambda *args:{}),safe_error=str,
        _job=metadata_only_job)
    monkeypatch.setattr('betalens_db_manager.manager.DatabaseManagerService',lambda settings:service)
    monkeypatch.setattr('betalens_db_manager.schema_contract.compare_structure',lambda *args:{'ready':True})
    monkeypatch.setattr('betalens_db_manager.publishing.existing',lambda *args:{})
    monkeypatch.setattr(store,'sync',lambda *args:None)
    writes=[]
    class Result(dict):pass
    def apply(db,definition,intent):
        writes.append((definition.name,intent))
        if fail_write:raise RuntimeError('write unavailable')
        result=Result(applied=len(intent['changes']),verified=True)
        result.applied_keys={change['row_key'] for change in intent['changes']}
        return result
    monkeypatch.setattr('betalens_db_manager.pipeline_writer.pg_apply',apply)
    with registry_context(registry):
        row=write_rows(tmp_path/'other.parquet',spec,[{'code':'A'}])
        snapshots=[('child',{'tables':{'trade_status_daily':[{'path':'/must-not-be-read'}],'other':[row]}})]
        if fail_write:
            with pytest.raises(RuntimeError,match='write unavailable'):
                write_cohorts(SimpleNamespace(s=service,store=store),'parent',snapshots,'postgres')
        else:
            result=write_cohorts(SimpleNamespace(s=service,store=store),'parent',snapshots,'postgres')
            assert result[0]['status']=='completed'
    assert [name for name,_ in writes]==['other']
    receipts=store.receipts('parent')
    assert len(receipts)==1 and receipts[0]['manifest']['table']=='other'
    assert receipts[0]['status']==('failed' if fail_write else 'completed')
    if fail_write:
        with state.connect() as c:
            event=json.loads(c.execute("SELECT value FROM event_log WHERE event='pipeline_table_write_failed'").fetchone()[0])
        assert event=={'target':'postgres','table':'other','child_ids':['child'],'error':'write unavailable'}


def test_scope_removal_preserves_completed_audit_and_only_prunes_remaining_work(tmp_path):
    state=State(tmp_path/'state');store=PipelineStore(state)
    plan={'pipeline_version':2,'revision':'manager-v1','request':{
        'tables':['trade_status_daily','other'],'targets':['postgres','clickhouse']},
        'mapping_snapshots':{'trade_status_daily':{},'other':{}}}
    parent=state.create_job(plan,'target')['id']
    children=[state.create_job({**plan,'pipeline_parent':parent},'target')['id'] for _ in range(2)]
    for number,child in enumerate(children):
        store.put(parent,str(number),'normalize',{})
        store.finish(parent,str(number),{'child_id':child,'tables':{
            'trade_status_daily':[{'path':'immutable-status.parquet'}],'other':[{'path':'other.parquet'}]}})
        for target in ('postgres','clickhouse'):
            if number==1 and target=='clickhouse':continue
            marker='stored:'+child+':'+target
            store.put(parent,marker,'stored',{'child_id':child,'target':target});store.finish(parent,marker,{})
    store.prepare(parent,'finished','postgres',{'table':'trade_status_daily','path':'original-journal'})
    store.acknowledge(parent,'finished','postgres',{'verified':True})
    store.prepare(parent,'waiting','clickhouse',{'table':'trade_status_daily','path':'pending-journal'})
    with state.connect() as c:
        snapshots=list(c.execute('SELECT hash,value FROM plan_snapshots'))
    result=remove_destination(state,parent,'trade_status_daily',tmp_path/'private'/'backup.json')
    assert result['children_changed']==1 and result['cached_cohorts_changed']==1
    assert result['pending_receipts_cancelled']==1
    assert state.plan(parent)['request']['tables']==['other']
    assert state.plan(children[0])['request']['tables']==['trade_status_daily','other']
    assert state.plan(children[1])['request']['tables']==['other']
    assert 'trade_status_daily' in store.get(parent,'0')['result']['tables']
    assert store.get(parent,'1')['result']['tables']=={'other':[{'path':'other.parquet'}]}
    assert store.receipt(parent,'finished','postgres')['status']=='completed'
    assert store.receipt(parent,'waiting','clickhouse')['status']=='cancelled'
    with state.connect() as c:assert list(c.execute('SELECT hash,value FROM plan_snapshots'))==snapshots
    backup=json.loads((tmp_path/'private'/'backup.json').read_text())
    assert len(backup['plans'])==2 and 'trade_status_daily' in json.loads(backup['units'][0]['result'])['tables']
    assert remove_destination(state,parent,'trade_status_daily',tmp_path/'never-written')==result


def test_scope_removal_refuses_a_live_writer(tmp_path):
    state=State(tmp_path/'state');PipelineStore(state)
    parent=state.create_job({'pipeline_version':2,'revision':'manager-v1',
        'request':{'tables':['trade_status_daily','other'],'targets':['postgres']}},'target')['id']
    state.enqueue(parent,'pipeline_run')
    with pytest.raises(ValueError,match='执行器停止'):
        remove_destination(state,parent,'trade_status_daily',tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()
    assert state.plan(parent)['request']['tables']==['trade_status_daily','other']
