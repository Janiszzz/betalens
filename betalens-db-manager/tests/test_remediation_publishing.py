"""Failure injection at commit/rollback boundaries; no production connections."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import pytest

from betalens_db_manager.canonical import canonical
from betalens_db_manager.clickhouse import ClickHouse
from betalens_db_manager.commit_protocol import manifest_rows, remaining_changes, NeedsReconciliation
from betalens_db_manager.comparison import compare, row_hash, row_key, typed
from betalens_db_manager.publishing import publish, rollback
from betalens_db_manager.state import State
from betalens_db_manager.config import Settings

SPEC=canonical('security_dim')


class Target:
    def __init__(self):
        self.rows={};self.writes=0;self.inverses=0;self.control={}
        self.fail_final=False;self.fail_inverse=False;self.source_job=None
    def _find_existing(self,spec,rows):
        return {tuple(r.get(k) for k in spec.keys):deepcopy(self.rows[row_key(spec,r)]) for r in rows if row_key(spec,r) in self.rows}
    def compare(self,spec,rows,mode):return compare(spec,rows,list(self._find_existing(spec,rows).values()),mode)
    def commit_rows(self,job,target,spec,rows,mode,manifest):
        assert isinstance(manifest,dict), 'orchestrator must pass the original durable manifest'
        changes=remaining_changes(spec,manifest,list(self._find_existing(spec,manifest_rows(manifest)).values()))
        self.writes+=len(changes)
        for c in changes:self.rows[c['row_key']]=deepcopy(c['after'])
        return manifest['counts']
    def apply_rollback(self,spec,audits,rollback_id,connection=None):
        for a in audits:
            assert row_hash(spec,self.rows[a['row_key']])==a['after_hash']
            self.inverses+=1
            if a['before_value'] is None:self.rows.pop(a['row_key'])
            else:self.rows[a['row_key']]=deepcopy(a['before_value'])
    def refresh_coverage(self,name):pass
    def connect(self,*args,**kwargs):return nullcontext()
    def control_write(self,table,rows):
        if self.fail_final and table=='import_task' and any(r['status']=='completed' for r in rows):raise RuntimeError('final mirror fault')
        if self.fail_inverse and table=='import_row_audit' and any(r['task_id']!=self.source_job for r in rows):raise RuntimeError('inverse mirror fault')
        for r in rows:self.control.setdefault(table,{})[str(r.get('id',r.get('task_id',r.get('rollback_task_id'))))]=deepcopy(r)


class Service:
    def __init__(self,path):
        self.settings=SimpleNamespace(state_dir=path)
        self.state=State(path);self.pg=Target();self.ch=Target();self.rows=[]
    def _job(self,job,*args):return self.state.job(job)
    def _build_canonical(self,item):return {SPEC.name:self.rows},[]
    safe_error=staticmethod(str)
    def create(self,count=1):
        job=self.state.create_job({'request':{'targets':['postgres'],'sources':['file'],'mode':'refresh_source'}},'isolated')['id']
        self.pg.source_job=job
        self.rows=[typed(SPEC,{'security_code':f'{i+1:06}.SZ','asset_type':'stock','name':'intended','import_task_id':job}) for i in range(count)]
        c=self.pg.compare(SPEC,self.rows,'refresh_source')
        self.state.update_job(job,status='previewed',preview={'token':'confirmed','tables':[{'target':'postgres','table':SPEC.name,'signature':c['signature'],'counts':c['counts']}]})
        return job


def test_stage_race_keeps_concurrent_value_and_original_before(tmp_path,monkeypatch):
    s=Service(tmp_path);job=s.create();old={**s.rows[0],'name':'old','import_task_id':'older'}
    s.pg.rows[row_key(SPEC,old)]=old
    c=s.pg.compare(SPEC,s.rows,'refresh_source')
    s.state.update_job(job,preview={'token':'confirmed','tables':[{'target':'postgres','table':SPEC.name,'signature':c['signature'],'counts':c['counts']}]})
    original=s.state.stage_commit
    def race(*args):
        value=original(*args)
        s.pg.rows[row_key(SPEC,old)]={**old,'name':'concurrent','import_task_id':'later'}
        return value
    monkeypatch.setattr(s.state,'stage_commit',race)
    result=publish(s,job,'confirmed')
    assert result['status']=='partial' and s.pg.writes==0
    assert next(iter(s.pg.rows.values()))['name']=='concurrent'
    assert s.state.table_commit(job,'postgres',SPEC.name)['comparison']['changes'][0]['before']['name']=='old'


def test_prepared_without_audits_is_reconciled_before_real_rollback(tmp_path,monkeypatch):
    s=Service(tmp_path);job=s.create();original=s.state.finish_commit
    monkeypatch.setattr(s.state,'finish_commit',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('finish fault')))
    assert publish(s,job,'confirmed')['status']=='partial'
    assert s.pg.rows and not s.state.audits(job)
    monkeypatch.setattr(s.state,'finish_commit',original)
    result=rollback(s,job)
    assert result['results'][0]['status']=='completed' and not s.pg.rows
    assert s.state.job(job)['status']=='rolled_back' and len(s.state.audits(job))==1


def test_partial_batch_does_not_fabricate_audits_for_unwritten_rows(tmp_path,monkeypatch):
    s=Service(tmp_path);job=s.create(2)
    def partial(job,target,spec,rows,mode,manifest):
        c=manifest['changes'][0];s.pg.rows[c['row_key']]=deepcopy(c['after']);raise RuntimeError('partial batch')
    monkeypatch.setattr(s.pg,'commit_rows',partial)
    assert publish(s,job,'confirmed')['status']=='partial'
    assert len(s.pg.rows)==1
    result=rollback(s,job)
    assert result['results'][0]['status']=='completed' and not s.pg.rows
    assert len(s.state.audits(job))==1


def test_final_sync_failure_is_durable_and_repeat_does_not_write_business(tmp_path):
    s=Service(tmp_path);job=s.create();s.pg.fail_final=True
    result=publish(s,job,'confirmed')
    assert result['status']=='audit_pending' and result['data_status']=='completed'
    assert result['audit_sync_status']=='pending' and s.state.sync_records(job)[0]['error']
    s.pg.fail_final=False
    assert publish(s,job,'confirmed')['status']=='completed'
    assert s.pg.control['import_task'][job]['status']=='completed' and s.pg.writes==1


def test_rollback_sync_failure_survives_restart_and_retry(tmp_path):
    s=Service(tmp_path);job=s.create();assert publish(s,job,'confirmed')['status']=='completed'
    s.pg.fail_inverse=True
    result=rollback(s,job);rid=result['results'][0]['rollback_task_id']
    assert result['results'][0]['audit_sync_status']=='pending'
    assert State(tmp_path).sync_records(rid)[0]['error'] and not s.pg.rows
    s.pg.fail_inverse=False
    again=rollback(s,job)
    assert again['results'][0]['audit_sync_status']=='synced' and s.pg.inverses==1
    assert any(r['task_id']==rid for r in s.pg.control['import_row_audit'].values())


def test_manifest_guards_skipped_keys_and_cannot_be_overwritten(tmp_path):
    s=Service(tmp_path);job=s.create(2);old=[{**r,'name':'old'} for r in s.rows]
    c=compare(SPEC,[old[0],s.rows[1]],old,'refresh_source')
    assert c['counts']['skipped']==1
    with pytest.raises(NeedsReconciliation):remaining_changes(SPEC,c,[{**old[0],'name':'changed'},old[1]])
    s.state.stage_commit(job,'postgres',SPEC.name,c)
    modified=deepcopy(c);modified['changes'][0]['after']['name']='replacement'
    with pytest.raises(ValueError,match='不可覆盖'):s.state.stage_commit(job,'postgres',SPEC.name,modified)


def test_later_modification_blocks_rollback_without_overwriting(tmp_path):
    s=Service(tmp_path);job=s.create();publish(s,job,'confirmed')
    for r in s.pg.rows.values():r['name']='later';r['import_task_id']='later'
    result=rollback(s,job)
    assert result['results'][0]['status']=='blocked'
    assert next(iter(s.pg.rows.values()))['name']=='later' and s.pg.inverses==0


def test_clickhouse_uncontrolled_write_and_rollback_fail_before_io(tmp_path,monkeypatch):
    ch=ClickHouse(Settings(config_dir=tmp_path/'config',state_dir=tmp_path/'state'))
    monkeypatch.setattr(ch,'execute',lambda *a,**k:pytest.fail('no SQL may run without isolation'))
    with pytest.raises(NeedsReconciliation,match='受控独占写入口'):ch.apply_rollback(SPEC,[],'rollback')
    with pytest.raises(NeedsReconciliation,match='受控独占写入口'):ch.commit_rows('task','clickhouse',SPEC,[],'fill_missing',{})


def test_missing_commit_evidence_cannot_report_rollback_success(tmp_path):
    s=Service(tmp_path);job=s.create();s.state.update_job(job,status='partial')
    result=rollback(s,job)
    assert result['results'][0]['status']=='needs_reconciliation'
    assert s.state.job(job)['status']!='rolled_back'


def test_all_control_sync_failures_keep_completed_business_state(tmp_path,monkeypatch):
    s=Service(tmp_path);job=s.create();original=s.pg.control_write
    monkeypatch.setattr(s.pg,'control_write',lambda *a:(_ for _ in ()).throw(RuntimeError('mirror offline')))
    result=publish(s,job,'confirmed')
    assert result['status']=='audit_pending' and result['data_status']=='completed'
    receipt=s.state.ledger_rows('import_target_status',job)[0]
    assert receipt['status']=='completed' and s.pg.writes==1
    monkeypatch.setattr(s.pg,'control_write',original)
    assert publish(s,job,'confirmed')['status']=='completed' and s.pg.writes==1


def test_nullable_schema_addition_does_not_invalidate_historical_rollback(tmp_path,monkeypatch):
    from dataclasses import replace
    s=Service(tmp_path);job=s.create();publish(s,job,'confirmed')
    added=replace(SPEC,fields=SPEC.fields+({'name':'later_optional','type':'text','nullable':True},))
    monkeypatch.setattr('betalens_db_manager.publishing.canonical',lambda name:added)
    for row in s.pg.rows.values():row['later_optional']=None
    s.state.ledger_put('schema_operation','migration',{'plan':{'table':SPEC.name,'targets':[
        {'target':'postgres','actual':{'fields':list(SPEC.fields)}}]},'results':[{'target':'postgres','status':'completed'}]})
    assert rollback(s,job)['results'][0]['status']=='completed' and not s.pg.rows


def test_clickhouse_gate_identity_cannot_change_with_tmpdir_or_loopback_alias(tmp_path,monkeypatch):
    from betalens_db_manager.ch_writer import LocalDedicatedWriter
    monkeypatch.setenv('TMPDIR',str(tmp_path))
    a=LocalDedicatedWriter(Settings(ch_url='http://localhost:8123',ch_database='sample'))
    b=LocalDedicatedWriter(Settings(ch_url='http://127.0.0.1:8123/',ch_database='sample'))
    assert a.lock==b.lock and a.lock.parent.parent.name=='tmp'


def test_finished_unknown_query_still_requires_original_operation_recovery(tmp_path,monkeypatch):
    from betalens_db_manager.ch_writer import LocalDedicatedWriter
    from betalens_db_manager.util import atomic_json
    import uuid
    gate=LocalDedicatedWriter(Settings(ch_url='http://localhost:8123',ch_database='test_'+uuid.uuid4().hex))
    monkeypatch.setattr(gate,'verify',lambda adapter:None)
    adapter=SimpleNamespace(execute=lambda query:'0',_json_rows=lambda value:[{'type':'QueryFinish'}],_literal=lambda value:repr(value))
    atomic_json(gate.journal,{'inflight':True,'query_id':'past','operation':'commit:original'})
    try:
        with pytest.raises(NeedsReconciliation,match='先恢复原操作'):
            with gate.exclusive(adapter,'commit:unrelated'):pytest.fail('must remain blocked')
        with gate.exclusive(adapter,'commit:original'):pass
    finally:
        gate.journal.unlink(missing_ok=True);gate.lock.unlink(missing_ok=True)


@pytest.mark.parametrize('query',['DELETE FROM example WHERE code=1','DETACH TABLE example','ATTACH TABLE example','WITH 1 AS x INSERT INTO example SELECT x'])
def test_unlisted_clickhouse_mutations_cannot_bypass_writer_gate(tmp_path,monkeypatch,query):
    from contextlib import contextmanager
    ch=ClickHouse(Settings(config_dir=tmp_path/'config',state_dir=tmp_path/'state',ch_writer_mode='local_dedicated'))
    @contextmanager
    def blocked(operation):
        raise NeedsReconciliation('gate checked')
        yield
    monkeypatch.setattr(ch,'write_window',blocked)
    monkeypatch.setattr('requests.post',lambda *a,**k:pytest.fail('SQL reached server before writer verification'))
    with pytest.raises(NeedsReconciliation,match='gate checked'):ch.execute(query)
