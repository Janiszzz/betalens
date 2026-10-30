from pathlib import Path
import json

from fastapi.testclient import TestClient
import pytest

from betalens_db_manager.config import Settings
from betalens_db_manager.monitor import create_monitor_app
from betalens_db_manager.pipeline_store import PipelineStore
from betalens_db_manager.state import State
from betalens_db_manager.util import now


@pytest.fixture
def task(tmp_path):
    settings=Settings(state_dir=tmp_path/'state',config_dir=tmp_path/'config',token='private-monitor-test-token')
    state=State(settings.state_dir)
    job=state.create_job({'request':{'tables':['quotes'],'targets':['postgres','clickhouse'],
                                  'start_date':'2026-01-01','end_date':'2026-01-03'}},settings.fingerprint)
    state.update_job(job['id'],status='running')
    state.ledger_put('pipeline_work_plan',job['id'],{'basis':'logical_requests','request_ids':['one','two','three'],
        'batch_ids':['a','b','c'],'requests_total':3,'batches_total':3,'version':'frozen','frozen_at':now()})
    for key,status in [('one','downloaded'),('two','pending'),('three','failed')]:
        state.slice(job['id'],key,'daily',{'source_id':'tushare','endpoint':'daily','parameters':{}},status=status)
    store=PipelineStore(state)
    for key,status in [('a','completed'),('b','failed'),('c','pending')]:
        store.put(job['id'],key,'normalize',{})
        if status!='pending':store.finish(job['id'],key,{'child_id':'child_'+key},error='rejected' if status=='failed' else None)
    store.put(job['id'],'stored:a:pg','stored',{'child_id':'child_a','target':'postgres'})
    store.finish(job['id'],'stored:a:pg',{})
    store.prepare(job['id'],'a:quotes:0','postgres',{'table':'quotes'})
    store.acknowledge(job['id'],'a:quotes:0','postgres',{'applied':42})
    with state.connect() as conn:
        conn.executemany('INSERT INTO event_log(job,created,event,value) VALUES(?,?,?,?)',
                        [(job['id'],now(),'request_started',json.dumps({'endpoint':'daily','message':str(i)})) for i in range(800)])
        conn.execute('INSERT INTO event_log(job,created,event,value) VALUES(?,?,?,?)',('other',now(),'request_started','{}'))
    (settings.state_dir/'worker-heartbeat.json').write_text(json.dumps({'at':now(),'status':'running','job':job['id']}))
    return settings,state,job['id']


def test_latest_logs_and_fixed_progress_do_not_initialize_or_modify_state(task,monkeypatch):
    settings,state,job=task
    before=state.path.read_bytes()
    monkeypatch.setattr(State,'__init__',lambda *_:pytest.fail('监控不得初始化状态数据库'))
    with TestClient(create_monitor_app(settings)) as client:
        assert client.get('/api/health').json()=={'status':'ok','read_only':True}
        rows=client.get('/api/tasks/'+job+'/logs?limit=10000').json()
        assert len(rows)==500 and rows[0]['id']==301 and rows[-1]['id']==800
        status=client.get('/api/tasks/'+job+'/status').json()
        progress=status['progress']
        assert progress['requests']=={'total':3,'processed':2,'failed':1}
        assert progress['batches']=={'total':3,'processed':2,'passed':1,'rejected':1}
        assert progress['writes']['postgres']['written']==1
        assert progress['writes']['clickhouse']['written']==0
        assert status['writes'][0]['applied']==42 and status['worker']['fresh']
        assert 'mapping_snapshots' not in str(status) and 'source_contracts' not in str(status)
        assert client.post('/api/tasks/'+job+'/status').status_code==405
        assert client.post('/api/tasks/'+job+'/start',json={}).status_code==404
    assert state.path.read_bytes()==before


def test_legacy_log_secrets_are_redacted_and_raw_value_is_not_returned(task):
    settings,state,job=task
    with state.connect() as conn:
        conn.execute('INSERT INTO event_log(job,created,event,value) VALUES(?,?,?,?)',
                     (job,now(),'request_started',json.dumps({'message':'token='+settings.token,'api_key':'other-private-value'})))
    with TestClient(create_monitor_app(settings)) as client:
        response=client.get('/api/tasks/'+job+'/logs?limit=1')
        assert settings.token not in response.text and 'other-private-value' not in response.text
        assert 'value' not in response.json()[0]


def test_wrong_state_path_is_never_created_and_bad_ids_are_rejected(tmp_path):
    settings=Settings(state_dir=tmp_path/'missing',config_dir=tmp_path/'unused')
    with TestClient(create_monitor_app(settings)) as client:
        assert client.get('/api/health').status_code==503
        assert client.get('/api/tasks/invalid/status').status_code==404
        assert client.get('/api/tasks/invalid/logs').status_code==404
    assert not settings.state_dir.exists()
