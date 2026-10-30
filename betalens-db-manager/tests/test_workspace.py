from pathlib import Path
from dataclasses import replace
import json
import pytest
from fastapi.testclient import TestClient
from test_manager import manager,table_definition,live_manager
from betalens_db_manager.api import create_app
from betalens_db_manager.schema_bundle import SchemaBundle
from betalens_db_manager.state import State
from betalens_db_manager.util import dumps,now


def test_manifest_is_complete_stable_and_preserves_native_control_types(tmp_path):
    s=manager(tmp_path);s.registry.save(table_definition('workspace_quotes'));bundle=SchemaBundle(s)
    first=bundle.manifest();second=bundle.manifest();assert first['version']==second['version']
    assert Path(first['path']).name=='schema-manifest.yaml' and Path(first['path']).exists()
    entries={e['name']:e for e in first['tables']}
    assert {'workspace_quotes','import_task','import_batch_receipt','trade_calendar_versions','reference_scope'}<=set(entries)
    assert next(f for f in entries['import_task']['fields'] if f['name']=='request')['physical_types']=={'postgres':'jsonb','clickhouse':'String'}
    assert 'GENERATED ALWAYS AS IDENTITY' in entries['import_row_audit']['fields'][0]['physical_types']['postgres']
    assert entries['workspace_quotes']['role']=='business' and entries['reference_scope']['role']=='audit'
    assert entries['trade_calendar_versions']['dependencies']==['reference_revision','reference_scope','reference_publication_event']
    assert 'import_row_audit' in entries['workspace_quotes']['dependencies']


def test_inspection_and_plan_only_query_metadata_and_are_target_scoped(tmp_path,monkeypatch):
    s=manager(tmp_path);bundle=SchemaBundle(s);observed=[]
    monkeypatch.setattr(bundle,'_inspect_pg',lambda:observed.append('pg') or {'database_exists':False,'tables':{}})
    monkeypatch.setattr(bundle,'_inspect_ch',lambda:(_ for _ in ()).throw(AssertionError('不得检查未选库')))
    result=bundle.inspect(['postgres'],['market_daily_raw']);assert observed==['pg']
    assert result['postgres']['tables'][0]['status']=='missing'
    plan=bundle.plan({'postgres':['market_daily_raw']})
    assert plan['expanded']['postgres']==sorted(plan['expanded']['postgres'],key=lambda n:(n=='market_daily_raw',n))
    assert set(a['action'] for a in plan['actions'])=={'create'}
    with pytest.raises(ValueError):bundle.plan({'postgres':['unknown_table']})
    with pytest.raises(ValueError):bundle.inspect(['bad'])
    with pytest.raises(ValueError):bundle.inspect([])


def test_batch_api_confirmation_survives_redaction_and_rejects_stale_preview(tmp_path,monkeypatch):
    s=manager(tmp_path)
    snapshot={'database_exists':False,'tables':{}}
    monkeypatch.setattr(SchemaBundle,'_inspect_pg',lambda _:snapshot)
    with TestClient(create_app(s)) as client:
        result=client.post('/api/schema/batch-plan',json={'selections':{'postgres':['security_dim']}})
        assert result.status_code==200,result.text
        plan=result.json();assert len(plan['token'])==64 and '[redacted]' not in plan['token']
        assert s.state.ledger_get('schema_bundle_plan',plan['token'])['snapshot_hash']==plan['snapshot_hash']
        snapshot['tables']['security_dim']={'exists':True,'fields':[],'keys':[],'indexes':[]}
        stale=client.post('/api/schema/batch-apply',json={'token':plan['token']})
        assert stale.status_code==409 and '重新预览' in stale.text
        snapshot['tables'].clear()
        second=client.post('/api/schema/batch-plan',json={'selections':{'postgres':['security_dim']}}).json()
        s.registry.save(table_definition('new_definition'))
        assert '配置或定义已变化' in client.post('/api/schema/batch-apply',json={'token':second['token']}).text


def test_unavailable_database_does_not_hide_other_target_metadata(tmp_path,monkeypatch):
    s=manager(tmp_path);bundle=SchemaBundle(s)
    monkeypatch.setattr(bundle,'_inspect_pg',lambda:(_ for _ in ()).throw(PermissionError('metadata denied')))
    monkeypatch.setattr(bundle,'_inspect_ch',lambda:{'database_exists':False,'tables':{}})
    plan=bundle.plan({'postgres':['security_dim'],'clickhouse':['security_dim']})
    assert all(a['action']=='blocked' and 'metadata denied' in a['error'] for a in plan['actions'] if a['target']=='postgres')
    assert all(a['action']=='create' for a in plan['actions'] if a['target']=='clickhouse')


def test_history_is_compact_filtered_and_hides_children(tmp_path):
    s=manager(tmp_path)
    one=s.state.create_job({'revision':'manager-v1','request':{'tables':['one','two'],'targets':['postgres'],'start_date':'2026-01-01','end_date':'2026-01-02'},'large_secret_contract':{'fields':['x']*1000}},s.settings.fingerprint)
    s.state.create_job({'revision':'manager-v1','pipeline_parent':one['id'],'request':{'tables':['one'],'targets':['postgres']}},s.settings.fingerprint)
    s.state.create_job({'revision':'reference-rollback-v1','request':{'targets':['postgres']}},s.settings.fingerprint)
    s.state.create_job({'revision':'manager-v1','request':{'tables':['three'],'targets':['clickhouse']}},'another-target')
    reference=s.state.create_job({'revision':'reference-v1','request':{'modules':['calendar'],'targets':['postgres']}},s.settings.fingerprint)
    with TestClient(create_app(s)) as client:
        r=client.get('/api/task-history?target=postgres&table=two').json()
        assert r['total']==1 and r['rows'][0]['id']==one['id']
        assert 'large_secret_contract' not in json.dumps(r) and 'plan' not in r['rows'][0]
        assert client.get('/api/task-history?limit=1').json()['total']==3
        assert client.get('/api/task-history?target=clickhouse').json()['rows'][0]['target_matches'] is False
        assert client.get('/api/task-history?target=postgres%27').status_code==409
        calendar=client.get('/api/task-history?table=trade_calendar_versions').json()
        assert calendar['total']==1 and calendar['rows'][0]['id']==reference['id']
        assert calendar['rows'][0]['tables']==['trade_calendar_versions']
        s.state.ledger_put('import_rollback','revoke',{'source_task_id':one['id'],'rollback_task_id':'revoke','target':'postgres','status':'blocked','conflicts':1})
        detail=client.get('/api/task-history/'+one['id']).json()
        assert detail['rollback_summary'][0]['conflicts']==1


def test_config_save_as_preserves_original_and_current_service(tmp_path):
    s=manager(tmp_path,env_file=tmp_path/'active.env',token='original-secret')
    s.settings.write_private_env({'TUSHARE_TOKEN':'original-secret','PGDATABASE':'betalens','BETALENS_DB_MANAGER_STATE':str(s.settings.state_dir),'BETALENS_DB_MANAGER_CONFIG':str(s.settings.config_dir)})
    before=s.settings.env_file.read_bytes();destination=tmp_path/'new.env'
    with TestClient(create_app(s)) as client:
        result=client.post('/api/config',json={'env_file':str(destination),'PGDATABASE':'new_db','TUSHARE_TOKEN':''})
        assert result.status_code==200,result.text
        data=result.json();assert data['restart_required'] and 'original-secret' not in result.text
        assert s.settings.pg_database=='betalens' and s.settings.env_file.read_bytes()==before
        assert 'original-secret' in destination.read_text() and destination.stat().st_mode&0o777==0o600
        assert data['active_path']==str(s.settings.env_file) and str(destination) in data['restart_command']
        assert client.post('/api/config',json={'env_file':str(destination)}).status_code==409


def test_schedule_retirement_preserves_running_and_manual_jobs(tmp_path):
    s=manager(tmp_path);queued=s.state.create_job({'request':{'targets':['postgres']}},s.settings.fingerprint)
    running=s.state.create_job({'request':{'targets':['postgres']}},s.settings.fingerprint)
    manual=s.state.create_job({'request':{'targets':['postgres']}},s.settings.fingerprint)
    s.state.enqueue(queued['id'],'update',{'schedule':'legacy'});running_task=s.state.enqueue(running['id'],'update',{'schedule':'legacy'})
    s.state.enqueue(manual['id'],'download')
    with s.state.connect() as c:
        c.execute("INSERT INTO schedules VALUES('legacy','legacy','{}',1,0,1,'2026-01-01',NULL)")
        c.execute("UPDATE tasks SET status='running' WHERE id=?",(running_task,))
        c.execute("INSERT INTO tasks(id,job,operation,payload,status,created) VALUES(?,?,'preview',?,'queued',?)",('legacy-queued',running['id'],dumps({'schedule':'legacy'}),now()))
        c.execute("UPDATE jobs SET status='downloading' WHERE id=?",(running['id'],))
        c.execute("DELETE FROM ledger WHERE table_name='migration' AND row_key='retire_schedules_v1'")
    State(s.settings.state_dir)
    assert s.state.job(queued['id'])['tasks'][0]['status']=='cancelled'
    assert s.state.job(running['id'])['tasks'][0]['status']=='running'
    assert s.state.job(running['id'])['status']=='downloading'
    assert s.state.job(manual['id'])['tasks'][0]['status']=='queued'
    with s.state.connect() as c:assert c.execute('SELECT enabled,next_run FROM schedules').fetchone()[0]==0


def test_real_batch_create_missing_only_with_dependencies_and_schema_drift(live_manager,monkeypatch):
    s=live_manager;s.registry.save(table_definition('workspace_quotes'));bundle=SchemaBundle(s)
    plan=bundle.plan({'postgres':['workspace_quotes','trade_calendar_versions'],'clickhouse':['workspace_quotes']})
    assert not any(a['name']=='market_daily_raw' for a in plan['actions'])
    result=bundle.apply(plan['token']);assert result['status']=='completed',result
    assert bundle.apply(plan['token'])==result
    with s.pg.connect() as c,c.cursor() as cur:
        from psycopg2 import sql
        from betalens_db_manager.storage import relation
        cur.execute(sql.SQL('INSERT INTO {} (code,business_date,price) VALUES(%s,%s,%s)').format(relation(s.settings,'workspace_quotes')),('keep','2026-10-20','12.12345678'))
        cur.execute(sql.SQL('ALTER TABLE {} ADD COLUMN untouched text').format(relation(s.settings,'workspace_quotes')))
    second=bundle.plan({'postgres':['workspace_quotes']})
    assert next(a for a in second['actions'] if a['name']=='workspace_quotes')['status']=='drifted'
    done=bundle.apply(second['token']);assert done['status']=='partial'
    with s.pg.connect(readonly=True) as c,c.cursor() as cur:
        cur.execute(sql.SQL('SELECT code,price,untouched FROM {}').format(relation(s.settings,'workspace_quotes')))
        row=cur.fetchone();assert row[0]=='keep' and str(row[1])=='12.12345678'
    # Record the real metadata queries. Reading any user relation fails this check.
    from contextlib import contextmanager
    pg_connect=s.pg.connect;ch_execute=s.ch.execute;queries=[]
    class Cursor:
        def __init__(self,cur):self.cur=cur
        def __getattr__(self,name):return getattr(self.cur,name)
        def execute(self,query,*args):
            text=query if isinstance(query,str) else query.as_string(self.cur)
            queries.append(text);return self.cur.execute(query,*args)
        def __enter__(self):return self
        def __exit__(self,*args):self.cur.close()
    class Connection:
        def __init__(self,conn):self.conn=conn
        def cursor(self):return Cursor(self.conn.cursor())
    @contextmanager
    def metadata_connection(*args,**kwargs):
        assert kwargs.get('readonly') is True
        with pg_connect(*args,**kwargs) as conn:yield Connection(conn)
    def metadata_execute(query,*args,**kwargs):
        queries.append(query);return ch_execute(query,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(s.pg,'connect',metadata_connection);patch.setattr(s.ch,'execute',metadata_execute)
        all_status=bundle.inspect(['postgres','clickhouse'])
    assert queries
    import re
    for query in queries:
        assert not re.search(r'\bCOUNT\s*\(|\bFINAL\b',query,re.I),query
        relations=re.findall(r'\b(?:FROM|JOIN)\s+([\w.]+)',query,re.I)
        assert all(name=='pg_database' or name=='pg_indexes' or name.startswith(('information_schema.','system.')) for name in relations),query
    assert all_status['postgres']['connected'] and all_status['clickhouse']['connected']
    assert next(r for r in all_status['postgres']['tables'] if r['name']=='reference_scope')['status']=='ready'
    assert next(r for r in all_status['clickhouse']['tables'] if r['name']=='trade_calendar_versions')['status']=='missing'


def test_empty_databases_can_create_entire_manifest_with_one_target_denied(live_manager,monkeypatch):
    s=live_manager;bundle=SchemaBundle(s);names=list(bundle.entries())
    with monkeypatch.context() as patch:
        patch.setattr(SchemaBundle,'_inspect_pg',lambda _:(_ for _ in ()).throw(PermissionError('permission denied for metadata')))
        denied=bundle.plan({'postgres':['security_dim'],'clickhouse':['security_dim']})
        partial=bundle.apply(denied['token']);assert partial['status']=='partial'
        assert all('permission denied' in r['error'] for r in partial['results'] if r['target']=='postgres')
        assert all(r['status']=='created' for r in partial['results'] if r['target']=='clickhouse')
    plan=bundle.plan({'postgres':names,'clickhouse':names})
    original=(s.settings,s.pg,s.ch);switched=[];create=SchemaBundle._create_pg
    def save_another_connection(current,entry):
        create(current,entry)
        if not switched:
            from betalens_db_manager.storage import Postgres
            from betalens_db_manager.clickhouse import ClickHouse
            s.settings=replace(s.settings,pg_database=s.settings.pg_database+'_other',ch_database=s.settings.ch_database+'_other')
            s.pg=Postgres(s.settings);s.ch=ClickHouse(s.settings);switched.append(True)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(SchemaBundle,'_create_pg',save_another_connection)
            complete=bundle.apply(plan['token'])
    finally:s.settings,s.pg,s.ch=original
    assert switched and complete['status']=='completed',[r for r in complete['results'] if r['status']=='failed']
    assert len(complete['results'])==2*len(names)
    with s.pg.connect(readonly=True,database='postgres') as c,c.cursor() as cur:
        cur.execute('SELECT 1 FROM pg_database WHERE datname=%s',(s.settings.pg_database+'_other',));assert cur.fetchone() is None
    assert s.ch.execute('EXISTS DATABASE `'+s.settings.ch_database+'_other`')=='0'
    inspected=bundle.inspect()
    assert all(target['connected'] and all(t['status']=='ready' for t in target['tables']) for target in inspected.values()),inspected
    repeat=bundle.apply(bundle.plan({'postgres':names,'clickhouse':names})['token'])
    assert repeat['status']=='completed' and all(r['status']=='existing' for r in repeat['results'])
