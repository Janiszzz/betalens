"""Opt-in five-day archived replay. Always writes isolated fixture databases."""
import json,os,sqlite3,time
from copy import deepcopy
from pathlib import Path
import pytest
from test_manager import live_manager
from betalens_db_manager.canonical import registry_context
from betalens_db_manager.config import Settings
from betalens_db_manager.pipeline import Pipeline
from betalens_db_manager.pipeline_options import options
from betalens_db_manager.pipeline_store import PipelineStore
from betalens_db_manager.util import dumps


@pytest.mark.skipif(os.environ.get('BETALENS_PIPELINE_BENCHMARK')!='1',reason='explicit offline archived benchmark only')
def test_five_day_archived_replay(live_manager,monkeypatch):
    s=live_manager;base=Settings.from_env('/home/janiszzz/betalens-infra/tushare.private.env')
    conn=sqlite3.connect('file:'+str(base.state_dir/'tasks.sqlite3')+'?mode=ro',uri=True);conn.row_factory=sqlite3.Row
    source='6a4103d5847c4d57ba2122054b106408'
    old=json.loads(conn.execute('SELECT plan FROM jobs WHERE id=?',(source,)).fetchone()['plan'])
    slices=[dict(r) for r in conn.execute('SELECT * FROM slices WHERE job=?',(source,))];conn.close()
    from betalens_db_manager.source import TushareSource
    monkeypatch.setattr(TushareSource,'fetch',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('offline benchmark forbids network')))
    with registry_context(s.registry):s.initialize(s.schema_plan()['token'])
    results=[]
    for attempt in range(3):
        plan=deepcopy(old);plan['pipeline_version']=2;plan['request'].update(options({}),min_free_bytes=0)
        job=s.state.create_job(plan,s.settings.fingerprint)
        for part in slices:
            s.state.slice(job['id'],part['id'],part['dataset'],json.loads(part['request']),status=part['status'],archive=part['archive'],sha=part['sha'],rows=part['rows'],result=json.loads(part['result']) if part['result'] else None)
        started=time.monotonic();result=Pipeline(s).run(job['id']);elapsed=time.monotonic()-started
        assert result['status']=='completed',result.get('error')
        receipts=PipelineStore(s.state).receipts(job['id'])
        results.append({'attempt':attempt+1,'seconds':round(elapsed,3),'tables':[{k:r['result'][k] for k in ('table','counts')}|{'target':r['target']} for r in receipts]})
        with registry_context(s.registry):
            for name in plan['request']['tables']:
                from psycopg2 import sql
                from betalens_db_manager.storage import relation
                assert s.settings.pg_database.startswith('manager_test_') and s.settings.ch_database.startswith('manager_test_')
                with s.pg.connect() as c,c.cursor() as cur:cur.execute(sql.SQL('TRUNCATE {}').format(relation(s.settings,name)))
                s.ch.execute('TRUNCATE TABLE '+s.ch.table(name))
    report={'old_local_seconds':167.69,'runs':results,'network':'disabled','isolated':True}
    output=Path('/tmp/betalens-pipeline-benchmark.json');output.write_text(dumps(report))
    print('\n'+dumps(report))
