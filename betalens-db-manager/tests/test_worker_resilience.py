"""Real lock contention, worker ownership and bounded metadata transactions."""
import json,math,os,signal,socket,sqlite3,subprocess,sys,threading,time
from contextlib import contextmanager
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pytest,requests
from test_manager import manager
from betalens_db_manager.config import Settings
from betalens_db_manager.service import TushareImportService
from betalens_db_manager.worker import retry_control,WorkerStopping,work
from betalens_db_manager.worker_supervisor import worker_status
from betalens_db_manager.state import State


def test_mapping_registration_batches_all_tables_without_reading_plan_under_write_lock(tmp_path,monkeypatch):
    s=manager(tmp_path)
    plan=s.compile({'pipeline_version':2,'tables':list(s.registry.catalog()),'start_date':'2005-01-01','end_date':'2026-10-27'})
    item=s.state.create_job(plan,s.settings.fingerprint)
    original=s.state.connect;transactions=[]
    @contextmanager
    def traced():
        with original() as c:
            c.set_trace_callback(lambda sql:transactions.append(sql) if sql in ('BEGIN ','COMMIT') else None)
            yield c
    monkeypatch.setattr(s.state,'connect',traced)
    monkeypatch.setattr(s.state,'_job_targets',lambda *_:(_ for _ in ()).throw(AssertionError('冻结的目标范围不得逐字段重新读取整份计划')))
    started=time.monotonic();s._record_mapping_plan(item);elapsed=time.monotonic()-started
    rows=s.state.ledger_rows('meta_source_mapping',item['id']);assert len(rows)>1000
    assert len(transactions)==2*math.ceil(len(rows)/128)
    sync=s.state.sync_records(item['id']);assert {r['target'] for r in sync}=={'postgres','clickhouse'}
    assert all(r['revision']==math.ceil(len(rows)/128) and r['acknowledged']==0 for r in sync)
    assert s.state.job(item['id'])['tasks']==[]
    print(json.dumps({'initial_requests':len(plan['requests']),'mapping_records':len(rows),'metadata_seconds':round(elapsed,3),'write_transactions':len(transactions)//2}))


def test_single_metadata_write_resolves_targets_before_transaction_and_closes_connection(tmp_path,monkeypatch):
    s=manager(tmp_path);job=s.state.create_job({'request':{'targets':['postgres','clickhouse']}},s.settings.fingerprint)
    original=s.state._job_targets
    def targets(conn,task):
        assert not conn.in_transaction
        return original(conn,task)
    monkeypatch.setattr(s.state,'_job_targets',targets)
    s.state.ledger_put('meta_source_mapping','one',{'task_id':job['id'],'field':'price'})
    assert len(s.state.sync_records(job['id']))==2
    with s.state.connect() as conn:assert conn.execute('SELECT 1').fetchone()[0]==1
    with pytest.raises(sqlite3.ProgrammingError):conn.execute('SELECT 1')


def test_real_write_lock_is_retried_then_claimed_exactly_once(tmp_path,capsys):
    s=TushareImportService(Settings(state_dir=tmp_path));s.state.busy_timeout=.05
    job=s.state.create_job({'pipeline_version':2,'request':{'targets':['postgres'],'run_in_background':True}},s.settings.fingerprint)
    task=s.state.enqueue(job['id'],'pipeline_run');assert s.state.job(job['id'])['status']=='queued'
    locked=threading.Event()
    def writer():
        with sqlite3.connect(s.state.path) as c:
            c.execute('BEGIN IMMEDIATE');locked.set();time.sleep(.3)
    holder=threading.Thread(target=writer);holder.start();assert locked.wait(2)
    claimed=retry_control('claim',lambda:s.state.claim(allow_ui=False),s.settings);holder.join()
    assert claimed['id']==task and s.state.claim() is None
    assert s.state.job(job['id'])['tasks'][0]['status']=='running'
    assert 'sqlite_busy_retry' in capsys.readouterr().out


def test_busy_retry_obeys_stop_and_does_not_retry_unrelated_sql_errors(tmp_path):
    s=Settings(state_dir=tmp_path);stopped=threading.Event()
    timer=threading.Timer(.05,stopped.set);timer.start()
    with pytest.raises(WorkerStopping):retry_control('claim',lambda:(_ for _ in ()).throw(sqlite3.OperationalError('database is locked')),s,stopped.is_set)
    timer.join()
    with pytest.raises(sqlite3.OperationalError,match='no such table'):
        retry_control('claim',lambda:(_ for _ in ()).throw(sqlite3.OperationalError('no such table')),s)


def test_cancelled_queue_stays_stopped_until_an_explicit_new_start(tmp_path):
    s=TushareImportService(Settings(state_dir=tmp_path))
    job=s.state.create_job({'pipeline_version':2,'request':{'targets':['postgres'],'run_in_background':True}},s.settings.fingerprint)
    s.state.enqueue(job['id'],'pipeline_run');s.state.cancel(job['id'])
    assert s.state.claim() is None and s.state.job(job['id'])['status']=='cancelled'
    assert s.state.job(job['id'])['cancel']==1
    s.state.enqueue(job['id'],'pipeline_run')
    assert s.state.job(job['id'])['cancel']==0 and s.state.job(job['id'])['status']=='queued'
    assert s.state.claim()['job']==job['id']


def test_worker_loop_survives_busy_claim_and_restores_signal_handlers(tmp_path,monkeypatch,capsys):
    s=TushareImportService(Settings(state_dir=tmp_path));claim=s.state.claim;calls=[]
    def busy_once(*args,**kwargs):
        calls.append(True)
        if len(calls)==1:raise sqlite3.OperationalError('database is locked')
        return claim(*args,**kwargs)
    monkeypatch.setattr(s.state,'claim',busy_once)
    previous=signal.getsignal(signal.SIGTERM);work(s,once=True)
    assert len(calls)==2 and signal.getsignal(signal.SIGTERM)==previous
    assert 'sqlite_busy_retry' in capsys.readouterr().out
    assert json.loads((tmp_path/'worker-heartbeat.json').read_text())['status']=='stopped'


def test_worker_health_requires_live_recent_heartbeat(tmp_path):
    s=Settings(state_dir=tmp_path);beat=tmp_path/'worker-heartbeat.json'
    beat.write_text(json.dumps({'pid':os.getpid(),'at':datetime.now(timezone.utc).isoformat(),'status':'idle'}))
    assert worker_status(s)['available']
    beat.write_text(json.dumps({'pid':os.getpid(),'at':(datetime.now(timezone.utc)-timedelta(seconds=60)).isoformat(),'status':'running'}))
    assert not worker_status(s)['available']


@pytest.mark.skipif(os.name!='posix' or not Path('/proc').exists(),reason='Linux subprocess ownership test')
def test_serving_restarts_killed_worker_and_stops_replacement_on_shutdown(tmp_path):
    s=Settings(config_dir=tmp_path/'config',state_dir=tmp_path/'state',env_file=tmp_path/'credentials.env')
    s.write_private_env({'BETALENS_DB_MANAGER_CONFIG':str(s.config_dir),'BETALENS_DB_MANAGER_STATE':str(s.state_dir)})
    state=State(s.state_dir)
    interrupted=state.create_job({'pipeline_version':2,'request':{'targets':['postgres'],'run_in_background':True}},s.fingerprint)
    state.enqueue(interrupted['id'],'pipeline_run');state.claim()
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=subprocess.Popen([sys.executable,'-m','betalens_db_manager','--env',str(s.env_file),'serve','--port',str(port)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    pids=[]
    def wait_health(previous=None):
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            try:
                result=requests.get(f'http://127.0.0.1:{port}/api/worker-status',timeout=.3).json()
                pid=result.get('heartbeat',{}).get('pid')
                if result.get('available') and pid!=previous:return result
            except (requests.RequestException,ValueError):pass
            time.sleep(.1)
        raise AssertionError('worker did not become healthy')
    try:
        first=wait_health();pids.append(first['heartbeat']['pid']);os.kill(pids[0],signal.SIGKILL)
        assert state.job(interrupted['id'])['status']=='interrupted'
        assert state.job(interrupted['id'])['tasks'][0]['status']=='interrupted'
        replacement=wait_health(pids[0]);pids.append(replacement['heartbeat']['pid'])
        assert replacement['supervisor']['restarts']>=1
        assert not Path(f'/proc/{pids[0]}').exists()
        server.terminate();assert server.wait(timeout=12)==0
        assert not Path(f'/proc/{pids[1]}').exists()
    finally:
        if server.poll() is None:server.terminate();server.wait(timeout=15)
        for pid in pids:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass


@pytest.mark.skipif(os.name!='posix' or not Path('/proc').exists(),reason='Linux process group ownership test')
def test_crashed_workers_execution_child_is_stopped_before_replacement(tmp_path):
    from betalens_db_manager.worker_supervisor import WorkerSupervisor
    script="import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(child.pid,flush=True); time.sleep(60)"
    worker=subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,text=True,start_new_session=True)
    execution=int(worker.stdout.readline())
    try:
        worker.kill();worker.wait(timeout=5)
        WorkerSupervisor(Settings(state_dir=tmp_path))._stop_tree(worker)
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            stat=Path(f'/proc/{execution}/stat')
            if not stat.exists() or stat.read_text().split()[2]=='Z':break
            time.sleep(.05)
        else:raise AssertionError('execution child survived worker replacement')
    finally:
        try:os.killpg(worker.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        worker.wait(timeout=5);worker.stdout.close()
