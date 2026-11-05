"""One durable worker per state directory for explicitly started tasks."""
from __future__ import annotations

import json
import signal
import os
import sqlite3
import time
import uuid
from datetime import datetime,timedelta
from contextlib import contextmanager

from .manager import DatabaseManagerService
from .util import dumps, now, digest
from .config import Settings


class WorkerStopping(Exception):pass


def sqlite_busy(exc):
    code=getattr(exc,'sqlite_errorcode',None)
    return isinstance(exc,sqlite3.OperationalError) and ((code is not None and code&255 in {sqlite3.SQLITE_BUSY,sqlite3.SQLITE_LOCKED}) or 'database is locked' in str(exc).lower() or 'database table is locked' in str(exc).lower())


def diagnostic(settings,event,**values):
    from .security import public_data
    print(dumps(public_data({'at':now(),'pid':os.getpid(),'event':event,**values},settings)),flush=True)


def retry_control(phase,call,settings,stopped=lambda:False):
    """Retry rolled-back SQLite control operations, never a business write."""
    attempt=0
    while not stopped():
        try:return call()
        except sqlite3.OperationalError as exc:
            if not sqlite_busy(exc):raise
            delay=min(5,.2*2**min(attempt,5));attempt+=1
            diagnostic(settings,'sqlite_busy_retry',phase=phase,attempt=attempt,retry_seconds=delay,error=str(exc))
            deadline=time.monotonic()+delay
            while not stopped() and time.monotonic()<deadline:time.sleep(min(.1,max(0,deadline-time.monotonic())))
    raise WorkerStopping()


def heartbeat(service,status,job=None,jobs=None):
    from .util import atomic_json
    try:atomic_json(service.settings.state_dir/'worker-heartbeat.json',{'pid':os.getpid(),'at':now(),'status':status,'job':job,'jobs':jobs or ([] if job is None else [job])})
    except OSError as exc:diagnostic(service.settings,'worker_heartbeat_failed',error=str(exc))


@contextmanager
def worker_lock(service):
    path = service.settings.state_dir / "worker.lock"
    handle = path.open("a+")
    try:
        if __import__("os").name == "nt":
            import msvcrt
            handle.seek(0); handle.write("0"); handle.flush(); handle.seek(0)
            acquire=lambda:msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            acquire=lambda:fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The HTTP port may close before the old worker finishes its slice.
        deadline=time.monotonic()+15
        while True:
            try:
                acquire();break
            except OSError as exc:
                if exc.errno not in {11,13}:raise
                if time.monotonic()>=deadline:raise ValueError('当前状态目录已有运行中的 worker') from None
                time.sleep(.2)
        yield
    finally:
        handle.close()


def execute_task(settings,task):
    from .manager import DatabaseManagerService
    from dataclasses import replace
    current=Settings.from_env(settings.env_file) if settings.env_file else settings
    if current.state_dir!=settings.state_dir or current.config_dir!=settings.config_dir:raise ValueError('环境目录已改变，请重启管理器')
    service=retry_control('initialize_task',lambda:DatabaseManagerService(current),current)
    def control(phase,call):return retry_control(phase,call,current)
    attempt=uuid.uuid4().hex
    control('attempt_started',lambda:service.state.ledger_put('execution_attempt',attempt,{'task_id':task['job'],'attempt_id':attempt,'operation':task['operation'],'started_at':now(),'status':'running','config_revision':digest([current.public(),current.env_file.stat().st_mtime_ns if current.env_file and current.env_file.exists() else None])}))
    control('execution_started',lambda:service.state.log(task['job'],'execution_started',attempt_id=attempt,operation=task['operation']))
    try:
        service.state.ensure_execution(task['job'])
        service.state.check_cancel(task['job'])
        operation,job=task['operation'],task['job']
        if operation=='pipeline_run':
            from .pipeline import Pipeline
            Pipeline(service).run(job)
        elif operation=='pipeline_review':
            from .review_service import ReviewService
            ReviewService(service).execute(task['payload']['decision_id'])
        elif operation=='pipeline_commit':
            from .pipeline import Pipeline
            Pipeline(service).commit(job)
        elif operation=='pipeline_validate':
            from .pipeline_validation import ValidationRunner
            ValidationRunner(service).run(job,task['payload'].get('validation',{}))
        elif operation=='pipeline_rollback':
            from .pipeline import Pipeline
            Pipeline(service).rollback(job,task['payload'].get('targets'))
        elif operation=='pipeline_sync':
            from .pipeline_store import PipelineStore
            for target in service.state.job_header(job)['request']['targets']:
                PipelineStore(service.state).sync(service,job,target)
        else:raise ValueError('未知操作')
        control('operation_finished',lambda:service.state.finish(task['id']))
        status='completed';error=None
    except BaseException as exc:
        error=service.safe_error(exc);status='failed'
        control('operation_failed',lambda:service.state.finish(task['id'],error))
        control('operation_error',lambda:service.state.update_job(task['job'],error=error))
        control('execution_failed',lambda:service.state.log(task['job'],'execution_failed',error=error))
    control('attempt_finished',lambda:service.state.ledger_put('execution_attempt',attempt,{**(service.state.ledger_get('execution_attempt',attempt) or {}),'finished_at':now(),'status':status,'error':error}))


def clean_cache(service):
    """All task archives are retained as durable audit evidence.

    Historical v1 evidence is read-only; v2 cache retention is explicit and
    separate from worker scheduling. Never expire either automatically.
    """
    return None


def interrupt_attempts(service,job,reason):
    """Record hard exits durably, including the reference publisher's attempt."""
    for kind in ('execution_attempt','reference_attempt'):
        for attempt in service.state.ledger_rows(kind,job):
            if attempt.get('status')=='running':
                service.state.ledger_put(kind,attempt['attempt_id'],{**attempt,'status':'interrupted','finished_at':now(),'error':reason})


def work(service: DatabaseManagerService, once: bool = False):
    """Own at most one ingestion and one independent validation process."""
    import multiprocessing
    from .management_api import session_online
    stop=False;owned=False;children={};claimed_once=False
    def signal_stop(*_):
        nonlocal stop
        stop=True
    previous={sig:signal.signal(sig,signal_stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    service.state.busy_timeout=1
    def control(phase,call):return retry_control(phase,call,service.settings,lambda:stop)
    def recover():
        with service.state.connect() as conn:
            abandoned=[r[0] for r in conn.execute("SELECT DISTINCT job FROM tasks WHERE status='running'")]
            conn.execute("UPDATE tasks SET status='interrupted',error='执行器已退出，需显式恢复并核验目标库' WHERE status='running'")
            for job in abandoned:conn.execute("UPDATE jobs SET status='interrupted',error='执行器已退出，需显式恢复并核验目标库',updated=? WHERE id=? AND status NOT IN ('completed','rolled_back')",(now(),job))
        return abandoned
    def cancelled(job):
        with service.state.connect() as conn:return bool(conn.execute('SELECT cancel FROM jobs WHERE id=?',(job,)).fetchone()[0])
    def finish_interrupted(task):
        with service.state.connect() as conn:
            running=conn.execute("SELECT status FROM tasks WHERE id=?",(task['id'],)).fetchone()
            if running and running[0]=='running':
                conn.execute("UPDATE tasks SET status='interrupted',error='进程中断；写入状态须在恢复前核验' WHERE id=?",(task['id'],))
                conn.execute("UPDATE jobs SET status='interrupted',error='进程中断；已保存检查点，恢复前核验目标库',updated=? WHERE id=?",(now(),task['job']))
                return True
        return False
    try:
        with worker_lock(service):
            owned=True
            for job in control('recover',recover):control('recover_attempts',lambda:interrupt_attempts(service,job,'执行器已退出，目标写入待核验'))
            was_online=False;beat=0
            config_stamp=service.settings.env_file.stat().st_mtime_ns if service.settings.env_file and service.settings.env_file.exists() else None
            while not stop or children:
                online=True
                try:online=session_online(service.state)
                except sqlite3.OperationalError as exc:
                    if not sqlite_busy(exc):raise
                    diagnostic(service.settings,'sqlite_busy_retry',phase='sessions',error=str(exc))
                for identity,entry in list(children.items()):
                    child,task=entry['process'],entry['task']
                    if not child.is_alive():
                        child.join()
                        if not stop and control('finish',lambda:finish_interrupted(task)):
                            control('finish_attempts',lambda:interrupt_attempts(service,task['job'],'进程中断，目标写入待核验'))
                        del children[identity]
                        continue
                    requested=False
                    try:requested=cancelled(task['job'])
                    except sqlite3.OperationalError as exc:
                        if not sqlite_busy(exc):raise
                        diagnostic(service.settings,'sqlite_busy_retry',phase='monitor',job=task['job'],error=str(exc))
                    requires_page=not once and not task['payload'].get('background',False)
                    if stop or requested or (requires_page and not online):
                        if entry['cancel_at'] is None:entry['cancel_at']=time.monotonic()
                        try:
                            if not requested:service.state.cancel(task['job'])
                            if not entry['stop_logged']:
                                service.state.log(task['job'],'stop_requested',reason='管理页面已关闭' if not online else '用户停止')
                                entry['stop_logged']=True
                        except sqlite3.OperationalError as exc:
                            if not sqlite_busy(exc):raise
                        elapsed=time.monotonic()-entry['cancel_at']
                        if elapsed>=65:child.kill()
                        elif elapsed>=60:child.terminate()
                if time.monotonic()-beat>=5:
                    jobs=[v['task']['job'] for v in children.values()]
                    heartbeat(service,'running' if jobs else 'idle',jobs[0] if jobs else None,jobs)
                    beat=time.monotonic()
                if stop:
                    if children:time.sleep(.25)
                    continue
                if once and claimed_once:
                    if not children:break
                    time.sleep(.25);continue
                try:
                    # Reload only the scheduler; running children retain the
                    # configuration they resolved at their execution boundary.
                    if hasattr(service,'reload_config') and service.settings.env_file and service.settings.env_file.exists():
                        current_stamp=service.settings.env_file.stat().st_mtime_ns
                        if current_stamp!=config_stamp:
                            try:control('reload_config',service.reload_config)
                            except ValueError as exc:diagnostic(service.settings,'worker_config_restart_required',error=str(exc))
                            config_stamp=current_stamp
                    if was_online and not online:
                        def offline_jobs():
                            with service.state.connect() as conn:return [r[0] for r in conn.execute("SELECT DISTINCT t.job FROM tasks t JOIN jobs j ON j.id=t.job WHERE t.status='queued' AND json_extract(j.plan,'$.pipeline_version')=2 AND coalesce(json_extract(t.payload,'$.background'),0)!=1")]
                        for job in control('offline_queue',offline_jobs):
                            control('cancel_offline',lambda:service.state.cancel(job))
                            control('cancel_status',lambda:service.state.update_job(job,status='cancelled',error='最后一个管理页面已关闭；请显式恢复'))
                    was_online=online
                    active_page=control('sessions',lambda:session_online(service.state,include_grace=False))
                    task=control('claim',lambda:service.state.claim(allow_ui=bool(active_page or once))) if len(children)<2 else None
                    if task:
                        claimed_once=True
                        child=multiprocessing.get_context('spawn').Process(target=execute_task,args=(service.settings,task))
                        try:child.start()
                        except Exception as exc:
                            message=service.safe_error(exc)
                            control('spawn_failed',lambda:service.state.finish(task['id'],message))
                            control('spawn_status',lambda:service.state.update_job(task['job'],status='interrupted',error='执行子进程启动失败：'+message))
                            diagnostic(service.settings,'worker_spawn_failed',job=task['job'],error=message)
                            if once:break
                            continue
                        children[task['id']]={'process':child,'task':task,'cancel_at':None,'stop_logged':False}
                        beat=0
                        continue  # Immediately admit an independent second job.
                    if once and not children:break
                    time.sleep(.25 if children else .5)
                except WorkerStopping:continue
    except WorkerStopping:pass
    finally:
        for entry in children.values():
            child=entry['process']
            if child.is_alive():child.terminate();child.join(5)
            if child.is_alive():child.kill();child.join(5)
        if owned:heartbeat(service,'stopped')
        for sig,handler in previous.items():signal.signal(sig,handler)
