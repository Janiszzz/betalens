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

from .service import TushareImportService
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


def heartbeat(service,status,job=None):
    from .util import atomic_json
    try:atomic_json(service.settings.state_dir/'worker-heartbeat.json',{'pid':os.getpid(),'at':now(),'status':status,'job':job})
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
        elif operation.startswith('reference_'):
            from .reference_service import ReferenceWorkflow
            reference=ReferenceWorkflow(service)
            if operation=='reference_download':reference.download(job);reference.preview(job)
            elif operation=='reference_preview':reference.preview(job)
            elif operation=='reference_commit':reference.commit(job,task['payload']['token'],task['payload'].get('accept_partial',False))
            elif operation=='reference_rollback':reference.rollback(job,task['payload'].get('targets'))
            else:raise ValueError('未知参考表任务操作')
        elif operation in {'download','update','file_download'}:
            service.download(job);service.preview(job)
        elif operation=='commit':service.commit(job,task['payload']['token'],task['payload'].get('accept_partial',False))
        elif operation=='resume':service.resume(job)
        elif operation=='preview':service.preview(job)
        elif operation=='rollback':service.rollback(job,task['payload'].get('targets'))
        elif operation=='sync_audit':
            from .publishing import operation_lock,retry_audit_sync
            if service.state.job(job)['plan'].get('pipeline_version')==2:
                from .pipeline_store import PipelineStore
                for target in service.state.job(job)['plan']['request']['targets']:PipelineStore(service.state).sync(service,job,target)
            else:
                with operation_lock(service):retry_audit_sync(service,job)
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
    from pathlib import Path
    from datetime import timezone
    cutoff=datetime.now(timezone.utc)-timedelta(days=7)
    with service.state.connect() as conn:
        protected={r[0] for r in conn.execute("SELECT s.archive FROM slices s JOIN jobs j ON j.id=s.job WHERE s.archive IS NOT NULL AND (j.status NOT IN ('completed','rolled_back') OR j.updated>?)",(cutoff.isoformat(),))}
    for summary in service.state.jobs(limit=1000):
        if summary['status'] not in {'completed','rolled_back'} or datetime.fromisoformat(summary['updated'])>cutoff:continue
        item=service.state.job(summary['id'])
        # Reference archives and normalized snapshots are durable audit/rollback
        # evidence. Their retention is separate from the legacy transient cache.
        if item['plan'].get('revision','').startswith('reference-') or item['plan'].get('pipeline_version')==2:continue
        if item['plan'].get('cache_purged'):continue
        if any(p.get('archive') in protected for p in item['slices']):continue
        for part in item['slices']:
            if part.get('archive'):
                path=Path(part['archive']);path.unlink(missing_ok=True);path.with_suffix('.meta.json').unlink(missing_ok=True);path.with_name(path.name.removesuffix('.json.gz')+'.response.json.gz').unlink(missing_ok=True)
        if item['plan'].get('file') and item['slices']:
            sample=next((p for p in item['slices'] if p.get('archive')),None)
            if sample:
                stored=Path(sample['archive']).parent/('source'+''.join(Path(item['plan']['file']['name']).suffixes))
                if item['id'] in stored.parts:stored.unlink(missing_ok=True)
        item['plan']['cache_purged']=True
        with service.state.connect() as conn:conn.execute('UPDATE jobs SET plan=? WHERE id=?',(dumps(item['plan']),item['id']))


def interrupt_attempts(service,job,reason):
    """Record hard exits durably, including the reference publisher's attempt."""
    for kind in ('execution_attempt','reference_attempt'):
        for attempt in service.state.ledger_rows(kind,job):
            if attempt.get('status')=='running':
                service.state.ledger_put(kind,attempt['attempt_id'],{**attempt,'status':'interrupted','finished_at':now(),'error':reason})


def work(service: TushareImportService, once: bool = False):
    import multiprocessing
    from .management_api import session_online
    stop=False;owned=False;child=None
    def signal_stop(*_):
        nonlocal stop
        stop=True
    previous={sig:signal.signal(sig,signal_stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    # Short waits keep cancellation responsive; contention is retried outside transactions.
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
    try:
        with worker_lock(service):
            owned=True
            for job in control('recover',recover):control('recover_attempts',lambda:interrupt_attempts(service,job,'执行器已退出，目标写入待核验'))
            was_online=False;cleaned=0;beat=0
            config_stamp=service.settings.env_file.stat().st_mtime_ns if service.settings.env_file and service.settings.env_file.exists() else None
            while not stop:
                if time.monotonic()-beat>=5:heartbeat(service,'idle');beat=time.monotonic()
                try:
                    if hasattr(service,'reload_config') and service.settings.env_file and service.settings.env_file.exists():
                        current_stamp=service.settings.env_file.stat().st_mtime_ns
                        if current_stamp!=config_stamp:
                            try:control('reload_config',service.reload_config)
                            except ValueError as exc:diagnostic(service.settings,'worker_config_restart_required',error=str(exc))
                            config_stamp=current_stamp
                    online=control('sessions',lambda:session_online(service.state))
                    if was_online and not online:
                        def offline_jobs():
                            with service.state.connect() as conn:return [r[0] for r in conn.execute("SELECT DISTINCT job FROM tasks WHERE status='queued' AND coalesce(json_extract(payload,'$.background'),0)!=1")]
                        for job in control('offline_queue',offline_jobs):
                            control('cancel_offline',lambda:service.state.cancel(job))
                            control('cancel_status',lambda:service.state.update_job(job,status='cancelled',error='最后一个管理页面已关闭；请显式恢复'))
                    active_page=control('sessions',lambda:session_online(service.state,include_grace=False));was_online=online
                    task=control('claim',lambda:service.state.claim(allow_ui=bool(active_page or once)))
                    if task:
                        heartbeat(service,'running',task['job'])
                        child=multiprocessing.get_context('spawn').Process(target=execute_task,args=(service.settings,task))
                        try:child.start()
                        except Exception as exc:
                            message=service.safe_error(exc)
                            control('spawn_failed',lambda:service.state.finish(task['id'],message))
                            control('spawn_status',lambda:service.state.update_job(task['job'],status='interrupted',error='执行子进程启动失败：'+message))
                            diagnostic(service.settings,'worker_spawn_failed',job=task['job'],error=message)
                            if once:break
                            continue
                        cancel_at=None;stop_logged=False
                        while child.is_alive():
                            online=True;requested=False
                            try:
                                online=session_online(service.state);requested=cancelled(task['job'])
                            except sqlite3.OperationalError as exc:
                                if not sqlite_busy(exc):raise
                                diagnostic(service.settings,'sqlite_busy_retry',phase='monitor',job=task['job'],error=str(exc))
                            requires_page=not once and not task['payload'].get('background',False)
                            if stop or requested or (requires_page and not online):
                                if cancel_at is None:cancel_at=time.monotonic()
                                try:
                                    if not requested:service.state.cancel(task['job'])
                                    if not stop_logged:
                                        service.state.log(task['job'],'stop_requested',reason='管理页面已关闭' if not online else '用户停止')
                                        stop_logged=True
                                except sqlite3.OperationalError as exc:
                                    if not sqlite_busy(exc):raise
                                if time.monotonic()-cancel_at>=60:child.terminate();child.join(5)
                                if child.is_alive() and time.monotonic()-cancel_at>=65:child.kill()
                            if time.monotonic()-beat>=5:heartbeat(service,'running',task['job']);beat=time.monotonic()
                            child.join(.25)
                        child.join()
                        def finish_interrupted():
                            with service.state.connect() as conn:
                                running=conn.execute("SELECT status FROM tasks WHERE id=?",(task['id'],)).fetchone()
                                if running and running[0]=='running':
                                    conn.execute("UPDATE tasks SET status='interrupted',error='进程中断；写入状态须在恢复前核验' WHERE id=?",(task['id'],))
                                    conn.execute("UPDATE jobs SET status='interrupted',error='进程中断；已保存检查点，恢复前核验目标库',updated=? WHERE id=?",(now(),task['job']))
                                    return True
                            return False
                        # A shutdown can leave recovery to the next worker, rather than blocking its exit.
                        if not stop and control('finish',finish_interrupted):control('finish_attempts',lambda:interrupt_attempts(service,task['job'],'进程中断，目标写入待核验'))
                        if once:break
                    elif once:break
                    else:
                        if time.monotonic()-cleaned>3600:
                            try:clean_cache(service)
                            except Exception as exc:diagnostic(service.settings,'worker_cache_cleanup_failed',error=service.safe_error(exc))
                            cleaned=time.monotonic()
                        time.sleep(.5)
                except WorkerStopping:break
    except WorkerStopping:pass
    finally:
        if child is not None and child.is_alive():
            child.terminate();child.join(5)
            if child.is_alive():child.kill();child.join(5)
        if owned:heartbeat(service,'stopped')
        for sig,handler in previous.items():signal.signal(sig,handler)
