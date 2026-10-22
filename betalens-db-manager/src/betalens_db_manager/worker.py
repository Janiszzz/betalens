"""One durable worker per state directory, with opt-in Shanghai-time schedules."""
from __future__ import annotations

import json
import signal
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta

from .service import SHANGHAI, TushareImportService
from .util import dumps, now, digest
from .config import Settings


def next_run(hour: int, minute: int) -> str:
    current = datetime.now(SHANGHAI)
    result = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if result <= current: result += timedelta(days=1)
    return result.isoformat()


class Scheduler:
    def __init__(self, service: TushareImportService):
        self.service = service

    def list(self):
        with self.service.state.connect() as conn:
            rows = [dict(row) for row in conn.execute("SELECT * FROM schedules ORDER BY label")]
        for row in rows:
            row["request"] = json.loads(row["request"]); row["enabled"] = bool(row["enabled"])
        return rows

    def save(self, label: str, request: dict, hour: int = 21, minute: int = 30, enabled: bool = False, id: str | None = None):
        if not 0 <= hour < 24 or not 0 <= minute < 60: raise ValueError("无效的更新时间")
        # Validate without creating a job or silently expanding data scope.
        for name in request.get("datasets", []):
            from .catalog import dataset
            dataset(name)
        if not (request.get("datasets") or request.get("tables")) or not label.strip(): raise ValueError("需要名称与数据集")
        request=(self.service.compile(request) if request.get('tables') else self.service._make_plan(request))['request']
        id = id or uuid.uuid4().hex
        with self.service.state.connect() as conn:
            conn.execute("INSERT INTO schedules VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET label=excluded.label,request=excluded.request,hour=excluded.hour,minute=excluded.minute,enabled=excluded.enabled,next_run=excluded.next_run",
                         (id, label, dumps(request), hour, minute, int(enabled), next_run(hour, minute) if enabled else None, None))
        if not enabled:
            with self.service.state.connect() as conn:
                conn.execute("UPDATE tasks SET status='cancelled' WHERE status='queued' AND json_extract(payload,'$.schedule')=?", (id,))
        return next(s for s in self.list() if s["id"] == id)

    def reset_missed(self):
        # Keep the saved opt-in state; never immediately catch up on a restart.
        with self.service.state.connect() as conn:
            for row in conn.execute("SELECT id,hour,minute,next_run FROM schedules WHERE enabled=1").fetchall():
                if not row["next_run"] or datetime.fromisoformat(row["next_run"]) <= datetime.now(SHANGHAI):
                    conn.execute("UPDATE schedules SET next_run=? WHERE id=?", (next_run(row["hour"], row["minute"]), row["id"]))

    def tick(self):
        for row in self.list():
            if not row["enabled"] or not row["next_run"] or datetime.fromisoformat(row["next_run"]) > datetime.now(SHANGHAI): continue
            previous = self.service.state.job(row["last_job"]) if row["last_job"] else None
            if previous and previous["status"] not in {"completed", "failed", "cancelled", "blocked", "pending_sync"}: continue
            request = dict(row["request"])
            request["end_date"] = (datetime.now(SHANGHAI).date()-timedelta(days=1)).isoformat()
            # A saved lookback range is explicit; scheduling does not change its start date.
            job = self.service.plan(request)
            self.service.state.enqueue(job["id"], "update", {"schedule": row["id"]})
            with self.service.state.connect() as conn:
                conn.execute("UPDATE schedules SET next_run=?,last_job=? WHERE id=? AND enabled=1", (next_run(row["hour"], row["minute"]), job["id"], row["id"]))


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
    service=DatabaseManagerService(current)
    attempt=uuid.uuid4().hex
    service.state.ledger_put('execution_attempt',attempt,{'task_id':task['job'],'attempt_id':attempt,'operation':task['operation'],'started_at':now(),'status':'running','config_revision':digest([current.public(),current.env_file.stat().st_mtime_ns if current.env_file and current.env_file.exists() else None])})
    service.state.log(task['job'],'execution_started',attempt_id=attempt,operation=task['operation'])
    try:
        service.state.check_cancel(task['job'])
        operation,job=task['operation'],task['job']
        if operation in {'download','update','file_download'}:
            service.download(job);service.preview(job)
        elif operation=='commit':service.commit(job,task['payload']['token'],task['payload'].get('accept_partial',False))
        elif operation=='resume':service.resume(job)
        elif operation=='preview':service.preview(job)
        elif operation=='rollback':service.rollback(job,task['payload'].get('targets'))
        else:raise ValueError('未知操作')
        service.state.finish(task['id'])
        status='completed';error=None
    except BaseException as exc:
        error=service.safe_error(exc);status='failed';service.state.finish(task['id'],error)
        service.state.update_job(task['job'],error=error)
        service.state.log(task['job'],'execution_failed',error=error)
    service.state.ledger_put('execution_attempt',attempt,{**(service.state.ledger_get('execution_attempt',attempt) or {}),'finished_at':now(),'status':status,'error':error})


def clean_cache(service):
    from pathlib import Path
    from datetime import timezone
    cutoff=datetime.now(timezone.utc)-timedelta(days=7)
    with service.state.connect() as conn:
        protected={r[0] for r in conn.execute("SELECT s.archive FROM slices s JOIN jobs j ON j.id=s.job WHERE s.archive IS NOT NULL AND (j.status NOT IN ('completed','rolled_back') OR j.updated>?)",(cutoff.isoformat(),))}
    for summary in service.state.jobs(limit=1000):
        if summary['status'] not in {'completed','rolled_back'} or datetime.fromisoformat(summary['updated'])>cutoff:continue
        item=service.state.job(summary['id'])
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


def work(service: TushareImportService, once: bool = False):
    import multiprocessing
    from .config import Settings
    from .management_api import session_online
    stop=False
    def signal_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    with worker_lock(service):
        with service.state.connect() as conn:
            abandoned=[r[0] for r in conn.execute("SELECT DISTINCT job FROM tasks WHERE status='running'")]
            conn.execute("UPDATE tasks SET status='interrupted',error='执行器已退出，需显式恢复并核验目标库' WHERE status='running'")
            for job in abandoned:conn.execute("UPDATE jobs SET status='interrupted',error='执行器已退出，需显式恢复并核验目标库',updated=? WHERE id=? AND status NOT IN ('completed','rolled_back')",(now(),job))
        scheduler=Scheduler(service);scheduler.reset_missed();was_online=False;cleaned=0
        config_stamp=service.settings.env_file.stat().st_mtime_ns if service.settings.env_file and service.settings.env_file.exists() else None
        while not stop:
            if hasattr(service,'reload_config') and service.settings.env_file and service.settings.env_file.exists():
                current_stamp=service.settings.env_file.stat().st_mtime_ns
                if current_stamp!=config_stamp:
                    try:service.reload_config()
                    except ValueError:pass  # Directory changes require an explicit service restart.
                    config_stamp=current_stamp
            online=session_online(service.state)
            if online and not was_online:scheduler.reset_missed()
            if was_online and not online:
                with service.state.connect() as conn:
                    queued=[r[0] for r in conn.execute("SELECT DISTINCT job FROM tasks WHERE status='queued'")]
                for job in queued:
                    service.state.cancel(job);service.state.update_job(job,status='cancelled',error='最后一个管理页面已关闭；请显式恢复')
            active_page=session_online(service.state,include_grace=False)
            if active_page:scheduler.tick()
            was_online=online
            task=service.state.claim() if active_page or once else None
            if task:
                child=multiprocessing.get_context('spawn').Process(target=execute_task,args=(service.settings,task))
                child.start();cancel_at=None
                while child.is_alive():
                    online=session_online(service.state)
                    requires_page=not once
                    item=service.state.job(task['job'])
                    if stop or item['cancel'] or (requires_page and not online):
                        service.state.cancel(task['job'])
                        if cancel_at is None:
                            cancel_at=time.monotonic();service.state.log(task['job'],'stop_requested',reason='管理页面已关闭' if not online else '用户停止')
                        if time.monotonic()-cancel_at>=60:child.terminate();child.join(5)
                        if child.is_alive() and time.monotonic()-cancel_at>=65:child.kill()
                    child.join(.25)
                child.join()
                with service.state.connect() as conn:
                    running=conn.execute("SELECT status FROM tasks WHERE id=?",(task['id'],)).fetchone()
                    if running and running[0]=='running':
                        conn.execute("UPDATE tasks SET status='interrupted',error='进程中断；写入状态须在恢复前核验' WHERE id=?",(task['id'],))
                if running and running[0]=='running':
                    service.state.update_job(task['job'],status='interrupted',error='进程中断；已保存检查点，恢复前核验目标库')
                    for attempt in service.state.ledger_rows('execution_attempt',task['job']):
                        if attempt.get('status')=='running':service.state.ledger_put('execution_attempt',attempt['attempt_id'],{**attempt,'status':'interrupted','finished_at':now(),'error':'进程中断，目标写入待核验'})
                if once:break
            elif once:break
            else:
                if time.monotonic()-cleaned>3600:clean_cache(service);cleaned=time.monotonic()
                time.sleep(.5)
