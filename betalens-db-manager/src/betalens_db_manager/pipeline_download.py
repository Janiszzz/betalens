"""Independent archive producer; downstream work only consumes small group IDs."""
from collections import defaultdict
from contextvars import copy_context
from pathlib import Path
from queue import Queue,Empty,Full
from threading import Event,RLock,Thread
import shutil,time

from .resource_budget import BUDGET
from .util import now


class DownloadBuffer:
    def __init__(self,pipeline,downloader,groups,request):
        self.pipeline=pipeline;self.downloader=downloader;self.groups=groups;self.request=request
        self.service=pipeline.s;self.job=downloader.job
        self.queue=Queue(maxsize=BUDGET.download_ready_groups)
        self.stopped=Event();self.done=Event();self.lock=RLock();self.error=None
        self.started_at=now();self.finished_at=None;self.completed=0;self.failures=0;self.bytes=0;self.paused=False
        self.paths={};self.owners=defaultdict(set);self.group_paths=defaultdict(set)
        self.started=set();self.allowed_to_finish=set();self.writes=defaultdict(set)
        self.released=set();self.last_publish=0;self.last_disk=0
        units=pipeline.store.units(self.job)
        stored={(u['payload'].get('child_id'),u['payload'].get('target')) for u in units if u['kind']=='stored' and u['status']=='completed'}
        for unit in units:
            if unit['kind']=='normalize' and unit['status']=='completed':
                child=(unit.get('result') or {}).get('child_id')
                if all((child,t) in stored for t in request['targets']):self.released.add(unit['id'])
        context=copy_context()
        self.thread=Thread(target=lambda:context.run(self._run),name='archive-producer')

    def __enter__(self):
        self.thread.start();return self

    def __exit__(self,*_):self.close()

    def check_cancel(self):
        if self.stopped.is_set():raise InterruptedError('下载派发已停止')
        self.service.state.check_cancel(self.job)

    def admit(self,group,head):
        self.check_cancel()
        instant=time.monotonic()
        with self.lock:
            if instant-self.last_disk>=1:
                if shutil.disk_usage(self.service.settings.state_dir).free<self.request['min_free_bytes']:
                    raise RuntimeError('剩余磁盘不足，原始归档已保存，请释放空间后恢复')
                self.last_disk=instant
            if not self.paused and self.bytes>=self.request['backlog_high_bytes']:
                self.paused=True;self.allowed_to_finish=set(self.started)
                self.service.state.log(self.job,'download_buffer_paused',buffered_source_bytes=self.bytes)
                self.publish(force=True)
            if self.paused and self.bytes<=self.request['backlog_low_bytes']:
                self.paused=False;self.allowed_to_finish.clear()
                self.service.state.log(self.job,'download_buffer_resumed',buffered_source_bytes=self.bytes)
                self.publish(force=True)
            if self.paused:
                if not self.request['auto_commit']:
                    raise RuntimeError('待提交缓存达到积压上限，请提交已有缓存或分段执行')
                if self.pipeline.failed_targets:
                    raise RuntimeError('目标库失败且缓存达到积压上限，请修复失败目标后恢复')
                # Finish the already-started head: otherwise the consumer may
                # never receive the group needed to release the byte budget.
                if group not in self.released and not (group==head and group in self.allowed_to_finish):return False
            self.started.add(group)
            return True

    def archived(self,group,request):
        prepared=self.downloader.prepared(request)[0]
        part=self.service.state.get_slice(self.job,self.downloader.key(prepared))
        if not part or not part.get('archive'):return
        path=str(Path(part['archive']).resolve())
        size=Path(path).stat().st_size
        with self.lock:
            if group in self.released:return
            if path not in self.paths:self.paths[path]=size;self.bytes+=size
            self.owners[path].add(group);self.group_paths[group].add(path)
            self.publish()

    def release(self,group):
        """Release admission budget, retaining every archived evidence file."""
        with self.lock:
            self.released.add(group)
            for path in self.group_paths.pop(group,()):
                self.owners[path].discard(group)
                if not self.owners[path]:
                    self.bytes-=self.paths.pop(path);del self.owners[path]
            self.publish(force=self.done.is_set() or self.bytes==0)

    def written(self,group,target):
        with self.lock:
            self.writes[group].add(target)
            if all(t in self.writes[group] for t in self.request['targets']):self.release(group)

    def publish(self,force=False):
        with self.lock:
            if not force and time.monotonic()-self.last_publish<5:return
            self.last_publish=time.monotonic()
            value={'task_id':self.job,'download_status':('partial' if self.error or self.failures else 'completed') if self.done.is_set() else 'buffered' if self.paused else 'running',
                   'started_at':self.started_at,'updated_at':now(),'completed_groups':self.completed,
                   'total_groups':len(self.groups),'ready_groups':self.queue.qsize(),'buffered_source_bytes':self.bytes,
                   'independent_producer':True}
            if self.done.is_set():value['downloaded_at']=self.finished_at
            self.service.state.ledger_put('pipeline_progress',self.job,value)

    def _run(self):
        iterator=None
        try:
            self.publish(force=True)
            iterator=self.downloader.run_groups(self.groups,self.request['download_workers'],lookahead=BUDGET.download_group_lookahead,flow=self)
            for key,_,failures in iterator:
                self.completed+=1;self.failures+=len(failures)
                while True:
                    self.check_cancel()
                    try:self.queue.put((key,failures),timeout=.1);break
                    except Full:pass
                self.publish()
        except BaseException as exc:self.error=exc
        finally:
            if iterator is not None:iterator.close()
            self.finished_at=now();self.done.set();self.publish(force=True)

    def __iter__(self):
        while True:
            self.check_cancel()
            try:yield self.queue.get(timeout=.1)
            except Empty:
                if self.done.is_set():
                    if self.error is not None:raise self.error
                    return

    def close(self):
        self.stopped.set();self.downloader.stop()
        self.thread.join()
