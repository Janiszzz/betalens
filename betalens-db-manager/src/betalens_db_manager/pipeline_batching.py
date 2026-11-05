"""Bounded FIFO target lanes that coalesce only already eligible cohorts."""
from concurrent.futures import Future
from threading import Condition,Thread
from time import monotonic
from .resource_budget import BUDGET


class WriteLane:
    def __init__(self,work,*,max_wait,row_limit,max_cohorts=8,name='write',capacity=BUDGET.pending_writes):
        self.work=work;self.max_wait=max_wait;self.row_limit=row_limit;self.max_cohorts=max_cohorts
        self.capacity=capacity
        self.condition=Condition();self.pending=[];self.closed=False
        self.thread=Thread(target=self._run,name=name);self.thread.start()

    def submit(self,child,snapshot):
        future=Future()
        with self.condition:
            while len(self.pending)>=self.capacity and not self.closed:self.condition.wait()
            if self.closed:raise RuntimeError('写入队列已关闭')
            self.pending.append((child,snapshot,future,monotonic()));self.condition.notify()
        return future

    def close(self):
        with self.condition:self.closed=True;self.condition.notify_all()

    def join(self):self.thread.join()

    def _run(self):
        while True:
            with self.condition:
                while not self.pending and not self.closed:self.condition.wait()
                if not self.pending:return
                while not self.closed and len(self.pending)<self.max_cohorts:
                    # Large cohorts start immediately; small arrivals get a
                    # bounded opportunity to share a physical table write.
                    rows=sum(r.get('rows',0) for _,s,_,_ in self.pending for records in s['tables'].values() for r in records)
                    left=self.max_wait-(monotonic()-self.pending[0][3])
                    if rows>=self.row_limit or left<=0:break
                    self.condition.wait(left)
                group=self.pending[:self.max_cohorts];del self.pending[:len(group)]
                self.condition.notify_all()
            active=[entry for entry in group if entry[2].set_running_or_notify_cancel()]
            if not active:continue
            try:
                results=self.work([(c,s) for c,s,_,_ in active])
                if len(results)!=len(active):raise RuntimeError('写入批次结果数量不符')
            except BaseException as exc:
                for _,_,future,_ in active:future.set_exception(exc)
            else:
                for entry,result in zip(active,results):entry[2].set_result(result)
