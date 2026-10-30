"""Own and restart one worker; never re-enqueue interrupted data operations."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime,timezone

from .util import atomic_json,now


def worker_status(settings):
    result={}
    for name in ('supervisor','heartbeat'):
        try:result[name]=json.loads((settings.state_dir/('worker-'+name+'.json')).read_text())
        except (OSError,ValueError):result[name]={}
    beat=result['heartbeat'];pid=beat.get('pid');alive=False
    if isinstance(pid,int) and pid>0:
        if os.name=='nt':alive=True  # Fresh heartbeats avoid Windows TerminateProcess semantics.
        else:
            try:os.kill(pid,0);alive=True
            except ProcessLookupError:pass
            except PermissionError:alive=True
    try:age=(datetime.now(timezone.utc)-datetime.fromisoformat(beat['at'])).total_seconds()
    except (KeyError,ValueError):age=None
    result.update(available=alive and age is not None and age<30 and beat.get('status') in {'idle','running'},heartbeat_age_seconds=age)
    return result


class WorkerSupervisor:
    def __init__(self,settings):
        self.settings=settings;self.stop=threading.Event();self.started=threading.Event();self.child=None;self.restarts=0
        self.thread=threading.Thread(target=self._run,name='db-manager-worker-supervisor',daemon=True)

    def publish(self,status,**values):
        try:atomic_json(self.settings.state_dir/'worker-supervisor.json',{'pid':os.getpid(),'worker_pid':self.child.pid if self.child else None,'at':now(),'status':status,'restarts':self.restarts,**values})
        except OSError as exc:
            from .worker import diagnostic
            diagnostic(self.settings,'worker_status_write_failed',error=str(exc))

    def start(self):
        self.thread.start();self.started.wait(5)

    def _stop_tree(self,child,graceful=False):
        if os.name=='posix':
            try:
                if graceful and child.poll() is None:child.terminate()
                else:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:return
            try:child.wait(timeout=70 if graceful else 3)
            except subprocess.TimeoutExpired:pass
            # A crashed worker may leave its execution child behind in the group.
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=5)
        elif child.poll() is None:
            child.terminate()
            try:child.wait(timeout=70 if graceful else 3)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
                child.wait(timeout=5)

    def _run(self):
        from .worker import diagnostic
        delay=.5
        while not self.stop.is_set():
            started=time.monotonic()
            try:
                command=[sys.executable,'-m','betalens_db_manager']
                if self.settings.env_file:command.extend(['--env',str(self.settings.env_file)])
                command.append('worker')
                with (self.settings.state_dir/'worker.log').open('a') as log:
                    self.child=subprocess.Popen(command,stdout=log,stderr=log,start_new_session=os.name=='posix')
                self.publish('monitoring');self.started.set()
                while self.child.poll() is None and not self.stop.wait(.25):pass
                if self.stop.is_set():
                    self._stop_tree(self.child,graceful=True);break
                code=self.child.returncode
                self._stop_tree(self.child)
                self.restarts+=1
                self.publish('restarting',exit_code=code,retry_seconds=delay)
                diagnostic(self.settings,'worker_restarting',worker_pid=self.child.pid,exit_code=code,retry_seconds=delay)
            except Exception as exc:
                from .security import safe_error
                if self.child and self.child.poll() is None:self._stop_tree(self.child)
                self.publish('restarting',error=safe_error(exc,self.settings),retry_seconds=delay);self.started.set()
                diagnostic(self.settings,'worker_launch_failed',error=safe_error(exc,self.settings),retry_seconds=delay)
            if self.stop.wait(delay):break
            delay=.5 if time.monotonic()-started>30 else min(30,delay*2)
        self.publish('stopped')

    def close(self):
        self.stop.set();self.thread.join(80)
        if self.thread.is_alive() and self.child:self._stop_tree(self.child)
