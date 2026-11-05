"""One deployment budget shared by ingestion and independent validation."""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import os
import threading
import time
from pathlib import Path
from collections import deque


@dataclass(frozen=True)
class ResourceBudget:
    name: str = 'throughput_90'
    download_workers: int = 12
    download_max_workers: int = 16
    cpu_workers: int = 3
    validation_io_workers: int = 6
    readonly_per_target: int = 2
    pg_connections: int = 6
    ch_threads: int = 3
    ch_read_memory_bytes: int = 2 * 1024**3
    ch_key_rows: int = 10000
    ch_key_bytes: int = 8 * 1024**2
    processing_memory_bytes: int = 6 * 1024**3
    manager_high_bytes: int = 7 * 1024**3
    minimum_available_bytes: int = 4 * 1024**3
    pending_normalizations: int = 6
    pending_writes: int = 8
    download_group_lookahead: int = 16
    download_ready_groups: int = 1024

    def public(self):
        return asdict(self)


BUDGET = ResourceBudget()
_read_locks = {}
_guard = threading.Lock()


@contextmanager
def target_read_slot(target):
    # All service instances within one execution process share the same slots.
    with _guard:
        semaphore = _read_locks.setdefault((os.getpid(), target), threading.BoundedSemaphore(BUDGET.readonly_per_target))
    with semaphore:
        yield


@contextmanager
def shared_slot(root, name, count, cancelled=lambda: None):
    """Process-wide deployment slots; the OS releases leases after a crash."""
    directory = Path(root) / 'resource-slots' / name
    directory.mkdir(parents=True, exist_ok=True)
    handle = None
    while handle is None:
        cancelled()
        for number in range(count):
            candidate = (directory / str(number)).open('a+b')
            try:
                if os.name == 'nt':
                    import msvcrt
                    if not candidate.tell():candidate.write(b'0');candidate.flush()
                    candidate.seek(0)
                    msvcrt.locking(candidate.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, PermissionError):
                candidate.close()
                continue
            handle = candidate
            break
        if handle is None:time.sleep(.05)
    try:
        yield
    finally:
        handle.close()


def memory_available():
    try:
        with open('/proc/meminfo') as handle:
            for line in handle:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def io_pressure():
    try:
        with open('/proc/pressure/io') as handle:
            for line in handle:
                if line.startswith('full '):
                    return float(dict(v.split('=') for v in line.split()[1:])['avg10'])
    except (OSError, ValueError, KeyError):
        pass
    return None


def cgroup_usage():
    try:
        relative=Path('/proc/self/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
        root=Path('/sys/fs/cgroup')/relative
        memory=int((root/'memory.current').read_text())
        cpu=dict(line.split() for line in (root/'cpu.stat').read_text().splitlines())
        return memory,int(cpu['usage_usec'])/1e6
    except (OSError,ValueError,KeyError,IndexError):return None,None


class DownloadPressure:
    """Sample every five seconds; tune admission only at a 30-second boundary."""
    def __init__(self,service,job,ceiling):
        self.service=service;self.job=job;self.ceiling=ceiling
        self.limit=min(BUDGET.download_workers,ceiling);self.samples=deque(maxlen=6)
        self.last_sample=-float('inf');self.last_adjust=time.monotonic();self.last_cpu=None

    def admission(self):
        instant=time.monotonic()
        if instant-self.last_sample<5:return self.limit
        available=memory_available();pressure=io_pressure();memory,cpu=cgroup_usage()
        cpu_ratio=None
        if cpu is not None and self.last_cpu is not None:cpu_ratio=max(0,cpu-self.last_cpu)/max(.01,instant-self.last_sample)/3.8
        self.last_cpu=cpu;self.last_sample=instant
        self.samples.append({'available_bytes':available,'manager_bytes':memory,'cpu_ratio':cpu_ratio,'io_full_avg10':pressure})
        low_memory=available is not None and available<BUDGET.minimum_available_bytes or memory is not None and memory>BUDGET.manager_high_bytes
        if low_memory:self.limit=min(self.limit,min(4,self.ceiling))
        if instant-self.last_adjust>=30:
            io_busy=len(self.samples)==6 and all((r['io_full_avg10'] or 0)>5 for r in self.samples)
            if low_memory or io_busy:self.limit=max(min(4,self.ceiling),self.limit-1)
            elif all((r['cpu_ratio'] or 0)<.8 and (r['manager_bytes'] or 0)<.8*BUDGET.manager_high_bytes for r in self.samples):self.limit=min(self.ceiling,self.limit+1)
            self.last_adjust=instant
            self.service.state.log(self.job,'resource_pressure',profile=BUDGET.name,download_limit=self.limit,**self.samples[-1])
        return self.limit
