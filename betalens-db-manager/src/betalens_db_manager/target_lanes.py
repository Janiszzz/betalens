"""Bounded independent database lanes; always join both before reporting failure."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from time import monotonic
from typing import Any


@dataclass
class LaneResult:
    target: str
    value: Any = None
    error: Exception | None = None
    elapsed_ms: int = 0


def target_lanes(targets, action, executor=None):
    targets=list(dict.fromkeys(targets))
    if not targets:return []
    if set(targets)-{'postgres','clickhouse'}:raise ValueError('目标库无效')
    def invoke(target):
        started=monotonic()
        try:return LaneResult(target,value=action(target),elapsed_ms=round((monotonic()-started)*1000))
        except Exception as exc:return LaneResult(target,error=exc,elapsed_ms=round((monotonic()-started)*1000))
    pool=executor or ThreadPoolExecutor(max_workers=len(targets),thread_name_prefix='target-lane')
    try:
        futures=[pool.submit(copy_context().run,invoke,target) for target in targets]
        return [future.result() for future in futures]
    finally:
        if executor is None:pool.shutdown(wait=True)


def lane_values(results):
    for result in results:
        if result.error is not None:raise result.error
    return [result.value for result in results]
