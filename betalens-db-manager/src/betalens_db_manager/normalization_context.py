"""Short-lived immutable input reuse, discarded at every consumption boundary."""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

CURRENT = ContextVar('normalization_inputs', default=None)


class Inputs:
    def __init__(self, item, budget=32*1024**2):
        self.item = item
        self.budget = budget
        self.used = 0
        self.archives = OrderedDict()
        self.binding_plans = {}

    def put(self, key, data, size):
        # Uncompressed wire bytes are conservative only for strings; account
        # for Python objects as well, bounding this cache below process memory.
        cost = size * 8
        if cost > self.budget:return
        while self.archives and self.used + cost > self.budget:
            _, (_, old_size) = self.archives.popitem(last=False)
            self.used -= old_size
        self.archives[key] = (data, cost)
        self.used += cost


@contextmanager
def normalization_inputs(item):
    token = CURRENT.set(Inputs(item))
    try:yield
    finally:CURRENT.reset(token)
