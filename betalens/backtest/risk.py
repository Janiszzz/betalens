"""Close-observed fixed exits anchored to the first actual entry of a side."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RiskConfig:
    stop_loss: float | None = None
    take_profit: float | None = None
    execution_metric: str = "open"

    def __post_init__(self):
        for key in ("stop_loss", "take_profit"):
            value = getattr(self, key)
            if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
                raise ValueError(f"risk.{key} 必须为有限正数或 None")
        if not isinstance(self.execution_metric, str) or not self.execution_metric:
            raise ValueError("risk.execution_metric 必须为非空价格字段名")

    @classmethod
    def parse(cls, value=None):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("risk 必须为配置字典或 RiskConfig")
        unknown = set(value) - {"stop_loss", "take_profit", "execution_metric"}
        if unknown:
            raise ValueError(f"未知 risk 配置: {sorted(unknown)}")
        options = dict(value)
        for key in ("stop_loss", "take_profit"):
            if options.get(key) is not None:
                number = float(options[key])
                if isinstance(options[key], bool) or not np.isfinite(number) or number <= 0:
                    raise ValueError(f"risk.{key} 必须为有限正数或 None")
                options[key] = number
        if not isinstance(options.get("execution_metric", "open"), str):
            raise ValueError("risk.execution_metric 必须为价格字段名")
        return cls(**options)

    @property
    def enabled(self):
        return self.stop_loss is not None or self.take_profit is not None


def validate_risk_config(risk=None, signal_params=None):
    config = RiskConfig.parse(risk)
    params = signal_params or {}
    signal = params.get("signal_weight") or {}
    state = signal.get("state") or params.get("state") or {}
    legacy = bool(state.get("enabled", False)) and any(state.get(k) is not None for k in ("stop_loss", "take_profit"))
    legacy = legacy or any(params.get(k) is not None for k in ("state_stop_loss", "state_take_profit"))
    if config.enabled and legacy:
        raise ValueError("统一回测 risk 与信号层止盈止损不能同时启用；请关闭旧 state 止盈止损")
    return config


class RiskManager:
    def __init__(self, config):
        self.config = RiskConfig.parse(config)
        self.entries = {}
        self.pending = {}
        self.exited_at = {}
        self.events = []

    def accepts(self, code, target_ts):
        return code not in self.pending and (code not in self.exited_at or target_ts > self.exited_at[code])

    def on_fill(self, code, before, after, price, ts):
        if not self.config.enabled:
            return
        old_side = int(np.sign(before)) if abs(before) > 1e-8 else 0
        new_side = int(np.sign(after)) if abs(after) > 1e-8 else 0
        if old_side != new_side:
            self.entries.pop(code, None)
            if new_side:
                self.entries[code] = (new_side, float(price), ts)
        if not new_side and code in self.pending:
            event = self.pending.pop(code)
            event.update(status="executed", execution_ts=ts)
            self.exited_at[code] = ts

    def observe(self, code, price, ts):
        if not self.config.enabled or code in self.pending or code not in self.entries:
            return None
        side, entry, entry_ts = self.entries[code]
        gain = side * (price / entry - 1.0)
        reason = None
        if self.config.stop_loss is not None and gain <= -self.config.stop_loss + 1e-12:
            reason = "stop_loss"
        elif self.config.take_profit is not None and gain >= self.config.take_profit - 1e-12:
            reason = "take_profit"
        if reason is None:
            return None
        event = dict(event_id=len(self.events) + 1, code=code, reason=reason,
                     trigger_ts=ts, entry_ts=entry_ts, entry_price=entry,
                     observed_price=float(price), trigger_return=gain, side=side,
                     execution_metric=self.config.execution_metric, status="pending",
                     execution_ts=None)
        self.events.append(event)
        self.pending[code] = event
        return event
