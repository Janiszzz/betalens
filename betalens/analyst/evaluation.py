"""Small actual-result interface shared by reports and mining workers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .trades import build_trade_cycles, trade_cycle_stats


@dataclass
class BacktestEvaluation:
    nav: pd.Series
    daily_returns: pd.Series
    daily_weight: pd.DataFrame
    daily_amount: pd.Series
    rebalance_log: pd.DataFrame
    synthetic_settlements: pd.DataFrame
    risk_events: pd.DataFrame
    trade_cycles: pd.DataFrame
    evaluation_version: str = "actual-v1"

    @classmethod
    def from_backtest(cls, bt):
        cycles = build_trade_cycles(bt.rebalance_log, bt.synthetic_settlements,
                                    bt.daily_position_value, bt.daily_amount.index)
        return cls(bt.nav.copy(), bt.daily_returns.copy(), bt.daily_weight.copy(), bt.daily_amount.copy(),
                   bt.rebalance_log.copy(), bt.synthetic_settlements.copy(), bt.risk_events.copy(), cycles)

    def window(self, start, end):
        lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        def daily(value):
            days = pd.DatetimeIndex(value.index).normalize()
            return value.loc[(days >= lo) & (days <= hi)].copy()
        returns = daily(self.daily_returns)
        baseline = pd.Series([1.], index=pd.DatetimeIndex([lo], name="datetime"))
        nav = pd.concat([baseline, (1 + returns).cumprod()]).rename("nav") if len(returns) else pd.Series(dtype=float)
        def rows(frame, key):
            if frame is None or frame.empty or key not in frame:
                return frame.copy() if frame is not None else pd.DataFrame()
            days = pd.to_datetime(frame[key]).dt.normalize()
            return frame.loc[(days >= lo) & (days <= hi)].copy()
        return BacktestEvaluation(nav, returns, daily(self.daily_weight), daily(self.daily_amount),
            rows(self.rebalance_log, "datetime"), rows(self.synthetic_settlements, "datetime"),
            rows(self.risk_events, "trigger_ts"), rows(self.trade_cycles, "exit_ts"))

    def metrics(self, *, annualizer=252, signal_count=0):
        from .metrics import performance_from_returns, turnover_from_trades
        result = performance_from_returns(self.daily_returns, annualizer)
        result["turnover"] = turnover_from_trades(self.rebalance_log, self.daily_amount, annualizer)["avg_oneway"]
        stock = self.daily_weight.drop(columns="cash", errors="ignore")
        exposure = stock.abs().sum(axis=1)
        cycle_stats = trade_cycle_stats(self.trade_cycles)
        result.update({key: cycle_stats[key] for key in ('trade_count', 'win_rate', 'odds')})
        result.update(avg_position=float(exposure.mean()) if len(exposure) else 0.,
            open_ratio=float((exposure > 1e-8).mean()) if len(exposure) else 0.,
            max_gross_exposure=float(exposure.max()) if len(exposure) else 0., signal_count=signal_count)
        return result
