"""Betalens public package exports.

Keep package initialization lightweight so importing a focused submodule such
as ``betalens.datafeed`` does not eagerly import the full analyst/factor stack.
"""

from .datafeed import (
    Datafeed,
    FillStrategy,
    get_absolute_trade_days,
    trade_days_offset,
)

__all__ = [
    "Datafeed",
    "FillStrategy",
    "get_absolute_trade_days",
    "trade_days_offset",
    "BacktestBase",
    "BacktestDataError",
    "DateMismatchError",
    "CodeMismatchError",
    "EventStudy",
]


def __getattr__(name):
    if name in {"BacktestBase", "BacktestDataError", "DateMismatchError", "CodeMismatchError"}:
        from .backtest import BacktestBase, BacktestDataError, CodeMismatchError, DateMismatchError

        values = {
            "BacktestBase": BacktestBase,
            "BacktestDataError": BacktestDataError,
            "DateMismatchError": DateMismatchError,
            "CodeMismatchError": CodeMismatchError,
        }
        return values[name]
    if name == "EventStudy":
        from .eventstudy import EventStudy

        return EventStudy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "1.1.0"
