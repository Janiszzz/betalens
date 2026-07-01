"""Lightweight datafeed package exports used by the factor pipeline."""

from .core import Datafeed, func_timer, get_absolute_trade_days, trade_days_offset
from .industry import build_industry_records, get_industry_members, query_industry
from .validation import FillStrategy, DataValidator
from .universe import get_index_universe, get_index_universe_date

__all__ = [
    "Datafeed",
    "FillStrategy",
    "DataValidator",
    "func_timer",
    "get_absolute_trade_days",
    "trade_days_offset",
    "build_industry_records",
    "get_industry_members",
    "query_industry",
    "get_index_universe",
    "get_index_universe_date",
]

__version__ = "2.3.1"
__author__ = "Janis"
__date__ = "2025-11-04"
