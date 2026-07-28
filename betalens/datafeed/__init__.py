"""Lightweight datafeed package exports used by the factor pipeline."""

from .core import Datafeed, get_absolute_trade_days, trade_days_offset
from .industry import get_industry_members, query_industry
from .validation import FillStrategy
from .universe import get_index_universe, get_index_universe_date

__all__ = [
    "Datafeed",
    "FillStrategy",
    "get_absolute_trade_days",
    "trade_days_offset",
    "get_industry_members",
    "query_industry",
    "get_index_universe",
    "get_index_universe_date",
]

__version__ = "2.3.1"
__author__ = "Janis"
__date__ = "2025-11-04"
