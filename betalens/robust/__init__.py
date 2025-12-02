"""
Robust模块 - 稳健性检验

提供因子稳健性检验和增量分析功能。
"""

from .robust import (
    RobustTest,
    panel,
    fake_fund,
    bootstrap_fake_fund,
    parse_name_dates,
    get_interval,
    gen_date_pairs
)

__all__ = [
    'RobustTest',
    'panel',
    'fake_fund',
    'bootstrap_fake_fund',
    'parse_name_dates',
    'get_interval',
    'gen_date_pairs'
]

