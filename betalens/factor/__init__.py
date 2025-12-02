"""
Factor模块 - 因子分析

提供因子构建、分组和权重生成功能。
"""

from .factor import (
    get_tradable_pool,
    single_factor,
    get_single_factor_weight,
    describe_labeled_pool
)

__all__ = [
    'get_tradable_pool',
    'single_factor',
    'get_single_factor_weight',
    'describe_labeled_pool'
]

