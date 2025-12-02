"""
Betalens - 量化分析与回测框架

子模块：
- datafeed: 数据管理和数据库交互工具包
- analyst: 绩效分析模块
- backtest: 回测模块
- factor: 因子分析模块
- robust: 稳健性检验模块
"""

from .datafeed import Datafeed, get_absolute_trade_days, trade_days_offset
from .backtest import BacktestBase
from .analyst import PortfolioAnalyzer, ReportExporter

__all__ = [
    # Datafeed
    'Datafeed',
    'get_absolute_trade_days',
    'trade_days_offset',
    # Backtest
    'BacktestBase',
    # Analyst
    'PortfolioAnalyzer',
    'ReportExporter',
]

__version__ = '1.0.0'
__author__ = 'Janis'
