"""
Betalens - 因子框架

子模块：
- datafeed: 数据管理和数据库交互工具包
- analyst: 分析师模块
- backtest: 回测模块
- factor: 因子模块
"""

# 导入datafeed子模块的所有公开API
from .datafeed import (
    Datafeed,
    FillStrategy,
    DataValidator,
    func_timer,
    get_absolute_trade_days,
    trade_days_offset,
)

# 导入backtest异常类
from .backtest import (
    BacktestBase,
    BacktestDataError,
    DateMismatchError,
    CodeMismatchError,
)

# 导入其他子模块（保持原有导入方式）
import betalens.analyst
import betalens.backtest
import betalens.factor

# 定义公开的API
__all__ = [
    'Datafeed',
    'DataValidator',
    'FillStrategy',
    'func_timer',
    'get_absolute_trade_days',
    'trade_days_offset',
    'BacktestBase',
    'BacktestDataError',
    'DateMismatchError',
    'CodeMismatchError',
]

__version__ = '1.1.0'
