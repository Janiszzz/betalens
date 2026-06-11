"""
Analyst模块 - 策略评价与报告工具

主要类：
- Analyst: 一键评价门面（推荐入口，from_backtest 直接吃回测实例）
- PortfolioAnalyzer: 投资组合绩效分析器
- ReportExporter: 分年度/时段/基准报告导出（兼容旧接口）

子模块：
- metrics: 纯函数指标库（回撤/风险/交易持仓/归因/滚动）
- naming: 证券代码→中文名映射（查库+缓存）
- plotting: matplotlib PNG + plotly 交互图

快速上手：
    from betalens.analyst import Analyst
    a = Analyst.from_backtest(bt, benchmark=hs300_bt)
    a.report(to_excel='report.xlsx', to_html='report.html')
"""

from .analyst import Analyst, PortfolioAnalyzer, ReportExporter
from . import metrics, naming, plotting

__all__ = [
    'Analyst',
    'PortfolioAnalyzer',
    'ReportExporter',
    'metrics',
    'naming',
    'plotting',
]

__version__ = '2.0.0'
__author__ = 'Janis'
