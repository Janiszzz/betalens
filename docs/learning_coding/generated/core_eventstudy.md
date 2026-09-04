# core_eventstudy：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-e01684539837"></a>
## betalens/eventstudy/__init__.py

[打开源码](../../../betalens/eventstudy/__init__.py) · 4 行 · 说明来源：文件族规则

- **作用**：包导出/包标识
- **输入**：import 请求
- **输出**：模块导出与符号；见静态 imports
- **副作用/维护重点**：初始化可能导入子模块；__all__ 与真实导出需结合源码阅读

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .eventstudy import EventStudy
```

<a id="file-92a9334f6d00"></a>
## betalens/eventstudy/eventstudy.py

[打开源码](../../../betalens/eventstudy/eventstudy.py) · 792 行 · 说明来源：人工文件说明

- **作用**：标准交易日事件窗口研究
- **输入**：Datafeed、事件 Series、代码与窗口
- **输出**：价格/收益矩阵、统计、比较结果和图形
- **副作用/维护重点**：查行情与日历；Day 0、重叠和缺失不可压缩错位

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from betalens.datafeed import get_absolute_trade_days
from typing import Optional, List, Union, Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_get_event_dates](../../../betalens/eventstudy/eventstudy.py#L19) | _get_event_dates(events: pd.Series) -&gt; pd.DatetimeIndex | pd.DatetimeIndex | 从事件序列中提取事件发生日期 |
| [_calc_returns](../../../betalens/eventstudy/eventstudy.py#L24) | _calc_returns(prices: pd.Series) -&gt; pd.Series | pd.Series | 计算日收益率 |
| [_normalize_date](../../../betalens/eventstudy/eventstudy.py#L29) | _normalize_date(value: pd.Timestamp) -&gt; pd.Timestamp | pd.Timestamp | Normalize a timestamp to a timezone-naive calendar date. |
| [_get_day0_trade_day_loc](../../../betalens/eventstudy/eventstudy.py#L37) | _get_day0_trade_day_loc(trade_days: pd.DatetimeIndex, event_date: pd.Timestamp, market_close_hour: int=15) -&gt; Optional[int] | Optional[int] | Locate the event's cost-price day on the standard trade calendar. |
| [_standard_trade_window](../../../betalens/eventstudy/eventstudy.py#L52) | _standard_trade_window(event_dates: pd.DatetimeIndex, window_before: int, window_after: int, market_close_hour: int, exchange: str) -&gt; pd.DatetimeIndex | pd.DatetimeIndex | Load standard trade days covering every event window that fits. |
| [_prices_on_trade_days](../../../betalens/eventstudy/eventstudy.py#L98) | _prices_on_trade_days(prices: pd.Series, trade_days: pd.DatetimeIndex) -&gt; pd.Series | pd.Series | Align observed prices to the standard calendar without shifting gaps. |
| [_get_window_returns](../../../betalens/eventstudy/eventstudy.py#L111) | _get_window_returns(prices: pd.Series, event_date: pd.Timestamp, window_before: int, window_after: int, market_close_hour: int=15, trade_days: Optional[pd.DatetimeIndex]=None) -&gt; Optional[pd.Series] | Optional[pd.Series] | Return the observed daily close-to-close returns around an event. |
| [_get_window_prices](../../../betalens/eventstudy/eventstudy.py#L145) | _get_window_prices(prices: pd.Series, event_date: pd.Timestamp, window_before: int, window_after: int, market_close_hour: int, trade_days: pd.DatetimeIndex) -&gt; Optional[pd.Series] | Optional[pd.Series] | Return close prices normalized to zero on the event anchor day. |
| [_stats_frame](../../../betalens/eventstudy/eventstudy.py#L171) | _stats_frame(returns_df: pd.DataFrame) -&gt; pd.DataFrame | pd.DataFrame | Compute the standard event-study statistics for each relative day. |
| [_compute_stats](../../../betalens/eventstudy/eventstudy.py#L179) | _compute_stats(returns: pd.Series) -&gt; dict | dict | 计算收益率统计量: 均值、标准差、上涨概率、胜率、t统计量、样本数 |
| [_compute_period_stats](../../../betalens/eventstudy/eventstudy.py#L216) | _compute_period_stats(returns_df: pd.DataFrame, event_dates: pd.DatetimeIndex, periods: pd.Series) -&gt; pd.DataFrame | pd.DataFrame | 按时间段分组计算统计量 |
| [EventStudy](../../../betalens/eventstudy/eventstudy.py#L239) | class EventStudy() | 类定义；构造/属性见方法与字段 | 事件研究分析器 |
| [EventStudy.__init__](../../../betalens/eventstudy/eventstudy.py#L242) | __init__(self, datafeed) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [EventStudy._get_stock_window_returns](../../../betalens/eventstudy/eventstudy.py#L245) | _get_stock_window_returns(self, code: str, event_dates: pd.DatetimeIndex, start: str, end: str, metric: str, window_before: int, window_after: int, market_close_hour: int, trade_days: pd.DatetimeIndex, benchmark_returns: Optional[pd.Series]=None, benchmark_prices: Optional[pd.Series]=None) -&gt; tuple[pd.DataFrame, pd.DataFrame, Optional[str]] | tuple[pd.DataFrame, pd.DataFrame, Optional[str]] | 获取单个股票在所有事件窗口的收益率矩阵 Returns: (returns, normalized prices, error) |
| [EventStudy._calc_holding_returns](../../../betalens/eventstudy/eventstudy.py#L321) | _calc_holding_returns(self, returns_df: pd.DataFrame, holding_periods: Optional[dict]=None, holding_start_offset: int=0) -&gt; tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame] | Calculate fixed holding returns and their cross-event statistics. |
| [EventStudy.analyze](../../../betalens/eventstudy/eventstudy.py#L364) | analyze(self, events: pd.Series, code: Union[str, List[str]], window_before: int=5, window_after: int=5, metric: str='收盘价(元)', periods: Optional[pd.Series]=None, holding_periods: Optional[dict]=None, holding_start_offset: int=0, market_close_hour: int=15, benchmark_code: Optional[str]=None, multi_asset_mode: Literal['aggregate', 'compare']='aggregate', exchange: str='SHSE') -&gt; dict | dict | 分析事件前后的收益率表现 Args: events: 事件序列，index为精确到秒的datetime，值为1表示事件发生 code: 证券代码，可以是单个代码(str)或多个代码列表(List[str]) 当为列表时，计算所有股票在每个时间点的平均收益率 window_before: 事件前窗口期数（相对于事件后第一个收益点） window_after: 事件后窗口期数 metric: 价格指标名称 periods: 可选的时间分段序列 holding_periods: 持有期字典，如{'days': [1,2,3,4,5], 'months': [1,3,6,9,12]} holding_…（完整内容见 inventory.json/源码） |
| [EventStudy.plot_bar](../../../betalens/eventstudy/eventstudy.py#L557) | plot_bar(self, daily_stats: pd.DataFrame, title: str='事件前后平均收益率', figsize: tuple=(12, 6), save_path: Optional[str]=None) -&gt; None | None | 类型1：柱状图展示 -n~n 平均收益率序列 Args: daily_stats: 每日收益统计DataFrame（来自analyze结果的daily_stats） title: 图表标题 figsize: 图表尺寸 save_path: 保存路径，如不提供则显示图表 |
| [EventStudy.plot_multi_stocks](../../../betalens/eventstudy/eventstudy.py#L610) | plot_multi_stocks(self, events: pd.Series, codes: List[str], event_index: int=0, window_before: int=10, window_after: int=10, metric: str='收盘价(元)', market_close_hour: int=15, title: Optional[str]=None, figsize: tuple=(14, 8), save_path: Optional[str]=None, exchange: str='SHSE') -&gt; None | None | 折线图展示给定股票池在特定事件的归一化价格曲线。 Args: events: 事件序列 codes: 股票代码列表 event_index: 选择第几个事件（0表示第一个事件） window_before: 事件前窗口 window_after: 事件后窗口 metric: 价格指标 market_close_hour: 市场收盘时间 title: 图表标题 figsize: 图表尺寸 save_path: 保存路径 |
| [EventStudy.plot_events_lines](../../../betalens/eventstudy/eventstudy.py#L709) | plot_events_lines(self, events: pd.Series, code: str, window_before: int=10, window_after: int=10, metric: str='收盘价(元)', market_close_hour: int=15, title: Optional[str]=None, figsize: tuple=(14, 8), max_events: Optional[int]=None, save_path: Optional[str]=None, exchange: str='SHSE') -&gt; None | None | 折线图叠加同一标的在多个事件周围的归一化价格曲线。 Args: events: 事件序列 code: 股票代码 window_before: 事件前窗口 window_after: 事件后窗口 metric: 价格指标 market_close_hour: 市场收盘时间 title: 图表标题 figsize: 图表尺寸 max_events: 最多展示的事件数量，None表示全部展示 save_path: 保存路径 |

