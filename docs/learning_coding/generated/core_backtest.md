# core_backtest：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-b96113b139e4"></a>
## betalens/backtest/__init__.py

[打开源码](../../../betalens/backtest/__init__.py) · 18 行 · 说明来源：文件族规则

- **作用**：Backtest模块 - 回测功能
- **输入**：import 请求
- **输出**：模块导出与符号；见静态 imports
- **副作用/维护重点**：初始化可能导入子模块；__all__ 与真实导出需结合源码阅读

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .backtest import BacktestBase, BacktestDataError, DateMismatchError, CodeMismatchError
```

<a id="file-c07be2fe05e7"></a>
## betalens/backtest/backtest.py

[打开源码](../../../betalens/backtest/backtest.py) · 1586 行 · 说明来源：人工文件说明

- **作用**：权重到实际持仓、损益与净值
- **输入**：权重矩阵、资金、价格/状态或预载面板
- **输出**：实例属性 nav、actual_weight、position、rebalance_log 等
- **副作用/维护重点**：构造即执行回测；默认查库；dump 写 Excel

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from betalens.datafeed import Datafeed
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import sys
import warnings
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [BacktestDataError](../../../betalens/backtest/backtest.py#L11) | class BacktestDataError(Exception) | 类定义；构造/属性见方法与字段 | 回测数据异常基类 |
| [DateMismatchError](../../../betalens/backtest/backtest.py#L16) | class DateMismatchError(BacktestDataError) | 类定义；构造/属性见方法与字段 | 日期不匹配异常 |
| [CodeMismatchError](../../../betalens/backtest/backtest.py#L21) | class CodeMismatchError(BacktestDataError) | 类定义；构造/属性见方法与字段 | 标的代码不匹配异常 |
| [format_data_sample](../../../betalens/backtest/backtest.py#L28) | format_data_sample(df, max_rows=3, max_cols=5) | 无返回注解；return: 'None'; '空DataFrame'; '\n '.join(sample_info) | 格式化数据样本用于错误信息 |
| [validate_weight_input](../../../betalens/backtest/backtest.py#L58) | validate_weight_input(weight) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 验证 weight 输入格式 Args: weight: 待验证的权重数据 Raises: BacktestDataError: 当格式不符合要求时 |
| [validate_query_result](../../../betalens/backtest/backtest.py#L138) | validate_query_result(df, expected_columns, query_name='数据库查询') | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 验证数据库查询结果 Args: df: 查询返回的 DataFrame expected_columns: 期望的列名列表 query_name: 查询名称（用于错误信息） Raises: BacktestDataError: 当格式不符合要求时 |
| [validate_pivot_result](../../../betalens/backtest/backtest.py#L178) | validate_pivot_result(df, expected_codes=None, index_levels=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 验证 pivot_table 结果 Args: df: pivot_table 后的 DataFrame expected_codes: 期望的标的代码列表 index_levels: 期望的索引层级名称列表 Raises: BacktestDataError: 当格式不符合要求时 |
| [validate_index_alignment](../../../betalens/backtest/backtest.py#L238) | validate_index_alignment(df1, df2, name1='DataFrame1', name2='DataFrame2') | 无返回注解；return: True; False | 验证两个 DataFrame 的索引是否对齐 Args: df1: 第一个 DataFrame df2: 第二个 DataFrame name1: 第一个 DataFrame 的名称 name2: 第二个 DataFrame 的名称 Returns: bool: 是否对齐 |
| [validate_calculation_inputs](../../../betalens/backtest/backtest.py#L277) | validate_calculation_inputs(*args, **kwargs) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 验证计算前的输入数据 Args: *args: 要验证的 DataFrame/Series **kwargs: 命名参数，格式为 name=df |
| [BacktestBase](../../../betalens/backtest/backtest.py#L358) | class BacktestBase(object) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [BacktestBase.__init__](../../../betalens/backtest/backtest.py#L359) | __init__(self, weight, symbol, amount, ftc=0.0, ptc=0.0, verbose=True, metric='收盘价(元)', time_tolerance=24, table_name='daily_market', check_trade_status=True, trade_status_mode='to_cash', trade_status_table='trade_status', lot_size=100, *, preloaded_cost_price=None, preloaded_close_price=None, preloaded_trade_status=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [BacktestBase.melt_weights](../../../betalens/backtest/backtest.py#L441) | melt_weights(self) | 无返回注解；return: 0; 1 | 无 docstring，需阅读函数体 |
| [BacktestBase._pivot_nearest_prices](../../../betalens/backtest/backtest.py#L449) | _pivot_nearest_prices(self, raw, metric, weight_codes) | 无返回注解；return: (prices, actual_dt) | 把 query_nearest_{after,before} 返回的长表转为宽表。 每个 input_ts 一行，避免不同 code 的真实 datetime 不同导致行爆炸。 Returns: prices: DataFrame, index=input_ts(DatetimeIndex), columns=code, 值=metric actual_dt: DataFrame, 同形状，值=每格真实成交 datetime（审计用） |
| [BacktestBase.get_trade_status](../../../betalens/backtest/backtest.py#L494) | get_trade_status(self) | 无返回注解；return: None | 从数据库提取调仓日的个券交易状态（一等流程，与 get_rebalance_data 并列）。 建立两份实例数据供审计与后续处理： self.trade_status: 长表 DataFrame，列 code/datetime/value/status_text/name value: 1=正常交易, 0=停牌等异常, -1=未上市/无法交易 self.trade_status_matrix: 宽表矩阵，index=调仓日(与 weight 对齐), columns=code, 值为 value（-1/0/1）；查询失败或关闭检查时为 None 提取完成后，若 check_trade_stat…（完整内容见 inventory.json/源码） |
| [BacktestBase._normalize_preloaded_trade_status](../../../betalens/backtest/backtest.py#L567) | _normalize_preloaded_trade_status(self, status, weight_codes) | 无返回注解；return: long_status | Validate a cached status panel and return the legacy long shape. |
| [BacktestBase._apply_trade_status](../../../betalens/backtest/backtest.py#L630) | _apply_trade_status(self) | 无返回注解；return: None | 按 trade_status_mode 处理 self.weight 中停牌（value==0）的持仓。 仅“停牌”(value==0) 视为异常需要处理；未上市(-1) 交由 get_rebalance_data 的 missing_codes 逻辑处理。 模式（trade_status_mode）： 'to_cash' : 默认。停牌股当期权重置0，资金留现金（假设买卖失败） 'hold' : 停牌无法调仓，沿用上一调仓日权重（持仓被动冻结） 'redistribute': 停牌股权重按比例分给当期可交易持仓，整行重新归一 'as_normal' : 忽略停牌，假设仍能正常买卖，仅统计提示…（完整内容见 inventory.json/源码） |
| [BacktestBase.get_rebalance_data](../../../betalens/backtest/backtest.py#L722) | get_rebalance_data(self) | 无返回注解；return: self.cost_price | 获取调仓日数据，包含日期和标的匹配验证 Raises: DateMismatchError: 当权重日期在数据库中无对应数据时 CodeMismatchError: 当权重标的在数据库中无数据时（严格模式） |
| [BacktestBase._normalize_preloaded_cost_price](../../../betalens/backtest/backtest.py#L936) | _normalize_preloaded_cost_price(self, raw, *, metric, weight_codes, weight_ts, ranges) | 无返回注解；return: result | Normalize cached execution prices to query_nearest_in_range_after shape. |
| [BacktestBase.get_position_data](../../../betalens/backtest/backtest.py#L1005) | get_position_data(self) | 无返回注解；return: self.amount | 无 docstring，需阅读函数体 |
| [BacktestBase.get_daily_position_data](../../../betalens/backtest/backtest.py#L1164) | get_daily_position_data(self) | 无返回注解；return: self.daily_amount | 无 docstring，需阅读函数体 |
| [BacktestBase._normalize_preloaded_close_price](../../../betalens/backtest/backtest.py#L1488) | _normalize_preloaded_close_price(self, raw, *, query_codes, start, end) | 无返回注解；return: result | Normalize cached daily valuation prices to query_time_range shape. |
| [BacktestBase.dump_to_excel](../../../betalens/backtest/backtest.py#L1543) | dump_to_excel(self, filepath: str) -&gt; str | str | 将 bt 实例的所有数据导出到一个 Excel 文件备查（每个属性一个 sheet） - DataFrame / Series 直接写入对应 sheet - 标量参数汇总到 'meta' sheet Args: filepath: 输出 Excel 路径 Returns: 实际写入的文件路径 |

