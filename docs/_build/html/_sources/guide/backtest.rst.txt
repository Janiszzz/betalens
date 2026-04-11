回测模块
========

`betalens.backtest` 聚焦「调仓权重 → 日度净值」的最短路径，以 :class:`BacktestBase` 为核心。

权重矩阵规范
------------

- 行索引为时间类型（DatetimeIndex），表示调仓时间
- 列代表证券代码；建议额外添加 ``cash`` 列以保留现金头寸
- 所有行需归一化（多头权重合计 1、空头合计 -1），框架内部会按成本价归一

.. code-block:: python

   import pandas as pd

   weights = pd.DataFrame(
       data=[[0.5, 0.5, 0.0], [-0.5, 0.5, 1.0]],
       index=pd.to_datetime(["2024-01-31 15:00", "2024-02-29 15:00"]),
       columns=["000001.SZ", "000002.SZ", "cash"],
   )

回测执行
--------

.. code-block:: python

   from betalens.backtest import BacktestBase

   engine = BacktestBase(
       weight=weights,
       symbol="Demo",
       amount=5_000_000,
       ftc=0.0002,  # 固定费用
       ptc=0.0,     # 百分比费用
       verbose=True,
   )

   nav = engine.nav                   # 日度净值 Series
   daily_amount = engine.daily_amount # 每日市值
   position = engine.position         # 每日持仓数量

主要属性
--------

- ``nav``: 归一化净值序列（基于初始金额）
- ``daily_amount``: 每日市值序列
- ``position``: 每日持仓数量 DataFrame
- ``cost_price``: 调仓成本价 DataFrame（MultiIndex）
- ``cost_ret``: 调仓区间收益率 DataFrame
- ``start`` / ``end``: 回测起止时间
- ``initial_amount``: 初始资金

内部步骤
--------

1. **get_rebalance_data**：
   - 调用 :class:`betalens.datafeed.Datafeed` 获取调仓日收盘价
   - 验证日期匹配和标的匹配
   - 构造 ``cost_price`` 和 ``cost_ret``

2. **get_position_data**：
   - 根据上一期权重计算调仓后仓位
   - 计算累计资产 ``amount``

3. **get_daily_position_data**：
   - 查询调仓区间的全部交易日
   - 前向填充头寸
   - 得到 ``daily_amount`` 和 ``nav``

异常处理
--------

框架提供详细的异常类和错误信息：

.. code-block:: python

   from betalens.backtest import (
       BacktestDataError,
       DateMismatchError,
       CodeMismatchError
   )

   try:
       engine = BacktestBase(weight=weights, symbol="Demo", amount=1_000_000)
   except DateMismatchError as e:
       print(f"日期不匹配: {e}")
   except CodeMismatchError as e:
       print(f"标的不匹配: {e}")
   except BacktestDataError as e:
       print(f"数据错误: {e}")

数据验证
--------

框架自动执行多层数据验证：

1. **权重输入验证**：检查DataFrame格式、索引类型、数值类型、NaN/Inf值
2. **查询结果验证**：检查数据库返回的列是否完整
3. **日期匹配检查**：确保权重日期在数据库中有对应数据
4. **标的匹配检查**：警告并置零缺失标的的权重
5. **计算前验证**：检查输入数据的完整性

常见扩展
--------

**多资产权重展开**：

.. code-block:: python

   from betalens.factor.factor import get_single_factor_weight

   # 从因子模块生成的权重可直接使用
   weights = get_single_factor_weight(labeled_pool, params)
   weights["cash"] = 0
   engine = BacktestBase(weight=weights, symbol="Factor", amount=1_000_000)

**成本拆解**：

.. code-block:: python

   # 访问每次调仓对应的成交价
   cost_price = engine.cost_price  # MultiIndex: (input_ts, datetime)

**持仓导出**：

.. code-block:: python

   # 每日持仓数量（股数）
   position = engine.position
   position.to_csv("position.csv")

**连接绩效分析**：

.. code-block:: python

   from betalens.analyst import PortfolioAnalyzer, ReportExporter

   analyzer = PortfolioAnalyzer(engine.nav)
   print(f"Sharpe: {analyzer.sharpe_ratio():.4f}")
   print(f"Max Drawdown: {analyzer.max_drawdown():.2%}")

   exporter = ReportExporter(analyzer)
   exporter.generate_annual_report()

若需要更复杂的交易撮合或期货保证金处理，可在此基础上继承 ``BacktestBase`` 扩展新的回测基类。


