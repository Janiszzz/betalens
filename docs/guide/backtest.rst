回测模块
========

`betalens.backtest` 聚焦「调仓权重 → 日度净值」的最短路径，当前以 :class:`BacktestBase` 为核心。

权重矩阵规范
------------

- 行索引必须命名为 ``input_ts``，表示调仓时间（DatetimeIndex 或 MultiIndex 均可）
- 列代表证券代码；建议额外添加 ``cash`` 列以保留现金头寸
- 所有行需归一化（多头权重合计 1、空头合计 -1），框架内部会按成本价归一

.. code-block:: python

   import pandas as pd

   weights = pd.DataFrame(
       data=[[0.5, 0.5, 0.0], [-0.5, 0.5, 1.0]],
       index=pd.to_datetime(["2024-01-31 15:00", "2024-02-29 15:00"]),
       columns=["000001.SZ", "000002.SZ", "cash"],
   )
   weights.index.name = "input_ts"

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
   pnl = engine.daily_amount.diff()   # 绝对收益
   trades = engine.trades             # 成交统计（预留字段）

内部步骤
--------

1. ``get_rebalance_data``：调用 :class:`betalens.datafeed.Datafeed` 获取调仓日收盘价，并构造 ``cost_ret``
2. ``get_position_data``：根据上一期权重计算调仓后仓位与累计资产
3. ``get_daily_position_data``：查询调仓区间的全部交易日，前向填充头寸，得到 ``daily_amount`` 与 ``nav``

你可以覆写这些方法或继承 ``BacktestBase`` 来插入自定义逻辑（例如交易成本、换手限制）。

常见扩展
--------

- **多资产权重展开**：使用 ``factor.get_single_factor_weight`` 生成的权重直接透视为宽表即可。
- **成本拆解**：访问 ``engine.cost_price``（MultiIndex 数据帧）即可获取每次调仓对应的成交价。
- **持仓导出**：``engine.position`` 保存了每日数量（股数），可与行情 merge 输出。

若需要更复杂的交易撮合或期货保证金处理，可在此基础上扩展新的回测基类。


