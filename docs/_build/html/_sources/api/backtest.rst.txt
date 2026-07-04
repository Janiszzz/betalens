Backtest API
============

核心包：:mod:`betalens.backtest`

.. automodule:: betalens.backtest
   :members:
   :undoc-members:
   :show-inheritance:

backtest.backtest
-----------------

.. automodule:: betalens.backtest.backtest
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

BacktestBase 当前签名
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   BacktestBase(
       weight, symbol, amount,
       ftc=0.0, ptc=0.0, verbose=True,
       metric='收盘价(元)', time_tolerance=24, table_name='daily_market',
       check_trade_status=True, trade_status_mode='to_cash',
       trade_status_table='trade_status', lot_size=100,
   )

从目标调仓权重计算日频净值。回测按整数手成交，未成交资金转入 ``cash``；换仓前可查询 ``trade_status`` 并按 ``trade_status_mode`` 处理停牌持仓。

:param weight: 目标权重 DataFrame，行是调仓时间，列是证券代码，建议包含 ``cash``。
:param symbol: 策略名称。
:param amount: 初始资金。
:param ftc: 固定交易费用。
:param ptc: 百分比交易费用。
:param verbose: 是否打印过程信息。
:param metric: 成交价格指标，默认 ``收盘价(元)``。
:param time_tolerance: 价格查询时间容差，单位小时，默认 ``24``。
:param table_name: 价格表名，默认 ``daily_market``。
:param check_trade_status: 是否检查交易状态，默认 ``True``。
:param trade_status_mode: 停牌处理模式：``to_cash`` / ``hold`` / ``redistribute`` / ``as_normal`` / ``report_only``。
:param trade_status_table: 交易状态表名，默认 ``trade_status``。
:param lot_size: 一手股数，默认 ``100``。

关键属性包括 ``nav``、``actual_weight``、``daily_position_value``、``daily_pnl``、``daily_pnl_total``、``rebalance_log``、``position``、``trade_status``、``trade_status_matrix``。
