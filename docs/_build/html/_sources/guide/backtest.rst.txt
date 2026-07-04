回测模块
========

``betalens.backtest`` 将调仓权重转换为日频净值、持仓、损益和调仓审计记录。核心类是 ``BacktestBase``。

权重矩阵
--------

* 行索引为 ``DatetimeIndex``，表示调仓日。
* 列为证券代码，建议显式包含 ``cash``。
* 多头权重合计通常为 1，空头权重合计通常为 -1。
* 输入 ``weight`` 是目标权重；整数手撮合后的真实权重在 ``engine.actual_weight``。

基本用法
--------

.. code-block:: python

   from betalens.backtest import BacktestBase

   engine = BacktestBase(
       weight=weights,
       symbol="DemoFactor",
       amount=100_000_000,
       ftc=0.0,
       ptc=0.0,
       verbose=True,
       metric="收盘价(元)",
       time_tolerance=24,
       table_name="daily_market",
       check_trade_status=True,
       trade_status_mode="to_cash",
       trade_status_table="trade_status",
       lot_size=100,
   )

   nav = engine.nav
   actual_weight = engine.actual_weight
   rebalance_log = engine.rebalance_log

关键参数
--------

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - 参数
     - 含义
     - 默认值
   * - ``metric``
     - 成交价格指标
     - ``收盘价(元)``
   * - ``time_tolerance``
     - 价格查询时间容差，单位小时
     - ``24``
   * - ``table_name``
     - 价格所在数据表
     - ``daily_market``
   * - ``lot_size``
     - 一手股数，按整数手成交
     - ``100``
   * - ``check_trade_status``
     - 是否在换仓前检查交易状态
     - ``True``
   * - ``trade_status_mode``
     - 停牌持仓处理模式
     - ``to_cash``

整数手成交
----------

每个调仓日会按当期总资产把目标权重转换为股数，并用 ``np.trunc(目标股数 / lot_size)`` 向零截断为整数手。未能成交的资金转入 ``cash``。因此：

* ``engine.actual_weight`` 才是用于日频持仓和净值计算的权重。
* ``engine.position`` 中股票列均为 ``lot_size`` 的整数倍。
* ``engine.amount`` 是各调仓日分配前总资产，由上一期实际持仓收益逐期递推。
* 资金太小买不起一手时，目标标的会被置 0 并保留现金。

交易状态处理
------------

回测会查询 ``trade_status`` 表并在换仓前处理停牌持仓。``Datafeed.query_trade_status`` 在 Python 端还原三态：``1`` 正常、``0`` 停牌、``-1`` 未上市。

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 模式
     - 行为
   * - ``to_cash``
     - 停牌标的目标权重转为现金。
   * - ``hold``
     - 停牌标的尽量沿用上一期持仓。
   * - ``redistribute``
     - 停牌权重在同侧可交易标的间再分配。
   * - ``as_normal``
     - 停牌标的按正常交易处理。
   * - ``report_only``
     - 仅记录状态，不调整权重。

关闭检查：

.. code-block:: python

   engine = BacktestBase(weights, "NoStatus", 1_000_000, check_trade_status=False)

主要产出
--------

* ``nav``：日频归一化净值。
* ``actual_weight``：整数手成交后的调仓权重，含 ``cash``。
* ``daily_position_value``：日频逐标的持仓市值，含 ``cash``。
* ``daily_pnl`` / ``daily_pnl_total``：逐标的和组合日频损益。
* ``rebalance_log``：调仓长表，含目标权重、真实权重、价格、股数、手数、市值。
* ``trade_status`` / ``trade_status_matrix``：交易状态审计数据。
* ``position``：日频持仓数量。
* ``cost_price`` / ``cost_ret``：调仓价格与调仓区间收益。

导出与评价
----------

.. code-block:: python

   from betalens.analyst import Analyst

   engine.dump_to_excel("backtest_dump.xlsx")
   Analyst.from_backtest(engine, name="DemoFactor").report(
       to_excel="report.xlsx",
       to_html="report.html",
   )

``dump_to_excel`` 会把回测产出写入多 sheet，``Analyst.from_excel`` 可以从导出的 xlsx 读回评价。

更多 API 细节请参阅 :doc:`../api/backtest`。
