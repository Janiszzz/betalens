Backtest API
============

.. automodule:: betalens.backtest
   :members:
   :undoc-members:
   :show-inheritance:

BacktestBase
------------

.. code-block:: text

   BacktestBase(
       weight, symbol, amount, ftc=0.0, ptc=0.0, verbose=True,
       metric="close", table_name="market_daily_post_adjusted",
       check_trade_status=True, trade_status_mode="to_cash",
       trade_status_table="stock_suspension_intervals_versions", *,
       end_date=None, price_time_map=None, risk=None,
   )

目标权重驱动的统一回测。股票数量按板块规则确定，收益使用后复权价格，
成交股数及费用使用原始价格。``daily_weight`` 为日末漂移权重，
``actual_weight`` 为成交事件后的实际权重。``position`` 为分红再投等效股数。

``RiskConfig`` 的 ``stop_loss/take_profit`` 默认均为空，``execution_metric`` 默认 open。
每日收盘以本轮首次实际建仓后复权价格检查，触发后只匹配更晚的成交时点。

``time_tolerance``、``lot_size`` 和 ``preloaded_*`` 已删除。
参见 :doc:`../guide/backtest` 的时间、费用、分红与风控约定。

.. automodule:: betalens.backtest.risk
   :members:
