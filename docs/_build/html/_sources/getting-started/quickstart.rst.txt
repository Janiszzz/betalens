10 分钟快速上手
==============

本节演示 Betalens 的标准流水线：调仓日 → 可交易池 → 因子预查询 → 分组 → 权重 → 回测 → 报告。

1. 准备调仓日与可交易池
-----------------------

.. code-block:: python

   from betalens.datafeed import get_absolute_trade_days
   from betalens.factor.factor import get_tradable_pool

   days = get_absolute_trade_days("2020-04-30", "2024-04-30", "Y")
   date_ranges, code_ranges = get_tradable_pool(days)

``get_tradable_pool`` 基于 ``trade_status`` 表筛选正常交易证券。默认 ``include_abnormal=False``，只纳入状态为 ``1`` 的证券；传入 ``include_abnormal=True`` 时可把停牌等异常状态也纳入候选池。

2. 批量预查询因子数据
---------------------

.. code-block:: python

   from betalens.factor.factor import pre_query_characteristic_data

   data = pre_query_characteristic_data(
       date_list=days,
       metric="股息率(报告期)",
       time_tolerance=24 * 2 * 365,
       table_name="fundamentals",
       date_ranges=date_ranges,
       code_ranges=code_ranges,
   )

``pre_query_characteristic_data`` 默认查询 ``fundamentals`` 表。``time_tolerance`` 的单位是小时，默认值 ``24*2*365`` 表示最多回看约两年。

3. 单因子分组
-------------

.. code-block:: python

   from betalens.factor.factor import single_characteristic, describe_labeled_pool

   labeled = single_characteristic(
       pre_queried_data=data,
       metric="股息率(报告期)",
       quantiles={"股息率(报告期)": 10},
   )
   print(describe_labeled_pool(labeled))

分组输出以 ``(input_ts, code)`` 为 MultiIndex，并新增 ``股息率(报告期)_label`` 标签列。

4. 生成权重
-----------

.. code-block:: python

   from betalens.factor.factor import get_single_factor_weight

   weights = get_single_factor_weight(labeled, {
       "factor_key": "股息率(报告期)",
       "mode": "classic-long-short",
   })
   weights["cash"] = 0

权重矩阵要求行是调仓时间、列是证券代码。建议显式补 ``cash`` 列；多头合计 1，空头合计 -1。

5. 回测
-------

.. code-block:: python

   from betalens.backtest import BacktestBase

   engine = BacktestBase(
       weight=weights,
       symbol="DemoFactor",
       amount=1_000_000,
       table_name="daily_market",
       metric="收盘价(元)",
       time_tolerance=24,
   )
   print(engine.nav.tail())

回测按整数手成交。输入的 ``weight`` 是目标权重，真正参与净值和持仓计算的是 ``engine.actual_weight``。

6. 绩效分析
-----------

.. code-block:: python

   from betalens.analyst import Analyst

   analyst = Analyst.from_backtest(engine, name="DemoFactor")
   analyst.report(to_excel="report.xlsx", to_html="report.html")

``Analyst.from_backtest`` 会自动读取回测实例的 ``nav``、``actual_weight``、``daily_position_value``、``daily_pnl`` 和 ``rebalance_log``。只有净值时也能生成收益/回撤/风险指标，持仓和归因类指标会自动跳过。

7. 双因子与多因子分组
---------------------

.. code-block:: python

   from betalens.factor.factor import (
       double_characteristic,
       get_double_factor_weight,
       multi_characteristic,
       get_multi_factor_weight,
   )

   size = pre_query_characteristic_data(days, "市值", date_ranges=date_ranges, code_ranges=code_ranges)
   bm = pre_query_characteristic_data(days, "账面市值比", date_ranges=date_ranges, code_ranges=code_ranges)

   double_labeled = double_characteristic(
       size,
       bm,
       metric1="市值",
       metric2="账面市值比",
       quantiles1={"市值": 5},
       quantiles2={"账面市值比": 5},
       sort_method="dependent",
   )
   double_weights = get_double_factor_weight(double_labeled, {
       "factor_key1": "市值",
       "factor_key2": "账面市值比",
       "mode": "freeplay",
       "long_combinations": [(0, 4)],
       "short_combinations": [(4, 0)],
   })
   double_weights["cash"] = 0

   multi_labeled = multi_characteristic(
       [size, bm],
       [
           {"name": "市值", "quantiles": 5, "method": "dependent"},
           {"name": "账面市值比", "quantiles": 5, "method": "dependent"},
       ],
   )
   multi_weights = get_multi_factor_weight(multi_labeled, {
       "mode": "freeplay",
       "long_combinations": [(0, 4)],
       "short_combinations": [(4, 0)],
   })
   multi_weights["cash"] = 0

后续章节会分别展开数据、因子、回测、评价、Dashboard 和参数挖掘。
