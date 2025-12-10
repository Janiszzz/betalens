10 分钟快速上手
==============

下面通过一个贯穿式示例演示 Betalens 的典型流水线：数据拉取 → 因子处理 → 权重生成 → 回测 → 绩效分析。

1. 准备调仓日与可交易标的
-------------------------

.. code-block:: python

   from betalens.datafeed import Datafeed, get_absolute_trade_days
   from betalens.factor.factor import get_tradable_pool

   # 生成调仓日序列（每年4月30日）
   trading_days = get_absolute_trade_days("2015-04-30", "2024-04-30", "Y")
   
   # 获取可交易股票池
   date_ranges, code_ranges = get_tradable_pool(trading_days)

`get_tradable_pool` 会对每个调仓日过滤交易状态为1的证券，返回日期序列与候选代码列表。

2. 批量预查询因子数据
---------------------

.. code-block:: python

   from betalens.factor.factor import pre_query_factor_data

   # 预查询因子数据，time_tolerance为时间容差（小时）
   pre_queried_data = pre_query_factor_data(
       date_list=trading_days,
       metric="股息率(报告期)",
       time_tolerance=24*2*365,  # 2年
       table_name="fundamental_data",
       date_ranges=date_ranges,
       code_ranges=code_ranges
   )

`pre_query_factor_data` 返回格式化的DataFrame，包含 input_ts、code、因子值、datetime、diff_hours 等列。

3. 单因子分组打标签
-------------------

.. code-block:: python

   from betalens.factor.factor import single_factor, describe_labeled_pool

   # 单因子分组（分10组）
   labeled_pool = single_factor(
       pre_queried_data=pre_queried_data,
       metric="股息率(报告期)",
       quantiles={"股息率(报告期)": 10}
   )

   # 查看分组统计
   summary = describe_labeled_pool(labeled_pool)
   print(summary)

`single_factor` 按 ``quantiles`` 配置打标签，结果以 ``input_ts`` 与 ``code`` 为 MultiIndex。

4. 派生多空权重
---------------

.. code-block:: python

   from betalens.factor.factor import get_single_factor_weight

   # 经典多空模式
   weights = get_single_factor_weight(
       labeled_pool,
       params={
           "factor_key": "股息率(报告期)",
           "mode": "classic-long-short",
       }
   )

   # 或自定义多空组合（freeplay模式）
   weights = get_single_factor_weight(
       labeled_pool,
       params={
           "factor_key": "股息率(报告期)",
           "mode": "freeplay",
           "long": [9],     # 做多第9组（最高）
           "short": [0],    # 做空第0组（最低）
       }
   )
   weights["cash"] = 0

`weights` 现为宽表（行：调仓时间，列：证券），可直接馈送回测模块。

5. 回测组合净值
---------------

.. code-block:: python

   from betalens.backtest import BacktestBase

   engine = BacktestBase(weight=weights, symbol="DemoFactor", amount=1_000_000)
   nav = engine.nav
   nav.plot(title="DemoFactor NAV")

`BacktestBase` 会自动调用 :class:`betalens.datafeed.Datafeed` 获取收盘价，并推导每日持仓。

6. 绩效分析与报告
-----------------

.. code-block:: python

   from betalens.analyst import PortfolioAnalyzer, ReportExporter

   analyzer = PortfolioAnalyzer(nav_series=nav)
   exporter = ReportExporter(analyzer)

   print("Sharpe:", analyzer.sharpe_ratio())
   print("Max Drawdown:", analyzer.max_drawdown())
   exporter.generate_annual_report()
   exporter.generate_custom_report("2020-01-01", "2024-04-30")

`ReportExporter` 支持 CLI 表格与 Excel 输出，可搭配基准序列生成超额收益分析。

7. 双因子分组（Double Sort）
----------------------------

.. code-block:: python

   from betalens.factor.factor import double_factor, get_double_factor_weight

   # 预查询两个因子的数据
   data1 = pre_query_factor_data(trading_days, "市值", date_ranges=date_ranges, code_ranges=code_ranges)
   data2 = pre_query_factor_data(trading_days, "账面市值比", date_ranges=date_ranges, code_ranges=code_ranges)

   # 双因子分组（条件排序：先按市值分组，再在组内按账面市值比分组）
   labeled_pool = double_factor(
       pre_queried_data1=data1,
       pre_queried_data2=data2,
       metric1="市值",
       metric2="账面市值比",
       quantiles1={"市值": 5},
       quantiles2={"账面市值比": 5},
       sort_method='dependent'  # 或 'independent'
   )

   # 生成双因子权重
   weights = get_double_factor_weight(
       labeled_pool,
       params={
           "factor_key1": "市值",
           "factor_key2": "账面市值比",
           "mode": "freeplay",
           "long_combinations": [(0, 4), (1, 4)],   # 小市值+高BM
           "short_combinations": [(4, 0), (4, 1)],  # 大市值+低BM
       }
   )

8. 多因子分组（Multi-Factor Sort）
----------------------------------

.. code-block:: python

   from betalens.factor.factor import multi_factor, get_multi_factor_weight

   # 预查询多个因子
   data_list = [
       pre_query_factor_data(trading_days, "市值", date_ranges=date_ranges, code_ranges=code_ranges),
       pre_query_factor_data(trading_days, "账面市值比", date_ranges=date_ranges, code_ranges=code_ranges),
       pre_query_factor_data(trading_days, "动量", date_ranges=date_ranges, code_ranges=code_ranges),
   ]

   # 多因子配置
   factors = [
       {'name': '市值', 'quantiles': 5, 'method': 'dependent'},
       {'name': '账面市值比', 'quantiles': 5, 'method': 'dependent'},
       {'name': '动量', 'quantiles': 3, 'method': 'independent'},
   ]

   labeled_pool = multi_factor(data_list, factors)

   # 生成多因子权重
   weights = get_multi_factor_weight(
       labeled_pool,
       params={
           "mode": "freeplay",
           "long_combinations": [(0, 4, 2)],   # 小市值+高BM+高动量
           "short_combinations": [(4, 0, 0)],  # 大市值+低BM+低动量
       }
   )

9. 启用稳健性检验（可选）
-------------------------

.. code-block:: python

   import pandas as pd
   from betalens.robust import RobustTest

   fund = nav.pct_change().dropna().rename("factor_nav")
   factors = pd.DataFrame(...)  # 因子收益序列

   test = RobustTest(fund=fund, factor=factors)
   orthogonals, tvalues = test.neu()
   eff_names, pvalues, pdf = test.bootstrap_once(n_bootstraps=500)

`RobustTest` 基于 Harvey & Liu (2021) "Lucky Factors" 思路，可配合 bootstrap 过滤伪因子。

通过以上步骤即可完成端到端的样板流程。后续章节将对每个子模块的高级特性进行详细说明。


