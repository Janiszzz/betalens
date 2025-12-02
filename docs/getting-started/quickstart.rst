10 分钟快速上手
==============

下面通过一个贯穿式示例演示 Betalens 的典型流水线：数据拉取 → 因子处理 → 权重生成 → 回测 → 绩效分析。

1. 准备调仓日与可交易标的
-------------------------

.. code-block:: python

   import pandas as pd
   from betalens.datafeed import Datafeed, get_absolute_trade_days
   from betalens.factor import get_tradable_pool

   trading_days = get_absolute_trade_days(
       start_date="2023-01-01",
       end_date="2023-06-30",
       freq="M",  # 每月末调仓
   )
   date_ranges, code_ranges = get_tradable_pool(trading_days)

`get_tradable_pool` 会对每个调仓日过滤不可交易证券，返回日期序列与候选代码列表。

2. 查询并分组单因子
-------------------

.. code-block:: python

   from betalens.factor import single_factor

   labeled_pool = single_factor(
       date_ranges=date_ranges,
       code_ranges=code_ranges,
       metric="股息率(报告期)",
       quantiles={"股息率(报告期)": 5},
   )

`single_factor` 会按 ``quantiles`` 配置打标签，结果以 ``input_ts`` 与代码为 MultiIndex。

3. 派生多空权重
---------------

.. code-block:: python

   from betalens.factor import get_single_factor_weight

   weights = get_single_factor_weight(
       labeled_pool,
       params={
           "factor_key": "股息率(报告期)",
           "mode": "classic-long-short",
           "long": [4],
           "short": [0],
       },
   )
   weights["cash"] = 0
   weights.index.name = "input_ts"

`weights` 现为宽表（行：调仓时间，列：证券），可直接馈送回测模块。

4. 回测组合净值
---------------

.. code-block:: python

   from betalens.backtest import BacktestBase

   engine = BacktestBase(weight=weights, symbol="DemoFactor", amount=1_000_000)
   nav = engine.nav
   nav.plot(title="DemoFactor NAV")

`BacktestBase` 会自动调用 :class:`betalens.datafeed.Datafeed` 获取收盘价，并推导每日持仓。

5. 绩效分析与报告
-----------------

.. code-block:: python

   from betalens.analyst import PortfolioAnalyzer, ReportExporter

   analyzer = PortfolioAnalyzer(nav_series=nav)
   exporter = ReportExporter(analyzer)

   print("Sharpe:", analyzer.sharpe_ratio())
   exporter.generate_annual_report()
   exporter.generate_custom_report("2023-01-01", "2023-06-30")

`ReportExporter` 支持 CLI 表格与 Excel 输出，可搭配基准序列生成超额收益分析。

6. 启用稳健性检验（可选）
-------------------------

.. code-block:: python

   import pandas as pd
   from betalens.robust import RobustTest

   fund = nav.pct_change().dropna().rename("factor_nav")
   factors = pd.concat(
       [weights.drop(columns="cash").mul(0.5).sum(axis=1)],
       axis=1
   ).rename(columns={0: "DemoFactor"})

   test = RobustTest(fund=fund, factor=factors)
   orthogonals, tvalues = test.neu()

`RobustTest` 基于 Harvey & Liu (2021) “Lucky Factors” 思路，可配合 bootstrap 过滤伪因子。

通过以上步骤即可完成端到端的样板流程。后续章节将对每个子模块的高级特性进行详细说明。


