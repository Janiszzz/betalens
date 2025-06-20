.. _getting_started:

############
快速入门
############

本教程将通过一个完整的示例，带您了解如何使用 ``betalens`` 框架来构建、回测和分析一个简单的单因子策略。

我们将以 **“股息率因子”** 为例，执行一个经典的多空策略：买入股息率最高的股票，卖出股息率最低的股票。

整个流程包括以下几个步骤：

1.  **环境准备**: 导入所需的模块。
2.  **确定回测周期**: 设置策略的调仓日期。
3.  **构建股票池**: 在每个调仓日，获取当时市场上所有可正常交易的股票。
4.  **计算因子值**: 为股票池中的每只股票匹配其最新的股息率因子值，并进行分组。
5.  **生成策略权重**: 根据因子分组结果，生成多空组合的持仓权重。
6.  **执行策略回测**: 使用回测引擎计算策略的净值曲线。
7.  **分析策略表现**: 使用分析模块生成绩效报告。

--------------------

完整代码示例
=============

下面的 Python 脚本展示了上述所有步骤的实现。代码中的注释详细解释了每一步的功能和目的。

.. code-block:: python
   :linenos:

   # 步骤 1: 环境准备
   # 导入所有需要的模块
   import betalens.datafeed as datafeed
   import betalens.factor as factor
   import betalens.backtest as backtest
   import betalens.analyst as analyst
   import matplotlib.pyplot as plt

   # 步骤 2: 确定回测周期
   # 使用 `get_absolute_trade_days` 获取从2015到2024年每年度的最后一个交易日。
   # 这是确定策略调仓频率的第一步。
   rebalance_dates = datafeed.get_absolute_trade_days(
       "2015-04-30", "2024-04-30", "Y"
   )

   # 步骤 3: 构建股票池
   # `get_tradable_pool` 会连接数据库，查询在指定调仓日处于可交易状态的所有股票，
   # 从而避免买入停牌或退市的股票。
   print("正在获取可交易股票池...")
   trade_dates, stock_pool = factor.get_tradable_pool(rebalance_dates)

   # 步骤 4: 计算因子值并分组
   # 这是因子构建的核心。`single_factor` 函数为每个调仓日的每只股票，查询其最新的因子值，
   # 并使用 pandas.qcut 将它们分为10组。
   print("正在计算因子暴露并分组...")
   factor_metric = "股息率(报告期)"
   quantiles = {factor_metric: 10} # 将股票按因子值进行十分位分组
   labeled_pool = factor.single_factor(
       trade_dates, stock_pool, factor_metric, quantiles
   )

   # (可选) 打印因子分组的描述性统计，以检查分组是否均匀、合理
   print("因子分组描述:")
   print(factor.describe_labeled_pool(labeled_pool))

   # 步骤 5: 生成策略权重
   # `get_single_factor_weight` 根据分组结果 (`_label` 列) 生成权重。
   # 'classic-long-short' 模式会自动做多标签值最大的组，做空标签值最小的组。
   # 因此也要注意，如SMB等因子的逻辑与之相反。
   print("正在生成策略权重...")
   weight_params = {
       'factor_key': factor_metric,
       'mode': 'classic-long-short',
   }
   weights = factor.get_single_factor_weight(labeled_pool, weight_params)

   # 为回测引擎添加空的 cash 列
   weights['cash'] = 0

   # 步骤 6: 执行策略回测
   # `BacktestBase` 是回测引擎。只需传入权重和初始资金，
   # 它就会自动处理价格获取、交易模拟和净值计算。
   print("正在执行回测...")
   backtest_engine = backtest.BacktestBase(
       weight=weights,
       symbol="dividend_yield_factor",
       amount=1000000
   )

   # 步骤 7: 分析策略表现
   # `PortfolioAnalyzer` 和 `ReportExporter` 是绩效分析工具。
   # 前者计算各种指标，后者则将这些指标格式化为易读的表格。
   print("正在生成绩效报告...")
   analyzer = analyst.PortfolioAnalyzer(backtest_engine.nav)
   exporter = analyst.ReportExporter(analyzer)

   print("\n--- 分年度绩效报告 ---")
   exporter.generate_annual_report()

   # 绘制最终的净值曲线
   plt.figure(figsize=(12, 7))
   backtest_engine.nav.plot(title='股息率因子策略净值曲线')
   plt.grid(True)
   plt.show()

