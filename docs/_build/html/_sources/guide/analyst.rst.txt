绩效分析模块
============

`betalens.analyst` 提供从净值序列出发的指标计算与报告导出功能，适用于回测与实时组合。

PortfolioAnalyzer
-----------------

.. code-block:: python

   import pandas as pd
   from betalens.analyst import PortfolioAnalyzer

   nav = pd.Series(..., index=pd.date_range("2023-01-01", periods=250))
   analyzer = PortfolioAnalyzer(
       nav_series=nav,
       risk_free_rate=0.02,  # 年化无风险利率
       annualizer=252,       # 年化因子
       window=30             # 滚动窗口
   )

核心指标
~~~~~~~~

.. code-block:: python

   # 累计收益率
   print(analyzer.total_return())

   # 年化收益率
   print(analyzer.annualized_return())

   # 年化波动率
   print(analyzer.annualized_volatility())

   # 夏普比率
   print(analyzer.sharpe_ratio())

   # 最大回撤
   print(analyzer.max_drawdown())

   # 卡玛比率（年化收益/最大回撤）
   print(analyzer.calmar_ratio())

   # 恢复天数
   print(analyzer.recovery_days())

滚动指标
~~~~~~~~

.. code-block:: python

   # 滚动最大回撤
   rolling_mdd = analyzer.rolling_max_drawdown()

   # 滚动胜率
   rolling_wr = analyzer.rolling_win_rate()

关键属性
~~~~~~~~

- ``nav``：净值序列（按日期排序）
- ``returns``：日度收益（pct_change 后去 NA）
- ``risk_free_rate``：年化无风险利率
- ``annualizer``：年化因子（默认252）
- ``window``：滚动统计窗口

ReportExporter
--------------

.. code-block:: python

   from betalens.analyst import ReportExporter

   exporter = ReportExporter(analyzer)

分年度报告
~~~~~~~~~~

.. code-block:: python

   exporter.generate_annual_report()
   exporter.generate_annual_report(excel_path="annual_report.xlsx")

自定义时段报告
~~~~~~~~~~~~~~

.. code-block:: python

   exporter.generate_custom_report("2024-01-01", "2024-06-30")
   exporter.generate_custom_report("2024-01-01", "2024-06-30", excel_path="custom.xlsx")

基准对比报告
~~~~~~~~~~~~

.. code-block:: python

   # 创建基准分析器
   benchmark_nav = pd.Series(...)
   benchmark = PortfolioAnalyzer(benchmark_nav)

   # 带基准的报告导出器
   exporter = ReportExporter(analyzer, benchmark_analyzer=benchmark)
   exporter.generate_benchmark_report()

   # 访问超额收益分析器
   excess_analyzer = exporter.excess
   print(f"超额收益: {excess_analyzer.total_return():.2%}")

指定分析区间
~~~~~~~~~~~~

.. code-block:: python

   # 在构造函数中裁剪分析区间
   exporter = ReportExporter(
       analyzer,
       benchmark_analyzer=benchmark,
       start_date="2023-01-01",
       end_date="2024-12-31"
   )

输出方式
~~~~~~~~

- **CLI 表格**：基于 PrettyTable，美观易读
- **Excel**：传入 ``excel_path`` 即可将结果写入多工作表

高级用法
--------

滚动指标绘图
~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(2, 1, figsize=(12, 8))

   # NAV 与滚动最大回撤
   axes[0].plot(analyzer.nav, label='NAV')
   axes[0].legend()

   axes[1].plot(analyzer.rolling_max_drawdown(), label='Rolling MDD', color='red')
   axes[1].legend()

   plt.tight_layout()
   plt.show()

自定义指标
~~~~~~~~~~

.. code-block:: python

   class CustomAnalyzer(PortfolioAnalyzer):
       def sortino_ratio(self, target_return=0):
           """索提诺比率"""
           excess_returns = self.returns - self.risk_free_rate / self.annualizer
           downside_returns = excess_returns[excess_returns < target_return]
           downside_std = downside_returns.std() * np.sqrt(self.annualizer)
           return excess_returns.mean() * self.annualizer / downside_std

   analyzer = CustomAnalyzer(nav)
   print(f"Sortino: {analyzer.sortino_ratio():.4f}")

多策略对比
~~~~~~~~~~

.. code-block:: python

   strategies = {
       "Strategy A": nav_a,
       "Strategy B": nav_b,
       "Benchmark": nav_benchmark
   }

   results = []
   for name, nav in strategies.items():
       a = PortfolioAnalyzer(nav)
       results.append({
           "Strategy": name,
           "Return": a.total_return(),
           "Volatility": a.annualized_volatility(),
           "Sharpe": a.sharpe_ratio(),
           "MaxDD": a.max_drawdown()
       })

   comparison = pd.DataFrame(results)
   print(comparison)

常见问题
--------

- 若 ``generate_custom_report`` 提示「数据不足」，请确认 NAV 序列包含目标日期，并保持频率一致。
- ``recovery_days`` 默认寻找首次回到历史高点的日期，若 NAV 从未恢复，将抛出索引错误，可自行捕获。
- 基准序列会自动裁剪到与策略NAV相同的日期范围。

更多 API 细节请参阅 :doc:`../api/analyst`。


