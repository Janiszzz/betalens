绩效分析模块
============

`betalens.analyst` 提供从净值序列出发的指标计算与报告导出功能，适用于回测与实时组合。

PortfolioAnalyzer
-----------------

.. code-block:: python

   import pandas as pd
   from betalens.analyst import PortfolioAnalyzer

   nav = pd.Series(..., index=pd.date_range("2023-01-01", periods=250))
   analyzer = PortfolioAnalyzer(nav, risk_free_rate=0.02, annualizer=252)

   print(analyzer.total_return())
   print(analyzer.annualized_return())
   print(analyzer.sharpe_ratio())
   print(analyzer.max_drawdown())

关键属性

- ``nav``：净值序列（按日期排序）
- ``returns``：日度收益（pct_change 后去 NA）
- ``risk_free_rate``：年化无风险利率
- ``window``：滚动统计窗口（用于 ``rolling_max_drawdown`` / ``rolling_win_rate``）

ReportExporter
--------------

.. code-block:: python

   from betalens.analyst import ReportExporter

   exporter = ReportExporter(analyzer)
   exporter.generate_annual_report()                 # 分年度指标
   exporter.generate_custom_report("2024-01-01", "2024-06-30")

可选参数

- ``benchmark_analyzer``：若提供，将输出基准与超额对比，并暴露 ``exporter.excess`` 分析器
- ``start_date`` / ``end_date``：在构造函数中裁剪分析区间

输出方式

- CLI 表格：基于 PrettyTable，美观易读
- Excel：``generate_*`` 方法传入 ``excel_path`` 即可将结果写入多工作表

高级用法
--------

- **滚动指标**：``PortfolioAnalyzer.rolling_max_drawdown`` / ``rolling_win_rate`` 输出新的 Series，可直接与 NAV 绘图
- **联动报告**：通过 ``benchmark_analyzer`` 构建组合 vs. 指数的对比，同时取得 ``exporter.excess`` 做超额回测
- **自定义指标**：继承 ``PortfolioAnalyzer`` 添加新方法，再交给 ``ReportExporter``，即可自动加入表格

常见问题
--------

- 若 ``generate_custom_report`` 提示「数据不足」，请确认 NAV 序列包含目标日期，并保持频率一致。
- ``recovery_days`` 默认寻找首次回到历史高点的日期，若 NAV 从未恢复，将抛出索引错误，可自行捕获。

更多 API 细节请参阅 :doc:`../api/analyst`。


