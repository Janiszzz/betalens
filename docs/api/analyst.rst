Analyst API
===========

.. automodule:: betalens.analyst
   :members:
   :undoc-members:
   :show-inheritance:

analyst.analyst
---------------

PortfolioAnalyzer
~~~~~~~~~~~~~~~~~

.. py:class:: PortfolioAnalyzer(nav_series, risk_free_rate=0.0, annualizer=252, window=30)

   投资组合绩效分析器。

   :param nav_series: 净值序列（pd.Series，日期索引）
   :param risk_free_rate: 年化无风险利率
   :param annualizer: 年化因子（默认252个交易日）
   :param window: 滚动统计窗口

   **属性**

   .. py:attribute:: nav
      :type: pd.Series

      净值序列（按日期排序）

   .. py:attribute:: returns
      :type: pd.Series

      日度收益序列

   **核心指标方法**

   .. py:method:: total_return()

      累计收益率。

   .. py:method:: annualized_return()

      年化收益率。

   .. py:method:: annualized_volatility()

      年化波动率。

   .. py:method:: sharpe_ratio()

      夏普比率（年化）。

   .. py:method:: max_drawdown()

      最大回撤。

   .. py:method:: calmar_ratio()

      卡玛比率（年化收益/最大回撤）。

   .. py:method:: recovery_days()

      恢复天数。

   **滚动指标方法**

   .. py:method:: rolling_max_drawdown()

      滚动最大回撤。

   .. py:method:: rolling_win_rate()

      滚动胜率。

ReportExporter
~~~~~~~~~~~~~~

.. py:class:: ReportExporter(analyzer, benchmark_analyzer=None, start_date=None, end_date=None)

   绩效报告导出器。

   :param analyzer: PortfolioAnalyzer实例
   :param benchmark_analyzer: 基准分析器实例（可选）
   :param start_date: 分析区间开始日期（可选）
   :param end_date: 分析区间结束日期（可选）

   **属性**

   .. py:attribute:: analyzer
      :type: PortfolioAnalyzer

      组合分析器

   .. py:attribute:: benchmark
      :type: PortfolioAnalyzer

      基准分析器（如果提供）

   .. py:attribute:: excess
      :type: PortfolioAnalyzer

      超额收益分析器（如果提供基准）

   **方法**

   .. py:method:: generate_annual_report(excel_path=None)

      生成分年度绩效报告。

      :param excel_path: Excel输出路径（可选）

   .. py:method:: generate_custom_report(start_date, end_date, excel_path=None)

      生成指定时段绩效报告。

      :param start_date: 开始日期
      :param end_date: 结束日期
      :param excel_path: Excel输出路径（可选）

   .. py:method:: generate_benchmark_report(excel_path=None)

      生成基准对比报告。需要在构造时提供 benchmark_analyzer。

      :param excel_path: Excel输出路径（可选）


