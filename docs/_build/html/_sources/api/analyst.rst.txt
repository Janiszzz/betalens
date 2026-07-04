Analyst API
===========

``betalens.analyst`` 提供策略评价门面、兼容旧接口的组合分析器、报告导出器，以及指标、命名和绘图工具。

Analyst
-------

.. py:class:: Analyst(analyzer, name='组合')

   策略评价一键门面。

   :param analyzer: ``PortfolioAnalyzer`` 实例。
   :param name: 组合名称。

   .. py:classmethod:: from_backtest(bt, benchmark=None, risk_free_rate=0.0, annualizer=252, window=30, name='组合')

      从回测实例构建，自动抽取 ``nav``、``actual_weight``、``daily_position_value``、``daily_pnl`` 和 ``rebalance_log``。

   .. py:classmethod:: from_excel(filepath, benchmark=None, name='组合', **kwargs)

      从 ``BacktestBase.dump_to_excel`` 导出的 xlsx 读回构建。

   .. py:method:: report(to_excel=None, to_html=None, show_plots=False)

      打印 CLI 报告，并可导出 Excel 和 HTML。返回指标摘要 dict。

   .. py:method:: print_report()
   .. py:method:: plots()
   .. py:method:: interactive_plots()
   .. py:method:: to_excel(filepath)
   .. py:method:: to_html(filepath)
   .. py:method:: summary_df()
   .. py:method:: top_holdings_df(top=10)
   .. py:method:: contribution_df(top=15)
   .. py:method:: trade_pnl_df()
   .. py:method:: monthly_table()

      常用输出和明细表接口。代码列会尽量映射为中文名标签。

PortfolioAnalyzer
-----------------

.. py:class:: PortfolioAnalyzer(nav_series, risk_free_rate=0.0, annualizer=252, window=30, weight=None, daily_position_value=None, daily_pnl=None, rebalance_log=None, benchmark=None)

   投资组合指标计算器。

   :param nav_series: 净值序列。
   :param risk_free_rate: 年化无风险利率。
   :param annualizer: 年化因子，默认 252。
   :param window: 滚动统计窗口。
   :param weight: 调仓权重。
   :param daily_position_value: 日频持仓金额。
   :param daily_pnl: 日频逐标的损益。
   :param rebalance_log: 调仓记录。
   :param benchmark: 基准净值。

   .. py:method:: summary()

      返回扁平指标 dict。

   .. py:method:: summary_grouped()

      返回按收益、回撤、风险、持仓、基准相对等类别分组的指标 dict。

   兼容方法包括 ``total_return``、``annualized_return``、``annualized_volatility``、``sharpe_ratio``、``max_drawdown``、``calmar_ratio``、``rolling_max_drawdown`` 和 ``rolling_win_rate``。

ReportExporter
--------------

.. py:class:: ReportExporter(analyzer, benchmark_analyzer=None, start_date=None, end_date=None)

   兼容旧接口的报告导出器。

   .. py:method:: generate_annual_report(excel_path=None)
   .. py:method:: generate_custom_report(start_date, end_date, excel_path=None)
   .. py:method:: generate_benchmark_report(excel_path=None)

analyst.analyst
---------------

.. automodule:: betalens.analyst.analyst
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

analyst.metrics
---------------

.. automodule:: betalens.analyst.metrics
   :members:
   :undoc-members:

analyst.naming
--------------

.. automodule:: betalens.analyst.naming
   :members:
   :undoc-members:

analyst.plotting
----------------

.. automodule:: betalens.analyst.plotting
   :members:
   :undoc-members:
