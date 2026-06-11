Analyst API
===========

.. automodule:: betalens.analyst
   :members:
   :undoc-members:
   :show-inheritance:

analyst.analyst
---------------

Analyst（门面）
~~~~~~~~~~~~~~~

.. py:class:: Analyst(analyzer, name='组合')

   策略评价一键门面。

   :param analyzer: PortfolioAnalyzer 实例
   :param name: 组合名称（报告标题用）

   .. py:classmethod:: from_backtest(bt, benchmark=None, risk_free_rate=0.0, annualizer=252, window=30, name='组合')

      从回测实例构建。自动抽取 ``nav`` / ``actual_weight`` /
      ``daily_position_value`` / ``daily_pnl`` / ``rebalance_log``。

      :param bt: BacktestBase 实例（须已完成回测）
      :param benchmark: 基准 nav Series 或另一个 bt 实例

   .. py:classmethod:: from_excel(filepath, benchmark=None, name='组合', **kwargs)

      从 ``bt.dump_to_excel`` 导出的 xlsx 读回构建（需含 nav sheet）。

   .. py:method:: report(to_excel=None, to_html=None, show_plots=False)

      一键报告：CLI 打印 + 可选导出 Excel / HTML。返回 ``summary()`` dict。

   .. py:method:: print_report()

      命令行打印全部指标分组表（PrettyTable）。

   .. py:method:: plots()

      返回 ``{名称: PNG bytes}``，供 ``st.image`` / 嵌入。

   .. py:method:: interactive_plots()

      返回 ``{名称: plotly Figure}``，供 dashboard / HTML。

   .. py:method:: to_excel(filepath)

      导出 Excel：指标汇总 + 各明细表分 sheet。

   .. py:method:: to_html(filepath)

      导出独立 HTML 报告（内嵌 plotly 交互图）。

   .. py:method:: summary_df()
   .. py:method:: top_holdings_df(top=10)
   .. py:method:: contribution_df(top=15)
   .. py:method:: trade_pnl_df()
   .. py:method:: monthly_table()

      各类明细表（DataFrame），代码列已转中文名标签。

   .. py:attribute:: name_map

      代码→中文名 dict（懒加载查库）。

PortfolioAnalyzer
~~~~~~~~~~~~~~~~~

.. py:class:: PortfolioAnalyzer(nav_series, risk_free_rate=0.0, annualizer=252, window=30, weight=None, daily_position_value=None, daily_pnl=None, rebalance_log=None, benchmark=None)

   投资组合分析器。

   :param nav_series: 净值序列（pd.Series，日期索引）
   :param risk_free_rate: 年化无风险利率
   :param annualizer: 年化因子（默认252个交易日）
   :param window: 滚动统计窗口
   :param weight: 调仓权重 DataFrame（换手/持仓类指标需要）
   :param daily_position_value: 日频持仓金额 DataFrame
   :param daily_pnl: 日频损益 DataFrame
   :param rebalance_log: 调仓记录 DataFrame
   :param benchmark: 基准 nav Series

   .. py:method:: summary()

      返回全部标量指标的扁平 dict。

   .. py:method:: summary_grouped()

      按 收益/回撤/风险分布/交易持仓/基准相对 分组的 dict。

   **兼容旧接口方法**

   ``total_return`` / ``annualized_return``（几何）/ ``annualized_volatility`` /
   ``sharpe_ratio`` / ``max_drawdown`` / ``calmar_ratio`` /
   ``rolling_max_drawdown`` / ``rolling_win_rate``（已修正为窗口内胜率）。

ReportExporter
~~~~~~~~~~~~~~

.. py:class:: ReportExporter(analyzer, benchmark_analyzer=None, start_date=None, end_date=None)

   绩效报告导出器（兼容旧接口）。

   :param analyzer: PortfolioAnalyzer实例
   :param benchmark_analyzer: 基准分析器实例（可选）
   :param start_date: 分析区间开始日期（可选）
   :param end_date: 分析区间结束日期（可选）

   .. py:attribute:: excess

      超额收益分析器（提供基准时）。

   .. py:method:: generate_annual_report(excel_path=None)

      生成分年度绩效报告。

   .. py:method:: generate_custom_report(start_date, end_date, excel_path=None)

      生成指定时段绩效报告。

   .. py:method:: generate_benchmark_report(excel_path=None)

      生成基准对比报告。需构造时提供 benchmark_analyzer。

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
