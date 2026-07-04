策略评价模块
============

``betalens.analyst`` 提供回测后的评价、报表和可视化。推荐入口是 ``Analyst`` 门面；``PortfolioAnalyzer`` 和 ``ReportExporter`` 保留为兼容旧用法。

推荐用法
--------

.. code-block:: python

   from betalens.analyst import Analyst

   analyst = Analyst.from_backtest(bt, benchmark=hs300_bt, name="DemoFactor")
   summary = analyst.report(
       to_excel="report.xlsx",
       to_html="report.html",
       show_plots=False,
   )

``from_backtest`` 会自动抽取 ``nav``、``actual_weight``、``daily_position_value``、``daily_pnl`` 和 ``rebalance_log``。``benchmark`` 可传基准净值 Series、基准代码或另一个回测实例。

从 Excel 读回
-------------

.. code-block:: python

   from betalens.analyst import Analyst

   analyst = Analyst.from_excel("backtest_dump.xlsx", name="UploadedRun")
   analyst.to_html("report.html")

该入口用于读取 ``BacktestBase.dump_to_excel()`` 的产物。若只有 ``nav`` sheet，收益、回撤和风险指标仍可计算，持仓、归因和交易类指标会跳过。

输出
----

.. code-block:: python

   analyst.print_report()
   png_figs = analyst.plots()
   plotly_figs = analyst.interactive_plots()
   analyst.to_excel("report.xlsx")
   analyst.to_html("report.html")

常见图表键包括 ``净值曲线``、``回撤曲线``、``滚动胜率``、``滚动夏普``、``月度收益``；有持仓数据时增加 ``权重堆积``，有损益数据时增加 ``收益贡献``。

指标分组
--------

.. code-block:: python

   grouped = analyst.an.summary_grouped()
   flat = analyst.an.summary()
   table = analyst.summary_df()

指标分为收益、回撤、风险分布、交易持仓、基准相对等类别。底层纯函数位于 ``betalens.analyst.metrics``，可直接用于自定义报告。

证券名称映射
------------

``betalens.analyst.naming`` 会从 ``daily_market``、``daily_index``、``daily_fund``、``daily_bond`` 表读取代码对应的最新 ``name``。查库失败或缺失名称时会降级为原代码，不阻断评价。

兼容旧接口
----------

.. code-block:: python

   from betalens.analyst import PortfolioAnalyzer, ReportExporter

   analyzer = PortfolioAnalyzer(bt.nav, weight=bt.actual_weight)
   exporter = ReportExporter(analyzer)
   exporter.generate_annual_report(excel_path="annual.xlsx")

旧接口适合已有脚本迁移；新代码优先使用 ``Analyst.from_backtest(...).report(...)``。

Dashboard 集成
--------------

当前 Dashboard 是 FastAPI + React/Vite 应用，位于 ``dashboard/``。后端从回测和 Analyst 对象序列化指标、图表、持仓和交易明细，前端通过分页接口读取大表，并提供 ``dump``、``report``、``html``、``profiling`` 等下载入口。详见 :doc:`dashboard`。

常见问题
--------

* 持仓或归因指标缺失：确认回测实例包含 ``actual_weight``、``daily_position_value``、``daily_pnl``、``rebalance_log``。
* 中文名显示为原代码：数据库连接不可用或对应表缺少 ``name`` 记录，属于正常降级。
* 交互 HTML 生成失败：安装 ``plotly`` 或 ``.[viz]`` 可选依赖。

更多 API 细节请参阅 :doc:`../api/analyst`。
