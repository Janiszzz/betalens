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

``from_backtest`` 使用 ``daily_returns``、``daily_weight`` 和实际资产金额，
并读取费用、再投余份结算、风控事件及 ``trade_cycles``。``benchmark`` 可传
基准净值 Series、基准代码或另一个回测实例。代码基准默认读取后复权收盘价，
保留开始日前一个收盘点，再按业务日期与策略日收益对齐；缺失行情不会填充。

年化天数仅计日收益观测，不计初始资金点。月度、年度和指定区间报告均保留
首日收益，包括初次建仓费用。

交易与研究口径
--------------

交易统计按完整方向持仓周期计算：从空仓建仓，到平仓或反向开仓；同方向加仓、
减仓合并到同一周期。净损益包含实际费用和再投余份结算，净收益率的分母为
累计开仓名义金额。未平仓周期单独展示，不计入交易胜率和年度交易统计。
旧文件缺少核算字段时标注 ``legacy_incomplete``，仅保留可识别的价格配对，
不生成净交易统计。

IC 使用 ``research_schedule`` 和 ``research_forward_returns`` 的显式研究区间。
止盈止损等额外成交不会改变预测期限；新回测缺少研究区间时不从成交事件推断 IC。
研究分组净值在原始调仓时等权、区间内自然漂移，不扣费用，也不应用手数和风控。

从 Excel 读回
-------------

.. code-block:: python

   from betalens.analyst import Analyst

   analyst = Analyst.from_excel("backtest_dump.xlsx", name="UploadedRun")
   analyst.to_html("report.html")

该入口用于读取 ``BacktestBase.dump_to_excel()`` 的产物，恢复实际日收益、交易周期、
研究区间和计算版本，使其与内存分析保持一致。若只有 ``nav`` sheet，收益、回撤和
风险指标仍可计算。报告的“评价口径”和“诊断”记录版本与缺失数据说明。

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

``betalens.analyst.naming`` 使用 ``security_dim`` 的当前展示名称；历史研究名称由 Datafeed 的 as_of 接口另行查询。缺失名称时展示代码。

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

* 持仓或归因指标缺失：确认回测实例包含 ``daily_weight``、``daily_position_value``、``daily_pnl``、``rebalance_log``。
* 中文名显示为原代码：数据库连接不可用或对应表缺少 ``name`` 记录，属于正常降级。
* 交互 HTML 生成失败：安装 ``plotly`` 或 ``.[viz]`` 可选依赖。

更多 API 细节请参阅 :doc:`../api/analyst`。
