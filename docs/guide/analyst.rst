策略评价模块
============

``betalens.analyst`` 提供策略评价：从回测结果出发计算多维指标、生成报告、导出图表。

三层结构：

- **门面 Analyst**（推荐入口）：``from_backtest`` 直接吃回测实例，``report()`` 一键输出。
- **PortfolioAnalyzer**：持净值/持仓数据，组合指标库计算。
- **ReportExporter**：分年度/时段/基准报告（兼容旧接口）。

子模块：``metrics``（纯函数指标库）、``naming``（代码→中文名）、``plotting``（PNG + plotly 交互图）。

一键评价（推荐）
----------------

.. code-block:: python

   from betalens.backtest import BacktestBase
   from betalens.analyst import Analyst

   bt = BacktestBase(weight=weights, symbol="strategy", amount=1_000_000)

   # 从回测实例构建：自动抽取 nav/actual_weight/daily_position_value/daily_pnl/rebalance_log
   a = Analyst.from_backtest(bt, benchmark=hs300_bt, name="我的策略")

   a.report()                                      # CLI 打印全部指标分组表
   a.report(to_excel="r.xlsx", to_html="r.html")   # 同时导出 Excel + 交互 HTML

``benchmark`` 可传基准净值 Series 或另一个回测实例；缺省时不计算基准相对指标。
只有 nav 时，持仓/归因/基准类指标自动跳过。

从 Excel 构建
~~~~~~~~~~~~~

dashboard 上传场景：读回 ``bt.dump_to_excel()`` 导出的 xlsx。

.. code-block:: python

   a = Analyst.from_excel("dump.xlsx", benchmark=bench_nav, name="组合")

输出方式
~~~~~~~~

.. code-block:: python

   figs  = a.plots()              # {名称: PNG bytes}，供 st.image / 嵌入
   ifigs = a.interactive_plots()  # {名称: plotly Figure}，供 dashboard / HTML

   a.summary_df()        # 全指标明细表
   a.top_holdings_df()   # 最频繁持仓（索引已转中文名）
   a.contribution_df()   # 收益贡献分解
   a.trade_pnl_df()      # 逐笔盈亏
   a.monthly_table()     # 月度收益矩阵

   a.to_excel("r.xlsx")  # 指标汇总 + 各明细表分 sheet
   a.to_html("r.html")   # 独立 HTML（内嵌 plotly 交互图）

图名称键：``净值曲线`` / ``回撤曲线`` / ``滚动胜率`` / ``滚动夏普`` / ``月度收益``，
有持仓数据时多 ``权重堆积``，有损益数据时多 ``收益贡献``。

评价指标一览
------------

通过 ``a.an.summary()`` 或 ``summary_grouped()`` 获取，按类别分组：

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 类别
     - 指标
   * - 收益
     - 累计收益、年化收益（几何）、年化波动率、夏普比率、索提诺比率
   * - 回撤
     - 最大回撤、卡玛比率、溃疡指数、Martin比率、痛苦指数、痛苦比率、最长回撤期
   * - 风险分布
     - 下行偏差、VaR(95%)、CVaR(95%)、偏度、峰度
   * - 交易持仓
     - 单边换手率(年化)、平均单边换手、平均持仓数、平均持仓寿命、权重HHI、前5集中度
   * - 基准相对
     - Beta、Alpha、跟踪误差、信息比率、相对基准胜率

指标库 metrics（纯函数）
------------------------

可独立调用，输入 ``nav`` / ``returns`` / ``weight`` / ``daily_pnl`` / ``rebalance_log``。

.. code-block:: python

   from betalens.analyst import metrics as M

   M.ulcer_index(nav)             # 溃疡指数
   M.sortino_ratio(returns)       # 索提诺
   M.value_at_risk(returns, 0.05) # VaR
   M.max_drawdown_duration(nav)   # 最长回撤期（天）
   M.turnover(weight)             # 换手率 dict: per_period/avg_oneway/avg_twoway/annualized
   M.top_holdings(weight, top=10) # 最频繁持仓
   M.weight_hhi(weight)           # 权重堆积 HHI（逐期）
   M.return_contribution(daily_pnl)   # 收益贡献分解
   M.trade_pnl(rebalance_log)         # 逐笔盈亏
   M.monthly_returns_table(nav)       # 月度收益矩阵

权重类函数自动剔除 ``cash`` 列。

中文名映射 naming
-----------------

.. code-block:: python

   from betalens.analyst import naming

   name_map = naming.get_name_map(["000300.SH", "000001.SZ"])  # 查库 name 列
   naming.label("000300.SH", name_map)   # → '沪深300(000300.SH)'

默认查 ``daily_market/daily_index/daily_fund/daily_bond`` 取每个代码最新名称，带缓存；
查库失败静默降级为返回原代码，不阻断分析。Analyst 的展示层自动用它把代码转为中文标签。

PortfolioAnalyzer
-----------------

.. code-block:: python

   from betalens.analyst import PortfolioAnalyzer

   analyzer = PortfolioAnalyzer(
       nav_series=nav,             # 必填
       risk_free_rate=0.0, annualizer=252, window=30,
       weight=None,                # 换手/持仓类指标需要
       daily_position_value=None,  # 权重堆积面积图需要
       daily_pnl=None,             # 收益贡献分解需要
       rebalance_log=None,         # 逐笔盈亏需要
       benchmark=None)             # 基准相对指标需要

   analyzer.summary()          # 全部标量指标 dict
   analyzer.summary_grouped()  # 按类别分组

旧用法 ``PortfolioAnalyzer(nav)`` 仍可用。兼容方法：``total_return`` / ``annualized_return``
/ ``annualized_volatility`` / ``sharpe_ratio`` / ``max_drawdown`` / ``calmar_ratio`` /
``rolling_max_drawdown`` / ``rolling_win_rate``。

.. note::

   ``annualized_return`` 已改为几何年化；``rolling_win_rate`` 已修正为窗口内胜率（落 [0,1]）。
   旧版 ``recovery_days()`` 已移除，改用 ``metrics.max_drawdown_duration(nav)``。

ReportExporter（兼容旧接口）
----------------------------

.. code-block:: python

   from betalens.analyst import ReportExporter

   exporter = ReportExporter(analyzer, benchmark_analyzer=benchmark,
                             start_date=None, end_date=None)  # 可选裁剪区间
   exporter.generate_annual_report(excel_path=None)    # 分年度
   exporter.generate_custom_report("2024-01-01", "2024-06-30")
   exporter.generate_benchmark_report()                # 需构造时传 benchmark_analyzer
   exporter.excess                                      # 超额收益分析器

Dashboard 集成
--------------

dashboard「💼 策略评价」页（``dashboard/views/portfolio.py``）：上传 ``bt.dump_to_excel``
导出的 xlsx → 指标卡 + plotly 交互图 + 明细表 tabs + Excel/HTML 下载。

依赖
----

- 核心：``matplotlib``、``prettytable``、``openpyxl``
- 交互图：``plotly``（可选，未装时 ``plot()`` 的 PNG 仍可用，``interactive_plots`` 抛 ImportError）
- 看板：``streamlit``

常见问题
--------

- ``generate_custom_report`` 提示"数据不足" → 确认 NAV 含目标日期且频率一致。
- 持仓/归因/基准指标缺失 → 检查回测是否产出 ``actual_weight``/``daily_position_value``/``daily_pnl``/``rebalance_log``，或是否传入 benchmark。
- 中文名显示为原代码 → 数据库连接不可用或该代码无 name 记录，属正常降级。

更多 API 细节请参阅 :doc:`../api/analyst`。
