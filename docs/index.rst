Betalens 文档
=============

**Betalens** 是一个面向量化研究的 Python 框架，覆盖数据入库与查询、因子分组、回测、绩效评价、事件研究、稳健性检验，以及浏览器 Dashboard。

核心能力
--------

* **Datafeed**：只读 PostgreSQL 数据访问、交易日、行业、指数成分和交易状态查询。
* **DB Manager / Infra**：独立的数据接入与管理工作台，维护同等地位的 PG/CH 目标、来源映射、审计、可信版本和完整性检查。
* **Factor**：可交易池、单/双/多因子分组、因子预处理、IC/回归/择时统计、参数挖掘。
* **Backtest**：目标权重到日频净值，整数手成交，停牌状态处理，调仓审计日志。
* **Analyst**：从回测实例生成指标表、Excel 报告和 plotly 交互 HTML。
* **Dashboard**：FastAPI + React/Vite 前端，用浏览器发现因子、配参数、跑回测、下载报告。

快速安装
--------

准备好 Python 3.10+ 和 PostgreSQL 13+ 后，在 PowerShell 执行：

.. code-block:: powershell

   git clone https://github.com/Janiszzz/betalens.git
   cd betalens
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -e ".[full]"

数据库安装、连接配置、schema 初始化和数据导入不能由上面三条命令替代。第一次使用请完整
阅读 :doc:`getting-started/installation`，并逐项执行其中的验收命令。

当前本地基础设施使用 ``betalens`` 库；三域业务参考数据从
``http://localhost:8010/integrity`` 管理。见 :doc:`guide/local-databases` 和
:doc:`guide/db-manager-reference`。下方研究示例使用旧研究查询合同，
新表不会自动生成旧兼容视图，也不能据建表成功推断回测链路已适配。

快速示例
--------

.. code-block:: python

   from betalens.datafeed import get_absolute_trade_days
   from betalens.factor.factor import (
       get_tradable_pool,
       pre_query_characteristic_data,
       single_characteristic,
       get_single_factor_weight,
   )
   from betalens.backtest import BacktestBase
   from betalens.analyst import Analyst

   days = get_absolute_trade_days("2020-04-30", "2024-04-30", "Y")
   date_ranges, code_ranges = get_tradable_pool(days)

   data = pre_query_characteristic_data(
       days,
       "股息率(报告期)",
       date_ranges=date_ranges,
       code_ranges=code_ranges,
   )
   labeled = single_characteristic(data, "股息率(报告期)", {"股息率(报告期)": 10})

   weights = get_single_factor_weight(labeled, {
       "factor_key": "股息率(报告期)",
       "mode": "classic-long-short",
   })
   weights["cash"] = 0

   engine = BacktestBase(
       weight=weights,
       symbol="Dividend",
       amount=1_000_000,
       table_name="daily_market",
   )
   Analyst.from_backtest(engine, name="Dividend").report(
       to_excel="report.xlsx",
       to_html="report.html",
   )

文档目录
--------

.. toctree::
   :maxdepth: 2
   :caption: 快速开始

   getting-started/project-overview
   getting-started/installation
   getting-started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: 用户指南

   guide/datafeed
   guide/factor
   guide/backtest
   guide/analyst
   guide/eventstudy
   guide/robust
   guide/dashboard
   guide/factor-pipeline
   guide/factor-mining
   guide/infra
   guide/db-manager-reference
   guide/local-databases

.. toctree::
   :maxdepth: 2
   :caption: API 参考

   api/datafeed
   api/factor
   api/backtest
   api/analyst
   api/eventstudy
   api/robust
   api/infra

实施与验收记录
--------------

以下记录对应各自日期的计划与验收；当前运行合同以用户指南和 API 参考为准。

.. toctree::
   :maxdepth: 1

   plans/tushare-database-iteration-20261018
   plans/db-manager-audit-remediation-20261023
   plans/db-manager-audit-remediation-20261023-results
   plans/db-manager-mapping-audit-20261024
   plans/db-manager-production-smoke-20261025

索引
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
