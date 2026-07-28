Betalens 文档
=============

**Betalens** 是一个面向量化研究的 Python 框架，覆盖数据入库与查询、因子分组、回测、绩效评价、事件研究、稳健性检验，以及浏览器 Dashboard。

核心能力
--------

* **Datafeed**：只读 PostgreSQL 数据访问、交易日、行业、指数成分和交易状态查询。
* **Database Manager**：建库、迁移、Excel/EDE/Wind 数据适配、批量写入和条件删除。
* **Factor**：可交易池、单/双/多因子分组、因子预处理、IC/回归/择时统计、参数挖掘。
* **Backtest**：目标权重到日频净值，整数手成交，停牌状态处理，调仓审计日志。
* **Analyst**：从回测实例生成指标表、Excel 报告和 plotly 交互 HTML。
* **Dashboard**：FastAPI + React/Vite 前端，用浏览器发现因子、配参数、跑回测、下载报告。

快速安装
--------

.. code-block:: powershell

   git clone https://github.com/Janiszzz/betalens.git
   cd betalens
   python -m pip install -e .
   python -m pip install -r requirements.txt

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
   guide/db-manager

.. toctree::
   :maxdepth: 2
   :caption: API 参考

   api/datafeed
   api/factor
   api/backtest
   api/analyst
   api/eventstudy
   api/robust
   api/db-manager

索引
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
