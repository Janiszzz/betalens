.. Betalens documentation master file

Betalens 文档
=============

**Betalens** 是一个用于量化因子分析和回测的 Python 框架。

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

特性
----

* 📊 **因子分析** - 支持单因子/双因子/多因子分组、打标签、生成多空权重
* 📈 **数据管理** - PostgreSQL 数据库接口，支持时间序列查询、Wind数据抓取
* 🔄 **回测框架** - 多资产多权重回测，自动获取价格数据，详细的异常处理
* 📋 **绩效分析** - 计算夏普比率、最大回撤等指标，分年度/自定义时段报告
* 🧪 **稳健性检验** - 基于Lucky Factors的因子增量检验、Bootstrap重采样

快速安装
--------

.. code-block:: bash

   pip install betalens

或从源码安装：

.. code-block:: bash

   git clone https://github.com/Janiszzz/betalens.git
   cd betalens/gitworks
   pip install -e .

快速示例
--------

.. code-block:: python

   from betalens.datafeed import Datafeed, get_absolute_trade_days
   from betalens.factor.factor import (
       get_tradable_pool, pre_query_characteristic_data,
       single_factor, get_single_factor_weight
   )
   from betalens.backtest import BacktestBase
   from betalens.analyst import PortfolioAnalyzer, ReportExporter

   # 1. 准备数据
   trading_days = get_absolute_trade_days("2020-04-30", "2024-04-30", "Y")
   date_ranges, code_ranges = get_tradable_pool(trading_days)

   # 2. 查询因子并分组
   data = pre_query_characteristic_data(trading_days, "股息率(报告期)",
                                date_ranges=date_ranges, code_ranges=code_ranges)
   labeled_pool = single_characteristic(data, "股息率(报告期)", {"股息率(报告期)": 10})

   # 3. 生成权重
   weights = get_single_factor_weight(labeled_pool, {
       "factor_key": "股息率(报告期)",
       "mode": "classic-long-short"
   })
   weights["cash"] = 0

   # 4. 回测
   engine = BacktestBase(weight=weights, symbol="Dividend", amount=1_000_000)

   # 5. 绩效分析
   analyzer = PortfolioAnalyzer(engine.nav)
   print(f"Sharpe: {analyzer.sharpe_ratio():.4f}")
   print(f"Max Drawdown: {analyzer.max_drawdown():.2%}")

   exporter = ReportExporter(analyzer)
   exporter.generate_annual_report()

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
   guide/robust

.. toctree::
   :maxdepth: 2
   :caption: API 参考

   api/datafeed
   api/factor
   api/backtest
   api/analyst
   api/robust

索引
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


