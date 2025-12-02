.. Betalens documentation master file

Betalens 文档
=============

**Betalens** 是一个用于量化分析和回测的 Python 框架。

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

特性
----

* 📊 **因子分析** - 支持单因子/多因子分组、打标签、生成多空权重
* 📈 **数据管理** - PostgreSQL 数据库接口，支持时间序列查询
* 🔄 **回测框架** - 多资产多权重回测，自动获取价格数据
* 📋 **绩效分析** - 计算夏普比率、最大回撤等指标，生成报告
* 🧪 **稳健性检验** - 因子增量检验、Bootstrap 重采样

快速安装（逗你玩的）
--------

.. code-block:: bash

   pip install betalens

或从源码安装：

.. code-block:: bash

   git clone https://github.com/Janiszzz/betalens.git
   cd betalens
   pip install -e .

快速示例
--------

.. code-block:: python

   from betalens.datafeed import Datafeed
   from betalens.backtest import BacktestBase
   from betalens.analyst import PortfolioAnalyzer, ReportExporter

   # 获取数据
   data = Datafeed("daily_market_data")
   params = {
       'codes': ['000001.SZ'],
       'datetimes': ['2024-01-01 10:00:00'],
       'metric': "收盘价(元)",
   }
   price = data.query_nearest_before(params)

   # 回测
   bb = BacktestBase(weight=weights, symbol="", amount=1000000)

   # 绩效分析
   analyzer = PortfolioAnalyzer(bb.nav)
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
   guide/backtest
   guide/analyst
   guide/factor
   guide/robust

.. toctree::
   :maxdepth: 2
   :caption: API 参考

   api/datafeed
   api/backtest
   api/analyst
   api/factor
   api/robust

.. toctree::
   :maxdepth: 1
   :caption: 其他

   changelog

索引
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


