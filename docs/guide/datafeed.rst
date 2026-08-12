数据模块
========

``betalens.datafeed`` 是研究运行时的只读数据访问层，负责连接 PostgreSQL、查询时间序列、读取交易日、处理行业和指数成分。建库、迁移、文件解析、批量写入和 GUI 管理由 ``betalens_db_manager`` 承担，详见 :doc:`db-manager`。

配置
----

配置模板位于 ``betalens/datafeed/config.example.json``。仓库内本地开发可复制为已忽略的
``config.local.json`` 后修改数据库信息：

.. code-block:: powershell

   Copy-Item betalens\datafeed\config.example.json betalens\datafeed\config.local.json

模板和 Database Manager 的默认数据库名均为 ``datafeed``。首次安装请先完成
:doc:`../getting-started/installation` 中的 PostgreSQL 与 schema 初始化步骤。

配置优先级：

1. 运行时传入的 ``db_config``。
2. ``BETALENS_DB_*`` 环境变量。
3. ``BETALENS_CONFIG`` 指定的配置文件。
4. 用户配置 ``%APPDATA%\betalens\config.json``。
5. 仓库本地配置 ``betalens/datafeed/config.local.json``。
6. 旧的 ``betalens/datafeed/config.json``。
7. 内置默认值。

.. code-block:: python

   from betalens.datafeed.config import get_config

   config = get_config()
   dbname = config.get("database.dbname")
   config.set("logging.log_dir", "./logs")
   config.save()

Datafeed 查询
--------------

.. code-block:: python

   from betalens.datafeed import Datafeed

   data = Datafeed("daily_market")
   prices = data.query_time_range(
       codes=["000001.SZ"],
       start_date="2024-01-01",
       end_date="2024-01-31",
       metric="收盘价(元)",
   )
   data.close()

常用表名：

* ``daily_market``：股票日行情。
* ``daily_index``：指数日行情。
* ``daily_fund``：基金日行情。
* ``daily_bond``：债券日行情。
* ``fundamentals``：基本面和财务指标。
* ``industry``：行业分类。
* ``index_universe``：指数成分。
* ``trade_status``：交易状态。

最近时点查询
------------

.. code-block:: python

   data = Datafeed("fundamentals")
   df = data.query_nearest_before({
       "codes": ["000001.SZ", "000002.SZ"],
       "datetimes": ["2024-04-30 15:00:01"],
       "metric": "股息率(报告期)",
       "time_tolerance": 24 * 365,
   })

``query_nearest_before`` 和 ``query_nearest_after`` 用于 PIT 查询。``time_tolerance`` 单位是小时。

交易日
------

.. code-block:: python

   from betalens.datafeed import get_absolute_trade_days, trade_days_offset

   month_ends = get_absolute_trade_days("2024-01-01", "2024-12-31", "M")
   nib_days = get_absolute_trade_days("2024-01-01", "2024-12-31", "D", exchange="NIB")
   next_day = trade_days_offset("2024-01-31", 1, period="D")

``get_absolute_trade_days`` 仅从本地 ``trade_calendar`` 数据集读取，不访问在线服务；
``exchange`` 默认 ``SHSE``。``period`` 支持 ``D``、``W``、``M``、``Q``、``S``、``Y``，
非日频返回每个周期最后一个交易日。首次使用前，需用数据库管理器导入交易日历。

行业
----

.. code-block:: python

   from betalens.datafeed import query_industry, get_industry_members

   industry = query_industry(
       cursor,
       codes=["000001.SZ"],
       date="2024-06-30",
       scheme="申万一级行业",
   )
   members = get_industry_members(cursor, "银行", "2024-06-30")

行业查询采用 point-in-time 口径：取查询日之前最近生效记录，避免前视。

指数成分
--------

``index_universe`` 复用长格式时序表，每个生效日写入一组指数成分，直到下次调整。

.. code-block:: python

   from betalens.datafeed import (
       get_index_universe,
       get_index_universe_date,
       get_index_universe_panel,
   )

   codes = get_index_universe(cursor, "000906.SH", "2024-03-01")
   effective_date = get_index_universe_date(cursor, "000906.SH", "2024-03-01")
   panel = get_index_universe_panel(
       cursor,
       "000906.SH",
       ["2024-03-01", "2024-03-04"],
   )

``get_index_universe_panel`` 返回 ``{date: set[str]}``。规范化数据库使用单次批量查询；旧表结构自动逐日回退。也可通过 ``Datafeed("index_universe").get_index_universe_panel(...)`` 调用。

入库建议使用数据库管理工具：

.. code-block:: python

   from betalens_db_manager import ImportJobRunner

   runner = ImportJobRunner()
   record = runner.run(
       path="中证800成分.xlsx",
       import_type="index_universe",
       table="index_universe",
       options={"index_code": "000906.SH", "index_name": "中证800"},
       mode="insert_only",
   )

交易状态
--------

``trade_status`` 稀疏存储异常状态和首次正常交易日锚点。查询时在 Python 端还原：

* ``1``：正常交易。
* ``0``：停牌等异常状态。
* ``-1``：未上市或尚无正常交易锚点。

.. code-block:: python

   ts = Datafeed("trade_status")
   status = ts.query_trade_status({
       "codes": ["000001.SZ", "000002.SZ"],
       "dates": ["2024-01-31", "2024-02-29"],
   })

回测和 ``get_tradable_pool`` 都依赖该表。停牌持仓在 ``BacktestBase`` 中由 ``trade_status_mode`` 控制。

文件与 Wind 入库
----------------

``Datafeed`` 不提供文件解析或写入方法。Excel/CSV、EDE 和 Wind 适配器位于 ``betalens_db_manager.adapters``，文件任务由数据库管理器统一执行预览、校验、冲突检查和写入。

.. code-block:: python

   from betalens_db_manager import ImportJobRunner

   runner = ImportJobRunner()
   preview = runner.preview("data.xlsx", import_type="ede")
   record = runner.run("data.xlsx", import_type="ede", table="daily_market")

在线 Wind 数据先转换为标准六列长表，再交给写入器：

.. code-block:: python

   from betalens_db_manager import DatabaseWriter
   from betalens_db_manager.adapters import fetch_daily_market

   frame = fetch_daily_market(["000001.SZ"], "2024-01-01", "2024-01-31")
   result = DatabaseWriter().write("daily_market", frame, mode="upsert")

更多 API 细节请参阅 :doc:`../api/datafeed`，数据库管理流程见 :doc:`db-manager`。
