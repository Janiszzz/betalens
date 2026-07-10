数据库管理工具
==============

``betalens_db_manager`` 是 Betalens 的本地数据库管理层，负责 schema 创建与核验、表预览、文件导入、冲突检查、导入记录和桌面 GUI。研究代码仍通过 ``betalens.datafeed`` 查询数据；入库和管理工作优先放到本工具。

主要对象
--------

.. code-block:: python

   from betalens_db_manager import (
       SchemaManager,
       DatabaseClient,
       QueryRequest,
       ImportJobRunner,
       ImportRecordStore,
   )

* ``SchemaManager``：创建数据库、创建表、创建索引、验证 schema。
* ``DatabaseClient``：连接数据库、表概览、schema 查询、分页查询。
* ``ImportJobRunner``：预览和执行文件导入。
* ``ImportRecordStore``：记录导入任务和日志路径。

支持表
------

当前白名单：

* ``daily_market``
* ``fundamentals``
* ``macro``
* ``factors``
* ``industry``
* ``index_universe``
* ``trade_status``

所有入库目标统一为长表字段：``datetime``、``code``、``name``、``metric``、``value``，可选 ``remark``。

Schema 管理
-----------

.. code-block:: python

   from betalens_db_manager import SchemaManager

   manager = SchemaManager()
   manager.ensure_schema(tables=["daily_market", "fundamentals"], create_database=True)
   report = manager.verify_schema()

``ensure_schema`` 可创建数据库、创建表、创建索引和注释。``verify_schema`` 用于检查本地库是否符合 Betalens 约定。

查询与预览
----------

.. code-block:: python

   from betalens_db_manager import DatabaseClient, QueryRequest

   client = DatabaseClient()
   overview = client.table_overview()
   schema = client.table_schema("daily_market")
   page = client.query_table(QueryRequest(
       table="daily_market",
       limit=100,
       filters={"code": "000001.SZ"},
   ))

查询层会校验表名白名单，并设置默认语句超时。

文件导入
--------

.. code-block:: python

   from betalens_db_manager import ImportJobRunner

   runner = ImportJobRunner()

   preview = runner.preview("行情.xlsx", import_type="ede")
   record = runner.run(
       path="行情.xlsx",
       import_type="ede",
       table="daily_market",
       mode="insert_only",
   )

支持导入类型：

* ``ede``：解析 EDE 导出宽表，生成标准长表。
* ``wind_long``：读取已是长表格式的 Wind 数据。
* ``index_universe``：整理指数成分表，写入 ``index_universe``。
* ``trade_status``：整理交易状态宽表，写入 ``trade_status`` 稀疏记录。

导入模式：

* ``insert_only``：只插入新行，遇到冲突保留原数据。
* ``upsert``：按唯一键更新已有行。

指数成分导入
------------

.. code-block:: python

   runner.run(
       path="中证800成分.xlsx",
       import_type="index_universe",
       table="index_universe",
       options={
           "index_code": "000906.SH",
           "index_name": "中证800",
           "sheet_name": "Sheet2",
       },
   )

``remark`` 会保存成分股列表；查询侧用 ``get_index_universe`` 按 PIT 生效日展开。

交易状态导入
------------

.. code-block:: python

   runner.run(
       path="交易状态.xlsx",
       import_type="trade_status",
       table="trade_status",
       options={"sheet_name": "Sheet1"},
   )

导入器会把停牌等异常状态写为 ``value=0``，并写入正常交易锚点 ``value=1``。查询侧还原为 ``1``、``0``、``-1`` 三态。

CLI 与 GUI
----------

核心逻辑在 Python 包内，可被 CLI 或 GUI 调用。GUI 依赖 ``PySide6``，用于表概览、查询、导入预览、执行导入和查看任务记录。运行前建议先安装：

.. code-block:: powershell

   python -m pip install -e ".[db,gui]"
   .\betalens-db-manager\run.bat

迁移提示
--------

``Datafeed`` 中的文件入库方法仍保留兼容入口，但会发出迁移提示。新流程优先用 ``betalens_db_manager``，让查询层和管理层职责分离。
