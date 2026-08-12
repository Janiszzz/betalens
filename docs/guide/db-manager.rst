数据库管理工具
==============

``betalens_db_manager`` 是 Betalens 唯一的数据库管理层。建库、migration、分区、索引、
兼容视图、文件导入、更新和条件删除都由该包负责；``betalens.datafeed`` 只读。

统一服务
--------

Python、CLI、BAT 和 PySide6 GUI 最终调用同一个门面：

.. code-block:: python

   from betalens_db_manager import DatabaseManager

   manager = DatabaseManager()
   schema_plan = manager.plan()
   manifest_plan = manager.plan_manifest(r"D:\data\imports.yaml")
   report = manager.bootstrap(
       create_database_if_missing=True,
       create_compat_views=True,
       verify=True,
       manifest=manifest_plan,
   )

``SchemaManager``、``DatabaseClient``、``DatabaseWriter``、``ImportJobRunner``、
``ImportRecordStore`` 和 ``QueryRequest`` 继续作为兼容 API。

手工一键初始化
--------------

先执行只读计划，再由用户手工初始化真实 ``datafeed`` 库：

.. code-block:: powershell

   python -m betalens_db_manager plan
   .\betalens_db_manager\init_local.bat

等价命令及独立核验：

.. code-block:: powershell

   python -m betalens_db_manager init --yes
   python -m betalens_db_manager verify --deep

``init_local.bat`` 优先使用仓库 ``.venv``，否则使用当前 ``python``，不会下载依赖。
初始化可重复执行；migration checksum 使用规范化 LF，历史 ``0001`` 至 ``0008`` 兼容
LF/CRLF checksum。``0009`` 的早期阻断式历史审计 checksum 也被接受，以便已部署数据库
平滑升级；其余新版 migration 只接受规范 checksum，且禁止 schema 降级。

初始化使用不可变的版本化 schema 契约，并在提交前后执行核验。报告区分
``failed_before_commit``、``committed_verification_failed``、``completed_with_errors`` 和
``completed``。这些内部实现不会出现在面向新手的 GUI 中。
旧表等价审计发现差异时会记录警告，但不会再回滚完整的新 schema 建立；旧表仍保留在
``betalens_legacy`` 供后续人工核对。

配置和任务记录
--------------

连接配置优先级为 CLI 参数、``BETALENS_DB_*`` 环境变量、用户本地配置和示例默认值。
GUI 顶部直接填写 host、port、dbname、user 和密码；这些值只用于当前进程，密码不会写入
本地文件。

schema、单文件和 Manifest 任务统一保存到
``logs/database-manager/jobs.sqlite3``，逐文件 item、取消状态、报告路径和 resume checkpoint
也在该 SQLite 文件中。PostgreSQL 不增加 job 表。

物理结构
--------

``betalens`` schema 包含 15 张基础物理表：

* 维度：``entity_dim``、``entity_name_history``、``metric_dim``、``metric_alias``、
  ``industry_scheme_dim``、``industry_dim``。
* 事实/PIT：``market_daily_fact``、``observation_fact``、``industry_membership``、
  ``index_snapshot``、``index_constituent``、``trade_status_event``。
* 管理：``schema_migration``、``dataset_coverage``。
* 日历：``trade_calendar_day``，以 ``(exchange, trade_date)`` 保存每个交易所的交易日序列。

``market_daily_fact`` 以 ``(entity_id, trade_date)`` 为主键且不分区；
``observation_fact`` 按 ``available_at`` 年度分区。``public`` 下提供十一个只读兼容视图：
``daily_market``、``daily_index``、``daily_fund``、``daily_bond``、``fundamentals``、
``macro``、``factors``、``industry``、``index_universe``、``trade_status`` 和
``trade_calendar``。

Manifest 批量导入
-----------------

Manifest 当前版本是 ``version: 1``。路径相对清单目录解析；文件和 glob 结果稳定排序；
同一文件被两个任务匹配会在任何写入前报错。文件名只给 adapter 建议，不能决定导入类型。

.. code-block:: yaml

   version: 1
   defaults:
     mode: insert_only
     on_rejected: fail
     options:
       chunk_rows: 100000

   imports:
     - id: daily-market-folder
       path: market/**/*.csv
       target: daily_market
       adapter: standard_long

     - id: roe-ede
       path: fundamentals/roe.xlsx
       target: fundamentals
       adapter: ede
       mode: upsert
       options:
         date_from: filename

导入整个文件夹时使用 ``*`` 或 ``**`` glob；不同结构必须拆成不同的
``target + adapter`` 清单项。旧 ``table/import_type`` 和 ``wind_long`` 名称仍接受。

运行前会解析所有文件、校验 adapter/options/target、计算 SHA-256，并生成 preview token：

.. code-block:: powershell

   python -m betalens_db_manager plan --manifest D:\data\imports.yaml
   python -m betalens_db_manager import --manifest D:\data\imports.yaml --resume
   python -m betalens_db_manager init --yes --manifest D:\data\imports.yaml

每个文件只创建一次 staging；CSV/CSV.GZ/Parquet 默认每十万行分块 COPY，Excel 整表读取。
数据库内统一完成去重、跨 chunk 冲突检查、维度解析、insert/upsert 和有限冲突采样。
一个文件失败会回滚该文件并继续后续文件；存在失败时总状态为
``completed_with_errors`` 且 CLI 返回非零。取消会回滚当前文件，保留之前成功文件。

``remark`` 必须是 JSON object。无效日期、空键、非有限数值和无法转换的结构字段进入
``*.rejected.csv``，其中包含 source_file、source_row、field、raw_value 和 reason。

适配器
------

GUI 默认使用 ``auto``：六列文件走 ``standard_long``，日期/代码/名称宽表走
``wind_wide``，其余普通行情或观测宽表按 ``ede`` 规则预处理。正式 adapter 还包括
``standard_long``、``wind_wide``、``ede``、``industry``、``index_universe``、
``trade_status`` 和 ``trade_calendar``；``wind_long`` 是 ``wind_wide`` 的兼容别名。CSV/XLSX/Parquet 可通过
``options.column_map`` 显式映射输入列。EDE 和行情 adapter 会按
逻辑目标及核心指标可用时间对齐：核心 OHLCV 写入 ``market_daily_fact``，扩展指标写入
``observation_fact``。

``trade_calendar`` 接受宽表：首行每个非空列标题为交易所代码，列值为交易日。例如
``SHSE``、``NIB`` 两列的 Excel 可直接通过 GUI 选择 ``trade_calendar``，或在 Manifest 中
使用 ``target: trade_calendar`` 和 ``adapter: trade_calendar`` 导入。重复
``(exchange, trade_date)`` 会幂等跳过。

查询、写入和删除
----------------

.. code-block:: python

   from betalens_db_manager import DatabaseClient, DatabaseWriter, DeleteRequest, QueryRequest

   client = DatabaseClient()
   page = client.query_table(QueryRequest(
       table="daily_market",
       code="000001.SZ",
       metric="收盘价(元)",
       start_date="2024-01-01",
       end_date="2024-12-31",
       limit=500,
   ))

   writer = DatabaseWriter(client)
   result = writer.write("daily_market", frame, mode="upsert")
   deleted = writer.delete(DeleteRequest(
       table="daily_market", codes=["000001.SZ"],
       metric="收盘价(元)", start_date="2024-01-01", end_date="2024-12-31",
   ))

``DeleteRequest`` 必须包含代码、指标或日期条件，禁止无条件删除。
``QueryRequest.page_token`` 用于稳定 keyset 翻页。

GUI
---

.. code-block:: powershell

   python -m pip install -e ".[db,gui]"
   python -m betalens_db_manager

GUI 离线启动，不因目标库尚未创建而弹异常。界面只有四个标签页：

* ``数据表``：显示逻辑表、物理存储、行数和契约元信息；选择任意表后可一键补齐其所需
  的 Betalens 数据库结构，不删除已有数据。
* ``文件导入``：可选择一个或多个文件，也可选择一个文件夹。文件夹会递归扫描 CSV、
  CSV.GZ、Excel 和 Parquet；选择目标表和文件类型后先检查，再导入通过检查的文件。
* ``查询与诊断``：提供筛选查询、受限的只读 SQL 和脏数据检查，并可导出当前有限结果。
* ``联网更新``：为后续数据源 API 接入预留，当前为空。

GUI 不展示 migration 历史、旧库迁移、Manifest 编辑器或任务日志页面。``trade_status``
覆盖仅截至 ``2015-06-01`` 时仍会在数据表的警告列中标红。

仓库没有已删除 ``daily_market`` 的全量历史源。未提供 Manifest 时，一键初始化只复现完整
空 schema，并报告缺少行情数据。
