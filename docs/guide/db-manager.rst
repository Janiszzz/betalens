数据库管理工具
==============

``betalens_db_manager`` 是 Betalens 唯一的数据库管理层。当前 PostgreSQL schema 已在本地
完成安装；该包负责现有结构检查、观测分区、文件导入、更新和条件删除，
``betalens.datafeed`` 只读。

统一服务
--------

当前工作区已停用 v1–v10 迁移、版本记录和旧 schema verifier。
数据库结构由人工维护；已有数据导入、冲突检查、任务记录和查询继续使用原有实现。

.. code-block:: python

   from betalens_db_manager import DatabaseManager

   manager = DatabaseManager()
   inspection = manager.inspect_schema()
   manifest_plan = manager.plan_manifest(r"D:\data\imports.yaml")
   report = manager.run_manifest(manifest_plan)

``inspect_schema()`` 只读检查现有表、字段类型和兼容视图能否查询。
它不校验迁移历史、索引定义、指标种子或数据质量，也不会自动修补结构。
``ensure_dataset(table)`` 只检查指定数据集；缺少表或字段会明确报错。

``bootstrap()``、``bootstrap_local()``、旧 ``verify()`` 和 CLI ``init/verify``
已停用，调用时返回错误，不会创建空数据库后报告成功。
``init_local.bat`` 同样退出并提示新的使用方式。

.. code-block:: console

   python -m betalens_db_manager inspect
   python -m betalens_db_manager plan --manifest D:\data\imports.yaml
   python -m betalens_db_manager import --manifest D:\data\imports.yaml

新机器需先恢复数据库备份或单独准备所需结构。导入历史观测数据时，仍由
``SchemaManager.ensure_observation_partitions()`` 为已存在的父表补充年度分区。

配置和任务记录
--------------

连接配置优先级为 CLI 参数、``BETALENS_DB_*`` 环境变量、用户本地配置和示例默认值。
GUI 顶部直接填写 host、port、dbname、user 和密码；这些值只用于当前进程，密码不会写入
本地文件。

单文件和 Manifest 导入任务统一保存到
``logs/database-manager/jobs.sqlite3``，逐文件 item、取消状态、报告路径和 resume checkpoint
也在该 SQLite 文件中。PostgreSQL 不增加 job 表。

物理结构
--------

``betalens`` schema 包含 14 张基础物理表：

* 维度：``entity_dim``、``entity_name_history``、``metric_dim``、``metric_alias``、
  ``industry_scheme_dim``、``industry_dim``。
* 事实/PIT：``market_daily_fact``、``observation_fact``、``industry_membership``、
  ``index_snapshot``、``index_constituent``、``trade_status_event``。
* 管理：``dataset_coverage``。
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

* ``数据表``：显示逻辑表、物理存储、行数和当前静态 schema 元信息；“检查所选表”仅执行只读检查。
* ``文件导入``：可选择一个或多个文件，也可选择一个文件夹。文件夹会递归扫描 CSV、
  CSV.GZ、Excel 和 Parquet；选择目标表和文件类型后先检查，再导入通过检查的文件。
* ``查询与诊断``：提供筛选查询、受限的只读 SQL 和脏数据检查，并可导出当前有限结果。
* ``联网更新``：为后续数据源 API 接入预留，当前为空。

GUI 不展示已停用的 migration 历史或旧库迁移、Manifest 编辑器或任务日志页面。``trade_status``
覆盖仅截至 ``2015-06-01`` 时仍会在数据表的警告列中标红。

仓库不附带历史行情数据。导入前请准备自己的源文件，并核对日期、代码、指标单位和可用时间。
