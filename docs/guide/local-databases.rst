本地库与运行约定
================

本页记录当前 WSL 开发机的非秘密部署信息，整理核对日期为 **2026-10-26**。
这些是本机现状，不是其他机器的自动配置，也不代表历史数据已全部补齐。
DB Manager 使用自己的环境文件，与研究层 ``BETALENS_DB_*`` 配置独立。

连接与账号现状
--------------

.. list-table:: 本机连接
   :header-rows: 1
   :widths: 23 37 40

   * - 项目
     - PostgreSQL
     - ClickHouse
   * - 地址
     - ``127.0.0.1:5432``
     - ``http://127.0.0.1:8123``
   * - 数据库
     - ``betalens``
     - ``betalens``
   * - 数据 schema
     - ``betalens``
     - 数据库直接限定表名
   * - 当前账号
     - ``betalens``，具有 superuser/createdb/createrole
     - ``betalens``，当前管理员
   * - DB Manager 写入
     - 事务写入
     - ``CLICKHOUSE_WRITER_MODE=local_admin``，本机受控写入

当前两库并未落实“迁移管理员、日常写账号、业务只读账号”的角色拆分；
该方案仍为后续工作，不能把当前账号描述成最小权限专用账号。
CH ``local_admin`` 继续使用现有管理员，执行同一受控锁、查询检查点和恢复日志；
它不能阻止其他客户端使用管理员凭据直接写库。维护时需避免旁路写入。
新环境默认写入模式仍为 ``disabled``，需按实际部署选择 ``local_dedicated`` 或显式
``local_admin``，不能从这份本机记录推导出自动切换策略。

PG 和 CH 是同等目标，任务默认双选，也可单选；不是 PG 提交后自动同步 CH 的主从模式。
每库的实际入库时间、版本序号、核验和发布结果独立。

配置、状态与目录
----------------

当前仓库位于 ``~/betalens``。下面用 ``~`` 表示运行 WSL 服务的 Linux 用户目录：

.. list-table:: 本机路径与默认路径
   :header-rows: 1
   :widths: 25 39 36

   * - 用途
     - 当前部署
     - 新环境默认
   * - 管理器 Python
     - ``~/betalens/betalens-db-manager/.venv/bin/python``
     - 安装时自行建立的独立虚拟环境
   * - 私有环境文件
     - ``~/betalens-infra/tushare.private.env``
     - ``~/.config/betalens/db-manager/credentials.env``
   * - 表定义与映射
     - ``~/.config/betalens/db-manager/``
     - 同左，可由 XDG 或环境变量覆盖
   * - 状态目录
     - ``~/betalens-infra/tushare/``
     - ``~/.local/share/betalens-db-manager/``
   * - 原始响应归档
     - ``~/betalens-infra/tushare/archive/``
     - ``<state_dir>/archive/``

历史目录名中的 ``tushare`` 只是当前磁盘路径，不代表数据库中存在同名供应商库。
表定义为 ``tables/<table_id>.yaml``，通用字段映射为 ``mappings/<table_id>.yaml``，
来源实例为 ``sources.yaml``。SQLite ``tasks.sqlite3`` 位于状态目录，保存任务、尝试、
请求参数、事件日志及审计索引。参考域缓存位于 ``reference-cache/<task_id>/audit.json``。

Windows 原生运行的默认目录为
``%LOCALAPPDATA%\Betalens\DBManager\config`` 和
``%LOCALAPPDATA%\Betalens\DBManager\state``。
Windows 浏览器访问 WSL 服务仍使用上述 Linux 路径。
秘密文件必须在 Git 工作树外，包括被忽略的目录；WSL 使用 Linux 文件系统，
目录权限 ``0700``、文件 ``0600``，Windows 使用受限 ACL。
文档和技能只保存环境变量名称及文件位置，不保存密码、API Token 或私有环境文件正文。

启动与页面
----------

已安装的本机环境使用：

.. code-block:: bash

   cd ~/betalens
   betalens-db-manager/.venv/bin/python -m betalens_db_manager \
     --env "$HOME/betalens-infra/tushare.private.env" serve --port 8010

``serve`` 默认启动待命 worker；已有外部 worker 才使用 ``--no-worker``。
启动不自动拉取数据、不自动恢复中断任务。端口占用时先检查现有服务，避免启动第二个执行器。

* ``/tables``：通用本地表及结构管理。
* ``/tasks``：通用更新与审计。
* ``/integrity``：行业、日历和停牌区间的完整性、版本任务及发布审计。
* ``/mappings``、``/sources``：通用字段映射与订阅源。
* ``/settings``、``/schedules``：私有配置及显式启用的调度。

浏览器访问 ``http://localhost:8010/integrity``；以实际端口和服务状态为准。
关闭最后一个管理页面后，经过重连宽限停止任务子进程，管理服务保持可访问。
自动调度默认关闭。已暂停的历史补全不因服务重启或文档示例自动恢复。

现有表清单
----------

本次只读核对，PG ``betalens`` schema 与 CH ``betalens`` 数据库各有 **40 张表**，
名称一致：24 张通用业务表、10 张管理表、3 张参考版本表及 3 张参考元数据表。
当前 CH 表均为 ``ReplacingMergeTree``；参考表的排序键包含修订号，保留历史版本。
今后增表后以实际清单和合同为准，不把该计数作为固定验收常量。

.. list-table:: 业务表分组
   :header-rows: 1
   :widths: 23 77

   * - 业务域
     - 表
   * - 证券档案
     - ``security_dim``、``stock_profile``、``fund_profile``、``index_profile``、``etf_profile``、``security_code_history``
   * - 行情、指标、财报
     - ``market_daily_raw``、``market_daily_post_adjusted``、``stock_quote_supplement_daily``、``fundamental_periodic``
   * - 基金观测
     - ``fund_nav_daily``、``fund_nav_observation``、``fund_shares_daily``
   * - 行业、成分
     - ``industry_scheme_dim``、``industry_dim``、``security_industry_membership``、``industry_membership_record``、``index_membership``
   * - 日历、状态、事件
     - ``trade_calendar``、``trade_status_daily``、``corporate_action``、``dividend_event``、``suspension_event``、``security_name_history``
   * - 新参考版本
     - ``stock_industry_membership_versions``、``trade_calendar_versions``、``stock_suspension_intervals_versions``

管理表为 ``meta_table/meta_column/meta_source_mapping/meta_source_capability``、
``import_task/import_source_archive/import_target_status/import_row_audit/import_rollback``、
``dataset_coverage``。新增的可信发布元数据为
``reference_revision/reference_scope/reference_publication_event``，保存在对应目标库中。
通用表清单与三域参考表清单分别由 ``/api/table-registry`` 和
``/api/reference/tables`` 提供；不能以通用清单的数量代替全部物理表数量。

只有三个新参考域采用本轮的追加版本/发布门槛，不能推断行情、财报和所有旧表也已改为
追加版本。旧 ``trade_calendar/security_industry_membership/suspension_event`` 保留，
但旧行及旧任务完成状态不会自动成为新参考域的可信基线。
详细规则见 :doc:`db-manager-reference`。

研究查询边界
------------

当前本地库不创建 ``daily_market/daily_index/daily_fund/trade_status`` 兼容视图。
``betalens.datafeed`` 和 ``BacktestBase`` 中这些名字属于旧研究查询合同；
新表存在、连接成功或把 ``BETALENS_DB_NAME`` 改为 ``betalens``，都不代表旧查询可直接运行。
本轮没有改写旧业务 API，也不把三域参考表当作旧接口的替代视图。

三域管理查询通过 ``POST /api/integrity/check`` 或默认可信读取的
``GET /api/query/{version_table}``，审计全历史使用
``GET /api/reference/versions?target=postgres&audit=true``。
直接 SQL 读物理版本表会包含未发布或已撤销数据，不能省略目标库发布凭证筛选。

验收快照与覆盖限制
------------------

2026-10-25 的三域真实验收使用 ``000001.SZ``、2026-10-12 至 2026-10-14、
两套行业分类、1—3 级和三个交易所日历。两库各追加 15 行、18 份范围凭证，
各有 30 项检查通过；业务字段哈希一致，随后已执行撤销发布。
因此这批样本在审计历史中仍可见，默认业务读取不再采用它们。
该结果是小样本验收，不是历史全量或全市场完整性证明。

本机验收材料位于 ``~/betalens-infra/reference-acceptance-20261025/``：
``report.json``、``audit.json``、``cross-target-verification.json``、
``final-service-check.json``。浏览器隔离验收材料位于
``~/.cache/betalens-reference-browser/``，测试库与临时测试账号已清理。
运行状态会变化，使用前重新查看任务、调度及范围检查，不把快照中的“无活动任务”视为永久保证。
