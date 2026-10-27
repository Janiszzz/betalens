独立数据库管理
==============

``betalens-db-manager`` 是独立安装的 FastAPI + React 组件，负责本地表结构、订阅源、
字段映射、请求任务、缓存审计、入库与回退，不依赖 Betalens 因子、Dashboard 或回测代码。
Python 包名为 ``betalens_db_manager``；这是当前组件的包名，不能与已删除的旧 Windows GUI 混淆。

PostgreSQL 和 ClickHouse 是同等目标库，默认同时选择，也可单独更新，分别报告执行和核验结果。
默认 PG 数据库及 schema 均为 ``betalens``，CH 数据库为 ``betalens``。

首次使用与启动
--------------

在仓库根目录创建组件虚拟环境，并以开发模式安装：

.. code-block:: bash

   python3 -m venv betalens-db-manager/.venv
   betalens-db-manager/.venv/bin/python -m pip install -e './betalens-db-manager[files]'
   betalens-db-manager/.venv/bin/betalens-db-manager serve --port 8010

Windows 原生运行使用虚拟环境的 ``Scripts\python.exe`` 和
``Scripts\betalens-db-manager.exe``。浏览器访问 ``http://127.0.0.1:8010``；
Windows 浏览器可通过 localhost 转发访问 WSL 服务。服务仅监听本机，静态前端随包安装。
已有私有环境文件时，在子命令前指定 ``--env /private/path/credentials.env``。

``serve`` 默认管理一个 worker，worker 为每次数据操作启动独立子进程。
仅在外部已管理 worker 时使用 ``serve --no-worker``。
修改前端后，在 ``betalens-db-manager/frontend`` 分别执行 ``npm ci`` 和 ``npm run build``。

服务启动不会自动建库、删除表、入库或恢复历史补全。全新机器可用 ``services up``
启动组件提供的 Compose 数据库模板；已有数据库直接配置连接。建库建表仍需单独预览和确认。

页面与工作流
------------

.. list-table:: 管理入口
   :header-rows: 1
   :widths: 20 18 62

   * - 页面
     - 路径
     - 用途
   * - 本地表
     - ``/tables``
     - 表清单、时间分布、结构检查、建表及变更。
   * - 更新与审计
     - ``/tasks``
     - 普通注册表的请求、日志、缓存、字段校验、确认入库和回退。
   * - 业务完整性
     - ``/integrity``
     - 行业、日历、停牌范围检查、追加版本、可信发布及撤销。
   * - 字段映射
     - ``/mappings``
     - 查询源多选、字段连线、联合键与来源优先级。
   * - 订阅源
     - ``/sources``
     - 来源实例、查询合同及配置状态。
   * - 配置
     - ``/settings``
     - 数据库、鉴权、目录、限频及连接测试。
   * - 自动更新
     - ``/schedules``
     - 显式启用的周期请求；默认关闭。

页面支持直接访问、刷新及浏览器前进、后退。普通更新流程为：

#. 查看本地表业务日期分布、最新业务日期和最后实际写入时间。
#. 指定表、证券、日期和目标库，核对自动生成的实际 API 请求及依赖。
#. 登记任务后手工拉取，查看脱敏参数、HTTP 状态、耗时、分页及重试日志。
#. 审计原始值、标准化结果、单位和日期、字段来源、拒收记录及 before/after。
#. 确认审计版本后写入，各库分别核验业务键、行数和内容哈希。
#. 查看结果或回退；数据、映射、结构或目标状态变化后重新审计。

表级日期分布只证明当天存在记录，不能证明某只股票或全市场完整。
三个业务参考域使用请求与发布范围凭证检查，详见 :doc:`db-manager-reference`。

本地表、YAML 与多源映射
----------------------

内置和自定义表由同一个 ``TableRegistry`` 管理，导入程序读取注册定义。
建表时填写中文名称、说明、粒度、字段、精度、单位、业务主键和时间规则，
保存草案后预览 PG/CH DDL，再确认部署。系统维护任务、来源、哈希、版本及
``ingested_at`` 等受保护审计字段，源字段不能覆盖它们。

配置目录中的可编辑定义：

* ``tables/<table_id>.yaml``：表定义。
* ``mappings/<table_id>.yaml``：查询绑定、字段、共同键和转换规则。
* ``sources.yaml``：订阅源实例及查询合同；鉴权使用环境变量引用。

SQLite 保存有效版本、任务、执行尝试、日志、操作状态和审计索引；目标库元数据记录已部署结构。
直接编辑 YAML 不自动执行 DDL；无效修改显示错误，保留上一有效版本。
数据任务固定使用创建时的结构、映射及来源合同快照。

映射页用下拉多选添加或移除查询源，已有来源显示勾选，重复绑定会被识别。
关系图展示源字段到本地字段的连接，也可下拉选择。单查询普通字段一一对应；
多查询合并必须使用完整业务键，可含多个联合字段。重复业务列要明确主备优先级，
源内同键多条不同值会阻止不明确的合并。

手工表的 ``format_version: 2`` 映射用 ``join_keys`` 指定共同键，并保留相应字段连线。
未使用源字段在 ``ignored_fields`` 说明原因。草稿可以保存，存在阻断问题时不能更新。
数值转换用 Decimal，前端以字符串传输精确值；零值有效，转换错误不能作为空值静默回退。

PG/CH 保持逻辑字段、单位及业务键语义一致，分别映射为
``numeric/date/timestamptz`` 与 ``Decimal/Date32/DateTime64`` 等物理类型。
增列、改类型、改主键、重命名、删列或删表先做结构预览；已有数据新增必填列或收紧可空性时
先要求可审计回填。归档与物理删除分开，破坏性迁移单独确认。

来源能力与 Tushare 完整获取
--------------------------

Tushare 只是订阅源之一。``SourceRegistry`` 支持 HTTP API、Python 适配器及
``betalens_db_manager.sources`` 入口点；Python 扩展使用 ``SubscriptionAdapter`` 和 ``FetchContext``。
来源负责提数，公共引擎负责映射、审计及数据库写入。Wind 未配置 SDK 和字段合同时明确不可用。
CSV、CSV.GZ、XLSX、Parquet 通过显式映射导入，保留原文件和转换证据。

Tushare 显式请求合同字段，按字段名取值，不依赖列顺序。缺少必需字段、出现未登记字段、
重复或交叠分页会阻止相关响应提交。权限不足、空响应、合同覆盖与字段实际验证分别报告，
不能据此声称数据完整。字段核查包含源字段、本地列、规则、实际响应与数据库读回证据。

每任务可设置每分钟请求次数，未指定用 ``TUSHARE_RPM``，不会由 Token 推断会员等级。
实际请求同时遵守接口限频。HTTP 429 遵守 ``Retry-After``；日或小时配额耗尽时保留检查点。
支持 offset 的查询按实际返回行数推进至空页，短页不直接终止。
非分页查询触及已登记上限时，按不重叠日期、证券或文档枚举分区重新请求。
证券分区使用重新获取并归档的完整源端清单，不能用残缺本地清单代替。
最小分区仍触及上限或无法证明覆盖时停止，不将截断响应标为完整。
每页和分区保存检查点、归档哈希，网页展示“请求分片与获取完整性”。

普通业务表与回退边界
--------------------

普通表包括证券维度及股票、基金、指数档案，原始和后复权行情，日指标、基金净值与份额、
行业字典和成员记录、分红及停复牌明细、财务宽表等。登记源字段需要明确去向；
仅存压缩归档不计作完整落列。财报区别 ``ann_date``、``f_ann_date`` 和报告修订版本。

行情 ``vol`` 从手转股或份乘 100，``amount`` 从千元转元乘 1000，股本及市值从万转基本单位乘 10000。
基金份额、申万行业行情等使用各自合同单位，不套用其他查询规则。
股票、基金后复权价格为原价乘源复权因子，指数保留原生值。后复权表保留原价、因子、公式和版本，
成交量、金额不随价格复权；缺因子明确标记不可用。

``event_date``、``available_at``、``downloaded_at``、``ingested_at`` 分别表示业务日期、实际可得、
下载和目标库实际写入时间。历史补列不能把当前时间冒充旧入库时间；可得时间依据和无法证明的
历史修订单独报告，不宣称所有表具有严格历史可知性。

普通表按业务键比较并 upsert。回退创建新任务，仅当当前行仍匹配原 after 时恢复 before 或删除新增行，
后续修改产生冲突。PG 用事务；CH 在受控窗口用临时表重建交换并保存检查点。
三个新参考域使用追加版本与撤销发布，不能混用这两种回退语义。

不创建 ``daily_market``、``daily_fund``、``daily_index``、``trade_status`` 兼容视图。
旧 ``BacktestBase`` 和业务查询 API 不自动转向新表，后续查询重构需明确接入新合同。

私有配置与目录
--------------

Linux/WSL 默认配置为 ``~/.config/betalens/db-manager/``，状态为
``~/.local/share/betalens-db-manager/``。Windows 原生分别为
``%LOCALAPPDATA%\Betalens\DBManager\config\`` 和同级 ``state\``。
Windows 浏览器访问 WSL 时，文件仍写入 WSL。

环境文件选择顺序为：显式 ``--env``、``BETALENS_DB_MANAGER_ENV``、``BETALENS_INFRA_ENV``、
默认配置目录下 ``credentials.env``。文件中的同名值优先于进程环境。

.. list-table:: 主要环境变量
   :header-rows: 1
   :widths: 48 52

   * - 变量
     - 用途
   * - ``PGHOST``、``PGPORT``、``PGDATABASE``、``PGSCHEMA``、``PGUSER``、``PGPASSWORD``
     - PostgreSQL 连接。
   * - ``CLICKHOUSE_URL``、``CLICKHOUSE_DB``、``CLICKHOUSE_USER``、``CLICKHOUSE_PASSWORD``
     - CH HTTP 连接；URL 不含鉴权值。
   * - ``CLICKHOUSE_WRITER_MODE``
     - ``disabled``、``local_dedicated`` 或 ``local_admin``。
   * - ``TUSHARE_TOKEN``、``TUSHARE_RPM``
     - Tushare 凭证及默认分钟请求上限。
   * - ``BETALENS_DB_MANAGER_CONFIG``
     - 表、映射和来源 YAML 目录。
   * - ``BETALENS_DB_MANAGER_STATE``
     - 状态目录，优先于旧名 ``BETALENS_INFRA_STATE``。
   * - ``BETALENS_INFRA_ARCHIVE``
     - 原始响应归档目录；默认状态目录下 ``archive/``。

网页可编辑连接、鉴权和限频。秘密输入留空保留原值，响应只显示是否配置。
保存后新执行尝试读取新配置，正在执行的子进程使用启动快照。改变配置或状态目录时返回
``restart_required``，需重启服务。凭据文件检查真实路径、符号链接及父目录 Git 标记，
即使被 ``.gitignore`` 忽略也不能写入工作树。WSL 凭据使用 Linux 文件系统；
目录 ``0700``、文件 ``0600``，Windows 使用受限 ACL。任务、映射、日志和报告不保存鉴权值。

CH 默认 ``disabled``；``local_dedicated`` 用本机专用写账号，``local_admin`` 用用户显式配置的
现有本机管理员，执行目标锁、权限和写入日志核验。管理员模式无法阻止外部管理员绕过工具写入，
维护时需避免其他客户端并发修改。启动不会自动创建管理员、专用写账号或业务只读账号。

任务、会话与命令行
------------------

.. code-block:: bash

   betalens-db-manager --env /private/path/credentials.env doctor
   betalens-db-manager --env /private/path/credentials.env schema plan
   betalens-db-manager --env /private/path/credentials.env schema apply --token <schema-token>
   betalens-db-manager --env /private/path/credentials.env plan request.yaml
   betalens-db-manager --env /private/path/credentials.env download <job>
   betalens-db-manager --env /private/path/credentials.env preview <job>
   betalens-db-manager --env /private/path/credentials.env commit <job> --token <preview-token>
   betalens-db-manager --env /private/path/credentials.env job <job>
   betalens-db-manager --env /private/path/credentials.env query market_daily_raw --target clickhouse

CLI 的 ``download``、``commit``、``resume``、``rollback``、``load`` 进入持久化队列，仍能在网页找到。
常驻 worker 等待页面在线；显式 ``worker --once`` 可在终端执行一个排队操作。
``load --config`` 登记批次拉取，完成后仍需审计确认，不自动入库。
普通任务 ``resume`` 可核验检查点续接；参考任务中断后要创建关联新任务，重新审计确认。
参考版本查询使用 :doc:`db-manager-reference` 的专用接口，不能直接传给普通 CLI ``query``。

最后一个管理页面关闭前浏览器提示；关闭或断连后经过重连宽限停止任务子进程，服务继续可访问。
任务和检查点保留，重新打开不自动恢复。自动更新默认关闭，仅显式开启且页面在线时调度，
错过的调度不补跑，周期任务也只拉取至待审计。
完成任务源缓存七天后清理，未完成任务引用的归档、长期审计和回退证据保留。

``rebuild`` 默认仅预览；``rebuild --confirm-drop`` 明确删除旧表，不是日常初始化或参考表升级命令。
结构和覆盖清单不是数据备份。已有生产数据应纳入清单并预览兼容迁移，不恢复已暂停历史补全。

测试与验收
----------

单元、隔离库测试在 ``betalens-db-manager/tests``，浏览器测试在 ``frontend/e2e``。
使用组件虚拟环境，真实库测试通过私有配置创建隔离测试库并清理：

.. code-block:: bash

   betalens-db-manager/.venv/bin/python -m pytest betalens-db-manager/tests -q
   BETALENS_INFRA_ENV=/private/path/credentials.env BETALENS_INFRA_INTEGRATION=1 \
     betalens-db-manager/.venv/bin/python -m pytest betalens-db-manager/tests/test_reference_storage.py -q

``betalens-db-manager/scripts/production_reference_smoke.py`` 默认只保存建表预览；
``--execute`` 会新增结构、真实提数、发布、读回并撤销，保留历史版本及审计，
应作为明确授权的生产验收操作。脚本不自动恢复历史补全。
