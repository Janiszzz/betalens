独立数据基础设施
================

``betalens-db-manager`` 独立安装和运行，维护数据接入、数据库结构、字段合同、
元数据、归档、审计和回退。组件不导入 ``betalens``、研究 Dashboard 或已删除的
``betalens_db_manager``。后复权行情按数据合同生成，暂不计算量化因子。

首次使用
--------

.. code-block:: bash

   python3 -m venv /path/to/infra-venv
   /path/to/infra-venv/bin/python -m pip install -e './betalens-db-manager[files]'
   cp betalens-db-manager/infra.env.example /private/path/infra.env
   chmod 600 /private/path/infra.env
   # 填写数据库连接和 TUSHARE_TOKEN
   /path/to/infra-venv/bin/betalens-db-manager --env /private/path/infra.env doctor
   /path/to/infra-venv/bin/betalens-db-manager --env /private/path/infra.env serve

访问 ``http://127.0.0.1:8010``；Windows 可通过 WSL localhost 转发访问。静态网页随包安装。
修改前端后在 ``betalens-db-manager/frontend`` 执行 ``npm ci && npm run build``。
全新机器可通过 ``services up`` 启动组件内 Compose 模板中的两个数据库，已有服务则直接配置连接。
服务启动和建库建表分开执行。

私有配置
--------

默认 PostgreSQL 数据库为 ``betalens``、数据 schema 为 ``betalens``，ClickHouse 数据库为
``betalens``。连接由 ``PGHOST/PGPORT/PGDATABASE/PGSCHEMA/PGUSER/PGPASSWORD`` 和
``CLICKHOUSE_URL/CLICKHOUSE_DB/CLICKHOUSE_USER/CLICKHOUSE_PASSWORD`` 配置。
``TUSHARE_TOKEN``、``TUSHARE_RPM`` 分别设置凭证和请求限频。

网页可编辑配置；秘密字段留空时保留原值。保存到仓库外的私有环境文件，采用原子替换和
``0600`` 权限；保存后重启服务生效。网页和日志不返回密码或 Token。
``BETALENS_INFRA_STATE`` 设置持久化任务目录；``BETALENS_INFRA_ARCHIVE`` 可单独设置下载归档目录。

建库建表与显式重建
------------------

网页查看结构计划后执行安装，或使用：

.. code-block:: bash

   betalens-db-manager --env /private/path/infra.env schema plan
   betalens-db-manager --env /private/path/infra.env schema apply --token <plan-token>
   betalens-db-manager --env /private/path/infra.env rebuild --dry-run
   betalens-db-manager --env /private/path/infra.env rebuild --confirm-drop

普通安装创建缺失的表、列和索引；不兼容的字段类型会停止安装。重建默认只读预览，
``--confirm-drop`` 才删除所列旧 schema/table。删除前保存对象、字段、行数和时间范围清单。
旧数据库名由 ``BETALENS_LEGACY_PGDATABASE`` 指定，默认 ``datafeed``；清理旧
``tushare/market/betalens`` schema，目标库重建 canonical 表；CH 清理所选数据库旧表。
清单是结构与覆盖审计记录，不是原始数据备份。

统一表合同
----------

``canonical.py`` 定义源无关表，``mapping.py`` 登记来源字段、目标字段、单位、转换规则和版本。
PG/CH 的事实表及管理表逻辑列集合一致，物理类型分别映射到 ``numeric/date/timestamptz`` 和
``Decimal/Date32/DateTime64``。核心表包括：

* ``security_dim``、``trade_calendar``。
* ``market_daily_raw``、``market_daily_post_adjusted``。
* ``trade_status_daily``。
* ``industry_scheme_dim``、``industry_dim``、``security_industry_membership``。
* ``index_membership``、``fundamental_periodic``、``corporate_action``。
* ``fund_nav_daily``、``fund_shares_daily``、``security_code_history``。

不创建 ``daily_market/daily_fund/daily_index/trade_status`` 兼容视图。
财务宽表保留 income/balancesheet/cashflow/fina_indicator 的全部登记数值输入，报表种类和
版本参与业务键，不把不同报表的公告日期混在一起。未入宽表的描述字段在字段映射中明确标记
``__raw_archive_only__``，原始响应始终保留。

行情量由手转股/份乘 100，金额由千元转元乘 1000，股本和市值由万转基本单位乘 10000。
申万行业行情和基金份额按各自合同转换。数值全程使用 Decimal，JSON 传输为字符串。
股票/基金后复权价格为原价乘源复权因子；指数保存原生指数值。后复权宽表同时保留
``raw_*``、因子、公式、计算版本，成交量和成交额不复权。缺因子时明确标记不可用，价格保持空值。
指数请求按源合同路由：中证 ``.CSI`` 代码保留原后缀，申万 ``.SI`` 代码使用
`sw_daily <https://tushare.pro/document/2?doc_id=327>`_；
`index_daily <https://tushare.pro/document/2?doc_id=95>`_ 不提供申万行业行情。

事实行带 ``event_date/available_at/downloaded_at``、可得时间依据、来源接口、字段映射版本、
任务 ID 和源记录哈希。源端仅有公告日期时采用下一日零点（上海）的保守日内假设；源端无法证明
历史修订的实际公开时间时，在依据字段说明，不将当前补拉版本当成严格历史 PIT 数据。

更新、审计与回退
----------------

网页“更新与审计”包含完整流程：选择 Tushare 或显式映射文件，选择目标库（默认同时勾选
PG/CH），下载归档，核对原始值、字段集合、单位转换、标准化宽表和 before/after，最后确认导入。
两个目标分别提交，任一失败显示 ``partial``；跨库键集合、字段哈希或行数不同也会显示差异。
“数据查询”可针对证券和日期范围检查交易日缺口，扣除已知上市/退市边界，并单独统计
未知交易状态和后复权价格不可用行。缺行情只标记为待核查缺口，不推定为停牌。

源字段按名称映射，必需字段缺失或未登记字段阻止发布，可选字段缺失显示说明。触及接口返回上限时
分片；不支持可靠分片/分页时停止，不把截断响应视为完整数据。权限不足作为源能力缺失记录；
``strict_capabilities`` 可要求这些缺失阻止整个任务。

本地 SQLite 保存提交前的 before/after 日志和每表进度，并将审计镜像写入选中的数据库。
只选择 CH 时无需 PG 可用。重复归档不产生重复逻辑行；中断恢复会核验已写入值后续接。
恢复下载不会绕过手工确认；已确认但部分失败的提交可用 ``resume`` 续接。

回退生成新任务 ID，保留原审计。只有当前行仍等于原任务的 after 值才允许逆操作。
后续任务修改过的行显示冲突，阻止未执行的逆操作。PG 在事务中回退；CH 以临时表重建受影响
分区、复制未受影响分区后原子交换发布，每表保存恢复进度。回退审计记录恢复前后值及目标状态。

基金和指数缺少等价于股票停复牌的权威逐证券接口；行情、成交量和交易日历只能产生
``derived_from_daily_quote`` 状态。股票结合上市/退市边界和 suspend_d；缺行情不等于停牌，
盘中停复牌也不会错误地标成全日停牌。行业归属保留三层分类、历史纳入/剔除日期和分类体系。
北交所新旧代码原样登记对照，上市日期不推定为代码变更生效日。

CLI 与历史补全
--------------

.. code-block:: bash

   betalens-db-manager --env /private/path/infra.env plan request.yaml
   betalens-db-manager --env /private/path/infra.env download <job>
   betalens-db-manager --env /private/path/infra.env preview <job>
   betalens-db-manager --env /private/path/infra.env commit <job> --token <preview-token>
   betalens-db-manager --env /private/path/infra.env resume <job>
   betalens-db-manager --env /private/path/infra.env rollback <job> --targets postgres clickhouse
   betalens-db-manager --env /private/path/infra.env query market_daily_raw --target clickhouse
   betalens-db-manager --env /private/path/infra.env load --config betalens-db-manager/initial_load.example.yaml

初始加载模板从 2004 年开始，基础信息包括退市股，A 股行情排除 B 股；逐月加载行情，指数和财报
分年份加载。四个财报 ``*_vip`` 接口均通过实测权限检查时，自动采用按报告期分页下载，
并限定到证券清单及其北交所历史代码；否则按证券组调用普通接口。两类接口使用相同字段合同。
再次执行同一个 manifest 会登记未完成批次的恢复操作；复用成功分片，并重新审计确认。
完整历史请求数量受证券数量、接口权限和请求限频影响；任务历史和覆盖范围可以区分样本验收与全量完成。
一次性真实验收脚本位于 ``scripts/verify_live_samples.py``，报告写入私有状态目录。

Wind 保留 SourceAdapter 接口，未配置 SDK 和字段合同时明确显示不可用。
文件支持 CSV/CSV.GZ/XLSX/Parquet，必须逐列指定目标字段；可为数值列显式登记单位乘数，
日期列可登记格式，时间列必须提供时区或自带时区。原文件和转换前的行均归档。
可得时间没有依据时保持未知。

自动更新默认关闭，用户主动勾选并保存后才启用。手动单次更新、安装、初始加载和服务启动不会
创建周期任务。调度按 Asia/Shanghai；重启不会立即补跑已错过的计划。

本地表全生命周期（0.3）
-----------------------

管理入口更名为 ``betalens-db-manager``，默认配置目录为 Linux/WSL
``~/.config/betalens/db-manager/``，Windows 原生为 ``%LOCALAPPDATA%/Betalens/DBManager/config``。
WSL 服务由 Windows 浏览器访问时仍使用 WSL 目录。凭据不得保存在任何 Git 工作树内。

在“本地表”填写业务字段、类型精度、业务主键和时间规则，保存 YAML 草案，预览 PG/CH DDL，
再确认执行。``ingested_at``、``import_task_id``、来源、哈希和版本等字段由系统维护。
表定义位于 ``tables/<table_id>.yaml``，映射位于 ``mappings/<table_id>.yaml``。
直接编辑 YAML 产生待校验变更，不自动执行 DDL；无效编辑保留上一有效运行版本。

“字段映射”允许拖动源字段到本地字段，或通过下拉选择；基本单位、日期格式和时区规则可编辑。
来源主备顺序写入 YAML，实际采用的来源和冲突进入任务证据。非默认输出字段也显式请求。
内置证券档案、分红、停复牌、名称历史和净值公告明细，完整保留对应查询粒度和字段。

“更新与审计”按本地表和日期自动匹配请求。拉取后先检查源缓存、单位、日期、拒收和 before/after，
再确认通过的记录入库。回退保留原记录，后续任务修改过的行产生冲突。
已有归档可创建补字段任务；无法恢复的字段需要重新拉取。数据库中新增的历史审计列保持未知，
最后实际写入时间可从历史审计读取，不把本次建列时间冒充旧数据入库时间。

网页会话管理执行器。最后一个页面关闭后，经过重连宽限停止任务子进程；任务记录和检查点保留。
自动更新默认关闭，仅页面在线时调度；定时任务也停止在待审计阶段。
CLI 数据操作进入同一队列，通常等待页面在线；显式 ``worker --once`` 可在终端执行一个操作。
``load`` 登记批次拉取任务，不再自动确认入库。重新打开网页不会自动恢复中断任务。

完成任务的源缓存七天后清理，但仍被未完成任务引用的归档继续保留；长期审计、任务参数和回退证据保留。

验收脚本 ``scripts/verify_manager_live.py`` 使用临时数据库核查 29 个接口及特殊证券样本；
``scripts/migrate_existing.py`` 默认只预览增量升级，``--apply`` 执行兼容增列和新建明细表，
遇到破坏性变化停止并要求单独结构计划。它们均不恢复历史全量补全。
