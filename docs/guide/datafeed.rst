数据订阅模块
============

``betalens.datafeed`` 是独立的只读订阅层，支持 PostgreSQL 和 ClickHouse。
数据库建立、字段配置、来源接入、写入、审计和发布由 ``betalens-db-manager`` 负责。
新查询仅支持当前统一宽表与参考版本表，不探测或回退到旧长表、旧规范化表和兼容视图。

.. important::

   实现严格 PIT 查询，与实际数据具备历史可知性证据，是两个验收条件。
   当前参考处理器的 ``available_at`` 为空，部分财报时间只是推定，指数成分缺少完整快照证明。
   因此连接和表结构正确，也可能得到缺失值或 ``PITUnavailableError``。
   不允许用业务日期、下载时间或当前值补造历史证据。

连接与价格口径
--------------

独立订阅 JSON 模板在 ``betalens/datafeed/config.example.json``。
指定 ``config_file`` 或 ``BETALENS_CONFIG``，也可使用用户配置目录。
构造参数／``db_config`` 高于环境变量，环境变量高于文件。
旧 PG ``database`` 配置节和 ``BETALENS_DB_*`` 变量仍可读取；配置错误直接报错，不连接默认库。
完整规则见模块目录的 ``CONFIG_README.md``。

.. code-block:: python

   from betalens.datafeed import Datafeed

   with Datafeed("market_daily_raw", config_file="/path/to/subscription.json",
                 backend="clickhouse") as data:
       prices = data.query_time_range(
           codes=["000001.SZ"], start_date="2024-01-01", end_date="2024-01-31",
           metric="close", filters={"asset_type": "stock"}, as_of="2024-01-31 16:00:00",
       )

``backend`` 支持 ``postgres`` 和 ``clickhouse``，默认从配置读取，未设置为 PostgreSQL。
价格口径直接由表名决定：``market_daily_post_adjusted`` 为后复权，``market_daily_raw`` 为未复权。
指数原生点位请使用 ``market_daily_raw`` 并指定 ``filters={"asset_type": "index"}``。
不再提供 ``adjustment`` 参数或配置项，不隐式切换任何表。
单次查询使用固定配置，新实例读取最新配置。管理服务无需在线。

.. note::

   直接以后复权价格计算成交手数是现有回测引擎的缺陷，
   不能通过“复权计量”将其合理化。本次没有修改回测、factor 或 mining 的业务实现。
   回测使用未复权价格进行成交、手数和权重换算，以及查询失败后按正常交易继续等问题，
   均为后续修复项；本模块通过测试不代表整个研究链路已经满足 PIT。

直接指定表与字段
----------------

``table_name`` 是实际数据库表名，``metric`` 是实际数值列名。
已删除 ``registry.py``，不保留 ``daily_market/fundamentals`` 等逻辑数据集、
中文指标、``prev_close`` 等字段别名，也不通过指标推断资产类型或报表口径。
例如前收盘价直接使用 ``previous_close``，收入直接使用 ``fundamental_periodic.revenue``。
缺表、缺字段明确报错；``metric=None`` 报错，调用方必须明确选择字段。

* 行情：``market_daily_raw`` 或 ``market_daily_post_adjusted``。
  ``filters={"asset_type": "stock"}`` 限定股票；不传此条件时不替调用方筛选资产类型。
* 财报：``fundamental_periodic``，必须在 ``filters`` 中同时指定实际的
  ``statement_type`` 与 ``report_type``，例如 ``income/1`` 或 ``fina_indicator/unspecified``。
* 行业专用接口：``stock_industry_membership_versions``。
* 交易状态专用接口：``stock_suspension_intervals_versions``，联合范围证据计算状态。
* 本地日历：``trade_calendar_versions``。
* 指数成分专用接口：``index_membership``，必须提供完整快照证据。

``filters`` 为字段名到非空标量的字典，仅表示相等条件；与 codes、时间范围共同筛选，
字段名经校验，值通过数据库参数绑定，不接受 SQL 片段。
普通数值时序查询要求表具备 ``security_code/event_date/available_at/availability_basis/data_quality_status``；
数值列直接由实际结构校验，无需增加注册项。参考版本表及完整成分集合必须走专用接口，
不能通过通用数值查询绕过发布、完整范围核验。

同一公告时刻发布多个报告期时选择最新报告期；不同可得时刻的旧报告修订仍是独立观测，
nearest 保持按可得时间匹配，不暗改为按报告期末匹配。

字段值保持数据库原生单位；有 ``meta_column`` 条目时在 ``remark.unit`` 中提供单位，
没有条目时为 ``None``，不通过中文别名猜单位，不自动执行股／手、元／万元或报告期／TTM 转换。

固定输出合同
------------

* nearest：``code/input_ts/datetime/diff_hours/{metric}/name``。
* 范围查询：``datetime/code/name/metric/value/remark``。
* 交易状态：``code/datetime/value/status_text/name``，``value`` 为 pandas ``Int8``，允许 ``NA``。
* 行业：保留 ``code/query_date/effective_dt/sec_name/industry_value/remark/ind_name/ind_code/scheme``。
* 指数单日返回代码列表，面板返回 ``{date: set[str]}``，交易日返回 ``list[date]``。

两个后端输出相同列顺序、dtype、缺失值和稳定排序。数值指标为 ``float64``；
字符串及 remark 为 ``object``，时间列为上海本地无时区 ``datetime64[ns]``。
内部以 UTC 比较，无时区用户输入按上海时间解释，数据库 timestamp 按 UTC 读取。
空结果保留完整字段和类型。nearest 保留输入顺序与重复输入，范围查询按时间倒序、代码和指标排序。

时序输出的 ``datetime`` 是有证据的可得时间，不是报告期或下载时间；
``remark`` 提供业务日期、报告期、版本、实际表名、原生单位及资产／报表类型。

查询边界
--------

.. code-block:: python

   with Datafeed("fundamental_periodic") as data:
       values = data.query_nearest_before({
           "codes": ["000001.SZ"], "datetimes": ["2024-04-30 15:00:00"],
           "metric": "revenue", "time_tolerance": 24 * 365,
           "filters": {"statement_type": "income", "report_type": "1"},
       })

before 包含输入时点，after 严格晚于输入时点。区间 before 为闭区间，
区间 after 为开区间；容差单位为小时，与区间取交集。
范围查询的纯日期结束值包含整天，带时间的结束值包含该时刻。
``as_of`` 进一步限制历史可知性；省略时范围查询返回范围内可证明的可得时间序列。

after 查询用于后续结果观察，其结果在输入时点尚不可知，不能当作该时点的因子信息。
无证据或超出容差的 nearest 输入仍保留一行缺失结果。

严格 PIT 与可信范围
-------------------

普通时序记录必须有 ``available_at``，且 ``data_quality_status`` 为 ``accepted/verified``。
``availability_basis`` 只接受内容版本级证明：

* ``source_version_publication_time``：来源证明该内容版本在此时刻公开。
* ``verified_historical_version``：经核验的历史版本及可得时间。

这些标记是生产者需提供证据的合同，不是让使用者给旧数据改标签即可通过。
``unknown/assumed/download`` 等依据及只有事件公开时间但没有内容版本证明的记录，均不放行。
行情已被覆盖的旧版本不能由查询端恢复；整行时间证明必须覆盖该行中查询字段及复权因子。

参考域先读取目标库发布事件，再核验修订头、范围清单、内容／键哈希和行数。
未发布和已撤销版本不可见。每次按整范围选择历史时点可知的最新完整快照，
不跨版本拼值、不将新版本空值从旧版本补回。

完整范围必须在已有 ``reference_scope.payload.evidence`` 中带 ``availability`` 对象，
其中 ``available_at/availability_basis`` 证明整个快照在当时可知；每条非空记录也必须满足行级证明。
当前管理器没有提供该历史证明，故默认无法用于严格 PIT。
零事件停牌范围同样需要范围级证明，上市范围和日历依赖也必须有历史可知证据。
这是管理端待补合同，datafeed 不生成或写入这些证据。

交易状态：``1`` 为已证实上市、开市且该时点未报告停牌，``0`` 为已证实停牌，
``-1`` 为有证据的未上市，其他情况为 ``NA``。
``1`` 不承诺涨跌停、委托限制等所有成交条件均已检查。
纯日期状态查询按该日 00:00 解释；日内状态请传完整时间，不能擅自按盘后信息还原盘前状态。

指数成分当前表缺少完整性证据，接口明确报错。查询端预留可选的
``snapshot_complete``、``snapshot_member_count`` 整数字段合同；只有全部成分在时点可知、
完整标记为 1 且数量吻合才能返回。当前物理表不因此自动增列。
空成分快照缺乏可表示的范围证明时仍报错，不能用“没有行”代替空集合。

交易日与名称
------------

``get_absolute_trade_days`` 和 ``trade_days_offset`` 只读本地可信日历，不访问 AkShare 或其他在线日历。
支持 ``D/W/M/Q/H/S/Y``；``H`` 与 ``S`` 均指自然半年度最后交易日。
交易所参数默认 ``SSE``；不再将 ``SHSE`` 转换为 ``SSE``。
传入 ``SHSE`` 时只查询该字面值，不会回退到上交所日历；缺失交易所或自然日不能推断为休市。
偏移保留原输入时间，非交易日起点先定位不早于该日的交易周期，再应用 offset。

``query_names(codes)`` 查询当前展示名称；传 ``as_of`` 查询历史名称。
历史查询没有对应证据时留空，不补当前名称。
独立查询函数可传入 ``Datafeed``；PG 旧游标入口保留，CH 不模拟 PG 游标。

错误与结构调整
--------------

``SchemaError`` 表示缺表、缺必需字段或逻辑类型不符；``VersionConflictError`` 表示版本冲突或发布内容核验失败；
``PITUnavailableError`` 表示无法用原返回容器表达缺失的完整覆盖证据。
不存在的数值字段使用 ``SchemaError``，参数错误使用 ``ValueError``；
专用接口不支持的表或绕过完整范围验证的请求使用 ``NotImplementedError``。
记录级证据不足返回缺失，而非填充、推定或自动切换后端。

新增无关列不影响固定业务输出。结构缓存仅在当前实例内使用；
数据库字段调整后新建实例。索引、DDL 和数据修复仍由 db-manager 执行。
两个目标独立发布，因此实际数据覆盖可能不同，结构一致并不代表已同步。

其他别名与转换：待审批清单
--------------------------

以下仅盘点并保留。用户已批准的表／指标映射及 ``SHSE → SSE`` 已移除；
下列项目需逐项批准后修改，不与非订阅 API 的迁移审批混为一谈。

.. list-table:: 2026-10-30 盘点（datafeed 范围）
   :header-rows: 1
   :widths: 6 34 32 28

   * - 编号
     - 现有转换
     - 删除后的建议行为
     - 影响
   * - A
     - ``industry._scheme`` 将“申万／中信一二三级行业”拆为体系代码和层级；全角括号转半角。
     - 显式传 ``scheme_code`` 与 ``level``，不再解析中文名称。
     - 因子中性化和 mining 的 ``industry_scheme`` 调用需要另行适配。
   * - B
     - 未注明版本的申万同时匹配 ``SW2014/SW2021``；中信同时匹配 ``CITICS/CITIC``。
     - 必须使用数据库中的精确体系代码，不合并版本或拼写。
     - 旧查询可能变为参数错误或无匹配；需明确研究使用的行业版本。
   * - C
     - ``_scheme_label`` 将代码译为中文；行业代码提取数字并转换为 ``industry_value`` 浮点数；数字 industry 输入自动按 value 查找。
     - 使用原始 ``scheme_code/industry_code``；反查完全由显式 ``by`` 决定。
     - 会改变行业输出合同及按数值查询的调用；原始代码的前缀／前导零不再丢失。
   * - D
     - 周期 ``H/S`` 均为半年度；period 和 exchange 输入自动转大写。
     - 半年度只保留 ``H``；参数使用精确大写值。
     - ``S``、``d``、``sse`` 等旧输入需修改；默认 SSE 不受影响。
   * - E
     - 订阅配置将旧 ``database`` 节作为 ``postgres`` 节；仍回退模块内 ``config.local.json/config.json``。
     - 只读取明确的 ``postgres/clickhouse`` 节与独立订阅配置来源。
     - 旧本地配置需要人工迁移；不改写凭据文件。
   * - F
     - 业务输出将 ``security_code/available_at`` 整理为 ``code/datetime``，行业还有 ``sec_name/ind_code/ind_name``。
     - 如批准，另行设计原字段输出合同。
     - 属于既定固定返回结构，建议保留；直接删除会影响所有消费者。

``readers.logical_type`` 的 PG/CH 类型统一、UTC／上海时区处理、缺失值和 dtype 规范化，
以及 ``reference`` 中实际发布 module 值与版本表的对应关系，属于双库执行和 PIT 核验合同，
不是外部查询别名，继续保留。状态接口的“交易状态”和指数接口的“universe”是派生输出的固定标签，
未用于查找或替换数据库字段。

补充检查：待修复清单（2026-10-30）
----------------------------------

本次只检查和登记，不改变接口、不删除兼容。以下包含上表 A—F 的具体化条目。
检查范围为 datafeed 及仓库内直接调用方；使用隔离的内存样本和临时配置复核，
没有连接生产数据库。SQL 参数问题已在 SQL 生成层复核，未据此声称所有真实后端均已执行验证。

.. list-table:: 配置入口
   :header-rows: 1
   :widths: 9 46 45

   * - 编号
     - 现状与位置
     - 待审批修复
   * - R01
     - ``config.py:ConfigManager/get_config/get_database_config`` 与新 loader 并存。旧入口读取新 postgres 配置节时，database 节仍会合入默认 datafeed 库；文件错误会回退默认配置。get_config 只接受首次路径。
     - 统一只读加载入口；旧读写 API 和全局配置状态按迁移审批处理。新 Datafeed 本身没有调用这条默认回退路径，不能混为一谈。
   * - R02
     - ``load_subscription_config`` 仍读取旧 database 节，并回退模块内 config.local.json/config.json；两套加载器对 XDG_CONFIG_HOME 的处理也不同。对应 E。
     - 明确唯一配置格式、查找顺序；删除旧兼容前列出调用方和需迁移配置，不自动改写用户凭据。
   * - R03
     - 未知配置键未报错；statment_timeout_ms 拼错后实际仍使用 120000。int 强制转换把 connect_timeout=1.9 截为 1。pool 清理连接参数时还会丢弃非白名单键。
     - 校验配置键、类型和后端适用范围；拒绝拼写错误、小数和布尔整数，不静默忽略或截断。CH 不应接受看似生效的 PG 连接池配置。

.. list-table:: 参数含义与校验
   :header-rows: 1
   :widths: 9 46 45

   * - R04
     - ``core._require`` 只检查必需键存在；nearest 的未知键被忽略。time_tolernace=1 的样本返回了距输入 681 小时的值，正确 time_tolerance=1 返回缺失。
     - 优先拒绝未知键和错误容器；审批后改为显式关键字参数，不新增请求实体。容差名明确为 tolerance_hours。
   * - R05
     - ``query._query_fields/readers.select_sql`` 只检查 filters 值是标量，绑定类型主要按 Python 类型判断。CH 把 float 绑成 String、bool 绑成 Int64 的字符串 True；无时区 datetime 被按 UTC 处理。
     - 按实际数据库字段类型验证和绑定；用户时间统一先按上海时区解释，不能调用 stored=True 路径。拒绝不支持的类型组合。
   * - R06
     - ``contracts.timestamp/date_only`` 使用宽松 Timestamp 解析和字符串长度判断整日。2024-01-03 与 20240103 的结束边界不同，整数 20240103 被解释为纳秒。limit=1.0 通过整数校验后在 iloc 切片报 TypeError。
     - 规定明确的日期／时间输入格式；整数日期不推断，日期和时刻的端点规则分开。统一 limit、offset、timeout、tolerance 的数值校验。
   * - R07
     - ``metric`` 在时序查询中已经表示物理列，但签名仍允许 None；在交易状态和指数接口中仅接受固定标签“交易状态”和“universe”。
     - 数值接口明确为必填 column；专用接口移除无选择意义的 metric 参数，避免同名参数承担不同职责。
   * - R08
     - log_dir 被丢弃；独立查询的 logger 未使用；cursor 实际可为 Datafeed、reader 或 PG 游标。query_names 不使用实例 table_name，而固定查 security_dim/security_name_history。
     - 删除无效参数前审批；数据源参数命名明确。名称查询与绑定数值表的接口分开，避免用多个行情表重复尝试同一名称查询。
   * - R09
     - start_date/end_date 和 get_latest_date/get_available_dates 实际指 available_at；并非 trade_date 或 period_end。as_of=None 在名称、日历、范围查询中的默认行为不同。codes 与 datetimes 是笛卡尔积，limit 是全局条数。
     - 明确 available_start/end、观察时点、业务日期和报告期；容器语义、全局 limit 和无 as_of 行为写入签名说明。若新增业务日期筛选，仍单独执行 PIT 截点约束。
   * - R10
     - ``industry._scheme/_industry_frame/get_industry_members`` 解析复合中文体系、括号和层级；中信一级行业（2021）的年份被忽略。exact 只约束未指定版本的申万；数字输入可覆盖 by 的选择。对应 A—C。
     - 使用明确 scheme_code/level/industry_code，精确查询不再依赖 exact 模糊开关或参数值类型猜测。中文显示名与查询键分开。
   * - R11
     - ``remark`` 为无固定子结构的 object 字典，存放业务日期、报告期、版本、单位；行业同时在 remark 和独立列存相同字段。status_text 混合状态显示和缺失原因，industry_value 从代码提取数字后转 float。
     - 审批返回合同后，将常用业务字段显式列出或至少固定 remark 子结构；保留原始字符串代码，缺失原因提供固定 reason_code，显示文字不作为程序判断键。
   * - R12
     - ``universe.get_index_universe_panel`` 接受完整时刻，但输出键截为 date。同一天 10:00、15:00 分别查询可得到 A、B，合并查询却只剩 B。
     - 日期接口拒绝带时间输入，或另定以完整观察时点为键的返回合同；不得静默覆盖日内请求。现有返回类型变更需批准。
   * - R13
     - ``contracts.proven`` 缺少 data_quality_status 时默认 accepted；``industry._industry_frame`` 缺少 membership_status 时默认 mapped。前者在证据字典样本中已复现；普通行是否受影响取决于其必需字段校验。
     - 明确哪些状态必须存在；缺失时返回未知或证据错误，不以成功状态补缺。与管理端证据合同一起确认。
   * - R17
     - ``core._period_days/trade_days_offset`` 同时接受 H/S 并自动大写；offset 在非 D 周期下表示周期数，函数名却称 trade_days。对应 D。
     - 统一周期枚举；明确采样周期与交易日偏移的不同含义，删除别名需批准。

.. list-table:: 结构解析与后续依赖
   :header-rows: 1
   :widths: 9 46 45

   * - R14
     - ``readers.logical_type`` 按子串识别类型。Array(Int64) 被识别为 integer、Array(String) 为 text；不同 Decimal 精度都归为 numeric。
     - 双库类型规范化继续保留，但只解析已支持的标量类型和包装；数组／复合类型明确拒绝。需要精度合同的字段保留精度与小数位校验。
   * - R15
     - ``reference._documents/known_scopes/suspension_status`` 解析 payload.selector/evidence 的多层 JSON，部分字段直接索引；结构错误可能变成 KeyError。_documents 检查所有实际列，新增非合同列也可能阻断。
     - 为发布与证据 JSON 明确版本、必需字段、类型和统一错误；业务视图与哈希规范化分开。不能为减少解析而删除发布、撤销、完整范围或哈希核验；合同调整需与 db-manager 协调。
   * - R16
     - ``query.build_nearest_query/build_nearest_in_range_query`` 仍生成旧长表 code/datetime/metric/value SQL；run_query/cursor/conn 仍是 PG 兼容入口，不受双库业务合同保证。
     - 原有迁移清单继续待批准；旧构造函数不得看作新库查询实现。仓库内本次未发现这些旧构造函数的业务调用，不代表外部导入无人使用。
   * - R18
     - backtest 默认 daily_market/收盘价(元)，factor 默认 fundamentals/trade_status，mining、行业中性化和模板仍使用旧表名／行业参数；analyst.naming 仍按多个旧行情表循环取名称。
     - 后续单独适配调用方及文档、测试。不能因存在旧调用而在 datafeed 恢复映射；本轮不修改这些业务模块，也不混入复权成交手数问题的修复。

优先处理会改变查询结果或隐瞒输入错误的 R03—R06、R12—R14。
配置退役、接口重命名及输出结构调整仍需审批；UTC／上海时区转换、双库结果类型统一、
严格 PIT 和完整范围验证必须保留。

迁移清单（尚未实施）
--------------------

* ``FillStrategy`` 及 validation 的空值、日期、去重、修复函数：建议迁入轻量通用预处理模块。
* ``pivot_to_wide/align_to_dates``：建议同归通用预处理。
* ``calculate_returns``：建议迁入分析模块。
* ``ConfigManager.save/set/__setitem__``：配置写入建议归管理端；新查询运行时仅使用只读加载。
* ``run_query/cursor/conn``：继续弃用，不属于固定格式或严格 PIT 保证；尚未删除。
* 公开 SQL 构造及输入组合函数：建议转内部实现；当前导入路径仍保留。

验证与性能
----------

``tests/test_datafeed.py`` 提供离线合同测试；``tests/test_datafeed_integration.py``
通过显式测试配置连接双库，只创建并清理唯一的 ``datafeed_test_*`` 临时空间。
``tests/benchmark_datafeed.py`` 比较 SQL 候选筛选与 Python 批量匹配，先校验结果一致，
再记录预热后五次耗时、P95、Python 峰值内存、传输量和扫描量。
不清理共享数据库缓存，因此首次耗时不能视为严格冷缓存实验。

真实库建成后应按代表性业务负载复测，尤其是长历史、财报修订密度、完整快照范围与同时更新情形。
只读实测不能替代管理端补齐历史可知性证据。

实测默认选择（2026-10-29）
~~~~~~~~~~~~~~~~~~~~~~~~
本机 PG 16.15、CH 25.8.33.6，合成行情为 5000 证券 × 20 交易日，财报为 200 证券 × 20 修订。
下表为预热后五次端到端中位数（毫秒），包含取数、解码、PIT 处理与 DataFrame 构造。
.. list-table:: SQL 候选筛选与 Python 批量匹配
   :header-rows: 1

   * - 后端
     - 场景
     - Python ms
     - SQL ms
     - Python / SQL 峰值内存 MB
   * - postgres
     - single_code_many_dates
     - 10.0
     - 12.7
     - 0.1 / 0.1
   * - postgres
     - market_single_date
     - 4323.4
     - 1150.5
     - 156.3 / 13.7
   * - postgres
     - market_four_dates
     - 4378.8
     - 3719.0
     - 125.2 / 53.9
   * - postgres
     - financial_revisions
     - 245.2
     - 158.8
     - 6.5 / 2.3
   * - clickhouse
     - single_code_many_dates
     - 23.8
     - 43.4
     - 0.1 / 0.1
   * - clickhouse
     - market_single_date
     - 5382.4
     - 4131.5
     - 189.9 / 13.5
   * - clickhouse
     - market_four_dates
     - 7022.4
     - 14284.0
     - 152.1 / 53.8
   * - clickhouse
     - financial_revisions
     - 179.9
     - 514.7
     - 6.6 / 2.3

默认规则只覆盖已测得优势的 nearest-before 点查询：PG 在至少 200 个不同证券时使用 SQL；
CH 在至少 5000 个不同证券且仅一个不同查询时点时使用 SQL。其他形状采用批量 Python，
包括 after 和区间查询。SQL 每批最多 512 个不同输入对，仍由共用逻辑检查版本冲突及整理输出。
参考域保留完整清单／内容核验，不能用一个未核验的 SQL max 替换。

内存是 tracemalloc 测得的 Python 分配峰值，不包含数据库进程 RSS。
扫描量在 PG 使用独立 EXPLAIN ANALYZE 的扫描节点行数，在 CH 使用响应 summary，
不同后端统计口径不同；只用于同一后端方案对照。未执行生产清缓存或索引变更。
这些是初始选择，不代表长历史或其他硬件上的性能保证。

上述性能数据是去除 registry 前的初始实验，查询匹配算法未因此重新测定。
当前接口直接读取物理字段及其原生单位，不再维护订阅端的字段单位映射。
传输字节数为解码结果重新编码 JSON 的估算，不是网络抓包统计。

本轮额外运行了现有回测、事件研究、因子分组、预处理与 mining 测试。
``test_failed_run_updates_manifest_and_audit_log`` 未模拟交易日历，会尝试真实研究数据库连接；
在当前缺少研究配置的环境中失败，使用修改前 datafeed 复测也在该位置失败。
此测试环境依赖单独记录，没有修改 mining 的业务实现或其既有测试。

本机只读样本核对（2026-10-29）：000001.SZ、2026-10-12 至 2026-10-14，
两库后复权收盘价接口均返回具有固定列的空表；这不能区分尚无记录与记录不满足 PIT 证据，
也不代表全库没有数据。同期 SSE 日历均明确抛出 ``PITUnavailableError``。
双库 ``datafeed_test_*`` 临时测试命名空间已全部清理，生产业务数据未改写。
