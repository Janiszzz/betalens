数据订阅模块
============

``betalens.datafeed`` 是 PostgreSQL / ClickHouse 的只读订阅层。
数据库配置持久化、DDL、导入、审计和可信发布由 db-manager 负责。
订阅接口直接查询物理表和字段，共用严格 PIT 与固定 DataFrame 返回规则。

连接配置
--------

配置模板为 ``betalens/datafeed/config.example.json``。
文件只接受 ``backend/postgres/clickhouse``；配置不使用旧 database、logging 或 adjustment 节。
只读加载器为 ``load_subscription_config``；旧 ConfigManager、全局配置和读写配置 API 已删除。

优先级：显式构造参数／db_config > 环境变量 > 所选文件。
文件选择为 config_file > BETALENS_CONFIG > 用户配置目录 betalens/config.json。
Windows 使用 APPDATA，Linux 使用 XDG_CONFIG_HOME，未设置时为 ~/.config。
不再回退模块内 config.local.json/config.json，也不连接内置默认库。
显式文件不存在、格式错误、未知配置键或必需连接字段缺失均报错。
每个实例固定其配置，文件修改仅影响新实例。
新版 JSON 可配置 ``pit_policy``；优先级为构造参数 > 文件 > research。
独立查询函数可通过同名关键字覆盖单次查询，不改变实例策略。

PG 环境变量为 ``BETALENS_PG_HOST/PORT/DATABASE/SCHEMA/USER/PASSWORD``；
CH 为 ``BETALENS_CH_URL/DATABASE/USER/PASSWORD``，后端为 BETALENS_BACKEND。
旧 BETALENS_DB_* 不再接受；私有配置文件不会被程序自动改写。
无文件时可使用完整 db_config 或对应新版环境变量。

.. code-block:: python

   from betalens.datafeed import Datafeed

   with Datafeed("market_daily_post_adjusted", backend="clickhouse",
                 config_file="/path/to/subscription.json") as data:
       result = data.query_time_range(
           codes=["000001.SZ"], metric="close",
           start_date="2024-01-01", end_date="2024-01-31",
           filters={"asset_type": "stock"}, as_of="2024-01-31 16:00:00",
       )

PG 使用连接池与只读 REPEATABLE READ 事务；CH 使用 HTTP 参数绑定、readonly=1 和 FINAL。
CH 无跨请求事务，参考查询先固定发布事件，再核验不可变清单。
连接、超时、版本冲突均抛出异常，不自动切换后端。
订阅接口已移除 logger/log_dir、run_query、cursor、conn 和旧 SQL 构造／输入组合函数。
独立查询函数的 source 参数接受 Datafeed；不再接受原生 PG 游标。

物理表、字段与筛选
------------------

``table_name`` 是实际表名，``metric`` 是实际数值字段；registry 已删除。
不保留逻辑数据集、中文指标及 prev_close 等别名；前收盘价直接使用 previous_close。
metric 必须明确指定；数值保持数据库原生单位，不转换元／万元、股／手或报告期／TTM。

* 后复权行情：market_daily_post_adjusted。
* 未复权及指数原生点位：market_daily_raw。
* 财报：fundamental_periodic，必须同时指定 statement_type/report_type。
* 行业研究查询：security_industry_membership，联合 industry_dim/industry_scheme_dim。
* 严格行业查询：stock_industry_membership_versions。
* 研究状态／风险查询：trade_status_daily，联合 suspension_event/上市范围。
* 严格状态查询：stock_suspension_intervals_versions。
* 研究日历：trade_calendar；严格日历：trade_calendar_versions。
* 指数完整成分：index_membership。

filters 是物理字段到非空标量的字典，仅表示相等条件。
值按目标字段类型校验并绑定，支持的时间条件遵循下面的上海输入规则。
不指定 asset_type 时不会替调用方筛选资产；因子 YAML 使用明确的 query_filters。
普通时序表要求 security_code/event_date/available_at/availability_basis/data_quality_status。
参考域和成分集合必须使用专用接口，不能经普通数值查询绕过完整范围证明。

.. code-block:: python

   with Datafeed("fundamental_periodic") as data:
       result = data.query_nearest_before({
           "codes": ["000001.SZ"], "datetimes": ["2024-04-30 18:00:00"],
           "metric": "revenue", "time_tolerance": 24 * 365,
           "filters": {"statement_type": "income", "report_type": "1"},
       })

行业版本与分级
--------------

行业查询使用 scheme_code 和 level，level 仅接受整数 1、2、3。
精确版本示例为 SW2014、SW2021、CITICS；中文复合名称、exact 开关和 CITIC 拼写兼容已移除。
研究模式未指定 scheme_code 时，只考虑申万：选择观察时点已可得且已生效的最新方案。
方案日期从 industry_scheme_dim 读取，不解析方案名称，不按当前年份猜测版本。
各层级共用同一方案选择；公司归属采用 [valid_from, valid_to)，缺失归属留空，重叠报错。
源端行业层级字段为 L1/L2/L3，查询 level 分别为整数 1/2/3。
一次性补录的方案日期是研究假设，不升级为严格内容版本证明；SW2014 归属未下载时仍为空。

strict 模式须显式查询 stock_industry_membership_versions，仍核验可信发布、完整范围、
classification_version 与边界证据。版本表在 research 下也不绕过发布核验，且不自动改读普通表。

.. code-block:: python

   with Datafeed("security_industry_membership") as data:
       level1 = data.query_industry(["000001.SZ"], ["2024-04-30 10:00"], level=1)
       level3 = data.query_industry(["000001.SZ"], ["2024-04-30 10:00"], level=3)
       specified = data.query_industry(
           ["000001.SZ"], ["2024-04-30 10:00"], scheme_code="SW2021", level=2,
       )

因子 YAML 的行业选择也是结构化字段，无需解析中文名称：

.. code-block:: yaml

   factor_spec:
     table_name: market_daily_post_adjusted
     query_filters: {asset_type: stock}
     inputs: {close_wide: close}
     industry_scheme: {scheme_code: null, level: 1}
     industry_inputs:
       subindustry_wide: {scheme_code: null, level: 3}
     backtest_metric: close

时间输入、端点与日历
--------------------

用户无时区 datetime 按 Asia/Shanghai 解释；带时区输入尊重其偏移并转 UTC。
数据库返回的无时区 timestamp 按 UTC 存储解释，不能再次当成上海时间输入。
例如上海 15:00 与同日 07:00Z 查询同一时刻，filters、nearest、范围和 as_of 规则一致。
旧时间输出列保持上海本地无时区 datetime64[ns]。

字符串仅接受 YYYY-MM-DD 或 ISO 日期时间，精度最多微秒；不接受 YYYYMMDD、
数字日期、相对日期或按字符串长度猜测的格式。date 对象表示整日，datetime 表示明确时刻。
范围的纯日期结束值包含整天，带时间的结束值包含该时刻。
start_date/end_date 当前筛选 available_at；业务日和报告期仍分别是 event_date/period_end。
get_available_dates/get_latest_date 返回有证据的可得时刻。

before 包含锚点；after 严格晚于锚点。区间 before 为闭区间，区间 after 为开区间。
time_tolerance 单位为小时，与区间取交集；bool、小数 limit/offset 等错误输入明确拒绝。
nearest 的 codes × datetimes 为笛卡尔积，保留重复输入；范围 limit 为全局条数。
after 的结果在输入时点尚不可知，仅用于后续观察。

``get_absolute_trade_days`` 和 ``trade_periods_offset`` 默认只读本地 trade_calendar，
研究模式信任源端静态日历，因此可查询观察时点之后的日历安排；仍要求自然日完整覆盖。
严格模式须同时指定 pit_policy="strict"、table_name="trade_calendar_versions"。
周期严格为 D/W/M/Q/H/Y；半年度统一为 H，表示自然半年最后交易日。
旧 S、自动大小写转换及 trade_days_offset 名称已移除。
trade_periods_offset 的 offset 表示交易周期数，period=D 时才是交易日数；保留输入时间。
exchange 使用精确 SSE/SZSE/BSE，默认 SSE；不保留 SHSE 别名。
缺自然日、范围证据不足及超出覆盖范围时不会推定休市或使用在线日历。

指数面板 get_index_universe_panel 的键为 date，因此仅接受 YYYY-MM-DD 或 date。
传入 datetime（含午夜 Timestamp）明确报错，防止两个日内观察结果覆盖同一日期键。
日内观察使用 get_index_universe 或 get_index_universe_date 的单时点接口。

PIT 策略与返回合同
------------------

默认 research 信任源端历史数据，接受已登记的次日、公告次日和生效日期推定可得时间。
查询以 available_at 比较，未知标签、空可得时间、下载及入库时刻均不能直接放行。
范围 remark 记录 pit_policy 和 availability_basis；nearest 保留原有六列。
研究模式不重建历史修订，同一时间的不同值仍报冲突，不按哈希或下载顺序猜测胜者。
后复权读取现有表，不重新计算复权、不根据因子跳变创造股息或股份。
缺因子的后复权价格保持缺失，原始表仍可独立使用。

strict 模式仍要求 accepted/verified 质量状态及 source_version_publication_time 或
verified_historical_version 内容证明；推定依据不满足门槛。
财报继续区分报告期、公告、报告类型；同一可得时刻的多个报告期选择最新报告期。

参考域先排除未发布／撤销版本，再核验修订清单、范围、内容／键哈希与行数。
完整范围及非空记录均需历史可得性证明；范围按整体选择，不跨版本拼值。
指数缺少 snapshot_complete/snapshot_member_count 完整性证据时明确报错；
空快照缺少独立范围表示时同样不能解释为已知空集合。订阅连接不增列或补造证据。
完整标记只能由管理端依据完整源响应或连续分页链补录，不能凭权重和或行数猜测。

交易状态为 1/0/-1 或 NA；研究模式可采用带依据的源端日频状态。
无停牌行不能自动解释为正常；日内时段无法解析时返回未知。
新增 query_trade_risks(source, table_name, codes, dates, *, pit_policy=None)，实例方法为
query_trade_risks(codes, dates)，返回 code/datetime/is_listed/is_st/is_suspended/name/remark。
三个标志均为可空 Int8。ST 与停牌分别表达；历史名称推断 ST 时在 remark 注明假设。
风险接口不决定成交，也不因 ST 或未知状态拒绝普通行情查询。
停牌区间左闭右开；纯日期观察按当日 00:00 解释。
query_names 未传 as_of 时查询当前展示名称，传入时只返回有证据的历史名称，缺失不回填当前名。

* nearest：code/input_ts/datetime/diff_hours/{metric}/name。
* 范围：datetime/code/name/metric/value/remark。
* 交易状态：code/datetime/value/status_text/name，value 为可空 Int8。
* 行业：code/query_date/effective_dt/sec_name/industry_value/remark/ind_name/ind_code/scheme。
* 指数单日为代码列表，面板为 {date: set[str]}，日历为 list[date]。

两库统一列序、float64 数值、缺失值、时间与稳定排序，空结果保留完整结构。
remark 包含业务日期、报告期、版本、表名、原生单位及报表类型。
PG/CH 类型只解析明确支持的标量及 Nullable/LowCardinality 包装，数组、复合和未知必需类型报错。
Decimal 原始精度保留在执行层；meta_column 明确声明 precision/scale 时交叉核验，
条件值超出实际精度时拒绝截断。无单位或精度条目时不猜测语义。

SchemaError 表示缺表／字段、类型或精度合同错误；VersionConflictError 表示版本冲突；
PITUnavailableError 表示完整覆盖或默认行业版本证据不足；参数错误为 ValueError。
新增无关列不会改变普通时序返回结构；结构调整后新建实例。
参考发布 JSON 的完整合同和错误统一仍是 R15 的待办，不能因此放松哈希验证。

实施状态与后续事项
------------------

已实施 B/D/E、R01/R02、R05/R06、R08 的订阅日志参数清理、R10/R12/R13/R14/R16/R17，
并将 R18 的仓库内调用、因子配置及 Dashboard 默认字段改为新表字段。
配置写入和旧原生 SQL 兼容入口已经删除。

其余接口改名、nearest 参数字典的严格未知键校验（R04）、metric 固定标签清理（R07）、
时间参数命名调整（R09）、remark／行业数值代码及完整输出合同调整（R11/F）、参考 JSON 合同（R15）
仍待后续批准。FillStrategy、validation、pivot_to_wide/align_to_dates 和 calculate_returns
尚留原导入路径，通用预处理／分析模块迁移未在本轮实施。

直接以后复权价格计算成交手数仍是回测引擎缺陷。本轮不修改回测及因子业务计算，
成交价格、手数及权重换算另行修复；查询失败后按正常交易继续等问题也不在本轮修复。
严格 PIT 程序正确与生产数据证据齐备分别验收，测试通过不等于整个回测链路已满足 PIT。

验证与性能
----------

``tests/test_datafeed.py`` 提供离线合同测试；``tests/test_datafeed_integration.py``
通过显式测试配置连接双库，只创建并清理唯一的 ``datafeed_test_*`` 临时空间。
``tests/benchmark_datafeed.py`` 比较 SQL 候选筛选与 Python 批量匹配，先校验结果一致，
再记录预热后五次耗时、P95、Python 分配峰值、进程 RSS 峰值、传输量和扫描量。
不清理共享数据库缓存，因此首次耗时不能视为严格冷缓存实验。

真实库建成后应按代表性业务负载复测，尤其是长历史、财报修订密度、完整快照范围与同时更新情形。
只读实测不能替代管理端补齐历史可知性证据。

实测默认选择（2026-11-08）
~~~~~~~~~~~~~~~~~~~~~~~~
5000 证券 × 20 行情日、200 证券 × 20 财报版本，预热后五次完整计时。
下表为端到端中位数；完整 P95、RSS、传输与扫描量见 :doc:`../plans/datafeed-research-acceptance-20261108`。

.. list-table:: SQL 与 Python 查询中位数（毫秒）
   :header-rows: 1

   * - 后端
     - 场景
     - Python
     - SQL
   * - postgres
     - single_code_many_dates
     - 9.0
     - 11.0
   * - postgres
     - market_single_date
     - 2931.3
     - 1052.9
   * - postgres
     - market_four_dates
     - 3976.8
     - 4289.2
   * - postgres
     - financial_revisions
     - 169.9
     - 384.8
   * - clickhouse
     - single_code_many_dates
     - 15.8
     - 25.2
   * - clickhouse
     - market_single_date
     - 3456.8
     - 6638.9
   * - clickhouse
     - market_four_dates
     - 4173.7
     - 23006.8
   * - clickhouse
     - financial_revisions
     - 173.6
     - 403.9

默认 SQL 仅用于 PostgreSQL 行情 nearest-before 点查询：至少 5000 个不同证券，
不超过四个不同查询时点。单截面 SQL 明显更快；四时点耗时差异小于 10%，
SQL 分配峰值约为 60 MB，而批量 Python 约为 140 MB，因此保留 SQL。
财报、小批量、after／区间和 ClickHouse 默认使用 Python；不扩大到未测长历史。
SQL 每批最多 512 个不同输入对，仍由共用逻辑核验同时间冲突并整理返回结构。
严格参考域保留完整清单及内容核验，不用未核验的 SQL max 替代。

分配峰值由 tracemalloc 测得，RSS 峰值采样客户端进程，含样本和解释器、不含数据库服务。
传输字节为解码记录重编码 JSON 的估算；PG 扫描量来自独立 EXPLAIN ANALYZE，CH 来自响应 summary。
首次耗时不代表严格冷缓存；未清理共享服务缓存，也未调整生产索引。
