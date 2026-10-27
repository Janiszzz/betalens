# 本地映射核查与字段补全计划

日期：2026-10-24。账号分权计划暂不执行。本文件记录只读检查结果与后续实施方案，不表示已经修改映射、代码、表结构或业务数据。

## 1. 本次已完成的核查

- 读取 `/home/janiszzz/.config/betalens/db-manager/mappings/` 和同级 `tables/` 的全部 44 个 YAML，覆盖 22 张本地表。
- 对照 25 份 Tushare 官方文档的完整输出字段表，包括非默认字段；25 份本地基础合同的字段名集合全部一致。4 个 VIP 查询复用对应财报合同，共计 29 个已登记查询。此结论不代表实际权限、类型、单位、返回非空率及存储值已经全部验证。
- 核对执行代码、界面和审计判据。没有发现 YAML 指向本地不存在列、已登记源合同不存在字段或完全重复的字段连接；仍发现生产依赖、字段语义和版本计算问题。
- 对存在无直接映射列的表，只读查询 PG/CH，每表每库最多 200 行，优先使用覆盖统计所示的最新事件日期。行情样本日期为 2026-10-16；财报样本报告期为 2024-12-31。没有执行全量空值扫描，也没有调用需要 Token 的数据接口。

按 YAML 的 `system` 标记统计，共有 752 个业务列，其中 65 个没有直接连接。这不是 65 个确定缺陷：其中包含常量、跨表复制、内置计算、版本号以及仅适用于后复权表的列。

| 本地表 | 业务列数 | 有 YAML 连接 | 无直接连接的业务列 |
|---|---:|---:|---|
| corporate_action | 8 | 4 | event_type、old_name、cash_dividend、share_dividend |
| dividend_event | 17 | 16 | record_version |
| fund_nav_daily | 9 | 9 | 无 |
| fund_nav_observation | 10 | 9 | record_version |
| fund_profile | 25 | 25 | 无 |
| fund_shares_daily | 3 | 3 | 无 |
| fundamental_periodic | 494 | 492 | statement_type、version_id |
| index_membership | 4 | 4 | 无 |
| index_profile | 13 | 13 | 无 |
| industry_dim | 8 | 7 | scheme_code |
| industry_membership_record | 13 | 11 | scheme_code、record_version |
| industry_scheme_dim | 3 | 1 | scheme_name、description |
| market_daily_post_adjusted | 43 | 15 | 28 列，主要是原始宽表继承列及复权说明 |
| market_daily_raw | 43 | 30 | asset_type、average_price、pcf、4 个调整说明列及 6 个 raw_* 列 |
| security_code_history | 5 | 4 | valid_from |
| security_dim | 7 | 6 | asset_type |
| security_industry_membership | 6 | 4 | scheme_code、membership_status |
| security_name_history | 7 | 6 | record_version |
| stock_profile | 17 | 17 | 无 |
| suspension_event | 5 | 4 | record_version |
| trade_calendar | 4 | 4 | 无 |
| trade_status_daily | 8 | 3 | asset_type、is_tradable、status_name、status_source、confidence |

全量源字段是否保留，应按“该字段是否进入至少一张合适粒度的类型化表”检查，而不是要求每张投影表都重复保存源查询全部字段。例如 stock_profile 承接完整股票档案，security_dim 是其统一证券投影。

## 2. 确定发现与补全边界

### 2.1 页面显示与实际生产路径不一致

MappingEditor 的右侧来源只统计字段连接，没有表达常量、系统字段、复制和计算节点。内置或混合映射又跳过本地列反向缺口检查。

实际双库样本中，后复权表的成交量、均价、估值、股本、复权方法等已有值；交易状态的五个派生列也有值。财报的 statement_type/version_id、行业的 scheme_code 等同样由代码生成。不能将这些列统一显示为缺映射。

`field_audit` 的 `verified_stored` 当前只要求“源字段出现在归档且某个目的表有表级验证”，没有检查目标库、业务行、目的列、采用来源和转换后数值；文件任务也不能仅凭存在切片就算响应验证成功。

### 2.2 后复权单表任务缺少完整依赖

`market_daily_post_adjusted.yaml` 没有 daily_basic 绑定，任务编译也不会自动为它补齐该依赖。单独更新后复权表，与同时更新原始表和后复权表，可能得到不同的估值、股本等列完整度。现有样本中有值不代表这个单表更新路径已经正确。

必须显式声明行情、daily_basic、复权因子及名称查询的输入依赖。依赖可以拉取并参与计算，但不能因此写入用户没有选择更新的表。

### 2.3 无来源、不适用与不能推断的日期

- 当前注册接口与已核对 daily_basic 文档没有市现率 pcf。不能将 ETF 申赎篮子 PCF 当作市现率；本轮不引入新的市现率因子计算，只登记缺失能力和潜在财务输入。
- bse_mapping 的 list_date 是上市日期，不是代码切换生效日期。security_code_history.valid_from 的两个库各 200 条样本均为空；不能为消除 NULL 直接把 list_date 填进去。
- 原始行情表中的复权说明和 raw_* 辅助列目前没有赋值路径，属于两表共用宽表设计带来的不适用列，应说明适用范围，不制造数据。
- corporate_action.old_name 没有直接源字段。只有历史名称区间能证明前后关系时，才可派生旧名称；第一条历史或区间不完整时保留未知。
- corporate_action.share_dividend 没有现成生产路径。按其“送股比例”语义，应对应 dividend.stk_bo_rate；stk_div 是送转合计，stk_co_rate 是转增比例，不能混为一列。[分红文档](https://tushare.pro/document/2?doc_id=103)
- corporate_action 的分红日期当前使用 end_date（分红年度），改名 effective_date 又接 end_date（名称区间结束），均需按事件含义修正。分红保留完整明细及报告期/公告/除权/派息日期；已实施分红摘要以 ex_date 为生效依据，缺少实际日期时不能拿报告期替代。

### 2.4 财报字段与修订版本

- income.rd_exp 是“研发费用”，fina_indicator.rd_exp 是“研发投入合计”。当前同名列元信息不能准确表达两者。尽管 statement_type 主键避免了直接合成同一行，仍需拆分语义列。[利润表](https://tushare.pro/document/2?doc_id=33)、[财务指标](https://tushare.pro/document/2?doc_id=79)
- 现有财报代码先用 f_ann_date 覆盖 announcement_date 计算 version_id，再补回 ann_date/f_ann_date/end_type。纯函数复现确认：仅 ann_date 或仅 end_type 变化时 version_id 不变，存在修订覆盖风险。
- 财报宽表按 statement_type 分行，某接口不提供的其他报表列属于不适用，不能据此认定应把所有报表强行合为一行。

### 2.5 净值版本与空表证据不足

fund_nav_daily 只有 security_code/nav_date 键。同一净值日存在不同公告版本时，当前最终合并路径会按顺序保留首个非空值，也可能跨版本补空；不是明确的“最新公告版本”规则。

应由 fund_nav_observation 保存完整版本，再生成 daily 当前值；公告日期不同取较晚公告，同公告日不同数值或缺公告依据时保留冲突，不能任选一条。

dividend_event、fund_nav_observation、industry_membership_record、security_name_history、suspension_event 本次两库查询均未取到记录，应标为无样本、待导入验证，不能由空表推导映射正确或错误。

### 2.6 新查询与单位核查

- bak_daily.avg_price 可作为股票均价的独立来源或交叉核验；该接口覆盖约从 2017 年中开始。其股本单位为亿，不能套 daily_basic 的万股转换；部分量额单位未在表中明确，未验证前禁止自动并入统一行情。[备用行情](https://tushare.pro/document/2?doc_id=255)
- etf_basic 可补 ETF 生命周期、扩位简称、基准指数和投资通道。建立 etf_profile 完整承接其字段，再投影到统一证券表，避免把 fund_basic.status 与 ETF 上市状态混淆。[ETF 档案](https://tushare.pro/document/2?doc_id=385)
- 现有两条指数样本的 average_price 为成交额/成交量量级，不能当成指数点位均价。当前代码已对指数排除该计算，但旧数据尚需形成独立修正预览；不能仅靠新代码断言历史行已经修正。
- 行业分类应继续显式固定版本。检查已确认当前请求生成器已设置 index_classify.src=SW2021，并对成分默认请求 Y/N；本轮不将其误报为尚未修复的默认参数缺陷。
- 部分 YAML 的 source_unit 保存的是字段描述，例如“股票代码”“开盘价”；须区分说明与实际计量单位，并逐接口核对倍率，而不是按字段同名套用转换。

## 3. 后续实施方案

### A. 建立逐列生产规则与双向审计

扩展 YAML 的可版本化本地列生产规则：直接映射、常量、跨表复制、已注册计算、系统生成、不适用、暂无来源。每项记录资产/报表适用范围、输入依赖、单位和处理器版本；复杂计算只引用已注册处理器，不执行任意表达式。

现有 v1 YAML 先生成迁移预览，再发布新版本；保留用户修改、旧任务规则和归档。缺少生产者元信息不再因为“表是 builtin”而免检。源字段允许明确标记“在另一张类型化表完整保存”，并引用具体列。

新增只读审计接口 `GET /api/tables/{table}/field-coverage`，合并返回本地列、生产者、源字段、依赖、适用范围、空值原因和分目标库证据。保留现有 field-audit 接口兼容，并明确区分源字段审计与本地列覆盖。

字段实证按任务、目标库、表、列和业务键核对：源有值、转换成功、冲突处理时被采用、提交后回读一致。统计适用数、源空值数、选中数、写入/保留数、核验数和不一致数；只有表级哈希时标为“表已核验”。不把现有值保留、拒收或源端全 NULL 伪装成新源字段已写入。

### B. 修复数据路径与语义

1. 后复权依赖显式化，使单表任务与多表任务基于相同源归档得到相同结果，转换只执行一次。
2. 显式登记派生字段及输入；无依据字段保留 NULL 与原因，不用默认零或随意日期填满。
3. 完整业务内容规范化后再计算财报版本，包含 ann_date、f_ann_date、end_type 和修订标识；VIP 与普通接口的同内容不制造重复版本。
4. 增加独立研发费用与研发投入列；旧 rd_exp 保留并标为待迁移/弃用，历史值按 statement_type 与归档证据迁移，来源不明则不猜测。
5. 保留净值观察版本，再按明确公告规则生成 daily 值；同公告日冲突阻止自动发布当前值。
6. 分红、改名摘要由对应明细生成，纠正日期和送股/转增语义。原始表与后复权表的不适用列不强行填值。

### C. 补充数据字典与来源能力

保留已对齐的 25 份基础合同及 4 个 VIP 变体，新增 etf_basic 和 bak_daily 合同。两个新查询各自用完整业务表承接全部输出字段；bak_daily 使用独立 stock_quote_supplement_daily，语义及单位验证后才允许向统一行情投影。

实施时执行显式 fields 的小样本请求，记录非默认字段、真实返回、权限与覆盖起点。无权限、源端未返回、文档单位不明确均单独报告；不购买权限，不把文档存在当作账号已可用。

### D. 页面与历史修正

映射图保留直接连线，增加常量、计算和依赖节点；本地列显示来源类型、输入、公式版本或留空原因。“仅看未映射”只筛选适用且缺少有效生产者的列，另提供“未验证”“源端空值”“不适用”筛选。

展示逐表、逐资产/报表类型、逐目标库的覆盖表格；空表显示待验证，不能显示 100% 完整。

历史修正优先从归档重审，旧版本号通过迁移记录关联，不覆盖旧审计或旧确认哈希。缺少归档才生成精确的重新拉取请求。所有生产补写、类型/列迁移均独立预览并确认，本轮不恢复历史全量补全，也不执行账号分权计划。

## 4. 验收条件

- 22 张表的所有本地列都有明确生产规则或留空原因；每个官方源字段有具体类型化存储去向，纯归档保留不计完整落表。
- 单独更新后复权表仍能获得应有的每日指标；不向未选择的原始表写入。
- 字段重排、单位倍率、零值、缺字段、源 NULL、主备未采用、列转换错误和单库失败都有正确审计结果。
- 修改 ann_date、f_ann_date、end_type 或财报内容会得到相应新版本；重复执行稳定。同名不同义研发字段不混写。
- 同净值日不同公告版本不拼接；同公告日冲突不静默选首条。
- bse 上市日期不会被误填为代码变更日期，指数汇总成交均价不会被称为指数价格，分红报告期不会被称为除权日。
- 股票/基金/指数、四类财报、申万/中信、名称/分红/停复牌明细均有对应样本；无权限或无样本明确列出，不能算验证通过。
- Playwright 验证新增生产节点、字段筛选、YAML 保存重载、单表更新依赖、审计缓存及确认前无业务写入；隔离双库验证入库、重放与回退。
