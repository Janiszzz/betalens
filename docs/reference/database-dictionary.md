# 当前数据库逐字段字典

采集时间：**2026-11-05T20:05:46.289427+08:00**。两库各 **41 张表、1,359 个字段位置**；相同字段名在不同表重复计数。

原生类型、可空性、PG 主键和 CH 排序键来自本轮只读目录查询；中文说明、单位及生成规则来自当前本机 YAML 和管理器代码。单位属于声明，不表示每列都已有实际非空数据。

入口：[数据库结构讲解](../guide/database-schema.md) · [Excel 字典](../_static/database-dictionary-20261105.xlsx) · [完整结构快照](../_static/database-schema-20261105.json)

阅读方式：`PK` 表示实际 PG 主键成员；“可空”为 PG/CH 顺序；CH 排序键不提供 PG 式唯一性约束。空单位显示“未声明/不适用”，不猜测具体量纲。源字段旁的乘数是当前映射声明，已入库值不能重复乘。

本字典列出全部实际字段，包括已弃用但仍存在的字段；这不赋予它们新的查询别名或兼容保证。

## 表目录

| 分组 | 表 | 中文名 | 字段数 | PG / CH 逻辑行数 |
| --- | --- | --- | ---: | ---: |
| 证券与档案 | [security_dim](#security-dim) | 证券基本信息 | 21 | 50,428 / 50,428 |
| 证券与档案 | [stock_profile](#stock-profile) | 股票档案 | 31 | 5,926 / 5,926 |
| 证券与档案 | [fund_profile](#fund-profile) | 基金档案 | 39 | 32,639 / 32,639 |
| 证券与档案 | [index_profile](#index-profile) | 指数档案 | 27 | 11,863 / 11,863 |
| 证券与档案 | [etf_profile](#etf-profile) | ETF档案 | 28 | 1,912 / 1,912 |
| 证券与档案 | [security_code_history](#security-code-history) | 证券代码对照 | 19 | 248 / 248 |
| 行情与财报 | [market_daily_raw](#market-daily-raw) | 原始日频行情 | 57 | 360,561 / 360,561 |
| 行情与财报 | [market_daily_post_adjusted](#market-daily-post-adjusted) | 后复权日频行情 | 57 | 360,561 / 360,561 |
| 行情与财报 | [stock_quote_supplement_daily](#stock-quote-supplement-daily) | 股票备用行情 | 45 | 106,000 / 106,000 |
| 行情与财报 | [fundamental_periodic](#fundamental-periodic) | 定期财务数据 | 510 | 0 / 0 |
| 基金观测 | [fund_nav_daily](#fund-nav-daily) | 基金日净值 | 23 | 488,682 / 488,682 |
| 基金观测 | [fund_nav_observation](#fund-nav-observation) | 基金净值公告记录 | 24 | 488,682 / 488,682 |
| 基金观测 | [fund_shares_daily](#fund-shares-daily) | 基金份额 | 17 | 34,341 / 34,341 |
| 行业与指数 | [industry_scheme_dim](#industry-scheme-dim) | 行业分类体系 | 17 | 2 / 2 |
| 行业与指数 | [industry_dim](#industry-dim) | 行业分类 | 22 | 928 / 928 |
| 行业与指数 | [industry_membership_record](#industry-membership-record) | 行业归属原始记录 | 27 | 11,435 / 11,435 |
| 行业与指数 | [security_industry_membership](#security-industry-membership) | 证券行业归属 | 20 | 34,305 / 34,305 |
| 行业与指数 | [index_membership](#index-membership) | 指数成分 | 18 | 2,954,624 / 2,954,624 |
| 日历与事件 | [trade_calendar](#trade-calendar) | 交易日历 | 18 | 60 / 60 |
| 日历与事件 | [trade_status_daily](#trade-status-daily) | 每日可交易状态 | 22 | 966,219 / 966,219 |
| 日历与事件 | [corporate_action](#corporate-action) | 证券事件 | 22 | 148,074 / 148,074 |
| 日历与事件 | [dividend_event](#dividend-event) | 分红方案 | 31 | 3 / 3 |
| 日历与事件 | [suspension_event](#suspension-event) | 停复牌事件 | 19 | 196 / 196 |
| 日历与事件 | [security_name_history](#security-name-history) | 证券名称历史 | 21 | 5,939 / 5,939 |
| 参考域版本 | [stock_industry_membership_versions](#stock-industry-membership-versions) | 股票历史行业分类版本 | 30 | 0 / 0 |
| 参考域版本 | [trade_calendar_versions](#trade-calendar-versions) | 交易日历版本 | 25 | 0 / 0 |
| 参考域版本 | [stock_suspension_intervals_versions](#stock-suspension-intervals-versions) | 股票停牌区间版本 | 30 | 0 / 0 |
| 参考域发布凭证 | [reference_revision](#reference-revision) | 参考域修订头 | 8 | 0 / 0 |
| 参考域发布凭证 | [reference_scope](#reference-scope) | 完整范围凭证 | 9 | 0 / 0 |
| 参考域发布凭证 | [reference_publication_event](#reference-publication-event) | 可信发布事件 | 7 | 0 / 0 |
| 管理与审计 | [meta_table](#meta-table) | 数据表元信息 | 12 | 0 / 0 |
| 管理与审计 | [meta_column](#meta-column) | 字段元信息 | 14 | 0 / 0 |
| 管理与审计 | [meta_source_mapping](#meta-source-mapping) | 来源字段映射 | 14 | 0 / 0 |
| 管理与审计 | [meta_source_capability](#meta-source-capability) | 数据源能力 | 9 | 0 / 0 |
| 管理与审计 | [import_task](#import-task) | 导入任务 | 10 | 0 / 0 |
| 管理与审计 | [import_source_archive](#import-source-archive) | 源数据归档 | 9 | 0 / 0 |
| 管理与审计 | [import_target_status](#import-target-status) | 目标库导入状态 | 10 | 0 / 0 |
| 管理与审计 | [import_batch_receipt](#import-batch-receipt) | 批次镜像与回执 | 9 | 12,282 / 8,394 |
| 管理与审计 | [import_row_audit](#import-row-audit) | 行级变更审计 | 11 | 0 / 0 |
| 管理与审计 | [import_rollback](#import-rollback) | 任务回退 | 10 | 0 / 0 |
| 管理与审计 | [dataset_coverage](#dataset-coverage) | 数据覆盖范围 | 7 | 349,792 / 349,792 |

<a id="security-dim"></a>

## security_dim

**证券基本信息**：统一证券标识和当前档案。实际主键是 security_code + asset_type；跨资产连接不能只按代码。当前名称不等于历史名称。

定义说明：股票、基金和指数统一证券标识及生命周期。声明粒度：每个证券一行。实际字段 **21** 个；逻辑行数 PG **50,428**、CH **50,428**。

PG 主键：`security_code, asset_type`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code, asset_type`。

声明时间字段：`未单独声明`；时间规则：`snapshot`。声明主键：`security_code, asset_type`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | 证券代码 | text | String | 否/否 | 未声明/不适用 | tushare:stock_basic.ts_code × 1 [asset_type=stock]；tushare:fund_basic.ts_code × 1 [asset_type=fund]；tushare:index_basic.ts_code × 1 [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `asset_type` **PK** | 资产类型；实际业务维度，如 stock/fund/index；证券总表主键还包含该维度。债券为查询预留维度，当前任务未导入。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(asset_scope)：按查询合同的资产范围生成 |
| `name` | 证券名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.name × 1 [asset_type=stock]；tushare:fund_basic.name × 1 [asset_type=fund]；tushare:index_basic.name × 1 [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `exchange` | 交易所 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.exchange × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `list_date` | 上市日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:stock_basic.list_date × 1 [asset_type=stock]；tushare:fund_basic.list_date × 1 [asset_type=fund]；tushare:index_basic.list_date × 1 [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `delist_date` | 退市日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:stock_basic.delist_date × 1 [asset_type=stock]；tushare:fund_basic.delist_date × 1 [asset_type=fund]；源字段 [asset_type=stock]；源字段 [asset_type=fund] |
| `status` | 当前状态 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.list_status × 1 [asset_type=stock]；tushare:fund_basic.status × 1 [asset_type=fund]；源字段 [asset_type=stock]；源字段 [asset_type=fund] |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="stock-profile"></a>

## stock_profile

**股票档案**：股票基础信息当前快照，保留 stock_basic 字段；上市日期不等于代码切换日期。

定义说明：股票档案；完整保存查询字段和版本。声明粒度：源记录业务粒度。实际字段 **31** 个；逻辑行数 PG **5,926**、CH **5,926**。

PG 主键：`security_code`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code`。

声明时间字段：`未单独声明`；时间规则：`snapshot`。声明主键：`security_code`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | TS代码 | text | String | 否/否 | 未声明/不适用 | tushare:stock_basic.ts_code × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `symbol` | 股票代码 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.symbol × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `name` | 股票名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.name × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `area` | 地域 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.area × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `industry` | 所属行业 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.industry × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `fullname` | 股票全称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.fullname × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `enname` | 英文全称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.enname × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `cnspell` | 拼音缩写 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.cnspell × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `market` | 市场类型（主板/创业板/科创板/CDR） | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.market × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `exchange` | 交易所代码 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.exchange × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `curr_type` | 交易货币 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.curr_type × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `list_status` | 上市状态 L上市 D退市 G过会未交易 P暂停上市 UN未上市 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.list_status × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `list_date` | 上市日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:stock_basic.list_date × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `delist_date` | 退市日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:stock_basic.delist_date × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `is_hs` | 是否沪深港通标的，N否 H沪股通 S深股通 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.is_hs × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `act_name` | 实控人名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.act_name × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `act_ent_type` | 实控人企业性质 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:stock_basic.act_ent_type × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="fund-profile"></a>

## fund_profile

**基金档案**：基金基础信息当前快照，包含场内/场外基金；不要把所有基金视为交易所 ETF。

定义说明：基金档案；完整保存查询字段和版本。声明粒度：源记录业务粒度。实际字段 **39** 个；逻辑行数 PG **32,639**、CH **32,639**。

PG 主键：`security_code`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code`。

声明时间字段：`未单独声明`；时间规则：`snapshot`。声明主键：`security_code`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | 基金代码 | text | String | 否/否 | 未声明/不适用 | tushare:fund_basic.ts_code × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `name` | 简称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:fund_basic.name × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `management` | 管理人 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:fund_basic.management × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `custodian` | 托管人 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:fund_basic.custodian × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `fund_type` | 投资类型 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:fund_basic.fund_type × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `found_date` | 成立日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:fund_basic.found_date × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `due_date` | 到期日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:fund_basic.due_date × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `list_date` | 上市时间 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:fund_basic.list_date × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `issue_date` | 发行日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:fund_basic.issue_date × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `delist_date` | 退市日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:fund_basic.delist_date × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `issue_amount` | 发行份额(亿) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 份 | tushare:fund_basic.issue_amount × 100000000（源单位 亿份） [asset_type=fund]；源字段 [asset_type=fund] |
| `m_fee` | 管理费 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fund_basic.m_fee × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `c_fee` | 托管费 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fund_basic.c_fee × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `duration_year` | 存续期 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fund_basic.duration_year × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `p_value` | 面值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fund_basic.p_value × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `min_amount` | 起点金额(万元) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:fund_basic.min_amount × 10000（源单位 万元） [asset_type=fund]；源字段 [asset_type=fund] |
| `exp_return` | 预期收益率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fund_basic.exp_return × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `benchmark` | 业绩比较基准 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:fund_basic.benchmark × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `status` | 存续状态D摘牌/已到期 I发行 L已上市/存续中 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:fund_basic.status × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `invest_type` | 投资风格 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:fund_basic.invest_type × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `type` | 基金类型 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:fund_basic.type × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `trustee` | 受托人 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:fund_basic.trustee × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `purc_startdate` | 日常申购起始日 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:fund_basic.purc_startdate × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `redm_startdate` | 日常赎回起始日 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:fund_basic.redm_startdate × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `market` | E场内O场外 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:fund_basic.market × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="index-profile"></a>

## index_profile

**指数档案**：指数基础档案当前快照；名称、市场、编制机构等属性不自动具备历史版本证据。

定义说明：指数档案；完整保存查询字段和版本。声明粒度：源记录业务粒度。实际字段 **27** 个；逻辑行数 PG **11,863**、CH **11,863**。

PG 主键：`security_code`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code`。

声明时间字段：`未单独声明`；时间规则：`snapshot`。声明主键：`security_code`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | TS代码 | text | String | 否/否 | 未声明/不适用 | tushare:index_basic.ts_code × 1 [asset_type=index]；源字段 [asset_type=index] |
| `name` | 简称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_basic.name × 1 [asset_type=index]；源字段 [asset_type=index] |
| `fullname` | 指数全称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_basic.fullname × 1 [asset_type=index]；源字段 [asset_type=index] |
| `market` | 市场 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_basic.market × 1 [asset_type=index]；源字段 [asset_type=index] |
| `publisher` | 发布方 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_basic.publisher × 1 [asset_type=index]；源字段 [asset_type=index] |
| `index_type` | 指数风格 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_basic.index_type × 1 [asset_type=index]；源字段 [asset_type=index] |
| `category` | 指数类别 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_basic.category × 1 [asset_type=index]；源字段 [asset_type=index] |
| `base_date` | 基期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:index_basic.base_date × 1 [asset_type=index]；源字段 [asset_type=index] |
| `base_point` | 基点 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:index_basic.base_point × 1 [asset_type=index]；源字段 [asset_type=index] |
| `list_date` | 发布日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:index_basic.list_date × 1 [asset_type=index]；源字段 [asset_type=index] |
| `weight_rule` | 加权方式 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_basic.weight_rule × 1 [asset_type=index]；源字段 [asset_type=index] |
| `desc` | 描述 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_basic.desc × 1 [asset_type=index]；源字段 [asset_type=index] |
| `exp_date` | 终止日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:index_basic.exp_date × 1 [asset_type=index]；源字段 [asset_type=index] |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="etf-profile"></a>

## etf_profile

**ETF档案**：ETF 独立档案，与 fund_profile 通过 security_code 联系；保留 ETF 专用字段。

定义说明：境内交易所ETF完整基础信息，含QDII；当前快照不代表历史档案。声明粒度：每个ETF一行。实际字段 **28** 个；逻辑行数 PG **1,912**、CH **1,912**。

PG 主键：`security_code`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code`。

声明时间字段：`未单独声明`；时间规则：`snapshot`。声明主键：`security_code`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | 基金交易代码 | text | String | 否/否 | 未声明/不适用 | tushare:etf_basic.ts_code × 1；源字段 |
| `csname` | ETF中文简称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:etf_basic.csname × 1；源字段 |
| `extname` | ETF扩位简称（交易所简称） | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:etf_basic.extname × 1；源字段 |
| `cname` | 基金中文全称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:etf_basic.cname × 1；源字段 |
| `index_code` | ETF基准指数代码 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:etf_basic.index_code × 1；源字段 |
| `index_name` | ETF基准指数中文全称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:etf_basic.index_name × 1；源字段 |
| `setup_date` | 设立日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:etf_basic.setup_date × 1；源字段 |
| `list_date` | 上市日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:etf_basic.list_date × 1；源字段 |
| `list_status` | 存续状态（L上市 D退市 P待上市） | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:etf_basic.list_status × 1；源字段 |
| `exchange` | 交易所（SH上交所 SZ深交所） | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:etf_basic.exchange × 1；源字段 |
| `mgr_name` | 基金管理人简称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:etf_basic.mgr_name × 1；源字段 |
| `custod_name` | 基金托管人名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:etf_basic.custod_name × 1；源字段 |
| `mgt_fee` | 基金管理人收取的费用；官方未明确单位及计费周期；保留源端原值，不推断百分数或金额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:etf_basic.mgt_fee × 1；源字段 |
| `etf_type` | 基金投资通道类型（境内、QDII） | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:etf_basic.etf_type × 1；源字段 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="security-code-history"></a>

## security_code_history

**证券代码对照**：新旧证券代码对照。valid_from 未知时保留空，不能将 list_date 当成代码切换生效日。

定义说明：证券新旧代码关系；上市日期不代表代码变更生效日。声明粒度：旧代码和新代码一行。实际字段 **19** 个；逻辑行数 PG **248**、CH **248**。

PG 主键：`previous_code, security_code`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`previous_code, security_code`。

声明时间字段：`valid_from`；时间规则：`observations`。声明主键：`previous_code, security_code`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `previous_code` **PK** | 旧代码 | text | String | 否/否 | 未声明/不适用 | tushare:bse_mapping.o_code × 1；源字段 |
| `security_code` **PK** | 新代码 | text | String | 否/否 | 未声明/不适用 | tushare:bse_mapping.n_code × 1；源字段 |
| `security_name` | 证券名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:bse_mapping.name × 1；源字段 |
| `list_date` | 上市日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:bse_mapping.list_date × 1；源字段 |
| `valid_from` | 代码变更生效日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；未提供：bse_mapping.list_date为上市日，不能证明代码切换生效日 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="market-daily-raw"></a>

## market_daily_raw

**原始日频行情**：股票/基金为未复权价格，指数为原生点位。成交量额、股本、市值已按映射统一单位；不能再重复换算。

定义说明：股票、基金和指数的未复权日频宽表。声明粒度：资产和交易日一行。实际字段 **57** 个；逻辑行数 PG **360,561**、CH **360,561**。

PG 主键：`asset_type, security_code, trade_date`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`asset_type, security_code, trade_date`。

声明时间字段：`trade_date`；时间规则：`trading_day`。声明主键：`asset_type, security_code, trade_date`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `asset_type` **PK** | 资产类型；实际业务维度，如 stock/fund/index；证券总表主键还包含该维度。债券为查询预留维度，当前任务未导入。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(asset_scope)：按查询合同的资产范围生成 |
| `security_code` **PK** | 证券代码 | text | String | 否/否 | 未声明/不适用 | tushare:daily.ts_code × 1 [asset_type=stock]；tushare:daily_basic.ts_code × 1 [asset_type=stock]；tushare:fund_daily.ts_code × 1 [asset_type=fund]；tushare:index_daily.ts_code × 1 [asset_type=index]；tushare:sw_daily.ts_code × 1 [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `security_name` | 证券名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:sw_daily.name × 1 [asset_type=index]；复制：优先源行情名称，否则取证券档案名称；档案快照不证明历史名称；源字段 [asset_type=index] |
| `trade_date` **PK** | 交易日期 | date | Date32 | 否/否 | 未声明/不适用 | tushare:daily.trade_date × 1 [asset_type=stock]；tushare:daily_basic.trade_date × 1 [asset_type=stock]；tushare:fund_daily.trade_date × 1 [asset_type=fund]；tushare:index_daily.trade_date × 1 [asset_type=index]；tushare:sw_daily.trade_date × 1 [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `open` | 开盘价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.open × 1 [asset_type=stock]；tushare:fund_daily.open × 1（源单位 元） [asset_type=fund]；tushare:index_daily.open × 1（源单位 指数点） [asset_type=index]；tushare:sw_daily.open × 1（源单位 指数点） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `high` | 最高价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.high × 1 [asset_type=stock]；tushare:fund_daily.high × 1（源单位 元） [asset_type=fund]；tushare:index_daily.high × 1（源单位 指数点） [asset_type=index]；tushare:sw_daily.high × 1（源单位 指数点） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `low` | 最低价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.low × 1 [asset_type=stock]；tushare:fund_daily.low × 1（源单位 元） [asset_type=fund]；tushare:index_daily.low × 1（源单位 指数点） [asset_type=index]；tushare:sw_daily.low × 1（源单位 指数点） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `close` | 收盘价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.close × 1 [asset_type=stock]；tushare:daily_basic.close × 1 [asset_type=stock]；tushare:fund_daily.close × 1（源单位 元） [asset_type=fund]；tushare:index_daily.close × 1（源单位 指数点） [asset_type=index]；tushare:sw_daily.close × 1（源单位 指数点） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `previous_close` | 前收盘价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.pre_close × 1 [asset_type=stock]；tushare:fund_daily.pre_close × 1（源单位 元） [asset_type=fund]；tushare:index_daily.pre_close × 1（源单位 指数点） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `change` | 涨跌 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.change × 1 [asset_type=stock]；tushare:fund_daily.change × 1（源单位 元） [asset_type=fund]；tushare:index_daily.change × 1（源单位 指数点） [asset_type=index]；tushare:sw_daily.change × 1（源单位 指数点） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `pct_change` | 涨跌幅 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:daily.pct_chg × 1（源单位 %） [asset_type=stock]；tushare:fund_daily.pct_chg × 1（源单位 %） [asset_type=fund]；tushare:index_daily.pct_chg × 1（源单位 %） [asset_type=index]；tushare:sw_daily.pct_change × 1 [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `volume` | 成交量 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 股/份 | tushare:daily.vol × 100（源单位 手） [asset_type=stock]；tushare:fund_daily.vol × 100（源单位 手） [asset_type=fund]；tushare:index_daily.vol × 100（源单位 手） [asset_type=index]；tushare:sw_daily.vol × 10000（源单位 万股） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `amount` | 成交额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:daily.amount × 1000（源单位 千元） [asset_type=stock]；tushare:fund_daily.amount × 1000（源单位 千元） [asset_type=fund]；tushare:index_daily.amount × 1000（源单位 千元） [asset_type=index]；tushare:sw_daily.amount × 10000（源单位 万元） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `average_price` | 均价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | 未登记直接来源；查看生成规则；计算(daily_average)：成交额/成交量；指数不适用，零成交量保留空值 [asset_type=['stock', 'fund']] |
| `turnover_rate` | 换手率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:daily_basic.turnover_rate × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock] |
| `volume_ratio` | 量比 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:daily_basic.volume_ratio × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `pe` | 市盈率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:daily_basic.pe × 1 [asset_type=stock]；tushare:sw_daily.pe × 1 [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=index] |
| `pb` | 市净率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:daily_basic.pb × 1 [asset_type=stock]；tushare:sw_daily.pb × 1 [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=index] |
| `ps` | 市销率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:daily_basic.ps × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `pcf` | 市现率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；未提供：已登记接口没有市现率；ETF申赎PCF不等于市现率 |
| `total_shares` | 总股本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 股 | tushare:daily_basic.total_share × 10000（源单位 万股） [asset_type=stock]；源字段 [asset_type=stock] |
| `float_shares` | 流通股本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 股 | tushare:daily_basic.float_share × 10000（源单位 万股） [asset_type=stock]；源字段 [asset_type=stock] |
| `free_shares` | 自由流通股本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 股 | tushare:daily_basic.free_share × 10000（源单位 万股） [asset_type=stock]；源字段 [asset_type=stock] |
| `total_market_value` | 总市值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:daily_basic.total_mv × 10000（源单位 万元） [asset_type=stock]；tushare:sw_daily.total_mv × 10000（源单位 万元） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=index] |
| `circulating_market_value` | 流通市值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:daily_basic.circ_mv × 10000（源单位 万元） [asset_type=stock]；tushare:sw_daily.float_mv × 10000（源单位 万元） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=index] |
| `adjustment_factor` | 复权因子 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；不适用：原始行情不使用后复权辅助列 |
| `adjustment_method` | 复权方式 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；不适用：原始行情不使用后复权辅助列 |
| `adjustment_formula` | 复权公式 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；不适用：原始行情不使用后复权辅助列 |
| `adjustment_version` | 复权计算版本 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；不适用：原始行情不使用后复权辅助列 |
| `raw_open` | 原始开盘价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | 未登记直接来源；查看生成规则；不适用：原始行情不使用后复权辅助列 |
| `raw_high` | 原始最高价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | 未登记直接来源；查看生成规则；不适用：原始行情不使用后复权辅助列 |
| `raw_low` | 原始最低价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | 未登记直接来源；查看生成规则；不适用：原始行情不使用后复权辅助列 |
| `raw_close` | 原始收盘价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | 未登记直接来源；查看生成规则；不适用：原始行情不使用后复权辅助列 |
| `raw_previous_close` | 原始前收盘价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | 未登记直接来源；查看生成规则；不适用：原始行情不使用后复权辅助列 |
| `raw_change` | 原始涨跌额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | 未登记直接来源；查看生成规则；不适用：原始行情不使用后复权辅助列 |
| `turnover_rate_free` | 自由流通换手率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:daily_basic.turnover_rate_f × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock] |
| `pe_ttm` | 滚动市盈率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:daily_basic.pe_ttm × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `ps_ttm` | 滚动市销率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:daily_basic.ps_ttm × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `dividend_yield` | 股息率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:daily_basic.dv_ratio × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock] |
| `dividend_yield_ttm` | 滚动股息率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:daily_basic.dv_ttm × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock] |
| `limit_status` | 涨跌停状态 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:daily_basic.limit_status × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `ah_volume` | 盘后成交量 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 股/份 | tushare:daily.ah_vol × 100（源单位 手） [asset_type=stock]；源字段 [asset_type=stock] |
| `ah_amount` | 盘后成交额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:daily.ah_amount × 1000（源单位 千元） [asset_type=stock]；源字段 [asset_type=stock] |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Date32 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Int64 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="market-daily-post-adjusted"></a>

## market_daily_post_adjusted

**后复权日频行情**：股票/基金按 raw_price × adjustment_factor 派生价格，指数保持点位。raw_* 保留原价格，量额不复权；缺因子时价格空。后复权价格不能直接用于回测成交手数计算。

定义说明：保存价格后复权值和复权计算依据的日频宽表。声明粒度：资产和交易日一行。实际字段 **57** 个；逻辑行数 PG **360,561**、CH **360,561**。

PG 主键：`asset_type, security_code, trade_date`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`asset_type, security_code, trade_date`。

声明时间字段：`trade_date`；时间规则：`trading_day`。声明主键：`asset_type, security_code, trade_date`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `asset_type` **PK** | 资产类型；实际业务维度，如 stock/fund/index；证券总表主键还包含该维度。债券为查询预留维度，当前任务未导入。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(asset_scope)：按查询合同的资产范围生成 |
| `security_code` **PK** | 证券代码 | text | String | 否/否 | 未声明/不适用 | tushare:adj_factor.ts_code × 1 [asset_type=stock]；tushare:fund_adj.ts_code × 1 [asset_type=fund]；tushare:daily.ts_code × 1 [asset_type=stock]；tushare:fund_daily.ts_code × 1 [asset_type=fund]；tushare:index_daily.ts_code × 1 [asset_type=index]；tushare:sw_daily.ts_code × 1 [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `security_name` | 证券名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `trade_date` **PK** | 交易日期 | date | Date32 | 否/否 | 未声明/不适用 | tushare:adj_factor.trade_date × 1 [asset_type=stock]；tushare:fund_adj.trade_date × 1 [asset_type=fund]；tushare:daily.trade_date × 1 [asset_type=stock]；tushare:fund_daily.trade_date × 1 [asset_type=fund]；tushare:index_daily.trade_date × 1 [asset_type=index]；tushare:sw_daily.trade_date × 1 [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `open` | 开盘价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.open × 1 [asset_type=stock]；内置派生规则；tushare:fund_daily.open × 1（源单位 元） [asset_type=fund]；内置派生规则；tushare:index_daily.open × 1（源单位 指数点） [asset_type=index]；内置派生规则；tushare:sw_daily.open × 1（源单位 指数点） [asset_type=index]；内置派生规则；计算(post_adjustment)：Decimal(raw_price)*Decimal(adjustment_factor) [asset_type=stock]；计算(post_adjustment)：Decimal(raw_price)*Decimal(adjustment_factor) [asset_type=fund]；计算(post_adjustment)：native_index_value [asset_type=index] |
| `high` | 最高价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.high × 1 [asset_type=stock]；内置派生规则；tushare:fund_daily.high × 1（源单位 元） [asset_type=fund]；内置派生规则；tushare:index_daily.high × 1（源单位 指数点） [asset_type=index]；内置派生规则；tushare:sw_daily.high × 1（源单位 指数点） [asset_type=index]；内置派生规则；计算(post_adjustment)：Decimal(raw_price)*Decimal(adjustment_factor) [asset_type=stock]；计算(post_adjustment)：Decimal(raw_price)*Decimal(adjustment_factor) [asset_type=fund]；计算(post_adjustment)：native_index_value [asset_type=index] |
| `low` | 最低价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.low × 1 [asset_type=stock]；内置派生规则；tushare:fund_daily.low × 1（源单位 元） [asset_type=fund]；内置派生规则；tushare:index_daily.low × 1（源单位 指数点） [asset_type=index]；内置派生规则；tushare:sw_daily.low × 1（源单位 指数点） [asset_type=index]；内置派生规则；计算(post_adjustment)：Decimal(raw_price)*Decimal(adjustment_factor) [asset_type=stock]；计算(post_adjustment)：Decimal(raw_price)*Decimal(adjustment_factor) [asset_type=fund]；计算(post_adjustment)：native_index_value [asset_type=index] |
| `close` | 收盘价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.close × 1 [asset_type=stock]；内置派生规则；tushare:fund_daily.close × 1（源单位 元） [asset_type=fund]；内置派生规则；tushare:index_daily.close × 1（源单位 指数点） [asset_type=index]；内置派生规则；tushare:sw_daily.close × 1（源单位 指数点） [asset_type=index]；内置派生规则；计算(post_adjustment)：Decimal(raw_price)*Decimal(adjustment_factor) [asset_type=stock]；计算(post_adjustment)：Decimal(raw_price)*Decimal(adjustment_factor) [asset_type=fund]；计算(post_adjustment)：native_index_value [asset_type=index] |
| `previous_close` | 前收盘价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.pre_close × 1 [asset_type=stock]；内置派生规则；tushare:fund_daily.pre_close × 1（源单位 元） [asset_type=fund]；内置派生规则；tushare:index_daily.pre_close × 1（源单位 指数点） [asset_type=index]；内置派生规则；计算(post_adjustment)：Decimal(raw_price)*Decimal(adjustment_factor) [asset_type=stock]；计算(post_adjustment)：Decimal(raw_price)*Decimal(adjustment_factor) [asset_type=fund]；计算(post_adjustment)：native_index_value [asset_type=index] |
| `change` | 涨跌 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.change × 1 [asset_type=stock]；内置派生规则；tushare:fund_daily.change × 1（源单位 元） [asset_type=fund]；内置派生规则；tushare:index_daily.change × 1（源单位 指数点） [asset_type=index]；内置派生规则；tushare:sw_daily.change × 1（源单位 指数点） [asset_type=index]；内置派生规则；计算(post_adjustment)：Decimal(raw_price)*Decimal(adjustment_factor) [asset_type=stock]；计算(post_adjustment)：Decimal(raw_price)*Decimal(adjustment_factor) [asset_type=fund]；计算(post_adjustment)：native_index_value [asset_type=index] |
| `pct_change` | 涨跌幅 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `volume` | 成交量 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 股/份 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `amount` | 成交额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `average_price` | 均价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | 未登记直接来源；查看生成规则；计算(post_adjustment)：原始均价乘复权因子；指数不适用 [asset_type=['stock', 'fund']] |
| `turnover_rate` | 换手率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `volume_ratio` | 量比 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `pe` | 市盈率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `pb` | 市净率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `ps` | 市销率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `pcf` | 市现率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；未提供：已登记接口没有市现率；ETF申赎PCF不等于市现率 |
| `total_shares` | 总股本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 股 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `float_shares` | 流通股本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 股 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `free_shares` | 自由流通股本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 股 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `total_market_value` | 总市值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `circulating_market_value` | 流通市值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `adjustment_factor` | 复权因子 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:adj_factor.adj_factor × 1 [asset_type=stock]；tushare:fund_adj.adj_factor × 1 [asset_type=fund]；不适用：指数保留原生点位，不使用股票/基金复权因子 [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund] |
| `adjustment_method` | 复权方式 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(post_adjustment)：原生指数值或raw_price × adjustment_factor；缺因子不伪造 |
| `adjustment_formula` | 复权公式 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(post_adjustment)：原生指数值或raw_price × adjustment_factor；缺因子不伪造 |
| `adjustment_version` | 复权计算版本 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(post_adjustment)：原生指数值或raw_price × adjustment_factor；缺因子不伪造 |
| `raw_open` | 原始开盘价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.open × 1 [asset_type=stock]；tushare:fund_daily.open × 1（源单位 元） [asset_type=fund]；tushare:index_daily.open × 1（源单位 指数点） [asset_type=index]；tushare:sw_daily.open × 1（源单位 指数点） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `raw_high` | 原始最高价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.high × 1 [asset_type=stock]；tushare:fund_daily.high × 1（源单位 元） [asset_type=fund]；tushare:index_daily.high × 1（源单位 指数点） [asset_type=index]；tushare:sw_daily.high × 1（源单位 指数点） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `raw_low` | 原始最低价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.low × 1 [asset_type=stock]；tushare:fund_daily.low × 1（源单位 元） [asset_type=fund]；tushare:index_daily.low × 1（源单位 指数点） [asset_type=index]；tushare:sw_daily.low × 1（源单位 指数点） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `raw_close` | 原始收盘价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.close × 1 [asset_type=stock]；tushare:fund_daily.close × 1（源单位 元） [asset_type=fund]；tushare:index_daily.close × 1（源单位 指数点） [asset_type=index]；tushare:sw_daily.close × 1（源单位 指数点） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `raw_previous_close` | 原始前收盘价 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.pre_close × 1 [asset_type=stock]；tushare:fund_daily.pre_close × 1（源单位 元） [asset_type=fund]；tushare:index_daily.pre_close × 1（源单位 指数点） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `raw_change` | 原始涨跌额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/指数点 | tushare:daily.change × 1 [asset_type=stock]；tushare:fund_daily.change × 1（源单位 元） [asset_type=fund]；tushare:index_daily.change × 1（源单位 指数点） [asset_type=index]；tushare:sw_daily.change × 1（源单位 指数点） [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `turnover_rate_free` | 自由流通换手率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `pe_ttm` | 滚动市盈率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `ps_ttm` | 滚动市销率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `dividend_yield` | 股息率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `dividend_yield_ttm` | 滚动股息率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `limit_status` | 涨跌停状态 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `ah_volume` | 盘后成交量 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 股/份 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `ah_amount` | 盘后成交额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | 未登记直接来源；查看生成规则；复制：继承本次原始行情和每日指标；成交量额不复权 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Date32 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Int64 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="stock-quote-supplement-daily"></a>

## stock_quote_supplement_daily

**股票备用行情**：bak_daily 的补充观测，保留源端字段与单位。不能凭同名或近似名字直接替代统一行情字段。

定义说明：股票备用行情完整源字段；2017年中左右开始，早期可能缺日；保留源端单位，单位未明确字段不得自动投影统一行情。声明粒度：每个股票和交易日一行。实际字段 **45** 个；逻辑行数 PG **106,000**、CH **106,000**。

PG 主键：`security_code, trade_date`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code, trade_date`。

声明时间字段：`trade_date`；时间规则：`observations`。声明主键：`security_code, trade_date`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | 股票代码 | text | String | 否/否 | 未声明/不适用 | tushare:bak_daily.ts_code × 1；源字段 |
| `trade_date` **PK** | 交易日期 | date | Date32 | 否/否 | 未声明/不适用 | tushare:bak_daily.trade_date × 1；源字段 |
| `name` | 股票名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:bak_daily.name × 1；源字段 |
| `pct_change` | 涨跌幅；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.pct_change × 1；源字段 |
| `close` | 收盘价；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.close × 1；源字段 |
| `change` | 涨跌额；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.change × 1；源字段 |
| `open` | 开盘价；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.open × 1；源字段 |
| `high` | 最高价；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.high × 1；源字段 |
| `low` | 最低价；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.low × 1；源字段 |
| `pre_close` | 昨收价；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.pre_close × 1；源字段 |
| `vol_ratio` | 量比；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.vol_ratio × 1；源字段 |
| `turn_over` | 换手率；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.turn_over × 1；源字段 |
| `swing` | 振幅；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.swing × 1；源字段 |
| `vol` | 成交量；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.vol × 1；源字段 |
| `amount` | 成交额；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.amount × 1；源字段 |
| `selling` | 内盘（主动卖） | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 手 | tushare:bak_daily.selling × 1（源单位 手）；源字段 |
| `buying` | 外盘（主动买） | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 手 | tushare:bak_daily.buying × 1（源单位 手）；源字段 |
| `total_share` | 总股本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 亿股 | tushare:bak_daily.total_share × 1（源单位 亿股）；源字段 |
| `float_share` | 流通股本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 亿股 | tushare:bak_daily.float_share × 1（源单位 亿股）；源字段 |
| `pe` | 动态市盈率；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.pe × 1；源字段 |
| `industry` | 所属行业 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:bak_daily.industry × 1；源字段 |
| `area` | 所属地域 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:bak_daily.area × 1；源字段 |
| `float_mv` | 流通市值；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.float_mv × 1；源字段 |
| `total_mv` | 总市值；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.total_mv × 1；源字段 |
| `avg_price` | 平均价；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.avg_price × 1；源字段 |
| `strength` | 强弱度 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:bak_daily.strength × 1（源单位 %）；源字段 |
| `activity` | 活跃度 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:bak_daily.activity × 1（源单位 %）；源字段 |
| `avg_turnover` | 笔换手；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.avg_turnover × 1；源字段 |
| `attack` | 攻击波 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:bak_daily.attack × 1（源单位 %）；源字段 |
| `interval_3` | 近3月涨幅；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.interval_3 × 1；源字段 |
| `interval_6` | 近6月涨幅；官方未明确单位；保留源端原值，不进行统一行情单位转换 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:bak_daily.interval_6 × 1；源字段 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="fundamental-periodic"></a>

## fundamental_periodic

**定期财务数据**：510 字段财报宽表，不同 statement_type 适用不同字段。主键含报告期、报表类型、报告类型和内容版本；稀疏空值并不自动表示采集失败。rd_exp 已弃用。

定义说明：利润表、资产负债表、现金流量表和财务指标的统一宽表。声明粒度：证券、报告期、报告类型和版本一行。实际字段 **510** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`security_code, period_end, statement_type, report_type, version_id`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code, period_end, statement_type, report_type, version_id`。

声明时间字段：`period_end`；时间规则：`quarter_end`。声明主键：`security_code, period_end, statement_type, report_type, version_id`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | 证券代码 | text | String | 否/否 | 未声明/不适用 | tushare:income.ts_code × 1 [asset_type=stock]；tushare:balancesheet.ts_code × 1 [asset_type=stock]；tushare:cashflow.ts_code × 1 [asset_type=stock]；tushare:fina_indicator.ts_code × 1 [asset_type=stock]；tushare:income_vip.ts_code × 1 [asset_type=stock]；tushare:balancesheet_vip.ts_code × 1 [asset_type=stock]；tushare:cashflow_vip.ts_code × 1 [asset_type=stock]；tushare:fina_indicator_vip.ts_code × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income]；源字段 [asset_type=stock,statement_type=balancesheet]；源字段 [asset_type=stock,statement_type=cashflow]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `period_end` **PK** | 报告期末；报告期末；报告期末，区别于 announcement_date / actual_announcement_date / available_at；当前管理器财报任务按报告期规划。 | date | Date32 | 否/否 | 未声明/不适用 | tushare:income.end_date × 1 [asset_type=stock]；tushare:balancesheet.end_date × 1 [asset_type=stock]；tushare:cashflow.end_date × 1 [asset_type=stock]；tushare:fina_indicator.end_date × 1 [asset_type=stock]；tushare:income_vip.end_date × 1 [asset_type=stock]；tushare:balancesheet_vip.end_date × 1 [asset_type=stock]；tushare:cashflow_vip.end_date × 1 [asset_type=stock]；tushare:fina_indicator_vip.end_date × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income]；源字段 [asset_type=stock,statement_type=balancesheet]；源字段 [asset_type=stock,statement_type=cashflow]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `announcement_date` | 公告日期；公告日期；源 ann_date；公告日期不是报告期，也不必等于真实可得时刻。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:income.ann_date × 1 [asset_type=stock]；tushare:balancesheet.ann_date × 1 [asset_type=stock]；tushare:cashflow.ann_date × 1 [asset_type=stock]；tushare:fina_indicator.ann_date × 1 [asset_type=stock]；tushare:income_vip.ann_date × 1 [asset_type=stock]；tushare:balancesheet_vip.ann_date × 1 [asset_type=stock]；tushare:cashflow_vip.ann_date × 1 [asset_type=stock]；tushare:fina_indicator_vip.ann_date × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income]；源字段 [asset_type=stock,statement_type=balancesheet]；源字段 [asset_type=stock,statement_type=cashflow]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `report_type` **PK** | 报告类型；报告类型；源报告类型代码，保留原代码；财务指标接口未提供时使用明确值 unspecified，不等同报告类型 1。 | text | String | 否/否 | 未声明/不适用 | tushare:income.report_type × 1 [asset_type=stock]；tushare:balancesheet.report_type × 1 [asset_type=stock]；tushare:cashflow.report_type × 1 [asset_type=stock]；tushare:income_vip.report_type × 1 [asset_type=stock]；tushare:balancesheet_vip.report_type × 1 [asset_type=stock]；tushare:cashflow_vip.report_type × 1 [asset_type=stock]；常量：财务指标接口不提供报告类型，使用显式unspecified [statement_type=fina_indicator]；源字段 [asset_type=stock,statement_type=income]；源字段 [asset_type=stock,statement_type=balancesheet]；源字段 [asset_type=stock,statement_type=cashflow] |
| `statement_type` **PK** | 报表种类；报表种类；income / balancesheet / cashflow / fina_indicator；普通和 VIP 接口使用同一语义。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(asset_scope)：普通/VIP接口归一为相同报表类型 |
| `version_id` **PK** | 财报内容版本；财报内容版本身份；普通/VIP接口使用同一报表身份。不能按哈希字典顺序或下载先后推断修订可得时间。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(business_version)：完整规范化业务记录的内容哈希 |
| `company_type` | 公司类型 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:income.comp_type × 1 [asset_type=stock]；tushare:balancesheet.comp_type × 1 [asset_type=stock]；tushare:cashflow.comp_type × 1 [asset_type=stock]；tushare:income_vip.comp_type × 1 [asset_type=stock]；tushare:balancesheet_vip.comp_type × 1 [asset_type=stock]；tushare:cashflow_vip.comp_type × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income]；源字段 [asset_type=stock,statement_type=balancesheet]；源字段 [asset_type=stock,statement_type=cashflow] |
| `update_flag` | 修订标记 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:income.update_flag × 1 [asset_type=stock]；tushare:balancesheet.update_flag × 1 [asset_type=stock]；tushare:cashflow.update_flag × 1 [asset_type=stock]；tushare:fina_indicator.update_flag × 1 [asset_type=stock]；tushare:income_vip.update_flag × 1 [asset_type=stock]；tushare:balancesheet_vip.update_flag × 1 [asset_type=stock]；tushare:cashflow_vip.update_flag × 1 [asset_type=stock]；tushare:fina_indicator_vip.update_flag × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income]；源字段 [asset_type=stock,statement_type=balancesheet]；源字段 [asset_type=stock,statement_type=cashflow]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `revenue` | 营业收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:income.revenue × 1 [asset_type=stock]；tushare:income_vip.revenue × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `net_income` | 净利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:income.n_income × 1 [asset_type=stock]；tushare:income_vip.n_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `total_assets` | 总资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:balancesheet.total_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.total_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `total_liabilities` | 总负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:balancesheet.total_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.total_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `operating_cashflow` | 经营活动现金流 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:cashflow.n_cashflow_act × 1 [asset_type=stock]；tushare:cashflow_vip.n_cashflow_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `roe` | 净资产收益率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.roe × 1 [asset_type=stock]；tushare:fina_indicator_vip.roe × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `basic_eps` | 基本每股收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.basic_eps × 1 [asset_type=stock]；tushare:income_vip.basic_eps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `diluted_eps` | 稀释每股收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.diluted_eps × 1 [asset_type=stock]；tushare:income_vip.diluted_eps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `total_revenue` | 营业总收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.total_revenue × 1 [asset_type=stock]；tushare:income_vip.total_revenue × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `int_income` | 利息收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.int_income × 1 [asset_type=stock]；tushare:income_vip.int_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `prem_earned` | 已赚保费 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.prem_earned × 1 [asset_type=stock]；tushare:income_vip.prem_earned × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `comm_income` | 手续费及佣金收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.comm_income × 1 [asset_type=stock]；tushare:income_vip.comm_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `n_commis_income` | 手续费及佣金净收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.n_commis_income × 1 [asset_type=stock]；tushare:income_vip.n_commis_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `n_oth_income` | 其他经营净收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.n_oth_income × 1 [asset_type=stock]；tushare:income_vip.n_oth_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `n_oth_b_income` | 加:其他业务净收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.n_oth_b_income × 1 [asset_type=stock]；tushare:income_vip.n_oth_b_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `prem_income` | 保险业务收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.prem_income × 1 [asset_type=stock]；tushare:income_vip.prem_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `out_prem` | 减:分出保费 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.out_prem × 1 [asset_type=stock]；tushare:income_vip.out_prem × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `une_prem_reser` | 提取未到期责任准备金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.une_prem_reser × 1 [asset_type=stock]；tushare:income_vip.une_prem_reser × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `reins_income` | 其中:分保费收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.reins_income × 1 [asset_type=stock]；tushare:income_vip.reins_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `n_sec_tb_income` | 代理买卖证券业务净收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.n_sec_tb_income × 1 [asset_type=stock]；tushare:income_vip.n_sec_tb_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `n_sec_uw_income` | 证券承销业务净收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.n_sec_uw_income × 1 [asset_type=stock]；tushare:income_vip.n_sec_uw_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `n_asset_mg_income` | 受托客户资产管理业务净收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.n_asset_mg_income × 1 [asset_type=stock]；tushare:income_vip.n_asset_mg_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `oth_b_income` | 其他业务收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.oth_b_income × 1 [asset_type=stock]；tushare:income_vip.oth_b_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `fv_value_chg_gain` | 加:公允价值变动净收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.fv_value_chg_gain × 1 [asset_type=stock]；tushare:income_vip.fv_value_chg_gain × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `invest_income` | 加:投资净收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.invest_income × 1 [asset_type=stock]；tushare:income_vip.invest_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `ass_invest_income` | 其中:对联营企业和合营企业的投资收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.ass_invest_income × 1 [asset_type=stock]；tushare:income_vip.ass_invest_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `forex_gain` | 加:汇兑净收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.forex_gain × 1 [asset_type=stock]；tushare:income_vip.forex_gain × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `total_cogs` | 营业总成本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.total_cogs × 1 [asset_type=stock]；tushare:income_vip.total_cogs × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `oper_cost` | 减:营业成本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.oper_cost × 1 [asset_type=stock]；tushare:income_vip.oper_cost × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `int_exp` | 减:利息支出 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.int_exp × 1 [asset_type=stock]；tushare:income_vip.int_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `comm_exp` | 减:手续费及佣金支出 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.comm_exp × 1 [asset_type=stock]；tushare:income_vip.comm_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `biz_tax_surchg` | 减:营业税金及附加 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.biz_tax_surchg × 1 [asset_type=stock]；tushare:income_vip.biz_tax_surchg × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `sell_exp` | 减:销售费用 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.sell_exp × 1 [asset_type=stock]；tushare:income_vip.sell_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `admin_exp` | 减:管理费用 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.admin_exp × 1 [asset_type=stock]；tushare:income_vip.admin_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `fin_exp` | 减:财务费用 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.fin_exp × 1 [asset_type=stock]；tushare:income_vip.fin_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `assets_impair_loss` | 减:资产减值损失 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.assets_impair_loss × 1 [asset_type=stock]；tushare:income_vip.assets_impair_loss × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `prem_refund` | 退保金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.prem_refund × 1 [asset_type=stock]；tushare:income_vip.prem_refund × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `compens_payout` | 赔付总支出 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.compens_payout × 1 [asset_type=stock]；tushare:income_vip.compens_payout × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `reser_insur_liab` | 提取保险责任准备金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.reser_insur_liab × 1 [asset_type=stock]；tushare:income_vip.reser_insur_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `div_payt` | 保户红利支出 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.div_payt × 1 [asset_type=stock]；tushare:income_vip.div_payt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `reins_exp` | 分保费用 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.reins_exp × 1 [asset_type=stock]；tushare:income_vip.reins_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `oper_exp` | 营业支出 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.oper_exp × 1 [asset_type=stock]；tushare:income_vip.oper_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `compens_payout_refu` | 减:摊回赔付支出 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.compens_payout_refu × 1 [asset_type=stock]；tushare:income_vip.compens_payout_refu × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `insur_reser_refu` | 减:摊回保险责任准备金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.insur_reser_refu × 1 [asset_type=stock]；tushare:income_vip.insur_reser_refu × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `reins_cost_refund` | 减:摊回分保费用 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.reins_cost_refund × 1 [asset_type=stock]；tushare:income_vip.reins_cost_refund × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `other_bus_cost` | 其他业务成本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.other_bus_cost × 1 [asset_type=stock]；tushare:income_vip.other_bus_cost × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `operate_profit` | 营业利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.operate_profit × 1 [asset_type=stock]；tushare:income_vip.operate_profit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `non_oper_income` | 加:营业外收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.non_oper_income × 1 [asset_type=stock]；tushare:income_vip.non_oper_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `non_oper_exp` | 减:营业外支出 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.non_oper_exp × 1 [asset_type=stock]；tushare:income_vip.non_oper_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `nca_disploss` | 其中:减:非流动资产处置净损失 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.nca_disploss × 1 [asset_type=stock]；tushare:income_vip.nca_disploss × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `total_profit` | 利润总额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.total_profit × 1 [asset_type=stock]；tushare:income_vip.total_profit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `income_tax` | 所得税费用 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.income_tax × 1 [asset_type=stock]；tushare:income_vip.income_tax × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `net_income_attributable` | 净利润(不含少数股东损益) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.n_income_attr_p × 1 [asset_type=stock]；tushare:income_vip.n_income_attr_p × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `minority_gain` | 少数股东损益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.minority_gain × 1 [asset_type=stock]；tushare:income_vip.minority_gain × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `oth_compr_income` | 其他综合收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.oth_compr_income × 1 [asset_type=stock]；tushare:income_vip.oth_compr_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `t_compr_income` | 综合收益总额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.t_compr_income × 1 [asset_type=stock]；tushare:income_vip.t_compr_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `compr_inc_attr_p` | 归属于母公司(或股东)的综合收益总额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.compr_inc_attr_p × 1 [asset_type=stock]；tushare:income_vip.compr_inc_attr_p × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `compr_inc_attr_m_s` | 归属于少数股东的综合收益总额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.compr_inc_attr_m_s × 1 [asset_type=stock]；tushare:income_vip.compr_inc_attr_m_s × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `ebit` | 息税前利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.ebit × 1 [asset_type=stock]；tushare:fina_indicator.ebit × 1 [asset_type=stock]；tushare:income_vip.ebit × 1 [asset_type=stock]；tushare:fina_indicator_vip.ebit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ebitda` | 息税折旧摊销前利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.ebitda × 1 [asset_type=stock]；tushare:fina_indicator.ebitda × 1 [asset_type=stock]；tushare:income_vip.ebitda × 1 [asset_type=stock]；tushare:fina_indicator_vip.ebitda × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `insurance_exp` | 保险业务支出 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.insurance_exp × 1 [asset_type=stock]；tushare:income_vip.insurance_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `undist_profit` | 年初未分配利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.undist_profit × 1 [asset_type=stock]；tushare:income_vip.undist_profit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `distable_profit` | 可分配利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.distable_profit × 1 [asset_type=stock]；tushare:income_vip.distable_profit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `rd_exp` | 研发指标（已弃用）；deprecated：历史 income.rd_exp 与 fina_indicator.rd_exp 共用列；保留历史值，新数据分别写 research_and_development_expense（研发费用）和 research_and_development_investment（研发投入合计）；已弃用的歧义旧列，禁止新查询混用；income.rd_exp 与 fina_indicator.rd_exp 分别落入两个 research_and_development_* 字段。 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；未提供：旧同名不同义字段已弃用；历史迁移需按报表类型和归档证明 |
| `fin_exp_int_exp` | 财务费用:利息费用 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.fin_exp_int_exp × 1 [asset_type=stock]；tushare:income_vip.fin_exp_int_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `fin_exp_int_inc` | 财务费用:利息收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.fin_exp_int_inc × 1 [asset_type=stock]；tushare:income_vip.fin_exp_int_inc × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `transfer_surplus_rese` | 盈余公积转入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.transfer_surplus_rese × 1 [asset_type=stock]；tushare:income_vip.transfer_surplus_rese × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `transfer_housing_imprest` | 住房周转金转入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.transfer_housing_imprest × 1 [asset_type=stock]；tushare:income_vip.transfer_housing_imprest × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `transfer_oth` | 其他转入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.transfer_oth × 1 [asset_type=stock]；tushare:income_vip.transfer_oth × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `adj_lossgain` | 调整以前年度损益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.adj_lossgain × 1 [asset_type=stock]；tushare:income_vip.adj_lossgain × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `withdra_legal_surplus` | 提取法定盈余公积 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.withdra_legal_surplus × 1 [asset_type=stock]；tushare:income_vip.withdra_legal_surplus × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `withdra_legal_pubfund` | 提取法定公益金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.withdra_legal_pubfund × 1 [asset_type=stock]；tushare:income_vip.withdra_legal_pubfund × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `withdra_biz_devfund` | 提取企业发展基金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.withdra_biz_devfund × 1 [asset_type=stock]；tushare:income_vip.withdra_biz_devfund × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `withdra_rese_fund` | 提取储备基金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.withdra_rese_fund × 1 [asset_type=stock]；tushare:income_vip.withdra_rese_fund × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `withdra_oth_ersu` | 提取任意盈余公积金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.withdra_oth_ersu × 1 [asset_type=stock]；tushare:income_vip.withdra_oth_ersu × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `workers_welfare` | 职工奖金福利 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.workers_welfare × 1 [asset_type=stock]；tushare:income_vip.workers_welfare × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `distr_profit_shrhder` | 可供股东分配的利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.distr_profit_shrhder × 1 [asset_type=stock]；tushare:income_vip.distr_profit_shrhder × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `prfshare_payable_dvd` | 应付优先股股利 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.prfshare_payable_dvd × 1 [asset_type=stock]；tushare:income_vip.prfshare_payable_dvd × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `comshare_payable_dvd` | 应付普通股股利 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.comshare_payable_dvd × 1 [asset_type=stock]；tushare:income_vip.comshare_payable_dvd × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `capit_comstock_div` | 转作股本的普通股股利 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.capit_comstock_div × 1 [asset_type=stock]；tushare:income_vip.capit_comstock_div × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `net_after_nr_lp_correct` | 扣除非经常性损益后的净利润（更正前） | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.net_after_nr_lp_correct × 1 [asset_type=stock]；tushare:income_vip.net_after_nr_lp_correct × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `credit_impa_loss` | 信用减值损失 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.credit_impa_loss × 1 [asset_type=stock]；tushare:cashflow.credit_impa_loss × 1 [asset_type=stock]；tushare:income_vip.credit_impa_loss × 1 [asset_type=stock]；tushare:cashflow_vip.credit_impa_loss × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income]；源字段 [asset_type=stock,statement_type=cashflow] |
| `net_expo_hedging_benefits` | 净敞口套期收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.net_expo_hedging_benefits × 1 [asset_type=stock]；tushare:income_vip.net_expo_hedging_benefits × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `oth_impair_loss_assets` | 其他资产减值损失 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.oth_impair_loss_assets × 1 [asset_type=stock]；tushare:income_vip.oth_impair_loss_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `total_opcost` | 营业总成本（二） | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.total_opcost × 1 [asset_type=stock]；tushare:income_vip.total_opcost × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `amodcost_fin_assets` | 以摊余成本计量的金融资产终止确认收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.amodcost_fin_assets × 1 [asset_type=stock]；tushare:income_vip.amodcost_fin_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `oth_income` | 其他收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.oth_income × 1 [asset_type=stock]；tushare:income_vip.oth_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `asset_disp_income` | 资产处置收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.asset_disp_income × 1 [asset_type=stock]；tushare:income_vip.asset_disp_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `continued_net_profit` | 持续经营净利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.continued_net_profit × 1 [asset_type=stock]；tushare:income_vip.continued_net_profit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `end_net_profit` | 终止经营净利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.end_net_profit × 1 [asset_type=stock]；tushare:income_vip.end_net_profit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `total_share` | 期末总股本（股） | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.total_share × 1 [asset_type=stock]；tushare:balancesheet_vip.total_share × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `cap_rese` | 资本公积金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.cap_rese × 1 [asset_type=stock]；tushare:balancesheet_vip.cap_rese × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `undistr_porfit` | 未分配利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.undistr_porfit × 1 [asset_type=stock]；tushare:balancesheet_vip.undistr_porfit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `surplus_rese` | 盈余公积金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.surplus_rese × 1 [asset_type=stock]；tushare:balancesheet_vip.surplus_rese × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `special_rese` | 专项储备 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.special_rese × 1 [asset_type=stock]；tushare:balancesheet_vip.special_rese × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `money_cap` | 货币资金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.money_cap × 1 [asset_type=stock]；tushare:balancesheet_vip.money_cap × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `trad_asset` | 交易性金融资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.trad_asset × 1 [asset_type=stock]；tushare:balancesheet_vip.trad_asset × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `notes_receiv` | 应收票据 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.notes_receiv × 1 [asset_type=stock]；tushare:balancesheet_vip.notes_receiv × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `accounts_receiv` | 应收账款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.accounts_receiv × 1 [asset_type=stock]；tushare:balancesheet_vip.accounts_receiv × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_receiv` | 其他应收款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_receiv × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_receiv × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `prepayment` | 预付款项 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.prepayment × 1 [asset_type=stock]；tushare:balancesheet_vip.prepayment × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `div_receiv` | 应收股利 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.div_receiv × 1 [asset_type=stock]；tushare:balancesheet_vip.div_receiv × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `int_receiv` | 应收利息 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.int_receiv × 1 [asset_type=stock]；tushare:balancesheet_vip.int_receiv × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `inventories` | 存货 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.inventories × 1 [asset_type=stock]；tushare:balancesheet_vip.inventories × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `amor_exp` | 待摊费用 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.amor_exp × 1 [asset_type=stock]；tushare:balancesheet_vip.amor_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `nca_within_1y` | 一年内到期的非流动资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.nca_within_1y × 1 [asset_type=stock]；tushare:balancesheet_vip.nca_within_1y × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `sett_rsrv` | 结算备付金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.sett_rsrv × 1 [asset_type=stock]；tushare:balancesheet_vip.sett_rsrv × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `loanto_oth_bank_fi` | 拆出资金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.loanto_oth_bank_fi × 1 [asset_type=stock]；tushare:balancesheet_vip.loanto_oth_bank_fi × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `premium_receiv` | 应收保费 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.premium_receiv × 1 [asset_type=stock]；tushare:balancesheet_vip.premium_receiv × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `reinsur_receiv` | 应收分保账款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.reinsur_receiv × 1 [asset_type=stock]；tushare:balancesheet_vip.reinsur_receiv × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `reinsur_res_receiv` | 应收分保合同准备金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.reinsur_res_receiv × 1 [asset_type=stock]；tushare:balancesheet_vip.reinsur_res_receiv × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `pur_resale_fa` | 买入返售金融资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.pur_resale_fa × 1 [asset_type=stock]；tushare:balancesheet_vip.pur_resale_fa × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_cur_assets` | 其他流动资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_cur_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_cur_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `total_cur_assets` | 流动资产合计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.total_cur_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.total_cur_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `fa_avail_for_sale` | 可供出售金融资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.fa_avail_for_sale × 1 [asset_type=stock]；tushare:balancesheet_vip.fa_avail_for_sale × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `htm_invest` | 持有至到期投资 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.htm_invest × 1 [asset_type=stock]；tushare:balancesheet_vip.htm_invest × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `lt_eqt_invest` | 长期股权投资 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.lt_eqt_invest × 1 [asset_type=stock]；tushare:balancesheet_vip.lt_eqt_invest × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `invest_real_estate` | 投资性房地产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.invest_real_estate × 1 [asset_type=stock]；tushare:balancesheet_vip.invest_real_estate × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `time_deposits` | 定期存款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.time_deposits × 1 [asset_type=stock]；tushare:balancesheet_vip.time_deposits × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_assets` | 其他资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `lt_rec` | 长期应收款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.lt_rec × 1 [asset_type=stock]；tushare:balancesheet_vip.lt_rec × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `fix_assets` | 固定资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.fix_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.fix_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `cip` | 在建工程 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.cip × 1 [asset_type=stock]；tushare:balancesheet_vip.cip × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `const_materials` | 工程物资 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.const_materials × 1 [asset_type=stock]；tushare:balancesheet_vip.const_materials × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `fixed_assets_disp` | 固定资产清理 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.fixed_assets_disp × 1 [asset_type=stock]；tushare:balancesheet_vip.fixed_assets_disp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `produc_bio_assets` | 生产性生物资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.produc_bio_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.produc_bio_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oil_and_gas_assets` | 油气资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oil_and_gas_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.oil_and_gas_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `intan_assets` | 无形资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.intan_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.intan_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `r_and_d` | 研发支出 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.r_and_d × 1 [asset_type=stock]；tushare:balancesheet_vip.r_and_d × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `goodwill` | 商誉 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.goodwill × 1 [asset_type=stock]；tushare:balancesheet_vip.goodwill × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `lt_amor_exp` | 长期待摊费用 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.lt_amor_exp × 1 [asset_type=stock]；tushare:balancesheet_vip.lt_amor_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `defer_tax_assets` | 递延所得税资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.defer_tax_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.defer_tax_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `decr_in_disbur` | 发放贷款及垫款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.decr_in_disbur × 1 [asset_type=stock]；tushare:balancesheet_vip.decr_in_disbur × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_nca` | 其他非流动资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_nca × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_nca × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `total_nca` | 非流动资产合计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.total_nca × 1 [asset_type=stock]；tushare:balancesheet_vip.total_nca × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `cash_reser_cb` | 现金及存放中央银行款项 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.cash_reser_cb × 1 [asset_type=stock]；tushare:balancesheet_vip.cash_reser_cb × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `depos_in_oth_bfi` | 存放同业和其它金融机构款项 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.depos_in_oth_bfi × 1 [asset_type=stock]；tushare:balancesheet_vip.depos_in_oth_bfi × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `prec_metals` | 贵金属 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.prec_metals × 1 [asset_type=stock]；tushare:balancesheet_vip.prec_metals × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `deriv_assets` | 衍生金融资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.deriv_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.deriv_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `rr_reins_une_prem` | 应收分保未到期责任准备金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.rr_reins_une_prem × 1 [asset_type=stock]；tushare:balancesheet_vip.rr_reins_une_prem × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `rr_reins_outstd_cla` | 应收分保未决赔款准备金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.rr_reins_outstd_cla × 1 [asset_type=stock]；tushare:balancesheet_vip.rr_reins_outstd_cla × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `rr_reins_lins_liab` | 应收分保寿险责任准备金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.rr_reins_lins_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.rr_reins_lins_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `rr_reins_lthins_liab` | 应收分保长期健康险责任准备金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.rr_reins_lthins_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.rr_reins_lthins_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `refund_depos` | 存出保证金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.refund_depos × 1 [asset_type=stock]；tushare:balancesheet_vip.refund_depos × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `ph_pledge_loans` | 保户质押贷款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.ph_pledge_loans × 1 [asset_type=stock]；tushare:balancesheet_vip.ph_pledge_loans × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `refund_cap_depos` | 存出资本保证金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.refund_cap_depos × 1 [asset_type=stock]；tushare:balancesheet_vip.refund_cap_depos × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `indep_acct_assets` | 独立账户资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.indep_acct_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.indep_acct_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `client_depos` | 其中：客户资金存款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.client_depos × 1 [asset_type=stock]；tushare:balancesheet_vip.client_depos × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `client_prov` | 其中：客户备付金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.client_prov × 1 [asset_type=stock]；tushare:balancesheet_vip.client_prov × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `transac_seat_fee` | 其中:交易席位费 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.transac_seat_fee × 1 [asset_type=stock]；tushare:balancesheet_vip.transac_seat_fee × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `invest_as_receiv` | 应收款项类投资 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.invest_as_receiv × 1 [asset_type=stock]；tushare:balancesheet_vip.invest_as_receiv × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `lt_borr` | 长期借款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.lt_borr × 1 [asset_type=stock]；tushare:balancesheet_vip.lt_borr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `st_borr` | 短期借款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.st_borr × 1 [asset_type=stock]；tushare:balancesheet_vip.st_borr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `cb_borr` | 向中央银行借款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.cb_borr × 1 [asset_type=stock]；tushare:balancesheet_vip.cb_borr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `depos_ib_deposits` | 吸收存款及同业存放 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.depos_ib_deposits × 1 [asset_type=stock]；tushare:balancesheet_vip.depos_ib_deposits × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `loan_oth_bank` | 拆入资金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.loan_oth_bank × 1 [asset_type=stock]；tushare:balancesheet_vip.loan_oth_bank × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `trading_fl` | 交易性金融负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.trading_fl × 1 [asset_type=stock]；tushare:balancesheet_vip.trading_fl × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `notes_payable` | 应付票据 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.notes_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.notes_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `acct_payable` | 应付账款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.acct_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.acct_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `adv_receipts` | 预收款项 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.adv_receipts × 1 [asset_type=stock]；tushare:balancesheet_vip.adv_receipts × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `sold_for_repur_fa` | 卖出回购金融资产款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.sold_for_repur_fa × 1 [asset_type=stock]；tushare:balancesheet_vip.sold_for_repur_fa × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `comm_payable` | 应付手续费及佣金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.comm_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.comm_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `payroll_payable` | 应付职工薪酬 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.payroll_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.payroll_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `taxes_payable` | 应交税费 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.taxes_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.taxes_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `int_payable` | 应付利息 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.int_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.int_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `div_payable` | 应付股利 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.div_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.div_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_payable` | 其他应付款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `acc_exp` | 预提费用 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.acc_exp × 1 [asset_type=stock]；tushare:balancesheet_vip.acc_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `deferred_inc` | 递延收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.deferred_inc × 1 [asset_type=stock]；tushare:balancesheet_vip.deferred_inc × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `st_bonds_payable` | 应付短期债券 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.st_bonds_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.st_bonds_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `payable_to_reinsurer` | 应付分保账款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.payable_to_reinsurer × 1 [asset_type=stock]；tushare:balancesheet_vip.payable_to_reinsurer × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `rsrv_insur_cont` | 保险合同准备金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.rsrv_insur_cont × 1 [asset_type=stock]；tushare:balancesheet_vip.rsrv_insur_cont × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `acting_trading_sec` | 代理买卖证券款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.acting_trading_sec × 1 [asset_type=stock]；tushare:balancesheet_vip.acting_trading_sec × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `acting_uw_sec` | 代理承销证券款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.acting_uw_sec × 1 [asset_type=stock]；tushare:balancesheet_vip.acting_uw_sec × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `non_cur_liab_due_1y` | 一年内到期的非流动负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.non_cur_liab_due_1y × 1 [asset_type=stock]；tushare:balancesheet_vip.non_cur_liab_due_1y × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_cur_liab` | 其他流动负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_cur_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_cur_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `total_cur_liab` | 流动负债合计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.total_cur_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.total_cur_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `bond_payable` | 应付债券 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.bond_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.bond_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `lt_payable` | 长期应付款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.lt_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.lt_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `specific_payables` | 专项应付款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.specific_payables × 1 [asset_type=stock]；tushare:balancesheet_vip.specific_payables × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `estimated_liab` | 预计负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.estimated_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.estimated_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `defer_tax_liab` | 递延所得税负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.defer_tax_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.defer_tax_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `defer_inc_non_cur_liab` | 递延收益-非流动负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.defer_inc_non_cur_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.defer_inc_non_cur_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_ncl` | 其他非流动负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_ncl × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_ncl × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `total_ncl` | 非流动负债合计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.total_ncl × 1 [asset_type=stock]；tushare:balancesheet_vip.total_ncl × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `depos_oth_bfi` | 同业和其它金融机构存放款项 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.depos_oth_bfi × 1 [asset_type=stock]；tushare:balancesheet_vip.depos_oth_bfi × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `deriv_liab` | 衍生金融负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.deriv_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.deriv_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `depos` | 吸收存款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.depos × 1 [asset_type=stock]；tushare:balancesheet_vip.depos × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `agency_bus_liab` | 代理业务负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.agency_bus_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.agency_bus_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_liab` | 其他负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `prem_receiv_adva` | 预收保费 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.prem_receiv_adva × 1 [asset_type=stock]；tushare:balancesheet_vip.prem_receiv_adva × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `depos_received` | 存入保证金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.depos_received × 1 [asset_type=stock]；tushare:balancesheet_vip.depos_received × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `ph_invest` | 保户储金及投资款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.ph_invest × 1 [asset_type=stock]；tushare:balancesheet_vip.ph_invest × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `reser_une_prem` | 未到期责任准备金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.reser_une_prem × 1 [asset_type=stock]；tushare:balancesheet_vip.reser_une_prem × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `reser_outstd_claims` | 未决赔款准备金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.reser_outstd_claims × 1 [asset_type=stock]；tushare:balancesheet_vip.reser_outstd_claims × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `reser_lins_liab` | 寿险责任准备金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.reser_lins_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.reser_lins_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `reser_lthins_liab` | 长期健康险责任准备金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.reser_lthins_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.reser_lthins_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `indept_acc_liab` | 独立账户负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.indept_acc_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.indept_acc_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `pledge_borr` | 其中:质押借款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.pledge_borr × 1 [asset_type=stock]；tushare:balancesheet_vip.pledge_borr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `indem_payable` | 应付赔付款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.indem_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.indem_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `policy_div_payable` | 应付保单红利 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.policy_div_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.policy_div_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `treasury_share` | 减:库存股 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.treasury_share × 1 [asset_type=stock]；tushare:balancesheet_vip.treasury_share × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `ordin_risk_reser` | 一般风险准备 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.ordin_risk_reser × 1 [asset_type=stock]；tushare:balancesheet_vip.ordin_risk_reser × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `forex_differ` | 外币报表折算差额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.forex_differ × 1 [asset_type=stock]；tushare:balancesheet_vip.forex_differ × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `invest_loss_unconf` | 未确认的投资损失 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.invest_loss_unconf × 1 [asset_type=stock]；tushare:balancesheet_vip.invest_loss_unconf × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `minority_int` | 少数股东权益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.minority_int × 1 [asset_type=stock]；tushare:balancesheet_vip.minority_int × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `total_hldr_eqy_exc_min_int` | 股东权益合计(不含少数股东权益) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.total_hldr_eqy_exc_min_int × 1 [asset_type=stock]；tushare:balancesheet_vip.total_hldr_eqy_exc_min_int × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `total_hldr_eqy_inc_min_int` | 股东权益合计(含少数股东权益) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.total_hldr_eqy_inc_min_int × 1 [asset_type=stock]；tushare:balancesheet_vip.total_hldr_eqy_inc_min_int × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `total_liab_hldr_eqy` | 负债及股东权益总计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.total_liab_hldr_eqy × 1 [asset_type=stock]；tushare:balancesheet_vip.total_liab_hldr_eqy × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `lt_payroll_payable` | 长期应付职工薪酬 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.lt_payroll_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.lt_payroll_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_comp_income` | 其他综合收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_comp_income × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_comp_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_eqt_tools` | 其他权益工具 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_eqt_tools × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_eqt_tools × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_eqt_tools_p_shr` | 其他权益工具(优先股) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_eqt_tools_p_shr × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_eqt_tools_p_shr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `lending_funds` | 融出资金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.lending_funds × 1 [asset_type=stock]；tushare:balancesheet_vip.lending_funds × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `acc_receivable` | 应收款项 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.acc_receivable × 1 [asset_type=stock]；tushare:balancesheet_vip.acc_receivable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `st_fin_payable` | 应付短期融资款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.st_fin_payable × 1 [asset_type=stock]；tushare:balancesheet_vip.st_fin_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `payables` | 应付款项 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.payables × 1 [asset_type=stock]；tushare:balancesheet_vip.payables × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `hfs_assets` | 持有待售的资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.hfs_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.hfs_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `hfs_sales` | 持有待售的负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.hfs_sales × 1 [asset_type=stock]；tushare:balancesheet_vip.hfs_sales × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `cost_fin_assets` | 以摊余成本计量的金融资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.cost_fin_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.cost_fin_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `fair_value_fin_assets` | 以公允价值计量且其变动计入其他综合收益的金融资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.fair_value_fin_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.fair_value_fin_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `cip_total` | 在建工程(合计)(元) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.cip_total × 1 [asset_type=stock]；tushare:balancesheet_vip.cip_total × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_pay_total` | 其他应付款(合计)(元) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_pay_total × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_pay_total × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `long_pay_total` | 长期应付款(合计)(元) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.long_pay_total × 1 [asset_type=stock]；tushare:balancesheet_vip.long_pay_total × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `debt_invest` | 债权投资(元) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.debt_invest × 1 [asset_type=stock]；tushare:balancesheet_vip.debt_invest × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_debt_invest` | 其他债权投资(元) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_debt_invest × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_debt_invest × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_eq_invest` | 其他权益工具投资(元) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_eq_invest × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_eq_invest × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_illiq_fin_assets` | 其他非流动金融资产(元) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_illiq_fin_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_illiq_fin_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_eq_ppbond` | 其他权益工具:永续债(元) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_eq_ppbond × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_eq_ppbond × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `receiv_financing` | 应收款项融资 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.receiv_financing × 1 [asset_type=stock]；tushare:balancesheet_vip.receiv_financing × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `use_right_assets` | 使用权资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.use_right_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.use_right_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `lease_liab` | 租赁负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.lease_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.lease_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `contract_assets` | 合同资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.contract_assets × 1 [asset_type=stock]；tushare:balancesheet_vip.contract_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `contract_liab` | 合同负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.contract_liab × 1 [asset_type=stock]；tushare:balancesheet_vip.contract_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `accounts_receiv_bill` | 应收票据及应收账款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.accounts_receiv_bill × 1 [asset_type=stock]；tushare:balancesheet_vip.accounts_receiv_bill × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `accounts_pay` | 应付票据及应付账款 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.accounts_pay × 1 [asset_type=stock]；tushare:balancesheet_vip.accounts_pay × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `oth_rcv_total` | 其他应收款(合计)（元） | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.oth_rcv_total × 1 [asset_type=stock]；tushare:balancesheet_vip.oth_rcv_total × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `fix_assets_total` | 固定资产(合计)(元) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:balancesheet.fix_assets_total × 1 [asset_type=stock]；tushare:balancesheet_vip.fix_assets_total × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=balancesheet] |
| `net_profit` | 净利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.net_profit × 1 [asset_type=stock]；tushare:cashflow_vip.net_profit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `finan_exp` | 财务费用 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.finan_exp × 1 [asset_type=stock]；tushare:cashflow_vip.finan_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_fr_sale_sg` | 销售商品、提供劳务收到的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_fr_sale_sg × 1 [asset_type=stock]；tushare:cashflow_vip.c_fr_sale_sg × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `recp_tax_rends` | 收到的税费返还 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.recp_tax_rends × 1 [asset_type=stock]；tushare:cashflow_vip.recp_tax_rends × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_depos_incr_fi` | 客户存款和同业存放款项净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_depos_incr_fi × 1 [asset_type=stock]；tushare:cashflow_vip.n_depos_incr_fi × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_incr_loans_cb` | 向中央银行借款净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_incr_loans_cb × 1 [asset_type=stock]；tushare:cashflow_vip.n_incr_loans_cb × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_inc_borr_oth_fi` | 向其他金融机构拆入资金净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_inc_borr_oth_fi × 1 [asset_type=stock]；tushare:cashflow_vip.n_inc_borr_oth_fi × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `prem_fr_orig_contr` | 收到原保险合同保费取得的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.prem_fr_orig_contr × 1 [asset_type=stock]；tushare:cashflow_vip.prem_fr_orig_contr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_incr_insured_dep` | 保户储金净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_incr_insured_dep × 1 [asset_type=stock]；tushare:cashflow_vip.n_incr_insured_dep × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_reinsur_prem` | 收到再保业务现金净额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_reinsur_prem × 1 [asset_type=stock]；tushare:cashflow_vip.n_reinsur_prem × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_incr_disp_tfa` | 处置交易性金融资产净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_incr_disp_tfa × 1 [asset_type=stock]；tushare:cashflow_vip.n_incr_disp_tfa × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `ifc_cash_incr` | 收取利息和手续费净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.ifc_cash_incr × 1 [asset_type=stock]；tushare:cashflow_vip.ifc_cash_incr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_incr_disp_faas` | 处置可供出售金融资产净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_incr_disp_faas × 1 [asset_type=stock]；tushare:cashflow_vip.n_incr_disp_faas × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_incr_loans_oth_bank` | 拆入资金净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_incr_loans_oth_bank × 1 [asset_type=stock]；tushare:cashflow_vip.n_incr_loans_oth_bank × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_cap_incr_repur` | 回购业务资金净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_cap_incr_repur × 1 [asset_type=stock]；tushare:cashflow_vip.n_cap_incr_repur × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_fr_oth_operate_a` | 收到其他与经营活动有关的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_fr_oth_operate_a × 1 [asset_type=stock]；tushare:cashflow_vip.c_fr_oth_operate_a × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_inf_fr_operate_a` | 经营活动现金流入小计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_inf_fr_operate_a × 1 [asset_type=stock]；tushare:cashflow_vip.c_inf_fr_operate_a × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_paid_goods_s` | 购买商品、接受劳务支付的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_paid_goods_s × 1 [asset_type=stock]；tushare:cashflow_vip.c_paid_goods_s × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_paid_to_for_empl` | 支付给职工以及为职工支付的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_paid_to_for_empl × 1 [asset_type=stock]；tushare:cashflow_vip.c_paid_to_for_empl × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_paid_for_taxes` | 支付的各项税费 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_paid_for_taxes × 1 [asset_type=stock]；tushare:cashflow_vip.c_paid_for_taxes × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_incr_clt_loan_adv` | 客户贷款及垫款净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_incr_clt_loan_adv × 1 [asset_type=stock]；tushare:cashflow_vip.n_incr_clt_loan_adv × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_incr_dep_cbob` | 存放央行和同业款项净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_incr_dep_cbob × 1 [asset_type=stock]；tushare:cashflow_vip.n_incr_dep_cbob × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_pay_claims_orig_inco` | 支付原保险合同赔付款项的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_pay_claims_orig_inco × 1 [asset_type=stock]；tushare:cashflow_vip.c_pay_claims_orig_inco × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `pay_handling_chrg` | 支付手续费的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.pay_handling_chrg × 1 [asset_type=stock]；tushare:cashflow_vip.pay_handling_chrg × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `pay_comm_insur_plcy` | 支付保单红利的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.pay_comm_insur_plcy × 1 [asset_type=stock]；tushare:cashflow_vip.pay_comm_insur_plcy × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `oth_cash_pay_oper_act` | 支付其他与经营活动有关的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.oth_cash_pay_oper_act × 1 [asset_type=stock]；tushare:cashflow_vip.oth_cash_pay_oper_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `st_cash_out_act` | 经营活动现金流出小计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.st_cash_out_act × 1 [asset_type=stock]；tushare:cashflow_vip.st_cash_out_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `oth_recp_ral_inv_act` | 收到其他与投资活动有关的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.oth_recp_ral_inv_act × 1 [asset_type=stock]；tushare:cashflow_vip.oth_recp_ral_inv_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_disp_withdrwl_invest` | 收回投资收到的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_disp_withdrwl_invest × 1 [asset_type=stock]；tushare:cashflow_vip.c_disp_withdrwl_invest × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_recp_return_invest` | 取得投资收益收到的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_recp_return_invest × 1 [asset_type=stock]；tushare:cashflow_vip.c_recp_return_invest × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_recp_disp_fiolta` | 处置固定资产、无形资产和其他长期资产收回的现金净额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_recp_disp_fiolta × 1 [asset_type=stock]；tushare:cashflow_vip.n_recp_disp_fiolta × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_recp_disp_sobu` | 处置子公司及其他营业单位收到的现金净额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_recp_disp_sobu × 1 [asset_type=stock]；tushare:cashflow_vip.n_recp_disp_sobu × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `stot_inflows_inv_act` | 投资活动现金流入小计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.stot_inflows_inv_act × 1 [asset_type=stock]；tushare:cashflow_vip.stot_inflows_inv_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_pay_acq_const_fiolta` | 购建固定资产、无形资产和其他长期资产支付的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_pay_acq_const_fiolta × 1 [asset_type=stock]；tushare:cashflow_vip.c_pay_acq_const_fiolta × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_paid_invest` | 投资支付的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_paid_invest × 1 [asset_type=stock]；tushare:cashflow_vip.c_paid_invest × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_disp_subs_oth_biz` | 取得子公司及其他营业单位支付的现金净额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_disp_subs_oth_biz × 1 [asset_type=stock]；tushare:cashflow_vip.n_disp_subs_oth_biz × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `oth_pay_ral_inv_act` | 支付其他与投资活动有关的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.oth_pay_ral_inv_act × 1 [asset_type=stock]；tushare:cashflow_vip.oth_pay_ral_inv_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_incr_pledge_loan` | 质押贷款净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_incr_pledge_loan × 1 [asset_type=stock]；tushare:cashflow_vip.n_incr_pledge_loan × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `stot_out_inv_act` | 投资活动现金流出小计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.stot_out_inv_act × 1 [asset_type=stock]；tushare:cashflow_vip.stot_out_inv_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_cashflow_inv_act` | 投资活动产生的现金流量净额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_cashflow_inv_act × 1 [asset_type=stock]；tushare:cashflow_vip.n_cashflow_inv_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_recp_borrow` | 取得借款收到的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_recp_borrow × 1 [asset_type=stock]；tushare:cashflow_vip.c_recp_borrow × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `proc_issue_bonds` | 发行债券收到的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.proc_issue_bonds × 1 [asset_type=stock]；tushare:cashflow_vip.proc_issue_bonds × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `oth_cash_recp_ral_fnc_act` | 收到其他与筹资活动有关的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.oth_cash_recp_ral_fnc_act × 1 [asset_type=stock]；tushare:cashflow_vip.oth_cash_recp_ral_fnc_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `stot_cash_in_fnc_act` | 筹资活动现金流入小计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.stot_cash_in_fnc_act × 1 [asset_type=stock]；tushare:cashflow_vip.stot_cash_in_fnc_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `free_cashflow` | 企业自由现金流量 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.free_cashflow × 1 [asset_type=stock]；tushare:cashflow_vip.free_cashflow × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_prepay_amt_borr` | 偿还债务支付的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_prepay_amt_borr × 1 [asset_type=stock]；tushare:cashflow_vip.c_prepay_amt_borr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_pay_dist_dpcp_int_exp` | 分配股利、利润或偿付利息支付的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_pay_dist_dpcp_int_exp × 1 [asset_type=stock]；tushare:cashflow_vip.c_pay_dist_dpcp_int_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `incl_dvd_profit_paid_sc_ms` | 其中:子公司支付给少数股东的股利、利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.incl_dvd_profit_paid_sc_ms × 1 [asset_type=stock]；tushare:cashflow_vip.incl_dvd_profit_paid_sc_ms × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `oth_cashpay_ral_fnc_act` | 支付其他与筹资活动有关的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.oth_cashpay_ral_fnc_act × 1 [asset_type=stock]；tushare:cashflow_vip.oth_cashpay_ral_fnc_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `stot_cashout_fnc_act` | 筹资活动现金流出小计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.stot_cashout_fnc_act × 1 [asset_type=stock]；tushare:cashflow_vip.stot_cashout_fnc_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_cash_flows_fnc_act` | 筹资活动产生的现金流量净额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_cash_flows_fnc_act × 1 [asset_type=stock]；tushare:cashflow_vip.n_cash_flows_fnc_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `eff_fx_flu_cash` | 汇率变动对现金的影响 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.eff_fx_flu_cash × 1 [asset_type=stock]；tushare:cashflow_vip.eff_fx_flu_cash × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `n_incr_cash_cash_equ` | 现金及现金等价物净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.n_incr_cash_cash_equ × 1 [asset_type=stock]；tushare:cashflow_vip.n_incr_cash_cash_equ × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_cash_equ_beg_period` | 期初现金及现金等价物余额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_cash_equ_beg_period × 1 [asset_type=stock]；tushare:cashflow_vip.c_cash_equ_beg_period × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_cash_equ_end_period` | 期末现金及现金等价物余额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_cash_equ_end_period × 1 [asset_type=stock]；tushare:cashflow_vip.c_cash_equ_end_period × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `c_recp_cap_contrib` | 吸收投资收到的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.c_recp_cap_contrib × 1 [asset_type=stock]；tushare:cashflow_vip.c_recp_cap_contrib × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `incl_cash_rec_saims` | 其中:子公司吸收少数股东投资收到的现金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.incl_cash_rec_saims × 1 [asset_type=stock]；tushare:cashflow_vip.incl_cash_rec_saims × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `uncon_invest_loss` | 未确认投资损失 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.uncon_invest_loss × 1 [asset_type=stock]；tushare:cashflow_vip.uncon_invest_loss × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `prov_depr_assets` | 加:资产减值准备 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.prov_depr_assets × 1 [asset_type=stock]；tushare:cashflow_vip.prov_depr_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `depr_fa_coga_dpba` | 固定资产折旧、油气资产折耗、生产性生物资产折旧 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.depr_fa_coga_dpba × 1 [asset_type=stock]；tushare:cashflow_vip.depr_fa_coga_dpba × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `amort_intang_assets` | 无形资产摊销 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.amort_intang_assets × 1 [asset_type=stock]；tushare:cashflow_vip.amort_intang_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `lt_amort_deferred_exp` | 长期待摊费用摊销 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.lt_amort_deferred_exp × 1 [asset_type=stock]；tushare:cashflow_vip.lt_amort_deferred_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `decr_deferred_exp` | 待摊费用减少 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.decr_deferred_exp × 1 [asset_type=stock]；tushare:cashflow_vip.decr_deferred_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `incr_acc_exp` | 预提费用增加 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.incr_acc_exp × 1 [asset_type=stock]；tushare:cashflow_vip.incr_acc_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `loss_disp_fiolta` | 处置固定、无形资产和其他长期资产的损失 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.loss_disp_fiolta × 1 [asset_type=stock]；tushare:cashflow_vip.loss_disp_fiolta × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `loss_scr_fa` | 固定资产报废损失 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.loss_scr_fa × 1 [asset_type=stock]；tushare:cashflow_vip.loss_scr_fa × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `loss_fv_chg` | 公允价值变动损失 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.loss_fv_chg × 1 [asset_type=stock]；tushare:cashflow_vip.loss_fv_chg × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `invest_loss` | 投资损失 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.invest_loss × 1 [asset_type=stock]；tushare:cashflow_vip.invest_loss × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `decr_def_inc_tax_assets` | 递延所得税资产减少 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.decr_def_inc_tax_assets × 1 [asset_type=stock]；tushare:cashflow_vip.decr_def_inc_tax_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `incr_def_inc_tax_liab` | 递延所得税负债增加 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.incr_def_inc_tax_liab × 1 [asset_type=stock]；tushare:cashflow_vip.incr_def_inc_tax_liab × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `decr_inventories` | 存货的减少 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.decr_inventories × 1 [asset_type=stock]；tushare:cashflow_vip.decr_inventories × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `decr_oper_payable` | 经营性应收项目的减少 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.decr_oper_payable × 1 [asset_type=stock]；tushare:cashflow_vip.decr_oper_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `incr_oper_payable` | 经营性应付项目的增加 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.incr_oper_payable × 1 [asset_type=stock]；tushare:cashflow_vip.incr_oper_payable × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `others` | 其他 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.others × 1 [asset_type=stock]；tushare:cashflow_vip.others × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `im_net_cashflow_oper_act` | 经营活动产生的现金流量净额(间接法) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.im_net_cashflow_oper_act × 1 [asset_type=stock]；tushare:cashflow_vip.im_net_cashflow_oper_act × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `conv_debt_into_cap` | 债务转为资本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.conv_debt_into_cap × 1 [asset_type=stock]；tushare:cashflow_vip.conv_debt_into_cap × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `conv_copbonds_due_within_1y` | 一年内到期的可转换公司债券 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.conv_copbonds_due_within_1y × 1 [asset_type=stock]；tushare:cashflow_vip.conv_copbonds_due_within_1y × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `fa_fnc_leases` | 融资租入固定资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.fa_fnc_leases × 1 [asset_type=stock]；tushare:cashflow_vip.fa_fnc_leases × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `im_n_incr_cash_equ` | 现金及现金等价物净增加额(间接法) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.im_n_incr_cash_equ × 1 [asset_type=stock]；tushare:cashflow_vip.im_n_incr_cash_equ × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `net_dism_capital_add` | 拆出资金净增加额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.net_dism_capital_add × 1 [asset_type=stock]；tushare:cashflow_vip.net_dism_capital_add × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `net_cash_rece_sec` | 代理买卖证券收到的现金净额(元) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.net_cash_rece_sec × 1 [asset_type=stock]；tushare:cashflow_vip.net_cash_rece_sec × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `use_right_asset_dep` | 使用权资产折旧 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.use_right_asset_dep × 1 [asset_type=stock]；tushare:cashflow_vip.use_right_asset_dep × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `oth_loss_asset` | 其他资产减值损失 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.oth_loss_asset × 1 [asset_type=stock]；tushare:cashflow_vip.oth_loss_asset × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `end_bal_cash` | 现金的期末余额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.end_bal_cash × 1 [asset_type=stock]；tushare:cashflow_vip.end_bal_cash × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `beg_bal_cash` | 减:现金的期初余额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.beg_bal_cash × 1 [asset_type=stock]；tushare:cashflow_vip.beg_bal_cash × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `end_bal_cash_equ` | 加:现金等价物的期末余额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.end_bal_cash_equ × 1 [asset_type=stock]；tushare:cashflow_vip.end_bal_cash_equ × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `beg_bal_cash_equ` | 减:现金等价物的期初余额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:cashflow.beg_bal_cash_equ × 1 [asset_type=stock]；tushare:cashflow_vip.beg_bal_cash_equ × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=cashflow] |
| `eps` | 基本每股收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.eps × 1 [asset_type=stock]；tushare:fina_indicator_vip.eps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `dt_eps` | 稀释每股收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.dt_eps × 1 [asset_type=stock]；tushare:fina_indicator_vip.dt_eps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `total_revenue_ps` | 每股营业总收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.total_revenue_ps × 1 [asset_type=stock]；tushare:fina_indicator_vip.total_revenue_ps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `revenue_ps` | 每股营业收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.revenue_ps × 1 [asset_type=stock]；tushare:fina_indicator_vip.revenue_ps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `capital_rese_ps` | 每股资本公积 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.capital_rese_ps × 1 [asset_type=stock]；tushare:fina_indicator_vip.capital_rese_ps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `surplus_rese_ps` | 每股盈余公积 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.surplus_rese_ps × 1 [asset_type=stock]；tushare:fina_indicator_vip.surplus_rese_ps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `undist_profit_ps` | 每股未分配利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.undist_profit_ps × 1 [asset_type=stock]；tushare:fina_indicator_vip.undist_profit_ps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `extra_item` | 非经常性损益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.extra_item × 1 [asset_type=stock]；tushare:fina_indicator_vip.extra_item × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `profit_dedt` | 扣除非经常性损益后的净利润（扣非净利润） | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.profit_dedt × 1 [asset_type=stock]；tushare:fina_indicator_vip.profit_dedt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `gross_margin` | 毛利 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.gross_margin × 1 [asset_type=stock]；tushare:fina_indicator_vip.gross_margin × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `current_ratio` | 流动比率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.current_ratio × 1 [asset_type=stock]；tushare:fina_indicator_vip.current_ratio × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `quick_ratio` | 速动比率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.quick_ratio × 1 [asset_type=stock]；tushare:fina_indicator_vip.quick_ratio × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `cash_ratio` | 保守速动比率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.cash_ratio × 1 [asset_type=stock]；tushare:fina_indicator_vip.cash_ratio × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `invturn_days` | 存货周转天数 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.invturn_days × 1 [asset_type=stock]；tushare:fina_indicator_vip.invturn_days × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `arturn_days` | 应收账款周转天数 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.arturn_days × 1 [asset_type=stock]；tushare:fina_indicator_vip.arturn_days × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `inv_turn` | 存货周转率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.inv_turn × 1 [asset_type=stock]；tushare:fina_indicator_vip.inv_turn × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ar_turn` | 应收账款周转率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ar_turn × 1 [asset_type=stock]；tushare:fina_indicator_vip.ar_turn × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ca_turn` | 流动资产周转率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ca_turn × 1 [asset_type=stock]；tushare:fina_indicator_vip.ca_turn × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `fa_turn` | 固定资产周转率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.fa_turn × 1 [asset_type=stock]；tushare:fina_indicator_vip.fa_turn × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `assets_turn` | 总资产周转率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.assets_turn × 1 [asset_type=stock]；tushare:fina_indicator_vip.assets_turn × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `op_income` | 经营活动净收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.op_income × 1 [asset_type=stock]；tushare:fina_indicator_vip.op_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `valuechange_income` | 价值变动净收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.valuechange_income × 1 [asset_type=stock]；tushare:fina_indicator_vip.valuechange_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `interst_income` | 利息费用 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.interst_income × 1 [asset_type=stock]；tushare:fina_indicator_vip.interst_income × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `daa` | 折旧与摊销 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.daa × 1 [asset_type=stock]；tushare:fina_indicator_vip.daa × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `fcff` | 企业自由现金流量 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.fcff × 1 [asset_type=stock]；tushare:fina_indicator_vip.fcff × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `fcfe` | 股权自由现金流量 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.fcfe × 1 [asset_type=stock]；tushare:fina_indicator_vip.fcfe × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `current_exint` | 无息流动负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.current_exint × 1 [asset_type=stock]；tushare:fina_indicator_vip.current_exint × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `noncurrent_exint` | 无息非流动负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.noncurrent_exint × 1 [asset_type=stock]；tushare:fina_indicator_vip.noncurrent_exint × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `interestdebt` | 带息债务 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.interestdebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.interestdebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `netdebt` | 净债务 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.netdebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.netdebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `tangible_asset` | 有形资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.tangible_asset × 1 [asset_type=stock]；tushare:fina_indicator_vip.tangible_asset × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `working_capital` | 营运资金 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.working_capital × 1 [asset_type=stock]；tushare:fina_indicator_vip.working_capital × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `networking_capital` | 营运流动资本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.networking_capital × 1 [asset_type=stock]；tushare:fina_indicator_vip.networking_capital × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `invest_capital` | 全部投入资本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.invest_capital × 1 [asset_type=stock]；tushare:fina_indicator_vip.invest_capital × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `retained_earnings` | 留存收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.retained_earnings × 1 [asset_type=stock]；tushare:fina_indicator_vip.retained_earnings × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `diluted2_eps` | 期末摊薄每股收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.diluted2_eps × 1 [asset_type=stock]；tushare:fina_indicator_vip.diluted2_eps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `bps` | 每股净资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.bps × 1 [asset_type=stock]；tushare:fina_indicator_vip.bps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ocfps` | 每股经营活动产生的现金流量净额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ocfps × 1 [asset_type=stock]；tushare:fina_indicator_vip.ocfps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `retainedps` | 每股留存收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.retainedps × 1 [asset_type=stock]；tushare:fina_indicator_vip.retainedps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `cfps` | 每股现金流量净额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.cfps × 1 [asset_type=stock]；tushare:fina_indicator_vip.cfps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ebit_ps` | 每股息税前利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ebit_ps × 1 [asset_type=stock]；tushare:fina_indicator_vip.ebit_ps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `fcff_ps` | 每股企业自由现金流量 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.fcff_ps × 1 [asset_type=stock]；tushare:fina_indicator_vip.fcff_ps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `fcfe_ps` | 每股股东自由现金流量 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.fcfe_ps × 1 [asset_type=stock]；tushare:fina_indicator_vip.fcfe_ps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `netprofit_margin` | 销售净利率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.netprofit_margin × 1 [asset_type=stock]；tushare:fina_indicator_vip.netprofit_margin × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `grossprofit_margin` | 销售毛利率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.grossprofit_margin × 1 [asset_type=stock]；tushare:fina_indicator_vip.grossprofit_margin × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `cogs_of_sales` | 销售成本率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.cogs_of_sales × 1 [asset_type=stock]；tushare:fina_indicator_vip.cogs_of_sales × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `expense_of_sales` | 销售期间费用率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.expense_of_sales × 1 [asset_type=stock]；tushare:fina_indicator_vip.expense_of_sales × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `profit_to_gr` | 净利润/营业总收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.profit_to_gr × 1 [asset_type=stock]；tushare:fina_indicator_vip.profit_to_gr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `saleexp_to_gr` | 销售费用/营业总收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.saleexp_to_gr × 1 [asset_type=stock]；tushare:fina_indicator_vip.saleexp_to_gr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `adminexp_of_gr` | 管理费用/营业总收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.adminexp_of_gr × 1 [asset_type=stock]；tushare:fina_indicator_vip.adminexp_of_gr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `finaexp_of_gr` | 财务费用/营业总收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.finaexp_of_gr × 1 [asset_type=stock]；tushare:fina_indicator_vip.finaexp_of_gr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `impai_ttm` | 资产减值损失/营业总收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.impai_ttm × 1 [asset_type=stock]；tushare:fina_indicator_vip.impai_ttm × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `gc_of_gr` | 营业总成本/营业总收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.gc_of_gr × 1 [asset_type=stock]；tushare:fina_indicator_vip.gc_of_gr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `op_of_gr` | 营业利润/营业总收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.op_of_gr × 1 [asset_type=stock]；tushare:fina_indicator_vip.op_of_gr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ebit_of_gr` | 息税前利润/营业总收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ebit_of_gr × 1 [asset_type=stock]；tushare:fina_indicator_vip.ebit_of_gr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `roe_waa` | 加权平均净资产收益率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.roe_waa × 1 [asset_type=stock]；tushare:fina_indicator_vip.roe_waa × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `roe_dt` | 净资产收益率(扣除非经常损益) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.roe_dt × 1 [asset_type=stock]；tushare:fina_indicator_vip.roe_dt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `roa` | 总资产报酬率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.roa × 1 [asset_type=stock]；tushare:fina_indicator_vip.roa × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `npta` | 总资产净利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.npta × 1 [asset_type=stock]；tushare:fina_indicator_vip.npta × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `roic` | 投入资本回报率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.roic × 1 [asset_type=stock]；tushare:fina_indicator_vip.roic × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `roe_yearly` | 年化净资产收益率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.roe_yearly × 1 [asset_type=stock]；tushare:fina_indicator_vip.roe_yearly × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `roa2_yearly` | 年化总资产报酬率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.roa2_yearly × 1 [asset_type=stock]；tushare:fina_indicator_vip.roa2_yearly × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `roe_avg` | 平均净资产收益率(增发条件) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.roe_avg × 1 [asset_type=stock]；tushare:fina_indicator_vip.roe_avg × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `opincome_of_ebt` | 经营活动净收益/利润总额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.opincome_of_ebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.opincome_of_ebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `investincome_of_ebt` | 价值变动净收益/利润总额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.investincome_of_ebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.investincome_of_ebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `n_op_profit_of_ebt` | 营业外收支净额/利润总额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.n_op_profit_of_ebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.n_op_profit_of_ebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `tax_to_ebt` | 所得税/利润总额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.tax_to_ebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.tax_to_ebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `dtprofit_to_profit` | 扣除非经常损益后的净利润/净利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.dtprofit_to_profit × 1 [asset_type=stock]；tushare:fina_indicator_vip.dtprofit_to_profit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `salescash_to_or` | 销售商品提供劳务收到的现金/营业收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.salescash_to_or × 1 [asset_type=stock]；tushare:fina_indicator_vip.salescash_to_or × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ocf_to_or` | 经营活动产生的现金流量净额/营业收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ocf_to_or × 1 [asset_type=stock]；tushare:fina_indicator_vip.ocf_to_or × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ocf_to_opincome` | 经营活动产生的现金流量净额/经营活动净收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ocf_to_opincome × 1 [asset_type=stock]；tushare:fina_indicator_vip.ocf_to_opincome × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `capitalized_to_da` | 资本支出/折旧和摊销 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.capitalized_to_da × 1 [asset_type=stock]；tushare:fina_indicator_vip.capitalized_to_da × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `debt_to_assets` | 资产负债率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.debt_to_assets × 1 [asset_type=stock]；tushare:fina_indicator_vip.debt_to_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `assets_to_eqt` | 权益乘数 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.assets_to_eqt × 1 [asset_type=stock]；tushare:fina_indicator_vip.assets_to_eqt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `dp_assets_to_eqt` | 权益乘数(杜邦分析) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.dp_assets_to_eqt × 1 [asset_type=stock]；tushare:fina_indicator_vip.dp_assets_to_eqt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ca_to_assets` | 流动资产/总资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ca_to_assets × 1 [asset_type=stock]；tushare:fina_indicator_vip.ca_to_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `nca_to_assets` | 非流动资产/总资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.nca_to_assets × 1 [asset_type=stock]；tushare:fina_indicator_vip.nca_to_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `tbassets_to_totalassets` | 有形资产/总资产 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.tbassets_to_totalassets × 1 [asset_type=stock]；tushare:fina_indicator_vip.tbassets_to_totalassets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `int_to_talcap` | 带息债务/全部投入资本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.int_to_talcap × 1 [asset_type=stock]；tushare:fina_indicator_vip.int_to_talcap × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `eqt_to_talcapital` | 归属于母公司的股东权益/全部投入资本 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.eqt_to_talcapital × 1 [asset_type=stock]；tushare:fina_indicator_vip.eqt_to_talcapital × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `currentdebt_to_debt` | 流动负债/负债合计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.currentdebt_to_debt × 1 [asset_type=stock]；tushare:fina_indicator_vip.currentdebt_to_debt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `longdeb_to_debt` | 非流动负债/负债合计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.longdeb_to_debt × 1 [asset_type=stock]；tushare:fina_indicator_vip.longdeb_to_debt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ocf_to_shortdebt` | 经营活动产生的现金流量净额/流动负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ocf_to_shortdebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.ocf_to_shortdebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `debt_to_eqt` | 产权比率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.debt_to_eqt × 1 [asset_type=stock]；tushare:fina_indicator_vip.debt_to_eqt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `eqt_to_debt` | 归属于母公司的股东权益/负债合计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.eqt_to_debt × 1 [asset_type=stock]；tushare:fina_indicator_vip.eqt_to_debt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `eqt_to_interestdebt` | 归属于母公司的股东权益/带息债务 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.eqt_to_interestdebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.eqt_to_interestdebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `tangibleasset_to_debt` | 有形资产/负债合计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.tangibleasset_to_debt × 1 [asset_type=stock]；tushare:fina_indicator_vip.tangibleasset_to_debt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `tangasset_to_intdebt` | 有形资产/带息债务 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.tangasset_to_intdebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.tangasset_to_intdebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `tangibleasset_to_netdebt` | 有形资产/净债务 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.tangibleasset_to_netdebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.tangibleasset_to_netdebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ocf_to_debt` | 经营活动产生的现金流量净额/负债合计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ocf_to_debt × 1 [asset_type=stock]；tushare:fina_indicator_vip.ocf_to_debt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ocf_to_interestdebt` | 经营活动产生的现金流量净额/带息债务 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ocf_to_interestdebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.ocf_to_interestdebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ocf_to_netdebt` | 经营活动产生的现金流量净额/净债务 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ocf_to_netdebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.ocf_to_netdebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ebit_to_interest` | 已获利息倍数(EBIT/利息费用) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ebit_to_interest × 1 [asset_type=stock]；tushare:fina_indicator_vip.ebit_to_interest × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `longdebt_to_workingcapital` | 长期债务与营运资金比率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.longdebt_to_workingcapital × 1 [asset_type=stock]；tushare:fina_indicator_vip.longdebt_to_workingcapital × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ebitda_to_debt` | 息税折旧摊销前利润/负债合计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ebitda_to_debt × 1 [asset_type=stock]；tushare:fina_indicator_vip.ebitda_to_debt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `turn_days` | 营业周期 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.turn_days × 1 [asset_type=stock]；tushare:fina_indicator_vip.turn_days × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `roa_yearly` | 年化总资产净利率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.roa_yearly × 1 [asset_type=stock]；tushare:fina_indicator_vip.roa_yearly × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `roa_dp` | 总资产净利率(杜邦分析) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.roa_dp × 1 [asset_type=stock]；tushare:fina_indicator_vip.roa_dp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `fixed_assets` | 固定资产合计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.fixed_assets × 1 [asset_type=stock]；tushare:fina_indicator_vip.fixed_assets × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `profit_prefin_exp` | 扣除财务费用前营业利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.profit_prefin_exp × 1 [asset_type=stock]；tushare:fina_indicator_vip.profit_prefin_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `non_op_profit` | 非营业利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.non_op_profit × 1 [asset_type=stock]；tushare:fina_indicator_vip.non_op_profit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `op_to_ebt` | 营业利润／利润总额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.op_to_ebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.op_to_ebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `nop_to_ebt` | 非营业利润／利润总额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.nop_to_ebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.nop_to_ebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ocf_to_profit` | 经营活动产生的现金流量净额／营业利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.ocf_to_profit × 1 [asset_type=stock]；tushare:fina_indicator_vip.ocf_to_profit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `cash_to_liqdebt` | 货币资金／流动负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.cash_to_liqdebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.cash_to_liqdebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `cash_to_liqdebt_withinterest` | 货币资金／带息流动负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.cash_to_liqdebt_withinterest × 1 [asset_type=stock]；tushare:fina_indicator_vip.cash_to_liqdebt_withinterest × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `op_to_liqdebt` | 营业利润／流动负债 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.op_to_liqdebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.op_to_liqdebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `op_to_debt` | 营业利润／负债合计 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.op_to_debt × 1 [asset_type=stock]；tushare:fina_indicator_vip.op_to_debt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `roic_yearly` | 年化投入资本回报率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.roic_yearly × 1 [asset_type=stock]；tushare:fina_indicator_vip.roic_yearly × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `total_fa_trun` | 固定资产合计周转率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.total_fa_trun × 1 [asset_type=stock]；tushare:fina_indicator_vip.total_fa_trun × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `profit_to_op` | 利润总额／营业收入 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.profit_to_op × 1 [asset_type=stock]；tushare:fina_indicator_vip.profit_to_op × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_opincome` | 经营活动单季度净收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_opincome × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_opincome × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_investincome` | 价值变动单季度净收益 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_investincome × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_investincome × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_dtprofit` | 扣除非经常损益后的单季度净利润 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_dtprofit × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_dtprofit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_eps` | 每股收益(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_eps × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_eps × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_netprofit_margin` | 销售净利率(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_netprofit_margin × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_netprofit_margin × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_gsprofit_margin` | 销售毛利率(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_gsprofit_margin × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_gsprofit_margin × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_exp_to_sales` | 销售期间费用率(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_exp_to_sales × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_exp_to_sales × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_profit_to_gr` | 净利润／营业总收入(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_profit_to_gr × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_profit_to_gr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_saleexp_to_gr` | 销售费用／营业总收入 (单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_saleexp_to_gr × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_saleexp_to_gr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_adminexp_to_gr` | 管理费用／营业总收入 (单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_adminexp_to_gr × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_adminexp_to_gr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_finaexp_to_gr` | 财务费用／营业总收入 (单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_finaexp_to_gr × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_finaexp_to_gr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_impair_to_gr_ttm` | 资产减值损失／营业总收入(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_impair_to_gr_ttm × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_impair_to_gr_ttm × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_gc_to_gr` | 营业总成本／营业总收入 (单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_gc_to_gr × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_gc_to_gr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_op_to_gr` | 营业利润／营业总收入(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_op_to_gr × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_op_to_gr × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_roe` | 净资产收益率(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_roe × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_roe × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_dt_roe` | 净资产单季度收益率(扣除非经常损益) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_dt_roe × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_dt_roe × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_npta` | 总资产净利润(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_npta × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_npta × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_opincome_to_ebt` | 经营活动净收益／利润总额(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_opincome_to_ebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_opincome_to_ebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_investincome_to_ebt` | 价值变动净收益／利润总额(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_investincome_to_ebt × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_investincome_to_ebt × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_dtprofit_to_profit` | 扣除非经常损益后的净利润／净利润(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_dtprofit_to_profit × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_dtprofit_to_profit × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_salescash_to_or` | 销售商品提供劳务收到的现金／营业收入(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_salescash_to_or × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_salescash_to_or × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_ocf_to_sales` | 经营活动产生的现金流量净额／营业收入(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_ocf_to_sales × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_ocf_to_sales × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_ocf_to_or` | 经营活动产生的现金流量净额／经营活动净收益(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.q_ocf_to_or × 1 [asset_type=stock]；tushare:fina_indicator_vip.q_ocf_to_or × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `basic_eps_yoy` | 基本每股收益同比增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.basic_eps_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.basic_eps_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `dt_eps_yoy` | 稀释每股收益同比增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.dt_eps_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.dt_eps_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `cfps_yoy` | 每股经营活动产生的现金流量净额同比增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.cfps_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.cfps_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `op_yoy` | 营业利润同比增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.op_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.op_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ebt_yoy` | 利润总额同比增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.ebt_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.ebt_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `netprofit_yoy` | 归属母公司股东的净利润同比增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.netprofit_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.netprofit_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `dt_netprofit_yoy` | 归属母公司股东的净利润-扣除非经常损益同比增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.dt_netprofit_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.dt_netprofit_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `ocf_yoy` | 经营活动产生的现金流量净额同比增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.ocf_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.ocf_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `roe_yoy` | 净资产收益率(摊薄)同比增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.roe_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.roe_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `bps_yoy` | 每股净资产相对年初增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.bps_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.bps_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `assets_yoy` | 资产总计相对年初增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.assets_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.assets_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `eqt_yoy` | 归属母公司的股东权益相对年初增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.eqt_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.eqt_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `tr_yoy` | 营业总收入同比增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.tr_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.tr_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `or_yoy` | 营业收入同比增长率(%) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.or_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.or_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_gr_yoy` | 营业总收入同比增长率(%)(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.q_gr_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.q_gr_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_gr_qoq` | 营业总收入环比增长率(%)(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.q_gr_qoq × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.q_gr_qoq × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_sales_yoy` | 营业收入同比增长率(%)(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.q_sales_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.q_sales_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_sales_qoq` | 营业收入环比增长率(%)(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.q_sales_qoq × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.q_sales_qoq × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_op_yoy` | 营业利润同比增长率(%)(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.q_op_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.q_op_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_op_qoq` | 营业利润环比增长率(%)(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.q_op_qoq × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.q_op_qoq × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_profit_yoy` | 净利润同比增长率(%)(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.q_profit_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.q_profit_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_profit_qoq` | 净利润环比增长率(%)(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.q_profit_qoq × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.q_profit_qoq × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_netprofit_yoy` | 归属母公司股东的净利润同比增长率(%)(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.q_netprofit_yoy × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.q_netprofit_yoy × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `q_netprofit_qoq` | 归属母公司股东的净利润环比增长率(%)(单季度) | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:fina_indicator.q_netprofit_qoq × 1（源单位 %） [asset_type=stock]；tushare:fina_indicator_vip.q_netprofit_qoq × 1（源单位 %） [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `equity_yoy` | 净资产同比增长率 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.equity_yoy × 1 [asset_type=stock]；tushare:fina_indicator_vip.equity_yoy × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |
| `actual_announcement_date` | 实际公告日期；实际公告日期；源 f_ann_date；只有日期时不能据此制造精确发布时间或历史修订顺序。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:income.f_ann_date × 1 [asset_type=stock]；tushare:balancesheet.f_ann_date × 1 [asset_type=stock]；tushare:cashflow.f_ann_date × 1 [asset_type=stock]；tushare:income_vip.f_ann_date × 1 [asset_type=stock]；tushare:balancesheet_vip.f_ann_date × 1 [asset_type=stock]；tushare:cashflow_vip.f_ann_date × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income]；源字段 [asset_type=stock,statement_type=balancesheet]；源字段 [asset_type=stock,statement_type=cashflow] |
| `end_type` | 报告期类型 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:income.end_type × 1 [asset_type=stock]；tushare:balancesheet.end_type × 1 [asset_type=stock]；tushare:cashflow.end_type × 1 [asset_type=stock]；tushare:income_vip.end_type × 1 [asset_type=stock]；tushare:balancesheet_vip.end_type × 1 [asset_type=stock]；tushare:cashflow_vip.end_type × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income]；源字段 [asset_type=stock,statement_type=balancesheet]；源字段 [asset_type=stock,statement_type=cashflow] |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Date32 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Int64 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `research_and_development_expense` | 研发费用；研发费用；来源 income.rd_exp，两个接口含义不同，禁止合并 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:income.rd_exp × 1 [asset_type=stock]；tushare:income_vip.rd_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=income] |
| `research_and_development_investment` | 研发投入合计；研发投入合计；来源 fina_indicator.rd_exp，两个接口含义不同，禁止合并 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fina_indicator.rd_exp × 1 [asset_type=stock]；tushare:fina_indicator_vip.rd_exp × 1 [asset_type=stock]；源字段 [asset_type=stock,statement_type=fina_indicator] |

<a id="fund-nav-daily"></a>

## fund_nav_daily

**基金日净值**：按 security_code + nav_date 保存规范日净值。实际主键不含 announcement_date，同一净值日的公告内容版本不能仅靠本表区分。

定义说明：基金单位净值、累计净值和资产净值。声明粒度：基金、净值日期和公告日期一行。实际字段 **23** 个；逻辑行数 PG **488,682**、CH **488,682**。

PG 主键：`security_code, nav_date`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code, nav_date`。

声明时间字段：`nav_date`；时间规则：`observations`。声明主键：`security_code, nav_date`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | 基金代码 | text | String | 否/否 | 未声明/不适用 | tushare:fund_nav.ts_code × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `nav_date` **PK** | 净值日期 | date | Date32 | 否/否 | 未声明/不适用 | tushare:fund_nav.nav_date × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `announcement_date` | 公告日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:fund_nav.ann_date × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `unit_nav` | 单位净值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:fund_nav.unit_nav × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `accum_nav` | 累计净值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:fund_nav.accum_nav × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `accum_div` | 累计分红 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:fund_nav.accum_div × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `net_asset` | 资产净值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:fund_nav.net_asset × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `total_netasset` | 合计资产净值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:fund_nav.total_netasset × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `adj_nav` | 复权单位净值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元 | tushare:fund_nav.adj_nav × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Date32 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Int64 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="fund-nav-observation"></a>

## fund_nav_observation

**基金净值公告记录**：保留源净值公告记录，按 security_code + record_version 区分内容；与规范日净值表的粒度不同。

定义说明：基金净值公告记录；完整保存查询字段和版本。声明粒度：源记录业务粒度。实际字段 **24** 个；逻辑行数 PG **488,682**、CH **488,682**。

PG 主键：`security_code, record_version`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code, record_version`。

声明时间字段：`nav_date`；时间规则：`observations`。声明主键：`security_code, record_version`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | TS代码 | text | String | 否/否 | 未声明/不适用 | tushare:fund_nav.ts_code × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `ann_date` | 公告日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:fund_nav.ann_date × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `nav_date` | 净值日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:fund_nav.nav_date × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `unit_nav` | 单位净值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fund_nav.unit_nav × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `accum_nav` | 累计净值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fund_nav.accum_nav × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `accum_div` | 累计分红 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fund_nav.accum_div × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `net_asset` | 资产净值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fund_nav.net_asset × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `total_netasset` | 合计资产净值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fund_nav.total_netasset × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `adj_nav` | 复权单位净值 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:fund_nav.adj_nav × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `record_version` **PK** | 记录内容版本；由业务内容形成的记录身份，参与事件/观察表的主键；不代表历史公开先后顺序。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(business_version)：完整规范化业务记录的内容哈希 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="fund-shares-daily"></a>

## fund_shares_daily

**基金份额**：基金份额变动日观测；份额不是成交量，不能用作每日成交股数。

定义说明：基金每日份额规模。声明粒度：基金和变动日期一行。实际字段 **17** 个；逻辑行数 PG **34,341**、CH **34,341**。

PG 主键：`security_code, trade_date`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code, trade_date`。

声明时间字段：`trade_date`；时间规则：`observations`。声明主键：`security_code, trade_date`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | 基金代码 | text | String | 否/否 | 未声明/不适用 | tushare:fund_share.ts_code × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `trade_date` **PK** | 变动日期 | date | Date32 | 否/否 | 未声明/不适用 | tushare:fund_share.trade_date × 1 [asset_type=fund]；源字段 [asset_type=fund] |
| `shares` | 基金份额 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 份 | tushare:fund_share.fd_share × 10000（源单位 万份） [asset_type=fund]；源字段 [asset_type=fund] |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Date32 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Int64 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="industry-scheme-dim"></a>

## industry_scheme_dim

**行业分类体系**：普通分类体系字典。当前普通处理路径使用 SW2021/CITIC；参考域的精确体系代码使用 SW2014/SW2021/CITICS，不能混淆或自动补别名。

定义说明：申万、中信等行业分类体系。声明粒度：每个分类体系一行。实际字段 **17** 个；逻辑行数 PG **2**、CH **2**。

PG 主键：`scheme_code`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`scheme_code`。

声明时间字段：`未单独声明`；时间规则：`snapshot`。声明主键：`scheme_code`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `scheme_code` **PK** | 分类体系代码；按本表合同解释精确体系及年份；普通表与参考域的实际值可能不同，不自动转换别名。 | text | String | 否/否 | 未声明/不适用 | tushare:index_classify.src × 1；计算(industry_scheme)：按查询接口和显式分类版本生成；中信固定CITIC，申万固定SW2021；源字段 |
| `scheme_name` | 体系名称 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(industry_scheme)：按SW2021/CITIC分别生成分类名称和说明 |
| `description` | 说明；本记录/表/字段的补充说明，具体对象由所在表决定。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(industry_scheme)：按SW2021/CITIC分别生成分类名称和说明 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="industry-dim"></a>

## industry_dim

**行业分类**：行业字典及父子关系。与归属关联时同时使用 scheme_code 和 industry_code。level 的实际类型为 text，参考版本中的 level 是整数。

定义说明：行业代码、名称、层级和父子关系。声明粒度：体系和行业代码一行。实际字段 **22** 个；逻辑行数 PG **928**、CH **928**。

PG 主键：`scheme_code, industry_code`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`scheme_code, industry_code`。

声明时间字段：`未单独声明`；时间规则：`snapshot`。声明主键：`scheme_code, industry_code`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `scheme_code` **PK** | 分类体系代码；按本表合同解释精确体系及年份；普通表与参考域的实际值可能不同，不自动转换别名。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(industry_scheme)：固定申万SW2021或中信CITIC版本 |
| `industry_code` **PK** | 行业代码 | text | String | 否/否 | 未声明/不适用 | tushare:index_classify.index_code × 1；tushare:index_member_all.l1_code × 1；内置派生规则；tushare:index_member_all.l2_code × 1；内置派生规则；tushare:index_member_all.l3_code × 1；内置派生规则；tushare:ci_index_member.l1_code × 1；内置派生规则；tushare:ci_index_member.l2_code × 1；内置派生规则；tushare:ci_index_member.l3_code × 1；内置派生规则；计算(industry_membership)：expand_industry_levels；源字段 |
| `industry_name` | 行业名称 | text | String | 否/否 | 未声明/不适用 | tushare:index_classify.industry_name × 1；tushare:index_member_all.l1_name × 1；内置派生规则；tushare:index_member_all.l2_name × 1；内置派生规则；tushare:index_member_all.l3_name × 1；内置派生规则；tushare:ci_index_member.l1_name × 1；内置派生规则；tushare:ci_index_member.l2_name × 1；内置派生规则；tushare:ci_index_member.l3_name × 1；内置派生规则；计算(industry_membership)：expand_industry_levels；源字段 |
| `parent_code` | 父行业代码 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_classify.parent_code × 1；源字段 |
| `level` | 行业层级 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_classify.level × 1；源字段 |
| `classification_code` | 行业代码 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_classify.industry_code × 1；源字段 |
| `is_pub` | 是否发布了指数 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_classify.is_pub × 1；源字段 |
| `src` | 行业分类（SW申万） | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_classify.src × 1；源字段 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="industry-membership-record"></a>

## industry_membership_record

**行业归属原始记录**：保留源行业归属宽记录，包含 l1/l2/l3 字段、纳入剔除日期、Y/N 标记及内容版本。源有效区间不证明历史公开时间。

定义说明：行业归属原始记录；完整保存查询字段和版本。声明粒度：源记录业务粒度。实际字段 **27** 个；逻辑行数 PG **11,435**、CH **11,435**。

PG 主键：`scheme_code, security_code, record_version`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`scheme_code, security_code, record_version`。

声明时间字段：`event_date`；时间规则：`observations`。声明主键：`scheme_code, security_code, record_version`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `l1_code` | 一级行业代码 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_member_all.l1_code × 1；tushare:ci_index_member.l1_code × 1；源字段 |
| `l1_name` | 一级行业名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_member_all.l1_name × 1；tushare:ci_index_member.l1_name × 1；源字段 |
| `l2_code` | 二级行业代码 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_member_all.l2_code × 1；tushare:ci_index_member.l2_code × 1；源字段 |
| `l2_name` | 二级行业名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_member_all.l2_name × 1；tushare:ci_index_member.l2_name × 1；源字段 |
| `l3_code` | 三级行业代码 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_member_all.l3_code × 1；tushare:ci_index_member.l3_code × 1；源字段 |
| `l3_name` | 三级行业名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_member_all.l3_name × 1；tushare:ci_index_member.l3_name × 1；源字段 |
| `security_code` **PK** | 成分股票代码 | text | String | 否/否 | 未声明/不适用 | tushare:index_member_all.ts_code × 1；tushare:ci_index_member.ts_code × 1；源字段 |
| `name` | 成分股票名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_member_all.name × 1；tushare:ci_index_member.name × 1；源字段 |
| `in_date` | 纳入日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:index_member_all.in_date × 1；tushare:ci_index_member.in_date × 1；源字段 |
| `out_date` | 剔除日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:index_member_all.out_date × 1；tushare:ci_index_member.out_date × 1；源字段 |
| `is_new` | 是否最新Y是N否 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:index_member_all.is_new × 1；tushare:ci_index_member.is_new × 1；源字段 |
| `scheme_code` **PK** | 分类体系代码；按本表合同解释精确体系及年份；普通表与参考域的实际值可能不同，不自动转换别名。；普通路径当前为 SW2021/CITIC；不等同参考域版本选择。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(industry_scheme)：固定申万SW2021或中信CITIC版本 |
| `record_version` **PK** | 记录内容版本；由业务内容形成的记录身份，参与事件/观察表的主键；不代表历史公开先后顺序。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(business_version)：完整规范化业务记录的内容哈希 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="security-industry-membership"></a>

## security_industry_membership

**证券行业归属**：从源记录展开的普通行业归属。没有参考域发布凭证；新 PIT 行业查询不能直接将其当可信版本表。

定义说明：证券在申万或中信行业中的历史归属。声明粒度：体系、证券、行业和有效期一行。实际字段 **20** 个；逻辑行数 PG **34,305**、CH **34,305**。

PG 主键：`scheme_code, security_code, industry_code, valid_from`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`scheme_code, security_code, industry_code, valid_from`。

声明时间字段：`valid_from`；时间规则：`observations`。声明主键：`scheme_code, security_code, industry_code, valid_from`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `scheme_code` **PK** | 分类体系代码；按本表合同解释精确体系及年份；普通表与参考域的实际值可能不同，不自动转换别名。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(industry_scheme)：固定申万SW2021或中信CITIC版本 |
| `security_code` **PK** | 证券代码 | text | String | 否/否 | 未声明/不适用 | tushare:index_member_all.ts_code × 1；tushare:ci_index_member.ts_code × 1；源字段 |
| `industry_code` **PK** | 行业代码 | text | String | 否/否 | 未声明/不适用 | tushare:index_member_all.l1_code × 1；内置派生规则；tushare:index_member_all.l2_code × 1；内置派生规则；tushare:index_member_all.l3_code × 1；内置派生规则；tushare:ci_index_member.l1_code × 1；内置派生规则；tushare:ci_index_member.l2_code × 1；内置派生规则；tushare:ci_index_member.l3_code × 1；内置派生规则；计算(industry_membership)：expand_industry_levels |
| `valid_from` **PK** | 纳入日期 | date | Date32 | 否/否 | 未声明/不适用 | tushare:index_member_all.in_date × 1；tushare:ci_index_member.in_date × 1；源字段 |
| `valid_to` | 剔除日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:index_member_all.out_date × 1；tushare:ci_index_member.out_date × 1；源字段 |
| `membership_status` | 归属状态 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(industry_membership)：依据纳入、剔除日期派生成分状态 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Date32 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Int64 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="index-membership"></a>

## index_membership

**指数成分**：指数、成分、trade_date 一行，weight 单位为 %。当前没有 snapshot_complete/snapshot_member_count，行数与权重和不构成完整快照证明。

定义说明：指数成分及历史权重。声明粒度：指数、成分证券和日期一行。实际字段 **18** 个；逻辑行数 PG **2,954,624**、CH **2,954,624**。

PG 主键：`index_code, security_code, trade_date`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`index_code, security_code, trade_date`。

声明时间字段：`trade_date`；时间规则：`observations`。声明主键：`index_code, security_code, trade_date`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `index_code` **PK** | 指数代码 | text | String | 否/否 | 未声明/不适用 | tushare:index_weight.index_code × 1 [asset_type=index]；源字段 [asset_type=index] |
| `security_code` **PK** | 成分证券代码 | text | String | 否/否 | 未声明/不适用 | tushare:index_weight.con_code × 1 [asset_type=index]；源字段 [asset_type=index] |
| `trade_date` **PK** | 交易日期 | date | Date32 | 否/否 | 未声明/不适用 | tushare:index_weight.trade_date × 1 [asset_type=index]；源字段 [asset_type=index] |
| `weight` | 权重 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | % | tushare:index_weight.weight × 1 [asset_type=index]；源字段 [asset_type=index] |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Date32 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Int64 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="trade-calendar"></a>

## trade_calendar

**交易日历**：普通交易所日历，包含开市与休市自然日记录。范围缺日未知；严格 PIT 日历使用参考版本及完整发布范围。

定义说明：各交易所交易日和前一交易日。声明粒度：交易所和日期一行。实际字段 **18** 个；逻辑行数 PG **60**、CH **60**。

PG 主键：`exchange, trade_date`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`exchange, trade_date`。

声明时间字段：`trade_date`；时间规则：`calendar_day`。声明主键：`exchange, trade_date`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `exchange` **PK** | 交易所 | text | String | 否/否 | 未声明/不适用 | tushare:trade_cal.exchange × 1；源字段 |
| `trade_date` **PK** | 交易日期 | date | Date32 | 否/否 | 未声明/不适用 | tushare:trade_cal.cal_date × 1；源字段 |
| `is_open` | 是否交易；是否交易；普通日历的开市标志；行缺失不代表 0。 | bigint | Int64 | 否/否 | 未声明/不适用 | tushare:trade_cal.is_open × 1；源字段 |
| `previous_trade_date` | 上一交易日 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:trade_cal.pretrade_date × 1；源字段 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Date32 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Int64 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="trade-status-daily"></a>

## trade_status_daily

**每日可交易状态**：普通派生可交易状态，需同时查看 status_source/confidence。它不等于新版 Datafeed 从可信停牌范围核验的 1/0/-1/NA 状态。

定义说明：证券在每个交易日的可交易状态及来源可信度。声明粒度：资产和交易日一行。实际字段 **22** 个；逻辑行数 PG **966,219**、CH **966,219**。

PG 主键：`asset_type, security_code, trade_date`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`asset_type, security_code, trade_date`。

声明时间字段：`trade_date`；时间规则：`trading_day`。声明主键：`asset_type, security_code, trade_date`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `asset_type` **PK** | 资产类型；实际业务维度，如 stock/fund/index；证券总表主键还包含该维度。债券为查询预留维度，当前任务未导入。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(asset_scope)：按查询合同的资产范围生成 |
| `security_code` **PK** | 证券代码 | text | String | 否/否 | 未声明/不适用 | tushare:suspend_d.ts_code × 1 [asset_type=stock]；tushare:daily.ts_code × 1 [asset_type=stock]；tushare:fund_daily.ts_code × 1 [asset_type=fund]；tushare:index_daily.ts_code × 1 [asset_type=index]；tushare:sw_daily.ts_code × 1 [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `trade_date` **PK** | 交易日期 | date | Date32 | 否/否 | 未声明/不适用 | tushare:suspend_d.trade_date × 1 [asset_type=stock]；tushare:daily.trade_date × 1 [asset_type=stock]；tushare:fund_daily.trade_date × 1 [asset_type=fund]；tushare:index_daily.trade_date × 1 [asset_type=index]；tushare:sw_daily.trade_date × 1 [asset_type=index]；源字段 [asset_type=stock]；源字段 [asset_type=fund]；源字段 [asset_type=index] |
| `is_tradable` | 是否可交易 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(trade_status)：股票按交易日历/停复牌/上市边界；基金指数按行情存在性派生 |
| `status_code` | 状态代码 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:suspend_d.suspend_type × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `status_name` | 状态名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(trade_status)：股票按交易日历/停复牌/上市边界；基金指数按行情存在性派生 |
| `status_source` | 状态来源 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(trade_status)：股票按交易日历/停复牌/上市边界；基金指数按行情存在性派生 |
| `confidence` | 状态可信度 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(trade_status)：股票按交易日历/停复牌/上市边界；基金指数按行情存在性派生 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Date32 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Int64 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="corporate-action"></a>

## corporate_action

**证券事件**：证券事件摘要。cash_dividend 是税后每股现金分红，share_dividend 仅送股比例，不含资本公积转增。不能代替完整分红方案表。

定义说明：分红、改名和复权因子等证券事件。声明粒度：证券和事件日期一行。实际字段 **22** 个；逻辑行数 PG **148,074**、CH **148,074**。

PG 主键：`security_code, event_type, event_date`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code, event_type, event_date`。

声明时间字段：`event_date`；时间规则：`observations`。声明主键：`security_code, event_type, event_date`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | 证券代码 | text | String | 否/否 | 未声明/不适用 | tushare:namechange.ts_code × 1；tushare:dividend.ts_code × 1；tushare:adj_factor.ts_code × 1 [asset_type=stock]；tushare:fund_adj.ts_code × 1 [asset_type=fund]；源字段；源字段 [asset_type=stock]；源字段 [asset_type=fund] |
| `event_type` **PK** | 事件类型；本表登记的事件类别；参考发布表使用 publish/revoke，证券事件表使用业务事件类别。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(event_summary)：已实施分红以ex_date为生效日；改名以start_date；送股比例stk_bo_rate |
| `effective_date` | 生效日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:namechange.start_date × 1；源字段 |
| `adjustment_factor` | 复权因子 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:adj_factor.adj_factor × 1 [asset_type=stock]；tushare:fund_adj.adj_factor × 1 [asset_type=fund]；源字段 [asset_type=stock]；源字段 [asset_type=fund] |
| `old_name` | 旧名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；未提供：只在完整、无歧义的紧邻名称区间证明后生成；当前不推断 |
| `new_name` | 新名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:namechange.name × 1；源字段 |
| `cash_dividend` | 现金分红 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 元/股 | tushare:dividend.cash_div × 1；源字段 |
| `share_dividend` | 送股比例 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:dividend.stk_bo_rate × 1；源字段 |
| `event_date` **PK** | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Date32 | 否/否 | 未声明/不适用 | tushare:namechange.start_date × 1；tushare:dividend.ex_date × 1；tushare:adj_factor.trade_date × 1 [asset_type=stock]；tushare:fund_adj.trade_date × 1 [asset_type=fund]；源字段；源字段 [asset_type=stock]；源字段 [asset_type=fund]；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Int64 | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="dividend-event"></a>

## dividend_event

**分红方案**：源分红方案完整记录，按内容版本区分；预案、股东大会、实施等进度不能混同生效。

定义说明：分红方案；完整保存查询字段和版本。声明粒度：源记录业务粒度。实际字段 **31** 个；逻辑行数 PG **3**、CH **3**。

PG 主键：`security_code, record_version`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code, record_version`。

声明时间字段：`event_date`；时间规则：`observations`。声明主键：`security_code, record_version`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | TS代码 | text | String | 否/否 | 未声明/不适用 | tushare:dividend.ts_code × 1；源字段 |
| `end_date` | 分红年度 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:dividend.end_date × 1；源字段 |
| `ann_date` | 公告日(预案，决案) | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:dividend.ann_date × 1；源字段 |
| `div_proc` | 实施进度 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:dividend.div_proc × 1；源字段 |
| `stk_div` | 每股送转 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:dividend.stk_div × 1；源字段 |
| `stk_bo_rate` | 每股送股比例 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:dividend.stk_bo_rate × 1；源字段 |
| `stk_co_rate` | 每股转增比例 | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:dividend.stk_co_rate × 1；源字段 |
| `cash_div` | 每股分红（税后） | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:dividend.cash_div × 1；源字段 |
| `cash_div_tax` | 每股分红（税前） | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 未声明/不适用 | tushare:dividend.cash_div_tax × 1；源字段 |
| `record_date` | 股权登记日 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:dividend.record_date × 1；源字段 |
| `ex_date` | 除权除息日 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:dividend.ex_date × 1；源字段 |
| `pay_date` | 派息日 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:dividend.pay_date × 1；源字段 |
| `div_listdate` | 红股上市日 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:dividend.div_listdate × 1；源字段 |
| `imp_ann_date` | 实施公告日 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:dividend.imp_ann_date × 1；源字段 |
| `base_date` | 基准日 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:dividend.base_date × 1；源字段 |
| `base_share` | 基准股本（万） | numeric(38,12) | Nullable(Decimal(38, 12)) | 是/是 | 股 | tushare:dividend.base_share × 10000（源单位 万股）；源字段 |
| `record_version` **PK** | 记录内容版本；由业务内容形成的记录身份，参与事件/观察表的主键；不代表历史公开先后顺序。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(business_version)：完整规范化业务记录的内容哈希 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="suspension-event"></a>

## suspension_event

**停复牌事件**：普通源停复牌事件，保留 suspend_type 和原始时段，不自动提供严格可交易性结论。

定义说明：停复牌事件；完整保存查询字段和版本。声明粒度：源记录业务粒度。实际字段 **19** 个；逻辑行数 PG **196**、CH **196**。

PG 主键：`security_code, record_version`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code, record_version`。

声明时间字段：`trade_date`；时间规则：`observations`。声明主键：`security_code, record_version`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | TS代码 | text | String | 否/否 | 未声明/不适用 | tushare:suspend_d.ts_code × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `trade_date` | 停复牌日期（覆盖从停牌到覆盖期间的连续日期） | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:suspend_d.trade_date × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `suspend_timing` | 日内停牌时间段（日内停牌才有值，否则为空值） | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:suspend_d.suspend_timing × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `suspend_type` | 停复牌类型：S-停牌，R-复牌 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:suspend_d.suspend_type × 1 [asset_type=stock]；源字段 [asset_type=stock] |
| `record_version` **PK** | 记录内容版本；由业务内容形成的记录身份，参与事件/观察表的主键；不代表历史公开先后顺序。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(business_version)：完整规范化业务记录的内容哈希 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="security-name-history"></a>

## security_name_history

**证券名称历史**：源名称变更观察，按内容版本区分。Datafeed 仅使用有可得性证据的有效名称，不以当前名称回填。

定义说明：证券名称历史；完整保存查询字段和版本。声明粒度：源记录业务粒度。实际字段 **21** 个；逻辑行数 PG **5,939**、CH **5,939**。

PG 主键：`security_code, record_version`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`security_code, record_version`。

声明时间字段：`event_date`；时间规则：`observations`。声明主键：`security_code, record_version`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` **PK** | TS代码 | text | String | 否/否 | 未声明/不适用 | tushare:namechange.ts_code × 1；源字段 |
| `name` | 证券名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:namechange.name × 1；源字段 |
| `start_date` | 开始日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:namechange.start_date × 1；源字段 |
| `end_date` | 结束日期；结束日期；当前名称查询采用 start_date <= 观察日期 <= end_date；空结束日仍需对应版本可得证据。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:namechange.end_date × 1；源字段 |
| `ann_date` | 公告日期 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | tushare:namechange.ann_date × 1；源字段 |
| `change_reason` | 变更原因 | text | Nullable(String) | 是/是 | 未声明/不适用 | tushare:namechange.change_reason × 1；源字段 |
| `record_version` **PK** | 记录内容版本；由业务内容形成的记录身份，参与事件/观察表的主键；不代表历史公开先后顺序。 | text | String | 否/否 | 未声明/不适用 | 未登记直接来源；查看生成规则；计算(business_version)：完整规范化业务记录的内容哈希 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 未登记直接来源；查看生成规则；系统生成：由任务、目标库及时间证据生成；未知可得时间保留空值 |

<a id="stock-industry-membership-versions"></a>

## stock_industry_membership_versions

**股票历史行业分类版本**：追加版本的行业历史完整范围。必须先核验发布、撤销、内容哈希与历史可得性，再按年份版本/层级/有效区间选唯一归属。

定义说明：追加版本；仅可信发布范围可供业务读取。股票×分类体系×层级完整历史快照中的一个有效区间。声明粒度：股票×分类体系×层级完整历史快照中的一个有效区间。实际字段 **30** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`revision_id, scope_id, record_id`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`revision_id, scope_id, record_id`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`revision_id, scope_id, record_id`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` | 股票代码 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `scheme_code` | 分类体系代码；按本表合同解释精确体系及年份；普通表与参考域的实际值可能不同，不自动转换别名。；精确值如 SW2014/SW2021/CITICS，年份不能省略后合并。 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `classification_version` | 分类版本；分类版本；必须与 scheme_code 一致，并由版本字典匹配证据支持。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `level` | 行业层级 | bigint | Int64 | 否/否 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `industry_code` | 行业代码 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `industry_name` | 行业名称 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `valid_from` | 生效日 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `valid_to` | 失效日（不含）；失效日（不含）；失效日不含；有效区间为 [valid_from, valid_to)，端点含义须有 boundary_basis 证据。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `boundary_basis` | 区间边界依据；区间边界依据；严格查询认可 source_effective_interval / source_in_date_open_end / verified_left_closed_right_open。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `membership_status` | 归属状态；归属状态；必须证实 mapped；unknown/conflict 等状态不能当作有效归属。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `raw_record` | 源记录 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `revision_id` **PK** | 参考域修订编号；同一目标库一次不可变参考域修订的身份；与发布事件和完整范围凭证一起使用。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `scope_id` **PK** | 完整范围编号；标识公司×体系×层级、交易所×自然日、公司×日期等完整业务范围；不是单条内容记录。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `record_id` **PK** | 范围内记录编号；在 revision_id、scope_id 下定位一条版本记录；零事件范围可以没有任何记录。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `attempt_id` | 执行尝试编号；区分同一任务的执行尝试，保留中断和恢复的来源身份。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `ingestion_seq` | 目标库单调入库序号；目标库内参考修订排序，避免入库时钟倒退；不能代替历史可得时间。 | bigint | Int64 | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；index_member_all, ci_index_member；参考域/管理系统合同 |

<a id="trade-calendar-versions"></a>

## trade_calendar_versions

**交易日历版本**：追加版本的交易所×自然日记录。每个完整日历范围一条记录，缺日不能推断休市；BSE 的参考 SSE 依据需显式保存。

定义说明：追加版本；仅可信发布范围可供业务读取。交易所×自然日期。声明粒度：交易所×自然日期。实际字段 **25** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`revision_id, scope_id, record_id`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`revision_id, scope_id, record_id`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`revision_id, scope_id, record_id`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `exchange` | 交易所 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `trade_date` | 自然日期 | date | Date32 | 否/否 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `is_open` | 是否开市；是否开市；1 开市，0 休市，NULL 未知；仅有可知完整自然日范围时可用于日历 API。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `previous_trade_date` | 上一交易日 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `reference_exchange` | 参考交易所；参考交易所；例如 BSE 显式参考 SSE；不是将查询 exchange 名称自动转换的兼容别名。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `calendar_basis` | 日历依据 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `revision_id` **PK** | 参考域修订编号；同一目标库一次不可变参考域修订的身份；与发布事件和完整范围凭证一起使用。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `scope_id` **PK** | 完整范围编号；标识公司×体系×层级、交易所×自然日、公司×日期等完整业务范围；不是单条内容记录。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `record_id` **PK** | 范围内记录编号；在 revision_id、scope_id 下定位一条版本记录；零事件范围可以没有任何记录。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `attempt_id` | 执行尝试编号；区分同一任务的执行尝试，保留中断和恢复的来源身份。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `ingestion_seq` | 目标库单调入库序号；目标库内参考修订排序，避免入库时钟倒退；不能代替历史可得时间。 | bigint | Int64 | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；trade_cal；参考域/管理系统合同 |

<a id="stock-suspension-intervals-versions"></a>

## stock_suspension_intervals_versions

**股票停牌区间版本**：完整股票日期范围内的一条停复牌事件；时间区间左闭右开。零事件可由 reference_scope 表达，不能因本表无行就判断正常。

定义说明：追加版本；仅可信发布范围可供业务读取。股票×业务日期完整事件集中的一条停复牌事件。声明粒度：股票×业务日期完整事件集中的一条停复牌事件。实际字段 **30** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`revision_id, scope_id, record_id`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`revision_id, scope_id, record_id`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`revision_id, scope_id, record_id`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `security_code` | 股票代码 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `trade_date` | 业务日期 | date | Date32 | 否/否 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `suspend_type` | 停复牌类型；停复牌类型；源 S/R 停复牌类型；不能脱离原始时段解释为正常交易状态。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `raw_timing` | 原始停牌时段 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `start_at` | 停牌开始（含）；停牌开始（含）；包含起点；按上海源时间解析并以 UTC 存储。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `end_at` | 停牌结束（不含）；停牌结束（不含）；不包含终点；与 start_at 构成 [start_at,end_at)。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `is_full_day` | 是否整日 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `time_precision` | 时间精度 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `parse_status` | 时段解析状态；时段解析状态；parsed 仍需发布、完整范围与可得性证据；解析失败保留未知。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `parse_reason` | 解析说明 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `raw_record` | 源记录 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `revision_id` **PK** | 参考域修订编号；同一目标库一次不可变参考域修订的身份；与发布事件和完整范围凭证一起使用。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `scope_id` **PK** | 完整范围编号；标识公司×体系×层级、交易所×自然日、公司×日期等完整业务范围；不是单条内容记录。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `record_id` **PK** | 范围内记录编号；在 revision_id、scope_id 下定位一条版本记录；零事件范围可以没有任何记录。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `attempt_id` | 执行尝试编号；区分同一任务的执行尝试，保留中断和恢复的来源身份。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `ingestion_seq` | 目标库单调入库序号；目标库内参考修订排序，避免入库时钟倒退；不能代替历史可得时间。 | bigint | Int64 | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `event_date` | 业务发生日期；行情通常对应交易日，财报通常对应报告期；具体看本表时间字段。它不表示历史公开时间。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `available_at` | 记录可得时间；写入层允许保存真实或推定时刻；必须结合 availability_basis、质量和内容版本证据判断，非空不等于严格 PIT 可用。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `downloaded_at` | 下载时间；本地实际取得源响应的时刻，不能替代历史公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `availability_basis` | 可得时间依据；说明 available_at 如何得出。当前严格 PIT 仅接受 source_version_publication_time 或 verified_historical_version；推定依据不满足门槛。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `source_id` | 订阅源实例；来源注册中的具体实例标识，不是证券代码。 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `source_endpoint` | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `source_row_no` | 源记录行号；定位归档响应中的源行；不是跨版本修订顺序。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `source_record_hash` | 源记录哈希；用于内容核对和追溯；内容哈希本身不证明发布时间或修订先后。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | String | 否/否 | 未声明/不适用 | 参考域系统生成；参考域/管理系统合同 |
| `data_quality_status` | 数据质量状态；accepted/verified 仅是 PIT 的必要条件之一；mixed_sources、unavailable_missing_factor 等状态不能直接作为严格已知记录。 | text | String | 否/否 | 未声明/不适用 | 参考域固定处理器；suspend_d；参考域/管理系统合同 |

<a id="reference-revision"></a>

## reference_revision

**参考域修订头**：不可变修订头，payload 保留范围清单、输入和来源合同等；不是发布状态表。

定义说明：不可变业务参考版本及发布凭证。声明粒度：版本或发布事件。实际字段 **8** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`revision_id`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`revision_id`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`revision_id`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `revision_id` **PK** | 参考域修订编号；同一目标库一次不可变参考域修订的身份；与发布事件和完整范围凭证一起使用。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `import_task_id` | 导入任务编号；与源任务、归档、批次及审计关联的任务 ID；任务完成不等于参考域已发布。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `attempt_id` | 执行尝试编号；区分同一任务的执行尝试，保留中断和恢复的来源身份。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `module` | 参考域模块；industry/calendar/suspension；每份修订只处理一个独立模块。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `ingestion_seq` | 目标库单调入库序号；目标库内参考修订排序，避免入库时钟倒退；不能代替历史可得时间。 | bigint | Int64 | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `ingested_at` | 实际入库时间；目标库写入该内容版本的时刻；PG、CH 分别记录，不能替代公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `manifest_hash` | 修订清单哈希；覆盖修订头与完整范围清单；发布事件必须指向同一份已核验清单。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `payload` | 完整 JSON 载荷；以 text/String 保存的规范化 JSON。包含物理投影外的 selector、evidence、scope_ids 等，必须与投影字段及哈希相互核验。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

<a id="reference-scope"></a>

## reference_scope

**完整范围凭证**：修订下完整范围凭证。selector、evidence、dependencies 等保存在 payload；row_count=0 可以是有证明的完整零事件范围。

定义说明：不可变业务参考版本及发布凭证。声明粒度：版本或发布事件。实际字段 **9** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`revision_id, scope_id`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`revision_id, scope_id`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`revision_id, scope_id`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `revision_id` **PK** | 参考域修订编号；同一目标库一次不可变参考域修订的身份；与发布事件和完整范围凭证一起使用。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `scope_id` **PK** | 完整范围编号；标识公司×体系×层级、交易所×自然日、公司×日期等完整业务范围；不是单条内容记录。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `module` | 参考域模块；industry/calendar/suspension；每份修订只处理一个独立模块。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `selector_hash` | 范围选择条件哈希；对 payload.selector 的内容校验，防止完整范围被换成另一组条件。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `row_count` | 记录总数；修订、范围或目标库核验的预期/实际记录总数，具体含义由所在表决定。 | bigint | Int64 | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `content_hash` | 范围内容哈希；对范围内规范化记录排序后的内容校验；必须按合同序列化，不能自行拼接。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `key_hash` | 记录键哈希；对键集合的校验值，参考域用于证明范围记录身份。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `complete` | 范围完整标志；0/1；表示声明的范围是否完整，不表示历史可得时间已证明，也不表示已发布。；完整范围标志；需同时核对 payload.evidence.availability 和记录级可得性。 | bigint | Int64 | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `payload` | 完整 JSON 载荷；以 text/String 保存的规范化 JSON。包含物理投影外的 selector、evidence、scope_ids 等，必须与投影字段及哈希相互核验。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

<a id="reference-publication-event"></a>

## reference_publication_event

**可信发布事件**：追加 publish/revoke 事件决定参考修订可见性；完成任务、已入库、已核验与已发布分别判断。

定义说明：不可变业务参考版本及发布凭证。声明粒度：版本或发布事件。实际字段 **7** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`event_id`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`event_id`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`event_id`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `event_id` **PK** | 发布事件编号；不可变 publish/revoke 事件的身份。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `event_seq` | 发布事件序号；目标库发布事件顺序；与记录业务日期和实际公开时间分开。 | bigint | Int64 | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `revision_id` | 参考域修订编号；同一目标库一次不可变参考域修订的身份；与发布事件和完整范围凭证一起使用。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `event_type` | 事件类型；本表登记的事件类别；参考发布表使用 publish/revoke，证券事件表使用业务事件类别。；publish 发布；revoke 撤销。撤销不删除版本物理行。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `rollback_task_id` | 回退任务编号；发起撤销或回退的任务；参考域通过追加 revoke 撤销可见性，不删除版本记录。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `created_at` | 创建时间；任务、审计或发布事件的本地创建时刻；不是数据公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `payload` | 完整 JSON 载荷；以 text/String 保存的规范化 JSON。包含物理投影外的 selector、evidence、scope_ids 等，必须与投影字段及哈希相互核验。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

<a id="meta-table"></a>

## meta_table

**数据表元信息**：库内表元信息镜像。当前为空，字典说明来自本机表定义及实际数据库目录。

定义说明：数据表元信息登记、查询与历史追溯。声明粒度：登记键一行。实际字段 **12** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`table_name`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`table_name`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`table_name`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `table_name` **PK** | 物理表名；目标数据库中的实际表名。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `table_zh_name` | 表中文名；供阅读的表标签，不是逻辑数据集映射。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `description` | 说明；本记录/表/字段的补充说明，具体对象由所在表决定。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `grain` | 记录粒度；一行代表的业务对象组合；仍应核对物理主键。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `domain` | 业务域；表定义的归类标签，不是 datafeed registry。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `updated_at` | 更新时间；管理元信息/状态行最后更新时刻。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `schema_version` | 结构版本；处理时采用的表结构/合同版本，不是业务生效年份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `business_keys` | 声明业务键；序列化保存的业务键定义；不一定等于版本表物理键。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `time_field` | 声明时间字段；表查询/任务范围使用的日期字段；不能一概理解为公告日。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `temporal_rule` | 时间规则；表定义中的时间处理策略。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `sources` | 声明来源集合；表元信息中登记的来源说明。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

<a id="meta-column"></a>

## meta_column

**字段元信息**：库内字段元信息镜像。当前为空，单位为 YAML 声明，未由本表读回核验。

定义说明：字段元信息登记、查询与历史追溯。声明粒度：登记键一行。实际字段 **14** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`table_name, column_name`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`table_name, column_name`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`table_name, column_name`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `table_name` **PK** | 物理表名；目标数据库中的实际表名。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `column_name` **PK** | 物理字段名；目标表中的实际字段名。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `column_zh_name` | 字段中文名；供阅读的字段标签，不是查询字段别名。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `description` | 说明；本记录/表/字段的补充说明，具体对象由所在表决定。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `unit` | 字段单位；目标字段的声明单位；空字符串不能自动解释为元、百分比或无量纲。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `data_type` | 声明数据类型；逻辑类型说明；实际原生类型以数据库目录为准。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `ordinal` | 字段顺序；字段在表定义中的顺序号。 | integer | Int64 | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `updated_at` | 更新时间；管理元信息/状态行最后更新时刻。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `date_role` | 时间角色；区分业务日期、公告日期、可得时间及本地审计时间的元数据说明。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `precision` | 数值总精度；Decimal/Numeric 的有效数字总位数。 | integer | Nullable(Int64) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `scale` | 小数位数；Decimal/Numeric 的小数部分位数。 | integer | Nullable(Int64) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `nullable` | 声明是否可空；元数据中的0/1标志；本字典另外列出两库实际可空性。 | integer | Nullable(Int64) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `system_generated` | 系统生成标志；元数据中的0/1标志，说明是否由管理器生成。 | integer | Nullable(Int64) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

<a id="meta-source-mapping"></a>

## meta_source_mapping

**来源字段映射**：库内来源字段映射镜像。当前为空，实际映射由当前本机 YAML 读取。

定义说明：来源字段映射登记、查询与历史追溯。声明粒度：登记键一行。实际字段 **14** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`source_name, source_endpoint, source_field, canonical_table, canonical_column`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`source_name, source_endpoint, source_field, canonical_table, canonical_column`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`source_name, source_endpoint, source_field, canonical_table, canonical_column`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `source_name` **PK** | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `source_endpoint` **PK** | 来源接口；产生此记录的源接口；合并/派生行可能记录多个接口，不能据此恢复逐字段历史版本。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `source_field` **PK** | 来源字段；源接口原始响应的字段名称。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `canonical_table` **PK** | 目标规范表；源字段映射指向的物理业务表。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `canonical_column` **PK** | 目标规范字段；源字段映射指向的物理字段。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `source_type` | 来源类型声明；源字段的数据类型。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `target_type` | 目标类型声明；目标字段的数据类型。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `source_unit` | 来源单位；源响应使用的单位；可能不同于统一表单位。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `target_unit` | 目标单位；完成转换后存储的单位。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `transform_expression` | 转换规则说明；声明的单位/字段转换说明；不应作为任意代码执行。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `date_role` | 时间角色；区分业务日期、公告日期、可得时间及本地审计时间的元数据说明。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `required` | 必需字段标志；源字段合同中的0/1要求。 | integer | Int64 | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `description` | 说明；本记录/表/字段的补充说明，具体对象由所在表决定。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

<a id="meta-source-capability"></a>

## meta_source_capability

**数据源能力**：来源能力、权限及探测结果。空表不表示来源完全可用或完全不可用。

定义说明：数据源能力登记、查询与历史追溯。声明粒度：登记键一行。实际字段 **9** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`source_name, endpoint`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`source_name, endpoint`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`source_name, endpoint`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `source_name` **PK** | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `endpoint` **PK** | 来源接口名；登记或归档所对应的源接口。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `capability` | 来源能力；所探测的接口/字段/权限能力。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `status` | 状态；任务、目标、批次或能力探测状态；不可等同参考域可信发布。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `permission_note` | 权限说明；接口权限或点数门槛的说明。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `last_probe_at` | 最近探测时间；接口能力实际探测时刻。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `sample_rows` | 探测样本行数；能力探测实际获得的样本规模。 | integer | Nullable(Int64) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `response_fields` | 响应字段清单；归档或能力探测记录的响应字段集合。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `error` | 错误说明；执行失败或探测错误的说明，不能作为业务数据使用。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

<a id="import-task"></a>

## import_task

**导入任务**：导入任务镜像。管理器任务真值还保存在仓库外 SQLite，本表为空不表示没有完成任务。

定义说明：导入任务登记、查询与历史追溯。声明粒度：登记键一行。实际字段 **10** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`task_id`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`task_id`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`task_id`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `task_id` **PK** | 任务编号；本表登记的管理任务身份。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `source_name` | 来源类型；例如 tushare；区别于具体订阅源实例 source_id。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `targets` | 目标库集合；任务选择的数据库目标，序列化文本。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `status` | 状态；任务、目标、批次或能力探测状态；不可等同参考域可信发布。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `request` | 请求 JSON；任务冻结的请求参数；PG 为 jsonb，CH 为 JSON 字符串。 | jsonb | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `mapping_version` | 映射版本；本次字段映射/处理合同的版本标识。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `created_at` | 创建时间；任务、审计或发布事件的本地创建时刻；不是数据公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `started_at` | 开始时间；任务或目标库执行开始时刻。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `finished_at` | 结束时间；任务、目标库导入或回退的结束时刻。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `error` | 错误说明；执行失败或探测错误的说明，不能作为业务数据使用。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

<a id="import-source-archive"></a>

## import_source_archive

**源数据归档**：源响应归档索引，保存引用与哈希；原始文件在仓库外归档目录。

定义说明：源数据归档登记、查询与历史追溯。声明粒度：登记键一行。实际字段 **9** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`task_id, dataset, request_hash`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`task_id, dataset, request_hash`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`task_id, dataset, request_hash`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `task_id` **PK** | 任务编号；本表登记的管理任务身份。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `dataset` **PK** | 源数据集；下载请求所对应的来源数据集名称。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `endpoint` | 来源接口名；登记或归档所对应的源接口。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `request_hash` **PK** | 请求哈希；归档请求的参数身份，用于追溯和去重。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `archive_path` | 归档路径；原始源响应归档的引用，不是本表存放完整响应。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `response_sha256` | 响应内容校验值；原始响应归档的 SHA256。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `response_fields` | 响应字段清单；归档或能力探测记录的响应字段集合。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `fetched_at` | 抓取时间；原始归档响应的实际抓取时刻。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `rows` | 记录行数；覆盖或归档记录的行数；不能据行数推断历史可得性和范围完整。 | integer | Int64 | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

<a id="import-target-status"></a>

## import_target_status

**目标库导入状态**：按任务和目标库登记的执行镜像；不能充当参考域 publish 事件。

定义说明：目标库导入状态登记、查询与历史追溯。声明粒度：登记键一行。实际字段 **10** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`task_id, target`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`task_id, target`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`task_id, target`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `task_id` **PK** | 任务编号；本表登记的管理任务身份。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `target` **PK** | 目标库；postgres 或 clickhouse；每库分别提交、核验和记录状态。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `status` | 状态；任务、目标、批次或能力探测状态；不可等同参考域可信发布。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `row_count` | 记录总数；修订、范围或目标库核验的预期/实际记录总数，具体含义由所在表决定。 | bigint | Nullable(Int64) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `table_count` | 表数量；目标库导入涉及的表数。 | integer | Nullable(Int64) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `key_hash` | 记录键哈希；对键集合的校验值，参考域用于证明范围记录身份。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `field_hash` | 字段内容哈希；对目标库读回字段内容的核验值。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `started_at` | 开始时间；任务或目标库执行开始时刻。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `finished_at` | 结束时间；任务、目标库导入或回退的结束时刻。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `error` | 错误说明；执行失败或探测错误的说明，不能作为业务数据使用。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

<a id="import-batch-receipt"></a>

## import_batch_receipt

**批次镜像与回执**：批次镜像与目标回执。两库记录数不同，应按 task_id/batch_id/target 核对，不把它当业务数据行数差异。

定义说明：批次镜像与回执登记、查询与历史追溯。声明粒度：登记键一行。实际字段 **9** 个；逻辑行数 PG **12,282**、CH **8,394**。

PG 主键：`task_id, batch_id, target`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`task_id, batch_id, target`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`task_id, batch_id, target`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `task_id` **PK** | 任务编号；本表登记的管理任务身份。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `batch_id` **PK** | 导入批次编号；任务内批次身份，与 task_id、target 组成回执键。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `target` **PK** | 目标库；postgres 或 clickhouse；每库分别提交、核验和记录状态。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `table_name` | 物理表名；目标数据库中的实际表名。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `status` | 状态；任务、目标、批次或能力探测状态；不可等同参考域可信发布。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `manifest` | 批次清单 JSON；批次冻结的写入/审计范围及相关证据；PG 为 jsonb，CH 为字符串。 | jsonb | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `result` | 执行结果 JSON；批次处理或镜像回执结果；不等同参考域发布凭证。 | jsonb | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `error` | 错误说明；执行失败或探测错误的说明，不能作为业务数据使用。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `updated_at` | 更新时间；管理元信息/状态行最后更新时刻。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

<a id="import-row-audit"></a>

## import_row_audit

**行级变更审计**：普通行级 before/after 变更审计；不同于参考域追加版本及 revoke。

定义说明：行级变更审计登记、查询与历史追溯。声明粒度：登记键一行。实际字段 **11** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`id`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`id`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`id`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `id` **PK** | 审计行编号；PG 为 identity bigint；CH 为 Int64，不能据此假设 CH 会自动自增。 | bigint | Int64 | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `task_id` | 任务编号；本表登记的管理任务身份。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `target` | 目标库；postgres 或 clickhouse；每库分别提交、核验和记录状态。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `table_name` | 物理表名；目标数据库中的实际表名。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `row_key` | 业务行键；被审计业务行的序列化主键。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `operation` | 变更操作；被审计的插入、更新、删除等操作种类。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `before_hash` | 变更前哈希；用于冲突检测和回退核对的旧内容哈希。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `after_hash` | 变更后哈希；用于提交核验和回退前置检查的新内容哈希。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `before_value` | 变更前内容；变更前的 JSON 快照；PG 为 jsonb，CH 为 JSON 字符串。 | jsonb | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `after_value` | 变更后内容；变更后的 JSON 快照；PG 为 jsonb，CH 为 JSON 字符串。 | jsonb | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `created_at` | 创建时间；任务、审计或发布事件的本地创建时刻；不是数据公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

<a id="import-rollback"></a>

## import_rollback

**任务回退**：普通任务回退结果，含恢复、删除及冲突计数；不是业务事件记录。

定义说明：任务回退登记、查询与历史追溯。声明粒度：登记键一行。实际字段 **10** 个；逻辑行数 PG **0**、CH **0**。

PG 主键：`rollback_task_id, target`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`rollback_task_id, target`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`rollback_task_id, target`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `rollback_task_id` **PK** | 回退任务编号；发起撤销或回退的任务；参考域通过追加 revoke 撤销可见性，不删除版本记录。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `source_task_id` | 被回退任务编号；要撤回写入结果的原任务身份。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `target` **PK** | 目标库；postgres 或 clickhouse；每库分别提交、核验和记录状态。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `status` | 状态；任务、目标、批次或能力探测状态；不可等同参考域可信发布。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `restored` | 恢复行数；回退时恢复旧内容的记录数。 | bigint | Int64 | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `deleted` | 删除行数；回退时撤除原新增内容的记录数。 | bigint | Int64 | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `conflicts` | 回退冲突数；当前内容不再符合回退前置条件的记录数。 | bigint | Int64 | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `created_at` | 创建时间；任务、审计或发布事件的本地创建时刻；不是数据公开时间。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `finished_at` | 结束时间；任务、目标库导入或回退的结束时刻。 | timestamp with time zone | Nullable(DateTime64(6, 'UTC')) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `error` | 错误说明；执行失败或探测错误的说明，不能作为业务数据使用。 | text | Nullable(String) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

<a id="dataset-coverage"></a>

## dataset_coverage

**数据覆盖范围**：按表、资产、证券记录覆盖首末业务日期及行数；不证明逐日连续、字段齐备、可信发布或 PIT 可用。

定义说明：数据覆盖范围登记、查询与历史追溯。声明粒度：登记键一行。实际字段 **7** 个；逻辑行数 PG **349,792**、CH **349,792**。

PG 主键：`table_name, asset_type, security_code`。CH 引擎：`ReplacingMergeTree`；ORDER BY：`table_name, asset_type, security_code`。

声明时间字段：`未单独声明`；时间规则：`未单独声明`。声明主键：`table_name, asset_type, security_code`。

| 字段 | 中文含义及说明 | PG 原生类型 | CH 原生类型 | 可空 PG/CH | 声明单位 | 来源、换算及生成规则 |
| --- | --- | --- | --- | --- | --- | --- |
| `table_name` **PK** | 物理表名；目标数据库中的实际表名。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `asset_type` **PK** | 资产类型；实际业务维度，如 stock/fund/index；证券总表主键还包含该维度。债券为查询预留维度，当前任务未导入。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `security_code` **PK** | 证券代码；本表统计对应的完整证券代码，需结合 asset_type 和 table_name 解释。 | text | String | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `first_event_date` | 最早业务日期；覆盖统计中已记录的最早 event_date，不证明中间逐日连续。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `last_event_date` | 最晚业务日期；覆盖统计中已记录的最晚 event_date，不证明历史可得性。 | date | Nullable(Date32) | 是/是 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `rows` | 记录行数；覆盖或归档记录的行数；不能据行数推断历史可得性和范围完整。 | bigint | Int64 | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |
| `updated_at` | 更新时间；管理元信息/状态行最后更新时刻。 | timestamp with time zone | DateTime64(6, 'UTC') | 否/否 | 未声明/不适用 | 管理器内部生成/镜像；参考域/管理系统合同 |

## 核对与范围

两库表名及字段名集合一致。原生类型保留各库表达：例如 PG numeric(38,12) 对应 CH Nullable(Decimal(38,12))；整数的原生位宽可能不同。管理 JSON 字段在 PG 使用 jsonb，在 CH 使用字符串。详细差异见结构快照和讲解。

当前表定义与实际数据库的字段名称集合核对：无缺列或额外列。

实际行数为独立只读查询快照，PG 使用 count(*)，CH 使用 FINAL 后 count(*)。这些数值不是全局同时刻事务快照，不证明覆盖完整或逐行内容一致。

生成依据：当前本机 tables/*.yaml、mappings/*.yaml；仓库 canonical.py/catalog.json、storage.py、reference_contracts.py、reference_storage.py；两库系统目录。没有导出凭据或业务行内容。
