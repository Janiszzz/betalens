# 生产表小样本验收结果（2026-10-25）

## 结果与范围

已执行用户明确批准的结构适配，并完成后续要求的四个 VIP 财报接口测试。累计通过 DB Manager 验证 24 张登记业务表、48 个案例，31 个登记接口均取得非空真实响应并完成审计、确认入库及回读。原始两轮报告均保留。

两库累计各核验 488 条标准化样本记录：新增 441 行、更新 31 行、跳过 16 行。合并后的独立只读复核确认 48 个案例、48 份目标表结构、最终业务键与字段哈希、净行数变化和 24 张表的样本跨库业务内容一致。10570 条字段×任务×目标库证据无缺失、重复或 mismatch。

19 张表发生实际写入；security_dim、security_code_history、security_industry_membership、industry_scheme_dim、trade_calendar 五张表的样本全部保留已有值。跳过和保留证明不等同于新来源值已写入。

每轮写入均复用该轮已经下载、审计的真实 HTTP 归档，创建关联子任务并按当前版本重新审计。原始请求、分页终结证据、父任务 HTTP 日志和归档继续保留。完整性只针对固定样本请求，不代表全市场或全部历史已补齐；历史补全和自动更新保持停止。

## VIP 增补与当前 CH 写入方式

按用户后续要求，不配置日常专用账号，继续使用已有 CH 管理员 `betalens`。仅将仓库外私有环境文件的 `CLICKHOUSE_WRITER_MODE` 改为 `local_admin`，账号、密码、Token 及其他配置不变；文件权限保持 0600。没有新增生产账号或执行永久账号分权。服务已重启，[配置页](http://localhost:8010/settings) 的实际写入能力为 `available=true`、`controlled_local_admin`。

管理员模式仍复用目标写锁、请求 journal、不确定写入核验以及提交和回退协议。该锁协调同一本机用户的管理器进程，不能阻止其他客户端直接使用管理员凭据写入；它不声称数据库侧全局独占。

VIP 样本为 2024 中报，income/fina_indicator 使用 000333.SZ，balancesheet/cashflow 使用 000001.SZ。每次显式请求全部登记字段，设置 limit=1，并持续请求 offset 直到空页。共保留 9 个响应归档，含 4 个终结空页；未出现未知字段、缺少登记输出字段、证券越界、报告期越界或重复页。

| VIP 接口 | 返回字段数 | 标准化行数 | PG 新增 | CH 新增 | 入库任务编号 |
|---|---:|---:|---:|---:|---|
| income_vip | 94 | 1 | 1 | 1 | `7dd54c2571894287ba4843eb3d243669` |
| balancesheet_vip | 158 | 1 | 1 | 1 | `b5e0c4c333dc4fc8afd5af842a3937a4` |
| cashflow_vip | 97 | 2 | 2 | 2 | `7a4ad06e478443a680edec8f65f0e8b6` |
| fina_indicator_vip | 167 | 1 | 1 | 1 | `9c193ad301684ce3956b86d895e409a6` |

所有 VIP 任务均完成两库字段哈希和键集合回读，无冲突、拒收或结构变更。cashflow 的两条源版本分别保留，没有合并成一条。

此次核查还修正财报日期过滤：income、balancesheet、cashflow 的 start/end 对应公告窗口；fina_indicator 对应报告期窗口；四者的 period 均精确校验报告期 end_date。公告窗口不再错误比较报告期，缺少验证日期或无效日期明确阻断。依据：[利润表](https://tushare.pro/document/2?doc_id=33)、[资产负债表](https://tushare.pro/document/2?doc_id=36)、[现金流量表](https://tushare.pro/document/2?doc_id=44)、[财务指标](https://tushare.pro/document/2?doc_id=79)。

VIP 的 1000 行仍只是客户端保守保护值，官方未在这些页面给出该分页保证；本轮实际证明的是所选单证券、报告期和 limit=1 的分页行为，不能外推为全部参数组合已验证。

## 已执行的结构适配

- PostgreSQL：22 张旧业务表的整数列原表扩容为 bigint；创建 etf_profile、stock_quote_supplement_daily；fundamental_periodic 增加独立的研发费用、研发投入列。
- ClickHouse：创建同名两张表，给 fundamental_periodic 增加同样两列。
- 共 24 个表级结构任务、27 个目标库操作完成；没有删列、改主键、表重建或旧业务行删除。每个旧表适配前后检查行数不变；整数扩容的 OID、权限及数值保留另有隔离库回归。
- 结构预览：`/home/janiszzz/betalens-infra/production-smoke-20261025-preview/ddl-preview.json`。最初自动审批因未单独确认 DDL 而拒绝执行；用户明确批准后才执行本节操作。
- 首轮生产测试的临时 CH 写账号仅限 betalens 和实际本机客户端地址，最终已删除；首轮没有改变日常配置。后续按用户要求仅启用管理员写入模式，仍未执行永久账号分权方案。

## 首轮逐表结果

PG 与 CH 的下列首轮 44 案例统计完全相同，VIP 的 4 案例另见上表；“样本行数”为各请求标准化结果之和。“冲突”与新增/修改/跳过不互斥。任务列为网页显示的 12 位前缀，完整编号和父任务见私有报告。

| 本地表 | 案例 | 样本行数 | 新增 | 修改 | 跳过 | 冲突 | 入库任务前缀 |
|---|---:|---:|---:|---:|---:|---:|---|
| corporate_action | 3 | 24 | 24 | 0 | 0 | 0 | `5cb1e5fdba9a`、`b3d633eeab7e`、`ccbecee179c8` |
| dividend_event | 1 | 73 | 73 | 0 | 0 | 0 | `b95f236b6c05` |
| etf_profile | 1 | 1 | 1 | 0 | 0 | 0 | `7b243d698cd7` |
| fund_nav_daily | 1 | 1 | 1 | 0 | 0 | 0 | `e3e42f3500e0` |
| fund_nav_observation | 1 | 1 | 1 | 0 | 0 | 0 | `d86ea7c335a8` |
| fund_profile | 1 | 1 | 1 | 0 | 0 | 0 | `cd171b7302de` |
| fund_shares_daily | 1 | 1 | 1 | 0 | 0 | 0 | `a8c3ea91cbef` |
| fundamental_periodic | 4 | 5 | 5 | 0 | 0 | 0 | `60703d4f952b`、`73bc56a10993`、`0ddfe2be401f`、`c7a4ae3180df` |
| index_membership | 1 | 300 | 300 | 0 | 0 | 0 | `39f2fa012382` |
| index_profile | 1 | 1 | 1 | 0 | 0 | 0 | `4e539db18433` |
| industry_dim | 2 | 34 | 0 | 31 | 3 | 0 | `55470f8ee136`、`64b3d8e7b0b9` |
| industry_membership_record | 2 | 2 | 2 | 0 | 0 | 0 | `c91c86292e53`、`01f785ba823a` |
| industry_scheme_dim | 2 | 2 | 0 | 0 | 2 | 2 | `e72498247d02`、`46bc44642984` |
| market_daily_post_adjusted | 4 | 4 | 4 | 0 | 0 | 0 | `f7c0f39d1cd0`、`ed33cb8ef1e9`、`d05625e1a1ac`、`b77a33f80005` |
| market_daily_raw | 4 | 4 | 4 | 0 | 0 | 0 | `eb7024d0309f`、`b16fba71af46`、`c27a81e2805d`、`cb1e8007aed1` |
| security_code_history | 1 | 1 | 0 | 0 | 1 | 0 | `b81653123622` |
| security_dim | 3 | 3 | 0 | 0 | 3 | 0 | `4d9497c2434b`、`0e36851a308f`、`5cc1b62ad32c` |
| security_industry_membership | 2 | 6 | 0 | 0 | 6 | 0 | `1d766ea35f94`、`5dd2ed917d8a` |
| security_name_history | 1 | 4 | 4 | 0 | 0 | 0 | `b5cb8f193b29` |
| stock_profile | 1 | 1 | 1 | 0 | 0 | 0 | `2e3de9d231ec` |
| stock_quote_supplement_daily | 1 | 1 | 1 | 0 | 0 | 0 | `d659cbc178eb` |
| suspension_event | 1 | 8 | 8 | 0 | 0 | 0 | `abf2a3523138` |
| trade_calendar | 1 | 1 | 0 | 0 | 1 | 0 | `f47a0a760def` |
| trade_status_daily | 4 | 4 | 4 | 0 | 0 | 0 | `fc80866acd0a`、`4187f644af54`、`6b47b9e98fe4`、`8ba6e235da59` |

行情样本日期为 2026-10-23；指数成分为上一完整月 2026-09，得到 300 行；财报为 2024 年报，其中研发字段使用非银行证券 000333.SZ。事件和区间表按单证券历史及业务日期语义过滤。

两处冲突仅涉及申万、中信行业体系的 description：已有“历史行业归属，以源端纳入和剔除日期为准”，新生成说明分别为“申万行业分类体系”“中信行业分类体系”。补空值模式保留原说明，并记录冲突。没有价格、数量或证券键冲突。

## 发现并修复的问题

- 股票档案自定义上市状态曾丢失证券过滤，真实全市场响应被审计拦截；已修复并按单证券重新提数通过。
- 补全中信分类码和财务指标报告类型的生成声明，保留必需业务键校验。
- 行业历史成员按有效区间与请求窗口相交筛选，避免丢失很早纳入但当前有效的成员。
- 名称接口输入日期为公告日期，输出 start/end 为生效区间；默认单证券历史提数后本地过滤，避免混淆日期角色。依据：[Tushare 名称变更](https://tushare.pro/document/2?doc_id=100)。
- 指数成分以 index_code 检查请求证券范围，不能把 con_code 当成请求指数；空月份保留无样本状态，改用上一完整月完成实测。依据：[Tushare 指数成分和权重](https://tushare.pro/document/2?doc_id=96)。
- 请求固定 endpoints，归档重审保留原参数、父任务和旧审计；不会修改用户 YAML 来临时选择接口。
- 字段审计区分本次写入与其他已登记但本任务未选择的类型化落点。
- 最后复核发现：没有 asset_type 物理列的档案、财报表曾误判字段“不适用”。现优先使用行上的实际维度，缺少资产列时使用原任务固定资产集合；无法判定时记 unknown_scope，不伪造验证。财报按报表类型判定列适用性，未选来源不参与执行。
- 新审计版本为 `column-verification-v2-scope-context`，未改变生产规则或 YAML 映射版本。全部 44 个任务的旧证据保留在不可变修复报告与 SQLite field_verification_revision；确认目标行仍匹配原提交清单后，重算最新字段证据。修复未写任何业务数据。
- 研发费用与研发投入均读到 16232771000 元，各自在对应 income/fina_indicator 行取得 PG/CH 新值写入证明，另一报表的同类列标记不适用。
- 验收工具现在要求每列×目标库证明完整；所有计划案例均通过才计表覆盖通过，计划失败和未通过案例返回非零退出码。

## 空值与未覆盖范围

- pcf（市现率）没有已登记来源；ETF 申赎 PCF 不等于市现率。旧混义 rd_exp 已弃用；BSE 上市日不能填作代码切换生效日；没有证据时不猜测 old_name。
- 原始行情的复权辅助列、指数的股票/基金复权因子及均价、其他报表类型的专属字段可为不适用。源端实际 NULL 保留，并与未选择、不可用、保留旧值分别报告。
- 基金、指数交易状态仍是行情存在性派生状态，不是权威停复牌状态。bak_daily 未明确单位的字段保留源值，不投影为统一行情字段。
- 四个 VIP 接口已完成上述小样本实测；其全市场和其他报告期不计为已验证。
- 本轮不批量修复历史错误值，也不恢复历史全量补全；新样本通过不能推断所有旧行已修正。

## 网页与证据

打开 [任务工作台](http://localhost:8010/tasks)，按上表任务前缀选择任务，查看“审计缓存”“写入结果”“逐列写入核验”和目标库回执。源 HTTP 日志在关联的父提数任务。当前没有任务编号直达 URL；`?table=` 可带入本地表选择，但不筛选任务列表。

私有报告目录：`/home/janiszzz/betalens-infra/production-smoke-20261025-executed/`。

- `report.json`：原始执行报告、结构操作、提交回执和逐表回读。其内字段证据数组是修复前快照，最新字段证据以下两份报告及网页为准。
- `field-verification-repair.json`：旧/新字段证据、原提交哈希复核、修复编号 `736e3e386ad24531aa55bd4dbd00e1f7`。
- `independent-readback.json`：独立只读复核；44 个案例、48 个目标表和 24 份跨库样本内容均 verified。
- `operational-checks.json`：服务、任务停止状态、账号清理与配置脱敏核查。
- 原提数材料继续保存在 `production-smoke-20261025-preview/` 和 `mapping-audit-20261024/`，未覆盖。

最新累计核验材料位于 `/home/janiszzz/betalens-infra/production-vip-20261025/`：

- `download-audit.json`：四个 VIP 的真实提数、分页、字段合同和审计。
- `committed/report.json`：VIP 两库生产入库和回读结果，使用现有管理员，没有创建临时账号。
- `combined-report.json`、`combined-independent-readback.json`：原 44 案例与 VIP 4 案例合并核验，31 个接口均有非空响应证据，所有案例 verified。
- `writer-configuration.json`、`operational-checks.json`：管理员模式配置变更、凭据未改变、现役网页能力和更新停止状态。

运行核查另外识别到 10 月 23–24 日创建的旧浏览器测试账号和两座旧 CH 测试库，它们没有 betalens 业务库写权限，与本轮生产临时账号及今日回归无关；仅记录，未擅自清理。

复现入口：`scripts/production_smoke.py`、`scripts/production_verify_readback.py`；本次证据修复脚本为 `scripts/repair_field_verification.py`。执行脚本默认不写业务库，必须显式 --execute；独立核验脚本的目标连接强制只读。

## 代码验证

此前完整回归 285 项、最后的请求/结构/归档针对性回归 42 项通过；本次审计范围修复及独立核验相关回归 53 项通过。真实归档的隔离库验收 13 组通过，涵盖普通股、退市股、北交所股票、ETF、指数、行业、财报和事件，并验证回退；这与本次保留样本的生产导入分开计数。已有浏览器验收 8 项通过，本次服务已重启加载审计修复。

VIP 增补相关验证：合同、分页与财报日期专项 67 项通过；相关流水线回归 43 项通过；管理员模式及发布回归 33 项通过，另有隔离 CH 管理员提交、锁、回读和回退 1 项通过，配置页浏览器验收 1 项通过，前端构建成功。以上是不同范围的验证记录，不按简单相加声称唯一测试数。

测试有一条现有 Starlette/httpx 弃用提示，不影响通过。历史补全和定时更新仍停止；服务保留可访问。日常 CH writer mode 现为用户选择的 local_admin；PG 也沿用原管理员账号，永久分权计划仍未执行。
