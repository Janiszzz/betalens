# db_migrations：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-14788f986bce"></a>
## betalens_db_manager/migrations/0001_bootstrap.sql

[打开源码](../../../betalens_db_manager/migrations/0001_bootstrap.sql) · 10 行 · 说明来源：人工迁移说明

- **作用**：创建 schema、迁移记录和基础覆盖元数据
- **输入**：迁移前数据库状态；由 SchemaManager 按版本执行
- **输出**：DDL/数据迁移后的数据库状态；无 Python 返回值
- **副作用/维护重点**：写库；已有迁移受校验和约束，按源码审查事务与兼容性

声明/SQL 操作/配置键定位：

- [L1](../../../betalens_db_manager/migrations/0001_bootstrap.sql#L1)：`CREATE SCHEMA IF NOT EXISTS betalens;`
- [L3](../../../betalens_db_manager/migrations/0001_bootstrap.sql#L3)：`CREATE TABLE IF NOT EXISTS betalens.schema_migration (`

<a id="file-ab0e4dfec545"></a>
## betalens_db_manager/migrations/0002_dimensions.sql

[打开源码](../../../betalens_db_manager/migrations/0002_dimensions.sql) · 141 行 · 说明来源：人工迁移说明

- **作用**：创建证券、名称历史和指标维度
- **输入**：迁移前数据库状态；由 SchemaManager 按版本执行
- **输出**：DDL/数据迁移后的数据库状态；无 Python 返回值
- **副作用/维护重点**：写库；已有迁移受校验和约束，按源码审查事务与兼容性

声明/SQL 操作/配置键定位：

- [L1](../../../betalens_db_manager/migrations/0002_dimensions.sql#L1)：`CREATE TABLE betalens.entity_dim (`
- [L13](../../../betalens_db_manager/migrations/0002_dimensions.sql#L13)：`CREATE INDEX idx_entity_dim_type_code ON betalens.entity_dim (entity_type, code);`
- [L15](../../../betalens_db_manager/migrations/0002_dimensions.sql#L15)：`CREATE TABLE betalens.entity_name_history (`
- [L23](../../../betalens_db_manager/migrations/0002_dimensions.sql#L23)：`CREATE INDEX idx_entity_name_history_asof`
- [L26](../../../betalens_db_manager/migrations/0002_dimensions.sql#L26)：`CREATE TABLE betalens.metric_dim (`
- [L44](../../../betalens_db_manager/migrations/0002_dimensions.sql#L44)：`CREATE INDEX idx_metric_dim_storage`
- [L47](../../../betalens_db_manager/migrations/0002_dimensions.sql#L47)：`CREATE TABLE betalens.metric_alias (`
- [L57](../../../betalens_db_manager/migrations/0002_dimensions.sql#L57)：`CREATE INDEX idx_metric_alias_metric ON betalens.metric_alias (metric_id);`
- [L59](../../../betalens_db_manager/migrations/0002_dimensions.sql#L59)：`INSERT INTO betalens.metric_dim`
- [L95](../../../betalens_db_manager/migrations/0002_dimensions.sql#L95)：`INSERT INTO betalens.metric_alias (logical_dataset, alias, metric_id)`
- [L99](../../../betalens_db_manager/migrations/0002_dimensions.sql#L99)：`INSERT INTO betalens.metric_alias (logical_dataset, alias, metric_id)`

<a id="file-4cfe0839318b"></a>
## betalens_db_manager/migrations/0003_market_fact.sql

[打开源码](../../../betalens_db_manager/migrations/0003_market_fact.sql) · 18 行 · 说明来源：人工迁移说明

- **作用**：创建日行情事实表与指标存储相关结构
- **输入**：迁移前数据库状态；由 SchemaManager 按版本执行
- **输出**：DDL/数据迁移后的数据库状态；无 Python 返回值
- **副作用/维护重点**：写库；已有迁移受校验和约束，按源码审查事务与兼容性

声明/SQL 操作/配置键定位：

- [L1](../../../betalens_db_manager/migrations/0003_market_fact.sql#L1)：`CREATE TABLE betalens.market_daily_fact (`
- [L16](../../../betalens_db_manager/migrations/0003_market_fact.sql#L16)：`CREATE INDEX idx_market_daily_fact_trade_date_entity`

<a id="file-f978032b260a"></a>
## betalens_db_manager/migrations/0004_observation_fact.sql

[打开源码](../../../betalens_db_manager/migrations/0004_observation_fact.sql) · 13 行 · 说明来源：人工迁移说明

- **作用**：创建按可得时间组织的通用观测事实表
- **输入**：迁移前数据库状态；由 SchemaManager 按版本执行
- **输出**：DDL/数据迁移后的数据库状态；无 Python 返回值
- **副作用/维护重点**：写库；已有迁移受校验和约束，按源码审查事务与兼容性

声明/SQL 操作/配置键定位：

- [L1](../../../betalens_db_manager/migrations/0004_observation_fact.sql#L1)：`CREATE TABLE betalens.observation_fact (`
- [L12](../../../betalens_db_manager/migrations/0004_observation_fact.sql#L12)：`CREATE INDEX idx_observation_metric_time_entity`

<a id="file-0b1ba7c894e5"></a>
## betalens_db_manager/migrations/0005_pit_and_metadata.sql

[打开源码](../../../betalens_db_manager/migrations/0005_pit_and_metadata.sql) · 70 行 · 说明来源：人工迁移说明

- **作用**：创建行业、指数成分、交易状态等 PIT 结构
- **输入**：迁移前数据库状态；由 SchemaManager 按版本执行
- **输出**：DDL/数据迁移后的数据库状态；无 Python 返回值
- **副作用/维护重点**：写库；已有迁移受校验和约束，按源码审查事务与兼容性

声明/SQL 操作/配置键定位：

- [L1](../../../betalens_db_manager/migrations/0005_pit_and_metadata.sql#L1)：`CREATE TABLE betalens.industry_scheme_dim (`
- [L7](../../../betalens_db_manager/migrations/0005_pit_and_metadata.sql#L7)：`CREATE TABLE betalens.industry_dim (`
- [L14](../../../betalens_db_manager/migrations/0005_pit_and_metadata.sql#L14)：`CREATE INDEX idx_industry_dim_scheme_name`
- [L17](../../../betalens_db_manager/migrations/0005_pit_and_metadata.sql#L17)：`CREATE TABLE betalens.industry_membership (`
- [L26](../../../betalens_db_manager/migrations/0005_pit_and_metadata.sql#L26)：`CREATE INDEX idx_industry_membership_entity_asof`
- [L28](../../../betalens_db_manager/migrations/0005_pit_and_metadata.sql#L28)：`CREATE INDEX idx_industry_membership_industry_asof`
- [L31](../../../betalens_db_manager/migrations/0005_pit_and_metadata.sql#L31)：`CREATE TABLE betalens.index_snapshot (`
- [L40](../../../betalens_db_manager/migrations/0005_pit_and_metadata.sql#L40)：`CREATE TABLE betalens.index_constituent (`
- [L49](../../../betalens_db_manager/migrations/0005_pit_and_metadata.sql#L49)：`CREATE INDEX idx_index_constituent_entity`
- [L52](../../../betalens_db_manager/migrations/0005_pit_and_metadata.sql#L52)：`CREATE TABLE betalens.trade_status_event (`
- [L60](../../../betalens_db_manager/migrations/0005_pit_and_metadata.sql#L60)：`CREATE INDEX idx_trade_status_event_date_entity`
- [L63](../../../betalens_db_manager/migrations/0005_pit_and_metadata.sql#L63)：`CREATE TABLE betalens.dataset_coverage (`

<a id="file-abf4b21cf55b"></a>
## betalens_db_manager/migrations/0006_migrate_legacy.sql

[打开源码](../../../betalens_db_manager/migrations/0006_migrate_legacy.sql) · 503 行 · 说明来源：人工迁移说明

- **作用**：将旧结构长表迁移进规范化结构
- **输入**：迁移前数据库状态；由 SchemaManager 按版本执行
- **输出**：DDL/数据迁移后的数据库状态；无 Python 返回值
- **副作用/维护重点**：写库；已有迁移受校验和约束，按源码审查事务与兼容性

声明/SQL 操作/配置键定位：

- [L1](../../../betalens_db_manager/migrations/0006_migrate_legacy.sql#L1)：`CREATE OR REPLACE FUNCTION betalens._migrate_legacy_long(`
- [L196](../../../betalens_db_manager/migrations/0006_migrate_legacy.sql#L196)：`DROP FUNCTION betalens._migrate_legacy_long(TEXT, TEXT);`
- [L203](../../../betalens_db_manager/migrations/0006_migrate_legacy.sql#L203)：`UPDATE betalens.entity_name_history history`
- [L210](../../../betalens_db_manager/migrations/0006_migrate_legacy.sql#L210)：`DO $$`
- [L281](../../../betalens_db_manager/migrations/0006_migrate_legacy.sql#L281)：`DO $$`
- [L373](../../../betalens_db_manager/migrations/0006_migrate_legacy.sql#L373)：`CREATE OR REPLACE FUNCTION betalens._migrate_legacy_names(p_table TEXT)`
- [L415](../../../betalens_db_manager/migrations/0006_migrate_legacy.sql#L415)：`DROP FUNCTION betalens._migrate_legacy_names(TEXT);`
- [L422](../../../betalens_db_manager/migrations/0006_migrate_legacy.sql#L422)：`UPDATE betalens.entity_name_history history`
- [L429](../../../betalens_db_manager/migrations/0006_migrate_legacy.sql#L429)：`DO $$`
- [L466](../../../betalens_db_manager/migrations/0006_migrate_legacy.sql#L466)：`DO $$`

<a id="file-afaa44f571f5"></a>
## betalens_db_manager/migrations/0007_compatibility_views.sql

[打开源码](../../../betalens_db_manager/migrations/0007_compatibility_views.sql) · 473 行 · 说明来源：人工迁移说明

- **作用**：保存旧关系并建立兼容视图
- **输入**：迁移前数据库状态；由 SchemaManager 按版本执行
- **输出**：DDL/数据迁移后的数据库状态；无 Python 返回值
- **副作用/维护重点**：写库；已有迁移受校验和约束，按源码审查事务与兼容性

声明/SQL 操作/配置键定位：

- [L1](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L1)：`CREATE SCHEMA IF NOT EXISTS betalens_legacy;`
- [L3](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L3)：`CREATE OR REPLACE FUNCTION betalens._assert_legacy_long_migrated(`
- [L80](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L80)：`DROP FUNCTION betalens._assert_legacy_long_migrated(TEXT, TEXT);`
- [L82](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L82)：`DO $$`
- [L164](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L164)：`DO $$`
- [L192](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L192)：`CREATE OR REPLACE FUNCTION betalens.entity_name_at(`
- [L218](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L218)：`CREATE OR REPLACE VIEW public.daily_market AS`
- [L257](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L257)：`CREATE OR REPLACE VIEW public.daily_index AS`
- [L296](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L296)：`CREATE OR REPLACE VIEW public.daily_fund AS`
- [L335](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L335)：`CREATE OR REPLACE VIEW public.daily_bond AS`
- [L374](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L374)：`CREATE OR REPLACE VIEW public.fundamentals AS`
- [L386](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L386)：`CREATE OR REPLACE VIEW public.macro AS`
- [L398](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L398)：`CREATE OR REPLACE VIEW public.factors AS`
- [L410](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L410)：`CREATE OR REPLACE VIEW public.industry AS`
- [L429](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L429)：`CREATE OR REPLACE VIEW public.index_universe AS`
- [L452](../../../betalens_db_manager/migrations/0007_compatibility_views.sql#L452)：`CREATE OR REPLACE VIEW public.trade_status AS`

<a id="file-c16ab4922b07"></a>
## betalens_db_manager/migrations/0008_finalize.sql

[打开源码](../../../betalens_db_manager/migrations/0008_finalize.sql) · 101 行 · 说明来源：人工迁移说明

- **作用**：初始化覆盖信息并完成迁移收尾
- **输入**：迁移前数据库状态；由 SchemaManager 按版本执行
- **输出**：DDL/数据迁移后的数据库状态；无 Python 返回值
- **副作用/维护重点**：写库；已有迁移受校验和约束，按源码审查事务与兼容性

声明/SQL 操作/配置键定位：

- [L1](../../../betalens_db_manager/migrations/0008_finalize.sql#L1)：`INSERT INTO betalens.dataset_coverage (logical_dataset)`
- [L58](../../../betalens_db_manager/migrations/0008_finalize.sql#L58)：`UPDATE betalens.dataset_coverage coverage`
- [L66](../../../betalens_db_manager/migrations/0008_finalize.sql#L66)：`COMMENT ON SCHEMA betalens IS`
- [L68](../../../betalens_db_manager/migrations/0008_finalize.sql#L68)：`COMMENT ON TABLE betalens.schema_migration IS`
- [L70](../../../betalens_db_manager/migrations/0008_finalize.sql#L70)：`COMMENT ON TABLE betalens.entity_dim IS`
- [L72](../../../betalens_db_manager/migrations/0008_finalize.sql#L72)：`COMMENT ON TABLE betalens.entity_name_history IS`
- [L74](../../../betalens_db_manager/migrations/0008_finalize.sql#L74)：`COMMENT ON TABLE betalens.metric_dim IS`
- [L76](../../../betalens_db_manager/migrations/0008_finalize.sql#L76)：`COMMENT ON TABLE betalens.metric_alias IS`
- [L78](../../../betalens_db_manager/migrations/0008_finalize.sql#L78)：`COMMENT ON TABLE betalens.industry_scheme_dim IS`
- [L80](../../../betalens_db_manager/migrations/0008_finalize.sql#L80)：`COMMENT ON TABLE betalens.industry_dim IS`
- [L82](../../../betalens_db_manager/migrations/0008_finalize.sql#L82)：`COMMENT ON TABLE betalens.market_daily_fact IS`
- [L84](../../../betalens_db_manager/migrations/0008_finalize.sql#L84)：`COMMENT ON COLUMN betalens.market_daily_fact.remark IS`
- [L86](../../../betalens_db_manager/migrations/0008_finalize.sql#L86)：`COMMENT ON TABLE betalens.observation_fact IS`
- [L88](../../../betalens_db_manager/migrations/0008_finalize.sql#L88)：`COMMENT ON COLUMN betalens.observation_fact.available_at IS`
- [L90](../../../betalens_db_manager/migrations/0008_finalize.sql#L90)：`COMMENT ON COLUMN betalens.observation_fact.period_end IS`
- [L92](../../../betalens_db_manager/migrations/0008_finalize.sql#L92)：`COMMENT ON TABLE betalens.industry_membership IS`
- [L94](../../../betalens_db_manager/migrations/0008_finalize.sql#L94)：`COMMENT ON TABLE betalens.index_snapshot IS`
- [L96](../../../betalens_db_manager/migrations/0008_finalize.sql#L96)：`COMMENT ON TABLE betalens.index_constituent IS`
- [L98](../../../betalens_db_manager/migrations/0008_finalize.sql#L98)：`COMMENT ON TABLE betalens.trade_status_event IS`
- [L100](../../../betalens_db_manager/migrations/0008_finalize.sql#L100)：`COMMENT ON TABLE betalens.dataset_coverage IS`

<a id="file-1171d4c99f63"></a>
## betalens_db_manager/migrations/0009_lifecycle_audit.sql

[打开源码](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql) · 713 行 · 说明来源：人工迁移说明

- **作用**：完善生命周期、名称历史及审计相关结构
- **输入**：迁移前数据库状态；由 SchemaManager 按版本执行
- **输出**：DDL/数据迁移后的数据库状态；无 Python 返回值
- **副作用/维护重点**：写库；已有迁移受校验和约束，按源码审查事务与兼容性

声明/SQL 操作/配置键定位：

- [L1](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L1)：`CREATE INDEX IF NOT EXISTS idx_industry_dim_scheme_name`
- [L4](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L4)：`CREATE OR REPLACE FUNCTION betalens.entity_name_at(`
- [L30](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L30)：`CREATE OR REPLACE VIEW public.daily_market AS`
- [L69](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L69)：`CREATE OR REPLACE VIEW public.daily_index AS`
- [L108](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L108)：`CREATE OR REPLACE VIEW public.daily_fund AS`
- [L147](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L147)：`CREATE OR REPLACE VIEW public.daily_bond AS`
- [L186](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L186)：`CREATE OR REPLACE VIEW public.fundamentals AS`
- [L198](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L198)：`CREATE OR REPLACE VIEW public.macro AS`
- [L210](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L210)：`CREATE OR REPLACE VIEW public.factors AS`
- [L222](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L222)：`CREATE OR REPLACE VIEW public.industry AS`
- [L241](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L241)：`CREATE OR REPLACE VIEW public.index_universe AS`
- [L264](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L264)：`CREATE OR REPLACE VIEW public.trade_status AS`
- [L287](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L287)：`CREATE OR REPLACE FUNCTION betalens._audit_legacy_long(p_dataset TEXT)`
- [L376](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L376)：`CREATE OR REPLACE FUNCTION betalens._audit_legacy_industry()`
- [L470](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L470)：`CREATE OR REPLACE FUNCTION betalens._audit_legacy_index_universe()`
- [L587](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L587)：`CREATE OR REPLACE FUNCTION betalens._audit_legacy_trade_status()`
- [L672](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L672)：`CREATE OR REPLACE FUNCTION betalens.assert_legacy_equivalence()`
- [L705](../../../betalens_db_manager/migrations/0009_lifecycle_audit.sql#L705)：`DO $$`

<a id="file-6e49529b11ba"></a>
## betalens_db_manager/migrations/0010_trade_calendar.sql

[打开源码](../../../betalens_db_manager/migrations/0010_trade_calendar.sql) · 27 行 · 说明来源：人工迁移说明

- **作用**：增加按交易所和日期存储的交易日历
- **输入**：迁移前数据库状态；由 SchemaManager 按版本执行
- **输出**：DDL/数据迁移后的数据库状态；无 Python 返回值
- **副作用/维护重点**：写库；已有迁移受校验和约束，按源码审查事务与兼容性

声明/SQL 操作/配置键定位：

- [L1](../../../betalens_db_manager/migrations/0010_trade_calendar.sql#L1)：`CREATE TABLE betalens.trade_calendar_day (`
- [L8](../../../betalens_db_manager/migrations/0010_trade_calendar.sql#L8)：`INSERT INTO betalens.dataset_coverage (logical_dataset)`
- [L12](../../../betalens_db_manager/migrations/0010_trade_calendar.sql#L12)：`CREATE OR REPLACE VIEW public.trade_calendar AS`
- [L22](../../../betalens_db_manager/migrations/0010_trade_calendar.sql#L22)：`COMMENT ON TABLE betalens.trade_calendar_day IS`
- [L24](../../../betalens_db_manager/migrations/0010_trade_calendar.sql#L24)：`COMMENT ON COLUMN betalens.trade_calendar_day.exchange IS`
- [L26](../../../betalens_db_manager/migrations/0010_trade_calendar.sql#L26)：`COMMENT ON COLUMN betalens.trade_calendar_day.trade_date IS`

<a id="file-2f12ab448721"></a>
## betalens_db_manager/migrations/__init__.py

[打开源码](../../../betalens_db_manager/migrations/__init__.py) · 2 行 · 说明来源：文件族规则

- **作用**：Packaged, immutable SQL migrations for :mod:`betalens_db_manager`.
- **输入**：import 请求
- **输出**：模块导出与符号；见静态 imports
- **副作用/维护重点**：初始化可能导入子模块；__all__ 与真实导出需结合源码阅读

