# Betalens DB Manager 审计修复结果

2026-10-23。实施范围为独立组件 `betalens-db-manager/`，没有修改业务包 `betalens/`，没有恢复历史补全任务，也没有登记已取消的自动任务。

代码修复、隔离双库测试和浏览器工作流已完成。生产表结构、ClickHouse 账号权限及运行中的生产服务尚未迁移或发布，不能将测试环境通过视为生产已升级。

| 缺陷 | 修复位置（相对组件 src/betalens_db_manager） | 验证与状态 |
|---|---|---|
| F01 固定提交前提 | `comparison.py`、`commit_protocol.py`、`publishing.py`、`storage.py`、`clickhouse.py`、`state.py` | 保存原 before/after 和全键检查，包含跳过行；stage 后并发改值仍被拒绝。真实 PG 两连接和 CH 两适配器测试通过。 |
| F02 CH 回退并发安全 | `ch_writer.py`、`clickhouse.py`、`schema_manager.py` | 缺少隔离能力时安全阻断。已实现并在隔离 CH 验证本机专用写入口；不同状态目录共享锁，临时表/原表保留检查点，未知查询必须先恢复原操作。生产专用账号尚未部署，生产回退能力仍受阻断。 |
| F03 prepared 无审计的回退 | `publishing.py:reconcile_audits`、`state.py:finish_commit` | 注入“目标已写入、finish_commit 失败”，回退先核验并仅补实际已写入子集的审计；缺证据进入待核验，后续修改产生冲突。 |
| F04 归档重放审计 | `response_validation.py`、`downloader.py`、`manager.py:replay_archive` | 重新检查 SHA、字段、行宽、原证券/日期范围、分页与完整拆分；重放保留原查询事实、范围角色和获取合同，不重新联网，也不借新合同掩盖错误证券。归档索引同步登记到关联任务。 |
| F05 builtin 空值遗漏 | `processors.py`、`mapping_store.py`、`manager.py` | 内置处理器按表/接口登记；自定义 direct 规则不能声明无执行器的 builtin。旧内置 YAML 有确定性兼容，声明输出缺失时拒收。 |
| F06 Int64 合同 | `integer_contract.py`、`schema_contract.py`、`comparison.py`、`storage.py`、`mapping_store.py` | 拒绝越界、小数、非有限值和不精确浮点数。PG bigint/CH Int64 的最大、最小值及 JS 安全整数之外的值真实往返一致。生产 PG 仍有旧 int32，未迁移表会被明确阻止提交。 |
| F07 可空性漂移 | `schema_contract.py`、`schema_manager.py`、`storage.py`、`management_api.py` | 统一检查类型、精度、可空性和键；统计历史 NULL；非空表新增必填列要求先审计回填；部署后重新读物理结构才能标为 ready。 |
| F08 凭据边界 | `security.py`、`source_registry.py`、`config.py`、`api.py`、`management_api.py`、`credential_audit.py` | header/query/body 仅持久化鉴权引用；递归拒绝嵌入凭据。REST/SSE/WS、日志、错误和导出统一脱敏；短凭据不破坏普通文本；HTTP 3xx 不携带鉴权跳转；原 wire 私有保存，脱敏导出使用独立哈希。历史扫描有预算，未扫完明确报告 incomplete。 |
| F09 快照参数丢失 | `manager.py` | 固定合同默认值 → 映射参数 → 任务参数的覆盖顺序，保留 0/false；快照不附加不支持的日期参数，模拟 HTTP 验证实际参数。 |
| F10 最终任务审计同步 | `state.py`、`publishing.py`、`worker.py`、`api.py`、前端 `main.tsx` | SQLite outbox 与本地状态一起持久化；数据状态独立于审计状态。重复确认、恢复或“重试同步审计”只补控制表，不重复业务写入。真实 PG 故障触发器的网页验收通过。 |
| F11 逆操作审计同步 | 同 F10，另含 `clickhouse.py:cleanup_rollback` | 回退来源任务与逆任务各自保留待办；重启后可补同步，重复回退不再执行逆操作。CH 备份只在核验和审计同步后清理，清理失败可再次同步重试。 |

测试证据保存在 `/tmp/betalens-remediation-acceptance/`，含脱敏日志、截图、结构盘点和证据索引。正式回归用例位于组件 `tests/test_remediation_*.py`，不依赖 `/tmp` 的原漏洞复现脚本。

- 最终完整回归：188 项全部通过，耗时 218.28 秒；包含写入口、只读盘点、私密归档和 Int64 边界测试。日志为 `betalens-remediation-release.log`。
- 浏览器：5 个既有流程通过；增加审计同步故障后，补同步与完整更新流程两项重跑通过，共覆盖 6 个不同场景。
- 前端：TypeScript 检查和 Vite 构建通过。
- 浏览器使用真实后端、隔离 PG/CH 和本地模拟 HTTP 源。确认前查业务表没有新增记录；补同步前后业务行逐值相同。
- 安全测试没有删除 `/tmp` 中现存的 Git 标记，也没有放宽私密路径保护。需要凭据文件的正式测试使用 Git 工作树外的测试目录。

新增操作入口：

```bash
# 在 betalens-db-manager/ 目录使用组件自己的开发环境
.venv/bin/python -m betalens_db_manager --env /path/outside/git/private.env migration-preview
.venv/bin/python -m betalens_db_manager --env /path/outside/git/private.env credentials-audit --max-seconds 60 --max-values 2000000
.venv/bin/python -m betalens_db_manager --env /path/outside/git/private.env sync-audit TASK_ID
```

`migration-preview` 仅读物理结构，输出待迁移字段，不执行 DDL；正常服务初始化仍会维护本地注册定义。`credentials-audit` 不初始化服务，只读既有配置和 SQLite。后者输出文件与字段位置，不输出值，也不改写旧归档或确认哈希。`sync-audit` 登记受管理操作，由 worker 执行；网页对应“重试同步审计”。

现有环境的只读盘点结果：

- PostgreSQL：22 张登记表需要整数类型迁移，全部包含 `source_row_no: integer → bigint`；`trade_calendar.is_open` 和 `trade_status_daily.is_tradable` 同样需要升级。没有执行这些 DDL。详细字段、键及类型见 `deployment-preview.json`。
- ClickHouse：22 张表逻辑结构匹配；当前 `CLICKHOUSE_WRITER_MODE=disabled`，提交和回退安全阻断。需先部署专用写账号、限定目标库写权限、移除其他普通写者权限，再设置 `local_dedicated` 并检查写入能力。
- 本机写入口支持 WSL/Linux、同步请求及本机 ReplacingMergeTree。不同 TMPDIR、状态目录、localhost/127.0.0.1 拼写不会生成独立锁。专用凭据必须由管理器独占；管理员属于可信维护人员，不能在维护窗口旁路写入。本实现不声称隔离恶意管理员、共享专用凭据的外部写者或跨主机写者。
- 历史凭据清点按 10 秒预算运行，扫描未完成，结果明确为 `incomplete`；已扫描部分没有已确认的凭据发现。不能据此宣称历史记录全部安全。可增加预算继续只读清点，必要时另建有审计记录的历史清理任务。

部署时应分别登记：PG 类型迁移任务、CH 专用账号与权限调整、生产服务重启记录。旧任务和归档继续保留；无法证明兼容的旧确认必须重新审计，不能自动扩大原授权范围。
