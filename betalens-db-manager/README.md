# Betalens DB Manager

独立运行的多来源数据库管理组件，不依赖 Betalens 的研究、因子或回测代码。

```bash
python -m pip install -e './betalens-db-manager[files]'
betalens-db-manager serve
```

浏览器访问 <http://127.0.0.1:8010>。从配置页填写数据库和来源鉴权，再从“本地表”建表或检查现有结构；所有结构变更先预览并确认。现有私有环境文件可通过 `--env /private/credentials.env` 继续使用，现有任务、归档和数据不会搬移。

页面分别使用 `/tables`、`/tasks`、`/mappings`、`/sources`、`/settings`、`/schedules` 路径，支持直接访问、刷新及浏览器前进/后退。映射页通过 `?table=dividend_event` 保留选中的本地表。

默认配置目录为 Linux/WSL `~/.config/betalens/db-manager/` 或 Windows `%LOCALAPPDATA%\Betalens\DBManager\config\`；状态和缓存分别位于 Linux `~/.local/share/betalens-db-manager/` 或 Windows 同级 `state` 目录。凭据禁止写入任意 Git 工作树；Linux 文件权限为 `0600`，Windows 使用受限 ACL。

工作流：查看业务日期覆盖 → 指定本地表和日期 → 预览自动生成的 API 请求 → 创建任务 → 拉取和查看日志 → 审计原始响应、转换及数据库差异 → 确认通过的记录入库 → 核验或回退。

“更新与审计”支持为每次任务设置 Tushare 每分钟请求次数；留空使用私有配置的 `TUSHARE_RPM`，不推断会员等级，也不会改变源端权限。预览表展示每个接口的实际限频、单次行数上限及获取策略；例如 `stock_basic` 另受官方 50 次/分钟限制。HTTP 429 遵守 `Retry-After`，分钟限频会暂停后重试，日/小时配额不足保留检查点并停止。

支持偏移分页的查询按实际返回行数推进，持续请求至空页，短页不会直接终止；重复或交叠页阻止提交。非分页查询达到已登记上限后，按不重叠日期范围、证券或文档登记的枚举条件拆分；证券分区使用重新从来源获取并归档的完整基础清单，不以本地残缺清单代替。父响应仅作审计证据，不重复入库。最小范围仍触及上限或无法证明分区完整时，任务明确阻断。每页及分区保存检查点，恢复时复用经哈希核验的归档；预览中的“请求分片与获取完整性”展示结果。规则针对已登记的 29 个接口，不承诺源端未披露或未返回的数据本身完整。

- 表定义位于 `tables/<table_id>.yaml`；映射位于 `mappings/<table_id>.yaml`。网页编辑和直接编辑使用同一校验逻辑；修改 YAML 不会自动执行 DDL。
- 字段映射页通过下拉多选添加/移除 API 查询源，关系图展示各源到本地表的字段连线。普通字段在单个查询内一一对应；多个查询源使用完整业务主键（支持联合键）合并。共同键允许共享，重复业务列须显式选择主备优先级；同源同键存在不同记录会阻止写入。
- 新建手工表的映射使用 `format_version: 2`：每个绑定用 `join_keys: [{source_field: ticker, column: code}, {source_field: day, column: business_date}]` 记录共同键，并同时保留对应的 `fields` 连线。未使用的源字段在绑定的 `ignored_fields` 中填写原因；不完整规则可保存为草稿，但不能发起更新。旧版内置派生规则继续兼容。
- PostgreSQL 与 ClickHouse 是独立目标，默认全选；单库错误有独立结果。实际入库时间与源业务日期、实际可得时间、下载时间分别保存。
- 内置 22 张业务表，覆盖 29 个 Tushare 接口的已登记输出字段。字段去向登记与实际验证分开报告；空响应、缺权限不代表数据完整。
- Tushare 只是来源之一。支持注册其他 HTTP API 和 Python 适配器（`SubscriptionAdapter` + `FetchContext`）；本地 CSV/XLSX/Parquet 可通过显式映射导入，Wind 未配置时明确报告不可用。
- 后台执行器每次操作使用独立子进程。关闭最后一个管理页面后停止网页任务；重新打开不自动续跑。自动更新默认关闭，启用后也只拉取至待审计状态。
- SQLite 保存请求、版本、操作、日志和回退证据。完成任务的源响应缓存七天后清理，未完成任务和长期审计保留。旧任务不能覆盖后续任务的数据。

CLI 的 `download`、`commit`、`resume`、`rollback` 和 `load` 登记到同一个执行器；CLI 操作同样等待管理页面在线，任务始终能在网页找到。`load` 拉取完成后同样需要审计确认，不自动入库。用 `worker --once` 执行一个排队操作。

```bash
# 后端与真实隔离库测试（连接信息仅从私有文件读取）
BETALENS_INFRA_ENV=/private/credentials.env BETALENS_INFRA_INTEGRATION=1 python -m pytest betalens-db-manager/tests -q
# 浏览器验收：真实后端、隔离 PG/CH、本地 HTTP 来源
cd betalens-db-manager/frontend
npm ci
npx playwright install --with-deps chromium
BETALENS_TEST_ENV=/private/credentials.env npx playwright test
```

完整运维说明见 [infra.rst](../docs/guide/infra.rst)。
