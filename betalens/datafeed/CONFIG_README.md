# 数据订阅连接配置

Datafeed 使用独立 JSON 配置，只读加载，不调用 db-manager 服务、不建表、不写入业务数据。
模板为 [config.example.json](config.example.json)。`reader` 是示例账号，不代表已经创建；使用实际连接账号。

```python
from betalens.datafeed import Datafeed

with Datafeed("market_daily_raw", config_file="/path/to/subscription.json",
              backend="clickhouse") as data:
    prices = data.query_time_range(
        codes=["000001.SZ"], start_date="2024-01-01", end_date="2024-02-01",
        metric="close", filters={"asset_type": "stock"}, as_of="2024-02-01 16:00:00",
    )
```

配置优先级：构造参数 / `db_config` > 环境变量 > 选中的文件。

- 文件选择：显式 `config_file` > `BETALENS_CONFIG` > 用户配置 > 旧 `config.local.json/config.json`。
- 用户配置：Windows 为 `%APPDATA%/betalens/config.json`；Linux 为 `$XDG_CONFIG_HOME/betalens/config.json`，未设置时 `~/.config/betalens/config.json`。
- 文件明确指定但不存在、JSON 无效、必需连接项缺少时直接报错，不连接内置默认库。
- 无文件时可通过完整 `db_config` 或环境变量连接。旧 JSON 的 `database` 节仍可作为 PG 配置读取。
- `backend` 为 `postgres`（默认）或 `clickhouse`。价格口径由表名明确选择，不再接受 `adjustment` 参数或配置项；旧配置含此项会明确报错，请移除。
- PG 环境变量保留 `BETALENS_DB_HOST/PORT/NAME/USER/PASSWORD`，增加 `BETALENS_DB_SCHEMA`。
- CH 使用 `BETALENS_CH_URL/DATABASE/USER/PASSWORD`；后端可用 `BETALENS_BACKEND` 选择。
- 配置按实例加载并固定；修改文件只影响新建实例，不支持查询中途热切换。

PG 使用只读、可重复读事务和连接池；CH 使用 HTTP 参数绑定与 `readonly=1`，不提供 PG 游标。
连接失败会抛出异常，不自动切换后端。两个后端的数据覆盖可能不同，结构一致不意味着数据已经同步。

`ConfigManager` 的 `get/set/save` 等旧入口尚未迁移，保留供外部代码兼容。新查询运行时不依赖其全局单例和内置默认值；变更运行参数请传构造参数，持久化配置由管理端维护。

`market_daily_post_adjusted` 为后复权，`market_daily_raw` 为未复权；后复权不适合直接用于真实成交手数换算。回测引擎中的该问题留待另行修复。
