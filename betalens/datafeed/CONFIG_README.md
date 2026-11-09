# 数据订阅连接配置

Datafeed 只读加载独立 JSON，不管理配置持久化、建表和数据写入。模板见 [config.example.json](config.example.json)。

```python
from betalens.datafeed import Datafeed

with Datafeed("market_daily_raw", backend="clickhouse",
              config_file="/path/to/subscription.json") as data:
    prices = data.query_time_range(
        codes=["000001.SZ"], metric="close", filters={"asset_type": "stock"},
        start_date="2024-01-01", end_date="2024-01-31", as_of="2024-01-31 16:00:00",
    )
```

- 顶层字段为 `backend/pit_policy/postgres/clickhouse`。PG 使用 `host/port/dbname/schema/user/password`，CH 使用 `url/database/user/password`。
- 连接参数优先级：显式 `db_config` > 环境变量 > 文件。`backend` 构造参数最高。
- 文件选择：`config_file` > `BETALENS_CONFIG` > 用户目录 `betalens/config.json`；Windows 使用 APPDATA，Linux 使用 XDG_CONFIG_HOME 或 `~/.config`。
- 不再回退包内 `config.local.json/config.json`，不接受 `database/logging/adjustment` 等旧配置键，也没有默认数据库连接。
- PG 环境变量为 `BETALENS_PG_HOST/PORT/DATABASE/SCHEMA/USER/PASSWORD`；CH 为 `BETALENS_CH_URL/DATABASE/USER/PASSWORD`。后端为 `BETALENS_BACKEND`。
- 旧 `BETALENS_DB_*` 会明确报错，不自动迁移私有凭据。无文件时可使用完整连接覆盖或新版环境变量。
- `connect_timeout/statement_timeout_ms` 为正整数；PG 额外支持正整数 `pool_min_connections/pool_max_connections`。未知键、错误类型及无效文件直接报错。
- `pit_policy` 为 `research`（默认）或 `strict`；构造参数优先于文件，推定时间不会变成严格证据。
- 新实例读取新配置；查询期间使用固定配置。旧 ConfigManager、get_config、get_database_config 及读写配置 API 已删除。

价格口径由物理表名明确选择：`market_daily_post_adjusted` 为后复权，`market_daily_raw` 为未复权；指数使用原生点位表。
无时区查询时间统一按上海解释，数据库时间按 UTC 读取，带时区输入尊重原时区。数值日期及宽松格式不接受。

后复权成交手数与权重换算是待修复的回测问题；本轮只接入新的查询合同。
