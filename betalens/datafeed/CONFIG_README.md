# Datafeed 配置管理

`betalens.datafeed` 使用外置 JSON 配置管理数据库、日志、Excel、Wind 和 EDE 解析参数。运行时查询层仍通过 `Datafeed` 访问数据；批量入库和 GUI 管理优先使用 `betalens_db_manager`。

## 快速开始

复制模板并修改本地数据库连接：

```powershell
Copy-Item betalens\datafeed\config.example.json betalens\datafeed\config.json
```

```python
from betalens.datafeed import Datafeed

df = Datafeed("daily_market")
prices = df.query_time_range(
    codes=["000001.SZ"],
    start_date="2024-01-01",
    end_date="2024-01-31",
    metric="收盘价(元)",
)
df.close()
```

## 配置文件

- `config.example.json`：模板文件，建议提交。
- `config.json`：本地配置文件，包含数据库密码时不要提交。

常见路径是 `betalens/datafeed/config.json`。配置读取逻辑会先找显式传入路径，再找默认配置，最后回落到内置默认值。

## 配置优先级

1. 运行时参数，例如 `Datafeed(..., db_config={...})`。
2. `config.json`。
3. 代码内置默认值。

```python
from betalens.datafeed import Datafeed

df = Datafeed(
    "daily_market",
    db_config={"dbname": "research", "user": "postgres"},
)
```

## 配置内容

### database

数据库连接信息：

```json
{
  "database": {
    "host": "localhost",
    "port": "5432",
    "dbname": "betalens",
    "user": "postgres",
    "password": "your_password"
  }
}
```

### logging

日志目录、日志级别和格式。默认日志目录通常是 `./logs`。

### excel

Excel/CSV 读取参数，包括候选编码、表头识别和交易时间对齐规则。

### wind

Wind 数据抓取字段映射，覆盖股票、指数、基金、债券等日频数据。

### ede

EDE 文件解析规则，包括日期提取、指标列元数据识别和清洗关键词。

## 读取和修改配置

```python
from betalens.datafeed.config import get_config

config = get_config()
db_name = config.get("database.dbname")

config.set("database.dbname", "new_database")
config.save()
```

使用独立配置文件：

```python
from betalens.datafeed.config import ConfigManager

config = ConfigManager("betalens/datafeed/config.prod.json")
```

## 与数据库管理工具的关系

`Datafeed` 是研究运行时查询层，适合读取 `daily_market`、`fundamentals`、`industry`、`index_universe`、`trade_status` 等表。

文件预览、冲突检查、导入记录、schema 创建和 GUI 管理建议使用 `betalens_db_manager`：

```python
from betalens_db_manager import ImportJobRunner

runner = ImportJobRunner()
preview = runner.preview("data.xlsx", import_type="ede")
record = runner.run("data.xlsx", import_type="ede", table="daily_market")
```

## 常见场景

### 开发环境

复制 `config.example.json` 为 `config.json`，填本地 PostgreSQL 连接即可。

### 生产或共享环境

创建单独的配置文件，例如 `config.prod.json`，并在代码中显式传入或用 `ConfigManager` 读取。

### 敏感信息

可以用环境变量覆盖密码：

```python
import os
from betalens.datafeed.config import get_config

config = get_config()
config.set("database.password", os.environ["DB_PASSWORD"])
```

## 注意事项

- 不要提交包含真实密码的 `config.json`、`config.prod.json`、`config.local.json`。
- 当前主行情表名是 `daily_market`，基本面表名是 `fundamentals`。
- `pre_query_characteristic_data` 的 `time_tolerance` 单位是小时。
- 回测默认从 `daily_market` 读取 `收盘价(元)`。

## 更多文档

- [Datafeed 指南](../../docs/guide/datafeed.rst)
- [数据库管理工具](../../docs/guide/db-manager.rst)
- [Datafeed API](../../docs/api/datafeed.rst)
