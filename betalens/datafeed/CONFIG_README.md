# Datafeed配置管理

## 快速开始

datafeed模块现已支持外置配置管理！所有默认参数都已迁移到`config.json`文件中。

### 立即使用

无需任何修改，模块会自动加载配置：

```python
from datafeed import Datafeed

# 自动从config.json加载配置
df = Datafeed("daily_market_data")
```

### 自定义配置

修改`config.json`文件中的参数即可：

```json
{
  "database": {
    "dbname": "my_database",
    "user": "my_user",
    "password": "my_password"
  }
}
```

## 配置文件

- **config.json**: 主配置文件（自动加载）
- **config.example.json**: 配置模板（可复制修改）

## 配置内容

### 数据库配置
- 数据库名称、用户名、密码
- 主机地址、端口

### 日志配置
- 日志目录、日志级别
- 日志格式

### Excel处理配置
- 支持的编码列表
- 交易时间对齐（开盘价09:30，其他15:00）

### Wind数据配置
- 股票、指数、基金、债券字段映射

### EDE格式配置
- 日期提取模式
- 列名识别规则
- 数据清理关键词

## 配置管理

### 读取配置

```python
from datafeed import get_config

config = get_config()
db_name = config.get('database.dbname')
```

### 修改配置

```python
from datafeed import get_config

config = get_config()
config.set('database.dbname', 'new_database')
config.save()  # 保存到文件
```

### 运行时覆盖

```python
from datafeed import Datafeed

# 临时覆盖配置
df = Datafeed(
    "table_name",
    db_config={'dbname': 'custom_db'}
)
```

## 配置优先级

1. **运行时参数** > 2. **config.json** > 3. **内置默认值**

## 完整文档

- 📖 [完整配置指南](docs/CONFIG_GUIDE.md)
- 💡 [使用示例代码](examples/config_usage_example.py)
- 📝 [更新日志](docs/changelogs/2025-11-22_v2.4.0_外置配置支持.md)

## 向后兼容

✅ 完全向后兼容，现有代码无需修改！

## 常见场景

### 场景1：开发环境
使用默认的`config.json`即可。

### 场景2：生产环境
1. 复制`config.example.json`为`config.prod.json`
2. 修改生产环境配置
3. 代码中指定配置文件：
```python
from datafeed.config import ConfigManager
config = ConfigManager('datafeed/config.prod.json')
```

### 场景3：敏感信息管理
使用环境变量：
```python
import os
from datafeed import get_config

config = get_config()
config.set('database.password', os.environ.get('DB_PASSWORD'))
```

## 注意事项

⚠️ **生产环境配置文件不要提交到版本控制！**

在`.gitignore`中添加：
```
datafeed/config.prod.json
datafeed/config.local.json
```

## 需要帮助？

查看详细文档：[docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md)

