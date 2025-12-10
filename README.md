# Betalens

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Betalens** 是一个用于量化分析和回测的 Python 框架，包含因子分析、数据获取、回测、绩效分析、稳健性检验等核心模块，适用于量化研究和策略开发。

## ✨ 特性

- 📊 **因子分析** - 支持单因子/多因子分组、打标签、生成多空权重
- 📈 **数据管理** - PostgreSQL 数据库接口，支持时间序列查询、EDE格式解析
- 🔄 **回测框架** - 多资产多权重回测，自动获取价格数据
- 📋 **绩效分析** - 计算夏普比率、最大回撤、年化收益等指标，生成报告
- 🧪 **稳健性检验** - 因子增量检验、Bootstrap 重采样

## 📦 安装

```bash
git clone https://github.com/Janiszzz/betalens.git
cd betalens
pip install -r requirements.txt
```

或使用 pip 安装（开发模式）：

```bash
pip install -e .
```

## 🚀 快速开始

```python
import pandas as pd
from betalens.datafeed import Datafeed, get_absolute_trade_days
from betalens.backtest import BacktestBase
from betalens.analyst import PortfolioAnalyzer, ReportExporter

# 1. 获取数据
data = Datafeed("daily_market_data")
date_ranges = get_absolute_trade_days('2020-01-01', '2024-01-01', 'W')

params = {
    'codes': ['000001.SZ'],
    'datetimes': date_ranges,
    'metric': "收盘价(元)",
    'time_tolerance': 48
}
price = data.query_nearest_before(params)

# 2. 构建权重
weights = pd.DataFrame(...)  # 你的权重逻辑
weights['cash'] = 1 - weights.sum(axis=1)

# 3. 回测
bb = BacktestBase(weight=weights, symbol="", amount=1000000)
bb.nav.plot()

# 4. 绩效分析
analyzer = PortfolioAnalyzer(bb.nav)
exporter = ReportExporter(analyzer)
exporter.generate_annual_report()
```

## 📁 项目结构

```
betalens/
├── betalens/              # 主包
│   ├── analyst/           # 绩效分析模块
│   ├── backtest/          # 回测模块
│   ├── datafeed/          # 数据管理模块
│   ├── factor/            # 因子分析模块
│   └── robust/            # 稳健性检验模块
├── docs/                  # 文档
├── examples/              # 示例代码
├── tests/                 # 测试代码
├── requirements.txt       # 依赖列表
├── setup.py              # 安装脚本
└── README.md             # 本文件
```

## 📖 文档

详细文档请访问 [docs/](docs/) 目录：

- [快速开始](docs/getting-started.md)
- [Datafeed 使用指南](docs/datafeed-guide.md)
- [Backtest 回测指南](docs/backtest-guide.md)
- [Analyst 分析指南](docs/analyst-guide.md)
- [API 参考](docs/api-reference.md)

## 🔧 依赖

- Python >= 3.8
- pandas >= 1.3.0
- numpy >= 1.20.0
- psycopg2 >= 2.9.0
- prettytable >= 3.0.0
- matplotlib >= 3.4.0
- openpyxl >= 3.0.0

完整依赖列表见 [requirements.txt](requirements.txt)

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系

- 作者：Janis
- GitHub：[@Janiszzz](https://github.com/Janiszzz)

---

如果这个项目对你有帮助，请给一个 ⭐ Star！

