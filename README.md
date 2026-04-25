# Betalens

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Betalens** 是一个用于量化分析和回测的 Python 框架，包含因子分析、数据获取、回测、绩效分析、稳健性检验等核心模块，适用于量化研究和策略开发。

## ✨ 特性

- 📊 **因子分析** - 支持单因子/多因子分组、打标签、生成多空权重
- 📈 **数据管理** - PostgreSQL 数据库接口，支持时间序列查询、EDE格式解析
- 🔄 **回测框架** - 多资产多权重回测，自动获取价格数据
- 📋 **绩效分析** - 计算夏普比率、最大回撤、年化收益等指标，生成报告
- 🔬 **事件研究** - 事件窗口收益分析，支持多标的、基准超额、可视化
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
│   ├── eventstudy/        # 事件研究模块
│   ├── factor/            # 因子分析模块
│   └── robust/            # 稳健性检验模块
├── docs/                  # Sphinx 文档（ReadTheDocs 部署）
├── test/                  # 测试代码
├── requirements.txt       # 依赖列表
├── pyproject.toml         # 包配置
└── README.md             # 本文件
```

## 📖 文档

完整文档基于 Sphinx 构建，可通过 ReadTheDocs 在线访问，或本地构建：

```bash
cd docs
pip install -r requirements.txt
pip install -e ..
sphinx-build -b html . _build/html
```

**文档结构：**

- **快速上手** - [安装指南](docs/getting-started/installation.rst) · [10 分钟快速上手](docs/getting-started/quickstart.rst)
- **用户指南** - [Datafeed](docs/guide/datafeed.rst) · [Factor](docs/guide/factor.rst) · [Backtest](docs/guide/backtest.rst) · [Analyst](docs/guide/analyst.rst) · [EventStudy](docs/guide/eventstudy.rst) · [Robust](docs/guide/robust.rst)
- **API 参考** - [Datafeed](docs/api/datafeed.rst) · [Factor](docs/api/factor.rst) · [Backtest](docs/api/backtest.rst) · [Analyst](docs/api/analyst.rst) · [EventStudy](docs/api/eventstudy.rst) · [Robust](docs/api/robust.rst)

## 🔧 依赖

- Python >= 3.10
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

