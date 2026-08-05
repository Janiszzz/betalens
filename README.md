# Betalens

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Betalens** 是一个面向量化研究的 Python 框架，覆盖数据入库与查询、因子分组、回测、绩效评价、事件研究、稳健性检验，以及浏览器 Dashboard。

## 特性

- **Datafeed**：PostgreSQL 数据访问、Excel/EDE/Wind 数据入库、交易日、行业、指数成分、交易状态。
- **Factor**：可交易池、单/双/多因子分组、因子预处理、IC/回归/择时统计、参数挖掘。
- **Backtest**：目标权重到日频净值，整数手成交，停牌状态处理，调仓审计日志。
- **Analyst**：从回测实例生成指标表、Excel 报告和 plotly 交互 HTML。
- **EventStudy / Robust**：事件窗口收益分析和 Lucky Factors 风格因子增量检验。
- **Dashboard**：FastAPI + React/Vite 前端，用浏览器发现因子、配参数、跑回测、下载报告。

## 安装

前置条件：Python 3.10+、PostgreSQL 13+；使用 Dashboard 还需要 Node.js 20.19+。

```powershell
git clone https://github.com/Janiszzz/betalens.git
cd betalens
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[full]"
```

复制数据库配置模板，填写本机 PostgreSQL 密码，然后初始化默认的 `datafeed` 数据库：

```powershell
Copy-Item betalens\datafeed\config.example.json betalens\datafeed\config.local.json
notepad betalens\datafeed\config.local.json
python -m betalens_db_manager plan
.\betalens_db_manager\init_local.bat
python -m betalens_db_manager verify
```

数据库、依赖包、数据导入、Dashboard 和常见故障的逐步说明见[从零安装教程](docs/getting-started/installation.rst)。

## 快速开始

```python
from betalens.datafeed import get_absolute_trade_days
from betalens.factor.factor import (
    get_tradable_pool,
    pre_query_characteristic_data,
    single_characteristic,
    get_single_factor_weight,
)
from betalens.backtest import BacktestBase
from betalens.analyst import Analyst

# 1. 调仓日 + 可交易池
days = get_absolute_trade_days("2020-04-30", "2024-04-30", "Y")
date_ranges, code_ranges = get_tradable_pool(days)

# 2. 查询因子并分组
data = pre_query_characteristic_data(
    days,
    "股息率(报告期)",
    table_name="fundamentals",
    date_ranges=date_ranges,
    code_ranges=code_ranges,
)
labeled = single_characteristic(data, "股息率(报告期)", {"股息率(报告期)": 10})

# 3. 生成多空权重
weights = get_single_factor_weight(labeled, {
    "factor_key": "股息率(报告期)",
    "mode": "classic-long-short",
})
weights["cash"] = 0

# 4. 回测。BacktestBase 默认从 daily_market 取 收盘价(元)，time_tolerance=24 小时。
engine = BacktestBase(
    weight=weights,
    symbol="Dividend",
    amount=1_000_000,
    table_name="daily_market",
)

# 5. 绩效评价
Analyst.from_backtest(engine, name="Dividend").report(
    to_excel="report.xlsx",
    to_html="report.html",
)
```

关键口径：

- 分组函数是真实接口 `single_characteristic` / `double_characteristic` / `multi_characteristic`。
- `pre_query_characteristic_data` 默认表是 `fundamentals`，`time_tolerance` 单位是小时。
- 回测按整数手成交；真实参与净值计算的是 `engine.actual_weight`，不是输入目标 `weight`。
- 绩效评价首选 `Analyst.from_backtest(...).report(...)`。

## Dashboard

```powershell
.\dashboard\run.bat
```

默认地址：

- 前端：`http://127.0.0.1:5173`
- 后端 Swagger：`http://127.0.0.1:8000/docs`

Dashboard 扫描 `betalens-factor/` 下的 YAML 因子配置，支持因子发现、参数编辑、运行日志、结果图表、持仓/交易明细分页、Excel/HTML/profiling 下载。详情见 [dashboard/README.md](dashboard/README.md)。

## 项目结构

```text
betalens/
├── betalens/              # 主包：datafeed / factor / backtest / analyst / eventstudy / robust
├── betalens-factor/       # 因子脚本、YAML 参数、运行产物和参数挖掘入口
├── betalens_db_manager/   # 数据库管理工具；稳定 Python import 名为 betalens_db_manager
├── dashboard/             # FastAPI + React/Vite Dashboard
├── docs/                  # Sphinx 文档
├── tests/                 # 测试
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 文档

本地构建：

```powershell
python -m pip install -r docs\requirements.txt
python -m sphinx -b html -n -W --keep-going docs docs\_build\html
```

文档入口：

- 快速开始：[项目全景](docs/getting-started/project-overview.rst) · [从零安装](docs/getting-started/installation.rst) · [10 分钟快速上手](docs/getting-started/quickstart.rst)
- 用户指南：[Datafeed](docs/guide/datafeed.rst) · [Factor](docs/guide/factor.rst) · [Backtest](docs/guide/backtest.rst) · [Analyst](docs/guide/analyst.rst) · [Dashboard](docs/guide/dashboard.rst) · [因子管线](docs/guide/factor-pipeline.rst) · [参数挖掘](docs/guide/factor-mining.rst) · [数据库管理](docs/guide/db-manager.rst)
- API 参考：[Datafeed](docs/api/datafeed.rst) · [Factor](docs/api/factor.rst) · [Backtest](docs/api/backtest.rst) · [Analyst](docs/api/analyst.rst) · [EventStudy](docs/api/eventstudy.rst) · [Robust](docs/api/robust.rst)

## 依赖

核心依赖见 [pyproject.toml](pyproject.toml) 和 [requirements.txt](requirements.txt)。可选依赖组：

- `viz`：plotly 交互图。
- `dashboard`：FastAPI / Uvicorn / Pydantic / PyArrow。
- `db`：PostgreSQL 文件导入、Parquet 和旧 XLS 支持。
- `gui`：PySide6 数据库管理 GUI。
- `full`：一次安装上述全部可选能力，推荐新用户使用。

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

## 联系

- 作者：Janis
- GitHub：[@Janiszzz](https://github.com/Janiszzz)
