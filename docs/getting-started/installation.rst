安装指南
========

本节汇总 Betalens 在 Windows 与 Linux 环境下的安装要点，帮助你在几分钟内启动整个框架。

环境要求
--------

- Python 3.10 及以上（建议开启虚拟环境）
- PostgreSQL 13+（`datafeed` 默认依赖）
- WindPy、psycopg2、openpyxl 等可选依赖根据实际功能按需启用
- Git ≥ 2.40（用于克隆源码及子模块）

推荐在新建虚拟环境后执行以下命令：

.. code-block:: powershell

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip

通过 PyPI 安装
--------------

.. code-block:: powershell

   pip install betalens

框架会自动拉取 analyst / backtest / datafeed / factor / robust 子模块。若你只需要特定组件，可使用 `pip install betalens[datafeed]` 的可选 extra（需在 `setup.py` 中开启）。

源码安装（推荐开发者）
----------------------

.. code-block:: powershell

   git clone https://github.com/Janiszzz/betalens.git
   cd betalens\gitworks
   pip install -e .

开发流程建议在仓库根目录安装依赖，以便直接引用 `tests` 与示例脚本：

.. code-block:: powershell

   pip install -r gitworks\requirements.txt
   pip install -r gitworks\docs\requirements.txt

数据库与配置
------------

`datafeed/config.example.json` 给出了完整模板。复制后按需修改数据库、日志、Wind 以及 EDE 相关配置：

.. code-block:: powershell

   Copy-Item datafeed\config.example.json datafeed\config.json

关键字段说明：

- ``database``：连接 PostgreSQL 的 host / port / dbname / user / password
- ``log``：日志目录（默认 ``./logs``，会在首次运行时创建）
- ``excel``：批量导入配置，如 key/value 列映射与时间字段
- ``wind``：WindPy 账号与抓取频率

配置类可通过 :func:`betalens.datafeed.config.get_config` 或 :class:`betalens.datafeed.ConfigManager` 在运行时热加载。

验证安装
--------

1. 数据查询自检

   .. code-block:: python

      from betalens.datafeed import Datafeed
      df = Datafeed("daily_market_data")
      latest = df.query_time_range(
          codes=["000001.SZ"],
          start_date="2024-01-01",
          end_date="2024-01-10",
          metric="收盘价(元)"
      )
      print(latest.tail())

2. 快速回测

   .. code-block:: python

      import pandas as pd
      from betalens.backtest import BacktestBase

      weights = pd.DataFrame(
          [[0.5, -0.5, 0.0]] * 5,
          index=pd.date_range("2024-01-01", periods=5, freq="D"),
          columns=["000001.SZ", "000002.SZ", "cash"]
      )
      weights.index.name = "input_ts"
      bb = BacktestBase(weight=weights, symbol="Demo", amount=1_000_000)
      print(bb.nav.tail())

常见问题
--------

- **psycopg2 编译失败**：优先安装 ``psycopg2-binary``，再在服务器环境切换为官方 wheel。
- **缺少 WindPy**：若不需要 Wind 接入，可在 ``autodoc_mock_imports``（conf.py）中加入 ``WindPy``，文档构建即可跳过。
- **ImportError: betalens.factor**：请确认已在仓库根目录或 ``gitworks`` 中执行命令，且环境变量 ``PYTHONPATH`` 包含当前路径。

完成上述步骤后，即可进入下一节的快速上手教程。


