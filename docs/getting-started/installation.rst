安装指南
========

环境要求
--------

* Python 3.10 及以上。
* PostgreSQL 13+，用于 ``datafeed`` 查询与本地数据库管理工具。
* 可选依赖：``plotly`` 用于交互图，``fastapi``/``uvicorn``/``pyarrow`` 用于 Dashboard，``PySide6`` 用于数据库管理 GUI，WindPy 用于 Wind 数据源。

推荐先创建虚拟环境：

.. code-block:: powershell

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip

源码安装
--------

在仓库根目录执行：

.. code-block:: powershell

   git clone https://github.com/Janiszzz/betalens.git
   cd betalens
   python -m pip install -e .
   python -m pip install -r requirements.txt

按需安装可选依赖：

.. code-block:: powershell

   python -m pip install -e ".[viz,dashboard,db,gui]"

文档构建依赖单独安装：

.. code-block:: powershell

   python -m pip install -r docs\requirements.txt

数据库与配置
------------

``betalens/datafeed/config.example.json`` 是配置模板。仓库内本地开发可复制为已忽略的
``config.local.json`` 后修改数据库连接：

.. code-block:: powershell

   Copy-Item betalens\datafeed\config.example.json betalens\datafeed\config.local.json

配置优先级为：运行时参数 > ``BETALENS_DB_*`` 环境变量 > ``BETALENS_CONFIG`` 指定文件
> ``%APPDATA%\betalens\config.json`` > ``betalens/datafeed/config.local.json``
> 旧 ``config.json`` > 内置默认值。

.. code-block:: json

   {
     "database": {
       "host": "localhost",
       "port": "5432",
       "dbname": "betalens",
       "user": "postgres",
       "password": "your_password"
     },
     "logging": {
       "log_dir": "./logs"
     }
   }

验证安装
--------

1. 导入自检

   .. code-block:: powershell

   python -c "import betalens; from betalens.datafeed import Datafeed; from betalens.factor.factor import single_characteristic; print(betalens.__version__)"

2. 数据查询自检

   .. code-block:: python

      from betalens.datafeed import Datafeed

      df = Datafeed("daily_market")
      latest = df.query_time_range(
          codes=["000001.SZ"],
          start_date="2024-01-01",
          end_date="2024-01-10",
          metric="收盘价(元)",
      )
      print(latest.tail())
      df.close()

3. 文档构建自检

   .. code-block:: powershell

      python -m sphinx -b html -n -W --keep-going docs docs\_build\html

常见问题
--------

* **数据库连接失败**：检查当前生效的用户配置或 ``betalens/datafeed/config.local.json``，确认 PostgreSQL 服务、库名、用户名和密码。
* **psycopg2 编译失败**：本地开发优先安装 ``psycopg2-binary`` 或 ``.[db]`` 可选依赖。
* **缺少 WindPy**：WindPy 是可选数据源；不抓取 Wind 数据时不影响核心回测。
* **文档构建提示缺 sphinx**：执行 ``python -m pip install -r docs\requirements.txt``。
