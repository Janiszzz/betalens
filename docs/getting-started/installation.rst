从零安装 Betalens
=================

这是一条面向第一次配置 Python 和 PostgreSQL 的完整安装路径。主流程以
**Windows 10/11 + PowerShell** 为例；macOS 和 Linux 用户可使用文末的命令对照。

完成后，你将拥有：

* 一个独立的 Python 虚拟环境；
* 一个可连接的 PostgreSQL ``datafeed`` 数据库；
* Betalens 主包、数据库管理器、Dashboard 和全部常用依赖；
* 已初始化的表、索引和 ``daily_market`` 等兼容视图；
* 一组可以逐项执行的安装验收命令。

.. contents:: 本页目录
   :local:
   :depth: 2

先理解安装顺序
--------------

整个过程按下面的顺序进行。前一项没有通过验收时，先不要继续下一项。

.. list-table:: 安装路线
   :header-rows: 1
   :widths: 8 34 36 22

   * - 步骤
     - 安装内容
     - 用途
     - 验收命令
   * - 1
     - Git、Python、PostgreSQL
     - 下载源码、运行框架、保存量化数据
     - ``git --version``、``python --version``、``psql --version``
   * - 2
     - Betalens 源码和 ``.venv``
     - 隔离本项目与电脑上的其他 Python 包
     - ``python -c "import sys; print(sys.executable)"``
   * - 3
     - Python 依赖和 Betalens
     - 安装主包、数据库、Dashboard、GUI 依赖
     - ``python -c "import betalens"``
   * - 4
     - 数据库配置和 schema
     - 建立 ``datafeed`` 库及标准表结构
     - ``python -m betalens_db_manager verify``
   * - 5
     - 研究数据和 Dashboard
     - 让查询、因子和回测真正有数据可用
     - Datafeed 查询、浏览器健康检查

.. important::

   PostgreSQL 是数据库服务，Betalens 是 Python 程序。安装 PostgreSQL 只会得到一个
   空数据库服务，不会自动包含行情和财务数据；安装 Betalens 也不会自动下载商业数据。
   schema 初始化和数据导入是两个独立步骤。

1. 准备基础软件
---------------

推荐版本
~~~~~~~~

.. list-table:: 软件要求
   :header-rows: 1
   :widths: 24 28 48

   * - 软件
     - 版本
     - 说明
   * - Windows
     - 10 或 11，64 位
     - 主教程使用 PowerShell，不建议在 CMD 和 PowerShell 之间来回切换
   * - Git
     - 当前稳定版
     - 安装时保留“Git from the command line”选项
   * - Python
     - 3.10 及以上；推荐 3.11 或 3.12，64 位
     - 从 python.org 安装时勾选 ``Add python.exe to PATH``
   * - PostgreSQL
     - 13 及以上，64 位
     - 需要 PostgreSQL Server；pgAdmin 和 Command Line Tools 推荐一并安装
   * - Node.js
     - 20.19+；推荐当前 LTS
     - 仅 Dashboard 前端需要，纯 Python 研究可暂不安装

安装完成后，**重新打开一个 PowerShell 窗口**，逐条执行：

.. code-block:: powershell

   git --version
   python --version
   python -m pip --version
   psql --version
   node --version
   npm --version

只使用核心 Python 功能时，最后两条可以暂时失败。如果 ``python`` 打开 Microsoft Store
或提示找不到命令，请关闭 Windows 的“管理应用执行别名”中的 ``python.exe`` 别名，或重新
安装 Python 并勾选 PATH 选项。

.. note::

   ``psql`` 找不到但 pgAdmin 能连接，通常只是 PostgreSQL 的 ``bin`` 目录没有加入 PATH。
   常见路径为 ``C:\Program Files\PostgreSQL\<版本>\bin``。可以继续使用 pgAdmin，
   也可以把实际 ``bin`` 路径加入用户 PATH 后重开 PowerShell。

2. 安装并检查 PostgreSQL
------------------------

Windows 安装器中的关键选项
~~~~~~~~~~~~~~~~~~~~~~~~~~

安装 PostgreSQL 时，建议保留以下组件：

* **PostgreSQL Server**：必须；真正保存数据的服务。
* **pgAdmin 4**：推荐；适合不熟悉命令行的用户查看数据库。
* **Command Line Tools**：推荐；提供本教程使用的 ``psql``。
* **Stack Builder**：Betalens 不依赖，可以不安装额外组件。

安装器会要求设置超级用户 ``postgres`` 的密码。请记住这个密码；后面初始化数据库时会用到。
端口没有冲突时保持默认 ``5432``，区域设置保持默认即可。

检查数据库服务
~~~~~~~~~~~~~~

在 PowerShell 执行：

.. code-block:: powershell

   Get-Service *postgres*
   psql -U postgres -h localhost -p 5432 -d postgres -c "SELECT version();"

第二条命令会提示输入安装时设置的密码。看到一行 PostgreSQL 版本信息即表示服务、端口、
用户和密码均可用。

.. admonition:: 使用 pgAdmin 验证

   如果不使用 ``psql``，打开 pgAdmin 4，在左侧展开 Servers，连接本机服务器，再打开
   Query Tool 执行 ``SELECT version();``。结果与命令行验收等价。

此时不必手工创建 ``datafeed`` 数据库。后面的 Betalens Database Manager 会在确认后创建
数据库并安装正确版本的 schema。默认 ``postgres`` 用户具备创建数据库的权限；如果使用
公司分配的普通账号，需要管理员授予 ``CREATEDB``，或请管理员先创建空的 ``datafeed`` 库。

3. 下载 Betalens 源码
---------------------

选择一个路径短、可写且空间充足的目录。以下示例使用 ``C:\dev``：

.. code-block:: powershell

   New-Item -ItemType Directory -Force C:\dev
   Set-Location C:\dev
   git clone https://github.com/Janiszzz/betalens.git
   Set-Location .\betalens
   git status

``git status`` 应显示当前分支和工作区状态。后续命令除非特别说明，都在仓库根目录执行，也就是
能看到 ``pyproject.toml``、``betalens``、``betalens_db_manager`` 和 ``dashboard`` 的目录。

.. warning::

   不要把仓库放进系统目录、Python 安装目录或需要管理员权限的目录。OneDrive 通常可以使用，
   但大规模数据、Node.js 依赖或回测产物频繁同步时可能变慢；新手优先使用 ``C:\dev``。

4. 创建独立 Python 环境
-----------------------

在仓库根目录执行：

.. code-block:: powershell

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip setuptools wheel

命令行开头通常会出现 ``(.venv)``。再确认当前 Python 确实来自仓库内的虚拟环境：

.. code-block:: powershell

   python -c "import sys; print(sys.executable)"

输出路径应以 ``<仓库路径>\.venv\Scripts\python.exe`` 结尾。

如果 PowerShell 提示“禁止运行脚本”，只为当前窗口临时放行后重新激活：

.. code-block:: powershell

   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1

以后每次打开新的 PowerShell，都要先进入仓库并执行激活命令。关闭窗口或执行
``deactivate`` 不会删除环境。

5. 安装依赖和 Betalens
-----------------------

推荐：完整安装
~~~~~~~~~~~~~~

新人建议一次装齐常用能力：

.. code-block:: powershell

   python -m pip install -e ".[full]"

``-e`` 表示源码可编辑安装。以后更新仓库内 Python 代码时通常不需要重复安装；修改
``pyproject.toml`` 中的依赖后，需要重新执行上面的命令。

``full`` 包含核心依赖，以及：

* ``db``：PostgreSQL 文件入库、Parquet 和旧 ``.xls`` 支持；
* ``viz``：Plotly 交互图和 HTML 报告；
* ``dashboard``：FastAPI 后端和大表分页；
* ``gui``：PySide6 数据库管理桌面界面。

按需安装
~~~~~~~~

磁盘或下载条件有限时，可以选择：

.. code-block:: powershell

   # 仅主包和核心研究能力
   python -m pip install -e .

   # 主包 + 数据库导入
   python -m pip install -e ".[db]"

   # 主包 + 数据库 + Dashboard，不安装桌面 GUI
   python -m pip install -e ".[db,viz,dashboard]"

``requirements.txt`` 面向完整本地开发环境，仍可用于维护者安装，但普通用户以
``pyproject.toml`` 的 extras 为准，避免重复执行两套依赖命令。

安装验收
~~~~~~~~

.. code-block:: powershell

   python -c "import betalens; print('Betalens', betalens.__version__)"
   python -c "import psycopg2, pandas, numpy; print('Python dependencies OK')"
   python -m pip check

预期看到 Betalens 版本、``Python dependencies OK`` 和 ``No broken requirements found.``。

6. 配置数据库连接
-----------------

推荐先复制仓库提供的模板。``config.local.json`` 已被 Git 忽略，不会被正常提交：

.. code-block:: powershell

   Copy-Item betalens\datafeed\config.example.json betalens\datafeed\config.local.json
   notepad betalens\datafeed\config.local.json

把连接信息改成自己的值。默认数据库名是 ``datafeed``，不是 ``betalens``：

.. code-block:: json

   {
     "database": {
       "dbname": "datafeed",
       "user": "postgres",
       "password": "替换为安装 PostgreSQL 时设置的密码",
       "host": "localhost",
       "port": "5432"
     },
     "logging": {
       "log_dir": "./logs",
       "log_level": "INFO"
     }
   }

保存后关闭记事本。不要把含真实密码的配置文件发送给别人或强制加入 Git。

也可以不用配置文件，只在当前 PowerShell 会话设置环境变量：

.. code-block:: powershell

   $env:BETALENS_DB_NAME = "datafeed"
   $env:BETALENS_DB_USER = "postgres"
   $env:BETALENS_DB_PASSWORD = "替换为你的密码"
   $env:BETALENS_DB_HOST = "localhost"
   $env:BETALENS_DB_PORT = "5432"

环境变量会覆盖配置文件。完整优先级是：运行时参数 > ``BETALENS_DB_*`` >
``BETALENS_CONFIG`` 指定文件 > 用户配置 > ``config.local.json`` > 旧配置 > 默认值。
详见 :doc:`../guide/datafeed`。

7. 初始化 Betalens 数据库
-------------------------

先查看只读计划。它不会修改 schema：

.. code-block:: powershell

   python -m betalens_db_manager plan

确认输出中的目标数据库是 ``datafeed``，再执行初始化：

.. code-block:: powershell

   .\betalens_db_manager\init_local.bat

等价的跨平台命令是：

.. code-block:: console

   python -m betalens_db_manager init --yes

初始化工具会按需创建 ``datafeed`` 数据库、执行版本化 migration、建立索引和兼容视图，
最后自动核验。命令可以重复执行；已完成的 migration 不会重复破坏数据。

再次独立验收：

.. code-block:: powershell

   python -m betalens_db_manager verify

成功报告的 ``status`` 应为 ``completed``。数据库中会有 ``betalens`` schema，以及
``daily_market``、``daily_index``、``daily_fund``、``daily_bond``、``fundamentals``、
``macro``、``factors``、``industry``、``index_universe`` 和 ``trade_status`` 等只读兼容视图。

.. admonition:: 已有旧数据库

   初始化器会规划旧表迁移并保留旧结构用于审计。先运行 ``plan``，阅读待执行 migration；
   生产数据库应先完成备份。数据库迁移、Manifest 和恢复细节见 :doc:`../guide/db-manager`。

8. 导入研究数据
---------------

新建的 schema 是空的。没有行情、交易状态和因子输入数据时，框架可以导入，但无法完成实际
回测。数据来源可以是自有 CSV/Excel/Parquet、EDE 导出或 Wind 数据。

图形界面导入
~~~~~~~~~~~~

完整安装后执行：

.. code-block:: powershell

   .\betalens_db_manager\run.bat

也可以运行 ``python -m betalens_db_manager``。在界面中填写连接信息，先预览文件和目标表，
检查字段映射、拒绝行和冲突，再执行导入。建议至少准备：

* ``daily_market``：日行情，回测默认读取 ``收盘价(元)``；
* ``trade_status``：上市、停牌等状态，可交易池和回测默认使用；
* ``fundamentals`` 或 ``factors``：因子研究所需指标；
* ``industry``、``index_universe``：行业中性化和指数成分研究按需导入。

批量清单导入
~~~~~~~~~~~~

大量文件建议从 ``betalens_db_manager/import_manifest.example.yaml`` 复制一份 Manifest，修改
数据路径、``target`` 和 ``adapter`` 后先预检：

.. code-block:: powershell

   python -m betalens_db_manager plan --manifest C:\data\imports.yaml
   python -m betalens_db_manager import --manifest C:\data\imports.yaml --yes
   python -m betalens_db_manager verify --deep

``--yes`` 会确认清单中的 upsert 可能更新已有值。第一次导入前务必阅读预检输出。各种输入
格式、字段约束和失败恢复见 :doc:`../guide/db-manager`。

9. 完整验收
------------

Python 包和数据库连接
~~~~~~~~~~~~~~~~~~~~

下面的命令会真正打开只读数据库连接，并读取最多一行行情：

.. code-block:: powershell

   python -c "from betalens.datafeed import Datafeed; feed=Datafeed('daily_market'); print(feed.query_time_range(limit=1)); feed.close()"

空数据库会打印空表，但不应出现连接或 relation 错误。导入数据后应看到一行结果。

数据库管理 CLI
~~~~~~~~~~~~~~

.. code-block:: powershell

   python -m betalens_db_manager --help
   python -m betalens_db_manager verify

Dashboard
~~~~~~~~~

先确认已经安装 Node.js，再启动前后端：

.. code-block:: powershell

   node --version
   npm --version
   .\dashboard\run.bat

首次启动时会自动执行 ``npm install``，耗时通常比后续启动长。两个终端窗口启动后访问：

* 前端：``http://127.0.0.1:5173``
* 后端健康检查：``http://127.0.0.1:8000/api/health``
* API 文档：``http://127.0.0.1:8000/docs``

首页能列出 ``betalens-factor`` 中的因子，健康检查返回正常状态，即表示 Dashboard 安装完成。
使用方式见 :doc:`../guide/dashboard`。

10. 日常启动顺序
----------------

安装只做一次。以后开始研究时通常只需：

.. code-block:: powershell

   Set-Location C:\dev\betalens
   .\.venv\Scripts\Activate.ps1
   python -m betalens_db_manager verify

需要 Dashboard 时再执行 ``.\dashboard\run.bat``。拉取代码更新后执行：

.. code-block:: powershell

   git pull
   python -m pip install -e ".[full]"
   python -m betalens_db_manager plan

确认数据库升级计划后，再运行 ``init_local.bat``。

macOS / Linux 命令对照
----------------------

先用系统包管理器安装 Git、Python 3、PostgreSQL 13+；需要 Dashboard 时再安装 Node.js 20.19+。
PostgreSQL 的服务启动和用户创建方式因发行版而异，但后续 Betalens 命令相同。

.. list-table:: Windows 与 macOS/Linux 对照
   :header-rows: 1
   :widths: 32 34 34

   * - 操作
     - Windows PowerShell
     - macOS / Linux
   * - 创建虚拟环境
     - ``python -m venv .venv``
     - ``python3 -m venv .venv``
   * - 激活环境
     - ``.\.venv\Scripts\Activate.ps1``
     - ``source .venv/bin/activate``
   * - 完整安装
     - ``python -m pip install -e ".[full]"``
     - ``python -m pip install -e '.[full]'``
   * - 初始化 schema
     - ``.\betalens_db_manager\init_local.bat``
     - ``python -m betalens_db_manager init --yes``
   * - 启动数据库 GUI
     - ``.\betalens_db_manager\run.bat``
     - ``python -m betalens_db_manager``
   * - 启动 Dashboard
     - ``.\dashboard\run.bat``
     - 分别运行后端和 ``dashboard/frontend`` 下的 ``npm run dev``

常见问题
--------

``password authentication failed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

用户名或密码不匹配。先用 ``psql -U postgres -h localhost -d postgres`` 验证密码，再检查
``config.local.json`` 或 ``BETALENS_DB_*``。修改环境变量后，重启正在运行的 Python 和
Dashboard 进程。

``connection refused`` 或连接超时
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PostgreSQL 服务未启动、host/port 错误或端口被防火墙拦截。Windows 先执行
``Get-Service *postgres*``，确认状态为 ``Running``；本机默认连接通常是
``localhost:5432``。

``permission denied to create database``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当前 PostgreSQL 用户没有 ``CREATEDB`` 权限。使用安装时创建的 ``postgres`` 用户初始化，
或让管理员先创建空 ``datafeed`` 数据库，再执行：

.. code-block:: powershell

   python -m betalens_db_manager init --yes --no-create-database

``No module named ...``
~~~~~~~~~~~~~~~~~~~~~~~

通常是虚拟环境未激活，或安装命令用了另一套 Python。检查
``python -c "import sys; print(sys.executable)"``，再在正确环境执行
``python -m pip install -e ".[full]"``。始终使用 ``python -m pip``，不要混用裸 ``pip``。

安装卡在 PySide6
~~~~~~~~~~~~~~~~

PySide6 体积较大，只用于数据库 GUI。可以先安装 ``.[db,viz,dashboard]`` 并使用 CLI；网络
恢复后再执行 ``python -m pip install -e ".[gui]"``。

``npm install`` 或 Vite 启动失败
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

先确认 ``node --version`` 至少为 20.19。删除或重装前端依赖前，先记录完整错误；常见原因是
Node.js 太旧、代理不可用或安全软件锁定 ``node_modules``。核心 Python 功能不依赖 Node.js。

查询提示表不存在
~~~~~~~~~~~~~~~~

Python 已连接到 PostgreSQL，但目标库尚未初始化，或配置指向了错误数据库。确认数据库名为
``datafeed``，依次执行 ``plan``、``init --yes`` 和 ``verify``。

查询返回空表
~~~~~~~~~~~~

schema 已安装但没有导入数据，或者代码、日期、指标名与数据库不一致。这不是安装失败。
先用数据库管理 GUI 查看表行数，再核对中文 metric 名和证券代码格式。

卸载或重建虚拟环境
~~~~~~~~~~~~~~~~~~

虚拟环境只包含可重新安装的 Python 包。退出正在使用它的程序并执行 ``deactivate`` 后，可以
删除仓库内的 ``.venv`` 再按第 4、5 步重建。**这不会删除 PostgreSQL 数据库**；数据库删除
是独立且不可逆的操作，本教程不提供自动删除命令。

下一步
------

安装和数据准备完成后，进入 :doc:`quickstart` 跑通“取数 → 分组 → 权重 → 回测 → 报告”。
模块职责、真实 API 和进阶流程可从文档首页按 Datafeed、Factor、Backtest、Analyst、
Database Manager 和 Dashboard 继续阅读。
