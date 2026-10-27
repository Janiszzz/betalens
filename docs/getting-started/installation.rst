从零安装 Betalens
=================

这是一条面向第一次配置 Python 和 PostgreSQL 的完整安装路径。主流程以
**Windows 10/11 + PowerShell** 为例；macOS 和 Linux 用户可使用文末的命令对照。

完成后，你将拥有：

* 一个独立的 Python 虚拟环境；
* 一个可连接的 PostgreSQL ``betalens`` 库，以及按需配置的 ClickHouse ``betalens`` 库；
* Betalens 主包、独立 Infra 工具、Dashboard 和全部常用依赖；
* 经明确确认建立的统一表和可审计更新任务；旧研究查询合同需另行适配；
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
     - 安装主包、数据库和 Dashboard 依赖；管理器独立安装
     - ``python -c "import betalens"``
   * - 4
     - 数据库配置和 schema
     - 配置 ``betalens`` 两库并预览确认结构
     - ``betalens-db-manager --env /private/path/infra.env doctor``
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

此时不必手工创建 ``betalens`` 数据库。独立 DB Manager 可在确认后建立通用结构；
参考域的 create-only 安装要求 PG 目标数据库已存在。若日常账号没有 ``CREATEDB``，
请管理员先创建空库，再单独预览并确认建表。账号分权与具体权限由部署者配置。

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
能看到 ``pyproject.toml``、``betalens``、``betalens-db-manager`` 和 ``dashboard`` 的目录。

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
* ``gui``：独立 Infra 网页。

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

6. 配置并启动独立 DB Manager
-----------------------------

DB Manager 不随主包安装，使用独立虚拟环境，以开发模式安装：

.. code-block:: powershell

   python -m venv betalens-db-manager\.venv
   $infraPython = ".\betalens-db-manager\.venv\Scripts\python.exe"
   & $infraPython -m pip install -e ".\betalens-db-manager[files]"
   $privateDir = Join-Path $env:LOCALAPPDATA "Betalens\DBManager\config"
   New-Item -ItemType Directory -Force $privateDir
   $privateEnv = Join-Path $privateDir "credentials.env"
   # 仅初次创建时复制，保留已有私密配置
   if (-not (Test-Path $privateEnv)) {
       Copy-Item betalens-db-manager\infra.env.example $privateEnv
   }
   & $infraPython -m betalens_db_manager --env $privateEnv serve --port 8010

访问 ``http://localhost:8010/settings``，填写数据库连接、密码及 API Token。
PG 的数据库与 schema 使用 ``betalens``，CH 数据库使用 ``betalens``；按需选择 CH 受控写入模式。
网页只显示秘密是否配置，密码框留空保留旧值。凭据保存到仓库外私有环境文件，
不得放在 Git 工作树内，即使该路径已被忽略。Windows 使用受限 ACL；WSL 使用 Linux
文件系统上的 ``0600`` 文件和 ``0700`` 目录。

Linux/WSL 对应启动及本机目录见 :doc:`../guide/infra` 和
:doc:`../guide/local-databases`。Windows 浏览器访问 WSL 服务时仍使用 WSL 配置。
DB Manager 的 ``PG*/CLICKHOUSE_*/TUSHARE_*`` 与研究层 ``BETALENS_DB_*``
是独立配置，不会相互自动同步。

7. 检查连接并确认结构
---------------------

可在网页配置页测试连接，或在组件环境中执行：

.. code-block:: powershell

   & $infraPython -m betalens_db_manager --env $privateEnv doctor
   & $infraPython -m betalens_db_manager --env $privateEnv schema plan

查看结构预览后，再按生成的令牌明确执行 ``schema apply --token <plan-token>``，
或在网页本地表页面确认建表。``/integrity`` 页为三个业务参考域单独预览和安装六张
版本/发布表，不迁移旧表。普通服务启动与 ``doctor`` 不会替代建表确认，
连接成功也不表示数据完整。遇到现有结构漂移先预览迁移，不能用删除重建代替检查。

8. 下载、审计与确认入库
-----------------------

``/tasks`` 管理通用本地表，``/integrity`` 管理股票历史行业、交易日历和股票停牌区间。
选择目标库、来源、证券和日期，登记请求后显式开始提数；核对实时 HTTP 日志、归档字段、
单位、时间、标准化值及 before/after，再确认入库。PG/CH 默认双选，但每库独立记录结果。
文件接入必须显式映射字段；Wind 未配置时不能执行。详见 :doc:`../guide/infra`。

三域版本追加后需经目标库核验及可信发布才能业务读取；旧行、未发布行和已撤销版本仍保留供审计。
完整性检查按请求范围凭证，不能用表中某天存在一行代替所选股票完整。
自动更新默认关闭；手动更新和首次安装不会恢复历史补全。

新建统一表不生成 ``daily_market/trade_status`` 等旧视图。研究层的 Datafeed 和
BacktestBase 尚需显式适配；先验证管理链路，再按实际业务合同准备研究查询。

9. 完整验收
------------

Python 包和数据库连接
~~~~~~~~~~~~~~~~~~~~

下面的命令适用于**已经准备好旧研究查询合同的数据库**，会打开只读连接并读取最多一行行情。
仅完成新 DB Manager 建表时，不应将缺少 ``daily_market`` 视为管理器安装失败：

.. code-block:: powershell

   python -c "from betalens.datafeed import Datafeed; feed=Datafeed('daily_market'); print(feed.query_time_range(limit=1)); feed.close()"

只有目标研究库已具备相应查询结构时，空数据会打印空表；新统一库需完成查询适配后再使用该示例。

Infra CLI
~~~~~~~~~~~~~~

.. code-block:: powershell

   & $infraPython -m betalens_db_manager --help
   & $infraPython -m betalens_db_manager --env $privateEnv doctor

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
   & .\betalens-db-manager\.venv\Scripts\python.exe -m betalens_db_manager --env $privateEnv doctor

需要 Dashboard 时再执行 ``.\dashboard\run.bat``。拉取代码更新后执行：

.. code-block:: powershell

   git pull
   python -m pip install -e ".[full]"
   & .\betalens-db-manager\.venv\Scripts\python.exe -m betalens_db_manager --env $privateEnv schema plan

代码更新不会自动修改数据库结构；根据检查结果单独处理需要的表结构变更。

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
   * - 检查管理连接
     - ``betalens-db-manager --env <private-env> doctor``
     - ``betalens-db-manager --env <private-env> doctor``
   * - 启动 Infra 网页
     - ``betalens-db-manager --env C:\private\infra.env serve``
     - ``betalens-db-manager --env /private/path/infra.env serve``
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

当前 PostgreSQL 用户没有 ``CREATEDB`` 权限。请管理员创建空库，随后 Infra 工具仍可执行结构初始化。

``No module named ...``
~~~~~~~~~~~~~~~~~~~~~~~

通常是虚拟环境未激活，或安装命令用了另一套 Python。检查
``python -c "import sys; print(sys.executable)"``，再在正确环境执行
``python -m pip install -e ".[full]"``。始终使用 ``python -m pip``，不要混用裸 ``pip``。

``npm install`` 或 Vite 启动失败
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

先确认 ``node --version`` 至少为 20.19。删除或重装前端依赖前，先记录完整错误；常见原因是
Node.js 太旧、代理不可用或安全软件锁定 ``node_modules``。核心 Python 功能不依赖 Node.js。

查询提示表不存在
~~~~~~~~~~~~~~~~

Python 已连接到 PostgreSQL，但目标库尚未初始化，或配置指向了错误数据库。确认数据库名为
管理连接是否为 ``betalens``，再查看表定义和结构预览；旧 Datafeed 查询还需确认其研究合同。
当前组件没有 ``inspect`` 子命令，不能用旧文档命令代替结构检查。

查询返回空表
~~~~~~~~~~~~

schema 已安装但没有导入数据，或者代码、日期、指标名与数据库不一致。这不是安装失败。
先用数据库管理网页查看业务日期、可信发布范围和行数，再核对研究合同、metric 和代码格式。

卸载或重建虚拟环境
~~~~~~~~~~~~~~~~~~

虚拟环境只包含可重新安装的 Python 包。退出正在使用它的程序并执行 ``deactivate`` 后，可以
删除仓库内的 ``.venv`` 再按第 4、5 步重建。**这不会删除 PostgreSQL 数据库**；数据库删除
是独立且不可逆的操作，本教程不提供自动删除命令。

下一步
------

安装和数据准备完成后，进入 :doc:`quickstart` 跑通“取数 → 分组 → 权重 → 回测 → 报告”。
模块职责、真实 API 和进阶流程可从文档首页按 Datafeed、Factor、Backtest、Analyst、
Infra 工具 和 Dashboard 继续阅读。
