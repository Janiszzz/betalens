项目全景
========

Betalens 不是单一的回测类，而是一套以 PostgreSQL 为数据底座、以 Python API 为研究核心、
以 YAML 因子目录和浏览器 Dashboard 为工作入口的量化研究框架。本页说明各部分负责什么、
数据如何流动，以及遇到问题时应该先看哪里。

整体数据流
----------

.. code-block:: text

   CSV / Excel / Parquet / EDE / Wind
                    |
                    v
      betalens_db_manager（建库、迁移、校验、导入）
                    |
                    v
        PostgreSQL datafeed 数据库
       betalens schema + public 兼容视图
                    |
                    v
       betalens.datafeed（只读、PIT 查询）
                    |
          +---------+----------+
          |                    |
          v                    v
   betalens.factor       betalens.eventstudy
   因子值/分组/权重         事件窗口收益
          |
          v
   betalens.backtest
   持仓/成交/损益/净值
          |
          v
   betalens.analyst
   指标/归因/Excel/HTML
          |
          v
   Dashboard（发现因子、配置、运行、查看和下载）

模块职责
--------

.. list-table:: 核心模块与边界
   :header-rows: 1
   :widths: 24 43 33

   * - 模块
     - 负责
     - 不负责
   * - ``betalens_db_manager``
     - 建库、migration、schema 核验、文件预览与导入、冲突处理、GUI
     - 研究运行时的行情查询和回测
   * - ``betalens.datafeed``
     - PostgreSQL 只读查询、交易日、行业、指数成分、交易状态
     - 建表、迁移和写入数据
   * - ``betalens.factor``
     - 可交易池、因子分组、预处理、IC/回归、profiling、参数挖掘
     - 执行成交和维护账户净值
   * - ``betalens.backtest``
     - 把目标权重转换为整数手持仓，处理交易状态，计算损益和净值
     - 生成完整绩效报告
   * - ``betalens.analyst``
     - 收益风险指标、持仓与贡献分析、Excel 和交互 HTML 报告
     - 修改回测成交结果
   * - ``betalens.eventstudy``
     - 单/多标的事件窗口、超额收益和固定持有期分析
     - 横截面因子组合构建
   * - ``betalens.robust``
     - Lucky Factors 风格增量检验和 bootstrap
     - 日常因子回测主流程
   * - ``dashboard``
     - 因子发现、参数编辑、任务日志、结果图表、分页明细和下载
     - 另建一套因子或回测逻辑

数据库结构
----------

默认连接的 PostgreSQL 数据库名是 ``datafeed``。其内部 ``betalens`` schema 保存规范化的
物理表，``public`` schema 暴露研究代码使用的兼容视图。

* 维度表维护证券、名称、指标和行业定义。
* ``market_daily_fact`` 保存日频市场事实。
* ``observation_fact`` 保存带 ``available_at`` 的 PIT 观测，并按年度分区。
* 行业、指数成分和交易状态使用独立事实/事件表。
* ``daily_market``、``fundamentals``、``industry``、``trade_status`` 等兼容视图是
  Datafeed 和既有因子脚本使用的稳定逻辑名称。

数据库结构必须通过 ``betalens_db_manager`` 安装和升级，不建议手工执行或改写 migration。
完整物理表、导入契约和兼容视图见 :doc:`../guide/db-manager`。

一次标准研究如何运行
--------------------

1. ``get_absolute_trade_days`` 生成调仓日。
2. ``get_tradable_pool`` 根据 ``trade_status`` 构建可交易池。
3. ``pre_query_characteristic_data`` 以 PIT 口径批量查询因子输入。
4. ``single_characteristic``、``double_characteristic`` 或
   ``multi_characteristic`` 完成截面分组。
5. 权重函数生成目标权重矩阵，并显式包含 ``cash`` 列。
6. ``BacktestBase`` 按价格、交易状态和整数手约束产生实际持仓与净值。
7. ``Analyst.from_backtest`` 读取真实回测结果，生成指标、图表和报告。

真正参与净值计算的是 ``BacktestBase.actual_weight``。排查异常结果时，依次查看
``actual_weight``、``position``、``trade_status_matrix`` 和 ``rebalance_log``，不要只检查
输入的目标权重。

因子目录与 Dashboard
--------------------

``betalens-factor`` 不会随 Python wheel 作为包代码安装，它是仓库内的研究资产目录：

.. code-block:: text

   betalens-factor/
     <factor_class>/
       class_<factor_class>.yaml
       factor_template_<factor_class>.py
       <FACTOR_NAME>/
         factor_<FACTOR_NAME>.py
         factor_<FACTOR_NAME>.yaml
         outputs/
         mining/

Dashboard 扫描类级和因子级 YAML，动态导入对应脚本，并把每次页面参数写入
``outputs/runs/<run_id>/run_config.yaml``。因此复核一次运行时，``run_config.yaml`` 是最终
参数口径。因子脚本规范见 :doc:`../guide/factor-pipeline`，Dashboard 运行链路见
:doc:`../guide/dashboard`。

仓库目录
--------

.. list-table:: 目录用途
   :header-rows: 1
   :widths: 30 70

   * - 路径
     - 内容
   * - ``betalens/``
     - 可安装的主 Python 包
   * - ``betalens_db_manager/``
     - 可安装的数据库管理包、migration、导入适配器和 GUI
   * - ``betalens-factor/``
     - 因子脚本、YAML、回测产物和参数挖掘入口
   * - ``dashboard/backend/``
     - FastAPI、任务队列、序列化和下载接口
   * - ``dashboard/frontend/``
     - React/Vite 单页应用
   * - ``docs/``
     - Sphinx 文档源文件
   * - ``tests/``
     - 核心、数据库、因子配置和 Dashboard 后端测试
   * - ``pyproject.toml``
     - 包版本、Python 要求、核心依赖和 extras 的唯一打包口径
   * - ``requirements.txt``
     - 完整本地开发环境依赖；普通用户优先使用 ``.[full]``

文档导航
--------

* 第一次部署：:doc:`installation`。
* 跑通标准研究：:doc:`quickstart`。
* 查询或配置问题：:doc:`../guide/datafeed`。
* 因子、分组和统计：:doc:`../guide/factor`。
* 成交、停牌和持仓：:doc:`../guide/backtest`。
* 指标和报告：:doc:`../guide/analyst`。
* 建库、导入和迁移：:doc:`../guide/db-manager`。
* 页面运行和下载：:doc:`../guide/dashboard`。

