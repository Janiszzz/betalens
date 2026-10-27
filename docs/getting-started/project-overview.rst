项目全景
========

Betalens 不是单一的回测类，而是一套以 PostgreSQL 为数据底座、以 Python API 为研究核心、
以 YAML 因子目录和浏览器 Dashboard 为工作入口的量化研究框架。本页说明各部分负责什么、
数据如何流动，以及遇到问题时应该先看哪里。

整体数据流
----------

.. code-block:: text

   Tushare / 其他 API / 显式合同文件
                    |
                    v
      betalens-db-manager（独立 Infra）
                    |
                    v
    PostgreSQL betalens + ClickHouse betalens
       统一表 + 来源/版本/可信发布审计
                    |
          研究层显式适配数据集合同
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
   * - ``betalens-db-manager``
     - 建库建表、来源映射、审计、任务、PG/CH 独立提交、参考域可信发布与完整性检查
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

当前管理器使用 PostgreSQL ``betalens`` 库的 ``betalens`` schema，
以及 ClickHouse ``betalens`` 库。两库是同等目标，按任务分别选择、核验与提交。

* 统一宽表包括证券档案、原始行情、后复权行情、财报、行业及事件明细。
* 股票历史行业、交易日历、股票停牌区间新增三个版本表，只追加，不覆盖旧版本。
* 三域可见性由目标库的修订、范围和不可变发布凭证共同决定，不以任务总状态放行。
* SQLite 保存本地编排、执行尝试、日志和索引，YAML 保存通用表定义及来源映射。
* 不自动建立 ``daily_market/fundamentals/industry/trade_status`` 等旧研究兼容视图。

研究包中既有 ``Datafeed``、``BacktestBase`` 和旧 fact/dim 示例仍使用原查询合同，
不等于当前物理库已经提供该合同。管理器建表和导入成功后，研究查询适配是独立工作。
部署及完整表清单见 :doc:`../guide/local-databases`；接入流程见
:doc:`../guide/infra`；三域语义见 :doc:`../guide/db-manager-reference`。

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
   * - ``betalens-db-manager/``
     - 独立发行包、配置、数据合同、接入服务及网页
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
* 建库、导入和迁移：:doc:`../guide/infra`。
* 页面运行和下载：:doc:`../guide/dashboard`。
