Dashboard
=========

``dashboard/`` 是 Betalens 的浏览器界面：FastAPI 后端 + React/Vite 前端。它不替代底层 Python API，而是把 ``betalens-factor/`` 中的因子 YAML 暴露为可配置、可运行、可下载的研究工作台。

启动
----

前端需要 Node.js 20.19+ 和 npm，后端需要安装 ``.[dashboard,viz]``；完整安装可直接使用
``python -m pip install -e ".[full]"``。第一次配置环境请先阅读
:doc:`../getting-started/installation`。

一键启动前后端：

.. code-block:: powershell

   .\dashboard\run.bat

或分别启动：

.. code-block:: powershell

   .\dashboard\run_backend.bat
   .\dashboard\run_frontend.bat

默认地址：

* 前端：``http://127.0.0.1:5173``
* 后端：``http://127.0.0.1:8000``
* Swagger：``http://127.0.0.1:8000/docs``

运行流程
--------

1. 前端请求 ``GET /api/factors``。
2. 后端扫描 ``betalens-factor/<class>/class_<class>.yaml`` 和 ``<class>/<name>/factor_<name>.yaml``。
3. 用户选择因子、调整 run 参数和 ``compute_kwargs``。
4. ``POST /api/runs`` 创建后台任务，后端写出本次仓库级 ``outputs/runs/<class>/<factor>/<run_id>/run_config.yaml``。
5. 后台线程调用 ``FactorPipeline(...).run(...)``，驱动 datafeed、factor、backtest、analyst。
6. 前端通过 ``GET /api/runs/{id}`` 轮询状态，通过 ``GET /api/runs/{id}/logs`` 接收 SSE 日志。
7. 完成后 ``GET /api/runs/{id}/result`` 读取指标、图表和表元数据。

收益概述图形
------------

每次回测完成后，结果页的“收益概述”（择时策略为“择时概览”）会自动展示与因子脚本静态图同口径的交互图，无需切换到单独的图形标签页：

* 截面因子：各分组累计收益曲线，以及按分组对比回测末期累计收益的直方图。
* 择时策略：净值、标的价格和仓位综合图；实际调仓日志中的买卖点标在标的价格曲线上。
* 全部策略：按平仓年份统计的单笔平均收益、胜率和交易次数。
* 截面因子 Profiling：分布、CDF、覆盖率、极值、自相关、换手和分布漂移。

图表数据直接来自本次 ``RunResult`` / ``BacktestBase``，不读取或转换落盘 PNG。

与离线实验看板共用展示
----------------------

``experiment-dashboard/`` 的因子首页使用指标表，支持 IC、Rank IC、ICIR、年化收益、
夏普、回撤、换手等列的筛选、排序、选列及本页 CSV 导出。
默认每行采用最新完整实验；没有完整快照时采用最新指标记录。可切换为最新入库记录。
来源实验 ID、回测区间和窗口聚合/整段指标口径均保留显示，缺失指标显示为空。
频率和股票池条件先筛实验，再选择该范围内的最新记录。

点击因子进入详情后，通过“历次实验”切换已入库记录；超过200条可加载更早记录。
窗口独立回测通过“结果范围”切换，不混用不同窗口的净值。仅指标候选保留实际可用的指标，
缺失的原始数据不从行情库补取。

在线与离线结果共用 ``dashboard/frontend/src/App.tsx`` 中的 ``BacktestResultView``、
``ResultNavigation`` 及原 Dashboard 样式。离线容器隔离页面样式，并通过
``BacktestResultClient`` 注入只读的归档数据接口；新增结果控件应继续通过该接口读取数据。
共用的 Python 转换逻辑位于 ``betalens/analyst/presentation.py``，旧的
``dashboard/backend/serialization.py`` 保留兼容导入。修改结果数据结构时维护这个共用模块，
并保留归档名称映射上下文，避免离线展示查询行情数据库。

在 ``dashboard/frontend/`` 执行 ``npm run build`` 会依次构建原 Dashboard 和离线看板，
将离线资源写入 Python 包。两个前端首次均需执行 ``npm ci``。
只构建在线前端可使用 ``npm run build:local``；打包发布前使用常规 ``build`` 同步两份资源。

离线 HTTP 新增 ``GET /api/catalog``、``GET /api/factors/{id}/history``，以及
``/api/experiments/{id}/result``、``profiling``、``result-table/{kind}``、
``result-download/{kind}``，全部为读取接口。

主要 API
--------

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - 方法
     - 路径
     - 用途
   * - GET
     - ``/api/health``
     - 健康检查。
   * - GET
     - ``/api/factors``
     - 发现因子，``refresh=true`` 可清缓存。
   * - GET
     - ``/api/factors/{class}/{name}``
     - 查看因子详情和脚本文档。
   * - POST
     - ``/api/runs``
     - 创建一次回测运行。
   * - GET
     - ``/api/runs/{id}``
     - 查询运行状态。
   * - GET
     - ``/api/runs/{id}/logs``
     - SSE 增量日志。
   * - GET
     - ``/api/runs/{id}/result``
     - 指标、图表、下载状态。
   * - GET
     - ``/api/runs/{id}/table/{kind}``
     - 分页读取 trades 或 positions。
   * - GET
     - ``/api/runs/{id}/download/{kind}``
     - 下载 dump、report、html、profiling。

结果与下载
----------

Dashboard 会把大表写入临时 parquet 后分页读取，避免一次性把交易或持仓明细塞进 JSON。常见下载物：

* ``dump``：``BacktestBase.dump_to_excel`` 的多 sheet 回测数据。
* ``report``：Analyst Excel 报告。
* ``html``：plotly 交互 HTML 报告。
* ``profiling``：因子体检输出。

新增因子
--------

Dashboard 不需要改代码即可发现新增因子。新增目录结构和脚本接口见 :doc:`factor-pipeline`。新增或修改 YAML 后，调用 ``GET /api/factors?refresh=true`` 或重启后端刷新缓存。

常见问题
--------

* 页面无因子：确认 ``betalens-factor/`` 下存在类级 YAML 和因子级 YAML。
* 参数改了不生效：刷新因子缓存；每次运行最终口径以 ``run_config.yaml`` 为准。
* 运行排队：后端默认单线程串行执行回测，避免多次回测抢数据库和内存。
* 下载还没出现：Excel dump 是后台线程异步落盘，稍后刷新运行结果。


保存到实验库
------------

截面因子和择时策略的参数面板均有“保存到实验库”复选框。库内默认关闭，
勾选结果写入有效 YAML 的 ``experiment_export.enabled``，不会修改源因子 YAML。
实验库位置由源配置的 ``experiment_export.store_path`` 指定，相对路径按仓库根解析。

开启后，在释放回测对象之前调用 ``ExperimentStore.save_run``；页面显示实验 ID。
运行状态的可选字段 ``experiment_export`` 返回 ``status/experiment_id/batch_id/error``。
入库失败时回测结果仍可查看，页面单独显示入库错误，运行目录保留入库回执。
关闭时不创建实验库。安装和离线查看方式见 :doc:`backtest`。

新报告统一位于仓库级 ``outputs/runs/<class>/<factor>/<run_id>/``，
临时分页表位于 ``outputs/cache/dashboard/<run_id>/``；清空任务只清理分页缓存，
不会删除实验库或报告。历史因子目录中的输出文件不自动搬迁。
