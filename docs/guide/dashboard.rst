Dashboard
=========

``dashboard/`` 是 Betalens 的浏览器界面：FastAPI 后端 + React/Vite 前端。它不替代底层 Python API，而是把 ``betalens-factor/`` 中的因子 YAML 暴露为可配置、可运行、可下载的研究工作台。

启动
----

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
4. ``POST /api/runs`` 创建后台任务，后端写出本次 ``outputs/runs/<run_id>/run_config.yaml``。
5. 后台线程调用 ``FactorPipeline(...).run(...)``，驱动 datafeed、factor、backtest、analyst。
6. 前端通过 ``GET /api/runs/{id}`` 轮询状态，通过 ``GET /api/runs/{id}/logs`` 接收 SSE 日志。
7. 完成后 ``GET /api/runs/{id}/result`` 读取指标、图表和表元数据。

收益概述图形
------------

每次回测完成后，结果页的“收益概述”（择时策略为“择时概览”）会自动展示与因子脚本静态图同口径的交互图，无需切换到单独的图形标签页：

* 截面因子：各分组累计收益曲线。
* 择时策略：净值、标的价格和仓位综合图；实际调仓日志中的买卖点标在标的价格曲线上。
* 全部策略：按平仓年份统计的单笔平均收益、胜率和交易次数。
* 截面因子 Profiling：分布、CDF、覆盖率、极值、自相关、换手和分布漂移。

图表数据直接来自本次 ``RunResult`` / ``BacktestBase``，不读取或转换落盘 PNG。

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
