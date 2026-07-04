因子参数挖掘
============

``betalens.factor.mining`` 提供通用参数挖掘框架，适合把因子的 ``window``、分组数、择时参数等做系统化搜索。它复用标准流水线：缓存数据、计算因子、PIT 股票池过滤、分组、权重、回测、指标排序。

因子侧 hook
-----------

参与挖掘的因子脚本通常提供：

.. code-block:: python

   def make_mining_spec(params):
       return FactorSpec(
           name="DISP",
           inputs={"close_wide": "收盘价(元)"},
           compute=compute_disp,
           compute_kwargs={"window": int(params["window"])},
           direction="negative",
           table_name="daily_market",
           index_code="000906.SH",
       )

   def mining_gid(params):
       return f"window={params['window']}"

   def mining_warmup_days(params):
       return int(params["window"])

   def mining_weight_hook(weights, task):
       return weights

   def mining_valid_report(params, rank, output_dir, start_date, end_date):
       return None

必需的是 ``make_mining_spec``；其他 hook 只有在需要稳定命名、预热期、权重过滤或验证报告时提供。

单段参数扫描
------------

.. code-block:: python

   from betalens.factor.mining import ParameterSweepConfig, run_parameter_sweep

   config = ParameterSweepConfig(
       factor_module="factor_DISP",
       output_dir="outputs/mining/sweep_window",
       span=("2020-01-01", "2024-12-31"),
       grid={"window": [60, 120, 252]},
       spec_factory="make_mining_spec",
       gid_factory="mining_gid",
       objective="sharpe",
       engine="vector",
       workers=1,
       rebuild_cache=False,
   )
   df = run_parameter_sweep(config)

``run_parameter_sweep`` 输出结果表，默认写 ``sweep_results.csv``。单参数粗筛建议先用 ``engine="vector"``，候选参数再用 ``engine="exact"`` 复核。

滚动窗口挖掘
------------

.. code-block:: python

   from betalens.factor.mining import RollingMiningConfig, run_walk_forward

   config = RollingMiningConfig(
       factor_module="factor_DISP",
       output_dir="outputs/mining/walkforward",
       grid={"window": [60, 120, 252]},
       train=("2018-01-01", "2021-12-31"),
       test=("2022-01-01", "2023-12-31"),
       valid=("2024-01-01", "2024-12-31"),
       train_schemes=[(252, 63), (504, 126)],
       test_schemes=[(252, 63)],
       objective="sharpe",
       engine="exact",
   )
   payload = run_walk_forward(config)

滚动流程：

* TRAIN：全网格滚动评估并统计冠军。
* TEST：只复测 TRAIN 入围候选。
* VALID：对 TEST 前几名做整段验证，可调用 ``valid_report_hook``。

缓存
----

挖掘会把宽表数据、PIT 股票池和元数据写入 cache 目录。默认 cache 在 ``output_dir`` 下，也可用 ``cache_dir`` 指定。

推荐忽略：

.. code-block:: text

   **/_sweep/_cache/
   **/_walkforward/_cache/

已有 `.gitignore` 包含这些目录。若历史缓存已被跟踪，需单独从索引移除。

引擎选择
--------

* ``vector``：用价格宽表快速近似回测，适合粗筛。
* ``exact``：调用 ``BacktestBase``，整数手、交易状态和资金递推口径更准，速度更慢。

内存保护
--------

多进程挖掘默认 ``max_memory_ratio=0.5``，会按缓存体量估算 worker 数。若单 worker 超出预算会快速失败。参数含义：

* ``workers``：目标并行数。
* ``max_memory_ratio``：可用内存占比上限。
* ``max_memory_bytes``：显式内存上限，优先级高于比例。
* ``cache_memory_multiplier``：缓存数据在 worker 内的放大系数。
* ``worker_memory_overhead_bytes``：单 worker 固定额外开销。

结果指标
--------

``metrics_from_nav`` 默认输出 ``ann_ret``、``sharpe``、``mdd`` 等指标。``objective`` 指定排序字段，``objective_higher_is_better`` 控制升降序。

实践建议
--------

* 小样本先跑 smoke 网格，确认 hook、数据和输出目录。
* 滚动窗口方案用 ``window_lengths × steps`` 生成，再过滤 ``step <= window_length``。
* 先用 ``vector`` 找候选，再用 ``exact`` 对前几名复核。
* 运行日志保留配置、参数组合、任务 START 行和进度条，便于恢复和审计。
