因子模块
========

``betalens.factor`` 覆盖因子研究的中段：可交易池、因子预查询、预处理、分组打标签、权重生成、统计检验和参数挖掘。标准入口在 ``betalens.factor.factor``，预处理在 ``betalens.factor.preprocessing``，统计检验在 ``betalens.factor.stats``，因子体检在 ``betalens.factor.profiling``，参数挖掘在 ``betalens.factor.mining``。

可交易池
--------

.. code-block:: python

   from betalens.datafeed import get_absolute_trade_days
   from betalens.factor.factor import get_tradable_pool

   days = get_absolute_trade_days("2020-04-30", "2024-04-30", "Y")
   date_ranges, code_ranges = get_tradable_pool(days)

要点：

* 基于 ``trade_status`` 表，按调仓日还原 ``1`` 正常、``0`` 停牌、``-1`` 未上市。
* 默认 ``include_abnormal=False``，只保留正常交易证券。
* ``include_abnormal=True`` 会把异常状态也纳入候选，适合做停牌处理或审计实验。
* 返回的 ``date_ranges`` 和 ``code_ranges`` 可复用于多个因子的预查询，减少重复查库。

因子预查询
----------

.. code-block:: python

   from betalens.factor.factor import pre_query_characteristic_data

   raw = pre_query_characteristic_data(
       date_list=days,
       metric="股息率(报告期)",
       time_tolerance=24 * 2 * 365,
       table_name="fundamentals",
       date_ranges=date_ranges,
       code_ranges=code_ranges,
   )

``pre_query_characteristic_data`` 默认 ``table_name="fundamentals"``。``time_tolerance`` 单位是小时，默认 ``24*2*365``。返回列通常包含 ``input_ts``、``code``、``{metric}``、``datetime``、``diff_hours`` 和 ``name``。

预处理与中性化
--------------

.. code-block:: python

   from betalens.factor.preprocessing import preprocess_factor

   cleaned = preprocess_factor(
       raw,
       metric="股息率(报告期)",
       winsorize_method="mad",
       standardize_method="zscore",
       industry_scheme="申万一级行业",
   )

``preprocess_factor`` 逐截面执行去空值、去极值、标准化和可选中性化。行业标签可由 ``industry_scheme`` 自动按 PIT 口径查询 ``industry`` 表；市值中性化需调用方准备 log 市值列。

常用低层函数：

* ``winsorize_factor``：截面去极值，支持 ``mad``、``percentile``、``std``。
* ``standardize_factor``：支持 ``zscore``、``rank``、``minmax``。
* ``neutralize_factor``：行业哑变量和 log 市值的截面 OLS 残差。
* ``neutralize_factor_by_factor``：用一个因子解释另一个因子并取残差。
* ``query_industry_panel``：为 ``(input_ts, code)`` 面板取 PIT 行业标签。

分组打标签
----------

.. code-block:: python

   from betalens.factor.factor import (
       single_characteristic,
       double_characteristic,
       multi_characteristic,
   )

   labeled = single_characteristic(cleaned, "股息率(报告期)", {"股息率(报告期)": 10})

   double_labeled = double_characteristic(
       size_data,
       bm_data,
       metric1="市值",
       metric2="账面市值比",
       quantiles1={"市值": 5},
       quantiles2={"账面市值比": 5},
       sort_method="dependent",
   )

   multi_labeled = multi_characteristic(
       [size_data, bm_data, momentum_data],
       [
           {"name": "市值", "quantiles": 5, "method": "dependent"},
           {"name": "账面市值比", "quantiles": 5, "method": "dependent"},
           {"name": "动量", "quantiles": 3, "method": "independent"},
       ],
   )

真实函数名是 ``single_characteristic``、``double_characteristic``、``multi_characteristic``。标签列名为 ``{metric}_label``，输出索引为 ``(input_ts, code)``。

权重生成
--------

.. code-block:: python

   from betalens.factor.factor import (
       get_single_factor_weight,
       get_double_factor_weight,
       get_multi_factor_weight,
   )

   weights = get_single_factor_weight(labeled, {
       "factor_key": "股息率(报告期)",
       "mode": "freeplay",
       "long": [9],
       "short": [0],
       "group_weights": {},
       "intra_group_allocation": {},
   })
   weights["cash"] = 0

模式：

* ``classic-long-short``：自动做多最高组、做空最低组。
* ``freeplay``：显式指定 ``long`` / ``short`` 或组合列表。
* ``group_weights`` 和 ``intra_group_allocation`` 可细化组间权重和组内分配。

因子体检
--------

``factor.profiling`` 关注因子值本身，不依赖未来收益，适合在回测前发现异常。

.. code-block:: python

   from betalens.factor.profiling import (
       describe_distribution,
       coverage_stats,
       detect_outliers,
       factor_autocorrelation,
       factor_turnover,
       factor_profile_payload,
   )

   distribution = describe_distribution(raw, metric="股息率(报告期)")
   coverage = coverage_stats(raw, metric="股息率(报告期)")
   payload = factor_profile_payload(raw, metric="股息率(报告期)")

常见检查包括分布稳定性、覆盖率、异常值、因子自相关、选股重合度、因子间相关与聚类。

统计检验
--------

.. code-block:: python

   from betalens.factor.stats import (
       calc_ic,
       calc_icir,
       summarize_ic,
       fama_macbeth,
       group_return_summary,
       run_timing_evaluation,
       run_cross_section_evaluation,
   )

   ic = calc_ic(factor_wide, return_wide, method="spearman")
   print(summarize_ic(ic))

``factor.stats`` 覆盖 IC/ICIR、Fama-MacBeth、分组收益、择时信号评估、滚动 IC、胜率、回归稳健性、截面综合评价和报告导出。绘图函数返回 matplotlib 图像，可用于报告或 Dashboard。

参数挖掘
--------

通用参数挖掘放在 ``betalens.factor.mining``，因子脚本只提供薄 hook。详见 :doc:`factor-mining`。

.. code-block:: python

   from betalens.factor.mining import ParameterSweepConfig, run_parameter_sweep

   config = ParameterSweepConfig(
       factor_module="factor_DISP",
       output_dir="outputs/mining",
       span=("2020-01-01", "2024-12-31"),
       grid={"window": [60, 120, 252]},
       engine="vector",
       workers=1,
   )
   result = run_parameter_sweep(config)

实践提示
--------

* metric 名必须与数据库 ``metric`` 列一致。
* 查询多个因子时先取一次 ``get_tradable_pool``，复用 ``date_ranges`` / ``code_ranges``。
* 极端值或覆盖率差的因子先做 ``preprocess_factor`` 或 profiling。
* 回测前补 ``cash`` 列，并在回测后检查 ``engine.actual_weight``。

更多函数级文档见 :doc:`../api/factor`。
