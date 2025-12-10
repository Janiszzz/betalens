因子模块
========

`betalens.factor` 集中封装了从可交易池筛选到分组打分、权重生成的常用步骤，支持单因子、双因子和多因子分组策略。

获取可交易池
------------

.. code-block:: python

   from betalens.factor.factor import get_tradable_pool
   from betalens.datafeed import get_absolute_trade_days

   rebalance_days = get_absolute_trade_days("2022-01-01", "2022-12-31", "M")
   date_ranges, code_ranges = get_tradable_pool(rebalance_days)

逻辑要点：

- 对每个调仓日调用 :class:`betalens.datafeed.Datafeed`，过滤交易状态 ``value == 1`` 的证券
- 返回 ``date_ranges``（日期列表）与 ``code_ranges``（与日期一一对应的代码列表）

批量预查询因子数据
------------------

.. code-block:: python

   from betalens.factor.factor import pre_query_factor_data

   pre_queried_data = pre_query_factor_data(
       date_list=rebalance_days,
       metric="股息率(报告期)",
       time_tolerance=24*2*365,  # 时间容差（小时）
       table_name="fundamental_data",
       date_ranges=date_ranges,  # 可选：复用可交易池
       code_ranges=code_ranges
   )

返回的 DataFrame 包含：

- ``input_ts``: 输入时间戳（调仓日期）
- ``code``: 股票代码
- ``{metric}``: 因子值列
- ``datetime``: 匹配到的数据时间戳
- ``diff_hours``: 时间差（小时）
- ``name``: 股票名称

单因子分组
----------

.. code-block:: python

   from betalens.factor.factor import single_factor

   labeled_pool = single_factor(
       pre_queried_data=pre_queried_data,
       metric="ROE",
       quantiles={"ROE": 10}
   )

特点：

- 支持自定义 ``quantiles``（整数代表等分桶）
- 自动 groupby ``input_ts`` 并添加 ``metric_label`` 列
- 以 ``input_ts`` 和 ``code`` 为 MultiIndex，便于后续透视

生成单因子权重
--------------

.. code-block:: python

   from betalens.factor.factor import get_single_factor_weight

   # 经典多空模式
   params = {
       "factor_key": "ROE",
       "mode": "classic-long-short",
   }
   weights = get_single_factor_weight(labeled_pool, params)

   # 自定义模式
   params = {
       "factor_key": "ROE",
       "mode": "freeplay",
       "long": [9],
       "short": [0],
       "group_weights": {  # 可选：组间权重
           "long": {9: 2, 8: 1},
           "short": {0: 2, 1: 1}
       },
       "intra_group_allocation": {  # 可选：组内分配方式
           "long": {9: {"method": "factor_value", "metric": "ROE", "order": "desc"}},
           "short": {0: {"method": "equal"}}
       }
   }
   weights = get_single_factor_weight(labeled_pool, params)
   weights["cash"] = 0

模式说明：

- ``classic-long-short``：自动选择最高/最低标签
- ``freeplay``：显式指定 ``long`` 与 ``short`` 标签列表，支持组间权重和组内分配

双因子分组（Double Sort）
-------------------------

双重排序用于同时考虑两个因子对投资组合的影响，分析因子间交互作用。

两种排序方法：

1. **独立排序（Independent Sort）**：
   - 对所有股票分别按两个因子独立分组
   - 取两个标签的交集形成N×M个组合
   - 适用场景：因子之间相关性较低时

2. **条件排序（Dependent Sort）**：
   - 先按主因子分组，然后在每个主因子组内按次因子分组
   - 形成N×M个投资组合，每组股票数量相对均匀
   - 适用场景：因子相关性较高，或需要控制某个因子影响时

.. code-block:: python

   from betalens.factor.factor import double_factor, describe_double_labeled_pool

   labeled_pool = double_factor(
       pre_queried_data1=data1,
       pre_queried_data2=data2,
       metric1="市值",
       metric2="账面市值比",
       quantiles1={"市值": 5},
       quantiles2={"账面市值比": 5},
       sort_method='dependent'  # 或 'independent'
   )

   # 查看分组统计
   count_pivot, mean_pivot1, mean_pivot2 = describe_double_labeled_pool(labeled_pool)

生成双因子权重
--------------

.. code-block:: python

   from betalens.factor.factor import get_double_factor_weight

   weights = get_double_factor_weight(
       labeled_pool,
       params={
           "factor_key1": "市值",
           "factor_key2": "账面市值比",
           "mode": "classic-long-short",  # 自动做多(max,max)，做空(min,min)
       }
   )

   # 或自定义组合
   weights = get_double_factor_weight(
       labeled_pool,
       params={
           "factor_key1": "市值",
           "factor_key2": "账面市值比",
           "mode": "freeplay",
           "long_combinations": [(0, 4), (1, 4)],
           "short_combinations": [(4, 0), (4, 1)],
       }
   )

多因子分组（Multi-Factor Sort）
-------------------------------

支持递归的独立排序和条件排序混合。

.. code-block:: python

   from betalens.factor.factor import multi_factor, describe_multi_labeled_pool

   factors = [
       {'name': '市值', 'quantiles': 5, 'method': 'dependent'},
       {'name': '账面市值比', 'quantiles': 5, 'method': 'dependent'},
       {'name': '动量', 'quantiles': 3, 'method': 'independent'},
   ]

   labeled_pool = multi_factor(
       pre_queried_data_list=[data1, data2, data3],
       factors=factors
   )

   # 查看分组统计
   stats = describe_multi_labeled_pool(labeled_pool, max_display_dims=2)

生成多因子权重
--------------

.. code-block:: python

   from betalens.factor.factor import get_multi_factor_weight

   weights = get_multi_factor_weight(
       labeled_pool,
       params={
           "mode": "classic-long-short",  # 所有因子最高组 vs 所有因子最低组
       }
   )

   # 或自定义组合
   weights = get_multi_factor_weight(
       labeled_pool,
       params={
           "mode": "freeplay",
           "long_combinations": [(0, 4, 2), (1, 4, 2)],
           "short_combinations": [(4, 0, 0)],
       }
   )

描述性统计
----------

.. code-block:: python

   from betalens.factor.factor import (
       describe_labeled_pool,
       describe_double_labeled_pool,
       describe_multi_labeled_pool
   )

   # 单因子
   summary = describe_labeled_pool(labeled_pool)

   # 双因子
   count_pivot, mean_pivot1, mean_pivot2 = describe_double_labeled_pool(labeled_pool)

   # 多因子
   stats = describe_multi_labeled_pool(labeled_pool)

输出同时包含 ``count`` 与 ``mean`` 透视结果，可快速检查分组是否均衡。

实践提示
--------

- 若指标存在极端值，建议在调用分组函数之前进行 winsor/标准化处理。
- ``get_tradable_pool`` 默认查询 ``fundamental_data`` 表，可酌情改写以适配自建数据库。
- 复用可交易池：当查询多个因子时，先调用一次 ``get_tradable_pool``，然后传入 ``date_ranges`` 和 ``code_ranges`` 复用。
- 对于 ``freeplay`` 模式，可通过 ``group_weights`` 和 ``intra_group_allocation`` 实现更精细的权重控制。

更多函数级文档见 :doc:`../api/factor`。


