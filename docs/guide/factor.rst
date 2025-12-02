因子模块
========

`betalens.factor` 集中封装了从可交易池筛选到分组打分、权重生成的常用步骤。

获取可交易池
------------

.. code-block:: python

   from betalens.factor import get_tradable_pool
   from betalens.datafeed import get_absolute_trade_days

   rebalance_days = get_absolute_trade_days("2022-01-01", "2022-12-31", "M")
   date_ranges, code_ranges = get_tradable_pool(rebalance_days)

逻辑要点

- 对每个调仓日调用 :class:`betalens.datafeed.Datafeed`，过滤交易状态 ``value == 1`` 的证券
- 返回 ``date_ranges``（日期列表）与 ``code_ranges``（与日期一一对应的代码列表）

单因子打分
----------

.. code-block:: python

   from betalens.factor import single_factor

   labeled_pool = single_factor(
       date_ranges=date_ranges,
       code_ranges=code_ranges,
       metric="ROE",
       quantiles={"ROE": 10},
   )

特点：

- 支持自定义 ``quantiles``（整数代表等分桶，列表可定义任意切分点）
- 自动 groupby ``input_ts`` 并添加 ``metric_label`` 列
- 以 ``input_ts`` 和 ``code`` 为 MultiIndex，便于后续透视

生成权重矩阵
------------

.. code-block:: python

   from betalens.factor import get_single_factor_weight

   params = {
       "factor_key": "ROE",
       "mode": "classic-long-short",
       "long": [9],
       "short": [0],
   }
   weights = get_single_factor_weight(labeled_pool, params)
   weights["cash"] = 0
   weights.index.name = "input_ts"

模式说明

- ``classic-long-short``：自动选择最高/最低标签
- ``freeplay``：显式指定 ``long`` 与 ``short`` 标签列表

函数内部会对每个调仓日分别正负归一，确保多头权重合计 1、空头 -1。

描述性统计
----------

.. code-block:: python

   from betalens.factor import describe_labeled_pool
   summary = describe_labeled_pool(labeled_pool)
   print(summary)

输出同时包含 ``count`` 与 ``mean`` 透视结果，可快速检查分组是否均衡。

实践提示
--------

- 若指标存在极端值，建议在调用 ``single_factor`` 之前进行 winsor/标准化处理。
- ``get_tradable_pool`` 默认查询 ``fundamental_data`` 表，可酌情改写以适配自建数据库。
- 对多因子策略，可用 ``single_factor`` 多次生成标签后合并，再自定义 ``get_single_factor_weight`` 逻辑。

更多函数级文档见 :doc:`../api/factor`。


