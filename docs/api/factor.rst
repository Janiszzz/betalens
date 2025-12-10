Factor API
==========

.. automodule:: betalens.factor
   :members:
   :undoc-members:
   :show-inheritance:

factor.factor
-------------

核心因子处理模块，提供可交易池获取、因子预查询、分组打标签、权重生成等功能。

数据准备函数
~~~~~~~~~~~~

.. py:function:: get_tradable_pool(date_list)

   获取可交易股票池。对每个调仓日查询交易状态为1的证券。

   :param date_list: 日期列表
   :return: (date_ranges, code_ranges) 元组

.. py:function:: pre_query_factor_data(date_list, metric, time_tolerance=17520, table_name="fundamental_data", date_ranges=None, code_ranges=None)

   批量预查询因子数据，生成符合因子排序函数要求的DataFrame。

   :param date_list: 调仓日期列表
   :param metric: 因子指标名称
   :param time_tolerance: 时间容差（小时），默认2年
   :param table_name: 数据库表名
   :param date_ranges: 可选，复用的日期范围
   :param code_ranges: 可选，复用的代码范围
   :return: DataFrame，包含 input_ts, code, metric, datetime, diff_hours 等列

单因子函数
~~~~~~~~~~

.. py:function:: single_factor(pre_queried_data, metric, quantiles)

   单因子分组打标签。

   :param pre_queried_data: 预查询的因子数据DataFrame
   :param metric: 因子指标名称
   :param quantiles: 分位数字典，如 {"ROE": 10}
   :return: 带标签的DataFrame，索引为(input_ts, code)

.. py:function:: get_single_factor_weight(labeled_pool, params)

   根据单因子标签生成多空权重。

   :param labeled_pool: 带标签的因子池
   :param params: 参数字典，包含 factor_key, mode, long, short 等
   :return: 权重DataFrame

.. py:function:: describe_labeled_pool(labeled_pool)

   描述打标签后的因子池统计信息。

   :return: 透视表，包含每个标签组的样本数和均值

双因子函数
~~~~~~~~~~

.. py:function:: double_factor(pre_queried_data1, pre_queried_data2, metric1, metric2, quantiles1, quantiles2, sort_method='dependent')

   双因子分组打标签（Double Sort）。

   :param pre_queried_data1: 主因子数据
   :param pre_queried_data2: 次因子数据
   :param metric1: 主因子名称
   :param metric2: 次因子名称
   :param quantiles1: 主因子分组数
   :param quantiles2: 次因子分组数
   :param sort_method: 'independent' 或 'dependent'
   :return: 带双标签的DataFrame

.. py:function:: get_double_factor_weight(labeled_pool, params)

   根据双因子标签生成多空权重。

   :param labeled_pool: 带双标签的因子池
   :param params: 参数字典，包含 factor_key1, factor_key2, mode, long_combinations, short_combinations 等
   :return: 权重DataFrame

.. py:function:: describe_double_labeled_pool(labeled_pool)

   描述双因子打标签后的统计信息。

   :return: (count_pivot, mean_pivot1, mean_pivot2) 元组

多因子函数
~~~~~~~~~~

.. py:function:: multi_factor(pre_queried_data_list, factors)

   多因子分组打标签（Multi-Factor Sort）。

   :param pre_queried_data_list: DataFrame列表
   :param factors: 因子配置列表，每个元素包含 name, quantiles, method
   :return: 带多标签的DataFrame

.. py:function:: get_multi_factor_weight(labeled_pool, params)

   根据多因子标签生成多空权重。

   :param labeled_pool: 带多标签的因子池
   :param params: 参数字典，包含 mode, long_combinations, short_combinations 等
   :return: 权重DataFrame

.. py:function:: describe_multi_labeled_pool(labeled_pool, max_display_dims=2)

   描述多因子打标签后的统计信息。

   :param max_display_dims: 最大显示维度
   :return: 统计信息字典


