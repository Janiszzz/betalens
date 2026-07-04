Factor API
==========

核心包：:mod:`betalens.factor`

.. automodule:: betalens.factor
   :members:
   :undoc-members:
   :show-inheritance:

factor.factor
-------------

可交易池、因子预查询、单/双/多因子分组与权重生成。

.. automodule:: betalens.factor.factor
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

重要口径
~~~~~~~~

* ``get_tradable_pool(date_list, include_abnormal=False)`` 基于 ``trade_status`` 表筛选可交易证券。
* ``pre_query_characteristic_data`` 默认 ``table_name="fundamentals"``，``time_tolerance`` 单位为小时。
* 分组函数真实名称为 ``single_characteristic``、``double_characteristic``、``multi_characteristic``。
* 权重生成后建议显式补 ``cash`` 列，再传给 ``BacktestBase``。

factor.preprocessing
--------------------

去极值、标准化、中性化、行业约束。

.. automodule:: betalens.factor.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

factor.profiling
----------------

因子值体检：分布、覆盖率、异常值、时序稳定性、换手、重合度、相关性与聚类。

.. automodule:: betalens.factor.profiling
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

factor.stats
------------

IC/ICIR、Fama-MacBeth、分组收益、择时评估、截面评价和图表。

.. automodule:: betalens.factor.stats
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

factor.mining
-------------

参数扫描和滚动窗口挖掘。

.. automodule:: betalens.factor.mining
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

factor.config
-------------

因子 YAML 读取、校验、路径解析和 run 参数解析。

.. automodule:: betalens.factor.config
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
