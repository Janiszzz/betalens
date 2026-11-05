Datafeed API
============

核心包：:mod:`betalens.datafeed`

业务接口使用当前宽表和可信参考版本表。PG/CH 共享返回合同；
可得时间、完整快照证据、价格口径及待迁移 API 见 :doc:`../guide/datafeed`。

``Datafeed(table_name, ..., backend=..., config_file=...)`` 直接使用物理表名。
``metric`` 必须是物理数值字段；无 registry、逻辑数据集和指标别名。
价格口径由表名决定，不再接受 ``adjustment``。
范围、日期、名称和交易日接口支持 ``as_of``。
时序和日期接口支持 ``filters={字段: 标量值}``；四种 nearest 在参数字典中传 ``filters``。
财报必须显式指定 ``statement_type/report_type``；日历默认 ``exchange="SSE"``，无 SHSE 转换。
``run_query/cursor/conn``、旧 SQL 构造函数和 ConfigManager 已移除。
独立查询函数仅接受 Datafeed 订阅实例；行业使用 ``scheme_code/level``。
半年度为 ``H``；偏移接口为 ``trade_periods_offset``，offset 表示交易周期数。

.. automodule:: betalens.datafeed
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

子模块
------

core
~~~~

.. automodule:: betalens.datafeed.core
   :members:
   :undoc-members:
   :show-inheritance:

validation
~~~~~~~~~~

.. automodule:: betalens.datafeed.validation
   :members:
   :undoc-members:
   :show-inheritance:

query
~~~~~

.. automodule:: betalens.datafeed.query
   :members:
   :undoc-members:
   :show-inheritance:

industry
~~~~~~~~

.. automodule:: betalens.datafeed.industry
   :members:
   :undoc-members:
   :show-inheritance:

universe
~~~~~~~~

.. automodule:: betalens.datafeed.universe
   :members:
   :undoc-members:
   :show-inheritance:

config
~~~~~~

.. automodule:: betalens.datafeed.config
   :members:
   :undoc-members:
   :show-inheritance:

错误类型
--------

.. automodule:: betalens.datafeed.contracts
   :members: DatafeedError, SchemaError, PITUnavailableError, VersionConflictError

