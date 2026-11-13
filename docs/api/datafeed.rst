Datafeed API
============

核心包：:mod:`betalens.datafeed`

业务接口使用当前宽表和可信参考版本表。PG/CH 共享返回合同；
可得时间、完整快照证据、价格口径及待迁移 API 见 :doc:`../guide/datafeed`。

``Datafeed(table_name, ..., backend=..., config_file=..., pit_policy=...)`` 直接使用物理表名。
``metric`` 必须是物理数值字段；无 registry、逻辑数据集和指标别名。
价格口径由表名决定，不再接受 ``adjustment``。默认 research，可显式指定 strict。
日历新增 table_name，默认 trade_calendar；strict 须显式使用 trade_calendar_versions。
query_trade_risks 分别返回上市、ST、停牌的可空标志及假设说明，不决定回测成交。
``query_trade_status_matrix(codes, dates)`` 在停牌参考表上返回可空 Int8 状态矩阵，
兼容逐日及压缩日期段凭证，并在重新核验版本后复用订阅内缓存。
范围、日期、名称和交易日接口支持 ``as_of``。
``query_daily_market(codes=None, start_date=None, end_date=None, metrics=("close",), *, as_of=None, filters=None)``
返回以 ``trade_date`` 为业务日期的日频长表，并保留 ``available_at``、资产类型和质量字段。
支持股票、基金和指数，供因子、基准与事件研究使用；股票回测执行仍使用配对价格接口。
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


status
~~~~~~

.. automodule:: betalens.datafeed.status
   :members:

daily
~~~~~

.. automodule:: betalens.datafeed.daily
   :members:
