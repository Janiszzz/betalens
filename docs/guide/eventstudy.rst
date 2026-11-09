事件研究模块
============

事件研究（Event Study）模块用于分析特定事件前后证券的收益率表现。支持单标的、多标的、基准超额收益等多种模式。

模块概述
--------

``EventStudy`` 接收一个 ``Datafeed`` 实例，通过 ``query_daily_market()`` 获取价格数据，
然后围绕事件日期计算窗口期收益率，并提供统计分析和可视化功能。
日频行情使用 ``trade_date`` 作为观测日期，可得时间 ``available_at`` 仍由 Datafeed
按 research/strict 策略保留和筛选。该接口支持股票、基金和指数，不依赖股票手数规则。

事件窗口统一使用 ``get_absolute_trade_days(..., "D")`` 返回的标准交易日历。
因此相对日 ``-n ... n`` 始终表示交易日，而不是自然日；行情缺失会保留为空值，
不会把后一个交易日错误地前移。

**核心流程：**

1. 准备事件序列（Series，index 为 datetime，值 1 表示事件发生）
2. 调用 ``analyze()`` 方法分析事件窗口收益
3. 查看统计结果或绘制图表

初始化
------

.. code-block:: python

   from betalens import Datafeed, EventStudy

   df_market = Datafeed("market_daily_post_adjusted")
   es = EventStudy(df_market)

基础事件分析
------------

.. code-block:: python

   import pandas as pd

   # 准备事件序列：index为精确到秒的datetime，值1表示事件发生
   events = pd.Series(0, index=pd.date_range('2020-01-01', '2024-12-31'))
   events['2021-03-15 10:30:00'] = 1
   events['2022-06-20 14:00:00'] = 1

   # 分析事件前后各20个交易日的收益率
   result = es.analyze(
       events=events,
       code='868008.WI',
       window_before=20,
       window_after=20,
       metric='close'
   )

``analyze()`` 返回一个字典，包含：

- ``daily_stats``: 每日收益统计（均值、标准差、上涨概率、t 统计量等）
- ``holding_stats``: 固定持有收益统计
- ``event_count``: 有效事件数
- ``returns_matrix``: 完整收益矩阵（行=相对天数，列=事件编号）
- ``holding_returns_matrix``: 每次事件的固定持有收益矩阵
- ``price_matrix``: 每次事件的价格曲线（Day 0 归一化为 0）
- ``holding_coverage``: 各事件、各持有期的有效标的覆盖率
- ``return_model``: ``per_asset_interval_then_average``

``daily_stats`` 使用各交易日自身的收盘价涨跌幅。Day 0 不再表示从成本价开始的
下一段收益，而是 Day 0 交易日当日的涨跌幅。

Day 0 日期规则
~~~~~~~~~~~~~~

- 事件在 15:00 前发生 → 当天为 Day 0
- 事件在 15:00 或之后发生 → 下一标准交易日为 Day 0

带时区时间先转换为上海时间。使用 ``metric='open'`` 时，边界为 09:30；
其他价格字段通过 ``price_time_map={字段: 'HH:MM'}`` 指定边界。
默认交易所为 ``SSE``，兼容 ``SHSE`` / ``XSHG`` 别名。

持有收益
--------

``analyze()`` 始终计算固定持有期收益，不需要指定模式。默认计算持有 1 至 5 日，
以及 1、3、6、9、12 个月（每月按 21 个交易日）的收益：
每只标的先独立复合持有区间内的日收益，再在同一事件内对有效标的求均值。
例如两只股票两日分别累计上涨 21% 和 0%，共性持有收益为 10.5%。

.. code-block:: python

   result = es.analyze(
       events=events,
       code='868008.WI',
       window_before=20,
       window_after=60,
       holding_periods={'days': [1, 2, 3, 4, 5], 'months': [1, 3, 6]}
   )

持有起点偏移
~~~~~~~~~~~~

通过 ``holding_start_offset`` 调整持有起始日：

.. code-block:: python

   # 从 Day -3 开始计算持有收益（提前建仓）
   result = es.analyze(
       events=events,
       code='868008.WI',
       window_before=20,
       window_after=20,
       holding_start_offset=-3
   )

基准超额收益
------------

通过 ``benchmark_code`` 参数传入基准代码，自动计算 持有标的收益 - 基准收益：
日收益使用当日收益之差；持有收益使用各自区间累计收益之差；``price_matrix``
也使用标的归一化曲线减去基准归一化曲线。

.. code-block:: python

   result = es.analyze(
       events=events,
       code='868008.WI',
       benchmark_code='000905.SH',
       window_before=20,
       window_after=20,
       metric='close'
   )

多标的平均模式
--------------

传入代码列表，自动计算所有股票在每个时间点的平均收益率：

.. code-block:: python

   result = es.analyze(
       events=events,
       code=['000905.SH', '000300.SH'],
       window_before=20,
       window_after=20,
       metric='close'
   )

   # 多标的模式额外返回：
   # result['stock_returns_dict'] — 每个股票的收益矩阵
   # result['valid_codes'] — 成功获取数据的代码列表

多标的比较模式
--------------

``multi_asset_mode='compare'`` 在保留顶层等权共性结果的同时，返回每个标的跨事件的统计和矩阵。事件文件无需增加标的列，同一份事件序列会应用于所有代码：

.. code-block:: python

   result = es.analyze(
       events=events,
       code=['000905.SH', '000300.SH', '000852.SH'],
       window_before=20,
       window_after=20,
       benchmark_code='000985.CSI',
       multi_asset_mode='compare'
   )

   common_stats = result['holding_stats']
   by_code = result['comparison']['by_code']
   hs300_stats = by_code['000300.SH']['holding_stats']

比较模式的口径如下：

- 顶层 ``daily_stats``、``holding_stats`` 和矩阵仍代表共性结果。
- 共性日收益在相同 ``event_id`` 和相对日上对可用标的求均值；持有收益先逐标的复合，再求均值。
- ``comparison['by_code']`` 提供每个代码的事件数、覆盖率、日度/累计统计和事件矩阵。
- 每个输入事件都有稳定 ``event_id``；某标的缺少某次事件行情时保留空值，不会让后续事件错位。
- 提供 ``benchmark_code`` 时，每个标的先计算相对同一基准的超额收益，再进行聚合与比较。
- 单个事件可以正常比较；由于样本数为 1，标准差和 t 统计为缺失值。

Dashboard 的“多标的处理”参数可选择“等权聚合”或“同图比较”。从单代码输入切换到多代码时，页面默认使用“同图比较”：日度平均收益以分组柱状图按代码列示，平均累计收益以多条曲线按代码列示；图例和悬浮提示同时显示股票代码与中文名称，名称缺失时回退为代码；共同均值仅作为默认隐藏、可从图例启用的参考序列。比较结果同时包含逐标的汇总表、按标的展开的日度/累计统计底表和单次事件下钻曲线；下钻曲线使用统一相对日轴，缺失窗口显示为空档而不是 0；事件级图表最多展示前 30 个事件。

可视化
------

柱状图：事件前后平均日收益
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   es.plot_bar(
       result['daily_stats'],
       title='事件前后平均收益率',
       save_path='bar_chart.png'   # 不传则直接显示
   )

多股票单事件对比
~~~~~~~~~~~~~~~~

.. code-block:: python

   es.plot_multi_stocks(
       events=events,
       codes=['000001.SZ', '000002.SZ', '600036.SH'],
       event_index=0,            # 第一个事件
       window_before=10,
       window_after=10,
       metric='close',
       save_path='multi_stocks.png'
   )

单股票多事件对比
~~~~~~~~~~~~~~~~

.. code-block:: python

   es.plot_events_lines(
       events=events,
       code='868008.WI',
       window_before=10,
       window_after=10,
       metric='close',
       max_events=10,            # 最多展示10个事件
       save_path='events_lines.png'
   )

``plot_multi_stocks`` 和 ``plot_events_lines`` 会复用 ``analyze()`` 的 ``price_matrix``，
仅比较事件前后的价格形态。所有曲线在 Day 0 对齐为 0，纵轴表示相对价格变化。

完整示例
--------

.. code-block:: python

   from betalens import Datafeed, EventStudy
   import pandas as pd

   # 1. 初始化
   df = Datafeed("market_daily_post_adjusted")
   es = EventStudy(df)

   # 2. 加载事件数据
   events_df = pd.read_excel('events.xlsx')
   events = events_df.set_index('date')['event']

   # 3. 分析
   result = es.analyze(
       events=events,
       code='868008.WI',
       window_before=20,
       window_after=20,
       metric='close'
   )

   # 4. 查看统计
   print(f"事件数: {result['event_count']}")
   print(result['daily_stats'])
   print(result['holding_stats'])

   # 5. 可视化
   es.plot_bar(result['daily_stats'], title='日收益', save_path='bar.png')
   es.plot_events_lines(events, '868008.WI', save_path='event_prices.png')

   df.close()
