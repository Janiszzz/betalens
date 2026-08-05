事件研究模块
============

事件研究（Event Study）模块用于分析特定事件前后证券的收益率表现。支持单标的、多标的、基准超额收益等多种模式。

模块概述
--------

``EventStudy`` 接收一个 ``Datafeed`` 实例，通过 ``query_time_range()`` 获取价格数据，
然后围绕事件日期计算窗口期收益率，并提供统计分析和可视化功能。

**核心流程：**

1. 准备事件序列（Series，index 为 datetime，值 1 表示事件发生）
2. 调用 ``analyze()`` 方法分析事件窗口收益
3. 查看统计结果或绘制图表

初始化
------

.. code-block:: python

   from betalens import Datafeed, EventStudy

   df_market = Datafeed("daily_market")
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
       metric='收盘价(元)'
   )

``analyze()`` 返回一个字典，包含：

- ``daily_stats``: 每日收益统计（均值、标准差、上涨概率、t 统计量等）
- ``cumulative_stats``: 累积收益统计
- ``event_count``: 有效事件数
- ``returns_matrix``: 完整收益矩阵（行=相对天数，列=事件编号）
- ``cumulative_returns_matrix``: 每次事件的累积收益矩阵（行=相对天数，列=事件编号）

Day 0 成本价规则
~~~~~~~~~~~~~~~~

- 事件在 15:00 前发生 → 当天收盘价为 Day 0 成本价
- 事件在 15:00 后发生 → 第二个交易日收盘价为 Day 0 成本价

分析模式
--------

flexible 模式（默认）
~~~~~~~~~~~~~~~~~~~~~

以 ``holding_start_offset`` 指定的持有起点为分界点计算双向累积收益，适合观察事件前后的整体走势：

.. code-block:: python

   result = es.analyze(
       events=events,
       code='868008.WI',
       window_before=20,
       window_after=20,
       mode='flexible'
   )

fixed 模式
~~~~~~~~~~

计算固定持有期的累积收益，适合精确衡量持有 N 天/月的回报：

.. code-block:: python

   result = es.analyze(
       events=events,
       code='868008.WI',
       window_before=20,
       window_after=60,
       mode='fixed',
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

.. code-block:: python

   result = es.analyze(
       events=events,
       code='868008.WI',
       benchmark_code='000905.SH',
       window_before=20,
       window_after=20,
       metric='收盘价(元)'
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
       metric='收盘价(元)'
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

   common_stats = result['cumulative_stats']
   by_code = result['comparison']['by_code']
   hs300_stats = by_code['000300.SH']['cumulative_stats']

比较模式的口径如下：

- 顶层 ``daily_stats``、``cumulative_stats`` 和矩阵仍代表共性结果。
- 共性先在相同 ``event_id`` 和相对日上对可用标的等权平均，再沿用 flexible/fixed 累计算法。
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

折线图：累积收益曲线
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 单标的
   es.plot_lines(result['cumulative_stats'], title='累积收益')

   # 多标的对比：传入 {代码: cumulative_stats} 字典
   es.plot_lines(
       {'000905.SH': result1['cumulative_stats'],
        '000300.SH': result2['cumulative_stats']},
       title='多标的累积收益对比'
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
       metric='收盘价(元)',
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
       metric='收盘价(元)',
       max_events=10,            # 最多展示10个事件
       save_path='events_lines.png'
   )

完整示例
--------

.. code-block:: python

   from betalens import Datafeed, EventStudy
   import pandas as pd

   # 1. 初始化
   df = Datafeed("daily_market")
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
       metric='收盘价(元)'
   )

   # 4. 查看统计
   print(f"事件数: {result['event_count']}")
   print(result['daily_stats'])
   print(result['cumulative_stats'])

   # 5. 可视化
   es.plot_bar(result['daily_stats'], title='日收益', save_path='bar.png')
   es.plot_lines(result['cumulative_stats'], title='累积收益', save_path='line.png')

   df.close()
