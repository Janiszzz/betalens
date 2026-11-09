EventStudy API
==============

EventStudy
~~~~~~~~~~

.. py:class:: EventStudy(datafeed)

   事件研究分析器。

   :param datafeed: :class:`betalens.datafeed.Datafeed` 实例，用于查询价格数据

   .. py:method:: analyze(events, code, window_before=5, window_after=5, metric='close', periods=None, holding_periods=None, holding_start_offset=0, market_close_hour=15, benchmark_code=None, multi_asset_mode='aggregate', exchange='SSE', price_time_map=None)

      分析事件前后的收益率表现。

      :param events: 事件序列（pd.Series），index 为精确到秒的 datetime，值 1 表示事件发生
      :param code: 证券代码（str 单标的 | List[str] 多标的取平均）
      :param window_before: 事件前窗口期数
      :param window_after: 事件后窗口期数
      :param metric: 物理价格字段，如 ``'close'`` / ``'open'``
      :param periods: 可选的时间分段序列（pd.Series），用于分组统计
      :param holding_periods: 持有期字典，如 ``{'days': [1,2,3,4,5], 'months': [1,3,6,9,12]}``；不传时使用该默认值
      :param holding_start_offset: 持有起点偏移天数（0=Day 0, -3=提前3天）
      :param market_close_hour: 市场收盘时间（小时），默认 15
      :param benchmark_code: 基准代码，提供时计算超额收益 = 标的收益 - 基准收益
      :param multi_asset_mode: 多标的处理方式；``'aggregate'`` 保持等权聚合，``'compare'`` 同时返回逐标的比较结果
      :param exchange: 标准交易日历交易所代码，默认 ``'SSE'``；兼容 ``'SHSE'``
      :param price_time_map: 自定义价格字段的观测时刻映射，格式 ``{字段: 'HH:MM'}``
      :return: 结果字典，包含以下键：

         - ``daily_stats`` (pd.DataFrame): 每日收益统计，index=day，列含 mean, std, positive_prob, odds, t_stat, count
         - ``holding_stats`` (pd.DataFrame): 固定持有收益统计，含持有期、均值、标准差、盈利概率、t 统计和样本数
         - ``event_count`` (int): 有效事件数
         - ``returns_matrix`` (pd.DataFrame): 收益矩阵，行=相对天数，列=事件编号
         - ``trade_days`` (pd.DatetimeIndex): 本次分析实际使用的标准交易日序列
         - ``holding_returns_matrix`` (pd.DataFrame): 每次事件的固定持有收益矩阵
         - ``holding_coverage`` (pd.DataFrame): 各事件、各持有期的标的覆盖率
         - ``return_model`` (str): ``per_asset_interval_then_average``
         - ``price_matrix`` (pd.DataFrame): 每次事件的价格曲线，Day 0 归一化为 0
         - ``stock_returns_dict`` (dict, 多标的模式): {代码: 收益矩阵}
         - ``valid_codes`` (list, 多标的模式): 有效代码列表
         - ``skipped_codes`` (list, 多标的模式): 无有效窗口的代码及原因
         - ``comparison`` (dict, compare 模式): 稳定事件映射和 ``by_code`` 逐标的统计、收益矩阵、累计矩阵、覆盖率
         - Dashboard 结果中的 ``assets``: 有效标的的 ``code``、中文 ``name`` 和图例 ``label``；名称查不到时 ``name`` 为 ``null``，``label`` 回退为代码
         - ``period_stats`` (pd.DataFrame, 仅单标的+periods): 分段统计

      .. note::

         ``daily_stats`` 使用每个交易日自身的价格涨跌幅。事件严格早于观测时刻时以当日为 Day 0，等于或晚于时以下一标准交易日为 Day 0。
         相对日由 ``get_absolute_trade_days(..., "D")`` 提供的标准交易日序列确定。
         compare 模式至少需要两个代码；日收益按相对日求均值，持有收益逐标的独立复合后求均值。
         超额持有收益为累计标的收益减累计基准收益。

   .. py:method:: plot_bar(daily_stats, title='事件前后平均收益率', figsize=(12, 6), save_path=None)

      柱状图展示事件前后平均日收益率。

      :param daily_stats: ``analyze()`` 返回的 ``daily_stats``
      :param title: 图表标题
      :param figsize: 图表尺寸
      :param save_path: 保存路径（None 则直接显示）

   .. py:method:: plot_multi_stocks(events, codes, event_index=0, window_before=10, window_after=10, metric='收盘价(元)', market_close_hour=15, title=None, figsize=(14, 8), save_path=None, exchange='SHSE')

      折线图展示多只股票在同一个事件前后的价格走势，各曲线在 Day 0 对齐为 0。

      :param events: 事件序列
      :param codes: 股票代码列表
      :param event_index: 选择第几个事件（0 = 第一个）
      :param window_before: 事件前窗口
      :param window_after: 事件后窗口
      :param metric: 价格指标
      :param market_close_hour: 收盘时间
      :param title: 标题（None 自动生成）
      :param figsize: 图表尺寸
      :param save_path: 保存路径
      :param exchange: 标准交易日历交易所代码，默认 ``'SHSE'``

   .. py:method:: plot_events_lines(events, code, window_before=10, window_after=10, metric='收盘价(元)', market_close_hour=15, title=None, figsize=(14, 8), max_events=None, save_path=None, exchange='SHSE')

      折线图叠加同一标的在多个事件前后的价格走势，各曲线在 Day 0 对齐为 0。

      :param events: 事件序列
      :param code: 股票代码
      :param window_before: 事件前窗口
      :param window_after: 事件后窗口
      :param metric: 价格指标
      :param market_close_hour: 收盘时间
      :param title: 标题（None 自动生成）
      :param figsize: 图表尺寸
      :param max_events: 最多展示事件数（None 全部展示）
      :param save_path: 保存路径
      :param exchange: 标准交易日历交易所代码，默认 ``'SHSE'``

辅助函数
~~~~~~~~

以下为模块内部辅助函数，通常不需要直接调用：

.. py:function:: _get_event_dates(events)

   从事件序列中提取事件发生日期（值为 1 的行）。

   :param events: pd.Series，index=datetime，值=0/1
   :return: pd.DatetimeIndex

.. py:function:: _calc_returns(prices)

   计算日收益率（pct_change）。

   :param prices: 价格序列
   :return: 收益率序列

.. py:function:: _get_day0_cost_price_loc(prices, event_date, market_close_hour=15)

   确定 Day 0 成本价在价格序列中的位置。

   :param prices: 价格序列（pd.Series，index=datetime）
   :param event_date: 事件时间戳
   :param market_close_hour: 收盘时间
   :return: int 位置索引，找不到返回 None

.. py:function:: _get_window_returns(returns, prices, event_date, window_before, window_after, market_close_hour=15)

   获取单个事件的窗口期收益率序列，索引重置为相对天数。

   :return: pd.Series（index=相对天数）或 None

.. py:function:: _aggregate_window_returns(all_returns)

   将多个事件的窗口收益率合并为矩阵。

   :param all_returns: Series 列表
   :return: pd.DataFrame（行=相对天数，列=事件编号）

.. py:function:: _compute_stats(returns)

   计算收益率统计量。

   :param returns: 收益率 Series
   :return: dict，包含 mean, std, positive_prob, odds, t_stat, count

.. py:function:: _compute_period_stats(returns_df, event_dates, periods)

   按时间段分组计算统计量。

   :param returns_df: 收益矩阵
   :param event_dates: 事件日期
   :param periods: 分段序列
   :return: pd.DataFrame
