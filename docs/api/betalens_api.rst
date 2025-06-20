###################
Betalens API 参考
###################

本文档详细介绍了 ``betalens`` 包中主要模块的公共 API。

.. contents:: 模块目录
   :local:

********************
betalens.datafeed
********************

.. currentmodule:: betalens.datafeed

``Datafeed`` 模块提供了一个与金融数据库（PostgreSQL）交互的接口，专门用于处理和查询金融时间序列数据。它的核心是 ``Datafeed`` 类，旨在简化行情数据、基本面数据等的入库、清洗和查询流程，为量化研究和回测提供支持。

主要功能包括：

- **数据入库**: 将标准格式的 CSV 文件（如日行情、指数、基金数据）批量导入数据库。
- **数据清洗**: 提供特定于数据源（如Wind EDE）的数据清洗和转换功能。
- **数据整合**: 将不同来源的数据（如财报数据和披露日期）进行合并。
- **高级查询**:
    - 按多种条件（日期范围、代码、指标）进行灵活查询。
    - 针对回测场景，高效查询指定时间点之前或之后最近的有效数据点。


Datafeed 类
-----------

.. class:: Datafeed(sheetname)

   初始化一个数据馈送实例，用于连接和操作指定的数据库表。

   在实例化时，此类会建立一个到本地 PostgreSQL 数据库表 "datafeed" 的连接。

   .. admonition:: Parameters
      :class: note

      :param sheetname: 要操作的数据库表的名称。
      :type sheetname: str

   **数据入库与处理方法**

   .. method:: insert_daily_market_data(data, table)

      将经过格式化的日度行情序列数据（如股票）插入到指定的表中。

      此方法首先会对输入的 DataFrame 进行转换（melt），将多个指标列转换为单一的键值对列，并根据指标名称（如“开盘价”）设置精确的时间戳，然后执行批量插入。

      :param data: 包含日度行情数据的 DataFrame。预期列应包括代码、简称、日期以及多个指标列（如“开盘价”）。
      :type data: pandas.DataFrame
      :param table: 目标数据库表的名称。注意：应符合betalens数据库规范。
      :type table: str
      :return: 成功返回 0，失败返回 1。
      :rtype: int

   .. method:: insert_daily_index_data(data, table)

      功能与 ``insert_daily_market_data`` 类似，专门用于处理指数的日度行情数据。

      :param data: 包含指数日度行情数据的 DataFrame。
      :type data: pandas.DataFrame
      :param table: 目标数据库表的名称。
      :type table: str
      :return: 成功返回 0，失败返回 1。
      :rtype: int

   .. method:: insert_daily_fund_data(data, table)

      功能与 ``insert_daily_market_data`` 类似，专门用于处理基金的日度行情数据。

      :param data: 包含基金日度行情数据的 DataFrame。
      :type data: pandas.DataFrame
      :param table: 目标数据库表的名称。
      :type table: str
      :return: 成功返回 0，失败返回 1。
      :rtype: int

   .. method:: insert_files(folder_path, insert_func)

      批量读取一个文件夹中的所有 CSV 文件，并使用指定的插入函数将数据导入数据库。

      :param folder_path: 存放 CSV 文件的文件夹路径。
      :type folder_path: str
      :param insert_func: 用于处理和插入单个文件数据的函数（例如 ``Datafeed.insert_daily_market_data``）。
      :type insert_func: function
      :return: 导入失败的文件路径列表。
      :rtype: list

   .. method:: insert_washed_data(df, table)

      将一个已经清洗干净的 DataFrame 批量插入到指定的数据库表中。

      :param df: 清洗后的数据。
      :type df: pandas.DataFrame
      :param table: 目标数据库表的名称。
      :type table: str
      :return: 成功返回 0，失败返回 1。
      :rtype: int

   **数据清洗方法**

   .. method:: wash_ede_data(filepath) -> pandas.DataFrame

      静态方法，用于清洗从特定数据源（Wind EDE）导出的 CSV 文件。它会将宽表格式转换为长表格式，并将Wind数据源的列名解析为 JSON 格式。

      :param filepath: 原始 CSV 文件的路径。
      :type filepath: str
      :return: 清洗和转换后的 DataFrame。
      :rtype: pandas.DataFrame

   .. method:: merge_fundamental_data(label_time_filepath, value_filepath, metric_name) -> pandas.DataFrame

      静态方法，用于合并基本面数据。它将一个包含“实际披露日期”的文件和一个包含“指标值”的文件根据代码和报告期进行合并。

      :param label_time_filepath: 包含实际披露日期的已清洗 CSV 文件路径。
      :type label_time_filepath: str
      :param value_filepath: 包含指标值的已清洗 CSV 文件路径。
      :type value_filepath: str
      :param metric_name: 要分配给合并后指标的名称。
      :type metric_name: str
      :return: 合并后的 DataFrame，包含 code, name, datetime, label_datetime, value, metric 等列。
      :rtype: pandas.DataFrame


   **数据查询方法**

   .. method:: query_data(params=None) -> pandas.DataFrame

      根据指定的多个条件从表中查询数据。

      :param params: 包含查询条件的字典。支持的键包括：
                     - 'start_date': (str) 开始日期, 'YYYY-MM-DD'
                     - 'end_date': (str) 结束日期, 'YYYY-MM-DD'
                     - 'code': (list[str]) 资产代码列表
                     - 'metric': (str) 指标名称
                     - 'label_start_date': (str) 报告期开始日期
                     - 'label_end_date': (str) 报告期结束日期
      :type params: dict, optional
      :return: 查询结果的 DataFrame。
      :rtype: pandas.DataFrame

      .. admonition:: Example
         :class: example

         .. code-block:: python

            db = Datafeed("daily_market_data")
            query_params = {
                'start_date': '2023-01-01',
                'end_date': '2023-03-31',
                'code': ['000001.SZ', '600519.SH'],
                'metric': '收盘价(元)'
            }
            price_data = db.query_data(params=query_params)
            print(price_data.head())


   .. method:: query_nearest_before(params) -> pandas.DataFrame

      对于给定的代码和时间戳列表，查询每个时间点 **之前** 最近的有效指标值。主要用于在回测中获取历史特征。

      :param params: 包含查询条件的字典。必须的键包括：
                     - 'codes': (list[str]) 资产代码列表
                     - 'datetimes': (list[str]) 目标时间戳列表, 'YYYY-MM-DD HH:MM:SS'
                     - 'metric': (str) 指标名称
                     可选的键：
                     - 'time_tolerance': (int) 允许的最大回溯时间间隔（小时）
      :type params: dict
      :return: 包含 code, input_ts, datetime, diff_hours, value 的 DataFrame。
      :rtype: pandas.DataFrame

      .. admonition:: Example
         :class: example

         .. code-block:: python

            db = Datafeed("daily_market_data")
            query_params = {
                'codes': ['000001.SZ'],
                'datetimes': ['2023-01-10 14:30:00'],
                'metric': '收盘价(元)',
                'time_tolerance': 72  # 最多回溯72小时
            }
            last_close = db.query_nearest_before(params=query_params)
            print(last_close)


   .. method:: query_nearest_after(params) -> pandas.DataFrame

      对于给定的代码和时间戳列表，查询每个时间点 **之后** 最近的有效指标值。主要用于在回测中获取未来价格以计算收益。

      :param params: 包含查询条件的字典。必须的键包括：
                     - 'codes': (list[str]) 资产代码列表
                     - 'datetimes': (list[str]) 目标时间戳列表, 'YYYY-MM-DD HH:MM:SS'
                     - 'metric': (str) 指标名称
                     可选的键：
                     - 'time_tolerance': (int) 允许的最大向前查找时间间隔（小时）
      :type params: dict
      :return: 包含 code, input_ts, datetime, diff_hours, value 的 DataFrame。
      :rtype: pandas.DataFrame


辅助函数
--------

.. function:: get_absolute_trade_days(begin_date, end_date, period) -> list

   获取指定日期区间内的交易日列表。依赖于 ``WindPy`` 库，且仅能提供周期末日。如获取2015-04-30~2017-04-30的年频日期，将返回[2015-12-31,2016-12-31,2017-04-30]。

   :param begin_date: 开始日期。
   :type begin_date: str
   :param end_date: 结束日期。
   :type end_date: str
   :param period: 周期（例如 'D' 表示日）。
   :type period: str
   :return: 交易日日期字符串列表。
   :rtype: list

使用示例
--------

下面的示例展示了如何使用 ``Datafeed`` 类来清洗、合并和检查基本面数据。

.. code-block:: python

   import pandas as pd
   from datafeed import Datafeed

   # 假设我们有两个从数据源导出的CSV文件：
   # 1. '定期报告实际披露日期.csv': 包含财报的实际披露日期
   # 2. '股息率(报告期).csv': 包含按报告期记录的股息率

   # 步骤 1: 实例化 Datafeed 类，指定目标表
   db = Datafeed("fundamental_data")

   # 步骤 2: 分别清洗两个数据文件
   # wash_ede_data 是静态方法，可以直接通过类名调用
   Datafeed.wash_ede_data("定期报告实际披露日期.csv")
   Datafeed.wash_ede_data("股息率(报告期).csv")
   # 这会生成两个清洗后的文件: *_washed.csv

   # 步骤 3: 合并清洗后的数据
   # merge_fundamental_data 也是静态方法
   merged_df = Datafeed.merge_fundamental_data(
       label_time_filepath="定期报告实际披露日期.csv_washed.csv",
       value_filepath="股息率(报告期).csv_washed.csv",
       metric_name="股息率(报告期)"
   )
   # 这会生成一个合并后的文件: *_mergeded.csv

   # 步骤 4: 检查合并后数据的完整性和唯一性
   checked_df = db.check_result(merged_df)
   print("数据检查完成，处理后的数据行数:", len(checked_df))

   # 步骤 5: 将最终的数据插入数据库
   # result = db.insert_washed_data(checked_df, db.sheet)
   # if result == 0:
   #     print("数据成功插入数据库！")

*****************
betalens.factor
*****************

.. currentmodule:: betalens.factor

``Factor`` 模块提供了一套用于执行单因子策略的工具函数。它整合了数据获取、因子计算、分组和权重生成等关键步骤，旨在简化从因子思想到策略权重的转换过程。

该模块的核心流程包括：

1.  **确定可交易股票池**: 根据给定的调仓日期，从数据库中筛选出处于正常交易状态的股票。
2.  **获取特征值**: 在每个调仓日，为股票池中的每只股票获取 **最新** 的特征值。
3.  **因子分组**: 根据特征值对股票进行排序和分位数分组。
4.  **生成多空权重**: 根据分组结果，为多头组合和空头组合生成相应的持仓权重。
5.  **回测**：持仓权重时间序列完成了对策略的全部描述。结合回测模块对策略进行回测。
6.  **因子描述**: 提供工具来描述因子的分组情况，例如计算每组的样本数和均值。

这些功能共同构成了一个基础的单因子测试框架，依赖于 ``betalens.datafeed`` 模块来与底层数据库交互。


.. function:: get_tradable_pool(date_list) -> tuple[list, list]

   根据给定的日期列表，获取每个日期对应的可交易股票池。

   函数会查询数据库中每个日期的股票交易状态，并筛选出状态为可交易（值为1）的股票。

   .. admonition:: Parameters
      :class: note

      :param date_list: 一个包含调仓日期的列表。日期应为 datetime 对象。
      :type date_list: list[datetime.datetime]

   :return: 一个元组，包含两个列表：
            - 第一个列表是去重后的实际有可交易股票的日期。若input无误，应与其相同。
            - 第二个列表是与日期列表对应的、包含可交易股票代码的嵌套列表。
   :rtype: tuple(list, list[list[str]])


.. function:: single_factor(date_ranges, code_ranges, metric, quantiles) -> pandas.DataFrame

   为每个调仓日的股票池，匹配其当日所能获取的最新的特征值，并进行分位数分组。

   该函数会为每个调仓日和对应的股票池，查询在该日之前最新特征值（使用 ``query_nearest_before``），然后对每个截面上的股票按因子值进行分位数分组。该函数的input为get_tradable_pool的output。

   .. admonition:: Parameters
      :class: note

      :param date_ranges: 调仓日期列表，通常来自 ``get_tradable_pool`` 的返回。
      :type date_ranges: list
      :param code_ranges: 与调仓日对应的可交易股票代码列表。
      :type code_ranges: list[list[str]]
      :param metric: 要查询的因子名称（在数据库中的 `metric` 字段）。注意，你需要提前确认数据库中有完整的该数据字段。
      :type metric: str
      :param quantiles: 用于 ``pandas.qcut`` 的分位数定义。可以是一个整数（例如10表示十分位数），或一个表示分位数边界的列表（例如 `[0, .2, .4, .6, .8, 1]`）。
      :type quantiles: int or list[float]

   :return: 一个以 `(input_ts, code)` 为多级索引的 DataFrame，包含了每个股票在每个调仓日的因子值和分组标签 (`_label`)。
   :rtype: pandas.DataFrame


.. function:: get_single_factor_weight(labeled_pool, params) -> pandas.DataFrame

   根据因子分组结果，生成多空组合的权重。

   函数支持两种模式：
   - ``classic-long-short``: 做多因子值最高的分组，做空因子值最低的分组。
   - ``freeplay``: 根据 `params` 中明确指定的 `long` 和 `short` 分组列表来构建多空组合。

   生成的权重会进行归一化处理：多头权重的和为1，空头权重的和为-1。

   .. admonition:: Parameters
      :class: note

      :param labeled_pool: 经过分组的因子暴露 DataFrame，通常是 ``single_factor`` 函数的输出。
      :type labeled_pool: pandas.DataFrame
      :param params: 一个包含权重生成逻辑的字典。
                     - 'factor_key': (str) 因子名称，用于构建列名。
                     - 'mode': (str) 模式选择，'classic-long-short' 或 'freeplay'。
                     - 'long': (list[int]) 如果模式是 'freeplay'，此列表指定哪些分组应作为多头。
                     - 'short': (list[int]) 如果模式是 'freeplay'，此列表指定哪些分组应作为空头。
      :type params: dict

   :return: 一个以调仓日为索引、股票代码为列的 DataFrame，值为每只股票在组合中的权重。
   :rtype: pandas.DataFrame

   .. admonition:: Example
      :class: example

      .. code-block:: python

         # 假设 labeled_pool 是 single_factor 的输出
         params = {
             'factor_key': '股息率(报告期)',
             'mode': 'classic-long-short',
         }
         weights = get_single_factor_weight(labeled_pool, params)
         print(weights.head())

         params_freeplay = {
             'factor_key': '股息率(报告期)',
             'mode': 'freeplay',
             'long': [9],  # 做多第9组 (最高)
             'short': [0, 1], # 做空第0和第1组 (最低)
         }
         weights_freeplay = get_single_factor_weight(labeled_pool, params_freeplay)
         print(weights_freeplay.head())


.. function:: describe_labeled_pool(labeled_pool) -> pandas.DataFrame

   生成一个透视表来描述和汇总因子分组情况。

   透视表以调仓日为索引，因子分组标签为列，可以展示每个分组在每个截面上的样本数量和因子均值。

   .. admonition:: Parameters
      :class: note

      :param labeled_pool: 经过分组的因子暴露 DataFrame，是 ``single_factor`` 函数的输出。
      :type labeled_pool: pandas.DataFrame

   :return: 一个描述因子分组统计特征的透视表。
   :rtype: pandas.DataFrame

使用示例
--------

下面的示例展示了从获取交易日、构建因子、生成权重到执行回测的完整流程。

.. code-block:: python

   import betalens.datafeed
   import betalens.backtest
   import betalens.analyst
   from factor import (
       get_tradable_pool,
       single_factor,
       get_single_factor_weight,
       describe_labeled_pool
   )

   # 1. 获取年度调仓日期
   date_list = betalens.datafeed.get_absolute_trade_days(
       "2015-04-30", "2024-04-30", "Y"
   )

   # 2. 获取每个调仓日的可交易股票池
   date_ranges, code_ranges = get_tradable_pool(date_list)

   # 3. 定义因子和分组方式
   metric = "股息率(报告期)"
   quantiles = {"股息率(报告期)": 10} # 十分位分组

   # 4. 计算因子暴露并进行分组
   labeled_pool = single_factor(date_ranges, code_ranges, metric, quantiles)

   # 5. (可选) 查看因子分组的描述性统计
   stats_table = describe_labeled_pool(labeled_pool)
   print(stats_table)

   # 6. 定义权重生成规则并生成权重
   params = {
       'factor_key': '股息率(报告期)',
       'mode': 'classic-long-short',
   }
   weights = get_single_factor_weight(labeled_pool, params)

   # 7. (可选) 使用生成的权重进行回测
   weights['cash'] = 0  # 可选，设置现金头寸
   backtest_engine = betalens.backtest.BacktestBase(
       weight=weights,
       amount=1000000
   )
   backtest_engine.nav.plot()

*******************
betalens.backtest
*******************

.. currentmodule:: betalens.backtest

``Backtest`` 模块提供了一个基础的向量化回测框架，封装在 ``BacktestBase`` 类中。它的主要功能是根据用户提供的资产权重序列，模拟策略的交易和持仓，并最终计算出每日的净值（NAV）曲线。

该回测引擎的核心逻辑包括：

1.  **数据获取**: 在每个调仓日，从数据库获取用于计算交易成本的资产价格。
2.  **市值计算**: 根据调仓日的资产价格和目标资金权重，计算调仓日的投资组合市值。
3.  **每日净值生成**: 基于调仓日的持仓手数，结合期间每日的收盘价，计算出每日的盯市价值和策略净值。

此类旨在简化回测流程，用户只需提供核心的权重信号，即可快速得到策略表现的初步结果。



BacktestBase 类
---------------

.. class:: BacktestBase(weight, symbol, amount, ftc=0.0, ptc=0.0, verbose=True)

   执行向量化回测的基础类。

   在实例化时，该类会自动执行整个回测流程，包括获取数据、计算每次调仓时的市值，以及生成每日的净值序列。回测结果会存储在类的属性中，最常用的是 ``nav``。

   .. admonition:: Parameters
      :class: note

      :param weight: 包含策略权重的 DataFrame。，以时间x代码形式储存权重值。
                     - **索引 (index)**: 必须是表示调仓日期的 DatetimeIndex。
                     - **列 (columns)**: 必须是资产的唯一标识代码（例如股票代码）。可以包含一个名为 'cash' 的列来表示现金头寸。
      :type weight: pandas.DataFrame
      :param symbol: 策略的标识符或名称（当前版本中未使用）。
      :type symbol: str
      :param amount: 初始投资金额。
      :type amount: int or float
      :param ftc: 固定交易成本（Fixed Transaction Costs）。当前版本中未实现。默认为 0.0。
      :type ftc: float, optional
      :param ptc: 比例交易成本（Proportional Transaction Costs）。当前版本中未实现。默认为 0.0。
      :type ptc: float, optional
      :param verbose: 是否打印详细的回测过程信息。当前版本中未实现。默认为 True。
      :type verbose: bool, optional


   **核心属性**

   .. attribute:: nav

      回测完成后生成的每日净值序列。索引为日期，值为当日净值。这是评估策略表现最核心的输出。

      :type: pandas.Series

   .. attribute:: daily_amount

      每日的投资组合总市值。

      :type: pandas.Series

   .. attribute:: position

      每日的资产持仓数量（单位：股/份）。

      :type: pandas.DataFrame

   .. attribute:: cost_price

      每次调仓时，用于执行交易的实际资产价格。

      :type: pandas.DataFrame

   **内部方法**
   下面的方法在 ``__init__`` 中被自动调用，代表了回测的主要步骤，用户通常不需要直接与它们交互。

  .. method:: get_rebalance_data()

      获取每个调仓日之后最近的有效价格作为交易成本价，并计算调仓期间的资产回报率。注意：这里隐含的行为是在调仓信号出现后，尽快进行调仓，而未必是在信号日调仓。这是为了不引入未来视野。

   .. method:: get_position_data()

      基于调仓期间的回报率和前次权重，计算出每个调仓节点上的投资组合总金额。

   .. method:: get_daily_position_data()

      将调仓日的持仓（单位：股/份）向前填充到每个交易日，并结合每日收盘价计算每日的盯市总价值（``daily_amount``）和净值（``nav``）。

使用示例
--------

下面的示例展示了如何使用一个虚拟的权重 DataFrame 来实例化 ``BacktestBase`` 并获取回测结果。

.. code-block:: python

   import pandas as pd
   from backtest import BacktestBase
   import matplotlib.pyplot as plt

   # 1. 创建一个虚拟的权重 DataFrame
   # 假设我们有三只股票，并且在每个调仓日都等权重持有它们。
   # 调仓频率为每月一次。
   rebalance_dates = pd.to_datetime(pd.date_range(
       start='2023-01-01',
       end='2024-01-01',
       freq='MS'
   ))
   assets = ['000001.SZ', '600519.SH', '000858.SZ']
   weights_df = pd.DataFrame(1/len(assets), index=rebalance_dates, columns=assets)

   # 添加 'cash' 列，表示无现金头寸
   #  'cash' 列是为了满足含空仓情况的策略
   weights_df['cash'] = 0

   # 2. 实例化 BacktestBase 类，并运行回测
   # 初始资金为 1,000,000
   bb = BacktestBase(weight=weights_df, symbol="equal_weight_strategy", amount=1000000)

   # 3. 访问并可视化回测结果
   # 打印每日净值序列的最后五行
   print("每日净值 (NAV):")
   print(bb.nav.tail())

   # 绘制净值曲线
   plt.figure(figsize=(10, 6))
   bb.nav.plot(title='Strategy Net Asset Value (NAV)')
   plt.xlabel('Date')
   plt.ylabel('NAV')
   plt.grid(True)
   plt.show()



*****************
betalens.analyst
*****************

.. currentmodule:: betalens.analyst

``Analyst`` 模块是用于投资组合绩效分析和报告生成的工具集。它包含两个核心类：``PortfolioAnalyzer`` 用于计算各种标准的风险和收益指标，而 ``ReportExporter`` 则用于将这些分析结果格式化为清晰、易读的报告。

该模块的主要功能包括：

- **绩效指标计算**: 计算累计收益率、年化收益率、年化波动率、夏普比率、最大回撤和卡玛比率等核心指标。
- **滚动指标分析**: 提供滚动窗口计算，以观察指标随时间的变化。
- **报告生成**:
    - 生成分年度的绩效总结。
    - 生成指定时间段的绩效报告。
    - 生成与基准组合的对比分析报告。
- **灵活输出**: 支持将报告输出到命令行（CLI）表格或 Excel 文件。



PortfolioAnalyzer 类
----------------------

.. class:: PortfolioAnalyzer(nav_series, risk_free_rate=0.0, annualizer=252, window=30)

   该类接收一个净值序列，并提供多种方法来计算投资组合的绩效指标。

   .. admonition:: Parameters
      :class: note

      :param nav_series: 包含日期索引的投资组合净值序列。默认日频。
      :type nav_series: pandas.Series
      :param risk_free_rate: 年化的无风险利率，用于计算夏普比率等指标。默认为 0.0。
      :type risk_free_rate: float, optional
      :param annualizer: 年化因子，即一年中的交易日数量。默认为 252。若输入为周频序列，应对应指定为52。
      :type annualizer: int, optional
      :param window: 用于计算滚动指标的窗口大小（天数）。默认为 30。
      :type window: int, optional

   **核心指标方法**

   .. method:: total_return() -> float

      计算整个期间的累计收益率。

   .. method:: annualized_return() -> float

      计算年化收益率。

   .. method:: annualized_volatility() -> float

      计算年化波动率。

   .. method:: sharpe_ratio() -> float

      计算年化夏普比率。

   .. method:: max_drawdown() -> float

      计算整个期间的最大回撤。

   .. method:: calmar_ratio() -> float

      计算卡玛比率（年化收益率 / 最大回撤）。


ReportExporter 类
-----------------

.. class:: ReportExporter(analyzer, benchmark_analyzer=None, start_date=None, end_date=None)

   该类使用 ``PortfolioAnalyzer`` 的实例来生成格式化的绩效报告。

   .. admonition:: Parameters
      :class: note

      :param analyzer: 要报告的投资组合的 ``PortfolioAnalyzer`` 实例。
      :type analyzer: PortfolioAnalyzer
      :param benchmark_analyzer: (可选) 作为对比基准的 ``PortfolioAnalyzer`` 实例。
      :type benchmark_analyzer: PortfolioAnalyzer, optional
      :param start_date: (可选) 如果提供，则所有报告将只针对从该日期开始的数据。
      :type start_date: str or datetime, optional
      :param end_date: (可选) 如果提供，则所有报告将只针对到该日期结束的数据。
      :type end_date: str or datetime, optional


   **报告生成方法**

   .. method:: generate_annual_report(excel_path=None)

      生成并打印分年度的绩效报告，包含每年的回报率、波动率、最大回撤和夏普比率。

      :param excel_path: (可选) 如果提供路径，则会将报告保存为 Excel 文件。
      :type excel_path: str, optional

   .. method:: generate_custom_report(start_date, end_date, excel_path=None)

      生成并打印指定时间段内的整体绩效报告。

      :param start_date: 报告的开始日期。
      :type start_date: str or datetime
      :param end_date: 报告的结束日期。
      :type end_date: str or datetime
      :param excel_path: (可选) 如果提供路径，则会将报告保存为 Excel 文件。
      :type excel_path: str, optional

   .. method:: generate_benchmark_report(excel_path=None)

      生成并打印投资组合与基准的对比报告。如果初始化时未提供 ``benchmark_analyzer``，则会引发 ``ValueError``。

      报告内容包括组合与基准各自的收益、波动率，以及超额收益。

      :param excel_path: (可选) 如果提供路径，则会将报告保存为 Excel 文件。
      :type excel_path: str, optional

使用示例
--------

下面的示例展示了如何使用 ``PortfolioAnalyzer`` 和 ``ReportExporter`` 来分析一个模拟的净值序列并生成多种报告。

.. code-block:: python

   import pandas as pd
   import numpy as np
   from analyst import PortfolioAnalyzer, ReportExporter

   # 1. 生成模拟的策略净值和基准净值序列
   dates = pd.date_range('2020-01-01', '2023-12-31')
   strategy_returns = np.random.normal(0.0008, 0.015, len(dates))
   benchmark_returns = np.random.normal(0.0005, 0.01, len(dates))

   strategy_nav = pd.Series(np.exp(strategy_returns.cumsum()), index=dates)
   benchmark_nav = pd.Series(np.exp(benchmark_returns.cumsum()), index=dates)


   # 2. 为策略和基准分别创建 PortfolioAnalyzer 实例
   strategy_analyzer = PortfolioAnalyzer(strategy_nav, risk_free_rate=0.02)
   benchmark_analyzer = PortfolioAnalyzer(benchmark_nav, risk_free_rate=0.02)

   # 3. 创建 ReportExporter 实例
   exporter = ReportExporter(analyzer=strategy_analyzer, benchmark_analyzer=benchmark_analyzer)

   # 4. 生成不同类型的报告
   print("--- 分年度绩效报告 ---")
   exporter.generate_annual_report()

   print("\n--- 2022年度绩效报告 ---")
   exporter.generate_custom_report('2022-01-01', '2022-12-31')

   print("\n--- 基准对比报告 ---")
   exporter.generate_benchmark_report(excel_path="performance_report.xlsx")