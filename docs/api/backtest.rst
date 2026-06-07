Backtest API
============

.. automodule:: betalens.backtest
   :members:
   :undoc-members:
   :show-inheritance:

backtest.backtest
-----------------

BacktestBase
~~~~~~~~~~~~

.. py:class:: BacktestBase(weight, symbol, amount, ftc=0.0, ptc=0.0, verbose=True, lot_size=100, check_trade_status=True, trade_status_mode='to_cash', trade_status_table='trade_status')

   回测基类，实现从调仓权重到日度净值的计算。按整数手成交：每个调仓日把目标权重
   换算为可成交的整数手，余款转入现金。换仓前从 ``trade_status`` 表提取交易状态，
   对停牌持仓按 ``trade_status_mode`` 处理。

   :param weight: 目标权重DataFrame，行为时间索引，列为证券代码
   :param symbol: 策略名称
   :param amount: 初始资金
   :param ftc: 固定交易费用
   :param ptc: 百分比交易费用
   :param verbose: 是否输出详细信息
   :param lot_size: 一手股数（默认100），非整数手无法成交
   :param check_trade_status: 是否在换仓前检查交易状态（默认 True）
   :param trade_status_mode: 停牌持仓处理模式，``to_cash``（默认）/``hold``/``redistribute``/``as_normal``/``report_only``
   :param trade_status_table: 交易状态表名（默认 ``trade_status``）

   **属性**

   .. py:attribute:: nav
      :type: pd.Series

      日度净值序列

   .. py:attribute:: actual_weight
      :type: pd.DataFrame

      实际成交权重（整数手股票权重 + ``cash`` 列，逐行合计为 1）。用于计算
      持仓与净值的真正权重表，区别于输入的目标 ``weight``。

   .. py:attribute:: amount
      :type: pd.Series

      各调仓日（分配前）总资产序列

   .. py:attribute:: daily_amount
      :type: pd.Series

      每日市值序列

   .. py:attribute:: position
      :type: pd.DataFrame

      每日持仓数量（股票列均为 ``lot_size`` 的整数倍）

   .. py:attribute:: cost_price
      :type: pd.DataFrame

      调仓成本价（index=input_ts）

   .. py:attribute:: cost_ret
      :type: pd.DataFrame

      调仓区间收益率

   .. py:attribute:: lot_size
      :type: int

      一手股数（默认 100）

   .. py:attribute:: trade_status
      :type: pd.DataFrame

      调仓日交易状态长表（code/datetime/value/status_text/name），审计用。
      关闭检查或查询失败时为 ``None``。

   .. py:attribute:: trade_status_matrix
      :type: pd.DataFrame

      调仓日 × code 的交易状态矩阵，值为 -1/0/1，index 与 ``weight`` 对齐。

   .. py:attribute:: start
      :type: datetime

      回测开始时间

   .. py:attribute:: end
      :type: datetime

      回测结束时间

   **方法**

   .. py:method:: get_trade_status()

      从 ``trade_status`` 表提取调仓日个券交易状态，建立 ``trade_status`` 长表与
      ``trade_status_matrix`` 矩阵，并按 ``trade_status_mode`` 处理停牌持仓的权重。
      在 ``get_rebalance_data`` 之前执行；``check_trade_status=False`` 时跳过。

   .. py:method:: get_rebalance_data()

      获取调仓日数据，包含日期和标的匹配验证。

   .. py:method:: get_position_data()

      按调仓日逐期迭代：用当期总资产把目标权重换算为整数手（向零截断，
      余款转现金），生成 ``actual_weight`` 与各期总资产 ``amount``。当期手数依赖
      当期总资产、总资产又由上期实际持仓收益递推，故无法向量化。

   .. py:method:: get_daily_position_data()

      用 ``actual_weight`` 计算每日持仓、市值和净值，现金正常参与每日重估。

异常类
~~~~~~

.. py:exception:: BacktestDataError

   回测数据异常基类。

.. py:exception:: DateMismatchError

   日期不匹配异常。当权重日期在数据库中无对应数据时抛出。

.. py:exception:: CodeMismatchError

   标的代码不匹配异常。

验证函数
~~~~~~~~

.. py:function:: validate_weight_input(weight)

   验证权重输入格式。

.. py:function:: validate_query_result(df, expected_columns, query_name)

   验证数据库查询结果。

.. py:function:: validate_pivot_result(df, expected_codes, index_levels)

   验证 pivot_table 结果。

.. py:function:: validate_index_alignment(df1, df2, name1, name2)

   验证两个DataFrame的索引是否对齐。

.. py:function:: validate_calculation_inputs(*args, **kwargs)

   验证计算前的输入数据。

.. py:function:: format_data_sample(df, max_rows, max_cols)

   格式化数据样本用于错误信息。


