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

.. py:class:: BacktestBase(weight, symbol, amount, ftc=0.0, ptc=0.0, verbose=True)

   回测基类，实现从调仓权重到日度净值的计算。

   :param weight: 权重DataFrame，行为时间索引，列为证券代码
   :param symbol: 策略名称
   :param amount: 初始资金
   :param ftc: 固定交易费用
   :param ptc: 百分比交易费用
   :param verbose: 是否输出详细信息

   **属性**

   .. py:attribute:: nav
      :type: pd.Series

      日度净值序列

   .. py:attribute:: daily_amount
      :type: pd.Series

      每日市值序列

   .. py:attribute:: position
      :type: pd.DataFrame

      每日持仓数量

   .. py:attribute:: cost_price
      :type: pd.DataFrame

      调仓成本价（MultiIndex: input_ts, datetime）

   .. py:attribute:: cost_ret
      :type: pd.DataFrame

      调仓区间收益率

   .. py:attribute:: start
      :type: datetime

      回测开始时间

   .. py:attribute:: end
      :type: datetime

      回测结束时间

   **方法**

   .. py:method:: get_rebalance_data()

      获取调仓日数据，包含日期和标的匹配验证。

   .. py:method:: get_position_data()

      计算调仓后仓位与累计资产。

   .. py:method:: get_daily_position_data()

      计算每日市值和净值。

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


