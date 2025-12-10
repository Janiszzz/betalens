Robust API
==========

.. automodule:: betalens.robust
   :members:
   :undoc-members:
   :show-inheritance:

robust.robust
-------------

RobustTest
~~~~~~~~~~

.. py:class:: RobustTest(fund, factor)

   因子增量检验类，基于 Harvey & Liu (2021) "Lucky Factors"。

   :param fund: 基金/组合收益序列（pd.Series）
   :param factor: 因子数据（pd.DataFrame）

   **属性**

   .. py:attribute:: X
      :type: pd.DataFrame

      因子数据

   .. py:attribute:: y
      :type: pd.Series

      资产收益数据

   .. py:attribute:: OX
      :type: pd.DataFrame

      正交化后的因子数据

   .. py:attribute:: T
      :type: pd.DataFrame

      t统计量

   **方法**

   .. py:method:: neu()

      去相关（正交化）。对每个因子进行单因子回归，得到正交化残差。

      :return: (OX, T) 元组

   .. py:method:: bootstrap_once(n_bootstraps=1000)

      Bootstrap检验。重复抽样计算最大统计量分布。

      :param n_bootstraps: 重采样次数
      :return: (eff_fct_name, modifd_P, max_statistic_pdf) 元组

   .. py:method:: work()

      完整工作流程。迭代执行去相关和Bootstrap直到收敛。

   .. py:staticmethod:: create_sample_dataframes()

      创建示例数据集用于测试。

      :return: (asset_returns, factor_values) 元组

辅助函数
~~~~~~~~

.. py:function:: panel(X, y)

   面板回归，单测alpha。

   :param X: 因子数据
   :param y: 收益数据
   :return: (B, OX, T, df_params) 元组

.. py:function:: bootstrap_fake_fund(X, B, OX, T, n_bootstraps=1000)

   Bootstrap检验伪基金。

   :return: (modifd_P, max_statistic_pdf) 元组

.. py:function:: parse_name_dates(s)

   解析基金经理任期字符串。

   :param s: 格式如 '姓名(开始日期-结束日期)'
   :return: 包含 name, start_date, end_date 的字典

.. py:function:: get_interval(df, start=None, end=None)

   获取DataFrame的时间区间切片。

.. py:function:: gen_date_pairs(start_time, end_time, interval='1Y')

   生成滚动时间段对。

   :param start_time: 开始时间
   :param end_time: 结束时间
   :param interval: 时间间隔
   :return: 时间戳对列表


