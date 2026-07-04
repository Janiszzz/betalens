Robust API
==========

``betalens.robust`` 提供基于 Lucky Factors 思路的因子增量检验和 bootstrap 检验。当前模块文件中仍保留历史实验脚本段，文档页避免在构建期 import 该模块。

RobustTest
----------

.. py:class:: RobustTest(fund, factor)

   因子增量检验类。

   :param fund: 基金或组合收益序列。
   :param factor: 因子收益 DataFrame。

   .. py:staticmethod:: create_sample_dataframes()

      创建示例数据。

   .. py:method:: neu()

      对因子做正交化，返回 ``(OX, T)``。

   .. py:method:: bootstrap_resample(data)

      对输入数据做一次 bootstrap 重采样。

   .. py:method:: max_statistic(data)

      计算 bootstrap 样本的最大统计量。

   .. py:method:: bootstrap_once(n_bootstraps=1000)

      重复抽样计算修正 p 值，返回 ``(eff_fct_name, modifd_P, max_statistic_pdf)``。

   .. py:method:: work()

      迭代执行正交化和 bootstrap，直到有效因子集合收敛。

辅助函数
--------

.. py:function:: panel(X, y)

   面板回归，返回 ``(B, OX, T, df_params)``。

.. py:function:: fake_fund(X, B, OX)

   根据回归结果构造伪基金收益。

.. py:function:: bootstrap_fake_fund(X, B, OX, T, n_bootstraps=1000)

   对伪基金做 bootstrap 检验。

.. py:function:: work(fund, fct)

   函数式完整检验入口。

.. py:function:: parse_name_dates(s)

   解析 ``姓名(开始日期-结束日期)`` 格式字符串。

.. py:function:: get_interval(df, start=None, end=None)

   按时间索引或 ``datetime`` 列切片。

.. py:function:: gen_date_pairs(start_time, end_time, interval='1Y')

   生成滚动时间段对。
