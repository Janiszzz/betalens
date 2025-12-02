稳健性检验
==========

`betalens.robust` 以 Harvey & Liu (2021) “Lucky Factors” 为蓝本，帮助判断新因子的增量价值。

RobustTest 流程
---------------

.. code-block:: python

   import pandas as pd
   from betalens.robust import RobustTest

   fund_returns = pd.Series(..., name="fund").dropna()
   factor_panel = pd.DataFrame(...).dropna(how="all")

   test = RobustTest(fund=fund_returns, factor=factor_panel)
   orthogonal_factors, t_values = test.neu()

   names, pvalues, pdf = test.bootstrap_once(n_bootstraps=500)

核心步骤

1. **构造样本**：拼接基金/组合收益与候选因子，错位一行用于预测下一期收益
2. **去相关**（``neu``）：逐个单因子回归，得到正交化残差 ``OX`` 以及 t 统计量
3. **Bootstrap**：重复抽样，计算最大统计量分布，从而校正多重检验偏差
4. **过滤**：根据 ``modifd_P``（修正后的 p 值）筛选通过阈值的因子

并行 & 扩展
-----------

- ``bootstrap_once`` 内部使用 :class:`concurrent.futures.ThreadPoolExecutor`，可根据 CPU 数量调整 ``n_bootstraps``
- ``create_sample_dataframes`` 提供随机数据集，可在单元测试中快速复现
- 如果要对基金经理任期、滚动区间做批量评估，可参考源码末尾的 ``parse_name_dates``、``gen_date_pairs`` 辅助函数

结果解读
--------

- ``OX``：正交化因子矩阵，可继续输入其它统计模型
- ``T``：原始 t 值，按基金名称索引
- ``modifd_P``：bootstrap 后的显著性水平；常用阈值 0.1 / 0.05
- 返回的 ``eff_fct_name`` 即通过显著性检验的因子集合

最佳实践
--------

- 输入数据需对齐日期并清除缺失值，避免线程池反复抛出异常
- Bootstrap 迭代次数建议大于 500，以获得平滑的分布
- 对海量因子面板，可在外层加入协程/分布式切片，分块调用 ``RobustTest``

更多 API 细节请查看 :doc:`../api/robust`。


