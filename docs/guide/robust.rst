稳健性检验
==========

`betalens.robust` 以 Harvey & Liu (2021) "Lucky Factors" 为蓝本，帮助判断新因子的增量价值，对抗过拟合。

模块定位
--------

稳健性检验模块处于因子研究 pipeline 的最下游，与因子挖掘模块高度隔离。其核心功能是：

- 输入多个因子收益序列
- 分析某个因子中性化之后的有效程度
- 输出检验结果，识别新因子被哪几个老因子解释

RobustTest 流程
---------------

.. code-block:: python

   import pandas as pd
   from betalens.robust import RobustTest

   # 准备数据
   fund_returns = pd.Series(..., name="fund").dropna()
   factor_panel = pd.DataFrame({
       'factor_1': [...],
       'factor_2': [...],
       'factor_3': [...]
   }, index=dates).dropna(how="all")

   # 创建检验实例
   test = RobustTest(fund=fund_returns, factor=factor_panel)

   # 步骤1: 去相关（正交化）
   orthogonal_factors, t_values = test.neu()

   # 步骤2: Bootstrap检验
   eff_names, pvalues, pdf = test.bootstrap_once(n_bootstraps=500)

核心步骤详解
------------

1. **构造样本**

   - 拼接基金/组合收益与候选因子
   - 错位一行用于预测下一期收益（因子滞后一期）

2. **去相关（neu方法）**

   - 逐个对每个因子进行单因子回归
   - 得到正交化残差 ``OX`` 
   - 计算原始 t 统计量 ``T``

3. **Bootstrap重采样**

   - 重复抽样 n_bootstraps 次
   - 计算最大统计量分布
   - 校正多重检验偏差

4. **过滤有效因子**

   - 根据修正后的 p 值（modifd_P）筛选
   - 常用阈值：0.1 / 0.05

结果解读
--------

.. code-block:: python

   # OX: 正交化因子矩阵
   print(test.OX)

   # T: 原始 t 值
   print(test.T)

   # modifd_P: bootstrap 后的显著性水平
   eff_names, modifd_P, max_statistic_pdf = test.bootstrap_once()
   print(modifd_P)

   # eff_fct_name: 通过显著性检验的因子集合
   print(eff_names)

面板回归（Panel）
-----------------

模块还提供面板回归工具：

.. code-block:: python

   from betalens.robust.robust import panel, bootstrap_fake_fund

   # 面板回归，单测alpha
   B, OX, T, df_params = panel(X, y)

   # Bootstrap检验
   modifd_P, max_statistic_pdf = bootstrap_fake_fund(X, B, OX, T, n_bootstraps=500)

辅助工具
--------

.. code-block:: python

   # 生成示例数据
   asset_returns, factor_values = RobustTest.create_sample_dataframes()

   # 解析基金经理任期字符串
   from betalens.robust.robust import parse_name_dates
   info = parse_name_dates('盛丰衍(20180711-20250101)')
   # {'name': '盛丰衍', 'start_date': datetime(...), 'end_date': datetime(...)}

   # 生成滚动时间段
   from betalens.robust.robust import gen_date_pairs
   pairs = gen_date_pairs('2020-01-01', '2024-12-31', interval='1Y')

并行 & 扩展
-----------

- ``bootstrap_once`` 内部使用 :class:`concurrent.futures.ThreadPoolExecutor`，自动并行化
- 可根据 CPU 数量调整 ``n_bootstraps``（建议 ≥ 500）
- 对于海量因子面板，可分块调用 ``RobustTest``

最佳实践
--------

**数据准备**

- 输入数据需对齐日期并清除缺失值
- 避免线程池反复抛出异常

**参数设置**

- Bootstrap 迭代次数建议大于 500，以获得平滑的分布
- 显著性阈值根据研究需求选择（0.1 宽松，0.05 严格）

**批量评估**

.. code-block:: python

   # 对多个基金/策略批量检验
   results = []
   for fund_name in fund_list:
       test = RobustTest(fund=returns[fund_name], factor=factor_panel)
       test.neu()
       eff_names, modifd_P, _ = test.bootstrap_once()
       results.append({
           'fund': fund_name,
           'effective_factors': list(eff_names),
           'p_values': modifd_P.to_dict()
       })

**滚动窗口检验**

.. code-block:: python

   from betalens.robust.robust import gen_date_pairs, get_interval

   # 生成滚动时间段
   intervals = gen_date_pairs('2015-01-01', '2024-12-31', interval='1Y')

   results = []
   for start, end in intervals:
       fund_subset = get_interval(fund_returns, start, end)
       factor_subset = get_interval(factor_panel, start, end)
       
       test = RobustTest(fund=fund_subset, factor=factor_subset)
       test.neu()
       eff_names, modifd_P, _ = test.bootstrap_once()
       results.append({
           'period': f"{start}~{end}",
           'effective_factors': list(eff_names)
       })

更多 API 细节请查看 :doc:`../api/robust`。


