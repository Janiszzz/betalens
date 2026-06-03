Factor API
==========

.. automodule:: betalens.factor
   :members:
   :undoc-members:
   :show-inheritance:

factor.factor
-------------

核心因子处理模块，提供可交易池获取、因子预查询、分组打标签、权重生成等功能。

数据准备函数
~~~~~~~~~~~~

.. py:function:: get_tradable_pool(date_list)

   获取可交易股票池。对每个调仓日查询交易状态为1的证券。

   :param date_list: 日期列表
   :return: (date_ranges, code_ranges) 元组

.. py:function:: pre_query_characteristic_data(date_list, metric, time_tolerance=17520, table_name="fundamental_data", date_ranges=None, code_ranges=None)

   批量预查询因子数据，生成符合因子排序函数要求的DataFrame。

   :param date_list: 调仓日期列表
   :param metric: 因子指标名称
   :param time_tolerance: 时间容差（小时），默认2年
   :param table_name: 数据库表名
   :param date_ranges: 可选，复用的日期范围
   :param code_ranges: 可选，复用的代码范围
   :return: DataFrame，包含 input_ts, code, metric, datetime, diff_hours 等列

单因子函数
~~~~~~~~~~

.. py:function:: single_characteristic(pre_queried_data, metric, quantiles)

   单因子分组打标签。

   :param pre_queried_data: 预查询的因子数据DataFrame
   :param metric: 因子指标名称
   :param quantiles: 分位数字典，如 {"ROE": 10}
   :return: 带标签的DataFrame，索引为(input_ts, code)

.. py:function:: get_single_factor_weight(labeled_pool, params)

   根据单因子标签生成多空权重。

   :param labeled_pool: 带标签的因子池
   :param params: 参数字典，包含 factor_key, mode, long, short 等
   :return: 权重DataFrame

.. py:function:: describe_labeled_pool(labeled_pool)

   描述打标签后的因子池统计信息。

   :return: 透视表，包含每个标签组的样本数和均值

双因子函数
~~~~~~~~~~

.. py:function:: double_characteristic(pre_queried_data1, pre_queried_data2, metric1, metric2, quantiles1, quantiles2, sort_method='dependent')

   双因子分组打标签（Double Sort）。

   :param pre_queried_data1: 主因子数据
   :param pre_queried_data2: 次因子数据
   :param metric1: 主因子名称
   :param metric2: 次因子名称
   :param quantiles1: 主因子分组数
   :param quantiles2: 次因子分组数
   :param sort_method: 'independent' 或 'dependent'
   :return: 带双标签的DataFrame

.. py:function:: get_double_factor_weight(labeled_pool, params)

   根据双因子标签生成多空权重。

   :param labeled_pool: 带双标签的因子池
   :param params: 参数字典，包含 factor_key1, factor_key2, mode, long_combinations, short_combinations 等
   :return: 权重DataFrame

.. py:function:: describe_double_labeled_pool(labeled_pool)

   描述双因子打标签后的统计信息。

   :return: (count_pivot, mean_pivot1, mean_pivot2) 元组

多因子函数
~~~~~~~~~~

.. py:function:: multi_characteristic(pre_queried_data_list, factors)

   多因子分组打标签（Multi-Factor Sort）。

   :param pre_queried_data_list: DataFrame列表
   :param factors: 因子配置列表，每个元素包含 name, quantiles, method
   :return: 带多标签的DataFrame

.. py:function:: get_multi_factor_weight(labeled_pool, params)

   根据多因子标签生成多空权重。

   :param labeled_pool: 带多标签的因子池
   :param params: 参数字典，包含 mode, long_combinations, short_combinations 等
   :return: 权重DataFrame

.. py:function:: describe_multi_labeled_pool(labeled_pool, max_display_dims=2)

   描述多因子打标签后的统计信息。

   :param max_display_dims: 最大显示维度
   :return: 统计信息字典

factor.preprocessing
--------------------

因子预处理模块，在 ``pre_query_characteristic_data()`` 之后、``single_characteristic()`` 之前调用。
提供截面级别的去极值、标准化、中性化，以及一键预处理流水线。

截面处理函数
~~~~~~~~~~~~

.. py:function:: winsorize_factor(factor_series, method='mad', n=3.0)

   截面去极值（单截面，index=code）。

   :param factor_series: 单截面因子值（pd.Series，index=code）
   :param method: ``'mad'`` 中位数绝对偏差（推荐）| ``'percentile'`` 百分位截尾 | ``'std'`` 均值±n倍标准差
   :param n: 阈值倍数（percentile 方法时为单侧截尾百分比，如 n=1 截 [1%, 99%]）
   :return: 去极值后的 Series

   .. code-block:: python

      >>> s = pd.Series({'A': 100, 'B': 2, 'C': 3, 'D': 1})
      >>> winsorize_factor(s, method='mad', n=3)

.. py:function:: standardize_factor(factor_series, method='zscore')

   截面标准化（单截面，index=code）。

   :param factor_series: 单截面因子值
   :param method: ``'zscore'`` (x-mean)/std | ``'rank'`` rank/N 映射到(0,1) | ``'minmax'`` 缩放到[0,1]
   :return: 标准化后的 Series

.. py:function:: neutralize_factor(factor_series, industry_labels=None, log_market_cap=None, return_stats=False)

   OLS 残差中性化（单截面）。对行业哑变量 + log(市值) 做截面 OLS，返回残差。
   行业标签既可直接传入，也可由 ``preprocess_factor(industry_scheme=...)`` 自动从 industry 表注入。

   :param factor_series: 因子值 Series，index=code（已标准化）
   :param industry_labels: 行业标签 Series，index=code（None 则跳过）
   :param log_market_cap: log 市值 Series，index=code（None 则跳过）
   :param return_stats: True 时额外返回回归诊断 dict（``n_obs`` / ``n_industry_dummies`` / ``r2`` / ``skipped``）
   :return: 残差 Series；``return_stats=True`` 时返回 ``(残差 Series, stats dict)``

   .. code-block:: python

      >>> neutralize_factor(s, industry_labels=ind, log_market_cap=lmc)
      >>> resid, stats = neutralize_factor(s, industry_labels=ind, return_stats=True)

因子对因子中性化
~~~~~~~~~~~~~~~~

.. py:function:: neutralize_factor_by_factor(factor_b_data, factor_a_data, metric_b, metric_a)

   用因子 A 对因子 B 做截面 OLS 中性化，返回残差作为"剔除 A 影响后的 B"。

   :param factor_b_data: 被解释因子的 ``pre_query_characteristic_data()`` 输出（列含 input_ts, code, {metric_b}）
   :param factor_a_data: 解释因子的输出（列含 input_ts, code, {metric_a}）
   :param metric_b: 被解释因子列名
   :param metric_a: 解释因子列名
   :return: 同 factor_b_data 格式的 DataFrame，{metric_b} 列替换为残差值

   .. code-block:: python

      >>> roe_pure = neutralize_factor_by_factor(roe_data, size_data, 'ROE', 'SIZE')
      >>> labeled = single_characteristic(roe_pure, 'ROE', quantiles={'ROE': 10})

行业中性化工具
~~~~~~~~~~~~~~

.. py:function:: query_industry_panel(pre_queried_data, scheme='申万一级行业', industry_table='industry', verbose=True)

   面板行业查询：为 ``pre_queried_data`` 的每个 (input_ts, code) 取 point-in-time 行业名。
   逐期调用 :func:`betalens.datafeed.query_industry`（datetime<=查询日 的最近一条，天然防前视），复用现有 API。

   :param pre_queried_data: 含 input_ts, code 列（``pre_query_characteristic_data()`` 的输出）
   :param scheme: 分类体系（metric），不带版本后缀时自动落到查询日生效的版本
   :param industry_table: 行业表名，默认 ``'industry'``
   :param verbose: True 时打印行业分布 / 缺失 / 面板平衡诊断
   :return: Series，MultiIndex=(input_ts, code)，值为 ind_name

   .. code-block:: python

      >>> ind_panel = query_industry_panel(pre_queried_data, '申万一级行业')
      >>> ind_panel.xs(ts)  # 取某期 code->行业

.. py:function:: filter_pool_by_industry(labeled_pool, industry_map, include_industries)

   将打标签的选股池限制在指定行业范围内。

   :param labeled_pool: ``single_characteristic()`` 的输出，MultiIndex(input_ts, code)
   :param industry_map: 行业映射表（列含 input_ts, code, industry）
   :param include_industries: 保留的行业列表，如 ``['银行', '非银金融']``
   :return: 过滤后的 labeled_pool

.. py:function:: apply_industry_weight_constraint(weights, industry_map, method='equal', target_weights=None)

   对已生成的权重矩阵施加行业权重约束。

   :param weights: ``get_single_factor_weight()`` 的输出
   :param industry_map: 行业映射表（列含 input_ts, code, industry）
   :param method: ``'equal'`` 全行业等权 | ``'market'`` 按目标比例 | ``'original'`` 不调整
   :param target_weights: method='market' 时使用，dict ``{industry: float}``
   :return: 调整后的权重 DataFrame

一键预处理流水线
~~~~~~~~~~~~~~~~

.. py:function:: preprocess_factor(pre_queried_data, metric, winsorize_method='mad', winsorize_n=3.0, standardize_method='zscore', industry_col=None, log_mktcap_col=None, industry_scheme=None, industry_table='industry', verbose=True)

   逐截面（按 input_ts）依次执行：去空值 → 去极值 → 标准化 → 中性化。

   行业标签来源：``industry_scheme`` 给定（推荐）则自动从 industry 表 point-in-time 查询，
   并打印行业分布 / 缺失 / 面板平衡及中性化执行摘要；``industry_col`` 给定则用调用方预先
   merge 的行业列（旧行为）。市值中性化仍由 ``log_mktcap_col`` 手动提供。

   :param pre_queried_data: ``pre_query_characteristic_data()`` 的输出
   :param metric: 因子列名
   :param winsorize_method: ``'mad'`` | ``'percentile'`` | ``'std'``
   :param winsorize_n: 去极值阈值
   :param standardize_method: ``'zscore'`` | ``'rank'`` | ``'minmax'``
   :param industry_col: pre_queried_data 中的行业列名（None 跳过）
   :param log_mktcap_col: pre_queried_data 中的 log 市值列名（None 跳过市值中性化）
   :param industry_scheme: 自动查 industry 表的分类体系名，如 ``'申万一级行业'``；给定即自动注入行业标签并打印诊断（优先于 industry_col）
   :param industry_table: 行业表名，默认 ``'industry'``
   :param verbose: True 时打印行业诊断与中性化执行摘要
   :return: 同 pre_queried_data 格式的 DataFrame，{metric} 列已替换为处理后的值

   .. code-block:: python

      >>> # 自动查表 + 诊断打印（推荐）
      >>> cleaned = preprocess_factor(raw_data, 'ROE', industry_scheme='申万一级行业')
      >>> # 旧用法：调用方自带行业列
      >>> cleaned = preprocess_factor(raw_data, 'ROE', industry_col='industry')
      >>> labeled = single_characteristic(cleaned, 'ROE', quantiles={'ROE': 10})

factor.stats
------------

因子统计检验模块，提供 IC/ICIR 分析、Fama-MacBeth 截面回归、分组收益统计。

IC / ICIR
~~~~~~~~~

.. py:function:: calc_ic(factor_data, return_data, method='spearman')

   逐截面计算 IC（Information Coefficient）。

   :param factor_data: 宽表（pd.DataFrame），index=input_ts，columns=code，值为因子值
   :param return_data: 宽表，index=input_ts，columns=code，值为持仓期收益率
   :param method: ``'spearman'`` Rank IC（推荐）| ``'pearson'`` 普通 IC
   :return: pd.Series，index=input_ts，name='IC'

   .. code-block:: python

      >>> ic = calc_ic(factor_wide, return_wide)
      >>> print(ic.mean(), ic.std())

.. py:function:: calc_icir(ic_series, window=None)

   计算 ICIR = mean(IC) / std(IC)。

   :param ic_series: ``calc_ic()`` 的输出
   :param window: None 返回全样本 float；整数返回滚动 Series
   :return: float（全样本）或 pd.Series（滚动）

   .. code-block:: python

      >>> icir = calc_icir(ic)                   # 全样本
      >>> rolling_icir = calc_icir(ic, window=12)  # 滚动12期

.. py:function:: summarize_ic(ic_series)

   IC 统计摘要。

   :param ic_series: IC 序列
   :return: dict，包含：

      - ``IC均值``: float
      - ``IC_std``: float
      - ``ICIR``: float
      - ``胜率(IC>0)``: float
      - ``t统计量``: float
      - ``p值``: float

   .. code-block:: python

      >>> summary = summarize_ic(ic)
      >>> pd.Series(summary)

Fama-MacBeth 截面回归
~~~~~~~~~~~~~~~~~~~~~

.. py:function:: fama_macbeth(factor_data_dict, return_data, industry_dummies=None)

   Fama-MacBeth 两步法截面回归。

   第一步：每截面期 t 做 OLS ``R_i = α_t + Σ(λ_k,t * F_k,i) + ε_i``

   第二步：对 λ_k 时间序列做 t 检验。

   :param factor_data_dict: dict，格式为 {因子名: 宽表 DataFrame}（index=date, columns=code）
   :param return_data: 宽表（index=date, columns=code），持仓期超额收益
   :param industry_dummies: 可选，行业哑变量宽表（控制变量，不纳入 λ 报告）
   :return: pd.DataFrame，index=factor_name，columns=['lambda_mean', 'lambda_std', 't_stat', 'p_value', 'n_periods']

   .. code-block:: python

      >>> fm = fama_macbeth({'ROE': roe_wide, 'PE': pe_wide}, return_wide)
      >>> print(fm[['lambda_mean', 't_stat']])

分组收益统计
~~~~~~~~~~~~

.. py:function:: group_return_summary(labeled_pool, return_data, metric)

   计算各分组在持仓期内的等权平均收益。

   :param labeled_pool: ``single_characteristic()`` 输出，MultiIndex(input_ts, code)，含 {metric}_label 列
   :param return_data: 宽表（index=input_ts, columns=code），持仓期收益率
   :param metric: 因子名（用于找标签列 ``{metric}_label``）
   :return: pd.DataFrame，index=input_ts，columns=['G1'...'GN', 'long_short']

   ``long_short`` = G_max - G_min（自动判断因子方向）

   .. code-block:: python

      >>> gr = group_return_summary(labeled_pool, return_wide, 'ROE')
      >>> gr.cumsum().plot()


