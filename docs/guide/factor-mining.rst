因子参数挖掘
============

``betalens.factor.mining`` 提供两阶段参数挖掘入口。参数空间、执行模式、
滑动窗口和筛选规则放在 ``parameter_space.yaml``；进程数、缓存和输出目录
放在 ``performance.yaml``。

入口
----

.. code-block:: python

   from betalens.factor.mining import run_mining

   result = run_mining(
       "betalens-factor/alpha101/mining/parameter_space.yaml",
       "betalens-factor/alpha101/mining/performance.yaml",
   )

``result.launch_id`` 标识本次启动；``result.factor_runs`` 为逐因子结果。每个
``FactorMiningResult`` 提供 ``factor_id``、``run_id``、``run_dir``、运行状态、
粗细搜窗口结果、候选汇总和最终赢家。

截面预处理可在 ``parameter_space.yaml`` 顶层统一配置，也可在单个 ``factors``
条目中覆盖：

.. code-block:: yaml

   preprocessing:
     winsorize_method: none       # none | mad | percentile | std
     winsorize_n: 3.0
     standardize_method: zscore  # none | zscore | rank | minmax
     neutralize: true

``neutralize`` 只控制 OLS 是否执行，行业和市值控制变量仍由因子 ``FactorSpec`` 的
``use_industry`` / ``use_mktcap`` 决定。未配置时沿用因子原有预处理行为。

搜索流程可以包含五个阶段：QMC 在宽边界内均匀覆盖数量级，TPE 根据已完成批次
向优良区域收敛；若优胜候选持续集中于同一参数边界，则扩大该侧边界并启动新的
Optuna study；之后在优胜候选邻域生成细粒度 Grid，最后对赢家参数施加随机扰动
验证局部稳定性。Optuna 只负责分布和 trial，worker 只接收普通参数字典，不负责
缓存或进程调度。

Alpha101 自动参数空间
----------------------

Alpha101 可以在 ``parameter_space.yaml`` 中逐因子显式声明参数空间；也可以将
``factors`` 设为 ``all``，由 ``alpha101_parameters.aggregate_mining_factors()``
为 ALPHA1 至 ALPHA101 生成参数定义。显式参数映射不会调用自动生成逻辑；显式
因子写成 ``parameters: auto`` 时会只为该因子调用自动生成逻辑。

一次挖掘多个 Alpha 时，可以直接使用名称列表。每个名称会自动补齐
``alpha101_mining``、``precomputed`` 和自动参数空间：

.. code-block:: yaml

   factors: [ALPHA3, ALPHA7, ALPHA8]

需要逐因子覆盖配置时，也支持带 ``id`` 的映射列表：

.. code-block:: yaml

   factors:
     - id: ALPHA3
       module: alpha101_mining
       execution_mode: precomputed
       parameters: auto

自动生成首先读取 ``alpha101_formulas.py`` 中各 ``alphaN`` 函数的关键字默认值，
再根据参数名后缀分类。``*_window``、``*_lag``、``*_threshold``、
``*_exponent`` 和 ``*_weight`` 可搜索；其他参数固定为论文默认值。默认按照公式
签名顺序最多放开前五个可搜索参数，可通过 ``max_dimensions`` 调整。

.. list-table:: 论文默认值 ``d`` 周围的参考点
   :header-rows: 1

   * - 参数种类
     - 参考点规则
   * - 整数 window / lag
     - ``[d/m, d, d*m]``，并保证不小于 1
   * - 浮点 window / lag
     - ``[d/m, d, d*m]``
   * - weight
     - 默认搜索类型级硬边界定义的完整 ``[0, 1]``
   * - threshold
     - ``[d-s, d, d+s]``，其中 ``s=max(abs(d), 0.05)*m``
   * - exponent
     - 保持符号并按 ``[abs(d)/m, abs(d)*m]`` 扩展
   * - 其他
     - ``[d]``，即固定值

这些参考点只用于计算 ``low`` 和 ``high``，不是离散候选集合。``m`` 是
``range_multiplier``，默认 10。例如 ALPHA3 的相关系数窗口默认值为
10，自动得到 ``low=1``、``high=100``、``step=1`` 和 ``scale=log``；QMC 可以
覆盖 1 至 100 的不同数量级。``type_limits`` 是不可突破的类型级硬边界，默认
window 为 1 至 1260、lag 为 1 至 504、weight 为 0 至 1、threshold 为 -10 至
10、exponent 为 0.01 至 100。该规则不读取历史数据，也不保证得到统计意义上的
最优搜索范围。

显式 Alpha 配置可写 ``parameters: auto``；``factors: all`` 则为全部 Alpha 自动
生成空间。两种写法都从顶层 ``alpha101_parameter_generation`` 读取：

.. code-block:: yaml

   alpha101_parameter_generation:
     range_multiplier: 10
     max_dimensions: 5
     type_limits:
       window: {low: 1, high: 1260}
       lag: {low: 1, high: 504}

宽搜、收敛与扩边
------------------

``sampler: qmc`` 使用 Optuna ``QMCSampler``，``qmc_type`` 可选 ``sobol`` 或
``halton``。Sobol 的 ``n_trials`` 宜使用 2 的幂。QMC 只改善给定边界内的覆盖，
不会自行修改 ``low/high``。

``search.refine`` 使用新的 TPE study，先将前序宽搜最好的 ``bootstrap_top_k`` 个
completed trial 及真实目标值导入为历史，再按 ``batch_size`` 评价并回传目标值后
生成下一批候选。``search.expansion`` 检查前 ``boundary_top_k`` 个候选；当至少
``winner_ratio`` 的候选位于参数轴同一端 ``boundary_tolerance`` 范围内时，只向
该侧扩大 ``range_multiplier`` 倍。每轮扩边都创建新 study，最多运行
``max_rounds`` 轮，并受 ``parameter_limits`` 约束。

赢家扰动验证
------------

``search.stability`` 在前 ``top_k`` 名赢家附近生成随机参数误差。线性参数按完整
跨度的 ``radius_ratio`` 扰动，对数参数在对数轴扰动，整数会重新按 step 对齐，
categorical 和 bool 保持赢家值。扰动候选照常经历全部滑动窗口评价。

当有效扰动比例达到 ``minimum_valid_ratio``，且至少 ``required_pass_ratio`` 的
扰动目标值退化不超过 ``max_objective_degradation`` 时，赢家标记为 ``stable``。
``require_pass: false`` 只做审计标注；设为 true 才会剔除未通过者。详细结果写入
审计工作簿的“稳定性验证”、候选汇总和窗口表现。

参数定义可用字段如下：

* ``type``：``int`` / ``float`` / ``categorical`` / ``bool``；
* ``low`` / ``high``：数值参数的闭区间边界；
* ``step``：正数步长；对数整数只能为 1，对数浮点不能设置；
* ``scale``：``linear`` / ``log``，使用 ``log`` 时 ``low`` 必须大于 0；
* ``choices``：categorical 的非空候选列表；bool 省略时默认为 ``[false, true]``。

因子 hook
---------

因子模块提供 ``make_mining_spec(params) -> MiningSpec``：

* ``precomputed``：完整区间只计算一次因子；没有窗口变换时完整净值只回测一次，
  各窗口表现从净值切片得到。逐调仓期 Rank IC 也按唯一的
  ``(信号日, 调仓日, 下一调仓日)`` 预计算一次，重叠窗口只做序列切片和聚合。
* ``rolling_fit``：通过 ``fit_window`` 为每个窗口独立拟合并生成权重。
* ``window_transform``：可在不重算因子的情况下做窗口级权重变换，例如 DISP 的
  pretom 过滤。

缓存
----

``cache.data_enabled: true`` 时，每个因子任务在自己的 ``cache/datasets`` 中使用
NPY/memmap 保存行情、成交价、PIT、交易状态和行业面板；重叠窗口只加载时间切片。
``cache/input_manifest.json`` 描述输入缓存，``cache/results.sqlite3`` 增量保存搜索
进度和计算结果。关闭数据缓存时直接查询数据库，但结果数据库仍会生成。

多进程运行时，实际 worker 数还会受物理 CPU 核数和内存预算约束。框架为新启动的
worker 默认设置单线程 BLAS；如果调用环境已显式设置 ``OPENBLAS_NUM_THREADS``、
``OMP_NUM_THREADS``、``MKL_NUM_THREADS`` 或 ``NUMEXPR_NUM_THREADS``，则保留调用方设置。

输出
----

每个因子、每次运行使用独立目录：

.. code-block:: text

   outputs/mining/ALPHA3/20260819_220000_a1b2c3d4/
     metadata.yaml
     cache/
       input_manifest.json
       datasets/
       results.sqlite3
     audit/
       挖掘审计.xlsx
       运行日志.log

``metadata.yaml`` 记录配置指纹、环境、Git 状态、运行状态、结果计数和异常。
``挖掘审计.xlsx`` 包含运行概览、参数空间、搜索进度、全部窗口表现、候选汇总、
IC 分析、赢家参数、赢家汇总和错误。``IC分析`` sheet 同时记录窗口 IC IR 和逐期
Rank IC 序列；IC IR 按窗口内逐期 Rank IC 的均值除以样本标准差计算，不额外年化。
旧版扁平 CSV/Parquet、``run_manifest.json``、
``selected_candidates.yaml`` 和 ``selected_nav`` 不再生成。

热力图报告
-----------

任务结束时，审计目录会按“滑动窗口组合 × 参数对”生成 total 热力图，以及
``热力图报告.json``。例如：

::

   热力图_total_252日窗口_21日步长_参数对01.png

所有可变参数自动做两两组合；三个参数会产生三个参数对。每张图包含夏普比率、
年化收益率、卡玛比率和最大回撤四个面板，单元格取相同参数坐标下全部窗口实例和
搜索阶段记录的平均值。粗搜、自适应收敛、扩边、细搜及稳定性验证均纳入统计，
但时间区间只展示完整 ``evaluation.span``（total），不再生成 train/test/valid 图。

单参数因子降级为“参数 × 候选”热力图；完全没有参数变化时生成单值图。
``热力图报告.json`` 记录窗口长度、滑动步长、参数轴、聚合方法和实际图像路径；
图像路径同时记录在 ``metadata.yaml`` 的 ``heatmap_paths`` 和
``FactorMiningResult.heatmap_paths``。
