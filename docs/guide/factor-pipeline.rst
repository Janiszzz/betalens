因子脚本与 YAML 管线
===================

``betalens-factor/`` 存放可运行因子。因子脚本只声明算子和配置转换，取数、分组、权重、回测、评价由通用 ``FactorPipeline`` 负责。

目录结构
--------

.. code-block:: text

   betalens-factor/
     factor_template.py
     <factor_class>/
       class_<factor_class>.yaml
       factor_template_<factor_class>.py
       mining/
         parameter_space.yaml
         performance.yaml
         run.py
       <FACTOR_NAME>/
         factor_<FACTOR_NAME>.py
         factor_<FACTOR_NAME>.yaml
   outputs/                           # 仓库根目录
     runs/<class>/<factor>/<run_id>/
     mining/<class>/<factor>/<run_id>/
     experiments.lmdb/
     cache/dashboard/<run_id>/

规则：

* 类目录必须有 ``class_<class>.yaml``。
* 因子目录必须有 ``factor_<name>.py`` 和 ``factor_<name>.yaml``。
* Dashboard 按 ``<class>/<name>/factor_<name>.py`` 动态导入脚本。
* 新运行写入仓库级 ``outputs/runs/<class>/<factor>/<run_id>/``，每次使用独立目录；旧产物保留原位。
* 参数挖掘在类目录的 ``mining/`` 中集中配置，单因子 YAML 不包含挖掘参数。

YAML 结构
---------

每个脚本读取同目录同 stem 的完整 YAML。例如 ``factor_RSI_FAST.py`` 读取 ``factor_RSI_FAST.yaml``。

.. code-block:: yaml

   meta:
     class: tdx
     name: RSI_FAST
     source: 来源
     formula: 公式
     logic: 逻辑
   factor_spec:
     inputs:
       close_wide: close
     industry_inputs: {}
     compute_kwargs:
       window: 4
     direction: positive
     table_name: market_daily_post_adjusted
     query_filters: {asset_type: stock}
     index_code: 000906.SH
     use_industry: false
     use_mktcap: false
     winsorize_method: none
     winsorize_n: 3.0
     standardize_method: none
     neutralize: false
     industry_scheme: {scheme_code: null, level: 1}
     backtest_metric: close
     required_history_bars: 0
     mask_inputs_by_pit: false
   weight:
     mode: freeplay
     long_groups: null
     short_groups: null
     group_weights: {}
     intra_group_allocation: {}
   run:
     start_date: '2024-01-01'
     end_date: '2024-12-31'
     rebal_freq: W
     grouping_mode: equal_count
     n_quantiles: 20
     initial_amount: 100000000
     include_profiling: true
     dump_excel: true
     warmup_days: null
     output_dir: ../../../outputs/runs/tdx/RSI_FAST

不做类级、因子级、脚本级多层覆盖；具体运行以完整 YAML 为唯一参数源。Dashboard 提交后会生成本次运行的 ``run_config.yaml``，再用这份文件构造 ``FactorSpec`` 和 run 参数。

脚本最小接口
------------

.. code-block:: python

   from pathlib import Path
   from betalens.factor.config import (
       load_yaml_config,
       factor_spec_options,
   )
   from factor_template import FactorPipeline, FactorSpec

   _CONFIG_FILE = Path(__file__).with_suffix(".yaml")

   def load_config(path: str | Path = _CONFIG_FILE) -> dict:
       return load_yaml_config(path, required_sections=("meta", "factor_spec", "weight", "run"))

   def compute_my_factor(close_wide, window):
       return close_wide.pct_change(window)

   def build_spec(config: dict, config_path: str | Path = _CONFIG_FILE) -> FactorSpec:
       options = factor_spec_options(config, config_path)
       return FactorSpec(
           name=config["meta"]["name"],
           compute=compute_my_factor,
           **options,
       )

   spec = build_spec(load_config())

   def run_from_config(config_path: str | Path = _CONFIG_FILE):
       from factor_runtime import run_factor_from_config
       return run_factor_from_config(config_path, build_spec, FactorPipeline)

要求：

* import 时不跑回测、不写文件。
* 暴露 ``spec``、``FactorPipeline``、``build_spec``。
* ``compute`` 的参数名必须覆盖 ``factor_spec.inputs`` 的 key，并接收 ``compute_kwargs``。
* CLI 只需要支持 ``--config PATH`` 选择完整 YAML。

FactorSpec
----------

``FactorSpec`` 描述一个因子的运行口径：

* ``name``：因子名和输出前缀。
* ``inputs``：``{算子参数名: 数据库 metric}``。
* ``industry_inputs``：可选的 ``{算子参数名: {scheme_code: null, level: 3}}``；用于公式内部的行业中性化。
* ``compute``：宽表算子函数。
* ``required_history_bars``：公式累计需要的交易日历史根数；未显式设置 ``warmup_days`` 时用于自动推断取数起点。
* ``mask_inputs_by_pit``：为 ``true`` 时，在公式计算前逐交易日按 ``index_code`` 的 PIT 成分掩码过滤所有输入。
* ``direction``：``positive`` 高分组做多，``negative`` 低分组做多。
* ``table_name``：输入数据表，常用 ``market_daily_post_adjusted``。
* ``index_code``：PIT 指数成分过滤。
* ``winsorize_method``：截尾方法，支持 ``none``、``mad``、``percentile``、``std``；``winsorize_n`` 为对应阈值。
* ``standardize_method``：标准化方法，支持 ``none``、``zscore``、``rank``、``minmax``。
* ``neutralize``：中性化总开关；``use_industry`` / ``use_mktcap`` 决定使用行业或市值控制变量。
* ``weight_mode``、``long_groups``、``short_groups``：权重生成口径。
* ``backtest_metric``：回测成交价指标。

``run.grouping_mode`` 支持两种截面分组口径：

* ``equal_count``：优先保证每组股票数相当，严格生成 ``n_quantiles`` 组；相同特征值可能跨组。
* ``value``：相同特征值绝不跨组，重复分位边界会合并，因此实际组数可能少于 ``n_quantiles``。

``classic-long-short`` 始终逐期选择实际最大/最小标签。``freeplay`` 在
``equal_count`` 下可使用数值标签；在 ``value`` 下必须使用 ``max`` / ``min``，
以避免实际组数变化时数值标签含义漂移。Profiling Excel 会额外输出
``group_balance_by_date`` 和 ``group_balance_summary``，展示组间数量差异、
同值边界和相邻组特征值区分度。

``FactorPipeline.run(..., warmup_days=N)`` 可用日历日显式覆盖自动预热；省略或在 YAML 中设为 ``null`` 时，管线根据 ``required_history_bars`` 与算子参数推断。旧 YAML 无需增加上述可选字段：曾启用行业/市值中性化的配置继续默认执行 ``mad(3) → zscore → neutralize``，未启用中性化的配置继续默认不做预处理。

日频数据与研究区间
------------------

通用、LiqDemand 和 Alpha101 模板通过 ``query_daily_market`` 读取业务日行情，
以 ``trade_date`` 对齐输入，保留 ``available_at`` 的可得时间元数据。
``research_schedule`` 明确记录 ``signal_date``、``signal_ts``、``target_ts``
和 ``forward_end_ts``：上一交易日信号映射到本次调仓日 00:10 的目标，
随后按所选价格字段匹配实际执行点。输入可得时间晚于目标时点时不提交该次新目标，
并在 schedule 中记录原因。

分组研究读取完整因子样本的价格。分组在研究调仓点等权、期间自然漂移，
不使用主策略的费用、手数和风控；缺少成员价格时保留缺失并输出覆盖率。
IC 使用原始研究调仓之间的显式区间收益，不从含风控的成交事件推导下一期。

Alpha101 目录
-------------

``betalens-factor/alpha101/alpha101_formulas.py`` 是 101 个论文公式、输入依赖与精确回看长度的注册表。每个因子的截面版和择时版脚本/YAML 由 ``tools/generate_catalog.py`` 生成；CI 或本地校验使用：

.. code-block:: powershell

   python betalens-factor\alpha101\tools\generate_catalog.py --check

择时版以中证 800 日频 PIT 截面作为 Alpha101 排名和行业中性化的公式上下文，并始终把 YAML 中 ``stock_code`` 指定的股票加入计算截面；指数成分资格不限制目标股票产生信号。目标股票自身数据或因子值无效时目标仓位为零。若请求开始日期早于目标股票全部必要行情字段的共同起点，管线自动将有效开始日期后移，并在运行日志中提示请求日期、目标代码和调整后日期。默认 ``rolling_z`` 信号使用过去 120 个有效因子观测的历史上轨、1 倍标准差、满仓做空，目标时间沿用调仓日 00:10 的共同映射。

RunResult
---------

``FactorPipeline.run`` 返回 ``RunResult``，可继续兼容旧解包：

.. code-block:: python

   result = FactorPipeline(spec).run("2024-01-01", "2024-12-31")
   bt = result.backtest
   analyst = result.analyst
   profiling = result.profiling

   bt2, analyst2 = result

常见增量产物包括 ``profiling``、``neutralize_stats``、``factor_values``、``pit_validation``。

Dashboard 发现规则
------------------

Dashboard 扫描类级和因子级 YAML，读取 ``meta``、``factor_spec``、``weight``、``run`` 默认值，并动态加载脚本 docstring 和 ``compute_kwargs``。新增因子后用 ``GET /api/factors?refresh=true`` 清缓存。


实验入库与运行合同
------------------

所有因子脚本的 ``run_from_config`` 与 Dashboard 共用 ``factor_runtime``，保持
``RunResult`` 及 ``bt, analyst = result`` 解包方式。直接调用管线仍返回计算结果；
配置驱动入口负责运行目录、配置快照与可选入库。

.. code-block:: yaml

   experiment_export:
     enabled: false
     store_path: outputs/experiments.lmdb

旧 YAML 缺少此节时默认关闭。``enabled`` 必须是布尔值；关闭时不打开实验库、
不检查或加载 LMDB 可选依赖。开启前需安装 ``betalens[experiments]``。
实验库相对路径固定按仓库根解析，与当前工作目录和 YAML 所在目录无关。
``run.output_dir`` 继续按源 YAML 所在目录解析；库内默认目录会追加独立运行 ID，
显式自定义目录保持有效。旧 ``outputs/runs/manual`` 默认值也会迁移至集中目录。

运行目录保存 ``run_config.yaml`` 和 ``experiment_export.json``。入库失败时计算结果
和报告继续保留，回执及 Dashboard 分别显示计算状态和入库状态。
母因子键统一为 ``<class>:<name>:<strategy_type>``；截面类型归一为
``cross_sectional``。同一运行 ID 重复保存相同内容返回相同实验 ID，内容不一致报冲突。
有效配置、因子库源码内容哈希及回测计算版本随实验保留。

输入面板的行是上海本地交易日，列是唯一证券代码；日期和证券轴共同对齐，
缺失值保留 NaN。重复业务日/证券键及 compute 输出轴不一致直接报错。
Alpha101 的数据库 ``pct_change`` 单位为百分数，公式入口仅除以 100 一次。
截面 freeplay 的空多空分组在取数前拒绝；择时不受这一分组约束。
行业继续使用显式版本表来源和 ``scheme_code/level``，不切换数据源掩盖覆盖缺口。
LiqDemand 复用公共预处理、回测和评价步骤，PreTOM 按完整月份确定窗口，
即使用户仅回测某月的一部分也保持月末定义一致。

后期批量跟踪接口
----------------

本次只保留以下设计约定，不提交空实现或调度任务：

* ``run_batch(batch_config)``：复用统一执行入口，逐因子记录状态与重试。
* ``resume_tracking(tracking_id, end_date)``：关联连续策略实验。
* ``checkpoint_ref``、``data_watermark``：引用计算检查点和输入数据边界。

真正续算需要回测引擎公开现金、持仓、未完成订单、风控及指标状态的导出/恢复合同，
并通过分段续算与全量运行的一致性验收。当前引擎从初始现金启动，本次未增加续算能力。


本地验收边界（2026-11-12）
--------------------------

209 个条目的配置、导入和合成面板计算通过；单次/挖掘入库、Dashboard 执行及离线
HTTP 回读均使用隔离实验库验证。两个前端通过类型检查与生产构建。
真实 PostgreSQL 查询返回 2026-10-12 至 10-16、两只股票的 5×2 行情面板。
完整真实回测尚未验收通过：较长预热区间缺少 SSE 日历覆盖，缩短范围后
在 ``query_trade_status → load_reference`` 的 PostgreSQL 参考版本读取处超时。
验收未修改数据库、核心包或可信范围规则。XICHOU 原始脚本仍缺失，外围挖掘入口
会在搜索前报告具体依赖，通用发现测试使用可用条目和自包含样本。
