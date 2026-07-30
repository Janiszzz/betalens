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
       <FACTOR_NAME>/
         factor_<FACTOR_NAME>.py
         factor_<FACTOR_NAME>.yaml
         outputs/
           legacy/
           runs/
         mining/
           sweep_window.py
           sweep_window.yaml

规则：

* 类目录必须有 ``class_<class>.yaml``。
* 因子目录必须有 ``factor_<name>.py`` 和 ``factor_<name>.yaml``。
* Dashboard 按 ``<class>/<name>/factor_<name>.py`` 动态导入脚本。
* 新运行写入 ``outputs/runs/<run_id-or-manual>/``。
* 参数挖掘入口放在 ``mining/``。

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
       close_wide: 收盘价(元)
     industry_inputs: {}
     compute_kwargs:
       window: 4
     direction: positive
     table_name: daily_market
     index_code: 000906.SH
     use_industry: false
     use_mktcap: false
     industry_scheme: 申万一级行业
     backtest_metric: 收盘价(元)
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
     n_quantiles: 20
     initial_amount: 100000000
     include_profiling: true
     dump_excel: true
     warmup_days: null
     output_dir: outputs/runs/manual

不做类级、因子级、脚本级多层覆盖；具体运行以完整 YAML 为唯一参数源。Dashboard 提交后会生成本次运行的 ``run_config.yaml``，再用这份文件构造 ``FactorSpec`` 和 run 参数。

脚本最小接口
------------

.. code-block:: python

   from pathlib import Path
   from betalens.factor.config import (
       load_yaml_config,
       factor_spec_options,
       run_parameters,
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
       config = load_config(config_path)
       kwargs = run_parameters(config, config_path)
       start = kwargs.pop("start_date")
       end = kwargs.pop("end_date")
       return FactorPipeline(build_spec(config, config_path)).run(start, end, **kwargs)

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
* ``industry_inputs``：可选的 ``{算子参数名: PIT 行业分类体系}``；用于公式内部的行业中性化。
* ``compute``：宽表算子函数。
* ``required_history_bars``：公式累计需要的交易日历史根数；未显式设置 ``warmup_days`` 时用于自动推断取数起点。
* ``mask_inputs_by_pit``：为 ``true`` 时，在公式计算前逐交易日按 ``index_code`` 的 PIT 成分掩码过滤所有输入。
* ``direction``：``positive`` 高分组做多，``negative`` 低分组做多。
* ``table_name``：输入数据表，常用 ``daily_market``。
* ``index_code``：PIT 指数成分过滤。
* ``use_industry`` / ``use_mktcap``：行业和市值中性化。
* ``weight_mode``、``long_groups``、``short_groups``：权重生成口径。
* ``backtest_metric``：回测成交价指标。

``FactorPipeline.run(..., warmup_days=N)`` 可用日历日显式覆盖自动预热；省略或在 YAML 中设为 ``null`` 时，管线根据 ``required_history_bars`` 与算子参数推断。旧 YAML 无需增加上述可选字段。

Alpha101 目录
-------------

``betalens-factor/alpha101/alpha101_formulas.py`` 是 101 个论文公式、输入依赖与精确回看长度的注册表。每个因子的截面版和择时版脚本/YAML 由 ``tools/generate_catalog.py`` 生成；CI 或本地校验使用：

.. code-block:: powershell

   python betalens-factor\alpha101\tools\generate_catalog.py --check

择时版以中证 800 日频 PIT 截面作为 Alpha101 排名和行业中性化的公式上下文，并始终把 YAML 中 ``stock_code`` 指定的股票加入计算截面；指数成分资格不限制目标股票产生信号。目标股票自身数据或因子值无效时目标仓位为零。若请求开始日期早于目标股票全部必要行情字段的共同起点，管线自动将有效开始日期后移，并在运行日志中提示请求日期、目标代码和调整后日期。默认 ``rolling_z`` 信号使用过去 120 个有效因子观测的历史上轨、1 倍标准差、满仓做空，执行时间为信号日后 10 分钟。

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
