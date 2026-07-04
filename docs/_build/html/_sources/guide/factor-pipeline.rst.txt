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
     compute_kwargs:
       window: 4
     direction: positive
     table_name: daily_market
     index_code: 000906.SH
     use_industry: false
     use_mktcap: false
     industry_scheme: 申万一级行业
     backtest_metric: 收盘价(元)
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
* ``compute``：宽表算子函数。
* ``direction``：``positive`` 高分组做多，``negative`` 低分组做多。
* ``table_name``：输入数据表，常用 ``daily_market``。
* ``index_code``：PIT 指数成分过滤。
* ``use_industry`` / ``use_mktcap``：行业和市值中性化。
* ``weight_mode``、``long_groups``、``short_groups``：权重生成口径。
* ``backtest_metric``：回测成交价指标。

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
