# 因子脚本开发标准

`betalens-factor/` 按“因子类 -> 因子 -> 主脚本 / 参数 / 产物 / 辅助脚本”组织。
Dashboard 只以 YAML 发现因子；JSON 参数文件已废弃。

## 目录结构

```text
betalens-factor/
  examples/
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
        <mining_script>.py
        <mining_script>.yaml
        outputs/
      tools/
        <tool_script>.py
        <tool_script>.yaml
        outputs/
```

规则：

- 类目录必须有 `class_<class>.yaml`，因子目录必须有 `factor_<name>.py` 和 `factor_<name>.yaml`。
- 主因子脚本保留在因子根目录，Dashboard 按 `<class>/<name>/factor_<name>.py` 导入。
- 历史结果放 `outputs/legacy/`；新运行结果放 `outputs/runs/<run_id-or-manual>/`。
- 参数挖掘入口放 `mining/`，其它报告/诊断脚本放 `tools/`。

## YAML 参数

每个可运行脚本只读取同目录唯一 YAML，默认同 stem：

- `factor_X.yaml` 对应 `factor_X.py`
- `sweep_window.yaml` 对应 `sweep_window.py`
- `report_X_triggers.yaml` 对应 `report_X_triggers.py`

主因子 YAML 的顶层结构：

```yaml
meta:
  class: tdx
  name: SAMPLE
  source: 来源
  formula: 公式
  logic: 逻辑
factor_spec:
  inputs: {close_wide: 收盘价(元)}
  compute_kwargs: {window: 20}
  direction: positive
  table_name: daily_market
  index_code: 000906.SH
  use_industry: true
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
```

原则：

- 不做类级、因子级、脚本级参数覆盖；具体脚本 YAML 必须是完整参数文件。
- 脚本内不设置业务默认参数，算子函数的计算参数必须由 `compute_kwargs` 提供。
- CLI 只支持 `--config PATH` 选择完整 YAML，不支持零散覆盖字段。

## 脚本接口

每个主因子脚本需要：

- import 时不跑回测、不写文件。
- 暴露 `spec`、`FactorPipeline`、`build_spec(config, config_path)`。
- `if __name__ == "__main__"` 中只解析 `--config`，读取 YAML 后运行。

Dashboard 运行时会把页面参数写成 `outputs/runs/<run_id>/run_config.yaml`，然后用这份完整 YAML 构造 `FactorSpec` 和 `run_kwargs`。
