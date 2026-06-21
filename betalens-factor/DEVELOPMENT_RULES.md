# 因子脚本开发标准

本文档描述 `dashboard/backend` 当前实际扫描规则。新增因子必须满足这些约定，才能被 Dashboard 自动发现、展示详情并运行。

## 目录结构

因子按“因子类别”组织在 `betalens-factor/` 下：

```text
betalens-factor/
  {factor_class}/
    spec_{factor_class}.json
    factor_template_{factor_class}.py      # 可选但推荐，类别内公共模板
    {FACTOR_NAME}/
      factor_{FACTOR_NAME}.py
```

扫描规则：

- Dashboard 只扫描 `betalens-factor/` 的一级子目录。
- 目录名以 `.` 或 `__` 开头会被忽略。
- 只有存在 `spec_{目录名}.json` 的目录会被视为因子类别。
- 因子脚本路径必须严格为 `{factor_class}/{name}/factor_{name}.py`。
- `spec` 中 `factors[].name` 必须和子目录名、脚本文件名一致。

示例：

```text
betalens-factor/tdx/spec_tdx.json
betalens-factor/tdx/XICHOU/factor_XICHOU.py
```

## spec JSON 规范

`spec_{factor_class}.json` 至少应包含：

```json
{
  "class": "tdx",
  "source": "因子来源说明",
  "defaults": {
    "direction": "positive",
    "index_code": "000906.SH",
    "use_industry": true,
    "use_mktcap": false,
    "industry_scheme": "申万一级行业",
    "backtest_metric": "收盘价(元)",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "rebal_freq": "W",
    "n_quantiles": 80,
    "include_profiling": true
  },
  "factors": [
    {
      "name": "MYFACTOR",
      "formula": "因子公式",
      "logic": "因子逻辑和方向说明",
      "inputs": {
        "close_wide": "收盘价(元)"
      },
      "compute_kwargs": {
        "window": 20
      }
    }
  ]
}
```

Dashboard 使用字段：

- `source`: 首页和详情页展示。
- `defaults`: 运行参数默认值，会进入参数面板。
- `factors[].name`: 因子唯一名称。
- `factors[].formula`: 首页公式摘要。
- `factors[].logic`: 首页逻辑摘要。
- `factors[].inputs`: 首页输入展示，也应与脚本 `FactorSpec.inputs` 对齐。
- `factors[].compute_kwargs`: 详情页算子参数默认值。

可扩展字段允许存在，但后端不会自动理解，除非同步修改 `dashboard/backend/factors.py` 和前端类型。

## 因子脚本接口

每个 `factor_{name}.py` 必须可被 Python import，并暴露：

```python
spec = FactorSpec(...)
FactorPipeline
```

Dashboard 运行时会：

1. import `factor_{name}.py`
2. 读取模块变量 `spec`
3. 读取模块变量或导入项 `FactorPipeline`
4. 根据页面参数用 `dataclasses.replace(spec, ...)` 覆盖部分字段
5. 调用 `FactorPipeline(spec).run(...)`

因此脚本必须满足：

- 顶层 import 不应启动回测、写文件或执行耗时逻辑。
- 回测只能放在 `if __name__ == "__main__":` 下。
- `spec` 必须是 dataclass 类型的 `FactorSpec`，支持 `dataclasses.replace()`。
- `FactorPipeline.run()` 必须接受 Dashboard 传入的关键字参数：
  - `rebal_freq`
  - `n_quantiles`
  - `initial_amount`
  - `output_dir`
  - `include_profiling`
  - `dump_excel`

Dashboard 会覆盖的 `FactorSpec` 字段：

- `direction`
- `index_code`
- `use_industry`
- `use_mktcap`
- `industry_scheme`
- `backtest_metric`
- `compute_kwargs`

如果自定义模板不支持这些字段，会导致运行失败。

## 算子函数规范

推荐形态：

```python
def compute_myfactor(close_wide, window=20):
    factor = close_wide.pct_change(window)
    return factor.replace([np.inf, -np.inf], np.nan)
```

要求：

- 输入参数名必须与 `FactorSpec.inputs` 的 key 一致。
- 输入是宽表 `DataFrame`：index 为日期时间，columns 为证券代码。
- 输出必须是同形状或可对齐的宽表 `DataFrame`。
- 输出值应为数值型；无效值使用 `NaN`，不要返回字符串。
- 不要在算子内查询数据库、写文件或修改全局状态。

## 最小脚本模板

```python
import sys
from pathlib import Path

import numpy as np

_CLASS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CLASS_DIR))

from factor_template_tdx import FactorSpec, FactorPipeline


def compute_myfactor(close_wide, window=20):
    return close_wide.pct_change(window).replace([np.inf, -np.inf], np.nan)


spec = FactorSpec(
    name="MYFACTOR",
    inputs={"close_wide": "收盘价(元)"},
    compute=compute_myfactor,
    direction="positive",
    compute_kwargs={"window": 20},
    index_code="000906.SH",
    use_industry=True,
    industry_scheme="申万一级行业",
    backtest_metric="收盘价(元)",
)


if __name__ == "__main__":
    FactorPipeline(spec).run(
        "2024-01-01",
        "2024-12-31",
        rebal_freq="W",
        n_quantiles=20,
        output_dir=str(Path(__file__).resolve().parent),
    )
```

## 输出约定

Dashboard 运行时传入 `dump_excel=False`，先展示内存结果，再后台异步生成下载文件。

模板 `run()` 返回对象应包含：

- `backtest`
- `analyst`
- `factor_values`，推荐包含列：
  - `信号日`
  - `股票代码`
  - `因子值`
  - `分组`

`factor_values` 会被 Dashboard 用于“调仓日三维持仓列表”中展示持仓对应因子值，并写入 `{name}_dump.xlsx` 的 `factor_values` sheet。

## 常见错误

- `spec` 中名字为 `XICHOU`，但目录或脚本不是 `XICHOU/factor_XICHOU.py`。
- 脚本只在 `__main__` 中定义 `spec`，导致 import 后 Dashboard 找不到。
- 顶层直接运行回测，导致详情页加载脚本时阻塞。
- `inputs` 的 key 与 compute 函数参数不一致。
- 指标名与数据库不一致，例如应使用 `收盘价(元)` 却写成 `收盘价`。
- 自定义 `FactorSpec` 缺少 Dashboard 会覆盖的字段。
