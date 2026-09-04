# research_course：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-0204328a800a"></a>
## docs/learning/first_factor.py

[打开源码](../../learning/first_factor.py) · 70 行 · 说明来源：文件族规则

- **作用**：研究员课程练习或模板
- **输入**：教学样本/课次参数或研究配置
- **输出**：演算输出或模板管线结果
- **副作用/维护重点**：不是生产入口；真实回测模板与离线实验必须区分

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from betalens.factor.factor import single_characteristic, get_single_factor_weight
from pathlib import Path
import pandas as pd
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [main](../../learning/first_factor.py#L16) | main() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |

<a id="file-58f614fdc62d"></a>
## docs/learning/labs.py

[打开源码](../../learning/labs.py) · 235 行 · 说明来源：文件族规则

- **作用**：研究员课程练习或模板
- **输入**：教学样本/课次参数或研究配置
- **输出**：演算输出或模板管线结果
- **副作用/维护重点**：不是生产入口；真实回测模板与离线实验必须区分

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.backtest import BacktestBase
from betalens.factor.factor import single_characteristic, get_single_factor_weight
from betalens.factor.preprocessing import winsorize_factor, standardize_factor, neutralize_factor
from betalens.factor.stats import calc_ic
from pathlib import Path
import argparse
import importlib.util
import numpy as np
import pandas as pd
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [lab02](../../learning/labs.py#L19) | lab02() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 表格、筛选与宽长转换。 |
| [lab03](../../learning/labs.py#L33) | lab03() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 收益、复利与未来收益的标签。 |
| [lab08](../../learning/labs.py#L45) | lab08() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 缺失值、重复键和单位。 |
| [lab09](../../learning/labs.py#L60) | lab09() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 自己编写动量公式。 |
| [lab09.momentum](../../learning/labs.py#L64) | momentum(close_wide, window) | 无返回注解；return: close_wide / close_wide.shift(window) - 1 | 无 docstring，需阅读函数体 |
| [lab10](../../learning/labs.py#L73) | lab10() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 使用真实预处理函数。 |
| [lab11](../../learning/labs.py#L91) | lab11() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 同分值分组和显式多空选择。 |
| [lab13](../../learning/labs.py#L111) | lab13() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 真实 BacktestBase，使用内存价格，关闭外部交易状态查询。 |
| [lab15](../../learning/labs.py#L129) | lab15() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 调用真实绩效指标，区分回撤幅度和负号表示。 |
| [lab16](../../learning/labs.py#L153) | lab16() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 真实 Rank IC：正确顺序、反向顺序、样本不足。 |
| [lab21](../../learning/labs.py#L169) | lab21() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 教学择时阈值，历史窗口不包括当前观测。 |
| [lab22](../../learning/labs.py#L184) | lab22() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 事件聚合：先平均再复利与先复利再平均不同。 |
| [lab23](../../learning/labs.py#L198) | lab23() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 从纯噪声候选中挑冠军，观察选择偏差。 |
| [main](../../learning/labs.py#L218) | main() | 无返回注解；return: None | 无 docstring，需阅读函数体 |

<a id="file-f4c44b805e65"></a>
## docs/learning/templates/research/COURSE_MOM/factor_COURSE_MOM.py

[打开源码](../../learning/templates/research/COURSE_MOM/factor_COURSE_MOM.py) · 73 行 · 说明来源：文件族规则

- **作用**：COURSE_MOM：完整的动量教学因子。市场回测需要真实数据库。；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters
from factor_template import FactorPipeline, FactorSpec
from pathlib import Path
import argparse
import numpy as np
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_find_repo_root](../../learning/templates/research/COURSE_MOM/factor_COURSE_MOM.py#L15) | _find_repo_root() | 无返回注解；return: directory | 无 docstring，需阅读函数体 |
| [load_config](../../learning/templates/research/COURSE_MOM/factor_COURSE_MOM.py#L33) | load_config(path=_CONFIG_FILE) | 无返回注解；return: load_yaml_config(path, required_sections=('meta', 'factor_spec', 'weight', 'run')) | 无 docstring，需阅读函数体 |
| [compute_momentum](../../learning/templates/research/COURSE_MOM/factor_COURSE_MOM.py#L37) | compute_momentum(close_wide, *, window) | 无返回注解；return: values.replace([np.inf, -np.inf], np.nan) | 无 docstring，需阅读函数体 |
| [build_spec](../../learning/templates/research/COURSE_MOM/factor_COURSE_MOM.py#L44) | build_spec(config, config_path=_CONFIG_FILE) | 无返回注解；return: FactorSpec(name=str(config['meta']['name']), compute=compute_momentum, **options) | 无 docstring，需阅读函数体 |
| [run_from_config](../../learning/templates/research/COURSE_MOM/factor_COURSE_MOM.py#L56) | run_from_config(config_path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../learning/templates/research/COURSE_MOM/factor_COURSE_MOM.py#L65) | main() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |

<a id="file-58afd291d184"></a>
## docs/learning/templates/research/COURSE_MOM/factor_COURSE_MOM.yaml

[打开源码](../../learning/templates/research/COURSE_MOM/factor_COURSE_MOM.yaml) · 40 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../learning/templates/research/COURSE_MOM/factor_COURSE_MOM.yaml#L1)：`meta:`
- [L8](../../learning/templates/research/COURSE_MOM/factor_COURSE_MOM.yaml#L8)：`factor_spec:`
- [L23](../../learning/templates/research/COURSE_MOM/factor_COURSE_MOM.yaml#L23)：`weight:`
- [L29](../../learning/templates/research/COURSE_MOM/factor_COURSE_MOM.yaml#L29)：`run:`

<a id="file-1fb012006aba"></a>
## docs/learning/templates/research/class_research.yaml

[打开源码](../../learning/templates/research/class_research.yaml) · 3 行 · 说明来源：文件族规则

- **作用**：因子类别发现元数据
- **输入**：维护者填写的参数
- **输出**：类别信息；不能替代完整因子运行 YAML
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../learning/templates/research/class_research.yaml#L1)：`class: research`
- [L2](../../learning/templates/research/class_research.yaml#L2)：`template_module: factor_template`
- [L3](../../learning/templates/research/class_research.yaml#L3)：`source: Betalens 研究员课程，教学用途`

