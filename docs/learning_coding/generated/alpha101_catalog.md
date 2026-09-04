# alpha101_catalog：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-f4929c92773c"></a>
## betalens-factor/alpha101/ALPHA1/factor_ALPHA1.py

[打开源码](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1.py) · 88 行 · 说明来源：文件族规则

- **作用**：ALPHA1 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha1](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1.py#L29) | compute_alpha1(close_wide, returns_wide, *, returns_threshold=0, returns_stddev_window=20, base_power_exponent=2.0, signed_power_base_argmax_window=5, rank_ts_argmax_signed_power_center=0.5) | 无返回注解；return: compute_alpha(1, close_wide=close_wide, returns_wide=returns_wide, returns_threshold=returns_threshold, returns_stddev_window=returns_stddev_window, base_power_exponent=base_power_exponent, signed_power_base_argmax_window=signed_power_base_argmax_window, rank_ts_argmax_signed_power_cen…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1.py#L51) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1.py#L71) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1.py#L80) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-e0733fbb1843"></a>
## betalens-factor/alpha101/ALPHA1/factor_ALPHA1.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1.yaml) · 43 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1.yaml#L7)：`factor_spec:`
- [L27](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1.yaml#L27)：`weight:`
- [L33](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1.yaml#L33)：`run:`

<a id="file-6fe90f79ce3e"></a>
## betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.py) · 91 行 · 说明来源：文件族规则

- **作用**：ALPHA1 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha1_timing](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.py#L29) | compute_alpha1_timing(close_wide, returns_wide, *, returns_threshold=0, returns_stddev_window=20, base_power_exponent=2.0, signed_power_base_argmax_window=5, rank_ts_argmax_signed_power_center=0.5, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(1, close_wide=close_wide, returns_wide=returns_wide, returns_threshold=returns_threshold, returns_stddev_window=returns_stddev_window, base_power_exponent=base_power_exponent, signed_power_base_argmax_window=signed_power_base_argmax_window, rank_ts_argmax_signed_power_cen…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.py#L54) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.py#L74) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.py#L83) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-42de59f6afe0"></a>
## betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.yaml) · 52 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.yaml#L8)：`factor_spec:`
- [L36](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.yaml#L36)：`weight:`
- [L42](../../../betalens-factor/alpha101/ALPHA1/factor_ALPHA1_timing.yaml#L42)：`run:`

<a id="file-d05cc06824ba"></a>
## betalens-factor/alpha101/ALPHA10/factor_ALPHA10.py

[打开源码](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10.py) · 86 行 · 说明来源：文件族规则

- **作用**：ALPHA10 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha10](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10.py#L29) | compute_alpha10(close_wide, *, close_delta_lag=1, change_minimum_window=4, ts_min_change_threshold=0, change_maximum_window=4, ts_max_change_threshold=0) | 无返回注解；return: compute_alpha(10, close_wide=close_wide, close_delta_lag=close_delta_lag, change_minimum_window=change_minimum_window, ts_min_change_threshold=ts_min_change_threshold, change_maximum_window=change_maximum_window, ts_max_change_threshold=ts_max_change_threshold) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10.py#L49) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10.py#L69) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10.py#L78) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-5b1b630312ff"></a>
## betalens-factor/alpha101/ALPHA10/factor_ALPHA10.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10.yaml) · 43 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10.yaml#L8)：`factor_spec:`
- [L27](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10.yaml#L27)：`weight:`
- [L33](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10.yaml#L33)：`run:`

<a id="file-3c8a70e1f574"></a>
## betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.py) · 89 行 · 说明来源：文件族规则

- **作用**：ALPHA10 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha10_timing](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.py#L29) | compute_alpha10_timing(close_wide, *, close_delta_lag=1, change_minimum_window=4, ts_min_change_threshold=0, change_maximum_window=4, ts_max_change_threshold=0, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(10, close_wide=close_wide, close_delta_lag=close_delta_lag, change_minimum_window=change_minimum_window, ts_min_change_threshold=ts_min_change_threshold, change_maximum_window=change_maximum_window, ts_max_change_threshold=ts_max_change_threshold) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.py#L52) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.py#L72) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.py#L81) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-afe80ea5ab94"></a>
## betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.yaml) · 51 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.yaml#L8)：`factor_spec:`
- [L35](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.yaml#L35)：`weight:`
- [L41](../../../betalens-factor/alpha101/ALPHA10/factor_ALPHA10_timing.yaml#L41)：`run:`

<a id="file-fa34f8de8078"></a>
## betalens-factor/alpha101/ALPHA100/factor_ALPHA100.py

[打开源码](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100.py) · 96 行 · 说明来源：文件族规则

- **作用**：ALPHA100 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha100](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100.py#L29) | compute_alpha100(close_wide, high_wide, low_wide, volume_wide, amount_wide, subindustry_wide, *, amount_average_window=20, close_rank_adv_correlation_window=5, close_argmin_window=30, scale_first_coefficient=1.5, amount_average_window_2=20) | 无返回注解；return: compute_alpha(100, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, amount_wide=amount_wide, subindustry_wide=subindustry_wide, amount_average_window=amount_average_window, close_rank_adv_correlation_window=close_rank_adv_correlation_window, close…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100.py#L59) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100.py#L79) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100.py#L88) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-4fcd634acf3c"></a>
## betalens-factor/alpha101/ALPHA100/factor_ALPHA100.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100.yaml) · 47 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100.yaml#L7)：`factor_spec:`
- [L31](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100.yaml#L31)：`weight:`
- [L37](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100.yaml#L37)：`run:`

<a id="file-2285bf651ec7"></a>
## betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.py) · 99 行 · 说明来源：文件族规则

- **作用**：ALPHA100 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha100_timing](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.py#L29) | compute_alpha100_timing(close_wide, high_wide, low_wide, volume_wide, amount_wide, subindustry_wide, *, amount_average_window=20, close_rank_adv_correlation_window=5, close_argmin_window=30, scale_first_coefficient=1.5, amount_average_window_2=20, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(100, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, amount_wide=amount_wide, subindustry_wide=subindustry_wide, amount_average_window=amount_average_window, close_rank_adv_correlation_window=close_rank_adv_correlation_window, close…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.py#L62) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.py#L82) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.py#L91) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-2a004b8fdbd3"></a>
## betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.yaml) · 56 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.yaml#L8)：`factor_spec:`
- [L40](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.yaml#L40)：`weight:`
- [L46](../../../betalens-factor/alpha101/ALPHA100/factor_ALPHA100_timing.yaml#L46)：`run:`

<a id="file-d32ff6b550f0"></a>
## betalens-factor/alpha101/ALPHA101/factor_ALPHA101.py

[打开源码](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101.py) · 84 行 · 说明来源：文件族规则

- **作用**：ALPHA101 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha101](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101.py#L29) | compute_alpha101(open_wide, close_wide, high_wide, low_wide, *, high_low_epsilon=0.001) | 无返回注解；return: compute_alpha(101, open_wide=open_wide, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, high_low_epsilon=high_low_epsilon) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101.py#L47) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101.py#L67) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101.py#L76) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-778ba29b8487"></a>
## betalens-factor/alpha101/ALPHA101/factor_ALPHA101.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101.yaml) · 41 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101.yaml#L7)：`factor_spec:`
- [L25](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101.yaml#L25)：`weight:`
- [L31](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101.yaml#L31)：`run:`

<a id="file-4381aac3b9f1"></a>
## betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.py) · 87 行 · 说明来源：文件族规则

- **作用**：ALPHA101 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha101_timing](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.py#L29) | compute_alpha101_timing(open_wide, close_wide, high_wide, low_wide, *, high_low_epsilon=0.001, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(101, open_wide=open_wide, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, high_low_epsilon=high_low_epsilon) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.py#L50) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.py#L70) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.py#L79) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-04155b5e913d"></a>
## betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.yaml) · 50 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.yaml#L8)：`factor_spec:`
- [L34](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.yaml#L34)：`weight:`
- [L40](../../../betalens-factor/alpha101/ALPHA101/factor_ALPHA101_timing.yaml#L40)：`run:`

<a id="file-5732d95dd387"></a>
## betalens-factor/alpha101/ALPHA11/factor_ALPHA11.py

[打开源码](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11.py) · 86 行 · 说明来源：文件族规则

- **作用**：ALPHA11 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha11](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11.py#L29) | compute_alpha11(close_wide, volume_wide, vwap_wide, *, spread_maximum_window=3, spread_minimum_window=3, volume_delta_lag=3) | 无返回注解；return: compute_alpha(11, close_wide=close_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, spread_maximum_window=spread_maximum_window, spread_minimum_window=spread_minimum_window, volume_delta_lag=volume_delta_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11.py#L49) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11.py#L69) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11.py#L78) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-c8bf5ea6675d"></a>
## betalens-factor/alpha101/ALPHA11/factor_ALPHA11.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11.yaml) · 42 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11.yaml#L7)：`factor_spec:`
- [L26](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11.yaml#L26)：`weight:`
- [L32](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11.yaml#L32)：`run:`

<a id="file-e5f1d4a31f5a"></a>
## betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.py) · 89 行 · 说明来源：文件族规则

- **作用**：ALPHA11 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha11_timing](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.py#L29) | compute_alpha11_timing(close_wide, volume_wide, vwap_wide, *, spread_maximum_window=3, spread_minimum_window=3, volume_delta_lag=3, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(11, close_wide=close_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, spread_maximum_window=spread_maximum_window, spread_minimum_window=spread_minimum_window, volume_delta_lag=volume_delta_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.py#L52) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.py#L72) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.py#L81) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-65877ddcabb4"></a>
## betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.yaml) · 51 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.yaml#L8)：`factor_spec:`
- [L35](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.yaml#L35)：`weight:`
- [L41](../../../betalens-factor/alpha101/ALPHA11/factor_ALPHA11_timing.yaml#L41)：`run:`

<a id="file-da112b69ef45"></a>
## betalens-factor/alpha101/ALPHA12/factor_ALPHA12.py

[打开源码](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12.py) · 82 行 · 说明来源：文件族规则

- **作用**：ALPHA12 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha12](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12.py#L29) | compute_alpha12(close_wide, volume_wide, *, volume_delta_lag=1, close_delta_lag=1) | 无返回注解；return: compute_alpha(12, close_wide=close_wide, volume_wide=volume_wide, volume_delta_lag=volume_delta_lag, close_delta_lag=close_delta_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12.py#L45) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12.py#L65) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12.py#L74) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-0af5da4b93d8"></a>
## betalens-factor/alpha101/ALPHA12/factor_ALPHA12.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12.yaml) · 41 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12.yaml#L7)：`factor_spec:`
- [L24](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12.yaml#L24)：`weight:`
- [L30](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12.yaml#L30)：`run:`

<a id="file-0e3b800cf4eb"></a>
## betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.py) · 85 行 · 说明来源：文件族规则

- **作用**：ALPHA12 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha12_timing](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.py#L29) | compute_alpha12_timing(close_wide, volume_wide, *, volume_delta_lag=1, close_delta_lag=1, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(12, close_wide=close_wide, volume_wide=volume_wide, volume_delta_lag=volume_delta_lag, close_delta_lag=close_delta_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.py#L48) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.py#L68) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.py#L77) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d846392587fe"></a>
## betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.yaml#L8)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA12/factor_ALPHA12_timing.yaml#L39)：`run:`

<a id="file-2e3499323c94"></a>
## betalens-factor/alpha101/ALPHA13/factor_ALPHA13.py

[打开源码](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13.py) · 80 行 · 说明来源：文件族规则

- **作用**：ALPHA13 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha13](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13.py#L29) | compute_alpha13(close_wide, volume_wide, *, rank_close_rank_volume_covariance_window=5) | 无返回注解；return: compute_alpha(13, close_wide=close_wide, volume_wide=volume_wide, rank_close_rank_volume_covariance_window=rank_close_rank_volume_covariance_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13.py#L43) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13.py#L63) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13.py#L72) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ed77e977030f"></a>
## betalens-factor/alpha101/ALPHA13/factor_ALPHA13.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13.yaml) · 39 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13.yaml#L7)：`factor_spec:`
- [L23](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13.yaml#L23)：`weight:`
- [L29](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13.yaml#L29)：`run:`

<a id="file-bcb99a4f1979"></a>
## betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.py) · 83 行 · 说明来源：文件族规则

- **作用**：ALPHA13 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha13_timing](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.py#L29) | compute_alpha13_timing(close_wide, volume_wide, *, rank_close_rank_volume_covariance_window=5, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(13, close_wide=close_wide, volume_wide=volume_wide, rank_close_rank_volume_covariance_window=rank_close_rank_volume_covariance_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.py#L46) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.py#L66) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.py#L75) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-00b52dffb2b9"></a>
## betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.yaml#L8)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA13/factor_ALPHA13_timing.yaml#L38)：`run:`

<a id="file-2b7f9cbb9a9d"></a>
## betalens-factor/alpha101/ALPHA14/factor_ALPHA14.py

[打开源码](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14.py) · 84 行 · 说明来源：文件族规则

- **作用**：ALPHA14 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha14](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14.py#L29) | compute_alpha14(open_wide, volume_wide, returns_wide, *, returns_delta_lag=3, open_volume_correlation_window=10) | 无返回注解；return: compute_alpha(14, open_wide=open_wide, volume_wide=volume_wide, returns_wide=returns_wide, returns_delta_lag=returns_delta_lag, open_volume_correlation_window=open_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14.py#L47) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14.py#L67) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14.py#L76) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d26b5f5de57e"></a>
## betalens-factor/alpha101/ALPHA14/factor_ALPHA14.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14.yaml) · 41 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14.yaml#L7)：`factor_spec:`
- [L25](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14.yaml#L25)：`weight:`
- [L31](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14.yaml#L31)：`run:`

<a id="file-86f10aef9e91"></a>
## betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.py) · 87 行 · 说明来源：文件族规则

- **作用**：ALPHA14 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha14_timing](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.py#L29) | compute_alpha14_timing(open_wide, volume_wide, returns_wide, *, returns_delta_lag=3, open_volume_correlation_window=10, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(14, open_wide=open_wide, volume_wide=volume_wide, returns_wide=returns_wide, returns_delta_lag=returns_delta_lag, open_volume_correlation_window=open_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.py#L50) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.py#L70) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.py#L79) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-b5fa221e0c69"></a>
## betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.yaml) · 50 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.yaml#L8)：`factor_spec:`
- [L34](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.yaml#L34)：`weight:`
- [L40](../../../betalens-factor/alpha101/ALPHA14/factor_ALPHA14_timing.yaml#L40)：`run:`

<a id="file-252d70cb8682"></a>
## betalens-factor/alpha101/ALPHA15/factor_ALPHA15.py

[打开源码](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15.py) · 82 行 · 说明来源：文件族规则

- **作用**：ALPHA15 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha15](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15.py#L29) | compute_alpha15(high_wide, volume_wide, *, rank_high_rank_volume_correlation_window=3, rank_correlation_high_sum_window=3) | 无返回注解；return: compute_alpha(15, high_wide=high_wide, volume_wide=volume_wide, rank_high_rank_volume_correlation_window=rank_high_rank_volume_correlation_window, rank_correlation_high_sum_window=rank_correlation_high_sum_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15.py#L45) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15.py#L65) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15.py#L74) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-dd3fae2c5754"></a>
## betalens-factor/alpha101/ALPHA15/factor_ALPHA15.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15.yaml) · 40 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15.yaml#L7)：`factor_spec:`
- [L24](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15.yaml#L24)：`weight:`
- [L30](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15.yaml#L30)：`run:`

<a id="file-4451b9f8f22f"></a>
## betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.py) · 85 行 · 说明来源：文件族规则

- **作用**：ALPHA15 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha15_timing](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.py#L29) | compute_alpha15_timing(high_wide, volume_wide, *, rank_high_rank_volume_correlation_window=3, rank_correlation_high_sum_window=3, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(15, high_wide=high_wide, volume_wide=volume_wide, rank_high_rank_volume_correlation_window=rank_high_rank_volume_correlation_window, rank_correlation_high_sum_window=rank_correlation_high_sum_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.py#L48) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.py#L68) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.py#L77) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ed61ac7be679"></a>
## betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.yaml#L8)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA15/factor_ALPHA15_timing.yaml#L39)：`run:`

<a id="file-1ed0730d0831"></a>
## betalens-factor/alpha101/ALPHA16/factor_ALPHA16.py

[打开源码](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16.py) · 80 行 · 说明来源：文件族规则

- **作用**：ALPHA16 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha16](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16.py#L29) | compute_alpha16(high_wide, volume_wide, *, rank_high_rank_volume_covariance_window=5) | 无返回注解；return: compute_alpha(16, high_wide=high_wide, volume_wide=volume_wide, rank_high_rank_volume_covariance_window=rank_high_rank_volume_covariance_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16.py#L43) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16.py#L63) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16.py#L72) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-2ce68da052d8"></a>
## betalens-factor/alpha101/ALPHA16/factor_ALPHA16.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16.yaml) · 39 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16.yaml#L7)：`factor_spec:`
- [L23](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16.yaml#L23)：`weight:`
- [L29](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16.yaml#L29)：`run:`

<a id="file-89d751b2833c"></a>
## betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.py) · 83 行 · 说明来源：文件族规则

- **作用**：ALPHA16 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha16_timing](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.py#L29) | compute_alpha16_timing(high_wide, volume_wide, *, rank_high_rank_volume_covariance_window=5, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(16, high_wide=high_wide, volume_wide=volume_wide, rank_high_rank_volume_covariance_window=rank_high_rank_volume_covariance_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.py#L46) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.py#L66) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.py#L75) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-64d3d9556bf8"></a>
## betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.yaml#L8)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA16/factor_ALPHA16_timing.yaml#L38)：`run:`

<a id="file-ec2790215241"></a>
## betalens-factor/alpha101/ALPHA17/factor_ALPHA17.py

[打开源码](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17.py) · 90 行 · 说明来源：文件族规则

- **作用**：ALPHA17 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha17](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17.py#L29) | compute_alpha17(close_wide, volume_wide, amount_wide, *, close_rank_window=10, close_delta_lag=1, delta_close_delta_lag=1, amount_average_window=20, volume_adv_rank_window=5) | 无返回注解；return: compute_alpha(17, close_wide=close_wide, volume_wide=volume_wide, amount_wide=amount_wide, close_rank_window=close_rank_window, close_delta_lag=close_delta_lag, delta_close_delta_lag=delta_close_delta_lag, amount_average_window=amount_average_window, volume_adv_rank_window=volume_adv_r…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17.py#L53) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17.py#L73) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17.py#L82) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-7e56e9c01c4c"></a>
## betalens-factor/alpha101/ALPHA17/factor_ALPHA17.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17.yaml) · 44 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17.yaml#L7)：`factor_spec:`
- [L28](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17.yaml#L28)：`weight:`
- [L34](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17.yaml#L34)：`run:`

<a id="file-7f0fe9196f3f"></a>
## betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.py) · 93 行 · 说明来源：文件族规则

- **作用**：ALPHA17 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha17_timing](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.py#L29) | compute_alpha17_timing(close_wide, volume_wide, amount_wide, *, close_rank_window=10, close_delta_lag=1, delta_close_delta_lag=1, amount_average_window=20, volume_adv_rank_window=5, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(17, close_wide=close_wide, volume_wide=volume_wide, amount_wide=amount_wide, close_rank_window=close_rank_window, close_delta_lag=close_delta_lag, delta_close_delta_lag=delta_close_delta_lag, amount_average_window=amount_average_window, volume_adv_rank_window=volume_adv_r…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.py#L56) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.py#L76) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.py#L85) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-f283c26834f9"></a>
## betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.yaml) · 53 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.yaml#L8)：`factor_spec:`
- [L37](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.yaml#L37)：`weight:`
- [L43](../../../betalens-factor/alpha101/ALPHA17/factor_ALPHA17_timing.yaml#L43)：`run:`

<a id="file-6342d0c44fb2"></a>
## betalens-factor/alpha101/ALPHA18/factor_ALPHA18.py

[打开源码](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18.py) · 82 行 · 说明来源：文件族规则

- **作用**：ALPHA18 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha18](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18.py#L29) | compute_alpha18(open_wide, close_wide, *, close_open_stddev_window=5, close_open_correlation_window=10) | 无返回注解；return: compute_alpha(18, open_wide=open_wide, close_wide=close_wide, close_open_stddev_window=close_open_stddev_window, close_open_correlation_window=close_open_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18.py#L45) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18.py#L65) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18.py#L74) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-1ea4011c851a"></a>
## betalens-factor/alpha101/ALPHA18/factor_ALPHA18.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18.yaml) · 40 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18.yaml#L7)：`factor_spec:`
- [L24](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18.yaml#L24)：`weight:`
- [L30](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18.yaml#L30)：`run:`

<a id="file-e1a5b3610af5"></a>
## betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.py) · 85 行 · 说明来源：文件族规则

- **作用**：ALPHA18 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha18_timing](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.py#L29) | compute_alpha18_timing(open_wide, close_wide, *, close_open_stddev_window=5, close_open_correlation_window=10, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(18, open_wide=open_wide, close_wide=close_wide, close_open_stddev_window=close_open_stddev_window, close_open_correlation_window=close_open_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.py#L48) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.py#L68) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.py#L77) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-02448e5a4a39"></a>
## betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.yaml#L8)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA18/factor_ALPHA18_timing.yaml#L39)：`run:`

<a id="file-e9a5232cc2b0"></a>
## betalens-factor/alpha101/ALPHA19/factor_ALPHA19.py

[打开源码](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19.py) · 88 行 · 说明来源：文件族规则

- **作用**：ALPHA19 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha19](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19.py#L29) | compute_alpha19(close_wide, returns_wide, *, close_delay_lag=7, close_delta_lag=7, rank_ts_sum_returns_offset=1, ts_sum_returns_offset=1, returns_sum_window=250) | 无返回注解；return: compute_alpha(19, close_wide=close_wide, returns_wide=returns_wide, close_delay_lag=close_delay_lag, close_delta_lag=close_delta_lag, rank_ts_sum_returns_offset=rank_ts_sum_returns_offset, ts_sum_returns_offset=ts_sum_returns_offset, returns_sum_window=returns_sum_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19.py#L51) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19.py#L71) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19.py#L80) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-070408140899"></a>
## betalens-factor/alpha101/ALPHA19/factor_ALPHA19.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19.yaml) · 43 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19.yaml#L7)：`factor_spec:`
- [L27](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19.yaml#L27)：`weight:`
- [L33](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19.yaml#L33)：`run:`

<a id="file-14d68c0b9085"></a>
## betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.py) · 91 行 · 说明来源：文件族规则

- **作用**：ALPHA19 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha19_timing](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.py#L29) | compute_alpha19_timing(close_wide, returns_wide, *, close_delay_lag=7, close_delta_lag=7, rank_ts_sum_returns_offset=1, ts_sum_returns_offset=1, returns_sum_window=250, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(19, close_wide=close_wide, returns_wide=returns_wide, close_delay_lag=close_delay_lag, close_delta_lag=close_delta_lag, rank_ts_sum_returns_offset=rank_ts_sum_returns_offset, ts_sum_returns_offset=ts_sum_returns_offset, returns_sum_window=returns_sum_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.py#L54) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.py#L74) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.py#L83) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-a5a032324bd9"></a>
## betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.yaml) · 52 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.yaml#L8)：`factor_spec:`
- [L36](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.yaml#L36)：`weight:`
- [L42](../../../betalens-factor/alpha101/ALPHA19/factor_ALPHA19_timing.yaml#L42)：`run:`

<a id="file-f6f87b634a57"></a>
## betalens-factor/alpha101/ALPHA2/factor_ALPHA2.py

[打开源码](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2.py) · 84 行 · 说明来源：文件族规则

- **作用**：ALPHA2 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha2](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2.py#L29) | compute_alpha2(open_wide, close_wide, volume_wide, *, volume_delta_lag=2, rank_delta_volume_rank_open_close_correlation_window=6) | 无返回注解；return: compute_alpha(2, open_wide=open_wide, close_wide=close_wide, volume_wide=volume_wide, volume_delta_lag=volume_delta_lag, rank_delta_volume_rank_open_close_correlation_window=rank_delta_volume_rank_open_close_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2.py#L47) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2.py#L67) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2.py#L76) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-1f9f5afac6f0"></a>
## betalens-factor/alpha101/ALPHA2/factor_ALPHA2.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2.yaml) · 41 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2.yaml#L7)：`factor_spec:`
- [L25](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2.yaml#L25)：`weight:`
- [L31](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2.yaml#L31)：`run:`

<a id="file-f6c8d18db801"></a>
## betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.py) · 87 行 · 说明来源：文件族规则

- **作用**：ALPHA2 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha2_timing](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.py#L29) | compute_alpha2_timing(open_wide, close_wide, volume_wide, *, volume_delta_lag=2, rank_delta_volume_rank_open_close_correlation_window=6, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(2, open_wide=open_wide, close_wide=close_wide, volume_wide=volume_wide, volume_delta_lag=volume_delta_lag, rank_delta_volume_rank_open_close_correlation_window=rank_delta_volume_rank_open_close_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.py#L50) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.py#L70) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.py#L79) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d9cb9b2c4dc4"></a>
## betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.yaml) · 50 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.yaml#L8)：`factor_spec:`
- [L34](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.yaml#L34)：`weight:`
- [L40](../../../betalens-factor/alpha101/ALPHA2/factor_ALPHA2_timing.yaml#L40)：`run:`

<a id="file-988ddbece890"></a>
## betalens-factor/alpha101/ALPHA20/factor_ALPHA20.py

[打开源码](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20.py) · 88 行 · 说明来源：文件族规则

- **作用**：ALPHA20 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha20](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20.py#L29) | compute_alpha20(open_wide, close_wide, high_wide, low_wide, *, high_delay_lag=1, close_delay_lag=1, low_delay_lag=1) | 无返回注解；return: compute_alpha(20, open_wide=open_wide, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, high_delay_lag=high_delay_lag, close_delay_lag=close_delay_lag, low_delay_lag=low_delay_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20.py#L51) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20.py#L71) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20.py#L80) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-3c6943908700"></a>
## betalens-factor/alpha101/ALPHA20/factor_ALPHA20.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20.yaml) · 43 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20.yaml#L7)：`factor_spec:`
- [L27](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20.yaml#L27)：`weight:`
- [L33](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20.yaml#L33)：`run:`

<a id="file-753ec61331c7"></a>
## betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.py) · 91 行 · 说明来源：文件族规则

- **作用**：ALPHA20 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha20_timing](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.py#L29) | compute_alpha20_timing(open_wide, close_wide, high_wide, low_wide, *, high_delay_lag=1, close_delay_lag=1, low_delay_lag=1, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(20, open_wide=open_wide, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, high_delay_lag=high_delay_lag, close_delay_lag=close_delay_lag, low_delay_lag=low_delay_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.py#L54) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.py#L74) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.py#L83) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-a2e17407b35b"></a>
## betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.yaml) · 52 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.yaml#L8)：`factor_spec:`
- [L36](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.yaml#L36)：`weight:`
- [L42](../../../betalens-factor/alpha101/ALPHA20/factor_ALPHA20_timing.yaml#L42)：`run:`

<a id="file-3fc784995732"></a>
## betalens-factor/alpha101/ALPHA21/factor_ALPHA21.py

[打开源码](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21.py) · 100 行 · 说明来源：文件族规则

- **作用**：ALPHA21 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha21](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21.py#L29) | compute_alpha21(close_wide, volume_wide, amount_wide, *, close_mean_window=8, close_mean_window_2=2, close_stddev_window=8, close_stddev_window_2=8, amount_average_window=20, volume_adv_threshold=1, condition1_true_value=-1.0, condition2_true_value=1.0, volume_condition_true_value=1.0, volume_condition_false_value=-1.0) | 无返回注解；return: compute_alpha(21, close_wide=close_wide, volume_wide=volume_wide, amount_wide=amount_wide, close_mean_window=close_mean_window, close_mean_window_2=close_mean_window_2, close_stddev_window=close_stddev_window, close_stddev_window_2=close_stddev_window_2, amount_average_window=amount_av…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21.py#L63) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21.py#L83) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21.py#L92) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-1ed0c57f5f7a"></a>
## betalens-factor/alpha101/ALPHA21/factor_ALPHA21.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21.yaml) · 50 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21.yaml#L8)：`factor_spec:`
- [L34](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21.yaml#L34)：`weight:`
- [L40](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21.yaml#L40)：`run:`

<a id="file-dfae8a64b493"></a>
## betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.py) · 103 行 · 说明来源：文件族规则

- **作用**：ALPHA21 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha21_timing](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.py#L29) | compute_alpha21_timing(close_wide, volume_wide, amount_wide, *, close_mean_window=8, close_mean_window_2=2, close_stddev_window=8, close_stddev_window_2=8, amount_average_window=20, volume_adv_threshold=1, condition1_true_value=-1.0, condition2_true_value=1.0, volume_condition_true_value=1.0, volume_condition_false_value=-1.0, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(21, close_wide=close_wide, volume_wide=volume_wide, amount_wide=amount_wide, close_mean_window=close_mean_window, close_mean_window_2=close_mean_window_2, close_stddev_window=close_stddev_window, close_stddev_window_2=close_stddev_window_2, amount_average_window=amount_av…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.py#L66) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.py#L86) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.py#L95) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-86ad7c4eef6c"></a>
## betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.yaml) · 58 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.yaml#L8)：`factor_spec:`
- [L42](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.yaml#L42)：`weight:`
- [L48](../../../betalens-factor/alpha101/ALPHA21/factor_ALPHA21_timing.yaml#L48)：`run:`

<a id="file-f57c2f97fb09"></a>
## betalens-factor/alpha101/ALPHA22/factor_ALPHA22.py

[打开源码](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22.py) · 86 行 · 说明来源：文件族规则

- **作用**：ALPHA22 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha22](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22.py#L29) | compute_alpha22(close_wide, high_wide, volume_wide, *, high_volume_correlation_window=5, correlation_high_volume_delta_lag=5, close_stddev_window=20) | 无返回注解；return: compute_alpha(22, close_wide=close_wide, high_wide=high_wide, volume_wide=volume_wide, high_volume_correlation_window=high_volume_correlation_window, correlation_high_volume_delta_lag=correlation_high_volume_delta_lag, close_stddev_window=close_stddev_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22.py#L49) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22.py#L69) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22.py#L78) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-55d9b0986987"></a>
## betalens-factor/alpha101/ALPHA22/factor_ALPHA22.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22.yaml) · 42 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22.yaml#L7)：`factor_spec:`
- [L26](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22.yaml#L26)：`weight:`
- [L32](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22.yaml#L32)：`run:`

<a id="file-e536345f4a24"></a>
## betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.py) · 89 行 · 说明来源：文件族规则

- **作用**：ALPHA22 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha22_timing](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.py#L29) | compute_alpha22_timing(close_wide, high_wide, volume_wide, *, high_volume_correlation_window=5, correlation_high_volume_delta_lag=5, close_stddev_window=20, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(22, close_wide=close_wide, high_wide=high_wide, volume_wide=volume_wide, high_volume_correlation_window=high_volume_correlation_window, correlation_high_volume_delta_lag=correlation_high_volume_delta_lag, close_stddev_window=close_stddev_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.py#L52) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.py#L72) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.py#L81) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-fe2089dd0e62"></a>
## betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.yaml) · 51 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.yaml#L8)：`factor_spec:`
- [L35](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.yaml#L35)：`weight:`
- [L41](../../../betalens-factor/alpha101/ALPHA22/factor_ALPHA22_timing.yaml#L41)：`run:`

<a id="file-d1c555a8fe59"></a>
## betalens-factor/alpha101/ALPHA23/factor_ALPHA23.py

[打开源码](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23.py) · 82 行 · 说明来源：文件族规则

- **作用**：ALPHA23 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha23](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23.py#L29) | compute_alpha23(high_wide, *, high_mean_window=20, high_delta_lag=2, high_ts_mean_false_value=0.0) | 无返回注解；return: compute_alpha(23, high_wide=high_wide, high_mean_window=high_mean_window, high_delta_lag=high_delta_lag, high_ts_mean_false_value=high_ts_mean_false_value) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23.py#L45) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23.py#L65) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23.py#L74) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ef9046d1f285"></a>
## betalens-factor/alpha101/ALPHA23/factor_ALPHA23.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23.yaml) · 40 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23.yaml#L7)：`factor_spec:`
- [L24](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23.yaml#L24)：`weight:`
- [L30](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23.yaml#L30)：`run:`

<a id="file-ce988da80287"></a>
## betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.py) · 85 行 · 说明来源：文件族规则

- **作用**：ALPHA23 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha23_timing](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.py#L29) | compute_alpha23_timing(high_wide, *, high_mean_window=20, high_delta_lag=2, high_ts_mean_false_value=0.0, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(23, high_wide=high_wide, high_mean_window=high_mean_window, high_delta_lag=high_delta_lag, high_ts_mean_false_value=high_ts_mean_false_value) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.py#L48) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.py#L68) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.py#L77) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-cc4df2ed6054"></a>
## betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.yaml#L8)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA23/factor_ALPHA23_timing.yaml#L39)：`run:`

<a id="file-23cced0ea987"></a>
## betalens-factor/alpha101/ALPHA24/factor_ALPHA24.py

[打开源码](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24.py) · 88 行 · 说明来源：文件族规则

- **作用**：ALPHA24 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha24](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24.py#L29) | compute_alpha24(close_wide, *, close_mean_window=100, ts_mean_close_delta_lag=100, close_delay_lag=100, trend_threshold=0.05, close_minimum_window=100, close_delta_lag=3) | 无返回注解；return: compute_alpha(24, close_wide=close_wide, close_mean_window=close_mean_window, ts_mean_close_delta_lag=ts_mean_close_delta_lag, close_delay_lag=close_delay_lag, trend_threshold=trend_threshold, close_minimum_window=close_minimum_window, close_delta_lag=close_delta_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24.py#L51) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24.py#L71) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24.py#L80) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-adf7d77a3b36"></a>
## betalens-factor/alpha101/ALPHA24/factor_ALPHA24.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24.yaml) · 44 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24.yaml#L8)：`factor_spec:`
- [L28](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24.yaml#L28)：`weight:`
- [L34](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24.yaml#L34)：`run:`

<a id="file-82c2c6232dea"></a>
## betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.py) · 91 行 · 说明来源：文件族规则

- **作用**：ALPHA24 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha24_timing](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.py#L29) | compute_alpha24_timing(close_wide, *, close_mean_window=100, ts_mean_close_delta_lag=100, close_delay_lag=100, trend_threshold=0.05, close_minimum_window=100, close_delta_lag=3, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(24, close_wide=close_wide, close_mean_window=close_mean_window, ts_mean_close_delta_lag=ts_mean_close_delta_lag, close_delay_lag=close_delay_lag, trend_threshold=trend_threshold, close_minimum_window=close_minimum_window, close_delta_lag=close_delta_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.py#L54) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.py#L74) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.py#L83) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-3e887a2ffda3"></a>
## betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.yaml) · 52 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.yaml#L8)：`factor_spec:`
- [L36](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.yaml#L36)：`weight:`
- [L42](../../../betalens-factor/alpha101/ALPHA24/factor_ALPHA24_timing.yaml#L42)：`run:`

<a id="file-e6eafbd16628"></a>
## betalens-factor/alpha101/ALPHA25/factor_ALPHA25.py

[打开源码](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25.py) · 86 行 · 说明来源：文件族规则

- **作用**：ALPHA25 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha25](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25.py#L29) | compute_alpha25(close_wide, high_wide, vwap_wide, returns_wide, amount_wide, *, amount_average_window=20) | 无返回注解；return: compute_alpha(25, close_wide=close_wide, high_wide=high_wide, vwap_wide=vwap_wide, returns_wide=returns_wide, amount_wide=amount_wide, amount_average_window=amount_average_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25.py#L49) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25.py#L69) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25.py#L78) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-3afb69155a77"></a>
## betalens-factor/alpha101/ALPHA25/factor_ALPHA25.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25.yaml) · 42 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25.yaml#L7)：`factor_spec:`
- [L26](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25.yaml#L26)：`weight:`
- [L32](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25.yaml#L32)：`run:`

<a id="file-e0e099d4dc89"></a>
## betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.py) · 89 行 · 说明来源：文件族规则

- **作用**：ALPHA25 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha25_timing](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.py#L29) | compute_alpha25_timing(close_wide, high_wide, vwap_wide, returns_wide, amount_wide, *, amount_average_window=20, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(25, close_wide=close_wide, high_wide=high_wide, vwap_wide=vwap_wide, returns_wide=returns_wide, amount_wide=amount_wide, amount_average_window=amount_average_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.py#L52) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.py#L72) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.py#L81) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-8eb9b386080f"></a>
## betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.yaml) · 51 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.yaml#L8)：`factor_spec:`
- [L35](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.yaml#L35)：`weight:`
- [L41](../../../betalens-factor/alpha101/ALPHA25/factor_ALPHA25_timing.yaml#L41)：`run:`

<a id="file-368258e9ecbb"></a>
## betalens-factor/alpha101/ALPHA26/factor_ALPHA26.py

[打开源码](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26.py) · 86 行 · 说明来源：文件族规则

- **作用**：ALPHA26 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha26](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26.py#L29) | compute_alpha26(high_wide, volume_wide, *, volume_rank_window=5, high_rank_window=5, ts_rank_volume_ts_rank_high_correlation_window=5, correlation_ts_rank_volume_maximum_window=3) | 无返回注解；return: compute_alpha(26, high_wide=high_wide, volume_wide=volume_wide, volume_rank_window=volume_rank_window, high_rank_window=high_rank_window, ts_rank_volume_ts_rank_high_correlation_window=ts_rank_volume_ts_rank_high_correlation_window, correlation_ts_rank_volume_maximum_window=correlation…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26.py#L49) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26.py#L69) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26.py#L78) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-160a5deec2e2"></a>
## betalens-factor/alpha101/ALPHA26/factor_ALPHA26.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26.yaml) · 42 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26.yaml#L7)：`factor_spec:`
- [L26](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26.yaml#L26)：`weight:`
- [L32](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26.yaml#L32)：`run:`

<a id="file-dc272d147357"></a>
## betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.py) · 89 行 · 说明来源：文件族规则

- **作用**：ALPHA26 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha26_timing](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.py#L29) | compute_alpha26_timing(high_wide, volume_wide, *, volume_rank_window=5, high_rank_window=5, ts_rank_volume_ts_rank_high_correlation_window=5, correlation_ts_rank_volume_maximum_window=3, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(26, high_wide=high_wide, volume_wide=volume_wide, volume_rank_window=volume_rank_window, high_rank_window=high_rank_window, ts_rank_volume_ts_rank_high_correlation_window=ts_rank_volume_ts_rank_high_correlation_window, correlation_ts_rank_volume_maximum_window=correlation…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.py#L52) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.py#L72) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.py#L81) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-da884a19081c"></a>
## betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.yaml) · 51 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.yaml#L8)：`factor_spec:`
- [L35](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.yaml#L35)：`weight:`
- [L41](../../../betalens-factor/alpha101/ALPHA26/factor_ALPHA26_timing.yaml#L41)：`run:`

<a id="file-1fa0537737dd"></a>
## betalens-factor/alpha101/ALPHA27/factor_ALPHA27.py

[打开源码](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27.py) · 90 行 · 说明来源：文件族规则

- **作用**：ALPHA27 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha27](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27.py#L29) | compute_alpha27(volume_wide, vwap_wide, *, rank_volume_rank_vwap_correlation_window=6, correlation_rank_volume_sum_window=2, ts_sum_correlation_rank_divisor=2.0, value_threshold=0.5, value_true_value=-1.0, value_false_value=1.0) | 无返回注解；return: compute_alpha(27, volume_wide=volume_wide, vwap_wide=vwap_wide, rank_volume_rank_vwap_correlation_window=rank_volume_rank_vwap_correlation_window, correlation_rank_volume_sum_window=correlation_rank_volume_sum_window, ts_sum_correlation_rank_divisor=ts_sum_correlation_rank_divisor, val…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27.py#L53) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27.py#L73) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27.py#L82) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-a253a8b7bf9d"></a>
## betalens-factor/alpha101/ALPHA27/factor_ALPHA27.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27.yaml) · 44 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27.yaml#L7)：`factor_spec:`
- [L28](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27.yaml#L28)：`weight:`
- [L34](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27.yaml#L34)：`run:`

<a id="file-9d74b5a7d51d"></a>
## betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.py) · 93 行 · 说明来源：文件族规则

- **作用**：ALPHA27 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha27_timing](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.py#L29) | compute_alpha27_timing(volume_wide, vwap_wide, *, rank_volume_rank_vwap_correlation_window=6, correlation_rank_volume_sum_window=2, ts_sum_correlation_rank_divisor=2.0, value_threshold=0.5, value_true_value=-1.0, value_false_value=1.0, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(27, volume_wide=volume_wide, vwap_wide=vwap_wide, rank_volume_rank_vwap_correlation_window=rank_volume_rank_vwap_correlation_window, correlation_rank_volume_sum_window=correlation_rank_volume_sum_window, ts_sum_correlation_rank_divisor=ts_sum_correlation_rank_divisor, val…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.py#L56) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.py#L76) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.py#L85) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-dab0bfcf6aa9"></a>
## betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.yaml) · 53 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.yaml#L8)：`factor_spec:`
- [L37](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.yaml#L37)：`weight:`
- [L43](../../../betalens-factor/alpha101/ALPHA27/factor_ALPHA27_timing.yaml#L43)：`run:`

<a id="file-7e39f0536a02"></a>
## betalens-factor/alpha101/ALPHA28/factor_ALPHA28.py

[打开源码](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28.py) · 88 行 · 说明来源：文件族规则

- **作用**：ALPHA28 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha28](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28.py#L29) | compute_alpha28(close_wide, high_wide, low_wide, amount_wide, *, amount_average_window=20, adv_low_correlation_window=5, high_low_divisor=2) | 无返回注解；return: compute_alpha(28, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, adv_low_correlation_window=adv_low_correlation_window, high_low_divisor=high_low_divisor) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28.py#L51) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28.py#L71) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28.py#L80) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-f0814fab5d3b"></a>
## betalens-factor/alpha101/ALPHA28/factor_ALPHA28.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28.yaml) · 43 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28.yaml#L7)：`factor_spec:`
- [L27](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28.yaml#L27)：`weight:`
- [L33](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28.yaml#L33)：`run:`

<a id="file-d9d4d0cc9083"></a>
## betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.py) · 91 行 · 说明来源：文件族规则

- **作用**：ALPHA28 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha28_timing](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.py#L29) | compute_alpha28_timing(close_wide, high_wide, low_wide, amount_wide, *, amount_average_window=20, adv_low_correlation_window=5, high_low_divisor=2, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(28, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, adv_low_correlation_window=adv_low_correlation_window, high_low_divisor=high_low_divisor) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.py#L54) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.py#L74) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.py#L83) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-c639db84c83d"></a>
## betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.yaml) · 52 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.yaml#L8)：`factor_spec:`
- [L36](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.yaml#L36)：`weight:`
- [L42](../../../betalens-factor/alpha101/ALPHA28/factor_ALPHA28_timing.yaml#L42)：`run:`

<a id="file-66bdaaf320d5"></a>
## betalens-factor/alpha101/ALPHA29/factor_ALPHA29.py

[打开源码](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29.py) · 94 行 · 说明来源：文件族规则

- **作用**：ALPHA29 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha29](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29.py#L29) | compute_alpha29(close_wide, returns_wide, *, close_center=1, close_delta_lag=5, rank_inner_minimum_window=2, ts_min_rank_inner_sum_window=1, rank_scale_ts_sum_product_window=1, product_rank_scale_minimum_window=5, returns_delay_lag=6, delay_returns_rank_window=5) | 无返回注解；return: compute_alpha(29, close_wide=close_wide, returns_wide=returns_wide, close_center=close_center, close_delta_lag=close_delta_lag, rank_inner_minimum_window=rank_inner_minimum_window, ts_min_rank_inner_sum_window=ts_min_rank_inner_sum_window, rank_scale_ts_sum_product_window=rank_scale_ts…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29.py#L57) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29.py#L77) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29.py#L86) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d5c13025038f"></a>
## betalens-factor/alpha101/ALPHA29/factor_ALPHA29.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29.yaml) · 46 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29.yaml#L7)：`factor_spec:`
- [L30](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29.yaml#L30)：`weight:`
- [L36](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29.yaml#L36)：`run:`

<a id="file-fa2814641367"></a>
## betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.py) · 97 行 · 说明来源：文件族规则

- **作用**：ALPHA29 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha29_timing](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.py#L29) | compute_alpha29_timing(close_wide, returns_wide, *, close_center=1, close_delta_lag=5, rank_inner_minimum_window=2, ts_min_rank_inner_sum_window=1, rank_scale_ts_sum_product_window=1, product_rank_scale_minimum_window=5, returns_delay_lag=6, delay_returns_rank_window=5, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(29, close_wide=close_wide, returns_wide=returns_wide, close_center=close_center, close_delta_lag=close_delta_lag, rank_inner_minimum_window=rank_inner_minimum_window, ts_min_rank_inner_sum_window=ts_min_rank_inner_sum_window, rank_scale_ts_sum_product_window=rank_scale_ts…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.py#L60) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.py#L80) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.py#L89) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ecadf5ba956f"></a>
## betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.yaml) · 55 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.yaml#L8)：`factor_spec:`
- [L39](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.yaml#L39)：`weight:`
- [L45](../../../betalens-factor/alpha101/ALPHA29/factor_ALPHA29_timing.yaml#L45)：`run:`

<a id="file-49571cbe4ea9"></a>
## betalens-factor/alpha101/ALPHA3/factor_ALPHA3.py

[打开源码](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3.py) · 80 行 · 说明来源：文件族规则

- **作用**：ALPHA3 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha3](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3.py#L29) | compute_alpha3(open_wide, volume_wide, *, rank_open_rank_volume_correlation_window=10) | 无返回注解；return: compute_alpha(3, open_wide=open_wide, volume_wide=volume_wide, rank_open_rank_volume_correlation_window=rank_open_rank_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3.py#L43) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3.py#L63) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3.py#L72) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-6c706ce36f0b"></a>
## betalens-factor/alpha101/ALPHA3/factor_ALPHA3.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3.yaml) · 39 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3.yaml#L7)：`factor_spec:`
- [L23](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3.yaml#L23)：`weight:`
- [L29](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3.yaml#L29)：`run:`

<a id="file-dc0e49231151"></a>
## betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.py) · 83 行 · 说明来源：文件族规则

- **作用**：ALPHA3 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha3_timing](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.py#L29) | compute_alpha3_timing(open_wide, volume_wide, *, rank_open_rank_volume_correlation_window=10, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(3, open_wide=open_wide, volume_wide=volume_wide, rank_open_rank_volume_correlation_window=rank_open_rank_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.py#L46) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.py#L66) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.py#L75) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ff24c9e1118f"></a>
## betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.yaml#L8)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA3/factor_ALPHA3_timing.yaml#L38)：`run:`

<a id="file-8bc136955bd9"></a>
## betalens-factor/alpha101/ALPHA30/factor_ALPHA30.py

[打开源码](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30.py) · 94 行 · 说明来源：文件族规则

- **作用**：ALPHA30 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha30](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30.py#L29) | compute_alpha30(close_wide, volume_wide, *, close_delay_lag=1, close_delay_lag_2=1, close_delay_lag_3=2, close_delay_lag_4=2, close_delay_lag_5=3, rank_direction_complement_base=1, volume_sum_window=5, volume_sum_window_2=20) | 无返回注解；return: compute_alpha(30, close_wide=close_wide, volume_wide=volume_wide, close_delay_lag=close_delay_lag, close_delay_lag_2=close_delay_lag_2, close_delay_lag_3=close_delay_lag_3, close_delay_lag_4=close_delay_lag_4, close_delay_lag_5=close_delay_lag_5, rank_direction_complement_base=rank_dir…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30.py#L57) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30.py#L77) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30.py#L86) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ebc3f8001419"></a>
## betalens-factor/alpha101/ALPHA30/factor_ALPHA30.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30.yaml) · 46 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30.yaml#L7)：`factor_spec:`
- [L30](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30.yaml#L30)：`weight:`
- [L36](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30.yaml#L36)：`run:`

<a id="file-5491da0c4c02"></a>
## betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.py) · 97 行 · 说明来源：文件族规则

- **作用**：ALPHA30 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha30_timing](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.py#L29) | compute_alpha30_timing(close_wide, volume_wide, *, close_delay_lag=1, close_delay_lag_2=1, close_delay_lag_3=2, close_delay_lag_4=2, close_delay_lag_5=3, rank_direction_complement_base=1, volume_sum_window=5, volume_sum_window_2=20, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(30, close_wide=close_wide, volume_wide=volume_wide, close_delay_lag=close_delay_lag, close_delay_lag_2=close_delay_lag_2, close_delay_lag_3=close_delay_lag_3, close_delay_lag_4=close_delay_lag_4, close_delay_lag_5=close_delay_lag_5, rank_direction_complement_base=rank_dir…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.py#L60) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.py#L80) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.py#L89) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d95ff48313f4"></a>
## betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.yaml) · 55 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.yaml#L8)：`factor_spec:`
- [L39](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.yaml#L39)：`weight:`
- [L45](../../../betalens-factor/alpha101/ALPHA30/factor_ALPHA30_timing.yaml#L45)：`run:`

<a id="file-461eabe3b7c5"></a>
## betalens-factor/alpha101/ALPHA31/factor_ALPHA31.py

[打开源码](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31.py) · 90 行 · 说明来源：文件族规则

- **作用**：ALPHA31 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha31](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31.py#L29) | compute_alpha31(close_wide, low_wide, amount_wide, *, close_delta_lag=10, rank_delta_close_decay_window=10, close_delta_lag_2=3, amount_average_window=20, adv_low_correlation_window=12) | 无返回注解；return: compute_alpha(31, close_wide=close_wide, low_wide=low_wide, amount_wide=amount_wide, close_delta_lag=close_delta_lag, rank_delta_close_decay_window=rank_delta_close_decay_window, close_delta_lag_2=close_delta_lag_2, amount_average_window=amount_average_window, adv_low_correlation_windo…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31.py#L53) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31.py#L73) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31.py#L82) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-b9c67da27601"></a>
## betalens-factor/alpha101/ALPHA31/factor_ALPHA31.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31.yaml) · 44 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31.yaml#L7)：`factor_spec:`
- [L28](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31.yaml#L28)：`weight:`
- [L34](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31.yaml#L34)：`run:`

<a id="file-6508b3f8b555"></a>
## betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.py) · 93 行 · 说明来源：文件族规则

- **作用**：ALPHA31 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha31_timing](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.py#L29) | compute_alpha31_timing(close_wide, low_wide, amount_wide, *, close_delta_lag=10, rank_delta_close_decay_window=10, close_delta_lag_2=3, amount_average_window=20, adv_low_correlation_window=12, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(31, close_wide=close_wide, low_wide=low_wide, amount_wide=amount_wide, close_delta_lag=close_delta_lag, rank_delta_close_decay_window=rank_delta_close_decay_window, close_delta_lag_2=close_delta_lag_2, amount_average_window=amount_average_window, adv_low_correlation_windo…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.py#L56) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.py#L76) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.py#L85) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-b938bbb9c64e"></a>
## betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.yaml) · 53 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.yaml#L8)：`factor_spec:`
- [L37](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.yaml#L37)：`weight:`
- [L43](../../../betalens-factor/alpha101/ALPHA31/factor_ALPHA31_timing.yaml#L43)：`run:`

<a id="file-7a8398bd06fc"></a>
## betalens-factor/alpha101/ALPHA32/factor_ALPHA32.py

[打开源码](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32.py) · 86 行 · 说明来源：文件族规则

- **作用**：ALPHA32 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha32](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32.py#L29) | compute_alpha32(close_wide, vwap_wide, *, close_mean_window=7, scale_correlation_vwap_coefficient=20, close_delay_lag=5, vwap_delay_close_correlation_window=230) | 无返回注解；return: compute_alpha(32, close_wide=close_wide, vwap_wide=vwap_wide, close_mean_window=close_mean_window, scale_correlation_vwap_coefficient=scale_correlation_vwap_coefficient, close_delay_lag=close_delay_lag, vwap_delay_close_correlation_window=vwap_delay_close_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32.py#L49) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32.py#L69) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32.py#L78) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-0248772f0830"></a>
## betalens-factor/alpha101/ALPHA32/factor_ALPHA32.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32.yaml) · 42 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32.yaml#L7)：`factor_spec:`
- [L26](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32.yaml#L26)：`weight:`
- [L32](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32.yaml#L32)：`run:`

<a id="file-d596ebf1f6e3"></a>
## betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.py) · 89 行 · 说明来源：文件族规则

- **作用**：ALPHA32 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha32_timing](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.py#L29) | compute_alpha32_timing(close_wide, vwap_wide, *, close_mean_window=7, scale_correlation_vwap_coefficient=20, close_delay_lag=5, vwap_delay_close_correlation_window=230, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(32, close_wide=close_wide, vwap_wide=vwap_wide, close_mean_window=close_mean_window, scale_correlation_vwap_coefficient=scale_correlation_vwap_coefficient, close_delay_lag=close_delay_lag, vwap_delay_close_correlation_window=vwap_delay_close_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.py#L52) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.py#L72) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.py#L81) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d087bc19db4c"></a>
## betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.yaml) · 51 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.yaml#L8)：`factor_spec:`
- [L35](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.yaml#L35)：`weight:`
- [L41](../../../betalens-factor/alpha101/ALPHA32/factor_ALPHA32_timing.yaml#L41)：`run:`

<a id="file-e657ba135c0e"></a>
## betalens-factor/alpha101/ALPHA33/factor_ALPHA33.py

[打开源码](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33.py) · 82 行 · 说明来源：文件族规则

- **作用**：ALPHA33 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha33](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33.py#L29) | compute_alpha33(open_wide, close_wide, *, open_close_coefficient=-1, open_close_complement_base=1) | 无返回注解；return: compute_alpha(33, open_wide=open_wide, close_wide=close_wide, open_close_coefficient=open_close_coefficient, open_close_complement_base=open_close_complement_base) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33.py#L45) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33.py#L65) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33.py#L74) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-cc64f7d85637"></a>
## betalens-factor/alpha101/ALPHA33/factor_ALPHA33.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33.yaml) · 40 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33.yaml#L7)：`factor_spec:`
- [L24](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33.yaml#L24)：`weight:`
- [L30](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33.yaml#L30)：`run:`

<a id="file-1379f3b75d22"></a>
## betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.py) · 85 行 · 说明来源：文件族规则

- **作用**：ALPHA33 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha33_timing](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.py#L29) | compute_alpha33_timing(open_wide, close_wide, *, open_close_coefficient=-1, open_close_complement_base=1, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(33, open_wide=open_wide, close_wide=close_wide, open_close_coefficient=open_close_coefficient, open_close_complement_base=open_close_complement_base) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.py#L48) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.py#L68) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.py#L77) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-9ee868cbea0c"></a>
## betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.yaml#L8)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA33/factor_ALPHA33_timing.yaml#L39)：`run:`

<a id="file-9a670dfc863c"></a>
## betalens-factor/alpha101/ALPHA34/factor_ALPHA34.py

[打开源码](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34.py) · 88 行 · 说明来源：文件族规则

- **作用**：ALPHA34 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha34](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34.py#L29) | compute_alpha34(close_wide, returns_wide, *, rank_stddev_returns_complement_base=1, returns_stddev_window=2, returns_stddev_window_2=5, rank_delta_close_complement_base=1, close_delta_lag=1) | 无返回注解；return: compute_alpha(34, close_wide=close_wide, returns_wide=returns_wide, rank_stddev_returns_complement_base=rank_stddev_returns_complement_base, returns_stddev_window=returns_stddev_window, returns_stddev_window_2=returns_stddev_window_2, rank_delta_close_complement_base=rank_delta_close_c…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34.py#L51) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34.py#L71) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34.py#L80) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-a71abd6b2456"></a>
## betalens-factor/alpha101/ALPHA34/factor_ALPHA34.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34.yaml) · 43 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34.yaml#L7)：`factor_spec:`
- [L27](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34.yaml#L27)：`weight:`
- [L33](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34.yaml#L33)：`run:`

<a id="file-810b2b1e8c7a"></a>
## betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.py) · 91 行 · 说明来源：文件族规则

- **作用**：ALPHA34 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha34_timing](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.py#L29) | compute_alpha34_timing(close_wide, returns_wide, *, rank_stddev_returns_complement_base=1, returns_stddev_window=2, returns_stddev_window_2=5, rank_delta_close_complement_base=1, close_delta_lag=1, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(34, close_wide=close_wide, returns_wide=returns_wide, rank_stddev_returns_complement_base=rank_stddev_returns_complement_base, returns_stddev_window=returns_stddev_window, returns_stddev_window_2=returns_stddev_window_2, rank_delta_close_complement_base=rank_delta_close_c…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.py#L54) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.py#L74) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.py#L83) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-8df2ae3ebb77"></a>
## betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.yaml) · 52 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.yaml#L8)：`factor_spec:`
- [L36](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.yaml#L36)：`weight:`
- [L42](../../../betalens-factor/alpha101/ALPHA34/factor_ALPHA34_timing.yaml#L42)：`run:`

<a id="file-d6eba0953206"></a>
## betalens-factor/alpha101/ALPHA35/factor_ALPHA35.py

[打开源码](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35.py) · 94 行 · 说明来源：文件族规则

- **作用**：ALPHA35 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha35](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35.py#L29) | compute_alpha35(close_wide, high_wide, low_wide, volume_wide, returns_wide, *, volume_rank_window=32, ts_rank_low_close_complement_base=1, low_close_high_rank_window=16, ts_rank_returns_complement_base=1, returns_rank_window=32) | 无返回注解；return: compute_alpha(35, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, returns_wide=returns_wide, volume_rank_window=volume_rank_window, ts_rank_low_close_complement_base=ts_rank_low_close_complement_base, low_close_high_rank_window=low_close_high_ran…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35.py#L57) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35.py#L77) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35.py#L86) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-064a4fac375e"></a>
## betalens-factor/alpha101/ALPHA35/factor_ALPHA35.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35.yaml) · 46 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35.yaml#L7)：`factor_spec:`
- [L30](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35.yaml#L30)：`weight:`
- [L36](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35.yaml#L36)：`run:`

<a id="file-08911b982b50"></a>
## betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.py) · 97 行 · 说明来源：文件族规则

- **作用**：ALPHA35 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha35_timing](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.py#L29) | compute_alpha35_timing(close_wide, high_wide, low_wide, volume_wide, returns_wide, *, volume_rank_window=32, ts_rank_low_close_complement_base=1, low_close_high_rank_window=16, ts_rank_returns_complement_base=1, returns_rank_window=32, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(35, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, returns_wide=returns_wide, volume_rank_window=volume_rank_window, ts_rank_low_close_complement_base=ts_rank_low_close_complement_base, low_close_high_rank_window=low_close_high_ran…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.py#L60) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.py#L80) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.py#L89) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-e8cd8f3ae4cb"></a>
## betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.yaml) · 55 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.yaml#L8)：`factor_spec:`
- [L39](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.yaml#L39)：`weight:`
- [L45](../../../betalens-factor/alpha101/ALPHA35/factor_ALPHA35_timing.yaml#L45)：`run:`

<a id="file-bc9f3683fb1e"></a>
## betalens-factor/alpha101/ALPHA36/factor_ALPHA36.py

[打开源码](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36.py) · 108 行 · 说明来源：文件族规则

- **作用**：ALPHA36 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha36](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36.py#L29) | compute_alpha36(open_wide, close_wide, volume_wide, vwap_wide, returns_wide, amount_wide, *, rank_correlation_close_coefficient=2.21, volume_delay_lag=1, close_open_delay_volume_correlation_window=15, rank_open_close_coefficient=0.7, rank_ts_rank_delay_coefficient=0.73, returns_delay_lag=6, delay_returns_rank_window=5, amount_average_window=20, vwap_adv_correlation_window=6, rank_open_close_coefficient_2=0.6, close_mean_window=200) | 无返回注解；return: compute_alpha(36, open_wide=open_wide, close_wide=close_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, returns_wide=returns_wide, amount_wide=amount_wide, rank_correlation_close_coefficient=rank_correlation_close_coefficient, volume_delay_lag=volume_delay_lag, close_open_delay_vol…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36.py#L71) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36.py#L91) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36.py#L100) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-0809dce0666d"></a>
## betalens-factor/alpha101/ALPHA36/factor_ALPHA36.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36.yaml) · 53 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36.yaml#L7)：`factor_spec:`
- [L37](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36.yaml#L37)：`weight:`
- [L43](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36.yaml#L43)：`run:`

<a id="file-775d10f6666b"></a>
## betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.py) · 111 行 · 说明来源：文件族规则

- **作用**：ALPHA36 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha36_timing](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.py#L29) | compute_alpha36_timing(open_wide, close_wide, volume_wide, vwap_wide, returns_wide, amount_wide, *, rank_correlation_close_coefficient=2.21, volume_delay_lag=1, close_open_delay_volume_correlation_window=15, rank_open_close_coefficient=0.7, rank_ts_rank_delay_coefficient=0.73, returns_delay_lag=6, delay_returns_rank_window=5, amount_average_window=20, vwap_adv_correlation_window=6, rank_open_close_coefficient_2=0.6, close_mean_window=200, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(36, open_wide=open_wide, close_wide=close_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, returns_wide=returns_wide, amount_wide=amount_wide, rank_correlation_close_coefficient=rank_correlation_close_coefficient, volume_delay_lag=volume_delay_lag, close_open_delay_vol…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.py#L74) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.py#L94) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.py#L103) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-fb2d73aa9c10"></a>
## betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.yaml) · 62 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.yaml#L8)：`factor_spec:`
- [L46](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.yaml#L46)：`weight:`
- [L52](../../../betalens-factor/alpha101/ALPHA36/factor_ALPHA36_timing.yaml#L52)：`run:`

<a id="file-923ace5ea4c4"></a>
## betalens-factor/alpha101/ALPHA37/factor_ALPHA37.py

[打开源码](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37.py) · 82 行 · 说明来源：文件族规则

- **作用**：ALPHA37 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha37](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37.py#L29) | compute_alpha37(open_wide, close_wide, *, open_close_delay_lag=1, delay_open_close_close_correlation_window=200) | 无返回注解；return: compute_alpha(37, open_wide=open_wide, close_wide=close_wide, open_close_delay_lag=open_close_delay_lag, delay_open_close_close_correlation_window=delay_open_close_close_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37.py#L45) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37.py#L65) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37.py#L74) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-478f8182ba59"></a>
## betalens-factor/alpha101/ALPHA37/factor_ALPHA37.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37.yaml) · 40 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37.yaml#L7)：`factor_spec:`
- [L24](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37.yaml#L24)：`weight:`
- [L30](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37.yaml#L30)：`run:`

<a id="file-e29289105283"></a>
## betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.py) · 85 行 · 说明来源：文件族规则

- **作用**：ALPHA37 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha37_timing](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.py#L29) | compute_alpha37_timing(open_wide, close_wide, *, open_close_delay_lag=1, delay_open_close_close_correlation_window=200, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(37, open_wide=open_wide, close_wide=close_wide, open_close_delay_lag=open_close_delay_lag, delay_open_close_close_correlation_window=delay_open_close_close_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.py#L48) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.py#L68) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.py#L77) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-97f5775f05ba"></a>
## betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.yaml#L8)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA37/factor_ALPHA37_timing.yaml#L39)：`run:`

<a id="file-13d052d1c723"></a>
## betalens-factor/alpha101/ALPHA38/factor_ALPHA38.py

[打开源码](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38.py) · 80 行 · 说明来源：文件族规则

- **作用**：ALPHA38 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha38](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38.py#L29) | compute_alpha38(open_wide, close_wide, *, close_rank_window=10) | 无返回注解；return: compute_alpha(38, open_wide=open_wide, close_wide=close_wide, close_rank_window=close_rank_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38.py#L43) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38.py#L63) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38.py#L72) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-9ea1487131b2"></a>
## betalens-factor/alpha101/ALPHA38/factor_ALPHA38.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38.yaml) · 39 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38.yaml#L7)：`factor_spec:`
- [L23](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38.yaml#L23)：`weight:`
- [L29](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38.yaml#L29)：`run:`

<a id="file-a6f6f4602bac"></a>
## betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.py) · 83 行 · 说明来源：文件族规则

- **作用**：ALPHA38 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha38_timing](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.py#L29) | compute_alpha38_timing(open_wide, close_wide, *, close_rank_window=10, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(38, open_wide=open_wide, close_wide=close_wide, close_rank_window=close_rank_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.py#L46) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.py#L66) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.py#L75) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-175ab607de52"></a>
## betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.yaml#L8)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA38/factor_ALPHA38_timing.yaml#L38)：`run:`

<a id="file-a5bf02ddc8be"></a>
## betalens-factor/alpha101/ALPHA39/factor_ALPHA39.py

[打开源码](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39.py) · 94 行 · 说明来源：文件族规则

- **作用**：ALPHA39 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha39](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39.py#L29) | compute_alpha39(close_wide, volume_wide, returns_wide, amount_wide, *, close_delta_lag=7, rank_decay_linear_volume_complement_base=1, amount_average_window=20, volume_adv_decay_window=9, rank_ts_sum_returns_offset=1, returns_sum_window=250) | 无返回注解；return: compute_alpha(39, close_wide=close_wide, volume_wide=volume_wide, returns_wide=returns_wide, amount_wide=amount_wide, close_delta_lag=close_delta_lag, rank_decay_linear_volume_complement_base=rank_decay_linear_volume_complement_base, amount_average_window=amount_average_window, volume_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39.py#L57) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39.py#L77) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39.py#L86) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-a91c660958cf"></a>
## betalens-factor/alpha101/ALPHA39/factor_ALPHA39.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39.yaml) · 46 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39.yaml#L7)：`factor_spec:`
- [L30](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39.yaml#L30)：`weight:`
- [L36](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39.yaml#L36)：`run:`

<a id="file-fe4885387889"></a>
## betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.py) · 97 行 · 说明来源：文件族规则

- **作用**：ALPHA39 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha39_timing](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.py#L29) | compute_alpha39_timing(close_wide, volume_wide, returns_wide, amount_wide, *, close_delta_lag=7, rank_decay_linear_volume_complement_base=1, amount_average_window=20, volume_adv_decay_window=9, rank_ts_sum_returns_offset=1, returns_sum_window=250, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(39, close_wide=close_wide, volume_wide=volume_wide, returns_wide=returns_wide, amount_wide=amount_wide, close_delta_lag=close_delta_lag, rank_decay_linear_volume_complement_base=rank_decay_linear_volume_complement_base, amount_average_window=amount_average_window, volume_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.py#L60) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.py#L80) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.py#L89) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-30acab8aac67"></a>
## betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.yaml) · 55 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.yaml#L8)：`factor_spec:`
- [L39](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.yaml#L39)：`weight:`
- [L45](../../../betalens-factor/alpha101/ALPHA39/factor_ALPHA39_timing.yaml#L45)：`run:`

<a id="file-cd1fce0ce707"></a>
## betalens-factor/alpha101/ALPHA4/factor_ALPHA4.py

[打开源码](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4.py) · 78 行 · 说明来源：文件族规则

- **作用**：ALPHA4 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha4](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4.py#L29) | compute_alpha4(low_wide, *, rank_low_rank_window=9) | 无返回注解；return: compute_alpha(4, low_wide=low_wide, rank_low_rank_window=rank_low_rank_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4.py#L41) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4.py#L61) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4.py#L70) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-01f7585dc31f"></a>
## betalens-factor/alpha101/ALPHA4/factor_ALPHA4.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4.yaml) · 38 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4.yaml#L7)：`factor_spec:`
- [L22](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4.yaml#L22)：`weight:`
- [L28](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4.yaml#L28)：`run:`

<a id="file-c123543ce7fd"></a>
## betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.py) · 81 行 · 说明来源：文件族规则

- **作用**：ALPHA4 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha4_timing](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.py#L29) | compute_alpha4_timing(low_wide, *, rank_low_rank_window=9, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(4, low_wide=low_wide, rank_low_rank_window=rank_low_rank_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.py#L44) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.py#L64) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.py#L73) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-28328ae3a6db"></a>
## betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.yaml) · 47 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.yaml#L8)：`factor_spec:`
- [L31](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.yaml#L31)：`weight:`
- [L37](../../../betalens-factor/alpha101/ALPHA4/factor_ALPHA4_timing.yaml#L37)：`run:`

<a id="file-b65f8dba3441"></a>
## betalens-factor/alpha101/ALPHA40/factor_ALPHA40.py

[打开源码](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40.py) · 82 行 · 说明来源：文件族规则

- **作用**：ALPHA40 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha40](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40.py#L29) | compute_alpha40(high_wide, volume_wide, *, high_stddev_window=10, high_volume_correlation_window=10) | 无返回注解；return: compute_alpha(40, high_wide=high_wide, volume_wide=volume_wide, high_stddev_window=high_stddev_window, high_volume_correlation_window=high_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40.py#L45) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40.py#L65) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40.py#L74) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-da41fef7709a"></a>
## betalens-factor/alpha101/ALPHA40/factor_ALPHA40.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40.yaml) · 40 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40.yaml#L7)：`factor_spec:`
- [L24](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40.yaml#L24)：`weight:`
- [L30](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40.yaml#L30)：`run:`

<a id="file-213b0d44a9ee"></a>
## betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.py) · 85 行 · 说明来源：文件族规则

- **作用**：ALPHA40 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha40_timing](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.py#L29) | compute_alpha40_timing(high_wide, volume_wide, *, high_stddev_window=10, high_volume_correlation_window=10, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(40, high_wide=high_wide, volume_wide=volume_wide, high_stddev_window=high_stddev_window, high_volume_correlation_window=high_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.py#L48) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.py#L68) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.py#L77) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-1bb35a8d5053"></a>
## betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.yaml#L8)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA40/factor_ALPHA40_timing.yaml#L39)：`run:`

<a id="file-7786e88c5d65"></a>
## betalens-factor/alpha101/ALPHA41/factor_ALPHA41.py

[打开源码](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41.py) · 79 行 · 说明来源：文件族规则

- **作用**：ALPHA41 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha41](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41.py#L29) | compute_alpha41(high_wide, low_wide, vwap_wide) | 无返回注解；return: compute_alpha(41, high_wide=high_wide, low_wide=low_wide, vwap_wide=vwap_wide) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41.py#L42) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41.py#L62) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41.py#L71) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-e594ec561ae4"></a>
## betalens-factor/alpha101/ALPHA41/factor_ALPHA41.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41.yaml) · 39 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41.yaml#L7)：`factor_spec:`
- [L23](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41.yaml#L23)：`weight:`
- [L29](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41.yaml#L29)：`run:`

<a id="file-b56afed968dc"></a>
## betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.py) · 83 行 · 说明来源：文件族规则

- **作用**：ALPHA41 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha41_timing](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.py#L29) | compute_alpha41_timing(high_wide, low_wide, vwap_wide, *, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(41, high_wide=high_wide, low_wide=low_wide, vwap_wide=vwap_wide) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.py#L46) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.py#L66) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.py#L75) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-3d19bdf3f8ac"></a>
## betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.yaml#L8)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA41/factor_ALPHA41_timing.yaml#L38)：`run:`

<a id="file-90fc57a16e51"></a>
## betalens-factor/alpha101/ALPHA42/factor_ALPHA42.py

[打开源码](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42.py) · 77 行 · 说明来源：文件族规则

- **作用**：ALPHA42 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha42](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42.py#L29) | compute_alpha42(close_wide, vwap_wide) | 无返回注解；return: compute_alpha(42, close_wide=close_wide, vwap_wide=vwap_wide) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42.py#L40) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42.py#L60) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42.py#L69) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-b6efd3206fe2"></a>
## betalens-factor/alpha101/ALPHA42/factor_ALPHA42.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42.yaml) · 38 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42.yaml#L7)：`factor_spec:`
- [L22](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42.yaml#L22)：`weight:`
- [L28](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42.yaml#L28)：`run:`

<a id="file-1941c3f9cd84"></a>
## betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.py) · 81 行 · 说明来源：文件族规则

- **作用**：ALPHA42 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha42_timing](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.py#L29) | compute_alpha42_timing(close_wide, vwap_wide, *, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(42, close_wide=close_wide, vwap_wide=vwap_wide) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.py#L44) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.py#L64) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.py#L73) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-3710fad47607"></a>
## betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.yaml) · 47 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.yaml#L8)：`factor_spec:`
- [L31](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.yaml#L31)：`weight:`
- [L37](../../../betalens-factor/alpha101/ALPHA42/factor_ALPHA42_timing.yaml#L37)：`run:`

<a id="file-d86629efe42a"></a>
## betalens-factor/alpha101/ALPHA43/factor_ALPHA43.py

[打开源码](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43.py) · 88 行 · 说明来源：文件族规则

- **作用**：ALPHA43 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha43](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43.py#L29) | compute_alpha43(close_wide, volume_wide, amount_wide, *, amount_average_window=20, volume_adv_rank_window=20, close_delta_lag=7, delta_close_rank_window=8) | 无返回注解；return: compute_alpha(43, close_wide=close_wide, volume_wide=volume_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, volume_adv_rank_window=volume_adv_rank_window, close_delta_lag=close_delta_lag, delta_close_rank_window=delta_close_rank_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43.py#L51) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43.py#L71) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43.py#L80) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-087bbac54183"></a>
## betalens-factor/alpha101/ALPHA43/factor_ALPHA43.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43.yaml) · 43 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43.yaml#L7)：`factor_spec:`
- [L27](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43.yaml#L27)：`weight:`
- [L33](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43.yaml#L33)：`run:`

<a id="file-6db7425419ac"></a>
## betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.py) · 91 行 · 说明来源：文件族规则

- **作用**：ALPHA43 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha43_timing](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.py#L29) | compute_alpha43_timing(close_wide, volume_wide, amount_wide, *, amount_average_window=20, volume_adv_rank_window=20, close_delta_lag=7, delta_close_rank_window=8, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(43, close_wide=close_wide, volume_wide=volume_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, volume_adv_rank_window=volume_adv_rank_window, close_delta_lag=close_delta_lag, delta_close_rank_window=delta_close_rank_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.py#L54) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.py#L74) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.py#L83) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-8603786ea902"></a>
## betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.yaml) · 52 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.yaml#L8)：`factor_spec:`
- [L36](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.yaml#L36)：`weight:`
- [L42](../../../betalens-factor/alpha101/ALPHA43/factor_ALPHA43_timing.yaml#L42)：`run:`

<a id="file-634f35ca4e4d"></a>
## betalens-factor/alpha101/ALPHA44/factor_ALPHA44.py

[打开源码](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44.py) · 80 行 · 说明来源：文件族规则

- **作用**：ALPHA44 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha44](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44.py#L29) | compute_alpha44(high_wide, volume_wide, *, high_rank_volume_correlation_window=5) | 无返回注解；return: compute_alpha(44, high_wide=high_wide, volume_wide=volume_wide, high_rank_volume_correlation_window=high_rank_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44.py#L43) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44.py#L63) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44.py#L72) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-33608f67a2e8"></a>
## betalens-factor/alpha101/ALPHA44/factor_ALPHA44.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44.yaml) · 39 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44.yaml#L7)：`factor_spec:`
- [L23](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44.yaml#L23)：`weight:`
- [L29](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44.yaml#L29)：`run:`

<a id="file-e0928ec50d28"></a>
## betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.py) · 83 行 · 说明来源：文件族规则

- **作用**：ALPHA44 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha44_timing](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.py#L29) | compute_alpha44_timing(high_wide, volume_wide, *, high_rank_volume_correlation_window=5, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(44, high_wide=high_wide, volume_wide=volume_wide, high_rank_volume_correlation_window=high_rank_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.py#L46) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.py#L66) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.py#L75) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-4cd4464f3d57"></a>
## betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.yaml#L8)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA44/factor_ALPHA44_timing.yaml#L38)：`run:`

<a id="file-09b7770e2131"></a>
## betalens-factor/alpha101/ALPHA45/factor_ALPHA45.py

[打开源码](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45.py) · 92 行 · 说明来源：文件族规则

- **作用**：ALPHA45 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha45](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45.py#L29) | compute_alpha45(close_wide, volume_wide, *, close_delay_lag=5, delay_close_sum_window=20, ts_sum_delay_close_divisor=20, close_volume_correlation_window=2, close_sum_window=5, close_sum_window_2=20, ts_sum_close_ts_sum_close_correlation_window=2) | 无返回注解；return: compute_alpha(45, close_wide=close_wide, volume_wide=volume_wide, close_delay_lag=close_delay_lag, delay_close_sum_window=delay_close_sum_window, ts_sum_delay_close_divisor=ts_sum_delay_close_divisor, close_volume_correlation_window=close_volume_correlation_window, close_sum_window=clo…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45.py#L55) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45.py#L75) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45.py#L84) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-998c9d66c9d0"></a>
## betalens-factor/alpha101/ALPHA45/factor_ALPHA45.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45.yaml) · 45 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45.yaml#L7)：`factor_spec:`
- [L29](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45.yaml#L29)：`weight:`
- [L35](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45.yaml#L35)：`run:`

<a id="file-2fb3856e0492"></a>
## betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.py) · 95 行 · 说明来源：文件族规则

- **作用**：ALPHA45 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha45_timing](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.py#L29) | compute_alpha45_timing(close_wide, volume_wide, *, close_delay_lag=5, delay_close_sum_window=20, ts_sum_delay_close_divisor=20, close_volume_correlation_window=2, close_sum_window=5, close_sum_window_2=20, ts_sum_close_ts_sum_close_correlation_window=2, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(45, close_wide=close_wide, volume_wide=volume_wide, close_delay_lag=close_delay_lag, delay_close_sum_window=delay_close_sum_window, ts_sum_delay_close_divisor=ts_sum_delay_close_divisor, close_volume_correlation_window=close_volume_correlation_window, close_sum_window=clo…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.py#L58) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.py#L78) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.py#L87) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ab84699883f9"></a>
## betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.yaml) · 54 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.yaml#L8)：`factor_spec:`
- [L38](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.yaml#L38)：`weight:`
- [L44](../../../betalens-factor/alpha101/ALPHA45/factor_ALPHA45_timing.yaml#L44)：`run:`

<a id="file-c0900bd93220"></a>
## betalens-factor/alpha101/ALPHA46/factor_ALPHA46.py

[打开源码](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46.py) · 96 行 · 说明来源：文件族规则

- **作用**：ALPHA46 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha46](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46.py#L29) | compute_alpha46(close_wide, *, trend_long_delay_lag=20, trend_first_short_delay_lag=10, trend_first_slope_divisor=10, trend_second_short_delay_lag=10, trend_second_slope_divisor=10, trend_threshold=0.25, trend_true_value=-1.0, trend_threshold_2=0, trend_true_value_2=1.0, close_delay_lag=1) | 无返回注解；return: compute_alpha(46, close_wide=close_wide, trend_long_delay_lag=trend_long_delay_lag, trend_first_short_delay_lag=trend_first_short_delay_lag, trend_first_slope_divisor=trend_first_slope_divisor, trend_second_short_delay_lag=trend_second_short_delay_lag, trend_second_slope_divisor=trend_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46.py#L59) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46.py#L79) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46.py#L88) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-2dd0ce060743"></a>
## betalens-factor/alpha101/ALPHA46/factor_ALPHA46.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46.yaml#L1)：`meta:`
- [L9](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46.yaml#L9)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46.yaml#L39)：`run:`

<a id="file-63ea7452a182"></a>
## betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.py) · 99 行 · 说明来源：文件族规则

- **作用**：ALPHA46 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha46_timing](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.py#L29) | compute_alpha46_timing(close_wide, *, trend_long_delay_lag=20, trend_first_short_delay_lag=10, trend_first_slope_divisor=10, trend_second_short_delay_lag=10, trend_second_slope_divisor=10, trend_threshold=0.25, trend_true_value=-1.0, trend_threshold_2=0, trend_true_value_2=1.0, close_delay_lag=1, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(46, close_wide=close_wide, trend_long_delay_lag=trend_long_delay_lag, trend_first_short_delay_lag=trend_first_short_delay_lag, trend_first_slope_divisor=trend_first_slope_divisor, trend_second_short_delay_lag=trend_second_short_delay_lag, trend_second_slope_divisor=trend_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.py#L62) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.py#L82) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.py#L91) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-c6e141a536f8"></a>
## betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.yaml) · 56 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.yaml#L8)：`factor_spec:`
- [L40](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.yaml#L40)：`weight:`
- [L46](../../../betalens-factor/alpha101/ALPHA46/factor_ALPHA46_timing.yaml#L46)：`run:`

<a id="file-449569935b6f"></a>
## betalens-factor/alpha101/ALPHA47/factor_ALPHA47.py

[打开源码](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47.py) · 92 行 · 说明来源：文件族规则

- **作用**：ALPHA47 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha47](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47.py#L29) | compute_alpha47(close_wide, high_wide, volume_wide, vwap_wide, amount_wide, *, close_divisor=1, amount_average_window=20, high_mean_window=5, vwap_delay_lag=5) | 无返回注解；return: compute_alpha(47, close_wide=close_wide, high_wide=high_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, close_divisor=close_divisor, amount_average_window=amount_average_window, high_mean_window=high_mean_window, vwap_delay_lag=vwap_delay_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47.py#L55) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47.py#L75) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47.py#L84) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-7d89bab8ad1b"></a>
## betalens-factor/alpha101/ALPHA47/factor_ALPHA47.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47.yaml) · 45 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47.yaml#L7)：`factor_spec:`
- [L29](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47.yaml#L29)：`weight:`
- [L35](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47.yaml#L35)：`run:`

<a id="file-670b7ff2c8c9"></a>
## betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.py) · 95 行 · 说明来源：文件族规则

- **作用**：ALPHA47 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha47_timing](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.py#L29) | compute_alpha47_timing(close_wide, high_wide, volume_wide, vwap_wide, amount_wide, *, close_divisor=1, amount_average_window=20, high_mean_window=5, vwap_delay_lag=5, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(47, close_wide=close_wide, high_wide=high_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, close_divisor=close_divisor, amount_average_window=amount_average_window, high_mean_window=high_mean_window, vwap_delay_lag=vwap_delay_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.py#L58) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.py#L78) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.py#L87) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-12eed1e8b8c4"></a>
## betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.yaml) · 54 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.yaml#L8)：`factor_spec:`
- [L38](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.yaml#L38)：`weight:`
- [L44](../../../betalens-factor/alpha101/ALPHA47/factor_ALPHA47_timing.yaml#L44)：`run:`

<a id="file-664ee2cbc61b"></a>
## betalens-factor/alpha101/ALPHA48/factor_ALPHA48.py

[打开源码](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48.py) · 96 行 · 说明来源：文件族规则

- **作用**：ALPHA48 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha48](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48.py#L29) | compute_alpha48(close_wide, subindustry_wide, *, close_delta_lag=1, close_delay_lag=1, delay_close_delta_lag=1, delta_close_delta_delay_close_correlation_window=250, close_delta_lag_2=1, close_delta_lag_3=1, close_delay_lag_2=1, delta_close_delay_power_exponent=2, delta_close_delay_sum_window=250) | 无返回注解；return: compute_alpha(48, close_wide=close_wide, subindustry_wide=subindustry_wide, close_delta_lag=close_delta_lag, close_delay_lag=close_delay_lag, delay_close_delta_lag=delay_close_delta_lag, delta_close_delta_delay_close_correlation_window=delta_close_delta_delay_close_correlation_window, …（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48.py#L59) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48.py#L79) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48.py#L88) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-0e03593f39b3"></a>
## betalens-factor/alpha101/ALPHA48/factor_ALPHA48.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48.yaml) · 47 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48.yaml#L7)：`factor_spec:`
- [L31](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48.yaml#L31)：`weight:`
- [L37](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48.yaml#L37)：`run:`

<a id="file-299ccec905ac"></a>
## betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.py) · 99 行 · 说明来源：文件族规则

- **作用**：ALPHA48 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha48_timing](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.py#L29) | compute_alpha48_timing(close_wide, subindustry_wide, *, close_delta_lag=1, close_delay_lag=1, delay_close_delta_lag=1, delta_close_delta_delay_close_correlation_window=250, close_delta_lag_2=1, close_delta_lag_3=1, close_delay_lag_2=1, delta_close_delay_power_exponent=2, delta_close_delay_sum_window=250, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(48, close_wide=close_wide, subindustry_wide=subindustry_wide, close_delta_lag=close_delta_lag, close_delay_lag=close_delay_lag, delay_close_delta_lag=delay_close_delta_lag, delta_close_delta_delay_close_correlation_window=delta_close_delta_delay_close_correlation_window, …（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.py#L62) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.py#L82) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.py#L91) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-114bcd7b0f4b"></a>
## betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.yaml) · 56 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.yaml#L8)：`factor_spec:`
- [L40](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.yaml#L40)：`weight:`
- [L46](../../../betalens-factor/alpha101/ALPHA48/factor_ALPHA48_timing.yaml#L46)：`run:`

<a id="file-799472d428f7"></a>
## betalens-factor/alpha101/ALPHA49/factor_ALPHA49.py

[打开源码](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49.py) · 92 行 · 说明来源：文件族规则

- **作用**：ALPHA49 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha49](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49.py#L29) | compute_alpha49(close_wide, *, trend_long_delay_lag=20, trend_first_short_delay_lag=10, trend_first_slope_divisor=10, trend_second_short_delay_lag=10, trend_second_slope_divisor=10, alpha_trend_threshold=-0.1, alpha_trend_true_value=1.0, close_delay_lag=1) | 无返回注解；return: compute_alpha(49, close_wide=close_wide, trend_long_delay_lag=trend_long_delay_lag, trend_first_short_delay_lag=trend_first_short_delay_lag, trend_first_slope_divisor=trend_first_slope_divisor, trend_second_short_delay_lag=trend_second_short_delay_lag, trend_second_slope_divisor=trend_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49.py#L55) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49.py#L75) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49.py#L84) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-2871249aff9c"></a>
## betalens-factor/alpha101/ALPHA49/factor_ALPHA49.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49.yaml) · 46 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49.yaml#L8)：`factor_spec:`
- [L30](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49.yaml#L30)：`weight:`
- [L36](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49.yaml#L36)：`run:`

<a id="file-aca9c0c19c8f"></a>
## betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.py) · 95 行 · 说明来源：文件族规则

- **作用**：ALPHA49 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha49_timing](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.py#L29) | compute_alpha49_timing(close_wide, *, trend_long_delay_lag=20, trend_first_short_delay_lag=10, trend_first_slope_divisor=10, trend_second_short_delay_lag=10, trend_second_slope_divisor=10, alpha_trend_threshold=-0.1, alpha_trend_true_value=1.0, close_delay_lag=1, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(49, close_wide=close_wide, trend_long_delay_lag=trend_long_delay_lag, trend_first_short_delay_lag=trend_first_short_delay_lag, trend_first_slope_divisor=trend_first_slope_divisor, trend_second_short_delay_lag=trend_second_short_delay_lag, trend_second_slope_divisor=trend_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.py#L58) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.py#L78) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.py#L87) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ab4a9758c78d"></a>
## betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.yaml) · 54 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.yaml#L8)：`factor_spec:`
- [L38](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.yaml#L38)：`weight:`
- [L44](../../../betalens-factor/alpha101/ALPHA49/factor_ALPHA49_timing.yaml#L44)：`run:`

<a id="file-cc0967c39654"></a>
## betalens-factor/alpha101/ALPHA5/factor_ALPHA5.py

[打开源码](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5.py) · 84 行 · 说明来源：文件族规则

- **作用**：ALPHA5 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha5](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5.py#L29) | compute_alpha5(open_wide, close_wide, vwap_wide, *, vwap_sum_window=10, ts_sum_vwap_divisor=10) | 无返回注解；return: compute_alpha(5, open_wide=open_wide, close_wide=close_wide, vwap_wide=vwap_wide, vwap_sum_window=vwap_sum_window, ts_sum_vwap_divisor=ts_sum_vwap_divisor) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5.py#L47) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5.py#L67) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5.py#L76) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-145b37b8a923"></a>
## betalens-factor/alpha101/ALPHA5/factor_ALPHA5.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5.yaml) · 41 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5.yaml#L7)：`factor_spec:`
- [L25](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5.yaml#L25)：`weight:`
- [L31](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5.yaml#L31)：`run:`

<a id="file-152f3b5a5d72"></a>
## betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.py) · 87 行 · 说明来源：文件族规则

- **作用**：ALPHA5 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha5_timing](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.py#L29) | compute_alpha5_timing(open_wide, close_wide, vwap_wide, *, vwap_sum_window=10, ts_sum_vwap_divisor=10, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(5, open_wide=open_wide, close_wide=close_wide, vwap_wide=vwap_wide, vwap_sum_window=vwap_sum_window, ts_sum_vwap_divisor=ts_sum_vwap_divisor) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.py#L50) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.py#L70) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.py#L79) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-dc1f7c53f452"></a>
## betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.yaml) · 50 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.yaml#L8)：`factor_spec:`
- [L34](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.yaml#L34)：`weight:`
- [L40](../../../betalens-factor/alpha101/ALPHA5/factor_ALPHA5_timing.yaml#L40)：`run:`

<a id="file-efaa2d06d8bc"></a>
## betalens-factor/alpha101/ALPHA50/factor_ALPHA50.py

[打开源码](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50.py) · 82 行 · 说明来源：文件族规则

- **作用**：ALPHA50 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha50](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50.py#L29) | compute_alpha50(volume_wide, vwap_wide, *, rank_volume_rank_vwap_correlation_window=5, rank_correlation_volume_maximum_window=5) | 无返回注解；return: compute_alpha(50, volume_wide=volume_wide, vwap_wide=vwap_wide, rank_volume_rank_vwap_correlation_window=rank_volume_rank_vwap_correlation_window, rank_correlation_volume_maximum_window=rank_correlation_volume_maximum_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50.py#L45) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50.py#L65) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50.py#L74) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-1f922c6bc2aa"></a>
## betalens-factor/alpha101/ALPHA50/factor_ALPHA50.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50.yaml) · 40 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50.yaml#L7)：`factor_spec:`
- [L24](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50.yaml#L24)：`weight:`
- [L30](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50.yaml#L30)：`run:`

<a id="file-c58ca28a6516"></a>
## betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.py) · 85 行 · 说明来源：文件族规则

- **作用**：ALPHA50 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha50_timing](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.py#L29) | compute_alpha50_timing(volume_wide, vwap_wide, *, rank_volume_rank_vwap_correlation_window=5, rank_correlation_volume_maximum_window=5, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(50, volume_wide=volume_wide, vwap_wide=vwap_wide, rank_volume_rank_vwap_correlation_window=rank_volume_rank_vwap_correlation_window, rank_correlation_volume_maximum_window=rank_correlation_volume_maximum_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.py#L48) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.py#L68) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.py#L77) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ca2db2115407"></a>
## betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.yaml#L8)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA50/factor_ALPHA50_timing.yaml#L39)：`run:`

<a id="file-83cbf8948d60"></a>
## betalens-factor/alpha101/ALPHA51/factor_ALPHA51.py

[打开源码](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51.py) · 92 行 · 说明来源：文件族规则

- **作用**：ALPHA51 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha51](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51.py#L29) | compute_alpha51(close_wide, *, trend_long_delay_lag=20, trend_first_short_delay_lag=10, trend_first_slope_divisor=10, trend_second_short_delay_lag=10, trend_second_slope_divisor=10, alpha_trend_threshold=-0.05, alpha_trend_true_value=1.0, close_delay_lag=1) | 无返回注解；return: compute_alpha(51, close_wide=close_wide, trend_long_delay_lag=trend_long_delay_lag, trend_first_short_delay_lag=trend_first_short_delay_lag, trend_first_slope_divisor=trend_first_slope_divisor, trend_second_short_delay_lag=trend_second_short_delay_lag, trend_second_slope_divisor=trend_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51.py#L55) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51.py#L75) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51.py#L84) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-6dd112b0285a"></a>
## betalens-factor/alpha101/ALPHA51/factor_ALPHA51.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51.yaml) · 46 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51.yaml#L8)：`factor_spec:`
- [L30](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51.yaml#L30)：`weight:`
- [L36](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51.yaml#L36)：`run:`

<a id="file-dd2997d4d6ff"></a>
## betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.py) · 95 行 · 说明来源：文件族规则

- **作用**：ALPHA51 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha51_timing](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.py#L29) | compute_alpha51_timing(close_wide, *, trend_long_delay_lag=20, trend_first_short_delay_lag=10, trend_first_slope_divisor=10, trend_second_short_delay_lag=10, trend_second_slope_divisor=10, alpha_trend_threshold=-0.05, alpha_trend_true_value=1.0, close_delay_lag=1, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(51, close_wide=close_wide, trend_long_delay_lag=trend_long_delay_lag, trend_first_short_delay_lag=trend_first_short_delay_lag, trend_first_slope_divisor=trend_first_slope_divisor, trend_second_short_delay_lag=trend_second_short_delay_lag, trend_second_slope_divisor=trend_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.py#L58) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.py#L78) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.py#L87) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ba67a718fd28"></a>
## betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.yaml) · 54 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.yaml#L8)：`factor_spec:`
- [L38](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.yaml#L38)：`weight:`
- [L44](../../../betalens-factor/alpha101/ALPHA51/factor_ALPHA51_timing.yaml#L44)：`run:`

<a id="file-8a162cabc615"></a>
## betalens-factor/alpha101/ALPHA52/factor_ALPHA52.py

[打开源码](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52.py) · 94 行 · 说明来源：文件族规则

- **作用**：ALPHA52 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha52](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52.py#L29) | compute_alpha52(low_wide, volume_wide, returns_wide, *, low_minimum_window=5, low_minimum_window_2=5, ts_min_low_delay_lag=5, returns_sum_window=240, returns_sum_window_2=20, ts_sum_returns_divisor=220, volume_rank_window=5) | 无返回注解；return: compute_alpha(52, low_wide=low_wide, volume_wide=volume_wide, returns_wide=returns_wide, low_minimum_window=low_minimum_window, low_minimum_window_2=low_minimum_window_2, ts_min_low_delay_lag=ts_min_low_delay_lag, returns_sum_window=returns_sum_window, returns_sum_window_2=returns_sum_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52.py#L57) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52.py#L77) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52.py#L86) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-4e8596e08075"></a>
## betalens-factor/alpha101/ALPHA52/factor_ALPHA52.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52.yaml) · 46 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52.yaml#L7)：`factor_spec:`
- [L30](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52.yaml#L30)：`weight:`
- [L36](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52.yaml#L36)：`run:`

<a id="file-f29e9df04147"></a>
## betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.py) · 97 行 · 说明来源：文件族规则

- **作用**：ALPHA52 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha52_timing](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.py#L29) | compute_alpha52_timing(low_wide, volume_wide, returns_wide, *, low_minimum_window=5, low_minimum_window_2=5, ts_min_low_delay_lag=5, returns_sum_window=240, returns_sum_window_2=20, ts_sum_returns_divisor=220, volume_rank_window=5, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(52, low_wide=low_wide, volume_wide=volume_wide, returns_wide=returns_wide, low_minimum_window=low_minimum_window, low_minimum_window_2=low_minimum_window_2, ts_min_low_delay_lag=ts_min_low_delay_lag, returns_sum_window=returns_sum_window, returns_sum_window_2=returns_sum_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.py#L60) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.py#L80) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.py#L89) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-6db7448e3d13"></a>
## betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.yaml) · 55 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.yaml#L8)：`factor_spec:`
- [L39](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.yaml#L39)：`weight:`
- [L45](../../../betalens-factor/alpha101/ALPHA52/factor_ALPHA52_timing.yaml#L45)：`run:`

<a id="file-0564af53a3f4"></a>
## betalens-factor/alpha101/ALPHA53/factor_ALPHA53.py

[打开源码](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53.py) · 82 行 · 说明来源：文件族规则

- **作用**：ALPHA53 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha53](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53.py#L29) | compute_alpha53(close_wide, high_wide, low_wide, *, oscillator_delta_lag=9) | 无返回注解；return: compute_alpha(53, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, oscillator_delta_lag=oscillator_delta_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53.py#L45) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53.py#L65) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53.py#L74) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-df51092fd9d8"></a>
## betalens-factor/alpha101/ALPHA53/factor_ALPHA53.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53.yaml) · 40 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53.yaml#L7)：`factor_spec:`
- [L24](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53.yaml#L24)：`weight:`
- [L30](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53.yaml#L30)：`run:`

<a id="file-cb55b809eb99"></a>
## betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.py) · 85 行 · 说明来源：文件族规则

- **作用**：ALPHA53 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha53_timing](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.py#L29) | compute_alpha53_timing(close_wide, high_wide, low_wide, *, oscillator_delta_lag=9, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(53, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, oscillator_delta_lag=oscillator_delta_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.py#L48) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.py#L68) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.py#L77) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-446b51e3f2c3"></a>
## betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.yaml#L8)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA53/factor_ALPHA53_timing.yaml#L39)：`run:`

<a id="file-347f9e051702"></a>
## betalens-factor/alpha101/ALPHA54/factor_ALPHA54.py

[打开源码](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54.py) · 86 行 · 说明来源：文件族规则

- **作用**：ALPHA54 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha54](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54.py#L29) | compute_alpha54(open_wide, close_wide, high_wide, low_wide, *, open_power_exponent=5, close_power_exponent=5) | 无返回注解；return: compute_alpha(54, open_wide=open_wide, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, open_power_exponent=open_power_exponent, close_power_exponent=close_power_exponent) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54.py#L49) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54.py#L69) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54.py#L78) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-3419b51df918"></a>
## betalens-factor/alpha101/ALPHA54/factor_ALPHA54.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54.yaml) · 42 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54.yaml#L7)：`factor_spec:`
- [L26](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54.yaml#L26)：`weight:`
- [L32](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54.yaml#L32)：`run:`

<a id="file-a23674b955cb"></a>
## betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.py) · 89 行 · 说明来源：文件族规则

- **作用**：ALPHA54 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha54_timing](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.py#L29) | compute_alpha54_timing(open_wide, close_wide, high_wide, low_wide, *, open_power_exponent=5, close_power_exponent=5, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(54, open_wide=open_wide, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, open_power_exponent=open_power_exponent, close_power_exponent=close_power_exponent) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.py#L52) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.py#L72) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.py#L81) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-f5a2bd41accb"></a>
## betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.yaml) · 51 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.yaml#L8)：`factor_spec:`
- [L35](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.yaml#L35)：`weight:`
- [L41](../../../betalens-factor/alpha101/ALPHA54/factor_ALPHA54_timing.yaml#L41)：`run:`

<a id="file-0cc0680a7ee4"></a>
## betalens-factor/alpha101/ALPHA55/factor_ALPHA55.py

[打开源码](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55.py) · 90 行 · 说明来源：文件族规则

- **作用**：ALPHA55 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha55](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55.py#L29) | compute_alpha55(close_wide, high_wide, low_wide, volume_wide, *, low_minimum_window=12, high_maximum_window=12, low_minimum_window_2=12, rank_stochastic_rank_volume_correlation_window=6) | 无返回注解；return: compute_alpha(55, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, low_minimum_window=low_minimum_window, high_maximum_window=high_maximum_window, low_minimum_window_2=low_minimum_window_2, rank_stochastic_rank_volume_correlation_window=rank_stoch…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55.py#L53) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55.py#L73) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55.py#L82) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-f1448c8f756b"></a>
## betalens-factor/alpha101/ALPHA55/factor_ALPHA55.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55.yaml) · 44 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55.yaml#L7)：`factor_spec:`
- [L28](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55.yaml#L28)：`weight:`
- [L34](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55.yaml#L34)：`run:`

<a id="file-5fa0f5642f88"></a>
## betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.py) · 93 行 · 说明来源：文件族规则

- **作用**：ALPHA55 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha55_timing](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.py#L29) | compute_alpha55_timing(close_wide, high_wide, low_wide, volume_wide, *, low_minimum_window=12, high_maximum_window=12, low_minimum_window_2=12, rank_stochastic_rank_volume_correlation_window=6, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(55, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, low_minimum_window=low_minimum_window, high_maximum_window=high_maximum_window, low_minimum_window_2=low_minimum_window_2, rank_stochastic_rank_volume_correlation_window=rank_stoch…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.py#L56) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.py#L76) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.py#L85) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-b4199452b94b"></a>
## betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.yaml) · 53 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.yaml#L8)：`factor_spec:`
- [L37](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.yaml#L37)：`weight:`
- [L43](../../../betalens-factor/alpha101/ALPHA55/factor_ALPHA55_timing.yaml#L43)：`run:`

<a id="file-38e192392092"></a>
## betalens-factor/alpha101/ALPHA56/factor_ALPHA56.py

[打开源码](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56.py) · 84 行 · 说明来源：文件族规则

- **作用**：ALPHA56 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha56](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56.py#L29) | compute_alpha56(returns_wide, cap_wide, *, returns_sum_window=10, returns_sum_window_2=2, ts_sum_returns_sum_window=3) | 无返回注解；return: compute_alpha(56, returns_wide=returns_wide, cap_wide=cap_wide, returns_sum_window=returns_sum_window, returns_sum_window_2=returns_sum_window_2, ts_sum_returns_sum_window=ts_sum_returns_sum_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56.py#L47) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56.py#L67) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56.py#L76) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-6ca83f818843"></a>
## betalens-factor/alpha101/ALPHA56/factor_ALPHA56.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56.yaml) · 41 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56.yaml#L7)：`factor_spec:`
- [L25](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56.yaml#L25)：`weight:`
- [L31](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56.yaml#L31)：`run:`

<a id="file-c957ad0dfa9c"></a>
## betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.py) · 87 行 · 说明来源：文件族规则

- **作用**：ALPHA56 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha56_timing](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.py#L29) | compute_alpha56_timing(returns_wide, cap_wide, *, returns_sum_window=10, returns_sum_window_2=2, ts_sum_returns_sum_window=3, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(56, returns_wide=returns_wide, cap_wide=cap_wide, returns_sum_window=returns_sum_window, returns_sum_window_2=returns_sum_window_2, ts_sum_returns_sum_window=ts_sum_returns_sum_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.py#L50) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.py#L70) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.py#L79) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-2b20db131fb1"></a>
## betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.yaml) · 50 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.yaml#L8)：`factor_spec:`
- [L34](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.yaml#L34)：`weight:`
- [L40](../../../betalens-factor/alpha101/ALPHA56/factor_ALPHA56_timing.yaml#L40)：`run:`

<a id="file-15af0eed3936"></a>
## betalens-factor/alpha101/ALPHA57/factor_ALPHA57.py

[打开源码](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57.py) · 82 行 · 说明来源：文件族规则

- **作用**：ALPHA57 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha57](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57.py#L29) | compute_alpha57(close_wide, vwap_wide, *, close_argmax_window=30, rank_ts_argmax_close_decay_window=2) | 无返回注解；return: compute_alpha(57, close_wide=close_wide, vwap_wide=vwap_wide, close_argmax_window=close_argmax_window, rank_ts_argmax_close_decay_window=rank_ts_argmax_close_decay_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57.py#L45) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57.py#L65) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57.py#L74) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-37c859ddf13b"></a>
## betalens-factor/alpha101/ALPHA57/factor_ALPHA57.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57.yaml) · 40 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57.yaml#L7)：`factor_spec:`
- [L24](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57.yaml#L24)：`weight:`
- [L30](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57.yaml#L30)：`run:`

<a id="file-b5fce03da44c"></a>
## betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.py) · 85 行 · 说明来源：文件族规则

- **作用**：ALPHA57 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha57_timing](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.py#L29) | compute_alpha57_timing(close_wide, vwap_wide, *, close_argmax_window=30, rank_ts_argmax_close_decay_window=2, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(57, close_wide=close_wide, vwap_wide=vwap_wide, close_argmax_window=close_argmax_window, rank_ts_argmax_close_decay_window=rank_ts_argmax_close_decay_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.py#L48) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.py#L68) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.py#L77) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-c1426ccb8eae"></a>
## betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.yaml#L8)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA57/factor_ALPHA57_timing.yaml#L39)：`run:`

<a id="file-dec8e3a544ed"></a>
## betalens-factor/alpha101/ALPHA58/factor_ALPHA58.py

[打开源码](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58.py) · 86 行 · 说明来源：文件族规则

- **作用**：ALPHA58 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha58](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58.py#L29) | compute_alpha58(volume_wide, vwap_wide, sector_wide, *, indneutralize_vwap_sector_volume_correlation_window=3.92795, value_decay_window=7.89291, decay_linear_value_rank_window=5.50322) | 无返回注解；return: compute_alpha(58, volume_wide=volume_wide, vwap_wide=vwap_wide, sector_wide=sector_wide, indneutralize_vwap_sector_volume_correlation_window=indneutralize_vwap_sector_volume_correlation_window, value_decay_window=value_decay_window, decay_linear_value_rank_window=decay_linear_value_ran…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58.py#L49) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58.py#L69) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58.py#L78) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-3d2d26f9ec8b"></a>
## betalens-factor/alpha101/ALPHA58/factor_ALPHA58.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58.yaml) · 42 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58.yaml#L7)：`factor_spec:`
- [L26](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58.yaml#L26)：`weight:`
- [L32](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58.yaml#L32)：`run:`

<a id="file-52b639bab165"></a>
## betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.py) · 89 行 · 说明来源：文件族规则

- **作用**：ALPHA58 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha58_timing](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.py#L29) | compute_alpha58_timing(volume_wide, vwap_wide, sector_wide, *, indneutralize_vwap_sector_volume_correlation_window=3.92795, value_decay_window=7.89291, decay_linear_value_rank_window=5.50322, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(58, volume_wide=volume_wide, vwap_wide=vwap_wide, sector_wide=sector_wide, indneutralize_vwap_sector_volume_correlation_window=indneutralize_vwap_sector_volume_correlation_window, value_decay_window=value_decay_window, decay_linear_value_rank_window=decay_linear_value_ran…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.py#L52) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.py#L72) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.py#L81) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-3202f6db6df8"></a>
## betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.yaml) · 51 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.yaml#L8)：`factor_spec:`
- [L35](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.yaml#L35)：`weight:`
- [L41](../../../betalens-factor/alpha101/ALPHA58/factor_ALPHA58_timing.yaml#L41)：`run:`

<a id="file-e1869c852327"></a>
## betalens-factor/alpha101/ALPHA59/factor_ALPHA59.py

[打开源码](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59.py) · 92 行 · 说明来源：文件族规则

- **作用**：ALPHA59 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha59](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59.py#L29) | compute_alpha59(volume_wide, vwap_wide, industry_wide, *, vwap_mix_weight=0.728317, mixed_complement_base=1, mixed_complement_weight=0.728317, indneutralize_mixed_industry_volume_correlation_window=4.25197, value_decay_window=16.2289, decay_linear_value_rank_window=8.19648) | 无返回注解；return: compute_alpha(59, volume_wide=volume_wide, vwap_wide=vwap_wide, industry_wide=industry_wide, vwap_mix_weight=vwap_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, indneutralize_mixed_industry_volume_correlation_window=indneutrali…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59.py#L55) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59.py#L75) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59.py#L84) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-10053228be6a"></a>
## betalens-factor/alpha101/ALPHA59/factor_ALPHA59.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59.yaml) · 45 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59.yaml#L7)：`factor_spec:`
- [L29](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59.yaml#L29)：`weight:`
- [L35](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59.yaml#L35)：`run:`

<a id="file-0f0c2e781bba"></a>
## betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.py) · 95 行 · 说明来源：文件族规则

- **作用**：ALPHA59 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha59_timing](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.py#L29) | compute_alpha59_timing(volume_wide, vwap_wide, industry_wide, *, vwap_mix_weight=0.728317, mixed_complement_base=1, mixed_complement_weight=0.728317, indneutralize_mixed_industry_volume_correlation_window=4.25197, value_decay_window=16.2289, decay_linear_value_rank_window=8.19648, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(59, volume_wide=volume_wide, vwap_wide=vwap_wide, industry_wide=industry_wide, vwap_mix_weight=vwap_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, indneutralize_mixed_industry_volume_correlation_window=indneutrali…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.py#L58) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.py#L78) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.py#L87) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-a1d055966d5e"></a>
## betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.yaml) · 54 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.yaml#L8)：`factor_spec:`
- [L38](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.yaml#L38)：`weight:`
- [L44](../../../betalens-factor/alpha101/ALPHA59/factor_ALPHA59_timing.yaml#L44)：`run:`

<a id="file-a03db78b700b"></a>
## betalens-factor/alpha101/ALPHA6/factor_ALPHA6.py

[打开源码](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6.py) · 80 行 · 说明来源：文件族规则

- **作用**：ALPHA6 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha6](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6.py#L29) | compute_alpha6(open_wide, volume_wide, *, open_volume_correlation_window=10) | 无返回注解；return: compute_alpha(6, open_wide=open_wide, volume_wide=volume_wide, open_volume_correlation_window=open_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6.py#L43) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6.py#L63) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6.py#L72) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-9901b402a217"></a>
## betalens-factor/alpha101/ALPHA6/factor_ALPHA6.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6.yaml) · 39 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6.yaml#L7)：`factor_spec:`
- [L23](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6.yaml#L23)：`weight:`
- [L29](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6.yaml#L29)：`run:`

<a id="file-9a9dde09ac3e"></a>
## betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.py) · 83 行 · 说明来源：文件族规则

- **作用**：ALPHA6 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha6_timing](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.py#L29) | compute_alpha6_timing(open_wide, volume_wide, *, open_volume_correlation_window=10, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(6, open_wide=open_wide, volume_wide=volume_wide, open_volume_correlation_window=open_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.py#L46) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.py#L66) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.py#L75) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-4b5a5066c766"></a>
## betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.yaml#L8)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA6/factor_ALPHA6_timing.yaml#L38)：`run:`

<a id="file-deb558734a27"></a>
## betalens-factor/alpha101/ALPHA60/factor_ALPHA60.py

[打开源码](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60.py) · 86 行 · 说明来源：文件族规则

- **作用**：ALPHA60 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha60](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60.py#L29) | compute_alpha60(close_wide, high_wide, low_wide, volume_wide, *, scale_rank_oscillator_coefficient=2, close_argmax_window=10) | 无返回注解；return: compute_alpha(60, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, scale_rank_oscillator_coefficient=scale_rank_oscillator_coefficient, close_argmax_window=close_argmax_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60.py#L49) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60.py#L69) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60.py#L78) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-98a78563ad59"></a>
## betalens-factor/alpha101/ALPHA60/factor_ALPHA60.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60.yaml) · 42 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60.yaml#L7)：`factor_spec:`
- [L26](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60.yaml#L26)：`weight:`
- [L32](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60.yaml#L32)：`run:`

<a id="file-a9eab01881cf"></a>
## betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.py) · 89 行 · 说明来源：文件族规则

- **作用**：ALPHA60 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha60_timing](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.py#L29) | compute_alpha60_timing(close_wide, high_wide, low_wide, volume_wide, *, scale_rank_oscillator_coefficient=2, close_argmax_window=10, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(60, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, scale_rank_oscillator_coefficient=scale_rank_oscillator_coefficient, close_argmax_window=close_argmax_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.py#L52) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.py#L72) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.py#L81) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-8c1bbe9823b8"></a>
## betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.yaml) · 51 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.yaml#L8)：`factor_spec:`
- [L35](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.yaml#L35)：`weight:`
- [L41](../../../betalens-factor/alpha101/ALPHA60/factor_ALPHA60_timing.yaml#L41)：`run:`

<a id="file-673af130e35c"></a>
## betalens-factor/alpha101/ALPHA61/factor_ALPHA61.py

[打开源码](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61.py) · 84 行 · 说明来源：文件族规则

- **作用**：ALPHA61 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha61](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61.py#L29) | compute_alpha61(vwap_wide, amount_wide, *, vwap_minimum_window=16.1219, amount_average_window=180, vwap_adv_correlation_window=17.9282) | 无返回注解；return: compute_alpha(61, vwap_wide=vwap_wide, amount_wide=amount_wide, vwap_minimum_window=vwap_minimum_window, amount_average_window=amount_average_window, vwap_adv_correlation_window=vwap_adv_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61.py#L47) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61.py#L67) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61.py#L76) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-96fe8bba0513"></a>
## betalens-factor/alpha101/ALPHA61/factor_ALPHA61.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61.yaml) · 41 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61.yaml#L7)：`factor_spec:`
- [L25](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61.yaml#L25)：`weight:`
- [L31](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61.yaml#L31)：`run:`

<a id="file-a2acb2220edc"></a>
## betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.py) · 87 行 · 说明来源：文件族规则

- **作用**：ALPHA61 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha61_timing](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.py#L29) | compute_alpha61_timing(vwap_wide, amount_wide, *, vwap_minimum_window=16.1219, amount_average_window=180, vwap_adv_correlation_window=17.9282, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(61, vwap_wide=vwap_wide, amount_wide=amount_wide, vwap_minimum_window=vwap_minimum_window, amount_average_window=amount_average_window, vwap_adv_correlation_window=vwap_adv_correlation_window) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.py#L50) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.py#L70) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.py#L79) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-e5ef83fc95d6"></a>
## betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.yaml) · 50 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.yaml#L8)：`factor_spec:`
- [L34](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.yaml#L34)：`weight:`
- [L40](../../../betalens-factor/alpha101/ALPHA61/factor_ALPHA61_timing.yaml#L40)：`run:`

<a id="file-bac83c1fd4c5"></a>
## betalens-factor/alpha101/ALPHA62/factor_ALPHA62.py

[打开源码](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62.py) · 92 行 · 说明来源：文件族规则

- **作用**：ALPHA62 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha62](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62.py#L29) | compute_alpha62(open_wide, high_wide, low_wide, vwap_wide, amount_wide, *, amount_average_window=20, adv_sum_window=22.4101, vwap_ts_sum_adv_correlation_window=9.91009, high_low_divisor=2) | 无返回注解；return: compute_alpha(62, open_wide=open_wide, high_wide=high_wide, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window, vwap_ts_sum_adv_correlation_window=vwap_ts_sum_adv_correlation_window, high_low_divis…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62.py#L55) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62.py#L75) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62.py#L84) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-1b64f5e963d1"></a>
## betalens-factor/alpha101/ALPHA62/factor_ALPHA62.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62.yaml) · 45 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62.yaml#L7)：`factor_spec:`
- [L29](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62.yaml#L29)：`weight:`
- [L35](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62.yaml#L35)：`run:`

<a id="file-9f9b65e387d0"></a>
## betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.py) · 95 行 · 说明来源：文件族规则

- **作用**：ALPHA62 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha62_timing](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.py#L29) | compute_alpha62_timing(open_wide, high_wide, low_wide, vwap_wide, amount_wide, *, amount_average_window=20, adv_sum_window=22.4101, vwap_ts_sum_adv_correlation_window=9.91009, high_low_divisor=2, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(62, open_wide=open_wide, high_wide=high_wide, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window, vwap_ts_sum_adv_correlation_window=vwap_ts_sum_adv_correlation_window, high_low_divis…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.py#L58) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.py#L78) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.py#L87) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-383d256bbd9a"></a>
## betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.yaml) · 54 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.yaml#L8)：`factor_spec:`
- [L38](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.yaml#L38)：`weight:`
- [L44](../../../betalens-factor/alpha101/ALPHA62/factor_ALPHA62_timing.yaml#L44)：`run:`

<a id="file-b51b47b97063"></a>
## betalens-factor/alpha101/ALPHA63/factor_ALPHA63.py

[打开源码](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63.py) · 102 行 · 说明来源：文件族规则

- **作用**：ALPHA63 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha63](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63.py#L29) | compute_alpha63(open_wide, close_wide, vwap_wide, amount_wide, industry_wide, *, indneutralize_close_industry_delta_lag=2.25164, delta_indneutralize_close_decay_window=8.22237, vwap_mix_weight=0.318108, mixed_complement_base=1, mixed_complement_weight=0.318108, amount_average_window=180, adv_sum_window=37.2467, mixed_ts_sum_adv_correlation_window=13.557, correlation_mixed_ts_sum_decay_window=12.2883) | 无返回注解；return: compute_alpha(63, open_wide=open_wide, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, indneutralize_close_industry_delta_lag=indneutralize_close_industry_delta_lag, delta_indneutralize_close_decay_window=delta_indneutralize_close_decay…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63.py#L65) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63.py#L85) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63.py#L94) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d0f51e3084a9"></a>
## betalens-factor/alpha101/ALPHA63/factor_ALPHA63.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63.yaml) · 50 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63.yaml#L7)：`factor_spec:`
- [L34](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63.yaml#L34)：`weight:`
- [L40](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63.yaml#L40)：`run:`

<a id="file-2312379b5235"></a>
## betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.py) · 105 行 · 说明来源：文件族规则

- **作用**：ALPHA63 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha63_timing](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.py#L29) | compute_alpha63_timing(open_wide, close_wide, vwap_wide, amount_wide, industry_wide, *, indneutralize_close_industry_delta_lag=2.25164, delta_indneutralize_close_decay_window=8.22237, vwap_mix_weight=0.318108, mixed_complement_base=1, mixed_complement_weight=0.318108, amount_average_window=180, adv_sum_window=37.2467, mixed_ts_sum_adv_correlation_window=13.557, correlation_mixed_ts_sum_decay_window=12.2883, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(63, open_wide=open_wide, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, indneutralize_close_industry_delta_lag=indneutralize_close_industry_delta_lag, delta_indneutralize_close_decay_window=delta_indneutralize_close_decay…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.py#L68) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.py#L88) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.py#L97) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-307266b4c94a"></a>
## betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.yaml) · 59 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.yaml#L8)：`factor_spec:`
- [L43](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.yaml#L43)：`weight:`
- [L49](../../../betalens-factor/alpha101/ALPHA63/factor_ALPHA63_timing.yaml#L49)：`run:`

<a id="file-b42ae85f857e"></a>
## betalens-factor/alpha101/ALPHA64/factor_ALPHA64.py

[打开源码](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64.py) · 108 行 · 说明来源：文件族规则

- **作用**：ALPHA64 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha64](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64.py#L29) | compute_alpha64(open_wide, high_wide, low_wide, vwap_wide, amount_wide, *, open_mix_weight=0.178404, mixed1_complement_base=1, mixed1_complement_weight=0.178404, mixed1_sum_window=12.7054, amount_average_window=120, adv_sum_window=12.7054, ts_sum_mixed1_ts_sum_adv_correlation_window=16.6208, high_low_divisor=2, high_low_mix_weight=0.178404, mixed2_complement_base=1, mixed2_complement_weight=0.178404, mixed2_delta_lag=3.69741) | 无返回注解；return: compute_alpha(64, open_wide=open_wide, high_wide=high_wide, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, open_mix_weight=open_mix_weight, mixed1_complement_base=mixed1_complement_base, mixed1_complement_weight=mixed1_complement_weight, mixed1_sum_window=mixed1_sum_w…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64.py#L71) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64.py#L91) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64.py#L100) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-76a44bfdd80b"></a>
## betalens-factor/alpha101/ALPHA64/factor_ALPHA64.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64.yaml) · 53 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64.yaml#L7)：`factor_spec:`
- [L37](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64.yaml#L37)：`weight:`
- [L43](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64.yaml#L43)：`run:`

<a id="file-c867b0355bf5"></a>
## betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.py) · 111 行 · 说明来源：文件族规则

- **作用**：ALPHA64 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha64_timing](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.py#L29) | compute_alpha64_timing(open_wide, high_wide, low_wide, vwap_wide, amount_wide, *, open_mix_weight=0.178404, mixed1_complement_base=1, mixed1_complement_weight=0.178404, mixed1_sum_window=12.7054, amount_average_window=120, adv_sum_window=12.7054, ts_sum_mixed1_ts_sum_adv_correlation_window=16.6208, high_low_divisor=2, high_low_mix_weight=0.178404, mixed2_complement_base=1, mixed2_complement_weight=0.178404, mixed2_delta_lag=3.69741, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(64, open_wide=open_wide, high_wide=high_wide, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, open_mix_weight=open_mix_weight, mixed1_complement_base=mixed1_complement_base, mixed1_complement_weight=mixed1_complement_weight, mixed1_sum_window=mixed1_sum_w…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.py#L74) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.py#L94) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.py#L103) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-44c0c183dfd9"></a>
## betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.yaml) · 62 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.yaml#L8)：`factor_spec:`
- [L46](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.yaml#L46)：`weight:`
- [L52](../../../betalens-factor/alpha101/ALPHA64/factor_ALPHA64_timing.yaml#L52)：`run:`

<a id="file-9351149a4599"></a>
## betalens-factor/alpha101/ALPHA65/factor_ALPHA65.py

[打开源码](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65.py) · 94 行 · 说明来源：文件族规则

- **作用**：ALPHA65 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha65](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65.py#L29) | compute_alpha65(open_wide, vwap_wide, amount_wide, *, open_mix_weight=0.00817205, mixed_complement_base=1, mixed_complement_weight=0.00817205, amount_average_window=60, adv_sum_window=8.6911, mixed_ts_sum_adv_correlation_window=6.40374, open_minimum_window=13.635) | 无返回注解；return: compute_alpha(65, open_wide=open_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, open_mix_weight=open_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65.py#L57) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65.py#L77) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65.py#L86) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-7cac2776969b"></a>
## betalens-factor/alpha101/ALPHA65/factor_ALPHA65.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65.yaml) · 46 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65.yaml#L7)：`factor_spec:`
- [L30](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65.yaml#L30)：`weight:`
- [L36](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65.yaml#L36)：`run:`

<a id="file-29ddbe6fa5f8"></a>
## betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.py) · 97 行 · 说明来源：文件族规则

- **作用**：ALPHA65 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha65_timing](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.py#L29) | compute_alpha65_timing(open_wide, vwap_wide, amount_wide, *, open_mix_weight=0.00817205, mixed_complement_base=1, mixed_complement_weight=0.00817205, amount_average_window=60, adv_sum_window=8.6911, mixed_ts_sum_adv_correlation_window=6.40374, open_minimum_window=13.635, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(65, open_wide=open_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, open_mix_weight=open_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.py#L60) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.py#L80) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.py#L89) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-77d674dc1eef"></a>
## betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.yaml) · 55 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.yaml#L8)：`factor_spec:`
- [L39](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.yaml#L39)：`weight:`
- [L45](../../../betalens-factor/alpha101/ALPHA65/factor_ALPHA65_timing.yaml#L45)：`run:`

<a id="file-315b9c4af69a"></a>
## betalens-factor/alpha101/ALPHA66/factor_ALPHA66.py

[打开源码](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66.py) · 92 行 · 说明来源：文件族规则

- **作用**：ALPHA66 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha66](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66.py#L29) | compute_alpha66(open_wide, high_wide, low_wide, vwap_wide, *, vwap_delta_lag=3.51013, delta_vwap_decay_window=7.23052, high_low_divisor=2, right_base_decay_window=11.4157, decay_linear_right_base_rank_window=6.72611) | 无返回注解；return: compute_alpha(66, open_wide=open_wide, high_wide=high_wide, low_wide=low_wide, vwap_wide=vwap_wide, vwap_delta_lag=vwap_delta_lag, delta_vwap_decay_window=delta_vwap_decay_window, high_low_divisor=high_low_divisor, right_base_decay_window=right_base_decay_window, decay_linear_right_bas…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66.py#L55) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66.py#L75) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66.py#L84) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-2bfdc122c312"></a>
## betalens-factor/alpha101/ALPHA66/factor_ALPHA66.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66.yaml) · 45 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66.yaml#L7)：`factor_spec:`
- [L29](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66.yaml#L29)：`weight:`
- [L35](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66.yaml#L35)：`run:`

<a id="file-526203dbded7"></a>
## betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.py) · 95 行 · 说明来源：文件族规则

- **作用**：ALPHA66 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha66_timing](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.py#L29) | compute_alpha66_timing(open_wide, high_wide, low_wide, vwap_wide, *, vwap_delta_lag=3.51013, delta_vwap_decay_window=7.23052, high_low_divisor=2, right_base_decay_window=11.4157, decay_linear_right_base_rank_window=6.72611, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(66, open_wide=open_wide, high_wide=high_wide, low_wide=low_wide, vwap_wide=vwap_wide, vwap_delta_lag=vwap_delta_lag, delta_vwap_decay_window=delta_vwap_decay_window, high_low_divisor=high_low_divisor, right_base_decay_window=right_base_decay_window, decay_linear_right_bas…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.py#L58) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.py#L78) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.py#L87) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-26cadafeb738"></a>
## betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.yaml) · 54 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.yaml#L8)：`factor_spec:`
- [L38](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.yaml#L38)：`weight:`
- [L44](../../../betalens-factor/alpha101/ALPHA66/factor_ALPHA66_timing.yaml#L44)：`run:`

<a id="file-6b649f8cdccb"></a>
## betalens-factor/alpha101/ALPHA67/factor_ALPHA67.py

[打开源码](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67.py) · 90 行 · 说明来源：文件族规则

- **作用**：ALPHA67 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha67](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67.py#L29) | compute_alpha67(high_wide, vwap_wide, amount_wide, sector_wide, subindustry_wide, *, high_minimum_window=2.14593, amount_average_window=20, indneutralize_vwap_sector_indneutralize_subindustry_adv_correlation_wind=6.02936) | 无返回注解；return: compute_alpha(67, high_wide=high_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, sector_wide=sector_wide, subindustry_wide=subindustry_wide, high_minimum_window=high_minimum_window, amount_average_window=amount_average_window, indneutralize_vwap_sector_indneutralize_subindustry_adv…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67.py#L53) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67.py#L73) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67.py#L82) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-8c935eaace15"></a>
## betalens-factor/alpha101/ALPHA67/factor_ALPHA67.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67.yaml) · 44 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67.yaml#L7)：`factor_spec:`
- [L28](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67.yaml#L28)：`weight:`
- [L34](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67.yaml#L34)：`run:`

<a id="file-a2324336596d"></a>
## betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.py) · 93 行 · 说明来源：文件族规则

- **作用**：ALPHA67 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha67_timing](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.py#L29) | compute_alpha67_timing(high_wide, vwap_wide, amount_wide, sector_wide, subindustry_wide, *, high_minimum_window=2.14593, amount_average_window=20, indneutralize_vwap_sector_indneutralize_subindustry_adv_correlation_wind=6.02936, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(67, high_wide=high_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, sector_wide=sector_wide, subindustry_wide=subindustry_wide, high_minimum_window=high_minimum_window, amount_average_window=amount_average_window, indneutralize_vwap_sector_indneutralize_subindustry_adv…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.py#L56) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.py#L76) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.py#L85) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-e3a88bc835b3"></a>
## betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.yaml) · 53 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.yaml#L8)：`factor_spec:`
- [L37](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.yaml#L37)：`weight:`
- [L43](../../../betalens-factor/alpha101/ALPHA67/factor_ALPHA67_timing.yaml#L43)：`run:`

<a id="file-841ac1e4a455"></a>
## betalens-factor/alpha101/ALPHA68/factor_ALPHA68.py

[打开源码](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68.py) · 96 行 · 说明来源：文件族规则

- **作用**：ALPHA68 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha68](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68.py#L29) | compute_alpha68(close_wide, high_wide, low_wide, amount_wide, *, amount_average_window=15, rank_high_rank_adv_correlation_window=8.91644, correlation_rank_high_rank_window=13.9333, close_mix_weight=0.518371, mixed_complement_base=1, mixed_complement_weight=0.518371, mixed_delta_lag=1.06157) | 无返回注解；return: compute_alpha(68, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, rank_high_rank_adv_correlation_window=rank_high_rank_adv_correlation_window, correlation_rank_high_rank_window=correlation_rank_high_ra…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68.py#L59) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68.py#L79) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68.py#L88) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-71f3e9182843"></a>
## betalens-factor/alpha101/ALPHA68/factor_ALPHA68.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68.yaml) · 47 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68.yaml#L7)：`factor_spec:`
- [L31](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68.yaml#L31)：`weight:`
- [L37](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68.yaml#L37)：`run:`

<a id="file-151bfbfb0d7e"></a>
## betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.py) · 99 行 · 说明来源：文件族规则

- **作用**：ALPHA68 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha68_timing](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.py#L29) | compute_alpha68_timing(close_wide, high_wide, low_wide, amount_wide, *, amount_average_window=15, rank_high_rank_adv_correlation_window=8.91644, correlation_rank_high_rank_window=13.9333, close_mix_weight=0.518371, mixed_complement_base=1, mixed_complement_weight=0.518371, mixed_delta_lag=1.06157, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(68, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, rank_high_rank_adv_correlation_window=rank_high_rank_adv_correlation_window, correlation_rank_high_rank_window=correlation_rank_high_ra…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.py#L62) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.py#L82) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.py#L91) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-b20ce11371fd"></a>
## betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.yaml) · 56 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.yaml#L8)：`factor_spec:`
- [L40](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.yaml#L40)：`weight:`
- [L46](../../../betalens-factor/alpha101/ALPHA68/factor_ALPHA68_timing.yaml#L46)：`run:`

<a id="file-9728e8e4de12"></a>
## betalens-factor/alpha101/ALPHA69/factor_ALPHA69.py

[打开源码](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69.py) · 98 行 · 说明来源：文件族规则

- **作用**：ALPHA69 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha69](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69.py#L29) | compute_alpha69(close_wide, vwap_wide, amount_wide, industry_wide, *, indneutralize_vwap_industry_delta_lag=2.72412, delta_indneutralize_vwap_maximum_window=4.79344, close_mix_weight=0.490655, mixed_complement_base=1, mixed_complement_weight=0.490655, amount_average_window=20, mixed_adv_correlation_window=4.92416, correlation_mixed_adv_rank_window=9.0615) | 无返回注解；return: compute_alpha(69, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, indneutralize_vwap_industry_delta_lag=indneutralize_vwap_industry_delta_lag, delta_indneutralize_vwap_maximum_window=delta_indneutralize_vwap_maximum_window, close_mix_we…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69.py#L61) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69.py#L81) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69.py#L90) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-accb560cb1a2"></a>
## betalens-factor/alpha101/ALPHA69/factor_ALPHA69.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69.yaml#L7)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69.yaml#L38)：`run:`

<a id="file-ab7a1cc23e25"></a>
## betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.py) · 101 行 · 说明来源：文件族规则

- **作用**：ALPHA69 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha69_timing](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.py#L29) | compute_alpha69_timing(close_wide, vwap_wide, amount_wide, industry_wide, *, indneutralize_vwap_industry_delta_lag=2.72412, delta_indneutralize_vwap_maximum_window=4.79344, close_mix_weight=0.490655, mixed_complement_base=1, mixed_complement_weight=0.490655, amount_average_window=20, mixed_adv_correlation_window=4.92416, correlation_mixed_adv_rank_window=9.0615, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(69, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, indneutralize_vwap_industry_delta_lag=indneutralize_vwap_industry_delta_lag, delta_indneutralize_vwap_maximum_window=delta_indneutralize_vwap_maximum_window, close_mix_we…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.py#L64) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.py#L84) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.py#L93) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-cb1b72e6d473"></a>
## betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.yaml) · 57 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.yaml#L8)：`factor_spec:`
- [L41](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.yaml#L41)：`weight:`
- [L47](../../../betalens-factor/alpha101/ALPHA69/factor_ALPHA69_timing.yaml#L47)：`run:`

<a id="file-3e958223e09a"></a>
## betalens-factor/alpha101/ALPHA7/factor_ALPHA7.py

[打开源码](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7.py) · 88 行 · 说明来源：文件族规则

- **作用**：ALPHA7 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha7](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7.py#L29) | compute_alpha7(close_wide, volume_wide, amount_wide, *, close_delta_lag=7, change_rank_window=60, amount_average_window=20, volume_adv_false_value=-1.0) | 无返回注解；return: compute_alpha(7, close_wide=close_wide, volume_wide=volume_wide, amount_wide=amount_wide, close_delta_lag=close_delta_lag, change_rank_window=change_rank_window, amount_average_window=amount_average_window, volume_adv_false_value=volume_adv_false_value) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7.py#L51) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7.py#L71) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7.py#L80) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ae385ee28331"></a>
## betalens-factor/alpha101/ALPHA7/factor_ALPHA7.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7.yaml) · 44 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7.yaml#L8)：`factor_spec:`
- [L28](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7.yaml#L28)：`weight:`
- [L34](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7.yaml#L34)：`run:`

<a id="file-5167e1851771"></a>
## betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.py) · 91 行 · 说明来源：文件族规则

- **作用**：ALPHA7 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha7_timing](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.py#L29) | compute_alpha7_timing(close_wide, volume_wide, amount_wide, *, close_delta_lag=7, change_rank_window=60, amount_average_window=20, volume_adv_false_value=-1.0, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(7, close_wide=close_wide, volume_wide=volume_wide, amount_wide=amount_wide, close_delta_lag=close_delta_lag, change_rank_window=change_rank_window, amount_average_window=amount_average_window, volume_adv_false_value=volume_adv_false_value) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.py#L54) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.py#L74) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.py#L83) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-062d276e63bb"></a>
## betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.yaml) · 52 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.yaml#L8)：`factor_spec:`
- [L36](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.yaml#L36)：`weight:`
- [L42](../../../betalens-factor/alpha101/ALPHA7/factor_ALPHA7_timing.yaml#L42)：`run:`

<a id="file-14c9e98fe932"></a>
## betalens-factor/alpha101/ALPHA70/factor_ALPHA70.py

[打开源码](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70.py) · 90 行 · 说明来源：文件族规则

- **作用**：ALPHA70 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha70](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70.py#L29) | compute_alpha70(close_wide, vwap_wide, amount_wide, industry_wide, *, vwap_delta_lag=1.29456, amount_average_window=50, indneutralize_close_industry_adv_correlation_window=17.8256, correlation_indneutralize_close_rank_window=17.9171) | 无返回注解；return: compute_alpha(70, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, vwap_delta_lag=vwap_delta_lag, amount_average_window=amount_average_window, indneutralize_close_industry_adv_correlation_window=indneutralize_close_industry_adv_correlati…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70.py#L53) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70.py#L73) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70.py#L82) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-9b8f9ff81356"></a>
## betalens-factor/alpha101/ALPHA70/factor_ALPHA70.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70.yaml) · 44 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70.yaml#L7)：`factor_spec:`
- [L28](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70.yaml#L28)：`weight:`
- [L34](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70.yaml#L34)：`run:`

<a id="file-ecd699dee503"></a>
## betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.py) · 93 行 · 说明来源：文件族规则

- **作用**：ALPHA70 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha70_timing](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.py#L29) | compute_alpha70_timing(close_wide, vwap_wide, amount_wide, industry_wide, *, vwap_delta_lag=1.29456, amount_average_window=50, indneutralize_close_industry_adv_correlation_window=17.8256, correlation_indneutralize_close_rank_window=17.9171, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(70, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, vwap_delta_lag=vwap_delta_lag, amount_average_window=amount_average_window, indneutralize_close_industry_adv_correlation_window=indneutralize_close_industry_adv_correlati…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.py#L56) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.py#L76) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.py#L85) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-e85c97ecdb7e"></a>
## betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.yaml) · 53 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.yaml#L8)：`factor_spec:`
- [L37](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.yaml#L37)：`weight:`
- [L43](../../../betalens-factor/alpha101/ALPHA70/factor_ALPHA70_timing.yaml#L43)：`run:`

<a id="file-d1d5c4a24783"></a>
## betalens-factor/alpha101/ALPHA71/factor_ALPHA71.py

[打开源码](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71.py) · 102 行 · 说明来源：文件族规则

- **作用**：ALPHA71 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha71](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71.py#L29) | compute_alpha71(open_wide, close_wide, low_wide, vwap_wide, amount_wide, *, close_rank_window=3.43976, amount_average_window=180, adv_rank_window=12.0647, ts_rank_close_ts_rank_adv_correlation_window=18.0175, correlation_ts_rank_close_decay_window=4.20501, decay_linear_correlation_ts_rank_rank_window=15.6948, rank_vwap_low_power_exponent=2, rank_vwap_low_decay_window=16.4662, decay_linear_rank_vwap_rank_window=4.4388) | 无返回注解；return: compute_alpha(71, open_wide=open_wide, close_wide=close_wide, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, close_rank_window=close_rank_window, amount_average_window=amount_average_window, adv_rank_window=adv_rank_window, ts_rank_close_ts_rank_adv_correlation_window…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71.py#L65) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71.py#L85) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71.py#L94) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-69dc8b979812"></a>
## betalens-factor/alpha101/ALPHA71/factor_ALPHA71.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71.yaml) · 50 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71.yaml#L7)：`factor_spec:`
- [L34](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71.yaml#L34)：`weight:`
- [L40](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71.yaml#L40)：`run:`

<a id="file-6b343d0ff3f1"></a>
## betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.py) · 105 行 · 说明来源：文件族规则

- **作用**：ALPHA71 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha71_timing](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.py#L29) | compute_alpha71_timing(open_wide, close_wide, low_wide, vwap_wide, amount_wide, *, close_rank_window=3.43976, amount_average_window=180, adv_rank_window=12.0647, ts_rank_close_ts_rank_adv_correlation_window=18.0175, correlation_ts_rank_close_decay_window=4.20501, decay_linear_correlation_ts_rank_rank_window=15.6948, rank_vwap_low_power_exponent=2, rank_vwap_low_decay_window=16.4662, decay_linear_rank_vwap_rank_window=4.4388, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(71, open_wide=open_wide, close_wide=close_wide, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, close_rank_window=close_rank_window, amount_average_window=amount_average_window, adv_rank_window=adv_rank_window, ts_rank_close_ts_rank_adv_correlation_window…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.py#L68) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.py#L88) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.py#L97) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-1161e66d4ca1"></a>
## betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.yaml) · 59 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.yaml#L8)：`factor_spec:`
- [L43](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.yaml#L43)：`weight:`
- [L49](../../../betalens-factor/alpha101/ALPHA71/factor_ALPHA71_timing.yaml#L49)：`run:`

<a id="file-48ffdc7ca61a"></a>
## betalens-factor/alpha101/ALPHA72/factor_ALPHA72.py

[打开源码](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72.py) · 100 行 · 说明来源：文件族规则

- **作用**：ALPHA72 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha72](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72.py#L29) | compute_alpha72(high_wide, low_wide, volume_wide, vwap_wide, amount_wide, *, high_low_divisor=2, amount_average_window=40, high_low_adv_correlation_window=8.93345, correlation_adv_high_decay_window=10.1519, vwap_rank_window=3.72469, volume_rank_window=18.5188, ts_rank_vwap_ts_rank_volume_correlation_window=6.86671, correlation_ts_rank_vwap_decay_window=2.95011) | 无返回注解；return: compute_alpha(72, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, high_low_divisor=high_low_divisor, amount_average_window=amount_average_window, high_low_adv_correlation_window=high_low_adv_correlation_window, correlation_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72.py#L63) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72.py#L83) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72.py#L92) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-622469e61e82"></a>
## betalens-factor/alpha101/ALPHA72/factor_ALPHA72.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72.yaml#L7)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72.yaml#L39)：`run:`

<a id="file-1c10b4dcb35a"></a>
## betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.py) · 103 行 · 说明来源：文件族规则

- **作用**：ALPHA72 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha72_timing](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.py#L29) | compute_alpha72_timing(high_wide, low_wide, volume_wide, vwap_wide, amount_wide, *, high_low_divisor=2, amount_average_window=40, high_low_adv_correlation_window=8.93345, correlation_adv_high_decay_window=10.1519, vwap_rank_window=3.72469, volume_rank_window=18.5188, ts_rank_vwap_ts_rank_volume_correlation_window=6.86671, correlation_ts_rank_vwap_decay_window=2.95011, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(72, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, high_low_divisor=high_low_divisor, amount_average_window=amount_average_window, high_low_adv_correlation_window=high_low_adv_correlation_window, correlation_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.py#L66) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.py#L86) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.py#L95) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d5b02e5ac516"></a>
## betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.yaml) · 58 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.yaml#L8)：`factor_spec:`
- [L42](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.yaml#L42)：`weight:`
- [L48](../../../betalens-factor/alpha101/ALPHA72/factor_ALPHA72_timing.yaml#L48)：`run:`

<a id="file-a21afbfa9dd6"></a>
## betalens-factor/alpha101/ALPHA73/factor_ALPHA73.py

[打开源码](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73.py) · 96 行 · 说明来源：文件族规则

- **作用**：ALPHA73 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha73](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73.py#L29) | compute_alpha73(open_wide, low_wide, vwap_wide, *, vwap_delta_lag=4.72775, delta_vwap_decay_window=2.91864, open_mix_weight=0.147155, mixed_complement_base=1, mixed_complement_weight=0.147155, mixed_delta_lag=2.03608, mixed_delta_decay_window=3.33829, decay_linear_mixed_delta_rank_window=16.7411) | 无返回注解；return: compute_alpha(73, open_wide=open_wide, low_wide=low_wide, vwap_wide=vwap_wide, vwap_delta_lag=vwap_delta_lag, delta_vwap_decay_window=delta_vwap_decay_window, open_mix_weight=open_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, …（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73.py#L59) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73.py#L79) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73.py#L88) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-effc7266c6aa"></a>
## betalens-factor/alpha101/ALPHA73/factor_ALPHA73.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73.yaml) · 47 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73.yaml#L7)：`factor_spec:`
- [L31](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73.yaml#L31)：`weight:`
- [L37](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73.yaml#L37)：`run:`

<a id="file-a49201d6bf05"></a>
## betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.py) · 99 行 · 说明来源：文件族规则

- **作用**：ALPHA73 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha73_timing](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.py#L29) | compute_alpha73_timing(open_wide, low_wide, vwap_wide, *, vwap_delta_lag=4.72775, delta_vwap_decay_window=2.91864, open_mix_weight=0.147155, mixed_complement_base=1, mixed_complement_weight=0.147155, mixed_delta_lag=2.03608, mixed_delta_decay_window=3.33829, decay_linear_mixed_delta_rank_window=16.7411, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(73, open_wide=open_wide, low_wide=low_wide, vwap_wide=vwap_wide, vwap_delta_lag=vwap_delta_lag, delta_vwap_decay_window=delta_vwap_decay_window, open_mix_weight=open_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, …（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.py#L62) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.py#L82) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.py#L91) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d78ec1043ac0"></a>
## betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.yaml) · 56 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.yaml#L8)：`factor_spec:`
- [L40](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.yaml#L40)：`weight:`
- [L46](../../../betalens-factor/alpha101/ALPHA73/factor_ALPHA73_timing.yaml#L46)：`run:`

<a id="file-386b42ec5d57"></a>
## betalens-factor/alpha101/ALPHA74/factor_ALPHA74.py

[打开源码](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74.py) · 98 行 · 说明来源：文件族规则

- **作用**：ALPHA74 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha74](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74.py#L29) | compute_alpha74(close_wide, high_wide, volume_wide, vwap_wide, amount_wide, *, amount_average_window=30, adv_sum_window=37.4843, close_ts_sum_adv_correlation_window=15.1365, high_mix_weight=0.0261661, mixed_complement_base=1, mixed_complement_weight=0.0261661, rank_mixed_rank_volume_correlation_window=11.4791) | 无返回注解；return: compute_alpha(74, close_wide=close_wide, high_wide=high_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window, close_ts_sum_adv_correlation_window=close_ts_sum_adv_correlation_window, high…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74.py#L61) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74.py#L81) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74.py#L90) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-7e6643bee937"></a>
## betalens-factor/alpha101/ALPHA74/factor_ALPHA74.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74.yaml#L7)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74.yaml#L38)：`run:`

<a id="file-627fc533a1da"></a>
## betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.py) · 101 行 · 说明来源：文件族规则

- **作用**：ALPHA74 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha74_timing](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.py#L29) | compute_alpha74_timing(close_wide, high_wide, volume_wide, vwap_wide, amount_wide, *, amount_average_window=30, adv_sum_window=37.4843, close_ts_sum_adv_correlation_window=15.1365, high_mix_weight=0.0261661, mixed_complement_base=1, mixed_complement_weight=0.0261661, rank_mixed_rank_volume_correlation_window=11.4791, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(74, close_wide=close_wide, high_wide=high_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window, close_ts_sum_adv_correlation_window=close_ts_sum_adv_correlation_window, high…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.py#L64) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.py#L84) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.py#L93) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-774c8ea64a9e"></a>
## betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.yaml) · 57 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.yaml#L8)：`factor_spec:`
- [L41](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.yaml#L41)：`weight:`
- [L47](../../../betalens-factor/alpha101/ALPHA74/factor_ALPHA74_timing.yaml#L47)：`run:`

<a id="file-db04a53f3cad"></a>
## betalens-factor/alpha101/ALPHA75/factor_ALPHA75.py

[打开源码](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75.py) · 88 行 · 说明来源：文件族规则

- **作用**：ALPHA75 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha75](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75.py#L29) | compute_alpha75(low_wide, volume_wide, vwap_wide, amount_wide, *, vwap_volume_correlation_window=4.24304, amount_average_window=50, rank_low_rank_adv_correlation_window=12.4413) | 无返回注解；return: compute_alpha(75, low_wide=low_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, vwap_volume_correlation_window=vwap_volume_correlation_window, amount_average_window=amount_average_window, rank_low_rank_adv_correlation_window=rank_low_rank_adv_correlation_win…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75.py#L51) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75.py#L71) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75.py#L80) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-591560b54b13"></a>
## betalens-factor/alpha101/ALPHA75/factor_ALPHA75.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75.yaml) · 43 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75.yaml#L7)：`factor_spec:`
- [L27](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75.yaml#L27)：`weight:`
- [L33](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75.yaml#L33)：`run:`

<a id="file-5ae2a0a786cf"></a>
## betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.py) · 91 行 · 说明来源：文件族规则

- **作用**：ALPHA75 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha75_timing](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.py#L29) | compute_alpha75_timing(low_wide, volume_wide, vwap_wide, amount_wide, *, vwap_volume_correlation_window=4.24304, amount_average_window=50, rank_low_rank_adv_correlation_window=12.4413, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(75, low_wide=low_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, vwap_volume_correlation_window=vwap_volume_correlation_window, amount_average_window=amount_average_window, rank_low_rank_adv_correlation_window=rank_low_rank_adv_correlation_win…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.py#L54) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.py#L74) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.py#L83) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-3c7573ecfd01"></a>
## betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.yaml) · 52 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.yaml#L8)：`factor_spec:`
- [L36](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.yaml#L36)：`weight:`
- [L42](../../../betalens-factor/alpha101/ALPHA75/factor_ALPHA75_timing.yaml#L42)：`run:`

<a id="file-273eccb0d1e4"></a>
## betalens-factor/alpha101/ALPHA76/factor_ALPHA76.py

[打开源码](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76.py) · 96 行 · 说明来源：文件族规则

- **作用**：ALPHA76 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha76](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76.py#L29) | compute_alpha76(low_wide, vwap_wide, amount_wide, sector_wide, *, vwap_delta_lag=1.24383, delta_vwap_decay_window=11.8259, amount_average_window=81, indneutralize_low_sector_adv_correlation_window=8.14941, corr_rank_window=19.569, ts_rank_corr_decay_window=17.1543, decay_linear_ts_rank_corr_rank_window=19.383) | 无返回注解；return: compute_alpha(76, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, sector_wide=sector_wide, vwap_delta_lag=vwap_delta_lag, delta_vwap_decay_window=delta_vwap_decay_window, amount_average_window=amount_average_window, indneutralize_low_sector_adv_correlation_window=indne…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76.py#L59) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76.py#L79) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76.py#L88) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-0117f0fda6b3"></a>
## betalens-factor/alpha101/ALPHA76/factor_ALPHA76.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76.yaml) · 47 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76.yaml#L7)：`factor_spec:`
- [L31](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76.yaml#L31)：`weight:`
- [L37](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76.yaml#L37)：`run:`

<a id="file-1b146506922b"></a>
## betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.py) · 99 行 · 说明来源：文件族规则

- **作用**：ALPHA76 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha76_timing](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.py#L29) | compute_alpha76_timing(low_wide, vwap_wide, amount_wide, sector_wide, *, vwap_delta_lag=1.24383, delta_vwap_decay_window=11.8259, amount_average_window=81, indneutralize_low_sector_adv_correlation_window=8.14941, corr_rank_window=19.569, ts_rank_corr_decay_window=17.1543, decay_linear_ts_rank_corr_rank_window=19.383, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(76, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, sector_wide=sector_wide, vwap_delta_lag=vwap_delta_lag, delta_vwap_decay_window=delta_vwap_decay_window, amount_average_window=amount_average_window, indneutralize_low_sector_adv_correlation_window=indne…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.py#L62) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.py#L82) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.py#L91) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-6fd874ca0ea4"></a>
## betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.yaml) · 56 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.yaml#L8)：`factor_spec:`
- [L40](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.yaml#L40)：`weight:`
- [L46](../../../betalens-factor/alpha101/ALPHA76/factor_ALPHA76_timing.yaml#L46)：`run:`

<a id="file-b0f91d1709b4"></a>
## betalens-factor/alpha101/ALPHA77/factor_ALPHA77.py

[打开源码](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77.py) · 94 行 · 说明来源：文件族规则

- **作用**：ALPHA77 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha77](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77.py#L29) | compute_alpha77(high_wide, low_wide, vwap_wide, amount_wide, *, high_low_divisor=2, vwap_high_low_decay_window=20.0451, high_low_divisor_2=2, amount_average_window=40, high_low_adv_correlation_window=3.1614, correlation_adv_high_decay_window=5.64125) | 无返回注解；return: compute_alpha(77, high_wide=high_wide, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, high_low_divisor=high_low_divisor, vwap_high_low_decay_window=vwap_high_low_decay_window, high_low_divisor_2=high_low_divisor_2, amount_average_window=amount_average_window, high_low…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77.py#L57) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77.py#L77) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77.py#L86) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-5719b68e3aff"></a>
## betalens-factor/alpha101/ALPHA77/factor_ALPHA77.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77.yaml) · 46 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77.yaml#L7)：`factor_spec:`
- [L30](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77.yaml#L30)：`weight:`
- [L36](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77.yaml#L36)：`run:`

<a id="file-d5ab0ce0e90c"></a>
## betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.py) · 97 行 · 说明来源：文件族规则

- **作用**：ALPHA77 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha77_timing](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.py#L29) | compute_alpha77_timing(high_wide, low_wide, vwap_wide, amount_wide, *, high_low_divisor=2, vwap_high_low_decay_window=20.0451, high_low_divisor_2=2, amount_average_window=40, high_low_adv_correlation_window=3.1614, correlation_adv_high_decay_window=5.64125, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(77, high_wide=high_wide, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, high_low_divisor=high_low_divisor, vwap_high_low_decay_window=vwap_high_low_decay_window, high_low_divisor_2=high_low_divisor_2, amount_average_window=amount_average_window, high_low…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.py#L60) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.py#L80) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.py#L89) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-84d4892a0d0e"></a>
## betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.yaml) · 55 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.yaml#L8)：`factor_spec:`
- [L39](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.yaml#L39)：`weight:`
- [L45](../../../betalens-factor/alpha101/ALPHA77/factor_ALPHA77_timing.yaml#L45)：`run:`

<a id="file-6a9250151302"></a>
## betalens-factor/alpha101/ALPHA78/factor_ALPHA78.py

[打开源码](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78.py) · 98 行 · 说明来源：文件族规则

- **作用**：ALPHA78 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha78](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78.py#L29) | compute_alpha78(low_wide, volume_wide, vwap_wide, amount_wide, *, low_mix_weight=0.352233, mixed_complement_base=1, mixed_complement_weight=0.352233, mixed_sum_window=19.7428, amount_average_window=40, adv_sum_window=19.7428, ts_sum_mixed_ts_sum_adv_correlation_window=6.83313, rank_vwap_rank_volume_correlation_window=5.77492) | 无返回注解；return: compute_alpha(78, low_wide=low_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, low_mix_weight=low_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, mixed_sum_window=mixed_sum_window, amount_average_win…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78.py#L61) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78.py#L81) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78.py#L90) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-6618085808d7"></a>
## betalens-factor/alpha101/ALPHA78/factor_ALPHA78.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78.yaml#L7)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78.yaml#L38)：`run:`

<a id="file-2fa6c62ff896"></a>
## betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.py) · 101 行 · 说明来源：文件族规则

- **作用**：ALPHA78 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha78_timing](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.py#L29) | compute_alpha78_timing(low_wide, volume_wide, vwap_wide, amount_wide, *, low_mix_weight=0.352233, mixed_complement_base=1, mixed_complement_weight=0.352233, mixed_sum_window=19.7428, amount_average_window=40, adv_sum_window=19.7428, ts_sum_mixed_ts_sum_adv_correlation_window=6.83313, rank_vwap_rank_volume_correlation_window=5.77492, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(78, low_wide=low_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, low_mix_weight=low_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, mixed_sum_window=mixed_sum_window, amount_average_win…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.py#L64) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.py#L84) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.py#L93) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-5e61783ed561"></a>
## betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.yaml) · 57 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.yaml#L8)：`factor_spec:`
- [L41](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.yaml#L41)：`weight:`
- [L47](../../../betalens-factor/alpha101/ALPHA78/factor_ALPHA78_timing.yaml#L47)：`run:`

<a id="file-28bea6ce407e"></a>
## betalens-factor/alpha101/ALPHA79/factor_ALPHA79.py

[打开源码](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79.py) · 100 行 · 说明来源：文件族规则

- **作用**：ALPHA79 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha79](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79.py#L29) | compute_alpha79(open_wide, close_wide, vwap_wide, amount_wide, sector_wide, *, close_mix_weight=0.60733, mixed_complement_base=1, mixed_complement_weight=0.60733, indneutralize_mixed_sector_delta_lag=1.23438, vwap_rank_window=3.60973, amount_average_window=150, adv_rank_window=9.18637, ts_rank_vwap_ts_rank_adv_correlation_window=14.6644) | 无返回注解；return: compute_alpha(79, open_wide=open_wide, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, sector_wide=sector_wide, close_mix_weight=close_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, indneutralize_mixed_sect…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79.py#L63) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79.py#L83) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79.py#L92) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-51f9105ebdf2"></a>
## betalens-factor/alpha101/ALPHA79/factor_ALPHA79.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79.yaml#L7)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79.yaml#L39)：`run:`

<a id="file-6c31b23caf92"></a>
## betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.py) · 103 行 · 说明来源：文件族规则

- **作用**：ALPHA79 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha79_timing](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.py#L29) | compute_alpha79_timing(open_wide, close_wide, vwap_wide, amount_wide, sector_wide, *, close_mix_weight=0.60733, mixed_complement_base=1, mixed_complement_weight=0.60733, indneutralize_mixed_sector_delta_lag=1.23438, vwap_rank_window=3.60973, amount_average_window=150, adv_rank_window=9.18637, ts_rank_vwap_ts_rank_adv_correlation_window=14.6644, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(79, open_wide=open_wide, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, sector_wide=sector_wide, close_mix_weight=close_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, indneutralize_mixed_sect…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.py#L66) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.py#L86) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.py#L95) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-6e4fa6f87f84"></a>
## betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.yaml) · 58 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.yaml#L8)：`factor_spec:`
- [L42](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.yaml#L42)：`weight:`
- [L48](../../../betalens-factor/alpha101/ALPHA79/factor_ALPHA79_timing.yaml#L48)：`run:`

<a id="file-3492ae9e3dcf"></a>
## betalens-factor/alpha101/ALPHA8/factor_ALPHA8.py

[打开源码](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8.py) · 84 行 · 说明来源：文件族规则

- **作用**：ALPHA8 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha8](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8.py#L29) | compute_alpha8(open_wide, returns_wide, *, open_sum_window=5, returns_sum_window=5, base_delay_lag=10) | 无返回注解；return: compute_alpha(8, open_wide=open_wide, returns_wide=returns_wide, open_sum_window=open_sum_window, returns_sum_window=returns_sum_window, base_delay_lag=base_delay_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8.py#L47) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8.py#L67) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8.py#L76) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-c80eeff31e5f"></a>
## betalens-factor/alpha101/ALPHA8/factor_ALPHA8.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8.yaml) · 41 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8.yaml#L7)：`factor_spec:`
- [L25](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8.yaml#L25)：`weight:`
- [L31](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8.yaml#L31)：`run:`

<a id="file-8829674d4a84"></a>
## betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.py) · 87 行 · 说明来源：文件族规则

- **作用**：ALPHA8 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha8_timing](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.py#L29) | compute_alpha8_timing(open_wide, returns_wide, *, open_sum_window=5, returns_sum_window=5, base_delay_lag=10, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(8, open_wide=open_wide, returns_wide=returns_wide, open_sum_window=open_sum_window, returns_sum_window=returns_sum_window, base_delay_lag=base_delay_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.py#L50) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.py#L70) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.py#L79) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-96f97a4bc31b"></a>
## betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.yaml) · 50 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.yaml#L8)：`factor_spec:`
- [L34](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.yaml#L34)：`weight:`
- [L40](../../../betalens-factor/alpha101/ALPHA8/factor_ALPHA8_timing.yaml#L40)：`run:`

<a id="file-07c7e0eea7d7"></a>
## betalens-factor/alpha101/ALPHA80/factor_ALPHA80.py

[打开源码](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80.py) · 96 行 · 说明来源：文件族规则

- **作用**：ALPHA80 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha80](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80.py#L29) | compute_alpha80(open_wide, high_wide, amount_wide, industry_wide, *, open_mix_weight=0.868128, mixed_complement_base=1, mixed_complement_weight=0.868128, indneutralize_mixed_industry_delta_lag=4.04545, amount_average_window=10, high_adv_correlation_window=5.11456, correlation_high_adv_rank_window=5.53756) | 无返回注解；return: compute_alpha(80, open_wide=open_wide, high_wide=high_wide, amount_wide=amount_wide, industry_wide=industry_wide, open_mix_weight=open_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, indneutralize_mixed_industry_delta_lag=indneu…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80.py#L59) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80.py#L79) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80.py#L88) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-9381b95598f6"></a>
## betalens-factor/alpha101/ALPHA80/factor_ALPHA80.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80.yaml) · 47 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80.yaml#L7)：`factor_spec:`
- [L31](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80.yaml#L31)：`weight:`
- [L37](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80.yaml#L37)：`run:`

<a id="file-0add03c362a6"></a>
## betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.py) · 99 行 · 说明来源：文件族规则

- **作用**：ALPHA80 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha80_timing](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.py#L29) | compute_alpha80_timing(open_wide, high_wide, amount_wide, industry_wide, *, open_mix_weight=0.868128, mixed_complement_base=1, mixed_complement_weight=0.868128, indneutralize_mixed_industry_delta_lag=4.04545, amount_average_window=10, high_adv_correlation_window=5.11456, correlation_high_adv_rank_window=5.53756, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(80, open_wide=open_wide, high_wide=high_wide, amount_wide=amount_wide, industry_wide=industry_wide, open_mix_weight=open_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, indneutralize_mixed_industry_delta_lag=indneu…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.py#L62) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.py#L82) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.py#L91) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d177a879b75d"></a>
## betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.yaml) · 56 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.yaml#L8)：`factor_spec:`
- [L40](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.yaml#L40)：`weight:`
- [L46](../../../betalens-factor/alpha101/ALPHA80/factor_ALPHA80_timing.yaml#L46)：`run:`

<a id="file-eea4770e1e15"></a>
## betalens-factor/alpha101/ALPHA81/factor_ALPHA81.py

[打开源码](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81.py) · 92 行 · 说明来源：文件族规则

- **作用**：ALPHA81 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha81](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81.py#L29) | compute_alpha81(volume_wide, vwap_wide, amount_wide, *, amount_average_window=10, adv_sum_window=49.6054, vwap_ts_sum_adv_correlation_window=8.47743, left_constant=4, rank_corr_product_window=14.9655, rank_vwap_rank_volume_correlation_window=5.07914) | 无返回注解；return: compute_alpha(81, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window, vwap_ts_sum_adv_correlation_window=vwap_ts_sum_adv_correlation_window, left_constant=left_constant, rank_corr_product_win…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81.py#L55) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81.py#L75) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81.py#L84) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-8e9ed858c298"></a>
## betalens-factor/alpha101/ALPHA81/factor_ALPHA81.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81.yaml) · 45 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81.yaml#L7)：`factor_spec:`
- [L29](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81.yaml#L29)：`weight:`
- [L35](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81.yaml#L35)：`run:`

<a id="file-385ffe3cfd7a"></a>
## betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.py) · 95 行 · 说明来源：文件族规则

- **作用**：ALPHA81 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha81_timing](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.py#L29) | compute_alpha81_timing(volume_wide, vwap_wide, amount_wide, *, amount_average_window=10, adv_sum_window=49.6054, vwap_ts_sum_adv_correlation_window=8.47743, left_constant=4, rank_corr_product_window=14.9655, rank_vwap_rank_volume_correlation_window=5.07914, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(81, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window, vwap_ts_sum_adv_correlation_window=vwap_ts_sum_adv_correlation_window, left_constant=left_constant, rank_corr_product_win…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.py#L58) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.py#L78) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.py#L87) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-cd2009eded60"></a>
## betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.yaml) · 54 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.yaml#L8)：`factor_spec:`
- [L38](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.yaml#L38)：`weight:`
- [L44](../../../betalens-factor/alpha101/ALPHA81/factor_ALPHA81_timing.yaml#L44)：`run:`

<a id="file-56b338525abb"></a>
## betalens-factor/alpha101/ALPHA82/factor_ALPHA82.py

[打开源码](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82.py) · 90 行 · 说明来源：文件族规则

- **作用**：ALPHA82 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha82](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82.py#L29) | compute_alpha82(open_wide, volume_wide, sector_wide, *, open_delta_lag=1.46063, delta_open_decay_window=14.8717, indneutralize_volume_sector_open_correlation_window=17.4842, corr_decay_window=6.92131, decay_linear_corr_rank_window=13.4283) | 无返回注解；return: compute_alpha(82, open_wide=open_wide, volume_wide=volume_wide, sector_wide=sector_wide, open_delta_lag=open_delta_lag, delta_open_decay_window=delta_open_decay_window, indneutralize_volume_sector_open_correlation_window=indneutralize_volume_sector_open_correlation_window, corr_decay_w…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82.py#L53) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82.py#L73) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82.py#L82) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-76419b58262b"></a>
## betalens-factor/alpha101/ALPHA82/factor_ALPHA82.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82.yaml) · 44 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82.yaml#L7)：`factor_spec:`
- [L28](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82.yaml#L28)：`weight:`
- [L34](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82.yaml#L34)：`run:`

<a id="file-32b049a48b64"></a>
## betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.py) · 93 行 · 说明来源：文件族规则

- **作用**：ALPHA82 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha82_timing](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.py#L29) | compute_alpha82_timing(open_wide, volume_wide, sector_wide, *, open_delta_lag=1.46063, delta_open_decay_window=14.8717, indneutralize_volume_sector_open_correlation_window=17.4842, corr_decay_window=6.92131, decay_linear_corr_rank_window=13.4283, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(82, open_wide=open_wide, volume_wide=volume_wide, sector_wide=sector_wide, open_delta_lag=open_delta_lag, delta_open_decay_window=delta_open_decay_window, indneutralize_volume_sector_open_correlation_window=indneutralize_volume_sector_open_correlation_window, corr_decay_w…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.py#L56) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.py#L76) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.py#L85) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-7ff385eba426"></a>
## betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.yaml) · 53 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.yaml#L8)：`factor_spec:`
- [L37](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.yaml#L37)：`weight:`
- [L43](../../../betalens-factor/alpha101/ALPHA82/factor_ALPHA82_timing.yaml#L43)：`run:`

<a id="file-de0d764dacb8"></a>
## betalens-factor/alpha101/ALPHA83/factor_ALPHA83.py

[打开源码](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83.py) · 88 行 · 说明来源：文件族规则

- **作用**：ALPHA83 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha83](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83.py#L29) | compute_alpha83(close_wide, high_wide, low_wide, volume_wide, vwap_wide, *, close_mean_window=5, ratio_delay_lag=2) | 无返回注解；return: compute_alpha(83, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, close_mean_window=close_mean_window, ratio_delay_lag=ratio_delay_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83.py#L51) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83.py#L71) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83.py#L80) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-f1fb6218b8c6"></a>
## betalens-factor/alpha101/ALPHA83/factor_ALPHA83.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83.yaml) · 43 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83.yaml#L7)：`factor_spec:`
- [L27](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83.yaml#L27)：`weight:`
- [L33](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83.yaml#L33)：`run:`

<a id="file-2a85a6243231"></a>
## betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.py) · 91 行 · 说明来源：文件族规则

- **作用**：ALPHA83 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha83_timing](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.py#L29) | compute_alpha83_timing(close_wide, high_wide, low_wide, volume_wide, vwap_wide, *, close_mean_window=5, ratio_delay_lag=2, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(83, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, close_mean_window=close_mean_window, ratio_delay_lag=ratio_delay_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.py#L54) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.py#L74) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.py#L83) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ca96eba3d79f"></a>
## betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.yaml) · 52 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.yaml#L8)：`factor_spec:`
- [L36](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.yaml#L36)：`weight:`
- [L42](../../../betalens-factor/alpha101/ALPHA83/factor_ALPHA83_timing.yaml#L42)：`run:`

<a id="file-00803e89af77"></a>
## betalens-factor/alpha101/ALPHA84/factor_ALPHA84.py

[打开源码](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84.py) · 84 行 · 说明来源：文件族规则

- **作用**：ALPHA84 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha84](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84.py#L29) | compute_alpha84(close_wide, vwap_wide, *, vwap_maximum_window=15.3217, vwap_ts_max_rank_window=20.7127, close_delta_lag=4.96796) | 无返回注解；return: compute_alpha(84, close_wide=close_wide, vwap_wide=vwap_wide, vwap_maximum_window=vwap_maximum_window, vwap_ts_max_rank_window=vwap_ts_max_rank_window, close_delta_lag=close_delta_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84.py#L47) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84.py#L67) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84.py#L76) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-4c320e79ac0f"></a>
## betalens-factor/alpha101/ALPHA84/factor_ALPHA84.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84.yaml) · 41 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84.yaml#L7)：`factor_spec:`
- [L25](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84.yaml#L25)：`weight:`
- [L31](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84.yaml#L31)：`run:`

<a id="file-02786166ebc8"></a>
## betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.py) · 87 行 · 说明来源：文件族规则

- **作用**：ALPHA84 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha84_timing](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.py#L29) | compute_alpha84_timing(close_wide, vwap_wide, *, vwap_maximum_window=15.3217, vwap_ts_max_rank_window=20.7127, close_delta_lag=4.96796, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(84, close_wide=close_wide, vwap_wide=vwap_wide, vwap_maximum_window=vwap_maximum_window, vwap_ts_max_rank_window=vwap_ts_max_rank_window, close_delta_lag=close_delta_lag) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.py#L50) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.py#L70) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.py#L79) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-2d0f7dd5dfc3"></a>
## betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.yaml) · 50 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.yaml#L8)：`factor_spec:`
- [L34](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.yaml#L34)：`weight:`
- [L40](../../../betalens-factor/alpha101/ALPHA84/factor_ALPHA84_timing.yaml#L40)：`run:`

<a id="file-99752000e5e1"></a>
## betalens-factor/alpha101/ALPHA85/factor_ALPHA85.py

[打开源码](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85.py) · 102 行 · 说明来源：文件族规则

- **作用**：ALPHA85 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha85](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85.py#L29) | compute_alpha85(close_wide, high_wide, low_wide, volume_wide, amount_wide, *, high_mix_weight=0.876703, mixed_complement_base=1, mixed_complement_weight=0.876703, amount_average_window=30, mixed_adv_correlation_window=9.61331, high_low_divisor=2, high_low_rank_window=3.70596, volume_rank_window=10.1595, ts_rank_high_low_ts_rank_volume_correlation_window=7.11408) | 无返回注解；return: compute_alpha(85, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, amount_wide=amount_wide, high_mix_weight=high_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, amount_average_window=amount…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85.py#L65) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85.py#L85) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85.py#L94) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-c093a2a58e9d"></a>
## betalens-factor/alpha101/ALPHA85/factor_ALPHA85.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85.yaml) · 50 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85.yaml#L7)：`factor_spec:`
- [L34](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85.yaml#L34)：`weight:`
- [L40](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85.yaml#L40)：`run:`

<a id="file-38810f88571d"></a>
## betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.py) · 105 行 · 说明来源：文件族规则

- **作用**：ALPHA85 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha85_timing](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.py#L29) | compute_alpha85_timing(close_wide, high_wide, low_wide, volume_wide, amount_wide, *, high_mix_weight=0.876703, mixed_complement_base=1, mixed_complement_weight=0.876703, amount_average_window=30, mixed_adv_correlation_window=9.61331, high_low_divisor=2, high_low_rank_window=3.70596, volume_rank_window=10.1595, ts_rank_high_low_ts_rank_volume_correlation_window=7.11408, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(85, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, amount_wide=amount_wide, high_mix_weight=high_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, amount_average_window=amount…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.py#L68) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.py#L88) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.py#L97) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-e9355840bf8e"></a>
## betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.yaml) · 59 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.yaml#L8)：`factor_spec:`
- [L43](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.yaml#L43)：`weight:`
- [L49](../../../betalens-factor/alpha101/ALPHA85/factor_ALPHA85_timing.yaml#L49)：`run:`

<a id="file-4d22420e12b5"></a>
## betalens-factor/alpha101/ALPHA86/factor_ALPHA86.py

[打开源码](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86.py) · 88 行 · 说明来源：文件族规则

- **作用**：ALPHA86 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha86](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86.py#L29) | compute_alpha86(close_wide, vwap_wide, amount_wide, *, amount_average_window=20, adv_sum_window=14.7444, close_ts_sum_adv_correlation_window=6.00049, correlation_close_ts_sum_rank_window=20.4195) | 无返回注解；return: compute_alpha(86, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window, close_ts_sum_adv_correlation_window=close_ts_sum_adv_correlation_window, correlation_close_ts_sum_rank_window=correlation_c…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86.py#L51) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86.py#L71) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86.py#L80) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-5070dcb0d673"></a>
## betalens-factor/alpha101/ALPHA86/factor_ALPHA86.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86.yaml) · 43 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86.yaml#L7)：`factor_spec:`
- [L27](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86.yaml#L27)：`weight:`
- [L33](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86.yaml#L33)：`run:`

<a id="file-e84e77ed5e3b"></a>
## betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.py) · 91 行 · 说明来源：文件族规则

- **作用**：ALPHA86 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha86_timing](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.py#L29) | compute_alpha86_timing(close_wide, vwap_wide, amount_wide, *, amount_average_window=20, adv_sum_window=14.7444, close_ts_sum_adv_correlation_window=6.00049, correlation_close_ts_sum_rank_window=20.4195, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(86, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window, close_ts_sum_adv_correlation_window=close_ts_sum_adv_correlation_window, correlation_close_ts_sum_rank_window=correlation_c…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.py#L54) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.py#L74) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.py#L83) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-34c66a30d60e"></a>
## betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.yaml) · 52 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.yaml#L8)：`factor_spec:`
- [L36](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.yaml#L36)：`weight:`
- [L42](../../../betalens-factor/alpha101/ALPHA86/factor_ALPHA86_timing.yaml#L42)：`run:`

<a id="file-75cafd9d21db"></a>
## betalens-factor/alpha101/ALPHA87/factor_ALPHA87.py

[打开源码](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87.py) · 100 行 · 说明来源：文件族规则

- **作用**：ALPHA87 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha87](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87.py#L29) | compute_alpha87(close_wide, vwap_wide, amount_wide, industry_wide, *, close_mix_weight=0.369701, mixed_complement_base=1, mixed_complement_weight=0.369701, mixed_delta_lag=1.91233, delta_mixed_decay_window=2.65461, amount_average_window=81, indneutralize_industry_adv_close_correlation_window=13.4132, corr_decay_window=4.89768, decay_linear_corr_rank_window=14.4535) | 无返回注解；return: compute_alpha(87, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, close_mix_weight=close_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, mixed_delta_lag=mixed_delta_lag, delta_mi…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87.py#L63) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87.py#L83) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87.py#L92) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ca6eb91986ba"></a>
## betalens-factor/alpha101/ALPHA87/factor_ALPHA87.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87.yaml#L7)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87.yaml#L39)：`run:`

<a id="file-fc1c5e94a59f"></a>
## betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.py) · 103 行 · 说明来源：文件族规则

- **作用**：ALPHA87 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha87_timing](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.py#L29) | compute_alpha87_timing(close_wide, vwap_wide, amount_wide, industry_wide, *, close_mix_weight=0.369701, mixed_complement_base=1, mixed_complement_weight=0.369701, mixed_delta_lag=1.91233, delta_mixed_decay_window=2.65461, amount_average_window=81, indneutralize_industry_adv_close_correlation_window=13.4132, corr_decay_window=4.89768, decay_linear_corr_rank_window=14.4535, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(87, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, close_mix_weight=close_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, mixed_delta_lag=mixed_delta_lag, delta_mi…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.py#L66) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.py#L86) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.py#L95) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-2596e31c1a79"></a>
## betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.yaml) · 58 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.yaml#L8)：`factor_spec:`
- [L42](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.yaml#L42)：`weight:`
- [L48](../../../betalens-factor/alpha101/ALPHA87/factor_ALPHA87_timing.yaml#L48)：`run:`

<a id="file-84359d8f3eb7"></a>
## betalens-factor/alpha101/ALPHA88/factor_ALPHA88.py

[打开源码](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88.py) · 98 行 · 说明来源：文件族规则

- **作用**：ALPHA88 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha88](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88.py#L29) | compute_alpha88(open_wide, close_wide, high_wide, low_wide, amount_wide, *, rank_close_high_decay_window=8.06882, close_rank_window=8.44728, amount_average_window=60, adv_rank_window=20.6966, ts_rank_close_ts_rank_adv_correlation_window=8.01266, corr_decay_window=6.65053, decay_linear_corr_rank_window=2.61957) | 无返回注解；return: compute_alpha(88, open_wide=open_wide, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, amount_wide=amount_wide, rank_close_high_decay_window=rank_close_high_decay_window, close_rank_window=close_rank_window, amount_average_window=amount_average_window, adv_rank_window=ad…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88.py#L61) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88.py#L81) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88.py#L90) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-5bd1f20164d0"></a>
## betalens-factor/alpha101/ALPHA88/factor_ALPHA88.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88.yaml#L7)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88.yaml#L38)：`run:`

<a id="file-90dec0a28311"></a>
## betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.py) · 101 行 · 说明来源：文件族规则

- **作用**：ALPHA88 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha88_timing](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.py#L29) | compute_alpha88_timing(open_wide, close_wide, high_wide, low_wide, amount_wide, *, rank_close_high_decay_window=8.06882, close_rank_window=8.44728, amount_average_window=60, adv_rank_window=20.6966, ts_rank_close_ts_rank_adv_correlation_window=8.01266, corr_decay_window=6.65053, decay_linear_corr_rank_window=2.61957, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(88, open_wide=open_wide, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, amount_wide=amount_wide, rank_close_high_decay_window=rank_close_high_decay_window, close_rank_window=close_rank_window, amount_average_window=amount_average_window, adv_rank_window=ad…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.py#L64) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.py#L84) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.py#L93) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-1821814f64e4"></a>
## betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.yaml) · 57 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.yaml#L8)：`factor_spec:`
- [L41](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.yaml#L41)：`weight:`
- [L47](../../../betalens-factor/alpha101/ALPHA88/factor_ALPHA88_timing.yaml#L47)：`run:`

<a id="file-60a529d302ab"></a>
## betalens-factor/alpha101/ALPHA89/factor_ALPHA89.py

[打开源码](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89.py) · 96 行 · 说明来源：文件族规则

- **作用**：ALPHA89 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha89](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89.py#L29) | compute_alpha89(low_wide, vwap_wide, amount_wide, industry_wide, *, amount_average_window=10, low_adv_correlation_window=6.94279, correlation_low_adv_decay_window=5.51607, decay_linear_correlation_low_rank_window=3.79744, indneutralize_vwap_industry_delta_lag=3.48158, delta_indneutralize_vwap_decay_window=10.1466, decay_linear_delta_indneutralize_rank_window=15.3012) | 无返回注解；return: compute_alpha(89, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, amount_average_window=amount_average_window, low_adv_correlation_window=low_adv_correlation_window, correlation_low_adv_decay_window=correlation_low_adv_decay_window, decay_l…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89.py#L59) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89.py#L79) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89.py#L88) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-ae4e29bc89aa"></a>
## betalens-factor/alpha101/ALPHA89/factor_ALPHA89.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89.yaml) · 47 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89.yaml#L7)：`factor_spec:`
- [L31](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89.yaml#L31)：`weight:`
- [L37](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89.yaml#L37)：`run:`

<a id="file-b60c2f5c9455"></a>
## betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.py) · 99 行 · 说明来源：文件族规则

- **作用**：ALPHA89 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha89_timing](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.py#L29) | compute_alpha89_timing(low_wide, vwap_wide, amount_wide, industry_wide, *, amount_average_window=10, low_adv_correlation_window=6.94279, correlation_low_adv_decay_window=5.51607, decay_linear_correlation_low_rank_window=3.79744, indneutralize_vwap_industry_delta_lag=3.48158, delta_indneutralize_vwap_decay_window=10.1466, decay_linear_delta_indneutralize_rank_window=15.3012, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(89, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, amount_average_window=amount_average_window, low_adv_correlation_window=low_adv_correlation_window, correlation_low_adv_decay_window=correlation_low_adv_decay_window, decay_l…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.py#L62) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.py#L82) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.py#L91) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-af5e6e0c20b6"></a>
## betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.yaml) · 56 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.yaml#L8)：`factor_spec:`
- [L40](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.yaml#L40)：`weight:`
- [L46](../../../betalens-factor/alpha101/ALPHA89/factor_ALPHA89_timing.yaml#L46)：`run:`

<a id="file-1edf69094b89"></a>
## betalens-factor/alpha101/ALPHA9/factor_ALPHA9.py

[打开源码](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9.py) · 86 行 · 说明来源：文件族规则

- **作用**：ALPHA9 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha9](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9.py#L29) | compute_alpha9(close_wide, *, close_delta_lag=1, change_minimum_window=5, ts_min_change_threshold=0, change_maximum_window=5, ts_max_change_threshold=0) | 无返回注解；return: compute_alpha(9, close_wide=close_wide, close_delta_lag=close_delta_lag, change_minimum_window=change_minimum_window, ts_min_change_threshold=ts_min_change_threshold, change_maximum_window=change_maximum_window, ts_max_change_threshold=ts_max_change_threshold) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9.py#L49) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9.py#L69) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9.py#L78) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d55dd9007d3c"></a>
## betalens-factor/alpha101/ALPHA9/factor_ALPHA9.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9.yaml) · 43 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9.yaml#L8)：`factor_spec:`
- [L27](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9.yaml#L27)：`weight:`
- [L33](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9.yaml#L33)：`run:`

<a id="file-2a4817cfa222"></a>
## betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.py) · 89 行 · 说明来源：文件族规则

- **作用**：ALPHA9 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha9_timing](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.py#L29) | compute_alpha9_timing(close_wide, *, close_delta_lag=1, change_minimum_window=5, ts_min_change_threshold=0, change_maximum_window=5, ts_max_change_threshold=0, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(9, close_wide=close_wide, close_delta_lag=close_delta_lag, change_minimum_window=change_minimum_window, ts_min_change_threshold=ts_min_change_threshold, change_maximum_window=change_maximum_window, ts_max_change_threshold=ts_max_change_threshold) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.py#L52) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.py#L72) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.py#L81) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-551f6fd52c7a"></a>
## betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.yaml) · 51 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.yaml#L8)：`factor_spec:`
- [L35](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.yaml#L35)：`weight:`
- [L41](../../../betalens-factor/alpha101/ALPHA9/factor_ALPHA9_timing.yaml#L41)：`run:`

<a id="file-3283d2fcf9bc"></a>
## betalens-factor/alpha101/ALPHA90/factor_ALPHA90.py

[打开源码](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90.py) · 90 行 · 说明来源：文件族规则

- **作用**：ALPHA90 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha90](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90.py#L29) | compute_alpha90(close_wide, low_wide, amount_wide, subindustry_wide, *, close_maximum_window=4.66719, amount_average_window=40, indneutralize_subindustry_adv_low_correlation_window=5.38375, correlation_low_indneutralize_rank_window=3.21856) | 无返回注解；return: compute_alpha(90, close_wide=close_wide, low_wide=low_wide, amount_wide=amount_wide, subindustry_wide=subindustry_wide, close_maximum_window=close_maximum_window, amount_average_window=amount_average_window, indneutralize_subindustry_adv_low_correlation_window=indneutralize_subindustry…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90.py#L53) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90.py#L73) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90.py#L82) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-35132c4881a7"></a>
## betalens-factor/alpha101/ALPHA90/factor_ALPHA90.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90.yaml) · 44 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90.yaml#L7)：`factor_spec:`
- [L28](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90.yaml#L28)：`weight:`
- [L34](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90.yaml#L34)：`run:`

<a id="file-6b9b9041e7c8"></a>
## betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.py) · 93 行 · 说明来源：文件族规则

- **作用**：ALPHA90 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha90_timing](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.py#L29) | compute_alpha90_timing(close_wide, low_wide, amount_wide, subindustry_wide, *, close_maximum_window=4.66719, amount_average_window=40, indneutralize_subindustry_adv_low_correlation_window=5.38375, correlation_low_indneutralize_rank_window=3.21856, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(90, close_wide=close_wide, low_wide=low_wide, amount_wide=amount_wide, subindustry_wide=subindustry_wide, close_maximum_window=close_maximum_window, amount_average_window=amount_average_window, indneutralize_subindustry_adv_low_correlation_window=indneutralize_subindustry…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.py#L56) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.py#L76) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.py#L85) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-db4419712897"></a>
## betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.yaml) · 53 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.yaml#L8)：`factor_spec:`
- [L37](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.yaml#L37)：`weight:`
- [L43](../../../betalens-factor/alpha101/ALPHA90/factor_ALPHA90_timing.yaml#L43)：`run:`

<a id="file-d3cbfe2317e5"></a>
## betalens-factor/alpha101/ALPHA91/factor_ALPHA91.py

[打开源码](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91.py) · 98 行 · 说明来源：文件族规则

- **作用**：ALPHA91 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha91](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91.py#L29) | compute_alpha91(close_wide, volume_wide, vwap_wide, amount_wide, industry_wide, *, indneutralize_close_industry_volume_correlation_window=9.74928, corr1_decay_window=16.398, decay_linear_corr1_decay_window=3.83219, decay_linear_corr1_rank_window=4.8667, amount_average_window=30, vwap_adv_correlation_window=4.01303, correlation_vwap_adv_decay_window=2.6809) | 无返回注解；return: compute_alpha(91, close_wide=close_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, indneutralize_close_industry_volume_correlation_window=indneutralize_close_industry_volume_correlation_window, corr1_decay_window=corr1_decay_win…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91.py#L61) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91.py#L81) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91.py#L90) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-99afce5c7680"></a>
## betalens-factor/alpha101/ALPHA91/factor_ALPHA91.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91.yaml#L7)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91.yaml#L38)：`run:`

<a id="file-379211040955"></a>
## betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.py) · 101 行 · 说明来源：文件族规则

- **作用**：ALPHA91 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha91_timing](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.py#L29) | compute_alpha91_timing(close_wide, volume_wide, vwap_wide, amount_wide, industry_wide, *, indneutralize_close_industry_volume_correlation_window=9.74928, corr1_decay_window=16.398, decay_linear_corr1_decay_window=3.83219, decay_linear_corr1_rank_window=4.8667, amount_average_window=30, vwap_adv_correlation_window=4.01303, correlation_vwap_adv_decay_window=2.6809, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(91, close_wide=close_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, indneutralize_close_industry_volume_correlation_window=indneutralize_close_industry_volume_correlation_window, corr1_decay_window=corr1_decay_win…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.py#L64) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.py#L84) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.py#L93) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-170e60915958"></a>
## betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.yaml) · 57 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.yaml#L8)：`factor_spec:`
- [L41](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.yaml#L41)：`weight:`
- [L47](../../../betalens-factor/alpha101/ALPHA91/factor_ALPHA91_timing.yaml#L47)：`run:`

<a id="file-8ca21a96a32f"></a>
## betalens-factor/alpha101/ALPHA92/factor_ALPHA92.py

[打开源码](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92.py) · 98 行 · 说明来源：文件族规则

- **作用**：ALPHA92 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha92](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92.py#L29) | compute_alpha92(open_wide, close_wide, high_wide, low_wide, amount_wide, *, high_low_divisor=2, condition_decay_window=14.7221, decay_linear_condition_rank_window=18.8683, amount_average_window=30, rank_low_rank_adv_correlation_window=7.58555, correlation_rank_low_decay_window=6.94024, decay_linear_correlation_rank_rank_window=6.80584) | 无返回注解；return: compute_alpha(92, open_wide=open_wide, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, amount_wide=amount_wide, high_low_divisor=high_low_divisor, condition_decay_window=condition_decay_window, decay_linear_condition_rank_window=decay_linear_condition_rank_window, amount…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92.py#L61) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92.py#L81) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92.py#L90) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-0e9c204682e7"></a>
## betalens-factor/alpha101/ALPHA92/factor_ALPHA92.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92.yaml#L7)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92.yaml#L38)：`run:`

<a id="file-0eee0089b6b3"></a>
## betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.py) · 101 行 · 说明来源：文件族规则

- **作用**：ALPHA92 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha92_timing](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.py#L29) | compute_alpha92_timing(open_wide, close_wide, high_wide, low_wide, amount_wide, *, high_low_divisor=2, condition_decay_window=14.7221, decay_linear_condition_rank_window=18.8683, amount_average_window=30, rank_low_rank_adv_correlation_window=7.58555, correlation_rank_low_decay_window=6.94024, decay_linear_correlation_rank_rank_window=6.80584, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(92, open_wide=open_wide, close_wide=close_wide, high_wide=high_wide, low_wide=low_wide, amount_wide=amount_wide, high_low_divisor=high_low_divisor, condition_decay_window=condition_decay_window, decay_linear_condition_rank_window=decay_linear_condition_rank_window, amount…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.py#L64) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.py#L84) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.py#L93) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-f754034a0b6c"></a>
## betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.yaml) · 57 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.yaml#L8)：`factor_spec:`
- [L41](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.yaml#L41)：`weight:`
- [L47](../../../betalens-factor/alpha101/ALPHA92/factor_ALPHA92_timing.yaml#L47)：`run:`

<a id="file-ff95bf4d0c7c"></a>
## betalens-factor/alpha101/ALPHA93/factor_ALPHA93.py

[打开源码](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93.py) · 100 行 · 说明来源：文件族规则

- **作用**：ALPHA93 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha93](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93.py#L29) | compute_alpha93(close_wide, vwap_wide, amount_wide, industry_wide, *, amount_average_window=81, indneutralize_vwap_industry_adv_correlation_window=17.4193, correlation_indneutralize_vwap_decay_window=19.848, decay_linear_correlation_indneutralize_rank_window=7.54455, close_mix_weight=0.524434, mixed_complement_base=1, mixed_complement_weight=0.524434, mixed_delta_lag=2.77377, delta_mixed_decay_window=16.2664) | 无返回注解；return: compute_alpha(93, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, amount_average_window=amount_average_window, indneutralize_vwap_industry_adv_correlation_window=indneutralize_vwap_industry_adv_correlation_window, correlation_indneutral…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93.py#L63) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93.py#L83) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93.py#L92) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-7bb126539f35"></a>
## betalens-factor/alpha101/ALPHA93/factor_ALPHA93.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93.yaml) · 49 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93.yaml#L7)：`factor_spec:`
- [L33](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93.yaml#L33)：`weight:`
- [L39](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93.yaml#L39)：`run:`

<a id="file-8043387c0d74"></a>
## betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.py) · 103 行 · 说明来源：文件族规则

- **作用**：ALPHA93 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha93_timing](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.py#L29) | compute_alpha93_timing(close_wide, vwap_wide, amount_wide, industry_wide, *, amount_average_window=81, indneutralize_vwap_industry_adv_correlation_window=17.4193, correlation_indneutralize_vwap_decay_window=19.848, decay_linear_correlation_indneutralize_rank_window=7.54455, close_mix_weight=0.524434, mixed_complement_base=1, mixed_complement_weight=0.524434, mixed_delta_lag=2.77377, delta_mixed_decay_window=16.2664, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(93, close_wide=close_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, amount_average_window=amount_average_window, indneutralize_vwap_industry_adv_correlation_window=indneutralize_vwap_industry_adv_correlation_window, correlation_indneutral…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.py#L66) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.py#L86) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.py#L95) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-da573a99676e"></a>
## betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.yaml) · 58 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.yaml#L8)：`factor_spec:`
- [L42](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.yaml#L42)：`weight:`
- [L48](../../../betalens-factor/alpha101/ALPHA93/factor_ALPHA93_timing.yaml#L48)：`run:`

<a id="file-ac90a8d6b18f"></a>
## betalens-factor/alpha101/ALPHA94/factor_ALPHA94.py

[打开源码](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94.py) · 90 行 · 说明来源：文件族规则

- **作用**：ALPHA94 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha94](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94.py#L29) | compute_alpha94(vwap_wide, amount_wide, *, vwap_minimum_window=11.5783, vwap_rank_window=19.6462, amount_average_window=60, adv_rank_window=4.02992, ts_rank_vwap_ts_rank_adv_correlation_window=18.0926, correlation_ts_rank_vwap_rank_window=2.70756) | 无返回注解；return: compute_alpha(94, vwap_wide=vwap_wide, amount_wide=amount_wide, vwap_minimum_window=vwap_minimum_window, vwap_rank_window=vwap_rank_window, amount_average_window=amount_average_window, adv_rank_window=adv_rank_window, ts_rank_vwap_ts_rank_adv_correlation_window=ts_rank_vwap_ts_rank_adv…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94.py#L53) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94.py#L73) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94.py#L82) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-9b249c3209b7"></a>
## betalens-factor/alpha101/ALPHA94/factor_ALPHA94.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94.yaml) · 44 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94.yaml#L7)：`factor_spec:`
- [L28](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94.yaml#L28)：`weight:`
- [L34](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94.yaml#L34)：`run:`

<a id="file-163e512cce52"></a>
## betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.py) · 93 行 · 说明来源：文件族规则

- **作用**：ALPHA94 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha94_timing](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.py#L29) | compute_alpha94_timing(vwap_wide, amount_wide, *, vwap_minimum_window=11.5783, vwap_rank_window=19.6462, amount_average_window=60, adv_rank_window=4.02992, ts_rank_vwap_ts_rank_adv_correlation_window=18.0926, correlation_ts_rank_vwap_rank_window=2.70756, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(94, vwap_wide=vwap_wide, amount_wide=amount_wide, vwap_minimum_window=vwap_minimum_window, vwap_rank_window=vwap_rank_window, amount_average_window=amount_average_window, adv_rank_window=adv_rank_window, ts_rank_vwap_ts_rank_adv_correlation_window=ts_rank_vwap_ts_rank_adv…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.py#L56) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.py#L76) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.py#L85) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-2702c045bae8"></a>
## betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.yaml) · 53 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.yaml#L8)：`factor_spec:`
- [L37](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.yaml#L37)：`weight:`
- [L43](../../../betalens-factor/alpha101/ALPHA94/factor_ALPHA94_timing.yaml#L43)：`run:`

<a id="file-8622a8a93942"></a>
## betalens-factor/alpha101/ALPHA95/factor_ALPHA95.py

[打开源码](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95.py) · 98 行 · 说明来源：文件族规则

- **作用**：ALPHA95 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha95](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95.py#L29) | compute_alpha95(open_wide, high_wide, low_wide, amount_wide, *, open_minimum_window=12.4105, high_low_divisor=2, high_low_sum_window=19.1351, amount_average_window=40, adv_sum_window=19.1351, ts_sum_high_low_ts_sum_adv_correlation_window=12.8742, right_constant=5, rank_corr_rank_window=11.7584) | 无返回注解；return: compute_alpha(95, open_wide=open_wide, high_wide=high_wide, low_wide=low_wide, amount_wide=amount_wide, open_minimum_window=open_minimum_window, high_low_divisor=high_low_divisor, high_low_sum_window=high_low_sum_window, amount_average_window=amount_average_window, adv_sum_window=adv_s…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95.py#L61) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95.py#L81) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95.py#L90) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-514dda6fba18"></a>
## betalens-factor/alpha101/ALPHA95/factor_ALPHA95.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95.yaml#L7)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95.yaml#L38)：`run:`

<a id="file-b71a3818ccc1"></a>
## betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.py) · 101 行 · 说明来源：文件族规则

- **作用**：ALPHA95 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha95_timing](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.py#L29) | compute_alpha95_timing(open_wide, high_wide, low_wide, amount_wide, *, open_minimum_window=12.4105, high_low_divisor=2, high_low_sum_window=19.1351, amount_average_window=40, adv_sum_window=19.1351, ts_sum_high_low_ts_sum_adv_correlation_window=12.8742, right_constant=5, rank_corr_rank_window=11.7584, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(95, open_wide=open_wide, high_wide=high_wide, low_wide=low_wide, amount_wide=amount_wide, open_minimum_window=open_minimum_window, high_low_divisor=high_low_divisor, high_low_sum_window=high_low_sum_window, amount_average_window=amount_average_window, adv_sum_window=adv_s…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.py#L64) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.py#L84) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.py#L93) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d0a1b17cf0a5"></a>
## betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.yaml) · 57 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.yaml#L8)：`factor_spec:`
- [L41](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.yaml#L41)：`weight:`
- [L47](../../../betalens-factor/alpha101/ALPHA95/factor_ALPHA95_timing.yaml#L47)：`run:`

<a id="file-66137b094b81"></a>
## betalens-factor/alpha101/ALPHA96/factor_ALPHA96.py

[打开源码](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96.py) · 102 行 · 说明来源：文件族规则

- **作用**：ALPHA96 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha96](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96.py#L29) | compute_alpha96(close_wide, volume_wide, vwap_wide, amount_wide, *, rank_vwap_rank_volume_correlation_window=3.83878, correlation_rank_vwap_decay_window=4.16783, decay_linear_correlation_rank_rank_window=8.38151, close_rank_window=7.45404, amount_average_window=60, adv_rank_window=4.13242, ts_rank_close_ts_rank_adv_correlation_window=3.65459, corr_argmax_window=12.6556, ts_argmax_corr_decay_window=14.0365, decay_linear_ts_argmax_corr_rank_window=13.4143) | 无返回注解；return: compute_alpha(96, close_wide=close_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, rank_vwap_rank_volume_correlation_window=rank_vwap_rank_volume_correlation_window, correlation_rank_vwap_decay_window=correlation_rank_vwap_decay_window, decay_linear_correla…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96.py#L65) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96.py#L85) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96.py#L94) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-778541ee8268"></a>
## betalens-factor/alpha101/ALPHA96/factor_ALPHA96.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96.yaml) · 50 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96.yaml#L7)：`factor_spec:`
- [L34](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96.yaml#L34)：`weight:`
- [L40](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96.yaml#L40)：`run:`

<a id="file-246db38fe38f"></a>
## betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.py) · 105 行 · 说明来源：文件族规则

- **作用**：ALPHA96 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha96_timing](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.py#L29) | compute_alpha96_timing(close_wide, volume_wide, vwap_wide, amount_wide, *, rank_vwap_rank_volume_correlation_window=3.83878, correlation_rank_vwap_decay_window=4.16783, decay_linear_correlation_rank_rank_window=8.38151, close_rank_window=7.45404, amount_average_window=60, adv_rank_window=4.13242, ts_rank_close_ts_rank_adv_correlation_window=3.65459, corr_argmax_window=12.6556, ts_argmax_corr_decay_window=14.0365, decay_linear_ts_argmax_corr_rank_window=13.4143, stock_code=None, signal_weight=Non…（完整内容见 inventory.json/源码） | 无返回注解；return: compute_alpha(96, close_wide=close_wide, volume_wide=volume_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, rank_vwap_rank_volume_correlation_window=rank_vwap_rank_volume_correlation_window, correlation_rank_vwap_decay_window=correlation_rank_vwap_decay_window, decay_linear_correla…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.py#L68) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.py#L88) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.py#L97) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-dd0f795c9ef9"></a>
## betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.yaml) · 59 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.yaml#L8)：`factor_spec:`
- [L43](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.yaml#L43)：`weight:`
- [L49](../../../betalens-factor/alpha101/ALPHA96/factor_ALPHA96_timing.yaml#L49)：`run:`

<a id="file-fe23d6d56213"></a>
## betalens-factor/alpha101/ALPHA97/factor_ALPHA97.py

[打开源码](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97.py) · 106 行 · 说明来源：文件族规则

- **作用**：ALPHA97 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha97](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97.py#L29) | compute_alpha97(low_wide, vwap_wide, amount_wide, industry_wide, *, low_mix_weight=0.721001, mixed_complement_base=1, mixed_complement_weight=0.721001, indneutralize_mixed_industry_delta_lag=3.3705, delta_indneutralize_mixed_decay_window=20.4523, low_rank_window=7.87871, amount_average_window=60, adv_rank_window=17.255, ts_rank_low_ts_rank_adv_correlation_window=4.97547, corr_rank_window=18.5925, ts_rank_corr_decay_window=15.7152, decay_linear_ts_rank_corr_rank_window=6.71659) | 无返回注解；return: compute_alpha(97, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, low_mix_weight=low_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, indneutralize_mixed_industry_delta_lag=indneutral…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97.py#L69) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97.py#L89) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97.py#L98) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-0385ba97140c"></a>
## betalens-factor/alpha101/ALPHA97/factor_ALPHA97.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97.yaml) · 52 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97.yaml#L7)：`factor_spec:`
- [L36](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97.yaml#L36)：`weight:`
- [L42](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97.yaml#L42)：`run:`

<a id="file-fa8a9be2ecd1"></a>
## betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.py) · 109 行 · 说明来源：文件族规则

- **作用**：ALPHA97 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha97_timing](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.py#L29) | compute_alpha97_timing(low_wide, vwap_wide, amount_wide, industry_wide, *, low_mix_weight=0.721001, mixed_complement_base=1, mixed_complement_weight=0.721001, indneutralize_mixed_industry_delta_lag=3.3705, delta_indneutralize_mixed_decay_window=20.4523, low_rank_window=7.87871, amount_average_window=60, adv_rank_window=17.255, ts_rank_low_ts_rank_adv_correlation_window=4.97547, corr_rank_window=18.5925, ts_rank_corr_decay_window=15.7152, decay_linear_ts_rank_corr_rank_window=6.71659, stock_code=…（完整内容见 inventory.json/源码） | 无返回注解；return: compute_alpha(97, low_wide=low_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, industry_wide=industry_wide, low_mix_weight=low_mix_weight, mixed_complement_base=mixed_complement_base, mixed_complement_weight=mixed_complement_weight, indneutralize_mixed_industry_delta_lag=indneutral…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.py#L72) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.py#L92) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.py#L101) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d8462301a0d5"></a>
## betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.yaml) · 61 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.yaml#L8)：`factor_spec:`
- [L45](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.yaml#L45)：`weight:`
- [L51](../../../betalens-factor/alpha101/ALPHA97/factor_ALPHA97_timing.yaml#L51)：`run:`

<a id="file-d3a3ffc98dc5"></a>
## betalens-factor/alpha101/ALPHA98/factor_ALPHA98.py

[打开源码](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98.py) · 98 行 · 说明来源：文件族规则

- **作用**：ALPHA98 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha98](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98.py#L29) | compute_alpha98(open_wide, vwap_wide, amount_wide, *, amount_average_window=5, adv_sum_window=26.4719, vwap_ts_sum_adv_correlation_window=4.58418, correlation_vwap_ts_sum_decay_window=7.18088, amount_average_window_2=15, rank_open_rank_adv_correlation_window=20.8187, corr_argmin_window=8.62571, ts_argmin_corr_rank_window=6.95668, ts_rank_ts_argmin_corr_decay_window=8.07206) | 无返回注解；return: compute_alpha(98, open_wide=open_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window, vwap_ts_sum_adv_correlation_window=vwap_ts_sum_adv_correlation_window, correlation_vwap_ts_sum_decay_window=correlation_vwap_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98.py#L61) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98.py#L81) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98.py#L90) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-34db958d3244"></a>
## betalens-factor/alpha101/ALPHA98/factor_ALPHA98.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98.yaml) · 48 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98.yaml#L7)：`factor_spec:`
- [L32](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98.yaml#L32)：`weight:`
- [L38](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98.yaml#L38)：`run:`

<a id="file-b652fad34312"></a>
## betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.py) · 101 行 · 说明来源：文件族规则

- **作用**：ALPHA98 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha98_timing](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.py#L29) | compute_alpha98_timing(open_wide, vwap_wide, amount_wide, *, amount_average_window=5, adv_sum_window=26.4719, vwap_ts_sum_adv_correlation_window=4.58418, correlation_vwap_ts_sum_decay_window=7.18088, amount_average_window_2=15, rank_open_rank_adv_correlation_window=20.8187, corr_argmin_window=8.62571, ts_argmin_corr_rank_window=6.95668, ts_rank_ts_argmin_corr_decay_window=8.07206, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(98, open_wide=open_wide, vwap_wide=vwap_wide, amount_wide=amount_wide, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window, vwap_ts_sum_adv_correlation_window=vwap_ts_sum_adv_correlation_window, correlation_vwap_ts_sum_decay_window=correlation_vwap_…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.py#L64) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.py#L84) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.py#L93) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-5b0d72c60117"></a>
## betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.yaml) · 57 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.yaml#L8)：`factor_spec:`
- [L41](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.yaml#L41)：`weight:`
- [L47](../../../betalens-factor/alpha101/ALPHA98/factor_ALPHA98_timing.yaml#L47)：`run:`

<a id="file-e94a52062d43"></a>
## betalens-factor/alpha101/ALPHA99/factor_ALPHA99.py

[打开源码](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99.py) · 94 行 · 说明来源：文件族规则

- **作用**：ALPHA99 cross-sectional factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha99](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99.py#L29) | compute_alpha99(high_wide, low_wide, volume_wide, amount_wide, *, high_low_divisor=2, high_low_sum_window=19.8975, amount_average_window=60, adv_sum_window=19.8975, ts_sum_high_low_ts_sum_adv_correlation_window=8.8136, low_volume_correlation_window=6.28259) | 无返回注解；return: compute_alpha(99, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, amount_wide=amount_wide, high_low_divisor=high_low_divisor, high_low_sum_window=high_low_sum_window, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window, ts_sum_high_low_ts_sum_adv…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99.py#L57) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99.py#L77) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99.py#L86) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-b1ac751e6119"></a>
## betalens-factor/alpha101/ALPHA99/factor_ALPHA99.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99.yaml) · 46 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99.yaml#L7)：`factor_spec:`
- [L30](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99.yaml#L30)：`weight:`
- [L36](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99.yaml#L36)：`run:`

<a id="file-55e511090dae"></a>
## betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.py

[打开源码](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.py) · 97 行 · 说明来源：文件族规则

- **作用**：ALPHA99 single-stock timing strategy.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.py#L25) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_alpha99_timing](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.py#L29) | compute_alpha99_timing(high_wide, low_wide, volume_wide, amount_wide, *, high_low_divisor=2, high_low_sum_window=19.8975, amount_average_window=60, adv_sum_window=19.8975, ts_sum_high_low_ts_sum_adv_correlation_window=8.8136, low_volume_correlation_window=6.28259, stock_code=None, signal_weight=None) | 无返回注解；return: compute_alpha(99, high_wide=high_wide, low_wide=low_wide, volume_wide=volume_wide, amount_wide=amount_wide, high_low_divisor=high_low_divisor, high_low_sum_window=high_low_sum_window, amount_average_window=amount_average_window, adv_sum_window=adv_sum_window, ts_sum_high_low_ts_sum_adv…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.py#L60) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.py#L80) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.py#L89) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-67fd7ae1b536"></a>
## betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.yaml

[打开源码](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.yaml) · 55 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.yaml#L1)：`meta:`
- [L8](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.yaml#L8)：`factor_spec:`
- [L39](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.yaml#L39)：`weight:`
- [L45](../../../betalens-factor/alpha101/ALPHA99/factor_ALPHA99_timing.yaml#L45)：`run:`

