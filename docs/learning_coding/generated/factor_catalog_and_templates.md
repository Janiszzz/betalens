# factor_catalog_and_templates：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-57d1cb3b3ec8"></a>
## betalens-factor/LiqDemand/DISP/factor_DISP.py

[打开源码](../../../betalens-factor/LiqDemand/DISP/factor_DISP.py) · 123 行 · 说明来源：文件族规则

- **作用**：DISP dispensability factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from betalens.factor.mining import MiningSpec
from factor_template_liqdemand import FactorSpec, LiqDemandPipeline, clean_inf, get_pretom_dates
from pathlib import Path
from typing import Mapping, Any
import argparse
import dataclasses
import logging
import pandas as pd
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/LiqDemand/DISP/factor_DISP.py#L43) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_disp](../../../betalens-factor/LiqDemand/DISP/factor_DISP.py#L47) | compute_disp(close_wide, window) | 无返回注解；return: clean_inf(-ratio) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/LiqDemand/DISP/factor_DISP.py#L53) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [_require_param](../../../betalens-factor/LiqDemand/DISP/factor_DISP.py#L65) | _require_param(params: Mapping[str, Any], key: str) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [make_mining_spec](../../../betalens-factor/LiqDemand/DISP/factor_DISP.py#L71) | make_mining_spec(params) | 无返回注解；return: MiningSpec(factor_spec=factor_spec, execution_mode='precomputed', window_transform=_mining_window_transform if bool(_require_param(params, 'pretom_only')) else None, warmup_days=mining_warmup_days) | 无 docstring，需阅读函数体 |
| [_mining_window_transform](../../../betalens-factor/LiqDemand/DISP/factor_DISP.py#L86) | _mining_window_transform(weights, window, context) | 无返回注解；return: weights; weights.loc[keep] | 无 docstring，需阅读函数体 |
| [mining_warmup_days](../../../betalens-factor/LiqDemand/DISP/factor_DISP.py#L101) | mining_warmup_days(params) | 无返回注解；return: int(window * 1.5) + 60 | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/LiqDemand/DISP/factor_DISP.py#L106) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: LiqDemandPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/LiqDemand/DISP/factor_DISP.py#L115) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-a62d52cd1c9e"></a>
## betalens-factor/LiqDemand/DISP/factor_DISP.yaml

[打开源码](../../../betalens-factor/LiqDemand/DISP/factor_DISP.yaml) · 38 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/LiqDemand/DISP/factor_DISP.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/LiqDemand/DISP/factor_DISP.yaml#L7)：`factor_spec:`
- [L19](../../../betalens-factor/LiqDemand/DISP/factor_DISP.yaml#L19)：`weight:`
- [L25](../../../betalens-factor/LiqDemand/DISP/factor_DISP.yaml#L25)：`run:`

<a id="file-a07e32b9e773"></a>
## betalens-factor/LiqDemand/DISP/mining/parameter_space.yaml

[打开源码](../../../betalens-factor/LiqDemand/DISP/mining/parameter_space.yaml) · 96 行 · 说明来源：文件族规则

- **作用**：参数空间、搜索与评价规则
- **输入**：维护者填写的参数
- **输出**：挖掘候选和窗口配置
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L2](../../../betalens-factor/LiqDemand/DISP/mining/parameter_space.yaml#L2)：`version: 1`
- [L4](../../../betalens-factor/LiqDemand/DISP/mining/parameter_space.yaml#L4)：`factor_class: DISP`
- [L7](../../../betalens-factor/LiqDemand/DISP/mining/parameter_space.yaml#L7)：`factors:`
- [L42](../../../betalens-factor/LiqDemand/DISP/mining/parameter_space.yaml#L42)：`evaluation:`
- [L54](../../../betalens-factor/LiqDemand/DISP/mining/parameter_space.yaml#L54)：`windows:`
- [L60](../../../betalens-factor/LiqDemand/DISP/mining/parameter_space.yaml#L60)：`search:`
- [L76](../../../betalens-factor/LiqDemand/DISP/mining/parameter_space.yaml#L76)：`selection:`

<a id="file-0e9c195fdb85"></a>
## betalens-factor/LiqDemand/DISP/mining/performance.yaml

[打开源码](../../../betalens-factor/LiqDemand/DISP/mining/performance.yaml) · 26 行 · 说明来源：文件族规则

- **作用**：挖掘资源、缓存和输出配置
- **输入**：维护者填写的参数
- **输出**：调度与存储参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/LiqDemand/DISP/mining/performance.yaml#L1)：`runtime:`
- [L11](../../../betalens-factor/LiqDemand/DISP/mining/performance.yaml#L11)：`cache:`
- [L15](../../../betalens-factor/LiqDemand/DISP/mining/performance.yaml#L15)：`output:`
- [L20](../../../betalens-factor/LiqDemand/DISP/mining/performance.yaml#L20)：`logging:`

<a id="file-18e6ccaad829"></a>
## betalens-factor/LiqDemand/DISP/mining/run.py

[打开源码](../../../betalens-factor/LiqDemand/DISP/mining/run.py) · 43 行 · 说明来源：人工文件说明

- **作用**：DISP 挖掘命令入口
- **输入**：同目录搜索和性能 YAML/命令参数
- **输出**：通用挖掘结果与输出目录
- **副作用/维护重点**：会运行搜索、查库和写产物；先阅读参数范围

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.factor.mining import run_mining
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [main](../../../betalens-factor/LiqDemand/DISP/mining/run.py#L20) | main() -&gt; int | int | 无 docstring，需阅读函数体 |

<a id="file-6ef60084b9db"></a>
## betalens-factor/LiqDemand/class_LiqDemand.yaml

[打开源码](../../../betalens-factor/LiqDemand/class_LiqDemand.yaml) · 21 行 · 说明来源：文件族规则

- **作用**：因子类别发现元数据
- **输入**：维护者填写的参数
- **输出**：类别信息；不能替代完整因子运行 YAML
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/LiqDemand/class_LiqDemand.yaml#L1)：`class: LiqDemand`
- [L2](../../../betalens-factor/LiqDemand/class_LiqDemand.yaml#L2)：`template_module: factor_template_liqdemand`
- [L3](../../../betalens-factor/LiqDemand/class_LiqDemand.yaml#L3)：`source: Nathan & Suominen (2026), The Liquidity-Demand Component of the Factor Zoo, SSRN 6909918`
- [L4](../../../betalens-factor/LiqDemand/class_LiqDemand.yaml#L4)：`defaults:`

<a id="file-839bf2e7bc21"></a>
## betalens-factor/LiqDemand/factor_template_liqdemand.py

[打开源码](../../../betalens-factor/LiqDemand/factor_template_liqdemand.py) · 287 行 · 说明来源：人工文件说明

- **作用**：流动性需求因子公共算子和管线
- **输入**：行情与类别参数
- **输出**：指标与运行结果
- **副作用/维护重点**：与 DISP 和类级默认配置一起阅读

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.analyst import Analyst
from betalens.backtest import BacktestBase
from betalens.datafeed import get_absolute_trade_days
from betalens.factor.factor import single_characteristic, get_single_factor_weight
from collections import defaultdict
from datetime import date, timedelta
from factor_template import FactorSpec, FactorPipeline, RunResult, fetch_daily_wide, wide_to_prequery, build_pit_universe, filter_long_by_pit_universe, infer_warmup_days, validate_weights_in_pit_universe, _labeled_to_factor_values, _expand_weights_to_factor_universe, append_grouped_profiling_excel
from pathlib import Path
import numpy as np
import pandas as pd
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [clean_inf](../../../betalens-factor/LiqDemand/factor_template_liqdemand.py#L80) | clean_inf(x) | 无返回注解；return: x.replace([np.inf, -np.inf], np.nan) | 把 ±inf 置为 NaN（算子末尾统一调用）。 |
| [get_pretom_dates](../../../betalens-factor/LiqDemand/factor_template_liqdemand.py#L85) | get_pretom_dates(start_date: str, end_date: str, lo: int=9, hi: int=4) -&gt; set[date] | set[date] | 每月 PreTOM 窗口 [τ−lo, τ−hi] 的交易日集合（论文 §2.2，默认倒数第 9~第 4，共 6 天）。 τ 为当月最后一个交易日。按 (年,月) 分组后，取该月交易日序列的 'month_days[-lo : -(hi-1) or None]'： -lo = 倒数第 lo 个（含） -(hi-1) = 倒数第 hi 个的下一个（不含），hi=4 时为 -3 'or None' = 当 hi==1 时 -(0)==0 会取空，需退化为 None 取到末尾 返回 set[date]，供权重表掩码使用（窗口内保留、窗口外清零）。 |
| [LiqDemandPipeline](../../../betalens-factor/LiqDemand/factor_template_liqdemand.py#L109) | class LiqDemandPipeline(FactorPipeline) | 类定义；构造/属性见方法与字段 | LiqDemand 类管线：在通用 FactorPipeline 主干上扩展 warmup 预热 + PreTOM 择时。 主干（取数→算子→profiling→中性化→分组→权重→回测→报告）逐字沿用通用实现， 仅插入两处本类逻辑，便于与 alpha101/tdx 口径对齐： 1. 取数区间向前扩 warmup_days 天，使 rolling(window) 在回测首日已就绪； 2. pretom_only=True 时，把非 PreTOM 交易日的权重整行清零（空仓）。 |
| [LiqDemandPipeline.run](../../../betalens-factor/LiqDemand/factor_template_liqdemand.py#L118) | run(self, start_date: str, end_date: str, *, rebal_freq: str='D', grouping_mode: str='equal_count', warmup_days: int=400, pretom_only: bool=True, pretom_lo: int=9, pretom_hi: int=4, universe: list &#124; None=None, n_quantiles: int=20, initial_amount: float=100000000.0, benchmark_code: str &#124; None=None, output_dir: str='.', extra_inputs: dict[str, pd.DataFrame] &#124; None=None, include_profiling: bool=True, dump_excel: bool=True, verbose: bool=True) -&gt; RunResult | RunResult | 无 docstring，需阅读函数体 |

<a id="file-79dca30ba712"></a>
## betalens-factor/alpha101/alpha101_formulas.py

[打开源码](../../../betalens-factor/alpha101/alpha101_formulas.py) · 939 行 · 说明来源：人工文件说明

- **作用**：101 个公式、依赖与回看需求注册
- **输入**：alpha 编号、宽表、参数
- **输出**：因子宽表/定义/历史需求
- **副作用/维护重点**：公式真源；核对 rank 轴、运算单位与累计回看

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from dataclasses import dataclass
from factor_template_alpha101 import clean_inf, correlation, covariance, decay_linear, delay, delta, elementwise_max, elementwise_min, indneutralize, product, rank, scale, sign, signed_power, stddev, ts_argmax, ts_argmin, ts_max, ts_mean, ts_min, ts_rank, ts_sum, where
from typing import Any, Callable, Mapping
import numpy as np
import pandas as pd
import re
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [AlphaDefinition](../../../betalens-factor/alpha101/alpha101_formulas.py#L177) | class AlphaDefinition() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：number: int; name: str; formula: str; inputs: Mapping[str, str]; industry_inputs: Mapping[str, str]; required_history_bars: int; compute: Callable[[dict[str, pd.DataFrame], Mapping[str, Any] &#124; None], pd.DataFrame]; parameters: Mapping[str, 'AlphaParameter'] |
| [AlphaParameter](../../../betalens-factor/alpha101/alpha101_formulas.py#L189) | class AlphaParameter() | 类定义；构造/属性见方法与字段 | One numeric formula literal exposed for controlled parameter mining.；字段：name: str; default: int &#124; float; kind: str; searchable: bool; source_line: int |
| [_adv](../../../betalens-factor/alpha101/alpha101_formulas.py#L199) | _adv(data: dict[str, pd.DataFrame], n: float) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_bool](../../../betalens-factor/alpha101/alpha101_formulas.py#L203) | _bool(frame: pd.DataFrame) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [alpha1](../../../betalens-factor/alpha101/alpha101_formulas.py#L207) | alpha1(d, *, returns_threshold=0, returns_stddev_window=20, base_power_exponent=2.0, signed_power_base_argmax_window=5, rank_ts_argmax_signed_power_center=0.5) | 无返回注解；return: rank(ts_argmax(signed_power(base, base_power_exponent), signed_power_base_argmax_window)) - rank_ts_argmax_signed_power_center | 无 docstring，需阅读函数体 |
| [alpha2](../../../betalens-factor/alpha101/alpha101_formulas.py#L212) | alpha2(d, *, volume_delta_lag=2, rank_delta_volume_rank_open_close_correlation_window=6) | 无返回注解；return: -correlation(rank(delta(np.log(d['volume']), volume_delta_lag)), rank((d['close'] - d['open']) / d['open']), rank_delta_volume_rank_open_close_correlation_window) | 无 docstring，需阅读函数体 |
| [alpha3](../../../betalens-factor/alpha101/alpha101_formulas.py#L216) | alpha3(d, *, rank_open_rank_volume_correlation_window=10) | 无返回注解；return: -correlation(rank(d['open']), rank(d['volume']), rank_open_rank_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [alpha4](../../../betalens-factor/alpha101/alpha101_formulas.py#L220) | alpha4(d, *, rank_low_rank_window=9) | 无返回注解；return: -ts_rank(rank(d['low']), rank_low_rank_window) | 无 docstring，需阅读函数体 |
| [alpha5](../../../betalens-factor/alpha101/alpha101_formulas.py#L224) | alpha5(d, *, vwap_sum_window=10, ts_sum_vwap_divisor=10) | 无返回注解；return: rank(d['open'] - ts_sum(d['vwap'], vwap_sum_window) / ts_sum_vwap_divisor) * -rank(d['close'] - d['vwap']).abs() | 无 docstring，需阅读函数体 |
| [alpha6](../../../betalens-factor/alpha101/alpha101_formulas.py#L228) | alpha6(d, *, open_volume_correlation_window=10) | 无返回注解；return: -correlation(d['open'], d['volume'], open_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [alpha7](../../../betalens-factor/alpha101/alpha101_formulas.py#L232) | alpha7(d, *, close_delta_lag=7, change_rank_window=60, amount_average_window=20, volume_adv_false_value=-1.0) | 无返回注解；return: where(_adv(d, amount_average_window) &lt; d['volume'], signal, volume_adv_false_value) | 无 docstring，需阅读函数体 |
| [alpha8](../../../betalens-factor/alpha101/alpha101_formulas.py#L238) | alpha8(d, *, open_sum_window=5, returns_sum_window=5, base_delay_lag=10) | 无返回注解；return: -rank(base - delay(base, base_delay_lag)) | 无 docstring，需阅读函数体 |
| [alpha9](../../../betalens-factor/alpha101/alpha101_formulas.py#L243) | alpha9(d, *, close_delta_lag=1, change_minimum_window=5, ts_min_change_threshold=0, change_maximum_window=5, ts_max_change_threshold=0) | 无返回注解；return: where(trend, change, -change) | 无 docstring，需阅读函数体 |
| [alpha10](../../../betalens-factor/alpha101/alpha101_formulas.py#L249) | alpha10(d, *, close_delta_lag=1, change_minimum_window=4, ts_min_change_threshold=0, change_maximum_window=4, ts_max_change_threshold=0) | 无返回注解；return: rank(where(trend, change, -change)) | 无 docstring，需阅读函数体 |
| [alpha11](../../../betalens-factor/alpha101/alpha101_formulas.py#L255) | alpha11(d, *, spread_maximum_window=3, spread_minimum_window=3, volume_delta_lag=3) | 无返回注解；return: (rank(ts_max(spread, spread_maximum_window)) + rank(ts_min(spread, spread_minimum_window))) * rank(delta(d['volume'], volume_delta_lag)) | 无 docstring，需阅读函数体 |
| [alpha12](../../../betalens-factor/alpha101/alpha101_formulas.py#L260) | alpha12(d, *, volume_delta_lag=1, close_delta_lag=1) | 无返回注解；return: sign(delta(d['volume'], volume_delta_lag)) * -delta(d['close'], close_delta_lag) | 无 docstring，需阅读函数体 |
| [alpha13](../../../betalens-factor/alpha101/alpha101_formulas.py#L264) | alpha13(d, *, rank_close_rank_volume_covariance_window=5) | 无返回注解；return: -rank(covariance(rank(d['close']), rank(d['volume']), rank_close_rank_volume_covariance_window)) | 无 docstring，需阅读函数体 |
| [alpha14](../../../betalens-factor/alpha101/alpha101_formulas.py#L268) | alpha14(d, *, returns_delta_lag=3, open_volume_correlation_window=10) | 无返回注解；return: -rank(delta(d['returns'], returns_delta_lag)) * correlation(d['open'], d['volume'], open_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [alpha15](../../../betalens-factor/alpha101/alpha101_formulas.py#L272) | alpha15(d, *, rank_high_rank_volume_correlation_window=3, rank_correlation_high_sum_window=3) | 无返回注解；return: -ts_sum(rank(correlation(rank(d['high']), rank(d['volume']), rank_high_rank_volume_correlation_window)), rank_correlation_high_sum_window) | 无 docstring，需阅读函数体 |
| [alpha16](../../../betalens-factor/alpha101/alpha101_formulas.py#L276) | alpha16(d, *, rank_high_rank_volume_covariance_window=5) | 无返回注解；return: -rank(covariance(rank(d['high']), rank(d['volume']), rank_high_rank_volume_covariance_window)) | 无 docstring，需阅读函数体 |
| [alpha17](../../../betalens-factor/alpha101/alpha101_formulas.py#L280) | alpha17(d, *, close_rank_window=10, close_delta_lag=1, delta_close_delta_lag=1, amount_average_window=20, volume_adv_rank_window=5) | 无返回注解；return: -rank(ts_rank(d['close'], close_rank_window)) * rank(delta(delta(d['close'], close_delta_lag), delta_close_delta_lag)) * rank(ts_rank(d['volume'] / _adv(d, amount_average_window), volume_adv_rank_window)) | 无 docstring，需阅读函数体 |
| [alpha18](../../../betalens-factor/alpha101/alpha101_formulas.py#L284) | alpha18(d, *, close_open_stddev_window=5, close_open_correlation_window=10) | 无返回注解；return: -rank(stddev((d['close'] - d['open']).abs(), close_open_stddev_window) + d['close'] - d['open'] + correlation(d['close'], d['open'], close_open_correlation_window)) | 无 docstring，需阅读函数体 |
| [alpha19](../../../betalens-factor/alpha101/alpha101_formulas.py#L288) | alpha19(d, *, close_delay_lag=7, close_delta_lag=7, rank_ts_sum_returns_offset=1, ts_sum_returns_offset=1, returns_sum_window=250) | 无返回注解；return: direction * (rank_ts_sum_returns_offset + rank(ts_sum_returns_offset + ts_sum(d['returns'], returns_sum_window))) | 无 docstring，需阅读函数体 |
| [alpha20](../../../betalens-factor/alpha101/alpha101_formulas.py#L293) | alpha20(d, *, high_delay_lag=1, close_delay_lag=1, low_delay_lag=1) | 无返回注解；return: -rank(d['open'] - delay(d['high'], high_delay_lag)) * rank(d['open'] - delay(d['close'], close_delay_lag)) * rank(d['open'] - delay(d['low'], low_delay_lag)) | 无 docstring，需阅读函数体 |
| [alpha21](../../../betalens-factor/alpha101/alpha101_formulas.py#L297) | alpha21(d, *, close_mean_window=8, close_mean_window_2=2, close_stddev_window=8, close_stddev_window_2=8, amount_average_window=20, volume_adv_threshold=1, condition1_true_value=-1.0, condition2_true_value=1.0, volume_condition_true_value=1.0, volume_condition_false_value=-1.0) | 无返回注解；return: where(condition1, condition1_true_value, where(condition2, condition2_true_value, where(volume_condition, volume_condition_true_value, volume_condition_false_value))) | 无 docstring，需阅读函数体 |
| [alpha22](../../../betalens-factor/alpha101/alpha101_formulas.py#L306) | alpha22(d, *, high_volume_correlation_window=5, correlation_high_volume_delta_lag=5, close_stddev_window=20) | 无返回注解；return: -delta(correlation(d['high'], d['volume'], high_volume_correlation_window), correlation_high_volume_delta_lag) * rank(stddev(d['close'], close_stddev_window)) | 无 docstring，需阅读函数体 |
| [alpha23](../../../betalens-factor/alpha101/alpha101_formulas.py#L310) | alpha23(d, *, high_mean_window=20, high_delta_lag=2, high_ts_mean_false_value=0.0) | 无返回注解；return: where(ts_mean(d['high'], high_mean_window) &lt; d['high'], -delta(d['high'], high_delta_lag), high_ts_mean_false_value) | 无 docstring，需阅读函数体 |
| [alpha24](../../../betalens-factor/alpha101/alpha101_formulas.py#L314) | alpha24(d, *, close_mean_window=100, ts_mean_close_delta_lag=100, close_delay_lag=100, trend_threshold=0.05, close_minimum_window=100, close_delta_lag=3) | 无返回注解；return: where(trend &lt;= trend_threshold, -(d['close'] - ts_min(d['close'], close_minimum_window)), -delta(d['close'], close_delta_lag)) | 无 docstring，需阅读函数体 |
| [alpha25](../../../betalens-factor/alpha101/alpha101_formulas.py#L319) | alpha25(d, *, amount_average_window=20) | 无返回注解；return: rank(-d['returns'] * _adv(d, amount_average_window) * d['vwap'] * (d['high'] - d['close'])) | 无 docstring，需阅读函数体 |
| [alpha26](../../../betalens-factor/alpha101/alpha101_formulas.py#L323) | alpha26(d, *, volume_rank_window=5, high_rank_window=5, ts_rank_volume_ts_rank_high_correlation_window=5, correlation_ts_rank_volume_maximum_window=3) | 无返回注解；return: -ts_max(correlation(ts_rank(d['volume'], volume_rank_window), ts_rank(d['high'], high_rank_window), ts_rank_volume_ts_rank_high_correlation_window), correlation_ts_rank_volume_maximum_window) | 无 docstring，需阅读函数体 |
| [alpha27](../../../betalens-factor/alpha101/alpha101_formulas.py#L327) | alpha27(d, *, rank_volume_rank_vwap_correlation_window=6, correlation_rank_volume_sum_window=2, ts_sum_correlation_rank_divisor=2.0, value_threshold=0.5, value_true_value=-1.0, value_false_value=1.0) | 无返回注解；return: where(value &gt; value_threshold, value_true_value, value_false_value) | 无 docstring，需阅读函数体 |
| [alpha28](../../../betalens-factor/alpha101/alpha101_formulas.py#L332) | alpha28(d, *, amount_average_window=20, adv_low_correlation_window=5, high_low_divisor=2) | 无返回注解；return: scale(correlation(_adv(d, amount_average_window), d['low'], adv_low_correlation_window) + (d['high'] + d['low']) / high_low_divisor - d['close']) | 无 docstring，需阅读函数体 |
| [alpha29](../../../betalens-factor/alpha101/alpha101_formulas.py#L336) | alpha29(d, *, close_center=1, close_delta_lag=5, rank_inner_minimum_window=2, ts_min_rank_inner_sum_window=1, rank_scale_ts_sum_product_window=1, product_rank_scale_minimum_window=5, returns_delay_lag=6, delay_returns_rank_window=5) | 无返回注解；return: left + ts_rank(delay(-d['returns'], returns_delay_lag), delay_returns_rank_window) | 无 docstring，需阅读函数体 |
| [alpha30](../../../betalens-factor/alpha101/alpha101_formulas.py#L342) | alpha30(d, *, close_delay_lag=1, close_delay_lag_2=1, close_delay_lag_3=2, close_delay_lag_4=2, close_delay_lag_5=3, rank_direction_complement_base=1, volume_sum_window=5, volume_sum_window_2=20) | 无返回注解；return: (rank_direction_complement_base - rank(direction)) * ts_sum(d['volume'], volume_sum_window) / ts_sum(d['volume'], volume_sum_window_2) | 无 docstring，需阅读函数体 |
| [alpha31](../../../betalens-factor/alpha101/alpha101_formulas.py#L347) | alpha31(d, *, close_delta_lag=10, rank_delta_close_decay_window=10, close_delta_lag_2=3, amount_average_window=20, adv_low_correlation_window=12) | 无返回注解；return: rank(rank(rank(decay_linear(-rank(rank(delta(d['close'], close_delta_lag))), rank_delta_close_decay_window)))) + rank(-delta(d['close'], close_delta_lag_2)) + sign(scale(correlation(_adv(d, amount_average_window), d['low'], adv_low_correlation_window))) | 无 docstring，需阅读函数体 |
| [alpha32](../../../betalens-factor/alpha101/alpha101_formulas.py#L351) | alpha32(d, *, close_mean_window=7, scale_correlation_vwap_coefficient=20, close_delay_lag=5, vwap_delay_close_correlation_window=230) | 无返回注解；return: scale(ts_mean(d['close'], close_mean_window) - d['close']) + scale_correlation_vwap_coefficient * scale(correlation(d['vwap'], delay(d['close'], close_delay_lag), vwap_delay_close_correlation_window)) | 无 docstring，需阅读函数体 |
| [alpha33](../../../betalens-factor/alpha101/alpha101_formulas.py#L355) | alpha33(d, *, open_close_coefficient=-1, open_close_complement_base=1) | 无返回注解；return: rank(open_close_coefficient * (open_close_complement_base - d['open'] / d['close'])) | 无 docstring，需阅读函数体 |
| [alpha34](../../../betalens-factor/alpha101/alpha101_formulas.py#L359) | alpha34(d, *, rank_stddev_returns_complement_base=1, returns_stddev_window=2, returns_stddev_window_2=5, rank_delta_close_complement_base=1, close_delta_lag=1) | 无返回注解；return: rank(rank_stddev_returns_complement_base - rank(stddev(d['returns'], returns_stddev_window) / stddev(d['returns'], returns_stddev_window_2)) + (rank_delta_close_complement_base - rank(delta(d['close'], close_delta_lag)))) | 无 docstring，需阅读函数体 |
| [alpha35](../../../betalens-factor/alpha101/alpha101_formulas.py#L363) | alpha35(d, *, volume_rank_window=32, ts_rank_low_close_complement_base=1, low_close_high_rank_window=16, ts_rank_returns_complement_base=1, returns_rank_window=32) | 无返回注解；return: ts_rank(d['volume'], volume_rank_window) * (ts_rank_low_close_complement_base - ts_rank(d['close'] + d['high'] - d['low'], low_close_high_rank_window)) * (ts_rank_returns_complement_base - ts_rank(d['returns'], returns_rank_window)) | 无 docstring，需阅读函数体 |
| [alpha36](../../../betalens-factor/alpha101/alpha101_formulas.py#L367) | alpha36(d, *, rank_correlation_close_coefficient=2.21, volume_delay_lag=1, close_open_delay_volume_correlation_window=15, rank_open_close_coefficient=0.7, rank_ts_rank_delay_coefficient=0.73, returns_delay_lag=6, delay_returns_rank_window=5, amount_average_window=20, vwap_adv_correlation_window=6, rank_open_close_coefficient_2=0.6, close_mean_window=200) | 无返回注解；return: rank_correlation_close_coefficient * rank(correlation(d['close'] - d['open'], delay(d['volume'], volume_delay_lag), close_open_delay_volume_correlation_window)) + rank_open_close_coefficient * rank(d['open'] - d['close']) + rank_ts_rank_delay_coefficient * rank(ts_rank(delay(-d['return…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [alpha37](../../../betalens-factor/alpha101/alpha101_formulas.py#L371) | alpha37(d, *, open_close_delay_lag=1, delay_open_close_close_correlation_window=200) | 无返回注解；return: rank(correlation(delay(d['open'] - d['close'], open_close_delay_lag), d['close'], delay_open_close_close_correlation_window)) + rank(d['open'] - d['close']) | 无 docstring，需阅读函数体 |
| [alpha38](../../../betalens-factor/alpha101/alpha101_formulas.py#L375) | alpha38(d, *, close_rank_window=10) | 无返回注解；return: -rank(ts_rank(d['close'], close_rank_window)) * rank(d['close'] / d['open']) | 无 docstring，需阅读函数体 |
| [alpha39](../../../betalens-factor/alpha101/alpha101_formulas.py#L379) | alpha39(d, *, close_delta_lag=7, rank_decay_linear_volume_complement_base=1, amount_average_window=20, volume_adv_decay_window=9, rank_ts_sum_returns_offset=1, returns_sum_window=250) | 无返回注解；return: -rank(delta(d['close'], close_delta_lag) * (rank_decay_linear_volume_complement_base - rank(decay_linear(d['volume'] / _adv(d, amount_average_window), volume_adv_decay_window)))) * (rank_ts_sum_returns_offset + rank(ts_sum(d['returns'], returns_sum_window))) | 无 docstring，需阅读函数体 |
| [alpha40](../../../betalens-factor/alpha101/alpha101_formulas.py#L383) | alpha40(d, *, high_stddev_window=10, high_volume_correlation_window=10) | 无返回注解；return: -rank(stddev(d['high'], high_stddev_window)) * correlation(d['high'], d['volume'], high_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [alpha41](../../../betalens-factor/alpha101/alpha101_formulas.py#L387) | alpha41(d) | 无返回注解；return: np.sqrt(d['high'] * d['low']) - d['vwap'] | 无 docstring，需阅读函数体 |
| [alpha42](../../../betalens-factor/alpha101/alpha101_formulas.py#L391) | alpha42(d) | 无返回注解；return: rank(d['vwap'] - d['close']) / rank(d['vwap'] + d['close']) | 无 docstring，需阅读函数体 |
| [alpha43](../../../betalens-factor/alpha101/alpha101_formulas.py#L395) | alpha43(d, *, amount_average_window=20, volume_adv_rank_window=20, close_delta_lag=7, delta_close_rank_window=8) | 无返回注解；return: ts_rank(d['volume'] / _adv(d, amount_average_window), volume_adv_rank_window) * ts_rank(-delta(d['close'], close_delta_lag), delta_close_rank_window) | 无 docstring，需阅读函数体 |
| [alpha44](../../../betalens-factor/alpha101/alpha101_formulas.py#L399) | alpha44(d, *, high_rank_volume_correlation_window=5) | 无返回注解；return: -correlation(d['high'], rank(d['volume']), high_rank_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [alpha45](../../../betalens-factor/alpha101/alpha101_formulas.py#L403) | alpha45(d, *, close_delay_lag=5, delay_close_sum_window=20, ts_sum_delay_close_divisor=20, close_volume_correlation_window=2, close_sum_window=5, close_sum_window_2=20, ts_sum_close_ts_sum_close_correlation_window=2) | 无返回注解；return: -rank(ts_sum(delay(d['close'], close_delay_lag), delay_close_sum_window) / ts_sum_delay_close_divisor) * correlation(d['close'], d['volume'], close_volume_correlation_window) * rank(correlation(ts_sum(d['close'], close_sum_window), ts_sum(d['close'], close_sum_window_2), ts_sum_close_t…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [_alpha_trend](../../../betalens-factor/alpha101/alpha101_formulas.py#L407) | _alpha_trend(d, *, long_delay_lag=20, first_short_delay_lag=10, first_slope_divisor=10, second_short_delay_lag=10, second_slope_divisor=10) | 无返回注解；return: (delay(d['close'], long_delay_lag) - delay(d['close'], first_short_delay_lag)) / first_slope_divisor - (delay(d['close'], second_short_delay_lag) - d['close']) / second_slope_divisor | 无 docstring，需阅读函数体 |
| [alpha46](../../../betalens-factor/alpha101/alpha101_formulas.py#L424) | alpha46(d, *, trend_long_delay_lag=20, trend_first_short_delay_lag=10, trend_first_slope_divisor=10, trend_second_short_delay_lag=10, trend_second_slope_divisor=10, trend_threshold=0.25, trend_true_value=-1.0, trend_threshold_2=0, trend_true_value_2=1.0, close_delay_lag=1) | 无返回注解；return: where(trend &gt; trend_threshold, trend_true_value, where(trend &lt; trend_threshold_2, trend_true_value_2, -(d['close'] - delay(d['close'], close_delay_lag)))) | 无 docstring，需阅读函数体 |
| [alpha47](../../../betalens-factor/alpha101/alpha101_formulas.py#L429) | alpha47(d, *, close_divisor=1, amount_average_window=20, high_mean_window=5, vwap_delay_lag=5) | 无返回注解；return: rank(close_divisor / d['close']) * d['volume'] / _adv(d, amount_average_window) * d['high'] * rank(d['high'] - d['close']) / ts_mean(d['high'], high_mean_window) - rank(d['vwap'] - delay(d['vwap'], vwap_delay_lag)) | 无 docstring，需阅读函数体 |
| [alpha48](../../../betalens-factor/alpha101/alpha101_formulas.py#L433) | alpha48(d, *, close_delta_lag=1, close_delay_lag=1, delay_close_delta_lag=1, delta_close_delta_delay_close_correlation_window=250, close_delta_lag_2=1, close_delta_lag_3=1, close_delay_lag_2=1, delta_close_delay_power_exponent=2, delta_close_delay_sum_window=250) | 无返回注解；return: numerator / denominator | 无 docstring，需阅读函数体 |
| [alpha49](../../../betalens-factor/alpha101/alpha101_formulas.py#L440) | alpha49(d, *, trend_long_delay_lag=20, trend_first_short_delay_lag=10, trend_first_slope_divisor=10, trend_second_short_delay_lag=10, trend_second_slope_divisor=10, alpha_trend_threshold=-0.1, alpha_trend_true_value=1.0, close_delay_lag=1) | 无返回注解；return: where(_alpha_trend(d, long_delay_lag=trend_long_delay_lag, first_short_delay_lag=trend_first_short_delay_lag, first_slope_divisor=trend_first_slope_divisor, second_short_delay_lag=trend_second_short_delay_lag, second_slope_divisor=trend_second_slope_divisor) &lt; alpha_trend_threshold, al…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [alpha50](../../../betalens-factor/alpha101/alpha101_formulas.py#L444) | alpha50(d, *, rank_volume_rank_vwap_correlation_window=5, rank_correlation_volume_maximum_window=5) | 无返回注解；return: -ts_max(rank(correlation(rank(d['volume']), rank(d['vwap']), rank_volume_rank_vwap_correlation_window)), rank_correlation_volume_maximum_window) | 无 docstring，需阅读函数体 |
| [alpha51](../../../betalens-factor/alpha101/alpha101_formulas.py#L448) | alpha51(d, *, trend_long_delay_lag=20, trend_first_short_delay_lag=10, trend_first_slope_divisor=10, trend_second_short_delay_lag=10, trend_second_slope_divisor=10, alpha_trend_threshold=-0.05, alpha_trend_true_value=1.0, close_delay_lag=1) | 无返回注解；return: where(_alpha_trend(d, long_delay_lag=trend_long_delay_lag, first_short_delay_lag=trend_first_short_delay_lag, first_slope_divisor=trend_first_slope_divisor, second_short_delay_lag=trend_second_short_delay_lag, second_slope_divisor=trend_second_slope_divisor) &lt; alpha_trend_threshold, al…（完整内容见 inventory.json/源码） | 无 docstring，需阅读函数体 |
| [alpha52](../../../betalens-factor/alpha101/alpha101_formulas.py#L452) | alpha52(d, *, low_minimum_window=5, low_minimum_window_2=5, ts_min_low_delay_lag=5, returns_sum_window=240, returns_sum_window_2=20, ts_sum_returns_divisor=220, volume_rank_window=5) | 无返回注解；return: (-ts_min(d['low'], low_minimum_window) + delay(ts_min(d['low'], low_minimum_window_2), ts_min_low_delay_lag)) * rank((ts_sum(d['returns'], returns_sum_window) - ts_sum(d['returns'], returns_sum_window_2)) / ts_sum_returns_divisor) * ts_rank(d['volume'], volume_rank_window) | 无 docstring，需阅读函数体 |
| [alpha53](../../../betalens-factor/alpha101/alpha101_formulas.py#L456) | alpha53(d, *, oscillator_delta_lag=9) | 无返回注解；return: -delta(oscillator, oscillator_delta_lag) | 无 docstring，需阅读函数体 |
| [alpha54](../../../betalens-factor/alpha101/alpha101_formulas.py#L461) | alpha54(d, *, open_power_exponent=5, close_power_exponent=5) | 无返回注解；return: -(d['low'] - d['close']) * d['open'] ** open_power_exponent / ((d['low'] - d['high']) * d['close'] ** close_power_exponent) | 无 docstring，需阅读函数体 |
| [alpha55](../../../betalens-factor/alpha101/alpha101_formulas.py#L465) | alpha55(d, *, low_minimum_window=12, high_maximum_window=12, low_minimum_window_2=12, rank_stochastic_rank_volume_correlation_window=6) | 无返回注解；return: -correlation(rank(stochastic), rank(d['volume']), rank_stochastic_rank_volume_correlation_window) | 无 docstring，需阅读函数体 |
| [alpha56](../../../betalens-factor/alpha101/alpha101_formulas.py#L470) | alpha56(d, *, returns_sum_window=10, returns_sum_window_2=2, ts_sum_returns_sum_window=3) | 无返回注解；return: -rank(ts_sum(d['returns'], returns_sum_window) / ts_sum(ts_sum(d['returns'], returns_sum_window_2), ts_sum_returns_sum_window)) * rank(d['returns'] * d['cap']) | 无 docstring，需阅读函数体 |
| [alpha57](../../../betalens-factor/alpha101/alpha101_formulas.py#L474) | alpha57(d, *, close_argmax_window=30, rank_ts_argmax_close_decay_window=2) | 无返回注解；return: -(d['close'] - d['vwap']) / decay_linear(rank(ts_argmax(d['close'], close_argmax_window)), rank_ts_argmax_close_decay_window) | 无 docstring，需阅读函数体 |
| [alpha58](../../../betalens-factor/alpha101/alpha101_formulas.py#L478) | alpha58(d, *, indneutralize_vwap_sector_volume_correlation_window=3.92795, value_decay_window=7.89291, decay_linear_value_rank_window=5.50322) | 无返回注解；return: -ts_rank(decay_linear(value, value_decay_window), decay_linear_value_rank_window) | 无 docstring，需阅读函数体 |
| [alpha59](../../../betalens-factor/alpha101/alpha101_formulas.py#L483) | alpha59(d, *, vwap_mix_weight=0.728317, mixed_complement_base=1, mixed_complement_weight=0.728317, indneutralize_mixed_industry_volume_correlation_window=4.25197, value_decay_window=16.2289, decay_linear_value_rank_window=8.19648) | 无返回注解；return: -ts_rank(decay_linear(value, value_decay_window), decay_linear_value_rank_window) | 无 docstring，需阅读函数体 |
| [alpha60](../../../betalens-factor/alpha101/alpha101_formulas.py#L489) | alpha60(d, *, scale_rank_oscillator_coefficient=2, close_argmax_window=10) | 无返回注解；return: -(scale_rank_oscillator_coefficient * scale(rank(oscillator)) - scale(rank(ts_argmax(d['close'], close_argmax_window)))) | 无 docstring，需阅读函数体 |
| [alpha61](../../../betalens-factor/alpha101/alpha101_formulas.py#L494) | alpha61(d, *, vwap_minimum_window=16.1219, amount_average_window=180, vwap_adv_correlation_window=17.9282) | 无返回注解；return: _bool(rank(d['vwap'] - ts_min(d['vwap'], vwap_minimum_window)) &lt; rank(correlation(d['vwap'], _adv(d, amount_average_window), vwap_adv_correlation_window))) | 无 docstring，需阅读函数体 |
| [alpha62](../../../betalens-factor/alpha101/alpha101_formulas.py#L498) | alpha62(d, *, amount_average_window=20, adv_sum_window=22.4101, vwap_ts_sum_adv_correlation_window=9.91009, high_low_divisor=2) | 无返回注解；return: -_bool(left &lt; rank(_bool(inner))) | 无 docstring，需阅读函数体 |
| [alpha63](../../../betalens-factor/alpha101/alpha101_formulas.py#L504) | alpha63(d, *, indneutralize_close_industry_delta_lag=2.25164, delta_indneutralize_close_decay_window=8.22237, vwap_mix_weight=0.318108, mixed_complement_base=1, mixed_complement_weight=0.318108, amount_average_window=180, adv_sum_window=37.2467, mixed_ts_sum_adv_correlation_window=13.557, correlation_mixed_ts_sum_decay_window=12.2883) | 无返回注解；return: -(left - right) | 无 docstring，需阅读函数体 |
| [alpha64](../../../betalens-factor/alpha101/alpha101_formulas.py#L511) | alpha64(d, *, open_mix_weight=0.178404, mixed1_complement_base=1, mixed1_complement_weight=0.178404, mixed1_sum_window=12.7054, amount_average_window=120, adv_sum_window=12.7054, ts_sum_mixed1_ts_sum_adv_correlation_window=16.6208, high_low_divisor=2, high_low_mix_weight=0.178404, mixed2_complement_base=1, mixed2_complement_weight=0.178404, mixed2_delta_lag=3.69741) | 无返回注解；return: -_bool(left &lt; rank(delta(mixed2, mixed2_delta_lag))) | 无 docstring，需阅读函数体 |
| [alpha65](../../../betalens-factor/alpha101/alpha101_formulas.py#L518) | alpha65(d, *, open_mix_weight=0.00817205, mixed_complement_base=1, mixed_complement_weight=0.00817205, amount_average_window=60, adv_sum_window=8.6911, mixed_ts_sum_adv_correlation_window=6.40374, open_minimum_window=13.635) | 无返回注解；return: -_bool(left &lt; rank(d['open'] - ts_min(d['open'], open_minimum_window))) | 无 docstring，需阅读函数体 |
| [alpha66](../../../betalens-factor/alpha101/alpha101_formulas.py#L524) | alpha66(d, *, vwap_delta_lag=3.51013, delta_vwap_decay_window=7.23052, high_low_divisor=2, right_base_decay_window=11.4157, decay_linear_right_base_rank_window=6.72611) | 无返回注解；return: -(left + right) | 无 docstring，需阅读函数体 |
| [alpha67](../../../betalens-factor/alpha101/alpha101_formulas.py#L531) | alpha67(d, *, high_minimum_window=2.14593, amount_average_window=20, indneutralize_vwap_sector_indneutralize_subindustry_adv_correlation_wind=6.02936) | 无返回注解；return: -np.power(base, exponent) | 无 docstring，需阅读函数体 |
| [alpha68](../../../betalens-factor/alpha101/alpha101_formulas.py#L537) | alpha68(d, *, amount_average_window=15, rank_high_rank_adv_correlation_window=8.91644, correlation_rank_high_rank_window=13.9333, close_mix_weight=0.518371, mixed_complement_base=1, mixed_complement_weight=0.518371, mixed_delta_lag=1.06157) | 无返回注解；return: -_bool(left &lt; rank(delta(mixed, mixed_delta_lag))) | 无 docstring，需阅读函数体 |
| [alpha69](../../../betalens-factor/alpha101/alpha101_formulas.py#L543) | alpha69(d, *, indneutralize_vwap_industry_delta_lag=2.72412, delta_indneutralize_vwap_maximum_window=4.79344, close_mix_weight=0.490655, mixed_complement_base=1, mixed_complement_weight=0.490655, amount_average_window=20, mixed_adv_correlation_window=4.92416, correlation_mixed_adv_rank_window=9.0615) | 无返回注解；return: -np.power(base, exponent) | 无 docstring，需阅读函数体 |
| [alpha70](../../../betalens-factor/alpha101/alpha101_formulas.py#L550) | alpha70(d, *, vwap_delta_lag=1.29456, amount_average_window=50, indneutralize_close_industry_adv_correlation_window=17.8256, correlation_indneutralize_close_rank_window=17.9171) | 无返回注解；return: -np.power(base, exponent) | 无 docstring，需阅读函数体 |
| [alpha71](../../../betalens-factor/alpha101/alpha101_formulas.py#L556) | alpha71(d, *, close_rank_window=3.43976, amount_average_window=180, adv_rank_window=12.0647, ts_rank_close_ts_rank_adv_correlation_window=18.0175, correlation_ts_rank_close_decay_window=4.20501, decay_linear_correlation_ts_rank_rank_window=15.6948, rank_vwap_low_power_exponent=2, rank_vwap_low_decay_window=16.4662, decay_linear_rank_vwap_rank_window=4.4388) | 无返回注解；return: elementwise_max(left, right) | 无 docstring，需阅读函数体 |
| [alpha72](../../../betalens-factor/alpha101/alpha101_formulas.py#L562) | alpha72(d, *, high_low_divisor=2, amount_average_window=40, high_low_adv_correlation_window=8.93345, correlation_adv_high_decay_window=10.1519, vwap_rank_window=3.72469, volume_rank_window=18.5188, ts_rank_vwap_ts_rank_volume_correlation_window=6.86671, correlation_ts_rank_vwap_decay_window=2.95011) | 无返回注解；return: left / right | 无 docstring，需阅读函数体 |
| [alpha73](../../../betalens-factor/alpha101/alpha101_formulas.py#L568) | alpha73(d, *, vwap_delta_lag=4.72775, delta_vwap_decay_window=2.91864, open_mix_weight=0.147155, mixed_complement_base=1, mixed_complement_weight=0.147155, mixed_delta_lag=2.03608, mixed_delta_decay_window=3.33829, decay_linear_mixed_delta_rank_window=16.7411) | 无返回注解；return: -elementwise_max(left, right) | 无 docstring，需阅读函数体 |
| [alpha74](../../../betalens-factor/alpha101/alpha101_formulas.py#L575) | alpha74(d, *, amount_average_window=30, adv_sum_window=37.4843, close_ts_sum_adv_correlation_window=15.1365, high_mix_weight=0.0261661, mixed_complement_base=1, mixed_complement_weight=0.0261661, rank_mixed_rank_volume_correlation_window=11.4791) | 无返回注解；return: -_bool(left &lt; right) | 无 docstring，需阅读函数体 |
| [alpha75](../../../betalens-factor/alpha101/alpha101_formulas.py#L582) | alpha75(d, *, vwap_volume_correlation_window=4.24304, amount_average_window=50, rank_low_rank_adv_correlation_window=12.4413) | 无返回注解；return: _bool(left &lt; right) | 无 docstring，需阅读函数体 |
| [alpha76](../../../betalens-factor/alpha101/alpha101_formulas.py#L588) | alpha76(d, *, vwap_delta_lag=1.24383, delta_vwap_decay_window=11.8259, amount_average_window=81, indneutralize_low_sector_adv_correlation_window=8.14941, corr_rank_window=19.569, ts_rank_corr_decay_window=17.1543, decay_linear_ts_rank_corr_rank_window=19.383) | 无返回注解；return: -elementwise_max(left, right) | 无 docstring，需阅读函数体 |
| [alpha77](../../../betalens-factor/alpha101/alpha101_formulas.py#L595) | alpha77(d, *, high_low_divisor=2, vwap_high_low_decay_window=20.0451, high_low_divisor_2=2, amount_average_window=40, high_low_adv_correlation_window=3.1614, correlation_adv_high_decay_window=5.64125) | 无返回注解；return: elementwise_min(left, right) | 无 docstring，需阅读函数体 |
| [alpha78](../../../betalens-factor/alpha101/alpha101_formulas.py#L601) | alpha78(d, *, low_mix_weight=0.352233, mixed_complement_base=1, mixed_complement_weight=0.352233, mixed_sum_window=19.7428, amount_average_window=40, adv_sum_window=19.7428, ts_sum_mixed_ts_sum_adv_correlation_window=6.83313, rank_vwap_rank_volume_correlation_window=5.77492) | 无返回注解；return: np.power(base, exponent) | 无 docstring，需阅读函数体 |
| [alpha79](../../../betalens-factor/alpha101/alpha101_formulas.py#L608) | alpha79(d, *, close_mix_weight=0.60733, mixed_complement_base=1, mixed_complement_weight=0.60733, indneutralize_mixed_sector_delta_lag=1.23438, vwap_rank_window=3.60973, amount_average_window=150, adv_rank_window=9.18637, ts_rank_vwap_ts_rank_adv_correlation_window=14.6644) | 无返回注解；return: _bool(left &lt; right) | 无 docstring，需阅读函数体 |
| [alpha80](../../../betalens-factor/alpha101/alpha101_formulas.py#L615) | alpha80(d, *, open_mix_weight=0.868128, mixed_complement_base=1, mixed_complement_weight=0.868128, indneutralize_mixed_industry_delta_lag=4.04545, amount_average_window=10, high_adv_correlation_window=5.11456, correlation_high_adv_rank_window=5.53756) | 无返回注解；return: -np.power(base, exponent) | 无 docstring，需阅读函数体 |
| [alpha81](../../../betalens-factor/alpha101/alpha101_formulas.py#L622) | alpha81(d, *, amount_average_window=10, adv_sum_window=49.6054, vwap_ts_sum_adv_correlation_window=8.47743, left_constant=4, rank_corr_product_window=14.9655, rank_vwap_rank_volume_correlation_window=5.07914) | 无返回注解；return: -_bool(left &lt; right) | 无 docstring，需阅读函数体 |
| [alpha82](../../../betalens-factor/alpha101/alpha101_formulas.py#L629) | alpha82(d, *, open_delta_lag=1.46063, delta_open_decay_window=14.8717, indneutralize_volume_sector_open_correlation_window=17.4842, corr_decay_window=6.92131, decay_linear_corr_rank_window=13.4283) | 无返回注解；return: -elementwise_min(left, right) | 无 docstring，需阅读函数体 |
| [alpha83](../../../betalens-factor/alpha101/alpha101_formulas.py#L636) | alpha83(d, *, close_mean_window=5, ratio_delay_lag=2) | 无返回注解；return: rank(delay(ratio, ratio_delay_lag)) * rank(rank(d['volume'])) / (ratio / (d['vwap'] - d['close'])) | 无 docstring，需阅读函数体 |
| [alpha84](../../../betalens-factor/alpha101/alpha101_formulas.py#L641) | alpha84(d, *, vwap_maximum_window=15.3217, vwap_ts_max_rank_window=20.7127, close_delta_lag=4.96796) | 无返回注解；return: signed_power(base, delta(d['close'], close_delta_lag)) | 无 docstring，需阅读函数体 |
| [alpha85](../../../betalens-factor/alpha101/alpha101_formulas.py#L646) | alpha85(d, *, high_mix_weight=0.876703, mixed_complement_base=1, mixed_complement_weight=0.876703, amount_average_window=30, mixed_adv_correlation_window=9.61331, high_low_divisor=2, high_low_rank_window=3.70596, volume_rank_window=10.1595, ts_rank_high_low_ts_rank_volume_correlation_window=7.11408) | 无返回注解；return: np.power(base, exponent) | 无 docstring，需阅读函数体 |
| [alpha86](../../../betalens-factor/alpha101/alpha101_formulas.py#L653) | alpha86(d, *, amount_average_window=20, adv_sum_window=14.7444, close_ts_sum_adv_correlation_window=6.00049, correlation_close_ts_sum_rank_window=20.4195) | 无返回注解；return: -_bool(left &lt; rank(d['close'] - d['vwap'])) | 无 docstring，需阅读函数体 |
| [alpha87](../../../betalens-factor/alpha101/alpha101_formulas.py#L658) | alpha87(d, *, close_mix_weight=0.369701, mixed_complement_base=1, mixed_complement_weight=0.369701, mixed_delta_lag=1.91233, delta_mixed_decay_window=2.65461, amount_average_window=81, indneutralize_industry_adv_close_correlation_window=13.4132, corr_decay_window=4.89768, decay_linear_corr_rank_window=14.4535) | 无返回注解；return: -elementwise_max(left, right) | 无 docstring，需阅读函数体 |
| [alpha88](../../../betalens-factor/alpha101/alpha101_formulas.py#L666) | alpha88(d, *, rank_close_high_decay_window=8.06882, close_rank_window=8.44728, amount_average_window=60, adv_rank_window=20.6966, ts_rank_close_ts_rank_adv_correlation_window=8.01266, corr_decay_window=6.65053, decay_linear_corr_rank_window=2.61957) | 无返回注解；return: elementwise_min(left, right) | 无 docstring，需阅读函数体 |
| [alpha89](../../../betalens-factor/alpha101/alpha101_formulas.py#L673) | alpha89(d, *, amount_average_window=10, low_adv_correlation_window=6.94279, correlation_low_adv_decay_window=5.51607, decay_linear_correlation_low_rank_window=3.79744, indneutralize_vwap_industry_delta_lag=3.48158, delta_indneutralize_vwap_decay_window=10.1466, decay_linear_delta_indneutralize_rank_window=15.3012) | 无返回注解；return: left - right | 无 docstring，需阅读函数体 |
| [alpha90](../../../betalens-factor/alpha101/alpha101_formulas.py#L679) | alpha90(d, *, close_maximum_window=4.66719, amount_average_window=40, indneutralize_subindustry_adv_low_correlation_window=5.38375, correlation_low_indneutralize_rank_window=3.21856) | 无返回注解；return: -np.power(base, exponent) | 无 docstring，需阅读函数体 |
| [alpha91](../../../betalens-factor/alpha101/alpha101_formulas.py#L685) | alpha91(d, *, indneutralize_close_industry_volume_correlation_window=9.74928, corr1_decay_window=16.398, decay_linear_corr1_decay_window=3.83219, decay_linear_corr1_rank_window=4.8667, amount_average_window=30, vwap_adv_correlation_window=4.01303, correlation_vwap_adv_decay_window=2.6809) | 无返回注解；return: -(left - right) | 无 docstring，需阅读函数体 |
| [alpha92](../../../betalens-factor/alpha101/alpha101_formulas.py#L692) | alpha92(d, *, high_low_divisor=2, condition_decay_window=14.7221, decay_linear_condition_rank_window=18.8683, amount_average_window=30, rank_low_rank_adv_correlation_window=7.58555, correlation_rank_low_decay_window=6.94024, decay_linear_correlation_rank_rank_window=6.80584) | 无返回注解；return: elementwise_min(left, right) | 无 docstring，需阅读函数体 |
| [alpha93](../../../betalens-factor/alpha101/alpha101_formulas.py#L699) | alpha93(d, *, amount_average_window=81, indneutralize_vwap_industry_adv_correlation_window=17.4193, correlation_indneutralize_vwap_decay_window=19.848, decay_linear_correlation_indneutralize_rank_window=7.54455, close_mix_weight=0.524434, mixed_complement_base=1, mixed_complement_weight=0.524434, mixed_delta_lag=2.77377, delta_mixed_decay_window=16.2664) | 无返回注解；return: left / right | 无 docstring，需阅读函数体 |
| [alpha94](../../../betalens-factor/alpha101/alpha101_formulas.py#L706) | alpha94(d, *, vwap_minimum_window=11.5783, vwap_rank_window=19.6462, amount_average_window=60, adv_rank_window=4.02992, ts_rank_vwap_ts_rank_adv_correlation_window=18.0926, correlation_ts_rank_vwap_rank_window=2.70756) | 无返回注解；return: -np.power(base, exponent) | 无 docstring，需阅读函数体 |
| [alpha95](../../../betalens-factor/alpha101/alpha101_formulas.py#L712) | alpha95(d, *, open_minimum_window=12.4105, high_low_divisor=2, high_low_sum_window=19.1351, amount_average_window=40, adv_sum_window=19.1351, ts_sum_high_low_ts_sum_adv_correlation_window=12.8742, right_constant=5, rank_corr_rank_window=11.7584) | 无返回注解；return: _bool(left &lt; right) | 无 docstring，需阅读函数体 |
| [alpha96](../../../betalens-factor/alpha101/alpha101_formulas.py#L719) | alpha96(d, *, rank_vwap_rank_volume_correlation_window=3.83878, correlation_rank_vwap_decay_window=4.16783, decay_linear_correlation_rank_rank_window=8.38151, close_rank_window=7.45404, amount_average_window=60, adv_rank_window=4.13242, ts_rank_close_ts_rank_adv_correlation_window=3.65459, corr_argmax_window=12.6556, ts_argmax_corr_decay_window=14.0365, decay_linear_ts_argmax_corr_rank_window=13.4143) | 无返回注解；return: -elementwise_max(left, right) | 无 docstring，需阅读函数体 |
| [alpha97](../../../betalens-factor/alpha101/alpha101_formulas.py#L726) | alpha97(d, *, low_mix_weight=0.721001, mixed_complement_base=1, mixed_complement_weight=0.721001, indneutralize_mixed_industry_delta_lag=3.3705, delta_indneutralize_mixed_decay_window=20.4523, low_rank_window=7.87871, amount_average_window=60, adv_rank_window=17.255, ts_rank_low_ts_rank_adv_correlation_window=4.97547, corr_rank_window=18.5925, ts_rank_corr_decay_window=15.7152, decay_linear_ts_rank_corr_rank_window=6.71659) | 无返回注解；return: -(left - right) | 无 docstring，需阅读函数体 |
| [alpha98](../../../betalens-factor/alpha101/alpha101_formulas.py#L734) | alpha98(d, *, amount_average_window=5, adv_sum_window=26.4719, vwap_ts_sum_adv_correlation_window=4.58418, correlation_vwap_ts_sum_decay_window=7.18088, amount_average_window_2=15, rank_open_rank_adv_correlation_window=20.8187, corr_argmin_window=8.62571, ts_argmin_corr_rank_window=6.95668, ts_rank_ts_argmin_corr_decay_window=8.07206) | 无返回注解；return: left - right | 无 docstring，需阅读函数体 |
| [alpha99](../../../betalens-factor/alpha101/alpha101_formulas.py#L741) | alpha99(d, *, high_low_divisor=2, high_low_sum_window=19.8975, amount_average_window=60, adv_sum_window=19.8975, ts_sum_high_low_ts_sum_adv_correlation_window=8.8136, low_volume_correlation_window=6.28259) | 无返回注解；return: -_bool(left &lt; right) | 无 docstring，需阅读函数体 |
| [alpha100](../../../betalens-factor/alpha101/alpha101_formulas.py#L747) | alpha100(d, *, amount_average_window=20, close_rank_adv_correlation_window=5, close_argmin_window=30, scale_first_coefficient=1.5, amount_average_window_2=20) | 无返回注解；return: -((scale_first_coefficient * scale(first) - scale(second)) * (d['volume'] / _adv(d, amount_average_window_2))) | 无 docstring，需阅读函数体 |
| [alpha101](../../../betalens-factor/alpha101/alpha101_formulas.py#L755) | alpha101(d, *, high_low_epsilon=0.001) | 无返回注解；return: (d['close'] - d['open']) / (d['high'] - d['low'] + high_low_epsilon) | 无 docstring，需阅读函数体 |
| [_parameter_kind](../../../betalens-factor/alpha101/alpha101_formulas.py#L759) | _parameter_kind(name: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_input_specs](../../../betalens-factor/alpha101/alpha101_formulas.py#L796) | _input_specs(formula: str) -&gt; tuple[dict[str, str], dict[str, str]] | tuple[dict[str, str], dict[str, str]] | 无 docstring，需阅读函数体 |
| [_definition](../../../betalens-factor/alpha101/alpha101_formulas.py#L812) | _definition(number: int) -&gt; AlphaDefinition | AlphaDefinition | 无 docstring，需阅读函数体 |
| [get_definition](../../../betalens-factor/alpha101/alpha101_formulas.py#L830) | get_definition(name_or_number: str &#124; int) -&gt; AlphaDefinition | AlphaDefinition | 无 docstring，需阅读函数体 |
| [resolve_compute_kwargs](../../../betalens-factor/alpha101/alpha101_formulas.py#L844) | resolve_compute_kwargs(name_or_number: str &#124; int, compute_kwargs: Mapping[str, Any] &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | Validate flat formula parameters while preserving paper defaults. |
| [default_compute_kwargs](../../../betalens-factor/alpha101/alpha101_formulas.py#L873) | default_compute_kwargs(name_or_number: str &#124; int) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [required_history_bars_for_alpha](../../../betalens-factor/alpha101/alpha101_formulas.py#L877) | required_history_bars_for_alpha(name_or_number: str &#124; int, compute_kwargs: Mapping[str, Any] &#124; None=None) -&gt; int | int | Return a conservative history requirement for a parameterized formula. |
| [compute_alpha](../../../betalens-factor/alpha101/alpha101_formulas.py#L894) | compute_alpha(name_or_number: str &#124; int, **wides) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |

<a id="file-c331801946dd"></a>
## betalens-factor/alpha101/alpha101_mining.py

[打开源码](../../../betalens-factor/alpha101/alpha101_mining.py) · 86 行 · 说明来源：人工文件说明

- **作用**：Alpha101 的挖掘 hook
- **输入**：候选参数
- **输出**：MiningSpec/计算函数配置
- **副作用/维护重点**：桥接通用挖掘和公式库

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha
from betalens.factor.mining import MiningSpec
from factor_template_alpha101 import FactorSpec
from typing import Any, Mapping
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_require](../../../betalens-factor/alpha101/alpha101_mining.py#L21) | _require(params: Mapping[str, Any], key: str) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [_alpha_id](../../../betalens-factor/alpha101/alpha101_mining.py#L27) | _alpha_id(params: Mapping[str, Any]) -&gt; int | int | 无 docstring，需阅读函数体 |
| [_formula_kwargs](../../../betalens-factor/alpha101/alpha101_mining.py#L32) | _formula_kwargs(params: Mapping[str, Any]) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [compute_alpha_mining](../../../betalens-factor/alpha101/alpha101_mining.py#L40) | compute_alpha_mining(**kwargs) | 无返回注解；return: compute_alpha(alpha_id, **kwargs) | 无 docstring，需阅读函数体 |
| [make_mining_spec](../../../betalens-factor/alpha101/alpha101_mining.py#L45) | make_mining_spec(params: Mapping[str, Any]) -&gt; MiningSpec | MiningSpec | 根据一个候选参数字典声明所选 Alpha 的计算与缓存输入。 |
| [mining_warmup_days](../../../betalens-factor/alpha101/alpha101_mining.py#L75) | mining_warmup_days(params: Mapping[str, Any]) -&gt; int | int | 按公式所需历史 bars 推导自然日预热长度，并保证至少 30 天。 |

<a id="file-fcaf4f4ce126"></a>
## betalens-factor/alpha101/alpha101_parameters.py

[打开源码](../../../betalens-factor/alpha101/alpha101_parameters.py) · 354 行 · 说明来源：人工文件说明

- **作用**：Alpha 参数空间推导
- **输入**：公式参数、生成边界规则
- **输出**：各因子搜索定义
- **副作用/维护重点**：参数范围不代表统计最优；与公式签名同步

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_formulas import AlphaParameter, default_compute_kwargs, get_definition
from betalens.factor.mining import validate_parameter_specs
from betalens.factor.mining_optuna import to_optuna_distribution
from typing import Any, Mapping, Sequence
import hashlib
import itertools
import json
import math
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [parameter_catalog](../../../betalens-factor/alpha101/alpha101_parameters.py#L42) | parameter_catalog(alpha_id: str &#124; int) -&gt; dict[str, AlphaParameter] | dict[str, AlphaParameter] | Return the ordered parameter catalog for one Alpha formula. |
| [_unique](../../../betalens-factor/alpha101/alpha101_parameters.py#L47) | _unique(values: Sequence[int &#124; float]) -&gt; list[int &#124; float] | list[int &#124; float] | 无 docstring，需阅读函数体 |
| [_type_limits](../../../betalens-factor/alpha101/alpha101_parameters.py#L55) | _type_limits(kind: str, overrides: Mapping[str, Mapping[str, Any]] &#124; None) -&gt; dict[str, float] &#124; None | dict[str, float] &#124; None | 无 docstring，需阅读函数体 |
| [_clamp_reference_points](../../../betalens-factor/alpha101/alpha101_parameters.py#L69) | _clamp_reference_points(values: Sequence[int &#124; float], default: int &#124; float, limits: Mapping[str, float] &#124; None, *, integer: bool=False) -&gt; list[int &#124; float] | list[int &#124; float] | 无 docstring，需阅读函数体 |
| [candidate_values](../../../betalens-factor/alpha101/alpha101_parameters.py#L87) | candidate_values(spec: AlphaParameter, *, range_multiplier: float=DEFAULT_RANGE_MULTIPLIER, type_limits: Mapping[str, Mapping[str, Any]] &#124; None=None) -&gt; list[int &#124; float] | list[int &#124; float] | 按参数种类返回宽范围参考点，并应用类型级硬边界。 |
| [default_search_space](../../../betalens-factor/alpha101/alpha101_parameters.py#L124) | default_search_space(alpha_id: str &#124; int, max_dimensions: int=DEFAULT_MAX_DIMENSIONS, *, range_multiplier: float=DEFAULT_RANGE_MULTIPLIER, type_limits: Mapping[str, Mapping[str, Any]] &#124; None=None) -&gt; dict[str, list[int &#124; float]] | dict[str, list[int &#124; float]] | 按公式参数顺序放开至多 ''max_dimensions'' 个可搜索参数并扩展边界。 |
| [mining_parameter_specs](../../../betalens-factor/alpha101/alpha101_parameters.py#L151) | mining_parameter_specs(alpha_id: str &#124; int, max_dimensions: int=DEFAULT_MAX_DIMENSIONS, *, range_multiplier: float=DEFAULT_RANGE_MULTIPLIER, type_limits: Mapping[str, Mapping[str, Any]] &#124; None=None) -&gt; dict[str, dict[str, Any]] | dict[str, dict[str, Any]] | 将宽范围参考点转换为 mining 使用的类型、边界、步长和尺度定义。 整数参数使用 ''step=1''；窗口和滞后参数使用 ''scale=log''，其余参数使用 ''scale=linear''。参考点本身不会作为 categorical 候选保留下来。 |
| [validate_mining_parameter_specs](../../../betalens-factor/alpha101/alpha101_parameters.py#L185) | validate_mining_parameter_specs(alpha_id: str &#124; int, parameter_specs: Mapping[str, Mapping[str, Any]]) -&gt; dict[str, dict[str, Any]] | dict[str, dict[str, Any]] | Validate aggregate parameter definitions against one Alpha formula. |
| [mining_optuna_distributions](../../../betalens-factor/alpha101/alpha101_parameters.py#L202) | mining_optuna_distributions(alpha_id: str &#124; int, parameter_specs: Mapping[str, Mapping[str, Any]]) -&gt; dict[str, Any] | dict[str, Any] | Build Optuna distributions after formula-aware validation. |
| [mining_parameter_limits](../../../betalens-factor/alpha101/alpha101_parameters.py#L213) | mining_parameter_limits(alpha_id: str &#124; int, *, type_limits: Mapping[str, Mapping[str, Any]] &#124; None=None) -&gt; dict[str, dict[str, float]] | dict[str, dict[str, float]] | 返回每个可搜索公式参数在自动扩边阶段不可突破的硬边界。 |
| [aggregate_mining_factors](../../../betalens-factor/alpha101/alpha101_parameters.py#L227) | aggregate_mining_factors(*, range_multiplier: float=DEFAULT_RANGE_MULTIPLIER, max_dimensions: int=DEFAULT_MAX_DIMENSIONS, type_limits: Mapping[str, Mapping[str, Any]] &#124; None=None) -&gt; dict[str, dict[str, Any]] | dict[str, dict[str, Any]] | 将 ''factors: all'' 展开为 ALPHA1 至 ALPHA101 的自动参数空间。 |
| [validate_search_space](../../../betalens-factor/alpha101/alpha101_parameters.py#L250) | validate_search_space(alpha_id: str &#124; int, search_space: Mapping[str, Sequence[Any]]) -&gt; dict[str, list[int &#124; float]] | dict[str, list[int &#124; float]] | Require one non-empty, numeric value list for every formula parameter. |
| [grid_candidate_count](../../../betalens-factor/alpha101/alpha101_parameters.py#L290) | grid_candidate_count(search_space: Mapping[str, Sequence[Any]]) -&gt; int | int | 无 docstring，需阅读函数体 |
| [formula_param_candidates](../../../betalens-factor/alpha101/alpha101_parameters.py#L294) | formula_param_candidates(alpha_id: str &#124; int, search_space: Mapping[str, Sequence[Any]], *, max_grid_candidates: int=256) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | Expand the complete grid and fail before execution if it exceeds the cap. |
| [formula_param_gid](../../../betalens-factor/alpha101/alpha101_parameters.py#L315) | formula_param_gid(alpha_id: str &#124; int, params: Mapping[str, Any]) -&gt; str | str | Create a deterministic, filesystem-safe candidate identifier. |
| [catalog_rows](../../../betalens-factor/alpha101/alpha101_parameters.py#L324) | catalog_rows(alpha_id: str &#124; int) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |

<a id="file-768c6870f501"></a>
## betalens-factor/alpha101/class_alpha101.yaml

[打开源码](../../../betalens-factor/alpha101/class_alpha101.yaml) · 17 行 · 说明来源：文件族规则

- **作用**：因子类别发现元数据
- **输入**：维护者填写的参数
- **输出**：类别信息；不能替代完整因子运行 YAML
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/class_alpha101.yaml#L1)：`class: alpha101`
- [L2](../../../betalens-factor/alpha101/class_alpha101.yaml#L2)：`template_module: factor_template_alpha101`
- [L3](../../../betalens-factor/alpha101/class_alpha101.yaml#L3)：`source: WorldQuant 101 Formulaic Alphas (Kakushadze 2016)`
- [L4](../../../betalens-factor/alpha101/class_alpha101.yaml#L4)：`defaults:`

<a id="file-02b6c181729c"></a>
## betalens-factor/alpha101/factor_template_alpha101.py

[打开源码](../../../betalens-factor/alpha101/factor_template_alpha101.py) · 457 行 · 说明来源：人工文件说明

- **作用**：Alpha 算子与截面/择时模板桥接
- **输入**：宽表、窗口、FactorSpec
- **输出**：算子结果与 RunResult
- **副作用/维护重点**：rank 是截面，ts_rank 是时序；择时仍需截面上下文

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.analyst import Analyst
from betalens.backtest import BacktestBase
from betalens.datafeed import get_absolute_trade_days
from betalens.factor.signal import build_signal_weights, infer_signal_warmup, resolve_timing_start_date
from dataclasses import dataclass, field
from factor_template import FactorPipeline as CorePipeline, FactorSpec as CoreFactorSpec
from factor_template import RunResult as CoreRunResult, _ensure_runtime, align_daily_wides, build_pit_universe, fetch_daily_wide, fetch_industry_wide, mask_wide_by_pit_universe
from pathlib import Path
from typing import Any, Callable
import math
import numpy as np
import pandas as pd
import re
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [FactorSpec](../../../betalens-factor/alpha101/factor_template_alpha101.py#L51) | class FactorSpec() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：name: str; inputs: dict[str, str]; compute: Callable[..., Any]; strategy_type: str = 'cross_section'; industry_inputs: dict[str, str] = field(default_factory=dict); required_history_bars: int = 0; mask_inputs_by_pit: bool = False; direction: str = 'positive'; compute_kwargs: di…（完整内容见 inventory.json/源码） |
| [FactorPipeline](../../../betalens-factor/alpha101/factor_template_alpha101.py#L74) | class FactorPipeline() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [FactorPipeline.__init__](../../../betalens-factor/alpha101/factor_template_alpha101.py#L75) | __init__(self, spec: FactorSpec) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [FactorPipeline.run](../../../betalens-factor/alpha101/factor_template_alpha101.py#L78) | run(self, *args, **kwargs) | 无返回注解；return: CorePipeline(core_spec).run(*args, **kwargs) | 无 docstring，需阅读函数体 |
| [window](../../../betalens-factor/alpha101/factor_template_alpha101.py#L99) | window(value) | 无返回注解；return: max(1, int(math.floor(float(value) + 0.5))) | Convert a paper window to the plan's nearest-integer convention. |
| [delta](../../../betalens-factor/alpha101/factor_template_alpha101.py#L106) | delta(x, n=1) | 无返回注解；return: x.diff(window(n)) | X 相对 n 周期前的时序差分。 |
| [delay](../../../betalens-factor/alpha101/factor_template_alpha101.py#L111) | delay(x, n=1) | 无返回注解；return: x.shift(window(n)) | X 的 n 周期时序滞后值。 |
| [sign](../../../betalens-factor/alpha101/factor_template_alpha101.py#L116) | sign(x) | 无返回注解；return: np.sign(x) | 逐元素符号（-1/0/1）。 |
| [rank](../../../betalens-factor/alpha101/factor_template_alpha101.py#L123) | rank(x) | 无返回注解；return: x.rank(axis=1, pct=True) | 同一日截面（按行）百分位排名，范围 (0,1]。 |
| [ts_rank](../../../betalens-factor/alpha101/factor_template_alpha101.py#L128) | ts_rank(x, n) | 无返回注解；return: x.rolling(n, min_periods=n).apply(lambda values: pd.Series(values).rank(method='average', pct=True).iloc[-1], raw=True) | n 周期窗口内当前值的时序百分位排名。 |
| [ts_min](../../../betalens-factor/alpha101/factor_template_alpha101.py#L139) | ts_min(x, n) | 无返回注解；return: x.rolling(n, min_periods=n).min() | n 周期内时序最小值。 |
| [ts_max](../../../betalens-factor/alpha101/factor_template_alpha101.py#L145) | ts_max(x, n) | 无返回注解；return: x.rolling(n, min_periods=n).max() | n 周期内时序最大值。 |
| [ts_sum](../../../betalens-factor/alpha101/factor_template_alpha101.py#L151) | ts_sum(x, n) | 无返回注解；return: x.rolling(n, min_periods=n).sum() | n 周期内时序求和。 |
| [ts_mean](../../../betalens-factor/alpha101/factor_template_alpha101.py#L157) | ts_mean(x, n) | 无返回注解；return: x.rolling(n, min_periods=n).mean() | 无 docstring，需阅读函数体 |
| [correlation](../../../betalens-factor/alpha101/factor_template_alpha101.py#L162) | correlation(x, y, n) | 无返回注解；return: x.rolling(n, min_periods=n).corr(y) | X 与 Y 的 n 周期滚动相关系数（逐列）。 |
| [covariance](../../../betalens-factor/alpha101/factor_template_alpha101.py#L168) | covariance(x, y, n) | 无返回注解；return: x.rolling(n, min_periods=n).cov(y) | X 与 Y 的 n 周期滚动协方差（逐列）。 |
| [stddev](../../../betalens-factor/alpha101/factor_template_alpha101.py#L174) | stddev(x, n) | 无返回注解；return: x.rolling(n, min_periods=n).std() | n 周期滚动标准差。 |
| [product](../../../betalens-factor/alpha101/factor_template_alpha101.py#L180) | product(x, n) | 无返回注解；return: x.rolling(n, min_periods=n).apply(np.prod, raw=True) | 无 docstring，需阅读函数体 |
| [ts_argmax](../../../betalens-factor/alpha101/factor_template_alpha101.py#L187) | ts_argmax(x, n) | 无返回注解；return: x.rolling(n, min_periods=n).apply(lambda values: float(np.argmax(values) + 1), raw=True) | 无 docstring，需阅读函数体 |
| [ts_argmin](../../../betalens-factor/alpha101/factor_template_alpha101.py#L194) | ts_argmin(x, n) | 无返回注解；return: x.rolling(n, min_periods=n).apply(lambda values: float(np.argmin(values) + 1), raw=True) | 无 docstring，需阅读函数体 |
| [decay_linear](../../../betalens-factor/alpha101/factor_template_alpha101.py#L201) | decay_linear(x, n) | 无返回注解；return: x.rolling(n, min_periods=n).apply(lambda values: float(np.dot(values, weights)), raw=True) | 无 docstring，需阅读函数体 |
| [scale](../../../betalens-factor/alpha101/factor_template_alpha101.py#L210) | scale(x, a=1.0) | 无返回注解；return: x.div(denominator, axis=0) * float(a) | 无 docstring，需阅读函数体 |
| [signed_power](../../../betalens-factor/alpha101/factor_template_alpha101.py#L215) | signed_power(x, exponent) | 无返回注解；return: np.sign(x) * np.power(np.abs(x), exponent) | 无 docstring，需阅读函数体 |
| [indneutralize](../../../betalens-factor/alpha101/factor_template_alpha101.py#L221) | indneutralize(x, groups) | 无返回注解；return: out | 无 docstring，需阅读函数体 |
| [where](../../../betalens-factor/alpha101/factor_template_alpha101.py#L237) | where(condition, when_true, when_false) | 无返回注解；return: when_true.where(condition, when_false) | 无 docstring，需阅读函数体 |
| [elementwise_min](../../../betalens-factor/alpha101/factor_template_alpha101.py#L245) | elementwise_min(x, y) | 无返回注解；return: pd.DataFrame(np.minimum(x, y), index=x.index, columns=x.columns) | 无 docstring，需阅读函数体 |
| [elementwise_max](../../../betalens-factor/alpha101/factor_template_alpha101.py#L252) | elementwise_max(x, y) | 无返回注解；return: pd.DataFrame(np.maximum(x, y), index=x.index, columns=x.columns) | 无 docstring，需阅读函数体 |
| [clean_inf](../../../betalens-factor/alpha101/factor_template_alpha101.py#L259) | clean_inf(x) | 无返回注解；return: x.replace([float('inf'), float('-inf')], float('nan')) | 把 ±inf 置为 NaN（算子末尾统一调用，防止除零污染）。 |
| [_timing_codes](../../../betalens-factor/alpha101/factor_template_alpha101.py#L264) | _timing_codes(params, universe) | 无返回注解；return: [stock_code]; values | 无 docstring，需阅读函数体 |
| [_with_timing_targets](../../../betalens-factor/alpha101/factor_template_alpha101.py#L279) | _with_timing_targets(pit_universe, target_codes) | 无返回注解；return: {day: {str(code) for code in members}.union(targets) for day, members in pit_universe.items()} | Keep PIT constituents as formula context while always admitting targets. |
| [TimingFactorPipeline](../../../betalens-factor/alpha101/factor_template_alpha101.py#L288) | class TimingFactorPipeline() | 类定义；构造/属性见方法与字段 | Shared full-cross-section-to-single-stock Alpha101 timing pipeline. |
| [TimingFactorPipeline.__init__](../../../betalens-factor/alpha101/factor_template_alpha101.py#L291) | __init__(self, spec: FactorSpec) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [TimingFactorPipeline.run](../../../betalens-factor/alpha101/factor_template_alpha101.py#L294) | run(self, start_date: str, end_date: str, *, rebal_freq: str='D', universe: list &#124; None=None, n_quantiles: int=10, initial_amount: float=100000000.0, benchmark_code: str &#124; None=None, output_dir: str='.', include_profiling: bool=False, dump_excel: bool=True, warmup_days: int &#124; None=None, verbose: bool=True) | 无返回注解；return: CoreRunResult(backtest=bt, analyst=analyst, profiling=None, neutralize_stats=None, factor_values=factor_values, pit_validation=None) | 无 docstring，需阅读函数体 |

<a id="file-d9cd32d64d2e"></a>
## betalens-factor/alpha101/mining/parameter_space.yaml

[打开源码](../../../betalens-factor/alpha101/mining/parameter_space.yaml) · 120 行 · 说明来源：文件族规则

- **作用**：参数空间、搜索与评价规则
- **输入**：维护者填写的参数
- **输出**：挖掘候选和窗口配置
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L2](../../../betalens-factor/alpha101/mining/parameter_space.yaml#L2)：`version: 1`
- [L4](../../../betalens-factor/alpha101/mining/parameter_space.yaml#L4)：`factor_class: alpha101`
- [L8](../../../betalens-factor/alpha101/mining/parameter_space.yaml#L8)：`factors:`
- [L31](../../../betalens-factor/alpha101/mining/parameter_space.yaml#L31)：`alpha101_parameter_generation:`
- [L44](../../../betalens-factor/alpha101/mining/parameter_space.yaml#L44)：`evaluation:`
- [L57](../../../betalens-factor/alpha101/mining/parameter_space.yaml#L57)：`windows:`
- [L63](../../../betalens-factor/alpha101/mining/parameter_space.yaml#L63)：`search:`
- [L109](../../../betalens-factor/alpha101/mining/parameter_space.yaml#L109)：`selection:`

<a id="file-7663d0e0162f"></a>
## betalens-factor/alpha101/mining/performance.yaml

[打开源码](../../../betalens-factor/alpha101/mining/performance.yaml) · 26 行 · 说明来源：文件族规则

- **作用**：挖掘资源、缓存和输出配置
- **输入**：维护者填写的参数
- **输出**：调度与存储参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/alpha101/mining/performance.yaml#L1)：`runtime:`
- [L11](../../../betalens-factor/alpha101/mining/performance.yaml#L11)：`cache:`
- [L15](../../../betalens-factor/alpha101/mining/performance.yaml#L15)：`output:`
- [L20](../../../betalens-factor/alpha101/mining/performance.yaml#L20)：`logging:`

<a id="file-345c17407b66"></a>
## betalens-factor/alpha101/mining/run.py

[打开源码](../../../betalens-factor/alpha101/mining/run.py) · 42 行 · 说明来源：人工文件说明

- **作用**：Alpha101 挖掘命令入口
- **输入**：搜索/性能 YAML 与命令参数
- **输出**：MiningResult 和搜索产物
- **副作用/维护重点**：可一次启动多个因子；不可将默认范围当快速冒烟测试

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.factor.mining import run_mining
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [main](../../../betalens-factor/alpha101/mining/run.py#L19) | main() -&gt; int | int | 无 docstring，需阅读函数体 |

<a id="file-f13bf768b75e"></a>
## betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.py

[打开源码](../../../betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.py) · 71 行 · 说明来源：文件族规则

- **作用**：ILLIQ_v2 factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template import FactorPipeline, FactorSpec
from pathlib import Path
import argparse
import numpy as np
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.py#L32) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_ILLIQ](../../../betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.py#L36) | compute_ILLIQ(close_wide, amount_wide, window) | 无返回注解；return: illiq_daily.rolling(window, min_periods=max(1, int(window) // 2)).mean() | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.py#L42) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.py#L54) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.py#L63) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-7273963b79c7"></a>
## betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.yaml

[打开源码](../../../betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.yaml) · 35 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.yaml#L7)：`factor_spec:`
- [L20](../../../betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.yaml#L20)：`weight:`
- [L26](../../../betalens-factor/citic_hf_behavior/ILLIQ_v2/factor_ILLIQ_v2.yaml#L26)：`run:`

<a id="file-7122ee51949f"></a>
## betalens-factor/citic_hf_behavior/class_citic_hf_behavior.yaml

[打开源码](../../../betalens-factor/citic_hf_behavior/class_citic_hf_behavior.yaml) · 17 行 · 说明来源：文件族规则

- **作用**：因子类别发现元数据
- **输入**：维护者填写的参数
- **输出**：类别信息；不能替代完整因子运行 YAML
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/citic_hf_behavior/class_citic_hf_behavior.yaml#L1)：`class: citic_hf_behavior`
- [L2](../../../betalens-factor/citic_hf_behavior/class_citic_hf_behavior.yaml#L2)：`template_module: factor_template`
- [L3](../../../betalens-factor/citic_hf_behavior/class_citic_hf_behavior.yaml#L3)：`source: 中信建投高频和行为金融学选股因子跟踪周报`
- [L4](../../../betalens-factor/citic_hf_behavior/class_citic_hf_behavior.yaml#L4)：`defaults:`

<a id="file-8fbde51b40fc"></a>
## betalens-factor/factor_template.py

[打开源码](../../../betalens-factor/factor_template.py) · 1085 行 · 说明来源：人工文件说明

- **作用**：通用截面研究管线与中间数据转换
- **输入**：FactorSpec、运行日期和参数
- **输出**：RunResult、回测/报告/体检等产物
- **副作用/维护重点**：查库、预热/PIT、分组、执行与导出；研究口径中心

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.analyst import Analyst as _Analyst
from betalens.analyst.metrics import group_nav
from betalens.analyst.metrics import match_trade_pairs
from betalens.backtest import BacktestBase as _BacktestBase
from betalens.datafeed import Datafeed as _Datafeed, get_absolute_trade_days as _get_absolute_trade_days, get_index_universe as _get_index_universe
from betalens.factor.factor import single_characteristic as _single_characteristic, get_single_factor_weight as _get_single_factor_weight
from betalens.factor.preprocessing import winsorize_factor as _winsorize_factor, standardize_factor as _standardize_factor, neutralize_factor as _neutralize_factor, query_industry_panel as _query_industry_panel
from betalens.factor.profiling import describe_distribution as _describe_distribution, coverage_stats as _coverage_stats, detect_outliers as _detect_outliers, distribution_stability as _distribution_stability, factor_profile_payload as _factor_profile_payload
from dataclasses import dataclass, field
from datafeed.validation import fix_null_values as _fix_null_values, FillStrategy as _FillStrategy
from typing import Callable, Any
import betalens.analyst.plotting as _P
import matplotlib
import matplotlib.pyplot as _plt
import numpy as _np
import numpy as np
import pandas as _pd
import pandas as pd
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_ensure_runtime](../../../betalens-factor/factor_template.py#L52) | _ensure_runtime() | 无返回注解；return: None | Load heavy runtime dependencies only when a pipeline actually runs. |
| [_ensure_profiling_runtime](../../../betalens-factor/factor_template.py#L109) | _ensure_profiling_runtime() | 无返回注解；return: None | 无 docstring，需阅读函数体 |
| [fetch_daily_wide](../../../betalens-factor/factor_template.py#L141) | fetch_daily_wide(metric, universe=None, start_date=None, end_date=None, table_name=DB_TABLE) | 无返回注解；return: pd.DataFrame(); df.pivot_table(index='datetime', columns='code', values='value').sort_index() | 无 docstring，需阅读函数体 |
| [align_daily_wides](../../../betalens-factor/factor_template.py#L156) | align_daily_wides(wides) | 无返回注解；return: dict(wides); aligned | Align daily metrics by trade date at that day's latest availability time. |
| [wide_to_prequery](../../../betalens-factor/factor_template.py#L187) | wide_to_prequery(wide_df, metric_name, signal_dates) | 无返回注解；return: long | 宽表 → betalens 长表（仅保留 signal_dates 当日截面）。 输出列与 pre_query_characteristic_data 对齐：input_ts/code/{metric}/datetime/ diff_hours，可直接喂给 preprocess / single_characteristic。 |
| [build_pit_universe](../../../betalens-factor/factor_template.py#L204) | build_pit_universe(signal_dates, index_code, table_name='index_universe') | 无返回注解；return: pit | 构建 {信号日: [成分股代码]} 的 point-in-time 成分股映射（防前视）。 |
| [mask_wide_by_pit_universe](../../../betalens-factor/factor_template.py#L214) | mask_wide_by_pit_universe(wide_df, pit_universe) | 无返回注解；return: wide_df; wide_df.where(mask) | Mask every row to constituents effective on that row's calendar date. |
| [fetch_industry_wide](../../../betalens-factor/factor_template.py#L230) | fetch_industry_wide(scheme, universe, dates, reference_index, chunk_size=30) | 无返回注解；return: pd.DataFrame(); pd.DataFrame(index=reference_index, columns=universe, dtype=object); out | Fetch a PIT industry label panel and align it to market-wide timestamps. |
| [filter_long_by_pit_universe](../../../betalens-factor/factor_template.py#L258) | filter_long_by_pit_universe(long_df, pit_universe) | 无返回注解；return: long_df; long_df.loc[mask].reset_index(drop=True) | 按 point-in-time 成分股逐期过滤长表。 某信号日成分股为空（指数无快照）时严格剔除该期，避免在无实时股票池 约束的情况下误选全市场股票。 |
| [filter_long_by_pit_universe._keep](../../../betalens-factor/factor_template.py#L267) | _keep(row) | 无返回注解；return: False; row['code'] in members | 无 docstring，需阅读函数体 |
| [infer_warmup_days](../../../betalens-factor/factor_template.py#L277) | infer_warmup_days(compute_kwargs, minimum=0) | 无返回注解；return: int(minimum or 0); max(int(minimum or 0), int(max(candidates) * 2 + 30)) | 根据常见窗口参数自动推断取数预热天数。 rolling/ewm/delta 类因子通常需要回测起点前的历史数据。这里把 window/lookback/period/span/lag/n 等整数参数视为交易日窗口，并按约 2 倍日历天加缓冲换算，保证年化窗口在回测首日附近已有完整历史。 |
| [validate_weights_in_pit_universe](../../../betalens-factor/factor_template.py#L298) | validate_weights_in_pit_universe(weights, pit_universe) | 无返回注解；return: pd.DataFrame(); pd.DataFrame(rows).set_index('input_ts').sort_index() | 校验每期非零权重股票是否都属于该期 PIT 股票池。 |
| [_labeled_to_factor_values](../../../betalens-factor/factor_template.py#L325) | _labeled_to_factor_values(labeled, name) | 无返回注解；return: factor_values.sort_values(['信号日', '分组', '因子值'], ascending=[True, False, False]).reset_index(drop=True) | 无 docstring，需阅读函数体 |
| [_expand_weights_to_factor_universe](../../../betalens-factor/factor_template.py#L337) | _expand_weights_to_factor_universe(weights, factor_values) | 无返回注解；return: weights; expanded.reindex(columns=codes, fill_value=0.0) | 保留全量因子股票代码，让回测收益矩阵覆盖所有分组。 未选分组的权重仍为 0，因此不会进入策略持仓；但 BacktestBase 会据此 查询这些股票的价格，供独立的分组净值图使用。 |
| [_factor_values_for_group_nav](../../../betalens-factor/factor_template.py#L361) | _factor_values_for_group_nav(factor_values, n_quantiles) | 无返回注解；return: factor_values; out | 把内部 0 基分组标签转换为 group_nav 使用的 1 基标签。 |
| [grouped_factor_statistics](../../../betalens-factor/factor_template.py#L382) | grouped_factor_statistics(labeled, name) | 无返回注解；return: (df, by_date_group, summary) | 基于 single_characteristic 的全量分组矩阵生成统计表。 |
| [group_balance_statistics](../../../betalens-factor/factor_template.py#L424) | group_balance_statistics(labeled, name) | 无返回注解；return: (by_date, pd.DataFrame()); (by_date, summary) | 评估逐期组间数量平衡和 firm characteristic 区分度。 |
| [append_grouped_profiling_excel](../../../betalens-factor/factor_template.py#L493) | append_grouped_profiling_excel(output_dir, name, labeled) | 无返回注解；return: {'group_stats_by_date': by_date_group, 'group_stats_summary': summary, 'group_factor_values': values, 'group_balance_by_date': balance_by_date, 'group_balance_summary': balance_summary} | 把全量分组矩阵与分组统计写入 profiling Excel。 |
| [_match_trade_pairs](../../../betalens-factor/factor_template.py#L519) | _match_trade_pairs(rebalance_log) | 无返回注解；return: match_trade_pairs(rebalance_log) | 兼容旧模板调用，实际口径由 analyst.metrics 统一维护。 |
| [_compute_group_nav](../../../betalens-factor/factor_template.py#L526) | _compute_group_nav(bt, factor_values, n_quantiles: int) | 无返回注解；return: group_nav(getattr(bt, 'cost_ret', None), _factor_values_for_group_nav(factor_values, n_quantiles), n_quantiles) | 从已有回测的 cost_ret 直接计算各分组等权净值，无需额外查库。 原理：bt.cost_ret 已记录每个调仓区间内各标的的价格变化率； factor_values 给出每个信号日各分组的成员；两者结合即可得到各组 等权持仓期收益，cumprod 后得到净值曲线（从 1.0 出发）。 Args: bt: 已完成回测的 BacktestBase 实例（含 cost_ret） factor_values: _labeled_to_factor_values 输出（信号日/股票代码/分组） n_quantiles: 分组数 Returns: DataFrame: index=调仓日, colu…（完整内容见 inventory.json/源码） |
| [FactorSpec](../../../betalens-factor/factor_template.py#L555) | class FactorSpec() | 类定义；构造/属性见方法与字段 | 声明一个因子的全部信息。 name: 因子名（输出文件前缀、长表列名） inputs: {算子参数名: 数据库 metric}，框架按此抓取每个宽表 compute: 算子函数；签名 = inputs 中所有 key + compute_kwargs direction: "positive"→高分组做多 (long=[n_q-1]) &#124; "negative"→低分组做多 (long=[0]) compute_kwargs: 透传给 compute 的额外关键字参数（如 window=20） table_name: Datafeed 数据表名 use_industry / use_mktcap…（完整内容见 inventory.json/源码） |
| [RunResult](../../../betalens-factor/factor_template.py#L597) | class RunResult() | 类定义；构造/属性见方法与字段 | FactorPipeline.run() 的统一结果容器。 支持 'bt, analyst = pipeline.run(...)' 解包（向后兼容旧调用方）； 新代码用 result.profiling / result.neutralize_stats 取增量产物。；字段：backtest: Any = None; analyst: Any = None; profiling: dict &#124; None = None; neutralize_stats: pd.DataFrame &#124; None = None; factor_values: pd.DataFrame &#124; None = None…（完整内容见 inventory.json/源码） |
| [RunResult.__iter__](../../../betalens-factor/factor_template.py#L611) | __iter__(self) | 无返回注解；return: iter((self.backtest, self.analyst)) | 无 docstring，需阅读函数体 |
| [FactorPipeline](../../../betalens-factor/factor_template.py#L619) | class FactorPipeline() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [FactorPipeline.__init__](../../../betalens-factor/factor_template.py#L620) | __init__(self, spec: FactorSpec) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [FactorPipeline._resolve_groups](../../../betalens-factor/factor_template.py#L623) | _resolve_groups(self, n_q: int) -&gt; tuple[list, list] | tuple[list, list] | 无 docstring，需阅读函数体 |
| [FactorPipeline._resolve_groups._as_list](../../../betalens-factor/factor_template.py#L628) | _as_list(value) | 无返回注解；return: []; [value]; list(value) | 无 docstring，需阅读函数体 |
| [FactorPipeline._preprocess_with_stats](../../../betalens-factor/factor_template.py#L641) | _preprocess_with_stats(self, prequery, metric, industry_scheme, mktcap_col, verbose) | 无返回注解；return: (processed, neu_df) | 逐截面 winsorize→standardize→neutralize，同时收集中性化诊断。 等价于 betalens.preprocess_factor，但额外返回逐期诊断 DataFrame （preprocess_factor 仅 print 不返回，故此处内联以便 dashboard 展示）。 Returns: (processed_df, neu_stats_df) neu_stats_df 列：input_ts/n_obs/n_industry_dummies/r2/skipped |
| [FactorPipeline._run_profiling](../../../betalens-factor/factor_template.py#L707) | _run_profiling(self, factor_wide, name, output_dir, verbose) | 无返回注解；return: results | 因子值体检：分布函数/集中度/p值阈值/时变稳定性 + PNG。 |
| [FactorPipeline.run](../../../betalens-factor/factor_template.py#L809) | run(self, start_date: str, end_date: str, *, rebal_freq: str='D', grouping_mode: str='equal_count', universe: list &#124; None=None, n_quantiles: int=20, initial_amount: float=100000000.0, benchmark_code: str &#124; None=None, output_dir: str='.', extra_inputs: dict[str, pd.DataFrame] &#124; None=None, include_profiling: bool=True, dump_excel: bool=True, warmup_days: int &#124; None=None, verbose: bool=True) -&gt; RunResult | RunResult | 运行完整管线: 取数 → 算子 → [profiling] → 中性化 → 分组 → 权重 → 回测 → 报告 返回 RunResult（可解包为 bt, analyst 向后兼容）。 股票池：index_code 给定时逐期 PIT 成分股过滤（防前视）；否则用静态 universe。 中性化：use_industry / use_mktcap 控制，诊断收入 RunResult.neutralize_stats。 dump_excel=False 时跳过 dump_to_excel（调用方可自行异步落盘，避免阻塞）。 |

<a id="file-7d5ea8e8fc1a"></a>
## betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.py

[打开源码](../../../betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.py) · 70 行 · 说明来源：文件族规则

- **作用**：RSI_FAST tdx factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_tdx import FactorPipeline, FactorSpec, REF, SMA, clean_inf
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.py#L30) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_rsi_fast](../../../betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.py#L34) | compute_rsi_fast(close_wide, window) | 无返回注解；return: clean_inf(up / ab * 100) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.py#L41) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.py#L53) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.py#L62) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-4bd031c517bb"></a>
## betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.yaml

[打开源码](../../../betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.yaml) · 34 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.yaml#L7)：`factor_spec:`
- [L19](../../../betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.yaml#L19)：`weight:`
- [L25](../../../betalens-factor/tdx/RSI_FAST/factor_RSI_FAST.yaml#L25)：`run:`

<a id="file-c703dbd74726"></a>
## betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.py

[打开源码](../../../betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.py) · 70 行 · 说明来源：文件族规则

- **作用**：RSI_SLOW tdx factor.；具体因子脚本/模板
- **输入**：完整 YAML、compute 的具名宽表和公式参数；精确签名见下
- **输出**：算子结果、spec 或管线 RunResult；运行时可生成报告
- **副作用/维护重点**：先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_tdx import FactorPipeline, FactorSpec, REF, SMA, clean_inf
from pathlib import Path
import argparse
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_config](../../../betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.py#L30) | load_config(path: str &#124; Path=_CONFIG_FILE) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [compute_rsi_slow](../../../betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.py#L34) | compute_rsi_slow(close_wide, window) | 无返回注解；return: clean_inf(up / ab * 100) | 无 docstring，需阅读函数体 |
| [build_spec](../../../betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.py#L41) | build_spec(config: dict, config_path: str &#124; Path=_CONFIG_FILE) -&gt; FactorSpec | FactorSpec | 无 docstring，需阅读函数体 |
| [run_from_config](../../../betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.py#L53) | run_from_config(config_path: str &#124; Path=_CONFIG_FILE) | 无返回注解；return: FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs) | 无 docstring，需阅读函数体 |
| [main](../../../betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.py#L62) | main() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-46cdfb650f79"></a>
## betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.yaml

[打开源码](../../../betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.yaml) · 34 行 · 说明来源：文件族规则

- **作用**：具体因子的完整运行参数
- **输入**：维护者填写的参数
- **输出**：meta/factor_spec/weight/run，经配置层转成运行参数
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.yaml#L1)：`meta:`
- [L7](../../../betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.yaml#L7)：`factor_spec:`
- [L19](../../../betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.yaml#L19)：`weight:`
- [L25](../../../betalens-factor/tdx/RSI_SLOW/factor_RSI_SLOW.yaml#L25)：`run:`

<a id="file-6a9c53d40024"></a>
## betalens-factor/tdx/class_tdx.yaml

[打开源码](../../../betalens-factor/tdx/class_tdx.yaml) · 17 行 · 说明来源：文件族规则

- **作用**：因子类别发现元数据
- **输入**：维护者填写的参数
- **输出**：类别信息；不能替代完整因子运行 YAML
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/tdx/class_tdx.yaml#L1)：`class: tdx`
- [L2](../../../betalens-factor/tdx/class_tdx.yaml#L2)：`template_module: factor_template_tdx`
- [L3](../../../betalens-factor/tdx/class_tdx.yaml#L3)：`source: 通达信(TDX)指标公式：RSI 快慢线 + 侯神吸筹能量`
- [L4](../../../betalens-factor/tdx/class_tdx.yaml#L4)：`defaults:`

<a id="file-6088c1e0f187"></a>
## betalens-factor/tdx/factor_template_tdx.py

[打开源码](../../../betalens-factor/tdx/factor_template_tdx.py) · 122 行 · 说明来源：人工文件说明

- **作用**：通达信公式算子及管线适配
- **输入**：日频宽表和算子参数
- **输出**：技术指标/因子与运行结果
- **副作用/维护重点**：平滑/窗口定义不能随意替换为近似函数

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from dataclasses import dataclass, field
from factor_template import FactorPipeline as CorePipeline, FactorSpec as CoreFactorSpec
from pathlib import Path
from typing import Any, Callable
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [FactorSpec](../../../betalens-factor/tdx/factor_template_tdx.py#L50) | class FactorSpec() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：name: str; inputs: dict[str, str]; compute: Callable[..., Any]; direction: str = 'positive'; compute_kwargs: dict[str, Any] = field(default_factory=dict); table_name: str = DB_TABLE; use_industry: bool = False; use_mktcap: bool = False; industry_scheme: str = '申万一级行业'; index_co…（完整内容见 inventory.json/源码） |
| [FactorPipeline](../../../betalens-factor/tdx/factor_template_tdx.py#L69) | class FactorPipeline() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [FactorPipeline.__init__](../../../betalens-factor/tdx/factor_template_tdx.py#L70) | __init__(self, spec: FactorSpec) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [FactorPipeline.run](../../../betalens-factor/tdx/factor_template_tdx.py#L73) | run(self, *args, **kwargs) | 无返回注解；return: CorePipeline(core_spec).run(*args, **kwargs) | 无 docstring，需阅读函数体 |
| [SMA](../../../betalens-factor/tdx/factor_template_tdx.py#L90) | SMA(x, n, m=1) | 无返回注解；return: x.ewm(alpha=m / n, adjust=False).mean() | TDX SMA(X,N,M)：加权移动平均，权重 alpha=M/N。 |
| [EMA](../../../betalens-factor/tdx/factor_template_tdx.py#L95) | EMA(x, n) | 无返回注解；return: x.ewm(span=n, adjust=False).mean() | TDX EMA(X,N)：指数移动平均。 |
| [MA](../../../betalens-factor/tdx/factor_template_tdx.py#L100) | MA(x, n) | 无返回注解；return: x.rolling(n).mean() | TDX MA(X,N)：N 周期简单移动平均。 |
| [REF](../../../betalens-factor/tdx/factor_template_tdx.py#L105) | REF(x, n=1) | 无返回注解；return: x.shift(n) | TDX REF(X,n)：n 周期前的值。 |
| [LLV](../../../betalens-factor/tdx/factor_template_tdx.py#L110) | LLV(x, n) | 无返回注解；return: x.rolling(n).min() | TDX LLV(X,N)：N 周期内最低值。 |
| [HHV](../../../betalens-factor/tdx/factor_template_tdx.py#L115) | HHV(x, n) | 无返回注解；return: x.rolling(n).max() | TDX HHV(X,N)：N 周期内最高值。 |
| [clean_inf](../../../betalens-factor/tdx/factor_template_tdx.py#L120) | clean_inf(x) | 无返回注解；return: x.replace([float('inf'), float('-inf')], float('nan')) | 把 ±inf 置为 NaN（算子末尾统一调用，防止除零污染）。 |

<a id="file-bcc6e681028c"></a>
## betalens-factor/tools/eventstudy/eventstudy.yaml

[打开源码](../../../betalens-factor/tools/eventstudy/eventstudy.yaml) · 15 行 · 说明来源：文件族规则

- **作用**：运行/构建声明式配置
- **输入**：维护者填写的参数
- **输出**：由对应读取器解释的配置对象
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L1](../../../betalens-factor/tools/eventstudy/eventstudy.yaml#L1)：`eventstudy:`

<a id="file-44dfa050b053"></a>
## betalens-factor/tools/eventstudy/run_eventstudy.py

[打开源码](../../../betalens-factor/tools/eventstudy/run_eventstudy.py) · 190 行 · 说明来源：人工文件说明

- **作用**：事件研究 CLI
- **输入**：事件 YAML、文件与运行参数
- **输出**：事件分析与导出产物
- **副作用/维护重点**：文件读取、数据库查询和结果导出

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.datafeed import Datafeed
from betalens.eventstudy.eventstudy import EventStudy
from betalens.factor.config import load_yaml_config, resolve_path, section
from pathlib import Path
from typing import Any
import argparse
import pandas as pd
import re
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_params](../../../betalens-factor/tools/eventstudy/run_eventstudy.py#L27) | load_params(config_path: str &#124; Path=PARAMS_FILE) -&gt; tuple[dict[str, Any], Path] | tuple[dict[str, Any], Path] | 无 docstring，需阅读函数体 |
| [parse_codes](../../../betalens-factor/tools/eventstudy/run_eventstudy.py#L33) | parse_codes(value: Any) -&gt; str &#124; list[str] | str &#124; list[str] | 无 docstring，需阅读函数体 |
| [parse_int_list](../../../betalens-factor/tools/eventstudy/run_eventstudy.py#L47) | parse_int_list(value: Any) -&gt; list[int] | list[int] | 无 docstring，需阅读函数体 |
| [build_holding_periods](../../../betalens-factor/tools/eventstudy/run_eventstudy.py#L55) | build_holding_periods(params: dict[str, Any]) -&gt; dict[str, list[int]] | dict[str, list[int]] | 无 docstring，需阅读函数体 |
| [read_events](../../../betalens-factor/tools/eventstudy/run_eventstudy.py#L62) | read_events(path: Path) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [_sheet_name](../../../betalens-factor/tools/eventstudy/run_eventstudy.py#L83) | _sheet_name(prefix: str, code: str, used: set[str]) -&gt; str | str | Create a unique Excel-safe sheet name for per-code comparison output. |
| [main](../../../betalens-factor/tools/eventstudy/run_eventstudy.py#L97) | main() -&gt; int | int | 无 docstring，需阅读函数体 |

