# core_factor：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-e157395e4dc9"></a>
## betalens/factor/__init__.py

[打开源码](../../../betalens/factor/__init__.py) · 5 行 · 说明来源：文件族规则

- **作用**：包导出/包标识
- **输入**：import 请求
- **输出**：模块导出与符号；见静态 imports
- **副作用/维护重点**：初始化可能导入子模块；__all__ 与真实导出需结合源码阅读

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .factor import *
from .mining import *
from .preprocessing import *
from .signal import *
```

<a id="file-fc0b9eb70b7f"></a>
## betalens/factor/config.py

[打开源码](../../../betalens/factor/config.py) · 178 行 · 说明来源：人工文件说明

- **作用**：完整 YAML 校验和参数映射
- **输入**：YAML 路径、配置分节
- **输出**：FactorSpec kwargs、run kwargs、绝对输出路径
- **副作用/维护重点**：write_yaml_config 写文件；路径相对 YAML 目录

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
import yaml
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [ConfigError](../../../betalens/factor/config.py#L9) | class ConfigError(ValueError) | 类定义；构造/属性见方法与字段 | Raised when a YAML parameter file is missing required structure. |
| [load_yaml_config](../../../betalens/factor/config.py#L13) | load_yaml_config(path: str &#124; Path, *, required_sections: Sequence[str]=()) -&gt; dict[str, Any] | dict[str, Any] | Load a YAML config file and validate required top-level sections. |
| [write_yaml_config](../../../betalens/factor/config.py#L32) | write_yaml_config(path: str &#124; Path, config: Mapping[str, Any]) -&gt; Path | Path | Write a complete runtime YAML config copy. |
| [section](../../../betalens/factor/config.py#L48) | section(config: Mapping[str, Any], name: str, *, context: str='config') -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [require_keys](../../../betalens/factor/config.py#L55) | require_keys(mapping: Mapping[str, Any], keys: Sequence[str], *, context: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [resolve_path](../../../betalens/factor/config.py#L61) | resolve_path(value: str &#124; Path, base_dir: str &#124; Path) -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [resolve_run_output_dir](../../../betalens/factor/config.py#L68) | resolve_run_output_dir(config: Mapping[str, Any], config_path: str &#124; Path) -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [factor_metadata](../../../betalens/factor/config.py#L74) | factor_metadata(config: Mapping[str, Any]) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [run_parameters](../../../betalens/factor/config.py#L89) | run_parameters(config: Mapping[str, Any], config_path: str &#124; Path) -&gt; dict[str, Any] | dict[str, Any] | Return concrete FactorPipeline.run kwargs from a complete factor YAML. |
| [factor_spec_options](../../../betalens/factor/config.py#L133) | factor_spec_options(config: Mapping[str, Any], config_path: str &#124; Path) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |

<a id="file-a7d119e96fb0"></a>
## betalens/factor/factor.py

[打开源码](../../../betalens/factor/factor.py) · 1466 行 · 说明来源：人工文件说明

- **作用**：可交易池、单/双/多特征分组与权重
- **输入**：查询日期/因子长表/标签/分组参数
- **输出**：股票池、带标签 MultiIndex 表、权重宽表
- **副作用/维护重点**：部分函数查库，分组和权重可离线；标签与同值规则是契约

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from betalens.datafeed import Datafeed
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [get_tradable_pool](../../../betalens/factor/factor.py#L53) | get_tradable_pool(date_list, include_abnormal=False) | 无返回注解；return: (date_ranges, code_ranges) | 获取可交易股票池（基于 trade_status 表） 注意不要在因子逻辑中引入是否停牌的未来函数。这要根据具体的因子而定： 如果属于盘后计算的因子，则可以假设剔除停牌股票（仅 value==1 正常交易），因为停牌状态在当日是已知的，不应影响因子计算。这会对策略产生流动性的影响，要注意。 Args: date_list: 日期列表 include_abnormal: 是否将异常交易状态（停牌等）的股票纳入计算 - False（默认）：仅纳入正常交易（value==1）的股票，保持原行为 - True：纳入所有已上市股票（value!=-1，即正常 + 异常停牌）， 仅排除首次正常交易日之前（…（完整内容见 inventory.json/源码） |
| [pre_query_characteristic_data](../../../betalens/factor/factor.py#L102) | pre_query_characteristic_data(date_list, metric, time_tolerance=24 * 2 * 365, table_name='fundamentals', date_ranges=None, code_ranges=None, include_abnormal=False) | 无返回注解；return: all_results | 批量预查询公司特征数据，生成符合特征排序函数要求的DataFrame 该函数先通过 get_tradable_pool 验证股票交易状态，获取可交易股票池， 然后批量查询多个调仓日期的公司特征数据，返回格式化的DataFrame， 可直接用于 single_characteristic、double_characteristic、multi_characteristic 等排序函数。 Args: date_list: 调仓日期列表，每个元素为 date 或 datetime 对象 metric: 公司特征指标名称（字符串），如 "股息率(报告期)"、"市值"、"账面市值比" time_tole…（完整内容见 inventory.json/源码） |
| [single_characteristic](../../../betalens/factor/factor.py#L214) | single_characteristic(pre_queried_data, metric, quantiles, grouping_mode='equal_count') | 无返回注解；return: labeled_pool | 单特征分组打标签 Args: pre_queried_data: DataFrame，包含所有日期的公司特征数据 必需列：input_ts, code, {metric}（特征值列） 可选列：datetime, diff_hours, name metric: 公司特征指标名称 quantiles: 分位数字典，如 {"股息率(报告期)": 10} grouping_mode: ''equal_count'' 或 ''value''。 ''equal_count'' 优先保证各组股票数相当且严格生成目标组数， 相同特征值必要时可能分到不同组；有效股票数少于目标组数时报错。 ''value'' …（完整内容见 inventory.json/源码） |
| [single_characteristic.single_sort](../../../betalens/factor/factor.py#L257) | single_sort(df, keys, quantile_dict, input_ts=None) | 无返回注解；return: df | 对单个截面分组，标签始终按特征值从低到高连续编号。 |
| [double_characteristic](../../../betalens/factor/factor.py#L333) | double_characteristic(pre_queried_data1, pre_queried_data2, metric1, metric2, quantiles1, quantiles2, sort_method='dependent') | 无返回注解；return: labeled_pool | 双特征分组打标签（Double Characteristic Sort） 支持独立排序（Independent Sort）和条件排序（Dependent Sort） Args: pre_queried_data1: DataFrame，包含主特征数据 必需列：input_ts, code, {metric1}（主特征值列） 可选列：datetime, diff_hours, name pre_queried_data2: DataFrame，包含次特征数据 必需列：input_ts, code, {metric2}（次特征值列） 可选列：datetime, diff_hours, name m…（完整内容见 inventory.json/源码） |
| [double_characteristic.independent_double_sort](../../../betalens/factor/factor.py#L377) | independent_double_sort(df, char1, char2, quantile_dict1, quantile_dict2) | 无返回注解；return: df | 独立排序：分别对两个特征独立分组 对全部股票分别按两个特征进行分组，然后取交集 |
| [double_characteristic.dependent_double_sort](../../../betalens/factor/factor.py#L408) | dependent_double_sort(df, char1, char2, quantile_dict1, quantile_dict2) | 无返回注解；return: df | 条件排序：先按主特征分组，再在每组内按次特征分组 |
| [_recursive_multi_characteristic_sort](../../../betalens/factor/factor.py#L471) | _recursive_multi_characteristic_sort(df, characteristics, current_index=0, parent_group=None) | 无返回注解；return: df; _recursive_multi_characteristic_sort(df, characteristics, current_index + 1, label_col); _recursive_multi_characteristic_sort(df, characteristics, current_index + 1, None) | 递归多特征分组排序核心函数 Args: df: 当前数据框 characteristics: 特征配置列表，每个元素为 {'name': str, 'quantiles': int, 'method': 'independent'/'dependent'} current_index: 当前处理的特征索引 parent_group: 父组标签（用于dependent排序） Returns: df: 添加了标签列的数据框 |
| [multi_characteristic](../../../betalens/factor/factor.py#L545) | multi_characteristic(pre_queried_data_list, characteristics) | 无返回注解；return: labeled_pool | 多特征分组打标签（Multi-Characteristic Sort） 支持递归的独立排序和条件排序混合 Args: pre_queried_data_list: DataFrame列表，每个DataFrame包含对应特征的数据 列表顺序应与characteristics配置列表的顺序一致 每个DataFrame必需列：input_ts, code, {metric}（特征值列） 可选列：datetime, diff_hours, name characteristics: 特征配置列表，每个元素为字典： { 'name': str, # 特征名称，如 '市值' 'quantiles': in…（完整内容见 inventory.json/源码） |
| [get_single_factor_weight](../../../betalens/factor/factor.py#L618) | get_single_factor_weight(labeled_pool, params) | 无返回注解；return: weights | 根据单特征标签生成多空因子权重（构建因子） 本函数基于公司特征分组结果，构建多空组合权重，该权重对应的收益率即为因子收益率。 Args: labeled_pool: 带标签的特征池DataFrame（由single_characteristic生成） params: 参数字典，包含： - 'factor_key': 特征名称 - 'mode': 'classic-long-short' 或 'freeplay' - 'long': 做多标签列表（freeplay模式） - 'short': 做空标签列表（freeplay模式） - 'grouping_mode': ''equal_count'…（完整内容见 inventory.json/源码） |
| [get_single_factor_weight._selector_list](../../../betalens/factor/factor.py#L650) | _selector_list(value) | 无返回注解；return: []; [value]; list(value) | 无 docstring，需阅读函数体 |
| [get_single_factor_weight._normalize_selector](../../../betalens/factor/factor.py#L657) | _normalize_selector(value) | 无返回注解；return: int(selector); selector; int(value) | 无 docstring，需阅读函数体 |
| [get_single_factor_weight._resolved_selectors](../../../betalens/factor/factor.py#L688) | _resolved_selectors(selectors, group) | 无返回注解；return: []; [(selector, max_label if selector == 'max' else min_label if selector == 'min' else selector) for selector in selectors] | 无 docstring，需阅读函数体 |
| [get_single_factor_weight._mapping_value](../../../betalens/factor/factor.py#L700) | _mapping_value(mapping, selector, label, default) | 无返回注解；return: mapping[key]; default | 无 docstring，需阅读函数体 |
| [get_single_factor_weight.f1](../../../betalens/factor/factor.py#L706) | f1(group) | 无返回注解；return: group | 经典多空始终按当期实际标签极值选组，兼容动态组数。 |
| [get_single_factor_weight.f2](../../../betalens/factor/factor.py#L720) | f2(group) | 无返回注解；return: group | 无 docstring，需阅读函数体 |
| [get_single_factor_weight.normalize_row](../../../betalens/factor/factor.py#L877) | normalize_row(row) | 无返回注解；return: row | 无 docstring，需阅读函数体 |
| [describe_labeled_pool](../../../betalens/factor/factor.py#L896) | describe_labeled_pool(labeled_pool) | 无返回注解；return: pivot | 描述打标签后的特征池统计信息 Args: labeled_pool: 带标签的特征池DataFrame Returns: pivot: 透视表，包含每个标签组的样本数和均值 |
| [describe_double_labeled_pool](../../../betalens/factor/factor.py#L927) | describe_double_labeled_pool(labeled_pool) | 无返回注解；return: (count_pivot, mean_pivot1, mean_pivot2) | 描述双特征打标签后的特征池统计信息 Args: labeled_pool: 带双标签的特征池DataFrame（由double_characteristic生成） Returns: count_pivot: 各组合的样本数统计 mean_pivot1: 主特征在各组合中的均值 mean_pivot2: 次特征在各组合中的均值 |
| [describe_multi_labeled_pool](../../../betalens/factor/factor.py#L990) | describe_multi_labeled_pool(labeled_pool, max_display_dims=2) | 无返回注解；return: {'count_pivot': count_pivot, 'mean_pivots': mean_pivots, 'characteristic_info': characteristics, 'display_characteristics': [f['name'] for f in display_chars]} | 描述多特征打标签后的特征池统计信息 Args: labeled_pool: 带多标签的特征池DataFrame（由multi_characteristic生成） max_display_dims: 最大显示维度（默认2，即显示前2个特征的交叉统计） Returns: stats_dict: 包含统计信息的字典 - 'count_pivot': 各组合的样本数统计（前max_display_dims个特征） - 'mean_pivots': 各特征在各组合中的均值（字典，key为特征名） - 'characteristic_info': 特征配置信息 |
| [get_multi_factor_weight](../../../betalens/factor/factor.py#L1068) | get_multi_factor_weight(labeled_pool, params) | 无返回注解；return: weights | 根据多特征标签生成多空因子权重（构建因子） 本函数基于公司特征分组结果，构建多空组合权重，该权重对应的收益率即为因子收益率。 Args: labeled_pool: 带多标签的特征池DataFrame（由multi_characteristic生成） params: 参数字典，包含： - 'mode': 'classic-long-short' 或 'freeplay' - 'long_combinations': 做多组合列表，如 [(0,4,2), (1,4,2)] 表示各特征标签组合 - 'short_combinations': 做空组合列表 - 对于 'classic-long-sh…（完整内容见 inventory.json/源码） |
| [get_multi_factor_weight.assign_weights](../../../betalens/factor/factor.py#L1094) | assign_weights(group) | 无返回注解；return: group | 无 docstring，需阅读函数体 |
| [get_multi_factor_weight.normalize_row](../../../betalens/factor/factor.py#L1276) | normalize_row(row) | 无返回注解；return: row | 无 docstring，需阅读函数体 |
| [get_double_factor_weight](../../../betalens/factor/factor.py#L1296) | get_double_factor_weight(labeled_pool, params) | 无返回注解；return: weights | 根据双特征标签生成多空因子权重（构建因子） 本函数基于公司特征分组结果，构建多空组合权重，该权重对应的收益率即为因子收益率。 Args: labeled_pool: 带双标签的特征池DataFrame（由double_characteristic生成） params: 参数字典，包含： - factor_key1: 主特征名称 - factor_key2: 次特征名称 - mode: 'classic-long-short' 或 'freeplay' - long_combinations: 做多组合列表，如 [(0,4), (1,4)] 表示主特征0/1组且次特征4组 - short_com…（完整内容见 inventory.json/源码） |
| [get_double_factor_weight.assign_weights](../../../betalens/factor/factor.py#L1321) | assign_weights(group) | 无返回注解；return: group | 无 docstring，需阅读函数体 |
| [get_double_factor_weight.normalize_row](../../../betalens/factor/factor.py#L1448) | normalize_row(row) | 无返回注解；return: row | 无 docstring，需阅读函数体 |

<a id="file-65e954bccf78"></a>
## betalens/factor/mining.py

[打开源码](../../../betalens/factor/mining.py) · 2422 行 · 说明来源：人工文件说明

- **作用**：参数挖掘调度和窗口评价
- **输入**：parameter_space.yaml、performance.yaml、MiningSpec hook
- **输出**：MiningResult 和逐因子结果
- **副作用/维护重点**：查库、进程/线程、缓存、日志和产物；先限制搜索范围

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_parameters import aggregate_mining_factors, mining_parameter_limits, mining_parameter_specs
from betalens.backtest import BacktestBase
from betalens.datafeed import Datafeed
from betalens.datafeed import get_absolute_trade_days
from betalens.factor.factor import get_single_factor_weight, single_characteristic
from betalens.factor.mining_audit import FactorMiningResult, MiningResult, MiningTask
from betalens.factor.mining_cache import CacheRequest, MiningCache
from betalens.factor.mining_cache import MiningCache
from betalens.factor.mining_optuna import create_coarse_study, seed_study_with_results
from betalens.factor.mining_optuna import create_fine_grid_study, detect_boundary_pressure, expand_parameter_specs, generate_fine_candidates, generate_perturbation_candidates
from betalens.factor.mining_optuna import suggest_params
from betalens.factor.mining_optuna import tell_trial
from betalens.factor.preprocessing import neutralize_factor, standardize_factor, winsorize_factor
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence
import hashlib
import importlib
import json
import logging
import math
import multiprocessing as mp
import numpy as np
import optuna
import os
import pandas as pd
import psutil
import sys
import threading
import time
import traceback
import uuid
import warnings
import yaml
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_ChineseLogFormatter](../../../betalens/factor/mining.py#L41) | class _ChineseLogFormatter(logging.Formatter) | 类定义；构造/属性见方法与字段 | Render logging metadata in Chinese for terminal and audit output. |
| [_ChineseLogFormatter.format](../../../betalens/factor/mining.py#L52) | format(self, record: logging.LogRecord) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_stage_name](../../../betalens/factor/mining.py#L63) | _stage_name(stage: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_mode_name](../../../betalens/factor/mining.py#L76) | _mode_name(mode: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_backend_name](../../../betalens/factor/mining.py#L83) | _backend_name(backend: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_engine_name](../../../betalens/factor/mining.py#L87) | _engine_name(engine: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_metric_name](../../../betalens/factor/mining.py#L91) | _metric_name(metric: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_aggregate_name](../../../betalens/factor/mining.py#L102) | _aggregate_name(aggregate: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_direction_name](../../../betalens/factor/mining.py#L112) | _direction_name(direction: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_yes_no](../../../betalens/factor/mining.py#L116) | _yes_no(value: Any) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_window_description](../../../betalens/factor/mining.py#L120) | _window_description(window_id: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_operation_name](../../../betalens/factor/mining.py#L128) | _operation_name(operation: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_heartbeat_fields](../../../betalens/factor/mining.py#L139) | _heartbeat_fields(fields: Mapping[str, Any]) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_configure_mining_logging](../../../betalens/factor/mining.py#L160) | _configure_mining_logging(log_path: Path, level: int, *, task_logs: bool, heartbeat_seconds: float) -&gt; Path | Path | Attach one live console and one run-specific audit file handler. |
| [_close_mining_logging](../../../betalens/factor/mining.py#L192) | _close_mining_logging() -&gt; None | None | 无 docstring，需阅读函数体 |
| [_configure_worker_logging](../../../betalens/factor/mining.py#L203) | _configure_worker_logging(log_queue, level: int, task_logs: bool, heartbeat_seconds: float) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_elapsed](../../../betalens/factor/mining.py#L219) | _elapsed(started: float) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_human_duration](../../../betalens/factor/mining.py#L223) | _human_duration(seconds: float) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_human_bytes](../../../betalens/factor/mining.py#L234) | _human_bytes(size: int) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_progress_due](../../../betalens/factor/mining.py#L243) | _progress_due(completed: int, total: int, *, updates: int=20) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [_heartbeat](../../../betalens/factor/mining.py#L251) | _heartbeat(operation: str, **fields) | 无返回注解；return: None | Emit liveness records while a single blocking operation is running. |
| [_heartbeat.report](../../../betalens/factor/mining.py#L261) | report() -&gt; None | None | 无 docstring，需阅读函数体 |
| [MiningSpec](../../../betalens/factor/mining.py#L280) | class MiningSpec() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：factor_spec: Any; execution_mode: Literal['precomputed', 'rolling_fit'] = 'precomputed'; fit_window: Callable &#124; None = None; window_transform: Callable &#124; None = None; warmup_days: int &#124; Callable = 30 |
| [MiningWindow](../../../betalens/factor/mining.py#L289) | class MiningWindow() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：window_id: str; start: str; end: str; length: int; step: int |
| [MiningData](../../../betalens/factor/mining.py#L298) | class MiningData() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：inputs: dict[str, pd.DataFrame]; price: pd.DataFrame; execution_price: pd.DataFrame; trade_status: pd.DataFrame &#124; None; pit: dict[Any, set[str]] &#124; None; universe: list[str]; industry_by_scheme: dict[str, pd.DataFrame] = field(default_factory=dict); cache_manifest_path: str &#124; No…（完整内容见 inventory.json/源码） |
| [_load_yaml](../../../betalens/factor/mining.py#L309) | _load_yaml(path: str &#124; Path) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [validate_parameter_specs](../../../betalens/factor/mining.py#L323) | validate_parameter_specs(parameters: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_validate_window_config](../../../betalens/factor/mining.py#L358) | _validate_window_config(config: Mapping[str, Any]) -&gt; tuple[list[int], list[int]] | tuple[list[int], list[int]] | 无 docstring，需阅读函数体 |
| [_candidate_id](../../../betalens/factor/mining.py#L369) | _candidate_id(factor_id: str, params: Mapping[str, Any]) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_import_spec](../../../betalens/factor/mining.py#L374) | _import_spec(module_name: str, params: Mapping[str, Any]) -&gt; MiningSpec | MiningSpec | 无 docstring，需阅读函数体 |
| [_warmup_days](../../../betalens/factor/mining.py#L399) | _warmup_days(spec: MiningSpec, params: Mapping[str, Any]) -&gt; int | int | 无 docstring，需阅读函数体 |
| [_system_memory](../../../betalens/factor/mining.py#L409) | _system_memory() -&gt; tuple[int &#124; None, int &#124; None] | tuple[int &#124; None, int &#124; None] | 无 docstring，需阅读函数体 |
| [_frame_bytes](../../../betalens/factor/mining.py#L418) | _frame_bytes(value: pd.DataFrame &#124; None) -&gt; int | int | 无 docstring，需阅读函数体 |
| [_effective_workers](../../../betalens/factor/mining.py#L424) | _effective_workers(requested: int, data: MiningData, ratio: float) -&gt; int | int | 无 docstring，需阅读函数体 |
| [_fetch_daily_wide](../../../betalens/factor/mining.py#L437) | _fetch_daily_wide(metric: str, universe: Sequence[str], start: str, end: str, table: str) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_align_daily_wides](../../../betalens/factor/mining.py#L478) | _align_daily_wides(wides: Mapping[str, pd.DataFrame]) -&gt; dict[str, pd.DataFrame] | dict[str, pd.DataFrame] | 无 docstring，需阅读函数体 |
| [_fetch_industry_wide](../../../betalens/factor/mining.py#L500) | _fetch_industry_wide(scheme: str, universe: Sequence[str], dates: Sequence[Any], index: pd.DatetimeIndex) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_fetch_trade_status](../../../betalens/factor/mining.py#L549) | _fetch_trade_status(universe: Sequence[str], dates: Sequence[Any]) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_build_pit](../../../betalens/factor/mining.py#L594) | _build_pit(dates: Sequence[Any], index_code: str) -&gt; dict[Any, set[str]] | dict[Any, set[str]] | 无 docstring，需阅读函数体 |
| [_mask_pit](../../../betalens/factor/mining.py#L628) | _mask_pit(wide: pd.DataFrame, pit: Mapping[Any, set[str]] &#124; None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_pit_fingerprint](../../../betalens/factor/mining.py#L640) | _pit_fingerprint(pit: Mapping[Any, set[str]] &#124; None) -&gt; str &#124; None | str &#124; None | 无 docstring，需阅读函数体 |
| [_cache_signature](../../../betalens/factor/mining.py#L653) | _cache_signature(spec: MiningSpec, start: str, end: str, performance: Mapping[str, Any], universe: Sequence[str], pit: Mapping[Any, set[str]] &#124; None) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_pit_from_frame](../../../betalens/factor/mining.py#L681) | _pit_from_frame(value: pd.DataFrame &#124; None) -&gt; dict[Any, set[str]] &#124; None | dict[Any, set[str]] &#124; None | 无 docstring，需阅读函数体 |
| [_slice_frame](../../../betalens/factor/mining.py#L687) | _slice_frame(value: pd.DataFrame &#124; None, start: Any, end: Any) -&gt; pd.DataFrame &#124; None | pd.DataFrame &#124; None | 无 docstring，需阅读函数体 |
| [_slice_data](../../../betalens/factor/mining.py#L695) | _slice_data(data: MiningData, start: Any, end: Any) -&gt; MiningData | MiningData | 无 docstring，需阅读函数体 |
| [_load_cached_data](../../../betalens/factor/mining.py#L719) | _load_cached_data(manifest_path: str &#124; Path) -&gt; MiningData | MiningData | 无 docstring，需阅读函数体 |
| [_fetch_data](../../../betalens/factor/mining.py#L760) | _fetch_data(spec: MiningSpec, span: tuple[str, str], performance: Mapping[str, Any], params: Mapping[str, Any], cache_dir: Path) -&gt; MiningData | MiningData | 无 docstring，需阅读函数体 |
| [_fetch_data.builder](../../../betalens/factor/mining.py#L793) | builder() | 无返回注解；return: payload | 无 docstring，需阅读函数体 |
| [_windows](../../../betalens/factor/mining.py#L866) | _windows(span: tuple[str, str], lengths: Sequence[int], steps: Sequence[int]) -&gt; list[MiningWindow] | list[MiningWindow] | 无 docstring，需阅读函数体 |
| [_sample_days](../../../betalens/factor/mining.py#L880) | _sample_days(days: Sequence[Any], frequency: str) -&gt; list[Any] | list[Any] | 无 docstring，需阅读函数体 |
| [_signal_pairs](../../../betalens/factor/mining.py#L891) | _signal_pairs(start: str, end: str, frequency: str, trade_days: Sequence[Any]) -&gt; list[tuple[Any, Any]] | list[tuple[Any, Any]] | 无 docstring，需阅读函数体 |
| [_weights_on_rebalance](../../../betalens/factor/mining.py#L899) | _weights_on_rebalance(weights: pd.DataFrame, pairs: Sequence[tuple[Any, Any]]) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_wide_to_long](../../../betalens/factor/mining.py#L908) | _wide_to_long(wide: pd.DataFrame, metric: str, signal_dates: Sequence[Any]) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_filter_pit](../../../betalens/factor/mining.py#L918) | _filter_pit(value: pd.DataFrame, pit: Mapping[Any, set[str]] &#124; None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_preprocess](../../../betalens/factor/mining.py#L925) | _preprocess(value: pd.DataFrame, factor: Any, signal_dates: Sequence[Any], data: MiningData) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_groups](../../../betalens/factor/mining.py#L956) | _groups(factor: Any, quantiles: int) -&gt; tuple[list[Any], list[Any]] | tuple[list[Any], list[Any]] | 无 docstring，需阅读函数体 |
| [_build_weights](../../../betalens/factor/mining.py#L966) | _build_weights(factor_wide: pd.DataFrame, spec: MiningSpec, params: Mapping[str, Any], data: MiningData, start: str, end: str, frequency: str) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [metrics_from_nav](../../../betalens/factor/mining.py#L998) | metrics_from_nav(nav: pd.Series &#124; pd.DataFrame) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_turnover](../../../betalens/factor/mining.py#L1014) | _turnover(weights: pd.DataFrame &#124; None) -&gt; float | float | 无 docstring，需阅读函数体 |
| [_daily_last](../../../betalens/factor/mining.py#L1025) | _daily_last(value: pd.DataFrame) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_rank_ic](../../../betalens/factor/mining.py#L1031) | _rank_ic(factor: pd.DataFrame, price: pd.DataFrame, pairs: Sequence[tuple[Any, Any]]) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_vector_nav](../../../betalens/factor/mining.py#L1055) | _vector_nav(weights: pd.DataFrame, price: pd.DataFrame) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [_exact_nav](../../../betalens/factor/mining.py#L1067) | _exact_nav(weights: pd.DataFrame, data: MiningData, spec: MiningSpec, amount: float, tolerance: int) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [_call_fit](../../../betalens/factor/mining.py#L1080) | _call_fit(callback: Callable, data: MiningData, params: Mapping[str, Any], window: MiningWindow, context: Mapping[str, Any]) | 无返回注解；return: callback(data, params, window, context); callback(data, params, window); callback(data, params) | 无 docstring，需阅读函数体 |
| [_evaluate_candidate](../../../betalens/factor/mining.py#L1090) | _evaluate_candidate(module: str, factor_id: str, params: Mapping[str, Any], stage: str, windows: Sequence[MiningWindow], data: MiningData, evaluation: Mapping[str, Any]) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_safe_evaluate](../../../betalens/factor/mining.py#L1259) | _safe_evaluate(module: str, factor_id: str, params: Mapping[str, Any], stage: str, windows: Sequence[MiningWindow], data: MiningData, evaluation: Mapping[str, Any]) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_initialize_worker](../../../betalens/factor/mining.py#L1277) | _initialize_worker(data: MiningData &#124; None, cache_manifest_path: str &#124; None, log_queue, log_level: int, task_logs: bool, heartbeat_seconds: float) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_worker_evaluate](../../../betalens/factor/mining.py#L1292) | _worker_evaluate(module: str, factor_id: str, params: Mapping[str, Any], stage: str, windows: Sequence[MiningWindow], evaluation: Mapping[str, Any]) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_summary](../../../betalens/factor/mining.py#L1298) | _summary(frame: pd.DataFrame, selection: Mapping[str, Any]) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_evaluate_stage](../../../betalens/factor/mining.py#L1393) | _evaluate_stage(module: str, factor_id: str, candidates: Sequence[Mapping[str, Any]], stage: str, windows: Sequence[MiningWindow], data: MiningData, evaluation: Mapping[str, Any], runtime: Mapping[str, Any], task: MiningTask, candidate_metadata: Mapping[str, Mapping[str, Any]] &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_evaluate_stage.report](../../../betalens/factor/mining.py#L1449) | report(completed: int, index: int, rows: list[dict[str, Any]], candidate_started: float) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_ask_candidates](../../../betalens/factor/mining.py#L1560) | _ask_candidates(study, parameter_specs: Mapping[str, Mapping[str, Any]], count: int, *, complete_for_sampling: bool=False) | 无返回注解；return: (trials, candidates) | 无 docstring，需阅读函数体 |
| [_tell_candidates](../../../betalens/factor/mining.py#L1579) | _tell_candidates(study, trials, candidates, factor_id: str, summary: pd.DataFrame) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_grid_specs](../../../betalens/factor/mining.py#L1591) | _grid_specs(parameter_specs: Mapping[str, Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]) -&gt; tuple[dict[str, list[Any]], dict[str, dict[str, Any]]] | tuple[dict[str, list[Any]], dict[str, dict[str, Any]]] | 无 docstring，需阅读函数体 |
| [_alpha_generation_options](../../../betalens/factor/mining.py#L1612) | _alpha_generation_options(parameters_config: Mapping[str, Any]) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_resolve_alpha_configs](../../../betalens/factor/mining.py#L1625) | _resolve_alpha_configs(parameters_config: Mapping[str, Any], factors: Any) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_selection_reason](../../../betalens/factor/mining.py#L1683) | _selection_reason(status: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_summary_records](../../../betalens/factor/mining.py#L1691) | _summary_records(frame: pd.DataFrame, stage: str) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_dedupe_candidates](../../../betalens/factor/mining.py#L1703) | _dedupe_candidates(factor_id: str, candidates: Sequence[Mapping[str, Any]]) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_evaluate_with_reuse](../../../betalens/factor/mining.py#L1718) | _evaluate_with_reuse(module: str, factor_id: str, candidates: Sequence[Mapping[str, Any]], stage: str, windows: Sequence[MiningWindow], data: MiningData, evaluation: Mapping[str, Any], runtime: Mapping[str, Any], task: MiningTask, *, previous: pd.DataFrame &#124; None=None, candidate_metadata: Mapping[str, Mapping[str, Any]] &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_run_sampled_stage](../../../betalens/factor/mining.py#L1777) | _run_sampled_stage(module: str, factor_id: str, parameter_specs: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any], stage: str, direction: str, windows: Sequence[MiningWindow], data: MiningData, evaluation: Mapping[str, Any], runtime: Mapping[str, Any], selection: Mapping[str, Any], task: MiningTask, *, previous: pd.DataFrame &#124; None=None) -&gt; tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame] | 无 docstring，需阅读函数体 |
| [_top_parameter_rows](../../../betalens/factor/mining.py#L1882) | _top_parameter_rows(summary: pd.DataFrame, parameter_specs: Mapping[str, Mapping[str, Any]], count: int) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_stability_results](../../../betalens/factor/mining.py#L1897) | _stability_results(selected: pd.DataFrame, stability_summary: pd.DataFrame, planned_by_parent: Mapping[str, int], config: Mapping[str, Any], direction: str) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_run_factor](../../../betalens/factor/mining.py#L1946) | _run_factor(task: MiningTask, factor_id: str, factor_config: Mapping[str, Any], parameters_config: Mapping[str, Any], performance: Mapping[str, Any], evaluation: Mapping[str, Any], windows: Sequence[MiningWindow]) -&gt; FactorMiningResult | FactorMiningResult | 无 docstring，需阅读函数体 |
| [_run_factor.broad_results](../../../betalens/factor/mining.py#L2032) | broad_results() -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_validate_performance_config](../../../betalens/factor/mining.py#L2248) | _validate_performance_config(performance: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_validate_search_config](../../../betalens/factor/mining.py#L2267) | _validate_search_config(parameters: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [run_mining](../../../betalens/factor/mining.py#L2303) | run_mining(parameter_config_path: str &#124; Path, performance_config_path: str &#124; Path) -&gt; MiningResult | MiningResult | Run one isolated mining task per factor and return the launch result. |

<a id="file-6984acfade1d"></a>
## betalens/factor/mining_audit.py

[打开源码](../../../betalens/factor/mining_audit.py) · 667 行 · 说明来源：人工文件说明

- **作用**：搜索结果持久化与人可读审计
- **输入**：候选、窗口表现、元数据
- **输出**：SQLite、metadata YAML、Excel 和热力图
- **副作用/维护重点**：写文件；指纹、失败状态和全候选记录属于结果的一部分

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from matplotlib import font_manager
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import itertools
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import platform
import sqlite3
import subprocess
import sys
import uuid
import yaml
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_json_default](../../../betalens/factor/mining_audit.py#L24) | _json_default(value: Any) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_json_payload](../../../betalens/factor/mining_audit.py#L32) | _json_payload(value: Mapping[str, Any]) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_safe_factor_id](../../../betalens/factor/mining_audit.py#L36) | _safe_factor_id(value: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_sha256](../../../betalens/factor/mining_audit.py#L41) | _sha256(path: Path) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_git_state](../../../betalens/factor/mining_audit.py#L49) | _git_state(path: Path) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_atomic_yaml](../../../betalens/factor/mining_audit.py#L68) | _atomic_yaml(path: Path, payload: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ResultStore](../../../betalens/factor/mining_audit.py#L77) | class ResultStore() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：path: Path; connection: sqlite3.Connection = field(init=False) |
| [ResultStore.__post_init__](../../../betalens/factor/mining_audit.py#L81) | __post_init__(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ResultStore.append](../../../betalens/factor/mining_audit.py#L97) | append(self, table: str, rows: Iterable[Mapping[str, Any]]) -&gt; int | int | 无 docstring，需阅读函数体 |
| [ResultStore.read](../../../betalens/factor/mining_audit.py#L114) | read(self, table: str) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [ResultStore.close](../../../betalens/factor/mining_audit.py#L120) | close(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_empty_safe](../../../betalens/factor/mining_audit.py#L125) | _empty_safe(frame: pd.DataFrame, fallback: str='暂无记录') -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_humanize_columns](../../../betalens/factor/mining_audit.py#L156) | _humanize_columns(frame: pd.DataFrame) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_write_sheet](../../../betalens/factor/mining_audit.py#L210) | _write_sheet(writer: pd.ExcelWriter, name: str, frame: pd.DataFrame, *, max_rows: int=1000000) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_style_workbook](../../../betalens/factor/mining_audit.py#L223) | _style_workbook(path: Path) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_total_range](../../../betalens/factor/mining_audit.py#L246) | _total_range(metadata: Mapping[str, Any]) -&gt; tuple[pd.Timestamp, pd.Timestamp] &#124; None | tuple[pd.Timestamp, pd.Timestamp] &#124; None | Resolve the complete evaluation range used by audit heatmaps. |
| [_heatmap_parameter_pairs](../../../betalens/factor/mining_audit.py#L262) | _heatmap_parameter_pairs(frame: pd.DataFrame, parameter_specs: Mapping[str, Any]) -&gt; list[tuple[str, str]] | list[tuple[str, str]] | 无 docstring，需阅读函数体 |
| [_heatmap_matrix](../../../betalens/factor/mining_audit.py#L277) | _heatmap_matrix(frame: pd.DataFrame, row_name: str, column_name: str, metric: str) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_write_heatmap_report](../../../betalens/factor/mining_audit.py#L296) | _write_heatmap_report(audit_dir: Path, metadata: Mapping[str, Any], window_results: pd.DataFrame) -&gt; list[str] | list[str] | Write parameter heatmap PNGs and a machine-readable chart index. |
| [FactorMiningResult](../../../betalens/factor/mining_audit.py#L416) | class FactorMiningResult() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：factor_id: str; run_id: str; run_dir: Path; status: str; coarse_window_results: pd.DataFrame = field(default_factory=pd.DataFrame); coarse_summary: pd.DataFrame = field(default_factory=pd.DataFrame); fine_window_results: pd.DataFrame = field(default_factory=pd.DataFrame); fine_…（完整内容见 inventory.json/源码） |
| [MiningResult](../../../betalens/factor/mining_audit.py#L434) | class MiningResult() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：launch_id: str; factor_runs: tuple[FactorMiningResult, ...] |
| [MiningTask](../../../betalens/factor/mining_audit.py#L440) | class MiningTask() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：factor_id: str; run_id: str; launch_id: str; run_dir: Path; cache_dir: Path; audit_dir: Path; metadata_path: Path; workbook_path: Path; log_path: Path; store: ResultStore; metadata: dict[str, Any] |
| [MiningTask.create](../../../betalens/factor/mining_audit.py#L454) | create(cls, output_root: str &#124; Path, factor_id: str, launch_id: str, *, factor_class: str &#124; None, parameter_path: Path, performance_path: Path, config: Mapping[str, Any]) -&gt; 'MiningTask' | 'MiningTask' | 无 docstring，需阅读函数体 |
| [MiningTask.write_metadata](../../../betalens/factor/mining_audit.py#L519) | write_metadata(self, **updates: Any) -&gt; None | None | 无 docstring，需阅读函数体 |
| [MiningTask.export_workbook](../../../betalens/factor/mining_audit.py#L523) | export_workbook(self) -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [MiningTask.finish](../../../betalens/factor/mining_audit.py#L605) | finish(self, result: FactorMiningResult, *, status: str, **updates: Any) -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-376c527978b2"></a>
## betalens/factor/mining_cache.py

[打开源码](../../../betalens/factor/mining_cache.py) · 395 行 · 说明来源：人工文件说明

- **作用**：不可变行情缓存发布和切片
- **输入**：DataFrame/PIT、缓存目录、请求指纹
- **输出**：manifest、NPY/memmap 与切片
- **副作用/维护重点**：读写缓存；过期输入不能因路径相同而复用

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json
import logging
import numpy as np
import os
import pandas as pd
import shutil
import time
import uuid
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_json_write](../../../betalens/factor/mining_cache.py#L22) | _json_write(path: Path, payload: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_fsync_file](../../../betalens/factor/mining_cache.py#L29) | _fsync_file(path: Path) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_file_sha256](../../../betalens/factor/mining_cache.py#L35) | _file_sha256(path: Path) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_safe_name](../../../betalens/factor/mining_cache.py#L43) | _safe_name(name: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_write_axis](../../../betalens/factor/mining_cache.py#L48) | _write_axis(root: Path, kind: str, payload: bytes, suffix: str) -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [_write_dates_axis](../../../betalens/factor/mining_cache.py#L66) | _write_dates_axis(root: Path, dates: np.ndarray) -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [_write_frame](../../../betalens/factor/mining_cache.py#L82) | _write_frame(root: Path, name: str, frame: pd.DataFrame, *, dtype: str &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_write_pit](../../../betalens/factor/mining_cache.py#L128) | _write_pit(root: Path, pit: Mapping[Any, set[str]] &#124; None, universe: list[str]) -&gt; dict[str, Any] &#124; None | dict[str, Any] &#124; None | 无 docstring，需阅读函数体 |
| [publish](../../../betalens/factor/mining_cache.py#L143) | publish(cache_dir: str &#124; Path, signature: str, *, inputs: Mapping[str, pd.DataFrame], price: pd.DataFrame, execution_price: pd.DataFrame, trade_status: pd.DataFrame, industry_by_scheme: Mapping[str, pd.DataFrame], pit: Mapping[Any, set[str]] &#124; None, universe: list[str], metadata: Mapping[str, Any]) -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [_iter_descriptors](../../../betalens/factor/mining_cache.py#L195) | _iter_descriptors(datasets: Mapping[str, Any]) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [_validate_descriptor](../../../betalens/factor/mining_cache.py#L204) | _validate_descriptor(root: Path, descriptor: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_load_descriptor](../../../betalens/factor/mining_cache.py#L227) | _load_descriptor(root: Path, descriptor: Mapping[str, Any] &#124; None) -&gt; dict[str, Any] &#124; None | dict[str, Any] &#124; None | 无 docstring，需阅读函数体 |
| [open_manifest](../../../betalens/factor/mining_cache.py#L244) | open_manifest(path: str &#124; Path) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [frame](../../../betalens/factor/mining_cache.py#L266) | frame(descriptor: Mapping[str, Any], start: Any=None, end: Any=None, columns=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [estimated_resident_bytes](../../../betalens/factor/mining_cache.py#L294) | estimated_resident_bytes(path: str &#124; Path) -&gt; int | int | 无 docstring，需阅读函数体 |
| [CacheRequest](../../../betalens/factor/mining_cache.py#L314) | class CacheRequest() | 类定义；构造/属性见方法与字段 | Description of one task-local mining input cache.；字段：directory: str &#124; Path; signature: str |
| [MiningCache](../../../betalens/factor/mining_cache.py#L321) | class MiningCache() | 类定义；构造/属性见方法与字段 | Small public facade over the task-local memmap cache. One task writes one ''input_manifest.json'' and one ''datasets'' directory. Workers only load immutable slices from those files. |
| [MiningCache.__init__](../../../betalens/factor/mining_cache.py#L328) | __init__(self, manifest_path: str &#124; Path) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [MiningCache.open_or_build](../../../betalens/factor/mining_cache.py#L333) | open_or_build(cls, request: CacheRequest, builder=None) -&gt; 'MiningCache' | 'MiningCache' | 无 docstring，需阅读函数体 |
| [MiningCache.universe](../../../betalens/factor/mining_cache.py#L370) | universe(self) -&gt; list[str] | list[str] | 无 docstring，需阅读函数体 |
| [MiningCache.load](../../../betalens/factor/mining_cache.py#L373) | load(self, name: str, start=None, end=None, columns=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |

<a id="file-7cc8c5ed809b"></a>
## betalens/factor/mining_optuna.py

[打开源码](../../../betalens/factor/mining_optuna.py) · 484 行 · 说明来源：人工文件说明

- **作用**：采样器、范围扩展与扰动适配
- **输入**：参数定义、已有评价、搜索设定
- **输出**：study、参数候选、grid/扰动计划
- **副作用/维护重点**：可选 Optuna 依赖；采样并不证明样本外有效

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import itertools
import json
import math
import optuna
import random
import warnings
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_optuna](../../../betalens/factor/mining_optuna.py#L18) | _optuna() | 无返回注解；return: optuna | 无 docstring，需阅读函数体 |
| [_choice_token](../../../betalens/factor/mining_optuna.py#L28) | _choice_token(value: Any) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [_decode_choice](../../../betalens/factor/mining_optuna.py#L34) | _decode_choice(value: Any) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [to_optuna_distribution](../../../betalens/factor/mining_optuna.py#L40) | to_optuna_distribution(spec: Mapping[str, Any]) | 无返回注解；return: optuna.distributions.CategoricalDistribution(tokens); optuna.distributions.IntDistribution(int(low), int(high), step=int(spec.get('step') or 1), log=log); optuna.distributions.FloatDistribution(float(low), float(high), step=spec.get('step'), log=log) | 无 docstring，需阅读函数体 |
| [suggest_params](../../../betalens/factor/mining_optuna.py#L62) | suggest_params(trial, parameter_specs: Mapping[str, Mapping[str, Any]]) -&gt; dict[str, Any] | dict[str, Any] | Suggest one ordinary parameter dictionary from an Optuna trial. |
| [create_coarse_study](../../../betalens/factor/mining_optuna.py#L92) | create_coarse_study(config: Mapping[str, Any], *, direction: str='maximize') | 无返回注解；return: optuna.create_study(sampler=sampler, direction=direction) | Create the sampler-only study used by the coarse stage. |
| [create_fine_grid_study](../../../betalens/factor/mining_optuna.py#L123) | create_fine_grid_study(search_space: Mapping[str, Sequence[Any]], *, direction: str='maximize', seed: int=20260818) | 无返回注解；return: optuna.create_study(sampler=optuna.samplers.GridSampler(encoded, seed=int(seed)), direction=direction) | Create a GridSampler study while preserving composite categorical values. |
| [tell_trial](../../../betalens/factor/mining_optuna.py#L143) | tell_trial(study, trial, value: float &#124; None) -&gt; None | None | Complete an externally evaluated trial, including GridSampler exhaustion. |
| [seed_study_with_results](../../../betalens/factor/mining_optuna.py#L154) | seed_study_with_results(study, parameter_specs: Mapping[str, Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]], values: Sequence[float]) -&gt; int | int | 将已完成的外部候选作为历史 trial 导入新的自适应 study。 |
| [_values](../../../betalens/factor/mining_optuna.py#L188) | _values(spec: Mapping[str, Any], count: int) -&gt; list[Any] | list[Any] | 无 docstring，需阅读函数体 |
| [_dedupe](../../../betalens/factor/mining_optuna.py#L217) | _dedupe(rows: Sequence[Mapping[str, Any]]) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_unique_values](../../../betalens/factor/mining_optuna.py#L227) | _unique_values(values: Sequence[Any]) -&gt; list[Any] | list[Any] | 无 docstring，需阅读函数体 |
| [generate_coarse_candidates](../../../betalens/factor/mining_optuna.py#L237) | generate_coarse_candidates(parameters: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | Generate sparse candidates using an ask/tell Optuna study. |
| [FineGridPlan](../../../betalens/factor/mining_optuna.py#L250) | class FineGridPlan() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：candidates: list[dict[str, Any]]; anchors: list[dict[str, Any]]; dimensions: dict[str, list[Any]]; local_bounds: dict[str, dict[str, Any]] |
| [_numeric_position](../../../betalens/factor/mining_optuna.py#L257) | _numeric_position(value: float, spec: Mapping[str, Any]) -&gt; float | float | 无 docstring，需阅读函数体 |
| [detect_boundary_pressure](../../../betalens/factor/mining_optuna.py#L266) | detect_boundary_pressure(parameters: Mapping[str, Mapping[str, Any]], winners: Sequence[Mapping[str, Any]], *, tolerance: float=0.1, winner_ratio: float=0.67) -&gt; dict[str, dict[str, Any]] | dict[str, dict[str, Any]] | 识别优胜候选持续靠近同一参数边界的维度。 |
| [expand_parameter_specs](../../../betalens/factor/mining_optuna.py#L303) | expand_parameter_specs(parameters: Mapping[str, Mapping[str, Any]], pressure: Mapping[str, Mapping[str, Any]], *, multiplier: float=3.0, limits: Mapping[str, Mapping[str, Any]] &#124; None=None) -&gt; dict[str, dict[str, Any]] | dict[str, dict[str, Any]] | 根据边界压力向命中侧扩展参数，并受硬边界限制。 |
| [PerturbationPlan](../../../betalens/factor/mining_optuna.py#L341) | class PerturbationPlan() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：candidates: list[dict[str, Any]]; metadata: dict[str, dict[str, Any]] |
| [generate_perturbation_candidates](../../../betalens/factor/mining_optuna.py#L346) | generate_perturbation_candidates(parameters: Mapping[str, Mapping[str, Any]], winners: Sequence[Mapping[str, Any]], *, perturbations_per_candidate: int=8, radius_ratio: float=0.1, seed: int=20260818) -&gt; PerturbationPlan | PerturbationPlan | 在赢家附近按参数尺度生成可复现的随机扰动候选。 |
| [generate_fine_candidates](../../../betalens/factor/mining_optuna.py#L408) | generate_fine_candidates(parameters: Mapping[str, Mapping[str, Any]], anchors: Sequence[Mapping[str, Any]], config: Mapping[str, Any], coarse_candidates: Sequence[Mapping[str, Any]] &#124; None=None) -&gt; FineGridPlan | FineGridPlan | Generate a bounded local grid around coarse winners. |

<a id="file-949c3892d2a8"></a>
## betalens/factor/preprocessing.py

[打开源码](../../../betalens/factor/preprocessing.py) · 657 行 · 说明来源：人工文件说明

- **作用**：去极值、标准化与中性化
- **输入**：Series/因子长表、行业、市值
- **输出**：因子残差、处理表、诊断
- **副作用/维护重点**：行业自动查询路径读库；样本不足可能 skipped

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from betalens.datafeed import Datafeed, query_industry
from betalens.datafeed.validation import fix_null_values, FillStrategy
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_statsmodels](../../../betalens/factor/preprocessing.py#L20) | _statsmodels() | 无返回注解；return: sm | 无 docstring，需阅读函数体 |
| [winsorize_factor](../../../betalens/factor/preprocessing.py#L30) | winsorize_factor(factor_series: pd.Series, method: str='mad', n: float=3.0) -&gt; pd.Series | pd.Series | 截面去极值（单截面，index=code）。 Args: factor_series: 单截面因子值，index=code method: 'mad'（中位数绝对偏差，推荐）&#124; 'percentile'（百分位截尾）&#124; 'std'（均值±n倍标准差） n: 阈值倍数（percentile 方法时为单侧截尾百分比，如 n=1 截 [1%, 99%]） Returns: 去极值后的 Series，index 不变 Example: &gt;&gt;&gt; s = pd.Series({'A': 100, 'B': 2, 'C': 3, 'D': 1}) &gt;&gt;&gt; winsorize_factor(s, method…（完整内容见 inventory.json/源码） |
| [standardize_factor](../../../betalens/factor/preprocessing.py#L72) | standardize_factor(factor_series: pd.Series, method: str='zscore') -&gt; pd.Series | pd.Series | 截面标准化（单截面，index=code）。 Args: factor_series: 单截面因子值 method: 'zscore'（(x-mean)/std）&#124; 'rank'（rank/N，结果在(0,1)）&#124; 'minmax'（缩放到[0,1]） Returns: 标准化后的 Series Example: &gt;&gt;&gt; standardize_factor(s, method='zscore') |
| [neutralize_factor](../../../betalens/factor/preprocessing.py#L115) | neutralize_factor(factor_series: pd.Series, industry_labels: pd.Series=None, log_market_cap: pd.Series=None, return_stats: bool=False) | 无返回注解；return: _ret(factor_series); _ret(result) | OLS 残差中性化（单截面）。 对行业哑变量 + log(市值) 做截面 OLS，返回残差。 参考 robust.py 中 neu() 的实现模式。 行业标签既可由调用方直接传入，也可由 preprocess_factor(industry_scheme=...) 自动从 industry 表（query_industry）按 point-in-time 注入。 Args: factor_series: 因子值 Series，index=code（已标准化） industry_labels: 行业标签 Series，index=code（传 None 则跳过） log_market_cap: …（完整内容见 inventory.json/源码） |
| [neutralize_factor._ret](../../../betalens/factor/preprocessing.py#L146) | _ret(series) | 无返回注解；return: (series, stats) if return_stats else series | 无 docstring，需阅读函数体 |
| [neutralize_factor_by_factor](../../../betalens/factor/preprocessing.py#L183) | neutralize_factor_by_factor(factor_b_data: pd.DataFrame, factor_a_data: pd.DataFrame, metric_b: str, metric_a: str) -&gt; pd.DataFrame | pd.DataFrame | 用因子A对因子B做截面OLS中性化，返回残差作为"剔除A影响后的B"。 使用场景： - 检验规模因子(SIZE)对盈利因子(ROE)的解释程度，取残差得到"纯ROE" - 对双因子做正交化，使两因子线性无关 Args: factor_b_data: 被解释因子的 pre_query_characteristic_data() 输出 （列含 input_ts, code, {metric_b}） factor_a_data: 解释因子的输出（列含 input_ts, code, {metric_a}） metric_b: 被解释因子列名（因子B） metric_a: 解释因子列名（因子A） Re…（完整内容见 inventory.json/源码） |
| [filter_pool_by_industry](../../../betalens/factor/preprocessing.py#L251) | filter_pool_by_industry(labeled_pool: pd.DataFrame, industry_map: pd.DataFrame, include_industries: list) -&gt; pd.DataFrame | pd.DataFrame | 将打标签的选股池限制在指定行业范围内。 用途：在单个或多个行业的股票池中按因子选股，完全隔离其他行业。 Args: labeled_pool: single_characteristic() 的输出， MultiIndex(input_ts, code) industry_map: 行业映射表，列须含 input_ts, code, industry （可复用 pre_queried_data 中的行业列，或单独查询） include_industries: 保留的行业列表，如 ['银行', '非银金融'] 传 None 或 [] 则不过滤（返回原表） Returns: 过滤后的 labele…（完整内容见 inventory.json/源码） |
| [apply_industry_weight_constraint](../../../betalens/factor/preprocessing.py#L286) | apply_industry_weight_constraint(weights: pd.DataFrame, industry_map: pd.DataFrame, method: str='equal', target_weights: Optional[dict]=None) -&gt; pd.DataFrame | pd.DataFrame | 对已生成的权重矩阵施加行业权重约束（行业中性化后处理）。 三种模式： - 'equal' : 全行业等权，多头各行业权重之和相等，空头同理 - 'market' : 按市场基准（target_weights 传入各行业目标比例之和=1） - 'original' : 不调整（直接返回，供流程统一调用） Args: weights: get_single_factor_weight() 的输出， index=input_ts，columns=code，值为权重（多头&gt;0，空头&lt;0） industry_map: 行业映射表，列含 input_ts, code, industry method: '…（完整内容见 inventory.json/源码） |
| [_rescale_side_by_industry](../../../betalens/factor/preprocessing.py#L348) | _rescale_side_by_industry(row: pd.Series, codes: pd.Index, ind_ts: pd.Series, method: str, target_weights: Optional[dict], sign: int) -&gt; pd.Series | pd.Series | 对多头或空头一侧按行业重新分配权重。 sign=1 → 多头（归一到 +1），sign=-1 → 空头（归一到 -1）。 |
| [query_industry_panel](../../../betalens/factor/preprocessing.py#L405) | query_industry_panel(pre_queried_data: pd.DataFrame, scheme: str='申万一级行业', industry_table: str='industry', verbose: bool=True) -&gt; pd.Series | pd.Series | 面板行业查询：为 pre_queried_data 的每个 (input_ts, code) 取 point-in-time 行业名。 逐期调用 datafeed.query_industry（datetime&lt;=查询日 的最近一条，天然防前视）， 复用现有 API，不另写 SQL。 Args: pre_queried_data: 含 input_ts, code 列（pre_query_characteristic_data 的输出） scheme: 分类体系（metric），不带版本后缀时自动落到查询日生效的版本 industry_table: 行业表名，默认 'industry' ver…（完整内容见 inventory.json/源码） |
| [_print_industry_diagnostics](../../../betalens/factor/preprocessing.py#L468) | _print_industry_diagnostics(ind_panel: pd.Series, pre_queried_data: pd.DataFrame, metric: Optional[str]=None) -&gt; None | None | 打印行业分布 / 缺失情况 / 面板平衡诊断（中文 [INFO] 风格）。 |
| [preprocess_factor](../../../betalens/factor/preprocessing.py#L540) | preprocess_factor(pre_queried_data: pd.DataFrame, metric: str, winsorize_method: str='mad', winsorize_n: float=3.0, standardize_method: str='zscore', industry_col: str=None, log_mktcap_col: str=None, industry_scheme: str=None, industry_table: str='industry', verbose: bool=True) -&gt; pd.DataFrame | pd.DataFrame | 逐截面（按 input_ts）依次执行： fix_null_values(drop) → winsorize_factor() → standardize_factor() → neutralize_factor() 行业中性化的标签有两种来源： - industry_scheme 给定（推荐）：自动从 industry 表（query_industry，point-in-time） 逐期查询行业，并打印行业分布/缺失/面板平衡及中性化执行摘要。 - industry_col 给定：使用调用方预先 merge 进 pre_queried_data 的行业列（旧行为）。 两者都不给则不做行业中性…（完整内容见 inventory.json/源码） |
| [_print_neutralize_summary](../../../betalens/factor/preprocessing.py#L642) | _print_neutralize_summary(neu_stats: list) -&gt; None | None | 打印中性化执行摘要：成功/跳过期数、平均 R^2、平均行业哑变量数。 |

<a id="file-14d5b7575b99"></a>
## betalens/factor/profiling.py

[打开源码](../../../betalens/factor/profiling.py) · 772 行 · 说明来源：人工文件说明

- **作用**：因子值体检与跨因子比较
- **输入**：因子宽/长表或多因子字典
- **输出**：覆盖、分布、自相关、重合度等结果
- **副作用/维护重点**：因子体检不等同未来收益评价

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from pathlib import Path
from scipy import stats as _scipy_stats
from scipy.cluster import hierarchy as _scipy_hier
from scipy.spatial.distance import squareform as _squareform
from typing import Any
import math
import numpy as np
import pandas as pd
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_to_wide](../../../betalens/factor/profiling.py#L50) | _to_wide(factor_data, metric=None) | 无返回注解；return: wide.sort_index() | 归一化为宽表：index=input_ts，columns=code，值为因子值。 自动判别输入： - 已是宽表（无 input_ts/code 列，index 为时间）：原样返回 - 长表（含 input_ts, code 列）：用 metric 列 pivot |
| [_to_long](../../../betalens/factor/profiling.py#L76) | _to_long(factor_data, metric='factor') | 无返回注解；return: factor_data.copy(); long | 归一化为长表：列 input_ts, code, {metric}。 |
| [describe_distribution](../../../betalens/factor/profiling.py#L92) | describe_distribution(factor_data, metric=None, by_period=True) -&gt; pd.DataFrame | pd.DataFrame | 因子值分布与值域统计（横截面体检）。 Args: factor_data: 长表（input_ts/code/metric）或宽表（index=ts, col=code） metric: 长表的因子列名（宽表可不传） by_period: True 逐截面统计后给出跨期均值与全样本两行视角； False 仅返回全样本汇总一行 Returns: DataFrame，index 含 '全样本' 及（by_period 时）'逐期均值'， columns=[count, mean, std, min, 1%, 25%, 50%, 75%, 99%, max, skew, kurt, 缺失率] Exa…（完整内容见 inventory.json/源码） |
| [describe_distribution._row](../../../betalens/factor/profiling.py#L112) | _row(s: pd.Series) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [coverage_stats](../../../betalens/factor/profiling.py#L143) | coverage_stats(factor_data, metric=None) -&gt; pd.DataFrame | pd.DataFrame | 因子覆盖度时变统计。 应对股票池随时间扩张：每期有效（非缺失）股票数与覆盖率， 用于发现某些时段因子大面积缺失（如早年财报字段未披露）。 Args: factor_data: 长表或宽表 metric: 长表因子列名 Returns: DataFrame，index=input_ts，columns=[有效数, 总数, 覆盖率] Example: &gt;&gt;&gt; cov = coverage_stats(pre_queried_data, 'ROE') &gt;&gt;&gt; cov['覆盖率'].plot() |
| [detect_outliers](../../../betalens/factor/profiling.py#L171) | detect_outliers(factor_data, metric=None, method='mad', n=3.0) -&gt; pd.DataFrame | pd.DataFrame | 逐期极值占比检测，提示是否需要 winsorize。 Args: factor_data: 长表或宽表 metric: 长表因子列名 method: 'mad'（中位数绝对偏差）&#124; 'std'（均值±n倍标准差） n: 阈值倍数 Returns: DataFrame，index=input_ts，columns=[下界, 上界, 极值数, 极值占比] 末行 'Total' 为全样本汇总 Example: &gt;&gt;&gt; detect_outliers(pre_queried_data, 'ROE', method='mad', n=3) |
| [detect_outliers._bounds](../../../betalens/factor/profiling.py#L190) | _bounds(v: pd.Series) | 无返回注解；return: (med - n * mad, med + n * mad); (mu - n * sd, mu + n * sd) | 无 docstring，需阅读函数体 |
| [factor_autocorrelation](../../../betalens/factor/profiling.py#L224) | factor_autocorrelation(factor_data, metric=None, lags=None, method='spearman') -&gt; pd.DataFrame | pd.DataFrame | 因子记忆性：截面 rank 自相关（corr(值_t, 值_{t-lag}) 逐期算后时序平均）。 高自相关 → 因子缓变、换手低；低自相关 → 因子快变、换手高、交易成本敏感。 Args: factor_data: 长表或宽表 metric: 长表因子列名 lags: 滞后期列表，默认 [1, 3, 6, 12] method: 'spearman'（rank，推荐）&#124; 'pearson' Returns: DataFrame，index=lag，columns=[自相关均值, 自相关std, 有效期数] Example: &gt;&gt;&gt; factor_autocorrelation(pre_que…（完整内容见 inventory.json/源码） |
| [factor_turnover](../../../betalens/factor/profiling.py#L268) | factor_turnover(factor_data, metric=None, quantile=0.2, side='top') -&gt; pd.Series | pd.Series | 头部组成分换手率：相邻期 top/bottom 分位组的成分变动比例。 turnover_t = 1 - &#124;持仓_t ∩ 持仓_{t-1}&#124; / &#124;持仓_t&#124; Args: factor_data: 长表或宽表 metric: 长表因子列名 quantile: 分位阈值（0.2 表示 top/bottom 20%） side: 'top'（因子值最高组）&#124; 'bottom'（最低组） Returns: Series，index=input_ts，name='turnover'（首期为 NaN） Example: &gt;&gt;&gt; to = factor_turnover(pre_queried_data,…（完整内容见 inventory.json/源码） |
| [factor_turnover._members](../../../betalens/factor/profiling.py#L289) | _members(row: pd.Series) -&gt; set | set | 无 docstring，需阅读函数体 |
| [selection_overlap](../../../betalens/factor/profiling.py#L313) | selection_overlap(factor_data, metric=None, quantile=0.2, side='top') -&gt; pd.DataFrame | pd.DataFrame | 相邻期选股重合度（Jaccard）：&#124;A∩B&#124; / &#124;A∪B&#124;。 与 factor_turnover 互补：turnover 看流出，Jaccard 看整体相似度。 Args: factor_data: 长表或宽表 metric: 长表因子列名 quantile: 分位阈值 side: 'top' &#124; 'bottom' Returns: DataFrame，index=input_ts，columns=[Jaccard, 持仓数]（首期 NaN） Example: &gt;&gt;&gt; ov = selection_overlap(pre_queried_data, 'ROE') &gt;&gt;&gt; ov['Jacca…（完整内容见 inventory.json/源码） |
| [selection_overlap._members](../../../betalens/factor/profiling.py#L334) | _members(row: pd.Series) -&gt; set | set | 无 docstring，需阅读函数体 |
| [distribution_stability](../../../betalens/factor/profiling.py#L355) | distribution_stability(factor_data, metric=None) -&gt; pd.DataFrame | pd.DataFrame | 分布漂移监测：逐期均值/标准差/偏度/峰度时序，判断因子分布是否随时间漂移。 Args: factor_data: 长表或宽表 metric: 长表因子列名 Returns: DataFrame，index=input_ts，columns=[mean, std, skew, kurt, 有效数] Example: &gt;&gt;&gt; ds = distribution_stability(pre_queried_data, 'ROE') &gt;&gt;&gt; ds[['mean', 'std']].plot() |
| [factor_profile_payload](../../../betalens/factor/profiling.py#L384) | factor_profile_payload(factor_data, metric=None, *, bins: int=60, p_values: tuple[float, ...]=(0.1, 0.05, 0.01), max_points: int=12000) -&gt; dict[str, Any] | dict[str, Any] | 构建可交互展示的因子值分布诊断数据。 该函数只返回结构化数据，不依赖 matplotlib/plotly。前端或 notebook 可据此 渲染分布函数、CDF、集中度、不同 p 值假设检验临界值对应的原因子值等。 Args: factor_data: 长表（input_ts/code/metric）或宽表（index=ts, col=code） metric: 长表的因子列名 bins: 全样本直方图分箱数 p_values: 显著性/尾部概率水平。输出会同时给出 H0: E[x]=0 的 均值检验阈值，以及经验分布单尾/双尾对应的原因子值临界点。 max_points: ECDF 最多返回…（完整内容见 inventory.json/源码） |
| [factor_profile_payload._clean](../../../betalens/factor/profiling.py#L413) | _clean(value) | 无返回注解；return: None; int(value); float(value); value | 无 docstring，需阅读函数体 |
| [_align_factor_dict](../../../betalens/factor/profiling.py#L584) | _align_factor_dict(factor_dict: dict) -&gt; dict | dict | 把 {因子名: 长表/宽表} 统一转成 {因子名: 宽表}。 |
| [cross_correlation](../../../betalens/factor/profiling.py#L589) | cross_correlation(factor_dict: dict, method='spearman') -&gt; pd.DataFrame | pd.DataFrame | 多因子平均截面相关矩阵（逐期算 corr 再时序平均，避免规模偏误）。 Args: factor_dict: {因子名: 长表/宽表} method: 'spearman'（rank，推荐）&#124; 'pearson' Returns: DataFrame，N×N 对称相关矩阵，index/columns=因子名 Example: &gt;&gt;&gt; cross_correlation({'ROE': roe, 'PE': pe, 'SIZE': size}) |
| [correlation_timeseries](../../../betalens/factor/profiling.py#L632) | correlation_timeseries(factor_dict: dict, pair: tuple, method='spearman') -&gt; pd.Series | pd.Series | 指定因子对的截面相关系数时序，观察相关性是否稳定（突变=轮动/风格切换信号）。 Args: factor_dict: {因子名: 长表/宽表} pair: 因子名二元组，如 ('ROE', 'SIZE') method: 'spearman' &#124; 'pearson' Returns: Series，index=input_ts，name='corr(A,B)' Example: &gt;&gt;&gt; correlation_timeseries({'ROE': roe, 'SIZE': size}, ('ROE', 'SIZE')) |
| [selection_coincidence](../../../betalens/factor/profiling.py#L662) | selection_coincidence(factor_dict: dict, quantile=0.2, side='top') -&gt; pd.DataFrame | pd.DataFrame | 两两因子选股重合度矩阵：各因子 top 组的平均 Jaccard 相似度（逐期算后平均）。 高重合 → 两因子选出的股票高度雷同，组合层面冗余（即便值相关性中等）。 Args: factor_dict: {因子名: 长表/宽表} quantile: 分位阈值 side: 'top' &#124; 'bottom' Returns: DataFrame，N×N 对称重合度矩阵（对角线为 1） Example: &gt;&gt;&gt; selection_coincidence({'ROE': roe, 'PE': pe}, quantile=0.2) |
| [selection_coincidence._members](../../../betalens/factor/profiling.py#L687) | _members(row: pd.Series) -&gt; set | set | 无 docstring，需阅读函数体 |
| [factor_clustering](../../../betalens/factor/profiling.py#L709) | factor_clustering(corr_matrix: pd.DataFrame, threshold=0.6) -&gt; dict | dict | 基于相关矩阵的层次聚类，提示冗余因子组。 距离 = 1 - &#124;corr&#124;，相关性高的因子聚为一类，超阈值即视为冗余候选。 无 scipy 时退化为简单的并查集分组（按 &#124;corr&#124; ≥ threshold 连边）。 Args: corr_matrix: cross_correlation 的输出（N×N） threshold: 归为同组的相关性阈值（&#124;corr&#124; ≥ threshold） Returns: dict: { 'clusters': [[因子名,...], ...], # 每个子列表为一个冗余组 'n_clusters': int, 'method': 'hierarchical'…（完整内容见 inventory.json/源码） |
| [factor_clustering.find](../../../betalens/factor/profiling.py#L747) | find(x) | 无返回注解；return: x | 无 docstring，需阅读函数体 |

<a id="file-60bd554734a5"></a>
## betalens/factor/signal.py

[打开源码](../../../betalens/factor/signal.py) · 779 行 · 说明来源：人工文件说明

- **作用**：择时信号到目标权重转换
- **输入**：因子序列/宽表、阈值与历史规则
- **输出**：SignalWeightResult：weights、factor_values、events
- **副作用/维护重点**：起始日期解析可查库；历史窗口不能用未来观测

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.datafeed import Datafeed
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import math
import numpy as np
import pandas as pd
import re
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [SignalWeightResult](../../../betalens/factor/signal.py#L16) | class SignalWeightResult() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：weights: pd.DataFrame; factor_values: pd.DataFrame; events: dict[str, pd.DataFrame] = field(default_factory=dict) |
| [resolve_timing_start_date](../../../betalens/factor/signal.py#L22) | resolve_timing_start_date(start_date: str, end_date: str, *, target_codes: Sequence[str], metrics: Sequence[str], table_name: str) -&gt; str | str | Move a timing run start to the first date with all required target data. |
| [_signal_config](../../../betalens/factor/signal.py#L79) | _signal_config(params: Mapping[str, Any] &#124; None) -&gt; Mapping[str, Any] | Mapping[str, Any] | 无 docstring，需阅读函数体 |
| [_setting](../../../betalens/factor/signal.py#L86) | _setting(params: Mapping[str, Any] &#124; None, key: str, default: Any=None, aliases: Sequence[str]=()) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [_optional_text](../../../betalens/factor/signal.py#L104) | _optional_text(value: Any) -&gt; str &#124; None | str &#124; None | 无 docstring，需阅读函数体 |
| [_normalize_codes](../../../betalens/factor/signal.py#L111) | _normalize_codes(value: Any) -&gt; list[str] | list[str] | 无 docstring，需阅读函数体 |
| [_resolve_codes](../../../betalens/factor/signal.py#L120) | _resolve_codes(factor_wide: pd.DataFrame, codes: Sequence[str] &#124; None=None, params: Mapping[str, Any] &#124; None=None, universe: Sequence[str] &#124; None=None) -&gt; list[str] | list[str] | 无 docstring，需阅读函数体 |
| [_daily_series](../../../betalens/factor/signal.py#L139) | _daily_series(series: pd.Series) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [_signal_index](../../../betalens/factor/signal.py#L145) | _signal_index(signal_dates: Sequence[Any]) -&gt; pd.DatetimeIndex | pd.DatetimeIndex | 无 docstring，需阅读函数体 |
| [_finite_float](../../../betalens/factor/signal.py#L150) | _finite_float(value: Any, default: float) -&gt; float | float | 无 docstring，需阅读函数体 |
| [_max_weight](../../../betalens/factor/signal.py#L158) | _max_weight(params: Mapping[str, Any] &#124; None, default: float=1.0) -&gt; float | float | 无 docstring，需阅读函数体 |
| [resolve_operator](../../../betalens/factor/signal.py#L163) | resolve_operator(direction: str='positive', operator: str &#124; None=None) -&gt; str | str | 无 docstring，需阅读函数体 |
| [resolve_side](../../../betalens/factor/signal.py#L181) | resolve_side(params: Mapping[str, Any] &#124; None=None, *, side: str &#124; None=None, direction: str='positive') -&gt; tuple[str, float] | tuple[str, float] | 无 docstring，需阅读函数体 |
| [_event_active](../../../betalens/factor/signal.py#L204) | _event_active(values: pd.Series &#124; pd.DataFrame, threshold: float, operator: str) | 无返回注解；return: active.fillna(False).astype(bool) | 无 docstring，需阅读函数体 |
| [_target_from_active](../../../betalens/factor/signal.py#L209) | _target_from_active(active: pd.Series, side_sign: float, max_weight: float) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [_order_factor_values](../../../betalens/factor/signal.py#L213) | _order_factor_values(df: pd.DataFrame) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_scale_stock_weights](../../../betalens/factor/signal.py#L218) | _scale_stock_weights(stock_weights: pd.DataFrame) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [cash_from_weights](../../../betalens/factor/signal.py#L233) | cash_from_weights(stock_weights: pd.DataFrame, *, scale: bool=True) -&gt; pd.DataFrame | pd.DataFrame | Add a cash column using net exposure: full short -1 implies cash 2. |
| [_with_execution_time](../../../betalens/factor/signal.py#L245) | _with_execution_time(weights: pd.DataFrame, execution_delay: pd.Timedelta &#124; None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_factor_value_frame](../../../betalens/factor/signal.py#L253) | _factor_value_frame(*, signal_index: pd.DatetimeIndex, code: str, factor: pd.Series, active: pd.Series, target_weight: pd.Series, extras: Mapping[str, pd.Series] &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [threshold_weight](../../../betalens/factor/signal.py#L276) | threshold_weight(*, factor_wide: pd.DataFrame, signal_dates: Sequence[Any], codes: Sequence[str] &#124; None=None, params: Mapping[str, Any] &#124; None=None, threshold: float &#124; None=None, operator: str &#124; None=None, side: str &#124; None=None, direction: str='positive', max_weight: float &#124; None=None, execution_delay: pd.Timedelta &#124; None=pd.Timedelta(minutes=10)) -&gt; SignalWeightResult | SignalWeightResult | 无 docstring，需阅读函数体 |
| [rolling_z_weight](../../../betalens/factor/signal.py#L329) | rolling_z_weight(*, factor_wide: pd.DataFrame, signal_dates: Sequence[Any], codes: Sequence[str] &#124; None=None, params: Mapping[str, Any] &#124; None=None, window: int &#124; None=None, sigma: float &#124; None=None, operator: str &#124; None=None, side: str &#124; None=None, direction: str='positive', max_weight: float &#124; None=None, execution_delay: pd.Timedelta &#124; None=pd.Timedelta(minutes=10)) -&gt; SignalWeightResult | SignalWeightResult | 无 docstring，需阅读函数体 |
| [_safe_div](../../../betalens/factor/signal.py#L398) | _safe_div(numerator: float, denominator: float) -&gt; float | float | 无 docstring，需阅读函数体 |
| [_slope](../../../betalens/factor/signal.py#L404) | _slope(values: Sequence[float]) -&gt; float | float | 无 docstring，需阅读函数体 |
| [_trend_label](../../../betalens/factor/signal.py#L412) | _trend_label(slope: float, net_change: float, eps: float=1e-10) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_event_ids](../../../betalens/factor/signal.py#L424) | _event_ids(factor: pd.Series, threshold: float, operator: str) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [event_table_for_code](../../../betalens/factor/signal.py#L429) | event_table_for_code(factor: pd.Series, high: pd.Series &#124; None, threshold: float, operator: str) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_history_before](../../../betalens/factor/signal.py#L511) | _history_before(events: pd.DataFrame, event_start: pd.Timestamp, window: int) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_wait_history_before](../../../betalens/factor/signal.py#L516) | _wait_history_before(events: pd.DataFrame, event_start: pd.Timestamp, window: int) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_dynamic_event_weight_for_code](../../../betalens/factor/signal.py#L526) | _dynamic_event_weight_for_code(factor: pd.Series, events: pd.DataFrame, params: Mapping[str, Any] &#124; None) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [event_history_weight](../../../betalens/factor/signal.py#L583) | event_history_weight(*, factor_wide: pd.DataFrame, signal_dates: Sequence[Any], high_wide: pd.DataFrame &#124; None=None, codes: Sequence[str] &#124; None=None, params: Mapping[str, Any] &#124; None=None, threshold: float &#124; None=None, operator: str &#124; None=None, side: str &#124; None=None, direction: str='positive', execution_delay: pd.Timedelta &#124; None=pd.Timedelta(minutes=10)) -&gt; SignalWeightResult | SignalWeightResult | 无 docstring，需阅读函数体 |
| [_resolve_method](../../../betalens/factor/signal.py#L648) | _resolve_method(params: Mapping[str, Any] &#124; None, method: str &#124; None=None) -&gt; str | str | 无 docstring，需阅读函数体 |
| [build_signal_weights](../../../betalens/factor/signal.py#L680) | build_signal_weights(*, factor_wide: pd.DataFrame, signal_dates: Sequence[Any], codes: Sequence[str] &#124; None=None, params: Mapping[str, Any] &#124; None=None, high_wide: pd.DataFrame &#124; None=None, direction: str='positive', method: str &#124; None=None, side: str &#124; None=None, execution_delay: pd.Timedelta &#124; None=pd.Timedelta(minutes=10)) -&gt; SignalWeightResult | SignalWeightResult | 无 docstring，需阅读函数体 |
| [infer_signal_warmup](../../../betalens/factor/signal.py#L727) | infer_signal_warmup(params: Mapping[str, Any] &#124; None, minimum: int=30) -&gt; int | int | 无 docstring，需阅读函数体 |
| [standard_timing_weight_hook](../../../betalens/factor/signal.py#L738) | standard_timing_weight_hook(weights: pd.DataFrame, task: Mapping[str, Any]) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |

<a id="file-96ca2a399d41"></a>
## betalens/factor/stats.py

[打开源码](../../../betalens/factor/stats.py) · 1586 行 · 说明来源：人工文件说明

- **作用**：IC、截面与择时统计及图形
- **输入**：因子/收益面板、评估参数
- **输出**：Series、统计字典、回归/图形/报告
- **副作用/维护重点**：调用者负责未来收益对齐；导出路径写文件

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from pathlib import Path
from scipy import stats as _scipy_stats
import io as _io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_statsmodels](../../../betalens/factor/stats.py#L17) | _statsmodels() | 无返回注解；return: sm | 无 docstring，需阅读函数体 |
| [calc_ic](../../../betalens/factor/stats.py#L33) | calc_ic(factor_data: pd.DataFrame, return_data: pd.DataFrame, method: str='spearman') -&gt; pd.Series | pd.Series | 逐截面计算 IC（Information Coefficient）。 Args: factor_data: 宽表，index=input_ts，columns=code，值为因子值 return_data: 宽表，index=input_ts，columns=code，值为持仓期收益率 method: 'spearman'（Rank IC，推荐）&#124; 'pearson'（普通 IC） Returns: Series，index=input_ts，name='IC' Example: &gt;&gt;&gt; ic = calc_ic(factor_wide, return_wide) &gt;&gt;&gt; print(ic.m…（完整内容见 inventory.json/源码） |
| [calc_icir](../../../betalens/factor/stats.py#L75) | calc_icir(ic_series: pd.Series, window: int=None) | 无返回注解；return: ic_series.mean() / std if std != 0 else np.nan; ic_series.rolling(window).mean() / ic_series.rolling(window).std() | 计算 ICIR = mean(IC) / std(IC)。 Args: ic_series: calc_ic() 的输出 window: None 返回全样本 float；整数返回滚动 Series Returns: float（全样本）或 Series（滚动） Example: &gt;&gt;&gt; icir = calc_icir(ic) # 全样本 &gt;&gt;&gt; rolling_icir = calc_icir(ic, window=12) # 滚动12期 |
| [summarize_ic](../../../betalens/factor/stats.py#L100) | summarize_ic(ic_series: pd.Series) -&gt; dict | dict | IC 统计摘要。 Returns: dict: { 'IC均值': float, 'IC_std': float, 'ICIR': float, '胜率(IC&gt;0)': float, 't统计量': float, 'p值': float } Example: &gt;&gt;&gt; summary = summarize_ic(ic) &gt;&gt;&gt; pd.Series(summary) |
| [_normalize_factor_long](../../../betalens/factor/stats.py#L145) | _normalize_factor_long(factor_values: pd.DataFrame) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [calc_ic_from_factor_values](../../../betalens/factor/stats.py#L171) | calc_ic_from_factor_values(factor_values: pd.DataFrame, price_data: pd.DataFrame, method: str='spearman', min_periods: int=5) -&gt; pd.Series | pd.Series | 基于因子长表和回测价格宽表计算逐期 IC。 Args: factor_values: 长表，至少含 input_ts/信号日、code/股票代码、factor/因子值。 price_data: 宽表，index=调仓/成交时间，columns=code，值为回测成交价。 method: 'spearman'（Rank IC，推荐）&#124; 'pearson'。 min_periods: 单期截面最少股票数。 Returns: Series，index=调仓时间，name='IC'。每期使用该调仓日前最近一期因子信号， 收益使用当前价格到下一期价格的持仓期收益。 |
| [_normal_cdf](../../../betalens/factor/stats.py#L230) | _normal_cdf(x) | 无返回注解；return: 0.5 * (1 + np.math.erf(x / np.sqrt(2))) | 标准正态 CDF，用于无 scipy 时近似 p 值。 |
| [fama_macbeth](../../../betalens/factor/stats.py#L239) | fama_macbeth(factor_data_dict: dict, return_data: pd.DataFrame, industry_dummies: pd.DataFrame=None) -&gt; pd.DataFrame | pd.DataFrame | Fama-MacBeth 两步法截面回归。 第一步：每个截面期 t 做 OLS： R_i = α_t + Σ(λ_k,t * F_k,i) + ε_i 第二步：对 λ_k 时间序列做 t 检验（Newey-West 可选）。 Args: factor_data_dict: {因子名: 宽表 DataFrame (index=date, columns=code)} return_data: 宽表 (index=date, columns=code)，持仓期超额收益 industry_dummies: 可选，宽表 (index=date, columns=行业哑变量名) （用于控制行业效应，不纳…（完整内容见 inventory.json/源码） |
| [group_return_summary](../../../betalens/factor/stats.py#L345) | group_return_summary(labeled_pool: pd.DataFrame, return_data: pd.DataFrame, metric: str) -&gt; pd.DataFrame | pd.DataFrame | 计算各分组在持仓期内的等权平均收益。 Args: labeled_pool: single_characteristic() 输出， MultiIndex(input_ts, code)，含 {metric}_label 列 return_data: 宽表 (index=input_ts, columns=code)，持仓期收益率 metric: 因子名（用于找 label 列：{metric}_label） Returns: DataFrame，index=input_ts，columns=['G1'...'GN', 'long_short'] long_short = G_max - G_…（完整内容见 inventory.json/源码） |
| [calc_timing_ic](../../../betalens/factor/stats.py#L403) | calc_timing_ic(factor: pd.Series, returns: pd.Series, periods: list=None, method: str='spearman', rolling_window: int=60) -&gt; dict | dict | 择时因子 IC 计算：滚动窗口内 factor 与 N 日前瞻收益的相关系数。 Args: factor: 因子时间序列，index=datetime returns: 收益率时间序列，index=datetime periods: 预测周期列表，如 [5, 10, 20] method: 'spearman' &#124; 'pearson' rolling_window: 滚动窗口大小 Returns: {period: {'ic_series': Series, 'stats': summarize_ic 结果}} Example: &gt;&gt;&gt; result = calc_timing_ic(fact…（完整内容见 inventory.json/源码） |
| [generate_timing_signals](../../../betalens/factor/stats.py#L476) | generate_timing_signals(factor: pd.Series, sigma: float=1.0, ma_window: int=250, extreme_quantile: float=0.1) -&gt; dict | dict | 生成 7 种择时信号。 Args: factor: 标准化后的因子序列 sigma: 阈值法的 σ 倍数 ma_window: 均线法窗口 extreme_quantile: 极值法分位数 Returns: {方法名: signal_series}，signal ∈ {1, 0, -1} Example: &gt;&gt;&gt; signals = generate_timing_signals(factor, sigma=1.0) &gt;&gt;&gt; signals['阈值法'].value_counts() |
| [generate_timing_signals._to_signal](../../../betalens/factor/stats.py#L510) | _to_signal(long_mask, short_mask) | 无返回注解；return: s | 无 docstring，需阅读函数体 |
| [test_timing_signal](../../../betalens/factor/stats.py#L528) | test_timing_signal(signal: pd.Series, returns: pd.Series, period: int=5) -&gt; dict | dict | 单个择时信号的绩效检验。 Returns: dict 含多头/空头样本数、胜率、均收益、盈亏比、综合胜率、t 统计量、p 值 Example: &gt;&gt;&gt; stats = test_timing_signal(signal, returns, period=5) &gt;&gt;&gt; stats['综合胜率'] |
| [test_timing_signal._stats](../../../betalens/factor/stats.py#L551) | _stats(rets, side) | 无返回注解；return: {f'{side}样本数': 0, f'{side}胜率': 0.0, f'{side}均收益': 0.0, f'{side}盈亏比': 0.0}; {f'{side}样本数': n, f'{side}胜率': win_rate, f'{side}均收益': avg_ret, f'{side}盈亏比': pnl_ratio} | 无 docstring，需阅读函数体 |
| [test_all_timing_signals](../../../betalens/factor/stats.py#L595) | test_all_timing_signals(signals_dict: dict, returns: pd.Series, period: int=5, is_ratio: float=0.7) -&gt; pd.DataFrame | pd.DataFrame | 批量检验所有信号方法，含样本内/外分割。 Returns: DataFrame，每行一个信号方法，列匹配 timing_report.xlsx sheet3 Example: &gt;&gt;&gt; df = test_all_timing_signals(signals, returns, period=5) &gt;&gt;&gt; df[['综合胜率', '是否显著']] |
| [timing_regression](../../../betalens/factor/stats.py#L638) | timing_regression(factor: pd.Series, returns: pd.Series, period: int=5) -&gt; dict | dict | 择时因子回归分析：fwd_return ~ α + β * factor。 Returns: dict: Alpha, Beta, t 值, p 值, R², 调整R², F统计量, 样本量 Example: &gt;&gt;&gt; reg = timing_regression(factor, returns, period=5) &gt;&gt;&gt; reg['Beta'], reg['Beta-P值'] |
| [timing_robustness](../../../betalens/factor/stats.py#L686) | timing_robustness(factor: pd.Series, returns: pd.Series, period: int=5, is_ratio: float=0.7, method: str='spearman') -&gt; dict | dict | 稳健性检验：样本内/外 IC 对比和 IC 衰减。 Returns: dict: 样本分割日期, IS/OOS IC 均值和 ICIR, IC 衰减幅度, 是否稳健 Example: &gt;&gt;&gt; rob = timing_robustness(factor, returns, period=5) &gt;&gt;&gt; rob['是否稳健'] |
| [timing_robustness._ic_stats](../../../betalens/factor/stats.py#L713) | _ic_stats(data) | 无返回注解；return: {'IC均值': np.nan, 'ICIR': np.nan}; {'IC均值': ic_mean, 'ICIR': icir} | 无 docstring，需阅读函数体 |
| [backtest_timing_signal](../../../betalens/factor/stats.py#L765) | backtest_timing_signal(signal: pd.Series, returns: pd.Series, benchmark_returns: pd.Series=None) -&gt; dict | dict | 择时信号回测绩效。 signal 在 t 时刻的值决定 t+1 的仓位：1=多头, -1=空头, 0=空仓。 Returns: dict: 总收益, 年化收益, 波动率, Sharpe, 最大回撤, Calmar, 日胜率, 交易次数等 Example: &gt;&gt;&gt; perf = backtest_timing_signal(signal, returns) &gt;&gt;&gt; perf['Sharpe(策略)'] |
| [compute_timing_score](../../../betalens/factor/stats.py#L869) | compute_timing_score(ic_stats: dict, signal_results: pd.DataFrame, regression_stats: dict, robustness_stats: dict) -&gt; dict | dict | 择时因子综合评分（4维度，0~1）。 - IC (30%): &#124;ICIR&#124; 和 IC&gt;0 占比 - 信号 (25%): 最佳综合胜率 - 回归 (20%): R² 和 β 显著性 - 稳健性 (25%): IS/OOS IC 一致性 Returns: {'IC': float, '信号': float, '回归': float, '稳健性': float, '综合评分': float, '评级': str} Example: &gt;&gt;&gt; score = compute_timing_score(ic_stats, signal_df, reg, rob) &gt;&gt;&gt; score['评级'] |
| [run_timing_evaluation](../../../betalens/factor/stats.py#L952) | run_timing_evaluation(factor: pd.Series, returns: pd.Series, periods: list=None, method: str='spearman', rolling_window: int=60, sigma: float=1.0, ma_window: int=250, is_ratio: float=0.7, factor_name: str='') -&gt; dict | dict | 一键运行择时因子评价，返回全部结果。 Args: factor: 因子时间序列 returns: 收益率时间序列 periods: 预测周期列表 method: IC 计算方法 rolling_window: 滚动 IC 窗口 sigma: 信号阈值 ma_window: 均线窗口 is_ratio: 样本内比例 factor_name: 因子名称 Returns: dict: 包含 ic_results, signals, signal_tests, regression, robustness, backtest, score, factor_std 等全部结果 Example: &gt;&gt;&gt; …（完整内容见 inventory.json/源码） |
| [export_timing_report](../../../betalens/factor/stats.py#L1035) | export_timing_report(results: dict) -&gt; bytes | bytes | 将择时评价结果导出为 Excel（6 sheet），返回 bytes。 Example: &gt;&gt;&gt; excel_bytes = export_timing_report(results) &gt;&gt;&gt; with open('report.xlsx', 'wb') as f: f.write(excel_bytes) |
| [_fig_to_bytes](../../../betalens/factor/stats.py#L1133) | _fig_to_bytes(fig) -&gt; bytes | bytes | 无 docstring，需阅读函数体 |
| [plot_factor_timeseries](../../../betalens/factor/stats.py#L1141) | plot_factor_timeseries(factor_std: pd.Series, signal: pd.Series=None, sigma: float=1.0, title: str='因子值（预处理后）') -&gt; bytes | bytes | 因子值时序图：折线 + ±σ 虚线 + 信号区域着色。 Example: &gt;&gt;&gt; img = plot_factor_timeseries(factor_std, signal) |
| [plot_rolling_ic](../../../betalens/factor/stats.py#L1174) | plot_rolling_ic(ic_series: pd.Series, ic_mean: float=None, period: int=5, title: str=None) -&gt; bytes | bytes | 滚动 IC 时序图。 Example: &gt;&gt;&gt; img = plot_rolling_ic(ic_series, ic_mean=0.15, period=5) |
| [plot_signal_avg_return](../../../betalens/factor/stats.py#L1207) | plot_signal_avg_return(signal_tests: pd.DataFrame, method_name: str='阈值法', title: str=None) -&gt; bytes | bytes | 各信号平均收益率柱状图（多头绿/空头红）。 Example: &gt;&gt;&gt; img = plot_signal_avg_return(signal_tests, method_name='阈值法') |
| [plot_win_rate_comparison](../../../betalens/factor/stats.py#L1243) | plot_win_rate_comparison(signal_tests: pd.DataFrame, title: str='三种方法胜率对比') -&gt; bytes | bytes | 各方法综合胜率柱状图 + 50% 基线。 Example: &gt;&gt;&gt; img = plot_win_rate_comparison(signal_tests) |
| [plot_ic_by_period](../../../betalens/factor/stats.py#L1269) | plot_ic_by_period(ic_results: dict, title: str='各预测周期 IC & ICIR') -&gt; bytes | bytes | 分组柱状图，双 Y 轴（IC 均值 + ICIR）。 Args: ic_results: calc_timing_ic 的返回值 Example: &gt;&gt;&gt; img = plot_ic_by_period(ic_results) |
| [plot_return_distribution](../../../betalens/factor/stats.py#L1311) | plot_return_distribution(returns: pd.Series, signal: pd.Series, period: int=5, title: str='按信号分组的收益分布') -&gt; bytes | bytes | 三色直方图（中性灰/多头绿/空头红）。 Example: &gt;&gt;&gt; img = plot_return_distribution(returns, signal) |
| [plot_factor_vs_return](../../../betalens/factor/stats.py#L1347) | plot_factor_vs_return(factor: pd.Series, returns: pd.Series, period: int=5, title: str=None) -&gt; bytes | bytes | 因子值 vs 未来收益率散点图 + OLS 拟合线。 Example: &gt;&gt;&gt; img = plot_factor_vs_return(factor, returns, period=5) |
| [plot_composite_score](../../../betalens/factor/stats.py#L1388) | plot_composite_score(scores: dict, title: str=None) -&gt; bytes | bytes | 综合评分横向柱状图。 Args: scores: compute_timing_score 的返回值 Example: &gt;&gt;&gt; img = plot_composite_score(score_dict) |
| [plot_group_cumulative_returns](../../../betalens/factor/stats.py#L1432) | plot_group_cumulative_returns(group_returns: pd.DataFrame, title: str='分组累积收益率') -&gt; bytes | bytes | 截面因子分组累积收益折线图。 Args: group_returns: group_return_summary 的输出 Example: &gt;&gt;&gt; img = plot_group_cumulative_returns(group_df) |
| [monotonicity_test](../../../betalens/factor/stats.py#L1466) | monotonicity_test(group_returns: pd.DataFrame) -&gt; dict | dict | 分组收益单调性检验。 检查各组平均收益是否单调递增/递减。 Args: group_returns: group_return_summary 的输出 Returns: dict: 各组均值, 是否单调, 方向, Spearman 相关系数 Example: &gt;&gt;&gt; mono = monotonicity_test(group_df) &gt;&gt;&gt; mono['是否单调'] |
| [run_cross_section_evaluation](../../../betalens/factor/stats.py#L1504) | run_cross_section_evaluation(factor_data: pd.DataFrame, return_data: pd.DataFrame, periods: list=None, method: str='spearman', n_groups: int=5, factor_name: str='') -&gt; dict | dict | 一键运行截面因子评价。 Args: factor_data: 宽表，index=datetime，columns=code return_data: 宽表，index=datetime，columns=code periods: 预测周期列表（用不同持仓期的 return_data 逐个计算） method: IC 方法 n_groups: 分组数 factor_name: 因子名称 Returns: dict: ic_series, ic_stats, fm_results, group_returns, mono_test, score Example: &gt;&gt;&gt; results = run…（完整内容见 inventory.json/源码） |

