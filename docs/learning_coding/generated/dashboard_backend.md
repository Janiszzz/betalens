# dashboard_backend：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-b9cec2a3485f"></a>
## dashboard/backend/__init__.py

[打开源码](../../../dashboard/backend/__init__.py) · 1 行 · 说明来源：文件族规则

- **作用**：FastAPI backend for the rebuilt betalens dashboard.
- **输入**：import 请求
- **输出**：模块导出与符号；见静态 imports
- **副作用/维护重点**：初始化可能导入子模块；__all__ 与真实导出需结合源码阅读

<a id="file-105f9bea0691"></a>
## dashboard/backend/eventstudy_dashboard.py

[打开源码](../../../dashboard/backend/eventstudy_dashboard.py) · 476 行 · 说明来源：人工文件说明

- **作用**：事件文件发现与事件结果适配
- **输入**：EventStudyRequest/事件文件
- **输出**：事件预览、统计和对比 JSON
- **副作用/维护重点**：校验文件路径、读文件及查库；独立于因子回测队列入口

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .factors import FACTOR_ROOT, REPO_ROOT
from __future__ import annotations
from betalens.analyst.naming import get_name_map
from betalens.datafeed import Datafeed
from betalens.eventstudy.eventstudy import EventStudy
from betalens.factor.config import load_yaml_config, section
from pathlib import Path
from typing import Any
import math
import numpy as np
import pandas as pd
import re
import tempfile
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [load_eventstudy_params](../../../dashboard/backend/eventstudy_dashboard.py#L26) | load_eventstudy_params() -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_clean_scalar](../../../dashboard/backend/eventstudy_dashboard.py#L31) | _clean_scalar(value: Any) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [_records](../../../dashboard/backend/eventstudy_dashboard.py#L46) | _records(df: pd.DataFrame &#124; None, index_name: str='day') -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_safe_event_path](../../../dashboard/backend/eventstudy_dashboard.py#L57) | _safe_event_path(file_id: str) -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [_read_event_frame](../../../dashboard/backend/eventstudy_dashboard.py#L69) | _read_event_frame(path: Path) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_event_series](../../../dashboard/backend/eventstudy_dashboard.py#L85) | _event_series(path: Path) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [discover_event_files](../../../dashboard/backend/eventstudy_dashboard.py#L91) | discover_event_files() -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_parse_codes](../../../dashboard/backend/eventstudy_dashboard.py#L149) | _parse_codes(value: Any) -&gt; str &#124; list[str] | str &#124; list[str] | 无 docstring，需阅读函数体 |
| [_asset_payload](../../../dashboard/backend/eventstudy_dashboard.py#L163) | _asset_payload(codes: Any) -&gt; list[dict[str, str &#124; None]] | list[dict[str, str &#124; None]] | 无 docstring，需阅读函数体 |
| [_parse_int_list](../../../dashboard/backend/eventstudy_dashboard.py#L191) | _parse_int_list(value: Any) -&gt; list[int] | list[int] | 无 docstring，需阅读函数体 |
| [_build_holding_periods](../../../dashboard/backend/eventstudy_dashboard.py#L204) | _build_holding_periods(params: dict[str, Any]) -&gt; dict[str, list[int]] | dict[str, list[int]] | 无 docstring，需阅读函数体 |
| [_param_value](../../../dashboard/backend/eventstudy_dashboard.py#L210) | _param_value(params: dict[str, Any], snake_name: str, camel_name: str, fallback: Any) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [_key_metric](../../../dashboard/backend/eventstudy_dashboard.py#L220) | _key_metric(rows: list[dict[str, Any]], day: int) -&gt; dict[str, Any] &#124; None | dict[str, Any] &#124; None | 无 docstring，需阅读函数体 |
| [_event_date_for_index](../../../dashboard/backend/eventstudy_dashboard.py#L227) | _event_date_for_index(event_dates: Any, event_idx: Any) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [_returns_matrix_records](../../../dashboard/backend/eventstudy_dashboard.py#L240) | _returns_matrix_records(df: pd.DataFrame &#124; None, event_dates: Any=None, max_events: int=30) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_price_matrix_records](../../../dashboard/backend/eventstudy_dashboard.py#L262) | _price_matrix_records(df: pd.DataFrame &#124; None, event_dates: Any=None, max_events: int=30) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_event_rows](../../../dashboard/backend/eventstudy_dashboard.py#L284) | _event_rows(path: Path) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_comparison_payload](../../../dashboard/backend/eventstudy_dashboard.py#L293) | _comparison_payload(raw: dict[str, Any]) -&gt; dict[str, Any] &#124; None | dict[str, Any] &#124; None | 无 docstring，需阅读函数体 |
| [run_event_study](../../../dashboard/backend/eventstudy_dashboard.py#L374) | run_event_study(params: dict[str, Any]) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |

<a id="file-9dd3cb796ae6"></a>
## dashboard/backend/factors.py

[打开源码](../../../dashboard/backend/factors.py) · 202 行 · 说明来源：人工文件说明

- **作用**：YAML 因子发现、详情与动态加载
- **输入**：类名/因子名、仓库目录
- **输出**：FactorSummary/Detail、配置、模块
- **副作用/维护重点**：扫描文件/缓存；import 因子脚本必须无回测副作用

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .schemas import FactorDetail, FactorSummary
from __future__ import annotations
from betalens.factor.config import load_yaml_config, section
from functools import lru_cache
from pathlib import Path
from typing import Any
import ast
import importlib.util
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_strategy_type](../../../dashboard/backend/factors.py#L21) | _strategy_type(meta: dict[str, Any]) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_display_inputs](../../../dashboard/backend/factors.py#L26) | _display_inputs(factor_spec: dict[str, Any]) -&gt; dict[str, str] | dict[str, str] | 无 docstring，需阅读函数体 |
| [_factor_yaml_paths](../../../dashboard/backend/factors.py#L36) | _factor_yaml_paths(factor_dir: Path) -&gt; list[Path] | list[Path] | Return factor YAMLs in a factor directory, canonical file first. |
| [_iter_factor_specs](../../../dashboard/backend/factors.py#L53) | _iter_factor_specs(class_dir: Path) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 扫描类目录下的因子子文件夹，读取各自 factor_*.yaml。 |
| [_iter_specs](../../../dashboard/backend/factors.py#L71) | _iter_specs() -&gt; list[tuple[str, Path, dict[str, Any]]] | list[tuple[str, Path, dict[str, Any]]] | 无 docstring，需阅读函数体 |
| [effective_factor_defaults](../../../dashboard/backend/factors.py#L90) | effective_factor_defaults(spec_data: dict[str, Any], factor_cfg: dict[str, Any]) -&gt; dict[str, Any] | dict[str, Any] | Return a flat runtime parameter view for the Dashboard form. |
| [discover_factors](../../../dashboard/backend/factors.py#L116) | discover_factors() -&gt; tuple[FactorSummary, ...] | tuple[FactorSummary, ...] | 无 docstring，需阅读函数体 |
| [get_factor_config](../../../dashboard/backend/factors.py#L138) | get_factor_config(factor_class: str, name: str) -&gt; tuple[Path, dict[str, Any], dict[str, Any]] | tuple[Path, dict[str, Any], dict[str, Any]] | 无 docstring，需阅读函数体 |
| [get_factor_detail](../../../dashboard/backend/factors.py#L159) | get_factor_detail(factor_class: str, name: str) -&gt; FactorDetail | FactorDetail | 无 docstring，需阅读函数体 |
| [load_factor_module](../../../dashboard/backend/factors.py#L185) | load_factor_module(script: Path) | 无返回注解；return: mod | 无 docstring，需阅读函数体 |
| [clear_factor_cache](../../../dashboard/backend/factors.py#L201) | clear_factor_cache() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-7a2bfea64afa"></a>
## dashboard/backend/main.py

[打开源码](../../../dashboard/backend/main.py) · 210 行 · 说明来源：人工文件说明

- **作用**：HTTP 路由、错误码与日志流
- **输入**：HTTP 路径/查询/请求体
- **输出**：JSON、SSE、FileResponse
- **副作用/维护重点**：调服务后可能启动任务或写产物；health 不检查 DB

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .eventstudy_dashboard import discover_event_files, run_event_study
from .factors import clear_factor_cache, discover_factors, get_factor_detail
from .runs import manager
from .schemas import EventStudyRequest, FactorDetail, FactorSummary, RunCreated, RunRequest, RunState
from .serialization import build_downloads
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
import asyncio
import json
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_manager](../../../dashboard/backend/main.py#L22) | _manager() | 无返回注解；return: manager | 无 docstring，需阅读函数体 |
| [_eventstudy](../../../dashboard/backend/main.py#L28) | _eventstudy() | 无返回注解；return: (discover_event_files, run_event_study) | 无 docstring，需阅读函数体 |
| [health](../../../dashboard/backend/main.py#L46) | health() -&gt; dict[str, str] | dict[str, str] | 无 docstring，需阅读函数体 |
| [list_factors](../../../dashboard/backend/main.py#L51) | list_factors(refresh: bool=False) | 无返回注解；return: list(discover_factors()) | 无 docstring，需阅读函数体 |
| [eventstudy_files](../../../dashboard/backend/main.py#L58) | eventstudy_files() | 无返回注解；return: discover_event_files() | 无 docstring，需阅读函数体 |
| [eventstudy_run](../../../dashboard/backend/main.py#L64) | eventstudy_run(request: EventStudyRequest) | 无返回注解；return: run_event_study(request.model_dump()) | 无 docstring，需阅读函数体 |
| [factor_detail](../../../dashboard/backend/main.py#L75) | factor_detail(factor_class: str, name: str) | 无返回注解；return: get_factor_detail(factor_class, name) | 无 docstring，需阅读函数体 |
| [create_run](../../../dashboard/backend/main.py#L83) | create_run(request: RunRequest) | 无返回注解；return: RunCreated(run_id=run.run_id) | 无 docstring，需阅读函数体 |
| [clear_runs](../../../dashboard/backend/main.py#L92) | clear_runs() | 无返回注解；return: _manager().clear() | 无 docstring，需阅读函数体 |
| [run_state](../../../dashboard/backend/main.py#L97) | run_state(run_id: str) | 无返回注解；return: _manager().get(run_id).to_state() | 无 docstring，需阅读函数体 |
| [run_result](../../../dashboard/backend/main.py#L105) | run_result(run_id: str) | 无返回注解；return: _manager().serialize_result(run_id) | 无 docstring，需阅读函数体 |
| [run_profiling](../../../dashboard/backend/main.py#L119) | run_profiling(run_id: str, date_from: str &#124; None=None, date_to: str &#124; None=None) | 无返回注解；return: _manager().factor_profile(run_id, date_from=date_from, date_to=date_to) | 无 docstring，需阅读函数体 |
| [run_table](../../../dashboard/backend/main.py#L133) | run_table(run_id: str, kind: str, request: Request, page: int=Query(1, ge=1), size: int=Query(50, ge=1, le=500), query: str &#124; None=None, date_from: str &#124; None=None, date_to: str &#124; None=None) | 无返回注解；return: _manager().table_page(run_id, kind, page=page, size=size, query=query, filters=filters or None, date_from=date_from, date_to=date_to) | 无 docstring，需阅读函数体 |
| [run_logs](../../../dashboard/backend/main.py#L169) | run_logs(run_id: str) | 无返回注解；return: StreamingResponse(event_stream(), media_type='text/event-stream') | 无 docstring，需阅读函数体 |
| [run_logs.event_stream](../../../dashboard/backend/main.py#L175) | event_stream() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [download](../../../dashboard/backend/main.py#L194) | download(run_id: str, kind: str) | 无返回注解；return: FileResponse(path, filename=path.name) | 无 docstring，需阅读函数体 |

<a id="file-0387907fe584"></a>
## dashboard/backend/requirements.txt

[打开源码](../../../dashboard/backend/requirements.txt) · 19 行 · 说明来源：人工文件说明

- **作用**：后端依赖声明
- **输入**：pip install -r
- **输出**：后端 Python 依赖环境
- **副作用/维护重点**：与 pyproject extras 核对版本，安装会改变环境

<a id="file-80458cab6992"></a>
## dashboard/backend/runs.py

[打开源码](../../../dashboard/backend/runs.py) · 498 行 · 说明来源：人工文件说明

- **作用**：内存任务队列与执行状态机
- **输入**：RunRequest、run_id
- **输出**：DashboardRun、RunState、结果/分页
- **副作用/维护重点**：单执行线程与独立 dump 线程；写配置产物；clear 不会强杀运行线程

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .factors import get_factor_config, load_factor_module
from .schemas import RunRequest, RunState, RunStatus
from .serialization import build_downloads
from .serialization import build_result_payload, write_factor_values_parquet
from .serialization import build_table, write_table_parquet
from .serialization import read_factor_profile
from .serialization import read_table_page
from __future__ import annotations
from betalens.analyst import Analyst
from betalens.factor.config import run_parameters, section, write_yaml_config
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import contextlib
import copy
import inspect
import io
import pandas as pd
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_normalize_group_list](../../../dashboard/backend/runs.py#L29) | _normalize_group_list(value: Any) -&gt; list[Any] &#124; None | list[Any] &#124; None | 无 docstring，需阅读函数体 |
| [LogBuffer](../../../dashboard/backend/runs.py#L51) | class LogBuffer(io.TextIOBase) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [LogBuffer.__init__](../../../dashboard/backend/runs.py#L52) | __init__(self, run: 'DashboardRun') | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [LogBuffer.writable](../../../dashboard/backend/runs.py#L55) | writable(self) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [LogBuffer.write](../../../dashboard/backend/runs.py#L58) | write(self, s: str) -&gt; int | int | 无 docstring，需阅读函数体 |
| [LogBuffer.flush](../../../dashboard/backend/runs.py#L63) | flush(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DashboardRun](../../../dashboard/backend/runs.py#L67) | class DashboardRun() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [DashboardRun.__init__](../../../dashboard/backend/runs.py#L68) | __init__(self, request: RunRequest) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [DashboardRun.append_log](../../../dashboard/backend/runs.py#L91) | append_log(self, text: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DashboardRun.log](../../../dashboard/backend/runs.py#L97) | log(self) -&gt; str | str | 无 docstring，需阅读函数体 |
| [DashboardRun.table_path](../../../dashboard/backend/runs.py#L101) | table_path(self, kind: str) -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [DashboardRun.factor_values_path](../../../dashboard/backend/runs.py#L104) | factor_values_path(self) -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [DashboardRun.cleanup](../../../dashboard/backend/runs.py#L107) | cleanup(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DashboardRun.mark_started](../../../dashboard/backend/runs.py#L111) | mark_started(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DashboardRun.mark_completed](../../../dashboard/backend/runs.py#L116) | mark_completed(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DashboardRun.mark_failed](../../../dashboard/backend/runs.py#L121) | mark_failed(self, exc: BaseException) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DashboardRun.elapsed_seconds](../../../dashboard/backend/runs.py#L128) | elapsed_seconds(self) -&gt; float | float | 无 docstring，需阅读函数体 |
| [DashboardRun.to_state](../../../dashboard/backend/runs.py#L134) | to_state(self) -&gt; RunState | RunState | 无 docstring，需阅读函数体 |
| [RunManager](../../../dashboard/backend/runs.py#L148) | class RunManager() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [RunManager.__init__](../../../dashboard/backend/runs.py#L149) | __init__(self) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [RunManager.create](../../../dashboard/backend/runs.py#L156) | create(self, request: RunRequest) -&gt; DashboardRun | DashboardRun | 无 docstring，需阅读函数体 |
| [RunManager.clear](../../../dashboard/backend/runs.py#L170) | clear(self) -&gt; dict[str, int] | dict[str, int] | 无 docstring，需阅读函数体 |
| [RunManager.get](../../../dashboard/backend/runs.py#L189) | get(self, run_id: str) -&gt; DashboardRun | DashboardRun | 无 docstring，需阅读函数体 |
| [RunManager._execute](../../../dashboard/backend/runs.py#L198) | _execute(self, run: DashboardRun) -&gt; None | None | 无 docstring，需阅读函数体 |
| [RunManager._build_run_config](../../../dashboard/backend/runs.py#L310) | _build_run_config(factor_cfg: dict[str, Any], run: DashboardRun, output_dir: Path) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [RunManager._flat_parameters](../../../dashboard/backend/runs.py#L382) | _flat_parameters(config: dict[str, Any]) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [RunManager._persist_tables](../../../dashboard/backend/runs.py#L404) | _persist_tables(self, run: DashboardRun) -&gt; None | None | 无 docstring，需阅读函数体 |
| [RunManager._dump_excel](../../../dashboard/backend/runs.py#L414) | _dump_excel(bt: Any, output_dir: Path, name: str, factor_values: Any=None, pit_validation: Any=None, neutralize_stats: Any=None) -&gt; None | None | 无 docstring，需阅读函数体 |
| [RunManager.serialize_result](../../../dashboard/backend/runs.py#L447) | serialize_result(self, run_id: str) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [RunManager.table_page](../../../dashboard/backend/runs.py#L458) | table_page(self, run_id: str, kind: str, page: int, size: int, query: str &#124; None=None, filters: dict[str, str] &#124; None=None, date_from: str &#124; None=None, date_to: str &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [RunManager.factor_profile](../../../dashboard/backend/runs.py#L484) | factor_profile(self, run_id: str, date_from: str &#124; None=None, date_to: str &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |

<a id="file-4fb540c77400"></a>
## dashboard/backend/schemas.py

[打开源码](../../../dashboard/backend/schemas.py) · 72 行 · 说明来源：人工文件说明

- **作用**：Pydantic 请求响应模型
- **输入**：JSON 字段
- **输出**：类型化模型或校验错误
- **副作用/维护重点**：parameters 是开放字典，深层语义还需业务校验

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Literal
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [FactorSummary](../../../dashboard/backend/schemas.py#L12) | class FactorSummary(BaseModel) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：factor_class: str; name: str; strategy_type: StrategyType = 'cross_sectional'; formula: str = ''; logic: str = ''; source: str = ''; inputs: dict[str, str] = Field(default_factory=dict); defaults: dict[str, Any] = Field(default_factory=dict) |
| [FactorDetail](../../../dashboard/backend/schemas.py#L23) | class FactorDetail(FactorSummary) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：doc: str = ''; compute_kwargs: dict[str, Any] = Field(default_factory=dict); script_path: str; factor_dir: str |
| [RunRequest](../../../dashboard/backend/schemas.py#L30) | class RunRequest(BaseModel) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：factor_class: str; name: str; parameters: dict[str, Any] = Field(default_factory=dict); compute_kwargs: dict[str, Any] = Field(default_factory=dict) |
| [RunCreated](../../../dashboard/backend/schemas.py#L37) | class RunCreated(BaseModel) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：run_id: str |
| [EventStudyRequest](../../../dashboard/backend/schemas.py#L41) | class EventStudyRequest(BaseModel) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：event_file: str &#124; None = None; code: str &#124; list[str] &#124; None = None; benchmark_code: str &#124; None = None; metric: str &#124; None = None; table_name: str &#124; None = None; mode: str &#124; None = None; multi_asset_mode: Literal['aggregate', 'compare'] &#124; None = None; window_before: int &#124; None =…（完整内容见 inventory.json/源码） |
| [RunState](../../../dashboard/backend/schemas.py#L57) | class RunState(BaseModel) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：run_id: str; status: RunStatus; factor_class: str; name: str; started_at: str &#124; None = None; finished_at: str &#124; None = None; elapsed_seconds: float = 0; error: str &#124; None = None; log_size: int = 0 |
| [DownloadInfo](../../../dashboard/backend/schemas.py#L69) | class DownloadInfo(BaseModel) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：kind: str; path: str &#124; None; exists: bool |

<a id="file-be7faaf40841"></a>
## dashboard/backend/serialization.py

[打开源码](../../../dashboard/backend/serialization.py) · 1373 行 · 说明来源：人工文件说明

- **作用**：研究对象到前端契约
- **输入**：回测/Analyst/因子值、分页参数
- **输出**：JSON、图表记录、Parquet 和下载状态
- **副作用/维护重点**：NaN、时间戳、列名、分页筛选须与 TS 对齐

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.analyst import metrics as M
from betalens.analyst.naming import get_name_map
from betalens.factor.profiling import factor_profile_payload
from pathlib import Path
from scipy import stats as scipy_stats
from typing import Any
import math
import numpy as np
import pandas as pd
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_clean_scalar](../../../dashboard/backend/serialization.py#L43) | _clean_scalar(value: Any) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [_name_map_for_codes](../../../dashboard/backend/serialization.py#L59) | _name_map_for_codes(codes: list[str] &#124; set[str] &#124; tuple[str, ...]) -&gt; dict[str, str] | dict[str, str] | 无 docstring，需阅读函数体 |
| [_label_code](../../../dashboard/backend/serialization.py#L75) | _label_code(code: str, name_map: dict[str, str] &#124; None=None) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_json_records](../../../dashboard/backend/serialization.py#L86) | _json_records(df: pd.DataFrame &#124; None, max_rows: int &#124; None=None) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_series_points](../../../dashboard/backend/serialization.py#L100) | _series_points(series: pd.Series &#124; None, name: str) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_wide_long_records](../../../dashboard/backend/serialization.py#L107) | _wide_long_records(df: pd.DataFrame &#124; None, value_name: str, top_n: int &#124; None=None) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_position_weight_records](../../../dashboard/backend/serialization.py#L125) | _position_weight_records(daily_position_value: pd.DataFrame &#124; None, top: int=10, max_codes: int=25) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_normalize_factor_values](../../../dashboard/backend/serialization.py#L180) | _normalize_factor_values(factor_values: pd.DataFrame &#124; None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_factor_values_for_group_nav](../../../dashboard/backend/serialization.py#L211) | _factor_values_for_group_nav(factor_values: pd.DataFrame &#124; None, n_quantiles: int) -&gt; pd.DataFrame &#124; None | pd.DataFrame &#124; None | 将内部 0 基分组标签转换成 group_nav 的 1 基标签。 |
| [_filter_factor_dates](../../../dashboard/backend/serialization.py#L233) | _filter_factor_dates(factor_df: pd.DataFrame, date_from: str &#124; None=None, date_to: str &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [build_factor_profile_payload](../../../dashboard/backend/serialization.py#L250) | build_factor_profile_payload(factor_values: pd.DataFrame &#124; None, date_from: str &#124; None=None, date_to: str &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [write_factor_values_parquet](../../../dashboard/backend/serialization.py#L286) | write_factor_values_parquet(factor_values: pd.DataFrame &#124; None, path: Path) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [read_factor_profile](../../../dashboard/backend/serialization.py#L296) | read_factor_profile(path: Path &#124; None, date_from: str &#124; None=None, date_to: str &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_factor_lookup_for_date](../../../dashboard/backend/serialization.py#L306) | _factor_lookup_for_date(factor_df: pd.DataFrame, dt: pd.Timestamp) -&gt; dict[str, dict[str, Any]] | dict[str, dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_rebalance_holding_records](../../../dashboard/backend/serialization.py#L329) | _rebalance_holding_records(bt: Any, factor_values: pd.DataFrame &#124; None=None) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_drawdown_interval](../../../dashboard/backend/serialization.py#L375) | _drawdown_interval(nav: pd.Series) -&gt; str &#124; None | str &#124; None | 无 docstring，需阅读函数体 |
| [build_metrics](../../../dashboard/backend/serialization.py#L388) | build_metrics(analyst: Any, bt: Any) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_empty_timing_payload](../../../dashboard/backend/serialization.py#L433) | _empty_timing_payload() -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_metric](../../../dashboard/backend/serialization.py#L453) | _metric(group: str, label: str, value: Any) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_finite_float](../../../dashboard/backend/serialization.py#L462) | _finite_float(value: Any) -&gt; float &#124; None | float &#124; None | 无 docstring，需阅读函数体 |
| [_numeric_series](../../../dashboard/backend/serialization.py#L470) | _numeric_series(series: Any) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [_numeric_frame](../../../dashboard/backend/serialization.py#L493) | _numeric_frame(df: Any) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_timing_weight_frames](../../../dashboard/backend/serialization.py#L508) | _timing_weight_frames(bt: Any) -&gt; tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series] | 无 docstring，需阅读函数体 |
| [_timing_primary_code](../../../dashboard/backend/serialization.py#L538) | _timing_primary_code(stock_weight: pd.DataFrame) -&gt; str &#124; None | str &#124; None | 无 docstring，需阅读函数体 |
| [_timing_price_series](../../../dashboard/backend/serialization.py#L547) | _timing_price_series(bt: Any, primary_code: str &#124; None) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [_timing_nav_price_records](../../../dashboard/backend/serialization.py#L561) | _timing_nav_price_records(nav: pd.Series, price: pd.Series, position: pd.Series) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_timing_trade_marker_records](../../../dashboard/backend/serialization.py#L582) | _timing_trade_marker_records(rebalance_log: Any, price: pd.Series, primary_code: str &#124; None) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | Return buy/sell events whose y values sit on the displayed price curve. |
| [_timing_position_records](../../../dashboard/backend/serialization.py#L629) | _timing_position_records(position: pd.Series, cash: pd.Series) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_drawdown_from_nav](../../../dashboard/backend/serialization.py#L645) | _drawdown_from_nav(nav: pd.Series) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [_timing_return_series](../../../dashboard/backend/serialization.py#L652) | _timing_return_series(nav: pd.Series) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [_timing_trade_segments](../../../dashboard/backend/serialization.py#L658) | _timing_trade_segments(stock_weight: pd.DataFrame, position: pd.Series, returns: pd.Series, daily_pnl: pd.Series, epsilon: float=1e-08) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_timing_trade_segments.side_of](../../../dashboard/backend/serialization.py#L675) | side_of(value: float) -&gt; int | int | 无 docstring，需阅读函数体 |
| [_timing_trade_segments.close_segment](../../../dashboard/backend/serialization.py#L682) | close_segment(end_i: int) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_timing_open_forward_returns](../../../dashboard/backend/serialization.py#L732) | _timing_open_forward_returns(nav: pd.Series, position: pd.Series) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_timing_trade_metrics](../../../dashboard/backend/serialization.py#L757) | _timing_trade_metrics(segments: list[dict[str, Any]]) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_timing_performance_metrics](../../../dashboard/backend/serialization.py#L777) | _timing_performance_metrics(nav: pd.Series, returns: pd.Series) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_timing_factor_series](../../../dashboard/backend/serialization.py#L811) | _timing_factor_series(factor_values: pd.DataFrame &#124; None, primary_code: str &#124; None=None) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [_forward_returns_from_nav](../../../dashboard/backend/serialization.py#L836) | _forward_returns_from_nav(nav: pd.Series, horizon: int) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [_rolling_rank_ic](../../../dashboard/backend/serialization.py#L845) | _rolling_rank_ic(aligned: pd.DataFrame, window: int) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [_ols_prediction_stats](../../../dashboard/backend/serialization.py#L859) | _ols_prediction_stats(aligned: pd.DataFrame) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_timing_prediction_payload](../../../dashboard/backend/serialization.py#L890) | _timing_prediction_payload(factor_values: pd.DataFrame &#124; None, nav: pd.Series, primary_code: str &#124; None, main_horizon: int=5) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [build_timing_payload](../../../dashboard/backend/serialization.py#L955) | build_timing_payload(bt: Any, factor_values: pd.DataFrame &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [build_chart_data](../../../dashboard/backend/serialization.py#L1030) | build_chart_data(bt: Any, factor_values: pd.DataFrame &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_nav_value_for_trade](../../../dashboard/backend/serialization.py#L1048) | _nav_value_for_trade(nav: pd.Series &#124; None, trade_date: Any) -&gt; float &#124; None | float &#124; None | 无 docstring，需阅读函数体 |
| [build_generated_chart_data](../../../dashboard/backend/serialization.py#L1060) | build_generated_chart_data(bt: Any, factor_values: pd.DataFrame &#124; None=None, n_quantiles: Any=None, precomputed: dict[str, Any] &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 复用脚本静态图口径，生成供 dashboard 渲染的结构化数据。 |
| [build_trade_table](../../../dashboard/backend/serialization.py#L1139) | build_trade_table(bt: Any) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [build_position_table](../../../dashboard/backend/serialization.py#L1149) | build_position_table(bt: Any) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [_lookup](../../../dashboard/backend/serialization.py#L1201) | _lookup(df: pd.DataFrame &#124; None, dt: pd.Timestamp, code: str) -&gt; float &#124; None | float &#124; None | 无 docstring，需阅读函数体 |
| [build_downloads](../../../dashboard/backend/serialization.py#L1208) | build_downloads(factor_dir: Path, name: str) -&gt; dict[str, dict[str, Any]] | dict[str, dict[str, Any]] | 无 docstring，需阅读函数体 |
| [build_result_payload](../../../dashboard/backend/serialization.py#L1223) | build_result_payload(run: Any, table_metas: dict[str, dict[str, Any]], factor_values: pd.DataFrame &#124; None=None, pit_validation: pd.DataFrame &#124; None=None, neutralize_stats: pd.DataFrame &#124; None=None, chart_data: dict[str, Any] &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 构建可 JSON 化的结果（指标+图表+表元数据）。巨表明细不在内,走 /table 分页。 不含 downloads —— 那个按需实时探测磁盘,因为 dump 是异步落盘的。 |
| [_table_meta](../../../dashboard/backend/serialization.py#L1266) | _table_meta(rows: list[dict[str, Any]]) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [build_table](../../../dashboard/backend/serialization.py#L1277) | build_table(bt: Any, kind: str) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [write_table_parquet](../../../dashboard/backend/serialization.py#L1285) | write_table_parquet(rows: list[dict[str, Any]], path: Path) -&gt; dict[str, Any] | dict[str, Any] | 把巨表落成 parquet,返回 {total, columns} 元数据。空表不落盘。 |
| [read_table_page](../../../dashboard/backend/serialization.py#L1295) | read_table_page(path: Path &#124; None, page: int=1, size: int=50, query: str &#124; None=None, filters: dict[str, str] &#124; None=None, date_from: str &#124; None=None, date_to: str &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 从 parquet 读取分页数据。 pyarrow 目前不能在任意文本搜索后直接只读目标页；这里保留 DataFrame 过滤，但避免先全量转成 Python records，降低大表接口的额外内存和 CPU。 |
| [paginate_table](../../../dashboard/backend/serialization.py#L1349) | paginate_table(rows: list[dict[str, Any]], page: int=1, size: int=50, query: str &#124; None=None, filters: dict[str, str] &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |

