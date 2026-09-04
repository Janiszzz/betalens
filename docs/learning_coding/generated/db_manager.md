# db_manager：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-897bd0749fe2"></a>
## betalens_db_manager/__init__.py

[打开源码](../../../betalens_db_manager/__init__.py) · 64 行 · 说明来源：文件族规则

- **作用**：Local database management tools for Betalens.
- **输入**：import 请求
- **输出**：模块导出与符号；见静态 imports
- **副作用/维护重点**：初始化可能导入子模块；__all__ 与真实导出需结合源码阅读

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .constants import ALLOWED_TABLES, ALLOWED_WRITE_TABLES, DEFAULT_LIMIT
from .db import DatabaseClient, QueryRequest
from .import_adapters import ADAPTERS, AdapterRegistry, ImportBatch, IndexSnapshotBatch, IndustryBatch, MarketBatch, ObservationBatch, TradeStatusBatch, TradeCalendarBatch
from .import_manifest import ManifestEntry, ManifestPlan, ManifestRunner
from .importers import DatabaseWriter, DeleteRequest, load_trade_calendar
from .job_store import JobStore
from .jobs import ImportJobRunner
from .manager import DatabaseManager
from .profiles import ConnectionProfile, ConnectionResolver, ProfileStore, ResolvedConnection
from .records import ImportRecordStore
from .registry import DATASETS, DatasetSpec, get_dataset
from .schema import SchemaManager
```

<a id="file-de4611a5aeb9"></a>
## betalens_db_manager/__main__.py

[打开源码](../../../betalens_db_manager/__main__.py) · 253 行 · 说明来源：人工文件说明

- **作用**：CLI/桌面启动分发
- **输入**：命令行、配置与确认选项
- **输出**：计划/运行 JSON、退出状态或 GUI
- **副作用/维护重点**：init/import 可写库；无参数启动桌面界面

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .gui import main as gui_main
from .manager import DatabaseManager
from .schema import SchemaManager
from .utils import json_default
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import argparse
import json
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_add_database_arguments](../../../betalens_db_manager/__main__.py#L16) | _add_database_arguments(parser: argparse.ArgumentParser) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_database_overrides](../../../betalens_db_manager/__main__.py#L25) | _database_overrides(args: argparse.Namespace) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [_add_schema_options](../../../betalens_db_manager/__main__.py#L33) | _add_schema_options(parser: argparse.ArgumentParser) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_build_parser](../../../betalens_db_manager/__main__.py#L50) | _build_parser() -&gt; argparse.ArgumentParser | argparse.ArgumentParser | 无 docstring，需阅读函数体 |
| [_target_version](../../../betalens_db_manager/__main__.py#L84) | _target_version(args: argparse.Namespace) -&gt; int &#124; None | int &#124; None | 无 docstring，需阅读函数体 |
| [_make_manager](../../../betalens_db_manager/__main__.py#L92) | _make_manager(args: argparse.Namespace) -&gt; DatabaseManager | DatabaseManager | 无 docstring，需阅读函数体 |
| [_json_print](../../../betalens_db_manager/__main__.py#L99) | _json_print(payload: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_confirm_schema](../../../betalens_db_manager/__main__.py#L103) | _confirm_schema(plan: Mapping[str, Any]) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [_has_upsert](../../../betalens_db_manager/__main__.py#L120) | _has_upsert(manifest_plan: Mapping[str, Any] &#124; None) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [_confirm_upsert](../../../betalens_db_manager/__main__.py#L127) | _confirm_upsert() -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [_run_plan](../../../betalens_db_manager/__main__.py#L135) | _run_plan(args: argparse.Namespace) -&gt; int | int | 无 docstring，需阅读函数体 |
| [_run_init](../../../betalens_db_manager/__main__.py#L147) | _run_init(args: argparse.Namespace) -&gt; int | int | 无 docstring，需阅读函数体 |
| [_run_import_command](../../../betalens_db_manager/__main__.py#L175) | _run_import_command(args: argparse.Namespace) -&gt; int | int | 无 docstring，需阅读函数体 |
| [_run_verify](../../../betalens_db_manager/__main__.py#L190) | _run_verify(args: argparse.Namespace) -&gt; int | int | 无 docstring，需阅读函数体 |
| [_load_manifest](../../../betalens_db_manager/__main__.py#L201) | _load_manifest(path: Path, manager: SchemaManager &#124; DatabaseManager &#124; None=None) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | Compatibility helper returning fully expanded, validated entries. |
| [_run_manifest](../../../betalens_db_manager/__main__.py#L212) | _run_manifest(path: Path, manager: SchemaManager &#124; DatabaseManager) -&gt; dict[str, Any] | dict[str, Any] | Compatibility wrapper used by older integration tests and callers. |
| [main](../../../betalens_db_manager/__main__.py#L224) | main(argv: list[str] &#124; None=None) -&gt; int | int | 无 docstring，需阅读函数体 |

<a id="file-baeb0a7a316d"></a>
## betalens_db_manager/adapters/__init__.py

[打开源码](../../../betalens_db_manager/adapters/__init__.py) · 41 行 · 说明来源：文件族规则

- **作用**：Source adapters owned by :mod:`betalens_db_manager`.
- **输入**：import 请求
- **输出**：模块导出与符号；见静态 imports
- **副作用/维护重点**：初始化可能导入子模块；__all__ 与真实导出需结合源码阅读

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .ede import clean_ede_dataframe, extract_date_from_filename, extract_date_from_metric_metadata, identify_code_name_columns, parse_metric_column, process_ede_file
from .files import apply_time_alignment, iter_file_chunks, read_csv_with_encoding, read_file
from .industry import build_industry_records
from .wind import fetch_daily_bond, fetch_daily_fund, fetch_daily_index, fetch_daily_market
```

<a id="file-d8838edeaf5f"></a>
## betalens_db_manager/adapters/ede.py

[打开源码](../../../betalens_db_manager/adapters/ede.py) · 275 行 · 说明来源：人工文件说明

- **作用**：Wind EDE 宽表解析
- **输入**：文件、列元信息、日期规则
- **输出**：规范化指标长表
- **副作用/维护重点**：文件读取；区分报告期和可得日期

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .files import read_file
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Iterable
import logging
import pandas as pd
import re
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_logger](../../../betalens_db_manager/adapters/ede.py#L47) | _logger(logger: logging.Logger &#124; None) -&gt; logging.Logger | logging.Logger | 无 docstring，需阅读函数体 |
| [extract_date_from_filename](../../../betalens_db_manager/adapters/ede.py#L51) | extract_date_from_filename(filepath: str &#124; Path, pattern: str=DEFAULT_DATE_PATTERN, default_time: str=DEFAULT_TIME, logger: logging.Logger &#124; None=None) -&gt; str &#124; None | str &#124; None | Extract an eight-digit date from a source filename. |
| [parse_metric_column](../../../betalens_db_manager/adapters/ede.py#L71) | parse_metric_column(column_name: str, logger: logging.Logger &#124; None=None) -&gt; tuple[str, dict[str, str]] | tuple[str, dict[str, str]] | Split an EDE metric header into its metric name and metadata. |
| [extract_date_from_metric_metadata](../../../betalens_db_manager/adapters/ede.py#L96) | extract_date_from_metric_metadata(metadata: dict[str, str], column_name: str, default_time: str=DEFAULT_TIME, logger: logging.Logger &#124; None=None) -&gt; str &#124; None | str &#124; None | Extract an eight-digit effective date embedded in an EDE header. |
| [clean_ede_dataframe](../../../betalens_db_manager/adapters/ede.py#L118) | clean_ede_dataframe(df: pd.DataFrame, keywords_to_remove: Iterable[str] &#124; None=None, logger: logging.Logger &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | Remove empty rows/columns and EDE footer or source-note rows. |
| [_find_column](../../../betalens_db_manager/adapters/ede.py#L138) | _find_column(columns: Iterable[object], candidates: Iterable[str]) -&gt; object &#124; None | object &#124; None | 无 docstring，需阅读函数体 |
| [identify_code_name_columns](../../../betalens_db_manager/adapters/ede.py#L148) | identify_code_name_columns(df: pd.DataFrame, code_column_names: Iterable[str] &#124; None=None, name_column_names: Iterable[str] &#124; None=None, logger: logging.Logger &#124; None=None) -&gt; tuple[object &#124; None, object &#124; None] | tuple[object &#124; None, object &#124; None] | Identify security code and name columns in an EDE frame. |
| [process_ede_file](../../../betalens_db_manager/adapters/ede.py#L167) | process_ede_file(filepath: str &#124; Path, date_from: str='filename', default_datetime: str &#124; None=None, code_column_names: Iterable[str] &#124; None=None, name_column_names: Iterable[str] &#124; None=None, logger: logging.Logger &#124; None=None, *, date_pattern: str=DEFAULT_DATE_PATTERN, default_time: str=DEFAULT_TIME, keywords_to_remove: Iterable[str] &#124; None=None) -&gt; tuple[pd.DataFrame &#124; None, list[dict[str, object]]] | tuple[pd.DataFrame &#124; None, list[dict[str, object]]] | Convert one EDE export to the standard six-column import boundary. |

<a id="file-637a53b269ae"></a>
## betalens_db_manager/adapters/files.py

[打开源码](../../../betalens_db_manager/adapters/files.py) · 201 行 · 说明来源：人工文件说明

- **作用**：CSV/Excel 读取、分块与时间对齐
- **输入**：源路径、编码、读取选项
- **输出**：DataFrame/迭代批次
- **副作用/维护重点**：文件 I/O；不能从依赖 pyarrow 推定此入口支持 Parquet

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, Iterator
import codecs
import gzip
import logging
import pandas as pd
import pyarrow.parquet as pq
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_logger](../../../betalens_db_manager/adapters/files.py#L29) | _logger(logger: logging.Logger &#124; None) -&gt; logging.Logger | logging.Logger | 无 docstring，需阅读函数体 |
| [read_csv_with_encoding](../../../betalens_db_manager/adapters/files.py#L33) | read_csv_with_encoding(filepath: str &#124; Path, *, encodings: Iterable[str] &#124; None=None, logger: logging.Logger &#124; None=None, **kwargs) -&gt; pd.DataFrame | pd.DataFrame | Read a CSV, trying common Chinese encodings when none is specified. |
| [read_file](../../../betalens_db_manager/adapters/files.py#L63) | read_file(filepath: str &#124; Path, *, encodings: Iterable[str] &#124; None=None, logger: logging.Logger &#124; None=None, **kwargs) -&gt; pd.DataFrame | pd.DataFrame | Read a supported local source file into a DataFrame. CSV, XLS and XLSX are deliberately the only accepted formats. Import jobs validate and normalize the returned frame before any database transaction. |
| [_detect_csv_encoding](../../../betalens_db_manager/adapters/files.py#L99) | _detect_csv_encoding(path: Path, encodings: Iterable[str] &#124; None=None) -&gt; str | str | 无 docstring，需阅读函数体 |
| [iter_file_chunks](../../../betalens_db_manager/adapters/files.py#L119) | iter_file_chunks(filepath: str &#124; Path, *, chunk_size: int=DEFAULT_CHUNK_SIZE, read_options: dict[str, Any] &#124; None=None, encodings: Iterable[str] &#124; None=None, logger: logging.Logger &#124; None=None) -&gt; Iterator[pd.DataFrame] | Iterator[pd.DataFrame] | Yield bounded DataFrames from CSV/CSV.GZ/Parquet or one Excel frame. |
| [apply_time_alignment](../../../betalens_db_manager/adapters/files.py#L169) | apply_time_alignment(df: pd.DataFrame, date_column: str='日期', metric_column: str='metric', open_metric_names: Iterable[str] &#124; None=None, open_time: str='09:30:01', other_time: str='15:00:01', inplace: bool=False, logger: logging.Logger &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | Align daily observations to their first known market timestamp. |

<a id="file-9c8c070304c1"></a>
## betalens_db_manager/adapters/industry.py

[打开源码](../../../betalens_db_manager/adapters/industry.py) · 48 行 · 说明来源：人工文件说明

- **作用**：行业源数据规范化
- **输入**：行业成员数据与分类选项
- **输出**：行业长表记录
- **副作用/维护重点**：生效日期与分类体系必须保留

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
import pandas as pd
import re
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [build_industry_records](../../../betalens_db_manager/adapters/industry.py#L10) | build_industry_records(df: pd.DataFrame, scheme: str='申万一级行业', code_col: str='code', name_col: str='name', date_col: str='effective_dt', ind_name_col: str='ind_name', ind_code_col: str &#124; None='ind_code') -&gt; pd.DataFrame | pd.DataFrame | Convert industry events to the standard six-column import frame. |
| [build_industry_records.numeric_code](../../../betalens_db_manager/adapters/industry.py#L27) | numeric_code(value) | 无返回注解；return: None; int(match.group()) if match else None | 无 docstring，需阅读函数体 |

<a id="file-945a99b92993"></a>
## betalens_db_manager/adapters/wind.py

[打开源码](../../../betalens_db_manager/adapters/wind.py) · 196 行 · 说明来源：人工文件说明

- **作用**：可选 WindPy 行情获取
- **输入**：代码、日期、指标选项
- **输出**：规范化行情表
- **副作用/维护重点**：调用外部 Wind 服务；不是通用离线测试路径

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .files import apply_time_alignment
from WindPy import w
from __future__ import annotations
from typing import Any, Iterable
import logging
import pandas as pd
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_wind_client](../../../betalens_db_manager/adapters/wind.py#L33) | _wind_client(client: Any &#124; None) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [fetch_daily_market](../../../betalens_db_manager/adapters/wind.py#L43) | fetch_daily_market(codes: Iterable[str], start_date: str, end_date: str, fields: Iterable[str] &#124; None=None, asset_type: str='stock', apply_time_stamps: bool=True, logger: logging.Logger &#124; None=None, *, field_names: Iterable[str] &#124; None=None, client: Any &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | Fetch daily Wind data and return the standard six-column long frame. |
| [fetch_daily_index](../../../betalens_db_manager/adapters/wind.py#L121) | fetch_daily_index(codes: Iterable[str], start_date: str, end_date: str, fields: Iterable[str] &#124; None=None, apply_time_stamps: bool=True, logger: logging.Logger &#124; None=None, *, field_names: Iterable[str] &#124; None=None, client: Any &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | Fetch daily index observations. |
| [fetch_daily_fund](../../../betalens_db_manager/adapters/wind.py#L147) | fetch_daily_fund(codes: Iterable[str], start_date: str, end_date: str, fields: Iterable[str] &#124; None=None, apply_time_stamps: bool=True, logger: logging.Logger &#124; None=None, *, field_names: Iterable[str] &#124; None=None, client: Any &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | Fetch daily fund observations. |
| [fetch_daily_bond](../../../betalens_db_manager/adapters/wind.py#L173) | fetch_daily_bond(codes: Iterable[str], start_date: str, end_date: str, fields: Iterable[str] &#124; None=None, apply_time_stamps: bool=True, logger: logging.Logger &#124; None=None, *, field_names: Iterable[str] &#124; None=None, client: Any &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | Fetch daily bond observations. |

<a id="file-41d76e9300ae"></a>
## betalens_db_manager/constants.py

[打开源码](../../../betalens_db_manager/constants.py) · 43 行 · 说明来源：人工文件说明

- **作用**：导入列、模式、日志路径等常量
- **输入**：模块导入
- **输出**：DB_COLUMNS、默认限制和模式名
- **副作用/维护重点**：标准必需五列加可选 remark；不是所有接口都有 unit 列

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .registry import LOGICAL_TABLES, WRITABLE_TABLES
from __future__ import annotations
from pathlib import Path
```

<a id="file-4863b03ad0dd"></a>
## betalens_db_manager/contracts.py

[打开源码](../../../betalens_db_manager/contracts.py) · 556 行 · 说明来源：人工文件说明

- **作用**：每版结构契约
- **输入**：schema 版本
- **输出**：SchemaContract：表列索引等
- **副作用/维护重点**：与 SQL migration 和 verify 同步

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
import hashlib
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [SchemaContract](../../../betalens_db_manager/contracts.py#L425) | class SchemaContract() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：version: int; tables: tuple[str, ...]; views: tuple[str, ...] = (); functions: tuple[str, ...] = (); columns: Mapping[str, tuple[tuple[str, str], ...]] = field(default_factory=dict); not_null_columns: Mapping[str, tuple[str, ...]] = field(default_factory=dict); identity_columns…（完整内容见 inventory.json/源码） |
| [get_schema_contract](../../../betalens_db_manager/contracts.py#L515) | get_schema_contract(version: int) -&gt; SchemaContract | SchemaContract | 无 docstring，需阅读函数体 |

<a id="file-f91cbc937870"></a>
## betalens_db_manager/db.py

[打开源码](../../../betalens_db_manager/db.py) · 908 行 · 说明来源：人工文件说明

- **作用**：管理端连接和分页查询
- **输入**：QueryRequest、数据库配置
- **输出**：查询记录、连接信息、分页 token
- **副作用/维护重点**：短连接 PostgreSQL；表名白名单与参数化必须保留

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .constants import ALLOWED_TABLES, DEFAULT_LIMIT, DEFAULT_STATEMENT_TIMEOUT_MS
from .registry import DATASETS, DatasetSpec, get_dataset
from .utils import clean_database_config
from __future__ import annotations
from betalens.datafeed.config import get_database_config
from dataclasses import dataclass
from psycopg2 import sql
from typing import Any
import base64
import json
import pandas as pd
import psycopg2
import psycopg2.extras
import re
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [QueryRequest](../../../betalens_db_manager/db.py#L39) | class QueryRequest() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：table: str; code: str &#124; None = None; metric: str &#124; None = None; start_date: str &#124; None = None; end_date: str &#124; None = None; limit: int = DEFAULT_LIMIT; page_token: str &#124; None = None |
| [DatabaseClient](../../../betalens_db_manager/db.py#L49) | class DatabaseClient() | 类定义；构造/属性见方法与字段 | Short-lived connections for inspecting logical Betalens datasets. |
| [DatabaseClient.__init__](../../../betalens_db_manager/db.py#L52) | __init__(self, db_config: dict[str, Any] &#124; None=None, statement_timeout_ms: int=DEFAULT_STATEMENT_TIMEOUT_MS) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [DatabaseClient.connect](../../../betalens_db_manager/db.py#L60) | connect(self) | 无返回注解；return: conn | 无 docstring，需阅读函数体 |
| [DatabaseClient.test_connection](../../../betalens_db_manager/db.py#L74) | test_connection(self) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseClient.validate_table](../../../betalens_db_manager/db.py#L81) | validate_table(self, table: str, *, writable: bool=False) -&gt; str | str | 无 docstring，需阅读函数体 |
| [DatabaseClient.make_page_token](../../../betalens_db_manager/db.py#L86) | make_page_token(row: dict[str, Any] &#124; pd.Series) -&gt; str | str | Build an opaque keyset token from the last row of a result page. |
| [DatabaseClient.parse_page_token](../../../betalens_db_manager/db.py#L99) | parse_page_token(token: str) -&gt; tuple[str, str, str] | tuple[str, str, str] | 无 docstring，需阅读函数体 |
| [DatabaseClient._has_new_schema](../../../betalens_db_manager/db.py#L107) | _has_new_schema(self, cur) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [DatabaseClient.table_overview](../../../betalens_db_manager/db.py#L114) | table_overview(self, *, include_checks: bool=False) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [DatabaseClient.coverage_checks](../../../betalens_db_manager/db.py#L178) | coverage_checks(self) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | Compute read-only logical row/date coverage for all datasets. |
| [DatabaseClient._coverage_checks](../../../betalens_db_manager/db.py#L187) | _coverage_checks(self, cur) -&gt; dict[str, dict[str, Any]] | dict[str, dict[str, Any]] | 无 docstring，需阅读函数体 |
| [DatabaseClient._physical_sizes](../../../betalens_db_manager/db.py#L211) | _physical_sizes(self, cur) -&gt; dict[str, int] | dict[str, int] | 无 docstring，需阅读函数体 |
| [DatabaseClient._pretty_bytes](../../../betalens_db_manager/db.py#L236) | _pretty_bytes(value: int) -&gt; str | str | 无 docstring，需阅读函数体 |
| [DatabaseClient._legacy_table_overview](../../../betalens_db_manager/db.py#L244) | _legacy_table_overview(self, cur) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [DatabaseClient.table_schema](../../../betalens_db_manager/db.py#L268) | table_schema(self, table: str) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseClient._legacy_table_schema](../../../betalens_db_manager/db.py#L343) | _legacy_table_schema(self, cur, table: str) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseClient.query_table](../../../betalens_db_manager/db.py#L369) | query_table(self, request: QueryRequest) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [DatabaseClient.execute_readonly_sql](../../../betalens_db_manager/db.py#L419) | execute_readonly_sql(self, query: str, *, limit: int=5000) -&gt; pd.DataFrame | pd.DataFrame | Execute one bounded, read-only SQL query for the desktop explorer. This intentionally is not a general ''run_query'' compatibility API. It is a GUI-facing read path guarded both syntactically and by a PostgreSQL read-only transaction. |
| [DatabaseClient.diagnose_data](../../../betalens_db_manager/db.py#L455) | diagnose_data(self, table: str, *, sample_limit: int=10) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | Return read-only, actionable integrity checks for one logical dataset. |
| [DatabaseClient._diagnose_market_observation](../../../betalens_db_manager/db.py#L572) | _diagnose_market_observation(self, cur, spec: DatasetSpec, limit: int) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [DatabaseClient._diagnose_query](../../../betalens_db_manager/db.py#L657) | _diagnose_query(cur, issue: str, query: str, params: list[Any], limit: int) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [DatabaseClient._diagnose_legacy_table](../../../betalens_db_manager/db.py#L671) | _diagnose_legacy_table(self, cur, table: str, limit: int) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [DatabaseClient._is_date_only](../../../betalens_db_manager/db.py#L690) | _is_date_only(value: str) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [DatabaseClient._resolve_metric](../../../betalens_db_manager/db.py#L694) | _resolve_metric(self, cur, table: str, metric: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [DatabaseClient._logical_source](../../../betalens_db_manager/db.py#L711) | _logical_source(self, spec: DatasetSpec) -&gt; tuple[str, list[Any]] | tuple[str, list[Any]] | 无 docstring，需阅读函数体 |
| [DatabaseClient._query_legacy](../../../betalens_db_manager/db.py#L798) | _query_legacy(self, cur, request: QueryRequest, limit: int) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [DatabaseClient.distinct_values](../../../betalens_db_manager/db.py#L835) | distinct_values(self, table: str, column: str, limit: int=100) -&gt; list[Any] | list[Any] | 无 docstring，需阅读函数体 |
| [DatabaseClient._distinct_query](../../../betalens_db_manager/db.py#L855) | _distinct_query(self, spec: DatasetSpec, column: str) -&gt; tuple[str, list[Any]] | tuple[str, list[Any]] | 无 docstring，需阅读函数体 |
| [DatabaseClient._legacy_date_range](../../../betalens_db_manager/db.py#L874) | _legacy_date_range(self, cur, table: str) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseClient._legacy_warnings](../../../betalens_db_manager/db.py#L886) | _legacy_warnings(self, cur, table: str) -&gt; list[str] | list[str] | 无 docstring，需阅读函数体 |

<a id="file-512c49ec65e0"></a>
## betalens_db_manager/gui.py

[打开源码](../../../betalens_db_manager/gui.py) · 23 行 · 说明来源：人工文件说明

- **作用**：GUI 公共兼容启动入口
- **输入**：启动调用
- **输出**：桌面 main 导出
- **副作用/维护重点**：UI 实现主要在 gui_app.py

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .gui_app import ConnectionBar, ExplorerPage, FileImportPage, MainWindow, OnlineUpdatePage, PandasTableModel, TableCatalogPage, main
```

<a id="file-a68ba68ce6fd"></a>
## betalens_db_manager/gui_app.py

[打开源码](../../../betalens_db_manager/gui_app.py) · 1303 行 · 说明来源：人工文件说明

- **作用**：PySide6 四页桌面界面
- **输入**：用户动作、controller 结果
- **输出**：表格、对话框、进度、窗口
- **副作用/维护重点**：Qt 主线程与后台 worker 通信；不得阻塞 UI

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .constants import DEFAULT_LIMIT, INSERT_ONLY, UPSERT
from .db import QueryRequest
from .gui_controller import ConnectionDraft, FileImportPlan, GuiController
from .import_adapters import ADAPTERS
from .registry import DATASETS
from .utils import json_default
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter, QStackedWidget, QTableView, QTabWidget, QVBoxLayout, QWidget
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import ctypes
import json
import os
import pandas as pd
import sys
import traceback
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_import_qt](../../../betalens_db_manager/gui_app.py#L39) | _import_qt() | 无返回注解；return: locals() | 无 docstring，需阅读函数体 |
| [_application_icon](../../../betalens_db_manager/gui_app.py#L129) | _application_icon() -&gt; QIcon | QIcon | 无 docstring，需阅读函数体 |
| [_configure_windows_app_id](../../../betalens_db_manager/gui_app.py#L133) | _configure_windows_app_id() -&gt; None | None | Give the Windows taskbar a stable identity for this application icon. |
| [_configure_cjk_font](../../../betalens_db_manager/gui_app.py#L146) | _configure_cjk_font(app: QApplication) -&gt; None | None | Register a system CJK font before creating widgets when Qt has none. |
| [PandasTableModel](../../../betalens_db_manager/gui_app.py#L164) | class PandasTableModel(QAbstractTableModel) | 类定义；构造/属性见方法与字段 | Small read-only DataFrame model used by all four pages. |
| [PandasTableModel.__init__](../../../betalens_db_manager/gui_app.py#L167) | __init__(self, data: pd.DataFrame &#124; Sequence[Mapping[str, Any]] &#124; None=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [PandasTableModel.frame](../../../betalens_db_manager/gui_app.py#L173) | frame(self) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [PandasTableModel.value](../../../betalens_db_manager/gui_app.py#L176) | value(self, row: int, column: str, default: Any=None) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [PandasTableModel.set_data](../../../betalens_db_manager/gui_app.py#L181) | set_data(self, data: pd.DataFrame &#124; Sequence[Mapping[str, Any]] &#124; None) -&gt; None | None | 无 docstring，需阅读函数体 |
| [PandasTableModel.rowCount](../../../betalens_db_manager/gui_app.py#L191) | rowCount(self, parent=QModelIndex()) | 无返回注解；return: 0 if parent.isValid() else len(self._df) | 无 docstring，需阅读函数体 |
| [PandasTableModel.columnCount](../../../betalens_db_manager/gui_app.py#L194) | columnCount(self, parent=QModelIndex()) | 无返回注解；return: 0 if parent.isValid() else len(self._df.columns) | 无 docstring，需阅读函数体 |
| [PandasTableModel.data](../../../betalens_db_manager/gui_app.py#L197) | data(self, index, role=Qt.DisplayRole) | 无返回注解；return: None; QColor('#fff0f0'); self._display(value) | 无 docstring，需阅读函数体 |
| [PandasTableModel._display](../../../betalens_db_manager/gui_app.py#L211) | _display(value: Any) -&gt; str | str | 无 docstring，需阅读函数体 |
| [PandasTableModel.headerData](../../../betalens_db_manager/gui_app.py#L223) | headerData(self, section, orientation, role=Qt.DisplayRole) | 无返回注解；return: None; str(self._df.columns[section]) if section &lt; len(self._df.columns) else ''; str(section + 1) | 无 docstring，需阅读函数体 |
| [RejectedRowsDialog](../../../betalens_db_manager/gui_app.py#L231) | class RejectedRowsDialog(QDialog) | 类定义；构造/属性见方法与字段 | Show the bounded rejected-row sample produced during file preflight. |
| [RejectedRowsDialog.__init__](../../../betalens_db_manager/gui_app.py#L234) | __init__(self, rows: Sequence[Mapping[str, Any]], parent=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [WorkerSignals](../../../betalens_db_manager/gui_app.py#L252) | class WorkerSignals(QObject) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [Worker](../../../betalens_db_manager/gui_app.py#L258) | class Worker(QRunnable) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [Worker.__init__](../../../betalens_db_manager/gui_app.py#L259) | __init__(self, operation: Callable[[], Any]) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [Worker.run](../../../betalens_db_manager/gui_app.py#L265) | run(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [AsyncPage](../../../betalens_db_manager/gui_app.py#L272) | class AsyncPage(QWidget) | 类定义；构造/属性见方法与字段 | Shared background execution with duplicate-action protection. |
| [AsyncPage.__init__](../../../betalens_db_manager/gui_app.py#L275) | __init__(self, pool: QThreadPool, parent=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [AsyncPage.start_task](../../../betalens_db_manager/gui_app.py#L280) | start_task(self, key: str, operation: Callable[[Callable[[dict[str, Any]], None]], Any], on_finished: Callable[[Any], None], on_error: Callable[[str], None] &#124; None=None, on_progress: Callable[[dict[str, Any]], None] &#124; None=None) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [AsyncPage.start_task.invoke](../../../betalens_db_manager/gui_app.py#L292) | invoke() | 无返回注解；return: operation(holder['worker'].signals.progress.emit) | 无 docstring，需阅读函数体 |
| [AsyncPage.start_task.complete](../../../betalens_db_manager/gui_app.py#L299) | complete(result, task_key=key) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [AsyncPage.start_task.failed](../../../betalens_db_manager/gui_app.py#L304) | failed(message, task_key=key) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [AsyncPage._task_state_changed](../../../betalens_db_manager/gui_app.py#L320) | _task_state_changed(self) -&gt; None | None | Hook for pages that need to disable duplicate buttons. |
| [AsyncPage.show_error](../../../betalens_db_manager/gui_app.py#L323) | show_error(self, message: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_set_table](../../../betalens_db_manager/gui_app.py#L327) | _set_table(table: QTableView, model: PandasTableModel) -&gt; None | None | 无 docstring，需阅读函数体 |
| [_json_text](../../../betalens_db_manager/gui_app.py#L334) | _json_text(payload: Any) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_progress_update](../../../betalens_db_manager/gui_app.py#L338) | _progress_update(bar: QProgressBar, status: QLabel, payload: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ConnectionBar](../../../betalens_db_manager/gui_app.py#L350) | class ConnectionBar(AsyncPage) | 类定义；构造/属性见方法与字段 | Single, password-in-memory connection bar shared by the window. |
| [ConnectionBar.__init__](../../../betalens_db_manager/gui_app.py#L355) | __init__(self, controller: GuiController, pool: QThreadPool, parent=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [ConnectionBar._draft](../../../betalens_db_manager/gui_app.py#L383) | _draft(self) -&gt; ConnectionDraft | ConnectionDraft | 无 docstring，需阅读函数体 |
| [ConnectionBar.connect](../../../betalens_db_manager/gui_app.py#L392) | connect(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ConnectionBar._connected](../../../betalens_db_manager/gui_app.py#L408) | _connected(self, result: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ConnectionBar._connect_error](../../../betalens_db_manager/gui_app.py#L413) | _connect_error(self, message: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ConnectionBar.set_result](../../../betalens_db_manager/gui_app.py#L418) | set_result(self, result: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TableCatalogPage](../../../betalens_db_manager/gui_app.py#L431) | class TableCatalogPage(AsyncPage) | 类定义；构造/属性见方法与字段 | Logical dataset list and safe one-click contract bootstrap. |
| [TableCatalogPage.__init__](../../../betalens_db_manager/gui_app.py#L436) | __init__(self, controller: GuiController, pool: QThreadPool, parent=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [TableCatalogPage.set_connection_state](../../../betalens_db_manager/gui_app.py#L467) | set_connection_state(self, state: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TableCatalogPage._update_actions](../../../betalens_db_manager/gui_app.py#L472) | _update_actions(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TableCatalogPage._task_state_changed](../../../betalens_db_manager/gui_app.py#L480) | _task_state_changed(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TableCatalogPage._selected_table](../../../betalens_db_manager/gui_app.py#L483) | _selected_table(self) -&gt; str &#124; None | str &#124; None | 无 docstring，需阅读函数体 |
| [TableCatalogPage._set_catalog](../../../betalens_db_manager/gui_app.py#L490) | _set_catalog(self, rows: Sequence[Mapping[str, Any]]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TableCatalogPage.refresh](../../../betalens_db_manager/gui_app.py#L506) | refresh(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TableCatalogPage._catalog_loaded](../../../betalens_db_manager/gui_app.py#L525) | _catalog_loaded(self, rows: Sequence[Mapping[str, Any]]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TableCatalogPage._catalog_error](../../../betalens_db_manager/gui_app.py#L530) | _catalog_error(self, message: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TableCatalogPage._selection_changed](../../../betalens_db_manager/gui_app.py#L535) | _selection_changed(self, *_args) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TableCatalogPage._show_metadata](../../../betalens_db_manager/gui_app.py#L541) | _show_metadata(self, table: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TableCatalogPage.create_selected](../../../betalens_db_manager/gui_app.py#L553) | create_selected(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TableCatalogPage._created](../../../betalens_db_manager/gui_app.py#L578) | _created(self, report: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TableCatalogPage._create_error](../../../betalens_db_manager/gui_app.py#L589) | _create_error(self, message: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage](../../../betalens_db_manager/gui_app.py#L595) | class FileImportPage(AsyncPage) | 类定义；构造/属性见方法与字段 | Single file and recursive folder import through one preflight flow. |
| [FileImportPage.__init__](../../../betalens_db_manager/gui_app.py#L598) | __init__(self, controller: GuiController, pool: QThreadPool, parent=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [FileImportPage.set_connection_state](../../../betalens_db_manager/gui_app.py#L668) | set_connection_state(self, state: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._target_changed](../../../betalens_db_manager/gui_app.py#L674) | _target_changed(self, *_args) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._refresh_adapters](../../../betalens_db_manager/gui_app.py#L678) | _refresh_adapters(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._update_actions](../../../betalens_db_manager/gui_app.py#L697) | _update_actions(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._task_state_changed](../../../betalens_db_manager/gui_app.py#L720) | _task_state_changed(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._has_rejection_details](../../../betalens_db_manager/gui_app.py#L723) | _has_rejection_details(self) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [FileImportPage._failed_file_paths](../../../betalens_db_manager/gui_app.py#L734) | _failed_file_paths(self) -&gt; tuple[Path, ...] | tuple[Path, ...] | 无 docstring，需阅读函数体 |
| [FileImportPage.choose_files](../../../betalens_db_manager/gui_app.py#L739) | choose_files(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage.choose_folder](../../../betalens_db_manager/gui_app.py#L749) | choose_folder(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._set_sources](../../../betalens_db_manager/gui_app.py#L754) | _set_sources(self, paths: Sequence[Path]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._scanned](../../../betalens_db_manager/gui_app.py#L768) | _scanned(self, files: Sequence[Path]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._scan_error](../../../betalens_db_manager/gui_app.py#L780) | _scan_error(self, message: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage.invalidate_plan](../../../betalens_db_manager/gui_app.py#L785) | invalidate_plan(self, *_args) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._options](../../../betalens_db_manager/gui_app.py#L789) | _options(self) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [FileImportPage._current_spec](../../../betalens_db_manager/gui_app.py#L803) | _current_spec(self) -&gt; tuple[str, str, str, dict[str, Any]] | tuple[str, str, str, dict[str, Any]] | 无 docstring，需阅读函数体 |
| [FileImportPage.preflight](../../../betalens_db_manager/gui_app.py#L809) | preflight(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._preflight_done](../../../betalens_db_manager/gui_app.py#L837) | _preflight_done(self, plan: FileImportPlan) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._preflight_error](../../../betalens_db_manager/gui_app.py#L846) | _preflight_error(self, message: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage.show_rejected_rows](../../../betalens_db_manager/gui_app.py#L852) | show_rejected_rows(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage.export_failed_files](../../../betalens_db_manager/gui_app.py#L885) | export_failed_files(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage.run_import](../../../betalens_db_manager/gui_app.py#L906) | run_import(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._import_done](../../../betalens_db_manager/gui_app.py#L931) | _import_done(self, report: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._import_error](../../../betalens_db_manager/gui_app.py#L956) | _import_error(self, message: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FileImportPage._progress](../../../betalens_db_manager/gui_app.py#L962) | _progress(self, payload: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage](../../../betalens_db_manager/gui_app.py#L966) | class ExplorerPage(AsyncPage) | 类定义；构造/属性见方法与字段 | Filtered logical queries, restricted SQL, and data-quality checks. |
| [ExplorerPage.__init__](../../../betalens_db_manager/gui_app.py#L969) | __init__(self, controller: GuiController, pool: QThreadPool, parent=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [ExplorerPage._build_filter_page](../../../betalens_db_manager/gui_app.py#L1009) | _build_filter_page(self) -&gt; QWidget | QWidget | 无 docstring，需阅读函数体 |
| [ExplorerPage._build_sql_page](../../../betalens_db_manager/gui_app.py#L1044) | _build_sql_page(self) -&gt; QWidget | QWidget | 无 docstring，需阅读函数体 |
| [ExplorerPage._build_diagnose_page](../../../betalens_db_manager/gui_app.py#L1063) | _build_diagnose_page(self) -&gt; QWidget | QWidget | 无 docstring，需阅读函数体 |
| [ExplorerPage.set_connection_state](../../../betalens_db_manager/gui_app.py#L1080) | set_connection_state(self, state: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage._set_online_controls](../../../betalens_db_manager/gui_app.py#L1086) | _set_online_controls(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage._task_state_changed](../../../betalens_db_manager/gui_app.py#L1096) | _task_state_changed(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage._mode_changed](../../../betalens_db_manager/gui_app.py#L1099) | _mode_changed(self, index: int) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage._filter_request](../../../betalens_db_manager/gui_app.py#L1105) | _filter_request(self, page_token: str &#124; None) -&gt; QueryRequest | QueryRequest | 无 docstring，需阅读函数体 |
| [ExplorerPage.start_filter_query](../../../betalens_db_manager/gui_app.py#L1116) | start_filter_query(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage._fetch_filter_page](../../../betalens_db_manager/gui_app.py#L1121) | _fetch_filter_page(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage._filter_done](../../../betalens_db_manager/gui_app.py#L1132) | _filter_done(self, frame: pd.DataFrame, request: QueryRequest) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage.next_page](../../../betalens_db_manager/gui_app.py#L1140) | next_page(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage.previous_page](../../../betalens_db_manager/gui_app.py#L1149) | previous_page(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage.run_sql](../../../betalens_db_manager/gui_app.py#L1155) | run_sql(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage._sql_done](../../../betalens_db_manager/gui_app.py#L1166) | _sql_done(self, frame: pd.DataFrame) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage.run_diagnosis](../../../betalens_db_manager/gui_app.py#L1170) | run_diagnosis(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage._diagnosis_done](../../../betalens_db_manager/gui_app.py#L1183) | _diagnosis_done(self, rows: Sequence[Mapping[str, Any]]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage._set_result](../../../betalens_db_manager/gui_app.py#L1187) | _set_result(self, frame: pd.DataFrame) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage._query_error](../../../betalens_db_manager/gui_app.py#L1193) | _query_error(self, message: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ExplorerPage.export_current](../../../betalens_db_manager/gui_app.py#L1198) | export_current(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [OnlineUpdatePage](../../../betalens_db_manager/gui_app.py#L1222) | class OnlineUpdatePage(QWidget) | 类定义；构造/属性见方法与字段 | Reserved empty page for a future connected-data update workflow. |
| [MainWindow](../../../betalens_db_manager/gui_app.py#L1228) | class MainWindow(QMainWindow) | 类定义；构造/属性见方法与字段 | The complete desktop app: exactly four beginner-oriented pages. |
| [MainWindow.__init__](../../../betalens_db_manager/gui_app.py#L1231) | __init__(self, controller: GuiController &#124; None=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [MainWindow._connection_changed](../../../betalens_db_manager/gui_app.py#L1262) | _connection_changed(self, result: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [MainWindow._schema_changed](../../../betalens_db_manager/gui_app.py#L1280) | _schema_changed(self, result: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [main](../../../betalens_db_manager/gui_app.py#L1284) | main(argv: list[str] &#124; None=None) -&gt; int | int | 无 docstring，需阅读函数体 |

<a id="file-035c4ba0651b"></a>
## betalens_db_manager/gui_controller.py

[打开源码](../../../betalens_db_manager/gui_controller.py) · 490 行 · 说明来源：人工文件说明

- **作用**：无 Qt 的界面业务控制
- **输入**：用户连接草稿、文件计划、操作请求
- **输出**：服务结果与可执行计划
- **副作用/维护重点**：防并发操作与失效预览；便于脱离 GUI 测试

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .constants import IMPORT_MODES, INSERT_ONLY
from .db import DatabaseClient, QueryRequest
from .import_adapters import ADAPTERS
from .jobs import ImportJobRunner
from .manager import DatabaseManager
from .registry import DATASETS, get_dataset
from .utils import json_default
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import hashlib
import json
import os
import threading
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [BusyOperationError](../../../betalens_db_manager/gui_controller.py#L29) | class BusyOperationError(RuntimeError) | 类定义；构造/属性见方法与字段 | Raised when a second database-changing GUI operation starts. |
| [StalePlanError](../../../betalens_db_manager/gui_controller.py#L33) | class StalePlanError(RuntimeError) | 类定义；构造/属性见方法与字段 | Raised when a file changed after the user checked it. |
| [OperationRegistry](../../../betalens_db_manager/gui_controller.py#L37) | class OperationRegistry() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [OperationRegistry.__init__](../../../betalens_db_manager/gui_controller.py#L38) | __init__(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [OperationRegistry.claim](../../../betalens_db_manager/gui_controller.py#L43) | claim(self, name: str) -&gt; Iterator[None] | Iterator[None] | 无 docstring，需阅读函数体 |
| [OperationRegistry.active](../../../betalens_db_manager/gui_controller.py#L54) | active(self) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [ConnectionDraft](../../../betalens_db_manager/gui_controller.py#L60) | class ConnectionDraft() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：dbname: str; user: str; host: str = 'localhost'; port: str = '5432'; password: str &#124; None = None |
| [ConnectionDraft.from_config](../../../betalens_db_manager/gui_controller.py#L68) | from_config(cls, config: Mapping[str, Any]) -&gt; 'ConnectionDraft' | 'ConnectionDraft' | 无 docstring，需阅读函数体 |
| [ConnectionDraft.as_config](../../../betalens_db_manager/gui_controller.py#L77) | as_config(self) -&gt; dict[str, str] | dict[str, str] | 无 docstring，需阅读函数体 |
| [FileImportItem](../../../betalens_db_manager/gui_controller.py#L100) | class FileImportItem() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：path: Path; source_sha256: str &#124; None; preview_token: str &#124; None; summary: Mapping[str, Any] = field(default_factory=dict); validation: Mapping[str, Any] = field(default_factory=dict); rejected_preview: tuple[Mapping[str, Any], ...] = (); error: str &#124; None = None |
| [FileImportItem.ready](../../../betalens_db_manager/gui_controller.py#L110) | ready(self) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [FileImportItem.as_dict](../../../betalens_db_manager/gui_controller.py#L113) | as_dict(self) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [FileImportPlan](../../../betalens_db_manager/gui_controller.py#L127) | class FileImportPlan() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：table: str; adapter: str; mode: str; options: Mapping[str, Any]; items: tuple[FileImportItem, ...]; fingerprint: str; source_label: str |
| [FileImportPlan.ready_items](../../../betalens_db_manager/gui_controller.py#L137) | ready_items(self) -&gt; tuple[FileImportItem, ...] | tuple[FileImportItem, ...] | 无 docstring，需阅读函数体 |
| [GuiController](../../../betalens_db_manager/gui_controller.py#L141) | class GuiController() | 类定义；构造/属性见方法与字段 | One-purpose actions for four simple pages, independent of PySide6. |
| [GuiController.__init__](../../../betalens_db_manager/gui_controller.py#L144) | __init__(self, manager: DatabaseManager &#124; None=None, *, client: DatabaseClient &#124; None=None, runner: ImportJobRunner &#124; None=None, manager_factory: Callable[[Mapping[str, Any]], DatabaseManager]=DatabaseManager) -&gt; None | None | 无 docstring，需阅读函数体 |
| [GuiController.connection_draft](../../../betalens_db_manager/gui_controller.py#L161) | connection_draft(self) -&gt; ConnectionDraft | ConnectionDraft | 无 docstring，需阅读函数体 |
| [GuiController.connect](../../../betalens_db_manager/gui_controller.py#L164) | connect(self, draft: ConnectionDraft &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [GuiController.is_online](../../../betalens_db_manager/gui_controller.py#L174) | is_online(self) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [GuiController.table_catalog](../../../betalens_db_manager/gui_controller.py#L177) | table_catalog(self) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [GuiController.table_metadata](../../../betalens_db_manager/gui_controller.py#L210) | table_metadata(self, table: str) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [GuiController.create_selected_table](../../../betalens_db_manager/gui_controller.py#L244) | create_selected_table(self, table: str) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [GuiController.discover_files](../../../betalens_db_manager/gui_controller.py#L254) | discover_files(paths: Sequence[str &#124; Path]) -&gt; list[Path] | list[Path] | 无 docstring，需阅读函数体 |
| [GuiController._plan_fingerprint](../../../betalens_db_manager/gui_controller.py#L271) | _plan_fingerprint(table: str, adapter: str, mode: str, options: Mapping[str, Any], items: Iterable[FileImportItem]) -&gt; str | str | 无 docstring，需阅读函数体 |
| [GuiController._file_workers](../../../betalens_db_manager/gui_controller.py#L294) | _file_workers(file_count: int, max_workers: int &#124; None) -&gt; int | int | Return a bounded worker count for file-level I/O and parsing. |
| [GuiController.preflight_import](../../../betalens_db_manager/gui_controller.py#L304) | preflight_import(self, paths: Sequence[str &#124; Path], *, table: str, adapter: str, mode: str=INSERT_ONLY, options: Mapping[str, Any] &#124; None=None, progress: ProgressCallback &#124; None=None, max_workers: int &#124; None=None) -&gt; FileImportPlan | FileImportPlan | 无 docstring，需阅读函数体 |
| [GuiController.run_import_plan](../../../betalens_db_manager/gui_controller.py#L385) | run_import_plan(self, plan: FileImportPlan, *, progress: ProgressCallback &#124; None=None, max_workers: int &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [GuiController.query](../../../betalens_db_manager/gui_controller.py#L464) | query(self, request: QueryRequest) | 无返回注解；return: self.client.query_table(request) | 无 docstring，需阅读函数体 |
| [GuiController.execute_sql](../../../betalens_db_manager/gui_controller.py#L469) | execute_sql(self, statement: str, *, limit: int=5000) | 无返回注解；return: self.client.execute_readonly_sql(statement, limit=limit) | 无 docstring，需阅读函数体 |
| [GuiController.diagnose_dirty_data](../../../betalens_db_manager/gui_controller.py#L474) | diagnose_dirty_data(self, table: str, *, sample_limit: int=10) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |

<a id="file-b6247cc64df5"></a>
## betalens_db_manager/import_adapters.py

[打开源码](../../../betalens_db_manager/import_adapters.py) · 1036 行 · 说明来源：人工文件说明

- **作用**：带目标类型的源数据适配注册
- **输入**：adapter 名、源文件、目标表、options
- **输出**：ImportBatch 子类型及拒绝行
- **副作用/维护重点**：读取文件、校验与时间对齐；注册后才能参与统一导入

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .adapters.ede import DEFAULT_CODE_COLUMNS, DEFAULT_DATE_PATTERN, DEFAULT_NAME_COLUMNS, DEFAULT_TIME, clean_ede_dataframe, extract_date_from_filename, extract_date_from_metric_metadata, identify_code_name_columns, parse_metric_column
from .adapters.files import DEFAULT_CHUNK_SIZE, iter_file_chunks
from .constants import DB_COLUMNS, INSERT_ONLY, UPSERT
from .registry import DATASETS
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Iterator, Mapping
import json
import logging
import numpy as np
import pandas as pd
import re
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [BatchKind](../../../betalens_db_manager/import_adapters.py#L41) | class BatchKind(str, Enum) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [ImportBatch](../../../betalens_db_manager/import_adapters.py#L51) | class ImportBatch() | 类定义；构造/属性见方法与字段 | One bounded, validated batch ready for ''DatabaseWriter''.；字段：table: str; frame: pd.DataFrame; rejected: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=['_source_row', '_errors'])); warnings: tuple[str, ...] = (); source_rows: int = 0; typed_fields: Mapping[str, Any] = field(defau…（完整内容见 inventory.json/源码） |
| [ImportBatch.__post_init__](../../../betalens_db_manager/import_adapters.py#L68) | __post_init__(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [MarketBatch](../../../betalens_db_manager/import_adapters.py#L83) | class MarketBatch(ImportBatch) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [ObservationBatch](../../../betalens_db_manager/import_adapters.py#L88) | class ObservationBatch(ImportBatch) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [IndustryBatch](../../../betalens_db_manager/import_adapters.py#L93) | class IndustryBatch(ImportBatch) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [IndexSnapshotBatch](../../../betalens_db_manager/import_adapters.py#L98) | class IndexSnapshotBatch(ImportBatch) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [TradeStatusBatch](../../../betalens_db_manager/import_adapters.py#L103) | class TradeStatusBatch(ImportBatch) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [TradeCalendarBatch](../../../betalens_db_manager/import_adapters.py#L108) | class TradeCalendarBatch(ImportBatch) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [AdapterContext](../../../betalens_db_manager/import_adapters.py#L124) | class AdapterContext() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：path: Path; table: str; options: Mapping[str, Any]; chunk_size: int; logger: logging.Logger |
| [AdapterSpec](../../../betalens_db_manager/import_adapters.py#L136) | class AdapterSpec() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：name: str; loader: AdapterLoader; allowed_targets: frozenset[str]; option_keys: frozenset[str] = frozenset(); required_options: frozenset[str] = frozenset(); aliases: tuple[str, ...] = () |
| [AdapterSpec.validate](../../../betalens_db_manager/import_adapters.py#L144) | validate(self, table: str, options: Mapping[str, Any], *, strict: bool) -&gt; None | None | 无 docstring，需阅读函数体 |
| [AdapterRegistry](../../../betalens_db_manager/import_adapters.py#L156) | class AdapterRegistry() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [AdapterRegistry.__init__](../../../betalens_db_manager/import_adapters.py#L157) | __init__(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [AdapterRegistry.register](../../../betalens_db_manager/import_adapters.py#L160) | register(self, spec: AdapterSpec) -&gt; None | None | 无 docstring，需阅读函数体 |
| [AdapterRegistry.resolve](../../../betalens_db_manager/import_adapters.py#L166) | resolve(self, name: str) -&gt; AdapterSpec | AdapterSpec | 无 docstring，需阅读函数体 |
| [AdapterRegistry.names](../../../betalens_db_manager/import_adapters.py#L172) | names(self, *, include_aliases: bool=True) -&gt; tuple[str, ...] | tuple[str, ...] | 无 docstring，需阅读函数体 |
| [AdapterRegistry.validate](../../../betalens_db_manager/import_adapters.py#L177) | validate(self, name: str, table: str, options: Mapping[str, Any] &#124; None=None, *, strict_options: bool=False) -&gt; AdapterSpec | AdapterSpec | 无 docstring，需阅读函数体 |
| [_batch_class](../../../betalens_db_manager/import_adapters.py#L242) | _batch_class(table: str) -&gt; type[ImportBatch] | type[ImportBatch] | 无 docstring，需阅读函数体 |
| [_apply_column_map](../../../betalens_db_manager/import_adapters.py#L246) | _apply_column_map(frame: pd.DataFrame, options: Mapping[str, Any]) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_parse_remark](../../../betalens_db_manager/import_adapters.py#L260) | _parse_remark(value: Any, *, text_key: str &#124; None) -&gt; tuple[Any, str &#124; None] | tuple[Any, str &#124; None] | 无 docstring，需阅读函数体 |
| [_canonicalize_market_time](../../../betalens_db_manager/import_adapters.py#L281) | _canonicalize_market_time(frame: pd.DataFrame, table: str) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_special_errors](../../../betalens_db_manager/import_adapters.py#L295) | _special_errors(row: pd.Series, table: str) -&gt; list[str] | list[str] | 无 docstring，需阅读函数体 |
| [_normalize_long](../../../betalens_db_manager/import_adapters.py#L316) | _normalize_long(source: pd.DataFrame, context: AdapterContext, *, row_offset: int) -&gt; ImportBatch | ImportBatch | 无 docstring，需阅读函数体 |
| [_iter_source](../../../betalens_db_manager/import_adapters.py#L454) | _iter_source(context: AdapterContext) -&gt; Iterator[pd.DataFrame] | Iterator[pd.DataFrame] | 无 docstring，需阅读函数体 |
| [_standard_long_loader](../../../betalens_db_manager/import_adapters.py#L466) | _standard_long_loader(context: AdapterContext, sources: Iterable[pd.DataFrame] &#124; None=None) -&gt; Iterator[ImportBatch] | Iterator[ImportBatch] | 无 docstring，需阅读函数体 |
| [_is_daily_wide_frame](../../../betalens_db_manager/import_adapters.py#L478) | _is_daily_wide_frame(frame: pd.DataFrame) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [_auto_loader](../../../betalens_db_manager/import_adapters.py#L486) | _auto_loader(context: AdapterContext) -&gt; Iterator[ImportBatch] | Iterator[ImportBatch] | Route ordinary user files to a bounded, column-based parser. The desktop importer should not make a first-time user decide whether a Wind/EDE export is a long or a wide table. This is deliberately limited to the three well-understood shapes below; special PIT datasets keep their dedicated adapters a…（完整内容见 inventory.json/源码） |
| [_wind_wide_loader](../../../betalens_db_manager/import_adapters.py#L521) | _wind_wide_loader(context: AdapterContext, sources: Iterable[pd.DataFrame] &#124; None=None) -&gt; Iterator[ImportBatch] | Iterator[ImportBatch] | 无 docstring，需阅读函数体 |
| [_ede_loader](../../../betalens_db_manager/import_adapters.py#L583) | _ede_loader(context: AdapterContext, sources: Iterable[pd.DataFrame] &#124; None=None) -&gt; Iterator[ImportBatch] | Iterator[ImportBatch] | 无 docstring，需阅读函数体 |
| [_industry_loader](../../../betalens_db_manager/import_adapters.py#L651) | _industry_loader(context: AdapterContext) -&gt; Iterator[ImportBatch] | Iterator[ImportBatch] | 无 docstring，需阅读函数体 |
| [_index_loader](../../../betalens_db_manager/import_adapters.py#L692) | _index_loader(context: AdapterContext) -&gt; Iterator[ImportBatch] | Iterator[ImportBatch] | 无 docstring，需阅读函数体 |
| [_trade_status_loader](../../../betalens_db_manager/import_adapters.py#L741) | _trade_status_loader(context: AdapterContext) -&gt; Iterator[ImportBatch] | Iterator[ImportBatch] | 无 docstring，需阅读函数体 |
| [_trade_calendar_loader](../../../betalens_db_manager/import_adapters.py#L795) | _trade_calendar_loader(context: AdapterContext) -&gt; Iterator[ImportBatch] | Iterator[ImportBatch] | 无 docstring，需阅读函数体 |
| [_register_defaults](../../../betalens_db_manager/import_adapters.py#L870) | _register_defaults() -&gt; None | None | 无 docstring，需阅读函数体 |
| [infer_adapter](../../../betalens_db_manager/import_adapters.py#L962) | infer_adapter(path: str &#124; Path) -&gt; str | str | Conservatively infer an adapter without treating index prices as constituents. |
| [load_import_batches](../../../betalens_db_manager/import_adapters.py#L979) | load_import_batches(import_type: str &#124; None, path: str &#124; Path, *, table: str, options: Mapping[str, Any] &#124; None=None, logger: logging.Logger &#124; None=None, strict_options: bool=False) -&gt; Iterator[ImportBatch] | Iterator[ImportBatch] | Load a source as bounded, typed import batches. |
| [collect_import_batches](../../../betalens_db_manager/import_adapters.py#L1009) | collect_import_batches(batches: Iterable[ImportBatch]) -&gt; tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame] | Materialize batches for the legacy DataFrame API. |

<a id="file-0b181dfe56e8"></a>
## betalens_db_manager/import_manifest.example.yaml

[打开源码](../../../betalens_db_manager/import_manifest.example.yaml) · 34 行 · 说明来源：文件族规则

- **作用**：运行/构建声明式配置
- **输入**：维护者填写的参数
- **输出**：由对应读取器解释的配置对象
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L2](../../../betalens_db_manager/import_manifest.example.yaml#L2)：`version: 1`
- [L4](../../../betalens_db_manager/import_manifest.example.yaml#L4)：`defaults:`
- [L10](../../../betalens_db_manager/import_manifest.example.yaml#L10)：`imports:`

<a id="file-b0dc95aa2a97"></a>
## betalens_db_manager/import_manifest.py

[打开源码](../../../betalens_db_manager/import_manifest.py) · 591 行 · 说明来源：人工文件说明

- **作用**：多文件清单预检与恢复
- **输入**：manifest YAML、默认值、检查点
- **输出**：ManifestPlan、逐文件运行结果
- **副作用/维护重点**：批量读写与检查点；整份清单不是单一事务

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .constants import INSERT_ONLY, IMPORT_MODES
from .contracts import LATEST_SCHEMA_VERSION
from .import_adapters import ADAPTERS, infer_adapter, load_import_batches
from .job_store import JobStore
from .jobs import ImportJobRunner
from .utils import file_sha256, json_default
from .validators import validate_import_frame
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
import glob
import hashlib
import json
import pandas as pd
import yaml
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [CheckpointStore](../../../betalens_db_manager/import_manifest.py#L35) | class CheckpointStore(Protocol) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [CheckpointStore.load](../../../betalens_db_manager/import_manifest.py#L36) | load(self, token: str) -&gt; dict[str, Any] &#124; None | dict[str, Any] &#124; None | 无 docstring，需阅读函数体 |
| [CheckpointStore.save](../../../betalens_db_manager/import_manifest.py#L38) | save(self, token: str, payload: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ManifestEntry](../../../betalens_db_manager/import_manifest.py#L42) | class ManifestEntry() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：item_id: str; group_id: str; position: int; path: Path; target: str; adapter: str; mode: str = INSERT_ONLY; options: dict[str, Any] = field(default_factory=dict); allow_unsafe_metrics: bool = False; on_rejected: str = 'fail'; sha256: str = ''; preview_token: str = '' |
| [ManifestEntry.table](../../../betalens_db_manager/import_manifest.py#L57) | table(self) -&gt; str | str | 无 docstring，需阅读函数体 |
| [ManifestEntry.import_type](../../../betalens_db_manager/import_manifest.py#L61) | import_type(self) -&gt; str | str | 无 docstring，需阅读函数体 |
| [ManifestEntry.as_dict](../../../betalens_db_manager/import_manifest.py#L64) | as_dict(self) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [ManifestPlan](../../../betalens_db_manager/import_manifest.py#L85) | class ManifestPlan() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：version: int; path: Path; token: str; entries: tuple[ManifestEntry, ...]; target_database: str; schema_version: int; on_error: str = 'continue'; warnings: tuple[str, ...] = (); previews: tuple[dict[str, Any], ...] = () |
| [ManifestPlan.as_dict](../../../betalens_db_manager/import_manifest.py#L96) | as_dict(self) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [ManifestRunner](../../../betalens_db_manager/import_manifest.py#L110) | class ManifestRunner() | 类定义；构造/属性见方法与字段 | Preflight every source, then commit one transaction per source file. |
| [ManifestRunner.__init__](../../../betalens_db_manager/import_manifest.py#L115) | __init__(self, job_runner: ImportJobRunner &#124; None=None, checkpoint_store: CheckpointStore &#124; None=None, *, target_database: str &#124; None=None, schema_version: int=LATEST_SCHEMA_VERSION) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ManifestRunner._normalize_options](../../../betalens_db_manager/import_manifest.py#L131) | _normalize_options(options: Mapping[str, Any] &#124; None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [ManifestRunner._expand_path](../../../betalens_db_manager/import_manifest.py#L140) | _expand_path(manifest_path: Path, value: Any) -&gt; list[Path] | list[Path] | 无 docstring，需阅读函数体 |
| [ManifestRunner._effective_value](../../../betalens_db_manager/import_manifest.py#L161) | _effective_value(raw: Mapping[str, Any], defaults: Mapping[str, Any], *keys: str) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [ManifestRunner._expected_hash](../../../betalens_db_manager/import_manifest.py#L171) | _expected_hash(value: Any, source: Path, sources: list[Path]) -&gt; str &#124; None | str &#124; None | 无 docstring，需阅读函数体 |
| [ManifestRunner.preflight](../../../betalens_db_manager/import_manifest.py#L187) | preflight(self, path: str &#124; Path) -&gt; ManifestPlan | ManifestPlan | 无 docstring，需阅读函数体 |
| [ManifestRunner._preview_source](../../../betalens_db_manager/import_manifest.py#L342) | _preview_source(self, entry: ManifestEntry) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [ManifestRunner.from_dict](../../../betalens_db_manager/import_manifest.py#L408) | from_dict(payload: Mapping[str, Any]) -&gt; ManifestPlan | ManifestPlan | 无 docstring，需阅读函数体 |
| [ManifestRunner.run](../../../betalens_db_manager/import_manifest.py#L438) | run(self, path_or_plan: str &#124; Path &#124; ManifestPlan &#124; Mapping[str, Any], *, resume: bool=True, on_error: str &#124; None=None, progress: ProgressCallback &#124; None=None, cancel: CancelCallback &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [ManifestRunner._emit](../../../betalens_db_manager/import_manifest.py#L578) | _emit(callback: ProgressCallback &#124; None, event: dict[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-dbbdf935cf84"></a>
## betalens_db_manager/importers.py

[打开源码](../../../betalens_db_manager/importers.py) · 1707 行 · 说明来源：人工文件说明

- **作用**：旧导入接口兼容与数据库写入器
- **输入**：标准表/批次、目标、写入模式/DeleteRequest
- **输出**：dry-run、写入或删除统计
- **副作用/维护重点**：临时 staging、COPY、事务和分区；负责实际写库

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .constants import DB_COLUMNS, INSERT_ONLY, UPSERT
from .db import DatabaseClient
from .import_adapters import ImportBatch, collect_import_batches, infer_adapter, load_import_batches
from .registry import DATASETS, get_dataset
from .schema import SchemaManager
from .utils import dataframe_summary
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from psycopg2 import sql
from typing import Any, Callable, Iterable, Sequence
import csv
import io
import json
import logging
import numpy as np
import pandas as pd
import psycopg2.extensions
import psycopg2.extras
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [normalize_import_frame](../../../betalens_db_manager/importers.py#L32) | normalize_import_frame(df: pd.DataFrame) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [normalize_import_frame.parse_remark](../../../betalens_db_manager/importers.py#L44) | parse_remark(value) | 无返回注解；return: None; value; parsed | 无 docstring，需阅读函数体 |
| [load_ede](../../../betalens_db_manager/importers.py#L66) | load_ede(path: str &#124; Path, date_from: str='filename', default_datetime: str &#124; None=None, logger: logging.Logger &#124; None=None, *, code_column_names: Sequence[str] &#124; None=None, name_column_names: Sequence[str] &#124; None=None, date_pattern: str='(\\d{8})', default_time: str='15:30:00', keywords_to_remove: Sequence[str] &#124; None=None, table: str='daily_market', column_map: dict[str, str] &#124; None=None, chunk_size: int=100000) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [load_wind_long](../../../betalens_db_manager/importers.py#L105) | load_wind_long(path: str &#124; Path, logger: logging.Logger &#124; None=None, *, open_metric_names: Sequence[str] &#124; None=None, open_time: str='09:30:01', other_time: str='15:00:01', table: str='daily_market', column_map: dict[str, str] &#124; None=None, chunk_size: int=100000) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [load_index_universe](../../../betalens_db_manager/importers.py#L136) | load_index_universe(path: str &#124; Path, index_code: str, index_name: str, sheet_name: str='Sheet2', seq_col: str='序号') -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [load_trade_status](../../../betalens_db_manager/importers.py#L155) | load_trade_status(path: str &#124; Path, sheet_name: str='Sheet1') -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [load_trade_calendar](../../../betalens_db_manager/importers.py#L169) | load_trade_calendar(path: str &#124; Path, sheet_name: str='Sheet1') -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [infer_import_type](../../../betalens_db_manager/importers.py#L183) | infer_import_type(path: str &#124; Path) -&gt; str | str | 无 docstring，需阅读函数体 |
| [load_import_frame](../../../betalens_db_manager/importers.py#L187) | load_import_frame(import_type: str &#124; None, path: str &#124; Path, options: dict[str, Any] &#124; None=None, logger: logging.Logger &#124; None=None, *, table: str &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [DeleteRequest](../../../betalens_db_manager/importers.py#L216) | class DeleteRequest() | 类定义；构造/属性见方法与字段 | A deliberately constrained logical deletion request.；字段：table: str; code: str &#124; None = None; codes: Sequence[str] &#124; None = None; metric: str &#124; None = None; start_date: str &#124; None = None; end_date: str &#124; None = None |
| [DeleteRequest.normalized_codes](../../../betalens_db_manager/importers.py#L226) | normalized_codes(self) -&gt; tuple[str, ...] | tuple[str, ...] | 无 docstring，需阅读函数体 |
| [DeleteRequest.validate](../../../betalens_db_manager/importers.py#L234) | validate(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter](../../../betalens_db_manager/importers.py#L240) | class DatabaseWriter() | 类定义；构造/属性见方法与字段 | Transactional set-based writer for standard six-column frames. |
| [DatabaseWriter.__init__](../../../betalens_db_manager/importers.py#L243) | __init__(self, client: DatabaseClient &#124; None=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [DatabaseWriter.dry_run](../../../betalens_db_manager/importers.py#L246) | dry_run(self, table: str, df: pd.DataFrame, conflict_sample_limit: int=20) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseWriter.write](../../../betalens_db_manager/importers.py#L280) | write(self, table: str, df: pd.DataFrame, mode: str=INSERT_ONLY, batch_size: int=5000) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseWriter.write_batches](../../../betalens_db_manager/importers.py#L295) | write_batches(self, table: str, batches: Iterable[pd.DataFrame &#124; ImportBatch], mode: str=INSERT_ONLY, batch_size: int=50000, progress: Callable[[dict[str, Any]], None] &#124; None=None, conflict_sample_limit: int=20) -&gt; dict[str, Any] | dict[str, Any] | Write bounded batches in one file-level transaction. |
| [DatabaseWriter.delete](../../../betalens_db_manager/importers.py#L428) | delete(self, request: DeleteRequest) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseWriter._prepare_frame](../../../betalens_db_manager/importers.py#L440) | _prepare_frame(self, df: pd.DataFrame, *, table: str &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [DatabaseWriter._stable_json](../../../betalens_db_manager/importers.py#L487) | _stable_json(value: Any) -&gt; str | str | 无 docstring，需阅读函数体 |
| [DatabaseWriter._copy_stage](../../../betalens_db_manager/importers.py#L492) | _copy_stage(self, cur, frame: pd.DataFrame, *, batch_size: int=50000, create: bool=True) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._validate_staged_conflicts](../../../betalens_db_manager/importers.py#L542) | _validate_staged_conflicts(cur, sample_limit: int=5) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._deduplicate_stage](../../../betalens_db_manager/importers.py#L578) | _deduplicate_stage(cur) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._drop_stage_tables](../../../betalens_db_manager/importers.py#L591) | _drop_stage_tables(cur) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._prepare_dimensions](../../../betalens_db_manager/importers.py#L605) | _prepare_dimensions(self, cur, table: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._validate_trade_calendar_stage](../../../betalens_db_manager/importers.py#L678) | _validate_trade_calendar_stage(cur) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._prepare_industry_dimensions](../../../betalens_db_manager/importers.py#L693) | _prepare_industry_dimensions(self, cur) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._prepare_index_dimensions](../../../betalens_db_manager/importers.py#L729) | _prepare_index_dimensions(self, cur) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._validate_trade_status_stage](../../../betalens_db_manager/importers.py#L795) | _validate_trade_status_stage(self, cur) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._prepare_metric_dimensions](../../../betalens_db_manager/importers.py#L806) | _prepare_metric_dimensions(self, cur, table: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._inspect_stage](../../../betalens_db_manager/importers.py#L870) | _inspect_stage(self, cur, table: str, sample_limit: int) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseWriter._inspect_market_observation](../../../betalens_db_manager/importers.py#L914) | _inspect_market_observation(self, cur) -&gt; str | str | 无 docstring，需阅读函数体 |
| [DatabaseWriter._core_value_sql](../../../betalens_db_manager/importers.py#L943) | _core_value_sql(fact_alias: str, resolved_alias: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [DatabaseWriter._inspect_industry](../../../betalens_db_manager/importers.py#L952) | _inspect_industry(self, cur) -&gt; str | str | 无 docstring，需阅读函数体 |
| [DatabaseWriter._inspect_index](../../../betalens_db_manager/importers.py#L982) | _inspect_index(self, cur) -&gt; str | str | 无 docstring，需阅读函数体 |
| [DatabaseWriter._inspect_trade_status](../../../betalens_db_manager/importers.py#L1025) | _inspect_trade_status(self, cur) -&gt; str | str | 无 docstring，需阅读函数体 |
| [DatabaseWriter._inspect_trade_calendar](../../../betalens_db_manager/importers.py#L1047) | _inspect_trade_calendar(self, cur) -&gt; str | str | 无 docstring，需阅读函数体 |
| [DatabaseWriter._write_new_schema](../../../betalens_db_manager/importers.py#L1063) | _write_new_schema(self, cur, table: str, mode: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._write_market_observation](../../../betalens_db_manager/importers.py#L1076) | _write_market_observation(self, cur, mode: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._write_industry](../../../betalens_db_manager/importers.py#L1172) | _write_industry(self, cur, mode: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._write_index](../../../betalens_db_manager/importers.py#L1225) | _write_index(self, cur, mode: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._write_trade_status](../../../betalens_db_manager/importers.py#L1291) | _write_trade_status(self, cur, mode: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._write_trade_calendar](../../../betalens_db_manager/importers.py#L1351) | _write_trade_calendar(cur) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._ensure_observation_partitions](../../../betalens_db_manager/importers.py#L1363) | _ensure_observation_partitions(self, cur, *, connection) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._update_coverage](../../../betalens_db_manager/importers.py#L1376) | _update_coverage(self, cur, table: str, frame: pd.DataFrame, inserted: int) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._legacy_dry_run](../../../betalens_db_manager/importers.py#L1398) | _legacy_dry_run(self, cur, table: str, frame: pd.DataFrame, sample_limit: int) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseWriter._temp_stage_exists](../../../betalens_db_manager/importers.py#L1426) | _temp_stage_exists(cur) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [DatabaseWriter._legacy_write](../../../betalens_db_manager/importers.py#L1431) | _legacy_write(self, cur, table: str, frame: pd.DataFrame, mode: str, batch_size: int, *, stage_loaded: bool=False, total_override: int &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseWriter._delete_new_schema](../../../betalens_db_manager/importers.py#L1478) | _delete_new_schema(self, cur, table: str, request: DeleteRequest) -&gt; int | int | 无 docstring，需阅读函数体 |
| [DatabaseWriter._delete_market_observation](../../../betalens_db_manager/importers.py#L1513) | _delete_market_observation(self, cur, table: str, request: DeleteRequest) -&gt; int | int | 无 docstring，需阅读函数体 |
| [DatabaseWriter._delete_industry](../../../betalens_db_manager/importers.py#L1590) | _delete_industry(self, cur, request: DeleteRequest) -&gt; int | int | 无 docstring，需阅读函数体 |
| [DatabaseWriter._refresh_industry_ranges](../../../betalens_db_manager/importers.py#L1611) | _refresh_industry_ranges(cur) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._delete_trade_status](../../../betalens_db_manager/importers.py#L1627) | _delete_trade_status(self, cur, request: DeleteRequest) -&gt; int | int | 无 docstring，需阅读函数体 |
| [DatabaseWriter._delete_conditions](../../../betalens_db_manager/importers.py#L1647) | _delete_conditions(request: DeleteRequest, code_expression: str, datetime_expression: str) -&gt; tuple[list[str], list[Any]] | tuple[list[str], list[Any]] | 无 docstring，需阅读函数体 |
| [DatabaseWriter._mark_coverage_stale](../../../betalens_db_manager/importers.py#L1671) | _mark_coverage_stale(self, cur, table: str, deleted: int) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseWriter._legacy_delete](../../../betalens_db_manager/importers.py#L1685) | _legacy_delete(self, cur, request: DeleteRequest) -&gt; int | int | 无 docstring，需阅读函数体 |

<a id="file-9d1aee965304"></a>
## betalens_db_manager/init_local.bat

[打开源码](../../../betalens_db_manager/init_local.bat) · 29 行 · 说明来源：文件族规则

- **作用**：Windows 启动/初始化包装脚本
- **输入**：当前环境、目录及可用程序
- **输出**：子进程、服务或初始化结果
- **副作用/维护重点**：执行前读脚本：可能启动 GUI、安装依赖或写数据库

<a id="file-c3d0cb04399b"></a>
## betalens_db_manager/job_store.py

[打开源码](../../../betalens_db_manager/job_store.py) · 434 行 · 说明来源：人工文件说明

- **作用**：本地 SQLite 任务持久化
- **输入**：任务/事件/检查点记录
- **输出**：历史查询和恢复数据
- **副作用/维护重点**：创建/更新本地 SQLite；不同于行情 PostgreSQL

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .constants import JOB_LOG_DIR, MANAGER_LOG_ROOT
from .utils import json_default
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
import json
import sqlite3
import uuid
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_now](../../../betalens_db_manager/job_store.py#L19) | _now() -&gt; str | str | 无 docstring，需阅读函数体 |
| [_json](../../../betalens_db_manager/job_store.py#L23) | _json(value: Any) -&gt; str | str | 无 docstring，需阅读函数体 |
| [_loads](../../../betalens_db_manager/job_store.py#L27) | _loads(value: str &#124; None, fallback: Any) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [JobStore](../../../betalens_db_manager/job_store.py#L36) | class JobStore() | 类定义；构造/属性见方法与字段 | Persist schema/import jobs without adding management tables to PostgreSQL. |
| [JobStore.__init__](../../../betalens_db_manager/job_store.py#L39) | __init__(self, path: str &#124; Path=DEFAULT_JOB_DATABASE, job_log_dir: str &#124; Path=JOB_LOG_DIR) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [JobStore._connect](../../../betalens_db_manager/job_store.py#L46) | _connect(self) -&gt; sqlite3.Connection | sqlite3.Connection | 无 docstring，需阅读函数体 |
| [JobStore._initialize](../../../betalens_db_manager/job_store.py#L53) | _initialize(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [JobStore.job_log_path](../../../betalens_db_manager/job_store.py#L119) | job_log_path(self, job_id: str) -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [JobStore.create_job](../../../betalens_db_manager/job_store.py#L122) | create_job(self, kind: str, *, job_id: str &#124; None=None, status: str='planned', target_database: str &#124; None=None, schema_version: int &#124; None=None, log_path: str &#124; Path &#124; None=None, report_path: str &#124; Path &#124; None=None, metadata: Mapping[str, Any] &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [JobStore.update_job](../../../betalens_db_manager/job_store.py#L159) | update_job(self, job_id: str, **changes: Any) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [JobStore.start_job](../../../betalens_db_manager/job_store.py#L194) | start_job(self, job_id: str) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [JobStore.finish_job](../../../betalens_db_manager/job_store.py#L197) | finish_job(self, job_id: str, status: str, *, result: Mapping[str, Any] &#124; None=None, error: str &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [JobStore.request_cancel](../../../betalens_db_manager/job_store.py#L215) | request_cancel(self, job_id: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [JobStore.is_cancel_requested](../../../betalens_db_manager/job_store.py#L218) | is_cancel_requested(self, job_id: str) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [JobStore.get_job](../../../betalens_db_manager/job_store.py#L222) | get_job(self, job_id: str) -&gt; dict[str, Any] &#124; None | dict[str, Any] &#124; None | 无 docstring，需阅读函数体 |
| [JobStore.list_jobs](../../../betalens_db_manager/job_store.py#L227) | list_jobs(self, *, limit: int=100, kind: str &#124; None=None, status: str &#124; None=None) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [JobStore._job_row](../../../betalens_db_manager/job_store.py#L252) | _job_row(row: sqlite3.Row) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [JobStore.create_item](../../../betalens_db_manager/job_store.py#L260) | create_item(self, job_id: str, *, item_id: str &#124; None=None, item_key: str &#124; None=None, status: str='pending', **fields: Any) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [JobStore.update_item](../../../betalens_db_manager/job_store.py#L291) | update_item(self, item_id: str, **changes: Any) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [JobStore.get_item](../../../betalens_db_manager/job_store.py#L317) | get_item(self, item_id: str) -&gt; dict[str, Any] &#124; None | dict[str, Any] &#124; None | 无 docstring，需阅读函数体 |
| [JobStore.list_items](../../../betalens_db_manager/job_store.py#L322) | list_items(self, job_id: str) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [JobStore._item_row](../../../betalens_db_manager/job_store.py#L331) | _item_row(row: sqlite3.Row) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [JobStore.save](../../../betalens_db_manager/job_store.py#L336) | save(self, token: str, payload: Mapping[str, Any]) -&gt; None | None | Save a ManifestRunner-compatible checkpoint payload. |
| [JobStore.load](../../../betalens_db_manager/job_store.py#L368) | load(self, token: str) -&gt; dict[str, Any] &#124; None | dict[str, Any] &#124; None | Load a ManifestRunner-compatible checkpoint payload. |
| [JobStore.append_record](../../../betalens_db_manager/job_store.py#L378) | append_record(self, record: Mapping[str, Any]) -&gt; None | None | Ingest a legacy ImportRecordStore record into the run table. |
| [JobStore.read_legacy_records](../../../betalens_db_manager/job_store.py#L418) | read_legacy_records(self) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |

<a id="file-eb1cf083fb8f"></a>
## betalens_db_manager/jobs.py

[打开源码](../../../betalens_db_manager/jobs.py) · 339 行 · 说明来源：人工文件说明

- **作用**：单文件导入任务编排
- **输入**：路径、目标、adapter、preview token、取消回调
- **输出**：任务记录和进度
- **副作用/维护重点**：SHA256/预览一致性、写库、日志和拒绝行文件；文件级事务

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .constants import DB_COLUMNS, INSERT_ONLY
from .db import DatabaseClient
from .import_adapters import infer_adapter, load_import_batches
from .importers import DatabaseWriter
from .records import ImportRecordStore
from .utils import file_sha256
from .validators import validate_import_frame
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import logging
import uuid
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [ImportCancelled](../../../betalens_db_manager/jobs.py#L22) | class ImportCancelled(RuntimeError) | 类定义；构造/属性见方法与字段 | Signal that the current file transaction must be rolled back. |
| [ImportJobRunner](../../../betalens_db_manager/jobs.py#L26) | class ImportJobRunner() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [ImportJobRunner.__init__](../../../betalens_db_manager/jobs.py#L27) | __init__(self, client: DatabaseClient &#124; None=None, store: ImportRecordStore &#124; None=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [ImportJobRunner.preview](../../../betalens_db_manager/jobs.py#L31) | preview(self, path: str &#124; Path, import_type: str &#124; None=None, options: dict[str, Any] &#124; None=None, *, table: str &#124; None=None, mode: str=INSERT_ONLY, conflict_sample_limit: int=20, inspect_database: bool=True) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [ImportJobRunner.run](../../../betalens_db_manager/jobs.py#L91) | run(self, path: str &#124; Path, table: str, import_type: str &#124; None=None, mode: str=INSERT_ONLY, options: dict[str, Any] &#124; None=None, allow_unsafe_metrics: bool=False, allow_nan_values: bool=False, progress: Callable[[str], None] &#124; None=None, on_rejected: str='fail', expected_sha256: str &#124; None=None, preview_token: str &#124; None=None, cancel_check: Callable[[], bool] &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [ImportJobRunner.run.valid_frames](../../../betalens_db_manager/jobs.py#L154) | valid_frames() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [ImportJobRunner.run.log_write_progress](../../../betalens_db_manager/jobs.py#L192) | log_write_progress(event: dict[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ImportJobRunner._default_target](../../../betalens_db_manager/jobs.py#L257) | _default_target(import_type: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [ImportJobRunner.preview_token](../../../betalens_db_manager/jobs.py#L266) | preview_token(source_sha256: str, table: str, import_type: str, mode: str, options: dict[str, Any]) -&gt; str | str | 无 docstring，需阅读函数体 |
| [ImportJobRunner._accumulate_batch_stats](../../../betalens_db_manager/jobs.py#L289) | _accumulate_batch_stats(stats: dict[str, Any], batch) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ImportJobRunner._finalize_stats](../../../betalens_db_manager/jobs.py#L303) | _finalize_stats(stats: dict[str, Any]) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [ImportJobRunner._build_logger](../../../betalens_db_manager/jobs.py#L314) | _build_logger(self, job_id: str, log_path: Path, progress: Callable[[str], None] &#124; None) -&gt; logging.Logger | logging.Logger | 无 docstring，需阅读函数体 |
| [_CallbackHandler](../../../betalens_db_manager/jobs.py#L329) | class _CallbackHandler(logging.Handler) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [_CallbackHandler.__init__](../../../betalens_db_manager/jobs.py#L330) | __init__(self, callback: Callable[[str], None], formatter: logging.Formatter) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [_CallbackHandler.emit](../../../betalens_db_manager/jobs.py#L335) | emit(self, record: logging.LogRecord) -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-1c720f6d6eec"></a>
## betalens_db_manager/manager.py

[打开源码](../../../betalens_db_manager/manager.py) · 628 行 · 说明来源：人工文件说明

- **作用**：无 Qt 的数据库服务门面
- **输入**：配置/profile、查询、迁移/导入请求
- **输出**：统一的状态、计划、查询和运行报告
- **副作用/维护重点**：构造服务与本地 JobStore；按调用读写 DB/文件

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .constants import MANAGER_LOG_ROOT
from .db import DatabaseClient, QueryRequest
from .import_manifest import ManifestRunner
from .importers import DatabaseWriter
from .job_store import JobStore
from .jobs import ImportJobRunner
from .profiles import ConnectionProfile, ConnectionResolver, ProfileStore, ResolvedConnection
from .records import ImportRecordStore
from .registry import get_dataset
from .schema import SchemaManager
from .utils import json_default
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import json
import logging
import uuid
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_serialize](../../../betalens_db_manager/manager.py#L33) | _serialize(value: Any) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [_write_json](../../../betalens_db_manager/manager.py#L47) | _write_json(path: Path, payload: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseManager](../../../betalens_db_manager/manager.py#L57) | class DatabaseManager() | 类定义；构造/属性见方法与字段 | Own the complete local database lifecycle and import workflow. |
| [DatabaseManager.__init__](../../../betalens_db_manager/manager.py#L60) | __init__(self, db_config: Mapping[str, Any] &#124; None=None, *, profile: str &#124; ConnectionProfile &#124; None=None, profile_store: ProfileStore &#124; None=None, resolver: ConnectionResolver &#124; None=None, job_store: JobStore &#124; None=None, import_statement_timeout_ms: int=MANIFEST_TIMEOUT_MS) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [DatabaseManager._rebuild_services](../../../betalens_db_manager/manager.py#L78) | _rebuild_services(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseManager.effective_config](../../../betalens_db_manager/manager.py#L91) | effective_config(self) -&gt; dict[str, str] | dict[str, str] | Return the runtime config, including the session-only password. |
| [DatabaseManager.connection_info](../../../betalens_db_manager/manager.py#L97) | connection_info(self) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseManager.profile_name](../../../betalens_db_manager/manager.py#L101) | profile_name(self) -&gt; str &#124; None | str &#124; None | 无 docstring，需阅读函数体 |
| [DatabaseManager.reconfigure](../../../betalens_db_manager/manager.py#L104) | reconfigure(self, overrides: Mapping[str, Any] &#124; None=None, *, profile: str &#124; ConnectionProfile &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseManager.save_profile](../../../betalens_db_manager/manager.py#L118) | save_profile(self, profile: ConnectionProfile, *, make_active: bool=True) -&gt; ConnectionProfile | ConnectionProfile | 无 docstring，需阅读函数体 |
| [DatabaseManager.delete_profile](../../../betalens_db_manager/manager.py#L121) | delete_profile(self, name: str) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [DatabaseManager.list_profiles](../../../betalens_db_manager/manager.py#L124) | list_profiles(self) -&gt; list[ConnectionProfile] | list[ConnectionProfile] | 无 docstring，需阅读函数体 |
| [DatabaseManager.probe_connection](../../../betalens_db_manager/manager.py#L127) | probe_connection(self) -&gt; dict[str, Any] | dict[str, Any] | Probe PostgreSQL without raising for an offline GUI startup. |
| [DatabaseManager.database_exists](../../../betalens_db_manager/manager.py#L148) | database_exists(self, dbname: str &#124; None=None) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [DatabaseManager.create_database](../../../betalens_db_manager/manager.py#L151) | create_database(self, dbname: str &#124; None=None) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [DatabaseManager.connect](../../../betalens_db_manager/manager.py#L154) | connect(self, dbname: str &#124; None=None) | 无返回注解；return: self.schema.connect(dbname) | 无 docstring，需阅读函数体 |
| [DatabaseManager.plan_schema](../../../betalens_db_manager/manager.py#L157) | plan_schema(self, *, target_version: int &#124; None=None, observation_years: Iterable[int] &#124; None=None, create_compat_views: bool=True) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseManager.plan](../../../betalens_db_manager/manager.py#L170) | plan(self, *, target_version: int &#124; None=None, observation_years: Iterable[int] &#124; None=None, manifest: str &#124; Path &#124; Mapping[str, Any] &#124; None=None, create_compat_views: bool=True) -&gt; dict[str, Any] | dict[str, Any] | Return a read-only combined schema/manifest plan. |
| [DatabaseManager._report_path](../../../betalens_db_manager/manager.py#L195) | _report_path(self, requested: str &#124; Path &#124; None, prefix: str) -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [DatabaseManager._begin_job](../../../betalens_db_manager/manager.py#L202) | _begin_job(self, kind: str, report_path: Path, *, metadata: Mapping[str, Any] &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseManager._schema_report_path](../../../betalens_db_manager/manager.py#L220) | _schema_report_path(report_path: Path) -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [DatabaseManager.bootstrap](../../../betalens_db_manager/manager.py#L223) | bootstrap(self, *, create_database_if_missing: bool=True, migrate_legacy: bool=True, create_compat_views: bool=True, verify: bool=True, observation_years: Iterable[int] &#124; None=None, report_path: str &#124; Path &#124; None=None, manifest: str &#124; Path &#124; Mapping[str, Any] &#124; None=None, resume: bool=True, progress: ProgressCallback &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | Bootstrap schema, optionally import a manifest, then verify again. |
| [DatabaseManager._failure_status](../../../betalens_db_manager/manager.py#L336) | _failure_status(report: Mapping[str, Any]) -&gt; str | str | 无 docstring，需阅读函数体 |
| [DatabaseManager.bootstrap_local](../../../betalens_db_manager/manager.py#L344) | bootstrap_local(self, **kwargs: Any) -&gt; dict[str, Any] | dict[str, Any] | Compatibility name for callers migrating from ''SchemaManager''. |
| [DatabaseManager.ensure_dataset](../../../betalens_db_manager/manager.py#L349) | ensure_dataset(self, table: str, *, report_path: str &#124; Path &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | Install the contract dependencies required by one logical dataset. PostgreSQL fact tables depend on shared dimensions, constraints and compatibility views. The safe operation is therefore an idempotent contract bootstrap, not a brittle bare ''CREATE TABLE'' for one name. |
| [DatabaseManager._manifest_runner](../../../betalens_db_manager/manager.py#L369) | _manifest_runner(self) | 无返回注解；return: ManifestRunner(job_runner=self.import_runner, checkpoint_store=self.job_store, target_database=str(self.db_config.get('dbname', ''))) | 无 docstring，需阅读函数体 |
| [DatabaseManager.plan_manifest](../../../betalens_db_manager/manager.py#L380) | plan_manifest(self, path: str &#124; Path) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseManager._execute_manifest](../../../betalens_db_manager/manager.py#L387) | _execute_manifest(self, plan: Any, *, resume: bool=True, progress: ProgressCallback &#124; None=None, cancel: Callable[[], bool] &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseManager._execute_manifest.callback](../../../betalens_db_manager/manager.py#L397) | callback(event: Any) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseManager.run_manifest](../../../betalens_db_manager/manager.py#L408) | run_manifest(self, path_or_plan: str &#124; Path &#124; Mapping[str, Any], *, resume: bool=True, progress: ProgressCallback &#124; None=None, report_path: str &#124; Path &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseManager._persist_manifest_items](../../../betalens_db_manager/manager.py#L456) | _persist_manifest_items(self, run_id: str, result: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [DatabaseManager.import_manifest](../../../betalens_db_manager/manager.py#L491) | import_manifest(self, *args: Any, **kwargs: Any) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseManager.preflight_import](../../../betalens_db_manager/manager.py#L494) | preflight_import(self, path: str &#124; Path, *, table: str, adapter: str &#124; None=None, import_type: str &#124; None=None, options: Mapping[str, Any] &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseManager.run_import](../../../betalens_db_manager/manager.py#L512) | run_import(self, path: str &#124; Path, *, table: str, adapter: str &#124; None=None, import_type: str &#124; None=None, mode: str='insert_only', options: Mapping[str, Any] &#124; None=None, progress: Callable[[str], None] &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseManager.verify](../../../betalens_db_manager/manager.py#L532) | verify(self, *, deep: bool=False, require_compat_views: bool=True, report_path: str &#124; Path &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseManager.verify_schema](../../../betalens_db_manager/manager.py#L581) | verify_schema(self, **kwargs: Any) -&gt; dict[str, Any] | dict[str, Any] | Direct compatibility delegate without creating a local verify job. |
| [DatabaseManager.dashboard_snapshot](../../../betalens_db_manager/manager.py#L586) | dashboard_snapshot(self) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseManager.table_schema](../../../betalens_db_manager/manager.py#L605) | table_schema(self, table: str) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [DatabaseManager.query](../../../betalens_db_manager/manager.py#L608) | query(self, request: QueryRequest) | 无返回注解；return: self.client.query_table(request) | 无 docstring，需阅读函数体 |
| [DatabaseManager.execute_readonly_sql](../../../betalens_db_manager/manager.py#L611) | execute_readonly_sql(self, query: str, *, limit: int=5000) | 无返回注解；return: self.client.execute_readonly_sql(query, limit=limit) | 无 docstring，需阅读函数体 |
| [DatabaseManager.diagnose_data](../../../betalens_db_manager/manager.py#L614) | diagnose_data(self, table: str, *, sample_limit: int=10) | 无返回注解；return: self.client.diagnose_data(table, sample_limit=sample_limit) | 无 docstring，需阅读函数体 |
| [DatabaseManager.list_jobs](../../../betalens_db_manager/manager.py#L617) | list_jobs(self, *, limit: int=100, kind: str &#124; None=None, status: str &#124; None=None) | 无返回注解；return: self.job_store.list_jobs(limit=limit, kind=kind, status=status) | 无 docstring，需阅读函数体 |
| [DatabaseManager.read_log](../../../betalens_db_manager/manager.py#L621) | read_log(path: str &#124; Path) -&gt; str | str | 无 docstring，需阅读函数体 |
| [DatabaseManager.cancel](../../../betalens_db_manager/manager.py#L624) | cancel(self, job_id: str) -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-d37e0ef025d5"></a>
## betalens_db_manager/profiles.py

[打开源码](../../../betalens_db_manager/profiles.py) · 246 行 · 说明来源：人工文件说明

- **作用**：连接档案选择与来源解释
- **输入**：用户 profile、覆盖配置、会话密码
- **输出**：ResolvedConnection 和显示配置
- **副作用/维护重点**：档案写盘不保存密码；effective 配置仍可能含密码

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.datafeed.config import DEFAULT_CONFIG, get_config
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json
import os
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [default_profile_path](../../../betalens_db_manager/profiles.py#L25) | default_profile_path() -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [ConnectionProfile](../../../betalens_db_manager/profiles.py#L31) | class ConnectionProfile() | 类定义；构造/属性见方法与字段 | A named, persistable connection profile. Passwords deliberately are not part of this type. They can only enter an effective connection through environment variables or runtime overrides.；字段：name: str; host: str = 'localhost'; port: str = '5432'; dbname: str = 'datafeed'; user: str = 'postgres' |
| [ConnectionProfile.from_mapping](../../../betalens_db_manager/profiles.py#L45) | from_mapping(cls, name: str, values: Mapping[str, Any]) -&gt; 'ConnectionProfile' | 'ConnectionProfile' | 无 docstring，需阅读函数体 |
| [ConnectionProfile.validate](../../../betalens_db_manager/profiles.py#L58) | validate(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ConnectionProfile.as_dict](../../../betalens_db_manager/profiles.py#L71) | as_dict(self) -&gt; dict[str, str] | dict[str, str] | 无 docstring，需阅读函数体 |
| [ProfileStore](../../../betalens_db_manager/profiles.py#L75) | class ProfileStore() | 类定义；构造/属性见方法与字段 | JSON persistence for non-secret connection settings. |
| [ProfileStore.__init__](../../../betalens_db_manager/profiles.py#L78) | __init__(self, path: str &#124; Path &#124; None=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [ProfileStore._read_document](../../../betalens_db_manager/profiles.py#L81) | _read_document(self) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [ProfileStore.list](../../../betalens_db_manager/profiles.py#L92) | list(self) -&gt; list[ConnectionProfile] | list[ConnectionProfile] | 无 docstring，需阅读函数体 |
| [ProfileStore.active_name](../../../betalens_db_manager/profiles.py#L101) | active_name(self) -&gt; str &#124; None | str &#124; None | 无 docstring，需阅读函数体 |
| [ProfileStore.get](../../../betalens_db_manager/profiles.py#L105) | get(self, name: str &#124; None=None) -&gt; ConnectionProfile &#124; None | ConnectionProfile &#124; None | 无 docstring，需阅读函数体 |
| [ProfileStore.save](../../../betalens_db_manager/profiles.py#L111) | save(self, profile: ConnectionProfile, *, make_active: bool=True) -&gt; ConnectionProfile | ConnectionProfile | 无 docstring，需阅读函数体 |
| [ProfileStore.delete](../../../betalens_db_manager/profiles.py#L127) | delete(self, name: str) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [ProfileStore._write_document](../../../betalens_db_manager/profiles.py#L142) | _write_document(self, payload: Mapping[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ResolvedConnection](../../../betalens_db_manager/profiles.py#L165) | class ResolvedConnection() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：config: dict[str, str]; sources: dict[str, str] = field(default_factory=dict); profile_name: str &#124; None = None |
| [ResolvedConnection.display_config](../../../betalens_db_manager/profiles.py#L170) | display_config(self) -&gt; dict[str, str] | dict[str, str] | 无 docstring，需阅读函数体 |
| [ResolvedConnection.as_dict](../../../betalens_db_manager/profiles.py#L176) | as_dict(self) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [ConnectionResolver](../../../betalens_db_manager/profiles.py#L184) | class ConnectionResolver() | 类定义；构造/属性见方法与字段 | Resolve defaults, local config, named profile, environment and CLI values. |
| [ConnectionResolver.__init__](../../../betalens_db_manager/profiles.py#L187) | __init__(self, profile_store: ProfileStore &#124; None=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [ConnectionResolver.resolve](../../../betalens_db_manager/profiles.py#L190) | resolve(self, overrides: Mapping[str, Any] &#124; None=None, *, profile: str &#124; ConnectionProfile &#124; None=None) -&gt; ResolvedConnection | ResolvedConnection | 无 docstring，需阅读函数体 |

<a id="file-ed9869060643"></a>
## betalens_db_manager/records.py

[打开源码](../../../betalens_db_manager/records.py) · 67 行 · 说明来源：人工文件说明

- **作用**：导入记录兼容门面
- **输入**：旧记录接口调用、JobStore
- **输出**：记录和日志路径
- **副作用/维护重点**：包装新存储，不要再建一套任务数据库

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .constants import IMPORT_RECORDS_FILE, JOB_LOG_DIR, MANAGER_LOG_ROOT
from .job_store import JobStore
from __future__ import annotations
from pathlib import Path
from typing import Any
import json
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [ImportRecordStore](../../../betalens_db_manager/records.py#L12) | class ImportRecordStore() | 类定义；构造/属性见方法与字段 | Retain the old append/read API on top of the shared SQLite JobStore. |
| [ImportRecordStore.__init__](../../../betalens_db_manager/records.py#L15) | __init__(self, records_file: Path=IMPORT_RECORDS_FILE, job_log_dir: Path=JOB_LOG_DIR, *, job_store: JobStore &#124; None=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [ImportRecordStore.job_log_path](../../../betalens_db_manager/records.py#L36) | job_log_path(self, job_id: str) -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [ImportRecordStore.append](../../../betalens_db_manager/records.py#L39) | append(self, record: dict[str, Any]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ImportRecordStore.read_all](../../../betalens_db_manager/records.py#L42) | read_all(self) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [ImportRecordStore._migrate_json_lines](../../../betalens_db_manager/records.py#L45) | _migrate_json_lines(self) -&gt; None | None | Import pre-redesign JSONL records once; upserts make this idempotent. |

<a id="file-5811a0034afc"></a>
## betalens_db_manager/registry.py

[打开源码](../../../betalens_db_manager/registry.py) · 132 行 · 说明来源：人工文件说明

- **作用**：管理端逻辑数据集、指标与可写性
- **输入**：表名/指标别名/写入要求
- **输出**：DatasetSpec/CoreMetric 或异常
- **副作用/维护重点**：与读层 registry 同步；不要用任意表名绕过注册

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [CoreMetric](../../../betalens_db_manager/registry.py#L16) | class CoreMetric() | 类定义；构造/属性见方法与字段 | A legacy metric represented by a column in ''market_daily_fact''.；字段：column: str; available_time: str |
| [DatasetSpec](../../../betalens_db_manager/registry.py#L24) | class DatasetSpec() | 类定义；构造/属性见方法与字段 | Storage and compatibility metadata for one logical dataset.；字段：name: str; storage: str; physical_tables: tuple[str, ...]; entity_type: str &#124; None = None; writable: bool = True |
| [get_dataset](../../../betalens_db_manager/registry.py#L106) | get_dataset(name: str, *, writable: bool=False) -&gt; DatasetSpec | DatasetSpec | Return a validated dataset specification. |
| [canonical_metric](../../../betalens_db_manager/registry.py#L118) | canonical_metric(metric: str, logical_dataset: str='daily_market') -&gt; str | str | Resolve the small static alias set used before a database is available. |
| [core_metric](../../../betalens_db_manager/registry.py#L131) | core_metric(metric: str) -&gt; CoreMetric &#124; None | CoreMetric &#124; None | 无 docstring，需阅读函数体 |

<a id="file-f04d8cd69b37"></a>
## betalens_db_manager/run.bat

[打开源码](../../../betalens_db_manager/run.bat) · 32 行 · 说明来源：文件族规则

- **作用**：Windows 启动/初始化包装脚本
- **输入**：当前环境、目录及可用程序
- **输出**：子进程、服务或初始化结果
- **副作用/维护重点**：执行前读脚本：可能启动 GUI、安装依赖或写数据库

<a id="file-8fed929085b1"></a>
## betalens_db_manager/schema.py

[打开源码](../../../betalens_db_manager/schema.py) · 1246 行 · 说明来源：人工文件说明

- **作用**：版本化建库迁移和核验
- **输入**：配置、目标版本、migration 资源
- **输出**：plan/init/verify 报告
- **副作用/维护重点**：建库、DDL、校验和检查；不能直接改已应用 migration

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .constants import ALLOWED_TABLES, MANAGER_LOG_ROOT
from .contracts import BASE_TABLES, COMPATIBILITY_VIEWS, COMPATIBILITY_VIEW_VERSION, CONSTRAINT_DEFINITION_FRAGMENTS, CORE_METRIC_SEEDS, DEFAULT_DEFINITION_FRAGMENTS, FINALIZE_VERSION, IDENTITY_COLUMNS, INDEX_DEFINITION_FRAGMENTS, LATEST_SCHEMA_VERSION, LEGACY_MIGRATION_VERSION, NOT_NULL_COLUMNS, REQUIRED_ALIASES, REQUIRED_CONSTRAINTS, REQUIRED_CONSTRAINT_TYPES, REQUIRED_INDEXES, TABLE_COLUMNS, VIEW_COLUMNS, VIEW_DEFINITION_FRAGMENTS, VIEW_DEFINITION_HASHES, get_schema_contract
from .utils import clean_database_config, json_default
from __future__ import annotations
from betalens.datafeed.config import get_database_config
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from importlib import resources
from pathlib import Path
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from typing import Any, Iterable
import hashlib
import json
import logging
import os
import psycopg2
import psycopg2.extras
import re
import time
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [Migration](../../../betalens_db_manager/schema.py#L74) | class Migration() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：version: int; name: str; resource_name: str; checksum: str; accepted_checksums: frozenset[str]; sql_text: str |
| [Migration.as_dict](../../../betalens_db_manager/schema.py#L82) | as_dict(self) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [MigrationChecksumError](../../../betalens_db_manager/schema.py#L91) | class MigrationChecksumError(RuntimeError) | 类定义；构造/属性见方法与字段 | Raised when an already applied migration was edited in place. |
| [SchemaDowngradeError](../../../betalens_db_manager/schema.py#L95) | class SchemaDowngradeError(RuntimeError) | 类定义；构造/属性见方法与字段 | Raised when a caller requests a target older than the installed schema. |
| [validate_database_name](../../../betalens_db_manager/schema.py#L99) | validate_database_name(dbname: str) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [canonicalize_migration_bytes](../../../betalens_db_manager/schema.py#L103) | canonicalize_migration_bytes(raw: bytes) -&gt; bytes | bytes | Return the platform-independent UTF-8/LF migration representation. |
| [migration_checksum_variants](../../../betalens_db_manager/schema.py#L109) | migration_checksum_variants(raw: bytes) -&gt; tuple[str, frozenset[str]] | tuple[str, frozenset[str]] | Return the canonical checksum plus accepted historical LF/CRLF hashes. |
| [_read_migration_bytes](../../../betalens_db_manager/schema.py#L119) | _read_migration_bytes(resource_name: str) -&gt; bytes | bytes | 无 docstring，需阅读函数体 |
| [load_migrations](../../../betalens_db_manager/schema.py#L127) | load_migrations() -&gt; tuple[Migration, ...] | tuple[Migration, ...] | 无 docstring，需阅读函数体 |
| [SchemaManager](../../../betalens_db_manager/schema.py#L169) | class SchemaManager() | 类定义；构造/属性见方法与字段 | Create, migrate, and strictly verify the local PostgreSQL schema. |
| [SchemaManager.__init__](../../../betalens_db_manager/schema.py#L172) | __init__(self, db_config: dict[str, Any] &#124; None=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [SchemaManager.validate_table](../../../betalens_db_manager/schema.py#L190) | validate_table(self, table_name: str) -&gt; str | str | 无 docstring，需阅读函数体 |
| [SchemaManager.connect](../../../betalens_db_manager/schema.py#L195) | connect(self, dbname: str &#124; None=None) | 无返回注解；return: psycopg2.connect(**cfg) | 无 docstring，需阅读函数体 |
| [SchemaManager.database_exists](../../../betalens_db_manager/schema.py#L201) | database_exists(self, dbname: str &#124; None=None) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [SchemaManager.create_database](../../../betalens_db_manager/schema.py#L212) | create_database(self, dbname: str &#124; None=None) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [SchemaManager._read_applied_migrations](../../../betalens_db_manager/schema.py#L237) | _read_applied_migrations(self, conn) -&gt; dict[int, dict[str, Any]] | dict[int, dict[str, Any]] | 无 docstring，需阅读函数体 |
| [SchemaManager._validate_applied_checksums](../../../betalens_db_manager/schema.py#L251) | _validate_applied_checksums(self, migrations: Iterable[Migration], applied: dict[int, dict[str, Any]]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [SchemaManager._reject_downgrade](../../../betalens_db_manager/schema.py#L269) | _reject_downgrade(applied: dict[int, dict[str, Any]], target_version: int) -&gt; None | None | 无 docstring，需阅读函数体 |
| [SchemaManager.plan_migration](../../../betalens_db_manager/schema.py#L276) | plan_migration(self, target_version: int &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [SchemaManager._apply_migrations](../../../betalens_db_manager/schema.py#L304) | _apply_migrations(self, conn, target_version: int) -&gt; tuple[list[dict[str, Any]], list[str], set[int]] | tuple[list[dict[str, Any]], list[str], set[int]] | 无 docstring，需阅读函数体 |
| [SchemaManager._legacy_observation_years](../../../betalens_db_manager/schema.py#L355) | _legacy_observation_years(self, conn) -&gt; set[int] | set[int] | 无 docstring，需阅读函数体 |
| [SchemaManager._normalize_partition_years](../../../betalens_db_manager/schema.py#L387) | _normalize_partition_years(years: Iterable[int] &#124; None=None) -&gt; set[int] | set[int] | 无 docstring，需阅读函数体 |
| [SchemaManager._partition_contract_errors](../../../betalens_db_manager/schema.py#L398) | _partition_contract_errors(cls, partition_rows: Iterable[dict[str, Any]], expected_years: Iterable[int] &#124; None=None) -&gt; tuple[list[str], set[int]] | tuple[list[str], set[int]] | 无 docstring，需阅读函数体 |
| [SchemaManager.ensure_observation_partitions](../../../betalens_db_manager/schema.py#L437) | ensure_observation_partitions(self, years: Iterable[int] &#124; None=None, *, connection=None) -&gt; list[str] | list[str] | Create observation partitions for explicit years. The current and next calendar year are always included. Importers call this method before loading historical years; they do not construct DDL. |
| [SchemaManager.write_failure_report](../../../betalens_db_manager/schema.py#L492) | write_failure_report(self, error: Exception &#124; str, *, stage: str, report_path: str &#124; Path &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | Persist a diagnostic report for failures before bootstrap starts. |
| [SchemaManager._bootstrap](../../../betalens_db_manager/schema.py#L523) | _bootstrap(self, *, create_database_if_missing: bool, migrate_legacy: bool, create_compat_views: bool, verify: bool, observation_years: Iterable[int] &#124; None, report_path: str &#124; Path &#124; None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [SchemaManager.bootstrap_local](../../../betalens_db_manager/schema.py#L691) | bootstrap_local(self: 'SchemaManager &#124; None'=None, *, create_database_if_missing: bool=True, migrate_legacy: bool=True, create_compat_views: bool=True, verify: bool=True, observation_years: Iterable[int] &#124; None=None, report_path: str &#124; Path &#124; None=None, db_config: dict[str, Any] &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | Create or upgrade a reproducible local database in one call. This intentionally supports both ''SchemaManager.bootstrap_local(...)'' and ''SchemaManager(config).bootstrap_local(...)''. ''db_config'' is accepted only for the class-style call. |
| [SchemaManager.ensure_schema](../../../betalens_db_manager/schema.py#L721) | ensure_schema(self, tables: list[str] &#124; None=None, force: bool=False, create_database_if_missing: bool=False, create_indexes: bool=True, create_comments: bool=True) -&gt; dict[str, Any] | dict[str, Any] | Compatibility wrapper around the atomic versioned bootstrap. Individual table creation is no longer supported because it can leave a schema whose views and routing metadata disagree with its fact tables. |
| [SchemaManager.table_exists](../../../betalens_db_manager/schema.py#L744) | table_exists(self, cur, table_name: str) -&gt; bool | bool | 无 docstring，需阅读函数体 |
| [SchemaManager._connection_scope](../../../betalens_db_manager/schema.py#L752) | _connection_scope(self, connection=None) | 无返回注解；return: None | 无 docstring，需阅读函数体 |
| [SchemaManager._verify_schema](../../../betalens_db_manager/schema.py#L762) | _verify_schema(self, *, require_compat_views: bool=True, expected_version: int &#124; None=None, expected_partition_years: Iterable[int] &#124; None=None, connection=None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [SchemaManager.verify_schema](../../../betalens_db_manager/schema.py#L1200) | verify_schema(self, *, require_compat_views: bool=True, expected_version: int &#124; None=None, expected_partition_years: Iterable[int] &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | Verify a committed schema using the contract for ''expected_version''. |
| [SchemaManager.verify_schema_precommit](../../../betalens_db_manager/schema.py#L1215) | verify_schema_precommit(self, connection, *, require_compat_views: bool=True, expected_version: int &#124; None=None, expected_partition_years: Iterable[int] &#124; None=None) -&gt; dict[str, Any] | dict[str, Any] | Verify uncommitted DDL using the migration transaction connection. |

<a id="file-ca10f44601c9"></a>
## betalens_db_manager/utils.py

[打开源码](../../../betalens_db_manager/utils.py) · 93 行 · 说明来源：人工文件说明

- **作用**：路径、哈希、JSON 和表格预览工具
- **输入**：路径/配置/对象/DataFrame
- **输出**：可序列化对象、摘要与路径
- **副作用/维护重点**：哈希读取文件，ensure_parent 创建目录

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
import hashlib
import json
import pandas as pd
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [repo_root](../../../betalens_db_manager/utils.py#L18) | repo_root() -&gt; Path | Path | 无 docstring，需阅读函数体 |
| [ensure_parent](../../../betalens_db_manager/utils.py#L22) | ensure_parent(path: Path) -&gt; None | None | 无 docstring，需阅读函数体 |
| [file_sha256](../../../betalens_db_manager/utils.py#L26) | file_sha256(path: str &#124; Path) -&gt; str | str | 无 docstring，需阅读函数体 |
| [clean_database_config](../../../betalens_db_manager/utils.py#L34) | clean_database_config(config: dict[str, Any]) -&gt; dict[str, Any] | dict[str, Any] | Return only psycopg2 connection keys from a datafeed config section. |
| [json_default](../../../betalens_db_manager/utils.py#L39) | json_default(value: Any) -&gt; Any | Any | 无 docstring，需阅读函数体 |
| [to_json_line](../../../betalens_db_manager/utils.py#L63) | to_json_line(record: dict[str, Any]) -&gt; str | str | 无 docstring，需阅读函数体 |
| [dataframe_preview](../../../betalens_db_manager/utils.py#L67) | dataframe_preview(df: pd.DataFrame, rows: int=100) -&gt; list[dict[str, Any]] | list[dict[str, Any]] | 无 docstring，需阅读函数体 |
| [dataframe_summary](../../../betalens_db_manager/utils.py#L75) | dataframe_summary(df: pd.DataFrame) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |

<a id="file-6e5a97be098f"></a>
## betalens_db_manager/validators.py

[打开源码](../../../betalens_db_manager/validators.py) · 74 行 · 说明来源：人工文件说明

- **作用**：导入前数据校验
- **输入**：规范化 DataFrame 和允许选项
- **输出**：ValidationReport
- **副作用/维护重点**：不应把被拒绝行悄悄视作成功导入

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .constants import REQUIRED_DB_COLUMNS
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [ValidationReport](../../../betalens_db_manager/validators.py#L15) | class ValidationReport() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：ok: bool; errors: list[str] = field(default_factory=list); warnings: list[str] = field(default_factory=list); stats: dict[str, Any] = field(default_factory=dict) |
| [validate_import_frame](../../../betalens_db_manager/validators.py#L22) | validate_import_frame(df: pd.DataFrame, allow_unsafe_metrics: bool=False, allow_nan_values: bool=False) -&gt; ValidationReport | ValidationReport | 无 docstring，需阅读函数体 |

