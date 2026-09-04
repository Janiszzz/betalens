# core_datafeed：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-3f481da29674"></a>
## betalens/datafeed/__init__.py

[打开源码](../../../betalens/datafeed/__init__.py) · 22 行 · 说明来源：文件族规则

- **作用**：Lightweight datafeed package exports used by the factor pipeline.
- **输入**：import 请求
- **输出**：模块导出与符号；见静态 imports
- **副作用/维护重点**：初始化可能导入子模块；__all__ 与真实导出需结合源码阅读

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .core import Datafeed, get_absolute_trade_days, trade_days_offset
from .industry import get_industry_members, query_industry
from .universe import get_index_universe, get_index_universe_date, get_index_universe_panel
from .validation import FillStrategy
```

<a id="file-a22b7673f6ea"></a>
## betalens/datafeed/config.example.json

[打开源码](../../../betalens/datafeed/config.example.json) · 18 行 · 说明来源：人工文件说明

- **作用**：本地连接配置模板
- **输入**：人工填写数据库及日志选项
- **输出**：配置管理器读取的字典
- **副作用/维护重点**：不是已生效凭据；复制前检查已有配置

<a id="file-d7a892129572"></a>
## betalens/datafeed/config.py

[打开源码](../../../betalens/datafeed/config.py) · 283 行 · 说明来源：人工文件说明

- **作用**：分层配置和缓存
- **输入**：运行参数、环境变量、配置文件
- **输出**：数据库/日志配置字典
- **副作用/维护重点**：save 可写配置；不要输出密码

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
import os
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [ConfigManager](../../../betalens/datafeed/config.py#L36) | class ConfigManager() | 类定义；构造/属性见方法与字段 | 配置管理器 |
| [ConfigManager.__init__](../../../betalens/datafeed/config.py#L39) | __init__(self, config_file: Optional[str]=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 初始化配置管理器 Args: config_file: 配置文件路径，默认为当前模块目录下的config.json |
| [ConfigManager.load](../../../betalens/datafeed/config.py#L72) | load(self) -&gt; None | None | 从文件加载配置 如果文件不存在或加载失败，使用默认配置 |
| [ConfigManager.save](../../../betalens/datafeed/config.py#L105) | save(self, config_file: Optional[str]=None) -&gt; None | None | 保存当前配置到文件 Args: config_file: 配置文件路径，默认使用初始化时的路径 |
| [ConfigManager.get](../../../betalens/datafeed/config.py#L131) | get(self, key_path: str, default: Any=None) -&gt; Any | Any | 获取配置值 Args: key_path: 配置键路径，使用点号分隔，如 'database.dbname' default: 默认值，如果键不存在则返回此值 Returns: 配置值 Example: &gt;&gt;&gt; config = ConfigManager() &gt;&gt;&gt; config.get('database.dbname') 'datafeed' &gt;&gt;&gt; config.get('database.port') '5432' |
| [ConfigManager.set](../../../betalens/datafeed/config.py#L160) | set(self, key_path: str, value: Any) -&gt; None | None | 设置配置值 Args: key_path: 配置键路径，使用点号分隔，如 'database.dbname' value: 配置值 Example: &gt;&gt;&gt; config = ConfigManager() &gt;&gt;&gt; config.set('database.dbname', 'my_database') &gt;&gt;&gt; config.get('database.dbname') 'my_database' |
| [ConfigManager.get_section](../../../betalens/datafeed/config.py#L186) | get_section(self, section: str) -&gt; Dict[str, Any] | Dict[str, Any] | 获取配置节 Args: section: 配置节名称，如 'database', 'logging' Returns: 配置节字典 |
| [ConfigManager._merge_config](../../../betalens/datafeed/config.py#L198) | _merge_config(self, base: Dict, override: Dict) -&gt; Dict | Dict | 合并两个配置字典（递归） Args: base: 基础配置 override: 覆盖配置 Returns: 合并后的配置 |
| [ConfigManager.config](../../../betalens/datafeed/config.py#L222) | config(self) -&gt; Dict[str, Any] | Dict[str, Any] | 获取完整配置字典 |
| [ConfigManager.__getitem__](../../../betalens/datafeed/config.py#L226) | __getitem__(self, key: str) -&gt; Any | Any | 支持字典式访问 |
| [ConfigManager.__setitem__](../../../betalens/datafeed/config.py#L230) | __setitem__(self, key: str, value: Any) -&gt; None | None | 支持字典式设置 |
| [get_config](../../../betalens/datafeed/config.py#L239) | get_config(config_file: Optional[str]=None) -&gt; ConfigManager | ConfigManager | 获取全局配置实例 Args: config_file: 配置文件路径，仅在首次调用时有效 Returns: ConfigManager实例 |
| [reset_config](../../../betalens/datafeed/config.py#L257) | reset_config() -&gt; None | None | 重置全局配置实例 |
| [get_database_config](../../../betalens/datafeed/config.py#L264) | get_database_config() -&gt; Dict[str, str] | Dict[str, str] | 获取数据库配置 |
| [get_logging_config](../../../betalens/datafeed/config.py#L280) | get_logging_config() -&gt; Dict[str, str] | Dict[str, str] | 获取日志配置 |

<a id="file-300038c0ad6b"></a>
## betalens/datafeed/core.py

[打开源码](../../../betalens/datafeed/core.py) · 411 行 · 说明来源：人工文件说明

- **作用**：研究取数门面与交易日历
- **输入**：逻辑表、代码、时间、指标
- **输出**：DataFrame、交易日列表或成分集合
- **副作用/维护重点**：Datafeed 构造即获取数据库连接；使用 close/context manager

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .industry import get_industry_members
from .industry import query_industry
from .pool import get_read_pool
from .query import _normalized_schema_available, build_query as _build_query, get_available_dates as _get_available_dates, get_latest_date as _get_latest_date, query_nearest_after as _query_nearest_after, query_nearest_before as _query_nearest_before, query_nearest_in_range_after as _query_nearest_in_range_after, query_nearest_in_range_before as _query_nearest_in_range_before, query_time_range as _query_time_range, query_trade_status as _query_trade_status
from .registry import get_dataset
from .universe import get_index_universe
from .universe import get_index_universe_date
from .universe import get_index_universe_panel
from __future__ import annotations
from datetime import date, datetime
from psycopg2 import sql as psql
from typing import Any
import akshare as ak
import logging
import pandas as pd
import psycopg2.extras
import warnings
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [Datafeed](../../../betalens/datafeed/core.py#L30) | class Datafeed() | 类定义；构造/属性见方法与字段 | Logical, read-only data access facade. ''table_name'' remains the legacy logical dataset name. The query module routes it to the normalized ''betalens'' schema when installed and falls back to a legacy read-only relation during rollout. |
| [Datafeed.__init__](../../../betalens/datafeed/core.py#L38) | __init__(self, table_name: str, db_config: dict[str, Any] &#124; None=None, log_dir: str &#124; None=None) -&gt; None | None | 无 docstring，需阅读函数体 |
| [Datafeed.cursor](../../../betalens/datafeed/core.py#L63) | cursor(self) | 无返回注解；return: self._cursor | Deprecated read-only cursor retained for third-party compatibility. |
| [Datafeed.__enter__](../../../betalens/datafeed/core.py#L72) | __enter__(self) -&gt; 'Datafeed' | 'Datafeed' | 无 docstring，需阅读函数体 |
| [Datafeed.__exit__](../../../betalens/datafeed/core.py#L75) | __exit__(self, exc_type, exc, traceback) -&gt; None | None | 无 docstring，需阅读函数体 |
| [Datafeed.run_query](../../../betalens/datafeed/core.py#L78) | run_query(self, conditions: list[str] &#124; None=None, params: list[Any] &#124; None=None, select_columns: str='*') -&gt; pd.DataFrame | pd.DataFrame | Execute a legacy SELECT on the logical relation. This compatibility method is deprecated. The pooled connection is read-only, so callers cannot use it to mutate the database. |
| [Datafeed.query_time_range](../../../betalens/datafeed/core.py#L103) | query_time_range(self, codes: list[str] &#124; None=None, start_date: str &#124; None=None, end_date: str &#124; None=None, metric: str &#124; None=None, limit: int &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [Datafeed.query_trade_status](../../../betalens/datafeed/core.py#L122) | query_trade_status(self, params: dict[str, Any] &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [Datafeed.query_nearest_after](../../../betalens/datafeed/core.py#L135) | query_nearest_after(self, params: dict[str, Any] &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [Datafeed.query_nearest_before](../../../betalens/datafeed/core.py#L147) | query_nearest_before(self, params: dict[str, Any] &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [Datafeed.query_nearest_in_range_after](../../../betalens/datafeed/core.py#L159) | query_nearest_in_range_after(self, params: dict[str, Any] &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [Datafeed.query_nearest_in_range_before](../../../betalens/datafeed/core.py#L173) | query_nearest_in_range_before(self, params: dict[str, Any] &#124; None=None) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [Datafeed.get_latest_date](../../../betalens/datafeed/core.py#L187) | get_latest_date(self, code: str &#124; None=None, metric: str &#124; None=None) | 无返回注解；return: _get_latest_date(cursor=self._cursor, table_name=self.sheet, code=code, metric=metric, logger=self.logger) | 无 docstring，需阅读函数体 |
| [Datafeed.get_available_dates](../../../betalens/datafeed/core.py#L196) | get_available_dates(self, code: str, metric: str, start_date: str &#124; None=None, end_date: str &#124; None=None) | 无返回注解；return: _get_available_dates(cursor=self._cursor, table_name=self.sheet, code=code, metric=metric, start_date=start_date, end_date=end_date, logger=self.logger) | 无 docstring，需阅读函数体 |
| [Datafeed.query_industry](../../../betalens/datafeed/core.py#L213) | query_industry(self, codes: list[str], dates, scheme: str='申万一级行业', *, exact: bool=False) -&gt; pd.DataFrame | pd.DataFrame | Return point-in-time industry memberships for code/date inputs. |
| [Datafeed.get_industry_members](../../../betalens/datafeed/core.py#L234) | get_industry_members(self, industry, date: str, scheme: str='申万一级行业', *, by: str='name', exact: bool=False) -&gt; pd.DataFrame | pd.DataFrame | Return members of one industry at a point in time. |
| [Datafeed.get_index_universe](../../../betalens/datafeed/core.py#L257) | get_index_universe(self, index_code: str, date: str) -&gt; list[str] | list[str] | Return the latest index constituent snapshot available at ''date''. |
| [Datafeed.get_index_universe_panel](../../../betalens/datafeed/core.py#L269) | get_index_universe_panel(self, index_code: str, dates) -&gt; dict[date, set[str]] | dict[date, set[str]] | Return latest point-in-time constituents for each requested date. |
| [Datafeed.get_index_universe_date](../../../betalens/datafeed/core.py#L281) | get_index_universe_date(self, index_code: str, date: str) | 无返回注解；return: get_index_universe_date(self._cursor, index_code=index_code, date=date, table_name=self.sheet, logger=self.logger) | Return the effective timestamp of an index constituent snapshot. |
| [Datafeed.query_names](../../../betalens/datafeed/core.py#L293) | query_names(self, codes: list[str]) -&gt; pd.DataFrame | pd.DataFrame | Resolve current names for a batch of entity codes. |
| [Datafeed._require](../../../betalens/datafeed/core.py#L321) | _require(params: dict[str, Any] &#124; None, keys: tuple[str, ...]) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [Datafeed.close](../../../betalens/datafeed/core.py#L327) | close(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [get_absolute_trade_days](../../../betalens/datafeed/core.py#L339) | get_absolute_trade_days(begin_date, end_date, period, exchange='SHSE') | 无返回注解；return: [value.date() for value in dates] | Return locally stored exchange trading dates, sampled by period end. |
| [trade_days_offset](../../../betalens/datafeed/core.py#L396) | trade_days_offset(begin_datetime, offset, period='D') | 无返回注解；return: datetime.combine(target.date(), original.time()) | Offset a date/time by exchange trading periods. |

<a id="file-bf00e03912b5"></a>
## betalens/datafeed/industry.py

[打开源码](../../../betalens/datafeed/industry.py) · 385 行 · 说明来源：人工文件说明

- **作用**：历史行业归属查询
- **输入**：cursor、证券、时点、分类体系
- **输出**：行业记录或成员列表
- **副作用/维护重点**：有数据库读取与新旧 schema 分支

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .query import _normalized_schema_available
from typing import Optional, List, Tuple, Union
import itertools
import logging
import pandas as pd
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_get_default_logger](../../../betalens/datafeed/industry.py#L35) | _get_default_logger() | 无返回注解；return: logger | 无 docstring，需阅读函数体 |
| [_explode_remark](../../../betalens/datafeed/industry.py#L47) | _explode_remark(df: pd.DataFrame) -&gt; pd.DataFrame | pd.DataFrame | 把 remark(JSONB-&gt;dict) 展开成 ind_name / ind_code / scheme 列 |
| [_explode_remark._get](../../../betalens/datafeed/industry.py#L54) | _get(r, k) | 无返回注解；return: r.get(k) if isinstance(r, dict) else None | 无 docstring，需阅读函数体 |
| [_scheme_clause](../../../betalens/datafeed/industry.py#L63) | _scheme_clause(scheme: str, exact: bool, col: str='t.metric') -&gt; Tuple[str, str] | Tuple[str, str] | 生成 metric 匹配子句与参数。 版本无关查询：scheme 不带版本后缀（如 '申万一级行业'）时用前缀匹配， 覆盖 '申万一级行业（旧版/2014/2021）' 等全部版本；配合 ORDER BY datetime DESC， 取 datetime&lt;=查询日 的最近一条 → 自动落到查询日生效的那个版本，无需硬编码版本边界。 带版本后缀（如 '申万一级行业（2021）'）时前缀匹配退化为精确，只命中该版本。 exact=True 则强制精确匹配（旧行为）。 Args: scheme: 分类体系名 exact: True 强制精确匹配 col: metric 列引用（带表别名前缀，如 '…（完整内容见 inventory.json/源码） |
| [query_industry](../../../betalens/datafeed/industry.py#L88) | query_industry(cursor, codes: List[str], dates: Union[str, List[str]], scheme: str='申万一级行业', table_name: str=DEFAULT_TABLE, exact: bool=False, logger: Optional[logging.Logger]=None) -&gt; pd.DataFrame | pd.DataFrame | 正查：每个 (code, date) 在该日所属的行业（point-in-time，取 datetime&lt;=date 的最近一条） Args: cursor: 数据库游标（建议 RealDictCursor） codes: 证券代码列表 dates: 查询日期，单个或列表，格式 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS' scheme: 分类体系（即 metric）。不带版本后缀（如 '申万一级行业'）时自动匹配全部 版本，最近一条天然落到查询日生效的版本；带后缀（如 '申万一级行业（2021）'） 则只查该版本。 table_name: 表名，默认 'indus…（完整内容见 inventory.json/源码） |
| [get_industry_members](../../../betalens/datafeed/industry.py#L175) | get_industry_members(cursor, industry: Union[str, int, float], date: str, scheme: str='申万一级行业', table_name: str=DEFAULT_TABLE, by: str='name', exact: bool=False, logger: Optional[logging.Logger]=None) -&gt; pd.DataFrame | pd.DataFrame | 反查：某日某行业的成分股（每只股票取 datetime&lt;=date 的最近归属，再筛目标行业） Args: cursor: 数据库游标 industry: 目标行业，可为行业名(str，匹配 remark-&gt;&gt;'ind_name') 或行业代码数值(int/float，匹配 value) date: 查询日期 scheme: 分类体系（metric）。不带版本后缀时自动匹配全部版本（最近一条天然落到 查询日生效的版本）；带后缀只查该版本。 table_name: 表名 by: 'name' 用行业名匹配，'value' 用行业代码数值匹配； industry 类型也会自动推断 exact: 强…（完整内容见 inventory.json/源码） |
| [_query_industry_normalized](../../../betalens/datafeed/industry.py#L252) | _query_industry_normalized(cursor, codes: List[str], dates: List[str], scheme: str, exact: bool, logger: logging.Logger) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_get_industry_members_normalized](../../../betalens/datafeed/industry.py#L323) | _get_industry_members_normalized(cursor, industry: Union[str, int, float], date: str, scheme: str, by: str, exact: bool, logger: logging.Logger) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |

<a id="file-4c0ff3abab41"></a>
## betalens/datafeed/pool.py

[打开源码](../../../betalens/datafeed/pool.py) · 123 行 · 说明来源：人工文件说明

- **作用**：线程安全只读连接池
- **输入**：数据库配置、连接数和超时
- **输出**：借出的连接/上下文
- **副作用/维护重点**：有连接和全局池生命周期；不能用于入库

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .config import get_database_config
from __future__ import annotations
from contextlib import contextmanager
from psycopg2.extensions import connection as Connection
from psycopg2.pool import ThreadedConnectionPool
from typing import Any, Iterator
import atexit
import threading
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_clean_config](../../../betalens/datafeed/pool.py#L21) | _clean_config(config: dict[str, Any] &#124; None) -&gt; dict[str, Any] | dict[str, Any] | 无 docstring，需阅读函数体 |
| [ReadOnlyConnectionPool](../../../betalens/datafeed/pool.py#L30) | class ReadOnlyConnectionPool() | 类定义；构造/属性见方法与字段 | A small pool whose checked-out sessions cannot modify the database. |
| [ReadOnlyConnectionPool.__init__](../../../betalens/datafeed/pool.py#L33) | __init__(self, db_config: dict[str, Any] &#124; None=None, min_connections: int=1, max_connections: int=10, statement_timeout_ms: int=120000) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ReadOnlyConnectionPool.acquire](../../../betalens/datafeed/pool.py#L48) | acquire(self) -&gt; Connection | Connection | 无 docstring，需阅读函数体 |
| [ReadOnlyConnectionPool.release](../../../betalens/datafeed/pool.py#L66) | release(self, conn: Connection &#124; None) -&gt; None | None | 无 docstring，需阅读函数体 |
| [ReadOnlyConnectionPool.connection](../../../betalens/datafeed/pool.py#L76) | connection(self) -&gt; Iterator[Connection] | Iterator[Connection] | 无 docstring，需阅读函数体 |
| [ReadOnlyConnectionPool.closeall](../../../betalens/datafeed/pool.py#L83) | closeall(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [get_read_pool](../../../betalens/datafeed/pool.py#L87) | get_read_pool(db_config: dict[str, Any] &#124; None=None, min_connections: int=1, max_connections: int=10, statement_timeout_ms: int=120000) -&gt; ReadOnlyConnectionPool | ReadOnlyConnectionPool | 无 docstring，需阅读函数体 |
| [close_all_pools](../../../betalens/datafeed/pool.py#L115) | close_all_pools() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-e3fa2d70263c"></a>
## betalens/datafeed/query.py

[打开源码](../../../betalens/datafeed/query.py) · 1656 行 · 说明来源：人工文件说明

- **作用**：参数化查询与新旧结构路由
- **输入**：cursor、逻辑表、时间范围和指标
- **输出**：查询长表、宽表、日期/状态
- **副作用/维护重点**：读取 PostgreSQL；nearest 与 in-range 的区间语义要保持

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .registry import CoreMetric, get_core_metric, get_dataset
from datetime import datetime, timedelta
from psycopg2 import sql as psql
from typing import Optional, List, Dict, Tuple, Any, Union
import itertools
import logging
import numpy as np
import pandas as pd
import re
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_connection_cache_key](../../../betalens/datafeed/query.py#L29) | _connection_cache_key(cursor) -&gt; tuple[Any, ...] | tuple[Any, ...] | 无 docstring，需阅读函数体 |
| [_normalized_schema_available](../../../betalens/datafeed/query.py#L46) | _normalized_schema_available(cursor) -&gt; bool | bool | Return true when the db-manager normalized schema is installed. |
| [_time_bounds](../../../betalens/datafeed/query.py#L62) | _time_bounds(start_date: Optional[str], end_date: Optional[str]) -&gt; tuple[pd.Timestamp &#124; None, pd.Timestamp &#124; None, bool] | tuple[pd.Timestamp &#124; None, pd.Timestamp &#124; None, bool] | 无 docstring，需阅读函数体 |
| [_trade_date_bounds](../../../betalens/datafeed/query.py#L77) | _trade_date_bounds(start: pd.Timestamp &#124; None, end: pd.Timestamp &#124; None, end_exclusive: bool, available_time) | 无返回注解；return: (lower, upper) | Convert fixed intraday availability bounds into indexable dates. |
| [_trade_date_bounds.naive](../../../betalens/datafeed/query.py#L85) | naive(value: pd.Timestamp) -&gt; pd.Timestamp | pd.Timestamp | 无 docstring，需阅读函数体 |
| [_resolve_metric](../../../betalens/datafeed/query.py#L106) | _resolve_metric(cursor, dataset: str, metric: str) | 无返回注解；return: cached; None; resolved | 无 docstring，需阅读函数体 |
| [_row_value](../../../betalens/datafeed/query.py#L149) | _row_value(row, key: str, position: int) | 无返回注解；return: row.get(key) if isinstance(row, dict) else row[position] | 无 docstring，需阅读函数体 |
| [_resolved_core_metric](../../../betalens/datafeed/query.py#L153) | _resolved_core_metric(row) -&gt; CoreMetric &#124; None | CoreMetric &#124; None | 无 docstring，需阅读函数体 |
| [_query_compatibility_view](../../../betalens/datafeed/query.py#L167) | _query_compatibility_view(cursor, table_name: str, codes: Optional[List[str]], start_date: Optional[str], end_date: Optional[str], metric: Optional[str], limit: Optional[int]) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_empty_nearest](../../../betalens/datafeed/query.py#L205) | _empty_nearest(codes, anchors, metric: str, ranges: bool=False, direction: str='before') -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_query_time_range_normalized](../../../betalens/datafeed/query.py#L236) | _query_time_range_normalized(cursor, table_name: str, codes: Optional[List[str]], start_date: Optional[str], end_date: Optional[str], metric: Optional[str], limit: Optional[int]) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_nearest_input_cte](../../../betalens/datafeed/query.py#L406) | _nearest_input_cte(ranges: bool) -&gt; psql.SQL | psql.SQL | 无 docstring，需阅读函数体 |
| [_query_nearest_normalized](../../../betalens/datafeed/query.py#L435) | _query_nearest_normalized(cursor, table_name: str, codes: List[str], anchors, metric: str, direction: str, time_tolerance: Optional[float], ranges: bool=False) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_query_trade_status_normalized](../../../betalens/datafeed/query.py#L604) | _query_trade_status_normalized(cursor, codes: Optional[List[str]], dates: List[str], logger: logging.Logger) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [_get_default_logger](../../../betalens/datafeed/query.py#L682) | _get_default_logger() | 无返回注解；return: logger | 获取默认logger |
| [build_query](../../../betalens/datafeed/query.py#L695) | build_query(table_name: str, conditions: Optional[List[str]]=None, params: Optional[List]=None, select_columns: str='*', order_by: Optional[str]=None, limit: Optional[int]=None) -&gt; Tuple[str, List] | Tuple[str, List] | 构建SQL查询 Args: table_name: 数据库表名 conditions: 条件列表 params: 参数列表 select_columns: 要选择的列 order_by: ORDER BY 子句（如 "datetime DESC"） limit: 最大返回行数 Returns: (SQL语句, 参数列表) |
| [generate_input_pairs](../../../betalens/datafeed/query.py#L734) | generate_input_pairs(codes: List[str], datetimes: List[str]) -&gt; List[Tuple[str, str]] | List[Tuple[str, str]] | 生成(code, datetime)笛卡尔积 Args: codes: 代码列表 datetimes: 时间戳列表 Returns: (code, datetime)元组列表 |
| [generate_input_range_pairs](../../../betalens/datafeed/query.py#L751) | generate_input_range_pairs(codes: List[str], ranges: List[Tuple[str, str]]) -&gt; List[Tuple[str, str, str]] | List[Tuple[str, str, str]] | 生成 (code, start_ts, end_ts) 笛卡尔积 Args: codes: 代码列表 ranges: (start_ts, end_ts) 区间列表 Returns: (code, start_ts, end_ts) 元组列表 |
| [build_nearest_in_range_query](../../../betalens/datafeed/query.py#L768) | build_nearest_in_range_query(table_name: str, input_tuples: List[Tuple[str, str, str]], metric: str, direction: str='after', time_tolerance: Optional[float]=None) -&gt; Tuple[str, List] | Tuple[str, List] | 构建区间内最近时点匹配查询 在每个 (code, start_ts, end_ts) 区间内，按方向查找距锚点最近的数据： - direction='after'：锚点为 start_ts，区间过滤 t.datetime &gt; start AND t.datetime &lt; end - direction='before'：锚点为 end_ts，区间过滤 t.datetime &lt;= end AND t.datetime &gt;= start Args: table_name: 表名 input_tuples: (code, start_ts, end_ts) 元组列表 metric: 指标名 dire…（完整内容见 inventory.json/源码） |
| [build_nearest_query](../../../betalens/datafeed/query.py#L860) | build_nearest_query(table_name: str, input_tuples: List[Tuple[str, str]], metric: str, direction: str='after', time_tolerance: Optional[float]=None) -&gt; Tuple[str, List] | Tuple[str, List] | 构建最近时点匹配查询 Args: table_name: 表名 input_tuples: (code, datetime)元组列表 metric: 指标名 direction: 查询方向，'after'（之后）或'before'（之前） time_tolerance: 时间容差（小时） Returns: (SQL语句, 参数列表) |
| [query_nearest_after](../../../betalens/datafeed/query.py#L947) | query_nearest_after(cursor, table_name: str, codes: List[str], datetimes: List[str], metric: str, time_tolerance: Optional[float]=None, logger: Optional[logging.Logger]=None) -&gt; pd.DataFrame | pd.DataFrame | 查询每个时点之后最近的有效值 用途：主要用于回测时提取价格 时间结构：最新特征 &lt;= 提数时点 &lt; 调仓时点 Args: cursor: 数据库游标 table_name: 表名 codes: 代码列表 datetimes: 时间戳列表，格式'YYYY-MM-DD HH:MM:SS' metric: 查询的指标名称 time_tolerance: 允许的最大时间间隔（单位：小时） logger: 日志记录器，如果为None则使用默认logger Returns: DataFrame，包含列： - code: 代码 - input_ts: 输入时间戳（提数时点） - datetime: 匹配到的…（完整内容见 inventory.json/源码） |
| [query_nearest_before](../../../betalens/datafeed/query.py#L1029) | query_nearest_before(cursor, table_name: str, codes: List[str], datetimes: List[str], metric: str, time_tolerance: Optional[float]=None, logger: Optional[logging.Logger]=None) -&gt; pd.DataFrame | pd.DataFrame | 查询每个时点之前最近的有效值 用途：主要用于回测时提取历史价格特征 时间结构：调仓时点 &lt;= 提数时点 &lt; 最新特征时点 Args: cursor: 数据库游标 table_name: 表名 codes: 代码列表 datetimes: 时间戳列表，格式'YYYY-MM-DD HH:MM:SS' metric: 查询的指标名称 time_tolerance: 允许的最大时间间隔（单位：小时） logger: 日志记录器，如果为None则使用默认logger Returns: DataFrame，包含列： - code: 代码 - input_ts: 输入时间戳（提数时点） - datetime…（完整内容见 inventory.json/源码） |
| [query_nearest_in_range_after](../../../betalens/datafeed/query.py#L1111) | query_nearest_in_range_after(cursor, table_name: str, codes: List[str], ranges: List[Tuple[str, str]], metric: str, time_tolerance: Optional[float]=None, logger: Optional[logging.Logger]=None) -&gt; pd.DataFrame | pd.DataFrame | 在每个 (start, end) 区间内查询距 start 最近的有效值（向后查） 时间结构：start &lt;= t.datetime - epsilon, t.datetime &lt; end，锚点 = start Args: cursor: 数据库游标 table_name: 表名 codes: 代码列表 ranges: (start, end) 区间列表，时间格式 'YYYY-MM-DD HH:MM:SS' metric: 指标名 time_tolerance: 锚点容差（小时），与区间共同生效 logger: 日志记录器 Returns: DataFrame: code, input_ts(…（完整内容见 inventory.json/源码） |
| [query_nearest_in_range_before](../../../betalens/datafeed/query.py#L1184) | query_nearest_in_range_before(cursor, table_name: str, codes: List[str], ranges: List[Tuple[str, str]], metric: str, time_tolerance: Optional[float]=None, logger: Optional[logging.Logger]=None) -&gt; pd.DataFrame | pd.DataFrame | 在每个 (start, end) 区间内查询距 end 最近的有效值（向前查） 时间结构：start &lt;= t.datetime &lt;= end，锚点 = end Args: cursor: 数据库游标 table_name: 表名 codes: 代码列表 ranges: (start, end) 区间列表，时间格式 'YYYY-MM-DD HH:MM:SS' metric: 指标名 time_tolerance: 锚点容差（小时），与区间共同生效 logger: 日志记录器 Returns: DataFrame: code, input_ts(=end), datetime, diff_hou…（完整内容见 inventory.json/源码） |
| [query_time_range](../../../betalens/datafeed/query.py#L1257) | query_time_range(cursor, table_name: str, codes: Optional[List[str]]=None, start_date: Optional[str]=None, end_date: Optional[str]=None, metric: Optional[str]=None, limit: Optional[int]=None, logger: Optional[logging.Logger]=None) -&gt; pd.DataFrame | pd.DataFrame | 查询指定时间范围的数据 Args: cursor: 数据库游标 table_name: 表名 codes: 代码列表，None表示所有代码 start_date: 开始日期 end_date: 结束日期 metric: 指标名称 limit: 最大返回行数，None表示不限制（按 datetime DESC 返回最新的 N 行） logger: 日志记录器，如果为None则使用默认logger Returns: DataFrame |
| [get_available_dates](../../../betalens/datafeed/query.py#L1332) | get_available_dates(cursor, table_name: str, code: str, metric: str, start_date: Optional[str]=None, end_date: Optional[str]=None, logger: Optional[logging.Logger]=None) -&gt; List[datetime] | List[datetime] | 获取指定代码和指标的可用日期列表 Args: cursor: 数据库游标 table_name: 表名 code: 代码 metric: 指标 start_date: 开始日期 end_date: 结束日期 logger: 日志记录器，如果为None则使用默认logger Returns: 日期列表 |
| [get_latest_date](../../../betalens/datafeed/query.py#L1402) | get_latest_date(cursor, table_name: str, code: Optional[str]=None, metric: Optional[str]=None, logger: Optional[logging.Logger]=None) -&gt; Optional[datetime] | Optional[datetime] | 获取最新的数据日期 Args: cursor: 数据库游标 table_name: 表名 code: 代码，None表示所有代码 metric: 指标，None表示所有指标 logger: 日志记录器，如果为None则使用默认logger Returns: 最新日期 |
| [query_trade_status](../../../betalens/datafeed/query.py#L1455) | query_trade_status(cursor, table_name: str, codes: Optional[List[str]], dates: List[str], metric: str='交易状态', logger: Optional[logging.Logger]=None) -&gt; pd.DataFrame | pd.DataFrame | 查询个券交易状态（适配稀疏存储） 表中仅存异常状态(value=0)与首次正常交易日锚点(value=1, remark.first_normal=true)。 本函数在 Python 端把稀疏记录解析为每个 (code, date) 的完整状态： -1 = 无法交易（首次正常交易日之前，视为未上市/未交易） 0 = 异常（停牌等，status_text 给出文本） 1 = 正常交易 Args: cursor: 数据库游标（RealDictCursor） table_name: 表名（trade_status） codes: 代码列表；None 表示全市场（取所有有锚点的代码） dates: …（完整内容见 inventory.json/源码） |
| [pivot_to_wide](../../../betalens/datafeed/query.py#L1563) | pivot_to_wide(df: pd.DataFrame, index_cols: List[str], pivot_col: str, value_col: str) -&gt; pd.DataFrame | pd.DataFrame | 将长格式数据转换为宽格式 Args: df: 长格式DataFrame index_cols: 索引列 pivot_col: 用于pivot的列（将变为新列名） value_col: 值列 Returns: 宽格式DataFrame |
| [align_to_dates](../../../betalens/datafeed/query.py#L1589) | align_to_dates(df: pd.DataFrame, target_dates: List[datetime], date_column: str='datetime', method: str='ffill') -&gt; pd.DataFrame | pd.DataFrame | 将数据对齐到目标日期序列 Args: df: 输入DataFrame target_dates: 目标日期列表 date_column: 日期列名 method: 填充方法，'ffill'或'bfill' Returns: 对齐后的DataFrame |
| [calculate_returns](../../../betalens/datafeed/query.py#L1627) | calculate_returns(df: pd.DataFrame, price_column: str, periods: List[int]=[1], group_by: Optional[str]=None) -&gt; pd.DataFrame | pd.DataFrame | 计算收益率 Args: df: 包含价格数据的DataFrame price_column: 价格列名 periods: 计算周期列表 group_by: 分组列（如code） Returns: 添加了收益率列的DataFrame |

<a id="file-ff958942dfcb"></a>
## betalens/datafeed/registry.py

[打开源码](../../../betalens/datafeed/registry.py) · 103 行 · 说明来源：人工文件说明

- **作用**：读层逻辑数据集与核心指标映射
- **输入**：数据集名、指标名
- **输出**：DatasetSpec/CoreMetric
- **副作用/维护重点**：与写层 registry 和兼容视图保持一致

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import time
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [DatasetSpec](../../../betalens/datafeed/registry.py#L13) | class DatasetSpec() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：logical_name: str; kind: str; entity_type: str &#124; None = None |
| [CoreMetric](../../../betalens/datafeed/registry.py#L20) | class CoreMetric() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：canonical_name: str; column: str; available_time: time |
| [_metrics](../../../betalens/datafeed/registry.py#L45) | _metrics(*items: tuple[str, str, tuple[str, ...], str, time]) -&gt; dict[tuple[str, str], CoreMetric] | dict[tuple[str, str], CoreMetric] | 无 docstring，需阅读函数体 |
| [get_dataset](../../../betalens/datafeed/registry.py#L98) | get_dataset(name: str) -&gt; DatasetSpec &#124; None | DatasetSpec &#124; None | 无 docstring，需阅读函数体 |
| [get_core_metric](../../../betalens/datafeed/registry.py#L102) | get_core_metric(dataset: str, metric: str) -&gt; CoreMetric &#124; None | CoreMetric &#124; None | 无 docstring，需阅读函数体 |

<a id="file-ad80ef4db74f"></a>
## betalens/datafeed/universe.py

[打开源码](../../../betalens/datafeed/universe.py) · 290 行 · 说明来源：人工文件说明

- **作用**：历史指数成分查询
- **输入**：cursor、指数代码、历史日期
- **输出**：集合、有效日期、date→set 面板
- **副作用/维护重点**：PIT 快照读取；不能拿最新成分填全历史

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .query import _normalized_schema_available
from .query import query_nearest_before
from datetime import date as Date
from typing import Iterable, Optional, List
import logging
import pandas as pd
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_normalize_query_dates](../../../betalens/datafeed/universe.py#L40) | _normalize_query_dates(dates: Iterable) -&gt; list[pd.Timestamp] | list[pd.Timestamp] | 无 docstring，需阅读函数体 |
| [get_index_universe_panel](../../../betalens/datafeed/universe.py#L52) | get_index_universe_panel(cursor, index_code: str, dates: Iterable, table_name: str=DEFAULT_TABLE, metric: str=DEFAULT_METRIC, logger: Optional[logging.Logger]=None) -&gt; dict[Date, set[str]] | dict[Date, set[str]] | Return point-in-time constituents for many dates in one call. |
| [_get_default_logger](../../../betalens/datafeed/universe.py#L121) | _get_default_logger() | 无返回注解；return: logger | 无 docstring，需阅读函数体 |
| [get_index_universe_date](../../../betalens/datafeed/universe.py#L133) | get_index_universe_date(cursor, index_code: str, date: str, table_name: str=DEFAULT_TABLE, metric: str=DEFAULT_METRIC, logger: Optional[logging.Logger]=None) | 无返回注解；return: None; pd.Timestamp(value); eff_dt | 返回某指数在某日实际生效的快照日期（point-in-time，取 datetime&lt;=date 的最近一条）。 复用 query.query_nearest_before 定位最近生效日。 Args: cursor: 数据库游标（建议 RealDictCursor） index_code: 指数代码，如 '000906.SH' date: 查询日期，'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS' table_name: 表名，默认 'index_universe' metric: 指标名，默认 'universe' logger: 日志器 Returns: 生效快照…（完整内容见 inventory.json/源码） |
| [get_index_universe](../../../betalens/datafeed/universe.py#L206) | get_index_universe(cursor, index_code: str, date: str, table_name: str=DEFAULT_TABLE, metric: str=DEFAULT_METRIC, logger: Optional[logging.Logger]=None) -&gt; List[str] | List[str] | 返回 index_code 在 date 当日生效的成分股代码列表（point-in-time）。 步骤：用 query.query_nearest_before 找到 &lt;=date 的最近生效快照日，再取该行 remark 中的 constituents 列表。该日前无可用股票池则返回空列表。 Args: cursor: 数据库游标（建议 RealDictCursor） index_code: 指数代码，如 '000906.SH' date: 查询日期，'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS' table_name: 表名，默认 'index_universe…（完整内容见 inventory.json/源码） |

<a id="file-abc11d466ee2"></a>
## betalens/datafeed/validation.py

[打开源码](../../../betalens/datafeed/validation.py) · 721 行 · 说明来源：人工文件说明

- **作用**：数据缺失、日期、重复检查修复
- **输入**：DataFrame、字段、FillStrategy
- **输出**：检查结果或修复后的表
- **副作用/维护重点**：修复策略可能改变样本，不能无条件填零

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from datetime import datetime
from enum import Enum
from typing import Optional, Union, List, Dict, Callable, Tuple, Any
import logging
import numpy as np
import pandas as pd
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [FillStrategy](../../../betalens/datafeed/validation.py#L18) | class FillStrategy(Enum) | 类定义；构造/属性见方法与字段 | 填充策略枚举 |
| [_get_default_logger](../../../betalens/datafeed/validation.py#L31) | _get_default_logger() | 无返回注解；return: logger | 获取默认logger |
| [check_null_values](../../../betalens/datafeed/validation.py#L44) | check_null_values(df: pd.DataFrame, columns: Optional[List[str]]=None, check_types: Optional[List[str]]=None, logger: Optional[logging.Logger]=None) -&gt; Dict[str, Any] | Dict[str, Any] | 检查空值、NaN、None Args: df: 待检查的DataFrame columns: 要检查的列名列表，None表示检查所有列 check_types: 检查类型列表，可选['null', 'nan', 'none', 'empty_string'] logger: 日志记录器，如果为None则使用默认logger Returns: 检查结果字典 |
| [check_datetime_column](../../../betalens/datafeed/validation.py#L122) | check_datetime_column(df: pd.DataFrame, date_column: str, expected_freq: Optional[str]=None, check_sorted: bool=True, check_duplicates: bool=True, check_format: bool=True, logger: Optional[logging.Logger]=None) -&gt; Dict[str, Any] | Dict[str, Any] | 检查日期列的各种问题 Args: df: DataFrame date_column: 日期列名 expected_freq: 期望的频率，如'D'(日), 'W'(周), 'M'(月), 'Q'(季度), 'Y'(年) check_sorted: 是否检查排序 check_duplicates: 是否检查重复 check_format: 是否检查格式 logger: 日志记录器，如果为None则使用默认logger Returns: 检查结果字典 |
| [fix_null_values](../../../betalens/datafeed/validation.py#L270) | fix_null_values(df: pd.DataFrame, strategy: Union[FillStrategy, str], columns: Optional[List[str]]=None, fill_value: Any=None, inplace: bool=False, logger: Optional[logging.Logger]=None) -&gt; pd.DataFrame | pd.DataFrame | 修复空值 Args: df: DataFrame strategy: 填充策略 columns: 要处理的列，None表示所有列 fill_value: 当strategy为FILL_VALUE时使用的填充值 inplace: 是否原地修改 logger: 日志记录器，如果为None则使用默认logger Returns: 修复后的DataFrame |
| [drop_duplicates_strict](../../../betalens/datafeed/validation.py#L371) | drop_duplicates_strict(df: pd.DataFrame, subset: Optional[List[str]]=None, keep: str='first', verify_all_fields: bool=True, ignore_cols: Optional[List[str]]=None, inplace: bool=False, logger: Optional[logging.Logger]=None) -&gt; Tuple[pd.DataFrame, Dict[str, Any]] | Tuple[pd.DataFrame, Dict[str, Any]] | 严格去重：确保只有完全相同的行才会被删除 Args: df: DataFrame subset: 用于判断重复的列，None表示所有列 keep: 'first', 'last', False（删除所有重复） verify_all_fields: 是否验证subset外的其他字段也相同 ignore_cols: 验证时忽略的列（如索引、时间戳等） inplace: 是否原地修改 logger: 日志记录器，如果为None则使用默认logger Returns: (修复后的DataFrame, 去重报告) |
| [fix_datetime_column](../../../betalens/datafeed/validation.py#L531) | fix_datetime_column(df: pd.DataFrame, date_column: str, fix_format: bool=True, fix_duplicates: Optional[str]='keep_first', fix_sort: bool=True, sort_order: str='ascending', dedupe_subset: Optional[List[str]]=None, verify_all_fields: bool=True, inplace: bool=False, logger: Optional[logging.Logger]=None) -&gt; pd.DataFrame | pd.DataFrame | 修复日期列的问题 Args: df: DataFrame date_column: 日期列名 fix_format: 是否修复格式（转换为datetime） fix_duplicates: 如何处理重复，None表示不处理 fix_sort: 是否排序 sort_order: 排序顺序，'ascending'或'descending' dedupe_subset: 去重时使用的列组合，None则使用[date_column] 推荐: ['code', 'metric', date_column] 避免误删不同metric的数据 verify_all_fields: 是否验证subset外的其他…（完整内容见 inventory.json/源码） |
| [validate_and_fix](../../../betalens/datafeed/validation.py#L627) | validate_and_fix(df: pd.DataFrame, validations: Dict[str, Dict], inplace: bool=False, logger: Optional[logging.Logger]=None) -&gt; Tuple[pd.DataFrame, Dict[str, Any]] | Tuple[pd.DataFrame, Dict[str, Any]] | 综合验证和修复 Args: df: DataFrame validations: 验证配置字典，格式如： { 'null_check': { 'columns': ['col1', 'col2'], 'fix_strategy': 'ffill' }, 'datetime_check': { 'column': 'date', 'expected_freq': 'D', 'fix_format': True, 'fix_duplicates': 'keep_first', 'fix_sort': True } } inplace: 是否原地修改 logger: 日志记录器，如果为None则使用…（完整内容见 inventory.json/源码） |

