# tests：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-7a183a1b3480"></a>
## dashboard/backend/test_eventstudy_dashboard.py

[打开源码](../../../dashboard/backend/test_eventstudy_dashboard.py) · 291 行 · 说明来源：文件族规则

- **作用**：回归测试：test_eventstudy_dashboard.py
- **输入**：fixture、合成样本与 mock；依赖由测试正文决定
- **输出**：断言通过/失败及测试报告
- **副作用/维护重点**：逐个测试签名和 docstring 见下；测试存在不代表当前已通过

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from . import eventstudy_dashboard
from .eventstudy_dashboard import _comparison_payload, _price_matrix_records, _asset_payload, _parse_codes, discover_event_files, run_event_study
from .schemas import EventStudyRequest
from __future__ import annotations
from betalens.eventstudy.eventstudy import EventStudy
from unittest.mock import patch
import numpy as np
import pandas as pd
import unittest
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [EventStudyDashboardTests](../../../dashboard/backend/test_eventstudy_dashboard.py#L23) | class EventStudyDashboardTests(unittest.TestCase) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_asset_payload_includes_name_and_falls_back_to_code](../../../dashboard/backend/test_eventstudy_dashboard.py#L24) | test_asset_payload_includes_name_and_falls_back_to_code(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_asset_name_lookup_failure_does_not_block_result](../../../dashboard/backend/test_eventstudy_dashboard.py#L34) | test_asset_name_lookup_failure_does_not_block_result(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_discover_event_files_reads_local_xlsx](../../../dashboard/backend/test_eventstudy_dashboard.py#L38) | test_discover_event_files_reads_local_xlsx(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_fixed_holding_returns_use_holding_start_offset](../../../dashboard/backend/test_eventstudy_dashboard.py#L52) | test_fixed_holding_returns_use_holding_start_offset(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_comparison_payload_is_json_safe_and_preserves_event_ids](../../../dashboard/backend/test_eventstudy_dashboard.py#L68) | test_comparison_payload_is_json_safe_and_preserves_event_ids(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_matrix_records_use_stable_event_id_for_date_lookup](../../../dashboard/backend/test_eventstudy_dashboard.py#L112) | test_matrix_records_use_stable_event_id_for_date_lookup(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_parse_codes_accepts_ascii_chinese_and_list_separators](../../../dashboard/backend/test_eventstudy_dashboard.py#L121) | test_parse_codes_accepts_ascii_chinese_and_list_separators(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_explicit_blank_code_does_not_fall_back_to_default](../../../dashboard/backend/test_eventstudy_dashboard.py#L131) | test_explicit_blank_code_does_not_fall_back_to_default(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_explicit_blank_code_does_not_fall_back_to_default.FakeDatafeed](../../../dashboard/backend/test_eventstudy_dashboard.py#L132) | class FakeDatafeed() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_explicit_blank_code_does_not_fall_back_to_default.FakeDatafeed.__init__](../../../dashboard/backend/test_eventstudy_dashboard.py#L133) | __init__(self, table_name: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_explicit_blank_code_does_not_fall_back_to_default.FakeDatafeed.close](../../../dashboard/backend/test_eventstudy_dashboard.py#L136) | close(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_explicit_blank_code_does_not_fall_back_to_default.FakeStudy](../../../dashboard/backend/test_eventstudy_dashboard.py#L139) | class FakeStudy() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_explicit_blank_code_does_not_fall_back_to_default.FakeStudy.__init__](../../../dashboard/backend/test_eventstudy_dashboard.py#L140) | __init__(self, datafeed) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_explicit_blank_code_does_not_fall_back_to_default.FakeStudy.analyze](../../../dashboard/backend/test_eventstudy_dashboard.py#L143) | analyze(self, **kwargs) | 无返回注解；return: {'event_count': 1, 'valid_codes': [kwargs['code']]} | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_missing_code_field_uses_configured_default](../../../dashboard/backend/test_eventstudy_dashboard.py#L173) | test_missing_code_field_uses_configured_default(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_missing_code_field_uses_configured_default.FakeDatafeed](../../../dashboard/backend/test_eventstudy_dashboard.py#L174) | class FakeDatafeed() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_missing_code_field_uses_configured_default.FakeDatafeed.__init__](../../../dashboard/backend/test_eventstudy_dashboard.py#L175) | __init__(self, table_name: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_missing_code_field_uses_configured_default.FakeDatafeed.close](../../../dashboard/backend/test_eventstudy_dashboard.py#L178) | close(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_missing_code_field_uses_configured_default.FakeStudy](../../../dashboard/backend/test_eventstudy_dashboard.py#L181) | class FakeStudy() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：received: dict = {} |
| [EventStudyDashboardTests.test_missing_code_field_uses_configured_default.FakeStudy.__init__](../../../dashboard/backend/test_eventstudy_dashboard.py#L184) | __init__(self, datafeed) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_missing_code_field_uses_configured_default.FakeStudy.analyze](../../../dashboard/backend/test_eventstudy_dashboard.py#L187) | analyze(self, **kwargs) | 无返回注解；return: {'event_count': 1, 'valid_codes': [kwargs['code']]} | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_run_forwards_compare_mode](../../../dashboard/backend/test_eventstudy_dashboard.py#L220) | test_run_forwards_compare_mode(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_run_forwards_compare_mode.FakeDatafeed](../../../dashboard/backend/test_eventstudy_dashboard.py#L231) | class FakeDatafeed() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_run_forwards_compare_mode.FakeDatafeed.__init__](../../../dashboard/backend/test_eventstudy_dashboard.py#L232) | __init__(self, table_name: str) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_run_forwards_compare_mode.FakeDatafeed.close](../../../dashboard/backend/test_eventstudy_dashboard.py#L235) | close(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_run_forwards_compare_mode.FakeStudy](../../../dashboard/backend/test_eventstudy_dashboard.py#L238) | class FakeStudy() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体；字段：received: dict = {} |
| [EventStudyDashboardTests.test_run_forwards_compare_mode.FakeStudy.__init__](../../../dashboard/backend/test_eventstudy_dashboard.py#L241) | __init__(self, datafeed) -&gt; None | None | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_run_forwards_compare_mode.FakeStudy.analyze](../../../dashboard/backend/test_eventstudy_dashboard.py#L244) | analyze(self, **kwargs) | 无返回注解；return: raw | 无 docstring，需阅读函数体 |
| [EventStudyDashboardTests.test_request_accepts_compare_mode](../../../dashboard/backend/test_eventstudy_dashboard.py#L285) | test_request_accepts_compare_mode(self) -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-4a0a2e5b70e5"></a>
## dashboard/backend/test_factor_yaml.py

[打开源码](../../../dashboard/backend/test_factor_yaml.py) · 168 行 · 说明来源：文件族规则

- **作用**：回归测试：test_factor_yaml.py
- **输入**：fixture、合成样本与 mock；依赖由测试正文决定
- **输出**：断言通过/失败及测试报告
- **副作用/维护重点**：逐个测试签名和 docstring 见下；测试存在不代表当前已通过

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .factors import clear_factor_cache, discover_factors, get_factor_config
from .runs import DashboardRun, RunManager
from .schemas import RunRequest
from __future__ import annotations
from betalens.factor.config import write_yaml_config
from pathlib import Path
import tempfile
import unittest
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [FactorYamlDashboardTests](../../../dashboard/backend/test_factor_yaml.py#L14) | class FactorYamlDashboardTests(unittest.TestCase) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [FactorYamlDashboardTests.test_discover_factors_reads_yaml_specs](../../../dashboard/backend/test_factor_yaml.py#L15) | test_discover_factors_reads_yaml_specs(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FactorYamlDashboardTests.test_discover_factors_exposes_complete_alpha101_catalog](../../../dashboard/backend/test_factor_yaml.py#L33) | test_discover_factors_exposes_complete_alpha101_catalog(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FactorYamlDashboardTests.test_get_factor_config_supports_multiple_yaml_specs_in_one_dir](../../../dashboard/backend/test_factor_yaml.py#L53) | test_get_factor_config_supports_multiple_yaml_specs_in_one_dir(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FactorYamlDashboardTests.test_run_config_copy_uses_output_dir_and_yaml_source](../../../dashboard/backend/test_factor_yaml.py#L65) | test_run_config_copy_uses_output_dir_and_yaml_source(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FactorYamlDashboardTests.test_run_config_normalizes_dashboard_group_inputs](../../../dashboard/backend/test_factor_yaml.py#L90) | test_run_config_normalizes_dashboard_group_inputs(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FactorYamlDashboardTests.test_run_config_rejects_empty_freeplay_groups](../../../dashboard/backend/test_factor_yaml.py#L109) | test_run_config_rejects_empty_freeplay_groups(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FactorYamlDashboardTests.test_timing_run_config_allows_empty_freeplay_groups](../../../dashboard/backend/test_factor_yaml.py#L125) | test_timing_run_config_allows_empty_freeplay_groups(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FactorYamlDashboardTests.test_run_config_preserves_nested_compute_kwargs](../../../dashboard/backend/test_factor_yaml.py#L144) | test_run_config_preserves_nested_compute_kwargs(self) -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-da602679d1fc"></a>
## dashboard/backend/test_serialization.py

[打开源码](../../../dashboard/backend/test_serialization.py) · 428 行 · 说明来源：文件族规则

- **作用**：回归测试：test_serialization.py
- **输入**：fixture、合成样本与 mock；依赖由测试正文决定
- **输出**：断言通过/失败及测试报告
- **副作用/维护重点**：逐个测试签名和 docstring 见下；测试存在不代表当前已通过

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from . import serialization as serialization_module
from .serialization import build_chart_data, build_factor_profile_payload, build_generated_chart_data, build_position_table, build_timing_payload, read_table_page, write_table_parquet
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
import unittest
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [TablePagingTests](../../../dashboard/backend/test_serialization.py#L22) | class TablePagingTests(unittest.TestCase) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_write_and_read_page](../../../dashboard/backend/test_serialization.py#L23) | test_write_and_read_page(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_filter_query_and_clean_values](../../../dashboard/backend/test_serialization.py#L39) | test_filter_query_and_clean_values(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_missing_table_returns_empty_page](../../../dashboard/backend/test_serialization.py#L59) | test_missing_table_returns_empty_page(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_unknown_filter_column_returns_empty_page](../../../dashboard/backend/test_serialization.py#L64) | test_unknown_filter_column_returns_empty_page(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_date_range_filter](../../../dashboard/backend/test_serialization.py#L73) | test_date_range_filter(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_position_weight_records_skip_zero_holdings](../../../dashboard/backend/test_serialization.py#L87) | test_position_weight_records_skip_zero_holdings(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_position_weight_records_skip_zero_holdings.FakeBacktest](../../../dashboard/backend/test_serialization.py#L88) | class FakeBacktest() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_rebalance_holdings_include_factor_values](../../../dashboard/backend/test_serialization.py#L113) | test_rebalance_holdings_include_factor_values(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_rebalance_holdings_include_factor_values.FakeBacktest](../../../dashboard/backend/test_serialization.py#L114) | class FakeBacktest() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_generated_charts_match_script_group_and_trade_logic](../../../dashboard/backend/test_serialization.py#L155) | test_generated_charts_match_script_group_and_trade_logic(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_generated_charts_match_script_group_and_trade_logic.FakeBacktest](../../../dashboard/backend/test_serialization.py#L156) | class FakeBacktest() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_generated_group_nav_converts_zero_based_factor_labels](../../../dashboard/backend/test_serialization.py#L197) | test_generated_group_nav_converts_zero_based_factor_labels(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_generated_group_nav_converts_zero_based_factor_labels.FakeBacktest](../../../dashboard/backend/test_serialization.py#L198) | class FakeBacktest() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_factor_profile_contains_all_static_panel_series](../../../dashboard/backend/test_serialization.py#L227) | test_factor_profile_contains_all_static_panel_series(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_position_table_skips_zero_quantity_rows](../../../dashboard/backend/test_serialization.py#L245) | test_position_table_skips_zero_quantity_rows(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_position_table_skips_zero_quantity_rows.FakeBacktest](../../../dashboard/backend/test_serialization.py#L246) | class FakeBacktest() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_timing_payload_summarizes_trades_and_position](../../../dashboard/backend/test_serialization.py#L259) | test_timing_payload_summarizes_trades_and_position(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_timing_payload_summarizes_trades_and_position.FakeBacktest](../../../dashboard/backend/test_serialization.py#L260) | class FakeBacktest() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_timing_payload_consumes_factor_values_with_timing_fields](../../../dashboard/backend/test_serialization.py#L299) | test_timing_payload_consumes_factor_values_with_timing_fields(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_timing_payload_consumes_factor_values_with_timing_fields.FakeBacktest](../../../dashboard/backend/test_serialization.py#L300) | class FakeBacktest() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_timing_payload_aligns_intraday_indexes_to_daily_records](../../../dashboard/backend/test_serialization.py#L358) | test_timing_payload_aligns_intraday_indexes_to_daily_records(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_timing_payload_aligns_intraday_indexes_to_daily_records.FakeBacktest](../../../dashboard/backend/test_serialization.py#L359) | class FakeBacktest() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_timing_prediction_falls_back_without_factor_values](../../../dashboard/backend/test_serialization.py#L411) | test_timing_prediction_falls_back_without_factor_values(self) -&gt; None | None | 无 docstring，需阅读函数体 |
| [TablePagingTests.test_timing_prediction_falls_back_without_factor_values.FakeBacktest](../../../dashboard/backend/test_serialization.py#L412) | class FakeBacktest() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |

<a id="file-97ac7aeb9559"></a>
## tests/test_backtest.py

[打开源码](../../../tests/test_backtest.py) · 55 行 · 说明来源：文件族规则

- **作用**：回归测试：test_backtest.py
- **输入**：fixture、合成样本与 mock；依赖由测试正文决定
- **输出**：断言通过/失败及测试报告
- **副作用/维护重点**：逐个测试签名和 docstring 见下；测试存在不代表当前已通过

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.backtest import BacktestBase
import numpy as np
import pandas as pd
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [test_buy_and_hold_nav_matches_normalized_price_series](../../../tests/test_backtest.py#L9) | test_buy_and_hold_nav_matches_normalized_price_series() -&gt; None | None | A zero-cost buy-and-hold portfolio should track the stock price exactly. |

<a id="file-58083f919dae"></a>
## tests/test_eventstudy.py

[打开源码](../../../tests/test_eventstudy.py) · 142 行 · 说明来源：文件族规则

- **作用**：回归测试：test_eventstudy.py
- **输入**：fixture、合成样本与 mock；依赖由测试正文决定
- **输出**：断言通过/失败及测试报告
- **副作用/维护重点**：逐个测试签名和 docstring 见下；测试存在不代表当前已通过

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.eventstudy.eventstudy import EventStudy, _get_window_prices, _get_window_returns
from unittest.mock import patch
import numpy as np
import pandas as pd
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [FakeDatafeed](../../../tests/test_eventstudy.py#L11) | class FakeDatafeed() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [FakeDatafeed.__init__](../../../tests/test_eventstudy.py#L12) | __init__(self, prices_by_code: dict[str, pd.Series]) -&gt; None | None | 无 docstring，需阅读函数体 |
| [FakeDatafeed.query_time_range](../../../tests/test_eventstudy.py#L15) | query_time_range(self, *, codes, start_date, end_date, metric) | 无返回注解；return: pd.DataFrame(columns=['datetime', 'value']); pd.DataFrame({'datetime': selected.index, 'value': selected.to_numpy()}) | 无 docstring，需阅读函数体 |
| [_calendar](../../../tests/test_eventstudy.py#L27) | _calendar(begin_date, end_date, period, exchange='SHSE') | 无返回注解；return: [value.date() for value in pd.bdate_range(begin_date, end_date)] | 无 docstring，需阅读函数体 |
| [test_daily_return_uses_the_day_close_to_close_change](../../../tests/test_eventstudy.py#L32) | test_daily_return_uses_the_day_close_to_close_change() -&gt; None | None | 无 docstring，需阅读函数体 |
| [test_price_window_is_aligned_to_zero_on_day0](../../../tests/test_eventstudy.py#L46) | test_price_window_is_aligned_to_zero_on_day0() -&gt; None | None | 无 docstring，需阅读函数体 |
| [test_holding_returns_are_always_produced_without_mode](../../../tests/test_eventstudy.py#L60) | test_holding_returns_are_always_produced_without_mode() -&gt; None | None | 无 docstring，需阅读函数体 |
| [test_analyze_returns_daily_holding_and_price_outputs_only](../../../tests/test_eventstudy.py#L71) | test_analyze_returns_daily_holding_and_price_outputs_only() -&gt; None | None | 无 docstring，需阅读函数体 |
| [test_insufficient_event_window_is_skipped_without_calendar_error](../../../tests/test_eventstudy.py#L93) | test_insufficient_event_window_is_skipped_without_calendar_error() -&gt; None | None | 无 docstring，需阅读函数体 |
| [test_all_insufficient_event_windows_return_empty_result_without_error](../../../tests/test_eventstudy.py#L121) | test_all_insufficient_event_windows_return_empty_result_without_error() -&gt; None | None | 无 docstring，需阅读函数体 |

<a id="file-f5389f5d4f0a"></a>
## tests/test_factor_grouping.py

[打开源码](../../../tests/test_factor_grouping.py) · 137 行 · 说明来源：文件族规则

- **作用**：回归测试：test_factor_grouping.py
- **输入**：fixture、合成样本与 mock；依赖由测试正文决定
- **输出**：断言通过/失败及测试报告
- **副作用/维护重点**：逐个测试签名和 docstring 见下；测试存在不代表当前已通过

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from betalens.factor.factor import get_single_factor_weight, single_characteristic
from factor_template import group_balance_statistics
from pathlib import Path
import pandas as pd
import pytest
import sys
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_sample](../../../tests/test_factor_grouping.py#L19) | _sample() -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [test_value_grouping_never_splits_equal_values_and_may_reduce_groups](../../../tests/test_factor_grouping.py#L28) | test_value_grouping_never_splits_equal_values_and_may_reduce_groups() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_equal_count_grouping_has_exact_group_count_and_balanced_sizes](../../../tests/test_factor_grouping.py#L41) | test_equal_count_grouping_has_exact_group_count_and_balanced_sizes() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_equal_count_grouping_rejects_too_few_stocks](../../../tests/test_factor_grouping.py#L55) | test_equal_count_grouping_rejects_too_few_stocks() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_classic_long_short_uses_actual_extreme_labels_for_value_groups](../../../tests/test_factor_grouping.py#L65) | test_classic_long_short_uses_actual_extreme_labels_for_value_groups() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_freeplay_selector_compatibility_depends_on_grouping_mode](../../../tests/test_factor_grouping.py#L89) | test_freeplay_selector_compatibility_depends_on_grouping_mode() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_group_balance_profiling_reports_count_and_value_separation](../../../tests/test_factor_grouping.py#L126) | test_group_balance_profiling_reports_count_and_value_separation() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |

<a id="file-3d0753a5285e"></a>
## tests/test_factor_mining.py

[打开源码](../../../tests/test_factor_mining.py) · 832 行 · 说明来源：文件族规则

- **作用**：回归测试：test_factor_mining.py
- **输入**：fixture、合成样本与 mock；依赖由测试正文决定
- **输出**：断言通过/失败及测试报告
- **副作用/维护重点**：逐个测试签名和 docstring 见下；测试存在不代表当前已通过

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from __future__ import annotations
from alpha101_parameters import mining_parameter_specs
from betalens.factor.mining_cache import CacheRequest, MiningCache
from betalens.factor.mining_optuna import create_coarse_study, create_fine_grid_study, detect_boundary_pressure, expand_parameter_specs, generate_coarse_candidates, generate_fine_candidates, generate_perturbation_candidates, seed_study_with_results, suggest_params, tell_trial
from pathlib import Path
from types import SimpleNamespace
import betalens.factor.mining as mining
import betalens.factor.mining_audit as mining_audit
import betalens.factor.mining_optuna as mining_optuna
import json
import numpy as np
import pandas as pd
import pytest
import sqlite3
import sys
import types
import yaml
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [test_alpha101_compact_factor_list_expands_automatic_configs](../../../tests/test_factor_mining.py#L33) | test_alpha101_compact_factor_list_expands_automatic_configs(monkeypatch) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_heatmap_uses_all_parameter_pairs_and_mean_aggregation](../../../tests/test_factor_mining.py#L53) | test_heatmap_uses_all_parameter_pairs_and_mean_aggregation(tmp_path) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_optuna_search_supports_log_and_composite_categories](../../../tests/test_factor_mining.py#L114) | test_optuna_search_supports_log_and_composite_categories() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_qmc_wide_search_boundary_expansion_and_perturbations_are_bounded](../../../tests/test_factor_mining.py#L156) | test_qmc_wide_search_boundary_expansion_and_perturbations_are_bounded() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_tpe_with_all_invalid_previous_results_does_not_raise_keyerror](../../../tests/test_factor_mining.py#L210) | test_tpe_with_all_invalid_previous_results_does_not_raise_keyerror() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_tpe_with_all_invalid_previous_results_does_not_raise_keyerror.Factor](../../../tests/test_factor_mining.py#L213) | class Factor() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [test_tpe_with_all_invalid_previous_results_does_not_raise_keyerror.Factor.compute](../../../tests/test_factor_mining.py#L219) | compute(**kwargs) | 无返回注解；return: kwargs['x'] | 无 docstring，需阅读函数体 |
| [test_alpha101_auto_space_uses_multiplier_dimensions_and_type_limits](../../../tests/test_factor_mining.py#L238) | test_alpha101_auto_space_uses_multiplier_dimensions_and_type_limits(monkeypatch) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_cache_open_or_build_and_slice](../../../tests/test_factor_mining.py#L262) | test_cache_open_or_build_and_slice(tmp_path) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_cache_open_or_build_and_slice.builder](../../../tests/test_factor_mining.py#L267) | builder() | 无返回注解；return: {'inputs': {'x': values}, 'price': values, 'execution_price': values, 'trade_status': pd.DataFrame(1, index=index, columns=['A']), 'industry_by_scheme': {}, 'pit': {day.date(): {'A'} for day in index}, 'universe': ['A'], 'metadata': {'version': 'test'}} | 无 docstring，需阅读函数体 |
| [_synthetic_data](../../../tests/test_factor_mining.py#L296) | _synthetic_data() -&gt; mining.MiningData | mining.MiningData | 无 docstring，需阅读函数体 |
| [test_execution_modes_compute_once_and_isolate_windows](../../../tests/test_factor_mining.py#L308) | test_execution_modes_compute_once_and_isolate_windows(monkeypatch) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_execution_modes_compute_once_and_isolate_windows.Factor](../../../tests/test_factor_mining.py#L316) | class Factor() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [test_execution_modes_compute_once_and_isolate_windows.Factor.compute](../../../tests/test_factor_mining.py#L321) | compute(self, **kwargs) | 无返回注解；return: kwargs['x'] | 无 docstring，需阅读函数体 |
| [test_execution_modes_compute_once_and_isolate_windows.vector_nav](../../../tests/test_factor_mining.py#L335) | vector_nav(weights, price) | 无返回注解；return: pd.Series(range(1, len(price) + 1), index=price.index, dtype=float) | 无 docstring，需阅读函数体 |
| [test_execution_modes_compute_once_and_isolate_windows.fit_window](../../../tests/test_factor_mining.py#L355) | fit_window(window_data, params, window, context) | 无返回注解；return: pd.DataFrame({'A': 1.0}, index=dates) | 无 docstring，需阅读函数体 |
| [test_execution_modes_compute_once_and_isolate_windows.rolling_factory](../../../tests/test_factor_mining.py#L362) | rolling_factory(params) | 无返回注解；return: value | 无 docstring，需阅读函数体 |
| [test_precomputed_window_transform_reuses_factor_and_retests_each_window](../../../tests/test_factor_mining.py#L382) | test_precomputed_window_transform_reuses_factor_and_retests_each_window(monkeypatch) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_precomputed_window_transform_reuses_factor_and_retests_each_window.Factor](../../../tests/test_factor_mining.py#L390) | class Factor() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [test_precomputed_window_transform_reuses_factor_and_retests_each_window.Factor.compute](../../../tests/test_factor_mining.py#L395) | compute(self, **kwargs) | 无返回注解；return: kwargs['x'] | 无 docstring，需阅读函数体 |
| [test_precomputed_window_transform_reuses_factor_and_retests_each_window.transform](../../../tests/test_factor_mining.py#L399) | transform(weights, window, context) | 无返回注解；return: weights | 无 docstring，需阅读函数体 |
| [test_precomputed_window_transform_reuses_factor_and_retests_each_window.vector_nav](../../../tests/test_factor_mining.py#L414) | vector_nav(weights, price) | 无返回注解；return: pd.Series(range(1, len(price) + 1), index=price.index, dtype=float) | 无 docstring，需阅读函数体 |
| [test_legacy_mining_section_has_migration_error](../../../tests/test_factor_mining.py#L432) | test_legacy_mining_section_has_migration_error(tmp_path) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_configuration_validation_rejects_unknown_fields_and_windows](../../../tests/test_factor_mining.py#L439) | test_configuration_validation_rejects_unknown_fields_and_windows() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_summary_supports_configured_metric_aggregation](../../../tests/test_factor_mining.py#L448) | test_summary_supports_configured_metric_aggregation() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_exact_engine_uses_preloaded_frames](../../../tests/test_factor_mining.py#L463) | test_exact_engine_uses_preloaded_frames() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_run_logging_is_live_and_persisted](../../../tests/test_factor_mining.py#L496) | test_run_logging_is_live_and_persisted(tmp_path, monkeypatch, capsys) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_run_logging_is_live_and_persisted.Factor](../../../tests/test_factor_mining.py#L499) | class Factor() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [test_run_logging_is_live_and_persisted.Factor.compute](../../../tests/test_factor_mining.py#L505) | compute(**kwargs) | 无返回注解；return: kwargs['x'] | 无 docstring，需阅读函数体 |
| [test_multistage_search_runs_qmc_tpe_expansion_grid_and_stability](../../../tests/test_factor_mining.py#L628) | test_multistage_search_runs_qmc_tpe_expansion_grid_and_stability(tmp_path, monkeypatch) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_multistage_search_runs_qmc_tpe_expansion_grid_and_stability.Factor](../../../tests/test_factor_mining.py#L631) | class Factor() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [test_multistage_search_runs_qmc_tpe_expansion_grid_and_stability.Factor.compute](../../../tests/test_factor_mining.py#L637) | compute(**kwargs) | 无返回注解；return: kwargs['x'] | 无 docstring，需阅读函数体 |
| [test_multi_factor_launch_creates_isolated_task_directories](../../../tests/test_factor_mining.py#L711) | test_multi_factor_launch_creates_isolated_task_directories(tmp_path, monkeypatch) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_multi_factor_launch_creates_isolated_task_directories.fake_run_factor](../../../tests/test_factor_mining.py#L743) | fake_run_factor(task, factor_id, *args, **kwargs) | 无返回注解；return: mining.FactorMiningResult(factor_id=factor_id, run_id=task.run_id, run_dir=task.run_dir, status='complete') | 无 docstring，需阅读函数体 |
| [test_legacy_performance_options_are_rejected](../../../tests/test_factor_mining.py#L761) | test_legacy_performance_options_are_rejected(tmp_path) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [test_failed_run_updates_manifest_and_audit_log](../../../tests/test_factor_mining.py#L781) | test_failed_run_updates_manifest_and_audit_log(tmp_path) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |

