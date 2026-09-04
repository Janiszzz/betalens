# core_robust：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-82993d6b018a"></a>
## betalens/robust/__init__.py

[打开源码](../../../betalens/robust/__init__.py) · 4 行 · 说明来源：文件族规则

- **作用**：包导出/包标识
- **输入**：import 请求
- **输出**：模块导出与符号；见静态 imports
- **副作用/维护重点**：初始化可能导入子模块；__all__ 与真实导出需结合源码阅读

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .robust import RobustTest
```

<a id="file-cd7713f874f1"></a>
## betalens/robust/newrobust.py

[打开源码](../../../betalens/robust/newrobust.py) · 229 行 · 说明来源：人工文件说明

- **作用**：另一版 Lucky Factors 实现
- **输入**：见本文件 RobustTest 签名
- **输出**：本实现的检验对象及结果
- **副作用/维护重点**：包 __init__ 未导出这一版本，不能按文件名认定已替换旧版

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
import concurrent.futures
import numpy as np
import pandas as pd
import statsmodels.api as sm
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [RobustTest](../../../betalens/robust/newrobust.py#L11) | class RobustTest() | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [RobustTest.__init__](../../../betalens/robust/newrobust.py#L12) | __init__(self, target: pd.Series, factors: pd.DataFrame) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [RobustTest._orthogonalize](../../../betalens/robust/newrobust.py#L20) | _orthogonalize(self, baseline_X=None) | 无返回注解；return: (self._OX, self._T) | 无 docstring，需阅读函数体 |
| [RobustTest._bootstrap_resample](../../../betalens/robust/newrobust.py#L38) | _bootstrap_resample(self, data) | 无返回注解；return: data.iloc[indices, :] | 无 docstring，需阅读函数体 |
| [RobustTest._max_statistic](../../../betalens/robust/newrobust.py#L43) | _max_statistic(self, data) | 无返回注解；return: np.max(t_stats) | 无 docstring，需阅读函数体 |
| [RobustTest._panel_regression](../../../betalens/robust/newrobust.py#L53) | _panel_regression(self, X, y) | 无返回注解；return: (model.params, model.resid, model.tvalues, model.params) | 无 docstring，需阅读函数体 |
| [RobustTest._fake_fund](../../../betalens/robust/newrobust.py#L58) | _fake_fund(self, X, B, OX) | 无返回注解；return: fake_y | 无 docstring，需阅读函数体 |
| [RobustTest._bootstrap_once](../../../betalens/robust/newrobust.py#L66) | _bootstrap_once(self, n_bootstraps=1000) | 无返回注解；return: (eff_factors, modified_p) | 无 docstring，需阅读函数体 |
| [RobustTest.incremental_test](../../../betalens/robust/newrobust.py#L83) | incremental_test(self, n_bootstraps=1000) -&gt; pd.DataFrame | pd.DataFrame | 因子增量检验，对应旧版 work() 迭代流程: 1. neu(): 正交化因子，计算单因子t统计量 2. bootstrap_once(): Bootstrap得到修正p值 3. 识别显著因子(p&lt;0.1)与非显著因子 4. 若收敛则返回；否则用显著因子回归y取残差，剔除显著因子后继续 |
| [RobustTest.alpha_test](../../../betalens/robust/newrobust.py#L142) | alpha_test(self, n_bootstraps=1000) -&gt; pd.DataFrame | pd.DataFrame | Alpha显著性检验，对应旧版 bootstrap_fake_fund 流程: 1. panel(): 多因子回归得到系数B、残差OX、t值T 2. fake_fund(): 用B重构+残差Bootstrap生成虚拟基金 3. 对虚拟基金重复回归，构建alpha的t分布 4. 计算真实alpha的修正p值 |
| [RobustTest.rolling_test](../../../betalens/robust/newrobust.py#L174) | rolling_test(self, interval='1Y', n_bootstraps=1000) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [RobustTest.segment_test](../../../betalens/robust/newrobust.py#L200) | segment_test(self, segments: list, n_bootstraps=1000) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |

<a id="file-1e086893dced"></a>
## betalens/robust/robust.py

[打开源码](../../../betalens/robust/robust.py) · 374 行 · 说明来源：人工文件说明

- **作用**：当前公开导出的 RobustTest 与旧辅助工具
- **输入**：fund Series、factor DataFrame、重采样次数
- **输出**：OX、t统计、修正概率等
- **副作用/维护重点**：滞后一期、联合删缺失和线程重采样；旧 work 入口需审查

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from datetime import datetime
from sympy.codegen.cfunctions import isnan
import concurrent.futures
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [RobustTest](../../../betalens/robust/robust.py#L15) | class RobustTest(object) | 类定义；构造/属性见方法与字段 | 无 docstring，需阅读函数体 |
| [RobustTest.__init__](../../../betalens/robust/robust.py#L16) | __init__(self, fund, factor) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [RobustTest.create_sample_dataframes](../../../betalens/robust/robust.py#L28) | create_sample_dataframes() | 无返回注解；return: (asset_returns, factor_values) | 无 docstring，需阅读函数体 |
| [RobustTest.neu](../../../betalens/robust/robust.py#L51) | neu(self) | 无返回注解；return: (self.OX, self.T) | 无 docstring，需阅读函数体 |
| [RobustTest.bootstrap_resample](../../../betalens/robust/robust.py#L67) | bootstrap_resample(self, data) | 无返回注解；return: bootstrapped_data | 无 docstring，需阅读函数体 |
| [RobustTest.max_statistic](../../../betalens/robust/robust.py#L72) | max_statistic(self, data) | 无返回注解；return: np.max(t_statistics) | 无 docstring，需阅读函数体 |
| [RobustTest.bootstrap_once](../../../betalens/robust/robust.py#L85) | bootstrap_once(self, n_bootstraps=1000) | 无返回注解；return: (eff_fct_name, modifd_P, max_statistic_pdf) | 无 docstring，需阅读函数体 |
| [RobustTest.work](../../../betalens/robust/robust.py#L104) | work(self) | 无返回注解；return: modifd_P | 无 docstring，需阅读函数体 |
| [panel](../../../betalens/robust/robust.py#L162) | panel(X, y) | 无返回注解；return: (B, OX, T, df_params.T) | 无 docstring，需阅读函数体 |
| [fake_fund](../../../betalens/robust/robust.py#L177) | fake_fund(X, B, OX) | 无返回注解；return: fake_y | 无 docstring，需阅读函数体 |
| [bootstrap_fake_fund](../../../betalens/robust/robust.py#L183) | bootstrap_fake_fund(X, B, OX, T, n_bootstraps=1000) | 无返回注解；return: (modifd_P['const'], max_statistic_pdf) | 无 docstring，需阅读函数体 |
| [work](../../../betalens/robust/robust.py#L229) | work(fund, fct) | 无返回注解；return: -1; pd.concat([modifd_P.reset_index(drop=True), df_params.reset_index(drop=True)], axis=1) | 无 docstring，需阅读函数体 |
| [parse_name_dates](../../../betalens/robust/robust.py#L244) | parse_name_dates(s) | 无返回注解；return: {'name': name_part, 'start_date': start_date, 'end_date': end_date} | 将字符串格式 '姓名(开始日期-结束日期)' 拆解为姓名和两个datetime对象 Args: s (str): 输入字符串，例如 '盛丰衍(20180711-20250101)' Returns: dict: 包含以下键的字典： - 'name': 提取的姓名 - 'start_date': 开始日期 datetime 对象 - 'end_date': 结束日期 datetime 对象 Raises: ValueError: 如果字符串格式不正确或无法解析日期 |
| [get_interval](../../../betalens/robust/robust.py#L281) | get_interval(df, start=None, end=None) | 无返回注解；return: df; df.loc[start:end]; df.loc[start:]; df.loc[:end]; df[mask] | 无 docstring，需阅读函数体 |
| [gen_date_pairs](../../../betalens/robust/robust.py#L331) | gen_date_pairs(start_time, end_time, interval='1Y') | 无返回注解；return: time_pairs | 无 docstring，需阅读函数体 |

