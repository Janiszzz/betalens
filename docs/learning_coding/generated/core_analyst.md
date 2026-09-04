# core_analyst：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-53b913cb1ee5"></a>
## betalens/analyst/__init__.py

[打开源码](../../../betalens/analyst/__init__.py) · 33 行 · 说明来源：文件族规则

- **作用**：Analyst模块 - 策略评价与报告工具
- **输入**：import 请求
- **输出**：模块导出与符号；见静态 imports
- **副作用/维护重点**：初始化可能导入子模块；__all__ 与真实导出需结合源码阅读

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from . import metrics, naming, plotting
from .analyst import Analyst, PortfolioAnalyzer, ReportExporter
```

<a id="file-4d1bd46c5884"></a>
## betalens/analyst/analyst.py

[打开源码](../../../betalens/analyst/analyst.py) · 694 行 · 说明来源：人工文件说明

- **作用**：评价门面及报告导出
- **输入**：回测实例/净值/Excel、可选基准
- **输出**：Analyst、汇总表、Excel/HTML 报告
- **副作用/维护重点**：基准代码可触发查库，报告会写文件；完整依赖需另验收

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from . import metrics as M
from . import plotting as P
from .naming import get_name_map, label
from betalens.datafeed import Datafeed
from betalens.factor.stats import calc_ic_from_factor_values
from betalens.factor.stats import summarize_ic
from datetime import datetime
from prettytable import PrettyTable
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_drawdown_interval](../../../betalens/analyst/analyst.py#L22) | _drawdown_interval(nav: pd.Series) -&gt; str &#124; None | str &#124; None | 无 docstring，需阅读函数体 |
| [_load_benchmark_nav_from_code](../../../betalens/analyst/analyst.py#L34) | _load_benchmark_nav_from_code(benchmark_code: str, start_date, end_date, metric: str, table_names: tuple[str, ...] &#124; list[str] &#124; None=None) -&gt; pd.Series &#124; None | pd.Series &#124; None | 无 docstring，需阅读函数体 |
| [PortfolioAnalyzer](../../../betalens/analyst/analyst.py#L77) | class PortfolioAnalyzer() | 类定义；构造/属性见方法与字段 | 投资组合分析器。 Args: nav_series: 净值序列（pd.Series, index=日期） risk_free_rate: 年化无风险利率 annualizer: 年化因子（日频 252） window: 滚动窗口默认值 weight: 调仓权重表（可选，换手/持仓类指标需要） daily_position_value: 日频持仓金额表（可选，权重堆积/面积图需要） daily_pnl: 日频损益表（可选，收益贡献分解需要） rebalance_log: 调仓记录表（可选，逐笔盈亏需要） benchmark: 基准净值 Series（可选） |
| [PortfolioAnalyzer.__init__](../../../betalens/analyst/analyst.py#L93) | __init__(self, nav_series, risk_free_rate=0.0, annualizer=252, window=30, weight=None, daily_position_value=None, daily_pnl=None, rebalance_log=None, benchmark=None, cost_price=None, factor_values=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [PortfolioAnalyzer.total_return](../../../betalens/analyst/analyst.py#L127) | total_return(self) | 无返回注解；return: M.total_return(self.nav) | 无 docstring，需阅读函数体 |
| [PortfolioAnalyzer.annualized_return](../../../betalens/analyst/analyst.py#L130) | annualized_return(self) | 无返回注解；return: M.annualized_return(self.nav, self.annualizer) | 无 docstring，需阅读函数体 |
| [PortfolioAnalyzer.annualized_volatility](../../../betalens/analyst/analyst.py#L133) | annualized_volatility(self) | 无返回注解；return: M.annualized_volatility(self.returns, self.annualizer) | 无 docstring，需阅读函数体 |
| [PortfolioAnalyzer.sharpe_ratio](../../../betalens/analyst/analyst.py#L136) | sharpe_ratio(self) | 无返回注解；return: M.sharpe_ratio(self.returns, self.risk_free_rate, self.annualizer) | 无 docstring，需阅读函数体 |
| [PortfolioAnalyzer.max_drawdown](../../../betalens/analyst/analyst.py#L139) | max_drawdown(self) | 无返回注解；return: M.max_drawdown(self.nav) | 无 docstring，需阅读函数体 |
| [PortfolioAnalyzer.calmar_ratio](../../../betalens/analyst/analyst.py#L142) | calmar_ratio(self) | 无返回注解；return: M.calmar_ratio(self.nav, self.annualizer) | 无 docstring，需阅读函数体 |
| [PortfolioAnalyzer.rolling_max_drawdown](../../../betalens/analyst/analyst.py#L145) | rolling_max_drawdown(self) | 无返回注解；return: M.rolling_max_drawdown(self.nav, self.window) | 无 docstring，需阅读函数体 |
| [PortfolioAnalyzer.rolling_win_rate](../../../betalens/analyst/analyst.py#L148) | rolling_win_rate(self) | 无返回注解；return: M.rolling_win_rate(self.returns, self.window) | 无 docstring，需阅读函数体 |
| [PortfolioAnalyzer.summary](../../../betalens/analyst/analyst.py#L153) | summary(self) -&gt; dict | dict | 返回全部标量指标，按类别分组的扁平 dict |
| [PortfolioAnalyzer.summary_grouped](../../../betalens/analyst/analyst.py#L230) | summary_grouped(self) -&gt; dict | dict | 按类别分组的指标 dict（用于分块展示） |
| [_fmt](../../../betalens/analyst/analyst.py#L251) | _fmt(v) | 无返回注解；return: str(int(v)); '-'; f'{v:.4f}'; str(v) | 统一格式化：百分比类用 %，比率类保留 4 位 |
| [_fmt_pct](../../../betalens/analyst/analyst.py#L262) | _fmt_pct(v) | 无返回注解；return: f'{v:.2%}'; _fmt(v) | 无 docstring，需阅读函数体 |
| [ReportExporter](../../../betalens/analyst/analyst.py#L276) | class ReportExporter() | 类定义；构造/属性见方法与字段 | 报告导出（兼容旧接口：分年度 / 时段 / 基准对比，CLI + Excel） |
| [ReportExporter.__init__](../../../betalens/analyst/analyst.py#L279) | __init__(self, analyzer, benchmark_analyzer=None, start_date=None, end_date=None) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [ReportExporter._format_percentage](../../../betalens/analyst/analyst.py#L292) | _format_percentage(self, value) | 无返回注解；return: f'{value:.2%}' if isinstance(value, (float, np.float64)) else str(value) | 无 docstring，需阅读函数体 |
| [ReportExporter.generate_annual_report](../../../betalens/analyst/analyst.py#L295) | generate_annual_report(self, excel_path=None) | 无返回注解；return: df | 分年度绩效报告 |
| [ReportExporter.generate_custom_report](../../../betalens/analyst/analyst.py#L316) | generate_custom_report(self, start_date, end_date, excel_path=None) | 无返回注解；return: None; df | 指定时段绩效报告 |
| [ReportExporter.generate_benchmark_report](../../../betalens/analyst/analyst.py#L335) | generate_benchmark_report(self, excel_path=None) | 无返回注解；return: df | 基准对比报告 |
| [ReportExporter._print_cli_table](../../../betalens/analyst/analyst.py#L352) | _print_cli_table(self, data_df, title='') | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [ReportExporter._export_to_excel](../../../betalens/analyst/analyst.py#L360) | _export_to_excel(self, data_df, file_path, sheet_name) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [Analyst](../../../betalens/analyst/analyst.py#L369) | class Analyst() | 类定义；构造/属性见方法与字段 | 策略评价一键门面。 用法： a = Analyst.from_backtest(bt, benchmark=hs300_bt) a.report() # CLI 打印全部指标表 a.report(to_excel='r.xlsx') # 同时导出 Excel a.report(to_html='r.html') # 导出交互 HTML（plotly） figs = a.plots() # {名称: PNG bytes}，供 st.image ifigs = a.interactive_plots() # {名称: plotly Figure}，供 dashboard |
| [Analyst.__init__](../../../betalens/analyst/analyst.py#L382) | __init__(self, analyzer: PortfolioAnalyzer, name: str='组合') | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [Analyst.from_backtest](../../../betalens/analyst/analyst.py#L388) | from_backtest(cls, bt, benchmark=None, risk_free_rate=0.0, annualizer=252, window=30, name='组合', benchmark_code=None, benchmark_metric='收盘价(元)', benchmark_table_names=None, factor_values=None) | 无返回注解；return: cls(an, name=name) | 从回测实例构建。自动抽取 nav / actual_weight / daily_position_value / daily_pnl / rebalance_log。 Args: bt: BacktestBase 实例（须已完成回测，含 nav 等属性） benchmark: 基准 nav Series 或另一个 bt 实例 benchmark_code: 基准代码；仅在 benchmark 未传入时按代码查库构造基准净值 benchmark_metric: 基准价格字段 factor_values: 因子长表；用于结合回测价格计算 IC name: 组合名称（用于报告标题） |
| [Analyst.from_excel](../../../betalens/analyst/analyst.py#L436) | from_excel(cls, filepath, benchmark=None, name='组合', **kwargs) | 无返回注解；return: cls(an, name=name) | 从 bt.dump_to_excel 导出的 xlsx 读回构建（dashboard 上传用） |
| [Analyst.from_excel._read](../../../betalens/analyst/analyst.py#L440) | _read(sheet, index_col=0) | 无返回注解；return: None; df | 无 docstring，需阅读函数体 |
| [Analyst.name_map](../../../betalens/analyst/analyst.py#L470) | name_map(self) -&gt; dict | dict | 无 docstring，需阅读函数体 |
| [Analyst.summary_df](../../../betalens/analyst/analyst.py#L482) | summary_df(self) -&gt; pd.DataFrame | pd.DataFrame | 全部指标的明细表（含格式化展示列） |
| [Analyst.top_holdings_df](../../../betalens/analyst/analyst.py#L491) | top_holdings_df(self, top=10) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [Analyst.contribution_df](../../../betalens/analyst/analyst.py#L498) | contribution_df(self, top=15) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [Analyst.trade_pnl_df](../../../betalens/analyst/analyst.py#L505) | trade_pnl_df(self) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [Analyst.monthly_table](../../../betalens/analyst/analyst.py#L513) | monthly_table(self) -&gt; pd.DataFrame | pd.DataFrame | 无 docstring，需阅读函数体 |
| [Analyst.print_report](../../../betalens/analyst/analyst.py#L518) | print_report(self) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 命令行打印全部指标表格（按类别分组） |
| [Analyst.plots](../../../betalens/analyst/analyst.py#L555) | plots(self) -&gt; dict | dict | 返回 {名称: PNG bytes}，供 st.image / 嵌入 |
| [Analyst.interactive_plots](../../../betalens/analyst/analyst.py#L581) | interactive_plots(self) -&gt; dict | dict | 返回 {名称: plotly Figure}，供 dashboard / HTML |
| [Analyst.to_excel](../../../betalens/analyst/analyst.py#L604) | to_excel(self, filepath: str) -&gt; str | str | 导出 Excel：指标汇总 + 各明细表 |
| [Analyst.to_html](../../../betalens/analyst/analyst.py#L624) | to_html(self, filepath: str) -&gt; str | str | 导出独立 HTML 报告（含 plotly 交互图 + 指标表） |
| [Analyst.report](../../../betalens/analyst/analyst.py#L652) | report(self, to_excel: str=None, to_html: str=None, show_plots: bool=False) | 无返回注解；return: self.an.summary() | 一键报告：CLI 打印 + 可选导出 Excel / HTML。 Args: to_excel: Excel 输出路径（None 不导出） to_html: HTML 输出路径（None 不导出） show_plots: 是否在 CLI 环境弹出静态图（matplotlib） |

<a id="file-841e0ef10030"></a>
## betalens/analyst/metrics.py

[打开源码](../../../betalens/analyst/metrics.py) · 513 行 · 说明来源：人工文件说明

- **作用**：纯绩效指标和收益明细计算
- **输入**：nav/returns/weight/持仓/交易日志
- **输出**：标量、Series、DataFrame 或字典
- **副作用/维护重点**：纯计算；年化频率、回撤符号和净值分母需核对

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
import numpy as np
import pandas as pd
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_to_returns](../../../betalens/analyst/metrics.py#L20) | _to_returns(nav: pd.Series) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [_drawdown_series](../../../betalens/analyst/metrics.py#L24) | _drawdown_series(nav: pd.Series) -&gt; pd.Series | pd.Series | 回撤序列（正数表示回撤幅度） |
| [excess_nav](../../../betalens/analyst/metrics.py#L31) | excess_nav(returns: pd.Series, bench_returns: pd.Series) -&gt; pd.Series | pd.Series | 由策略日收益 - 基准日收益构造超额净值。 |
| [total_return](../../../betalens/analyst/metrics.py#L47) | total_return(nav: pd.Series) -&gt; float | float | 无 docstring，需阅读函数体 |
| [annualized_return](../../../betalens/analyst/metrics.py#L52) | annualized_return(nav: pd.Series, annualizer: int=252) -&gt; float | float | 几何年化收益率 |
| [annualized_volatility](../../../betalens/analyst/metrics.py#L61) | annualized_volatility(returns: pd.Series, annualizer: int=252) -&gt; float | float | 无 docstring，需阅读函数体 |
| [sharpe_ratio](../../../betalens/analyst/metrics.py#L65) | sharpe_ratio(returns: pd.Series, rf: float=0.0, annualizer: int=252) -&gt; float | float | 无 docstring，需阅读函数体 |
| [max_drawdown](../../../betalens/analyst/metrics.py#L72) | max_drawdown(nav: pd.Series) -&gt; float | float | 无 docstring，需阅读函数体 |
| [calmar_ratio](../../../betalens/analyst/metrics.py#L76) | calmar_ratio(nav: pd.Series, annualizer: int=252) -&gt; float | float | 无 docstring，需阅读函数体 |
| [ulcer_index](../../../betalens/analyst/metrics.py#L85) | ulcer_index(nav: pd.Series) -&gt; float | float | 溃疡指数：回撤的均方根，惩罚深而久的回撤 |
| [martin_ratio](../../../betalens/analyst/metrics.py#L91) | martin_ratio(nav: pd.Series, rf: float=0.0, annualizer: int=252) -&gt; float | float | Martin 比率（UPI）= 年化超额收益 / 溃疡指数 |
| [pain_index](../../../betalens/analyst/metrics.py#L99) | pain_index(nav: pd.Series) -&gt; float | float | 痛苦指数：平均回撤深度 |
| [pain_ratio](../../../betalens/analyst/metrics.py#L104) | pain_ratio(nav: pd.Series, rf: float=0.0, annualizer: int=252) -&gt; float | float | 无 docstring，需阅读函数体 |
| [max_drawdown_duration](../../../betalens/analyst/metrics.py#L111) | max_drawdown_duration(nav: pd.Series) -&gt; int | int | 最长回撤持续期（距前高的最长天数，按数据点计） |
| [downside_deviation](../../../betalens/analyst/metrics.py#L125) | downside_deviation(returns: pd.Series, mar: float=0.0, annualizer: int=252) -&gt; float | float | 下行偏差（年化），mar 为最低可接受日收益 |
| [sortino_ratio](../../../betalens/analyst/metrics.py#L133) | sortino_ratio(returns: pd.Series, rf: float=0.0, annualizer: int=252) -&gt; float | float | 无 docstring，需阅读函数体 |
| [value_at_risk](../../../betalens/analyst/metrics.py#L141) | value_at_risk(returns: pd.Series, level: float=0.05) -&gt; float | float | 历史法 VaR，返回正数表示潜在损失幅度 |
| [conditional_var](../../../betalens/analyst/metrics.py#L148) | conditional_var(returns: pd.Series, level: float=0.05) -&gt; float | float | 历史法 CVaR（期望损失） |
| [skewness](../../../betalens/analyst/metrics.py#L159) | skewness(returns: pd.Series) -&gt; float | float | 无 docstring，需阅读函数体 |
| [kurtosis](../../../betalens/analyst/metrics.py#L163) | kurtosis(returns: pd.Series) -&gt; float | float | 无 docstring，需阅读函数体 |
| [rolling_win_rate](../../../betalens/analyst/metrics.py#L169) | rolling_win_rate(returns: pd.Series, window: int=30) -&gt; pd.Series | pd.Series | 滚动胜率：窗口内日收益&gt;0 的占比 |
| [rolling_sharpe](../../../betalens/analyst/metrics.py#L174) | rolling_sharpe(returns: pd.Series, window: int=60, rf: float=0.0, annualizer: int=252) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [rolling_max_drawdown](../../../betalens/analyst/metrics.py#L182) | rolling_max_drawdown(nav: pd.Series, window: int=60) -&gt; pd.Series | pd.Series | 无 docstring，需阅读函数体 |
| [turnover](../../../betalens/analyst/metrics.py#L191) | turnover(weight: pd.DataFrame, annualizer: int=252, include_cash: bool=False) -&gt; dict | dict | 换手率。逐期单边换手 = 0.5 * Σ&#124;w_t - w_{t-1}&#124;。 Returns: dict: 含 per_period(Series)、avg_oneway、avg_twoway、annualized |
| [_avg_period_days](../../../betalens/analyst/metrics.py#L217) | _avg_period_days(index: pd.DatetimeIndex) -&gt; float | float | 无 docstring，需阅读函数体 |
| [top_holdings](../../../betalens/analyst/metrics.py#L224) | top_holdings(weight: pd.DataFrame, top: int=10) -&gt; pd.DataFrame | pd.DataFrame | 最频繁持仓：按出现频率（权重非零的期数占比）+ 平均权重排序。 Returns: DataFrame: index=code, 列 freq(出现频率)、avg_weight(平均权重)、 max_weight(最大权重) |
| [weight_hhi](../../../betalens/analyst/metrics.py#L243) | weight_hhi(weight: pd.DataFrame) -&gt; pd.Series | pd.Series | 赫芬达尔指数（权重堆积/集中度），逐期 Σw²，越高越集中 |
| [top_n_concentration](../../../betalens/analyst/metrics.py#L251) | top_n_concentration(weight: pd.DataFrame, n: int=5) -&gt; pd.Series | pd.Series | 逐期前 N 大持仓权重之和 |
| [avg_holdings_count](../../../betalens/analyst/metrics.py#L259) | avg_holdings_count(weight: pd.DataFrame) -&gt; dict | dict | 持仓标的个数（逐期 + 平均） |
| [holding_period](../../../betalens/analyst/metrics.py#L268) | holding_period(weight: pd.DataFrame) -&gt; float | float | 平均持仓寿命（标的从建仓到清仓的平均持有期数） |
| [return_contribution](../../../betalens/analyst/metrics.py#L291) | return_contribution(daily_pnl: pd.DataFrame, top: int=15) -&gt; pd.DataFrame | pd.DataFrame | 收益贡献分解：各标的累计损益及占比。 Args: daily_pnl: 日频损益表（index=日, columns=code+cash） Returns: DataFrame: index=code, 列 pnl(累计损益)、contribution(占总损益比例) |
| [trade_pnl](../../../betalens/analyst/metrics.py#L308) | trade_pnl(rebalance_log: pd.DataFrame) -&gt; pd.DataFrame | pd.DataFrame | 逐标的盈亏配对统计（基于调仓记录的成交金额变化近似）。 对每个 code 按调仓日排序，相邻成交金额的差视为已实现/浮动盈亏的代理， 汇总成交次数、胜率、平均盈亏。 Returns: DataFrame: index=code, 列 trades(成交次数)、win_rate(盈利次数占比)、 avg_value(平均成交金额)、total_value(累计成交金额) |
| [match_trade_pairs](../../../betalens/analyst/metrics.py#L338) | match_trade_pairs(rebalance_log: pd.DataFrame &#124; None) -&gt; pd.DataFrame | pd.DataFrame | 按证券代码将买卖记录 FIFO 配对并计算单笔收益率。 |
| [annual_trade_performance](../../../betalens/analyst/metrics.py#L370) | annual_trade_performance(trade_pairs: pd.DataFrame &#124; None) -&gt; pd.DataFrame | pd.DataFrame | 按平仓年份汇总单笔平均收益、胜率和交易次数。 |
| [group_nav](../../../betalens/analyst/metrics.py#L393) | group_nav(cost_ret: pd.DataFrame &#124; None, factor_values: pd.DataFrame &#124; None, n_quantiles: int) -&gt; pd.DataFrame | pd.DataFrame | 由调仓区间收益与信号日分组成员计算各组等权净值。 |
| [beta](../../../betalens/analyst/metrics.py#L445) | beta(returns: pd.Series, bench_returns: pd.Series) -&gt; float | float | 无 docstring，需阅读函数体 |
| [alpha](../../../betalens/analyst/metrics.py#L453) | alpha(returns: pd.Series, bench_returns: pd.Series, rf: float=0.0, annualizer: int=252) -&gt; float | float | 无 docstring，需阅读函数体 |
| [tracking_error](../../../betalens/analyst/metrics.py#L463) | tracking_error(returns: pd.Series, bench_returns: pd.Series, annualizer: int=252) -&gt; float | float | 无 docstring，需阅读函数体 |
| [information_ratio](../../../betalens/analyst/metrics.py#L469) | information_ratio(returns: pd.Series, bench_returns: pd.Series, annualizer: int=252) -&gt; float | float | 无 docstring，需阅读函数体 |
| [win_rate_vs_benchmark](../../../betalens/analyst/metrics.py#L478) | win_rate_vs_benchmark(returns: pd.Series, bench_returns: pd.Series) -&gt; float | float | 无 docstring，需阅读函数体 |
| [profit_loss_counts](../../../betalens/analyst/metrics.py#L483) | profit_loss_counts(daily_pnl_total: pd.Series) -&gt; tuple[int, int] | tuple[int, int] | 无 docstring，需阅读函数体 |
| [profit_loss_ratio](../../../betalens/analyst/metrics.py#L488) | profit_loss_ratio(daily_pnl_total: pd.Series) -&gt; float | float | 无 docstring，需阅读函数体 |
| [monthly_returns_table](../../../betalens/analyst/metrics.py#L499) | monthly_returns_table(nav: pd.Series) -&gt; pd.DataFrame | pd.DataFrame | 月度收益矩阵：index=年, columns=月(1-12)+全年 |

<a id="file-1cc403d0bbed"></a>
## betalens/analyst/naming.py

[打开源码](../../../betalens/analyst/naming.py) · 123 行 · 说明来源：人工文件说明

- **作用**：证券代码名称映射和缓存
- **输入**：证券代码集合
- **输出**：名称字典/标签/重命名表
- **副作用/维护重点**：尝试查库，失败回退代码；缓存可能需清理

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from betalens import Datafeed
import pandas as pd
import warnings
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [get_name_map](../../../betalens/analyst/naming.py#L17) | get_name_map(codes, tables=None) -&gt; dict | dict | 查库取 code→中文名映射。 Args: codes: 代码可迭代对象 tables: 候选表列表，None 用默认（股/指/基/债） Returns: dict: {code: name}，查不到的 code 不在结果中 |
| [label](../../../betalens/analyst/naming.py#L82) | label(code: str, name_map: dict=None) -&gt; str | str | 生成展示标签：「中文名(代码)」，无名称时回落为代码本身。 Example: &gt;&gt;&gt; label('000300.SH', {'000300.SH': '沪深300'}) '沪深300(000300.SH)' |
| [rename_codes](../../../betalens/analyst/naming.py#L97) | rename_codes(obj, name_map: dict=None, axis: int=1) | 无返回注解；return: obj.rename(columns=mapping) if axis == 1 else obj.rename(index=mapping); obj.rename(index=mapping) | 把 DataFrame/Series 的 code 索引或列名替换为「中文名(代码)」标签。 Args: obj: DataFrame 或 Series name_map: code→name 映射，None 时自动查库 axis: 1=替换列名（DataFrame），0=替换索引 Returns: 替换标签后的副本 |
| [clear_cache](../../../betalens/analyst/naming.py#L121) | clear_cache() | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 清空名称缓存（测试或数据更新后调用） |

<a id="file-0e3196c1cb60"></a>
## betalens/analyst/plotting.py

[打开源码](../../../betalens/analyst/plotting.py) · 459 行 · 说明来源：人工文件说明

- **作用**：静态和交互图形
- **输入**：净值、指标、权重及损益表
- **输出**：matplotlib/Plotly 对象、图片字节、HTML 片段
- **副作用/维护重点**：图形依赖；保持坐标、单位与空值语义

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from . import metrics as M
from .naming import label
from matplotlib.lines import Line2D
import io as _io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [_fig_to_bytes](../../../betalens/analyst/plotting.py#L26) | _fig_to_bytes(fig) -&gt; bytes | bytes | 无 docstring，需阅读函数体 |
| [plot_nav](../../../betalens/analyst/plotting.py#L34) | plot_nav(nav: pd.Series, benchmark: pd.Series=None, title: str='净值曲线') -&gt; bytes | bytes | 净值曲线图（可叠加基准） |
| [plot_drawdown](../../../betalens/analyst/plotting.py#L51) | plot_drawdown(nav: pd.Series, title: str='回撤曲线') -&gt; bytes | bytes | 回撤面积图 |
| [plot_rolling_metric](../../../betalens/analyst/plotting.py#L63) | plot_rolling_metric(series: pd.Series, title: str, ylabel: str, color: str='#2ca02c') -&gt; bytes | bytes | 通用滚动指标折线图（滚动胜率/夏普/回撤等） |
| [plot_contribution_bar](../../../betalens/analyst/plotting.py#L74) | plot_contribution_bar(contrib: pd.DataFrame, name_map: dict=None, title: str='收益贡献 Top') -&gt; bytes | bytes | 收益贡献柱状图（横向，按 pnl 排序，中文名标签） |
| [plot_weight_concentration](../../../betalens/analyst/plotting.py#L89) | plot_weight_concentration(hhi: pd.Series, count: pd.Series, weight: 'pd.DataFrame &#124; None'=None, name_map: 'dict &#124; None'=None, top: int=8, max_codes: int=15, max_periods: int=12, title: str='权重堆积与持仓数') -&gt; bytes | bytes | HHI 集中度 + 持仓数双轴图；传入 weight 时在下方附时序持仓权重表一览 |
| [plot_weight_concentration._draw_dual](../../../betalens/analyst/plotting.py#L97) | _draw_dual(ax1) | 无显式 return/返回注解；可能以属性、副作用或 yield 输出 | 无 docstring，需阅读函数体 |
| [plot_monthly_heatmap](../../../betalens/analyst/plotting.py#L152) | plot_monthly_heatmap(table: pd.DataFrame, title: str='月度收益热力表') -&gt; bytes | bytes | 月度收益热力图 |
| [plot_group_nav](../../../betalens/analyst/plotting.py#L178) | plot_group_nav(group_nav: pd.DataFrame, title: str='分组净值曲线', n_quantiles: int=10) -&gt; bytes | bytes | 十分组净值曲线（截面因子用）。 Args: group_nav: DataFrame, index=date, columns=G1..Gn（各组净值，从1出发） title: 图表标题 n_quantiles: 分组数（用于图例列数优化） Returns: PNG bytes |
| [plot_timing_nav_with_trades](../../../betalens/analyst/plotting.py#L214) | plot_timing_nav_with_trades(nav: pd.Series, trade_pairs: pd.DataFrame, title: str='净值曲线（含买卖点）') -&gt; bytes | bytes | 择时策略净值曲线 + 买卖点标注。 Args: nav: 净值序列（index=date, values=nav） trade_pairs: DataFrame，含 buy_date, sell_date, return 列 title: 图表标题 Returns: PNG bytes |
| [plot_annual_trade_performance](../../../betalens/analyst/plotting.py#L266) | plot_annual_trade_performance(trade_pairs: pd.DataFrame, title: str='分年度交易表现') -&gt; bytes | bytes | 分年度柱状收益 + 胜率折线。 Args: trade_pairs: DataFrame，含 sell_date(Timestamp) 和 return(float) 列 title: 图表标题 Returns: PNG bytes |
| [_import_plotly](../../../betalens/analyst/plotting.py#L323) | _import_plotly() | 无返回注解；return: go | 无 docstring，需阅读函数体 |
| [ip_nav](../../../betalens/analyst/plotting.py#L336) | ip_nav(nav: pd.Series, benchmark: pd.Series=None, title: str='净值曲线') -&gt; 'object' | 'object' | 交互净值曲线（plotly） |
| [ip_drawdown](../../../betalens/analyst/plotting.py#L353) | ip_drawdown(nav: pd.Series, title: str='回撤曲线') -&gt; 'object' | 'object' | 无 docstring，需阅读函数体 |
| [ip_rolling](../../../betalens/analyst/plotting.py#L364) | ip_rolling(series: pd.Series, title: str, ylabel: str, color: str='#2ca02c') -&gt; 'object' | 'object' | 无 docstring，需阅读函数体 |
| [ip_contribution](../../../betalens/analyst/plotting.py#L375) | ip_contribution(contrib: pd.DataFrame, name_map: dict=None, title: str='收益贡献 Top') -&gt; 'object' | 'object' | 无 docstring，需阅读函数体 |
| [ip_weight_area](../../../betalens/analyst/plotting.py#L390) | ip_weight_area(daily_position_value: pd.DataFrame, name_map: dict=None, top: int=10, max_codes: int=25, title: str='持仓权重堆积') -&gt; 'object' | 'object' | 持仓权重堆积面积图。 选标逻辑（按标的，并集法）：取每个时点权重前 ''top'' 大标的的并集作为 显示标的——低换手时就是那固定十几只全显示；高换手时各期轮动的主力都能入选。 并集仍超过 ''max_codes'' 时，按各标的的峰值单期权重保留最重要的，其余与 未入选标的一并归入「其他」，避免色块过多显示不过来。 |
| [ip_monthly_heatmap](../../../betalens/analyst/plotting.py#L441) | ip_monthly_heatmap(table: pd.DataFrame, title: str='月度收益热力表') -&gt; 'object' | 'object' | 无 docstring，需阅读函数体 |
| [fig_to_html_div](../../../betalens/analyst/plotting.py#L457) | fig_to_html_div(fig) -&gt; str | str | plotly Figure → 可嵌入的 HTML div 片段（首图带 plotly.js CDN） |

