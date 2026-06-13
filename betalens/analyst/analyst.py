"""
策略评价模块

三层结构：
- PortfolioAnalyzer: 持有净值/持仓数据，组合 metrics 计算各类指标
- ReportExporter: 兼容旧接口的分年度/时段/基准报告（CLI + Excel）
- Analyst: 一键门面，from_backtest 直接吃回测实例，report() 串起全部输出

向后兼容：旧用法 PortfolioAnalyzer(nav_series) 仍可用，持仓类指标缺数据时返回 NaN。
"""
import numpy as np
import pandas as pd
from datetime import datetime

from prettytable import PrettyTable

from . import metrics as M
from . import plotting as P
from .naming import get_name_map, label


class PortfolioAnalyzer:
    """
    投资组合分析器。

    Args:
        nav_series: 净值序列（pd.Series, index=日期）
        risk_free_rate: 年化无风险利率
        annualizer: 年化因子（日频 252）
        window: 滚动窗口默认值
        weight: 调仓权重表（可选，换手/持仓类指标需要）
        daily_position_value: 日频持仓金额表（可选，权重堆积/面积图需要）
        daily_pnl: 日频损益表（可选，收益贡献分解需要）
        rebalance_log: 调仓记录表（可选，逐笔盈亏需要）
        benchmark: 基准净值 Series（可选）
    """

    def __init__(self, nav_series, risk_free_rate=0.0, annualizer=252, window=30,
                 weight=None, daily_position_value=None, daily_pnl=None,
                 rebalance_log=None, benchmark=None):
        self.nav = nav_series.sort_index()
        self.returns = self.nav.pct_change().dropna()
        self.risk_free_rate = risk_free_rate
        self.annualizer = annualizer
        self.window = window

        self.weight = weight
        self.daily_position_value = daily_position_value
        self.daily_pnl = daily_pnl
        self.rebalance_log = rebalance_log

        self.benchmark = None
        self.bench_returns = None
        if benchmark is not None:
            self.benchmark = benchmark.sort_index().reindex(self.nav.index).ffill()
            self.bench_returns = self.benchmark.pct_change().dropna()

    # ── 兼容旧接口的核心指标（委托 metrics）────────────────────────────────

    def total_return(self):
        return M.total_return(self.nav)

    def annualized_return(self):
        return M.annualized_return(self.nav, self.annualizer)

    def annualized_volatility(self):
        return M.annualized_volatility(self.returns, self.annualizer)

    def sharpe_ratio(self):
        return M.sharpe_ratio(self.returns, self.risk_free_rate, self.annualizer)

    def max_drawdown(self):
        return M.max_drawdown(self.nav)

    def calmar_ratio(self):
        return M.calmar_ratio(self.nav, self.annualizer)

    def rolling_max_drawdown(self):
        return M.rolling_max_drawdown(self.nav, self.window)

    def rolling_win_rate(self):
        return M.rolling_win_rate(self.returns, self.window)

    # ── 汇总：分组指标字典 ─────────────────────────────────────────────────

    def summary(self) -> dict:
        """返回全部标量指标，按类别分组的扁平 dict"""
        rf, ann = self.risk_free_rate, self.annualizer
        nav, ret = self.nav, self.returns
        out = {
            '累计收益': M.total_return(nav),
            '年化收益': M.annualized_return(nav, ann),
            '年化波动率': M.annualized_volatility(ret, ann),
            '夏普比率': M.sharpe_ratio(ret, rf, ann),
            '索提诺比率': M.sortino_ratio(ret, rf, ann),
            '最大回撤': M.max_drawdown(nav),
            '卡玛比率': M.calmar_ratio(nav, ann),
            '溃疡指数': M.ulcer_index(nav),
            'Martin比率': M.martin_ratio(nav, rf, ann),
            '痛苦指数': M.pain_index(nav),
            '痛苦比率': M.pain_ratio(nav, rf, ann),
            '最长回撤期(日)': M.max_drawdown_duration(nav),
            '下行偏差': M.downside_deviation(ret, 0.0, ann),
            'VaR(95%)': M.value_at_risk(ret, 0.05),
            'CVaR(95%)': M.conditional_var(ret, 0.05),
            '偏度': M.skewness(ret),
            '峰度': M.kurtosis(ret),
        }
        # 交易 / 持仓类（需 weight）
        if self.weight is not None:
            to = M.turnover(self.weight, ann)
            cnt = M.avg_holdings_count(self.weight)
            out.update({
                '单边换手率(年化)': to['annualized'],
                '平均单边换手': to['avg_oneway'],
                '平均持仓数': cnt['avg'],
                '平均持仓寿命(期)': M.holding_period(self.weight),
                '权重HHI(均值)': M.weight_hhi(self.weight).mean(),
                '前5集中度(均值)': M.top_n_concentration(self.weight, 5).mean(),
            })
        # 基准相对类
        if self.bench_returns is not None:
            br = self.bench_returns.reindex(ret.index)
            out.update({
                'Beta': M.beta(ret, br),
                'Alpha': M.alpha(ret, br, rf, ann),
                '跟踪误差': M.tracking_error(ret, br, ann),
                '信息比率': M.information_ratio(ret, br, ann),
                '相对基准胜率': M.win_rate_vs_benchmark(ret, br),
            })
        return out

    def summary_grouped(self) -> dict:
        """按类别分组的指标 dict（用于分块展示）"""
        s = self.summary()
        groups = {
            '收益': ['累计收益', '年化收益', '年化波动率', '夏普比率', '索提诺比率'],
            '回撤': ['最大回撤', '卡玛比率', '溃疡指数', 'Martin比率', '痛苦指数',
                     '痛苦比率', '最长回撤期(日)'],
            '风险分布': ['下行偏差', 'VaR(95%)', 'CVaR(95%)', '偏度', '峰度'],
            '交易持仓': ['单边换手率(年化)', '平均单边换手', '平均持仓数',
                         '平均持仓寿命(期)', '权重HHI(均值)', '前5集中度(均值)'],
            '基准相对': ['Beta', 'Alpha', '跟踪误差', '信息比率', '相对基准胜率'],
        }
        return {g: {k: s[k] for k in keys if k in s}
                for g, keys in groups.items()}


def _fmt(v):
    """统一格式化：百分比类用 %，比率类保留 4 位"""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return '-'
        return f"{v:.4f}"
    return str(v)


def _fmt_pct(v):
    if isinstance(v, (float, np.floating)) and not np.isnan(v):
        return f"{v:.2%}"
    return _fmt(v)


# 这些 key 用百分比展示
_PCT_KEYS = {'累计收益', '年化收益', '年化波动率', '最大回撤', '痛苦指数',
             '下行偏差', 'VaR(95%)', 'CVaR(95%)', '单边换手率(年化)', '平均单边换手',
             'Alpha', '跟踪误差', '相对基准胜率', '前5集中度(均值)'}


class ReportExporter:
    """报告导出（兼容旧接口：分年度 / 时段 / 基准对比，CLI + Excel）"""

    def __init__(self, analyzer, benchmark_analyzer=None, start_date=None, end_date=None):
        self.analyzer = analyzer
        if start_date:
            subset = self.analyzer.nav.loc[start_date:end_date]
            self.analyzer = PortfolioAnalyzer(subset)
        self.benchmark = None
        if benchmark_analyzer:
            subset = benchmark_analyzer.nav.loc[
                self.analyzer.nav.index[0]:self.analyzer.nav.index[-1]]
            self.benchmark = PortfolioAnalyzer(subset)
            excess_returns = (self.analyzer.returns - self.benchmark.returns + 1).cumprod()
            self.excess = PortfolioAnalyzer(excess_returns)

    def _format_percentage(self, value):
        return f"{value:.2%}" if isinstance(value, (float, np.float64)) else str(value)

    def generate_annual_report(self, excel_path=None):
        """分年度绩效报告"""
        annual_groups = self.analyzer.nav.groupby(pd.Grouper(freq='YE'))
        report_data = []
        for year, nav_series in annual_groups:
            if len(nav_series) < 5:
                continue
            ya = PortfolioAnalyzer(nav_series)
            report_data.append({
                'Year': year.year,
                'Return': ya.total_return(),
                'Volatility': ya.annualized_volatility(),
                'MaxDD': ya.max_drawdown(),
                'Sharpe': ya.sharpe_ratio(),
            })
        df = pd.DataFrame(report_data).set_index('Year')
        self._print_cli_table(df, title="分年度绩效报告")
        if excel_path:
            self._export_to_excel(df, excel_path, sheet_name="Annual")
        return df

    def generate_custom_report(self, start_date, end_date, excel_path=None):
        """指定时段绩效报告"""
        subset = self.analyzer.nav.loc[start_date:end_date]
        if len(subset) < 2:
            print(f"⚠️ 警告: {start_date}至{end_date}期间数据不足（仅{len(subset)}条），跳过")
            return None
        ca = PortfolioAnalyzer(subset)
        metrics = {
            '累计收益': ca.total_return(),
            '年化波动率': ca.annualized_volatility(),
            '最大回撤': ca.max_drawdown(),
            '夏普比率': ca.sharpe_ratio(),
        }
        df = pd.DataFrame([metrics])
        self._print_cli_table(df, title=f"{start_date}至{end_date}绩效")
        if excel_path:
            self._export_to_excel(df, excel_path, sheet_name="Custom")
        return df

    def generate_benchmark_report(self, excel_path=None):
        """基准对比报告"""
        if not self.benchmark:
            raise ValueError("未提供基准分析器")
        comparison = {
            '组合收益': self.analyzer.total_return(),
            '基准收益': self.benchmark.total_return(),
            '超额收益': self.excess.total_return(),
            '组合波动率': self.analyzer.annualized_volatility(),
            '基准波动率': self.benchmark.annualized_volatility(),
        }
        df = pd.DataFrame([comparison])
        self._print_cli_table(df, title="基准对比报告")
        if excel_path:
            self._export_to_excel(df, excel_path, sheet_name="Benchmark")
        return df

    def _print_cli_table(self, data_df, title=""):
        table = PrettyTable()
        table.title = title
        table.field_names = ["指标"] + list(data_df.columns)
        for index, row in data_df.iterrows():
            table.add_row([index] + [self._format_percentage(v) for v in row])
        print(table)

    def _export_to_excel(self, data_df, file_path, sheet_name):
        if not file_path.endswith('.xlsx'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            file_path = f"Report_{timestamp}.xlsx"
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            data_df.to_excel(writer, sheet_name=sheet_name, float_format="%.2f")
        print(f"报告已保存至: {file_path}")


class Analyst:
    """
    策略评价一键门面。

    用法：
        a = Analyst.from_backtest(bt, benchmark=hs300_bt)
        a.report()                    # CLI 打印全部指标表
        a.report(to_excel='r.xlsx')   # 同时导出 Excel
        a.report(to_html='r.html')    # 导出交互 HTML（plotly）
        figs = a.plots()              # {名称: PNG bytes}，供 st.image
        ifigs = a.interactive_plots() # {名称: plotly Figure}，供 dashboard
    """

    def __init__(self, analyzer: PortfolioAnalyzer, name: str = '组合'):
        self.an = analyzer
        self.name = name
        self._name_map = None

    @classmethod
    def from_backtest(cls, bt, benchmark=None, risk_free_rate=0.0,
                      annualizer=252, window=30, name='组合'):
        """
        从回测实例构建。自动抽取 nav / actual_weight / daily_position_value /
        daily_pnl / rebalance_log。

        Args:
            bt: BacktestBase 实例（须已完成回测，含 nav 等属性）
            benchmark: 基准 nav Series 或另一个 bt 实例
            name: 组合名称（用于报告标题）
        """
        bench_nav = None
        if benchmark is not None:
            bench_nav = getattr(benchmark, 'nav', benchmark)

        weight = getattr(bt, 'actual_weight', None)
        if weight is None:
            weight = getattr(bt, 'weight', None)

        an = PortfolioAnalyzer(
            nav_series=bt.nav,
            risk_free_rate=risk_free_rate,
            annualizer=annualizer,
            window=window,
            weight=weight,
            daily_position_value=getattr(bt, 'daily_position_value', None),
            daily_pnl=getattr(bt, 'daily_pnl', None),
            rebalance_log=getattr(bt, 'rebalance_log', None),
            benchmark=bench_nav,
        )
        return cls(an, name=name)

    @classmethod
    def from_excel(cls, filepath, benchmark=None, name='组合', **kwargs):
        """从 bt.dump_to_excel 导出的 xlsx 读回构建（dashboard 上传用）"""
        xls = pd.ExcelFile(filepath)

        def _read(sheet, index_col=0):
            if sheet not in xls.sheet_names:
                return None
            df = pd.read_excel(xls, sheet_name=sheet, index_col=index_col)
            if index_col == 0 and len(df.index):
                try:
                    df.index = pd.to_datetime(df.index)
                except (ValueError, TypeError):
                    pass
            return df

        nav_df = _read('nav')
        nav = nav_df.iloc[:, 0] if nav_df is not None else None
        if nav is None:
            raise ValueError("Excel 缺少 nav sheet")

        an = PortfolioAnalyzer(
            nav_series=nav,
            weight=_read('actual_weight') if 'actual_weight' in xls.sheet_names else _read('weight'),
            daily_position_value=_read('daily_position_value'),
            daily_pnl=_read('daily_pnl'),
            rebalance_log=_read('rebalance_log', index_col=None),
            benchmark=benchmark,
            **kwargs,
        )
        return cls(an, name=name)

    # ── 中文名映射 ─────────────────────────────────────────────────────────

    @property
    def name_map(self) -> dict:
        if self._name_map is None:
            codes = set()
            if self.an.weight is not None:
                codes.update(self.an.weight.columns)
            if self.an.daily_position_value is not None:
                codes.update(self.an.daily_position_value.columns)
            self._name_map = get_name_map(list(codes))
        return self._name_map

    # ── 数据表 ─────────────────────────────────────────────────────────────

    def summary_df(self) -> pd.DataFrame:
        """全部指标的明细表（含格式化展示列）"""
        rows = []
        for group, items in self.an.summary_grouped().items():
            for k, v in items.items():
                disp = _fmt_pct(v) if k in _PCT_KEYS else _fmt(v)
                rows.append({'类别': group, '指标': k, '数值': v, '展示': disp})
        return pd.DataFrame(rows)

    def top_holdings_df(self, top=10) -> pd.DataFrame:
        if self.an.weight is None:
            return pd.DataFrame()
        df = M.top_holdings(self.an.weight, top)
        df.index = [label(c, self.name_map) for c in df.index]
        return df

    def contribution_df(self, top=15) -> pd.DataFrame:
        if self.an.daily_pnl is None:
            return pd.DataFrame()
        df = M.return_contribution(self.an.daily_pnl, top)
        df.index = [label(c, self.name_map) for c in df.index]
        return df

    def trade_pnl_df(self) -> pd.DataFrame:
        if self.an.rebalance_log is None:
            return pd.DataFrame()
        df = M.trade_pnl(self.an.rebalance_log)
        if not df.empty:
            df.index = [label(c, self.name_map) for c in df.index]
        return df

    def monthly_table(self) -> pd.DataFrame:
        return M.monthly_returns_table(self.an.nav)

    # ── CLI 打印 ───────────────────────────────────────────────────────────

    def print_report(self):
        """命令行打印全部指标表格（按类别分组）"""
        print(f"\n{'='*50}\n  {self.name} 策略评价报告\n{'='*50}")
        for group, items in self.an.summary_grouped().items():
            if not items:
                continue
            table = PrettyTable()
            table.title = group
            table.field_names = ["指标", "数值"]
            table.align["指标"] = "l"
            table.align["数值"] = "r"
            for k, v in items.items():
                disp = _fmt_pct(v) if k in _PCT_KEYS else _fmt(v)
                table.add_row([k, disp])
            print(table)

        th = self.top_holdings_df()
        if not th.empty:
            t = PrettyTable()
            t.title = "最频繁持仓 Top10"
            t.field_names = ["标的", "出现频率", "平均权重", "最大权重"]
            for idx, row in th.iterrows():
                t.add_row([idx, f"{row['freq']:.1%}",
                           f"{row['avg_weight']:.2%}", f"{row['max_weight']:.2%}"])
            print(t)

        contrib = self.contribution_df()
        if not contrib.empty:
            t = PrettyTable()
            t.title = "收益贡献 Top15"
            t.field_names = ["标的", "累计损益", "贡献占比"]
            for idx, row in contrib.iterrows():
                t.add_row([idx, f"{row['pnl']:,.0f}", f"{row['contribution']:.2%}"])
            print(t)

    # ── 静态图（matplotlib PNG bytes）──────────────────────────────────────

    def plots(self) -> dict:
        """返回 {名称: PNG bytes}，供 st.image / 嵌入"""
        an = self.an
        figs = {
            '净值曲线': P.plot_nav(an.nav, an.benchmark, f'{self.name} 净值曲线'),
            '回撤曲线': P.plot_drawdown(an.nav),
            '滚动胜率': P.plot_rolling_metric(
                M.rolling_win_rate(an.returns, an.window),
                f'滚动胜率({an.window}日)', '胜率'),
            '滚动夏普': P.plot_rolling_metric(
                M.rolling_sharpe(an.returns, max(an.window, 60)),
                f'滚动夏普({max(an.window, 60)}日)', '夏普', color='#1f77b4'),
            '月度收益': P.plot_monthly_heatmap(self.monthly_table()),
        }
        if an.weight is not None:
            figs['权重堆积'] = P.plot_weight_concentration(
                M.weight_hhi(an.weight),
                M.avg_holdings_count(an.weight)['per_period'],
                weight=an.weight, name_map=self.name_map)
        if an.daily_pnl is not None:
            figs['收益贡献'] = P.plot_contribution_bar(
                M.return_contribution(an.daily_pnl), self.name_map)
        return figs

    # ── 交互图（plotly Figure）─────────────────────────────────────────────

    def interactive_plots(self) -> dict:
        """返回 {名称: plotly Figure}，供 dashboard / HTML"""
        an = self.an
        figs = {
            '净值曲线': P.ip_nav(an.nav, an.benchmark, f'{self.name} 净值曲线'),
            '回撤曲线': P.ip_drawdown(an.nav),
            '滚动胜率': P.ip_rolling(
                M.rolling_win_rate(an.returns, an.window),
                f'滚动胜率({an.window}日)', '胜率'),
            '滚动夏普': P.ip_rolling(
                M.rolling_sharpe(an.returns, max(an.window, 60)),
                f'滚动夏普({max(an.window, 60)}日)', '夏普', color='#1f77b4'),
            '月度收益': P.ip_monthly_heatmap(self.monthly_table()),
        }
        if an.daily_position_value is not None:
            figs['权重堆积'] = P.ip_weight_area(an.daily_position_value, self.name_map)
        if an.daily_pnl is not None:
            figs['收益贡献'] = P.ip_contribution(
                M.return_contribution(an.daily_pnl), self.name_map)
        return figs

    # ── 导出 ───────────────────────────────────────────────────────────────

    def to_excel(self, filepath: str) -> str:
        """导出 Excel：指标汇总 + 各明细表"""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            self.summary_df()[['类别', '指标', '展示']].to_excel(
                writer, sheet_name='指标汇总', index=False)
            th = self.top_holdings_df()
            if not th.empty:
                th.to_excel(writer, sheet_name='最频繁持仓')
            contrib = self.contribution_df()
            if not contrib.empty:
                contrib.to_excel(writer, sheet_name='收益贡献')
            tp = self.trade_pnl_df()
            if not tp.empty:
                tp.to_excel(writer, sheet_name='逐笔盈亏')
            mt = self.monthly_table()
            if not mt.empty:
                mt.to_excel(writer, sheet_name='月度收益')
        print(f"Excel 报告已保存至: {filepath}")
        return filepath

    def to_html(self, filepath: str) -> str:
        """导出独立 HTML 报告（含 plotly 交互图 + 指标表）"""
        parts = [
            "<html><head><meta charset='utf-8'>",
            "<style>body{font-family:'Microsoft YaHei',sans-serif;margin:24px;}"
            "table{border-collapse:collapse;margin:12px 0;}"
            "th,td{border:1px solid #ddd;padding:6px 12px;text-align:right;}"
            "th{background:#f5f5f5;}h1,h2{color:#333;}</style></head><body>",
            f"<h1>{self.name} 策略评价报告</h1>",
        ]
        # 指标表
        sdf = self.summary_df()
        parts.append("<h2>指标汇总</h2>")
        parts.append(sdf[['类别', '指标', '展示']].to_html(index=False, border=0))
        # 交互图
        first = True
        for title, fig in self.interactive_plots().items():
            parts.append(f"<h2>{title}</h2>")
            parts.append(fig.to_html(
                full_html=False,
                include_plotlyjs='cdn' if first else False))
            first = False
        parts.append("</body></html>")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(parts))
        print(f"HTML 报告已保存至: {filepath}")
        return filepath

    def report(self, to_excel: str = None, to_html: str = None,
               show_plots: bool = False):
        """
        一键报告：CLI 打印 + 可选导出 Excel / HTML。

        Args:
            to_excel: Excel 输出路径（None 不导出）
            to_html: HTML 输出路径（None 不导出）
            show_plots: 是否在 CLI 环境弹出静态图（matplotlib）
        """
        self.print_report()
        if to_excel:
            self.to_excel(to_excel)
        if to_html:
            self.to_html(to_html)
        if show_plots:
            import matplotlib.pyplot as plt
            for title, png in self.plots().items():
                import io
                img = plt.imread(io.BytesIO(png))
                plt.figure(figsize=(10, 4))
                plt.imshow(img)
                plt.axis('off')
                plt.title(title)
            plt.show()
        return self.an.summary()


#%%
# 使用示例
if __name__ == "__main__":
    dates = pd.date_range('2010-01-01', '2025-05-15')
    nav = np.exp(np.random.normal(0, 0.01, len(dates)).cumsum())
    analyzer = PortfolioAnalyzer(pd.Series(nav, index=dates))

    benchmark = PortfolioAnalyzer(pd.Series(np.ones(len(dates)), index=dates))
    exporter = ReportExporter(analyzer, benchmark)
    exporter.generate_annual_report()
    exporter.generate_custom_report('2024-01-01', '2024-12-31')
    exporter.generate_benchmark_report()

    # 新门面（无持仓数据时仅算 nav 类指标）
    Analyst(analyzer, name='示例组合').print_report()

