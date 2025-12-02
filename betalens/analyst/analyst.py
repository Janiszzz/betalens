import pandas as pd
import numpy as np
from datetime import datetime

class PortfolioAnalyzer:
    def __init__(self, nav_series, risk_free_rate=0.0, annualizer=252, window=30):
        """
        :param nav_series: pandas Series，包含日期索引的净值序列
        :param risk_free_rate: 无风险利率（年化）
        """
        self.nav = nav_series.sort_index()
        self.returns = self.nav.pct_change().dropna()
        self.risk_free_rate = risk_free_rate
        self.annualizer = annualizer
        self.window = window
        
    # 核心指标计算
    def total_return(self):
        """累计收益率"""
        return (self.nav.iloc[-1] / self.nav.iloc[0]) - 1
    
    def annualized_return(self):
        """年化收益率"""
        return self.returns.mean() * self.annualizer

    def annualized_volatility(self):
        """年化波动率"""
        return self.returns.std() * np.sqrt(self.annualizer)

    def sharpe_ratio(self):
        """夏普比率（年化）"""
        excess_returns = self.returns - self.risk_free_rate/self.annualizer
        return excess_returns.mean() / excess_returns.std() * np.sqrt(self.annualizer)

    def max_drawdown(self):
        """最大回撤"""
        peak = self.nav.expanding(min_periods=1).max()
        drawdown = (peak - self.nav)/peak
        return drawdown.max()

    # 滚动窗口计算（示例）
    def rolling_max_drawdown(self):
        """滚动最大回撤"""
        return self.nav.rolling(self.window).apply(
            lambda x: (x.expanding().max() - x).max() / x.expanding().max()
        )
    
    def rolling_win_rate(self):
        return (self.returns.rolling(self.window).sum() > 0).mean()/(len(self.returns)-self.window)*len(self.returns)
    
    # 高级指标（需辅助方法）
    def calmar_ratio(self):
        """卡玛比率（年化收益/最大回撤）"""
        annual_return = (1 + self.total_return())**(self.annualizer/len(self.nav)) - 1
        return annual_return / self.max_drawdown()

    def recovery_days(self):
        peak_idx = self.nav.expanding().idxmax()
        trough_idx = self.nav.idxmin()
        recovery_date = self.nav[self.nav >= self.nav.loc[peak_idx]].index[0]
        return (recovery_date - trough_idx).days
 

from prettytable import PrettyTable

class ReportExporter:
    def __init__(self, analyzer, benchmark_analyzer=None, start_date=None, end_date=None):
        """
        :param analyzer: PortfolioAnalyzer实例
        :param benchmark_analyzer: 基准分析器实例（可选）
        """
        self.analyzer = analyzer

        if(start_date):
            subset = self.analyzer.nav.loc[start_date:end_date]
            custom_analyzer = PortfolioAnalyzer(subset)
            self.analyzer = custom_analyzer
        if(benchmark_analyzer):
            subset = benchmark_analyzer.nav.loc[self.analyzer.nav.index[0]:self.analyzer.nav.index[-1]]
            self.benchmark = PortfolioAnalyzer(subset)
            #self.excess_returns = PortfolioAnalyzer(((self.analyzer.returns+1) / (self.benchmark.returns+1)).cumprod())
            excess_returns = (self.analyzer.returns - self.benchmark.returns + 1).cumprod()

            self.excess = PortfolioAnalyzer(excess_returns)

    def _format_percentage(self, value):
        """统一格式化百分比数值"""
        return f"{value:.2%}" if isinstance(value, (float, np.float64)) else str(value)

    def generate_annual_report(self, excel_path=None):
        """分年度报告生成"""
        # 按年分组计算指标（参考网页7的resample方法）
        annual_groups = self.analyzer.nav.groupby(pd.Grouper(freq='YE'))
        
        report_data = []
        for year, nav_series in annual_groups:
            if len(nav_series) < 5: continue  # 过滤无效年份
            year_analyzer = PortfolioAnalyzer(nav_series)
            report_data.append({
                'Year': year.year,
                'Return': year_analyzer.total_return(),
                'Volatility': year_analyzer.annualized_volatility(),
                'MaxDD': year_analyzer.max_drawdown(),
                'Sharpe': year_analyzer.sharpe_ratio()
            })
        
        df = pd.DataFrame(report_data).set_index('Year')
        self._print_cli_table(df, title="分年度绩效报告")
        if excel_path:
            self._export_to_excel(df, excel_path, sheet_name="Annual")

    def generate_custom_report(self, start_date, end_date, excel_path=None):
        """指定时段报告生成（参考网页11的时间切片）"""
        subset = self.analyzer.nav.loc[start_date:end_date]
        
        # 检查是否有足够的数据
        if len(subset) < 2:
            print(f"⚠️ 警告: {start_date}至{end_date}期间数据不足（仅{len(subset)}条），跳过报告生成")
            print(f"   可用数据范围: {self.analyzer.nav.index[0]} 至 {self.analyzer.nav.index[-1]}")
            return
        
        custom_analyzer = PortfolioAnalyzer(subset)
        
        metrics = {
            '累计收益': custom_analyzer.total_return(),
            '年化波动率': custom_analyzer.annualized_volatility(),
            '最大回撤': custom_analyzer.max_drawdown(),
            '夏普比率': custom_analyzer.sharpe_ratio()
        }
        
        df = pd.DataFrame([metrics])
        self._print_cli_table(df, title=f"{start_date}至{end_date}绩效")
        if excel_path:
            self._export_to_excel(df, excel_path, sheet_name="Custom")

    def generate_benchmark_report(self, excel_path=None):
        """基准对比报告（参考网页13的对比逻辑）"""
        if not self.benchmark:
            raise ValueError("未提供基准分析器")
            
        # 计算超额收益等指标
        comparison = {
            '组合收益': self.analyzer.total_return(),
            '基准收益': self.benchmark.total_return(),
            '超额收益': self.excess.total_return(),
            '组合波动率': self.analyzer.annualized_volatility(),
            '基准波动率': self.benchmark.annualized_volatility()
        }
        
        df = pd.DataFrame([comparison])
        self._print_cli_table(df, title="基准对比报告")
        if excel_path:
            self._export_to_excel(df, excel_path, sheet_name="Benchmark")



    def _print_cli_table(self, data_df, title=""):
        """命令行表格输出（参考网页5的格式化）"""
        table = PrettyTable()
        table.title = title
        table.field_names = ["指标"] + list(data_df.columns)
        
        for index, row in data_df.iterrows():
            table.add_row([index] + [self._format_percentage(v) for v in row])
        
        print(table)

    def _export_to_excel(self, data_df, file_path, sheet_name):
        """Excel生成（参考网页6的分表逻辑）"""
        if not file_path.endswith('.xlsx'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            file_path = f"Report_{timestamp}.xlsx"
            
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            data_df.to_excel(writer, sheet_name=sheet_name, float_format="%.2f")
        print(f"报告已保存至: {file_path}")
#%%
# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    dates = pd.date_range('2010-01-01', '2025-05-15')
    nav = np.exp(np.random.normal(0, 0.01, len(dates)).cumsum())
    analyzer = PortfolioAnalyzer(pd.Series(nav, index=dates))
    
    # 基准数据
    benchmark_nav = np.exp(np.random.normal(0, 0.01, len(dates)).cumsum())
    benchmark = PortfolioAnalyzer(pd.Series(benchmark_nav, index=dates))
    benchmark = PortfolioAnalyzer(pd.Series(np.ones(len(dates)), index=dates))
    # 生成报告
    exporter = ReportExporter(analyzer, benchmark)
    exporter.generate_annual_report()  # 分年度输出
    exporter.generate_custom_report('2024-01-01', '2024-12-31')  # 指定时段
    exporter.generate_benchmark_report()  # 基准对比

    exporter.analyzer.returns
    exporter.benchmark.returns
    exporter.excess.returns

    (analyzer.nav.iloc[-1] / analyzer.nav.iloc[0]) - 1

    analyzer.total_return()