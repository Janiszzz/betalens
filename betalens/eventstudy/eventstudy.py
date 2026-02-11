"""
事件研究模块 - 用于分析特定事件前后的收益率表现

betalens API 使用:
    - datafeed.query_time_range(): 查询时间范围内的数据
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def _get_event_dates(events: pd.Series) -> pd.DatetimeIndex:
    """从事件序列中提取事件发生日期"""
    return events[events == 1].index


def _calc_returns(prices: pd.Series) -> pd.Series:
    """计算日收益率"""
    return prices.pct_change()


def _get_day0_cost_price_loc(
    prices: pd.Series,
    event_date: pd.Timestamp,
    market_close_hour: int = 15
) -> Optional[int]:
    """确定Day 0的成本价位置

    - 事件在15:00前：当天收盘价为成本价
    - 事件在15:00后：第二天收盘价为成本价

    Returns:
        成本价在prices中的位置索引，如果找不到返回None
    """
    event_date_only = event_date.date()

    if event_date.hour < market_close_hour:
        # 收盘前：找当天或之后第一个价格
        valid_prices = prices[prices.index.date >= event_date_only]
    else:
        # 收盘后：找第二天或之后第一个价格
        next_day = event_date_only + timedelta(days=1)
        valid_prices = prices[prices.index.date >= next_day]

    if valid_prices.empty:
        return None

    return prices.index.get_loc(valid_prices.index[0])


def _get_window_returns(
    returns: pd.Series,
    prices: pd.Series,
    event_date: pd.Timestamp,
    window_before: int,
    window_after: int,
    market_close_hour: int = 15
) -> Optional[pd.Series]:
    """获取事件窗口期内的收益率序列，索引重置为相对事件日的天数

    关键：
    - 如果事件发生在当天收盘前（15:00前），当天收盘价为day0的成本价
    - 如果事件发生在当天收盘后，第二日收盘价为day0的成本价
    - Day 0的收益是从成本价到下一个交易日的收益
    """
    # 确定成本价位置
    cost_price_loc = _get_day0_cost_price_loc(prices, event_date, market_close_hour)
    if cost_price_loc is None:
        return None

    # Day 0的收益是从成本价到下一个价格的收益率
    # 对应returns中cost_price_loc+1位置的收益率
    first_return_loc = cost_price_loc + 1
    if first_return_loc >= len(returns):
        return None

    # 计算窗口范围：向前取window_before个点，向后取window_after个点
    start = max(0, first_return_loc - window_before)
    end = min(len(returns), first_return_loc + window_after + 1)

    window_ret = returns.iloc[start:end].copy()
    # 重置索引：first_return_loc对应day 0
    window_ret.index = range(-(first_return_loc - start), end - first_return_loc)

    return window_ret


def _aggregate_window_returns(
    all_returns: List[pd.Series]
) -> pd.DataFrame:
    """将多个事件的窗口收益率合并为矩阵，行=相对天数，列=事件编号"""
    if not all_returns:
        return pd.DataFrame()
    df = pd.DataFrame(all_returns).T
    df.columns = range(len(all_returns))
    return df


def _compute_stats(returns: pd.Series) -> dict:
    """计算收益率统计量: 均值、标准差、上涨概率、胜率、t统计量、样本数"""
    if returns.empty or returns.isna().all():
        return {
            'mean': np.nan,
            'std': np.nan,
            'positive_prob': np.nan,
            'odds': np.nan,
            't_stat': np.nan,
            'count': 0
        }
    clean = returns.dropna()
    n = len(clean)
    if n == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'positive_prob': np.nan,
            'odds': np.nan,
            't_stat': np.nan,
            'count': 0
        }
    mean = clean.mean()
    std = clean.std()
    pos_prob = (clean > 0).mean()
    odds = pos_prob / (1 - pos_prob) if pos_prob < 1 else np.inf
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else np.nan
    return {
        'mean': mean,
        'std': std,
        'positive_prob': pos_prob,
        'odds': odds,
        't_stat': t_stat,
        'count': n
    }


def _compute_period_stats(
    returns_df: pd.DataFrame,
    event_dates: pd.DatetimeIndex,
    periods: pd.Series
) -> pd.DataFrame:
    """按时间段分组计算统计量"""
    aligned_periods = periods.reindex(event_dates)
    results = []
    for period_val in aligned_periods.dropna().unique():
        mask = aligned_periods == period_val
        cols = [i for i, m in enumerate(mask) if m]
        if not cols:
            continue
        period_returns = returns_df[cols].values.flatten()
        period_returns = pd.Series(period_returns).dropna()
        stats = _compute_stats(period_returns)
        stats['period'] = period_val
        results.append(stats)
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).set_index('period').sort_index()


class EventStudy:
    """事件研究分析器"""

    def __init__(self, datafeed):
        self.datafeed = datafeed  # [betalens API] datafeed 实例

    def _get_stock_window_returns(
        self,
        code: str,
        event_dates: pd.DatetimeIndex,
        start: str,
        end: str,
        metric: str,
        window_before: int,
        window_after: int,
        market_close_hour: int
    ) -> pd.DataFrame:
        """获取单个股票在所有事件窗口的收益率矩阵

        Returns:
            DataFrame: 行为相对天数，列为事件编号，值为收益率
        """
        try:
            data = self.datafeed.query_time_range(
                codes=[code],
                start_date=start,
                end_date=end,
                metric=metric
            )

            if data.empty:
                return pd.DataFrame()

            prices = data.set_index('datetime')['value'].sort_index()
            prices = prices.astype(float)
            returns = _calc_returns(prices)

            all_window_returns = []
            for ed in event_dates:
                wr = _get_window_returns(returns, prices, ed, window_before, window_after, market_close_hour)
                if wr is not None and not wr.empty:
                    all_window_returns.append(wr)
                else:
                    # 如果这个事件没有数据，添加空Series保持对齐
                    all_window_returns.append(None)

            return _aggregate_window_returns([r for r in all_window_returns if r is not None])

        except Exception:
            # 如果获取数据失败，返回空DataFrame
            return pd.DataFrame()

    def _calc_cumulative_flexible(self, returns_df: pd.DataFrame, holding_start_offset: int = 0) -> pd.DataFrame:
        """模式一：累积收益率序列

        直接计算收益率序列的累积收益，不区分正反向，不设置0点
        从序列第一个值开始累积到最后

        Args:
            holding_start_offset: 保留参数以保持接口兼容，但不再使用
        """
        # 直接计算累积收益率
        return returns_df.add(1).cumprod() - 1

    def _calc_cumulative_fixed(
        self,
        returns_df: pd.DataFrame,
        holding_periods: Optional[dict],
        holding_start_offset: int = 0
    ) -> pd.DataFrame:
        """模式二：固定持有期收益

        - 持有后1,2,3,4,5天，1,3,6,9,12个月：从持有起点到目标日的累积收益
        - 持有前1,2,3,4,5天，1,3,6,9,12个月：从目标日到持有起点的累积收益

        Args:
            holding_periods: {'days': [1,2,3,4,5], 'months': [1,3,6,9,12]}
            holding_start_offset: 持有起点偏移天数，0表示从Day 0开始，n表示从Day n开始
        """
        if holding_periods is None:
            holding_periods = {
                'days': [1, 2, 3, 4, 5],
                'months': [1, 3, 6, 9, 12]
            }

        if 0 not in returns_df.index:
            raise ValueError("模式二需要Day 0作为参考点")

        # 确定实际的持有起点
        holding_start = holding_start_offset
        if holding_start not in returns_df.index:
            # 如果偏移后的起点不在数据中，找最接近的
            available_days = returns_df.index[returns_df.index >= holding_start]
            if available_days.empty:
                return pd.DataFrame()
            holding_start = available_days[0]

        holding_start_loc = returns_df.index.get_loc(holding_start)
        if not isinstance(holding_start_loc, int):
            return pd.DataFrame()

        # 构建目标持有期索引（相对于持有起点的偏移）
        target_days = []

        # 添加天数持有期
        if 'days' in holding_periods:
            target_days.extend(holding_periods['days'])

        # 添加月数持有期（假设1个月=21个交易日）
        if 'months' in holding_periods:
            for m in holding_periods['months']:
                target_days.append(m * 21)

        # 生成负数持有期，并调整为相对于持有起点的索引
        target_indices = []
        for d in target_days:
            target_indices.append(holding_start + d)  # 正向
            target_indices.append(holding_start - d)  # 反向
        target_indices = sorted(set(target_indices))

        # 筛选在数据范围内的索引
        min_idx = returns_df.index.min()
        max_idx = returns_df.index.max()
        target_indices = [d for d in target_indices if min_idx <= d <= max_idx]

        # 计算每个目标持有期的累积收益
        result_dict = {}
        for target_day in target_indices:
            if target_day >= holding_start:
                # 正向持有：从持有起点累积到target_day
                if target_day in returns_df.index:
                    end_loc = returns_df.index.get_loc(target_day)
                    if isinstance(end_loc, int):
                        period_returns = returns_df.iloc[holding_start_loc:end_loc+1]
                        cum_ret = period_returns.add(1).prod(axis=0) - 1
                        result_dict[target_day] = cum_ret
            else:
                # 反向持有：从target_day正向累积到持有起点
                if target_day in returns_df.index:
                    period_start_loc = returns_df.index.get_loc(target_day)
                    if isinstance(period_start_loc, int):
                        period_returns = returns_df.iloc[period_start_loc:holding_start_loc+1]
                        cum_ret = period_returns.add(1).prod(axis=0) - 1
                        result_dict[target_day] = cum_ret

        if not result_dict:
            return pd.DataFrame()

        return pd.DataFrame(result_dict).T
    
    def analyze(
        self,
        events: pd.Series,
        code: Union[str, List[str]],
        window_before: int = 5,
        window_after: int = 5,
        metric: str = '收盘价(元)',
        periods: Optional[pd.Series] = None,
        mode: str = 'flexible',
        holding_periods: Optional[dict] = None,
        holding_start_offset: int = 0,
        market_close_hour: int = 15,
        benchmark_code: Optional[str] = None
    ) -> dict:
        """
        分析事件前后的收益率表现

        Args:
            events: 事件序列，index为精确到秒的datetime，值为1表示事件发生
            code: 证券代码，可以是单个代码(str)或多个代码列表(List[str])
                  当为列表时，计算所有股票在每个时间点的平均收益率
            window_before: 事件前窗口期数（相对于事件后第一个收益点）
            window_after: 事件后窗口期数
            metric: 价格指标名称
            periods: 可选的时间分段序列
            mode: 展示模式，'flexible'=给定n展示前后n期，'fixed'=固定持有期
            holding_periods: 固定持有期字典，如{'days': [1,2,3,4,5], 'months': [1,3,6,9,12]}
            holding_start_offset: 持有起点偏移天数，0表示从Day 0开始，n表示从Day n开始
            market_close_hour: 市场收盘时间（小时），默认15点
            benchmark_code: 可选的业绩比较基准代码，如提供则计算超额收益=持有标的收益-基准收益

        Returns:
            包含 daily_stats, cumulative_stats, event_count, returns_matrix 的字典
            多标的模式额外包含: stock_returns_dict (每个股票的收益矩阵)

        Note:
            Day 0成本价判断：
            - 事件在15:00前：当天收盘价为成本价
            - 事件在15:00后：第二日收盘价为成本价
        """
        event_dates = _get_event_dates(events)
        if event_dates.empty:
            return {'error': 'no events'}

        start = (event_dates.min() - timedelta(days=window_before * 5)).strftime('%Y-%m-%d')
        end = (event_dates.max() + timedelta(days=window_after * 5)).strftime('%Y-%m-%d')

        # 判断是单标的还是多标的模式
        is_multi_stock = isinstance(code, list)

        if is_multi_stock:
            # 多标的模式：分别获取每个股票数据并计算平均
            stock_returns_dict = {}
            valid_codes = []

            print(f"多标的模式：正在获取 {len(code)} 只股票的数据...")
            for stock_code in code:
                returns_df = self._get_stock_window_returns(
                    stock_code, event_dates, start, end, metric,
                    window_before, window_after, market_close_hour
                )
                if not returns_df.empty:
                    stock_returns_dict[stock_code] = returns_df
                    valid_codes.append(stock_code)

            if not stock_returns_dict:
                return {'error': 'no valid stock data'}

            print(f"成功获取 {len(valid_codes)} 只股票的数据")

            # 计算所有股票在每个时间点的平均收益率
            # 使用所有股票数据的并集索引
            all_indices = set()
            all_columns = set()
            for df in stock_returns_dict.values():
                all_indices.update(df.index)
                all_columns.update(df.columns)

            all_indices = sorted(all_indices)
            all_columns = sorted(all_columns)

            # 创建平均收益率矩阵
            avg_returns_list = []
            for idx in all_indices:
                for col in all_columns:
                    values = []
                    for df in stock_returns_dict.values():
                        if idx in df.index and col in df.columns:
                            val = df.loc[idx, col]
                            if not pd.isna(val):
                                values.append(val)
                    if values:
                        avg_returns_list.append({
                            'day': idx,
                            'event': col,
                            'return': np.mean(values)
                        })

            # 构建平均收益率DataFrame
            if not avg_returns_list:
                return {'error': 'no valid average returns'}

            avg_df = pd.DataFrame(avg_returns_list)
            returns_df = avg_df.pivot(index='day', columns='event', values='return')
            returns_df = returns_df.sort_index()

        else:
            # 单标的模式：原有逻辑
            # [betalens API] 调用 datafeed.query_time_range() 获取价格数据
            data = self.datafeed.query_time_range(
                codes=[code],
                start_date=start,
                end_date=end,
                metric=metric
            )

            if data.empty:
                return {'error': 'no data'}

            # 保持datetime精度（精确到秒），以匹配精确的事件时间戳
            prices = data.set_index('datetime')['value'].sort_index()
            prices = prices.astype(float)  # 转换Decimal为float
            returns = _calc_returns(prices)

            # 如果提供了基准代码，获取基准收益率
            benchmark_returns = None
            benchmark_prices = None
            if benchmark_code:
                benchmark_data = self.datafeed.query_time_range(
                    codes=[benchmark_code],
                    start_date=start,
                    end_date=end,
                    metric=metric
                )
                if not benchmark_data.empty:
                    benchmark_prices = benchmark_data.set_index('datetime')['value'].sort_index()
                    benchmark_prices = benchmark_prices.astype(float)
                    benchmark_returns = _calc_returns(benchmark_prices)

            all_window_returns = []
            valid_event_dates = []
            for ed in event_dates:
                wr = _get_window_returns(returns, prices, ed, window_before, window_after, market_close_hour)
                if wr is not None and not wr.empty:
                    # 如果提供了基准，计算超额收益
                    if benchmark_returns is not None and benchmark_prices is not None:
                        benchmark_wr = _get_window_returns(benchmark_returns, benchmark_prices, ed, window_before, window_after, market_close_hour)
                        if benchmark_wr is not None and not benchmark_wr.empty:
                            # 对齐索引，计算超额收益
                            aligned_wr = wr.reindex(benchmark_wr.index)
                            excess_returns = aligned_wr - benchmark_wr
                            all_window_returns.append(excess_returns.dropna())
                            valid_event_dates.append(ed)
                    else:
                        all_window_returns.append(wr)
                        valid_event_dates.append(ed)

            if not all_window_returns:
                return {'error': 'no matching events'}

            returns_df = _aggregate_window_returns(all_window_returns)
        
        day_stats = {}
        for day in returns_df.index:
            day_stats[day] = _compute_stats(returns_df.loc[day])
        overall_stats = pd.DataFrame(day_stats).T
        overall_stats.index.name = 'day'

        # 计算累积收益
        if mode == 'flexible':
            # 模式一：以Day 0为分界点的双向累积
            cum_returns = self._calc_cumulative_flexible(returns_df, holding_start_offset)
        elif mode == 'fixed':
            # 模式二：固定持有期收益
            cum_returns = self._calc_cumulative_fixed(returns_df, holding_periods, holding_start_offset)
        else:
            raise ValueError(f"不支持的模式: {mode}")

        cum_stats = {}
        for day in cum_returns.index:
            cum_stats[day] = _compute_stats(cum_returns.loc[day])
        cumulative_stats = pd.DataFrame(cum_stats).T
        cumulative_stats.index.name = 'day'
        
        result = {
            'daily_stats': overall_stats,
            'cumulative_stats': cumulative_stats,
            'event_count': returns_df.shape[1] if not returns_df.empty else 0,
            'returns_matrix': returns_df
        }

        # 多标的模式：添加每个股票的收益矩阵
        if is_multi_stock:
            result['stock_returns_dict'] = stock_returns_dict
            result['valid_codes'] = valid_codes

        if periods is not None and not is_multi_stock:
            # 多标的模式下暂不支持periods分析
            period_stats = _compute_period_stats(
                returns_df,
                pd.DatetimeIndex(valid_event_dates),
                periods
            )
            result['period_stats'] = period_stats

        return result

    def plot_bar(
        self,
        daily_stats: pd.DataFrame,
        title: str = '事件前后平均收益率',
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None
    ) -> None:
        """
        类型1：柱状图展示 -n~n 平均收益率序列

        Args:
            daily_stats: 每日收益统计DataFrame（来自analyze结果的daily_stats）
            title: 图表标题
            figsize: 图表尺寸
            save_path: 保存路径，如不提供则显示图表
        """
        fig, ax = plt.subplots(figsize=figsize)

        days = daily_stats.index
        means = daily_stats['mean'].values
        stds = daily_stats['std'].values

        # 绘制柱状图
        colors = ['red' if m < 0 else 'green' for m in means]
        bars = ax.bar(days, means, color=colors, alpha=0.7, edgecolor='black')

        # 添加误差线
        ax.errorbar(days, means, yerr=stds, fmt='none', ecolor='gray', capsize=3, alpha=0.5)

        # 添加零线
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=0, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='事件发生点(Day 0)')

        # 设置标签
        ax.set_xlabel('相对事件发生的天数', fontsize=12)
        ax.set_ylabel('平均收益率', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 格式化y轴为百分比
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_lines(
        self,
        cumulative_stats: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        title: str = '事件前后平均累积收益率',
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
        show_std: bool = True
    ) -> None:
        """
        类型2：折线图展示平均累积收益率曲线（支持单标的或多标的对比）

        Args:
            cumulative_stats: 累积收益统计DataFrame（单标的）或字典（多标的）
                            - 单标的：传入analyze结果的cumulative_stats
                            - 多标的：传入字典 {股票代码: cumulative_stats, ...}
            title: 图表标题
            figsize: 图表尺寸
            save_path: 保存路径，如不提供则显示图表
            show_std: 是否显示标准差区间（仅单标的模式有效）
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 判断是单标的还是多标的模式
        if isinstance(cumulative_stats, dict):
            # 多标的模式：绘制多条曲线
            colors = cm.get_cmap('tab10')
            num_stocks = len(cumulative_stats)

            for idx, (code, stats_df) in enumerate(cumulative_stats.items()):
                days = stats_df.index.to_numpy()
                means = stats_df['mean'].to_numpy()

                color = colors(idx / max(num_stocks - 1, 1))
                ax.plot(days, means, marker='o', linewidth=2, markersize=4,
                       color=color, label=code, alpha=0.8)

            legend_label = '事件发生点(Day 0)'

        else:
            # 单标的模式：原有逻辑
            days = cumulative_stats.index.to_numpy()
            means = cumulative_stats['mean'].to_numpy()
            stds = cumulative_stats['std'].to_numpy()

            # 绘制平均累积收益折线
            ax.plot(days, means, marker='o', linewidth=2, markersize=5,
                    color='blue', label='平均累积收益', alpha=0.8)

            # 添加置信区间
            if show_std:
                ax.fill_between(days, means - stds, means + stds,
                               alpha=0.2, color='blue', label='±1标准差')

            legend_label = '事件发生点(Day 0)'

        # 添加零线和事件发生点
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=legend_label)

        # 设置标签
        ax.set_xlabel('相对事件发生的天数', fontsize=12)
        ax.set_ylabel('平均累积收益率', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # 格式化y轴为百分比
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_multi_stocks(
        self,
        events: pd.Series,
        codes: List[str],
        event_index: int = 0,
        window_before: int = 10,
        window_after: int = 10,
        metric: str = '收盘价(元)',
        market_close_hour: int = 15,
        title: Optional[str] = None,
        figsize: tuple = (14, 8),
        save_path: Optional[str] = None
    ) -> None:
        """
        类型2B：折线图展示给定股票池在特定事件的 -n~n 累积收益曲线
        所有折线在 t=0, y=0 处相交

        Args:
            events: 事件序列
            codes: 股票代码列表
            event_index: 选择第几个事件（0表示第一个事件）
            window_before: 事件前窗口
            window_after: 事件后窗口
            metric: 价格指标
            market_close_hour: 市场收盘时间
            title: 图表标题
            figsize: 图表尺寸
            save_path: 保存路径
        """
        event_dates = _get_event_dates(events)
        if event_dates.empty:
            print("错误：没有事件")
            return

        if event_index >= len(event_dates):
            print(f"错误：事件索引 {event_index} 超出范围（共 {len(event_dates)} 个事件）")
            return

        selected_event = event_dates[event_index]
        start = (selected_event - timedelta(days=window_before * 3)).strftime('%Y-%m-%d')
        end = (selected_event + timedelta(days=window_after * 3)).strftime('%Y-%m-%d')

        fig, ax = plt.subplots(figsize=figsize)

        # 为每只股票绘制累积收益曲线
        for code in codes:
            try:
                data = self.datafeed.query_time_range(
                    codes=[code],
                    start_date=start,
                    end_date=end,
                    metric=metric
                )

                if data.empty:
                    continue

                prices = data.set_index('datetime')['value'].sort_index()
                prices = prices.astype(float)
                returns = _calc_returns(prices)

                # 获取窗口收益
                window_returns = _get_window_returns(
                    returns, prices, selected_event,
                    window_before, window_after, market_close_hour
                )

                if window_returns is None or window_returns.empty:
                    continue

                # 计算累积收益（以Day 0为起点，归一化为0）
                cumulative = (1 + window_returns).cumprod() - 1
                cumulative = cumulative - cumulative[0]
                # 绘制折线
                ax.plot(cumulative.index, cumulative.values, marker='o', label=code, linewidth=2, markersize=4)

            except Exception:
                continue

        # 添加零点标记
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='事件发生点(Day 0)')

        # 设置标签
        ax.set_xlabel('相对事件发生的天数', fontsize=12)
        ax.set_ylabel('累积收益率', fontsize=12)
        if title is None:
            title = f'事件 {event_index+1} ({selected_event.strftime("%Y-%m-%d %H:%M:%S")}) - 多股票累积收益'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # 格式化y轴为百分比
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_events_lines(
        self,
        events: pd.Series,
        code: str,
        window_before: int = 10,
        window_after: int = 10,
        metric: str = '收盘价(元)',
        market_close_hour: int = 15,
        title: Optional[str] = None,
        figsize: tuple = (14, 8),
        max_events: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        类型3：折线图展示同一标的在多个事件周围的 -n~n 累积收益曲线
        所有折线在 t=0, y=0 处相交

        Args:
            events: 事件序列
            code: 股票代码
            window_before: 事件前窗口
            window_after: 事件后窗口
            metric: 价格指标
            market_close_hour: 市场收盘时间
            title: 图表标题
            figsize: 图表尺寸
            max_events: 最多展示的事件数量，None表示全部展示
            save_path: 保存路径
        """
        event_dates = _get_event_dates(events)
        if event_dates.empty:
            print("错误：没有事件")
            return

        # 限制展示的事件数量
        if max_events is not None and len(event_dates) > max_events:
            event_dates = event_dates[:max_events]
            print(f"注意：限制展示前 {max_events} 个事件")

        fig, ax = plt.subplots(figsize=figsize)

        # 为每个事件绘制累积收益曲线
        for idx, selected_event in enumerate(event_dates):
            try:
                start = (selected_event - timedelta(days=window_before * 3)).strftime('%Y-%m-%d')
                end = (selected_event + timedelta(days=window_after * 3)).strftime('%Y-%m-%d')

                data = self.datafeed.query_time_range(
                    codes=[code],
                    start_date=start,
                    end_date=end,
                    metric=metric
                )

                if data.empty:
                    continue

                prices = data.set_index('datetime')['value'].sort_index()
                prices = prices.astype(float)
                returns = _calc_returns(prices)

                # 获取窗口收益
                window_returns = _get_window_returns(
                    returns, prices, selected_event,
                    window_before, window_after, market_close_hour
                )

                if window_returns is None or window_returns.empty:
                    continue

                # 计算累积收益（归一化为0点）
                cumulative = (1 + window_returns).cumprod() - 1
                # 对齐0点：减去Day 0位置的值，使得所有曲线在Day 0处为0
                if 0 in cumulative.index:
                    day0_loc = cumulative.index.get_loc(0)
                    if isinstance(day0_loc, int) and day0_loc < len(cumulative):
                        cumulative = cumulative - cumulative.iloc[day0_loc]

                # 绘制折线
                event_label = f"事件{idx+1} ({selected_event.strftime('%Y-%m-%d')})"
                ax.plot(cumulative.index, cumulative.values,
                       marker='o', label=event_label, linewidth=1.5, markersize=3, alpha=0.7)

            except Exception as e:
                print(f"警告：事件 {selected_event} 处理失败: {e}")
                continue

        # 添加零点标记
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='事件发生点(Day 0)')

        # 设置标签
        ax.set_xlabel('相对事件发生的天数', fontsize=12)
        ax.set_ylabel('累积收益率', fontsize=12)
        if title is None:
            title = f'{code} - 多事件累积收益对比（共{len(event_dates)}个事件）'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        # 格式化y轴为百分比
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()

        plt.close()


