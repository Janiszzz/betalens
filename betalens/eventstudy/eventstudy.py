"""
事件研究模块 - 用于分析特定事件前后的收益率表现

betalens API 使用:
    - datafeed.query_time_range(): 查询时间范围内的数据
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Literal
import matplotlib.pyplot as plt

from betalens.datafeed import get_absolute_trade_days

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['simhei']  # 指定默认字体 (黑体)
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def _get_event_dates(events: pd.Series) -> pd.DatetimeIndex:
    """从事件序列中提取事件发生日期"""
    return events[events == 1].index


def _calc_returns(prices: pd.Series) -> pd.Series:
    """计算日收益率"""
    return prices.pct_change(fill_method=None)


def _normalize_date(value: pd.Timestamp) -> pd.Timestamp:
    """Normalize a timestamp to a timezone-naive calendar date."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _get_day0_trade_day_loc(
    trade_days: pd.DatetimeIndex,
    event_date: pd.Timestamp,
    market_close_hour: int = 15,
) -> Optional[int]:
    """Locate the event's cost-price day on the standard trade calendar."""
    event_timestamp = pd.Timestamp(event_date)
    event_day = _normalize_date(event_timestamp)
    side = "left" if event_timestamp.hour < market_close_hour else "right"
    location = int(trade_days.searchsorted(event_day, side=side))
    if location >= len(trade_days):
        return None
    return location


def _standard_trade_window(
    event_dates: pd.DatetimeIndex,
    window_before: int,
    window_after: int,
    market_close_hour: int,
    exchange: str,
) -> pd.DatetimeIndex:
    """Load standard trade days covering every event window that fits."""
    if window_before < 0 or window_after < 0:
        raise ValueError("window_before/window_after 不能为负数")

    first_event = min(_normalize_date(value) for value in event_dates)
    last_event = max(_normalize_date(value) for value in event_dates)
    padding = max(window_before, window_after + 1, 5) * 3 + 14

    for _ in range(6):
        begin = first_event - pd.Timedelta(days=padding)
        end = last_event + pd.Timedelta(days=padding)
        raw_days = get_absolute_trade_days(begin, end, "D", exchange=exchange)
        trade_days = pd.DatetimeIndex(pd.to_datetime(raw_days)).normalize().sort_values().unique()
        if trade_days.empty:
            padding *= 2
            continue
        positions = [
            _get_day0_trade_day_loc(trade_days, event_date, market_close_hour)
            for event_date in event_dates
        ]
        valid_positions = [
            int(position)
            for position in positions
            if position is not None
            and position >= window_before + 1
            and position + window_after < len(trade_days)
        ]
        if valid_positions:
            first_position = min(position - window_before - 1 for position in valid_positions)
            last_position = max(position + window_after for position in valid_positions)
            return trade_days[first_position:last_position + 1]
        padding *= 2

    # Some events can fall too close to the available calendar boundary.  They
    # are ignored by the window builders; an empty calendar lets ``analyze``
    # return a normal no-data result when none of the events is usable.
    return pd.DatetimeIndex([])


def _prices_on_trade_days(
    prices: pd.Series,
    trade_days: pd.DatetimeIndex,
) -> pd.Series:
    """Align observed prices to the standard calendar without shifting gaps."""
    normalized = prices.copy()
    normalized.index = pd.DatetimeIndex(
        [_normalize_date(value) for value in pd.to_datetime(normalized.index)]
    )
    normalized = normalized.groupby(level=0).last().sort_index().astype(float)
    return normalized.reindex(trade_days)


def _get_window_returns(
    prices: pd.Series,
    event_date: pd.Timestamp,
    window_before: int,
    window_after: int,
    market_close_hour: int = 15,
    trade_days: Optional[pd.DatetimeIndex] = None,
) -> Optional[pd.Series]:
    """Return the observed daily close-to-close returns around an event."""
    if trade_days is not None:
        calendar = pd.DatetimeIndex(trade_days).normalize().sort_values().unique()
        anchor_loc = _get_day0_trade_day_loc(
            calendar, event_date, market_close_hour
        )
        if anchor_loc is None:
            return None

        calendar_prices = _prices_on_trade_days(prices, calendar)
        calendar_returns = _calc_returns(calendar_prices)
        if (
            anchor_loc - window_before < 0
            or anchor_loc + window_after >= len(calendar_returns)
        ):
            return None
        relative_days = pd.RangeIndex(-window_before, window_after + 1)
        values = []
        for relative_day in relative_days:
            return_loc = anchor_loc + int(relative_day)
            values.append(calendar_returns.iloc[return_loc])
        return pd.Series(values, index=relative_days, dtype=float)

    raise ValueError("trade_days 是必需参数")


def _get_window_prices(
    prices: pd.Series,
    event_date: pd.Timestamp,
    window_before: int,
    window_after: int,
    market_close_hour: int,
    trade_days: pd.DatetimeIndex,
) -> Optional[pd.Series]:
    """Return close prices normalized to zero on the event anchor day."""
    calendar = pd.DatetimeIndex(trade_days).normalize().sort_values().unique()
    anchor_loc = _get_day0_trade_day_loc(calendar, event_date, market_close_hour)
    if anchor_loc is None:
        return None
    calendar_prices = _prices_on_trade_days(prices, calendar)
    start = anchor_loc - window_before
    end = anchor_loc + window_after + 1
    if start < 0 or end > len(calendar_prices):
        return None
    window = calendar_prices.iloc[start:end].copy()
    window.index = pd.RangeIndex(-window_before, window_after + 1)
    day0 = window.loc[0]
    if pd.isna(day0) or day0 == 0:
        return window * np.nan
    return window / day0 - 1


def _stats_frame(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the standard event-study statistics for each relative day."""
    stats = {day: _compute_stats(returns_df.loc[day]) for day in returns_df.index}
    result = pd.DataFrame(stats).T
    result.index.name = 'day'
    return result


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
        cols = [i for i, m in enumerate(mask) if m and i in returns_df.columns]
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
        market_close_hour: int,
        trade_days: pd.DatetimeIndex,
        benchmark_returns: Optional[pd.Series] = None,
        benchmark_prices: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
        """获取单个股票在所有事件窗口的收益率矩阵

        Returns:
            (returns, normalized prices, error)
        """
        try:
            data = self.datafeed.query_time_range(
                codes=[code],
                start_date=start,
                end_date=end,
                metric=metric
            )

            if data.empty:
                return pd.DataFrame(), pd.DataFrame(), 'no data'

            prices = data.set_index('datetime')['value'].sort_index()
            prices = prices.astype(float)
            event_windows = {}
            price_windows = {}
            for event_id, ed in enumerate(event_dates):
                wr = _get_window_returns(
                    prices, ed, window_before, window_after,
                    market_close_hour, trade_days,
                )
                if wr is None or wr.empty or wr.isna().all():
                    continue
                wp = _get_window_prices(
                    prices, ed, window_before, window_after,
                    market_close_hour, trade_days,
                )
                if benchmark_returns is not None and benchmark_prices is not None:
                    benchmark_wr = _get_window_returns(
                        benchmark_prices,
                        ed,
                        window_before,
                        window_after,
                        market_close_hour,
                        trade_days,
                    )
                    if benchmark_wr is None or benchmark_wr.empty or benchmark_wr.isna().all():
                        continue
                    wr = wr.subtract(benchmark_wr, fill_value=np.nan).dropna()
                    if wr.empty:
                        continue
                event_windows[event_id] = wr
                price_windows[event_id] = wp

            if not event_windows:
                return pd.DataFrame(), pd.DataFrame(), 'no matching events'
            result = pd.DataFrame(event_windows).sort_index()
            # Keep columns tied to the input event order.  A target can be
            # missing one event window without shifting every later event.
            result = result.reindex(columns=range(len(event_dates)))
            result.columns.name = 'event_id'
            prices_result = pd.DataFrame(price_windows).sort_index().reindex(columns=range(len(event_dates)))
            prices_result.columns.name = 'event_id'
            return result, prices_result, None

        except Exception as exc:
            return pd.DataFrame(), pd.DataFrame(), str(exc)

    def _calc_holding_returns(
        self,
        returns_df: pd.DataFrame,
        holding_periods: Optional[dict] = None,
        holding_start_offset: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate fixed holding returns and their cross-event statistics."""
        periods = holding_periods or {
            'days': [1, 2, 3, 4, 5],
            'months': [1, 3, 6, 9, 12],
        }
        if returns_df.empty or holding_start_offset not in returns_df.index:
            return pd.DataFrame(), pd.DataFrame()

        targets = [
            (f'{int(day)}日', int(day))
            for day in periods.get('days', [])
        ]
        targets.extend(
            (f'{int(month)}个月', int(month) * 21)
            for month in periods.get('months', [])
        )

        matrix_rows = {}
        labels = {}
        for label, offset in targets:
            target_day = holding_start_offset + offset
            if target_day not in returns_df.index:
                continue
            period_returns = returns_df.loc[holding_start_offset + 1:target_day]
            matrix_rows[target_day] = period_returns.add(1).prod(axis=0, skipna=False) - 1
            labels[target_day] = label

        if not matrix_rows:
            return pd.DataFrame(), pd.DataFrame()

        matrix = pd.DataFrame(matrix_rows).T
        matrix.index.name = 'holding_day'
        stats = _stats_frame(matrix)
        stats.index.name = 'holding_day'
        stats.insert(0, 'holding_period', [labels[day] for day in stats.index])
        return matrix, stats
    
    def analyze(
        self,
        events: pd.Series,
        code: Union[str, List[str]],
        window_before: int = 5,
        window_after: int = 5,
        metric: str = 'close',
        periods: Optional[pd.Series] = None,
        holding_periods: Optional[dict] = None,
        holding_start_offset: int = 0,
        market_close_hour: int = 15,
        benchmark_code: Optional[str] = None,
        multi_asset_mode: Literal['aggregate', 'compare'] = 'aggregate',
        exchange: str = 'SHSE',
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
            holding_periods: 持有期字典，如{'days': [1,2,3,4,5], 'months': [1,3,6,9,12]}
            holding_start_offset: 持有起点偏移天数，0表示从Day 0开始，n表示从Day n开始
            market_close_hour: 市场收盘时间（小时），默认15点
            benchmark_code: 可选的业绩比较基准代码，如提供则计算超额收益=持有标的收益-基准收益
            multi_asset_mode: 多标的处理模式，aggregate=等权平均，compare=同时返回逐标的比较
            exchange: 标准交易日历的交易所代码，默认SHSE

        Returns:
            包含 daily_stats, holding_stats, price_matrix 和 returns_matrix 的字典
            多标的模式额外包含: stock_returns_dict (每个股票的收益矩阵)

        Note:
            Day 0日期判断：事件在收盘前使用当日，收盘后使用下一交易日。
            Day 0收益为该交易日自身的收盘价涨跌幅。
        """
        event_dates = _get_event_dates(events)
        if event_dates.empty:
            return {'error': 'no events'}
        if multi_asset_mode not in {'aggregate', 'compare'}:
            raise ValueError(f"不支持的多标的模式: {multi_asset_mode}")
        holding_periods = holding_periods or {
            'days': [1, 2, 3, 4, 5],
            'months': [1, 3, 6, 9, 12],
        }
        holding_horizon = max(
            [0]
            + [int(day) for day in holding_periods.get('days', [])]
            + [int(month) * 21 for month in holding_periods.get('months', [])]
        )
        calculation_after = max(window_after, holding_start_offset + holding_horizon)
        trade_days = _standard_trade_window(
            event_dates, window_before, calculation_after, market_close_hour, exchange
        )
        if trade_days.empty:
            return {
                'daily_stats': pd.DataFrame(),
                'holding_stats': pd.DataFrame(),
                'event_count': 0,
                'event_dates': pd.DatetimeIndex(event_dates),
                'trade_days': trade_days,
                'returns_matrix': pd.DataFrame(),
                'holding_returns_matrix': pd.DataFrame(),
                'price_matrix': pd.DataFrame(),
            }
        start = trade_days[0].strftime('%Y-%m-%d')
        end = trade_days[-1].strftime('%Y-%m-%d')

        # 判断是单标的还是多标的模式
        is_multi_stock = isinstance(code, list)
        if multi_asset_mode == 'compare' and (not is_multi_stock or len(code) < 2):
            raise ValueError("compare 模式至少需要两个标的代码")

        benchmark_returns = None
        benchmark_prices = None
        if benchmark_code:
            benchmark_data = self.datafeed.query_time_range(
                codes=[benchmark_code],
                start_date=start,
                end_date=end,
                metric=metric,
            )
            if benchmark_data.empty:
                return {'error': 'no benchmark data'}
            benchmark_prices = benchmark_data.set_index('datetime')['value'].sort_index().astype(float)
            benchmark_returns = _calc_returns(benchmark_prices)

        codes = code if is_multi_stock else [code]
        stock_returns_full = {}
        stock_price_dict = {}
        valid_codes = []
        skipped_codes = []
        for stock_code in codes:
            stock_returns, stock_prices, error = self._get_stock_window_returns(
                stock_code, event_dates, start, end, metric,
                window_before, calculation_after, market_close_hour,
                trade_days, benchmark_returns, benchmark_prices,
            )
            if stock_returns.empty:
                skipped_codes.append({'code': stock_code, 'reason': error or 'no valid data'})
                continue
            stock_returns_full[stock_code] = stock_returns
            stock_price_dict[stock_code] = stock_prices.loc[-window_before:window_after]
            valid_codes.append(stock_code)

        if not stock_returns_full:
            return {'error': 'no valid stock data'}

        stacked_returns = pd.concat(stock_returns_full, names=['code', 'day'])
        returns_full = stacked_returns.groupby(level='day').mean().sort_index()
        returns_full = returns_full.dropna(axis=1, how='all')
        returns_df = returns_full.loc[-window_before:window_after]
        stacked_prices = pd.concat(stock_price_dict, names=['code', 'day'])
        price_matrix = stacked_prices.groupby(level='day').mean().sort_index()
        price_matrix = price_matrix.reindex(columns=returns_full.columns)

        overall_stats = _stats_frame(returns_df)
        holding_returns, holding_stats = self._calc_holding_returns(
            returns_full, holding_periods, holding_start_offset
        )
        
        result = {
            'daily_stats': overall_stats,
            'holding_stats': holding_stats,
            'event_count': returns_df.shape[1] if not returns_df.empty else 0,
            'event_dates': pd.DatetimeIndex(event_dates),
            'trade_days': trade_days,
            'returns_matrix': returns_df,
            'holding_returns_matrix': holding_returns,
            'price_matrix': price_matrix,
        }

        # 多标的模式：添加每个股票的收益矩阵
        if is_multi_stock:
            result['stock_returns_dict'] = {
                stock_code: matrix.loc[-window_before:window_after]
                for stock_code, matrix in stock_returns_full.items()
            }
            result['stock_price_dict'] = stock_price_dict
            result['valid_codes'] = valid_codes
            result['skipped_codes'] = skipped_codes

            if multi_asset_mode == 'compare':
                by_code = {}
                total_events = len(event_dates)
                for stock_code, stock_returns in stock_returns_full.items():
                    stock_daily_stats = _stats_frame(stock_returns)
                    stock_holding_returns, stock_holding_stats = self._calc_holding_returns(
                        stock_returns, holding_periods, holding_start_offset,
                    )
                    event_ids = [
                        int(event_id)
                        for event_id in stock_returns.columns
                        if stock_returns[event_id].notna().any()
                    ]
                    by_code[stock_code] = {
                        'event_count': len(event_ids),
                        'coverage': len(event_ids) / total_events if total_events else 0.0,
                        'event_ids': event_ids,
                        'event_dates': pd.DatetimeIndex([event_dates[event_id] for event_id in event_ids]),
                        'daily_stats': stock_daily_stats.loc[-window_before:window_after],
                        'holding_stats': stock_holding_stats,
                        'returns_matrix': stock_returns.loc[-window_before:window_after],
                        'holding_returns_matrix': stock_holding_returns,
                        'price_matrix': stock_price_dict[stock_code],
                    }
                result['comparison'] = {
                    'mode': 'compare',
                    'events': [
                        {'event_id': event_id, 'event_date': event_date}
                        for event_id, event_date in enumerate(event_dates)
                    ],
                    'valid_codes': valid_codes,
                    'skipped_codes': skipped_codes,
                    'by_code': by_code,
                }

        if periods is not None and not is_multi_stock:
            # 多标的模式下暂不支持periods分析
            period_stats = _compute_period_stats(
                returns_df,
                pd.DatetimeIndex(event_dates),
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
        ax.axvline(x=0, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='持有首日(Day 0)')

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

    def plot_multi_stocks(
        self,
        events: pd.Series,
        codes: List[str],
        event_index: int = 0,
        window_before: int = 10,
        window_after: int = 10,
        metric: str = 'close',
        market_close_hour: int = 15,
        title: Optional[str] = None,
        figsize: tuple = (14, 8),
        save_path: Optional[str] = None,
        exchange: str = 'SHSE',
    ) -> None:
        """
        折线图展示给定股票池在特定事件的归一化价格曲线。

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
        selected_events = pd.Series(
            [1], index=pd.DatetimeIndex([selected_event]), dtype=int
        )

        fig, ax = plt.subplots(figsize=figsize)

        if len(codes) >= 2:
            result = self.analyze(
                selected_events, codes, window_before, window_after, metric,
                market_close_hour=market_close_hour,
                multi_asset_mode='compare', exchange=exchange,
            )
            matrices = {
                code: item['price_matrix']
                for code, item in result.get('comparison', {}).get('by_code', {}).items()
            }
        elif codes:
            result = self.analyze(
                selected_events, codes[0], window_before, window_after, metric,
                market_close_hour=market_close_hour, exchange=exchange,
            )
            matrices = {codes[0]: result.get('price_matrix', pd.DataFrame())}
        else:
            matrices = {}

        for code, price_matrix in matrices.items():
            if price_matrix.empty:
                continue
            relative_price = price_matrix.iloc[:, 0]
            ax.plot(
                relative_price.index, relative_price.values, marker='o', label=code,
                linewidth=2, markersize=4,
            )

        # 添加零点标记
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='事件日(Day 0)')

        # 设置标签
        ax.set_xlabel('相对事件发生的天数', fontsize=12)
        ax.set_ylabel('相对价格（Day 0 = 0）', fontsize=12)
        if title is None:
            title = f'事件 {event_index+1} ({selected_event.strftime("%Y-%m-%d %H:%M:%S")}) - 多股票价格走势'
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
        metric: str = 'close',
        market_close_hour: int = 15,
        title: Optional[str] = None,
        figsize: tuple = (14, 8),
        max_events: Optional[int] = None,
        save_path: Optional[str] = None,
        exchange: str = 'SHSE',
    ) -> None:
        """
        折线图叠加同一标的在多个事件周围的归一化价格曲线。

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

        result = self.analyze(
            events.loc[event_dates], code, window_before, window_after, metric,
            market_close_hour=market_close_hour, exchange=exchange,
        )
        price_matrix = result.get('price_matrix', pd.DataFrame())
        result_dates = result.get('event_dates', pd.DatetimeIndex([]))

        fig, ax = plt.subplots(figsize=figsize)
        for event_id in price_matrix.columns:
            relative_price = price_matrix[event_id]
            event_position = int(event_id)
            event_date = result_dates[event_position]
            event_label = f"事件{event_position + 1} ({event_date.strftime('%Y-%m-%d')})"
            ax.plot(
                relative_price.index, relative_price.values, marker='o', label=event_label,
                linewidth=1.5, markersize=3, alpha=0.7,
            )

        # 添加零点标记
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='事件日(Day 0)')

        # 设置标签
        ax.set_xlabel('相对事件发生的天数', fontsize=12)
        ax.set_ylabel('相对价格（Day 0 = 0）', fontsize=12)
        if title is None:
            title = f'{code} - 多事件价格走势（共{len(event_dates)}个事件）'
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


