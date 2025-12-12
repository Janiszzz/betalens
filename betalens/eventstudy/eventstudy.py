import pandas as pd
import numpy as np
from typing import Optional, List, Union
from datetime import timedelta


def _get_event_dates(events: pd.Series) -> pd.DatetimeIndex:
    return events[events == 1].index


def _calc_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change()


def _get_window_returns(
    returns: pd.Series,
    event_date: pd.Timestamp,
    window_before: int,
    window_after: int
) -> Optional[pd.Series]:
    try:
        loc = returns.index.get_loc(event_date)
        start = max(0, loc - window_before)
        end = min(len(returns), loc + window_after + 1)
        window_ret = returns.iloc[start:end].copy()
        window_ret.index = range(-(loc - start), end - loc)
        return window_ret
    except KeyError:
        return None


def _aggregate_window_returns(
    all_returns: List[pd.Series]
) -> pd.DataFrame:
    if not all_returns:
        return pd.DataFrame()
    df = pd.DataFrame(all_returns).T
    df.columns = range(len(all_returns))
    return df


def _compute_stats(returns: pd.Series) -> dict:
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
    def __init__(self, datafeed):
        self.datafeed = datafeed
    
    def analyze(
        self,
        events: pd.Series,
        code: str,
        window_before: int = 5,
        window_after: int = 5,
        metric: str = '收盘价(元)',
        periods: Optional[pd.Series] = None
    ) -> dict:
        event_dates = _get_event_dates(events)
        if event_dates.empty:
            return {'error': 'no events'}
        
        start = (event_dates.min() - timedelta(days=window_before * 3)).strftime('%Y-%m-%d')
        end = (event_dates.max() + timedelta(days=window_after * 3)).strftime('%Y-%m-%d')
        
        data = self.datafeed.query_time_range(
            codes=[code],
            start_date=start,
            end_date=end,
            metric=metric
        )
        
        if data.empty:
            return {'error': 'no data'}
        
        prices = data.set_index('datetime')['value'].sort_index()
        returns = _calc_returns(prices)
        
        all_window_returns = []
        valid_event_dates = []
        for ed in event_dates:
            wr = _get_window_returns(returns, ed, window_before, window_after)
            if wr is not None and not wr.empty:
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
        
        cum_returns = returns_df.add(1).cumprod() - 1
        cum_stats = {}
        for day in cum_returns.index:
            cum_stats[day] = _compute_stats(cum_returns.loc[day])
        cumulative_stats = pd.DataFrame(cum_stats).T
        cumulative_stats.index.name = 'day'
        
        result = {
            'daily_stats': overall_stats,
            'cumulative_stats': cumulative_stats,
            'event_count': len(valid_event_dates),
            'returns_matrix': returns_df
        }
        
        if periods is not None:
            period_stats = _compute_period_stats(
                returns_df, 
                pd.DatetimeIndex(valid_event_dates), 
                periods
            )
            result['period_stats'] = period_stats
        
        return result

