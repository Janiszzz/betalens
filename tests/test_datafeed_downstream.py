from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from betalens.datafeed.daily import query_daily_market, daily_wide
from betalens.factor.research import (build_rebalance_schedule, weights_on_schedule,
    build_forward_returns, ideal_group_nav)
from betalens.analyst.trades import build_trade_cycles
from betalens.eventstudy.eventstudy import EventStudy, _get_day0_trade_day_loc
from datafeed_support import market_reader, quote
from test_backtest import bars, simulate


def test_daily_market_uses_trade_date_even_if_available_next_day():
    row = quote(at='2024-01-03 00:00', value=100, trade_date=pd.Timestamp('2024-01-02').date(),
                event_date=pd.Timestamp('2024-01-02').date())
    source = SimpleNamespace(_reader=market_reader([row]), pit_policy='strict')
    data = query_daily_market(source, 'market_daily_post_adjusted', ['A'], '2024-01-02', '2024-01-02')
    assert data.iloc[0].trade_date == pd.Timestamp('2024-01-02')
    assert data.iloc[0].available_at == pd.Timestamp('2024-01-03')
    wide = daily_wide(data, 'close')
    assert wide.index[0] == pd.Timestamp('2024-01-02')
    assert wide.attrs['availability_upper']['2024-01-02'].startswith('2024-01-03')


def test_daily_market_supports_stock_fund_and_index_without_share_rules():
    rows = [quote(code=code, asset_type=kind) for code, kind in [('A', 'stock'), ('B', 'fund'), ('C', 'index')]]
    source = SimpleNamespace(_reader=market_reader(rows), pit_policy='strict')
    data = query_daily_market(source, 'market_daily_post_adjusted', ['A', 'B', 'C'], metrics=['close'])
    assert set(data.asset_type) == {'stock', 'fund', 'index'}


def test_schedule_maps_previous_signal_day_and_blocks_late_availability():
    wide = pd.DataFrame({'A': [1.]}, index=pd.to_datetime(['2024-01-02']))
    wide.attrs['availability_upper'] = {'2024-01-02': '2024-01-03T00:00:00'}
    schedule = build_rebalance_schedule(['2024-01-02'], ['2024-01-03'], wides={'x': wide})
    weights = weights_on_schedule(wide, schedule)
    assert weights.index[0] == pd.Timestamp('2024-01-03 00:10')
    wide.attrs['availability_upper']['2024-01-02'] = '2024-01-03T10:00:00'
    rejected = build_rebalance_schedule(['2024-01-02'], ['2024-01-03'], wides={'x': wide})
    assert not rejected.valid.iloc[0]
    assert weights_on_schedule(wide, rejected).empty


def test_forward_price_matching_reports_missing_and_superseded_intervals():
    from betalens.factor.research import forward_return_matrix
    prices = pd.DataFrame({'A': [100., np.nan, 121., 130.], 'B': [0., 100., 110., np.nan]},
        index=pd.bdate_range('2024-01-02', periods=4))
    schedule = build_rebalance_schedule(['2024-01-01', '2024-01-02', '2024-01-03'],
        ['2024-01-02', '2024-01-03', '2024-01-04'])
    forward = build_forward_returns(prices, schedule, metric='open')
    first_a = forward.loc[forward.code.eq('A')].iloc[0]
    assert first_a['return'] == pytest.approx(.21)
    assert first_a.start_ts == pd.Timestamp('2024-01-02 09:30')
    assert first_a.end_ts == pd.Timestamp('2024-01-04 09:30')
    assert forward.loc[forward.code.eq('A')].iloc[1].reason == 'superseded_or_nonpositive_interval'
    assert forward.loc[forward.code.eq('B')].iloc[0].reason == 'invalid_price'
    matrix = forward_return_matrix(prices, schedule, metric='open')
    for code in prices:
        np.testing.assert_allclose(matrix[code], forward.loc[forward.code.eq(code), 'return'], equal_nan=True)


def test_group_nav_and_ic_use_original_schedule_not_risk_events():
    from betalens.factor.stats import calc_ic_from_factor_values
    days = pd.bdate_range('2024-01-02', periods=3)
    prices = pd.DataFrame({'A': [100., 110., 121.], 'B': [100., 100., 100.]}, index=days)
    schedule = build_rebalance_schedule(['2024-01-01', '2024-01-03'], ['2024-01-02', '2024-01-04'])
    factors = pd.DataFrame({'信号日': pd.to_datetime(['2024-01-01'] * 2 + ['2024-01-03'] * 2),
        '股票代码': ['A', 'B'] * 2, '因子值': [2., 1.] * 2, '分组': [0] * 4})
    forward = build_forward_returns(prices, schedule)
    ic = calc_ic_from_factor_values(factors, forward_returns=forward, min_periods=2)
    assert ic.iloc[0] == pytest.approx(1)
    nav, coverage = ideal_group_nav(prices, prices, factors, schedule, 1)
    np.testing.assert_allclose(nav.G1.iloc[:2], [1., 1.05])
    assert nav.G1.iloc[-1] == pytest.approx(1.105)
    assert coverage.coverage.eq(1).all()
    # An arbitrary additional exit price must not become an IC prediction boundary.
    events = pd.DataFrame({'A': [100., 90., 121.], 'B': [100.] * 3}, index=days)
    events.attrs['index_role'] = 'execution_events'
    with pytest.raises(ValueError, match='研究区间'):
        calc_ic_from_factor_values(factors, events)
    pd.testing.assert_series_equal(ic, calc_ic_from_factor_values(factors, events, forward_returns=forward, min_periods=2))


def _log(quantities, prices, fees):
    return pd.DataFrame([dict(datetime=day, code='A', signed_shares=q, shares=abs(q),
        price=p, adjusted_price=p, adjustment_factor=1., fee=fee,
        direction='buy' if q > 0 else 'sell', source='strategy', order_id=i)
        for i, (day, q, p, fee) in enumerate(zip(pd.bdate_range('2024-01-02', periods=len(prices)), quantities, prices, fees))])


def test_complete_cycle_combines_additions_partial_sales_and_fees():
    cycles = build_trade_cycles(_log([1000, 500, -500, -1000], [100, 110, 120, 130], [10] * 4))
    assert len(cycles) == 1
    cycle = cycles.iloc[0]
    assert cycle.status == 'closed'
    assert cycle.entry_value == 155000
    assert cycle.fees == 40
    assert cycle.pnl_net == 34960
    assert cycle.return_net == pytest.approx(34960 / 155000)


def test_direction_reversal_splits_cycle_and_allocates_fee():
    cycles = build_trade_cycles(_log([1000, -1500, 500], [100, 110, 100], [10, 15, 5]))
    assert cycles.side.tolist() == ['long', 'short']
    assert cycles.status.tolist() == ['closed', 'closed']
    np.testing.assert_allclose(cycles.fees, [20, 10])
    np.testing.assert_allclose(cycles.pnl_net, [9980, 4990])


def test_cycles_include_fraction_settlement_and_reconcile_account(simulate):
    panel = bars({'A': [100, 90, 108]}, factors={'A': [1, 10/9, 10/9]})
    targets = pd.DataFrame({'A': [10/11, 0]}, index=pd.to_datetime(['2024-01-02', '2024-01-04']))
    bt = simulate(panel, targets, amount=110000, ptc=.001)
    assert len(bt.trade_cycles) == 1
    assert bt.trade_cycles.iloc[0].status == 'closed'
    assert bt.trade_cycles.iloc[0].pnl_net == pytest.approx(bt.daily_amount.iloc[-1] - bt.initial_amount)
    assert bt.trade_cycles.iloc[0].fees == pytest.approx(bt.rebalance_log.fee.sum())


class DailyFeed:
    def __init__(self, prices):
        self.prices = prices
    def query_daily_market(self, codes, start_date, end_date, metrics):
        series = self.prices[codes[0]]
        return pd.DataFrame({'trade_date': series.index, 'value': series.to_numpy()})


@pytest.fixture
def calendar(monkeypatch):
    monkeypatch.setattr('betalens.eventstudy.eventstudy.get_absolute_trade_days',
        lambda begin, end, period, exchange='SSE': [d.date() for d in pd.bdate_range(begin, end)])


def test_event_holding_compounds_each_asset_then_averages(calendar):
    days = pd.bdate_range('2024-01-01', periods=5)
    feed = DailyFeed({'A': pd.Series([100, 100, 110, 121, 121], index=days),
                      'B': pd.Series([100] * 5, index=days)})
    events = pd.Series([1], index=pd.to_datetime(['2024-01-02 10:00']))
    result = EventStudy(feed).analyze(events, ['A', 'B'], window_before=0, window_after=2,
        holding_periods={'days': [2], 'months': []}, multi_asset_mode='compare')
    assert result['holding_returns_matrix'].iloc[0, 0] == pytest.approx(.105)
    assert result['comparison']['by_code']['A']['holding_returns_matrix'].iloc[0, 0] == pytest.approx(.21)


def test_event_excess_is_difference_of_interval_returns(calendar):
    days = pd.bdate_range('2024-01-01', periods=5)
    feed = DailyFeed({'A': pd.Series([100, 100, 110, 121, 121], index=days),
                     'I': pd.Series([100, 100, 105, 110.25, 110.25], index=days)})
    events = pd.Series([1], index=pd.to_datetime(['2024-01-02 10:00']))
    result = EventStudy(feed).analyze(events, 'A', benchmark_code='I', window_before=0, window_after=2,
        holding_periods={'days': [2], 'months': []})
    assert result['holding_returns_matrix'].iloc[0, 0] == pytest.approx(.21 - .1025)
    assert result['price_matrix'].loc[2].iloc[0] == pytest.approx(.21 - .1025)


def test_event_utc_and_exact_observation_time_use_shanghai_boundary():
    days = pd.bdate_range('2024-01-02', periods=3)
    assert _get_day0_trade_day_loc(days, pd.Timestamp('2024-01-02 07:00Z')) == 1
    assert _get_day0_trade_day_loc(days, pd.Timestamp('2024-01-02 06:59Z')) == 0
    assert _get_day0_trade_day_loc(days, pd.Timestamp('2024-01-02 09:30'), 9.5) == 1


def test_analysis_roundtrip_and_first_month_fees(simulate, tmp_path):
    from betalens.analyst import Analyst
    panel = bars({'A': [100, 110, 90]})
    targets = pd.DataFrame({'A': [.9, 0]}, index=pd.to_datetime(['2024-01-02', '2024-01-04']))
    bt = simulate(panel, targets, amount=100000, ftc=1, ptc=.001)
    direct = Analyst.from_backtest(bt)
    loaded = Analyst.from_excel(bt.dump_to_excel(tmp_path / 'dump.xlsx'))
    pd.testing.assert_series_equal(direct.an.returns, loaded.an.returns, check_names=False, check_freq=False)
    expected, actual = direct.an.summary(), loaded.an.summary()
    assert expected['最大回撤区间'] == actual['最大回撤区间']
    np.testing.assert_allclose(pd.to_numeric(pd.Series(expected), errors='coerce'), pd.to_numeric(pd.Series(actual), errors='coerce'), equal_nan=True)
    assert direct.monthly_table().loc[2024, 1] == pytest.approx(bt.nav.iloc[-1] - 1)
    assert direct.monthly_table().loc[2024, '全年'] == pytest.approx(bt.nav.iloc[-1] - 1)
    assert direct.an.annualized_return() == pytest.approx(bt.nav.iloc[-1] ** (252 / len(bt.daily_returns)) - 1)


def test_window_evaluation_keeps_first_return_and_actual_risk_metrics(simulate):
    from betalens.analyst import BacktestEvaluation
    panel = bars({'A': [100, 90, 80, 80]})
    targets = pd.DataFrame({'A': [.9]}, index=pd.to_datetime(['2024-01-02']))
    bt = simulate(panel, targets, risk={'stop_loss': .08})
    result = BacktestEvaluation.from_backtest(bt).window('2024-01-03', '2024-01-05')
    metrics = result.metrics()
    assert metrics['trade_count'] == 1
    assert metrics['win_rate'] == 0
    assert metrics['open_ratio'] == pytest.approx(1/3)
    assert (1 + result.daily_returns).prod() == pytest.approx(bt.daily_amount.iloc[-1] / bt.daily_amount.iloc[0])
    assert result.daily_returns.iloc[0] < 0


def test_benchmark_uses_adjusted_close_and_aligns_daily_returns(simulate, monkeypatch):
    from betalens.analyst import Analyst
    calls = []
    class BenchmarkFeed:
        def __init__(self, table):
            calls.append(table)
        def query_daily_market(self, codes, start_date, end_date, metrics):
            assert metrics == ['close']
            return pd.DataFrame(dict(trade_date=pd.bdate_range('2024-01-01', periods=4),
                code=['I'] * 4, metric=['close'] * 4, value=[100., 110., 121., 133.1],
                available_at=pd.bdate_range('2024-01-01', periods=4)))
        def close(self):
            pass
    monkeypatch.setattr('betalens.datafeed.Datafeed', BenchmarkFeed)
    bt = simulate(bars({'A': [100, 110, 120]}), pd.DataFrame({'A': [1.]}, index=pd.to_datetime(['2024-01-02'])))
    analyst = Analyst.from_backtest(bt, benchmark_code='I')
    assert calls == ['market_daily_post_adjusted']
    pd.testing.assert_index_equal(analyst.an.bench_returns.index, bt.daily_returns.index)
    np.testing.assert_allclose(analyst.an.bench_returns, .1)


def test_cached_input_availability_survives_memmap_roundtrip(tmp_path):
    from betalens.factor.mining_cache import MiningCache, CacheRequest
    from betalens.factor.mining import _load_cached_data
    wide = pd.DataFrame({'A': [1., 2.]}, index=pd.bdate_range('2024-01-02', periods=2))
    timing = {'2024-01-02': '2024-01-03T00:00:00', '2024-01-03': '2024-01-04T00:00:00'}
    def builder():
        return dict(inputs={'x': wide}, price=wide, execution_price=wide,
            trade_status=wide * 0 + 1, industry_by_scheme={}, pit=None, universe=['A'],
            metadata={'availability': {'x': timing}})
    cache = MiningCache.open_or_build(CacheRequest(tmp_path / 'cache', 'new-daily-axis'), builder)
    restored = _load_cached_data(cache.manifest_path)
    assert restored.inputs['x'].attrs['availability_upper'] == timing


def test_research_ic_is_reproducible_after_excel_roundtrip(simulate, tmp_path):
    from betalens.analyst import Analyst
    panel = bars({code: [100, 100 + i, 100 + i * 2] for i, code in enumerate('ABCDEF', 1)})
    bt = simulate(panel, pd.DataFrame({'A': [.5]}, index=pd.to_datetime(['2024-01-02'])))
    bt.factor_values = pd.DataFrame({'信号日': pd.to_datetime(['2024-01-01'] * 6),
        '股票代码': list('ABCDEF'), '因子值': list(range(1, 7)), '分组': [0] * 6})
    bt.research_schedule = build_rebalance_schedule(['2024-01-01', '2024-01-03'], ['2024-01-02', '2024-01-04'])
    prices = panel.pivot(index='trade_date', columns='code', values='close')
    bt.research_forward_returns = build_forward_returns(prices, bt.research_schedule)
    direct = Analyst.from_backtest(bt)
    loaded = Analyst.from_excel(bt.dump_to_excel(tmp_path / 'research.xlsx'))
    pd.testing.assert_series_equal(direct.an.ic_series, loaded.an.ic_series, check_freq=False)
    assert direct.an.ic_series.iloc[0] == pytest.approx(1.)
    # v2 files without explicit research periods never infer IC from event prices.
    del bt.research_forward_returns
    missing = Analyst.from_excel(bt.dump_to_excel(tmp_path / 'missing.xlsx'))
    assert missing.an.ic_series is None


def test_fraction_only_close_is_accounted_without_a_trade_row(simulate):
    factor = 1.015
    panel = bars({'A': [100, 100/factor, 100/factor, 100/factor]}, factors={'A': [1, factor, factor, factor]})
    targets = pd.DataFrame({'A': [1., 1.5/101.5, .5/101.5, 0.]}, index=pd.bdate_range('2024-01-02', periods=4))
    bt = simulate(panel, targets, amount=10000)
    assert len(bt.synthetic_settlements) == 1
    assert bt.synthetic_settlements.iloc[0].datetime not in set(bt.rebalance_log.datetime)
    assert bt.trade_cycles.iloc[0].status == 'closed'
    assert bt.trade_cycles.iloc[0].pnl_net == pytest.approx(0, abs=1e-8)


def test_actual_metrics_match_analyst_and_count_constant_return_days(simulate):
    from betalens.analyst import Analyst, BacktestEvaluation
    from betalens.analyst.metrics import performance_from_returns
    constant = performance_from_returns(pd.Series([.01, .01, .01]))
    assert constant['ann_ret'] == pytest.approx(1.01 ** 252 - 1)
    assert constant['n_days'] == 3
    loss = performance_from_returns(pd.Series([-.1, 0.]))
    assert loss['mdd'] == pytest.approx(.1)
    bt = simulate(bars({'A': [100, 90, 80, 80]}), pd.DataFrame({'A': [.9]}, index=pd.to_datetime(['2024-01-02'])), risk={'stop_loss': .08})
    report = Analyst.from_backtest(bt).an.summary()
    actual = BacktestEvaluation.from_backtest(bt).metrics()
    for key, label in [('ann_ret', '年化收益'), ('ann_vol', '年化波动率'), ('sharpe', '夏普比率'), ('mdd', '最大回撤'),
                       ('turnover', '平均单边换手'), ('trade_count', '交易次数'), ('win_rate', '交易胜率'), ('odds', '交易赔率')]:
        assert actual[key] == pytest.approx(report[label])


def test_attached_research_uses_full_factor_universe_despite_risk_exit(simulate, monkeypatch, tmp_path):
    from betalens.factor.research import attach_research
    from betalens.analyst import Analyst
    codes = list('ABCDEF')
    panel = bars({code: [100, 90 + i, 80 + i * 2, 90 + i * 2]
                  for i, code in enumerate(codes)})
    targets = pd.DataFrame({'A': [.9]}, index=pd.to_datetime(['2024-01-02']))
    bt = simulate(panel, targets, risk={'stop_loss': .08})
    assert bt.trade_cycles.iloc[0].status == 'closed'
    calls = []
    class ResearchFeed:
        def __init__(self, table):
            assert table == 'market_daily_post_adjusted'
        def query_daily_market(self, requested, start, end, metrics):
            calls.extend(requested)
            data = panel.loc[panel.code.isin(requested)]
            return data.melt(id_vars=['code', 'trade_date', 'available_at'], value_vars=metrics,
                var_name='metric', value_name='value')
        def close(self):
            pass
    monkeypatch.setattr('betalens.datafeed.Datafeed', ResearchFeed)
    schedule = build_rebalance_schedule(['2024-01-01', '2024-01-03'], ['2024-01-02', '2024-01-04'])
    factors = pd.DataFrame({'信号日': pd.to_datetime(['2024-01-01'] * 6 + ['2024-01-03'] * 6),
        '股票代码': codes * 2, '因子值': list(range(6)) * 2, '分组': [0] * 12})
    annotated = attach_research(bt, factors, schedule, 1)
    assert set(calls) == set(codes)
    assert set(annotated.columns) >= {'signal_date', 'signal_ts', 'target_ts', 'forward_end_ts'}
    direct = Analyst.from_backtest(bt)
    assert direct.an.ic_series.iloc[0] == pytest.approx(1.)
    expected = panel.pivot(index='trade_date', columns='code', values='close').loc['2024-01-04'].mean() / 100
    assert bt.research_group_nav.G1.loc['2024-01-04'].iloc[0] == pytest.approx(expected)
    loaded = Analyst.from_excel(bt.dump_to_excel(tmp_path / 'attached.xlsx'))
    pd.testing.assert_series_equal(direct.an.ic_series, loaded.an.ic_series)
    # Canonical research columns also work without the original display labels.
    prices = panel.pivot(index='trade_date', columns='code', values='close')
    canonical = annotated.drop(columns='信号日').rename(columns={'股票代码': 'code', '分组': 'group'})
    canonical_nav, _ = ideal_group_nav(prices, prices, canonical, schedule, 1)
    pd.testing.assert_frame_equal(canonical_nav, bt.research_group_nav)


def test_annual_and_custom_reports_keep_first_return_across_year_boundary():
    from betalens.analyst import PortfolioAnalyzer, ReportExporter
    returns = pd.Series([-.01, .1, .2], index=pd.to_datetime(['2023-12-29 15:00', '2024-01-02 15:00', '2024-01-03 15:00']))
    nav = pd.concat([pd.Series([1.], index=pd.to_datetime(['2023-12-29'])), (1 + returns).cumprod()])
    exporter = ReportExporter(PortfolioAnalyzer(nav, daily_returns=returns))
    annual = exporter.generate_annual_report()
    assert annual.loc[2023, 'Return'] == pytest.approx(-.01)
    assert annual.loc[2024, 'Return'] == pytest.approx(.32)
    custom = exporter.generate_custom_report('2024-01-02', '2024-01-03')
    assert custom['累计收益'].iloc[0] == pytest.approx(.32)
    assert ReportExporter(exporter.analyzer, start_date='2024-01-02').analyzer.total_return() == pytest.approx(.32)


def test_legacy_accounting_is_disclosed_and_excluded_from_net_statistics(tmp_path):
    from betalens.analyst import Analyst, PortfolioAnalyzer
    from betalens.analyst.metrics import annual_trade_performance
    legacy = pd.DataFrame(dict(code=['A', 'A'], datetime=pd.to_datetime(['2024-01-02', '2024-01-03']),
        direction=['buy', 'sell'], price=[100., 110.]))
    nav = pd.Series([1., 1.1], index=pd.to_datetime(['2024-01-02', '2024-01-03']))
    report = Analyst(PortfolioAnalyzer(nav, rebalance_log=legacy))
    assert report.an.summary()['交易次数'] == 0
    assert report.an.metadata['trade_accounting'] == 'legacy_incomplete'
    assert annual_trade_performance(report.trade_pnl_df()).empty
    destination = tmp_path / 'report.xlsx'
    report.to_excel(destination)
    with pd.ExcelFile(destination) as book:
        assert {'评价口径', '诊断', '逐笔盈亏'} <= set(book.sheet_names)
        summary = pd.read_excel(book, sheet_name='指标汇总')
        assert '交易次数' in summary['指标'].tolist()
