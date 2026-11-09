from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betalens.backtest import BacktestBase, BacktestDataError, RiskConfig
from betalens.backtest.execution import round_order, share_rule
from betalens.backtest import backtest as module


def bars(prices, *, opens=None, factors=None, markets=None, start="2024-01-02"):
    days = pd.bdate_range(start, periods=len(next(iter(prices.values()))))
    records = []
    for code, values in prices.items():
        for i, (day, close) in enumerate(zip(days, values)):
            factor = (factors or {}).get(code, [1.0] * len(days))[i]
            opening = (opens or {}).get(code, values)[i]
            records.append(dict(code=code, trade_date=day, market=(markets or {}).get(code, "主板"),
                adjustment_factor=factor, close=close * factor, raw_close=close,
                open=opening * factor, raw_open=opening,
                available_at=day + pd.Timedelta(hours=16), availability_basis="verified_historical_version"))
    return pd.DataFrame(records)


@pytest.fixture
def simulate(monkeypatch):
    storage = {}

    class Feed:
        pit_policy = "research"

        def __init__(self, *args, **kwargs):
            pass

        def query_backtest_bars(self, codes, start_date, end_date, metrics):
            result = storage["bars"].copy()
            return result[result.code.isin(codes)]

        def query_backtest_calendar(self, start, end):
            return pd.bdate_range(start, end)

        def query_trade_status(self, params):
            return pd.DataFrame([dict(code=code, datetime=pd.Timestamp(ts),
                value=storage.get("status", {}).get((pd.Timestamp(ts), code), 1))
                for ts in params["dates"] for code in params["codes"]])

        def close(self):
            pass

    monkeypatch.setattr(module, "Datafeed", Feed)

    def run(panel, weights, *, status=None, **kwargs):
        storage.update(bars=panel, status=status or {})
        options = dict(symbol="test", amount=1_000_000, check_trade_status=False,
                       end_date=str(panel.trade_date.max().date()), verbose=False)
        options.update(kwargs)
        return BacktestBase(weights, **options)
    return run


def test_sparse_targets_drift_without_daily_rebalancing(simulate):
    panel = bars({"A": [100, 110, 121], "B": [100, 100, 100]})
    weights = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=pd.to_datetime(["2024-01-02"]))
    bt = simulate(panel, weights)
    np.testing.assert_allclose(bt.daily_amount, [1_000_000, 1_050_000, 1_105_000])
    np.testing.assert_allclose(bt.daily_weight.A, [0.5, 55 / 105, 60.5 / 110.5])
    np.testing.assert_allclose(bt.position.A, 5000)
    assert len(bt.rebalance_log) == 2
    np.testing.assert_allclose(bt.daily_pnl_total, bt.daily_amount.diff().fillna(0))


def test_daily_rows_rebalance_even_if_targets_repeat(simulate):
    panel = bars({"A": [100, 200, 400], "B": [100, 100, 100]})
    weights = pd.DataFrame({"A": [0.5] * 3, "B": [0.5] * 3}, index=pd.bdate_range("2024-01-02", periods=3))
    bt = simulate(panel, weights, amount=2_000_000)
    np.testing.assert_allclose(bt.daily_amount, [2_000_000, 3_000_000, 4_500_000])
    assert len(bt.actual_weight) == 3


def test_buy_and_hold_nav_matches_normalized_price_series(simulate):
    prices = [100, 110, 90, 120, 130]
    panel = bars({"A": prices})
    weights = pd.DataFrame({"A": 1.0}, index=pd.bdate_range("2024-01-02", periods=5))
    bt = simulate(panel, weights, amount=100_000)
    np.testing.assert_allclose(bt.nav.iloc[1:], np.asarray(prices) / prices[0])
    np.testing.assert_allclose(bt.position.A, 1000)
    assert len(bt.rebalance_log) == 1


def test_timestamp_after_open_matches_next_open_and_splits_returns(simulate):
    panel = bars({"A": [110, 130, 140]}, opens={"A": [100, 120, 140]})
    weights = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-02 10:00"]))
    bt = simulate(panel, weights, metric="open", amount=120_000)
    assert bt.rebalance_log.iloc[0].datetime == pd.Timestamp("2024-01-03 09:30")
    np.testing.assert_allclose(bt.daily_amount, [120_000, 130_000, 140_000])


def test_exact_open_is_allowed_and_weekend_goes_forward(simulate):
    panel = bars({"A": [100, 100]}, start="2024-01-08")
    weights = pd.DataFrame({"A": [0.5, 1.0]}, index=pd.to_datetime(["2024-01-06 10:00", "2024-01-08 09:30"]))
    bt = simulate(panel, weights, metric="open")
    assert bt.order_log.iloc[0].status == "superseded"
    assert len(bt.rebalance_log) == 1
    assert bt.rebalance_log.iloc[0].datetime == pd.Timestamp("2024-01-08 09:30")


def test_explicit_end_keeps_unfilled_target(simulate):
    panel = bars({"A": [100, 100]})
    weights = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-02 10:00"]))
    bt = simulate(panel, weights, metric="open", end_date="2024-01-02")
    assert bt.rebalance_log.empty
    assert bt.order_log.iloc[0].status == "pending"
    np.testing.assert_allclose(bt.nav, 1)


def test_default_end_includes_last_matching_execution(simulate):
    panel = bars({"A": [100, 100]})
    weights = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-02 10:00"]))
    bt = simulate(panel, weights, metric="open", end_date=None)
    assert bt.end == pd.Timestamp("2024-01-03 15:00")
    assert len(bt.rebalance_log) == 1


@pytest.mark.parametrize("market,requested,expected", [("主板", 199, 100), ("创业板", 299, 200), ("科创板", 199, 0), ("科创板", 201, 201)])
def test_board_minimum_and_increment(market, requested, expected):
    assert round_order(requested, 0, share_rule(market)) == (expected, 0)


def test_star_minimum_applies_to_trade_delta_and_full_odd_close(simulate):
    panel = bars({"A": [100, 100, 100]}, markets={"A": "科创板"})
    weights = pd.DataFrame({"A": [0.2, 0.25, 0]}, index=pd.bdate_range("2024-01-02", periods=3))
    bt = simulate(panel, weights, amount=100_000)
    assert bt.rebalance_log.shares.tolist() == [200, 200]
    assert bt.position.A.tolist() == [200, 200, 0]
    assert round_order(-199, 199, share_rule("科创板")) == (-199, 0)


def test_fee_reservation_and_first_day_loss(simulate):
    panel = bars({"A": [100, 100]})
    weights = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
    bt = simulate(panel, weights, amount=100_000, ftc=10, ptc=.001)
    assert bt.rebalance_log.iloc[0].shares == 900
    assert bt.rebalance_log.iloc[0].fee == 100
    assert bt.daily_position_value.cash.iloc[0] == 9900
    assert bt.nav.iloc[0] == 1
    assert bt.nav.iloc[-1] == .999
    assert bt.daily_returns.iloc[0] == pytest.approx(-.001)
    np.testing.assert_allclose(bt.daily_pnl_total, [-100, 0])


def test_dividend_crossing_is_reinvestment_not_a_trade(simulate):
    panel = bars({"A": [100, 90, 90]}, factors={"A": [1, 10 / 9, 10 / 9]})
    weights = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
    bt = simulate(panel, weights, amount=100_000)
    np.testing.assert_allclose(bt.nav, 1)
    np.testing.assert_allclose(bt.position_units.A, 1000)
    assert bt.position.A.iloc[1] == pytest.approx(1000 * 10 / 9)
    assert len(bt.rebalance_log) == 1


def test_buy_on_ex_date_uses_current_factor(simulate):
    panel = bars({"A": [100, 90, 90]}, factors={"A": [1, 10 / 9, 10 / 9]})
    weights = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-03"]))
    bt = simulate(panel, weights, amount=100_000)
    assert bt.rebalance_log.iloc[0].shares == 1100
    np.testing.assert_allclose(bt.nav, 1)


def test_synthetic_fraction_settles_without_extra_fee(simulate):
    panel = bars({"A": [100, 90, 108]}, factors={"A": [1, 10 / 9, 10 / 9]})
    weights = pd.DataFrame({"A": [10 / 11, 0]}, index=pd.to_datetime(["2024-01-02", "2024-01-04"]))
    bt = simulate(panel, weights, amount=110_000, ptc=.001)
    assert bt.rebalance_log.shares.tolist() == [1000, 1111]
    assert bt.rebalance_log.fee.tolist() == pytest.approx([100, 119.988])
    assert len(bt.synthetic_settlements) == 1
    assert bt.synthetic_settlements.iloc[0].cash_change == pytest.approx(12)
    assert bt.position.A.iloc[-1] == 0
    assert bt.daily_amount.iloc[-1] == pytest.approx(129780.012)


def test_risk_exits_next_open_without_rebalancing_other_stock(simulate):
    panel = bars({"A": [100, 90, 80, 80], "B": [100, 110, 120, 130]})
    weights = pd.DataFrame({"A": [.5], "B": [.5]}, index=pd.to_datetime(["2024-01-02"]))
    bt = simulate(panel, weights, risk={"stop_loss": .08})
    risk_trades = bt.rebalance_log[bt.rebalance_log.source.eq("risk")]
    assert len(risk_trades) == 1
    assert risk_trades.iloc[0].datetime == pd.Timestamp("2024-01-04 09:30")
    assert risk_trades.iloc[0].price == 80
    assert bt.risk_events.iloc[0].status == "executed"
    assert bt.risk_events.iloc[0].trigger_ts == pd.Timestamp("2024-01-03 15:00")
    np.testing.assert_allclose(bt.position.B, 5000)
    assert bt.position.A.tolist() == [5000, 5000, 0, 0]


@pytest.mark.parametrize("side,prices,config", [(1, [100, 110, 120], {"take_profit": .1}), (-1, [100, 110, 120], {"stop_loss": .1}), (-1, [100, 90, 80], {"take_profit": .1})])
def test_fixed_risk_for_both_sides(simulate, side, prices, config):
    panel = bars({"A": prices})
    weights = pd.DataFrame({"A": [side]}, index=pd.to_datetime(["2024-01-02"]))
    bt = simulate(panel, weights, risk=config)
    assert len(bt.risk_events) == 1
    assert bt.position.A.iloc[-1] == 0


def test_risk_entry_anchor_survives_addition_and_reduction(simulate):
    panel = bars({"A": [100, 95, 91, 90, 89]})
    weights = pd.DataFrame({"A": [.5, .9, .25]}, index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    bt = simulate(panel, weights, risk={"stop_loss": .08})
    assert bt.risk_events.iloc[0].entry_price == 100
    assert bt.risk_events.iloc[0].trigger_ts == pd.Timestamp("2024-01-04 15:00")
    assert bt.position.A.iloc[-1] == 0


def test_risk_close_execution_is_strictly_after_trigger(simulate):
    panel = bars({"A": [100, 90, 80]})
    weights = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
    bt = simulate(panel, weights, risk={"stop_loss": .08, "execution_metric": "close"})
    event = bt.risk_events.iloc[0]
    assert event.execution_ts > event.trigger_ts
    assert event.execution_ts == pd.Timestamp("2024-01-04 15:00")


def test_risk_blocks_old_targets_and_accepts_only_post_exit_targets(simulate):
    panel = bars({"A": [100, 90, 80, 80]})
    weights = pd.DataFrame({"A": [1., 1., 1.]}, index=pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-05"]))
    bt = simulate(panel, weights, risk={"stop_loss": .08})
    assert bt.order_log[bt.order_log.target_ts.eq(pd.Timestamp("2024-01-04"))].iloc[0].status == "risk_blocked"
    assert bt.position.A.iloc[-1] > 0
    assert len(bt.risk_events) == 1
    assert bt.rebalance_log.iloc[-1].source == "strategy"


def test_pending_risk_at_end_and_missing_close_are_audited(simulate):
    panel = bars({"A": [100, np.nan, 90]})
    weights = pd.DataFrame({"A": [1.]}, index=pd.to_datetime(["2024-01-02"]))
    bt = simulate(panel, weights, risk={"stop_loss": .08})
    assert bt.risk_events.iloc[0].status == "pending"
    assert bt.risk_events.iloc[0].trigger_ts == pd.Timestamp("2024-01-04 15:00")
    assert bt.valuation_audit.missing_risk_observation.any()
    assert bt.position.A.iloc[-1] > 0


def test_target_status_applied_once_without_freezing_holdings(simulate):
    panel = bars({"A": [100, 100, 100]})
    weights = pd.DataFrame({"A": [1., .5, 0.]}, index=pd.bdate_range("2024-01-02", periods=3))
    bt = simulate(panel, weights, check_trade_status=True,
        status={(pd.Timestamp("2024-01-03"), "A"): 0})
    assert bt.weight.A.tolist() == [1, .5, 0]
    assert bt.effective_target_weight.A.tolist() == [1, 0, 0]
    assert bt.position.A.tolist() == [10000, 0, 0]


def test_bookkeeping_conservation_and_export(simulate, tmp_path):
    panel = bars({"A": [100, 90, 80, 85], "B": [100, 110, 100, 105]})
    weights = pd.DataFrame({"A": [.5], "B": [.5]}, index=pd.to_datetime(["2024-01-02"]))
    bt = simulate(panel, weights, ptc=.001, risk={"stop_loss": .08})
    np.testing.assert_allclose(bt.daily_weight.sum(axis=1), 1)
    np.testing.assert_allclose(bt.daily_position_value.sum(axis=1), bt.daily_amount)
    previous = bt.daily_amount.shift(1).fillna(bt.initial_amount)
    np.testing.assert_allclose(bt.daily_pnl_total, bt.daily_amount - previous, atol=1e-8)
    path = bt.dump_to_excel(tmp_path / "bt.xlsx")
    sheets = pd.ExcelFile(path).sheet_names
    assert {"risk_events", "daily_weight", "synthetic_settlements", "meta"} <= set(sheets)


def test_removed_options_invalid_weights_and_unsupported_board(simulate):
    panel = bars({"A": [100]})
    weights = pd.DataFrame({"A": [1.]}, index=pd.to_datetime(["2024-01-02"]))
    with pytest.raises(BacktestDataError, match="已移除"):
        simulate(panel, weights, time_tolerance=24)
    with pytest.raises(BacktestDataError, match="NaN"):
        simulate(panel, weights * np.nan)
    with pytest.raises(BacktestDataError, match="重复"):
        simulate(panel, pd.concat([weights, weights]))
    with pytest.raises(ValueError, match="板块"):
        simulate(bars({"A": [100]}, markets={"A": "北交所"}), weights)
    with pytest.raises(ValueError):
        RiskConfig.parse({"stop_loss": 0})


def test_all_cash_has_daily_path_without_stock_orders(simulate):
    panel = bars({"A": [100, 110, 120]})
    weights = pd.DataFrame({"cash": [1.]}, index=pd.to_datetime(["2024-01-02"]))
    bt = simulate(panel, weights)
    np.testing.assert_allclose(bt.nav, 1)
    assert bt.rebalance_log.empty
    assert bt.daily_weight.cash.tolist() == [1, 1, 1]


def test_existing_odd_shares_are_sold_without_splitting_odd_balance():
    assert round_order(-70, 150, share_rule("主板")) == (-50, 0)
    assert round_order(-170, 250, share_rule("创业板")) == (-150, 0)
    assert round_order(-100, 150, share_rule("科创板")) == (0, 0)


def test_custom_execution_price_is_distinct_from_closing_valuation(simulate):
    panel = bars({"A": [100, 110]})
    panel["high"], panel["raw_high"] = [125., 125.], [125., 125.]
    weights = pd.DataFrame({"A": [1.]}, index=pd.to_datetime(["2024-01-02"]))
    bt = simulate(panel, weights, metric="high", price_time_map={"high": "15:00"}, amount=100_000)
    assert bt.rebalance_log.iloc[0].price == 125
    np.testing.assert_allclose(bt.daily_amount, [80_000, 88_000])
    np.testing.assert_allclose(bt.daily_pnl_total, [-20_000, 8_000])


def test_normal_direction_flip_resets_risk_anchor(simulate):
    panel = bars({"A": [100, 110, 115]})
    weights = pd.DataFrame({"A": [1., -1.]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    bt = simulate(panel, weights, risk={"stop_loss": .08})
    assert bt.risk_events.empty
    assert bt._risk.entries["A"][1] == 110


def test_no_duplicate_fee_for_pending_risk_without_future_price(simulate):
    panel = bars({"A": [100, 90, 85, 80]}, opens={"A": [100, 90, np.nan, np.nan]})
    weights = pd.DataFrame({"A": [.9]}, index=pd.to_datetime(["2024-01-02"]))
    bt = simulate(panel, weights, risk={"stop_loss": .08}, ftc=1)
    assert len(bt.risk_events) == 1
    assert bt.risk_events.iloc[0].status == "pending"
    assert len(bt.rebalance_log) == 1
    assert bt.daily_fees.sum().sum() == 1


def test_risk_config_rejects_duplicate_signal_controls():
    from betalens.backtest import validate_risk_config
    with pytest.raises(ValueError, match="同时启用"):
        validate_risk_config({"stop_loss": .1}, {"signal_weight": {"state": {"enabled": True, "stop_loss": .1}}})
    with pytest.raises(ValueError, match="正数"):
        RiskConfig(stop_loss=-.1)


def test_turnover_and_trade_return_do_not_count_drift_or_dividend_as_trades(simulate):
    from betalens.analyst import metrics
    panel = bars({"A": [100, 90, 90]}, factors={"A": [1, 10 / 9, 10 / 9]})
    weights = pd.DataFrame({"A": [1., 0.]}, index=pd.to_datetime(["2024-01-02", "2024-01-04"]))
    bt = simulate(panel, weights)
    turnover = metrics.turnover_from_trades(bt.rebalance_log, bt.daily_amount)["per_period"]
    assert turnover.iloc[1] == 0
    pairs = metrics.match_trade_pairs(bt.rebalance_log, bt.synthetic_settlements)
    assert pairs.iloc[0]["return"] == pytest.approx(0)


def test_direct_and_mining_adapter_produce_same_nav(simulate):
    from types import SimpleNamespace
    from betalens.factor.mining import _backtest_nav, MiningSpec
    panel = bars({"A": [100, 90, 80]})
    weights = pd.DataFrame({"A": [.9]}, index=pd.to_datetime(["2024-01-02"]))
    config = {"stop_loss": .08}
    bt = simulate(panel, weights, risk=config, amount=100_000, ptc=.001)
    spec = MiningSpec(SimpleNamespace(name="test", compute_kwargs={}, backtest_metric="close"))
    # Cached mining frames intentionally contain no execution data.
    nav = _backtest_nav(weights, None, spec, 100_000, "2024-01-04", config, None, 0, .001)
    pd.testing.assert_series_equal(nav, bt.nav)


def test_dashboard_uses_drift_and_preserves_first_fee_loss(simulate):
    from dashboard.backend.serialization import build_timing_payload
    panel = bars({"A": [100, 100]})
    weights = pd.DataFrame({"A": [1.]}, index=pd.to_datetime(["2024-01-02"]))
    bt = simulate(panel, weights, amount=100_000, ftc=10, ptc=.001)
    payload = build_timing_payload(bt)
    metrics = {m["label"]: m["value"] for m in payload["metrics"]}
    assert metrics["累计收益"] == pytest.approx(-.001)
    assert metrics["最大回撤"] == pytest.approx(.001)
    assert len(payload["charts"]["position"]) == 2
