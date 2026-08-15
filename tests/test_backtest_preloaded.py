from __future__ import annotations

import pandas as pd
import pytest

from betalens.backtest import BacktestBase
from betalens.backtest.backtest import BacktestDataError


def _preloaded_inputs():
    dates = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02 09:40:00"),
        pd.Timestamp("2024-01-03 09:40:00"),
    ])
    weights = pd.DataFrame(
        {"A": [0.5, 0.4], "B": [0.5, 0.6], "cash": [0.0, 0.0]},
        index=dates,
    )
    prices = pd.DataFrame(
        {"A": [10.0, 11.0], "B": [20.0, 19.0]},
        index=dates,
    )
    status = pd.DataFrame({"A": [1, 1], "B": [1, 1]}, index=dates)
    return weights, prices, status


def test_backtest_preloaded_data_avoids_datafeed(monkeypatch) -> None:
    weights, prices, status = _preloaded_inputs()

    class RefuseDatafeed:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Datafeed must not be created for fully preloaded backtests")

    monkeypatch.setattr("betalens.backtest.backtest.Datafeed", RefuseDatafeed)
    engine = BacktestBase(
        weight=weights,
        symbol="PRELOADED",
        amount=1_000_000,
        metric="开盘价(元)",
        verbose=False,
        preloaded_cost_price=prices,
        preloaded_close_price=prices,
        preloaded_trade_status=status,
    )

    assert engine.data_sources == {
        "trade_status": "preloaded",
        "cost_price": "preloaded",
        "daily_price": "preloaded",
    }
    pd.testing.assert_index_equal(
        engine.cost_price.index.rename(None), weights.index.rename(None)
    )
    assert not engine.nav.empty


def test_backtest_rejects_invalid_preloaded_status() -> None:
    weights, prices, status = _preloaded_inputs()
    status.iloc[0, 0] = 2

    with pytest.raises(BacktestDataError, match="只能包含 -1、0、1"):
        BacktestBase(
            weight=weights,
            symbol="INVALID",
            amount=1_000_000,
            verbose=False,
            preloaded_cost_price=prices,
            preloaded_close_price=prices,
            preloaded_trade_status=status,
        )
