from __future__ import annotations

import sys
from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FACTOR_ROOT = REPO_ROOT / "betalens-factor"
ALPHA101_ROOT = FACTOR_ROOT / "alpha101"
for _path in (FACTOR_ROOT, ALPHA101_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from factor_template import (  # noqa: E402
    align_daily_wides,
    mask_wide_by_pit_universe,
    validate_weights_in_pit_universe,
)
from betalens.factor.signal import build_signal_weights, resolve_timing_start_date  # noqa: E402
from factor_template_alpha101 import _with_timing_targets  # noqa: E402


def test_daily_pit_mask_uses_each_rows_calendar_date() -> None:
    index = pd.to_datetime(["2024-01-02 15:00:01", "2024-01-03 15:00:01"])
    wide = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]},
        index=index,
    )
    pit = {
        date(2024, 1, 2): {"A", "B"},
        date(2024, 1, 3): {"B", "C"},
    }

    actual = mask_wide_by_pit_universe(wide, pit)

    assert actual.loc[index[0], ["A", "B"]].tolist() == [1.0, 3.0]
    assert pd.isna(actual.loc[index[0], "C"])
    assert pd.isna(actual.loc[index[1], "A"])
    assert actual.loc[index[1], ["B", "C"]].tolist() == [4.0, 6.0]
    pd.testing.assert_frame_equal(wide, pd.DataFrame(
        {"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]}, index=index
    ))


def test_daily_inputs_align_by_date_at_latest_availability_time() -> None:
    open_wide = pd.DataFrame(
        {"A": [10.0, 11.0]},
        index=pd.to_datetime(["2024-01-02 09:30:01", "2024-01-03 09:30:01"]),
    )
    close_wide = pd.DataFrame(
        {"A": [10.5, 10.8]},
        index=pd.to_datetime(["2024-01-02 15:00:01", "2024-01-03 15:00:01"]),
    )

    actual = align_daily_wides({"open": open_wide, "close": close_wide})

    expected_index = pd.to_datetime(["2024-01-02 15:00:01", "2024-01-03 15:00:01"])
    assert actual["open"].index.equals(expected_index)
    assert actual["close"].index.equals(expected_index)
    assert actual["open"]["A"].tolist() == [10.0, 11.0]
    assert actual["close"]["A"].tolist() == [10.5, 10.8]


def test_pit_weight_validation_ignores_cash_column() -> None:
    index = pd.to_datetime(["2024-01-02 15:10:01"])
    weights = pd.DataFrame({"A": [1.0], "cash": [0.0]}, index=index)

    result = validate_weights_in_pit_universe(weights, {date(2024, 1, 2): {"A"}})

    assert result.iloc[0]["passed"]
    assert result.iloc[0]["selected_count"] == 1


def test_timing_target_is_not_restricted_by_index_membership() -> None:
    first = date(2024, 1, 2)
    second = date(2024, 1, 3)
    context = {first: {"A", "B"}, second: {"B", "C"}}

    formula_context = _with_timing_targets(context, ["TARGET"])

    assert formula_context == {
        first: {"A", "B", "TARGET"},
        second: {"B", "C", "TARGET"},
    }
    assert context == {first: {"A", "B"}, second: {"B", "C"}}

    index = pd.to_datetime(["2024-01-02 15:00:01", "2024-01-03 15:00:01"])
    target = pd.DataFrame({"TARGET": [1.0, 2.0]}, index=index)
    masked = mask_wide_by_pit_universe(target, formula_context)
    assert masked["TARGET"].tolist() == [1.0, 2.0]


def test_timing_start_moves_to_latest_required_target_data_start() -> None:
    datafeed = MagicMock()
    datafeed.get_available_dates.side_effect = lambda code, metric, end_date: {
        "收盘价(元)": pd.to_datetime(["2018-06-11", "2018-06-12"]),
        "成交量(股)": pd.to_datetime(["2018-06-12", "2018-06-13"]),
    }[metric]
    output = StringIO()

    with patch("betalens.datafeed.Datafeed", return_value=datafeed), redirect_stdout(output):
        effective = resolve_timing_start_date(
            "2015-01-01",
            "2020-12-31",
            target_codes=["300750.SZ"],
            metrics=["收盘价(元)", "成交量(股)", "收盘价(元)"],
            table_name="daily_market",
        )

    assert effective == "2018-06-12"
    assert "300750.SZ" in output.getvalue()
    assert "2015-01-01" in output.getvalue()
    assert "2018-06-12" in output.getvalue()
    assert datafeed.get_available_dates.call_count == 2
    datafeed.close.assert_called_once_with()


def test_timing_start_keeps_later_requested_date_without_warning() -> None:
    datafeed = MagicMock()
    datafeed.get_available_dates.return_value = pd.to_datetime(["2018-06-11", "2018-06-12"])
    output = StringIO()

    with patch("betalens.datafeed.Datafeed", return_value=datafeed), redirect_stdout(output):
        effective = resolve_timing_start_date(
            "2019-01-02",
            "2020-12-31",
            target_codes=["300750.SZ"],
            metrics=["收盘价(元)"],
            table_name="daily_market",
        )

    assert effective == "2019-01-02"
    assert output.getvalue() == ""
    datafeed.close.assert_called_once_with()


def test_timing_target_with_nan_factor_holds_cash() -> None:
    index = pd.date_range("2024-01-01", periods=5)
    factor = pd.DataFrame({"300750.SZ": [1.0, 2.0, 3.0, 10.0, float("nan")]}, index=index)

    result = build_signal_weights(
        factor_wide=factor,
        signal_dates=index,
        codes=["300750.SZ"],
        params={
            "signal_weight": {
                "method": "rolling_z",
                "window": 3,
                "sigma": 1.0,
                "operator": "gt",
                "side": "short",
                "max_weight": 1.0,
            }
        },
    )

    assert result.weights["300750.SZ"].tolist() == [0.0, 0.0, 0.0, -1.0, 0.0]
    assert result.weights["cash"].tolist() == [1.0, 1.0, 1.0, 2.0, 1.0]
    assert (result.weights.index == index + pd.Timedelta(minutes=10)).all()


def test_timing_signal_configuration_supports_long_and_lt() -> None:
    index = pd.date_range("2024-01-01", periods=5)
    factor = pd.DataFrame({"300750.SZ": [3.0, 2.0, 1.0, -10.0, 0.0]}, index=index)

    result = build_signal_weights(
        factor_wide=factor,
        signal_dates=index,
        codes=["300750.SZ"],
        params={
            "signal_weight": {
                "method": "rolling_z",
                "window": 3,
                "sigma": 1.0,
                "operator": "lt",
                "side": "long",
                "max_weight": 0.6,
            }
        },
    )

    assert result.weights["300750.SZ"].tolist() == [0.0, 0.0, 0.0, 0.6, 0.0]
    assert result.weights["cash"].tolist() == [1.0, 1.0, 1.0, 0.4, 1.0]


def test_timing_rolling_window_uses_factor_history_before_signal_dates() -> None:
    full_index = pd.date_range("2024-01-01", periods=5)
    factor = pd.DataFrame({"300750.SZ": [1.0, 2.0, 3.0, 10.0, 4.0]}, index=full_index)

    result = build_signal_weights(
        factor_wide=factor,
        signal_dates=full_index[-2:],
        codes=["300750.SZ"],
        params={
            "signal_weight": {
                "method": "rolling_z",
                "window": 3,
                "sigma": 1.0,
                "operator": "gt",
                "side": "short",
                "max_weight": 1.0,
            }
        },
    )

    assert result.weights["300750.SZ"].tolist() == [-1.0, 0.0]
    assert result.factor_values["滚动均值"].tolist() == [2.0, 5.0]


def test_timing_rolling_window_uses_prior_valid_observations_across_gaps() -> None:
    full_index = pd.date_range("2024-01-01", periods=6)
    factor = pd.DataFrame(
        {"300750.SZ": [1.0, 2.0, float("nan"), 3.0, 10.0, float("nan")]},
        index=full_index,
    )

    result = build_signal_weights(
        factor_wide=factor,
        signal_dates=full_index[-2:],
        codes=["300750.SZ"],
        params={
            "signal_weight": {
                "method": "rolling_z",
                "window": 3,
                "sigma": 1.0,
                "operator": "gt",
                "side": "short",
            }
        },
    )

    assert result.weights["300750.SZ"].tolist() == [-1.0, 0.0]
    assert result.factor_values["滚动均值"].iloc[0] == 2.0
    assert pd.isna(result.factor_values["因子值"].iloc[1])
