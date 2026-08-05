from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from betalens.eventstudy.eventstudy import EventStudy


class FakeDatafeed:
    def __init__(self, prices: dict[str, pd.Series]) -> None:
        self.prices = prices

    def query_time_range(self, codes, start_date=None, end_date=None, metric=None):
        code = codes[0]
        series = self.prices.get(code)
        if series is None:
            return pd.DataFrame(columns=["datetime", "value"])
        return pd.DataFrame({"datetime": series.index, "value": series.values})


def _prices(start: str, periods: int, daily_return: float) -> pd.Series:
    index = pd.date_range(start, periods=periods, freq="B")
    values = 100 * np.power(1 + daily_return, np.arange(periods))
    return pd.Series(values, index=index)


class EventStudyCompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = pd.Series(
            [1, 1],
            index=pd.to_datetime(["2024-01-03 10:00:00", "2024-01-10 10:00:00"]),
        )
        self.feed = FakeDatafeed(
            {
                "A": _prices("2024-01-01", 15, 0.02),
                # The second event has no next close for B, but its event id
                # must remain present in the comparison matrix.
                "B": _prices("2024-01-01", 8, 0.03),
                "BENCH": _prices("2024-01-01", 15, 0.01),
            }
        )

    def test_compare_keeps_stable_event_ids_and_common_average(self) -> None:
        result = EventStudy(self.feed).analyze(
            self.events,
            code=["A", "B"],
            window_before=0,
            window_after=1,
            multi_asset_mode="compare",
        )

        comparison = result["comparison"]
        b_returns = comparison["by_code"]["B"]["returns_matrix"]

        self.assertEqual(list(b_returns.columns), [0, 1])
        self.assertTrue(b_returns[1].isna().all())
        self.assertEqual(comparison["by_code"]["B"]["event_ids"], [0])
        self.assertEqual(comparison["by_code"]["B"]["coverage"], 0.5)
        self.assertEqual(
            [item["event_id"] for item in comparison["events"]], [0, 1]
        )
        self.assertEqual(
            list(result["event_dates"]), list(self.events.index)
        )

        a_returns = comparison["by_code"]["A"]["returns_matrix"]
        self.assertAlmostEqual(
            result["returns_matrix"].loc[0, 0],
            (a_returns.loc[0, 0] + b_returns.loc[0, 0]) / 2,
        )
        self.assertAlmostEqual(
            result["returns_matrix"].loc[0, 1], a_returns.loc[0, 1]
        )
        self.assertEqual(comparison["by_code"]["B"]["daily_stats"].loc[0, "count"], 1)
        self.assertTrue(pd.isna(comparison["by_code"]["B"]["daily_stats"].loc[0, "t_stat"]))

    def test_compare_applies_benchmark_to_each_target(self) -> None:
        result = EventStudy(self.feed).analyze(
            self.events,
            code=["A", "B"],
            window_before=0,
            window_after=0,
            benchmark_code="BENCH",
            multi_asset_mode="compare",
        )

        by_code = result["comparison"]["by_code"]
        self.assertAlmostEqual(by_code["A"]["returns_matrix"].loc[0, 0], 0.01)
        self.assertAlmostEqual(by_code["B"]["returns_matrix"].loc[0, 0], 0.02)
        self.assertAlmostEqual(result["returns_matrix"].loc[0, 0], 0.015)

    def test_compare_preserves_each_assets_pre_event_cumulative_return(self) -> None:
        result = EventStudy(self.feed).analyze(
            self.events,
            code=["A", "B"],
            window_before=1,
            window_after=1,
            multi_asset_mode="compare",
        )

        by_code = result["comparison"]["by_code"]
        self.assertAlmostEqual(
            by_code["A"]["cumulative_returns_matrix"].loc[-1, 0],
            (1.02 ** 2) - 1,
        )
        self.assertAlmostEqual(
            by_code["B"]["cumulative_returns_matrix"].loc[-1, 0],
            (1.03 ** 2) - 1,
        )
        self.assertNotEqual(
            by_code["B"]["cumulative_returns_matrix"].loc[-1, 0],
            0,
        )

    def test_aggregate_remains_default_and_fixed_mode_is_available(self) -> None:
        aggregate = EventStudy(self.feed).analyze(
            self.events,
            code=["A", "B"],
            window_before=1,
            window_after=2,
        )
        fixed = EventStudy(self.feed).analyze(
            self.events,
            code=["A", "B"],
            window_before=1,
            window_after=2,
            mode="fixed",
            holding_periods={"days": [1], "months": []},
            multi_asset_mode="compare",
        )

        self.assertNotIn("comparison", aggregate)
        self.assertIn("comparison", fixed)
        self.assertFalse(fixed["cumulative_returns_matrix"].empty)

    def test_compare_requires_multiple_codes(self) -> None:
        with self.assertRaisesRegex(ValueError, "至少需要两个"):
            EventStudy(self.feed).analyze(
                self.events,
                code="A",
                multi_asset_mode="compare",
            )

    def test_compare_reports_codes_without_valid_data(self) -> None:
        result = EventStudy(self.feed).analyze(
            self.events,
            code=["A", "MISSING"],
            window_before=0,
            window_after=1,
            multi_asset_mode="compare",
        )

        comparison = result["comparison"]
        self.assertEqual(comparison["valid_codes"], ["A"])
        self.assertEqual(comparison["skipped_codes"][0]["code"], "MISSING")
        self.assertEqual(comparison["by_code"]["A"]["coverage"], 1.0)

    def test_post_close_event_uses_next_close(self) -> None:
        event = pd.Series([1], index=pd.to_datetime(["2024-01-03 16:00:00"]))
        result = EventStudy(self.feed).analyze(
            event,
            code="A",
            window_before=0,
            window_after=0,
        )

        # The next-close cost rule shifts the day-0 return by one business day.
        self.assertAlmostEqual(result["returns_matrix"].loc[0, 0], 0.02)


if __name__ == "__main__":
    unittest.main()
