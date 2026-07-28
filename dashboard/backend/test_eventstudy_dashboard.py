from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from betalens.eventstudy.eventstudy import EventStudy

from . import eventstudy_dashboard
from .eventstudy_dashboard import _comparison_payload, _cumulative_matrix_records, discover_event_files, run_event_study
from .schemas import EventStudyRequest


class EventStudyDashboardTests(unittest.TestCase):
    def test_discover_event_files_reads_local_xlsx(self) -> None:
        payload = discover_event_files()
        files = payload["files"]

        self.assertIn("defaults", payload)
        self.assertEqual(payload["defaults"]["event_file"], "1.春节假期.xlsx")
        self.assertGreaterEqual(len(files), 1)
        self.assertTrue(any(item["id"] == "1.春节假期.xlsx" for item in files))
        first = next(item for item in files if item["id"] == "1.春节假期.xlsx")
        self.assertIn("date", first["columns"])
        self.assertGreater(first["eventCount"], 0)

    def test_flexible_cumulative_uses_holding_start_offset(self) -> None:
        returns = pd.DataFrame(
            {
                0: {-1: 0.01, 0: 0.02, 1: 0.03, 2: 0.04},
                1: {-1: -0.01, 0: 0.01, 1: 0.02, 2: 0.03},
            }
        ).sort_index()

        cumulative = EventStudy(None)._calc_cumulative_flexible(returns, holding_start_offset=1)

        self.assertAlmostEqual(cumulative.loc[1, 0], 0.03)
        self.assertAlmostEqual(cumulative.loc[1, 1], 0.02)
        self.assertAlmostEqual(cumulative.loc[2, 0], (1.03 * 1.04) - 1)
        self.assertAlmostEqual(cumulative.loc[0, 0], (1.02 * 1.03) - 1)
        self.assertAlmostEqual(cumulative.loc[-1, 1], (0.99 * 1.01 * 1.02) - 1)

    def test_comparison_payload_is_json_safe_and_preserves_event_ids(self) -> None:
        daily = pd.DataFrame(
            {"mean": [0.01], "std": [0.0], "positive_prob": [1.0], "odds": [np.inf], "t_stat": [np.nan], "count": [1]},
            index=pd.Index([0], name="day"),
        )
        cumulative = daily.copy()
        raw = {
            "comparison": {
                "events": [
                    {"event_id": 0, "event_date": pd.Timestamp("2024-01-03 10:00:00")},
                    {"event_id": 1, "event_date": pd.Timestamp("2024-01-10 10:00:00")},
                ],
                "valid_codes": ["A", "B"],
                "skipped_codes": [{"code": "C", "reason": "no data"}],
                "by_code": {
                    "A": {
                        "event_count": 2,
                        "coverage": 1.0,
                        "daily_stats": daily,
                        "cumulative_stats": cumulative,
                        "cumulative_returns_matrix": pd.DataFrame({0: {0: 0.01}, 1: {0: 0.02}}),
                    },
                    "B": {
                        "event_count": 1,
                        "coverage": 0.5,
                        "daily_stats": daily,
                        "cumulative_stats": cumulative,
                        "cumulative_returns_matrix": pd.DataFrame({0: {0: 0.03}, 1: {0: np.nan}}),
                    },
                },
            }
        }

        payload = _comparison_payload(raw, 0)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["events"][1]["eventId"], 1)
        self.assertEqual(payload["events"][1]["eventDate"], "2024-01-10 10:00:00")
        self.assertEqual(payload["summaryByCode"][1]["coverage"], 0.5)
        self.assertIsNone(payload["summaryByCode"][0]["day0TStat"])
        self.assertEqual(len(payload["eventCumulativeByCode"]), 4)

    def test_matrix_records_use_stable_event_id_for_date_lookup(self) -> None:
        event_dates = pd.DatetimeIndex(["2024-01-03", "2024-01-10"])
        records = _cumulative_matrix_records(
            pd.DataFrame({1: {0: 0.01}}), event_dates=event_dates
        )

        self.assertEqual(records[0]["event"], "1")
        self.assertEqual(records[0]["eventDate"], "2024-01-10 00:00:00")

    def test_run_forwards_compare_mode(self) -> None:
        raw = {
            "event_count": 1,
            "valid_codes": ["A", "B"],
            "daily_stats": pd.DataFrame({"mean": [0.01], "t_stat": [np.nan], "positive_prob": [1.0]}, index=[0]),
            "cumulative_stats": pd.DataFrame({"mean": [0.01], "t_stat": [np.nan], "positive_prob": [1.0]}, index=[0]),
            "returns_matrix": pd.DataFrame({0: {0: 0.01}}),
            "cumulative_returns_matrix": pd.DataFrame({0: {0: 0.01}}),
            "event_dates": pd.DatetimeIndex(["2024-01-03"]),
        }

        class FakeDatafeed:
            def __init__(self, table_name: str) -> None:
                self.table_name = table_name

            def close(self) -> None:
                pass

        class FakeStudy:
            received: dict = {}

            def __init__(self, datafeed) -> None:
                self.datafeed = datafeed

            def analyze(self, **kwargs):
                type(self).received = kwargs
                return raw

        defaults = {
            "event_file": "1.春节假期.xlsx",
            "code": "A,B",
            "benchmark_code": "",
            "metric": "收盘价(元)",
            "table_name": "daily_market",
            "mode": "flexible",
            "multi_asset_mode": "aggregate",
            "window_before": 1,
            "window_after": 1,
            "holding_start_offset": 0,
            "market_close_hour": 15,
            "holding_days": "1",
            "holding_months": "",
        }
        event_path = eventstudy_dashboard.EVENT_ROOT / "1.春节假期.xlsx"
        with (
            patch.object(eventstudy_dashboard, "load_eventstudy_params", return_value=defaults),
            patch.object(eventstudy_dashboard, "_safe_event_path", return_value=event_path),
            patch.object(eventstudy_dashboard, "_event_series", return_value=pd.Series([1], index=pd.DatetimeIndex(["2024-01-03"]))),
            patch.object(eventstudy_dashboard, "_event_rows", return_value=[]),
            patch.object(eventstudy_dashboard, "Datafeed", FakeDatafeed),
            patch.object(eventstudy_dashboard, "EventStudy", FakeStudy),
        ):
            result = run_event_study({"multi_asset_mode": "compare"})

        self.assertEqual(FakeStudy.received["multi_asset_mode"], "compare")
        self.assertEqual(result["parameters"]["multiAssetMode"], "compare")

    def test_request_accepts_compare_mode(self) -> None:
        request = EventStudyRequest(code=["A", "B"], multi_asset_mode="compare")
        self.assertEqual(request.multi_asset_mode, "compare")


if __name__ == "__main__":
    unittest.main()
