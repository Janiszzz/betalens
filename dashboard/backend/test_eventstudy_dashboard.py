from __future__ import annotations

import unittest

import pandas as pd

from betalens.eventstudy.eventstudy import EventStudy

from .eventstudy_dashboard import discover_event_files


class EventStudyDashboardTests(unittest.TestCase):
    def test_discover_event_files_reads_local_xlsx(self) -> None:
        payload = discover_event_files()
        files = payload["files"]

        self.assertIn("defaults", payload)
        self.assertEqual(payload["defaults"]["event_file"], "events.xlsx")
        self.assertGreaterEqual(len(files), 1)
        self.assertTrue(any(item["id"] == "events.xlsx" for item in files))
        first = next(item for item in files if item["id"] == "events.xlsx")
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


if __name__ == "__main__":
    unittest.main()
