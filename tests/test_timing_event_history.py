from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FACTOR_DIR = REPO_ROOT / "betalens-factor" / "tdx" / "XICHOU"
for _path in (REPO_ROOT, FACTOR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import factor_XICHOU_timing as timing  # noqa: E402


class EventHistoryTimingTests(unittest.TestCase):
    def test_build_weights_uses_event_history_signal_weight_operator(self) -> None:
        idx = pd.date_range("2024-01-01", periods=16, freq="D")
        factor = pd.DataFrame(
            {"000001.SZ": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]},
            index=idx,
        )
        high = pd.DataFrame(
            {"000001.SZ": [10, 10, 9, 11, 10, 10, 9, 11, 10, 10, 9, 11, 10, 10, 9, 11]},
            index=idx,
        )

        weights, events = timing._build_weights(
            factor_wide=factor,
            high_wide=high,
            signal_dates=list(idx),
            codes=["000001.SZ"],
            direction="positive",
            params={
                "trigger_threshold": 0.5,
                "trigger_operator": "auto",
                "history_window": 3,
                "duration_quantile": 1.0,
                "exit_wait_quantile": 0.5,
                "min_history_events": 3,
                "default_exit_wait_days": 1,
                "max_weight": 1.0,
            },
        )

        daily_weights = weights.copy()
        daily_weights.index = daily_weights.index.normalize()
        self.assertEqual(daily_weights.loc["2024-01-13", "000001.SZ"], 0.0)
        self.assertEqual(daily_weights.loc["2024-01-14", "000001.SZ"], 1.0)
        self.assertEqual(daily_weights.loc["2024-01-15", "000001.SZ"], 1.0)
        self.assertEqual(daily_weights.loc["2024-01-16", "000001.SZ"], 1.0)
        self.assertEqual(daily_weights.loc["2024-01-14", "cash"], 0.0)
        self.assertIn("000001.SZ", events)
        self.assertEqual(len(events["000001.SZ"]), 4)


if __name__ == "__main__":
    unittest.main()
