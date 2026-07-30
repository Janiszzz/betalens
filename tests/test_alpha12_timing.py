from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FACTOR_DIR = REPO_ROOT / "betalens-factor" / "alpha101" / "ALPHA12"
for _path in (REPO_ROOT, FACTOR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from factor_ALPHA12 import compute_alpha12  # noqa: E402
import factor_ALPHA12_timing as timing  # noqa: E402
from betalens.factor.signal import build_signal_weights  # noqa: E402


class Alpha12TimingTests(unittest.TestCase):
    def test_compute_wrapper_ignores_timing_kwargs(self) -> None:
        close = pd.DataFrame(
            {"000001.SZ": [10.0, 9.8, 10.1]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        volume = pd.DataFrame(
            {"000001.SZ": [100.0, 120.0, 90.0]},
            index=close.index,
        )

        expected = compute_alpha12(close, volume)
        actual = timing.compute_alpha12_timing(
            close,
            volume,
            stock_code="000001.SZ",
            threshold_window=3,
            threshold_sigma=1.0,
            max_weight=1.0,
        )

        pd.testing.assert_frame_equal(actual, expected)

    def test_rolling_threshold_uses_past_window_only(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        factor_wide = pd.DataFrame({"000001.SZ": [1.0, 2.0, 3.0, 4.0, 3.0]}, index=idx)

        result = build_signal_weights(
            factor_wide=factor_wide,
            signal_dates=list(idx),
            codes=["000001.SZ"],
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
        weights, factor_values = result.weights, result.factor_values

        self.assertEqual(list(weights.index), list(idx + pd.Timedelta(minutes=10)))
        self.assertEqual(weights["000001.SZ"].tolist(), [0.0, 0.0, 0.0, -1.0, 0.0])
        self.assertEqual(weights["cash"].tolist(), [1.0, 1.0, 1.0, 2.0, 1.0])
        self.assertTrue(pd.isna(factor_values.loc[0, "历史阈值"]))
        self.assertTrue(pd.isna(factor_values.loc[1, "历史阈值"]))
        self.assertTrue(pd.isna(factor_values.loc[2, "历史阈值"]))
        self.assertAlmostEqual(float(factor_values.loc[3, "滚动均值"]), 2.0)
        self.assertAlmostEqual(float(factor_values.loc[3, "历史阈值"]), 3.0)
        self.assertAlmostEqual(float(factor_values.loc[4, "滚动均值"]), 3.0)
        self.assertAlmostEqual(float(factor_values.loc[4, "历史阈值"]), 4.0)
        self.assertEqual(factor_values["是否触发"].tolist(), [False, False, False, True, False])
        self.assertEqual(factor_values["目标仓位"].tolist(), [0.0, 0.0, 0.0, -1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
