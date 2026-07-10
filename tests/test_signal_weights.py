from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from betalens.factor import mining as mining_module  # noqa: E402
from betalens.factor.signal import (  # noqa: E402
    build_signal_weights,
    cash_from_weights,
    event_history_weight,
    rolling_z_weight,
    standard_timing_weight_hook,
    threshold_weight,
)


class SignalWeightTests(unittest.TestCase):
    def test_cash_from_weights_uses_net_exposure(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        weights = pd.DataFrame(
            {
                "000001.SZ": [1.0, -1.0, 0.6],
                "000002.SZ": [0.0, 0.0, -0.4],
            },
            index=idx,
        )

        actual = cash_from_weights(weights, scale=True)

        self.assertEqual(actual["cash"].tolist(), [0.0, 2.0, 0.8])

    def test_threshold_weight_supports_long_and_short_sides(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        factor = pd.DataFrame({"000001.SZ": [0.0, 0.2, 0.3]}, index=idx)

        long_result = threshold_weight(
            factor_wide=factor,
            signal_dates=idx,
            codes=["000001.SZ"],
            params={"trigger_threshold": 0.1, "max_weight": 1.0},
            side="long",
        )
        short_result = threshold_weight(
            factor_wide=factor,
            signal_dates=idx,
            codes=["000001.SZ"],
            params={"trigger_threshold": 0.1, "max_weight": 1.0},
            side="short",
        )

        self.assertEqual(long_result.weights["000001.SZ"].tolist(), [0.0, 1.0, 1.0])
        self.assertEqual(long_result.weights["cash"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(short_result.weights["000001.SZ"].tolist(), [0.0, -1.0, -1.0])
        self.assertEqual(short_result.weights["cash"].tolist(), [1.0, 2.0, 2.0])

    def test_rolling_z_weight_uses_past_window_only(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        factor = pd.DataFrame({"000001.SZ": [1.0, 2.0, 3.0, 4.0, 3.0]}, index=idx)

        result = rolling_z_weight(
            factor_wide=factor,
            signal_dates=idx,
            codes=["000001.SZ"],
            params={"threshold_window": 3, "threshold_sigma": 1.0, "max_weight": 1.0},
            side="short",
        )

        self.assertEqual(list(result.weights.index), list(idx + pd.Timedelta(minutes=10)))
        self.assertEqual(result.weights["000001.SZ"].tolist(), [0.0, 0.0, 0.0, -1.0, 0.0])
        self.assertEqual(result.weights["cash"].tolist(), [1.0, 1.0, 1.0, 2.0, 1.0])
        self.assertAlmostEqual(float(result.factor_values.loc[3, "滚动均值"]), 2.0)
        self.assertAlmostEqual(float(result.factor_values.loc[3, "历史阈值"]), 3.0)
        self.assertEqual(result.factor_values["是否触发"].tolist(), [False, False, False, True, False])

    def test_event_history_weight_waits_for_historical_events(self) -> None:
        idx = pd.date_range("2024-01-01", periods=16, freq="D")
        values = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        highs = [10, 10, 9, 11, 10, 10, 9, 11, 10, 10, 9, 11, 10, 10, 9, 11]
        factor = pd.DataFrame({"000001.SZ": values}, index=idx)
        high = pd.DataFrame({"000001.SZ": highs}, index=idx)

        result = event_history_weight(
            factor_wide=factor,
            high_wide=high,
            signal_dates=idx,
            codes=["000001.SZ"],
            params={
                "trigger_threshold": 0.5,
                "history_window": 3,
                "duration_quantile": 1.0,
                "exit_wait_quantile": 0.5,
                "min_history_events": 3,
                "default_exit_wait_days": 1,
                "max_weight": 1.0,
            },
            side="long",
        )

        daily_weights = result.weights.copy()
        daily_weights.index = daily_weights.index.normalize()
        self.assertEqual(daily_weights.loc["2024-01-13", "000001.SZ"], 0.0)
        self.assertEqual(daily_weights.loc["2024-01-14", "000001.SZ"], 1.0)
        self.assertEqual(daily_weights.loc["2024-01-15", "000001.SZ"], 1.0)
        self.assertEqual(daily_weights.loc["2024-01-16", "000001.SZ"], 1.0)
        self.assertIn("000001.SZ", result.events)
        self.assertEqual(len(result.events["000001.SZ"]), 4)

    def test_build_signal_weights_accepts_legacy_and_nested_params(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        factor = pd.DataFrame({"000001.SZ": [1.0, 2.0, 3.0, 4.0, 3.0]}, index=idx)

        legacy = build_signal_weights(
            factor_wide=factor,
            signal_dates=idx,
            codes=["000001.SZ"],
            params={"threshold_window": 3, "threshold_sigma": 1.0},
            side="short",
        )
        nested = build_signal_weights(
            factor_wide=factor,
            signal_dates=idx,
            codes=["000001.SZ"],
            params={"signal_weight": {"method": "rolling_z", "window": 3, "sigma": 1.0, "side": "short"}},
        )

        pd.testing.assert_frame_equal(legacy.weights, nested.weights)

    def test_standard_timing_weight_hook_works_with_mining_task_context(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        factor = pd.DataFrame({"000001.SZ": [1.0, 2.0, 3.0, 4.0, 3.0]}, index=idx)
        spec = SimpleNamespace(
            name="FAKE",
            compute_kwargs={"signal_weight": {"method": "rolling_z", "window": 3, "sigma": 1.0, "side": "short"}},
            direction="positive",
        )
        task = {
            "params": {},
            "context": {
                "factor_wide": factor,
                "input_wides": {},
                "price_wide": pd.DataFrame({"000001.SZ": [10, 11, 12, 11, 10]}, index=idx),
                "signal_dates": list(idx),
                "spec": spec,
                "universe": ["000001.SZ"],
            },
        }

        weights = standard_timing_weight_hook(pd.DataFrame(), task)

        self.assertEqual(weights["000001.SZ"].tolist(), [0.0, 0.0, 0.0, -1.0, 0.0])
        self.assertEqual(weights["cash"].tolist(), [1.0, 1.0, 1.0, 2.0, 1.0])

    def test_mining_timing_weight_mode_uses_weight_hook(self) -> None:
        module_name = "_fake_timing_mining_module"
        module = types.ModuleType(module_name)

        def make_mining_spec(params):
            del params
            return SimpleNamespace(
                name="FAKE_TIMING",
                inputs={"factor_wide": "factor"},
                compute=lambda factor_wide, **kwargs: factor_wide,
                compute_kwargs={"signal_weight": {"method": "threshold", "threshold": 0.1, "side": "long"}},
                weight_mode="timing",
                backtest_metric="price",
                direction="positive",
            )

        module.make_mining_spec = make_mining_spec
        sys.modules[module_name] = module

        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        original_cache = mining_module._CACHE_DATA
        original_pit = mining_module._PIT_UNIVERSE
        original_signal_dates = mining_module._task_signal_dates
        try:
            mining_module._CACHE_DATA = {
                "inputs": {"factor_wide": pd.DataFrame({"000001.SZ": [0.0, 0.2, 0.3, 0.0, 0.2]}, index=idx)},
                "price": pd.DataFrame({"000001.SZ": [10.0, 10.5, 11.0, 10.8, 11.2]}, index=idx),
                "universe": ["000001.SZ"],
            }
            mining_module._PIT_UNIVERSE = None
            mining_module._task_signal_dates = lambda start, end, fetch_start, rebal_freq: list(idx)
            out = mining_module._run_one_task(
                {
                    "params": {},
                    "win_start": "2024-01-01",
                    "win_end": "2024-01-05",
                    "scheme": "full",
                    "phase": "sweep",
                    "gid": "fake",
                    "factor_module": module_name,
                    "spec_factory": "make_mining_spec",
                    "weight_hook": "betalens.factor.signal.standard_timing_weight_hook",
                    "warmup_days": 1,
                    "rebal_freq": "D",
                    "engine": "vector",
                    "n_quantiles_param": "n_quantiles",
                    "initial_amount": 1_000_000,
                    "time_tolerance": 24,
                }
            )
        finally:
            mining_module._CACHE_DATA = original_cache
            mining_module._PIT_UNIVERSE = original_pit
            mining_module._task_signal_dates = original_signal_dates
            sys.modules.pop(module_name, None)

        self.assertNotIn("error", out)
        self.assertIn("sharpe", out)
        self.assertGreaterEqual(out["n_days"], 1)


if __name__ == "__main__":
    unittest.main()
