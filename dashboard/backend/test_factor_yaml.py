from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from betalens.factor.config import write_yaml_config

from .factors import clear_factor_cache, discover_factors, get_factor_config
from .runs import DashboardRun, RunManager
from .schemas import RunRequest


class FactorYamlDashboardTests(unittest.TestCase):
    def test_discover_factors_reads_yaml_specs(self) -> None:
        clear_factor_cache()
        factors = discover_factors()
        names = {(item.factor_class, item.name) for item in factors}

        self.assertIn(("tdx", "RSI_FAST"), names)
        self.assertIn(("LiqDemand", "DISP"), names)
        self.assertIn(("alpha101", "ALPHA12_timing"), names)
        disp = next(item for item in factors if item.factor_class == "LiqDemand" and item.name == "DISP")
        self.assertEqual(disp.defaults["n_quantiles"], 20)
        self.assertEqual(disp.defaults["direction"], "negative")
        self.assertEqual(disp.strategy_type, "cross_sectional")
        self.assertIn(("tdx", "XICHOU_timing"), names)
        xichou = next(item for item in factors if item.factor_class == "tdx" and item.name == "XICHOU")
        self.assertEqual(xichou.strategy_type, "cross_sectional")
        xichou_timing = next(item for item in factors if item.factor_class == "tdx" and item.name == "XICHOU_timing")
        self.assertEqual(xichou_timing.strategy_type, "timing")

    def test_discover_factors_exposes_complete_alpha101_catalog(self) -> None:
        clear_factor_cache()
        alpha101 = [item for item in discover_factors() if item.factor_class == "alpha101"]
        names = {item.name for item in alpha101}

        self.assertEqual(len(alpha101), 202)
        self.assertEqual(
            names,
            {
                name
                for number in range(1, 102)
                for name in (f"ALPHA{number}", f"ALPHA{number}_timing")
            },
        )
        self.assertEqual(sum(item.strategy_type == "cross_sectional" for item in alpha101), 101)
        self.assertEqual(sum(item.strategy_type == "timing" for item in alpha101), 101)

        alpha48 = next(item for item in alpha101 if item.name == "ALPHA48")
        self.assertEqual(alpha48.inputs["subindustry_wide"], "industry:申万三级行业")

    def test_get_factor_config_supports_multiple_yaml_specs_in_one_dir(self) -> None:
        script, _class_cfg, factor_cfg = get_factor_config("tdx", "XICHOU_timing")

        self.assertEqual(script.name, "factor_XICHOU_timing.py")
        self.assertEqual(factor_cfg["meta"]["name"], "XICHOU_timing")
        self.assertEqual(factor_cfg["meta"]["strategy_type"], "timing")

        alpha_script, _alpha_class_cfg, alpha_factor_cfg = get_factor_config("alpha101", "ALPHA12_timing")
        self.assertEqual(alpha_script.name, "factor_ALPHA12_timing.py")
        self.assertEqual(alpha_factor_cfg["meta"]["name"], "ALPHA12_timing")
        self.assertEqual(alpha_factor_cfg["meta"]["strategy_type"], "timing")

    def test_run_config_copy_uses_output_dir_and_yaml_source(self) -> None:
        _script, _class_cfg, factor_cfg = get_factor_config("tdx", "RSI_FAST")
        request = RunRequest(
            factor_class="tdx",
            name="RSI_FAST",
            parameters={
                "start_date": "2024-02-01",
                "end_date": "2024-02-29",
                "n_quantiles": 5,
                "long_groups": 4,
            },
            compute_kwargs={"window": 4},
        )
        run = DashboardRun(request)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            config = RunManager._build_run_config(factor_cfg, run, out_dir)
            path = write_yaml_config(out_dir / "run_config.yaml", config)

            self.assertTrue(path.exists())
            self.assertEqual(config["run"]["output_dir"], str(out_dir.resolve()))
            self.assertEqual(config["run"]["n_quantiles"], 5)
            self.assertEqual(config["factor_spec"]["compute_kwargs"], {"window": 4})
            self.assertEqual(config["weight"]["long_groups"], [4])

    def test_run_config_normalizes_dashboard_group_inputs(self) -> None:
        _script, _class_cfg, factor_cfg = get_factor_config("alpha101", "ALPHA12")
        request = RunRequest(
            factor_class="alpha101",
            name="ALPHA12",
            parameters={
                "weight_mode": "freeplay",
                "long_groups": "18,19",
                "short_groups": "0，1",
            },
        )
        run = DashboardRun(request)
        with tempfile.TemporaryDirectory() as tmp:
            config = RunManager._build_run_config(factor_cfg, run, Path(tmp) / "run")

        self.assertEqual(config["weight"]["mode"], "freeplay")
        self.assertEqual(config["weight"]["long_groups"], [18, 19])
        self.assertEqual(config["weight"]["short_groups"], [0, 1])

    def test_run_config_rejects_empty_freeplay_groups(self) -> None:
        _script, _class_cfg, factor_cfg = get_factor_config("alpha101", "ALPHA12")
        request = RunRequest(
            factor_class="alpha101",
            name="ALPHA12",
            parameters={
                "weight_mode": "freeplay",
                "long_groups": "",
                "short_groups": "",
            },
        )
        run = DashboardRun(request)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "freeplay"):
                RunManager._build_run_config(factor_cfg, run, Path(tmp) / "run")

    def test_timing_run_config_allows_empty_freeplay_groups(self) -> None:
        _script, _class_cfg, factor_cfg = get_factor_config("tdx", "XICHOU_timing")
        request = RunRequest(
            factor_class="tdx",
            name="XICHOU_timing",
            parameters={
                "weight_mode": "freeplay",
                "long_groups": "",
                "short_groups": "",
            },
        )
        run = DashboardRun(request)
        with tempfile.TemporaryDirectory() as tmp:
            config = RunManager._build_run_config(factor_cfg, run, Path(tmp) / "run")

        self.assertEqual(config["meta"]["strategy_type"], "timing")
        self.assertIsNone(config["weight"]["long_groups"])
        self.assertIsNone(config["weight"]["short_groups"])

    def test_run_config_preserves_nested_compute_kwargs(self) -> None:
        _script, _class_cfg, factor_cfg = get_factor_config("alpha101", "ALPHA12_timing")
        nested = {
            "stock_code": "000001.SZ",
            "signal_weight": {
                "method": "rolling_z",
                "window": 120,
                "sigma": 1.0,
                "side": "short",
            },
        }
        request = RunRequest(
            factor_class="alpha101",
            name="ALPHA12_timing",
            compute_kwargs=nested,
        )
        run = DashboardRun(request)
        with tempfile.TemporaryDirectory() as tmp:
            config = RunManager._build_run_config(factor_cfg, run, Path(tmp) / "run")

        self.assertEqual(config["factor_spec"]["compute_kwargs"], nested)


if __name__ == "__main__":
    unittest.main()
