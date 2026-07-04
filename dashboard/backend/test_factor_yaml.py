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
        disp = next(item for item in factors if item.factor_class == "LiqDemand" and item.name == "DISP")
        self.assertEqual(disp.defaults["n_quantiles"], 20)
        self.assertEqual(disp.defaults["direction"], "negative")

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


if __name__ == "__main__":
    unittest.main()
