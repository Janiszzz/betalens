from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from betalens.factor.config import ConfigError, load_yaml_config, resolve_run_output_dir


class FactorConfigTests(unittest.TestCase):
    def test_load_complete_factor_yaml_and_resolve_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "factor_SAMPLE.yaml"
            config_path.write_text(
                """
meta:
  name: SAMPLE
factor_spec:
  inputs:
    close_wide: 收盘价(元)
  compute_kwargs: {}
  direction: positive
  table_name: daily_market
  index_code: 000906.SH
  use_industry: true
  use_mktcap: false
  industry_scheme: 申万一级行业
  backtest_metric: 收盘价(元)
weight:
  mode: freeplay
  long_groups: null
  short_groups: null
run:
  start_date: '2024-01-01'
  end_date: '2024-12-31'
  rebal_freq: W
  n_quantiles: 10
  initial_amount: 100000000
  include_profiling: true
  dump_excel: true
  output_dir: outputs/runs/manual
""",
                encoding="utf-8",
            )

            config = load_yaml_config(config_path, required_sections=("meta", "factor_spec", "weight", "run"))
            output_dir = resolve_run_output_dir(config, config_path)

        self.assertEqual(config["meta"]["name"], "SAMPLE")
        self.assertEqual(output_dir, (base / "outputs" / "runs" / "manual").resolve())

    def test_missing_required_section_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.yaml"
            path.write_text("meta:\n  name: BROKEN\n", encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "section 'factor_spec'"):
                load_yaml_config(path, required_sections=("meta", "factor_spec"))


if __name__ == "__main__":
    unittest.main()
