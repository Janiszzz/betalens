"""ALPHA45 cross-sectional factor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_FACTOR_DIR = Path(__file__).resolve().parent
_CLASS_DIR = _FACTOR_DIR.parent
_FACTOR_ROOT = _CLASS_DIR.parent
_REPO_ROOT = _FACTOR_ROOT.parent
for _path in (_REPO_ROOT, _FACTOR_ROOT, _CLASS_DIR, _FACTOR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section  # noqa: E402
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha  # noqa: E402
from factor_template_alpha101 import FactorSpec, FactorPipeline  # noqa: E402


_CONFIG_FILE = _FACTOR_DIR / "factor_ALPHA45.yaml"
_REQUIRED_SECTIONS = ("meta", "factor_spec", "weight", "run")


def load_config(path: str | Path = _CONFIG_FILE) -> dict:
    return load_yaml_config(path, required_sections=_REQUIRED_SECTIONS)


def compute_alpha45(
    close_wide,
    volume_wide,
    *,
    close_delay_lag=5,
    delay_close_sum_window=20,
    ts_sum_delay_close_divisor=20,
    close_volume_correlation_window=2,
    close_sum_window=5,
    close_sum_window_2=20,
    ts_sum_close_ts_sum_close_correlation_window=2,
):
    return compute_alpha(
        45,
        close_wide=close_wide,
        volume_wide=volume_wide,
        close_delay_lag=close_delay_lag,
        delay_close_sum_window=delay_close_sum_window,
        ts_sum_delay_close_divisor=ts_sum_delay_close_divisor,
        close_volume_correlation_window=close_volume_correlation_window,
        close_sum_window=close_sum_window,
        close_sum_window_2=close_sum_window_2,
        ts_sum_close_ts_sum_close_correlation_window=ts_sum_close_ts_sum_close_correlation_window,
    )


def build_spec(config: dict, config_path: str | Path = _CONFIG_FILE) -> FactorSpec:
    options = factor_spec_options(config, config_path)
    parameter_names = set(get_definition(45).parameters)
    formula_kwargs = {
        name: value
        for name, value in options.get("compute_kwargs", {}).items()
        if name in parameter_names
    }
    options["required_history_bars"] = required_history_bars_for_alpha(45, formula_kwargs)
    return FactorSpec(
        name=str(section(config, "meta")["name"]),
        compute=compute_alpha45,
        strategy_type="cross_section",
        **options,
    )


spec = build_spec(load_config())


def run_from_config(config_path: str | Path = _CONFIG_FILE):
    config = load_config(config_path)
    kwargs = run_parameters(config, config_path)
    start_date = kwargs.pop("start_date")
    end_date = kwargs.pop("end_date")
    Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
    return FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALPHA45 from YAML.")
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="YAML parameter file")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
