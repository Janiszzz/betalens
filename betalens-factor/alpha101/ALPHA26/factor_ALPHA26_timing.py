"""ALPHA26 single-stock timing strategy."""
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
from factor_template_alpha101 import FactorSpec, TimingFactorPipeline as FactorPipeline  # noqa: E402


_CONFIG_FILE = _FACTOR_DIR / "factor_ALPHA26_timing.yaml"
_REQUIRED_SECTIONS = ("meta", "factor_spec", "weight", "run")


def load_config(path: str | Path = _CONFIG_FILE) -> dict:
    return load_yaml_config(path, required_sections=_REQUIRED_SECTIONS)


def compute_alpha26_timing(
    high_wide,
    volume_wide,
    *,
    volume_rank_window=5,
    high_rank_window=5,
    ts_rank_volume_ts_rank_high_correlation_window=5,
    correlation_ts_rank_volume_maximum_window=3,
    stock_code=None,
    signal_weight=None,
):
    del stock_code, signal_weight
    return compute_alpha(
        26,
        high_wide=high_wide,
        volume_wide=volume_wide,
        volume_rank_window=volume_rank_window,
        high_rank_window=high_rank_window,
        ts_rank_volume_ts_rank_high_correlation_window=ts_rank_volume_ts_rank_high_correlation_window,
        correlation_ts_rank_volume_maximum_window=correlation_ts_rank_volume_maximum_window,
    )


def build_spec(config: dict, config_path: str | Path = _CONFIG_FILE) -> FactorSpec:
    options = factor_spec_options(config, config_path)
    parameter_names = set(get_definition(26).parameters)
    formula_kwargs = {
        name: value
        for name, value in options.get("compute_kwargs", {}).items()
        if name in parameter_names
    }
    options["required_history_bars"] = required_history_bars_for_alpha(26, formula_kwargs)
    return FactorSpec(
        name=str(section(config, "meta")["name"]),
        compute=compute_alpha26_timing,
        strategy_type="timing",
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
    parser = argparse.ArgumentParser(description="Run ALPHA26 timing from YAML.")
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="YAML parameter file")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
