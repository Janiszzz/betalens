"""ALPHA12 continuous same-direction momentum factor."""
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

from alpha12_modified_common import (  # noqa: E402
    FactorPipeline,
    FactorSpec,
    build_factor_spec,
    compute_momentum,
    load_factor_config,
    run_factor_from_config,
)

_CONFIG_FILE = _FACTOR_DIR / "factor_ALPHA12_MOMENTUM.yaml"


def compute_alpha12_momentum(
    close_wide,
    volume_wide,
    *,
    volume_delta_lag,
    close_delta_lag,
    extreme_scale=2.0,
    trend_power=1.0,
):
    return compute_momentum(
        close_wide,
        volume_wide,
        volume_delta_lag=volume_delta_lag,
        close_delta_lag=close_delta_lag,
        extreme_scale=extreme_scale,
        trend_power=trend_power,
    )


def build_spec(config: dict, config_path: str | Path = _CONFIG_FILE) -> FactorSpec:
    return build_factor_spec(config, config_path, compute=compute_alpha12_momentum)


spec = build_spec(load_factor_config(_CONFIG_FILE))


def run_from_config(config_path: str | Path = _CONFIG_FILE):
    return run_factor_from_config(config_path, build_spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALPHA12_MOMENTUM from YAML.")
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="YAML parameter file")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
