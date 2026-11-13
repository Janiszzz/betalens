#%%
"""RSI_SLOW tdx factor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_FACTOR_DIR = Path(__file__).resolve().parent
_CLASS_DIR = _FACTOR_DIR.parent
_FACTOR_ROOT = _CLASS_DIR.parent
_REPO_ROOT = _FACTOR_ROOT.parent
for _path in (_REPO_ROOT, _CLASS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from betalens.factor.config import (  # noqa: E402
    factor_spec_options,
    load_yaml_config,
    section,
)
from factor_template_tdx import FactorPipeline, FactorSpec, REF, SMA, clean_inf  # noqa: E402


_CONFIG_FILE = _FACTOR_DIR / "factor_RSI_SLOW.yaml"
_REQUIRED_SECTIONS = ("meta", "factor_spec", "weight", "run")


def load_config(path: str | Path = _CONFIG_FILE) -> dict:
    return load_yaml_config(path, required_sections=_REQUIRED_SECTIONS)


def compute_rsi_slow(close_wide, window):
    diff = close_wide - REF(close_wide, 1)
    up = SMA(diff.clip(lower=0), window, 1)
    ab = SMA(diff.abs(), window, 1)
    return clean_inf(up / ab * 100)


def build_spec(config: dict, config_path: str | Path = _CONFIG_FILE) -> FactorSpec:
    options = factor_spec_options(config, config_path)
    return FactorSpec(
        name=str(section(config, "meta")["name"]),
        compute=compute_rsi_slow,
        **options,
    )


spec = build_spec(load_config())


def run_from_config(config_path: str | Path = _CONFIG_FILE):
    from factor_runtime import run_factor_from_config
    return run_factor_from_config(config_path, build_spec, FactorPipeline)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RSI_SLOW from its YAML parameter file.")
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="YAML parameter file")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
