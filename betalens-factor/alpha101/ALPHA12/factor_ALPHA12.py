#%%
"""Alpha#12 factor."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.getLogger("IndexUniverseQuery").setLevel(logging.WARNING)

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
    run_parameters,
    section,
)
from factor_template_alpha101 import (  # noqa: E402
    FactorPipeline,
    FactorSpec,
    clean_inf,
    delta,
    sign,
)


_CONFIG_FILE = _FACTOR_DIR / "factor_ALPHA12.yaml"
_REQUIRED_SECTIONS = ("meta", "factor_spec", "weight", "run")


def load_config(path: str | Path = _CONFIG_FILE) -> dict:
    return load_yaml_config(path, required_sections=_REQUIRED_SECTIONS)


def compute_alpha12(close_wide, volume_wide):
    return clean_inf(sign(delta(volume_wide, 1)) * (-1 * delta(close_wide, 1)))


def build_spec(config: dict, config_path: str | Path = _CONFIG_FILE) -> FactorSpec:
    options = factor_spec_options(config, config_path)
    return FactorSpec(
        name=str(section(config, "meta")["name"]),
        compute=compute_alpha12,
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
    parser = argparse.ArgumentParser(description="Run ALPHA12 from its YAML parameter file.")
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="YAML parameter file")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
