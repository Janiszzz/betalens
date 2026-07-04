#%%
"""DISP dispensability factor."""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Mapping, Any

logging.getLogger("IndexUniverseQuery").setLevel(logging.WARNING)

_FACTOR_DIR = Path(__file__).resolve().parent
_CLASS_DIR = _FACTOR_DIR.parent
_FACTOR_ROOT = _CLASS_DIR.parent
_REPO_ROOT = _FACTOR_ROOT.parent
for _path in (_REPO_ROOT, _CLASS_DIR, _FACTOR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from betalens.factor.config import (  # noqa: E402
    factor_spec_options,
    load_yaml_config,
    run_parameters,
    section,
)
from factor_template_liqdemand import (  # noqa: E402
    FactorSpec,
    LiqDemandPipeline,
    clean_inf,
    get_pretom_dates,
)


FactorPipeline = LiqDemandPipeline
_CONFIG_FILE = _FACTOR_DIR / "factor_DISP.yaml"
_REQUIRED_SECTIONS = ("meta", "factor_spec", "weight", "run")


def load_config(path: str | Path = _CONFIG_FILE) -> dict:
    return load_yaml_config(path, required_sections=_REQUIRED_SECTIONS)


def compute_disp(close_wide, window):
    min_periods = max(20, int(window) // 2)
    ratio = close_wide / close_wide.rolling(int(window), min_periods=min_periods).max()
    return clean_inf(-ratio)


def build_spec(config: dict, config_path: str | Path = _CONFIG_FILE) -> FactorSpec:
    options = factor_spec_options(config, config_path)
    return FactorSpec(
        name=str(section(config, "meta")["name"]),
        compute=compute_disp,
        **options,
    )


spec = build_spec(load_config())


def _require_param(params: Mapping[str, Any], key: str) -> Any:
    if key not in params:
        raise KeyError(f"mining params missing required key: {key}")
    return params[key]


def make_mining_spec(params):
    window = int(_require_param(params, "window"))
    return dataclasses.replace(
        spec,
        compute_kwargs={"window": window},
    )


def mining_gid(params):
    window = int(_require_param(params, "window"))
    pretom = _require_param(params, "pretom")
    pretom_only = bool(_require_param(params, "pretom_only"))
    n_quantiles = int(_require_param(params, "n_quantiles"))
    timing = "PT" if pretom_only else "DLY"
    return f"w{window}_p{int(pretom[0])}-{int(pretom[1])}_{timing}_q{n_quantiles}"


def mining_warmup_days(params):
    window = int(_require_param(params, "window"))
    return int(window * 1.5) + 60


def mining_weight_hook(weights, task):
    params = task["params"]
    if not bool(_require_param(params, "pretom_only")):
        return weights
    pretom = _require_param(params, "pretom")
    dates = get_pretom_dates(
        task["win_start"],
        task["win_end"],
        lo=int(pretom[0]),
        hi=int(pretom[1]),
    )
    keep = [ts.date() in dates for ts in weights.index]
    return weights.loc[keep]


def mining_valid_report(params, rank, output_dir, start_date, end_date):
    mining_spec = dataclasses.replace(make_mining_spec(params), name=f"DISP_valid{rank}")
    pretom = _require_param(params, "pretom")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    LiqDemandPipeline(mining_spec).run(
        start_date,
        end_date,
        warmup_days=mining_warmup_days(params),
        pretom_only=bool(_require_param(params, "pretom_only")),
        pretom_lo=int(pretom[0]),
        pretom_hi=int(pretom[1]),
        n_quantiles=int(_require_param(params, "n_quantiles")),
        output_dir=output_dir,
        include_profiling=False,
        dump_excel=False,
        verbose=False,
    )


def run_from_config(config_path: str | Path = _CONFIG_FILE):
    config = load_config(config_path)
    kwargs = run_parameters(config, config_path)
    start_date = kwargs.pop("start_date")
    end_date = kwargs.pop("end_date")
    Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
    return LiqDemandPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DISP from its YAML parameter file.")
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="YAML parameter file")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
