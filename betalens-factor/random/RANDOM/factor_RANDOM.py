"""Reproducible multi-distribution random cross-sectional benchmark factor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
from factor_template_random import FactorPipeline, FactorSpec  # noqa: E402


_CONFIG_FILE = _FACTOR_DIR / "factor_RANDOM.yaml"
_REQUIRED_SECTIONS = ("meta", "factor_spec", "weight", "run")


def load_config(path: str | Path = _CONFIG_FILE) -> dict:
    return load_yaml_config(path, required_sections=_REQUIRED_SECTIONS)


_DISTRIBUTION_ALIASES = {
    "gaussian": "normal",
    "exp": "exponential",
    "log_normal": "lognormal",
    "student-t": "student_t",
    "t": "student_t",
}
_SUPPORTED_DISTRIBUTIONS = {
    "uniform",
    "normal",
    "poisson",
    "exponential",
    "lognormal",
    "beta",
    "student_t",
}


def _finite_parameter(params: dict, name: str) -> float:
    if name not in params:
        raise ValueError(f"distribution_params 缺少参数: {name}")
    value = float(params[name])
    if not np.isfinite(value):
        raise ValueError(f"distribution_params.{name} 必须是有限数值")
    return value


def _draw_random_values(
    rng: np.random.Generator,
    shape: tuple[int, int],
    distribution: str,
    params: dict,
) -> np.ndarray:
    mode = str(distribution).strip().lower().replace("-", "_")
    mode = _DISTRIBUTION_ALIASES.get(mode, mode)
    if mode not in _SUPPORTED_DISTRIBUTIONS:
        supported = ", ".join(sorted(_SUPPORTED_DISTRIBUTIONS))
        raise ValueError(f"不支持的随机分布 {distribution!r}；可选: {supported}")

    if mode == "uniform":
        low = _finite_parameter(params, "low")
        high = _finite_parameter(params, "high")
        if high <= low:
            raise ValueError("uniform 要求 high > low")
        return rng.uniform(low, high, size=shape)

    if mode == "normal":
        mean = _finite_parameter(params, "mean")
        std = _finite_parameter(params, "std")
        if std <= 0:
            raise ValueError("normal 要求 std > 0")
        return rng.normal(mean, std, size=shape)

    if mode == "poisson":
        lam = _finite_parameter(params, "lam")
        tie_break_epsilon = _finite_parameter(params, "tie_break_epsilon")
        if lam < 0:
            raise ValueError("poisson 要求 lam >= 0")
        if not 0 < tie_break_epsilon < 1:
            raise ValueError("poisson 要求 0 < tie_break_epsilon < 1")
        draws = rng.poisson(lam, size=shape).astype(float)
        # Discrete draws contain ties. Seeded sub-integer jitter keeps the
        # Poisson count in floor(value) while making group assignment random.
        random_order = rng.random(shape).argsort(axis=1).argsort(axis=1)
        offsets = (random_order + 1) / (shape[1] + 1) * tie_break_epsilon
        return draws + offsets

    if mode == "exponential":
        scale = _finite_parameter(params, "scale")
        if scale <= 0:
            raise ValueError("exponential 要求 scale > 0")
        return rng.exponential(scale, size=shape)

    if mode == "lognormal":
        mean = _finite_parameter(params, "mean")
        sigma = _finite_parameter(params, "sigma")
        if sigma <= 0:
            raise ValueError("lognormal 要求 sigma > 0")
        return rng.lognormal(mean, sigma, size=shape)

    if mode == "beta":
        alpha = _finite_parameter(params, "alpha")
        beta = _finite_parameter(params, "beta")
        if alpha <= 0 or beta <= 0:
            raise ValueError("beta 要求 alpha > 0 且 beta > 0")
        return rng.beta(alpha, beta, size=shape)

    df = _finite_parameter(params, "df")
    if df <= 0:
        raise ValueError("student_t 要求 df > 0")
    return rng.standard_t(df, size=shape)


def compute_random(
    close_wide: pd.DataFrame,
    seed: int,
    distribution: str,
    distribution_params: dict,
) -> pd.DataFrame:
    """Generate seeded random scores while using close only as an availability mask."""
    ordered = close_wide.sort_index().sort_index(axis=1)
    rng = np.random.default_rng(int(seed))
    values = pd.DataFrame(
        _draw_random_values(
            rng,
            ordered.shape,
            distribution,
            dict(distribution_params),
        ),
        index=ordered.index,
        columns=ordered.columns,
    )
    values = values.where(ordered.notna())
    return values.reindex(index=close_wide.index, columns=close_wide.columns)


def build_spec(config: dict, config_path: str | Path = _CONFIG_FILE) -> FactorSpec:
    options = factor_spec_options(config, config_path)
    return FactorSpec(
        name=str(section(config, "meta")["name"]),
        compute=compute_random,
        **options,
    )


spec = build_spec(load_config())


def run_from_config(config_path: str | Path = _CONFIG_FILE):
    config = load_config(config_path)
    kwargs = run_parameters(config, config_path)
    start_date = kwargs.pop("start_date")
    end_date = kwargs.pop("end_date")
    Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
    return FactorPipeline(build_spec(config, config_path)).run(
        start_date,
        end_date,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RANDOM from its YAML parameter file."
    )
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="YAML parameter file")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
