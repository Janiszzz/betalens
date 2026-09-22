"""Continuous extreme-effect variants of ALPHA12.

The original ALPHA12 mixes price direction, volume direction and absolute
price scale in one expression. This module keeps the same two raw inputs,
but makes both changes dimensionless, preserves their true signs, and uses a
continuous strength score instead of bins or cross-sectional ranks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Mapping

import numpy as np
import pandas as pd

from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section
from factor_template_alpha101 import FactorPipeline, FactorSpec, clean_inf

Effect = Literal["momentum", "reversal"]
REQUIRED_SECTIONS = ("meta", "factor_spec", "weight", "run")
_EPSILON = 1e-12


def _positive_int(value, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1 or numeric != parsed:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return parsed


def validate_lag(value, name: str) -> int:
    return _positive_int(value, name, maximum=180)


def validate_extreme_scale(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("extreme_scale must be between 1 and 4")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("extreme_scale must be between 1 and 4") from exc
    if not np.isfinite(result) or result < 1.0 or result > 4.0:
        raise ValueError("extreme_scale must be between 1 and 4")
    return result


def validate_trend_power(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("trend_power must be between 0 and 2")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("trend_power must be between 0 and 2") from exc
    if not np.isfinite(result) or result < 0.0 or result > 2.0:
        raise ValueError("trend_power must be between 0 and 2")
    return result


def _validate_effect(effect: str) -> Effect:
    normalized = str(effect).strip().lower()
    if normalized not in {"momentum", "reversal"}:
        raise ValueError("effect must be 'momentum' or 'reversal'")
    return normalized  # type: ignore[return-value]


def _log_changes(close_wide, volume_wide, *, close_delta_lag, volume_delta_lag):
    close_lag = validate_lag(close_delta_lag, "close_delta_lag")
    volume_lag = validate_lag(volume_delta_lag, "volume_delta_lag")
    close_log = np.log(close_wide.where(close_wide > 0))
    volume_log = np.log(volume_wide.where(volume_wide > 0))
    price_change = close_log - close_log.shift(close_lag)
    volume_change = volume_log - volume_log.shift(volume_lag)
    return clean_inf(price_change), clean_inf(volume_change), close_log, volume_log


def _cross_sectional_mad_scale(values: pd.DataFrame) -> pd.DataFrame:
    """Scale absolute changes by a same-date robust cross-sectional scale."""
    median = values.median(axis=1, skipna=True)
    mad = values.sub(median, axis=0).abs().median(axis=1, skipna=True)
    fallback = values.abs().median(axis=1, skipna=True)
    standard_deviation = values.std(axis=1, skipna=True)
    scale = mad.where(mad > _EPSILON, fallback)
    scale = scale.where(scale > _EPSILON, standard_deviation)
    scale = scale.where(scale > _EPSILON, 1.0)
    return values.abs().div(scale.add(_EPSILON), axis=0)


def _efficiency_ratio(levels: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Kaufman-style path efficiency using only current and past observations."""
    increments = levels.diff().abs()
    path = increments.rolling(lag, min_periods=lag).sum()
    net = levels.diff(lag).abs()
    ratio = net.div(path.where(path > _EPSILON))
    return ratio.clip(lower=0.0, upper=1.0)


def compute_continuous(
    close_wide,
    volume_wide,
    *,
    volume_delta_lag,
    close_delta_lag,
    extreme_scale=2.0,
    trend_power=1.0,
    effect: Effect,
):
    """Compute one continuous ALPHA12 momentum or reversal signal."""
    close_lag = validate_lag(close_delta_lag, "close_delta_lag")
    volume_lag = validate_lag(volume_delta_lag, "volume_delta_lag")
    alpha = validate_extreme_scale(extreme_scale)
    beta = validate_trend_power(trend_power)
    selected_effect = _validate_effect(effect)

    price_change, volume_change, close_log, volume_log = _log_changes(
        close_wide,
        volume_wide,
        close_delta_lag=close_lag,
        volume_delta_lag=volume_lag,
    )
    price_strength = _cross_sectional_mad_scale(price_change)
    volume_strength = _cross_sectional_mad_scale(volume_change)
    price_extreme = np.tanh(price_strength).pow(alpha)
    volume_extreme = np.tanh(volume_strength).pow(alpha)

    price_efficiency = _efficiency_ratio(close_log, close_lag)
    volume_efficiency = _efficiency_ratio(volume_log, volume_lag)
    efficiency_product = price_efficiency * volume_efficiency
    trend_weight = efficiency_product.pow(beta / 2.0)
    trend_weight = trend_weight.where(efficiency_product.notna())

    strength = (price_extreme * volume_extreme).pow(0.5) * trend_weight
    valid = price_change.notna() & volume_change.notna() & strength.notna()
    product = price_change.mul(volume_change)
    same_direction = product.gt(0)
    selected = same_direction if selected_effect == "momentum" else product.lt(0)

    direction = np.sign(price_change)
    score = direction * strength
    if selected_effect == "reversal":
        score = -score
    # Keep valid non-target observations as explicit neutral zeros.
    score = score.where(selected, 0.0).where(valid)
    return clean_inf(score)


def compute_momentum(close_wide, volume_wide, **kwargs):
    return compute_continuous(close_wide, volume_wide, effect="momentum", **kwargs)


def compute_reversal(close_wide, volume_wide, **kwargs):
    return compute_continuous(close_wide, volume_wide, effect="reversal", **kwargs)


def required_history_bars(compute_kwargs: Mapping[str, object]) -> int:
    close_lag = validate_lag(compute_kwargs["close_delta_lag"], "close_delta_lag")
    volume_lag = validate_lag(compute_kwargs["volume_delta_lag"], "volume_delta_lag")
    validate_extreme_scale(compute_kwargs.get("extreme_scale", 2.0))
    validate_trend_power(compute_kwargs.get("trend_power", 1.0))
    return max(close_lag, volume_lag) + 1


def load_factor_config(path: str | Path) -> dict:
    return load_yaml_config(path, required_sections=REQUIRED_SECTIONS)


def build_factor_spec(config: Mapping[str, object], config_path: str | Path, *, compute: Callable) -> FactorSpec:
    options = factor_spec_options(config, config_path)
    options["required_history_bars"] = required_history_bars(options["compute_kwargs"])
    return FactorSpec(
        name=str(section(config, "meta")["name"]),
        compute=compute,
        strategy_type="cross_section",
        **options,
    )


def run_factor_from_config(config_path: str | Path, build_spec: Callable):
    config = load_factor_config(config_path)
    kwargs = run_parameters(config, config_path)
    start_date = kwargs.pop("start_date")
    end_date = kwargs.pop("end_date")
    Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
    return FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs)


__all__ = [
    "FactorPipeline", "FactorSpec", "build_factor_spec", "compute_continuous",
    "compute_momentum", "compute_reversal", "load_factor_config",
    "required_history_bars", "run_factor_from_config", "validate_extreme_scale",
    "validate_lag", "validate_trend_power",
]
