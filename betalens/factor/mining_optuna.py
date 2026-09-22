"""Optuna adapter for the two-stage mining search.

This module intentionally has no worker, cache, audit, or storage concerns.
Those belong to :mod:`betalens.factor.mining`.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import warnings
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _optuna():
    try:
        import optuna  # type: ignore
        return optuna
    except ImportError as exc:
        raise ImportError(
            "parameter mining requires Optuna; install it with `pip install optuna`"
        ) from exc


def _choice_token(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return "__json__:" + json.dumps(value, ensure_ascii=False, sort_keys=True)


def _decode_choice(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("__json__:"):
        return json.loads(value[len("__json__:"):])
    return value


def to_optuna_distribution(spec: Mapping[str, Any]):
    kind = str(spec.get("type", "float")).lower()
    optuna = _optuna()
    if kind in {"categorical", "choice", "bool", "boolean"}:
        choices = list(spec.get("choices", [False, True] if kind in {"bool", "boolean"} else []))
        tokens = [_choice_token(value) for value in choices]
        return optuna.distributions.CategoricalDistribution(tokens)
    low, high = spec.get("low"), spec.get("high")
    if low is None or high is None or float(low) > float(high):
        raise ValueError("numeric parameter requires low <= high")
    log = str(spec.get("scale", "linear")).lower() == "log"
    if kind in {"int", "integer"}:
        if log and int(low) < 1:
            raise ValueError("log integer parameters must have low >= 1")
        return optuna.distributions.IntDistribution(
            int(low), int(high), step=int(spec.get("step") or 1), log=log
        )
    return optuna.distributions.FloatDistribution(
        float(low), float(high), step=spec.get("step"), log=log
    )


def suggest_params(trial, parameter_specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Suggest one ordinary parameter dictionary from an Optuna trial."""
    output = {}
    for name, spec in parameter_specs.items():
        kind = str(spec.get("type", "float")).lower()
        if kind in {"categorical", "choice", "bool", "boolean"}:
            choices = list(
                spec.get("choices", [False, True] if kind in {"bool", "boolean"} else [])
            )
            value = trial.suggest_categorical(name, [_choice_token(item) for item in choices])
        elif kind in {"int", "integer"}:
            value = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step") or 1),
                log=str(spec.get("scale", "linear")).lower() == "log",
            )
        else:
            value = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                step=spec.get("step"),
                log=str(spec.get("scale", "linear")).lower() == "log",
            )
        output[name] = _decode_choice(value)
    return output


def create_coarse_study(config: Mapping[str, Any], *, direction: str = "maximize"):
    """Create the sampler-only study used by the coarse stage."""
    optuna = _optuna()
    seed = int(config.get("seed", 20260818))
    sampler_name = str(config.get("sampler", "random")).lower()
    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=max(1, int(config.get("n_startup_trials", 10))),
            multivariate=bool(config.get("multivariate", False)),
        )
    elif sampler_name == "qmc":
        qmc_type = str(config.get("qmc_type", "sobol")).lower()
        if qmc_type not in {"sobol", "halton"}:
            raise ValueError("qmc_type must be sobol or halton")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            sampler = optuna.samplers.QMCSampler(
                qmc_type=qmc_type,
                scramble=bool(config.get("scramble", True)),
                seed=seed,
                warn_asynchronous_seeding=False,
                warn_independent_sampling=False,
            )
    else:
        raise ValueError(f"unsupported coarse sampler: {sampler_name}")
    return optuna.create_study(sampler=sampler, direction=direction)


def create_fine_grid_study(
    search_space: Mapping[str, Sequence[Any]],
    *,
    direction: str = "maximize",
    seed: int = 20260818,
):
    """Create a GridSampler study while preserving composite categorical values."""
    optuna = _optuna()
    encoded = {
        name: [_choice_token(value) for value in values]
        for name, values in search_space.items()
    }
    if not encoded or any(not values for values in encoded.values()):
        raise ValueError("fine grid search space must contain non-empty dimensions")
    return optuna.create_study(
        sampler=optuna.samplers.GridSampler(encoded, seed=int(seed)),
        direction=direction,
    )


def tell_trial(study, trial, value: float | None) -> None:
    """Complete an externally evaluated trial, including GridSampler exhaustion."""
    try:
        study.tell(trial, value)
    except RuntimeError as exc:
        # GridSampler saves the final trial, then calls Study.stop(). Optuna only
        # permits that call from optimize(), not from the documented ask/tell API.
        if "Study.stop" not in str(exc):
            raise


def seed_study_with_results(
    study,
    parameter_specs: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    values: Sequence[float],
) -> int:
    """将已完成的外部候选作为历史 trial 导入新的自适应 study。"""
    if len(candidates) != len(values):
        raise ValueError("seed candidates and values must have the same length")
    optuna = _optuna()
    distributions = {
        name: to_optuna_distribution(spec)
        for name, spec in parameter_specs.items()
    }
    added = 0
    for candidate, value in zip(candidates, values):
        if value is None or not math.isfinite(float(value)):
            continue
        params = {}
        for name, spec in parameter_specs.items():
            raw = candidate[name]
            kind = str(spec.get("type", "float")).lower()
            params[name] = _choice_token(raw) if kind in {"categorical", "choice", "bool", "boolean"} else raw
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=float(value),
            )
        )
        added += 1
    return added


def _values(spec: Mapping[str, Any], count: int) -> list[Any]:
    kind = str(spec.get("type", "float")).lower()
    if kind in {"categorical", "choice", "bool", "boolean"}:
        return list(spec.get("choices", [False, True] if kind in {"bool", "boolean"} else []))
    low, high = float(spec["low"]), float(spec["high"])
    count = max(1, int(count))
    if count == 1 or low == high:
        values = [low]
    elif str(spec.get("scale", "linear")).lower() == "log":
        if low <= 0:
            raise ValueError("log parameter requires low > 0")
        values = [math.exp(math.log(low) + (math.log(high) - math.log(low)) * i / (count - 1)) for i in range(count)]
    else:
        values = [low + (high - low) * i / (count - 1) for i in range(count)]
    if kind in {"int", "integer"}:
        step = int(spec.get("step") or 1)
        return list(
            dict.fromkeys(
                max(int(low), min(int(high), int(low + round((value - low) / step) * step)))
                for value in values
            )
        )
    step = spec.get("step")
    if step:
        decimals = max(0, len(str(step).split(".")[-1])) if "." in str(step) else 0
        values = [round(round((value - low) / float(step)) * float(step) + low, decimals) for value in values]
    return list(dict.fromkeys(values))


def _dedupe(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output, seen = [], set()
    for row in rows:
        key = json.dumps(dict(row), sort_keys=True, ensure_ascii=False, default=str)
        if key not in seen:
            seen.add(key)
            output.append(dict(row))
    return output


def _unique_values(values: Sequence[Any]) -> list[Any]:
    output, seen = [], set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def generate_coarse_candidates(parameters: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Generate sparse candidates using an ask/tell Optuna study."""
    n_trials = max(1, int(config.get("n_trials", 32)))
    study = create_coarse_study(config)
    rows = []
    for _ in range(n_trials):
        trial = study.ask()
        rows.append(suggest_params(trial, parameters))
        study.tell(trial, 0.0)
    return _dedupe(rows)


@dataclass(frozen=True)
class FineGridPlan:
    candidates: list[dict[str, Any]]
    anchors: list[dict[str, Any]]
    dimensions: dict[str, list[Any]]
    local_bounds: dict[str, dict[str, Any]]


def _numeric_position(value: float, spec: Mapping[str, Any]) -> float:
    low, high = float(spec["low"]), float(spec["high"])
    if high == low:
        return 0.5
    if str(spec.get("scale", "linear")).lower() == "log" and low > 0 and value > 0:
        return (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    return (value - low) / (high - low)


def detect_boundary_pressure(
    parameters: Mapping[str, Mapping[str, Any]],
    winners: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 0.1,
    winner_ratio: float = 0.67,
) -> dict[str, dict[str, Any]]:
    """识别优胜候选持续靠近同一参数边界的维度。"""
    if not 0 <= float(tolerance) < 0.5:
        raise ValueError("boundary tolerance must be in [0, 0.5)")
    if not 0 < float(winner_ratio) <= 1:
        raise ValueError("boundary winner_ratio must be in (0, 1]")
    pressure = {}
    for name, spec in parameters.items():
        kind = str(spec.get("type", "float")).lower()
        if kind in {"categorical", "choice", "bool", "boolean"} or float(spec["low"]) == float(spec["high"]):
            continue
        positions = [_numeric_position(float(row[name]), spec) for row in winners if name in row]
        if not positions:
            continue
        low_ratio = sum(value <= tolerance for value in positions) / len(positions)
        high_ratio = sum(value >= 1 - tolerance for value in positions) / len(positions)
        sides = []
        if low_ratio >= winner_ratio:
            sides.append("low")
        if high_ratio >= winner_ratio:
            sides.append("high")
        if sides:
            pressure[name] = {
                "sides": sides,
                "low_ratio": low_ratio,
                "high_ratio": high_ratio,
                "winner_count": len(positions),
            }
    return pressure


def expand_parameter_specs(
    parameters: Mapping[str, Mapping[str, Any]],
    pressure: Mapping[str, Mapping[str, Any]],
    *,
    multiplier: float = 3.0,
    limits: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """根据边界压力向命中侧扩展参数，并受硬边界限制。"""
    if not math.isfinite(float(multiplier)) or float(multiplier) <= 1:
        raise ValueError("expansion multiplier must be finite and > 1")
    output = {name: dict(spec) for name, spec in parameters.items()}
    for name, hit in pressure.items():
        if name not in output:
            continue
        spec = output[name]
        low, high = float(spec["low"]), float(spec["high"])
        sides = set(hit.get("sides") or [])
        log_scale = str(spec.get("scale", "linear")).lower() == "log"
        if log_scale:
            proposed_low = low / float(multiplier) if "low" in sides else low
            proposed_high = high * float(multiplier) if "high" in sides else high
        else:
            width = max(high - low, abs(low), abs(high), 1e-12)
            proposed_low = low - width * (float(multiplier) - 1) if "low" in sides else low
            proposed_high = high + width * (float(multiplier) - 1) if "high" in sides else high
        hard = dict((limits or {}).get(name, {}))
        if hard.get("low") is not None:
            proposed_low = max(proposed_low, float(hard["low"]))
        if hard.get("high") is not None:
            proposed_high = min(proposed_high, float(hard["high"]))
        kind = str(spec.get("type", "float")).lower()
        if kind in {"int", "integer"}:
            proposed_low, proposed_high = int(math.floor(proposed_low)), int(math.ceil(proposed_high))
        spec["low"], spec["high"] = proposed_low, proposed_high
    return output


@dataclass(frozen=True)
class PerturbationPlan:
    candidates: list[dict[str, Any]]
    metadata: dict[str, dict[str, Any]]


def generate_perturbation_candidates(
    parameters: Mapping[str, Mapping[str, Any]],
    winners: Sequence[Mapping[str, Any]],
    *,
    perturbations_per_candidate: int = 8,
    radius_ratio: float = 0.1,
    seed: int = 20260818,
) -> PerturbationPlan:
    """在赢家附近按参数尺度生成可复现的随机扰动候选。"""
    if int(perturbations_per_candidate) < 1:
        raise ValueError("perturbations_per_candidate must be positive")
    if not 0 < float(radius_ratio) <= 0.5:
        raise ValueError("stability radius_ratio must be in (0, 0.5]")
    rng = random.Random(int(seed))
    candidates, metadata, seen = [], {}, set()
    for winner_rank, winner in enumerate(winners, 1):
        parent_id = str(winner.get("candidate_id", ""))
        base = {name: winner[name] for name in parameters if name in winner}
        attempts = 0
        generated = 0
        while generated < int(perturbations_per_candidate) and attempts < int(perturbations_per_candidate) * 20:
            attempts += 1
            candidate = dict(base)
            changed = False
            for name, spec in parameters.items():
                kind = str(spec.get("type", "float")).lower()
                if name not in candidate or kind in {"categorical", "choice", "bool", "boolean"}:
                    continue
                low, high = float(spec["low"]), float(spec["high"])
                if low == high:
                    continue
                center = float(candidate[name])
                if str(spec.get("scale", "linear")).lower() == "log":
                    radius = (math.log(high) - math.log(low)) * float(radius_ratio)
                    value = math.exp(math.log(center) + rng.uniform(-radius, radius))
                else:
                    value = center + rng.uniform(-1, 1) * (high - low) * float(radius_ratio)
                value = max(low, min(high, value))
                if kind in {"int", "integer"}:
                    step = int(spec.get("step") or 1)
                    value = int(low + round((value - low) / step) * step)
                    value = max(int(low), min(int(high), value))
                elif spec.get("step"):
                    step = float(spec["step"])
                    value = low + round((value - low) / step) * step
                    value = max(low, min(high, value))
                changed = changed or value != candidate[name]
                candidate[name] = value
            token = json.dumps(candidate, ensure_ascii=False, sort_keys=True, default=str)
            if not changed or token in seen:
                continue
            seen.add(token)
            candidates.append(candidate)
            metadata[token] = {
                "parent_candidate_id": parent_id,
                "winner_rank": winner_rank,
                "perturbation_index": generated + 1,
            }
            generated += 1
    return PerturbationPlan(candidates, metadata)


def generate_fine_candidates(
    parameters: Mapping[str, Mapping[str, Any]],
    anchors: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    coarse_candidates: Sequence[Mapping[str, Any]] | None = None,
) -> FineGridPlan:
    """Generate a bounded local grid around coarse winners."""
    if not anchors:
        return FineGridPlan([], [], {}, {})
    points = max(1, int(config.get("points_per_dimension", 7)))
    max_candidates = max(1, int(config.get("max_candidates", 256)))
    dimensions: dict[str, list[Any]] = {}
    local_bounds: dict[str, dict[str, Any]] = {}
    for name, spec in parameters.items():
        kind = str(spec.get("type", "float")).lower()
        if kind in {"categorical", "choice", "bool", "boolean"}:
            dimensions[name] = list(spec.get("choices", [False, True] if kind in {"bool", "boolean"} else []))
            local_bounds[name] = {"type": "categorical", "values": dimensions[name]}
            continue
        values = sorted(float(row[name]) for row in anchors if name in row)
        low, high = float(spec["low"]), float(spec["high"])
        sampled = sorted(
            set(
                float(row[name])
                for row in (coarse_candidates or [])
                if name in row
            )
        )
        local_lows, local_highs = [], []
        for anchor in values:
            position = bisect_left(sampled, anchor)
            if sampled:
                left = sampled[position - 1] if position > 0 else low
                next_position = position + 1 if position < len(sampled) and sampled[position] == anchor else position
                right = sampled[next_position] if next_position < len(sampled) else high
            else:
                gap = (high - low) / max(points - 1, 1)
                left, right = anchor - gap, anchor + gap
            local_lows.append(left)
            local_highs.append(right)
        local = dict(spec)
        local["low"] = max(low, min(local_lows))
        local["high"] = min(high, max(local_highs))
        dimensions[name] = _values(local, points)
        local_bounds[name] = {
            "type": kind,
            "global_low": spec["low"],
            "global_high": spec["high"],
            "local_low": local["low"],
            "local_high": local["high"],
            "values": dimensions[name],
        }
    names = list(dimensions)
    combinations = itertools.islice(
        itertools.product(*(dimensions[name] for name in names)),
        max_candidates,
    )
    candidates = [dict(zip(names, values)) for values in combinations]
    return FineGridPlan(
        candidates=_dedupe(candidates),
        anchors=[dict(value) for value in anchors],
        dimensions=dimensions,
        local_bounds=local_bounds,
    )


__all__ = [
    "create_coarse_study",
    "create_fine_grid_study",
    "FineGridPlan",
    "PerturbationPlan",
    "detect_boundary_pressure",
    "expand_parameter_specs",
    "generate_coarse_candidates",
    "generate_fine_candidates",
    "generate_perturbation_candidates",
    "suggest_params",
    "seed_study_with_results",
    "tell_trial",
    "to_optuna_distribution",
]
