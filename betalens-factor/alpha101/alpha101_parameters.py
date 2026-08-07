"""Parameter catalog and geometric candidate generation for Alpha101 mining."""
from __future__ import annotations

import itertools
import json
from typing import Any, Mapping

from alpha101_formulas import AlphaParameter, default_formula_params, get_definition


# Search ratios are deliberately multiplicative.  They cover several scales
# around the paper default without assuming that an additive step has the
# same meaning for a 3-day lookback and a 250-day lookback.
_GEOMETRIC_RATIOS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def parameter_catalog(alpha_id: str | int) -> dict[str, AlphaParameter]:
    """Return the stable parameter catalog for one Alpha formula."""
    return dict(get_definition(alpha_id).parameters)


def _unique(values) -> list[int | float]:
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _geometric(default: float, *, minimum: float = 0.0) -> list[float]:
    sign = -1.0 if default < 0 else 1.0
    magnitude = abs(float(default))
    if magnitude == 0:
        return [0.0]
    return _unique(
        [sign * max(minimum, magnitude * ratio) for ratio in _GEOMETRIC_RATIOS]
    )


def _bounded_weight(default: float) -> list[float]:
    """Search a [0, 1] mixing weight on a geometric odds scale."""
    value = min(1.0 - 1e-6, max(1e-6, float(default)))
    odds = value / (1.0 - value)
    values = []
    for ratio in _GEOMETRIC_RATIOS:
        scaled_odds = odds * ratio
        values.append(scaled_odds / (1.0 + scaled_odds))
    return _unique(values)


def candidate_values(spec: AlphaParameter) -> list[int | float]:
    """Return deterministic values spanning multiple multiplicative scales."""
    default = spec.default
    if spec.kind in {"window", "lag"}:
        return _unique(max(1, int(value + 0.5)) for value in _geometric(float(default), minimum=1.0))
    if spec.kind == "exponent":
        return _geometric(float(default), minimum=0.05)
    if spec.kind == "threshold":
        return _geometric(float(default), minimum=1e-8)
    center = float(default)
    if 0.1 <= center <= 0.9:
        return _bounded_weight(center)
    return _geometric(center, minimum=1e-8)


def formula_param_candidates(alpha_id: str | int, max_candidates: int = 256) -> list[dict[str, Any]]:
    """Build baseline, one-at-a-time, then pairwise candidates within a hard cap."""
    if int(max_candidates) < 1:
        raise ValueError("max_candidates must be >= 1")
    baseline = default_formula_params(alpha_id)
    specs = [spec for spec in parameter_catalog(alpha_id).values() if spec.searchable]
    candidates = [baseline]
    seen = {json.dumps(baseline, ensure_ascii=False, sort_keys=True, default=str)}

    def add(values: Mapping[str, Any]) -> bool:
        if len(candidates) >= int(max_candidates):
            return False
        candidate = dict(baseline)
        candidate.update(values)
        key = json.dumps(candidate, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
        return len(candidates) < int(max_candidates)

    deviations: dict[str, list[int | float]] = {}
    for spec in specs:
        values = [value for value in candidate_values(spec) if value != spec.default]
        deviations[spec.name] = values
        for value in values:
            if not add({spec.name: value}):
                return candidates

    for left, right in itertools.combinations(specs, 2):
        for left_value, right_value in itertools.product(deviations[left.name], deviations[right.name]):
            if not add({left.name: left_value, right.name: right_value}):
                return candidates
    return candidates


def formula_param_gid(alpha_id: str | int, params: Mapping[str, Any]) -> str:
    """Create a deterministic, filesystem-safe candidate identifier."""
    name = get_definition(alpha_id).name
    encoded = json.dumps(dict(params), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    # The full JSON remains in result rows; the compact id keeps filenames usable.
    import hashlib

    return f"{name}_{hashlib.sha1(encoded.encode('utf-8')).hexdigest()[:12]}"


def catalog_rows(alpha_id: str | int) -> list[dict[str, Any]]:
    definition = get_definition(alpha_id)
    return [
        {
            "alpha": definition.name,
            "parameter": spec.name,
            "kind": spec.kind,
            "default": spec.default,
            "searchable": spec.searchable,
            "source_line": spec.source_line,
        }
        for spec in definition.parameters.values()
    ]


__all__ = [
    "candidate_values",
    "catalog_rows",
    "formula_param_candidates",
    "formula_param_gid",
    "parameter_catalog",
]
