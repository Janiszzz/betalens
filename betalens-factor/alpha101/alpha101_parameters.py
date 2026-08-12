"""Explicit Alpha101 parameter catalogs and bounded grid expansion."""
from __future__ import annotations

import itertools
import json
import math
from typing import Any, Mapping, Sequence

from alpha101_formulas import AlphaParameter, default_compute_kwargs, get_definition


def parameter_catalog(alpha_id: str | int) -> dict[str, AlphaParameter]:
    """Return the ordered parameter catalog for one Alpha formula."""
    return dict(get_definition(alpha_id).parameters)


def _unique(values: Sequence[int | float]) -> list[int | float]:
    output: list[int | float] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def candidate_values(spec: AlphaParameter) -> list[int | float]:
    """Return a compact, auditable three-point range around a paper default."""
    default = spec.default
    if spec.kind in {"window", "lag"}:
        if isinstance(default, float):
            center = float(default)
            return _unique([max(1.0, center * 0.5), center, max(1.0, center * 2.0)])
        center = max(1, int(default))
        return _unique([max(1, int(math.floor(center * 0.5 + 0.5))), center, max(1, center * 2)])
    if spec.kind == "weight":
        center = float(default)
        return _unique([max(0.0, center * 0.5), center, min(1.0, center * 1.5)])
    if spec.kind == "threshold":
        center = float(default)
        spread = max(abs(center) * 0.5, 0.05)
        return _unique([center - spread, center, center + spread])
    if spec.kind == "exponent":
        center = float(default)
        return _unique([center * 0.5, center, center * 2.0])
    return [default]


def default_search_space(alpha_id: str | int, max_dimensions: int = 3) -> dict[str, list[int | float]]:
    """Build the explicit YAML search space, varying at most three parameters."""
    if int(max_dimensions) < 0:
        raise ValueError("max_dimensions must be >= 0")
    remaining = int(max_dimensions)
    search_space: dict[str, list[int | float]] = {}
    for name, spec in parameter_catalog(alpha_id).items():
        values = [spec.default]
        if spec.searchable and remaining > 0:
            proposed = candidate_values(spec)
            if len(proposed) > 1:
                values = proposed
                remaining -= 1
        search_space[name] = values
    return search_space


def validate_search_space(
    alpha_id: str | int,
    search_space: Mapping[str, Sequence[Any]],
) -> dict[str, list[int | float]]:
    """Require one non-empty, numeric value list for every formula parameter."""
    definition = get_definition(alpha_id)
    expected = set(definition.parameters)
    supplied = set(search_space)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"{definition.name} search_space mismatch: {'; '.join(details)}")

    validated: dict[str, list[int | float]] = {}
    for name, spec in definition.parameters.items():
        raw_values = search_space[name]
        if not isinstance(raw_values, (list, tuple)) or not raw_values:
            raise ValueError(f"{definition.name}.{name} must be a non-empty YAML list")
        values: list[int | float] = []
        for raw in raw_values:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"{definition.name}.{name} values must be numeric")
            value = float(raw) if isinstance(spec.default, float) or isinstance(raw, float) else int(raw)
            if not math.isfinite(float(value)):
                raise ValueError(f"{definition.name}.{name} values must be finite")
            if spec.kind in {"window", "lag"} and value < 1:
                raise ValueError(f"{definition.name}.{name} values must be >= 1")
            if spec.kind == "exponent" and value == 0:
                raise ValueError(f"{definition.name}.{name} values must be non-zero")
            if value not in values:
                values.append(value)
        validated[name] = values
    return validated


def grid_candidate_count(search_space: Mapping[str, Sequence[Any]]) -> int:
    return math.prod(len(values) for values in search_space.values())


def formula_param_candidates(
    alpha_id: str | int,
    search_space: Mapping[str, Sequence[Any]],
    *,
    max_grid_candidates: int = 256,
) -> list[dict[str, Any]]:
    """Expand the complete grid and fail before execution if it exceeds the cap."""
    validated = validate_search_space(alpha_id, search_space)
    count = grid_candidate_count(validated)
    if count > int(max_grid_candidates):
        raise ValueError(
            f"{get_definition(alpha_id).name} grid has {count} candidates, "
            f"exceeding max_grid_candidates={int(max_grid_candidates)}"
        )
    names = list(validated)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(validated[name] for name in names))
    ]


def formula_param_gid(alpha_id: str | int, params: Mapping[str, Any]) -> str:
    """Create a deterministic, filesystem-safe candidate identifier."""
    name = get_definition(alpha_id).name
    encoded = json.dumps(dict(params), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
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
    "default_compute_kwargs",
    "default_search_space",
    "formula_param_candidates",
    "formula_param_gid",
    "grid_candidate_count",
    "parameter_catalog",
    "validate_search_space",
]
