from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class ConfigError(ValueError):
    """Raised when a YAML parameter file is missing required structure."""


def load_yaml_config(
    path: str | Path,
    *,
    required_sections: Sequence[str] = (),
) -> dict[str, Any]:
    """Load a YAML config file and validate required top-level sections."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"YAML config not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping")
    for section_name in required_sections:
        section(loaded, section_name, context=str(config_path))
    return loaded


def write_yaml_config(path: str | Path, config: Mapping[str, Any]) -> Path:
    """Write a complete runtime YAML config copy."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(
            dict(config),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return out


def section(config: Mapping[str, Any], name: str, *, context: str = "config") -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"{context}: section '{name}' must be a mapping")
    return value


def require_keys(mapping: Mapping[str, Any], keys: Sequence[str], *, context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"{context}: missing required key(s): {', '.join(missing)}")


def resolve_path(value: str | Path, base_dir: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir) / path).resolve()


def resolve_run_output_dir(config: Mapping[str, Any], config_path: str | Path) -> Path:
    run = section(config, "run", context=str(config_path))
    require_keys(run, ("output_dir",), context=f"{config_path}: run")
    return resolve_path(run["output_dir"], Path(config_path).parent)


def factor_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    meta = section(config, "meta")
    factor_spec = section(config, "factor_spec")
    return {
        "name": meta.get("name", ""),
        "formula": meta.get("formula", ""),
        "logic": meta.get("logic", ""),
        "inputs": {
            **dict(factor_spec.get("inputs", {}) or {}),
            **dict(factor_spec.get("industry_inputs", {}) or {}),
        },
        "compute_kwargs": factor_spec.get("compute_kwargs", {}) or {},
    }


def run_parameters(config: Mapping[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Return concrete FactorPipeline.run kwargs from a complete factor YAML."""
    run = section(config, "run", context=str(config_path))
    require_keys(
        run,
        (
            "start_date",
            "end_date",
            "rebal_freq",
            "n_quantiles",
            "initial_amount",
            "include_profiling",
            "dump_excel",
            "output_dir",
        ),
        context=f"{config_path}: run",
    )
    params = {
        "start_date": str(run["start_date"]),
        "end_date": str(run["end_date"]),
        "rebal_freq": str(run["rebal_freq"]),
        "n_quantiles": int(run["n_quantiles"]),
        "initial_amount": float(run["initial_amount"]),
        "include_profiling": bool(run["include_profiling"]),
        "dump_excel": bool(run["dump_excel"]),
        "output_dir": str(resolve_run_output_dir(config, config_path)),
    }
    optional_bool = ("pretom_only", "verbose")
    optional_int = ("warmup_days", "pretom_lo", "pretom_hi")
    for key in optional_bool:
        if key in run:
            params[key] = bool(run[key])
    for key in optional_int:
        if key in run and run[key] is not None:
            params[key] = int(run[key])
    if "universe" in run:
        params["universe"] = run["universe"]
    if "benchmark_code" in run and run["benchmark_code"] is not None:
        params["benchmark_code"] = str(run["benchmark_code"])
    return params


def factor_spec_options(config: Mapping[str, Any], config_path: str | Path) -> dict[str, Any]:
    factor_spec = section(config, "factor_spec", context=str(config_path))
    weight = section(config, "weight", context=str(config_path))
    require_keys(
        factor_spec,
        (
            "inputs",
            "compute_kwargs",
            "direction",
            "table_name",
            "index_code",
            "use_industry",
            "use_mktcap",
            "industry_scheme",
            "backtest_metric",
        ),
        context=f"{config_path}: factor_spec",
    )
    require_keys(
        weight,
        ("mode", "long_groups", "short_groups"),
        context=f"{config_path}: weight",
    )
    options = {
        "inputs": dict(factor_spec["inputs"]),
        "compute_kwargs": dict(factor_spec["compute_kwargs"] or {}),
        "direction": str(factor_spec["direction"]),
        "table_name": str(factor_spec["table_name"]),
        "index_code": factor_spec.get("index_code"),
        "use_industry": bool(factor_spec["use_industry"]),
        "use_mktcap": bool(factor_spec["use_mktcap"]),
        "industry_scheme": str(factor_spec["industry_scheme"]),
        "backtest_metric": str(factor_spec["backtest_metric"]),
        "long_groups": weight.get("long_groups"),
        "short_groups": weight.get("short_groups"),
        "weight_mode": str(weight["mode"]),
        "group_weights": dict(weight.get("group_weights") or {}),
        "intra_group_allocation": dict(weight.get("intra_group_allocation") or {}),
    }
    if "industry_inputs" in factor_spec:
        options["industry_inputs"] = dict(factor_spec.get("industry_inputs") or {})
    if "required_history_bars" in factor_spec:
        options["required_history_bars"] = int(factor_spec.get("required_history_bars") or 0)
    if "mask_inputs_by_pit" in factor_spec:
        options["mask_inputs_by_pit"] = bool(factor_spec.get("mask_inputs_by_pit", False))
    return options
