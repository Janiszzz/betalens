"""Run bounded, resumable walk-forward mining for the Alpha101 class."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_CLASS_DIR = _SCRIPT_DIR.parent
_FACTOR_ROOT = _CLASS_DIR.parent
_REPO_ROOT = _FACTOR_ROOT.parent
for _path in (_REPO_ROOT, _FACTOR_ROOT, _CLASS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from betalens.factor.config import load_yaml_config, resolve_path, section  # noqa: E402
from betalens.factor.mining import RollingMiningConfig, run_walk_forward  # noqa: E402
from alpha101_formulas import default_compute_kwargs, get_definition  # noqa: E402
from alpha101_mining import mining_warmup_days  # noqa: E402
from alpha101_parameters import (  # noqa: E402
    catalog_rows,
    formula_param_candidates,
    grid_candidate_count,
    validate_search_space,
)


_CONFIG_FILE = _SCRIPT_DIR / "walkforward.yaml"


def _schemes(value: Any, name: str) -> list[tuple[int, int]]:
    raw = itertools.product(value["window_lengths"], value["steps"]) if isinstance(value, dict) else value
    schemes = []
    for pair in raw:
        win_len, step = (int(item) for item in pair)
        if step <= win_len and (win_len, step) not in schemes:
            schemes.append((win_len, step))
    if not schemes:
        raise ValueError(f"{name} produced no valid schemes")
    return schemes


def _paired_schemes(value: Any, name: str) -> list[tuple[int, int, int]]:
    if isinstance(value, dict):
        raw = itertools.product(
            value["train_lengths"],
            value["test_lengths"],
            value["steps"],
        )
    else:
        raw = value
    schemes = []
    for item in raw:
        if len(item) != 3:
            raise ValueError(f"{name} entries must be (train_length, test_length, step)")
        train_len, test_len, step = (int(part) for part in item)
        if train_len < 1 or test_len < 1 or step < 1:
            raise ValueError(f"{name} values must be positive")
        entry = (train_len, test_len, step)
        if entry not in schemes:
            schemes.append(entry)
    if not schemes:
        raise ValueError(f"{name} produced no valid schemes")
    return schemes


def _date_span(value: Any, name: str) -> tuple[str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain [start_date, end_date]")
    start, end = (str(item) for item in value)
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError(f"{name} start_date must not exceed end_date")
    return start, end


def _alpha_ids(value: Any) -> list[int]:
    if value == "all":
        return list(range(1, 102))
    ids = []
    for item in value:
        text = str(item).upper().replace("ALPHA", "")
        number = get_definition(int(text)).number
        if number not in ids:
            ids.append(number)
    if not ids:
        raise ValueError("mining.factors must select at least one Alpha")
    return ids


def _config_hash(mining: Mapping[str, Any], alpha_id: int, search_space: Mapping[str, Any]) -> str:
    payload = {"mining": dict(mining), "alpha_id": alpha_id, "search_space": dict(search_space)}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _factor_config(
    mining: Mapping[str, Any],
    alpha_id: int,
    output_dir: Path,
    cache_dir: Path,
    rebuild_cache: bool,
    formula_search_space: Mapping[str, list[int | float]],
    config_hash: str,
    config_path: Path,
) -> RollingMiningConfig:
    rolling_mode = str(mining.get("rolling_mode", "split")).lower()
    if rolling_mode == "paired":
        paired_schemes = _paired_schemes(mining["paired_schemes"], "paired_schemes")
        rolling_span = _date_span(mining["rolling_span"], "rolling_span")
        # RollingMiningConfig keeps train/test for backward compatibility;
        # paired mode uses rolling_span as the authoritative discovery range.
        train = rolling_span
        test = rolling_span
        train_schemes = []
        test_schemes = []
    elif rolling_mode == "split":
        paired_schemes = None
        rolling_span = None
        train = _date_span(mining["train"], "train")
        test = _date_span(mining["test"], "test")
        train_schemes = _schemes(mining["train_schemes"], "train_schemes")
        test_schemes = _schemes(mining["test_schemes"], "test_schemes")
    else:
        raise ValueError(f"unknown rolling_mode: {rolling_mode}")
    grid: dict[str, list[Any]] = {
        "alpha_id": [alpha_id],
        **{name: list(values) for name, values in formula_search_space.items()},
        "n_quantiles": [int(value) for value in mining["n_quantiles"]],
    }
    max_grid_candidates = int(mining.get("max_grid_candidates", 256))
    if str(mining.get("sampler", "grid")).lower() == "grid":
        count = grid_candidate_count(grid)
        if count > max_grid_candidates:
            raise ValueError(
                f"{get_definition(alpha_id).name} grid has {count} candidates, "
                f"exceeding max_grid_candidates={max_grid_candidates}"
            )
    paper_params = {
        "alpha_id": alpha_id,
        **default_compute_kwargs(alpha_id),
        "n_quantiles": int(mining["n_quantiles"][0]),
    }
    return RollingMiningConfig(
        factor_module="alpha101_mining",
        output_dir=output_dir,
        cache_dir=cache_dir,
        grid=grid,
        train=train,
        test=test,
        valid=_date_span(mining["valid"], "valid"),
        train_schemes=train_schemes,
        test_schemes=test_schemes,
        spec_factory="make_mining_spec",
        gid_factory="mining_gid",
        valid_report_hook="mining_valid_report",
        warmup_days_factory=mining_warmup_days,
        objective=str(mining["objective"]),
        objective_higher_is_better=bool(mining["objective_higher_is_better"]),
        report_top_n=int(mining["report_top_n"]),
        engine=str(mining["engine"]),
        workers=int(mining["workers"]),
        rebuild_cache=rebuild_cache,
        rebal_freq=str(mining["rebal_freq"]),
        initial_amount=float(mining["initial_amount"]),
        max_memory_ratio=float(mining["max_memory_ratio"]),
        max_warmup_days=int(mining.get("max_warmup_days", 5000)),
        max_windows_per_scheme=(
            None if mining.get("max_windows_per_scheme") is None else int(mining["max_windows_per_scheme"])
        ),
        rolling_mode=rolling_mode,
        paired_schemes=paired_schemes,
        rolling_span=rolling_span,
        sampler=str(mining.get("sampler", "grid")),
        paper_params=paper_params,
        n_trials=int(mining.get("n_trials", 96)),
        max_grid_candidates=max_grid_candidates,
        trial_batch_size=(
            None if mining.get("trial_batch_size") is None else int(mining["trial_batch_size"])
        ),
        random_seed=int(mining.get("random_seed", 20260810)),
        ic_coverage_min=float(mining.get("constraints", {}).get("ic_coverage_min", 0.80)),
        max_drawdown_max=float(mining.get("constraints", {}).get("max_drawdown_max", 0.35)),
        turnover_max=float(mining.get("constraints", {}).get("turnover_max", 1.00)),
        config_hash=config_hash,
        log_level=str(mining.get("log_level", "INFO")),
        storage_url=mining.get("storage_url"),
        config_path=str(config_path.resolve()),
    )


def _factor_search_space(alpha_id: int) -> dict[str, list[int | float]]:
    definition = get_definition(alpha_id)
    config_path = _CLASS_DIR / definition.name / f"factor_{definition.name}.yaml"
    config = load_yaml_config(config_path, required_sections=("factor_spec", "mining"))
    mining_section = section(config, "mining", context=str(config_path))
    if "search_space" not in mining_section:
        raise KeyError(f"{config_path} missing mining.search_space")
    return validate_search_space(alpha_id, mining_section["search_space"])


def run_from_config(config_path: str | Path = _CONFIG_FILE) -> pd.DataFrame:
    config_path = Path(config_path).resolve()
    cfg = load_yaml_config(config_path, required_sections=("mining",))
    mining = section(cfg, "mining", context=str(config_path))
    output_root = resolve_path(mining["output_dir"], config_path.parent)
    cache_dir = resolve_path(mining["cache_dir"], config_path.parent)
    output_root.mkdir(parents=True, exist_ok=True)
    alpha_ids = _alpha_ids(mining["factors"])
    resume = bool(mining.get("resume", True))
    fail_fast = bool(mining.get("fail_fast", False))
    statuses = []

    catalog = [row for alpha_id in alpha_ids for row in catalog_rows(alpha_id)]
    pd.DataFrame(catalog).to_csv(output_root / "parameter_catalog.csv", index=False, encoding="utf-8-sig")

    for position, alpha_id in enumerate(alpha_ids):
        name = get_definition(alpha_id).name
        factor_dir = output_root / name
        factor_dir.mkdir(parents=True, exist_ok=True)
        search_space = _factor_search_space(alpha_id)
        if str(mining.get("sampler", "grid")).lower() == "grid":
            candidates = formula_param_candidates(
                alpha_id,
                search_space,
                max_grid_candidates=int(mining.get("max_grid_candidates", 256)),
            )
        else:
            candidates = [default_compute_kwargs(alpha_id)]
        run_hash = _config_hash(mining, alpha_id, search_space)
        status_path = factor_dir / "status.json"
        if resume and status_path.exists():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if previous.get("status") == "completed" and previous.get("config_hash") == run_hash:
                statuses.append({"alpha": name, "status": "skipped", "candidates": len(candidates), "error": ""})
                continue

        try:
            result = run_walk_forward(
                _factor_config(
                    mining,
                    alpha_id,
                    factor_dir,
                    cache_dir,
                    rebuild_cache=bool(mining.get("rebuild_cache", False)) and position == 0,
                    formula_search_space=search_space,
                    config_hash=run_hash,
                    config_path=config_path,
                )
            )
            status_path.write_text(
                json.dumps(
                    {"status": "completed", "config_hash": run_hash, "candidates": len(candidates)},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            statuses.append({"alpha": name, "status": "completed", "candidates": len(candidates), "error": ""})
            del result
        except Exception as exc:  # Keep a 101-factor batch progressing after isolated failures.
            error = f"{type(exc).__name__}: {exc}"
            status_path.write_text(
                json.dumps(
                    {"status": "failed", "config_hash": run_hash, "candidates": len(candidates), "error": error},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            statuses.append({"alpha": name, "status": "failed", "candidates": len(candidates), "error": error})
            if fail_fast:
                break

        pd.DataFrame(statuses).to_csv(output_root / "batch_status.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(statuses)
    summary.to_csv(output_root / "batch_status.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="YAML parameter file")
    args = parser.parse_args()
    summary = run_from_config(args.config)
    return 1 if (not summary.empty and (summary["status"] == "failed").any()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
