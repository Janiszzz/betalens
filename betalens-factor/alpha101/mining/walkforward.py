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
from alpha101_formulas import get_definition  # noqa: E402
from alpha101_mining import mining_warmup_days  # noqa: E402
from alpha101_parameters import catalog_rows, formula_param_candidates  # noqa: E402


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


def _config_hash(mining: Mapping[str, Any], alpha_id: int, candidates: list[dict[str, Any]]) -> str:
    payload = {"mining": dict(mining), "alpha_id": alpha_id, "candidates": candidates}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _factor_config(
    mining: Mapping[str, Any],
    alpha_id: int,
    output_dir: Path,
    cache_dir: Path,
    rebuild_cache: bool,
) -> RollingMiningConfig:
    candidates = formula_param_candidates(alpha_id, max_candidates=int(mining.get("max_candidates", 256)))
    rolling_mode = str(mining.get("rolling_mode", "split")).lower()
    if rolling_mode == "paired":
        paired_schemes = _paired_schemes(mining["paired_schemes"], "paired_schemes")
        train_schemes = []
        test_schemes = []
    elif rolling_mode == "split":
        paired_schemes = None
        train_schemes = _schemes(mining["train_schemes"], "train_schemes")
        test_schemes = _schemes(mining["test_schemes"], "test_schemes")
    else:
        raise ValueError(f"unknown rolling_mode: {rolling_mode}")
    grid = {
        "alpha_id": [alpha_id],
        "formula_params": candidates,
        "n_quantiles": [int(value) for value in mining["n_quantiles"]],
    }
    return RollingMiningConfig(
        factor_module="alpha101_mining",
        output_dir=output_dir,
        cache_dir=cache_dir,
        grid=grid,
        train=tuple(str(value) for value in mining["train"]),
        test=tuple(str(value) for value in mining["test"]),
        valid=tuple(str(value) for value in mining["valid"]),
        train_schemes=train_schemes,
        test_schemes=test_schemes,
        spec_factory="make_mining_spec",
        gid_factory="mining_gid",
        valid_report_hook="mining_valid_report",
        warmup_days_factory=mining_warmup_days,
        objective=str(mining["objective"]),
        objective_higher_is_better=bool(mining["objective_higher_is_better"]),
        candidate_percentile=tuple(float(value) for value in mining["candidate_percentile"]),
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
    )


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
        candidates = formula_param_candidates(alpha_id, max_candidates=int(mining.get("max_candidates", 256)))
        run_hash = _config_hash(mining, alpha_id, candidates)
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
