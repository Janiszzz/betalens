#%%
"""DISP walk-forward parameter mining entrypoint."""
from __future__ import annotations

import argparse
import io
import itertools
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.getLogger("IndexUniverseQuery").setLevel(logging.WARNING)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
_FACTOR_DIR = _SCRIPT_DIR.parent
_CLASS_DIR = _FACTOR_DIR.parent
_FACTOR_ROOT = _CLASS_DIR.parent
_REPO_ROOT = _FACTOR_ROOT.parent
for _path in (_REPO_ROOT, _FACTOR_ROOT, _CLASS_DIR, _FACTOR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from betalens.factor.config import load_yaml_config, resolve_path, section  # noqa: E402
from betalens.factor.mining import RollingMiningConfig, run_walk_forward  # noqa: E402
from factor_DISP import mining_warmup_days  # noqa: E402


_CONFIG_FILE = _SCRIPT_DIR / "walkforward.yaml"


def log(message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def _factory(name: str | None):
    if name in (None, ""):
        return None
    if name == "mining_warmup_days":
        return mining_warmup_days
    raise ValueError(f"unsupported warmup_days_factory: {name}")


def _schemes(value, name: str):
    if isinstance(value, dict):
        lengths = value["window_lengths"]
        steps = value["steps"]
        raw = itertools.product(lengths, steps)
    else:
        raw = value

    schemes = []
    seen = set()
    for item in raw:
        win_len, step = (int(x) for x in item)
        if step > win_len:
            continue
        scheme = (win_len, step)
        if scheme not in seen:
            schemes.append(scheme)
            seen.add(scheme)
    if not schemes:
        raise ValueError(f"{name} produced no valid schemes; every step must be <= window_length")
    return schemes


def build_config(config_path: str | Path = _CONFIG_FILE) -> RollingMiningConfig:
    config_path = Path(config_path).resolve()
    cfg = load_yaml_config(config_path, required_sections=("mining",))
    mining = section(cfg, "mining", context=str(config_path))
    base_dir = config_path.parent
    return RollingMiningConfig(
        factor_module=str(mining["factor_module"]),
        spec_factory=str(mining["spec_factory"]),
        gid_factory=mining.get("gid_factory"),
        weight_hook=mining.get("weight_hook"),
        valid_report_hook=mining.get("valid_report_hook"),
        warmup_days_factory=_factory(mining.get("warmup_days_factory")),
        output_dir=resolve_path(mining["output_dir"], base_dir),
        cache_dir=resolve_path(mining["cache_dir"], base_dir),
        grid=dict(mining["grid"]),
        train=tuple(mining["train"]),
        test=tuple(mining["test"]),
        valid=tuple(mining["valid"]),
        train_schemes=_schemes(mining["train_schemes"], "train_schemes"),
        test_schemes=_schemes(mining["test_schemes"], "test_schemes"),
        objective=str(mining["objective"]),
        objective_higher_is_better=bool(mining["objective_higher_is_better"]),
        candidate_percentile=tuple(float(x) for x in mining["candidate_percentile"]),
        report_top_n=int(mining["report_top_n"]),
        engine=str(mining["engine"]),
        workers=int(mining["workers"]),
        rebuild_cache=bool(mining["rebuild_cache"]),
        max_memory_ratio=float(mining["max_memory_ratio"]),
        max_warmup_days=int(mining["max_warmup_days"]),
        max_windows_per_scheme=(
            None if mining.get("max_windows_per_scheme") is None else int(mining["max_windows_per_scheme"])
        ),
    )


def _grid_size(grid: dict) -> int:
    total = 1
    for values in grid.values():
        total *= len(values)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DISP walk-forward mining from YAML.")
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="YAML parameter file")
    args = parser.parse_args()
    config = build_config(args.config)
    log("Starting DISP walk-forward mining")
    log(f"engine={config.engine}  rebuild_cache={config.rebuild_cache}")
    log(f"train={config.train[0]}~{config.train[1]}  test={config.test[0]}~{config.test[1]}  valid={config.valid[0]}~{config.valid[1]}")
    log(f"grid combinations={_grid_size(config.grid)}")
    log(f"train_schemes={list(config.train_schemes)}  test_schemes={list(config.test_schemes)}  window_cap={config.max_windows_per_scheme}")
    log(f"workers_requested={config.workers}  max_memory_ratio={config.max_memory_ratio}")
    log(f"output_dir={config.output_dir}")
    log(f"cache_dir={config.cache_dir}")
    run_walk_forward(config)
    log("Finished DISP walk-forward mining")


if __name__ == "__main__":
    main()
