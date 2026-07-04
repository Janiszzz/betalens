#%%
"""DISP parameter sweep entrypoint."""
from __future__ import annotations

import argparse
import io
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
from betalens.factor.mining import ParameterSweepConfig, run_parameter_sweep  # noqa: E402
from factor_DISP import mining_warmup_days  # noqa: E402


_CONFIG_FILE = _SCRIPT_DIR / "sweep_window.yaml"


def log(message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def _factory(name: str | None):
    if name in (None, ""):
        return None
    if name == "mining_warmup_days":
        return mining_warmup_days
    raise ValueError(f"unsupported warmup_days_factory: {name}")


def build_config(config_path: str | Path = _CONFIG_FILE) -> ParameterSweepConfig:
    config_path = Path(config_path).resolve()
    cfg = load_yaml_config(config_path, required_sections=("mining",))
    mining = section(cfg, "mining", context=str(config_path))
    base_dir = config_path.parent
    return ParameterSweepConfig(
        factor_module=str(mining["factor_module"]),
        spec_factory=str(mining["spec_factory"]),
        gid_factory=mining.get("gid_factory"),
        weight_hook=mining.get("weight_hook"),
        warmup_days_factory=_factory(mining.get("warmup_days_factory")),
        output_dir=resolve_path(mining["output_dir"], base_dir),
        cache_dir=resolve_path(mining["cache_dir"], base_dir),
        span=tuple(mining["span"]),
        grid=dict(mining["grid"]),
        objective=str(mining["objective"]),
        objective_higher_is_better=bool(mining["objective_higher_is_better"]),
        engine=str(mining["engine"]),
        workers=int(mining["workers"]),
        rebuild_cache=bool(mining["rebuild_cache"]),
        max_memory_ratio=float(mining["max_memory_ratio"]),
        max_warmup_days=int(mining["max_warmup_days"]),
        results_filename=str(mining["results_filename"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DISP sweep from YAML.")
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="YAML parameter file")
    args = parser.parse_args()
    config = build_config(args.config)
    windows = list(config.grid["window"])
    log("Starting DISP window sweep")
    log(f"span={config.span[0]}~{config.span[1]}  engine={config.engine}  rebuild_cache={config.rebuild_cache}")
    log(f"window grid: count={len(windows)}  min={min(windows)}  max={max(windows)}")
    log(f"output_dir={config.output_dir}")
    log(f"cache_dir={config.cache_dir}")
    df = run_parameter_sweep(config)
    if df.empty:
        log("No valid result")
        return
    print("\n=== Sorted By Objective ===")
    print(df.to_string(index=False))
    best = df.iloc[0]
    print(f"\nBest gid = {best.get('gid')}  {config.objective}={best.get(config.objective)}")
    print(f"Saved: {Path(config.output_dir) / config.results_filename}")
    log("Finished DISP window sweep")


if __name__ == "__main__":
    main()
