"""运行 XICHOU timing 单标的参数挖掘。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FACTOR_DIR = SCRIPT_DIR.parent
CLASS_DIR = FACTOR_DIR.parent
FACTOR_ROOT = CLASS_DIR.parent
REPO_ROOT = FACTOR_ROOT.parent
for path in (REPO_ROOT, FACTOR_ROOT, CLASS_DIR, FACTOR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from betalens.factor.mining import run_mining  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameters",
        default=str(SCRIPT_DIR / "parameter_space.yaml"),
        help="参数空间配置文件路径",
    )
    parser.add_argument(
        "--performance",
        default=str(SCRIPT_DIR / "performance.yaml"),
        help="性能与输出配置文件路径",
    )
    args = parser.parse_args()
    result = run_mining(args.parameters, args.performance)
    for factor_run in result.factor_runs:
        print(
            f"timing 挖掘已结束：因子={factor_run.factor_id}，"
            f"最终入选={len(factor_run.selected_candidates)}，"
            f"任务目录={factor_run.run_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
