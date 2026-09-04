"""COURSE_MOM：完整的动量教学因子。市场回测需要真实数据库。

可在课程模板目录直接运行，也可按第 18 课复制至 betalens-factor/research。
import 仅加载配置和声明算子，不连接数据库、不启动回测、不写文件。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def _find_repo_root():
    for directory in Path(__file__).resolve().parents:
        if (directory / "pyproject.toml").is_file() and (directory / "betalens-factor").is_dir():
            return directory
    raise RuntimeError("请将教学脚本放在 Betalens 仓库内")


_REPO_ROOT = _find_repo_root()
for _path in (_REPO_ROOT, _REPO_ROOT / "betalens-factor"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters
from factor_template import FactorPipeline, FactorSpec

_CONFIG_FILE = Path(__file__).with_suffix(".yaml")


def load_config(path=_CONFIG_FILE):
    return load_yaml_config(path, required_sections=("meta", "factor_spec", "weight", "run"))


def compute_momentum(close_wide, *, window):
    if isinstance(window, bool) or int(window) != window or window < 1:
        raise ValueError("window 必须为正整数")
    values = close_wide / close_wide.shift(int(window)) - 1
    return values.replace([np.inf, -np.inf], np.nan)


def build_spec(config, config_path=_CONFIG_FILE):
    options = factor_spec_options(config, config_path)
    window = options["compute_kwargs"]["window"]
    if isinstance(window, bool) or int(window) != window or window < 1:
        raise ValueError("compute_kwargs.window 必须为正整数")
    options["required_history_bars"] = int(window) + 1
    return FactorSpec(name=str(config["meta"]["name"]), compute=compute_momentum, **options)


spec = build_spec(load_config())


def run_from_config(config_path=_CONFIG_FILE):
    config = load_config(config_path)
    kwargs = run_parameters(config, config_path)
    start_date = kwargs.pop("start_date")
    end_date = kwargs.pop("end_date")
    Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
    return FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="完整 YAML 配置路径")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
