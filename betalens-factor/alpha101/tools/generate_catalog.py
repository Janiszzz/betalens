"""Generate and verify the Alpha101 factor scripts and complete YAML specs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


CLASS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CLASS_DIR.parents[1]
for path in (REPO_ROOT, CLASS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha101_formulas import (  # noqa: E402
    ALPHA_DEFINITIONS,
    default_compute_kwargs,
    required_history_bars_for_alpha,
)
from alpha101_parameters import default_search_space  # noqa: E402


SOURCE = "WorldQuant 101 Formulaic Alphas (Kakushadze 2016)"


def _signature(definition, *, timing: bool) -> str:
    names = [*definition.inputs, *definition.industry_inputs]
    parameters = [f"{name}={spec.default!r}" for name, spec in definition.parameters.items()]
    timing_parameters = ["stock_code=None", "signal_weight=None"] if timing else []
    keyword_only = [*parameters, *timing_parameters]
    lines = [*(f"    {name}," for name in names)]
    if keyword_only:
        lines.append("    *,")
        lines.extend(f"    {parameter}," for parameter in keyword_only)
    return "\n" + "\n".join(lines) + "\n"


def _forward_args(definition) -> str:
    names = [*definition.inputs, *definition.industry_inputs]
    parameters = list(definition.parameters)
    return "\n".join(
        f"        {name}={name},"
        for name in [*names, *parameters]
    )


def render_script(definition, *, timing: bool) -> str:
    stem = f"factor_{definition.name}{'_timing' if timing else ''}"
    compute_name = f"compute_alpha{definition.number}{'_timing' if timing else ''}"
    pipeline_import = "TimingFactorPipeline as FactorPipeline" if timing else "FactorPipeline"
    strategy = "timing" if timing else "cross_section"
    timing_ignore = "    del stock_code, signal_weight\n" if timing else ""
    return f'''"""{definition.name} {'single-stock timing strategy' if timing else 'cross-sectional factor'}."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_FACTOR_DIR = Path(__file__).resolve().parent
_CLASS_DIR = _FACTOR_DIR.parent
_FACTOR_ROOT = _CLASS_DIR.parent
_REPO_ROOT = _FACTOR_ROOT.parent
for _path in (_REPO_ROOT, _FACTOR_ROOT, _CLASS_DIR, _FACTOR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section  # noqa: E402
from alpha101_formulas import compute_alpha, get_definition, required_history_bars_for_alpha  # noqa: E402
from factor_template_alpha101 import FactorSpec, {pipeline_import}  # noqa: E402


_CONFIG_FILE = _FACTOR_DIR / "{stem}.yaml"
_REQUIRED_SECTIONS = ("meta", "factor_spec", "weight", "run")


def load_config(path: str | Path = _CONFIG_FILE) -> dict:
    return load_yaml_config(path, required_sections=_REQUIRED_SECTIONS)


def {compute_name}({_signature(definition, timing=timing)}):
{timing_ignore}    return compute_alpha(
        {definition.number},
{_forward_args(definition)}
    )


def build_spec(config: dict, config_path: str | Path = _CONFIG_FILE) -> FactorSpec:
    options = factor_spec_options(config, config_path)
    parameter_names = set(get_definition({definition.number}).parameters)
    formula_kwargs = {{
        name: value
        for name, value in options.get("compute_kwargs", {{}}).items()
        if name in parameter_names
    }}
    options["required_history_bars"] = required_history_bars_for_alpha({definition.number}, formula_kwargs)
    return FactorSpec(
        name=str(section(config, "meta")["name"]),
        compute={compute_name},
        strategy_type="{strategy}",
        **options,
    )


spec = build_spec(load_config())


def run_from_config(config_path: str | Path = _CONFIG_FILE):
    config = load_config(config_path)
    kwargs = run_parameters(config, config_path)
    start_date = kwargs.pop("start_date")
    end_date = kwargs.pop("end_date")
    Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
    return FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run {definition.name}{' timing' if timing else ''} from YAML.")
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="YAML parameter file")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
'''


def factor_config(definition, *, timing: bool) -> dict:
    name = f"{definition.name}{'_timing' if timing else ''}"
    meta = {
        "class": "alpha101",
        "name": name,
        "source": SOURCE,
        "formula": definition.formula if not timing else f"{definition.name} rolling-z timing",
        "logic": (
            "严格按论文公式计算，高值作为正向截面因子。"
            if not timing
            else "以中证800日频PIT截面作为公式上下文并始终纳入目标股票；高于过去120个有效观测的均值+1σ时满仓做空，否则持币。"
        ),
    }
    if timing:
        meta["strategy_type"] = "timing"
    factor_spec = {
        "inputs": dict(definition.inputs),
        "industry_inputs": dict(definition.industry_inputs),
        "compute_kwargs": (
            {
                **default_compute_kwargs(definition.number),
                "stock_code": "300750.SZ",
                "signal_weight": {
                    "method": "rolling_z",
                    "window": 120,
                    "sigma": 1.0,
                    "operator": "gt",
                    "side": "short",
                    "max_weight": 1.0,
                },
            }
            if timing
            else default_compute_kwargs(definition.number)
        ),
        "direction": "positive",
        "table_name": "daily_market",
        "index_code": "000906.SH",
        "use_industry": False if timing else True,
        "use_mktcap": False,
        "industry_scheme": "申万一级行业",
        "backtest_metric": "收盘价(元)" if timing else "开盘价(元)",
        "required_history_bars": int(definition.required_history_bars),
        "mask_inputs_by_pit": True,
    }
    weight = {
        "mode": "freeplay" if timing else "classic-long-short",
        "long_groups": None,
        "short_groups": None,
        "group_weights": {},
        "intra_group_allocation": {},
    }
    run = {
        "start_date": "2025-01-01" if timing else "2024-01-01",
        "end_date": "2025-12-31",
        "rebal_freq": "D" if timing else "W",
        "n_quantiles": 10,
        "initial_amount": 100000000,
        "benchmark_code": "000906.SH",
        "include_profiling": False if timing else True,
        "dump_excel": True,
        "warmup_days": None,
        "output_dir": "outputs/runs/manual",
    }
    mining = {
        "sampler": "grid",
        "search_space": default_search_space(definition.number, max_dimensions=3),
    }
    return {"meta": meta, "factor_spec": factor_spec, "weight": weight, "run": run, "mining": mining}


def render_yaml(definition, *, timing: bool) -> str:
    return yaml.safe_dump(
        factor_config(definition, timing=timing),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )


def expected_files() -> dict[Path, str]:
    files = {}
    for definition in ALPHA_DEFINITIONS.values():
        factor_dir = CLASS_DIR / definition.name
        for timing in (False, True):
            suffix = "_timing" if timing else ""
            stem = f"factor_{definition.name}{suffix}"
            files[factor_dir / f"{stem}.py"] = render_script(definition, timing=timing)
            files[factor_dir / f"{stem}.yaml"] = render_yaml(definition, timing=timing)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify generated files without writing")
    args = parser.parse_args()
    mismatches = []
    for path, content in expected_files().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                mismatches.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    if mismatches:
        for path in mismatches:
            print(path.relative_to(CLASS_DIR))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
