from __future__ import annotations

import importlib.util
import ast
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from betalens.factor.config import load_yaml_config, section

from .schemas import FactorDetail, FactorSummary


REPO_ROOT = Path(__file__).resolve().parents[2]
FACTOR_ROOT = REPO_ROOT / "betalens-factor"
_FACTOR_REQUIRED_SECTIONS = ("meta", "factor_spec", "weight", "run")


def _iter_factor_specs(class_dir: Path) -> list[dict[str, Any]]:
    """扫描类目录下的因子子文件夹，读取各自 factor_{NAME}.yaml。"""
    factors: list[dict[str, Any]] = []
    for factor_dir in sorted(class_dir.iterdir()):
        if not factor_dir.is_dir() or factor_dir.name.startswith((".", "__")):
            continue
        factor_spec_path = factor_dir / f"factor_{factor_dir.name}.yaml"
        if not factor_spec_path.exists():
            continue
        try:
            factor_cfg = load_yaml_config(factor_spec_path, required_sections=_FACTOR_REQUIRED_SECTIONS)
        except Exception:
            continue
        factors.append(factor_cfg)
    return factors


def _iter_specs() -> list[tuple[str, Path, dict[str, Any]]]:
    if not FACTOR_ROOT.exists():
        return []
    specs: list[tuple[str, Path, dict[str, Any]]] = []
    for class_dir in sorted(FACTOR_ROOT.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith((".", "__")):
            continue
        spec_path = class_dir / f"class_{class_dir.name}.yaml"
        if not spec_path.exists():
            continue
        try:
            spec_data = load_yaml_config(spec_path)
        except Exception:
            continue
        spec_data["factors"] = _iter_factor_specs(class_dir)
        specs.append((class_dir.name, class_dir, spec_data))
    return specs


def _factor_script(class_dir: Path, factor_name: str) -> Path:
    return class_dir / factor_name / f"factor_{factor_name}.py"


def effective_factor_defaults(spec_data: dict[str, Any], factor_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a flat runtime parameter view for the Dashboard form."""
    del spec_data
    factor_spec = section(factor_cfg, "factor_spec")
    weight = section(factor_cfg, "weight")
    run = section(factor_cfg, "run")
    defaults = dict(run)
    defaults.update(
        {
            "direction": factor_spec.get("direction"),
            "index_code": factor_spec.get("index_code"),
            "use_industry": factor_spec.get("use_industry"),
            "use_mktcap": factor_spec.get("use_mktcap"),
            "industry_scheme": factor_spec.get("industry_scheme"),
            "backtest_metric": factor_spec.get("backtest_metric"),
            "weight_mode": weight.get("mode"),
            "long_groups": weight.get("long_groups"),
            "short_groups": weight.get("short_groups"),
            "group_weights": weight.get("group_weights", {}),
            "intra_group_allocation": weight.get("intra_group_allocation", {}),
        }
    )
    return defaults


@lru_cache(maxsize=1)
def discover_factors() -> tuple[FactorSummary, ...]:
    found: list[FactorSummary] = []
    for cls, _class_dir, spec_data in _iter_specs():
        source = spec_data.get("source", "")
        for factor in spec_data.get("factors", []):
            meta = section(factor, "meta")
            factor_spec = section(factor, "factor_spec")
            found.append(
                FactorSummary(
                    factor_class=cls,
                    name=meta.get("name", ""),
                    formula=meta.get("formula", ""),
                    logic=meta.get("logic", ""),
                    source=meta.get("source") or source,
                    inputs=factor_spec.get("inputs", {}),
                    defaults=effective_factor_defaults(spec_data, factor),
                )
            )
    return tuple(found)


def get_factor_config(factor_class: str, name: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    class_dir = FACTOR_ROOT / factor_class
    spec_path = class_dir / f"class_{factor_class}.yaml"
    if not spec_path.exists():
        raise FileNotFoundError(f"Factor class spec not found: {spec_path}")
    spec_data = load_yaml_config(spec_path)
    factor_spec_path = class_dir / name / f"factor_{name}.yaml"
    if not factor_spec_path.exists():
        raise FileNotFoundError(f"Factor spec not found: {factor_spec_path}")
    factor_cfg = load_yaml_config(factor_spec_path, required_sections=_FACTOR_REQUIRED_SECTIONS)
    script = _factor_script(class_dir, name)
    if not script.exists():
        raise FileNotFoundError(f"Factor script not found: {script}")
    return script, spec_data, factor_cfg


def get_factor_detail(factor_class: str, name: str) -> FactorDetail:
    script, spec_data, factor_cfg = get_factor_config(factor_class, name)
    meta = section(factor_cfg, "meta")
    factor_spec = section(factor_cfg, "factor_spec")
    doc = ""
    try:
        module = ast.parse(script.read_text(encoding="utf-8"))
        doc = ast.get_docstring(module) or ""
    except Exception as exc:
        doc = f"因子脚本可解析失败: {exc}"
    return FactorDetail(
        factor_class=factor_class,
        name=meta.get("name", name),
        formula=meta.get("formula", ""),
        logic=meta.get("logic", ""),
        source=meta.get("source") or spec_data.get("source", ""),
        inputs=factor_spec.get("inputs", {}),
        defaults=effective_factor_defaults(spec_data, factor_cfg),
        compute_kwargs=factor_spec.get("compute_kwargs", {}) or {},
        doc=doc,
        script_path=str(script),
        factor_dir=str(script.parent),
    )


def load_factor_module(script: Path):
    class_dir = script.parent.parent
    factor_root = class_dir.parent
    for path in (REPO_ROOT, factor_root, class_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    module_name = f"dashboard_factor_{class_dir.name}_{script.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load factor module from {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clear_factor_cache() -> None:
    discover_factors.cache_clear()
