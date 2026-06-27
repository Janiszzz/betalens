from __future__ import annotations

import importlib.util
import ast
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schemas import FactorDetail, FactorSummary


REPO_ROOT = Path(__file__).resolve().parents[2]
FACTOR_ROOT = REPO_ROOT / "betalens-factor"
_FACTOR_METADATA_KEYS = {
    "name",
    "formula",
    "logic",
    "inputs",
    "compute_kwargs",
    "compute_body",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_factor_specs(class_dir: Path) -> list[dict[str, Any]]:
    """扫描类目录下的因子子文件夹，读取各自 spec_{NAME}.json。"""
    factors: list[dict[str, Any]] = []
    for factor_dir in sorted(class_dir.iterdir()):
        if not factor_dir.is_dir() or factor_dir.name.startswith((".", "__")):
            continue
        factor_spec_path = factor_dir / f"spec_{factor_dir.name}.json"
        if not factor_spec_path.exists():
            continue
        try:
            factor_cfg = _read_json(factor_spec_path)
        except Exception:
            continue
        factor_cfg.setdefault("name", factor_dir.name)
        factors.append(factor_cfg)
    return factors


def _iter_specs() -> list[tuple[str, Path, dict[str, Any]]]:
    if not FACTOR_ROOT.exists():
        return []
    specs: list[tuple[str, Path, dict[str, Any]]] = []
    for class_dir in sorted(FACTOR_ROOT.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith((".", "__")):
            continue
        spec_path = class_dir / f"spec_{class_dir.name}.json"
        if not spec_path.exists():
            continue
        try:
            spec_data = _read_json(spec_path)
        except Exception:
            continue
        spec_data["factors"] = _iter_factor_specs(class_dir)
        specs.append((class_dir.name, class_dir, spec_data))
    return specs


def _factor_script(class_dir: Path, factor_name: str) -> Path:
    return class_dir / factor_name / f"factor_{factor_name}.py"


def effective_factor_defaults(spec_data: dict[str, Any], factor_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return class defaults overlaid with per-factor runtime overrides."""
    defaults = dict(spec_data.get("defaults", {}) or {})
    overrides = {
        key: value
        for key, value in factor_cfg.items()
        if key not in _FACTOR_METADATA_KEYS
    }
    defaults.update(overrides)
    return defaults


@lru_cache(maxsize=1)
def discover_factors() -> tuple[FactorSummary, ...]:
    found: list[FactorSummary] = []
    for cls, _class_dir, spec_data in _iter_specs():
        source = spec_data.get("source", "")
        for factor in spec_data.get("factors", []):
            found.append(
                FactorSummary(
                    factor_class=cls,
                    name=factor.get("name", ""),
                    formula=factor.get("formula", ""),
                    logic=factor.get("logic", ""),
                    source=source,
                    inputs=factor.get("inputs", {}),
                    defaults=effective_factor_defaults(spec_data, factor),
                )
            )
    return tuple(found)


def get_factor_config(factor_class: str, name: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    class_dir = FACTOR_ROOT / factor_class
    spec_path = class_dir / f"spec_{factor_class}.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"Factor class spec not found: {spec_path}")
    spec_data = _read_json(spec_path)
    factor_spec_path = class_dir / name / f"spec_{name}.json"
    if not factor_spec_path.exists():
        raise FileNotFoundError(f"Factor spec not found: {factor_spec_path}")
    factor_cfg = _read_json(factor_spec_path)
    factor_cfg.setdefault("name", name)
    script = _factor_script(class_dir, name)
    if not script.exists():
        raise FileNotFoundError(f"Factor script not found: {script}")
    return script, spec_data, factor_cfg


def get_factor_detail(factor_class: str, name: str) -> FactorDetail:
    script, spec_data, factor_cfg = get_factor_config(factor_class, name)
    doc = ""
    try:
        module = ast.parse(script.read_text(encoding="utf-8"))
        doc = ast.get_docstring(module) or ""
    except Exception as exc:
        doc = f"因子脚本可解析失败: {exc}"
    return FactorDetail(
        factor_class=factor_class,
        name=name,
        formula=factor_cfg.get("formula", ""),
        logic=factor_cfg.get("logic", ""),
        source=spec_data.get("source", ""),
        inputs=factor_cfg.get("inputs", {}),
        defaults=effective_factor_defaults(spec_data, factor_cfg),
        compute_kwargs=factor_cfg.get("compute_kwargs", {}),
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
