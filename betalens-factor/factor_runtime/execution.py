"""Config-driven execution; the core package remains independent of scripts."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import uuid

from betalens.factor.config import ConfigError, load_yaml_config, run_parameters, write_yaml_config

FACTOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FACTOR_ROOT.parent


def export_settings(config):
    value = config.get("experiment_export", {})
    if not isinstance(value, dict):
        raise ConfigError("experiment_export 必须是字典")
    unknown = value.keys() - {"enabled", "store_path", "capture_selected"}
    if unknown:
        raise ConfigError(f"未知 experiment_export 配置: {sorted(unknown)}")
    enabled = value.get("enabled", False)
    capture = value.get("capture_selected", True)
    if not isinstance(enabled, bool) or not isinstance(capture, bool):
        raise ConfigError("experiment_export.enabled/capture_selected 必须为布尔值")
    path = value.get("store_path", "outputs/experiments.lmdb")
    if not isinstance(path, str) or not path.strip():
        raise ConfigError("experiment_export.store_path 必须是非空路径")
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return {"enabled": enabled, "store_path": str(path.resolve()), "capture_selected": capture}


def _preflight_export(settings):
    if not settings["enabled"]:
        return
    for dependency in ("lmdb", "pyarrow"):
        if importlib.util.find_spec(dependency) is None:
            raise ConfigError(f"实验入库缺少 {dependency}，请安装 betalens[experiments]")
    path = Path(settings["store_path"])
    parent = path
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise ConfigError(f"实验库路径不可写: {path}")


def validate_factor_config(config):
    """Reject unusable portfolio choices before querying any market data."""
    if config["meta"].get("strategy_type", "cross_sectional") == "timing":
        return
    weight = config["weight"]
    if weight.get("mode") != "freeplay":
        return
    def selections(value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return []
        return list(value) if isinstance(value, (tuple, list)) else [value]
    groups = (selections(weight.get("long_groups")), selections(weight.get("short_groups")))
    if not any(groups):
        raise ConfigError("freeplay 模式必须至少设置 long_groups 或 short_groups（取数前校验）")
    mode = str(config["run"].get("grouping_mode", "equal_count")).replace("-", "_")
    if mode in {"value", "strict_value", "unequal"}:
        for values in groups:
            if any(not isinstance(v, str) or v.lower() not in {"max", "min"} for v in values):
                raise ConfigError("value 分组的 freeplay 模式只能使用 max/min 分组选择器")


def _strategy(value):
    return "timing" if value == "timing" else "cross_sectional"


def _identity(meta):
    category, name = str(meta["class"]), str(meta["name"])
    # These fields also become directory components.
    for value in (category, name):
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ConfigError("因子 class/name 必须是有效的单个目录名")
    strategy = _strategy(meta.get("strategy_type"))
    return category, name, strategy, f"{category}:{name}:{strategy}"


def _register(store, meta):
    category, name, strategy, key = _identity(meta)
    existing = next((factor for factor in store.factors() if factor["key"] == key), None)
    return existing["id"] if existing else store.register_factor(
        key=key, name=name, category=category, strategy_type=strategy)


def _provenance(config_path):
    checksum = hashlib.sha256()
    for path in sorted(FACTOR_ROOT.rglob("*.py")):
        if any(part in {"outputs", "__pycache__", ".venv"} for part in path.parts):
            continue
        checksum.update(path.relative_to(FACTOR_ROOT).as_posix().encode())
        checksum.update(path.read_bytes())
    return {"source_config": str(Path(config_path).resolve()), "source_hash": checksum.hexdigest()}


def _receipt(output_dir, receipt):
    path = Path(output_dir) / "experiment_export.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        print(f"[experiment] 回执写入失败: {exc}")
    print(f"[experiment] {receipt['status']}" + (f": {receipt['error']}" if receipt.get("error") else ""))


def export_run(result, config, config_path, run_id, output_dir):
    settings = export_settings(config)
    receipt = {"status": "disabled", "experiment_id": None, "batch_id": None, "error": None}
    if settings["enabled"]:
        try:
            from betalens.experiments import ExperimentStore
            with ExperimentStore(settings["store_path"]) as store:
                factor_id = _register(store, config["meta"])
                identity = store.save_run(result, factor_id=factor_id, config=config,
                    source_id=run_id, metadata={"run_id": run_id, **_provenance(config_path)})
            receipt.update(status="saved", experiment_id=identity)
        except Exception as exc:
            receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    result.experiment_export = receipt
    _receipt(output_dir, receipt)
    return receipt


def execute_factor(config, config_path, build_spec, pipeline_cls, *, run_id=None, output_dir=None):
    config = copy.deepcopy(config)
    config_path = Path(config_path).resolve()
    validate_factor_config(config)
    settings = export_settings(config)
    _preflight_export(settings)
    category, name, _, _ = _identity(config["meta"])
    run_id = run_id or uuid.uuid4().hex
    params = run_parameters(config, config_path)
    if output_dir is None:
        configured = config["run"]["output_dir"]
        resolved = Path(params["output_dir"])
        default = REPO_ROOT / "outputs" / "runs" / category / name
        output_dir = default / run_id if configured == "outputs/runs/manual" or resolved == default else resolved
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config["run"]["output_dir"] = str(output_dir)
    config["experiment_export"] = {"enabled": settings["enabled"], "store_path": settings["store_path"]}
    effective_path = write_yaml_config(output_dir / "run_config.yaml", config)
    params = run_parameters(config, effective_path)
    start, end = params.pop("start_date"), params.pop("end_date")
    signature = inspect.signature(pipeline_cls.run)
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        params = {key: value for key, value in params.items() if key in signature.parameters}
    result = pipeline_cls(build_spec(config, effective_path)).run(start, end, **params)
    export_run(result, config, config_path, run_id, output_dir)
    return result


def run_factor_from_config(config_path, build_spec, pipeline_cls):
    config = load_yaml_config(config_path, required_sections=("meta", "factor_spec", "weight", "run"))
    return execute_factor(config, config_path, build_spec, pipeline_cls)


def _mining_meta(run, parameters_path):
    """Use catalog identity rather than workflow labels such as factor_class=DISP."""
    context = getattr(run, "archive_context", {})
    strategy = _strategy(context.get("strategy_type"))
    names = {str(run.factor_id), str(context.get("name", ""))}
    candidates = []
    # Prefer the originating class, then look across the existing catalog.
    for path in sorted(FACTOR_ROOT.glob("*/*/factor_*.yaml")):
        meta = load_yaml_config(path).get("meta", {})
        if meta.get("name") in names and _strategy(meta.get("strategy_type")) == strategy:
            candidates.append((path, meta))
    local = [meta for path, meta in candidates if parameters_path.is_relative_to(path.parent.parent)]
    matches = local or [meta for _, meta in candidates]
    if len(matches) > 1:
        raise ConfigError(f"挖掘因子身份不唯一: {run.factor_id}")
    if matches:
        return matches[0]
    return {"class": context.get("factor_class") or parameters_path.parent.parent.name,
            "name": run.factor_id, "strategy_type": strategy}


def run_mining_from_config(parameters_path, performance_path):
    """Archive returned mining objects; disabled exports never capture winners."""
    from betalens.factor.mining import run_mining
    parameters_path, performance_path = Path(parameters_path).resolve(), Path(performance_path).resolve()
    performance = load_yaml_config(performance_path)
    settings = export_settings(performance)
    _preflight_export(settings)
    # Validate missing adapter dependencies before a potentially expensive search.
    from betalens.factor.mining import _resolve_alpha_configs
    parameters = load_yaml_config(parameters_path)
    for directory in (parameters_path.parent.parent, parameters_path.parent.parent.parent):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    factors = parameters.get("factors", {})
    if parameters.get("factor_class") == "alpha101":
        factors = _resolve_alpha_configs(parameters, factors)
    if isinstance(factors, dict):
        import importlib
        for value in factors.values():
            try:
                importlib.import_module(value["module"])
            except ImportError as exc:
                raise ConfigError(f"挖掘适配器 {value['module']} 缺少依赖: {exc}") from exc
    result = run_mining(parameters_path, performance_path,
                        capture_selected=settings["enabled"] and settings["capture_selected"])
    receipt = {"status": "disabled", "experiment_id": None, "batch_id": None, "error": None}
    if settings["enabled"]:
        try:
            from betalens.experiments import ExperimentStore
            with ExperimentStore(settings["store_path"]) as store:
                for run in result.factor_runs:
                    run.archive_context = {**getattr(run, "archive_context", {}),
                                           "provenance": _provenance(parameters_path)}
                factor_ids = {run.factor_id: _register(store, _mining_meta(run, parameters_path))
                              for run in result.factor_runs}
                batch = store.save_mining(result, factor_ids=factor_ids)
            receipt.update(status=batch["status"], batch_id=batch["id"], failures=batch["failures"])
        except Exception as exc:
            receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    for run in result.factor_runs:
        directory = Path(run.run_dir)
        write_yaml_config(directory / "source_parameter_space.yaml", parameters)
        write_yaml_config(directory / "source_performance.yaml", performance)
        _receipt(directory, receipt)
    result.experiment_export = receipt
    return result
