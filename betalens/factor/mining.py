"""Multi-stage, window-aware factor parameter mining."""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from betalens.factor.config import resolve_preprocessing_options
from betalens.factor.mining_audit import (
    FactorMiningResult,
    MiningResult,
    MiningTask,
)


_LOGGER_NAME = "betalens.factor.mining"
_LOGGER = logging.getLogger(_LOGGER_NAME)
_TASK_LOGS = False
_HEARTBEAT_SECONDS = 30.0


class _ChineseLogFormatter(logging.Formatter):
    """Render logging metadata in Chinese for terminal and audit output."""

    _LEVEL_NAMES = {
        logging.DEBUG: "调试",
        logging.INFO: "信息",
        logging.WARNING: "警告",
        logging.ERROR: "错误",
        logging.CRITICAL: "严重",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.level_cn = self._LEVEL_NAMES.get(record.levelno, record.levelname)
        process_name = str(record.processName)
        if process_name == "MainProcess":
            record.process_cn = "主进程"
        else:
            suffix = process_name.rsplit("-", 1)[-1]
            record.process_cn = f"工作进程-{suffix}" if suffix.isdigit() else f"工作进程 {process_name}"
        return super().format(record)


def _stage_name(stage: str) -> str:
    names = {
        "coarse": "宽范围粗搜",
        "refine": "自适应收敛搜索",
        "fine": "局部网格细搜",
        "stability": "赢家扰动验证",
    }
    value = str(stage).lower()
    if value.startswith("expansion_"):
        return f"第{value.rsplit('_', 1)[-1]}轮边界扩展搜索"
    return names.get(value, str(stage))


def _mode_name(mode: str) -> str:
    return {
        "precomputed": "预计算",
        "rolling_fit": "滚动拟合",
    }.get(str(mode).lower(), str(mode))


def _backend_name(backend: str) -> str:
    return {"process": "多进程", "serial": "单进程"}.get(str(backend).lower(), str(backend))


def _engine_name(engine: str) -> str:
    return {"vector": "向量回测", "exact": "精确回测"}.get(str(engine).lower(), str(engine))


def _metric_name(metric: str) -> str:
    return {
        "sharpe": "夏普比率",
        "ann_ret": "年化收益率",
        "mdd": "最大回撤",
        "calmar": "卡玛比率",
        "turnover": "换手率",
        "rank_ic": "Rank IC",
    }.get(str(metric).lower(), str(metric))


def _aggregate_name(aggregate: str) -> str:
    return {
        "mean": "均值",
        "median": "中位数",
        "min": "最小值",
        "max": "最大值",
        "p25": "25%分位数",
    }.get(str(aggregate).lower(), str(aggregate))


def _direction_name(direction: str) -> str:
    return {"maximize": "越大越好", "minimize": "越小越好"}.get(str(direction).lower(), str(direction))


def _yes_no(value: Any) -> str:
    return "是" if bool(value) else "否"


def _window_description(window_id: str) -> str:
    parts = str(window_id).split("/")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        length, step, index = map(int, parts)
        return f"{length}日窗口、每{step}日滑动、第{index + 1}个窗口"
    return str(window_id)


def _operation_name(operation: str) -> str:
    return {
        "data.query": "查询行情数据",
        "candidate.factor": "计算候选因子",
        "candidate.weights": "生成候选权重",
        "candidate.nav": "计算候选全程净值",
        "candidate.rank_ic": "预计算候选Rank IC序列",
        "candidate.fit_window": "拟合窗口参数",
        "candidate.window_nav": "计算窗口净值",
    }.get(operation, operation)


def _heartbeat_fields(fields: Mapping[str, Any]) -> str:
    labels = {
        "stage": "搜索阶段",
        "candidate": "候选",
        "window": "窗口",
        "engine": "回测引擎",
        "table": "数据表",
        "metric": "字段",
    }
    values = []
    for key, value in fields.items():
        if key == "stage":
            value = _stage_name(str(value))
        elif key == "window":
            value = _window_description(str(value))
        elif key == "engine":
            value = _engine_name(str(value))
        values.append(f"{labels.get(key, key)}={value}")
    return "，".join(values)


def _configure_mining_logging(
    log_path: Path,
    level: int,
    *,
    task_logs: bool,
    heartbeat_seconds: float,
) -> Path:
    """Attach one live console and one run-specific audit file handler."""
    global _HEARTBEAT_SECONDS, _TASK_LOGS
    _TASK_LOGS = bool(task_logs)
    _HEARTBEAT_SECONDS = max(0.0, float(heartbeat_seconds))
    for handler in list(_LOGGER.handlers):
        if getattr(handler, "_betalens_mining_handler", False):
            _LOGGER.removeHandler(handler)
            handler.close()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = _ChineseLogFormatter(
        "%(asctime)s %(level_cn)-2s [%(process_cn)s PID=%(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    audit = logging.FileHandler(log_path, encoding="utf-8")
    for handler in (console, audit):
        handler.setLevel(level)
        handler.setFormatter(formatter)
        handler._betalens_mining_handler = True
        _LOGGER.addHandler(handler)
    _LOGGER.setLevel(level)
    _LOGGER.propagate = False
    return log_path


def _close_mining_logging() -> None:
    global _HEARTBEAT_SECONDS, _TASK_LOGS
    for handler in list(_LOGGER.handlers):
        if getattr(handler, "_betalens_mining_handler", False):
            handler.flush()
            _LOGGER.removeHandler(handler)
            handler.close()
    _TASK_LOGS = False
    _HEARTBEAT_SECONDS = 30.0


def _configure_worker_logging(
    log_queue,
    level: int,
    task_logs: bool,
    heartbeat_seconds: float,
) -> None:
    global _HEARTBEAT_SECONDS, _TASK_LOGS
    _TASK_LOGS = bool(task_logs)
    _HEARTBEAT_SECONDS = max(0.0, float(heartbeat_seconds))
    for handler in list(_LOGGER.handlers):
        _LOGGER.removeHandler(handler)
    _LOGGER.addHandler(QueueHandler(log_queue))
    _LOGGER.setLevel(level)
    _LOGGER.propagate = False


def _elapsed(started: float) -> str:
    return _human_duration(time.perf_counter() - started)


def _human_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}秒"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}分{remainder:04.1f}秒"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}小时{minutes:02d}分"


def _human_bytes(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TiB"


def _progress_due(completed: int, total: int, *, updates: int = 20) -> bool:
    if completed <= 1 or completed >= total:
        return True
    interval = max(1, math.ceil(total / max(1, updates)))
    return completed % interval == 0


@contextmanager
def _heartbeat(operation: str, **fields):
    """Emit liveness records while a single blocking operation is running."""
    interval = _HEARTBEAT_SECONDS
    if interval <= 0:
        yield
        return
    stopped = threading.Event()
    started = time.perf_counter()
    suffix = _heartbeat_fields(fields)

    def report() -> None:
        while not stopped.wait(interval):
            _LOGGER.info(
                "任务仍在运行：环节=%s，已耗时=%s%s",
                _operation_name(operation),
                _elapsed(started),
                f"，{suffix}" if suffix else "",
            )

    thread = threading.Thread(target=report, name=f"mining-heartbeat-{operation}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=min(interval, 1.0))


@dataclass
class MiningSpec:
    factor_spec: Any
    execution_mode: Literal["precomputed", "rolling_fit"] = "precomputed"
    fit_window: Callable | None = None
    window_transform: Callable | None = None
    warmup_days: int | Callable = 30


@dataclass(frozen=True)
class MiningWindow:
    window_id: str
    start: str
    end: str
    length: int
    step: int


@dataclass
class MiningData:
    inputs: dict[str, pd.DataFrame]
    price: pd.DataFrame
    execution_price: pd.DataFrame
    trade_status: pd.DataFrame | None
    pit: dict[Any, set[str]] | None
    universe: list[str]
    industry_by_scheme: dict[str, pd.DataFrame] = field(default_factory=dict)
    cache_manifest_path: str | None = None


class CandidateEvaluation(list):
    """Window results plus the de-duplicated Rank IC observations for audit."""

    def __init__(
        self,
        window_rows: Sequence[Mapping[str, Any]] = (),
        rank_ic_series: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(window_rows)
        self.rank_ic_series = list(rank_ic_series)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    if "mining" in payload:
        raise ValueError(
            f"legacy `mining` section found in {path}; migrate it to parameter_space.yaml "
            "and performance.yaml"
        )
    return payload


def validate_parameter_specs(parameters: Mapping[str, Any]) -> None:
    for name, spec in parameters.items():
        if not isinstance(spec, Mapping):
            raise TypeError(f"parameter {name} must be a mapping")
        kind = str(spec.get("type", "float")).lower()
        allowed = {"type", "low", "high", "step", "scale", "choices"}
        unknown = sorted(set(spec) - allowed)
        if unknown:
            raise ValueError(f"parameter {name} has unknown fields: {unknown}")
        if kind in {"categorical", "choice", "bool", "boolean"}:
            choices = spec.get("choices", [False, True] if kind in {"bool", "boolean"} else None)
            if not choices:
                raise ValueError(f"parameter {name} requires non-empty choices")
            continue
        if kind not in {"int", "integer", "float"}:
            raise ValueError(f"unsupported parameter type for {name}: {kind}")
        if "low" not in spec or "high" not in spec:
            raise ValueError(f"numeric parameter {name} requires low and high")
        if float(spec["low"]) > float(spec["high"]):
            raise ValueError(f"parameter {name} low must not exceed high")
        scale = str(spec.get("scale", "linear")).lower()
        if scale not in {"linear", "log"}:
            raise ValueError(f"parameter {name} scale must be linear or log")
        if scale == "log" and float(spec["low"]) <= 0:
            raise ValueError(f"log parameter {name} requires low > 0")
        if kind in {"int", "integer"} and int(spec.get("step", 1)) < 1:
            raise ValueError(f"integer parameter {name} step must be positive")
        if kind == "float" and spec.get("step") is not None and float(spec["step"]) <= 0:
            raise ValueError(f"float parameter {name} step must be positive")
        if scale == "log" and kind == "float" and spec.get("step") is not None:
            raise ValueError(f"log float parameter {name} cannot use step")
        if scale == "log" and kind in {"int", "integer"} and spec.get("step", 1) != 1:
            raise ValueError(f"log integer parameter {name} requires step=1")


def _validate_window_config(config: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    lengths = [int(value) for value in config.get("lengths", [252])]
    steps = [int(value) for value in config.get("steps", [63])]
    if not lengths or not steps or any(value < 1 for value in lengths + steps):
        raise ValueError("windows.lengths and windows.steps must contain positive integers")
    invalid = [(length, step) for length in lengths for step in steps if step > length]
    if invalid:
        raise ValueError(f"window step must not exceed length: {invalid}")
    return lengths, steps


def _candidate_id(factor_id: str, params: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(params), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return f"{factor_id}_{hashlib.sha1(encoded.encode('utf-8')).hexdigest()[:12]}"


def _import_spec(
    module_name: str,
    params: Mapping[str, Any],
    preprocessing: Mapping[str, Any] | None = None,
) -> MiningSpec:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name or "." in module_name:
            raise
        factor_root = Path(__file__).resolve().parents[2] / "betalens-factor"
        matches = list(factor_root.rglob(f"{module_name}.py")) if factor_root.exists() else []
        if len(matches) != 1:
            raise ModuleNotFoundError(
                f"cannot uniquely resolve mining module {module_name!r}; matches={len(matches)}"
            ) from exc
        module_directory = str(matches[0].parent)
        if module_directory not in sys.path:
            sys.path.insert(0, module_directory)
        module = importlib.import_module(module_name)
    factory = getattr(module, "make_mining_spec", None)
    if factory is None:
        raise AttributeError(f"{module_name} must expose make_mining_spec(params)")
    value = factory(dict(params))
    if not isinstance(value, MiningSpec):
        raise TypeError(f"{module_name}.make_mining_spec(params) must return MiningSpec")
    factor = value.factor_spec
    legacy_enabled = bool(
        getattr(factor, "use_industry", False)
        or getattr(factor, "use_mktcap", False)
    )
    configured = {
        "winsorize_method": getattr(factor, "winsorize_method", None),
        "winsorize_n": getattr(factor, "winsorize_n", 3.0),
        "standardize_method": getattr(factor, "standardize_method", None),
        "neutralize": getattr(factor, "neutralize", None),
    }
    if preprocessing:
        resolve_preprocessing_options(
            preprocessing,
            legacy_enabled=legacy_enabled,
            context="parameter_space.preprocessing",
            strict=True,
        )
        configured.update(preprocessing)
    resolved = resolve_preprocessing_options(
        configured,
        legacy_enabled=legacy_enabled,
        context=f"{getattr(factor, 'name', module_name)} preprocessing",
    )
    for name, option in resolved.items():
        setattr(factor, name, option)
    return value


def _warmup_days(spec: MiningSpec, params: Mapping[str, Any]) -> int:
    value = spec.warmup_days
    if callable(value):
        try:
            value = value(params)
        except (KeyError, TypeError, ValueError):
            return 1200
    return max(0, int(value))


def _system_memory() -> tuple[int | None, int | None]:
    try:
        import psutil
        info = psutil.virtual_memory()
        return int(info.total), int(info.available)
    except Exception:
        return None, None


def _physical_cpu_count() -> int:
    try:
        import psutil
        return max(1, int(psutil.cpu_count(logical=False) or os.cpu_count() or 1))
    except Exception:
        return max(1, int(os.cpu_count() or 1))


def _frame_bytes(value: pd.DataFrame | None) -> int:
    if value is None:
        return 0
    return int(value.memory_usage(index=True, deep=True).sum())


def _effective_workers(requested: int, data: MiningData, ratio: float) -> int:
    requested = max(1, int(requested))
    cpu_budget = _physical_cpu_count()
    total, available = _system_memory()
    if not total or not available or ratio <= 0:
        return min(requested, cpu_budget)
    frames = [*data.inputs.values(), data.price, data.execution_price, *data.industry_by_scheme.values()]
    if data.trade_status is not None:
        frames.append(data.trade_status)
    per_worker = max(1, sum(_frame_bytes(value) for value in frames))
    budget = min(int(total * min(float(ratio), 1.0)), int(available * 0.85))
    return max(1, min(requested, cpu_budget, budget // per_worker))


@contextmanager
def _worker_thread_limits():
    """Keep spawned numeric workers from multiplying native thread pools."""
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    added = []
    for name in names:
        if name not in os.environ:
            os.environ[name] = "1"
            added.append(name)
    try:
        yield
    finally:
        for name in added:
            os.environ.pop(name, None)


def _fetch_daily_wide(metric: str, universe: Sequence[str], start: str, end: str, table: str) -> pd.DataFrame:
    from betalens.datafeed import Datafeed

    started = time.perf_counter()
    _LOGGER.info(
        "开始查询行情数据：数据表=%s，字段=%s，日期=%s 至 %s，证券数=%d",
        table,
        metric,
        start,
        end,
        len(universe),
    )
    data = Datafeed(table)
    try:
        with _heartbeat("data.query", table=table, metric=metric):
            result = data.query_time_range(codes=list(universe), start_date=start, end_date=end, metric=metric)
    finally:
        data.close()
    if result.empty:
        _LOGGER.warning(
            "行情数据查询结果为空：数据表=%s，字段=%s，耗时=%s",
            table,
            metric,
            _elapsed(started),
        )
        return pd.DataFrame()
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result["datetime"] = pd.to_datetime(result["datetime"])
    wide = result.pivot_table(index="datetime", columns="code", values="value").sort_index()
    _LOGGER.info(
        "行情数据查询完成：数据表=%s，字段=%s，原始记录=%d行，宽表=%d行 x %d列，耗时=%s",
        table,
        metric,
        len(result),
        wide.shape[0],
        wide.shape[1],
        _elapsed(started),
    )
    return wide


def _align_daily_wides(wides: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    nonempty = [value for value in wides.values() if value is not None and not value.empty]
    if not nonempty:
        return dict(wides)
    latest: dict[pd.Timestamp, pd.Timestamp] = {}
    for wide in nonempty:
        for value in pd.DatetimeIndex(wide.index):
            day = value.normalize()
            if day not in latest or value > latest[day]:
                latest[day] = value
    days = pd.DatetimeIndex(sorted(latest))
    canonical = pd.DatetimeIndex([latest[day] for day in days])
    output = {}
    for name, wide in wides.items():
        value = wide.copy()
        value.index = pd.DatetimeIndex(value.index).normalize()
        value = value.loc[~value.index.duplicated(keep="last")].reindex(days)
        value.index = canonical
        output[name] = value
    return output


def _fetch_industry_wide(scheme: str, universe: Sequence[str], dates: Sequence[Any], index: pd.DatetimeIndex) -> pd.DataFrame:
    from betalens.datafeed import Datafeed

    started = time.perf_counter()
    pieces = []
    data = Datafeed("industry")
    try:
        days = list(dict.fromkeys(pd.Timestamp(day).date() for day in dates))
        blocks = max(1, math.ceil(len(days) / 30))
        _LOGGER.info(
            "开始查询行业标签：行业体系=%s，交易日=%d，证券数=%d，分为%d批",
            scheme,
            len(days),
            len(universe),
            blocks,
        )
        for offset in range(0, len(days), 30):
            value = data.query_industry(codes=list(universe), dates=days[offset:offset + 30], scheme=scheme)
            if value is not None and not value.empty:
                pieces.append(value[["query_date", "code", "ind_name"]])
            completed = offset // 30 + 1
            if _progress_due(completed, blocks, updates=10):
                _LOGGER.info(
                    "行业标签查询进度：行业体系=%s，已完成%d/%d批，耗时=%s",
                    scheme,
                    completed,
                    blocks,
                    _elapsed(started),
                )
    finally:
        data.close()
    if not pieces:
        _LOGGER.warning("行业标签查询结果为空：行业体系=%s，耗时=%s", scheme, _elapsed(started))
        return pd.DataFrame(index=index, columns=universe, dtype=object)
    labels = pd.concat(pieces, ignore_index=True)
    labels["query_date"] = pd.to_datetime(labels["query_date"]).dt.normalize()
    pivot = labels.pivot_table(index="query_date", columns="code", values="ind_name", aggfunc="last")
    output = pivot.reindex(index=index.normalize(), columns=universe)
    output.index = index
    _LOGGER.info(
        "行业标签查询完成：行业体系=%s，结果=%d行 x %d列，耗时=%s",
        scheme,
        output.shape[0],
        output.shape[1],
        _elapsed(started),
    )
    return output


def _fetch_trade_status(universe: Sequence[str], dates: Sequence[Any]) -> pd.DataFrame:
    from betalens.datafeed import Datafeed

    started = time.perf_counter()
    index = pd.DatetimeIndex(sorted({pd.Timestamp(value).normalize() for value in dates}))
    result = pd.DataFrame(-1, index=index, columns=list(universe), dtype=np.int8)
    data = Datafeed("trade_status")
    try:
        blocks = max(1, math.ceil(len(index) / 120))
        _LOGGER.info(
            "开始查询交易状态：交易日=%d，证券数=%d，分为%d批",
            len(index),
            len(universe),
            blocks,
        )
        for offset in range(0, len(index), 120):
            block = index[offset:offset + 120]
            value = data.query_trade_status({"codes": list(universe), "dates": [day.strftime("%Y-%m-%d") for day in block]})
            if value is None or value.empty:
                continue
            value = value.copy()
            value["day"] = pd.to_datetime(value["datetime"]).dt.normalize()
            wide = value.pivot_table(index="day", columns="code", values="value", aggfunc="first")
            common_days = result.index.intersection(wide.index)
            common_codes = result.columns.intersection(wide.columns)
            result.loc[common_days, common_codes] = wide.loc[common_days, common_codes].astype(np.int8)
            completed = offset // 120 + 1
            if _progress_due(completed, blocks, updates=10):
                _LOGGER.info(
                    "交易状态查询进度：已完成%d/%d批，耗时=%s",
                    completed,
                    blocks,
                    _elapsed(started),
                )
    finally:
        data.close()
    _LOGGER.info(
        "交易状态查询完成：结果=%d行 x %d列，耗时=%s",
        result.shape[0],
        result.shape[1],
        _elapsed(started),
    )
    return result


def _build_pit(dates: Sequence[Any], index_code: str) -> dict[Any, set[str]]:
    from betalens.datafeed import Datafeed

    started = time.perf_counter()
    data = Datafeed("index_universe")
    result = {}
    try:
        total = len(dates)
        _LOGGER.info("开始构建时点证券池：指数=%s，交易日=%d", index_code, total)
        for position, day in enumerate(dates, 1):
            stamp = pd.Timestamp(day)
            result[stamp.date()] = set(data.get_index_universe(index_code, stamp.strftime("%Y-%m-%d")))
            if _progress_due(position, total, updates=20):
                _LOGGER.info(
                    "时点证券池构建进度：指数=%s，已完成%d/%d日，当前日期=%s，耗时=%s",
                    index_code,
                    position,
                    total,
                    stamp.strftime("%Y-%m-%d"),
                    _elapsed(started),
                )
    finally:
        data.close()
    securities = len({code for values in result.values() for code in values})
    _LOGGER.info(
        "时点证券池构建完成：指数=%s，交易日=%d，证券总数=%d，耗时=%s",
        index_code,
        len(result),
        securities,
        _elapsed(started),
    )
    return result


def _mask_pit(wide: pd.DataFrame, pit: Mapping[Any, set[str]] | None) -> pd.DataFrame:
    if wide.empty or not pit:
        return wide
    mask = pd.DataFrame(False, index=wide.index, columns=wide.columns)
    columns = set(map(str, wide.columns))
    for stamp in wide.index:
        keep = list(columns.intersection(map(str, pit.get(pd.Timestamp(stamp).date(), set()))))
        if keep:
            mask.loc[stamp, keep] = True
    return wide.where(mask)


def _pit_fingerprint(pit: Mapping[Any, set[str]] | None) -> str | None:
    if not pit:
        return None
    digest = hashlib.sha256()
    for day in sorted(pit, key=lambda value: pd.Timestamp(value)):
        digest.update(str(pd.Timestamp(day).date()).encode("utf-8"))
        digest.update(b"\0")
        for code in sorted(map(str, pit[day])):
            digest.update(code.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _cache_signature(
    spec: MiningSpec,
    start: str,
    end: str,
    performance: Mapping[str, Any],
    universe: Sequence[str],
    pit: Mapping[Any, set[str]] | None,
) -> str:
    factor = spec.factor_spec
    cache_config = performance.get("cache", {}) or {}
    payload = {
        "schema": 4,
        "dataset_version": cache_config.get("dataset_version", "default"),
        "start": start,
        "end": end,
        "table": getattr(factor, "table_name", "daily_market"),
        "index_code": getattr(factor, "index_code", None),
        "inputs": dict(getattr(factor, "inputs", {}) or {}),
        "industry_inputs": dict(getattr(factor, "industry_inputs", {}) or {}),
        "industry_scheme": getattr(factor, "industry_scheme", None),
        "winsorize_method": getattr(factor, "winsorize_method", None),
        "winsorize_n": getattr(factor, "winsorize_n", 3.0),
        "standardize_method": getattr(factor, "standardize_method", None),
        "neutralize": getattr(factor, "neutralize", None),
        "execution_price_field": getattr(factor, "backtest_metric", "收盘价(元)"),
        "valuation_price_field": "收盘价(元)",
        "universe": sorted(map(str, universe)),
        "pit_fingerprint": _pit_fingerprint(pit),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def _pit_from_frame(value: pd.DataFrame | None) -> dict[Any, set[str]] | None:
    if value is None or value.empty:
        return None
    return {pd.Timestamp(day).date(): set(value.columns[value.loc[day].fillna(False).astype(bool)]) for day in value.index}


def _slice_frame(value: pd.DataFrame | None, start: Any, end: Any) -> pd.DataFrame | None:
    if value is None or value.empty:
        return value
    index = pd.DatetimeIndex(value.index).normalize()
    keep = (index >= pd.Timestamp(start).normalize()) & (index <= pd.Timestamp(end).normalize())
    return value.loc[keep]


def _slice_data(data: MiningData, start: Any, end: Any) -> MiningData:
    start_day, end_day = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    pit = None
    if data.pit is not None:
        pit = {
            day: values
            for day, values in data.pit.items()
            if start_day <= pd.Timestamp(day).date() <= end_day
        }
    return MiningData(
        inputs={name: _slice_frame(value, start, end) for name, value in data.inputs.items()},
        price=_slice_frame(data.price, start, end),
        execution_price=_slice_frame(data.execution_price, start, end),
        trade_status=_slice_frame(data.trade_status, start, end),
        pit=pit,
        universe=list(data.universe),
        industry_by_scheme={
            name: _slice_frame(value, start, end)
            for name, value in data.industry_by_scheme.items()
        },
        cache_manifest_path=data.cache_manifest_path,
    )


def _load_cached_data(manifest_path: str | Path) -> MiningData:
    from betalens.factor.mining_cache import MiningCache

    started = time.perf_counter()
    _LOGGER.info("开始加载任务输入缓存：清单=%s", manifest_path)
    cache = MiningCache(manifest_path)
    manifest = cache.manifest
    pit_frame = cache.load("pit") if manifest.get("pit") else None
    loaded = MiningData(
        inputs={name: cache.load(name) for name in manifest.get("inputs", {})},
        price=cache.load("price"),
        execution_price=cache.load("execution_price"),
        trade_status=cache.load("trade_status") if manifest.get("trade_status") else None,
        pit=_pit_from_frame(pit_frame),
        universe=cache.universe,
        industry_by_scheme={
            name: cache.load(name)
            for name in manifest.get("industry_by_scheme", {})
        },
        cache_manifest_path=str(cache.manifest_path),
    )
    resident = sum(
        _frame_bytes(value)
        for value in [
            *loaded.inputs.values(),
            loaded.price,
            loaded.execution_price,
            loaded.trade_status,
            *loaded.industry_by_scheme.values(),
        ]
    )
    _LOGGER.info(
        "任务输入缓存加载完成：数据集=%d，证券数=%d，驻留内存约=%s，耗时=%s",
        len(loaded.inputs) + len(loaded.industry_by_scheme) + 3,
        len(loaded.universe),
        _human_bytes(resident),
        _elapsed(started),
    )
    return loaded


def _fetch_data(
    spec: MiningSpec,
    span: tuple[str, str],
    performance: Mapping[str, Any],
    params: Mapping[str, Any],
    cache_dir: Path,
) -> MiningData:
    from betalens.datafeed import get_absolute_trade_days

    started = time.perf_counter()
    factor = spec.factor_spec
    fetch_start = (pd.Timestamp(span[0]) - pd.Timedelta(days=_warmup_days(spec, params))).strftime("%Y-%m-%d")
    end = str(span[1])
    _LOGGER.info(
        "开始准备因子数据：因子=%s，评分区间=%s 至 %s，取数区间=%s 至 %s，预热天数=%d",
        getattr(factor, "name", "factor"),
        span[0],
        span[1],
        fetch_start,
        end,
        _warmup_days(spec, params),
    )
    days = list(sorted(get_absolute_trade_days(fetch_start, end, "D")))
    _LOGGER.info("交易日历准备完成：共%d个交易日", len(days))
    universe = list(performance.get("universe") or [])
    pit = None
    if getattr(factor, "index_code", None):
        pit = _build_pit(days, getattr(factor, "index_code"))
        universe = sorted({str(code) for values in pit.values() for code in values})
    if not universe:
        raise ValueError("a mining factor requires factor_spec.index_code or performance.universe")
    _LOGGER.info("证券池准备完成：共%d只证券", len(universe))

    def builder():
        build_started = time.perf_counter()
        input_specs = dict(getattr(factor, "inputs", {}) or {})
        _LOGGER.info("开始准备缓存数据：因子输入字段=%d个", len(input_specs))
        raw = {}
        for position, (name, metric) in enumerate(input_specs.items(), 1):
            _LOGGER.info(
                "开始读取因子输入：输入名=%s，字段=%s，进度=%d/%d",
                name,
                metric,
                position,
                len(input_specs),
            )
            raw[name] = _fetch_daily_wide(
                metric,
                universe,
                fetch_start,
                end,
                getattr(factor, "table_name", "daily_market"),
            )
        raw = _align_daily_wides(raw)
        inputs = {name: _mask_pit(value, pit) if getattr(factor, "mask_inputs_by_pit", False) else value for name, value in raw.items()}
        execution = _fetch_daily_wide(getattr(factor, "backtest_metric", "收盘价(元)"), universe, fetch_start, end, getattr(factor, "table_name", "daily_market"))
        price = _fetch_daily_wide("收盘价(元)", universe, fetch_start, end, getattr(factor, "table_name", "daily_market"))
        reference = next(iter(inputs.values()), price).index
        industry = {}
        schemes = dict(getattr(factor, "industry_inputs", {}) or {})
        if (
            getattr(factor, "neutralize", None) is not False
            and getattr(factor, "use_industry", False)
        ):
            schemes.setdefault("__neutralize_industry", getattr(factor, "industry_scheme", "申万一级行业"))
        for name, scheme in schemes.items():
            value = _fetch_industry_wide(scheme, universe, days, reference)
            if name == "__neutralize_industry":
                industry[scheme] = value
            else:
                inputs[name] = value
        payload = {
            "inputs": inputs,
            "price": price,
            "execution_price": execution,
            "trade_status": _fetch_trade_status(universe, days),
            "industry_by_scheme": industry,
            "pit": pit,
            "universe": universe,
            "metadata": {"span": [fetch_start, end], "factor": getattr(factor, "name", "factor")},
        }
        _LOGGER.info(
            "缓存数据准备完成：因子输入=%d个，行业数据=%d个，耗时=%s",
            len(inputs),
            len(industry),
            _elapsed(build_started),
        )
        return payload

    cache_config = performance.get("cache", {}) or {}
    if not bool(cache_config.get("data_enabled", True)):
        _LOGGER.info("缓存已关闭，将直接从数据源查询")
        payload = builder()
        loaded = MiningData(**{name: payload[name] for name in ("inputs", "price", "execution_price", "trade_status", "pit", "universe", "industry_by_scheme")})
        _LOGGER.info("因子数据准备完成：来源=直接查询，耗时=%s", _elapsed(started))
        return loaded
    from betalens.factor.mining_cache import CacheRequest, MiningCache
    cache = MiningCache.open_or_build(
        CacheRequest(
            cache_dir,
            _cache_signature(spec, fetch_start, end, performance, universe, pit),
        ),
        builder=builder,
    )
    loaded = _load_cached_data(cache.manifest_path)
    _LOGGER.info("因子数据准备完成：来源=任务输入缓存，耗时=%s", _elapsed(started))
    return loaded


def _windows(span: tuple[str, str], lengths: Sequence[int], steps: Sequence[int]) -> list[MiningWindow]:
    from betalens.datafeed import get_absolute_trade_days

    days = list(sorted(pd.Timestamp(day).normalize() for day in get_absolute_trade_days(span[0], span[1], "D")))
    output = []
    for length in map(int, lengths):
        for step in map(int, steps):
            if length < 1 or step < 1 or step > length:
                continue
            for number, offset in enumerate(range(0, len(days) - length + 1, step)):
                output.append(MiningWindow(f"{length}/{step}/{number}", days[offset].strftime("%Y-%m-%d"), days[offset + length - 1].strftime("%Y-%m-%d"), length, step))
    return output


def _sample_days(days: Sequence[Any], frequency: str) -> list[Any]:
    key = frequency.upper()
    if key == "D":
        return list(days)
    mapping = {"W": "W", "M": "M", "Q": "Q", "S": "2Q", "Y": "Y"}
    if key not in mapping:
        raise ValueError(f"unsupported rebalance frequency: {frequency}")
    values = pd.Series(pd.to_datetime(list(days)))
    return [value.date() for value in values.groupby(values.dt.to_period(mapping[key])).last()]


def _signal_pairs(start: str, end: str, frequency: str, trade_days: Sequence[Any]) -> list[tuple[Any, Any]]:
    days = sorted({pd.Timestamp(value).date() for value in trade_days})
    start_day, end_day = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    rebalances = _sample_days([day for day in days if start_day <= day <= end_day], frequency)
    lookup = {day: index for index, day in enumerate(days)}
    return [(days[lookup[day] - 1], day) for day in rebalances if lookup.get(day, 0) > 0]


def _weights_on_rebalance(weights: pd.DataFrame, pairs: Sequence[tuple[Any, Any]]) -> pd.DataFrame:
    mapping = {pd.Timestamp(signal).normalize(): pd.Timestamp(rebalance).normalize() for signal, rebalance in pairs}
    normalized = pd.DatetimeIndex(weights.index).normalize()
    keep = normalized.isin(mapping)
    output = weights.loc[keep].copy()
    output.index = pd.DatetimeIndex([mapping[day] + pd.Timedelta(minutes=10) for day in normalized[keep]])
    return output.sort_index()


def _wide_to_long(wide: pd.DataFrame, metric: str, signal_dates: Sequence[Any]) -> pd.DataFrame:
    dates = set(signal_dates)
    value = wide.loc[wide.index.map(lambda stamp: stamp.date() in dates)].stack().reset_index()
    value.columns = ["input_ts", "code", metric]
    value["input_ts"] = pd.to_datetime(value["input_ts"])
    value["datetime"] = value["input_ts"]
    value["diff_hours"] = 0.0
    return value


def _filter_pit(value: pd.DataFrame, pit: Mapping[Any, set[str]] | None) -> pd.DataFrame:
    if not pit or value.empty:
        return value
    keep = [row.code in pit.get(row.input_ts.date(), set()) for row in value.itertuples()]
    return value.loc[keep].reset_index(drop=True)


def _preprocess(value: pd.DataFrame, factor: Any, signal_dates: Sequence[Any], data: MiningData) -> pd.DataFrame:
    use_industry = bool(getattr(factor, "use_industry", False))
    use_mktcap = bool(getattr(factor, "use_mktcap", False))
    preprocessing = resolve_preprocessing_options(
        {
            "winsorize_method": getattr(factor, "winsorize_method", None),
            "winsorize_n": getattr(factor, "winsorize_n", 3.0),
            "standardize_method": getattr(factor, "standardize_method", None),
            "neutralize": getattr(factor, "neutralize", None),
        },
        legacy_enabled=bool(use_industry or use_mktcap),
        context=f"{getattr(factor, 'name', 'factor')} preprocessing",
    )
    if (
        preprocessing["winsorize_method"] == "none"
        and preprocessing["standardize_method"] == "none"
        and not (preprocessing["neutralize"] and (use_industry or use_mktcap))
    ):
        return value
    from betalens.factor.preprocessing import neutralize_factor, standardize_factor, winsorize_factor

    metric = factor.name
    value = value.dropna(subset=[metric]).copy()
    industry_panel = None
    if preprocessing["neutralize"] and use_industry:
        scheme = getattr(factor, "industry_scheme", "申万一级行业")
        cached = data.industry_by_scheme.get(scheme)
        if cached is not None and not cached.empty:
            industry_panel = _wide_to_long(cached, "__industry", signal_dates).set_index(["input_ts", "code"])["__industry"]
    mktcap_panel = None
    if preprocessing["neutralize"] and use_mktcap:
        market_cap = _fetch_daily_wide("A股流通市值(元)", data.universe, str(value.input_ts.min().date()), str(value.input_ts.max().date()), getattr(factor, "table_name", "daily_market"))
        if not market_cap.empty:
            mktcap_panel = _wide_to_long(np.log(market_cap.replace(0, np.nan)), "__mktcap", signal_dates).set_index(["input_ts", "code"])["__mktcap"]
    groups = []
    for stamp, group in value.groupby("input_ts"):
        section = group.set_index("code").copy()
        series = section[metric]
        if preprocessing["winsorize_method"] != "none":
            series = winsorize_factor(
                series,
                method=preprocessing["winsorize_method"],
                n=preprocessing["winsorize_n"],
            )
        if preprocessing["standardize_method"] != "none":
            series = standardize_factor(
                series,
                method=preprocessing["standardize_method"],
            )
        industry = industry_panel.xs(stamp, level="input_ts").reindex(series.index) if industry_panel is not None and stamp in industry_panel.index.get_level_values("input_ts") else None
        mktcap = mktcap_panel.xs(stamp, level="input_ts").reindex(series.index) if mktcap_panel is not None and stamp in mktcap_panel.index.get_level_values("input_ts") else None
        section[metric] = (
            neutralize_factor(series, industry_labels=industry, log_market_cap=mktcap)
            if preprocessing["neutralize"] and (industry is not None or mktcap is not None)
            else series
        )
        groups.append(section.reset_index())
    return pd.concat(groups, ignore_index=True) if groups else value.iloc[0:0]


def _groups(factor: Any, quantiles: int) -> tuple[list[Any], list[Any]]:
    if getattr(factor, "weight_mode", "freeplay") == "classic-long-short":
        return ["max"], ["min"]
    long_groups = list(getattr(factor, "long_groups", None) or [])
    short_groups = list(getattr(factor, "short_groups", None) or [])
    if not long_groups and not short_groups:
        raise ValueError("freeplay mode requires long_groups or short_groups")
    return long_groups, short_groups


def _build_weights(factor_wide: pd.DataFrame, spec: MiningSpec, params: Mapping[str, Any], data: MiningData, start: str, end: str, frequency: str) -> pd.DataFrame:
    from betalens.factor.factor import get_single_factor_weight, single_characteristic

    factor = spec.factor_spec
    pairs = _signal_pairs(start, end, frequency, data.price.index)
    signal_dates = [signal for signal, _ in pairs]
    if getattr(factor, "weight_mode", "freeplay") in {"event", "timing"}:
        weights = pd.DataFrame(index=pd.DatetimeIndex(signal_dates) + pd.Timedelta(minutes=10))
    else:
        value = _filter_pit(_wide_to_long(factor_wide, factor.name, signal_dates), data.pit)
        value = _preprocess(value, factor, signal_dates, data)
        if value.empty:
            return pd.DataFrame()
        quantiles = int(params.get("n_quantiles", 10))
        grouping_mode = str(params.get("grouping_mode", "equal_count"))
        labeled = single_characteristic(
            value,
            factor.name,
            {factor.name: quantiles},
            grouping_mode=grouping_mode,
        )
        long_groups, short_groups = _groups(factor, quantiles)
        weights = get_single_factor_weight(labeled, {
            "factor_key": factor.name,
            "mode": getattr(factor, "weight_mode", "freeplay"),
            "long": long_groups,
            "short": short_groups,
            "grouping_mode": grouping_mode,
        })
    return _weights_on_rebalance(weights, pairs)


def metrics_from_nav(nav: pd.Series | pd.DataFrame) -> dict[str, Any]:
    if isinstance(nav, pd.DataFrame):
        value = nav.iloc[:, 0] if nav.shape[1] else pd.Series(dtype=float)
    else:
        value = pd.Series(nav)
    value = pd.Series(value).dropna()
    returns = value.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return {"sharpe": 0.0, "ann_ret": 0.0, "ann_vol": 0.0, "mdd": 0.0, "calmar": 0.0, "n_days": int(len(returns))}
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    ann_vol = returns.std() * math.sqrt(252)
    wealth = (1 + returns).cumprod()
    mdd = float((1 - wealth / wealth.cummax()).max())
    return {"sharpe": round(float(ann_ret / ann_vol), 4), "ann_ret": round(float(ann_ret), 4), "ann_vol": round(float(ann_vol), 4), "mdd": round(mdd, 4), "calmar": round(float(ann_ret / mdd), 4) if mdd > 0 else 0.0, "n_days": int(len(returns))}


def _turnover(weights: pd.DataFrame | None) -> float:
    if weights is None or weights.empty:
        return 0.0
    columns = [name for name in weights if str(name).lower() != "cash"]
    if not columns:
        return 0.0
    values = weights.loc[:, columns].fillna(0.0).sort_index().reset_index(drop=True)
    values = pd.concat([pd.DataFrame([np.zeros(len(columns))], columns=columns), values], ignore_index=True)
    return float((0.5 * values.diff().abs().sum(axis=1).iloc[1:]).mean())


def _daily_last(value: pd.DataFrame) -> pd.DataFrame:
    result = value.copy()
    result.index = pd.DatetimeIndex(result.index).normalize()
    return result.groupby(level=0, sort=True).last()


def _rank_ic_key(
    signal: Any,
    rebalance: Any,
    next_rebalance: Any,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    return tuple(
        pd.Timestamp(value).normalize()
        for value in (signal, rebalance, next_rebalance)
    )


def _rank_ic_observations(
    factor: pd.DataFrame,
    price: pd.DataFrame,
    pair_groups: Sequence[Sequence[tuple[Any, Any]]],
) -> dict[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp], tuple[float, int]]:
    """Calculate every distinct forward-period Rank IC exactly once."""
    factor, price = _daily_last(factor), _daily_last(price)
    keys = dict.fromkeys(
        _rank_ic_key(pairs[index][0], pairs[index][1], pairs[index + 1][1])
        for pairs in pair_groups
        for index in range(max(0, len(pairs) - 1))
    )
    observations = {}
    for signal, rebalance, next_rebalance in keys:
        key = (signal, rebalance, next_rebalance)
        if signal not in factor.index or rebalance not in price.index or next_rebalance not in price.index:
            observations[key] = (np.nan, 0)
            continue
        section = pd.concat([factor.loc[signal].rename("signal"), (price.loc[next_rebalance] / price.loc[rebalance] - 1).rename("return")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        observations[key] = (np.nan, int(len(section)))
        if len(section) >= 3 and section.signal.nunique() > 1 and section["return"].nunique() > 1:
            correlation = section.signal.corr(section["return"], method="spearman")
            if pd.notna(correlation):
                observations[key] = (float(correlation), int(len(section)))
    return observations


def _rank_ic_metrics(
    observations: Mapping[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp], tuple[float, int]],
    pairs: Sequence[tuple[Any, Any]],
) -> dict[str, Any]:
    possible = max(0, len(pairs) - 1)
    selected = []
    for index in range(possible):
        key = _rank_ic_key(pairs[index][0], pairs[index][1], pairs[index + 1][1])
        if key in observations and pd.notna(observations[key][0]):
            selected.append((key[1], observations[key][0]))
    coverage = len(selected) / possible if possible else 0.0
    if not selected:
        return {
            "rank_ic": np.nan,
            "robust_rank_ic": np.nan,
            "mean_rank_ic": np.nan,
            "ic_ir": np.nan,
            "ic_coverage": coverage,
        }
    values = pd.Series([value for _, value in selected], dtype=float)
    standard_deviation = float(values.std())
    ic_ir = float(values.mean() / standard_deviation) if len(values) > 1 and standard_deviation > 0 else np.nan
    monthly = pd.Series(
        values.to_numpy(),
        index=pd.DatetimeIndex([day for day, _ in selected]),
    ).groupby(lambda day: day.to_period("M")).mean()
    median = float(monthly.median())
    robust = median - 0.25 * 1.4826 * float((monthly - median).abs().median())
    return {
        "rank_ic": robust,
        "robust_rank_ic": robust,
        "mean_rank_ic": float(values.mean()),
        "ic_ir": ic_ir,
        "ic_coverage": coverage,
    }


def _rank_ic(
    factor: pd.DataFrame,
    price: pd.DataFrame,
    pairs: Sequence[tuple[Any, Any]],
) -> dict[str, Any]:
    observations = _rank_ic_observations(factor, price, [pairs])
    return _rank_ic_metrics(observations, pairs)


def _rank_ic_audit_rows(
    observations: Mapping[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp], tuple[float, int]],
    *,
    factor_id: str,
    candidate_id: str,
    stage: str,
    params_json: str,
    window_id: str | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "factor_id": factor_id,
            "candidate_id": candidate_id,
            "stage": stage,
            "window_id": window_id,
            "params_json": params_json,
            "signal_date": signal,
            "rebalance_date": rebalance,
            "forward_end_date": next_rebalance,
            "rank_ic": value,
            "cross_section_n": count,
        }
        for (signal, rebalance, next_rebalance), (value, count) in observations.items()
    ]


def _vector_nav(weights: pd.DataFrame, price: pd.DataFrame) -> pd.Series:
    if weights is None or weights.empty or price.empty:
        return pd.Series(dtype=float)
    codes = [name for name in weights if name != "cash" and name in price]
    if not codes:
        return pd.Series(1.0, index=price.index)
    values = weights.copy()
    values.index = pd.DatetimeIndex(values.index).normalize()
    values = values.reindex(columns=codes).reindex(price.index, method="ffill").shift(1).fillna(0.0)
    return (1 + (values * price.loc[:, codes].pct_change()).sum(axis=1).fillna(0.0)).cumprod()


def _exact_nav(weights: pd.DataFrame, data: MiningData, spec: MiningSpec, amount: float, tolerance: int) -> pd.Series:
    from betalens.backtest import BacktestBase

    codes = [name for name in weights if name != "cash"]
    if not codes:
        return pd.Series(dtype=float)
    close = data.price.loc[:, [name for name in codes if name in data.price]]
    execution = data.execution_price.loc[:, [name for name in codes if name in data.execution_price]]
    status = data.trade_status.loc[:, [name for name in codes if name in data.trade_status]] if data.trade_status is not None else None
    engine = BacktestBase(weights, metric=getattr(spec.factor_spec, "backtest_metric", "收盘价(元)"), symbol=getattr(spec.factor_spec, "name", "mining"), amount=amount, time_tolerance=tolerance, table_name=getattr(spec.factor_spec, "table_name", "daily_market"), verbose=False, preloaded_cost_price=execution, preloaded_close_price=close, preloaded_trade_status=status)
    return pd.Series(engine.nav)


def _call_fit(callback: Callable, data: MiningData, params: Mapping[str, Any], window: MiningWindow, context: Mapping[str, Any]):
    try:
        return callback(data, params, window, context)
    except TypeError:
        try:
            return callback(data, params, window)
        except TypeError:
            return callback(data, params)


def _evaluate_candidate(module: str, factor_id: str, params: Mapping[str, Any], stage: str, windows: Sequence[MiningWindow], data: MiningData, evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate_started = time.perf_counter()
    params = dict(params)
    spec = _import_spec(module, params, evaluation.get("_preprocessing"))
    mode = str(spec.execution_mode).lower()
    if mode not in {"precomputed", "rolling_fit"}:
        raise ValueError(f"unsupported execution_mode={mode}")
    candidate = _candidate_id(factor_id, params)
    engine = str(evaluation.get("engine", "vector")).lower()
    amount = float(evaluation.get("initial_amount", 1e8))
    frequency = str(evaluation.get("rebal_freq", "D"))
    tolerance = int(evaluation.get("time_tolerance", 24))
    params_json = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
    window_pairs = {
        window.window_id: _signal_pairs(
            window.start,
            window.end,
            frequency,
            data.execution_price.index,
        )
        for window in windows
    }
    if _TASK_LOGS:
        _LOGGER.info(
            "%s候选开始计算：因子=%s，候选=%s，执行模式=%s，回测引擎=%s，窗口数=%d，参数=%s",
            _stage_name(stage),
            factor_id,
            candidate,
            _mode_name(mode),
            _engine_name(engine),
            len(windows),
            params_json,
        )
    full_factor = full_weights = full_nav = None
    full_rank_ic_observations = {}
    rank_ic_rows = []
    if mode == "precomputed":
        compute_started = time.perf_counter()
        if _TASK_LOGS:
            _LOGGER.info("%s候选 %s：开始计算全程因子", _stage_name(stage), candidate)
        with _heartbeat("candidate.factor", stage=stage, candidate=candidate):
            full_factor = spec.factor_spec.compute(**data.inputs, **getattr(spec.factor_spec, "compute_kwargs", {}))
        if _TASK_LOGS:
            _LOGGER.info(
                "%s候选 %s：全程因子计算完成，结果=%d行 x %d列，耗时=%s",
                _stage_name(stage),
                candidate,
                full_factor.shape[0],
                full_factor.shape[1],
                _elapsed(compute_started),
            )
        weights_started = time.perf_counter()
        with _heartbeat("candidate.weights", stage=stage, candidate=candidate):
            full_weights = _build_weights(full_factor, spec, params, data, evaluation["span"][0], evaluation["span"][1], frequency)
        if _TASK_LOGS:
            _LOGGER.info(
                "%s候选 %s：全程权重生成完成，结果=%d行 x %d列，耗时=%s",
                _stage_name(stage),
                candidate,
                full_weights.shape[0],
                full_weights.shape[1],
                _elapsed(weights_started),
            )
        if full_weights is None or full_weights.empty:
            factor_nonnull = int(full_factor.notna().sum().sum()) if isinstance(full_factor, pd.DataFrame) else 0
            pair_count = len(_signal_pairs(evaluation["span"][0], evaluation["span"][1], frequency, data.price.index))
            raise ValueError(
                "empty weights after factor/PIT/preprocessing: "
                f"factor_nonnull={factor_nonnull}, signal_pairs={pair_count}, "
                f"factor_shape={getattr(full_factor, 'shape', None)}"
            )
        if spec.window_transform is None:
            nav_started = time.perf_counter()
            with _heartbeat("candidate.nav", stage=stage, candidate=candidate, engine=engine):
                full_nav = _vector_nav(full_weights, data.price) if engine == "vector" else _exact_nav(full_weights, data, spec, amount, tolerance)
            if _TASK_LOGS:
                _LOGGER.info(
                    "%s候选 %s：全程净值回测完成，观测数=%d，耗时=%s",
                    _stage_name(stage),
                    candidate,
                    len(full_nav),
                    _elapsed(nav_started),
                )
        rank_ic_started = time.perf_counter()
        with _heartbeat("candidate.rank_ic", stage=stage, candidate=candidate):
            full_rank_ic_observations = _rank_ic_observations(
                full_factor,
                data.execution_price,
                list(window_pairs.values()),
            )
        rank_ic_rows.extend(_rank_ic_audit_rows(
            full_rank_ic_observations,
            factor_id=factor_id,
            candidate_id=candidate,
            stage=stage,
            params_json=params_json,
        ))
        if _TASK_LOGS:
            _LOGGER.info(
                "%s候选 %s：Rank IC序列预计算完成，唯一期间=%d，耗时=%s",
                _stage_name(stage),
                candidate,
                len(full_rank_ic_observations),
                _elapsed(rank_ic_started),
            )
    rows = []
    windows_started = time.perf_counter()
    for position, window in enumerate(windows, 1):
        window_started = time.perf_counter()
        report_progress = _TASK_LOGS and _progress_due(position, len(windows), updates=20)
        if report_progress:
            _LOGGER.info(
                "%s候选 %s：开始评价第%d/%d个窗口（%.1f%%），窗口方案=%s，评分区间=%s 至 %s",
                _stage_name(stage),
                candidate,
                position,
                len(windows),
                (position - 1) / len(windows) * 100,
                _window_description(window.window_id),
                window.start,
                window.end,
            )
        try:
            active_spec = spec
            factor, weights = full_factor, full_weights
            pairs = window_pairs[window.window_id]
            active_rank_ic_observations = full_rank_ic_observations
            if mode == "rolling_fit":
                active_spec = _import_spec(
                    module,
                    params,
                    evaluation.get("_preprocessing"),
                )
                if active_spec.execution_mode != "rolling_fit" or active_spec.fit_window is None:
                    raise ValueError("rolling_fit requires MiningSpec.fit_window")
                fit_start = pd.Timestamp(window.start) - pd.Timedelta(days=_warmup_days(active_spec, params))
                window_data = _slice_data(data, fit_start, window.end)
                with _heartbeat("candidate.fit_window", stage=stage, candidate=candidate, window=window.window_id):
                    result = _call_fit(active_spec.fit_window, window_data, params, window, {"spec": active_spec, "evaluation": evaluation})
                if isinstance(result, Mapping):
                    factor, weights = result.get("factor_wide"), result.get("weights")
                elif isinstance(result, pd.DataFrame):
                    weights = result
                else:
                    raise TypeError("fit_window must return DataFrame or mapping")
                if weights is None and factor is not None:
                    weights = _build_weights(factor, active_spec, params, window_data, window.start, window.end, frequency)
                if factor is not None:
                    active_rank_ic_observations = _rank_ic_observations(
                        factor,
                        window_data.execution_price,
                        [pairs],
                    )
                    rank_ic_rows.extend(_rank_ic_audit_rows(
                        active_rank_ic_observations,
                        factor_id=factor_id,
                        candidate_id=candidate,
                        stage=stage,
                        params_json=params_json,
                        window_id=window.window_id,
                    ))
                with _heartbeat("candidate.window_nav", stage=stage, candidate=candidate, window=window.window_id):
                    nav = _vector_nav(weights, window_data.price) if engine == "vector" else _exact_nav(weights, window_data, active_spec, amount, tolerance)
            elif spec.window_transform is not None:
                transformed = spec.window_transform(weights, window, {"data": data, "spec": spec, "params": params})
                weights = transformed if transformed is not None else weights
                with _heartbeat("candidate.window_nav", stage=stage, candidate=candidate, window=window.window_id):
                    nav = _vector_nav(weights, data.price) if engine == "vector" else _exact_nav(weights, data, spec, amount, tolerance)
            else:
                nav = full_nav
            if nav is None or len(nav) == 0:
                raise ValueError("empty nav")
            index = pd.DatetimeIndex(nav.index)
            keep = (index.normalize() >= pd.Timestamp(window.start)) & (index.normalize() <= pd.Timestamp(window.end))
            metrics = metrics_from_nav(pd.Series(nav.to_numpy()[keep], index=index[keep]))
            window_weights = _slice_frame(weights, window.start, window.end)
            metrics["turnover"] = _turnover(window_weights)
            if factor is not None:
                metrics.update(_rank_ic_metrics(active_rank_ic_observations, pairs))
            row = {"factor_id": factor_id, "candidate_id": candidate, "stage": stage, "window_id": window.window_id, "window_start": window.start, "window_end": window.end, "params_json": params_json, **params, **metrics, "error": None}
            if report_progress:
                window_elapsed = time.perf_counter() - windows_started
                eta = window_elapsed / position * (len(windows) - position)
                _LOGGER.info(
                    "%s候选 %s：完成第%d/%d个窗口（%.1f%%），窗口方案=%s，夏普=%.4f，最大回撤=%.2f%%，本窗口耗时=%s，预计剩余=%s",
                    _stage_name(stage),
                    candidate,
                    position,
                    len(windows),
                    position / len(windows) * 100,
                    _window_description(window.window_id),
                    float(metrics.get("sharpe", np.nan)),
                    float(metrics.get("mdd", np.nan)) * 100,
                    _elapsed(window_started),
                    _human_duration(eta),
                )
        except Exception as exc:
            row = {"factor_id": factor_id, "candidate_id": candidate, "stage": stage, "window_id": window.window_id, "window_start": window.start, "window_end": window.end, "params_json": params_json, **params, "error": f"{type(exc).__name__}: {exc}"}
            _LOGGER.exception(
                "%s候选 %s：第%d/%d个窗口计算失败，窗口方案=%s，错误=%s",
                _stage_name(stage),
                candidate,
                position,
                len(windows),
                _window_description(window.window_id),
                row["error"],
            )
        rows.append(row)
    failed = sum(row.get("error") is not None for row in rows)
    if _TASK_LOGS:
        _LOGGER.info(
            "%s候选计算完成：因子=%s，候选=%s，窗口总数=%d，成功=%d，失败=%d，总耗时=%s",
            _stage_name(stage),
            factor_id,
            candidate,
            len(rows),
            len(rows) - failed,
            failed,
            _elapsed(candidate_started),
        )
    return CandidateEvaluation(rows, rank_ic_rows)


def _safe_evaluate(module: str, factor_id: str, params: Mapping[str, Any], stage: str, windows: Sequence[MiningWindow], data: MiningData, evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        return _evaluate_candidate(module, factor_id, params, stage, windows, data, evaluation)
    except Exception as exc:
        _LOGGER.exception(
            "%s候选发生致命错误：因子=%s，候选=%s，错误类型=%s，错误信息=%s",
            _stage_name(stage),
            factor_id,
            _candidate_id(factor_id, params),
            type(exc).__name__,
            exc,
        )
        return CandidateEvaluation([
            {"factor_id": factor_id, "candidate_id": _candidate_id(factor_id, params), "stage": stage, "window_id": window.window_id, "window_start": window.start, "window_end": window.end, "params_json": json.dumps(dict(params), ensure_ascii=False, sort_keys=True, default=str), **dict(params), "error": f"{type(exc).__name__}: {exc}"}
            for window in windows
        ])


_WORKER_DATA: MiningData | None = None


def _initialize_worker(
    data: MiningData | None,
    cache_manifest_path: str | None,
    log_queue,
    log_level: int,
    task_logs: bool,
    heartbeat_seconds: float,
) -> None:
    global _WORKER_DATA
    _configure_worker_logging(log_queue, log_level, task_logs, heartbeat_seconds)
    _LOGGER.info("工作进程开始初始化：PID=%d，使用共享缓存=%s", os.getpid(), _yes_no(cache_manifest_path))
    _WORKER_DATA = _load_cached_data(cache_manifest_path) if cache_manifest_path else data
    _LOGGER.info("工作进程初始化完成：PID=%d", os.getpid())


def _worker_evaluate(module: str, factor_id: str, params: Mapping[str, Any], stage: str, windows: Sequence[MiningWindow], evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    if _WORKER_DATA is None:
        raise RuntimeError("mining worker data was not initialized")
    return _safe_evaluate(module, factor_id, params, stage, windows, _WORKER_DATA, evaluation)


def _summary(frame: pd.DataFrame, selection: Mapping[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    total = frame.groupby(["factor_id", "candidate_id"]).size().rename("window_count")
    valid = frame[frame["error"].isna()]
    if valid.empty:
        return pd.DataFrame()
    groups = valid.groupby(["factor_id", "candidate_id"])
    summary = (
        groups.size().rename("valid_window_count").reset_index()
        .merge(total.reset_index(), on=["factor_id", "candidate_id"], how="left")
    )
    excluded = {
        "stage", "window_id", "window_start", "window_end", "error",
        "factor_id", "candidate_id", "params_json", "parent_candidate_id",
        "winner_rank", "perturbation_index",
    }
    parameter_names: set[str] = set()
    if "params_json" in valid:
        for payload in valid["params_json"].dropna().head(10):
            try:
                parameter_names.update(json.loads(str(payload)))
            except (TypeError, ValueError):
                pass
    metric_columns = [
        name for name in valid.columns
        if name not in excluded
        and name not in parameter_names
        and pd.api.types.is_numeric_dtype(valid[name])
    ]
    for name in metric_columns:
        values = groups[name].agg(
            mean="mean",
            median="median",
            p25=lambda value: value.quantile(0.25),
            min="min",
            max="max",
        ).add_prefix(f"{name}_").reset_index()
        summary = summary.merge(values, on=["factor_id", "candidate_id"], how="left")
    if "mdd_max" in summary:
        summary["max_mdd"] = summary["mdd_max"]
    identity_columns = [
        name for name in frame.columns
        if name not in {
            "stage", "window_id", "window_start", "window_end", "ann_ret", "ann_vol",
            "sharpe", "mdd", "calmar", "turnover", "rank_ic", "robust_rank_ic",
            "mean_rank_ic", "ic_coverage", "n_days", "error", "factor_id", "candidate_id",
            *metric_columns,
        }
    ]
    if identity_columns:
        identity = valid.groupby(["factor_id", "candidate_id"], as_index=False)[identity_columns].first()
        summary = summary.merge(identity, on=["factor_id", "candidate_id"], how="left")
    summary["valid_window_ratio"] = summary.valid_window_count / summary.window_count.replace(0, np.nan)
    objective = selection.get("objective", {}) or {}
    metric, aggregate = str(objective.get("metric", "sharpe")), str(objective.get("aggregate", "median"))
    source = f"{metric}_{aggregate}"
    if source not in summary:
        aggregators = {
            "mean": "mean",
            "median": "median",
            "p25": lambda value: value.quantile(0.25),
            "min": "min",
            "max": "max",
        }
        if metric not in valid.columns:
            raise ValueError(f"unknown objective metric: {metric}")
        if aggregate not in aggregators:
            raise ValueError(f"unsupported objective aggregate: {aggregate}")
        source = f"{metric}_{aggregate}"
        values = (
            valid.groupby(["factor_id", "candidate_id"])[metric]
            .agg(aggregators[aggregate])
            .rename(source)
            .reset_index()
        )
        summary = summary.merge(values, on=["factor_id", "candidate_id"], how="left")
    summary["objective"] = summary[source]
    summary["selection_status"] = "candidate"
    operators = {">=": lambda left, right: left >= right, "<=": lambda left, right: left <= right, ">": lambda left, right: left > right, "<": lambda left, right: left < right, "==": lambda left, right: left == right}
    for rule in selection.get("filters", []) or []:
        name, op, target = str(rule.get("metric")), str(rule.get("op", ">=")), rule.get("value")
        if name not in summary:
            raise ValueError(f"unknown selection filter metric: {name}")
        if target is None:
            raise ValueError(f"selection filter {name} is missing value")
        if op not in operators:
            raise ValueError(f"unsupported selection operator: {op}")
        summary.loc[~operators[op](summary[name], target).fillna(False), "selection_status"] = "filtered"
    summary = summary.sort_values("objective", ascending=str(objective.get("direction", "maximize")).lower() == "minimize", na_position="last").reset_index(drop=True)
    eligible = summary.index[summary.selection_status == "candidate"]
    summary.loc[eligible[:int(selection.get("top_k", len(summary)))], "selection_status"] = "selected"
    return summary


def _evaluate_stage(
    module: str,
    factor_id: str,
    candidates: Sequence[Mapping[str, Any]],
    stage: str,
    windows: Sequence[MiningWindow],
    data: MiningData,
    evaluation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    task: MiningTask,
    candidate_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    stage_started = time.perf_counter()
    total_candidates = len(candidates)
    if not candidates:
        _LOGGER.info("%s已跳过：因子=%s，没有待计算候选", _stage_name(stage), factor_id)
        return pd.DataFrame()
    requested_workers = int(runtime.get("workers", 1))
    workers = _effective_workers(requested_workers, data, float(runtime.get("max_memory_ratio", 0.5)))
    main_file = str(getattr(sys.modules.get("__main__"), "__file__", "") or "")
    spawn_safe = not (sys.platform == "win32" and (not main_file or main_file.startswith("<")))
    use_processes = (
        str(runtime.get("backend", "process")).lower() == "process"
        and workers > 1
        and len(candidates) > 1
        and spawn_safe
    )
    if workers > 1 and not spawn_safe:
        warnings.warn(
            "Windows 多进程挖掘必须从 Python 文件启动；当前是交互式会话，将改用单进程运行",
            RuntimeWarning,
            stacklevel=2,
        )
    resident = sum(
        _frame_bytes(value)
        for value in [
            *data.inputs.values(),
            data.price,
            data.execution_price,
            data.trade_status,
            *data.industry_by_scheme.values(),
        ]
    )
    backend = "process" if use_processes else "serial"
    _LOGGER.info(
        "开始%s：因子=%s，候选数=%d，每个候选窗口数=%d，运行方式=%s，实际进程数=%d，配置进程数=%d，加载数据约=%s",
        _stage_name(stage),
        factor_id,
        total_candidates,
        len(windows),
        _backend_name(backend),
        workers if use_processes else 1,
        requested_workers,
        _human_bytes(resident),
    )

    def report(completed: int, index: int, rows: list[dict[str, Any]], candidate_started: float) -> None:
        failed = sum(row.get("error") is not None for row in rows)
        sharpe_values = [float(row["sharpe"]) for row in rows if row.get("error") is None and pd.notna(row.get("sharpe"))]
        elapsed_seconds = time.perf_counter() - stage_started
        eta = elapsed_seconds / completed * (total_candidates - completed) if completed else 0.0
        candidate_id = _candidate_id(factor_id, candidates[index])
        metadata = dict((candidate_metadata or {}).get(candidate_id, {}))
        if metadata:
            for row in rows:
                row.update(metadata)
            for row in getattr(rows, "rank_ic_series", ()):
                row.update(metadata)
        task.store.append("window_results", rows)
        task.store.append("rank_ic_series", getattr(rows, "rank_ic_series", ()))
        task.store.append(
            "errors",
            [row for row in rows if row.get("error") is not None],
        )
        task.store.append("search_progress", [{
            "event": "completed",
            "stage": stage,
            "factor_id": factor_id,
            "candidate_order": index + 1,
            "completed_order": completed,
            "candidate_id": candidate_id,
            "params_json": json.dumps(dict(candidates[index]), ensure_ascii=False, sort_keys=True, default=str),
            "valid_windows": len(rows) - failed,
            "failed_windows": failed,
            "sharpe_median": float(np.median(sharpe_values)) if sharpe_values else None,
            "candidate_elapsed_seconds": round(time.perf_counter() - candidate_started, 6),
        }])
        _LOGGER.info(
            "%s进度：因子=%s，已完成%d/%d个候选（%.1f%%），刚完成候选=%s，有效窗口=%d，失败窗口=%d，夏普中位数=%s，候选耗时=%s，预计剩余=%s",
            _stage_name(stage),
            factor_id,
            completed,
            total_candidates,
            completed / total_candidates * 100,
            candidate_id,
            len(rows) - failed,
            failed,
            f"{float(np.median(sharpe_values)):.4f}" if sharpe_values else "暂无",
            _elapsed(candidate_started),
            _human_duration(eta),
        )

    if use_processes:
        cache_manifest_path = data.cache_manifest_path
        initializer_data = None if cache_manifest_path else data
        context = mp.get_context("spawn" if sys.platform == "win32" else None)
        log_queue = context.Queue()
        listener = QueueListener(log_queue, *_LOGGER.handlers, respect_handler_level=True)
        listener.start()
        rows_by_index: dict[int, list[dict[str, Any]]] = {}
        try:
            with _worker_thread_limits():
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=context,
                    initializer=_initialize_worker,
                    initargs=(
                        initializer_data,
                        cache_manifest_path,
                        log_queue,
                        _LOGGER.level,
                        _TASK_LOGS,
                        _HEARTBEAT_SECONDS,
                    ),
                ) as executor:
                    futures = {}
                    for index, params in enumerate(candidates):
                        submitted = time.perf_counter()
                        future = executor.submit(
                            _worker_evaluate,
                            module,
                            factor_id,
                            params,
                            stage,
                            windows,
                            evaluation,
                        )
                        futures[future] = (index, submitted)
                    for completed, future in enumerate(as_completed(futures), 1):
                        index, submitted = futures[future]
                        rows = future.result()
                        rows_by_index[index] = rows
                        report(completed, index, rows, submitted)
        finally:
            listener.stop()
            log_queue.close()
            log_queue.join_thread()
        output = pd.DataFrame(
            [row for index in range(total_candidates) for row in rows_by_index[index]]
        )
    else:
        batches = []
        for index, params in enumerate(candidates):
            candidate_started = time.perf_counter()
            rows = _safe_evaluate(module, factor_id, params, stage, windows, data, evaluation)
            batches.append(rows)
            report(index + 1, index, rows, candidate_started)
        output = pd.DataFrame([row for rows in batches for row in rows])
    failures = int(output["error"].notna().sum()) if not output.empty and "error" in output else 0
    _LOGGER.info(
        "%s完成：因子=%s，候选数=%d，窗口结果=%d条，失败窗口=%d，总耗时=%s",
        _stage_name(stage),
        factor_id,
        total_candidates,
        len(output),
        failures,
        _elapsed(stage_started),
    )
    return output


def _ask_candidates(
    study,
    parameter_specs: Mapping[str, Mapping[str, Any]],
    count: int,
    *,
    complete_for_sampling: bool = False,
):
    from betalens.factor.mining_optuna import suggest_params

    trials, candidates = [], []
    for _ in range(max(0, int(count))):
        trial = study.ask()
        trials.append(trial)
        candidates.append(suggest_params(trial, parameter_specs))
        if complete_for_sampling:
            study.tell(trial, 0.0)
    return trials, candidates


def _tell_candidates(study, trials, candidates, factor_id: str, summary: pd.DataFrame) -> None:
    from betalens.factor.mining_optuna import tell_trial

    scores = {}
    if not summary.empty:
        scores = summary.set_index("candidate_id")["objective"].to_dict()
    for trial, params in zip(trials, candidates):
        candidate = _candidate_id(factor_id, {**dict(params), "factor_id": factor_id})
        score = scores.get(candidate)
        tell_trial(study, trial, None if score is None or not np.isfinite(score) else float(score))


def _grid_specs(
    parameter_specs: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[Any]], dict[str, dict[str, Any]]]:
    search_space = {}
    for name in parameter_specs:
        values, seen = [], set()
        for candidate in candidates:
            value = candidate[name]
            key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                values.append(value)
        search_space[name] = values
    specs = {}
    for name, values in search_space.items():
        source = parameter_specs[name]
        specs[name] = {"type": "categorical", "choices": values}
    return search_space, specs


def _alpha_generation_options(parameters_config: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(parameters_config.get("alpha101_parameter_generation") or {})
    allowed = {"range_multiplier", "max_dimensions", "type_limits"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown alpha101_parameter_generation options: {unknown}")
    if float(config.get("range_multiplier", 10)) <= 1:
        raise ValueError("alpha101_parameter_generation.range_multiplier must be > 1")
    if int(config.get("max_dimensions", 5)) < 1:
        raise ValueError("alpha101_parameter_generation.max_dimensions must be positive")
    return config


def _resolve_alpha_configs(
    parameters_config: Mapping[str, Any],
    factors: Any,
) -> dict[str, Any]:
    from alpha101_parameters import (
        aggregate_mining_factors,
        mining_parameter_limits,
        mining_parameter_specs,
    )

    options = _alpha_generation_options(parameters_config)
    if factors == "all":
        return aggregate_mining_factors(**options)
    if isinstance(factors, list):
        normalized: dict[str, Any] = {}
        for value in factors:
            # Compact form: ``factors: [ALPHA3, ALPHA7]``.  Alpha101 has a
            # single mining adapter, so these entries can use the automatic
            # parameter-space generator without repeating a full mapping.
            if isinstance(value, str):
                factor_id = value.strip().upper()
                if not factor_id.startswith("ALPHA") or not factor_id[5:].isdigit():
                    raise ValueError(
                        "alpha101 factors 列表中的字符串必须是 ALPHA1..ALPHA101"
                    )
                normalized[factor_id] = {
                    "module": "alpha101_mining",
                    "execution_mode": "precomputed",
                    "parameters": "auto",
                }
                continue
            if not isinstance(value, Mapping) or not value.get("id"):
                raise ValueError(
                    "factors 列表项必须是 Alpha 名称字符串，或包含 id 的配置映射"
                )
            factor_id = str(value["id"]).strip().upper()
            normalized[factor_id] = dict(value)
        factors = normalized
    if not isinstance(factors, Mapping):
        return factors
    resolved = {}
    for factor_id, raw in factors.items():
        factor = dict(raw) if isinstance(raw, Mapping) else raw
        if (
            isinstance(factor, Mapping)
            and str(factor_id).upper().startswith("ALPHA")
            and str(factor.get("parameters", "")).lower() == "auto"
        ):
            number = int(str(factor_id).upper().replace("ALPHA", ""))
            factor["parameters"] = mining_parameter_specs(number, **options)
            factor["parameter_limits"] = mining_parameter_limits(
                number,
                type_limits=options.get("type_limits"),
            )
        resolved[str(factor_id)] = factor
    return resolved


def _selection_reason(status: str) -> str:
    return {
        "selected": "通过筛选并进入排名范围",
        "filtered": "未通过筛选条件",
        "candidate": "通过筛选但未进入排名范围",
    }.get(str(status), str(status))


def _summary_records(frame: pd.DataFrame, stage: str) -> list[dict[str, Any]]:
    records = []
    for rank, row in enumerate(frame.to_dict(orient="records"), 1):
        records.append({
            "stage": stage,
            "rank": rank,
            **row,
            "selection_reason": _selection_reason(row.get("selection_status")),
        })
    return records


def _dedupe_candidates(
    factor_id: str,
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output, seen = [], set()
    for value in candidates:
        candidate = dict(value)
        candidate.setdefault("factor_id", factor_id)
        candidate_id = _candidate_id(factor_id, candidate)
        if candidate_id not in seen:
            seen.add(candidate_id)
            output.append(candidate)
    return output


def _evaluate_with_reuse(
    module: str,
    factor_id: str,
    candidates: Sequence[Mapping[str, Any]],
    stage: str,
    windows: Sequence[MiningWindow],
    data: MiningData,
    evaluation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    task: MiningTask,
    *,
    previous: pd.DataFrame | None = None,
    candidate_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    candidates = _dedupe_candidates(factor_id, candidates)
    previous = previous if previous is not None else pd.DataFrame()
    previous_ic = task.store.read("rank_ic_series") if not previous.empty else pd.DataFrame()
    reused, pending = [], []
    available = set(previous.candidate_id.astype(str)) if not previous.empty and "candidate_id" in previous else set()
    for value in candidates:
        candidate_id = _candidate_id(factor_id, value)
        if candidate_id not in available:
            pending.append(value)
            continue
        rows = previous.loc[previous.candidate_id.astype(str) == candidate_id].copy()
        if "window_id" in rows:
            rows = rows.drop_duplicates("window_id", keep="last")
        rows["stage"] = stage
        metadata = dict((candidate_metadata or {}).get(candidate_id, {}))
        for key, item in metadata.items():
            rows[key] = item
        reused.append(rows)
        task.store.append("window_results", rows.to_dict(orient="records"))
        if not previous_ic.empty and "candidate_id" in previous_ic:
            ic_rows = previous_ic.loc[
                previous_ic.candidate_id.astype(str) == candidate_id
            ].copy()
            dedupe_columns = [
                name for name in (
                    "window_id", "signal_date", "rebalance_date", "forward_end_date",
                )
                if name in ic_rows
            ]
            if dedupe_columns:
                ic_rows = ic_rows.drop_duplicates(dedupe_columns, keep="last")
            ic_rows["stage"] = stage
            task.store.append("rank_ic_series", ic_rows.to_dict(orient="records"))
        task.store.append("search_progress", [{
            "event": "completed",
            "stage": stage,
            "factor_id": factor_id,
            "candidate_id": candidate_id,
            "source": "复用已计算候选",
            "params_json": json.dumps(value, ensure_ascii=False, sort_keys=True, default=str),
            **metadata,
        }])
    evaluated = _evaluate_stage(
        module,
        factor_id,
        pending,
        stage,
        windows,
        data,
        evaluation,
        runtime,
        task,
        candidate_metadata=candidate_metadata,
    )
    parts = [*reused, evaluated]
    return pd.concat([value for value in parts if not value.empty], ignore_index=True) if any(
        not value.empty for value in parts
    ) else pd.DataFrame()


def _run_sampled_stage(
    module: str,
    factor_id: str,
    parameter_specs: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    stage: str,
    direction: str,
    windows: Sequence[MiningWindow],
    data: MiningData,
    evaluation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    selection: Mapping[str, Any],
    task: MiningTask,
    *,
    previous: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from betalens.factor.mining_optuna import create_coarse_study, seed_study_with_results

    n_trials = max(1, int(config.get("n_trials", 32)))
    default_sampler = "tpe" if stage == "refine" else "qmc"
    sampler = str(config.get("sampler", default_sampler)).lower()
    batch_size = n_trials if sampler == "qmc" else max(1, int(config.get("batch_size", 8)))
    study_config = dict(config)
    study_config.setdefault("sampler", sampler)
    study = create_coarse_study(study_config, direction=direction)
    if sampler == "tpe" and previous is not None and not previous.empty:
        seed_summary = _summary(previous, selection)
        if not seed_summary.empty and "objective" in seed_summary:
            seed_summary = seed_summary.dropna(subset=["objective"]).head(
                max(1, int(config.get("bootstrap_top_k", 16)))
            )
        else:
            seed_summary = pd.DataFrame()
            _LOGGER.warning(
                "TPE没有可导入的有效历史结果：因子=%s，前序候选全部无有效窗口；将从空study重新探索",
                factor_id,
            )
        seed_candidates = [
            {name: row[name] for name in parameter_specs}
            for row in seed_summary.to_dict(orient="records")
        ]
        seeded = seed_study_with_results(
            study,
            parameter_specs,
            seed_candidates,
            [float(value) for value in seed_summary.get("objective", [])],
        ) if not seed_summary.empty else 0
        _LOGGER.info(
            "TPE历史结果已导入：因子=%s，来源阶段候选=%d，作为先验的有效trial=%d",
            factor_id, len(seed_summary), seeded,
        )
    stage_parts = []
    for offset in range(0, n_trials, batch_size):
        trials, candidates = _ask_candidates(
            study,
            parameter_specs,
            min(batch_size, n_trials - offset),
            complete_for_sampling=sampler == "qmc",
        )
        for value in candidates:
            value.setdefault("factor_id", factor_id)
        source = f"Optuna {sampler.upper()}"
        task.store.append("search_progress", [{
            "event": "planned",
            "stage": stage,
            "factor_id": factor_id,
            "candidate_order": offset + index + 1,
            "trial_number": getattr(trial, "number", offset + index),
            "candidate_id": _candidate_id(factor_id, value),
            "source": source,
            "params_json": json.dumps(value, ensure_ascii=False, sort_keys=True, default=str),
        } for index, (trial, value) in enumerate(zip(trials, candidates))])
        known_parts = [value for value in [previous, *stage_parts] if value is not None and not value.empty]
        known = pd.concat(known_parts, ignore_index=True) if known_parts else pd.DataFrame()
        batch = _evaluate_with_reuse(
            module,
            factor_id,
            candidates,
            stage,
            windows,
            data,
            evaluation,
            runtime,
            task,
            previous=known,
        )
        if not batch.empty:
            stage_parts.append(batch)
        if sampler != "qmc":
            score_parts = [value for value in [known, batch] if not value.empty]
            score_frame = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame()
            score_summary = _summary(score_frame, selection)
            _tell_candidates(study, trials, candidates, factor_id, score_summary)
    frame = pd.concat(stage_parts, ignore_index=True) if stage_parts else pd.DataFrame()
    summary = _summary(frame, selection)
    records = _summary_records(summary, stage)
    task.store.append("candidate_summary", records)
    task.store.append("search_progress", [{"event": "summarized", **row} for row in records])
    _LOGGER.info(
        "%s筛选完成：因子=%s，采样器=%s，试验数=%d，去重候选=%d",
        _stage_name(stage), factor_id, sampler.upper(), n_trials, len(summary),
    )
    return frame, summary


def _top_parameter_rows(
    summary: pd.DataFrame,
    parameter_specs: Mapping[str, Mapping[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    selected = summary.loc[summary.selection_status == "selected"]
    source = selected if not selected.empty else summary
    return [
        {name: row[name] for name in parameter_specs if name in row}
        for row in source.head(max(1, int(count))).to_dict(orient="records")
    ]


def _stability_results(
    selected: pd.DataFrame,
    stability_summary: pd.DataFrame,
    planned_by_parent: Mapping[str, int],
    config: Mapping[str, Any],
    direction: str,
) -> pd.DataFrame:
    if selected.empty:
        return selected
    output = selected.copy()
    max_degradation = float(config.get("max_objective_degradation", 0.2))
    required_ratio = float(config.get("required_pass_ratio", 0.7))
    minimum_valid_ratio = float(config.get("minimum_valid_ratio", 0.8))
    values = []
    for row in output.to_dict(orient="records"):
        candidate_id = str(row["candidate_id"])
        planned = int(planned_by_parent.get(candidate_id, 0))
        nearby = stability_summary.loc[
            stability_summary.get(
                "parent_candidate_id",
                pd.Series(index=stability_summary.index, dtype=object),
            ).astype(str) == candidate_id
        ] if not stability_summary.empty else pd.DataFrame()
        objectives = pd.to_numeric(nearby.get("objective", pd.Series(dtype=float)), errors="coerce").dropna()
        winner_objective = float(row["objective"])
        scale = max(abs(winner_objective), 1e-12)
        if direction == "minimize":
            degradations = (objectives - winner_objective) / scale
        else:
            degradations = (winner_objective - objectives) / scale
        passed = degradations <= max_degradation
        valid_ratio = len(objectives) / planned if planned else 0.0
        pass_ratio = float(passed.mean()) if len(passed) else 0.0
        stable = planned > 0 and valid_ratio >= minimum_valid_ratio and pass_ratio >= required_ratio
        values.append({
            "stability_status": "not_tested" if planned == 0 else ("stable" if stable else "unstable"),
            "perturbation_count": planned,
            "valid_perturbation_count": len(objectives),
            "valid_perturbation_ratio": valid_ratio,
            "perturbation_pass_ratio": pass_ratio,
            "perturbed_objective_median": float(objectives.median()) if len(objectives) else np.nan,
            "objective_degradation_median": float(degradations.median()) if len(degradations) else np.nan,
        })
    output = pd.concat([output.reset_index(drop=True), pd.DataFrame(values)], axis=1)
    if bool(config.get("require_pass", False)):
        output = output.loc[output.stability_status == "stable"].reset_index(drop=True)
    return output


def _run_factor(
    task: MiningTask,
    factor_id: str,
    factor_config: Mapping[str, Any],
    parameters_config: Mapping[str, Any],
    performance: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    windows: Sequence[MiningWindow],
) -> FactorMiningResult:
    from betalens.factor.mining_optuna import (
        create_fine_grid_study,
        detect_boundary_pressure,
        expand_parameter_specs,
        generate_fine_candidates,
        generate_perturbation_candidates,
    )

    started = time.perf_counter()
    module = str(factor_config.get("module") or "")
    mode = str(factor_config.get("execution_mode", "")).lower()
    if not module:
        raise ValueError(f"factor {factor_id} is missing module")
    if mode not in {"precomputed", "rolling_fit"}:
        raise ValueError(f"factor {factor_id} requires execution_mode=precomputed or rolling_fit")
    parameter_specs = dict(factor_config.get("parameters") or {})
    if str(factor_id).upper().startswith("ALPHA"):
        number = int(str(factor_id).upper().replace("ALPHA", ""))
        parameter_specs.setdefault("alpha_id", {"type": "int", "low": number, "high": number, "step": 1})
    validate_parameter_specs(parameter_specs)
    parameter_limits = dict(factor_config.get("parameter_limits") or {})
    task.write_metadata(
        execution_mode=mode,
        resolved_parameter_specs=parameter_specs,
        parameter_limits=parameter_limits,
    )
    _LOGGER.info(
        "开始处理因子：因子=%s，模块=%s，执行模式=%s，待挖掘参数=%s，任务目录=%s",
        factor_id, module, _mode_name(mode), ",".join(parameter_specs), task.run_dir,
    )
    sample = {name: spec.get("high", (spec.get("choices") or [None])[0]) for name, spec in parameter_specs.items()}
    sample["factor_id"] = factor_id
    factor_evaluation = {**dict(evaluation), **dict(factor_config.get("evaluation") or {})}
    global_preprocessing = parameters_config.get("preprocessing") or {}
    factor_preprocessing = factor_config.get("preprocessing") or {}
    if not isinstance(factor_preprocessing, Mapping):
        raise ValueError(f"factor {factor_id} preprocessing must be a mapping")
    preprocessing = {
        **dict(global_preprocessing),
        **dict(factor_preprocessing),
    }
    imported_spec = _import_spec(module, sample, preprocessing)
    factor_evaluation["_preprocessing"] = preprocessing
    if mode != str(imported_spec.execution_mode).lower():
        raise ValueError(
            f"factor {factor_id} execution_mode={mode} does not match make_mining_spec={imported_spec.execution_mode}"
        )
    span = factor_evaluation["span"]
    data = _fetch_data(imported_spec, (str(span[0]), str(span[1])), performance, sample, task.cache_dir)
    cache_signature = _cache_signature(imported_spec, str(span[0]), str(span[1]), performance, data.universe, data.pit)
    task.write_metadata(cache_signature=cache_signature, cache_manifest=data.cache_manifest_path)

    search = parameters_config.get("search") or {}
    selection = parameters_config.get("selection") or {}
    direction = str((selection.get("objective") or {}).get("direction", "maximize")).lower()
    runtime = performance.get("runtime") or {}
    coarse_config = search.get("coarse") or {}
    _LOGGER.info(
        "开始宽范围搜索：因子=%s，采样器=%s，试验数=%d，参数边界=%s",
        factor_id,
        str(coarse_config.get("sampler", "qmc")).upper(),
        max(1, int(coarse_config.get("n_trials", 32))),
        json.dumps(parameter_specs, ensure_ascii=False, default=str),
    )
    coarse, coarse_summary = _run_sampled_stage(
        module, str(factor_id), parameter_specs, coarse_config, "coarse", direction,
        windows, data, factor_evaluation, runtime, selection, task,
    )

    stage_frames: dict[str, pd.DataFrame] = {"coarse": coarse}
    stage_summaries: dict[str, pd.DataFrame] = {"coarse": coarse_summary}
    broad_frames = [coarse] if not coarse.empty else []

    refine_config = dict(search.get("refine") or {})
    if refine_config and bool(refine_config.get("enabled", True)):
        refine_config.setdefault("sampler", "tpe")
        previous = pd.concat(broad_frames, ignore_index=True) if broad_frames else pd.DataFrame()
        refine, refine_summary = _run_sampled_stage(
            module, str(factor_id), parameter_specs, refine_config, "refine", direction,
            windows, data, factor_evaluation, runtime, selection, task, previous=previous,
        )
        stage_frames["refine"] = refine
        stage_summaries["refine"] = refine_summary
        if not refine.empty:
            broad_frames.append(refine)

    def broad_results() -> pd.DataFrame:
        if not broad_frames:
            return pd.DataFrame()
        value = pd.concat(broad_frames, ignore_index=True)
        return value.drop_duplicates(["candidate_id", "window_id"], keep="last")

    current_specs = {name: dict(spec) for name, spec in parameter_specs.items()}
    broad = broad_results()
    broad_summary = _summary(broad, selection)
    expansion_config = dict(search.get("expansion") or {})
    if expansion_config and bool(expansion_config.get("enabled", True)):
        max_rounds = max(1, int(expansion_config.get("max_rounds", 1)))
        boundary_top_k = max(1, int(expansion_config.get("boundary_top_k", 3)))
        for round_number in range(1, max_rounds + 1):
            winner_params = _top_parameter_rows(broad_summary, current_specs, boundary_top_k)
            pressure = detect_boundary_pressure(
                current_specs,
                winner_params,
                tolerance=float(expansion_config.get("boundary_tolerance", 0.1)),
                winner_ratio=float(expansion_config.get("winner_ratio", 0.67)),
            )
            if not pressure:
                _LOGGER.info("边界检查通过：因子=%s，优胜候选未持续集中在参数边界", factor_id)
                break
            expanded_specs = expand_parameter_specs(
                current_specs,
                pressure,
                multiplier=float(expansion_config.get("range_multiplier", 3.0)),
                limits=parameter_limits,
            )
            task.store.append("search_progress", [{
                "event": "boundary_check",
                "stage": f"expansion_{round_number}",
                "factor_id": factor_id,
                "source": "优胜候选触及边界，启动独立扩边study",
                "boundary_pressure_json": json.dumps(pressure, ensure_ascii=False, default=str),
                "previous_bounds_json": json.dumps(current_specs, ensure_ascii=False, default=str),
                "expanded_bounds_json": json.dumps(expanded_specs, ensure_ascii=False, default=str),
            }])
            if expanded_specs == current_specs:
                _LOGGER.info("边界扩展停止：因子=%s，参数已达到类型级硬边界", factor_id)
                break
            stage = f"expansion_{round_number}"
            round_config = dict(expansion_config)
            round_config.setdefault("sampler", "qmc")
            round_config["seed"] = int(expansion_config.get("seed", coarse_config.get("seed", 20260818))) + round_number
            _LOGGER.info(
                "启动第%d轮独立边界扩展搜索：因子=%s，触边参数=%s，新边界=%s",
                round_number, factor_id, ",".join(pressure),
                json.dumps(expanded_specs, ensure_ascii=False, default=str),
            )
            expanded, expanded_summary = _run_sampled_stage(
                module, str(factor_id), expanded_specs, round_config, stage, direction,
                windows, data, factor_evaluation, runtime, selection, task, previous=broad,
            )
            stage_frames[stage] = expanded
            stage_summaries[stage] = expanded_summary
            if not expanded.empty:
                broad_frames.append(expanded)
            current_specs = expanded_specs
            broad = broad_results()
            broad_summary = _summary(broad, selection)

    fine_config = search.get("fine") or {}
    anchors = _top_parameter_rows(
        broad_summary,
        current_specs,
        int(fine_config.get("top_k", 8)),
    )
    broad_candidates = []
    if not broad.empty:
        broad_candidates = [
            {name: row[name] for name in current_specs}
            for row in broad.drop_duplicates("candidate_id").to_dict(orient="records")
        ]
    fine_plan = generate_fine_candidates(
        current_specs, anchors, fine_config, coarse_candidates=broad_candidates,
    )
    fine_candidates = list(fine_plan.candidates)
    fine_trials = []
    fine_study = None
    if fine_candidates:
        fine_space, fine_specs = _grid_specs(current_specs, fine_candidates)
        fine_study = create_fine_grid_study(
            fine_space, direction=direction,
            seed=int(coarse_config.get("seed", 20260818)),
        )
        max_grid = math.prod(len(values) for values in fine_space.values())
        fine_trials, fine_candidates = _ask_candidates(
            fine_study, fine_specs,
            min(max_grid, max(1, int(fine_config.get("max_candidates", 256)))),
        )
        for value in fine_candidates:
            value.setdefault("factor_id", factor_id)
    plan_details = {
        "anchors_json": json.dumps(fine_plan.anchors, ensure_ascii=False, default=str),
        "local_bounds_json": json.dumps(fine_plan.local_bounds, ensure_ascii=False, default=str),
    }
    task.store.append("search_progress", [{
        "event": "planned", "stage": "fine", "factor_id": factor_id,
        "candidate_order": index + 1, "trial_number": getattr(trial, "number", index),
        "candidate_id": _candidate_id(str(factor_id), value), "source": "局部网格",
        "params_json": json.dumps(value, ensure_ascii=False, sort_keys=True, default=str),
        **plan_details,
    } for index, (trial, value) in enumerate(zip(fine_trials, fine_candidates))])

    _LOGGER.info(
        "局部网格已生成：因子=%s，锚点=%d，网格候选=%d",
        factor_id, len(anchors), len(fine_candidates),
    )
    fine = _evaluate_with_reuse(
        module, str(factor_id), fine_candidates, "fine", windows, data,
        factor_evaluation, runtime, task, previous=broad,
    )
    fine_summary = _summary(fine, selection)
    fine_records = _summary_records(fine_summary, "fine")
    task.store.append("candidate_summary", fine_records)
    task.store.append("search_progress", [{"event": "summarized", **row} for row in fine_records])
    _LOGGER.info(
        "细搜筛选完成：因子=%s，汇总候选=%d，最终入选=%d，被过滤=%d",
        factor_id,
        len(fine_summary),
        int((fine_summary.selection_status == "selected").sum()) if not fine_summary.empty else 0,
        int((fine_summary.selection_status == "filtered").sum()) if not fine_summary.empty else 0,
    )
    if fine_study is not None:
        _tell_candidates(fine_study, fine_trials, fine_candidates, str(factor_id), fine_summary)
    stage_frames["fine"] = fine
    stage_summaries["fine"] = fine_summary
    final_summary = fine_summary if not fine_summary.empty else broad_summary
    selected = final_summary[final_summary.selection_status == "selected"].copy() if not final_summary.empty else pd.DataFrame()

    stability = pd.DataFrame()
    stability_summary = pd.DataFrame()
    stability_config = dict(search.get("stability") or {})
    if stability_config and bool(stability_config.get("enabled", True)) and not selected.empty:
        stability_winners = selected.head(max(1, int(stability_config.get("top_k", 3)))).to_dict(orient="records")
        perturbation_plan = generate_perturbation_candidates(
            current_specs,
            stability_winners,
            perturbations_per_candidate=max(1, int(stability_config.get("perturbations_per_candidate", 8))),
            radius_ratio=float(stability_config.get("radius_ratio", 0.1)),
            seed=int(stability_config.get("seed", coarse_config.get("seed", 20260818))),
        )
        perturbations, perturbation_metadata = [], {}
        planned_by_parent: dict[str, int] = {}
        for value in perturbation_plan.candidates:
            token = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            metadata = dict(perturbation_plan.metadata[token])
            candidate = dict(value)
            candidate.setdefault("factor_id", factor_id)
            candidate_id = _candidate_id(str(factor_id), candidate)
            perturbations.append(candidate)
            perturbation_metadata[candidate_id] = metadata
            parent_id = str(metadata["parent_candidate_id"])
            planned_by_parent[parent_id] = planned_by_parent.get(parent_id, 0) + 1
        task.store.append("search_progress", [{
            "event": "planned",
            "stage": "stability",
            "factor_id": factor_id,
            "candidate_order": index + 1,
            "candidate_id": _candidate_id(str(factor_id), value),
            "source": "赢家邻域随机扰动",
            "params_json": json.dumps(value, ensure_ascii=False, sort_keys=True, default=str),
            **perturbation_metadata.get(_candidate_id(str(factor_id), value), {}),
        } for index, value in enumerate(perturbations)])
        previous_parts = [value for value in [broad, fine] if not value.empty]
        previous = pd.concat(previous_parts, ignore_index=True) if previous_parts else pd.DataFrame()
        _LOGGER.info(
            "开始赢家扰动验证：因子=%s，赢家数=%d，扰动候选=%d，扰动半径=%.1f%%",
            factor_id, len(stability_winners), len(perturbations),
            float(stability_config.get("radius_ratio", 0.1)) * 100,
        )
        stability = _evaluate_with_reuse(
            module, str(factor_id), perturbations, "stability", windows, data,
            factor_evaluation, runtime, task, previous=previous,
            candidate_metadata=perturbation_metadata,
        )
        stability_summary = _summary(stability, selection)
        stability_records = _summary_records(stability_summary, "stability")
        task.store.append("candidate_summary", stability_records)
        task.store.append("search_progress", [{"event": "summarized", **row} for row in stability_records])
        selected = _stability_results(
            selected,
            stability_summary,
            planned_by_parent,
            stability_config,
            direction,
        )
        stage_frames["stability"] = stability
        stage_summaries["stability"] = stability_summary
        stable_count = int((selected.get("stability_status") == "stable").sum()) if not selected.empty else 0
        _LOGGER.info(
            "赢家扰动验证完成：因子=%s，通过稳定性验证=%d/%d，是否强制通过=%s",
            factor_id, stable_count, len(stability_winners),
            _yes_no(stability_config.get("require_pass", False)),
        )
    winner_records = []
    for rank, row in enumerate(selected.to_dict(orient="records"), 1):
        winner_records.append({"rank": rank, **row})
    task.store.append("winners", winner_records)
    result = FactorMiningResult(
        factor_id=str(factor_id), run_id=task.run_id, run_dir=task.run_dir, status="complete",
        coarse_window_results=coarse, coarse_summary=coarse_summary,
        fine_window_results=fine, fine_summary=fine_summary,
        stability_window_results=stability, stability_summary=stability_summary,
        stage_window_results=stage_frames, stage_summaries=stage_summaries,
        selected_candidates=selected,
    )
    _LOGGER.info(
        "因子挖掘完成：因子=%s，宽搜候选=%d，细搜候选=%d，最终入选=%d，耗时=%s",
        factor_id, len(broad_summary), len(fine_summary), len(selected), _elapsed(started),
    )
    return result


def _validate_performance_config(performance: Mapping[str, Any]) -> None:
    output = performance.get("output") or {}
    cache = performance.get("cache") or {}
    legacy_output = {"window_results", "summary_results", "persist_full_nav"}.intersection(output)
    legacy_cache = {"enabled", "directory", "rebuild", "format"}.intersection(cache)
    if legacy_output or legacy_cache:
        keys = sorted([f"output.{key}" for key in legacy_output] + [f"cache.{key}" for key in legacy_cache])
        raise ValueError(f"legacy mining performance options are not supported: {', '.join(keys)}")
    runtime = performance.get("runtime") or {}
    if str(runtime.get("backend", "process")).lower() not in {"process", "serial"}:
        raise ValueError("runtime.backend must be process or serial")
    if int(runtime.get("workers", 1)) < 1 or int(runtime.get("chunk_size", 1)) < 1:
        raise ValueError("runtime.workers and runtime.chunk_size must be positive")
    if not 0 < float(runtime.get("max_memory_ratio", 0.5)) <= 1:
        raise ValueError("runtime.max_memory_ratio must be in (0, 1]")
    if not isinstance(cache.get("data_enabled", True), bool):
        raise ValueError("cache.data_enabled must be boolean")


def _validate_search_config(parameters: Mapping[str, Any]) -> None:
    preprocessing = parameters.get("preprocessing")
    if preprocessing is not None:
        if not isinstance(preprocessing, Mapping):
            raise ValueError("preprocessing must be a mapping")
        resolve_preprocessing_options(
            preprocessing,
            context="parameter_space.preprocessing",
            strict=True,
        )
    search = parameters.get("search") or {}
    for stage in ("coarse", "refine", "expansion"):
        config = search.get(stage) or {}
        if not config:
            continue
        sampler = str(config.get("sampler", "qmc" if stage != "refine" else "tpe")).lower()
        if sampler not in {"qmc", "random", "tpe"}:
            raise ValueError(f"search.{stage}.sampler must be qmc, random or tpe")
        if int(config.get("n_trials", 32)) < 1:
            raise ValueError(f"search.{stage}.n_trials must be positive")
        if sampler == "qmc" and str(config.get("qmc_type", "sobol")).lower() not in {"sobol", "halton"}:
            raise ValueError(f"search.{stage}.qmc_type must be sobol or halton")
    expansion = search.get("expansion") or {}
    if expansion:
        if float(expansion.get("range_multiplier", 3)) <= 1:
            raise ValueError("search.expansion.range_multiplier must be > 1")
        if not 0 <= float(expansion.get("boundary_tolerance", 0.1)) < 0.5:
            raise ValueError("search.expansion.boundary_tolerance must be in [0, 0.5)")
        if not 0 < float(expansion.get("winner_ratio", 0.67)) <= 1:
            raise ValueError("search.expansion.winner_ratio must be in (0, 1]")
    stability = search.get("stability") or {}
    if stability:
        ratios = {
            "radius_ratio": (0, 0.5),
            "required_pass_ratio": (0, 1),
            "minimum_valid_ratio": (0, 1),
        }
        for name, (low, high) in ratios.items():
            value = float(stability.get(name, 0.1 if name == "radius_ratio" else 0.7))
            if not low < value <= high:
                raise ValueError(f"search.stability.{name} must be in ({low}, {high}]")
        if float(stability.get("max_objective_degradation", 0.2)) < 0:
            raise ValueError("search.stability.max_objective_degradation must be >= 0")


def run_mining(parameter_config_path: str | Path, performance_config_path: str | Path) -> MiningResult:
    """Run one isolated mining task per factor and return the launch result."""
    parameter_path = Path(parameter_config_path).resolve()
    performance_path = Path(performance_config_path).resolve()
    factor_directory = parameter_path.parent.parent
    if str(factor_directory) not in sys.path:
        sys.path.insert(0, str(factor_directory))
    parameters_config = _load_yaml(parameter_path)
    performance = _load_yaml(performance_path)
    _validate_performance_config(performance)
    _validate_search_config(parameters_config)
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        pass
    if int(parameters_config.get("version", 1)) != 1:
        raise ValueError(f"unsupported mining parameter_space version: {parameters_config.get('version')}")
    factors = parameters_config.get("factors") or {}
    if factors == "all" and str(parameters_config.get("factor_class", "")).lower() != "alpha101":
        raise ValueError("factors: all is currently supported only for alpha101")
    if str(parameters_config.get("factor_class", "")).lower() == "alpha101":
        factors = _resolve_alpha_configs(parameters_config, factors)
    elif isinstance(factors, list):
        normalized = {}
        for value in factors:
            if not isinstance(value, Mapping) or not value.get("id"):
                raise ValueError(
                    "非 alpha101 因子的 factors 列表项必须是包含 id 的配置映射；"
                    "字符串列表无法推断 module"
                )
            normalized[str(value["id"])] = dict(value)
        factors = normalized
    if not isinstance(factors, Mapping) or not factors:
        raise ValueError("factors must be a non-empty mapping or `all`")
    evaluation = dict(parameters_config.get("evaluation") or {})
    span = tuple(evaluation.get("span") or parameters_config.get("span") or ())
    if len(span) != 2 or pd.Timestamp(span[0]) > pd.Timestamp(span[1]):
        raise ValueError("evaluation.span must contain an ordered [start, end]")
    evaluation["span"] = [str(span[0]), str(span[1])]
    lengths, steps = _validate_window_config(parameters_config.get("windows") or {})
    windows = _windows((str(span[0]), str(span[1])), lengths, steps)
    if not windows:
        raise ValueError("windows produced no complete windows")
    direction = str(((parameters_config.get("selection") or {}).get("objective") or {}).get("direction", "maximize")).lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("selection.objective.direction must be maximize or minimize")
    output_value = (performance.get("output") or {}).get("directory", "outputs/mining")
    output_root = Path(str(output_value))
    if not output_root.is_absolute():
        output_root = (performance_path.parent / output_root).resolve()
    launch_id = f"{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    log_config = performance.get("logging") or {}
    level_name = str(log_config.get("level", "INFO")).upper()
    levels = logging.getLevelNamesMapping()
    if level_name not in levels:
        raise ValueError(f"unsupported logging.level: {level_name}")
    heartbeat_seconds = float(log_config.get("heartbeat_seconds", 30))
    if heartbeat_seconds < 0:
        raise ValueError("logging.heartbeat_seconds must be >= 0")
    factor_runs = []
    resolved_parameters_config = {**parameters_config, "factors": factors}
    for factor_id, factor_config in factors.items():
        if not isinstance(factor_config, Mapping):
            raise TypeError(f"factor {factor_id} configuration must be a mapping")
        task = MiningTask.create(
            output_root, str(factor_id), launch_id,
            factor_class=parameters_config.get("factor_class"),
            parameter_path=parameter_path,
            performance_path=performance_path,
            config={"parameter_space": resolved_parameters_config, "performance": performance},
        )
        _configure_mining_logging(
            task.log_path, levels[level_name],
            task_logs=bool(log_config.get("task_logs", True)),
            heartbeat_seconds=heartbeat_seconds,
        )
        try:
            _LOGGER.info(
                "开始参数挖掘：启动编号=%s，运行编号=%s，因子=%s，任务目录=%s",
                launch_id, task.run_id, factor_id, task.run_dir,
            )
            result = _run_factor(
                task, str(factor_id), factor_config, resolved_parameters_config,
                performance, evaluation, windows,
            )
            task.finish(result, status="complete", windows={"lengths": lengths, "steps": steps, "count": len(windows)})
            factor_runs.append(result)
            _LOGGER.info("审计结果已生成：%s", task.workbook_path)
        except Exception as exc:
            _LOGGER.exception("参数挖掘失败：错误类型=%s，错误信息=%s", type(exc).__name__, exc)
            task.store.append("errors", [{
                "stage": "run", "factor_id": str(factor_id),
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
            }])
            failed = FactorMiningResult(
                factor_id=str(factor_id), run_id=task.run_id, run_dir=task.run_dir, status="failed",
            )
            task.finish(
                failed, status="failed", error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            raise
        finally:
            _close_mining_logging()
    return MiningResult(launch_id=launch_id, factor_runs=tuple(factor_runs))


__all__ = [
    "FactorMiningResult",
    "MiningData",
    "MiningResult",
    "MiningSpec",
    "MiningWindow",
    "metrics_from_nav",
    "run_mining",
    "validate_parameter_specs",
]
