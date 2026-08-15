from __future__ import annotations

import contextlib
import importlib
import itertools
import hashlib
import json
import os
import shutil
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


__all__ = [
    "ParameterSweepConfig",
    "RollingMiningConfig",
    "build_grid_combos",
    "gen_rolling_windows",
    "gen_rolling_train_test_windows",
    "mean_one_way_turnover",
    "metrics_from_nav",
    "robust_rank_ic_metrics",
    "run_parameter_sweep",
    "run_walk_forward",
    "tally_champions",
]


DEFAULT_WORKERS = min(10, max(1, (os.cpu_count() or 4) - 2))
DEFAULT_MAX_MEMORY_RATIO = 0.50
DEFAULT_CACHE_MEMORY_MULTIPLIER = 3.0
DEFAULT_WORKER_MEMORY_OVERHEAD_BYTES = 512 * 1024 * 1024
CACHE_SCHEMA_VERSION = 3


@dataclass
class CachePaths:
    data: Path
    pit: Path
    meta: Path


@dataclass
class ParameterSweepConfig:
    factor_module: str
    output_dir: str | Path
    span: tuple[str, str]
    grid: Mapping[str, Sequence[Any]]
    spec_factory: str = "make_mining_spec"
    gid_factory: str | None = None
    weight_hook: str | None = None
    warmup_days_factory: Callable[[Mapping[str, Any]], int] | None = None
    objective: str = "sharpe"
    objective_higher_is_better: bool = True
    engine: str = "exact"
    workers: int = 1
    cache_dir: str | Path | None = None
    rebuild_cache: bool = False
    rebal_freq: str = "D"
    n_quantiles_param: str = "n_quantiles"
    initial_amount: float = 1e8
    time_tolerance: int = 24 * 11
    max_warmup_days: int = 1200
    max_memory_ratio: float = DEFAULT_MAX_MEMORY_RATIO
    max_memory_bytes: int | None = None
    cache_memory_multiplier: float = DEFAULT_CACHE_MEMORY_MULTIPLIER
    worker_memory_overhead_bytes: int = DEFAULT_WORKER_MEMORY_OVERHEAD_BYTES
    universe: Sequence[str] | None = None
    results_filename: str = "sweep_results.csv"


@dataclass
class RollingMiningConfig:
    factor_module: str
    output_dir: str | Path
    grid: Mapping[str, Sequence[Any]]
    train: tuple[str, str]
    test: tuple[str, str]
    valid: tuple[str, str]
    train_schemes: Sequence[tuple[int, int]]
    test_schemes: Sequence[tuple[int, int]]
    spec_factory: str = "make_mining_spec"
    gid_factory: str | None = None
    weight_hook: str | None = None
    valid_report_hook: str | None = None
    warmup_days_factory: Callable[[Mapping[str, Any]], int] | None = None
    objective: str = "sharpe"
    objective_higher_is_better: bool = True
    candidate_percentile: tuple[float, float] = (0.50, 0.75)
    report_top_n: int = 3
    engine: str = "exact"
    workers: int = DEFAULT_WORKERS
    cache_dir: str | Path | None = None
    rebuild_cache: bool = False
    rebal_freq: str = "D"
    n_quantiles_param: str = "n_quantiles"
    initial_amount: float = 1e8
    time_tolerance: int = 24 * 11
    max_warmup_days: int = 1200
    max_memory_ratio: float = DEFAULT_MAX_MEMORY_RATIO
    max_memory_bytes: int | None = None
    cache_memory_multiplier: float = DEFAULT_CACHE_MEMORY_MULTIPLIER
    worker_memory_overhead_bytes: int = DEFAULT_WORKER_MEMORY_OVERHEAD_BYTES
    max_windows_per_scheme: int | None = None
    # ``split`` preserves the original two-stage behaviour.  ``paired``
    # creates a train window followed immediately by its test window and
    # advances both together, which is the usual walk-forward protocol.
    rolling_mode: str = "split"
    paired_schemes: Sequence[tuple[int, int, int]] | None = None
    rolling_span: tuple[str, str] | None = None
    universe: Sequence[str] | None = None
    sampler: str | None = None
    paper_params: Mapping[str, Any] | None = None
    n_trials: int = 96
    max_grid_candidates: int = 256
    random_seed: int = 20260810
    ic_coverage_min: float = 0.80
    max_drawdown_max: float = 0.35
    turnover_max: float = 1.00
    pruning_enabled: bool = False
    pruning_stages: Sequence[float] = (1.0,)
    pruning_reduction_factor: int = 3
    pruning_min_full_candidates: int = 3
    pruning_keep_paper: bool = True
    config_hash: str = ""
    log_level: str = "INFO"
    storage_url: str | None = None
    config_path: str | None = None
    runtime_root: str | Path | None = None
    min_workers: int = 10
    max_workers: int = 12
    max_active_windows: int = 12
    max_inflight_batches: int = 12
    grid_batch_max_candidates: int = 8
    grid_batch_target_seconds: float = 10.0
    grid_prefetch_batches: int = 24
    storage_queue_max_batches: int = 24
    audit_queue_max_events: int = 20_000
    console_trial_events: bool = False
    scheduler_schema_version: int = 3
    max_persist_overlap_batches: int = 1
    sqlite_batch_transactions: bool = True
    scheduler_wait_seconds: float = 0.25
    audit_incremental: bool = True
    resource_check_seconds: int = 15
    resource_wait_minutes: int = 30
    memory_low_watermark_ratio: float = 0.15
    memory_resume_watermark_ratio: float = 0.20
    audit_mirror_seconds: int = 60
    partial_refresh_seconds: int = 60
    partial_refresh_trials: int = 250
    snapshot_seconds: int = 300
    blas_threads: int = 1
    search_hash: str = ""
    runtime_hash: str = ""


_CACHE_DATA: dict[str, Any] | None = None
_PIT_UNIVERSE: dict[Any, set[str]] | None = None


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    value = float(value)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def _system_resource_snapshot() -> dict[str, int | None]:
    """Return physical and commit/pagefile capacity without opening a data source."""
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            if ok:
                return {
                    "physical_total": int(status.ullTotalPhys),
                    "physical_available": int(status.ullAvailPhys),
                    "commit_total": int(status.ullTotalPageFile),
                    "commit_available": int(status.ullAvailPageFile),
                }
        except Exception:
            pass
    else:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total_pages = int(os.sysconf("SC_PHYS_PAGES"))
            avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            return {
                "physical_total": page_size * total_pages,
                "physical_available": page_size * avail_pages,
                "commit_total": None,
                "commit_available": None,
            }
        except (AttributeError, OSError, ValueError):
            pass
    return {
        "physical_total": None,
        "physical_available": None,
        "commit_total": None,
        "commit_available": None,
    }


def _system_memory_snapshot() -> tuple[int | None, int | None]:
    resources = _system_resource_snapshot()
    return resources["physical_total"], resources["physical_available"]


def _memory_budget_bytes(config: Any) -> int | None:
    explicit = getattr(config, "max_memory_bytes", None)
    if explicit:
        return int(explicit)
    total, _ = _system_memory_snapshot()
    if not total:
        return None
    ratio = float(getattr(config, "max_memory_ratio", DEFAULT_MAX_MEMORY_RATIO))
    if ratio <= 0:
        return None
    return int(total * min(ratio, 1.0))


def _dataframe_memory_bytes(df: pd.DataFrame | pd.Series | None) -> int:
    if df is None:
        return 0
    try:
        usage = df.memory_usage(index=True, deep=True)
        if hasattr(usage, "sum"):
            return int(usage.sum())
        return int(usage)
    except Exception:
        return 0


def _cache_payload_memory_bytes(inputs: Mapping[str, Any], price: Any) -> int:
    total = 0
    seen = set()
    for obj in list(inputs.values()) + [price]:
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        total += _dataframe_memory_bytes(obj)
    return int(total)


def _clear_status_line():
    width = shutil.get_terminal_size((120, 20)).columns
    print("\r" + " " * max(width - 1, 1) + "\r", end="", flush=True)


def _render_progress_bar(
    done: int,
    total: int,
    start_time: float,
    *,
    submitted: int | None = None,
    active: int | None = None,
    final: bool = False,
):
    width = shutil.get_terminal_size((120, 20)).columns
    bar_width = 28
    ratio = (done / total) if total else 1.0
    filled = int(bar_width * ratio)
    bar = "#" * filled + "-" * (bar_width - filled)
    elapsed = time.time() - start_time
    eta = elapsed / done * (total - done) if done else 0.0
    extra = ""
    if submitted is not None:
        extra += f" submitted={submitted}/{total}"
    if active is not None:
        extra += f" active={active}"
    line = (
        f"  progress [{bar}] {done}/{total} {ratio:6.1%}"
        f" elapsed={elapsed:,.0f}s eta={eta:,.0f}s{extra}"
    )
    if len(line) > width - 1:
        line = line[: max(width - 4, 1)] + "..."
    print("\r" + line.ljust(max(width - 1, 1)), end="\n" if final else "", flush=True)


def _task_start_message(task: Mapping[str, Any], index: int, total: int) -> str:
    return (
        f"  START {index}/{total} "
        f"phase={task.get('phase')} scheme={task.get('scheme')} "
        f"window={task.get('win_start')}~{task.get('win_end')} "
        f"gid={task.get('gid')} params={task.get('params')}"
    )


def _log_task_start(task: Mapping[str, Any], index: int, total: int):
    _clear_status_line()
    print(_task_start_message(task, index, total), flush=True)


def _pit_memory_estimate_bytes(pit: Any) -> int:
    if not pit:
        return 0
    try:
        total = 0
        for key, codes in pit.items():
            total += 64 + len(str(key))
            total += sum(64 + len(str(code)) for code in codes)
        return int(total)
    except Exception:
        return 0


def _read_cache_memory_estimate(cache_paths: CachePaths) -> int:
    if cache_paths.data.exists():
        try:
            from betalens.factor.mining_cache import estimated_resident_bytes

            return estimated_resident_bytes(cache_paths.data)
        except Exception:
            pass
    return 0


def _estimate_worker_memory_bytes(config: Any, cache_paths: CachePaths) -> int:
    cache_bytes = _read_cache_memory_estimate(cache_paths)
    overhead = int(getattr(config, "worker_memory_overhead_bytes", DEFAULT_WORKER_MEMORY_OVERHEAD_BYTES))
    return max(1, int(cache_bytes + max(overhead, 0)))


def _effective_workers_for_memory(config: Any, cache_paths: CachePaths) -> int:
    requested = max(1, int(getattr(config, "workers", 1)))

    budget = _memory_budget_bytes(config)
    if not budget:
        return requested

    per_worker = _estimate_worker_memory_bytes(config, cache_paths)
    if per_worker > budget:
        raise MemoryError(
            "estimated memory for one mining worker "
            f"({_format_bytes(per_worker)}) exceeds configured cap "
            f"({_format_bytes(budget)}). Increase max_memory_ratio/"
            "max_memory_bytes, reduce cache size, or use a smaller universe/window."
        )
    allowed = max(1, int(budget // per_worker))
    effective = max(1, min(requested, allowed))
    total_mem, _ = _system_memory_snapshot()
    ratio = float(getattr(config, "max_memory_ratio", DEFAULT_MAX_MEMORY_RATIO))
    print(
        "  memory cap: "
        f"{_format_bytes(budget)}"
        f" ({min(max(ratio, 0.0), 1.0):.0%} of {_format_bytes(total_mem)}), "
        f"estimated/worker={_format_bytes(per_worker)}, "
        f"workers={effective}/{requested}"
    )
    if effective < requested:
        print("  memory guard reduced workers to stay within the configured cap")
    return effective


def _date_min(*spans: tuple[str, str]) -> str:
    return min(pd.Timestamp(span[0]) for span in spans).strftime("%Y-%m-%d")


def _date_max(*spans: tuple[str, str]) -> str:
    return max(pd.Timestamp(span[1]) for span in spans).strftime("%Y-%m-%d")


def _call_module_function(module_name: str, function_name: str, *args, **kwargs):
    if "." in function_name:
        target_module, target_function = function_name.rsplit(".", 1)
        module = importlib.import_module(target_module)
        return getattr(module, target_function)(*args, **kwargs)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)(*args, **kwargs)


def _load_spec(module_name: str, spec_factory: str, params: Mapping[str, Any]):
    return _call_module_function(module_name, spec_factory, params)


def _resolve_gid(module_name: str, gid_factory: str | None, params: Mapping[str, Any]) -> str:
    if gid_factory:
        return str(_call_module_function(module_name, gid_factory, params))
    return default_gid(params)


def default_gid(params: Mapping[str, Any]) -> str:
    parts = []
    for key, value in params.items():
        if isinstance(value, (tuple, list)):
            value_s = "-".join(str(v) for v in value)
        elif isinstance(value, bool):
            value_s = "T" if value else "F"
        else:
            value_s = str(value)
        parts.append(f"{key}{value_s}")
    return "_".join(parts)


def build_grid_combos(
    grid: Mapping[str, Sequence[Any]],
    *,
    factor_module: str | None = None,
    gid_factory: str | None = None,
) -> list[dict[str, Any]]:
    keys = list(grid)
    combos = []
    for values in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))
        gid = _resolve_gid(factor_module, gid_factory, params) if factor_module else default_gid(params)
        combos.append({"gid": gid, "params": params})
    return combos


def gen_rolling_windows(
    start: str,
    end: str,
    win_len: int,
    step: int,
    cap: int | None = None,
) -> list[tuple[str, str]]:
    from betalens.datafeed import get_absolute_trade_days

    days = sorted(get_absolute_trade_days(start, end, "D"))
    windows = []
    i = 0
    while i + win_len <= len(days):
        s = days[i]
        e = days[i + win_len - 1]
        windows.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
        i += step
    if cap and len(windows) > cap:
        return [windows[0], windows[-1]] if cap == 2 else windows[:cap]
    return windows


def gen_rolling_train_test_windows(
    start: str,
    end: str,
    train_len: int,
    test_len: int,
    step: int,
    cap: int | None = None,
) -> list[tuple[str, str, str, str]]:
    """Generate contiguous rolling train/test pairs without look-ahead.

    The test interval starts on the first trading day after its paired train
    interval.  Both intervals then advance by ``step`` trading days.  The
    helper deliberately works on absolute trading days, matching
    :func:`gen_rolling_windows` and the rest of the mining engine.
    """
    from betalens.datafeed import get_absolute_trade_days

    train_len = int(train_len)
    test_len = int(test_len)
    step = int(step)
    if train_len < 1 or test_len < 1 or step < 1:
        raise ValueError("train_len, test_len and step must be positive")
    days = sorted(get_absolute_trade_days(start, end, "D"))
    windows = []
    i = 0
    while i + train_len + test_len <= len(days):
        train_start = days[i]
        train_end = days[i + train_len - 1]
        test_start = days[i + train_len]
        test_end = days[i + train_len + test_len - 1]
        windows.append(
            (
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            )
        )
        i += step
    if cap and len(windows) > cap:
        return [windows[0], windows[-1]] if cap == 2 else windows[:cap]
    return windows


def infer_warmup_days_from_params(params: Mapping[str, Any], minimum: int = 30) -> int:
    candidates = []
    scan_items = list(params.items())
    for key, value in params.items():
        if isinstance(value, Mapping):
            scan_items.extend((f"{key}.{sub_key}", sub_value) for sub_key, sub_value in value.items())
    for key, value in scan_items:
        key_l = str(key).lower()
        if key_l not in {"n"} and not any(
            token in key_l for token in ("window", "lookback", "period", "span", "lag")
        ):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and np.isfinite(value) and value > 1:
            candidates.append(int(value))
    try:
        from betalens.factor.signal import infer_signal_warmup

        candidates.append(infer_signal_warmup(params, minimum=minimum))
    except Exception:
        pass
    if not candidates:
        return int(minimum)
    return max(int(minimum), int(max(candidates) * 2 + 30))


def _warmup_days(config: Any, params: Mapping[str, Any]) -> int:
    if config.warmup_days_factory is not None:
        return int(config.warmup_days_factory(params))
    return infer_warmup_days_from_params(params, minimum=30)


def fetch_daily_wide(
    metric: str,
    universe: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    table_name: str = "daily_market",
) -> pd.DataFrame:
    from betalens.datafeed import Datafeed

    data = Datafeed(table_name)
    try:
        df = data.query_time_range(
            codes=list(universe) if universe is not None else None,
            start_date=start_date,
            end_date=end_date,
            metric=metric,
        )
    finally:
        data.close()
    if df.empty:
        return pd.DataFrame()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.pivot_table(index="datetime", columns="code", values="value").sort_index()


def align_daily_wides(wides: Mapping[str, pd.DataFrame | None]) -> dict[str, pd.DataFrame | None]:
    """Align market panels to one canonical timestamp per calendar day.

    ``daily_market`` metrics are not guaranteed to be stored at the same
    intraday timestamp (for example, open at 09:30 and close/returns at
    15:00).  Alpha formulas operate on a panel, so keeping those native
    timestamps would make binary operators compare differently labelled
    frames.  Pick the latest timestamp observed on each day across all
    non-empty panels, normalize every panel to calendar dates, and restore
    that canonical timestamp index.
    """
    nonempty = [wide for wide in wides.values() if wide is not None and not wide.empty]
    if not nonempty:
        return dict(wides)

    latest_by_day: dict[pd.Timestamp, pd.Timestamp] = {}
    for wide in nonempty:
        for ts in pd.DatetimeIndex(wide.index):
            stamp = pd.Timestamp(ts)
            day = stamp.normalize()
            previous = latest_by_day.get(day)
            if previous is None or stamp > previous:
                latest_by_day[day] = stamp
    days = pd.DatetimeIndex(sorted(latest_by_day))
    canonical_index = pd.DatetimeIndex([latest_by_day[day] for day in days])

    aligned: dict[str, pd.DataFrame | None] = {}
    for name, wide in wides.items():
        if wide is None or wide.empty:
            aligned[name] = wide
            continue
        frame = wide.copy()
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        frame = frame.loc[~frame.index.duplicated(keep="last")].reindex(days)
        frame.index = canonical_index
        aligned[name] = frame
    return aligned


def fetch_industry_wide(
    scheme: str,
    universe: Sequence[str] | None,
    dates,
    reference_index: pd.DatetimeIndex | None = None,
    chunk_size: int = 30,
) -> pd.DataFrame:
    """Fetch a PIT industry label panel aligned to a market wide index."""
    if not universe or not dates:
        return pd.DataFrame(index=reference_index, columns=universe or [], dtype=object)

    from betalens.datafeed import Datafeed

    pieces = []
    data = Datafeed("industry")
    try:
        day_list = list(dict.fromkeys(pd.Timestamp(day).date() for day in dates))
        for offset in range(0, len(day_list), int(chunk_size)):
            frame = data.query_industry(
                codes=list(universe),
                dates=day_list[offset : offset + int(chunk_size)],
                scheme=scheme,
            )
            if frame is not None and not frame.empty:
                pieces.append(frame[["query_date", "code", "ind_name"]])
    finally:
        data.close()

    if not pieces:
        return pd.DataFrame(index=reference_index, columns=universe, dtype=object)
    labels = pd.concat(pieces, ignore_index=True)
    labels["query_date"] = pd.to_datetime(labels["query_date"]).dt.normalize()
    pivot = labels.pivot_table(
        index="query_date", columns="code", values="ind_name", aggfunc="last"
    )
    if reference_index is None:
        return pivot.reindex(columns=universe).sort_index()
    normalized = pd.DatetimeIndex(reference_index).normalize()
    out = pivot.reindex(index=normalized, columns=universe)
    out.index = pd.DatetimeIndex(reference_index)
    return out


def fetch_trade_status_wide(
    universe: Sequence[str],
    dates: Sequence[Any],
    *,
    chunk_size: int = 120,
) -> pd.DataFrame:
    """Fetch full PIT trade status in bounded date chunks."""
    from betalens.datafeed import Datafeed

    normalized = pd.DatetimeIndex(sorted({pd.Timestamp(value).normalize() for value in dates}))
    columns = [str(code) for code in universe]
    result = pd.DataFrame(-1, index=normalized, columns=columns, dtype=np.int8)
    data = Datafeed("trade_status")
    try:
        for offset in range(0, len(normalized), max(1, int(chunk_size))):
            block = normalized[offset:offset + max(1, int(chunk_size))]
            status = data.query_trade_status({
                "codes": columns,
                "dates": [day.strftime("%Y-%m-%d") for day in block],
            })
            if status is None or status.empty:
                continue
            status = status.copy()
            status["day"] = pd.to_datetime(status["datetime"]).dt.normalize()
            wide = status.pivot_table(index="day", columns="code", values="value", aggfunc="first")
            common_dates = result.index.intersection(wide.index)
            common_codes = result.columns.intersection(wide.columns)
            result.loc[common_dates, common_codes] = wide.loc[common_dates, common_codes].astype(np.int8)
    finally:
        data.close()
    return result


def mask_wide_by_pit_universe(wide_df: pd.DataFrame, pit_universe) -> pd.DataFrame:
    """Mask a wide panel to the PIT universe effective on each calendar date."""
    if wide_df is None or wide_df.empty or not pit_universe:
        return wide_df
    mask = pd.DataFrame(False, index=wide_df.index, columns=wide_df.columns)
    columns = set(map(str, wide_df.columns))
    for ts in wide_df.index:
        members = pit_universe.get(pd.Timestamp(ts).date(), set())
        keep = list(columns.intersection(map(str, members)))
        if keep:
            mask.loc[ts, keep] = True
    return wide_df.where(mask)


def _cache_signature(config: Any, spec: Any, fetch_start: str, end: str) -> str:
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "search_hash": str(getattr(config, "search_hash", "")),
        "fetch_start": fetch_start,
        "end": end,
        "table_name": getattr(spec, "table_name", "daily_market"),
        "index_code": getattr(spec, "index_code", None),
        "backtest_metric": getattr(spec, "backtest_metric", "收盘价(元)"),
        "mask_inputs_by_pit": bool(getattr(spec, "mask_inputs_by_pit", False)),
        "inputs": dict(getattr(spec, "inputs", {}) or {}),
        "industry_inputs": dict(getattr(spec, "industry_inputs", {}) or {}),
        "industry_scheme": getattr(spec, "industry_scheme", None),
        "universe": list(getattr(config, "universe", None) or []),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def wide_to_prequery(wide_df: pd.DataFrame, metric_name: str, signal_dates) -> pd.DataFrame:
    date_set = set(signal_dates)
    mask = wide_df.index.map(lambda ts: ts.date() in date_set)
    long = wide_df.loc[mask].stack().reset_index()
    long.columns = ["input_ts", "code", metric_name]
    long["input_ts"] = pd.to_datetime(long["input_ts"])
    long["datetime"] = long["input_ts"]
    long["diff_hours"] = 0.0
    return long


def build_pit_universe(signal_dates, index_code: str, table_name: str = "index_universe"):
    from betalens.datafeed import Datafeed

    data = Datafeed(table_name)
    pit = {}
    try:
        for d in signal_dates:
            date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
            pit[d] = set(data.get_index_universe(index_code, date_str))
    finally:
        data.close()
    return pit


def filter_long_by_pit_universe(long_df: pd.DataFrame, pit_universe) -> pd.DataFrame:
    if not pit_universe or long_df.empty:
        return long_df

    def _keep(row):
        members = pit_universe.get(row["input_ts"].date())
        return bool(members) and row["code"] in members

    return long_df.loc[long_df.apply(_keep, axis=1)].reset_index(drop=True)


def _resolve_groups(spec: Any, n_quantiles: int) -> tuple[list[Any], list[Any]]:
    long_groups = getattr(spec, "long_groups", None)
    short_groups = getattr(spec, "short_groups", None)
    if getattr(spec, "weight_mode", "freeplay") == "classic-long-short":
        return [n_quantiles - 1], [0]
    long_groups = long_groups or []
    short_groups = short_groups or []
    if not long_groups and not short_groups:
        raise ValueError("freeplay mode requires long_groups or short_groups")
    return long_groups, short_groups


def _preprocess_if_needed(
    prequery: pd.DataFrame,
    spec: Any,
    signal_dates,
    universe: Sequence[str],
    fetch_start: str,
    end_date: str,
    prepared: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    use_industry = bool(getattr(spec, "use_industry", False))
    use_mktcap = bool(getattr(spec, "use_mktcap", False))
    if not (use_industry or use_mktcap):
        return prequery

    from betalens.datafeed.validation import FillStrategy, fix_null_values
    from betalens.factor.preprocessing import (
        neutralize_factor,
        query_industry_panel,
        standardize_factor,
        winsorize_factor,
    )

    metric = spec.name
    data = fix_null_values(prequery, strategy=FillStrategy.DROP, columns=[metric])
    industry_scheme = getattr(spec, "industry_scheme", "申万一级行业") if use_industry else None
    industry_panel = None
    if industry_scheme:
        prepared = prepared or {}
        cached = (prepared.get("industry_by_scheme") or {}).get(industry_scheme)
        if cached is None:
            cached = (_CACHE_DATA or {}).get("industry_by_scheme", {}).get(industry_scheme)
        if isinstance(cached, Mapping) and "values" in cached:
            cached = _cache_frame(cached, fetch_start, end_date)
        if cached is not None and not cached.empty:
            cached_long = wide_to_prequery(cached, "__mining_industry", signal_dates)
            if not cached_long.empty:
                industry_panel = cached_long.set_index(["input_ts", "code"])["__mining_industry"]
        if industry_panel is None:
            industry_panel = query_industry_panel(
                data, scheme=industry_scheme, industry_table="industry", verbose=False
            )

    mktcap_col = None
    if use_mktcap:
        mktcap_wide = fetch_daily_wide(
            "A股流通市值(元)",
            universe=universe,
            start_date=fetch_start,
            end_date=end_date,
            table_name=getattr(spec, "table_name", "daily_market"),
        )
        if not mktcap_wide.empty:
            log_mktcap = np.log(mktcap_wide.replace(0, np.nan))
            lm_long = wide_to_prequery(log_mktcap, "log_mktcap", signal_dates)
            data = data.merge(
                lm_long[["input_ts", "code", "log_mktcap"]],
                on=["input_ts", "code"],
                how="left",
            )
            mktcap_col = "log_mktcap"

    groups = []
    for ts, group in data.groupby("input_ts"):
        sub = group.copy()
        series = sub.set_index("code")[metric]
        series = winsorize_factor(series, method="mad", n=3.0)
        series = standardize_factor(series, method="zscore")

        industry = None
        if industry_panel is not None and pd.Timestamp(ts) in industry_panel.index.get_level_values("input_ts"):
            industry = industry_panel.xs(pd.Timestamp(ts), level="input_ts").reindex(series.index)
        mktcap = sub.set_index("code")[mktcap_col] if mktcap_col and mktcap_col in sub.columns else None

        if industry is not None or mktcap is not None:
            series = neutralize_factor(series, industry_labels=industry, log_market_cap=mktcap)

        sub = sub.set_index("code")
        sub[metric] = series
        groups.append(sub.reset_index())
    if not groups:
        return data.iloc[0:0]
    return pd.concat(groups, ignore_index=True)


def metrics_from_nav(nav: pd.Series | pd.DataFrame) -> dict[str, Any]:
    if isinstance(nav, pd.DataFrame):
        nav = nav.iloc[:, 0] if nav.shape[1] else pd.Series(dtype=float)
    else:
        nav = pd.Series(nav)
    nav = nav.dropna()
    r = nav.pct_change().dropna()
    if len(r) < 2 or r.std() == 0:
        return {
            "sharpe": 0.0,
            "ann_ret": 0.0,
            "ann_vol": 0.0,
            "mdd": 0.0,
            "calmar": 0.0,
            "n_days": int(len(r)),
        }
    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    wealth = (1 + r).cumprod()
    mdd = float((1 - wealth / wealth.cummax()).max())
    calmar = ann_ret / mdd if mdd > 0 else 0.0
    return {
        "sharpe": round(float(sharpe), 4),
        "ann_ret": round(float(ann_ret), 4),
        "ann_vol": round(float(ann_vol), 4),
        "mdd": round(mdd, 4),
        "calmar": round(float(calmar), 4),
        "n_days": int(len(r)),
    }


def _vector_backtest(weights: pd.DataFrame, price_wide: pd.DataFrame) -> pd.Series:
    codes = [c for c in weights.columns if c != "cash" and c in price_wide.columns]
    if not codes or weights.empty:
        return pd.Series([1.0])

    reb_days = pd.DatetimeIndex(sorted({pd.Timestamp(ts).normalize() for ts in weights.index}))
    px = price_wide.loc[
        (price_wide.index >= reb_days[0]) &
        (price_wide.index <= reb_days[-1] + pd.Timedelta(days=40)),
        codes,
    ]
    if px.empty:
        return pd.Series([1.0])

    ret = px.pct_change()
    w = weights.copy()
    w.index = pd.DatetimeIndex(w.index).normalize()
    w = w.reindex(columns=codes).reindex(px.index, method="ffill").shift(1).fillna(0.0)
    port_ret = (w * ret).sum(axis=1).dropna()
    if port_ret.empty:
        return pd.Series([1.0])
    return (1 + port_ret).cumprod()


def _init_worker(cache_data_path: str, pit_path: str):
    global _CACHE_DATA, _PIT_UNIVERSE
    warnings.filterwarnings("ignore")
    from betalens.factor.mining_cache import open_manifest

    _CACHE_DATA = open_manifest(cache_data_path)
    _PIT_UNIVERSE = None


def _worker_private_bytes_probe(cache_data_path: str, pit_path: str) -> int:
    """Map the real cache in a short-lived worker and report private bytes."""
    _init_worker(cache_data_path, pit_path)
    materialized = []
    groups = [
        (_CACHE_DATA or {}).get("inputs", {}),
        (_CACHE_DATA or {}).get("industry_by_scheme", {}),
    ]
    descriptors = [
        (_CACHE_DATA or {}).get("price"),
        (_CACHE_DATA or {}).get("execution_price"),
        (_CACHE_DATA or {}).get("trade_status"),
        (_CACHE_DATA or {}).get("pit"),
    ]
    for group in groups:
        descriptors.extend(group.values())
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping) or "values" not in descriptor:
            continue
        dates = np.asarray(descriptor["dates"])
        if not len(dates):
            continue
        end = pd.Timestamp(int(dates[min(len(dates), 64) - 1]))
        materialized.append(_cache_frame(descriptor, end=end).copy())
    if os.name == "nt":
        try:
            import ctypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                ctypes.c_ulong,
            ]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            ok = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return int(counters.PrivateUsage)
        except Exception:
            pass
    try:
        import psutil

        memory = psutil.Process(os.getpid()).memory_full_info()
        return int(getattr(memory, "private", getattr(memory, "uss", memory.rss)))
    except Exception:
        return 0


def _cache_frame(descriptor: Any, start: Any = None, end: Any = None, columns=None) -> pd.DataFrame:
    if isinstance(descriptor, pd.DataFrame):
        result = descriptor
        if start is not None:
            result = result.loc[result.index >= pd.Timestamp(start)]
        if end is not None:
            result = result.loc[result.index <= pd.Timestamp(end)]
        if columns is not None:
            result = result.loc[:, [column for column in columns if column in result.columns]]
        return result
    from betalens.factor.mining_cache import frame

    return frame(descriptor, start=start, end=end, columns=columns)


def _task_signal_dates(start: str, end: str, fetch_start: str, rebal_freq: str):
    return [signal for signal, _rebalance in _task_signal_rebalance_pairs(start, end, fetch_start, rebal_freq)]


def _sample_trade_days(days: Sequence[Any], period: str) -> list[Any]:
    period_key = str(period).upper()
    if period_key == "D":
        return list(days)
    period_map = {"W": "W", "M": "M", "Q": "Q", "S": "2Q", "Y": "Y"}
    if period_key not in period_map:
        raise ValueError(f"unsupported rebalance frequency: {period}")
    values = pd.Series(pd.to_datetime(list(days)))
    return [value.date() for value in values.groupby(values.dt.to_period(period_map[period_key])).last()]


def _task_signal_rebalance_pairs(
    start: str,
    end: str,
    fetch_start: str,
    rebal_freq: str,
    trade_days: Sequence[Any] | None = None,
):
    if trade_days is None:
        from betalens.datafeed import get_absolute_trade_days

        all_trade_days = sorted(get_absolute_trade_days(fetch_start, end, "D"))
    else:
        all_trade_days = sorted({pd.Timestamp(value).date() for value in trade_days})
        fetch_day = pd.Timestamp(fetch_start).date()
        end_day = pd.Timestamp(end).date()
        all_trade_days = [day for day in all_trade_days if fetch_day <= day <= end_day]
    start_day = pd.Timestamp(start).date()
    rebalance_dates = _sample_trade_days(
        [day for day in all_trade_days if start_day <= day <= pd.Timestamp(end).date()],
        rebal_freq,
    )
    idx = {d: i for i, d in enumerate(all_trade_days)}
    return [
        (all_trade_days[idx[day] - 1], day)
        for day in rebalance_dates
        if idx.get(day, 0) > 0
    ]


def _weights_on_rebalance_days(
    weights: pd.DataFrame,
    signal_rebalance_pairs: Sequence[tuple[Any, Any]],
) -> pd.DataFrame:
    mapping = {
        pd.Timestamp(signal).normalize(): pd.Timestamp(rebalance).normalize()
        for signal, rebalance in signal_rebalance_pairs
    }
    normalized = pd.DatetimeIndex(weights.index).normalize()
    keep = normalized.isin(mapping)
    result = weights.loc[keep].copy()
    result.index = pd.DatetimeIndex(
        [mapping[day] + pd.Timedelta(minutes=10) for day in normalized[keep]]
    )
    return result.sort_index()


def _daily_last(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.copy()
    daily.index = pd.DatetimeIndex(daily.index).normalize()
    return daily.groupby(level=0, sort=True).last()


def robust_rank_ic_metrics(
    factor_wide: pd.DataFrame,
    execution_price: pd.DataFrame,
    signal_rebalance_pairs: Sequence[tuple[Any, Any]],
) -> dict[str, Any]:
    """Measure prior-day signal IC against current-to-next execution returns."""
    factor = _daily_last(factor_wide)
    price = _daily_last(execution_price)
    observations: list[tuple[pd.Timestamp, float]] = []
    possible = max(0, len(signal_rebalance_pairs) - 1)
    for index in range(possible):
        signal_day, rebalance_day = signal_rebalance_pairs[index]
        _next_signal, next_rebalance_day = signal_rebalance_pairs[index + 1]
        signal_ts = pd.Timestamp(signal_day).normalize()
        rebalance_ts = pd.Timestamp(rebalance_day).normalize()
        next_ts = pd.Timestamp(next_rebalance_day).normalize()
        if signal_ts not in factor.index or rebalance_ts not in price.index or next_ts not in price.index:
            continue
        future_return = price.loc[next_ts] / price.loc[rebalance_ts] - 1.0
        section = pd.concat(
            [factor.loc[signal_ts].rename("signal"), future_return.rename("future_return")],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(section) < 3 or section["signal"].nunique() < 2 or section["future_return"].nunique() < 2:
            continue
        value = section["signal"].corr(section["future_return"], method="spearman")
        if pd.notna(value):
            observations.append((rebalance_ts, float(value)))

    coverage = len(observations) / possible if possible else 0.0
    if not observations:
        return {
            "robust_rank_ic": float("nan"),
            "mean_rank_ic": float("nan"),
            "ic_coverage": float(coverage),
            "valid_ic_sections": 0,
            "possible_ic_sections": int(possible),
        }
    monthly = (
        pd.Series(
            [value for _date, value in observations],
            index=pd.DatetimeIndex([date for date, _value in observations]),
            dtype=float,
        )
        .groupby(lambda value: value.to_period("M"))
        .mean()
    )
    median = float(monthly.median())
    mad = float((monthly - median).abs().median())
    robust = median - 0.25 * 1.4826 * mad
    return {
        "robust_rank_ic": float(robust),
        "mean_rank_ic": float(np.mean([value for _date, value in observations])),
        "ic_coverage": float(coverage),
        "valid_ic_sections": int(len(observations)),
        "possible_ic_sections": int(possible),
    }


def mean_one_way_turnover(weights: pd.DataFrame) -> float:
    """Average 0.5 * absolute non-cash weight change at each rebalance."""
    columns = [column for column in weights.columns if str(column).lower() != "cash"]
    if not columns or weights.empty:
        return 0.0
    values = weights.loc[:, columns].fillna(0.0).sort_index()
    previous = pd.DataFrame([np.zeros(len(columns))], columns=columns)
    stacked = pd.concat([previous, values.reset_index(drop=True)], ignore_index=True)
    return float((0.5 * stacked.diff().abs().sum(axis=1).iloc[1:]).mean())


def _run_one_task_impl(
    task: dict[str, Any],
    prepared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from betalens.backtest import BacktestBase
    from betalens.factor.factor import get_single_factor_weight, single_characteristic

    if _CACHE_DATA is None:
        raise RuntimeError("mining cache is not initialized")

    params = dict(task["params"])
    out = dict(params)
    out.update({
        "win_start": task["win_start"],
        "win_end": task["win_end"],
        "scheme": task["scheme"],
        "phase": task["phase"],
        "gid": task["gid"],
    })
    for key in ("trial_number", "study_name", "train_start", "train_end", "test_start", "test_end"):
        if key in task:
            out[key] = task[key]
    if "rank" in task:
        out["rank"] = task["rank"]

    try:
        spec = _load_spec(task["factor_module"], task["spec_factory"], params)
        start, end = task["win_start"], task["win_end"]
        fetch_start = (
            pd.Timestamp(start) - pd.Timedelta(days=int(task["warmup_days"]))
        ).strftime("%Y-%m-%d")
        prepared = prepared or {}
        trade_status_descriptor = prepared.get("trade_status", _CACHE_DATA.get("trade_status"))
        cached_trade_days = (
            pd.DatetimeIndex(np.asarray(trade_status_descriptor["dates"], dtype=np.int64))
            if isinstance(trade_status_descriptor, Mapping) and "dates" in trade_status_descriptor
            else None
        )
        if cached_trade_days is None:
            first_input = next(iter((_CACHE_DATA.get("inputs") or {}).values()), None)
            if isinstance(first_input, pd.DataFrame):
                cached_trade_days = pd.DatetimeIndex(first_input.index)
            elif isinstance(first_input, Mapping) and "dates" in first_input:
                cached_trade_days = pd.DatetimeIndex(
                    np.asarray(first_input["dates"], dtype=np.int64)
                )
        signal_rebalance_pairs = _task_signal_rebalance_pairs(
            start,
            end,
            fetch_start,
            task["rebal_freq"],
            cached_trade_days,
        )
        signal_dates = [signal for signal, _rebalance in signal_rebalance_pairs]
        if not signal_dates:
            out["error"] = "no signal dates"
            return out

        wides = {}
        input_names = list(dict.fromkeys([
            *getattr(spec, "inputs", {}),
            *getattr(spec, "industry_inputs", {}),
        ]))
        for arg_name in input_names:
            wides[arg_name] = _cache_frame(
                (prepared.get("inputs") or {}).get(arg_name, _CACHE_DATA["inputs"][arg_name]),
                fetch_start,
                pd.Timestamp(end) + pd.Timedelta(days=1),
            )

        factor_wide = spec.compute(**wides, **getattr(spec, "compute_kwargs", {}))
        weight_mode = getattr(spec, "weight_mode", "freeplay")
        universe = _CACHE_DATA.get("universe") or []

        if weight_mode in ("event", "timing"):
            if not task.get("weight_hook"):
                raise ValueError(f"{weight_mode} weight_mode requires a mining weight_hook")
            weights = pd.DataFrame(index=pd.DatetimeIndex(signal_dates) + pd.Timedelta(minutes=10))
        else:
            prequery = wide_to_prequery(factor_wide, spec.name, signal_dates)
            prequery = filter_long_by_pit_universe(prequery, _PIT_UNIVERSE)
            if prequery.empty:
                out["error"] = "empty prequery"
                return out

            prequery = _preprocess_if_needed(
                prequery, spec, signal_dates, universe, fetch_start, end, prepared
            )
            if prequery.empty:
                out["error"] = "empty after preprocessing"
                return out

            n_quantiles_key = task["n_quantiles_param"]
            if n_quantiles_key not in params:
                raise KeyError(f"mining params missing required key: {n_quantiles_key}")
            n_quantiles = int(params[n_quantiles_key])
            labeled = single_characteristic(prequery, spec.name, {spec.name: n_quantiles})
            long_groups, short_groups = _resolve_groups(spec, n_quantiles)
            weights = get_single_factor_weight(labeled, {
                "factor_key": spec.name,
                "mode": weight_mode,
                "long": long_groups,
                "short": short_groups,
            })

        if task.get("weight_hook"):
            cached_price = _cache_frame(
                prepared.get("price", _CACHE_DATA["price"]),
                fetch_start,
                pd.Timestamp(end) + pd.Timedelta(days=40),
            )
            hook_task = dict(task)
            hook_task["context"] = {
                "factor_wide": factor_wide,
                "input_wides": wides,
                "price_wide": cached_price,
                "signal_dates": signal_dates,
                "fetch_start": fetch_start,
                "win_start": start,
                "win_end": end,
                "spec": spec,
                "universe": universe,
            }
            weights = _call_module_function(
                task["factor_module"],
                task["weight_hook"],
                weights,
                hook_task,
            )
        weights = _weights_on_rebalance_days(weights, signal_rebalance_pairs)
        if weights.empty:
            out["error"] = "empty weights"
            return out

        active_codes = [
            column for column in weights.columns
            if column != "cash" and weights[column].fillna(0.0).abs().sum() > 0
        ]
        weights = weights.loc[:, active_codes + (["cash"] if "cash" in weights.columns else [])]

        price_wide = _cache_frame(
            prepared.get("price", _CACHE_DATA["price"]),
            fetch_start,
            pd.Timestamp(end) + pd.Timedelta(days=40),
        )
        execution_wide = _cache_frame(
            prepared.get("execution_price", _CACHE_DATA.get("execution_price", _CACHE_DATA["price"])),
            fetch_start,
            pd.Timestamp(end) + pd.Timedelta(days=40),
        )
        if task["engine"] == "vector":
            nav = _vector_backtest(weights, price_wide)
            turnover_weights = weights
        else:
            stock_codes = [column for column in weights.columns if column != "cash"]
            execution_price = _cache_frame(
                prepared.get("execution_price", _CACHE_DATA["execution_price"]),
                weights.index.min(),
                weights.index.max() + pd.Timedelta(hours=task["time_tolerance"]),
                stock_codes,
            )
            trade_status = _cache_frame(
                prepared.get("trade_status", _CACHE_DATA["trade_status"]),
                weights.index.min().normalize(),
                weights.index.max().normalize() + pd.Timedelta(days=1),
                stock_codes,
            )
            close_price = _cache_frame(
                prepared.get("price", _CACHE_DATA["price"]),
                weights.index.min().normalize(),
                weights.index.max().normalize(),
                stock_codes,
            )
            bt = BacktestBase(
                weights,
                metric=getattr(spec, "backtest_metric", "收盘价(元)"),
                symbol=spec.name,
                amount=task["initial_amount"],
                time_tolerance=task["time_tolerance"],
                table_name=getattr(spec, "table_name", "daily_market"),
                verbose=False,
                preloaded_cost_price=execution_price,
                preloaded_close_price=close_price,
                preloaded_trade_status=trade_status,
            )
            nav = bt.nav
            turnover_weights = getattr(bt, "actual_weight", weights)
        out.update(metrics_from_nav(nav))
        out.update(robust_rank_ic_metrics(factor_wide, execution_wide, signal_rebalance_pairs))
        out["turnover"] = mean_one_way_turnover(turnover_weights)
        out["worker_pid"] = os.getpid()
        out["wide_rows"] = int(len(factor_wide))
        out["wide_columns"] = int(factor_wide.shape[1])
        out["weight_rows"] = int(len(turnover_weights))
        out["weight_columns"] = int(turnover_weights.shape[1])
        out["data_sources"] = getattr(bt, "data_sources", {}) if task["engine"] != "vector" else {"price": "memmap"}
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return out


def _run_one_task(task: dict[str, Any]) -> dict[str, Any]:
    log_path = task.get("task_log_path")
    if not log_path:
        return _run_one_task_impl(task)
    path = Path(str(log_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            started = time.perf_counter()
            print(
                f"worker_pid={os.getpid()} phase={task.get('phase')} "
                f"window={task.get('win_start')}~{task.get('win_end')} "
                f"engine={task.get('engine')} params={json.dumps(task.get('params', {}), ensure_ascii=False, sort_keys=True)}",
                flush=True,
            )
            result = _run_one_task_impl(task)
            result["task_log_path"] = str(path.resolve())
            result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            print(f"result={json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)}", flush=True)
            return result


def _prepare_task_batch(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if _CACHE_DATA is None:
        raise RuntimeError("mining cache is not initialized")
    if not tasks:
        return {}
    first = tasks[0]
    identity = (
        first.get("window_id"), first.get("phase"), first.get("stage_index"),
        first.get("win_start"), first.get("win_end"), first.get("engine"),
    )
    for task in tasks[1:]:
        current = (
            task.get("window_id"), task.get("phase"), task.get("stage_index"),
            task.get("win_start"), task.get("win_end"), task.get("engine"),
        )
        if current != identity:
            raise ValueError("a mining batch must contain one window, phase, stage, and engine")

    fetch_start = min(
        pd.Timestamp(task["win_start"]) - pd.Timedelta(days=int(task["warmup_days"]))
        for task in tasks
    )
    compute_end = pd.Timestamp(first["win_end"]) + pd.Timedelta(days=1)
    price_end = pd.Timestamp(first["win_end"]) + pd.Timedelta(days=40)
    input_names: set[str] = set()
    industry_schemes: set[str] = set()
    for task in tasks:
        spec = _load_spec(task["factor_module"], task["spec_factory"], task["params"])
        input_names.update(getattr(spec, "inputs", {}))
        input_names.update(getattr(spec, "industry_inputs", {}))
        if bool(getattr(spec, "use_industry", False)):
            industry_schemes.add(getattr(spec, "industry_scheme", "申万一级行业"))
    inputs = {
        name: _cache_frame(_CACHE_DATA["inputs"][name], fetch_start, compute_end)
        for name in sorted(input_names)
    }
    cached_industries = (_CACHE_DATA or {}).get("industry_by_scheme", {})
    industry_by_scheme = {
        scheme: _cache_frame(cached_industries[scheme], fetch_start, compute_end)
        for scheme in sorted(industry_schemes)
        if scheme in cached_industries
    }
    return {
        "inputs": inputs,
        "industry_by_scheme": industry_by_scheme,
        "price": _cache_frame(_CACHE_DATA["price"], fetch_start, price_end),
        "execution_price": _cache_frame(
            _CACHE_DATA.get("execution_price", _CACHE_DATA["price"]), fetch_start, price_end
        ),
        "trade_status": _CACHE_DATA.get("trade_status"),
    }


def _run_task_batch(tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate a homogeneous Grid micro-batch in one worker process."""
    if not tasks:
        return []
    batch_started = time.perf_counter()
    try:
        prepared = _prepare_task_batch(tasks)
    except Exception as exc:
        results = []
        for task in tasks:
            result = {
                **task,
                "error": f"{type(exc).__name__}: {exc}",
                "worker_pid": os.getpid(),
            }
            log_path = task.get("task_log_path")
            if log_path:
                path = Path(str(log_path))
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8", buffering=1) as stream:
                    stream.write(
                        f"worker_pid={os.getpid()} batch_id={task.get('batch_id')} "
                        f"stage={task.get('stage_index')} "
                        f"params={json.dumps(task.get('params', {}), ensure_ascii=False, sort_keys=True)}\n"
                    )
                    traceback.print_exception(type(exc), exc, exc.__traceback__, file=stream)
                    stream.write(f"result={json.dumps(result, ensure_ascii=False, default=str)}\n")
                result["task_log_path"] = str(path.resolve())
            results.append(result)
        return results
    results = []
    for task in tasks:
        log_path = task.get("task_log_path")
        started = time.perf_counter()
        try:
            if log_path:
                path = Path(str(log_path))
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8", buffering=1) as stream:
                    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                        print(
                            f"worker_pid={os.getpid()} phase={task.get('phase')} "
                            f"window={task.get('win_start')}~{task.get('win_end')} "
                            f"stage={task.get('stage_index')} engine={task.get('engine')} "
                            f"batch_id={task.get('batch_id')} "
                            f"params={json.dumps(task.get('params', {}), ensure_ascii=False, sort_keys=True)}",
                            flush=True,
                        )
                        result = _run_one_task_impl(task, prepared)
                        print(
                            f"result={json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)}",
                            flush=True,
                        )
                result["task_log_path"] = str(path.resolve())
            else:
                result = _run_one_task_impl(task, prepared)
        except Exception as exc:
            result = {**task, "error": f"{type(exc).__name__}: {exc}"}
        result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        result["worker_pid"] = os.getpid()
        results.append(result)
    batch_elapsed = round(time.perf_counter() - batch_started, 6)
    for result in results:
        result["batch_elapsed_seconds"] = batch_elapsed
        result["batch_size"] = len(tasks)
    return results


def _cache_paths(cache_dir: str | Path) -> CachePaths:
    cache = _as_path(cache_dir)
    ready = cache / "READY.json"
    manifest = cache / "manifest.json"
    if ready.exists():
        try:
            pointer = json.loads(ready.read_text(encoding="utf-8"))
            manifest = cache / str(pointer["manifest"])
        except Exception:
            pass
    return CachePaths(
        data=manifest,
        pit=manifest,
        meta=ready,
    )


def _cache_dir(config: Any) -> Path:
    if config.cache_dir is not None:
        return _as_path(config.cache_dir)
    return _as_path(config.output_dir) / "_cache"


def build_cache_for_config(config: Any, spans: Sequence[tuple[str, str]], sample_params: Mapping[str, Any]) -> CachePaths:
    """Build a cache containing PIT-safe market and industry inputs.

    Industry inputs are kept in the same ``inputs`` mapping as market panels so
    existing task workers can pass them directly to ``spec.compute``.  The
    neutralization panel is stored separately by scheme and reused by every
    candidate instead of querying the industry table once per task.
    """
    from betalens.datafeed import get_absolute_trade_days

    cache_dir = _cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = _load_spec(config.factor_module, config.spec_factory, sample_params)
    start = _date_min(*spans)
    end = _date_max(*spans)
    fetch_start = (
        pd.Timestamp(start) - pd.Timedelta(days=int(config.max_warmup_days))
    ).strftime("%Y-%m-%d")
    signature = _cache_signature(config, spec, fetch_start, end)
    ready_path = cache_dir / "READY.json"
    if ready_path.exists() and not config.rebuild_cache:
        try:
            pointer = json.loads(ready_path.read_text(encoding="utf-8"))
            pointed_manifest = cache_dir / str(pointer["manifest"])
            pointed = json.loads(pointed_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, KeyError):
            pointed_manifest = None
            pointed = {}
        if pointed_manifest is not None and pointed.get("cache_signature") == signature:
            print(f"[cache] hit: {pointed_manifest}")
            return CachePaths(data=pointed_manifest, pit=pointed_manifest, meta=ready_path)
    existing_manifest = cache_dir / signature / "manifest.json"
    if existing_manifest.exists() and not config.rebuild_cache:
        try:
            meta = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            meta = {}
        if meta.get("schema_version") == CACHE_SCHEMA_VERSION and meta.get("cache_signature") == signature:
            print(f"[cache] hit: {existing_manifest}")
            return CachePaths(data=existing_manifest, pit=existing_manifest, meta=cache_dir / "READY.json")

    print(f"[cache] fetch range: {fetch_start} ~ {end}")
    all_days = sorted(get_absolute_trade_days(fetch_start, end, "D"))
    pit_days = all_days if getattr(spec, "mask_inputs_by_pit", False) else [
        d for d in all_days if d >= pd.Timestamp(start).date()
    ]

    pit = None
    universe = list(config.universe) if config.universe is not None else None
    index_code = getattr(spec, "index_code", None)
    if index_code:
        t0 = time.time()
        pit = build_pit_universe(pit_days, index_code)
        universe = sorted({code for codes in pit.values() for code in codes})
        print(f"[cache] PIT universe: {len(universe)} codes, {len(pit)} dates, {time.time() - t0:.1f}s")
    elif universe is None:
        raise ValueError("config.universe is required when spec.index_code is empty")

    raw_inputs = {}
    inputs = {}
    metrics_by_arg = dict(getattr(spec, "inputs", {}) or {})
    for arg_name, metric in metrics_by_arg.items():
        t0 = time.time()
        wide = fetch_daily_wide(
            metric,
            universe=universe,
            start_date=fetch_start,
            end_date=end,
            table_name=getattr(spec, "table_name", "daily_market"),
        )
        raw_inputs[arg_name] = wide
        print(f"[cache] {arg_name} ({metric}): {wide.shape}, {time.time() - t0:.1f}s")

    execution_metric = getattr(spec, "backtest_metric", "收盘价(元)")
    input_price = next(
        (wide for arg, metric in metrics_by_arg.items() if metric == execution_metric for wide in [raw_inputs[arg]]),
        None,
    )
    if input_price is None:
        t0 = time.time()
        execution_price = fetch_daily_wide(
            execution_metric,
            universe=universe,
            start_date=fetch_start,
            end_date=end,
            table_name=getattr(spec, "table_name", "daily_market"),
        )
        print(f"[cache] execution price ({execution_metric}): {execution_price.shape}, {time.time() - t0:.1f}s")
    else:
        execution_price = input_price.copy()

    close_metric = "收盘价(元)"
    input_close = next(
        (wide for arg, metric in metrics_by_arg.items() if metric == close_metric for wide in [raw_inputs[arg]]),
        None,
    )
    if input_close is None:
        t0 = time.time()
        close_price = fetch_daily_wide(
            close_metric,
            universe=universe,
            start_date=fetch_start,
            end_date=end,
            table_name=getattr(spec, "table_name", "daily_market"),
        )
        print(f"[cache] valuation price ({close_metric}): {close_price.shape}, {time.time() - t0:.1f}s")
    else:
        close_price = input_close.copy()

    # Market metrics can have different intraday timestamps.  Align them
    # before deriving the reference index, PIT masks, or the cached price
    # panel so every formula receives identically labelled daily wides.
    aligned_market = align_daily_wides(raw_inputs)
    raw_inputs = {arg_name: aligned_market[arg_name] for arg_name in metrics_by_arg}
    price = close_price
    for arg_name, wide in raw_inputs.items():
        inputs[arg_name] = (
            mask_wide_by_pit_universe(wide, pit)
            if getattr(spec, "mask_inputs_by_pit", False)
            else wide
        )

    reference_index = next(
        (wide.index for wide in raw_inputs.values() if wide is not None and not wide.empty),
        pd.DatetimeIndex(all_days),
    )
    industry_by_scheme = {}
    industry_specs = dict(getattr(spec, "industry_inputs", {}) or {})
    schemes = dict(industry_specs)
    neutralize_scheme = getattr(spec, "industry_scheme", None) if getattr(spec, "use_industry", False) else None
    if neutralize_scheme:
        schemes.setdefault("__neutralize_industry", neutralize_scheme)
    for arg_name, scheme in schemes.items():
        t0 = time.time()
        wide = fetch_industry_wide(scheme, universe, all_days, reference_index=reference_index)
        if arg_name == "__neutralize_industry":
            industry_by_scheme[scheme] = mask_wide_by_pit_universe(wide, pit) if getattr(spec, "mask_inputs_by_pit", False) else wide
        else:
            inputs[arg_name] = mask_wide_by_pit_universe(wide, pit) if getattr(spec, "mask_inputs_by_pit", False) else wide
        print(f"[cache] {arg_name} ({scheme}): {wide.shape}, {time.time() - t0:.1f}s")

    t0 = time.time()
    trade_status = fetch_trade_status_wide(universe or [], all_days)
    print(f"[cache] trade_status: {trade_status.shape}, {time.time() - t0:.1f}s")
    from betalens.factor.mining_cache import publish

    manifest = publish(
        cache_dir,
        signature,
        inputs=inputs,
        price=price,
        execution_price=execution_price,
        trade_status=trade_status,
        industry_by_scheme=industry_by_scheme,
        pit=pit,
        universe=list(universe or []),
        metadata={
        "input_shapes": {name: list(wide.shape) for name, wide in inputs.items()},
        "price_shape": list(price.shape) if isinstance(price, pd.DataFrame) else None,
        "execution_price_shape": list(execution_price.shape),
        "industry_schemes": sorted(industry_by_scheme),
        "universe_size": len(universe or []),
        },
        rebuild=bool(config.rebuild_cache),
    )
    print(f"[cache] saved: {cache_dir}")
    return CachePaths(data=manifest, pit=manifest, meta=cache_dir / "READY.json")


def build_tasks(
    *,
    config: Any,
    phase: str,
    span: tuple[str, str],
    schemes: Sequence[tuple[int, int]],
    combos: Sequence[Mapping[str, Any]],
    cap: int | None = None,
) -> list[dict[str, Any]]:
    tasks = []
    for win_len, step in schemes:
        scheme = f"{win_len}/{step}"
        windows = gen_rolling_windows(span[0], span[1], win_len, step, cap=cap)
        for win_start, win_end in windows:
            for combo in combos:
                params = dict(combo["params"])
                tasks.append({
                    "params": params,
                    "gid": combo["gid"],
                    "factor_module": config.factor_module,
                    "spec_factory": config.spec_factory,
                    "weight_hook": config.weight_hook,
                    "warmup_days": _warmup_days(config, params),
                    "phase": phase,
                    "scheme": scheme,
                    "win_start": win_start,
                    "win_end": win_end,
                    "engine": config.engine,
                    "rebal_freq": config.rebal_freq,
                    "n_quantiles_param": config.n_quantiles_param,
                    "initial_amount": config.initial_amount,
                    "time_tolerance": config.time_tolerance,
                })
    return tasks


def build_paired_tasks(
    *,
    config: Any,
    phase: str,
    pairs: Sequence[tuple[str, str, str, str]],
    scheme: str,
    combos: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build tasks for a paired rolling train/test schedule."""
    tasks = []
    for train_start, train_end, test_start, test_end in pairs:
        if phase == "train":
            win_start, win_end = train_start, train_end
        elif phase == "test":
            win_start, win_end = test_start, test_end
        else:
            raise ValueError(f"paired tasks only support train/test, got {phase!r}")
        for combo in combos:
            params = dict(combo["params"])
            tasks.append({
                "params": params,
                "gid": combo["gid"],
                "factor_module": config.factor_module,
                "spec_factory": config.spec_factory,
                "weight_hook": config.weight_hook,
                "warmup_days": _warmup_days(config, params),
                "phase": phase,
                "scheme": scheme,
                "win_start": win_start,
                "win_end": win_end,
                "engine": config.engine,
                "rebal_freq": config.rebal_freq,
                "n_quantiles_param": config.n_quantiles_param,
                "initial_amount": config.initial_amount,
                "time_tolerance": config.time_tolerance,
            })
    return tasks


def build_sweep_tasks(config: ParameterSweepConfig, combos: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tasks = []
    for combo in combos:
        params = dict(combo["params"])
        tasks.append({
            "params": params,
            "gid": combo["gid"],
            "factor_module": config.factor_module,
            "spec_factory": config.spec_factory,
            "weight_hook": config.weight_hook,
            "warmup_days": _warmup_days(config, params),
            "phase": "sweep",
            "scheme": "full",
            "win_start": config.span[0],
            "win_end": config.span[1],
            "engine": config.engine,
            "rebal_freq": config.rebal_freq,
            "n_quantiles_param": config.n_quantiles_param,
            "initial_amount": config.initial_amount,
            "time_tolerance": config.time_tolerance,
        })
    return tasks


def run_tasks(
    config: Any,
    tasks: Sequence[dict[str, Any]],
    cache_paths: CachePaths,
    *,
    executor: ProcessPoolExecutor | None = None,
) -> pd.DataFrame:
    rows = []
    total = len(tasks)
    if total == 0:
        return pd.DataFrame()

    effective_workers = _effective_workers_for_memory(config, cache_paths)
    t0 = time.time()
    _render_progress_bar(0, total, t0, submitted=0, active=0)
    if effective_workers <= 1:
        if executor is None:
            _init_worker(str(cache_paths.data), str(cache_paths.pit))
        for i, task in enumerate(tasks, 1):
            _log_task_start(task, i, total)
            _render_progress_bar(i - 1, total, t0, submitted=i, active=1)
            if executor is None:
                rows.append(_run_one_task(task))
            else:
                rows.append(executor.submit(_run_one_task, task).result())
            _render_progress_bar(i, total, t0, submitted=i, active=0, final=i == total)
        return pd.DataFrame(rows)

    done = 0
    submitted = 0
    owns_executor = executor is None
    if executor is None:
        executor = ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=_init_worker,
            initargs=(str(cache_paths.data), str(cache_paths.pit)),
        )
    try:
        futures = set()
        future_meta = {}

        def _submit_until_full():
            nonlocal submitted
            while submitted < total and len(futures) < effective_workers:
                task = tasks[submitted]
                submitted += 1
                _log_task_start(task, submitted, total)
                future = executor.submit(_run_one_task, task)
                futures.add(future)
                future_meta[future] = submitted
                _render_progress_bar(
                    done,
                    total,
                    t0,
                    submitted=submitted,
                    active=len(futures),
                )

        _submit_until_full()
        while futures:
            finished, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                future_meta.pop(future, None)
                rows.append(future.result())
                done += 1
                _render_progress_bar(
                    done,
                    total,
                    t0,
                    submitted=submitted,
                    active=len(futures),
                    final=done == total,
                )
            _submit_until_full()
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        if owns_executor:
            executor.shutdown(wait=True, cancel_futures=True)
    return pd.DataFrame(rows)


def tally_champions(
    df: pd.DataFrame,
    *,
    objective: str = "sharpe",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    if df.empty or objective not in df.columns:
        return pd.DataFrame()
    ok = df[df["error"].isna()] if "error" in df.columns else df
    ok = ok[ok[objective].notna()]
    champions = []
    for (scheme, win_start, win_end), group in ok.groupby(["scheme", "win_start", "win_end"]):
        idx = group[objective].idxmax() if higher_is_better else group[objective].idxmin()
        best = group.loc[idx]
        champions.append({
            "scheme": scheme,
            "win_start": win_start,
            "win_end": win_end,
            "gid": best["gid"],
            objective: best[objective],
        })
    champion_df = pd.DataFrame(champions)
    if champion_df.empty:
        return champion_df
    return (
        champion_df.groupby("gid")
        .agg(
            wins_count=("gid", "size"),
            avg_champ_score=(objective, "mean"),
            champ_windows=("win_start", lambda s: list(s)),
        )
        .reset_index()
        .sort_values(["wins_count", "avg_champ_score"], ascending=[False, not higher_is_better])
        .reset_index(drop=True)
    )


def _sort_results(df: pd.DataFrame, objective: str, higher_is_better: bool) -> pd.DataFrame:
    if df.empty or objective not in df.columns:
        return df
    return df.sort_values(objective, ascending=not higher_is_better).reset_index(drop=True)


def run_parameter_sweep(config: ParameterSweepConfig) -> pd.DataFrame:
    output_dir = _as_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combos = build_grid_combos(
        config.grid,
        factor_module=config.factor_module,
        gid_factory=config.gid_factory,
    )
    if not combos:
        return pd.DataFrame()

    print(f"=== Parameter sweep (engine={config.engine}, workers={config.workers}) ===")
    print(f"grid combos={len(combos)}  span={config.span[0]}~{config.span[1]}")
    cache_paths = build_cache_for_config(config, [config.span], combos[0]["params"])
    tasks = build_sweep_tasks(config, combos)
    df = run_tasks(config, tasks, cache_paths)
    df = _sort_results(df, config.objective, config.objective_higher_is_better)
    df.to_csv(output_dir / config.results_filename, index=False, encoding="utf-8-sig")
    return df


def _run_paired_walk_forward(
    config: RollingMiningConfig,
    output_dir: Path,
    combos: Sequence[Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Run a true rolling train -> immediately-following-test schedule."""
    paired_schemes = list(config.paired_schemes or [])
    if not paired_schemes:
        raise ValueError("rolling_mode='paired' requires paired_schemes")
    rolling_span = config.rolling_span or (config.train[0], config.test[1])
    rolling_start, rolling_end = rolling_span

    cache_paths = build_cache_for_config(
        config,
        [rolling_span, config.valid],
        combos[0]["params"],
    )
    train_frames = []
    test_pairs: list[tuple[str, str, str, str, str]] = []
    print(f"\n[TRAIN->TEST] paired rolling schemes={paired_schemes}")
    for train_len, test_len, step in paired_schemes:
        scheme = f"paired/{train_len}/{test_len}/{step}"
        pairs = gen_rolling_train_test_windows(
            rolling_start,
            rolling_end,
            train_len,
            test_len,
            step,
            cap=config.max_windows_per_scheme,
        )
        if not pairs:
            print(f"  [{scheme}] no complete train/test pairs")
            continue
        test_pairs.extend((a, b, c, d, scheme) for a, b, c, d in pairs)
        tasks = build_paired_tasks(
            config=config,
            phase="train",
            pairs=pairs,
            scheme=scheme,
            combos=combos,
        )
        print(f"  [{scheme}] train windows={len(pairs)} tasks={len(tasks)}")
        train_frames.append(run_tasks(config, tasks, cache_paths))

    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    train_df.to_csv(output_dir / "train_results.csv", index=False, encoding="utf-8-sig")
    train_tally = tally_champions(
        train_df,
        objective=config.objective,
        higher_is_better=config.objective_higher_is_better,
    )
    train_tally.to_csv(output_dir / "train_champions.csv", index=False, encoding="utf-8-sig")
    if train_tally.empty:
        print("  [TRAIN] no valid result")
        return {"train_results": train_df, "train_champions": train_tally}

    counts = train_tally["wins_count"].values
    p_low, p_high = np.percentile(
        counts,
        [config.candidate_percentile[0] * 100, config.candidate_percentile[1] * 100],
    )
    candidates = train_tally[train_tally["wins_count"] >= p_low]
    candidates.to_csv(output_dir / "train_candidates.csv", index=False, encoding="utf-8-sig")
    print(
        f"  wins_count P{config.candidate_percentile[0] * 100:.0f}={p_low:.1f} "
        f"P{config.candidate_percentile[1] * 100:.0f}={p_high:.1f} candidates={len(candidates)}"
    )
    candidate_gids = set(candidates["gid"])
    candidate_combos = [combo for combo in combos if combo["gid"] in candidate_gids]

    test_frames = []
    for train_start, train_end, test_start, test_end, scheme in test_pairs:
        tasks = build_paired_tasks(
            config=config,
            phase="test",
            pairs=[(train_start, train_end, test_start, test_end)],
            scheme=scheme,
            combos=candidate_combos,
        )
        test_frames.append(run_tasks(config, tasks, cache_paths))
    test_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    test_df.to_csv(output_dir / "test_results.csv", index=False, encoding="utf-8-sig")
    test_tally = tally_champions(
        test_df,
        objective=config.objective,
        higher_is_better=config.objective_higher_is_better,
    )
    test_tally.to_csv(output_dir / "test_champions.csv", index=False, encoding="utf-8-sig")
    if test_tally.empty:
        print("  [TEST] no valid result")
        return {
            "train_results": train_df,
            "train_champions": train_tally,
            "train_candidates": candidates,
            "test_results": test_df,
            "test_champions": test_tally,
        }

    top = test_tally.head(config.report_top_n)
    print(f"  [TEST] top {len(top)}:")
    print(top.to_string(index=False))
    combo_by_gid = {combo["gid"]: combo for combo in combos}

    valid_tasks = []
    for rank, gid in enumerate(top["gid"].tolist(), 1):
        combo = combo_by_gid[gid]
        params = dict(combo["params"])
        valid_tasks.append({
            "params": params,
            "gid": combo["gid"],
            "rank": rank,
            "factor_module": config.factor_module,
            "spec_factory": config.spec_factory,
            "weight_hook": config.weight_hook,
            "warmup_days": _warmup_days(config, params),
            "phase": "valid",
            "scheme": "full",
            "win_start": config.valid[0],
            "win_end": config.valid[1],
            "engine": config.engine,
            "rebal_freq": config.rebal_freq,
            "n_quantiles_param": config.n_quantiles_param,
            "initial_amount": config.initial_amount,
            "time_tolerance": config.time_tolerance,
        })
    valid_df = run_tasks(config, valid_tasks, cache_paths)
    if not valid_df.empty and "rank" in valid_df.columns:
        valid_df = valid_df.sort_values("rank").reset_index(drop=True)
    valid_df.to_csv(output_dir / "valid_results.csv", index=False, encoding="utf-8-sig")

    if config.valid_report_hook:
        for rank, gid in enumerate(top["gid"].tolist(), 1):
            try:
                _call_module_function(
                    config.factor_module,
                    config.valid_report_hook,
                    combo_by_gid[gid]["params"],
                    rank,
                    str(output_dir),
                    config.valid[0],
                    config.valid[1],
                )
            except Exception as exc:
                print(f"  [valid report] #{rank} failed: {exc}")

    return {
        "train_results": train_df,
        "train_champions": train_tally,
        "train_candidates": candidates,
        "test_results": test_df,
        "test_champions": test_tally,
        "valid_results": valid_df,
    }


def run_walk_forward(config: RollingMiningConfig) -> dict[str, pd.DataFrame]:
    output_dir = _as_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.sampler:
        from betalens.factor.mining_optuna import run_optuna_walk_forward

        return run_optuna_walk_forward(config)
    combos = build_grid_combos(
        config.grid,
        factor_module=config.factor_module,
        gid_factory=config.gid_factory,
    )
    if not combos:
        return {}

    if config.rolling_mode == "paired":
        return _run_paired_walk_forward(config, _as_path(config.output_dir), combos)
    if config.rolling_mode != "split":
        raise ValueError(f"unknown rolling_mode: {config.rolling_mode!r}")

    print(
        f"=== Walk-forward mining (engine={config.engine}, workers={config.workers}) ==="
    )
    print(
        f"grid combos={len(combos)}  train schemes={list(config.train_schemes)}  "
        f"test schemes={list(config.test_schemes)}"
    )
    cache_paths = build_cache_for_config(
        config,
        [config.train, config.test, config.valid],
        combos[0]["params"],
    )

    print(f"\n[TRAIN] {config.train[0]}~{config.train[1]}")
    train_tasks = build_tasks(
        config=config,
        phase="train",
        span=config.train,
        schemes=config.train_schemes,
        combos=combos,
        cap=config.max_windows_per_scheme,
    )
    print(f"  tasks={len(train_tasks)}")
    train_df = run_tasks(config, train_tasks, cache_paths)
    train_df.to_csv(output_dir / "train_results.csv", index=False, encoding="utf-8-sig")
    train_tally = tally_champions(
        train_df,
        objective=config.objective,
        higher_is_better=config.objective_higher_is_better,
    )
    train_tally.to_csv(output_dir / "train_champions.csv", index=False, encoding="utf-8-sig")
    if train_tally.empty:
        print("  [TRAIN] no valid result")
        return {"train_results": train_df, "train_champions": train_tally}

    counts = train_tally["wins_count"].values
    p_low, p_high = np.percentile(
        counts,
        [config.candidate_percentile[0] * 100, config.candidate_percentile[1] * 100],
    )
    candidates = train_tally[train_tally["wins_count"] >= p_low]
    candidates.to_csv(output_dir / "train_candidates.csv", index=False, encoding="utf-8-sig")
    print(
        f"  wins_count P{config.candidate_percentile[0] * 100:.0f}={p_low:.1f} "
        f"P{config.candidate_percentile[1] * 100:.0f}={p_high:.1f}  "
        f"candidates={len(candidates)}"
    )
    print(train_tally.head(10).to_string(index=False))

    candidate_gids = set(candidates["gid"])
    candidate_combos = [combo for combo in combos if combo["gid"] in candidate_gids]

    print(f"\n[TEST] {config.test[0]}~{config.test[1]}  candidates={len(candidate_combos)}")
    test_tasks = build_tasks(
        config=config,
        phase="test",
        span=config.test,
        schemes=config.test_schemes,
        combos=candidate_combos,
        cap=config.max_windows_per_scheme,
    )
    print(f"  tasks={len(test_tasks)}")
    test_df = run_tasks(config, test_tasks, cache_paths)
    test_df.to_csv(output_dir / "test_results.csv", index=False, encoding="utf-8-sig")
    test_tally = tally_champions(
        test_df,
        objective=config.objective,
        higher_is_better=config.objective_higher_is_better,
    )
    test_tally.to_csv(output_dir / "test_champions.csv", index=False, encoding="utf-8-sig")
    if test_tally.empty:
        print("  [TEST] no valid result")
        return {
            "train_results": train_df,
            "train_champions": train_tally,
            "test_results": test_df,
            "test_champions": test_tally,
        }
    top = test_tally.head(config.report_top_n)
    print(f"  TEST top {len(top)}:")
    print(top.to_string(index=False))

    print(f"\n[VALID] {config.valid[0]}~{config.valid[1]}")
    combo_by_gid = {combo["gid"]: combo for combo in combos}
    valid_tasks = []
    for rank, gid in enumerate(top["gid"].tolist(), 1):
        combo = combo_by_gid[gid]
        params = dict(combo["params"])
        valid_tasks.append({
            "params": params,
            "gid": combo["gid"],
            "rank": rank,
            "factor_module": config.factor_module,
            "spec_factory": config.spec_factory,
            "weight_hook": config.weight_hook,
            "warmup_days": _warmup_days(config, params),
            "phase": "valid",
            "scheme": "full",
            "win_start": config.valid[0],
            "win_end": config.valid[1],
            "engine": config.engine,
            "rebal_freq": config.rebal_freq,
            "n_quantiles_param": config.n_quantiles_param,
            "initial_amount": config.initial_amount,
            "time_tolerance": config.time_tolerance,
        })

    valid_df = run_tasks(config, valid_tasks, cache_paths)
    if not valid_df.empty and "rank" in valid_df.columns:
        valid_df = valid_df.sort_values("rank").reset_index(drop=True)
    valid_df.to_csv(output_dir / "valid_results.csv", index=False, encoding="utf-8-sig")

    if config.valid_report_hook:
        for rank, gid in enumerate(top["gid"].tolist(), 1):
            params = combo_by_gid[gid]["params"]
            try:
                _call_module_function(
                    config.factor_module,
                    config.valid_report_hook,
                    params,
                    rank,
                    str(output_dir),
                    config.valid[0],
                    config.valid[1],
                )
            except Exception as exc:
                print(f"  [valid report] #{rank} failed: {exc}")

    if not valid_df.empty:
        best = valid_df.iloc[0]
        print(f"\n=== best gid={best['gid']}  valid {config.objective}={best.get(config.objective)} ===")
    print(f"output dir: {output_dir}")
    return {
        "train_results": train_df,
        "train_champions": train_tally,
        "train_candidates": candidates,
        "test_results": test_df,
        "test_champions": test_tally,
        "valid_results": valid_df,
    }
