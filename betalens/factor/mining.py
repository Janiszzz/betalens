from __future__ import annotations

import importlib
import itertools
import json
import os
import pickle
import shutil
import time
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
    "metrics_from_nav",
    "run_parameter_sweep",
    "run_walk_forward",
    "tally_champions",
]


DEFAULT_WORKERS = min(10, max(1, (os.cpu_count() or 4) - 2))
DEFAULT_MAX_MEMORY_RATIO = 0.50
DEFAULT_CACHE_MEMORY_MULTIPLIER = 3.0
DEFAULT_WORKER_MEMORY_OVERHEAD_BYTES = 512 * 1024 * 1024


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
    universe: Sequence[str] | None = None


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


def _system_memory_snapshot() -> tuple[int | None, int | None]:
    """Return (total_physical_bytes, available_physical_bytes)."""
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
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
        except Exception:
            return None, None
    else:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total_pages = int(os.sysconf("SC_PHYS_PAGES"))
            avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            return page_size * total_pages, page_size * avail_pages
        except (AttributeError, OSError, ValueError):
            return None, None
    return None, None


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
    if cache_paths.meta.exists():
        try:
            with open(cache_paths.meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            payload = int(meta.get("cache_payload_memory_bytes", 0))
            pickle_size = int(meta.get("pickle_bytes", 0))
            return max(payload, pickle_size)
        except Exception:
            pass
    size = 0
    for path in (cache_paths.data, cache_paths.pit):
        if path.exists():
            size += path.stat().st_size
    return int(size)


def _estimate_worker_memory_bytes(config: Any, cache_paths: CachePaths) -> int:
    cache_bytes = _read_cache_memory_estimate(cache_paths)
    multiplier = float(getattr(config, "cache_memory_multiplier", DEFAULT_CACHE_MEMORY_MULTIPLIER))
    overhead = int(getattr(config, "worker_memory_overhead_bytes", DEFAULT_WORKER_MEMORY_OVERHEAD_BYTES))
    return max(1, int(cache_bytes * max(multiplier, 1.0) + max(overhead, 0)))


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

    days = sorted(get_absolute_trade_days(start, end, "D", use_pmc=False))
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
    from betalens.datafeed import Datafeed, get_index_universe

    data = Datafeed(table_name)
    pit = {}
    try:
        for d in signal_dates:
            date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
            pit[d] = set(get_index_universe(data.cursor, index_code, date_str))
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
    industry_panel = query_industry_panel(data, scheme=industry_scheme, industry_table="industry", verbose=False) \
        if industry_scheme else None

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
    with open(cache_data_path, "rb") as f:
        _CACHE_DATA = pickle.load(f)
    with open(pit_path, "rb") as f:
        _PIT_UNIVERSE = pickle.load(f)


def _task_signal_dates(start: str, end: str, fetch_start: str, rebal_freq: str):
    from betalens.datafeed import get_absolute_trade_days

    rebalance_dates = get_absolute_trade_days(start, end, rebal_freq, use_pmc=False)
    all_trade_days = sorted(get_absolute_trade_days(fetch_start, end, "D", use_pmc=False))
    idx = {d: i for i, d in enumerate(all_trade_days)}
    return [all_trade_days[idx[d] - 1] for d in rebalance_dates if idx.get(d, 0) > 0]


def _run_one_task(task: dict[str, Any]) -> dict[str, Any]:
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
    if "rank" in task:
        out["rank"] = task["rank"]

    try:
        spec = _load_spec(task["factor_module"], task["spec_factory"], params)
        start, end = task["win_start"], task["win_end"]
        fetch_start = (
            pd.Timestamp(start) - pd.Timedelta(days=int(task["warmup_days"]))
        ).strftime("%Y-%m-%d")
        signal_dates = _task_signal_dates(start, end, fetch_start, task["rebal_freq"])
        if not signal_dates:
            out["error"] = "no signal dates"
            return out

        wides = {}
        for arg_name in spec.inputs:
            wide = _CACHE_DATA["inputs"][arg_name]
            wides[arg_name] = wide.loc[
                (wide.index >= pd.Timestamp(fetch_start)) &
                (wide.index <= pd.Timestamp(end) + pd.Timedelta(days=1))
            ]

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

            prequery = _preprocess_if_needed(prequery, spec, signal_dates, universe, fetch_start, end)
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
            weights.index = weights.index + pd.Timedelta(minutes=10)

        if task.get("weight_hook"):
            hook_task = dict(task)
            hook_task["context"] = {
                "factor_wide": factor_wide,
                "input_wides": wides,
                "price_wide": _CACHE_DATA["price"],
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
        if weights.empty:
            out["error"] = "empty weights"
            return out

        if task["engine"] == "vector":
            nav = _vector_backtest(weights, _CACHE_DATA["price"])
        else:
            bt = BacktestBase(
                weights,
                metric=getattr(spec, "backtest_metric", "收盘价(元)"),
                symbol=spec.name,
                amount=task["initial_amount"],
                time_tolerance=task["time_tolerance"],
                verbose=False,
            )
            nav = bt.nav
        out.update(metrics_from_nav(nav))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _cache_paths(cache_dir: str | Path) -> CachePaths:
    cache = _as_path(cache_dir)
    return CachePaths(
        data=cache / "mining_cache.pkl",
        pit=cache / "pit_universe.pkl",
        meta=cache / "mining_cache_meta.json",
    )


def _cache_dir(config: Any) -> Path:
    if config.cache_dir is not None:
        return _as_path(config.cache_dir)
    return _as_path(config.output_dir) / "_cache"


def build_cache_for_config(config: Any, spans: Sequence[tuple[str, str]], sample_params: Mapping[str, Any]) -> CachePaths:
    from betalens.datafeed import get_absolute_trade_days

    cache_dir = _cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(cache_dir)
    if paths.data.exists() and paths.pit.exists() and not config.rebuild_cache:
        print(f"[cache] hit: {paths.data.name}, {paths.pit.name}")
        return paths

    spec = _load_spec(config.factor_module, config.spec_factory, sample_params)
    start = _date_min(*spans)
    end = _date_max(*spans)
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=int(config.max_warmup_days))).strftime("%Y-%m-%d")
    print(f"[cache] fetch range: {fetch_start} ~ {end}")

    all_days = sorted(get_absolute_trade_days(fetch_start, end, "D", use_pmc=False))
    pit_days = [d for d in all_days if d >= pd.Timestamp(start).date()]

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

    inputs = {}
    metrics_by_arg = dict(getattr(spec, "inputs", {}))
    for arg_name, metric in metrics_by_arg.items():
        t0 = time.time()
        wide = fetch_daily_wide(
            metric,
            universe=universe,
            start_date=fetch_start,
            end_date=end,
            table_name=getattr(spec, "table_name", "daily_market"),
        )
        inputs[arg_name] = wide
        print(f"[cache] {arg_name} ({metric}): {wide.shape}, {time.time() - t0:.1f}s")

    price_metric = getattr(spec, "backtest_metric", "收盘价(元)")
    price = None
    for arg_name, metric in metrics_by_arg.items():
        if metric == price_metric:
            price = inputs[arg_name]
            break
    if price is None:
        t0 = time.time()
        price = fetch_daily_wide(
            price_metric,
            universe=universe,
            start_date=fetch_start,
            end_date=end,
            table_name=getattr(spec, "table_name", "daily_market"),
        )
        print(f"[cache] price ({price_metric}): {price.shape}, {time.time() - t0:.1f}s")

    with open(paths.data, "wb") as f:
        pickle.dump({"inputs": inputs, "price": price, "universe": universe}, f)
    with open(paths.pit, "wb") as f:
        pickle.dump(pit, f)
    meta = {
        "cache_payload_memory_bytes": int(
            _cache_payload_memory_bytes(inputs, price) + _pit_memory_estimate_bytes(pit)
        ),
        "pickle_bytes": int(paths.data.stat().st_size + paths.pit.stat().st_size),
        "input_shapes": {name: list(wide.shape) for name, wide in inputs.items()},
        "price_shape": list(price.shape) if isinstance(price, pd.DataFrame) else None,
        "universe_size": len(universe or []),
    }
    with open(paths.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[cache] saved: {cache_dir}")
    return paths


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


def run_tasks(config: Any, tasks: Sequence[dict[str, Any]], cache_paths: CachePaths) -> pd.DataFrame:
    rows = []
    total = len(tasks)
    if total == 0:
        return pd.DataFrame()

    effective_workers = _effective_workers_for_memory(config, cache_paths)
    t0 = time.time()
    _render_progress_bar(0, total, t0, submitted=0, active=0)
    if effective_workers <= 1:
        _init_worker(str(cache_paths.data), str(cache_paths.pit))
        for i, task in enumerate(tasks, 1):
            _log_task_start(task, i, total)
            _render_progress_bar(i - 1, total, t0, submitted=i, active=1)
            rows.append(_run_one_task(task))
            _render_progress_bar(i, total, t0, submitted=i, active=0, final=i == total)
        return pd.DataFrame(rows)

    done = 0
    submitted = 0
    with ProcessPoolExecutor(
        max_workers=effective_workers,
        initializer=_init_worker,
        initargs=(str(cache_paths.data), str(cache_paths.pit)),
    ) as executor:
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


def run_walk_forward(config: RollingMiningConfig) -> dict[str, pd.DataFrame]:
    output_dir = _as_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combos = build_grid_combos(
        config.grid,
        factor_module=config.factor_module,
        gid_factory=config.gid_factory,
    )
    if not combos:
        return {}

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
