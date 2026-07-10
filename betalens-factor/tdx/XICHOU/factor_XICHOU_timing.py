#%%
"""XICHOU single-stock timing factor."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

_FACTOR_DIR = Path(__file__).resolve().parent
_CLASS_DIR = _FACTOR_DIR.parent
_FACTOR_ROOT = _CLASS_DIR.parent
_REPO_ROOT = _FACTOR_ROOT.parent
for _path in (_REPO_ROOT, _FACTOR_ROOT, _CLASS_DIR, _FACTOR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from betalens.factor.config import (  # noqa: E402
    factor_spec_options,
    load_yaml_config,
    run_parameters,
    section,
)
from factor_template import RunResult, infer_warmup_days  # noqa: E402
from factor_template_tdx import FactorSpec  # noqa: E402
from factor_XICHOU import compute_xichou  # noqa: E402


_CONFIG_FILE = _FACTOR_DIR / "factor_XICHOU_timing.yaml"
_REQUIRED_SECTIONS = ("meta", "factor_spec", "weight", "run")
_FACTOR_VALUE_COLUMNS = ["信号日", "股票代码", "因子值", "分组", "目标仓位", "是否触发"]


def load_config(path: str | Path = _CONFIG_FILE) -> dict:
    return load_yaml_config(path, required_sections=_REQUIRED_SECTIONS)


def _param(params: Mapping[str, Any], key: str, default: Any) -> Any:
    return params[key] if key in params and params[key] is not None else default


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _formula_kwargs(params: Mapping[str, Any]) -> dict[str, int]:
    return {
        "sma_n": int(_param(params, "sma_n", 3)),
        "ema_n": int(_param(params, "ema_n", 3)),
        "llv_n": int(_param(params, "llv_n", 15)),
    }


def compute_xichou_timing(low_wide, high_wide=None, **kwargs):
    del high_wide
    return compute_xichou(low_wide, **_formula_kwargs(kwargs))


def build_spec(config: dict, config_path: str | Path = _CONFIG_FILE) -> FactorSpec:
    options = factor_spec_options(config, config_path)
    return FactorSpec(
        name=str(section(config, "meta")["name"]),
        compute=compute_xichou_timing,
        **options,
    )


spec = build_spec(load_config())


def _resolve_trigger_operator(direction: str, trigger_operator: str | None) -> str:
    op = str(trigger_operator or "auto").strip().lower()
    aliases = {
        "auto": "gt" if direction == "positive" else "lt",
        "gt": "gt",
        ">": "gt",
        "above": "gt",
        "greater": "gt",
        "lt": "lt",
        "<": "lt",
        "below": "lt",
        "less": "lt",
    }
    if op not in aliases:
        raise ValueError(f"trigger_operator supports auto/gt/lt only: {trigger_operator}")
    return aliases[op]


def _event_active(values: pd.Series | pd.DataFrame, threshold: float, operator: str):
    active = values > threshold if operator == "gt" else values < threshold
    return active.fillna(False).astype(bool)


def _daily_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").copy()
    out.index = pd.to_datetime(out.index).normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or abs(float(denominator)) < 1e-12:
        return np.nan
    return float(numerator) / float(denominator)


def _slope(values) -> float:
    y = pd.Series(values, dtype="float64").dropna().to_numpy()
    if len(y) < 2:
        return np.nan
    x = np.arange(len(y), dtype="float64")
    return float(np.polyfit(x, y, 1)[0])


def _trend_label(slope: float, net_change: float, eps: float = 1e-10) -> str:
    if pd.isna(slope):
        return "single"
    if slope > eps:
        return "up"
    if slope < -eps:
        return "down"
    if abs(float(net_change)) <= eps:
        return "flat"
    return "mixed"


def _event_table_for_code(
    factor: pd.Series,
    high: pd.Series,
    threshold: float,
    operator: str,
) -> pd.DataFrame:
    factor = _daily_series(factor)
    high = _daily_series(high).reindex(factor.index)
    active = _event_active(factor, threshold, operator)
    event_ids = (active & ~active.shift(1, fill_value=False)).cumsum().where(active)
    rows = []

    for event_id in event_ids.dropna().unique():
        dates = event_ids[event_ids == event_id].index
        values = factor.loc[dates].dropna().astype(float)
        if values.empty:
            continue
        start_date = pd.Timestamp(dates[0])
        end_date = pd.Timestamp(dates[-1])
        start_value = float(values.iloc[0])
        end_value = float(values.iloc[-1])
        max_date = pd.Timestamp(values.idxmax())
        min_date = pd.Timestamp(values.idxmin())
        max_value = float(values.loc[max_date])
        min_value = float(values.loc[min_date])
        net_change = end_value - start_value
        slope = _slope(values)
        trigger_high = high.loc[dates].max()
        post_dates = factor.index[factor.index > end_date]
        post_high = high.reindex(post_dates)
        hit = post_high[post_high > trigger_high].dropna().head(1)
        if hit.empty:
            new_high_date = pd.NaT
            wait_days = np.nan
        else:
            new_high_date = pd.Timestamp(hit.index[0])
            wait_days = int(post_dates.get_loc(new_high_date) + 1)
        rows.append(
            {
                "event_id": int(event_id),
                "start_date": start_date,
                "end_date": end_date,
                "duration": int(len(dates)),
                "factor_sum": float(values.sum()),
                "start_value": start_value,
                "end_value": end_value,
                "max_value": max_value,
                "max_date": max_date,
                "min_value": min_value,
                "min_date": min_date,
                "amplitude": max_value - min_value,
                "relative_amplitude": _safe_div(max_value - min_value, start_value),
                "net_change": net_change,
                "change_rate": _safe_div(net_change, start_value),
                "daily_slope": slope,
                "trend": _trend_label(slope, net_change),
                "up_days": int((values.diff() > 0).sum()),
                "down_days": int((values.diff() < 0).sum()),
                "up_day_ratio": _safe_div(float((values.diff() > 0).sum()), max(len(values) - 1, 1)),
                "mean_value": float(values.mean()),
                "median_value": float(values.median()),
                "trigger_high": float(trigger_high) if pd.notna(trigger_high) else np.nan,
                "new_high_date": new_high_date,
                "wait_days_to_new_high": wait_days,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "event_id",
                "start_date",
                "end_date",
                "duration",
                "factor_sum",
                "trigger_high",
                "new_high_date",
                "wait_days_to_new_high",
            ]
        )
    return pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)


def _history_before(events: pd.DataFrame, event_start: pd.Timestamp, window: int) -> pd.DataFrame:
    previous = events[events["end_date"] < event_start].sort_values("end_date")
    return previous.tail(int(window))


def _wait_history_before(events: pd.DataFrame, event_start: pd.Timestamp, window: int) -> pd.DataFrame:
    previous = events[
        (events["end_date"] < event_start)
        & events["new_high_date"].notna()
        & (events["new_high_date"] < event_start)
        & events["wait_days_to_new_high"].notna()
    ].sort_values("end_date")
    return previous.tail(int(window))


def _dynamic_event_weight_for_code(
    factor: pd.Series,
    events: pd.DataFrame,
    params: Mapping[str, Any],
) -> pd.Series:
    factor = _daily_series(factor)
    target = pd.Series(0.0, index=factor.index)
    if events.empty:
        return target

    history_window = int(_param(params, "history_window", 10))
    duration_q = min(max(float(_param(params, "duration_quantile", 0.65)), 0.0), 1.0)
    exit_q = min(max(float(_param(params, "exit_wait_quantile", 0.75)), 0.0), 1.0)
    min_history_events = int(_param(params, "min_history_events", 3))
    default_exit_wait_days = int(_param(params, "default_exit_wait_days", 5))
    max_weight = float(_param(params, "max_weight", 1.0))

    for event in events.itertuples(index=False):
        event_start = pd.Timestamp(event.start_date)
        event_end = pd.Timestamp(event.end_date)
        history = _history_before(events, event_start, history_window)
        if len(history) < min_history_events:
            continue

        factor_sum_level = pd.to_numeric(history["factor_sum"], errors="coerce").dropna().mean()
        if not np.isfinite(factor_sum_level) or abs(float(factor_sum_level)) <= 1e-12:
            continue

        open_day = int(math.ceil(history["duration"].quantile(duration_q)))
        open_day = max(open_day, 1)
        wait_history = _wait_history_before(events, event_start, history_window)
        if wait_history.empty:
            exit_wait_days = default_exit_wait_days
        else:
            exit_wait_days = int(math.ceil(wait_history["wait_days_to_new_high"].quantile(exit_q)))
        exit_wait_days = max(exit_wait_days, 0)

        event_values = factor.loc[(factor.index >= event_start) & (factor.index <= event_end)].dropna()
        cumulative = 0.0
        last_weight = 0.0
        for day_number, (date, value) in enumerate(event_values.items(), start=1):
            cumulative += float(value)
            if day_number < open_day:
                continue
            last_weight = min(max_weight, max(0.0, cumulative / factor_sum_level))
            target.loc[date] = max(target.loc[date], last_weight)

        if last_weight <= 0 or event_end not in target.index:
            continue
        end_pos = target.index.get_loc(event_end)
        hold_dates = target.index[end_pos + 1:end_pos + 1 + exit_wait_days]
        if len(hold_dates):
            target.loc[hold_dates] = np.maximum(target.loc[hold_dates], last_weight)

    return target.clip(lower=0.0, upper=max_weight)


def _fetch_daily_wide(
    metric: str,
    *,
    universe: list[str],
    start_date: str,
    end_date: str,
    table_name: str,
) -> pd.DataFrame:
    from betalens.datafeed import Datafeed

    data = Datafeed(table_name)
    try:
        raw = data.query_time_range(
            codes=universe,
            start_date=start_date,
            end_date=end_date,
            metric=metric,
        )
    finally:
        data.close()
    if raw.empty:
        return pd.DataFrame()
    raw = raw.copy()
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    return (
        raw.pivot_table(index="datetime", columns="code", values="value", aggfunc="first")
        .sort_index()
        .sort_index(axis=1)
    )


def _resolve_codes(params: Mapping[str, Any], universe: list | None) -> list[str]:
    stock_code = _optional_text(params.get("stock_code"))
    if stock_code:
        return [stock_code]
    if universe:
        return [str(code) for code in universe if _optional_text(code)]
    raise ValueError("XICHOU_timing requires compute_kwargs.stock_code or run.universe")


def _signal_dates(start_date: str, end_date: str, fetch_start: str, rebal_freq: str) -> list[pd.Timestamp]:
    from betalens.datafeed import get_absolute_trade_days

    rebalance_dates = get_absolute_trade_days(start_date, end_date, rebal_freq, use_pmc=False)
    all_trade_days = sorted(get_absolute_trade_days(fetch_start, end_date, "D", use_pmc=False))
    day_index = {day: i for i, day in enumerate(all_trade_days)}
    dates = [all_trade_days[day_index[day] - 1] for day in rebalance_dates if day_index.get(day, 0) > 0]
    return [pd.Timestamp(day).normalize() for day in dates]


def _build_weights(
    *,
    factor_wide: pd.DataFrame,
    high_wide: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    codes: list[str],
    direction: str,
    params: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    signal_index = pd.DatetimeIndex(signal_dates)
    operator = _resolve_trigger_operator(direction, str(_param(params, "trigger_operator", "auto")))
    threshold = float(_param(params, "trigger_threshold", 0.1))

    weights = pd.DataFrame(0.0, index=signal_index, columns=codes)
    events_by_code: dict[str, pd.DataFrame] = {}
    for code in codes:
        if code not in factor_wide.columns:
            events_by_code[code] = pd.DataFrame()
            continue
        high = high_wide[code] if code in high_wide.columns else pd.Series(index=factor_wide.index, dtype=float)
        events = _event_table_for_code(
            factor=factor_wide[code],
            high=high,
            threshold=threshold,
            operator=operator,
        )
        events_by_code[code] = events
        target = _dynamic_event_weight_for_code(factor_wide[code], events, params)
        weights[code] = target.reindex(signal_index).fillna(0.0).astype(float)

    stock_sum = weights.clip(lower=0.0).sum(axis=1)
    scale = pd.Series(1.0, index=weights.index)
    scale.loc[stock_sum > 1.0] = 1.0 / stock_sum.loc[stock_sum > 1.0]
    weights = weights.mul(scale, axis=0)
    weights["cash"] = (1.0 - weights.sum(axis=1)).clip(lower=0.0, upper=1.0)
    weights.index = weights.index + pd.Timedelta(minutes=10)
    return weights.fillna(0.0), events_by_code


def _build_factor_values(
    *,
    factor_wide: pd.DataFrame,
    weights: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    codes: list[str],
    direction: str,
    params: Mapping[str, Any],
) -> pd.DataFrame:
    signal_index = pd.DatetimeIndex(signal_dates)
    weight_by_day = weights.copy()
    weight_by_day.index = pd.DatetimeIndex(weight_by_day.index).normalize()
    operator = _resolve_trigger_operator(direction, str(_param(params, "trigger_operator", "auto")))
    threshold = float(_param(params, "trigger_threshold", 0.1))

    rows: list[dict[str, Any]] = []
    for code in codes:
        if code not in factor_wide.columns:
            continue
        factor = _daily_series(factor_wide[code]).reindex(signal_index)
        active = _event_active(factor, threshold, operator)
        code_weight = (
            weight_by_day[code].reindex(signal_index).fillna(0.0)
            if code in weight_by_day.columns
            else pd.Series(0.0, index=signal_index)
        )
        for date, value in factor.items():
            if pd.isna(value):
                continue
            target_weight = float(code_weight.loc[date])
            rows.append(
                {
                    "信号日": pd.Timestamp(date),
                    "股票代码": code,
                    "因子值": float(value),
                    "分组": 1 if target_weight > 1e-8 else 0,
                    "目标仓位": target_weight,
                    "是否触发": bool(active.loc[date]),
                }
            )
    if not rows:
        return pd.DataFrame(columns=_FACTOR_VALUE_COLUMNS)
    return pd.DataFrame(rows, columns=_FACTOR_VALUE_COLUMNS).sort_values(["信号日", "股票代码"]).reset_index(drop=True)


def _write_timing_artifacts(
    output_dir: Path,
    name: str,
    weights: pd.DataFrame,
    events_by_code: dict[str, pd.DataFrame],
    factor_values: pd.DataFrame,
) -> None:
    weights.to_csv(output_dir / f"{name}_weights.csv", encoding="utf-8-sig")
    factor_values.to_csv(output_dir / f"{name}_factor_values.csv", index=False, encoding="utf-8-sig")
    frames = []
    for code, events in events_by_code.items():
        if events.empty:
            continue
        event_df = events.copy()
        event_df.insert(0, "code", code)
        frames.append(event_df)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            output_dir / f"{name}_trigger_events.csv",
            index=False,
            encoding="utf-8-sig",
        )


class FactorPipeline:
    def __init__(self, spec: FactorSpec):
        self.spec = spec

    def run(
        self,
        start_date: str,
        end_date: str,
        *,
        rebal_freq: str = "D",
        universe: list | None = None,
        n_quantiles: int = 10,
        initial_amount: float = 1e8,
        benchmark_code: str | None = None,
        output_dir: str = ".",
        include_profiling: bool = False,
        dump_excel: bool = True,
        warmup_days: int | None = None,
        verbose: bool = True,
    ) -> RunResult:
        del n_quantiles, include_profiling
        from betalens.analyst import Analyst
        from betalens.backtest import BacktestBase

        sp = self.spec
        params = dict(sp.compute_kwargs or {})
        codes = _resolve_codes(params, universe)
        formula_params = _formula_kwargs(params)
        inferred_warmup = infer_warmup_days(formula_params, minimum=90)
        history_warmup = int(_param(params, "history_window", 10)) * 30 + 60
        warmup = int(warmup_days if warmup_days is not None else max(inferred_warmup, history_warmup))
        fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=warmup)).strftime("%Y-%m-%d")
        fetch_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        signals = _signal_dates(start_date, end_date, fetch_start, rebal_freq)
        if not signals:
            raise ValueError(f"no signal dates for {start_date}~{end_date}, rebal_freq={rebal_freq}")

        if verbose:
            print(
                f"XICHOU_timing: codes={codes} fetch={fetch_start}~{fetch_end} "
                f"signals={len(signals)} warmup_days={warmup}",
                flush=True,
            )

        input_wides: dict[str, pd.DataFrame] = {}
        for arg_name, metric in sp.inputs.items():
            if verbose:
                print(f"  fetch {arg_name}: {metric}", flush=True)
            wide = _fetch_daily_wide(
                metric=str(metric),
                universe=codes,
                start_date=fetch_start,
                end_date=fetch_end,
                table_name=sp.table_name,
            )
            if wide.empty:
                raise ValueError(f"empty input data: {arg_name} ({metric})")
            input_wides[arg_name] = wide

        low_wide = input_wides["low_wide"]
        high_wide = input_wides.get("high_wide", pd.DataFrame(index=low_wide.index))
        if verbose:
            print(f"  compute factor: {formula_params}", flush=True)
        factor_wide = sp.compute(**input_wides, **params)
        if factor_wide.empty:
            raise ValueError("empty XICHOU factor values")

        weights, events_by_code = _build_weights(
            factor_wide=factor_wide,
            high_wide=high_wide,
            signal_dates=signals,
            codes=codes,
            direction=sp.direction,
            params=params,
        )
        if weights.empty:
            raise ValueError("empty timing weights")

        factor_values = _build_factor_values(
            factor_wide=factor_wide,
            weights=weights,
            signal_dates=signals,
            codes=codes,
            direction=sp.direction,
            params=params,
        )

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_timing_artifacts(out_dir, sp.name, weights, events_by_code, factor_values)

        if verbose:
            nonzero_days = int((weights.drop(columns=["cash"], errors="ignore").abs().sum(axis=1) > 1e-8).sum())
            print(f"  weights: {weights.shape}, nonzero_days={nonzero_days}", flush=True)
            print("  run backtest", flush=True)
        bt = BacktestBase(
            weights,
            symbol=sp.name,
            amount=initial_amount,
            metric=sp.backtest_metric,
            table_name=sp.table_name,
            time_tolerance=24 * 11,
            verbose=verbose,
        )

        if verbose:
            print("  build analyst report", flush=True)
        analyst = Analyst.from_backtest(
            bt,
            name=sp.name,
            benchmark_code=benchmark_code,
            benchmark_metric=sp.backtest_metric,
            factor_values=factor_values,
        )
        analyst.report(
            to_excel=str(out_dir / f"{sp.name}_report.xlsx"),
            to_html=str(out_dir / f"{sp.name}_report.html"),
        )

        if dump_excel:
            dump_path = out_dir / f"{sp.name}_dump.xlsx"
            bt.dump_to_excel(str(dump_path))
            with pd.ExcelWriter(dump_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                factor_values.to_excel(writer, sheet_name="factor_values", index=False)
                for code, events in events_by_code.items():
                    if not events.empty:
                        events.to_excel(writer, sheet_name=f"events_{code}"[:31], index=False)

        return RunResult(
            backtest=bt,
            analyst=analyst,
            profiling=None,
            neutralize_stats=None,
            factor_values=factor_values,
            pit_validation=None,
        )


def run_from_config(config_path: str | Path = _CONFIG_FILE):
    config = load_config(config_path)
    kwargs = run_parameters(config, config_path)
    start_date = kwargs.pop("start_date")
    end_date = kwargs.pop("end_date")
    Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
    return FactorPipeline(build_spec(config, config_path)).run(start_date, end_date, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XICHOU timing factor from its YAML parameter file.")
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="YAML parameter file")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
