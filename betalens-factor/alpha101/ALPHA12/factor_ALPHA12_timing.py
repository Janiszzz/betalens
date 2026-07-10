"""ALPHA12 single-stock timing factor."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

_FACTOR_DIR = Path(__file__).resolve().parent
_CLASS_DIR = _FACTOR_DIR.parent
_FACTOR_ROOT = _CLASS_DIR.parent
_REPO_ROOT = _FACTOR_ROOT.parent
for _path in (_REPO_ROOT, _FACTOR_ROOT, _CLASS_DIR, _FACTOR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from betalens.factor.config import factor_spec_options, load_yaml_config, run_parameters, section  # noqa: E402
from betalens.factor.signal import infer_signal_warmup, rolling_z_weight  # noqa: E402
from factor_template import FactorSpec, RunResult, infer_warmup_days  # noqa: E402
from factor_ALPHA12 import compute_alpha12  # noqa: E402


_CONFIG_FILE = _FACTOR_DIR / "factor_ALPHA12_timing.yaml"
_REQUIRED_SECTIONS = ("meta", "factor_spec", "weight", "run")
_FACTOR_VALUE_COLUMNS = [
    "信号日",
    "股票代码",
    "因子值",
    "滚动均值",
    "滚动标准差",
    "历史阈值",
    "分组",
    "是否触发",
    "目标仓位",
]


def load_config(path: str | Path = _CONFIG_FILE) -> dict:
    return load_yaml_config(path, required_sections=_REQUIRED_SECTIONS)


def _param(params: Mapping[str, Any], key: str, default: Any) -> Any:
    return params[key] if key in params and params[key] is not None else default


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_codes(universe: Any) -> list[str]:
    if universe is None:
        return []
    if isinstance(universe, str):
        parts = re.split(r"[,\s，;；]+", universe.strip())
        return [part for part in parts if part]
    return [code for code in (_optional_text(item) for item in universe) if code]


def compute_alpha12_timing(close_wide, volume_wide, **kwargs):
    del kwargs
    return compute_alpha12(close_wide, volume_wide)


def build_spec(config: dict, config_path: str | Path = _CONFIG_FILE) -> FactorSpec:
    options = factor_spec_options(config, config_path)
    return FactorSpec(
        name=str(section(config, "meta")["name"]),
        compute=compute_alpha12_timing,
        **options,
    )


spec = build_spec(load_config())


def _daily_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").copy()
    out.index = pd.to_datetime(out.index).normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _resolve_codes(params: Mapping[str, Any], universe: Any) -> list[str]:
    stock_code = _optional_text(params.get("stock_code"))
    if stock_code:
        return [stock_code]

    codes = _normalize_codes(universe)
    if len(codes) == 1:
        return codes
    if codes:
        raise ValueError("ALPHA12_timing requires exactly one stock_code or a single-code run.universe")
    raise ValueError("ALPHA12_timing requires compute_kwargs.stock_code or run.universe")


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


def _signal_dates(start_date: str, end_date: str, fetch_start: str, rebal_freq: str) -> list[pd.Timestamp]:
    from betalens.datafeed import get_absolute_trade_days

    rebalance_dates = get_absolute_trade_days(start_date, end_date, rebal_freq, use_pmc=False)
    all_trade_days = sorted(get_absolute_trade_days(fetch_start, end_date, "D", use_pmc=False))
    day_index = {day: i for i, day in enumerate(all_trade_days)}
    dates = [all_trade_days[day_index[day] - 1] for day in rebalance_dates if day_index.get(day, 0) > 0]
    return [pd.Timestamp(day).normalize() for day in dates]


def _rolling_threshold(series: pd.Series, window: int, sigma: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    history = series.shift(1)
    rolling = history.rolling(window=window, min_periods=window)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std()
    threshold = rolling_mean + float(sigma) * rolling_std
    return rolling_mean, rolling_std, threshold


def _build_timing_outputs(
    *,
    factor_wide: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    codes: list[str],
    params: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_index = pd.DatetimeIndex(signal_dates).sort_values()
    if signal_index.empty:
        raise ValueError("no signal dates for timing run")

    stock_code = codes[0]
    if stock_code not in factor_wide.columns:
        raise ValueError(f"missing factor input for stock_code={stock_code}")

    factor_series = _daily_series(factor_wide[stock_code]).reindex(signal_index)
    if factor_series.dropna().empty:
        raise ValueError(f"empty factor data for {stock_code}")

    result = rolling_z_weight(
        factor_wide=factor_wide,
        signal_dates=signal_dates,
        codes=codes,
        params=params,
        side="short",
    )
    factor_values = result.factor_values.copy()
    for col in _FACTOR_VALUE_COLUMNS:
        if col not in factor_values.columns:
            factor_values[col] = pd.NA
    return result.weights, factor_values[_FACTOR_VALUE_COLUMNS]


def _write_timing_artifacts(output_dir: Path, name: str, weights: pd.DataFrame, factor_values: pd.DataFrame) -> None:
    weights.to_csv(output_dir / f"{name}_weights.csv", encoding="utf-8-sig")
    factor_values.to_csv(output_dir / f"{name}_factor_values.csv", index=False, encoding="utf-8-sig")


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
        threshold_window = int(_param(params, "threshold_window", 120))
        inferred_warmup = max(
            infer_warmup_days(params, minimum=threshold_window),
            infer_signal_warmup(params, minimum=threshold_window),
        )
        warmup = int(warmup_days if warmup_days is not None else max(inferred_warmup, threshold_window))
        fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=warmup)).strftime("%Y-%m-%d")
        fetch_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        signals = _signal_dates(start_date, end_date, fetch_start, rebal_freq)
        if not signals:
            raise ValueError(f"no signal dates for {start_date}~{end_date}, rebal_freq={rebal_freq}")

        if verbose:
            print(
                f"ALPHA12_timing: codes={codes} fetch={fetch_start}~{fetch_end} "
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

        if verbose:
            print("  compute factor", flush=True)
        factor_wide = sp.compute(**input_wides, **params)
        if factor_wide.empty:
            raise ValueError("empty ALPHA12 factor values")

        weights, factor_values = _build_timing_outputs(
            factor_wide=factor_wide,
            signal_dates=signals,
            codes=codes,
            params=params,
        )
        if weights.empty:
            raise ValueError("empty timing weights")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_timing_artifacts(out_dir, sp.name, weights, factor_values)

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
    parser = argparse.ArgumentParser(description="Run ALPHA12 timing factor from its YAML parameter file.")
    parser.add_argument("--config", default=str(_CONFIG_FILE), help="YAML parameter file")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
