"""Detach public outputs from engines; no DB handles or engine reconstruction."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any

import pandas as pd

from .models import ArchiveError, json_value


BACKTEST_TABLES = (
    "weight", "effective_target_weight", "actual_weight", "daily_weight",
    "cost_price", "raw_cost_price", "actual_datetime", "cost_ret", "amount",
    "position", "equivalent_position", "position_units", "daily_price", "daily_raw_price",
    "daily_amount", "nav", "daily_returns", "daily_position_value", "daily_pnl",
    "daily_pnl_total", "daily_fees", "rebalance_log", "order_log", "risk_events",
    "synthetic_settlements", "valuation_audit", "price_audit", "trade_status",
    "trade_status_matrix", "trade_cycles", "research_schedule", "research_forward_returns",
    "research_group_nav", "research_group_coverage", "factor_values",
)
ANALYSIS_TABLES = (
    "nav", "returns", "daily_returns", "weight", "daily_position_value", "daily_pnl",
    "rebalance_log", "benchmark", "cost_price", "factor_values", "daily_amount",
    "synthetic_settlements", "risk_events", "forward_returns", "research_schedule",
    "trade_cycles", "ic_series",
)


@dataclass
class Snapshot:
    tables: dict[str, Any] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)


def capture_analyst(analyst) -> Snapshot:
    an = analyst.an
    tables = {"analysis/" + key: getattr(an, key).copy(deep=True)
              for key in ANALYSIS_TABLES if isinstance(getattr(an, key, None), (pd.Series, pd.DataFrame))}
    if "analysis/nav" not in tables:
        raise ArchiveError("Analyst has no NAV")
    tables["nav"] = tables["analysis/nav"].copy()
    return Snapshot(tables=tables, metrics=an.summary(), metadata={
        "snapshot_kind": "analyst", "name": analyst.name,
        "analysis": {key: getattr(an, key) for key in ("risk_free_rate", "annualizer", "window")},
        "analysis_metadata": dict(an.metadata),
        "analysis_diagnostics": list(getattr(an, "diagnostics", ())),
        # Never invoke name_map: it may lazily query a market database.
        "name_map": dict(getattr(analyst, "_name_map", None) or {}),
    })


def capture_backtest(bt, analyst=None) -> Snapshot:
    if analyst is None:
        from betalens.analyst import Analyst
        analyst = Analyst.from_backtest(bt)
    snapshot = capture_analyst(analyst)
    snapshot.tables.update({key: getattr(bt, key).copy(deep=True) for key in BACKTEST_TABLES
                            if isinstance(getattr(bt, key, None), (pd.Series, pd.DataFrame))})
    snapshot.metadata.update({key: json_value(getattr(bt, key)) for key in (
        "symbol", "metric", "table_name", "initial_amount", "ftc", "ptc", "start", "end",
        "trades", "calculation_version", "risk_config", "price_time_map", "data_sources",
    ) if hasattr(bt, key)})
    snapshot.metadata["snapshot_kind"] = "backtest"
    return snapshot


def capture_run(result) -> Snapshot:
    snapshot = capture_backtest(result.backtest, getattr(result, "analyst", None))
    for name in ("profiling", "neutralize_stats", "factor_values", "pit_validation", "chart_data"):
        value = getattr(result, name, None)
        if isinstance(value, (pd.DataFrame, pd.Series)):
            snapshot.tables[name] = value.copy(deep=True)
        elif value is not None:
            snapshot.extras[name] = value
    return snapshot


def describe_spec(spec) -> dict:
    """Effective FactorSpec config, including callable identity, never the callable."""
    if is_dataclass(spec):
        values = asdict(spec)
    else:
        values = {}
        for cls in reversed(type(spec).__mro__):
            values.update({key: getattr(spec, key) if isinstance(value, (staticmethod, classmethod)) else value
                           for key, value in vars(cls).items()
                           if not key.startswith("_") and not isinstance(value, property)})
        values.update(vars(spec))
    def describe(value):
        if not callable(value):
            return json_value(value)
        identity = f"{getattr(value, '__module__', type(value).__module__)}:{getattr(value, '__qualname__', getattr(value, '__name__', type(value).__qualname__))}"
        from functools import partial
        if isinstance(value, partial):
            return {"callable": describe(value.func), "args": json_value(value.args), "kwargs": json_value(value.keywords or {})}
        return identity
    return {key: describe(value) for key, value in values.items()}
