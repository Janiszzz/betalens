"""Logical dataset registry for database-manager reads and writes.

The public names in this module are the compatibility contract.  Physical
table names are intentionally kept here so importers, the GUI and query tools
do not grow their own routing rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class CoreMetric:
    """A legacy metric represented by a column in ``market_daily_fact``."""

    column: str
    available_time: str


@dataclass(frozen=True)
class DatasetSpec:
    """Storage and compatibility metadata for one logical dataset."""

    name: str
    storage: str
    physical_tables: tuple[str, ...]
    entity_type: str | None = None
    writable: bool = True


CORE_MARKET_METRICS: Mapping[str, CoreMetric] = MappingProxyType(
    {
        "开盘价(元)": CoreMetric("open", "09:30:01"),
        "最高价(元)": CoreMetric("high", "15:00:01"),
        "最低价(元)": CoreMetric("low", "15:00:01"),
        "收盘价(元)": CoreMetric("close", "15:00:01"),
        "前收盘价": CoreMetric("prev_close", "09:30:01"),
        "前收盘价(元)": CoreMetric("prev_close", "09:30:01"),
        "成交量(股)": CoreMetric("volume", "15:00:01"),
        "成交金额(元)": CoreMetric("amount", "15:00:01"),
        "成交额(元)": CoreMetric("amount", "15:00:01"),
        "换手率(%)": CoreMetric("turnover_rate", "15:00:01"),
    }
)


_MARKET_TABLES = (
    "entity_dim",
    "entity_name_history",
    "market_daily_fact",
    "observation_fact",
    "metric_dim",
    "metric_alias",
)
_OBSERVATION_TABLES = (
    "entity_dim",
    "entity_name_history",
    "observation_fact",
    "metric_dim",
    "metric_alias",
)

DATASETS: Mapping[str, DatasetSpec] = MappingProxyType(
    {
        "daily_market": DatasetSpec("daily_market", "market", _MARKET_TABLES, "stock"),
        "daily_index": DatasetSpec("daily_index", "market", _MARKET_TABLES, "index"),
        "daily_fund": DatasetSpec("daily_fund", "market", _MARKET_TABLES, "fund"),
        "daily_bond": DatasetSpec("daily_bond", "market", _MARKET_TABLES, "bond"),
        "fundamentals": DatasetSpec("fundamentals", "observation", _OBSERVATION_TABLES, "stock"),
        "macro": DatasetSpec("macro", "observation", _OBSERVATION_TABLES, "macro"),
        "factors": DatasetSpec("factors", "observation", _OBSERVATION_TABLES, "stock"),
        "industry": DatasetSpec(
            "industry",
            "industry",
            ("entity_dim", "industry_scheme_dim", "industry_dim", "industry_membership"),
            "stock",
        ),
        "index_universe": DatasetSpec(
            "index_universe",
            "index_universe",
            ("entity_dim", "index_snapshot", "index_constituent"),
            "index",
        ),
        "trade_status": DatasetSpec(
            "trade_status",
            "trade_status",
            ("entity_dim", "trade_status_event"),
            "stock",
        ),
    }
)


LOGICAL_TABLES = tuple(DATASETS)
WRITABLE_TABLES = tuple(name for name, spec in DATASETS.items() if spec.writable)


def get_dataset(name: str, *, writable: bool = False) -> DatasetSpec:
    """Return a validated dataset specification."""

    try:
        spec = DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"非法逻辑表名: {name}") from exc
    if writable and not spec.writable:
        raise ValueError(f"逻辑表不可写: {name}")
    return spec


def canonical_metric(metric: str, logical_dataset: str = "daily_market") -> str:
    """Resolve the small static alias set used before a database is available."""

    if logical_dataset == "daily_market":
        return {
            "前收盘价(元)": "前收盘价",
            "成交额": "成交金额(元)",
            "成交额(元)": "成交金额(元)",
            "成交金额": "成交金额(元)",
        }.get(metric, metric)
    return metric


def core_metric(metric: str) -> CoreMetric | None:
    return CORE_MARKET_METRICS.get(metric)
