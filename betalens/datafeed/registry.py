"""Logical dataset and core metric registry used by the read layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time


NORMALIZED_SCHEMA = "betalens"


@dataclass(frozen=True)
class DatasetSpec:
    logical_name: str
    kind: str
    entity_type: str | None = None


@dataclass(frozen=True)
class CoreMetric:
    canonical_name: str
    column: str
    available_time: time


DATASETS = {
    "daily_market": DatasetSpec("daily_market", "market", "stock"),
    "daily_index": DatasetSpec("daily_index", "market", "index"),
    "daily_fund": DatasetSpec("daily_fund", "market", "fund"),
    "daily_bond": DatasetSpec("daily_bond", "market", "bond"),
    "fundamentals": DatasetSpec("fundamentals", "observation", "stock"),
    "macro": DatasetSpec("macro", "observation", "macro"),
    "factors": DatasetSpec("factors", "observation", "stock"),
    "industry": DatasetSpec("industry", "industry", "stock"),
    "index_universe": DatasetSpec("index_universe", "index_universe", "index"),
    "trade_status": DatasetSpec("trade_status", "trade_status", "stock"),
    "trade_calendar": DatasetSpec("trade_calendar", "trade_calendar"),
}


_OPEN = time(9, 30, 1)
_CLOSE = time(15, 0, 1)


def _metrics(
    *items: tuple[str, str, tuple[str, ...], str, time],
) -> dict[tuple[str, str], CoreMetric]:
    result: dict[tuple[str, str], CoreMetric] = {}
    for dataset, canonical_name, aliases, column, available_time in items:
        metric = CoreMetric(canonical_name, column, available_time)
        for alias in (canonical_name, *aliases):
            result[(dataset, alias)] = metric
    return result


CORE_METRICS = _metrics(
    ("daily_market", "开盘价(元)", ("开盘价",), "open", _OPEN),
    ("daily_market", "最高价(元)", ("最高价",), "high", _CLOSE),
    ("daily_market", "最低价(元)", ("最低价",), "low", _CLOSE),
    ("daily_market", "收盘价(元)", ("收盘价",), "close", _CLOSE),
    ("daily_market", "前收盘价", ("前收盘价(元)",), "prev_close", _OPEN),
    ("daily_market", "成交量(股)", ("成交量",), "volume", _CLOSE),
    (
        "daily_market",
        "成交金额(元)",
        ("成交额", "成交额(元)", "成交金额"),
        "amount",
        _CLOSE,
    ),
    ("daily_market", "换手率(%)", ("换手率",), "turnover_rate", _CLOSE),
    ("daily_index", "开盘价", ("开盘价(元)",), "open", _OPEN),
    ("daily_index", "最高价", ("最高价(元)",), "high", _CLOSE),
    ("daily_index", "最低价", ("最低价(元)",), "low", _CLOSE),
    ("daily_index", "收盘价", ("收盘价(元)",), "close", _CLOSE),
    ("daily_index", "前收盘价", ("前收盘价(元)",), "prev_close", _OPEN),
    ("daily_index", "成交量", (), "volume", _CLOSE),
    ("daily_index", "成交额", ("成交额(元)", "成交金额(元)"), "amount", _CLOSE),
    ("daily_index", "换手率", ("换手率(%)",), "turnover_rate", _CLOSE),
    ("daily_fund", "开盘价(元)", ("开盘价",), "open", _OPEN),
    ("daily_fund", "最高价(元)", ("最高价",), "high", _CLOSE),
    ("daily_fund", "最低价(元)", ("最低价",), "low", _CLOSE),
    ("daily_fund", "收盘价(元)", ("收盘价",), "close", _CLOSE),
    ("daily_fund", "前收盘价", ("前收盘价(元)",), "prev_close", _OPEN),
    ("daily_fund", "成交量(份)", ("成交量",), "volume", _CLOSE),
    ("daily_fund", "成交额(元)", ("成交金额(元)", "成交额"), "amount", _CLOSE),
    ("daily_fund", "换手率(%)", ("换手率",), "turnover_rate", _CLOSE),
    ("daily_bond", "开盘价(元)", ("开盘价",), "open", _OPEN),
    ("daily_bond", "最高价(元)", ("最高价",), "high", _CLOSE),
    ("daily_bond", "最低价(元)", ("最低价",), "low", _CLOSE),
    ("daily_bond", "收盘价(元)", ("收盘价",), "close", _CLOSE),
    ("daily_bond", "前收盘价", ("前收盘价(元)",), "prev_close", _OPEN),
    ("daily_bond", "成交量(手)", ("成交量",), "volume", _CLOSE),
    ("daily_bond", "成交额(元)", ("成交金额(元)", "成交额"), "amount", _CLOSE),
    ("daily_bond", "换手率(%)", ("换手率",), "turnover_rate", _CLOSE),
)


def get_dataset(name: str) -> DatasetSpec | None:
    return DATASETS.get(name)


def get_core_metric(dataset: str, metric: str) -> CoreMetric | None:
    return CORE_METRICS.get((dataset, metric))
