"""Typed, target-aware source adapters for database imports."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Iterator, Mapping

import numpy as np
import pandas as pd

from .adapters.ede import (
    DEFAULT_CODE_COLUMNS,
    DEFAULT_DATE_PATTERN,
    DEFAULT_NAME_COLUMNS,
    DEFAULT_TIME,
    clean_ede_dataframe,
    extract_date_from_filename,
    extract_date_from_metric_metadata,
    identify_code_name_columns,
    parse_metric_column,
)
from .adapters.files import DEFAULT_CHUNK_SIZE, iter_file_chunks
from .constants import DB_COLUMNS, INSERT_ONLY, UPSERT
from .registry import DATASETS


MARKET_TARGETS = frozenset(("daily_market", "daily_index", "daily_fund", "daily_bond"))
OBSERVATION_TARGETS = frozenset(("fundamentals", "macro", "factors"))
ALL_TARGETS = frozenset(DATASETS)
SPECIAL_TARGETS = frozenset(("industry", "index_universe", "trade_status", "trade_calendar"))
MISSING_VALUE_MARKERS = frozenset(("n/a",))
DAILY_WIDE_DATE_COLUMNS = frozenset(("date", "datetime", "日期", "交易日期"))


class BatchKind(str, Enum):
    MARKET = "market"
    OBSERVATION = "observation"
    INDUSTRY = "industry"
    INDEX = "index"
    TRADE_STATUS = "trade_status"
    TRADE_CALENDAR = "trade_calendar"


@dataclass
class ImportBatch:
    """One bounded, validated batch ready for ``DatabaseWriter``."""

    table: str
    frame: pd.DataFrame
    rejected: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["_source_row", "_errors"])
    )
    warnings: tuple[str, ...] = ()
    source_rows: int = 0
    # Adapter-specific fields are extracted before the writer sees the batch.
    # They keep industry/index/status semantics out of an untyped ``remark``
    # string while retaining the six-column public DataFrame boundary.
    typed_fields: Mapping[str, Any] = field(default_factory=dict)
    kind: ClassVar[BatchKind]
    allowed_storage: ClassVar[frozenset[str]]

    def __post_init__(self) -> None:
        if self.table not in DATASETS:
            raise ValueError(f"非法逻辑目标: {self.table}")
        storage = DATASETS[self.table].storage
        if storage not in self.allowed_storage:
            raise ValueError(f"{type(self).__name__} 不支持目标 {self.table} ({storage})")
        missing = [column for column in DB_COLUMNS if column not in self.frame.columns]
        if missing:
            raise ValueError(f"规范化批次缺少列: {missing}")
        self.frame = self.frame[list(DB_COLUMNS)].reset_index(drop=True)
        self.rejected = self.rejected.reset_index(drop=True)
        if not self.source_rows:
            self.source_rows = len(self.frame) + len(self.rejected)


class MarketBatch(ImportBatch):
    kind = BatchKind.MARKET
    allowed_storage = frozenset(("market",))


class ObservationBatch(ImportBatch):
    kind = BatchKind.OBSERVATION
    allowed_storage = frozenset(("observation",))


class IndustryBatch(ImportBatch):
    kind = BatchKind.INDUSTRY
    allowed_storage = frozenset(("industry",))


class IndexSnapshotBatch(ImportBatch):
    kind = BatchKind.INDEX
    allowed_storage = frozenset(("index_universe",))


class TradeStatusBatch(ImportBatch):
    kind = BatchKind.TRADE_STATUS
    allowed_storage = frozenset(("trade_status",))


class TradeCalendarBatch(ImportBatch):
    kind = BatchKind.TRADE_CALENDAR
    allowed_storage = frozenset(("trade_calendar",))


BATCH_CLASS_BY_STORAGE: Mapping[str, type[ImportBatch]] = {
    "market": MarketBatch,
    "observation": ObservationBatch,
    "industry": IndustryBatch,
    "index_universe": IndexSnapshotBatch,
    "trade_status": TradeStatusBatch,
    "trade_calendar": TradeCalendarBatch,
}


@dataclass(frozen=True)
class AdapterContext:
    path: Path
    table: str
    options: Mapping[str, Any]
    chunk_size: int
    logger: logging.Logger


AdapterLoader = Callable[[AdapterContext], Iterator[ImportBatch]]


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    loader: AdapterLoader
    allowed_targets: frozenset[str]
    option_keys: frozenset[str] = frozenset()
    required_options: frozenset[str] = frozenset()
    aliases: tuple[str, ...] = ()

    def validate(self, table: str, options: Mapping[str, Any], *, strict: bool) -> None:
        if table not in self.allowed_targets:
            raise ValueError(f"导入类型 {self.name} 不支持逻辑目标 {table}")
        missing = sorted(self.required_options - set(options))
        if missing:
            raise ValueError(f"导入类型 {self.name} 缺少 options: {missing}")
        if strict:
            unknown = sorted(set(options) - self.option_keys)
            if unknown:
                raise ValueError(f"导入类型 {self.name} 不支持 options: {unknown}")


class AdapterRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, AdapterSpec] = {}

    def register(self, spec: AdapterSpec) -> None:
        for key in (spec.name, *spec.aliases):
            if key in self._specs:
                raise ValueError(f"重复导入适配器: {key}")
            self._specs[key] = spec

    def resolve(self, name: str) -> AdapterSpec:
        try:
            return self._specs[str(name)]
        except KeyError as exc:
            raise ValueError(f"未知导入类型: {name}") from exc

    def names(self, *, include_aliases: bool = True) -> tuple[str, ...]:
        if include_aliases:
            return tuple(self._specs)
        return tuple(dict.fromkeys(spec.name for spec in self._specs.values()))

    def validate(
        self,
        name: str,
        table: str,
        options: Mapping[str, Any] | None = None,
        *,
        strict_options: bool = False,
    ) -> AdapterSpec:
        spec = self.resolve(name)
        spec.validate(table, options or {}, strict=strict_options)
        return spec


ADAPTERS = AdapterRegistry()


_COMMON_OPTIONS = frozenset(
    (
        "column_map",
        "chunk_size",
        "read_options",
        "canonicalize_market_time",
        "remark_text_key",
    )
)


_CORE_METRIC_TIMES: Mapping[str, Mapping[str, str]] = {
    "daily_market": {
        "开盘价": "09:30:01",
        "开盘价(元)": "09:30:01",
        "前收盘价": "09:30:01",
        "前收盘价(元)": "09:30:01",
        "最高价": "15:00:01",
        "最高价(元)": "15:00:01",
        "最低价": "15:00:01",
        "最低价(元)": "15:00:01",
        "收盘价": "15:00:01",
        "收盘价(元)": "15:00:01",
        "成交量": "15:00:01",
        "成交量(股)": "15:00:01",
        "成交额": "15:00:01",
        "成交额(元)": "15:00:01",
        "成交金额": "15:00:01",
        "成交金额(元)": "15:00:01",
        "换手率": "15:00:01",
        "换手率(%)": "15:00:01",
    },
    "daily_index": {
        "开盘价": "09:30:01", "开盘价(元)": "09:30:01",
        "前收盘价": "09:30:01", "前收盘价(元)": "09:30:01",
        "最高价": "15:00:01", "最高价(元)": "15:00:01",
        "最低价": "15:00:01", "最低价(元)": "15:00:01",
        "收盘价": "15:00:01", "收盘价(元)": "15:00:01",
        "成交量": "15:00:01", "成交额": "15:00:01",
        "成交额(元)": "15:00:01", "成交金额(元)": "15:00:01",
        "换手率": "15:00:01", "换手率(%)": "15:00:01",
    },
    "daily_fund": {},
    "daily_bond": {},
}
_CORE_METRIC_TIMES["daily_fund"] = dict(_CORE_METRIC_TIMES["daily_market"])
_CORE_METRIC_TIMES["daily_bond"] = dict(_CORE_METRIC_TIMES["daily_market"])


def _batch_class(table: str) -> type[ImportBatch]:
    return BATCH_CLASS_BY_STORAGE[DATASETS[table].storage]


def _apply_column_map(frame: pd.DataFrame, options: Mapping[str, Any]) -> pd.DataFrame:
    mapping = options.get("column_map") or {}
    if not isinstance(mapping, Mapping):
        raise ValueError("column_map 必须是 source_column: canonical_column 映射")
    unknown = sorted(str(column) for column in mapping if column not in frame.columns)
    if unknown:
        raise ValueError(f"column_map 源列不存在: {unknown}")
    renamed = frame.rename(columns=dict(mapping))
    duplicate = renamed.columns[renamed.columns.duplicated()].tolist()
    if duplicate:
        raise ValueError(f"column_map 产生重复列: {duplicate}")
    return renamed


def _parse_remark(value: Any, *, text_key: str | None) -> tuple[Any, str | None]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None
    if isinstance(value, dict):
        return value, None
    if isinstance(value, (list, bool, int, float)):
        return None, "remark 必须是 JSON object"
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None, None
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return None, "remark 必须是 JSON object"
        return parsed, None
    except json.JSONDecodeError:
        if text_key:
            return {text_key: text}, None
        return None, "remark 不是合法 JSON"


def _canonicalize_market_time(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    mapping = _CORE_METRIC_TIMES.get(table, {})
    if not mapping or frame.empty:
        return frame
    out = frame.copy()
    for metric, time_text in mapping.items():
        mask = out["metric"].eq(metric)
        if mask.any():
            out.loc[mask, "datetime"] = (
                out.loc[mask, "datetime"].dt.normalize() + pd.to_timedelta(time_text)
            )
    return out


def _special_errors(row: pd.Series, table: str) -> list[str]:
    errors: list[str] = []
    remark = row["remark"]
    if table == "industry":
        if not isinstance(remark, dict) or not remark.get("ind_name") or not remark.get("ind_code"):
            errors.append("industry 要求 remark.ind_name 和 remark.ind_code")
    elif table == "index_universe":
        constituents = remark.get("constituents") if isinstance(remark, dict) else None
        if row["metric"] != "universe" or not isinstance(constituents, list):
            errors.append("index_universe 要求 metric=universe 和 remark.constituents 数组")
        elif float(row["value"]) != float(len({json.dumps(v, sort_keys=True, default=str) for v in constituents})):
            errors.append("index_universe value 与去重成分数量不一致")
    elif table == "trade_status":
        if row["metric"] != "交易状态" or row["value"] not in (0.0, 1.0):
            errors.append("trade_status 仅接受 metric=交易状态 且 value 为 0/1")
    elif table == "trade_calendar":
        if row["metric"] != "交易日" or row["value"] != 1.0:
            errors.append("trade_calendar 仅接受 metric=交易日 且 value=1")
    return errors


def _normalize_long(
    source: pd.DataFrame,
    context: AdapterContext,
    *,
    row_offset: int,
) -> ImportBatch:
    frame = _apply_column_map(source, context.options).copy()
    if "note" in frame.columns and "remark" not in frame.columns:
        frame = frame.rename(columns={"note": "remark"})
    if "remark" not in frame.columns:
        frame["remark"] = None
    required = ("datetime", "code", "name", "metric", "value")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"标准长表缺少列: {missing}")

    working = frame[list(DB_COLUMNS)].copy()
    source_rows = len(working)
    working["_source_row"] = np.arange(row_offset + 1, row_offset + source_rows + 1)

    # N/A is an expected source-side missing-value marker.  It represents no
    # observation to import, rather than an invalid observation.
    value_text = working["value"].astype("string").str.strip().str.casefold()
    missing_value = working["value"].isna() | value_text.isin(MISSING_VALUE_MARKERS)
    working = working.loc[~missing_value].copy()

    # Error positions below are relative to the retained, importable source
    # rows.  Missing-value rows were intentionally skipped above.
    errors = [[] for _ in range(len(working))]
    retained_source = frame.loc[working.index]

    parsed_dt = pd.to_datetime(working["datetime"], errors="coerce")
    parsed_value = pd.to_numeric(working["value"], errors="coerce")
    finite = pd.Series(
        np.isfinite(parsed_value.to_numpy(dtype=float, na_value=np.nan)),
        index=working.index,
    )
    for position, bad in enumerate(parsed_dt.isna().to_numpy()):
        if bad:
            errors[position].append("datetime 无法解析")
    for position, bad in enumerate((~finite).to_numpy()):
        if bad:
            errors[position].append("value 不是有限数值")

    working["datetime"] = parsed_dt
    working["value"] = parsed_value
    for column in ("code", "name", "metric"):
        null_mask = working[column].isna()
        working[column] = working[column].fillna("").astype(str).str.strip()
        bad_mask = null_mask | working[column].eq("")
        for position, bad in enumerate(bad_mask.to_numpy()):
            if bad:
                errors[position].append(f"{column} 为空")

    parsed_remarks: list[Any] = []
    text_key = context.options.get("remark_text_key")
    for position, value in enumerate(working["remark"].tolist()):
        parsed, error = _parse_remark(value, text_key=text_key)
        parsed_remarks.append(parsed)
        if error:
            errors[position].append(error)
    working["remark"] = parsed_remarks

    for position, (_, row) in enumerate(working.iterrows()):
        if not errors[position]:
            errors[position].extend(_special_errors(row, context.table))
    rejected_mask = pd.Series([bool(items) for items in errors], index=working.index)
    # Keep the original row columns for compatibility, and add one structured
    # record per rejected field so callers can route diagnostics without
    # parsing a free-form error string.
    rejected_rows: list[dict[str, Any]] = []
    for position, bad in enumerate(rejected_mask.to_numpy()):
        if not bad:
            continue
        source_row = int(working.iloc[position]["_source_row"])
        combined = "; ".join(errors[position])
        original = retained_source.iloc[position].to_dict()
        for reason in errors[position]:
            field_name = "remark"
            for candidate in ("datetime", "code", "name", "metric", "value", "remark"):
                if candidate in reason:
                    field_name = candidate
                    break
            rejected_rows.append(
                {
                    **original,
                    "source_file": str(context.path),
                    "source_row": source_row,
                    "field": field_name,
                    "raw_value": original.get(field_name),
                    "reason": reason,
                    # Historical names remain available to old ImportJobRunner
                    # callers while new consumers use the typed names above.
                    "_source_row": source_row,
                    "_errors": combined,
                }
            )
    rejected = pd.DataFrame(rejected_rows)
    if rejected.empty:
        rejected = pd.DataFrame(
            columns=[
                *frame.columns,
                "source_file", "source_row", "field", "raw_value", "reason",
                "_source_row", "_errors",
            ]
        )

    valid = working.loc[~rejected_mask, list(DB_COLUMNS)].copy()
    if context.table in MARKET_TARGETS and context.options.get("canonicalize_market_time", True):
        valid = _canonicalize_market_time(valid, context.table)
    if context.table == "trade_calendar":
        valid["code"] = valid["code"].astype(str).str.strip().str.upper()
        valid["name"] = valid["code"]
        valid["datetime"] = valid["datetime"].dt.normalize()
    typed_fields: dict[str, Any] = {}
    if context.table == "industry":
        typed_fields = {
            "industry_name": valid["remark"].map(lambda item: item.get("ind_name") if isinstance(item, dict) else None).reset_index(drop=True),
            "industry_code": valid["remark"].map(lambda item: item.get("ind_code") if isinstance(item, dict) else None).reset_index(drop=True),
            "scheme": valid["remark"].map(lambda item: item.get("scheme") if isinstance(item, dict) else None).reset_index(drop=True),
        }
    elif context.table == "index_universe":
        typed_fields = {
            "constituents": valid["remark"].map(lambda item: item.get("constituents") if isinstance(item, dict) else None).reset_index(drop=True),
        }
    elif context.table == "trade_status":
        typed_fields = {
            "status_text": valid["remark"].map(lambda item: item.get("status") if isinstance(item, dict) else None).reset_index(drop=True),
        }
    return _batch_class(context.table)(
        table=context.table,
        frame=valid,
        rejected=rejected,
        source_rows=source_rows,
        typed_fields=typed_fields,
    )


def _iter_source(context: AdapterContext) -> Iterator[pd.DataFrame]:
    read_options = dict(context.options.get("read_options") or {})
    if "sheet_name" in context.options and context.path.suffix.lower() in {".xls", ".xlsx"}:
        read_options.setdefault("sheet_name", context.options["sheet_name"])
    yield from iter_file_chunks(
        context.path,
        chunk_size=context.chunk_size,
        read_options=read_options,
        logger=context.logger,
    )


def _standard_long_loader(
    context: AdapterContext,
    sources: Iterable[pd.DataFrame] | None = None,
) -> Iterator[ImportBatch]:
    offset = 0
    source_frames = _iter_source(context) if sources is None else sources
    for source in source_frames:
        batch = _normalize_long(source, context, row_offset=offset)
        offset += len(source)
        yield batch


def _is_daily_wide_frame(frame: pd.DataFrame) -> bool:
    columns = {str(column).strip().casefold() for column in frame.columns}
    code_columns = {name.casefold() for name in DEFAULT_CODE_COLUMNS}
    name_columns = {name.casefold() for name in DEFAULT_NAME_COLUMNS}
    date_columns = {name.casefold() for name in DAILY_WIDE_DATE_COLUMNS}
    return bool(columns & code_columns and columns & name_columns and columns & date_columns)


def _auto_loader(context: AdapterContext) -> Iterator[ImportBatch]:
    """Route ordinary user files to a bounded, column-based parser.

    The desktop importer should not make a first-time user decide whether a
    Wind/EDE export is a long or a wide table.  This is deliberately limited
    to the three well-understood shapes below; special PIT datasets keep their
    dedicated adapters and are never guessed from a filename.
    """

    source_iter = _iter_source(context)
    try:
        sample = next(source_iter)
    except StopIteration:
        return
    try:
        source_frames = chain((sample,), source_iter)
        frame = _apply_column_map(sample, context.options)
        columns = {str(column).strip().casefold() for column in frame.columns}
        if {"datetime", "code", "name", "metric", "value"}.issubset(columns):
            yield from _standard_long_loader(context, source_frames)
            return

        if _is_daily_wide_frame(frame):
            yield from _wind_wide_loader(context, source_frames)
            return

        # The remaining supported shape is an EDE-wide export.  Its cleaner,
        # code/name discovery and metric-header date parser live in adapters/ede.
        yield from _ede_loader(context, source_frames)
    finally:
        close = getattr(source_iter, "close", None)
        if callable(close):
            close()


def _wind_wide_loader(
    context: AdapterContext,
    sources: Iterable[pd.DataFrame] | None = None,
) -> Iterator[ImportBatch]:
    offset = 0
    source_frames = _iter_source(context) if sources is None else sources
    for source in source_frames:
        # CSV/Excel exports often end with an ``Unnamed:*`` column created by
        # a trailing delimiter.  It contains no observations and must not be
        # melted into one NaN metric for every source row.
        frame = _apply_column_map(source, context.options).dropna(axis=1, how="all")
        if {"datetime", "code", "name", "metric", "value"}.issubset(frame.columns):
            batch = _normalize_long(frame, context, row_offset=offset)
            offset += len(source)
            yield batch
            continue
        aliases = {
            "代码": "code",
            "简称": "name",
            "证券代码": "code",
            "证券简称": "name",
            "日期": "date",
            "datetime": "date",
            "Date": "date",
        }
        frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})
        date_column = str(context.options.get("date_column", "date"))
        code_column = str(context.options.get("code_column", "code"))
        name_column = str(context.options.get("name_column", "name"))
        required = {date_column, code_column, name_column}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Wind 宽表缺少列: {missing}")
        ignored = set(context.options.get("ignore_columns") or ())
        value_columns = context.options.get("value_columns")
        if value_columns is None:
            value_columns = [
                column
                for column in frame.columns
                if column not in required | ignored
                and not frame[column].dropna().astype(str).str.strip().eq("").all()
            ]
        else:
            value_columns = list(value_columns)
        if not value_columns:
            raise ValueError("Wind 宽表未找到非空指标列")
        long = frame.melt(
            id_vars=[date_column, code_column, name_column],
            value_vars=value_columns,
            var_name="metric",
            value_name="value",
        ).rename(
            columns={date_column: "datetime", code_column: "code", name_column: "name"}
        )
        other_time = str(context.options.get("other_time", "15:00:01"))
        long["datetime"] = pd.to_datetime(long["datetime"], errors="coerce").dt.normalize() + pd.to_timedelta(other_time)
        long["remark"] = None
        batch = _normalize_long(long, context, row_offset=offset)
        offset += len(long)
        yield batch


def _ede_loader(
    context: AdapterContext,
    sources: Iterable[pd.DataFrame] | None = None,
) -> Iterator[ImportBatch]:
    date_from = str(context.options.get("date_from", "filename"))
    if date_from not in {"filename", "metric"}:
        raise ValueError("date_from 必须是 filename 或 metric")
    source_frames = _iter_source(context) if sources is None else sources
    source_iter = iter(source_frames)
    try:
        sample = next(source_iter)
    except StopIteration:
        return
    if _is_daily_wide_frame(_apply_column_map(sample, context.options)):
        yield from _wind_wide_loader(context, chain((sample,), source_iter))
        return

    default_time = str(context.options.get("default_time", DEFAULT_TIME))
    file_datetime = None
    if date_from == "filename":
        file_datetime = extract_date_from_filename(
            context.path,
            pattern=str(context.options.get("date_pattern", DEFAULT_DATE_PATTERN)),
            default_time=default_time,
            logger=context.logger,
        )
    file_datetime = file_datetime or context.options.get("default_datetime")
    offset = 0
    for source in chain((sample,), source_iter):
        frame = clean_ede_dataframe(
            _apply_column_map(source, context.options),
            keywords_to_remove=context.options.get("keywords_to_remove"),
            logger=context.logger,
        )
        code_col, name_col = identify_code_name_columns(
            frame,
            context.options.get("code_column_names") or DEFAULT_CODE_COLUMNS,
            context.options.get("name_column_names") or DEFAULT_NAME_COLUMNS,
            logger=context.logger,
        )
        if code_col is None:
            raise ValueError("EDE 未找到代码列")
        keys = {code_col, name_col}
        parts: list[pd.DataFrame] = []
        for metric_col in (column for column in frame.columns if column not in keys):
            metric, metadata = parse_metric_column(str(metric_col), logger=context.logger)
            available_at = file_datetime
            if date_from == "metric" or available_at is None:
                available_at = extract_date_from_metric_metadata(
                    metadata, str(metric_col), default_time=default_time, logger=context.logger
                )
            part = pd.DataFrame(
                {
                    "datetime": available_at,
                    "code": frame[code_col],
                    "name": frame[name_col] if name_col is not None else frame[code_col],
                    "metric": metric,
                    "value": frame[metric_col],
                    "remark": [dict(metadata) for _ in range(len(frame))],
                }
            )
            parts.append(part)
        long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=DB_COLUMNS)
        batch = _normalize_long(long, context, row_offset=offset)
        offset += len(long)
        yield batch


def _industry_loader(context: AdapterContext) -> Iterator[ImportBatch]:
    offset = 0
    for source in _iter_source(context):
        frame = _apply_column_map(source, context.options)
        columns = {
            "code": str(context.options.get("code_column", "code")),
            "name": str(context.options.get("name_column", "name")),
            "datetime": str(context.options.get("date_column", "effective_dt")),
            "ind_name": str(context.options.get("industry_name_column", "ind_name")),
            "ind_code": str(context.options.get("industry_code_column", "ind_code")),
        }
        missing = sorted(set(columns.values()) - set(frame.columns))
        if missing:
            raise ValueError(f"industry 源缺少列: {missing}")
        scheme_column = context.options.get("scheme_column")
        default_scheme = str(context.options.get("scheme", "申万一级行业"))
        rows = pd.DataFrame(
            {
                "datetime": frame[columns["datetime"]],
                "code": frame[columns["code"]],
                "name": frame[columns["name"]],
                "metric": frame[scheme_column] if scheme_column else default_scheme,
                "value": [
                    float(match.group()) if (match := re.search(r"\d+", str(value))) else 0.0
                    for value in frame[columns["ind_code"]]
                ],
                "remark": [
                    {"ind_name": str(name), "ind_code": str(code), "scheme": str(scheme)}
                    for name, code, scheme in zip(
                        frame[columns["ind_name"]],
                        frame[columns["ind_code"]],
                        frame[scheme_column] if scheme_column else [default_scheme] * len(frame),
                    )
                ],
            }
        )
        batch = _normalize_long(rows, context, row_offset=offset)
        offset += len(rows)
        yield batch


def _index_loader(context: AdapterContext) -> Iterator[ImportBatch]:
    sources = [_apply_column_map(frame, context.options) for frame in _iter_source(context)]
    if not sources:
        return
    frame = pd.concat(sources, ignore_index=True)
    records: list[dict[str, Any]] = []
    row_layout = {"effective_at", "index_code", "constituent_code"}.issubset(frame.columns)
    if row_layout:
        for (effective_at, index_code), group in frame.groupby(["effective_at", "index_code"], sort=True):
            constituents: list[Any] = []
            for _, row in group.iterrows():
                member: dict[str, Any] = {"code": str(row["constituent_code"]).strip()}
                if "weight" in group.columns and pd.notna(row.get("weight")):
                    member["weight"] = row["weight"]
                constituents.append(member if len(member) > 1 else member["code"])
            records.append(
                {
                    "datetime": effective_at,
                    "code": str(index_code),
                    "name": str(group["index_name"].iloc[-1]) if "index_name" in group else str(index_code),
                    "metric": "universe",
                    "value": len({str(item.get("code") if isinstance(item, dict) else item) for item in constituents}),
                    "remark": {"constituents": constituents},
                }
            )
    else:
        index_code = str(context.options.get("index_code", "")).strip()
        index_name = str(context.options.get("index_name", index_code)).strip()
        if not index_code:
            raise ValueError("旧式指数成分矩阵要求 options.index_code")
        seq_col = context.options.get("seq_col", "序号")
        for column in (value for value in frame.columns if value != seq_col):
            effective_at = pd.to_datetime(column, errors="raise")
            constituents = list(dict.fromkeys(str(value).strip() for value in frame[column] if pd.notna(value)))
            if constituents:
                records.append(
                    {
                        "datetime": effective_at,
                        "code": index_code,
                        "name": index_name,
                        "metric": "universe",
                        "value": len(constituents),
                        "remark": {"constituents": constituents},
                    }
                )
    normalized = pd.DataFrame(records, columns=DB_COLUMNS)
    yield _normalize_long(normalized, context, row_offset=0)


def _trade_status_loader(context: AdapterContext) -> Iterator[ImportBatch]:
    layout = str(context.options.get("layout", "auto"))
    if layout in {"auto", "long"}:
        sources = list(_iter_source(context))
        if sources:
            frame = _apply_column_map(pd.concat(sources, ignore_index=True), context.options)
            if {"date", "code", "status"}.issubset(frame.columns):
                normal_values = set(context.options.get("normal_values") or ("交易", "正常交易", "1", 1, True))
                rows = pd.DataFrame(
                    {
                        "datetime": pd.to_datetime(frame["date"], errors="coerce").dt.normalize() + pd.Timedelta(hours=15, seconds=1),
                        "code": frame["code"],
                        "name": frame["name"] if "name" in frame else frame["code"],
                        "metric": "交易状态",
                        "value": [1.0 if value in normal_values else 0.0 for value in frame["status"]],
                        "remark": [{"status": str(value)} for value in frame["status"]],
                    }
                )
                yield _normalize_long(rows, context, row_offset=0)
                return
            if layout == "long":
                raise ValueError("trade_status long 布局要求 date/code/status 列")
    if context.path.suffix.lower() not in {".xls", ".xlsx"}:
        raise ValueError("trade_status matrix 布局仅支持 Excel；CSV/Parquet 请使用 long 布局")
    sheet = context.options.get("sheet_name", "Sheet1")
    names_row = pd.read_excel(context.path, sheet_name=sheet, header=None, nrows=1)
    frame = pd.read_excel(context.path, sheet_name=sheet, header=1)
    date_col = frame.columns[0]
    code_columns = [column for column in frame.columns[1:] if pd.notna(column) and not str(column).startswith("Unnamed")]
    names = {column: names_row.iloc[0, position + 1] for position, column in enumerate(frame.columns[1:])}
    normal_values = set(context.options.get("normal_values") or ("交易", "正常交易"))
    records: list[dict[str, Any]] = []
    for code in code_columns:
        first_normal = False
        for position, status in enumerate(frame[code]):
            if pd.isna(status):
                continue
            normal = status in normal_values
            if normal and first_normal:
                continue
            first_normal = first_normal or normal
            records.append(
                {
                    "datetime": pd.Timestamp(frame[date_col].iloc[position]).normalize() + pd.Timedelta(hours=15, seconds=1),
                    "code": str(code),
                    "name": names.get(code) or str(code),
                    "metric": "交易状态",
                    "value": 1.0 if normal else 0.0,
                    "remark": {"status": str(status), **({"first_normal": True} if normal else {})},
                }
            )
    yield _normalize_long(pd.DataFrame(records, columns=DB_COLUMNS), context, row_offset=0)


def _trade_calendar_loader(context: AdapterContext) -> Iterator[ImportBatch]:
    row_offset = 0
    for source in _iter_source(context):
        frame = _apply_column_map(source, context.options)
        records: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        source_rows = 0
        for column in frame.columns:
            raw_exchange = str(column).strip()
            exchange = raw_exchange.upper()
            blank_header = not raw_exchange or raw_exchange.casefold().startswith("unnamed:")
            invalid_header = blank_header or len(exchange) > 32
            for position, value in enumerate(frame[column].tolist()):
                if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                source_rows += 1
                source_row = row_offset + position + 2
                reason = None
                parsed = pd.NaT
                if invalid_header:
                    reason = "交易所列标题为空或超过 32 个字符"
                elif isinstance(value, (bool, int, float, np.number)):
                    reason = "trade_date 无法解析"
                else:
                    parsed = pd.to_datetime(value, errors="coerce")
                    if pd.isna(parsed):
                        reason = "trade_date 无法解析"
                if reason is not None:
                    rejected.append(
                        {
                            "exchange": None if blank_header else exchange,
                            "trade_date": value,
                            "source_file": str(context.path),
                            "source_row": source_row,
                            "field": "exchange" if invalid_header else "trade_date",
                            "raw_value": raw_exchange if invalid_header else value,
                            "reason": reason,
                            "_source_row": source_row,
                            "_errors": reason,
                        }
                    )
                    continue
                timestamp = pd.Timestamp(parsed)
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.tz_localize(None)
                records.append(
                    {
                        "datetime": timestamp.normalize(),
                        "code": exchange,
                        "name": exchange,
                        "metric": "交易日",
                        "value": 1.0,
                        "remark": None,
                    }
                )
        row_offset += len(frame)
        valid = pd.DataFrame(records, columns=DB_COLUMNS)
        rejected_frame = pd.DataFrame(rejected)
        if rejected_frame.empty:
            rejected_frame = pd.DataFrame(
                columns=(
                    "exchange", "trade_date", "source_file", "source_row",
                    "field", "raw_value", "reason", "_source_row", "_errors",
                )
            )
        yield TradeCalendarBatch(
            table=context.table,
            frame=valid,
            rejected=rejected_frame,
            source_rows=source_rows,
        )


def _register_defaults() -> None:
    ADAPTERS.register(
        AdapterSpec(
            "auto",
            _auto_loader,
            MARKET_TARGETS | OBSERVATION_TARGETS,
            _COMMON_OPTIONS
            | frozenset(
                (
                    "date_column", "code_column", "name_column", "value_columns",
                    "ignore_columns", "other_time", "date_from", "default_datetime",
                    "date_pattern", "default_time", "keywords_to_remove",
                    "code_column_names", "name_column_names",
                )
            ),
        )
    )
    ADAPTERS.register(
        AdapterSpec(
            "standard_long",
            _standard_long_loader,
            ALL_TARGETS,
            _COMMON_OPTIONS,
        )
    )
    ADAPTERS.register(
        AdapterSpec(
            "wind_wide",
            _wind_wide_loader,
            MARKET_TARGETS | OBSERVATION_TARGETS,
            _COMMON_OPTIONS
            | frozenset(("date_column", "code_column", "name_column", "value_columns", "ignore_columns", "other_time")),
            aliases=("wind_long",),
        )
    )
    ADAPTERS.register(
        AdapterSpec(
            "ede",
            _ede_loader,
            MARKET_TARGETS | OBSERVATION_TARGETS,
            _COMMON_OPTIONS
            | frozenset(
                (
                    "date_from", "default_datetime", "date_pattern", "default_time",
                    "keywords_to_remove", "code_column_names", "name_column_names",
                )
            ),
        )
    )
    ADAPTERS.register(
        AdapterSpec(
            "industry",
            _industry_loader,
            frozenset(("industry",)),
            _COMMON_OPTIONS
            | frozenset(
                (
                    "code_column", "name_column", "date_column", "industry_name_column",
                    "industry_code_column", "scheme", "scheme_column",
                )
            ),
        )
    )
    ADAPTERS.register(
        AdapterSpec(
            "index_universe",
            _index_loader,
            frozenset(("index_universe",)),
            _COMMON_OPTIONS | frozenset(("index_code", "index_name", "seq_col", "sheet_name")),
        )
    )
    ADAPTERS.register(
        AdapterSpec(
            "trade_status",
            _trade_status_loader,
            frozenset(("trade_status",)),
            _COMMON_OPTIONS | frozenset(("layout", "normal_values", "sheet_name")),
        )
    )
    ADAPTERS.register(
        AdapterSpec(
            "trade_calendar",
            _trade_calendar_loader,
            frozenset(("trade_calendar",)),
            _COMMON_OPTIONS | frozenset(("sheet_name",)),
        )
    )


_register_defaults()


def infer_adapter(path: str | Path) -> str:
    """Conservatively infer an adapter without treating index prices as constituents."""

    name = Path(path).name.lower()
    if "交易状态" in name or "trade_status" in name:
        return "trade_status"
    if "交易日" in name or "trade_calendar" in name:
        return "trade_calendar"
    if "成分" in name or "constituent" in name or "index_universe" in name:
        return "index_universe"
    if "行业" in name or "industry" in name:
        return "industry"
    if name.startswith("ede") or "_ede" in name:
        return "ede"
    return "auto"


def load_import_batches(
    import_type: str | None,
    path: str | Path,
    *,
    table: str,
    options: Mapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
    strict_options: bool = False,
) -> Iterator[ImportBatch]:
    """Load a source as bounded, typed import batches."""

    resolved_options = dict(options or {})
    adapter_name = import_type or infer_adapter(path)
    spec = ADAPTERS.validate(
        adapter_name,
        table,
        resolved_options,
        strict_options=strict_options,
    )
    chunk_size = int(resolved_options.get("chunk_size", DEFAULT_CHUNK_SIZE))
    context = AdapterContext(
        path=Path(path).expanduser().resolve(),
        table=table,
        options=resolved_options,
        chunk_size=max(1, chunk_size),
        logger=logger or logging.getLogger("betalens_db_manager.import_adapters"),
    )
    yield from spec.loader(context)


def collect_import_batches(batches: Iterable[ImportBatch]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize batches for the legacy DataFrame API."""

    batch_list = list(batches)
    frames = [batch.frame for batch in batch_list if not batch.frame.empty]
    rejected = [batch.rejected for batch in batch_list if not batch.rejected.empty]
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DB_COLUMNS),
        pd.concat(rejected, ignore_index=True) if rejected else pd.DataFrame(columns=["_source_row", "_errors"]),
    )


__all__ = [
    "ADAPTERS",
    "AdapterRegistry",
    "AdapterSpec",
    "BatchKind",
    "ImportBatch",
    "IndexSnapshotBatch",
    "IndustryBatch",
    "MarketBatch",
    "ObservationBatch",
    "TradeStatusBatch",
    "TradeCalendarBatch",
    "collect_import_batches",
    "infer_adapter",
    "load_import_batches",
]
