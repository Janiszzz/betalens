"""Parser for Wind EDE wide-table exports."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .files import read_file


DEFAULT_DATE_PATTERN = r"(\d{8})"
DEFAULT_TIME = "15:30:00"
DEFAULT_CODE_COLUMNS = (
    "证券代码",
    "code",
    "windcode",
    "代码",
    "Code",
    "WindCode",
)
DEFAULT_NAME_COLUMNS = (
    "证券简称",
    "name",
    "sec_name",
    "简称",
    "名称",
    "Name",
    "SecName",
)
DEFAULT_REMOVE_KEYWORDS = (
    "数据来源",
    "Wind",
    "来源:",
    "注:",
    "说明:",
    "Source:",
    "Note:",
    "Remark:",
)


def _logger(logger: logging.Logger | None) -> logging.Logger:
    return logger or logging.getLogger("betalens_db_manager.adapters.ede")


def extract_date_from_filename(
    filepath: str | Path,
    pattern: str = DEFAULT_DATE_PATTERN,
    default_time: str = DEFAULT_TIME,
    logger: logging.Logger | None = None,
) -> str | None:
    """Extract an eight-digit date from a source filename."""

    match = re.search(pattern, Path(filepath).stem)
    if match is None:
        _logger(logger).warning("未在文件名中找到日期: %s", Path(filepath).name)
        return None
    try:
        parsed = datetime.strptime(match.group(1), "%Y%m%d")
    except (IndexError, ValueError):
        _logger(logger).warning("文件名中的日期无效: %s", match.group(0))
        return None
    return f"{parsed:%Y-%m-%d} {default_time}"


def parse_metric_column(
    column_name: str,
    logger: logging.Logger | None = None,
) -> tuple[str, dict[str, str]]:
    """Split an EDE metric header into its metric name and metadata."""

    text = str(column_name).strip()
    brackets = re.findall(r"\[([^\]]+)\]", text)
    parts = [part.strip() for part in re.sub(r"\[[^\]]+\]", "", text).split() if part.strip()]
    if not parts:
        _logger(logger).warning("无法解析指标列: %s", text)
        return text, {"原始列名": text}

    metadata = {"原始列名": text}
    if brackets:
        metadata["日期说明"] = brackets[0]
    if len(parts) >= 2:
        metadata["值类型"] = parts[1]
    if len(brackets) >= 2:
        metadata["单位说明"] = brackets[1]
    if len(parts) >= 3:
        metadata["单位"] = parts[2]
    return parts[0], metadata


def extract_date_from_metric_metadata(
    metadata: dict[str, str],
    column_name: str,
    default_time: str = DEFAULT_TIME,
    logger: logging.Logger | None = None,
) -> str | None:
    """Extract an eight-digit effective date embedded in an EDE header."""

    candidates = (metadata.get("日期说明", ""), str(column_name))
    for candidate in candidates:
        match = re.search(r"(\d{8})", candidate)
        if match is None:
            continue
        try:
            parsed = datetime.strptime(match.group(1), "%Y%m%d")
        except ValueError:
            continue
        return f"{parsed:%Y-%m-%d} {default_time}"
    _logger(logger).debug("指标列中没有可用日期: %s", column_name)
    return None


def clean_ede_dataframe(
    df: pd.DataFrame,
    keywords_to_remove: Iterable[str] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Remove empty rows/columns and EDE footer or source-note rows."""

    frame = df.dropna(how="all").dropna(axis=1, how="all")
    for keyword in keywords_to_remove or DEFAULT_REMOVE_KEYWORDS:
        remove = pd.Series(False, index=frame.index)
        for column in frame.select_dtypes(include=("object", "string")).columns:
            remove |= frame[column].astype(str).str.contains(
                str(keyword), case=False, na=False, regex=False
            )
        if remove.any():
            _logger(logger).info("删除包含 %r 的 EDE 尾注行: %d", keyword, int(remove.sum()))
            frame = frame.loc[~remove]
    return frame.reset_index(drop=True)


def _find_column(columns: Iterable[object], candidates: Iterable[str]) -> object | None:
    values = list(columns)
    by_casefold = {str(value).strip().casefold(): value for value in values}
    for candidate in candidates:
        matched = by_casefold.get(str(candidate).strip().casefold())
        if matched is not None:
            return matched
    return None


def identify_code_name_columns(
    df: pd.DataFrame,
    code_column_names: Iterable[str] | None = None,
    name_column_names: Iterable[str] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[object | None, object | None]:
    """Identify security code and name columns in an EDE frame."""

    code_col = _find_column(df.columns, code_column_names or DEFAULT_CODE_COLUMNS)
    name_col = _find_column(df.columns, name_column_names or DEFAULT_NAME_COLUMNS)
    if code_col is None and len(df.columns):
        code_col = df.columns[0]
        _logger(logger).warning("未识别标准代码列，使用第一列: %s", code_col)
    if name_col is None and len(df.columns) > 1:
        name_col = df.columns[1]
        _logger(logger).warning("未识别标准名称列，使用第二列: %s", name_col)
    return code_col, name_col


def process_ede_file(
    filepath: str | Path,
    date_from: str = "filename",
    default_datetime: str | None = None,
    code_column_names: Iterable[str] | None = None,
    name_column_names: Iterable[str] | None = None,
    logger: logging.Logger | None = None,
    *,
    date_pattern: str = DEFAULT_DATE_PATTERN,
    default_time: str = DEFAULT_TIME,
    keywords_to_remove: Iterable[str] | None = None,
) -> tuple[pd.DataFrame | None, list[dict[str, object]]]:
    """Convert one EDE export to the standard six-column import boundary."""

    if date_from not in {"filename", "metric"}:
        raise ValueError("date_from 必须是 'filename' 或 'metric'")

    log = _logger(logger)
    errors: list[dict[str, object]] = []
    try:
        frame = clean_ede_dataframe(
            read_file(filepath, logger=log),
            keywords_to_remove=keywords_to_remove,
            logger=log,
        )
        if frame.empty:
            return None, [{"type": "empty_file", "message": "文件为空或清理后无数据"}]

        code_col, name_col = identify_code_name_columns(
            frame,
            code_column_names=code_column_names,
            name_column_names=name_column_names,
            logger=log,
        )
        if code_col is None:
            return None, [{"type": "missing_code_column", "message": "未找到代码列"}]

        file_datetime = None
        if date_from == "filename":
            file_datetime = extract_date_from_filename(
                filepath,
                pattern=date_pattern,
                default_time=default_time,
                logger=log,
            )
        file_datetime = file_datetime or default_datetime

        key_columns = {code_col, name_col}
        metric_columns = [column for column in frame.columns if column not in key_columns]
        if not metric_columns:
            return None, [{"type": "no_metric_columns", "message": "未找到指标列"}]

        parts: list[pd.DataFrame] = []
        for metric_col in metric_columns:
            metric_name, metadata = parse_metric_column(str(metric_col), logger=log)
            metric_datetime = None
            if date_from == "metric" or file_datetime is None:
                metric_datetime = extract_date_from_metric_metadata(
                    metadata,
                    str(metric_col),
                    default_time=default_time,
                    logger=log,
                )
            available_at = metric_datetime or file_datetime
            if available_at is None:
                errors.append(
                    {
                        "type": "missing_datetime",
                        "column": str(metric_col),
                        "message": "无法确定日期",
                    }
                )
                continue

            columns = [code_col, metric_col]
            if name_col is not None:
                columns.insert(1, name_col)
            long = frame.loc[:, columns].copy()
            rename = {code_col: "code", metric_col: "value"}
            if name_col is not None:
                rename[name_col] = "name"
            long = long.rename(columns=rename)
            if "name" not in long:
                long["name"] = long["code"]
            long["value"] = pd.to_numeric(
                long["value"].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )
            long = long.loc[long["value"].notna()]
            if long.empty:
                continue
            long["metric"] = metric_name
            long["datetime"] = pd.Timestamp(available_at)
            long["remark"] = [dict(metadata) for _ in range(len(long))]
            parts.append(long[["datetime", "code", "name", "metric", "value", "remark"]])

        if not parts:
            errors.append({"type": "no_valid_data", "message": "没有有效数据"})
            return None, errors

        result = pd.concat(parts, ignore_index=True)
        result["code"] = result["code"].astype(str).str.strip()
        result["name"] = result["name"].fillna(result["code"]).astype(str).str.strip()
        result = result.sort_values(["datetime", "code", "metric"], kind="stable")
        return result.reset_index(drop=True), errors
    except Exception as exc:
        log.exception("处理 EDE 文件失败: %s", filepath)
        errors.append({"type": "processing_error", "message": str(exc)})
        return None, errors
