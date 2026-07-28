"""Local CSV/Excel readers and market timestamp alignment."""

from __future__ import annotations

import codecs
import gzip
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd


DEFAULT_ENCODINGS = (
    "utf-8",
    "utf-8-sig",
    "gb18030",
    "gbk",
    "gb2312",
    "cp936",
    "big5",
    "latin1",
)
AUTO_DETECT_ENCODINGS = tuple(encoding for encoding in DEFAULT_ENCODINGS if encoding != "latin1")
DEFAULT_OPEN_METRICS = frozenset(("开盘价", "开盘价(元)", "前收盘价"))
DEFAULT_CHUNK_SIZE = 100_000


def _logger(logger: logging.Logger | None) -> logging.Logger:
    return logger or logging.getLogger("betalens_db_manager.adapters.files")


def read_csv_with_encoding(
    filepath: str | Path,
    *,
    encodings: Iterable[str] | None = None,
    logger: logging.Logger | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a CSV, trying common Chinese encodings when none is specified."""

    path = Path(filepath)
    log = _logger(logger)
    if "encoding" in kwargs:
        return pd.read_csv(path, **kwargs)

    attempted = tuple(encodings or DEFAULT_ENCODINGS)
    last_error: UnicodeError | None = None
    for encoding in attempted:
        try:
            frame = pd.read_csv(path, encoding=encoding, **kwargs)
            log.info("read CSV %s with encoding %s", path, encoding)
            return frame
        except UnicodeError as exc:
            last_error = exc

    message = f"无法读取 CSV {path}，已尝试编码: {', '.join(attempted)}"
    if last_error is not None:
        raise UnicodeError(message) from last_error
    raise UnicodeError(message)


def read_file(
    filepath: str | Path,
    *,
    encodings: Iterable[str] | None = None,
    logger: logging.Logger | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a supported local source file into a DataFrame.

    CSV, XLS and XLSX are deliberately the only accepted formats. Import jobs
    validate and normalize the returned frame before any database transaction.
    """

    path = Path(filepath).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv" or path.name.lower().endswith(".csv.gz"):
        return read_csv_with_encoding(
            path,
            encodings=encodings,
            logger=logger,
            **kwargs,
        )
    if suffix in {".xls", ".xlsx"}:
        if suffix == ".xlsx" and "engine" not in kwargs:
            kwargs["engine"] = "openpyxl"
        frame = pd.read_excel(path, **kwargs)
        _logger(logger).info("read Excel source %s", path)
        return frame
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"不支持的文件格式: {suffix or '<none>'}")


def _detect_csv_encoding(path: Path, encodings: Iterable[str] | None = None) -> str:
    # latin1 accepts every byte sequence, so using it as an automatic fallback
    # silently corrupts Chinese headers when a sampled multibyte sequence is
    # incomplete.  Callers can still request it explicitly through read_options.
    candidates = tuple(encodings or AUTO_DETECT_ENCODINGS)
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rb") as handle:
        sample = handle.read(1024 * 1024)
    for encoding in candidates:
        try:
            # A fixed-size sample can end in the middle of a GBK/UTF-8
            # character.  Incremental decoding validates the complete prefix
            # without treating that unfinished final character as corruption.
            codecs.getincrementaldecoder(encoding)(errors="strict").decode(sample, final=False)
            return encoding
        except UnicodeError:
            continue
    raise UnicodeError(f"无法识别 CSV 编码: {path}; 已尝试 {', '.join(candidates)}")


def iter_file_chunks(
    filepath: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    read_options: dict[str, Any] | None = None,
    encodings: Iterable[str] | None = None,
    logger: logging.Logger | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield bounded DataFrames from CSV/CSV.GZ/Parquet or one Excel frame."""

    path = Path(filepath).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    options = dict(read_options or {})
    size = max(1, int(chunk_size))
    lower_name = path.name.lower()

    if path.suffix.lower() == ".csv" or lower_name.endswith(".csv.gz"):
        options.pop("chunksize", None)
        encoding = options.pop("encoding", None) or _detect_csv_encoding(path, encodings)
        reader = pd.read_csv(
            path,
            encoding=encoding,
            compression="infer",
            chunksize=size,
            **options,
        )
        yield from reader
        return

    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("分块读取 Parquet 需要安装 pyarrow") from exc
        columns = options.pop("columns", None)
        if options:
            raise ValueError(f"Parquet 分块读取不支持参数: {sorted(options)}")
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=size, columns=columns):
            yield batch.to_pandas()
        return

    if path.suffix.lower() in {".xls", ".xlsx"}:
        yield read_file(path, logger=logger, **options)
        return

    raise ValueError(f"不支持的文件格式: {path.suffix or '<none>'}")


def apply_time_alignment(
    df: pd.DataFrame,
    date_column: str = "日期",
    metric_column: str = "metric",
    open_metric_names: Iterable[str] | None = None,
    open_time: str = "09:30:01",
    other_time: str = "15:00:01",
    inplace: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Align daily observations to their first known market timestamp."""

    if date_column not in df.columns:
        raise ValueError(f"日期列不存在: {date_column}")

    frame = df if inplace else df.copy()
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    invalid = dates.isna() & frame[date_column].notna()
    if invalid.any():
        samples = frame.loc[invalid, date_column].astype(str).head(5).tolist()
        raise ValueError(f"日期列包含无法解析的值: {samples}")

    if metric_column in frame.columns:
        open_mask = frame[metric_column].isin(set(open_metric_names or DEFAULT_OPEN_METRICS))
    else:
        open_mask = pd.Series(False, index=frame.index)
        _logger(logger).warning("指标列 %s 不存在，全部按收盘时间对齐", metric_column)

    open_delta = pd.to_timedelta(open_time)
    other_delta = pd.to_timedelta(other_time)
    frame[date_column] = dates.dt.normalize() + other_delta
    frame.loc[open_mask, date_column] = dates.loc[open_mask].dt.normalize() + open_delta
    return frame
