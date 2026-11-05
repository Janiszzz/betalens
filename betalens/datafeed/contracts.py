"""Small, shared read contracts. No database management or inferred timestamps."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from decimal import Decimal
from numbers import Integral, Real

import numpy as np
import pandas as pd


class DatafeedError(RuntimeError):
    """A subscription cannot satisfy its declared contract."""


class SchemaError(DatafeedError):
    """A required relation, column or logical type is missing."""


class PITUnavailableError(DatafeedError):
    """Historical availability or complete coverage cannot be established."""


class VersionConflictError(DatafeedError):
    """Equally ranked records disagree; neither is chosen arbitrarily."""


# Positive attestations about the CONTENT VERSION, not merely the event date.
# Producers must supply the underlying evidence; adding a label cannot recover
# an overwritten history. Unknown/assumed/download timestamps are not accepted.
PROVEN_AVAILABILITY = frozenset({
    "source_version_publication_time", "verified_historical_version",
})
ACCEPTED_QUALITY = frozenset({"accepted", "verified"})
RANGE_COLUMNS = ["datetime", "code", "name", "metric", "value", "remark"]
STATUS_COLUMNS = ["code", "datetime", "value", "status_text", "name"]
INDUSTRY_COLUMNS = ["code", "query_date", "effective_dt", "sec_name",
                    "industry_value", "remark", "ind_name", "ind_code", "scheme"]

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:\d{2})?")


def integer(value, name, *, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} 必须是整数")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}")
    return result


def hours(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, Decimal)):
        raise ValueError("time_tolerance 必须是非负小时数")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError("time_tolerance 必须是非负小时数")
    return result


def timestamp(value, *, stored=False):
    """User naive times are Shanghai; typed database timestamps are UTC."""
    if isinstance(value, str):
        if not (_DATE.fullmatch(value) or _TIMESTAMP.fullmatch(value)):
            raise ValueError("时间必须使用 YYYY-MM-DD 或 ISO 日期时间，精度最多微秒")
    elif not isinstance(value, (date, datetime, pd.Timestamp, np.datetime64)):
        raise ValueError("时间必须是 ISO 字符串或日期时间对象，不能使用数字日期")
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("时间不能为空")
    if result.tzinfo is None:
        result = result.tz_localize("UTC" if stored else "Asia/Shanghai")
    if result.nanosecond:
        raise ValueError("时间精度最多微秒")
    return result.tz_convert("UTC")


def local_time(value):
    if value is None or pd.isna(value):
        return pd.NaT
    return timestamp(value, stored=True).tz_convert("Asia/Shanghai").tz_localize(None)


def date_only(value):
    return isinstance(value, date) and not isinstance(value, datetime) or (
        isinstance(value, str) and _DATE.fullmatch(value) is not None
    )


def calendar_date(value):
    if not date_only(value):
        raise ValueError("日期接口仅接受 YYYY-MM-DD 或 date；日内观察请使用单时点接口")
    return timestamp(value).tz_convert("Asia/Shanghai").date()


def bounds(start=None, end=None):
    lower = timestamp(start) if start is not None else None
    upper = timestamp(end) if end is not None else None
    if start is not None and end is not None and date_only(start) and date_only(end) and lower > upper:
        raise ValueError("开始时间不能晚于结束时间")
    exclusive = end is not None and date_only(end)
    if exclusive:
        upper += pd.Timedelta(days=1)
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("开始时间不能晚于结束时间")
    return lower, upper, exclusive


def proven(row, as_of=None):
    if row.get("availability_basis") not in PROVEN_AVAILABILITY:
        return False
    if row.get("available_at") is None:
        return False
    if row.get("data_quality_status") not in ACCEPTED_QUALITY:
        return False
    available = timestamp(row["available_at"], stored=True)
    return as_of is None or available <= timestamp(as_of)


def frame(records, columns, *, numeric=(), times=(), integers=()):
    """The same dtypes are used for populated, unmatched and empty results."""
    result = pd.DataFrame.from_records(records, columns=columns)
    for column in columns:
        if column in numeric:
            result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
        elif column in integers:
            result[column] = pd.array(result[column], dtype="Int8")
        elif column in times:
            result[column] = pd.to_datetime(result[column]).astype("datetime64[ns]")
        else:
            result[column] = result[column].astype(object).where(result[column].notna(), None)
    return result


def json_value(value):
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Decimal):
        text = format(value, "f")
        return (text.rstrip("0").rstrip(".") if "." in text else text) if value else "0"
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(type(value).__name__)


def digest(value):
    """Match the manager's immutable reference manifests (no manager import)."""
    return hashlib.sha256(json.dumps(_safe_json(value), ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=json_value,
                                    allow_nan=False).encode()).hexdigest()


def _safe_json(value):
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 9007199254740991:
        return str(value)
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return value


def document(value):
    result = json.loads(value) if isinstance(value, str) else value
    if not isinstance(result, dict):
        raise DatafeedError("数据库证据必须是 JSON 对象")
    return result
