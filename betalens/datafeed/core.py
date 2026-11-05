"""Read-only, source-neutral business subscriptions for PostgreSQL/ClickHouse."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from . import query
from .config import load_subscription_config
from .contracts import frame, local_time, timestamp, integer, PITUnavailableError, VersionConflictError
from .readers import PostgresReader, ClickHouseReader, identifier, require_columns
from .reference import calendar_days, calendar_rows, load_reference


class Datafeed:
    def __init__(self, table_name, db_config=None, *, backend=None,
                 config_file=None):
        self.sheet = identifier(table_name)
        config = load_subscription_config(db_config, config_file=config_file, backend=backend)
        self.backend = config["backend"]
        self._reader = PostgresReader(config) if self.backend == "postgres" else ClickHouseReader(config)
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def query_time_range(self, codes=None, start_date=None, end_date=None, metric=None, limit=None, *, as_of=None, filters=None):
        return query.query_time_range(self, self.sheet, codes, start_date, end_date, metric, limit, as_of=as_of, filters=filters)

    def query_trade_status(self, params=None):
        params = self._require(params, ("dates",))
        return query.query_trade_status(self, self.sheet, params.get("codes"), params["dates"], params.get("metric", "交易状态"))

    def query_nearest_before(self, params=None):
        p = self._require(params, ("codes", "datetimes", "metric"))
        return query.query_nearest_before(self, self.sheet, p["codes"], p["datetimes"], p["metric"], p.get("time_tolerance"), filters=p.get("filters"))

    def query_nearest_after(self, params=None):
        p = self._require(params, ("codes", "datetimes", "metric"))
        return query.query_nearest_after(self, self.sheet, p["codes"], p["datetimes"], p["metric"], p.get("time_tolerance"), filters=p.get("filters"))

    def query_nearest_in_range_before(self, params=None):
        p = self._require(params, ("codes", "ranges", "metric"))
        return query.query_nearest_in_range_before(self, self.sheet, p["codes"], p["ranges"], p["metric"], p.get("time_tolerance"), filters=p.get("filters"))

    def query_nearest_in_range_after(self, params=None):
        p = self._require(params, ("codes", "ranges", "metric"))
        return query.query_nearest_in_range_after(self, self.sheet, p["codes"], p["ranges"], p["metric"], p.get("time_tolerance"), filters=p.get("filters"))

    def get_latest_date(self, code=None, metric=None, *, as_of=None, filters=None):
        return query.get_latest_date(self, self.sheet, code, metric, as_of=as_of, filters=filters)

    def get_available_dates(self, code, metric, start_date=None, end_date=None, *, as_of=None, filters=None):
        return query.get_available_dates(self, self.sheet, code, metric, start_date, end_date, as_of=as_of, filters=filters)

    def query_industry(self, codes, dates, scheme_code=None, *, level=1):
        from .industry import query_industry
        return query_industry(self, codes, dates, scheme_code, self.sheet, level=level)

    def get_industry_members(self, industry, date, scheme_code=None, *, by="name", level=1):
        from .industry import get_industry_members
        return get_industry_members(self, industry, date, scheme_code, self.sheet, by=by, level=level)

    def get_index_universe(self, index_code, date):
        from .universe import get_index_universe
        return get_index_universe(self, index_code, date, table_name=self.sheet)

    def get_index_universe_panel(self, index_code, dates):
        from .universe import get_index_universe_panel
        return get_index_universe_panel(self, index_code, dates, table_name=self.sheet)

    def get_index_universe_date(self, index_code, date):
        from .universe import get_index_universe_date
        return get_index_universe_date(self, index_code, date, table_name=self.sheet)

    def query_names(self, codes, *, as_of=None):
        cutoff = timestamp(as_of) if as_of is not None else None
        codes = list(dict.fromkeys(query._codes(codes, required=True)))
        if not codes:
            return frame([], ["code", "name"])
        with self._reader.snapshot():
            if as_of is not None:
                require_columns(self._reader, "security_name_history", {"name": "text"})
                history = query._name_rows(self._reader, codes)
                records = [{"code": code, "name": query.historical_name(history, code, cutoff)} for code in codes]
            else:
                require_columns(self._reader, "security_dim", {"security_code": "text", "name": "text"})
                rows = self._reader.rows("security_dim", ["security_code", "name"], [("security_code", "IN", codes)])
                names = {}
                for row in rows:
                    code, name = row["security_code"], row["name"]
                    if code in names and names[code] != name:
                        raise VersionConflictError("证券代码对应多个当前名称: " + code)
                    names[code] = name
                records = [{"code": code, "name": names.get(code)} for code in codes]
        return frame(records, ["code", "name"])

    @staticmethod
    def _require(params, keys):
        params = params or {}
        if not all(key in params for key in keys):
            raise ValueError(f"必须提供参数: {list(keys)}")
        return params

    def close(self):
        if not self._closed:
            self._reader.close()
            self._closed = True


def _period_days(days, period):
    key = period
    if not isinstance(key, str) or key not in {"D", "W", "M", "Q", "H", "Y"}:
        raise ValueError(f"不支持的 period: {period}")
    values = pd.Series(pd.to_datetime(days), dtype="datetime64[ns]")
    if key == "H":
        groups = values.dt.year * 2 + (values.dt.month - 1) // 6
        values = values.groupby(groups).last()
    elif key != "D":
        values = values.groupby(values.dt.to_period(key)).last()
    return [v.date() for v in values]


def get_absolute_trade_days(begin_date, end_date, period, exchange="SSE", *,
                            backend=None, config_file=None, db_config=None, as_of=None):
    _period_days([], period)
    begin_date, end_date = timestamp(begin_date), timestamp(end_date)
    if exchange not in {"SSE", "SZSE", "BSE"}:
        raise ValueError("exchange 必须是 SSE/SZSE/BSE")
    if begin_date > end_date:
        raise ValueError("开始日期不能晚于结束日期")
    with Datafeed("trade_calendar_versions", db_config, backend=backend, config_file=config_file) as data:
        with data._reader.snapshot():
            days = calendar_days(data._reader, begin_date, end_date, exchange, as_of)
    return _period_days(days, period)


def trade_periods_offset(begin_datetime, offset, period="D", *, exchange="SSE",
                      backend=None, config_file=None, db_config=None, as_of=None):
    """Shift by trading period ends; offset counts periods, not natural days."""
    if exchange not in {"SSE", "SZSE", "BSE"}:
        raise ValueError("exchange 必须是 SSE/SZSE/BSE")
    offset = integer(offset, "offset")
    _period_days([], period)
    original = local_time(timestamp(begin_datetime))
    cutoff = timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    with Datafeed("trade_calendar_versions", db_config, backend=backend, config_file=config_file) as data:
        with data._reader.snapshot():
            scopes = load_reference(data._reader, "calendar", exchange=exchange)
            rows = calendar_rows(scopes, cutoff)
    days = _period_days(sorted(pd.Timestamp(d).date() for d, row in rows.items() if row["is_open"] == 1), period)
    index = pd.DatetimeIndex(days)
    insertion = int(index.searchsorted(original.normalize(), side="left"))
    target = insertion + int(offset)
    if target < 0 or target >= len(index):
        raise IndexError("交易日偏移超出有证据的本地日历范围")
    selected = index[target]
    # A missing natural day could change an offset even if the target exists.
    first, last = min(original.normalize(), selected), max(original.normalize(), selected)
    key = period
    if key == "H":
        first = first.replace(month=1 if first.month <= 6 else 7, day=1)
        last = last.replace(month=6 if last.month <= 6 else 12, day=1) + pd.offsets.MonthEnd()
    elif key != "D":
        first = first.to_period(key).start_time.normalize()
        last = last.to_period(key).end_time.normalize()
    if any(day.date().isoformat() not in rows for day in pd.date_range(first, last)):
        raise PITUnavailableError("交易日偏移经过缺失或历史可知性不足的日历范围")
    return datetime.combine(selected.date(), original.time())
