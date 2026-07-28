"""Read-only PostgreSQL access for Betalens research workflows."""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from typing import Any

import pandas as pd
import psycopg2.extras
from psycopg2 import sql as psql

from .pool import get_read_pool
from .query import (
    _normalized_schema_available,
    build_query as _build_query,
    get_available_dates as _get_available_dates,
    get_latest_date as _get_latest_date,
    query_nearest_after as _query_nearest_after,
    query_nearest_before as _query_nearest_before,
    query_nearest_in_range_after as _query_nearest_in_range_after,
    query_nearest_in_range_before as _query_nearest_in_range_before,
    query_time_range as _query_time_range,
    query_trade_status as _query_trade_status,
)
from .registry import get_dataset


class Datafeed:
    """Logical, read-only data access facade.

    ``table_name`` remains the legacy logical dataset name. The query module
    routes it to the normalized ``betalens`` schema when installed and falls
    back to a legacy read-only relation during rollout.
    """

    def __init__(
        self,
        table_name: str,
        db_config: dict[str, Any] | None = None,
        log_dir: str | None = None,
    ) -> None:
        del log_dir  # File logging no longer belongs to each query object.
        self.sheet = str(table_name)
        if get_dataset(self.sheet) is None:
            raise ValueError(f"不支持的逻辑数据集: {self.sheet}")
        self.logger = logging.getLogger("betalens.datafeed")
        pool_min = int((db_config or {}).get("pool_min_connections", 1))
        pool_max = int((db_config or {}).get("pool_max_connections", 10))
        timeout = int((db_config or {}).get("statement_timeout_ms", 120_000))
        self._pool = get_read_pool(
            db_config,
            min_connections=pool_min,
            max_connections=pool_max,
            statement_timeout_ms=timeout,
        )
        self.conn = self._pool.acquire()
        self._cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        self._closed = False

    @property
    def cursor(self):
        """Deprecated read-only cursor retained for third-party compatibility."""
        warnings.warn(
            "Datafeed.cursor 已弃用；请使用 Datafeed 的结构化批量查询方法。",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._cursor

    def __enter__(self) -> "Datafeed":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def run_query(
        self,
        conditions: list[str] | None = None,
        params: list[Any] | None = None,
        select_columns: str = "*",
    ) -> pd.DataFrame:
        """Execute a legacy SELECT on the logical relation.

        This compatibility method is deprecated. The pooled connection is
        read-only, so callers cannot use it to mutate the database.
        """
        warnings.warn(
            "Datafeed.run_query() 已弃用；请使用结构化查询方法。",
            DeprecationWarning,
            stacklevel=2,
        )
        query, bound = _build_query(
            table_name=self.sheet,
            conditions=conditions,
            params=params,
            select_columns=select_columns,
        )
        self._cursor.execute(query, bound)
        return pd.DataFrame(self._cursor.fetchall())

    def query_time_range(
        self,
        codes: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        metric: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        return _query_time_range(
            cursor=self._cursor,
            table_name=self.sheet,
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            metric=metric,
            limit=limit,
            logger=self.logger,
        )

    def query_trade_status(self, params: dict[str, Any] | None = None) -> pd.DataFrame:
        params = params or {}
        if "dates" not in params:
            raise ValueError("必须提供参数: ['dates']")
        return _query_trade_status(
            cursor=self._cursor,
            table_name=self.sheet,
            codes=params.get("codes"),
            dates=params["dates"],
            metric=params.get("metric", "交易状态"),
            logger=self.logger,
        )

    def query_nearest_after(self, params: dict[str, Any] | None = None) -> pd.DataFrame:
        params = self._require(params, ("codes", "datetimes", "metric"))
        return _query_nearest_after(
            cursor=self._cursor,
            table_name=self.sheet,
            codes=params["codes"],
            datetimes=params["datetimes"],
            metric=params["metric"],
            time_tolerance=params.get("time_tolerance"),
            logger=self.logger,
        )

    def query_nearest_before(self, params: dict[str, Any] | None = None) -> pd.DataFrame:
        params = self._require(params, ("codes", "datetimes", "metric"))
        return _query_nearest_before(
            cursor=self._cursor,
            table_name=self.sheet,
            codes=params["codes"],
            datetimes=params["datetimes"],
            metric=params["metric"],
            time_tolerance=params.get("time_tolerance"),
            logger=self.logger,
        )

    def query_nearest_in_range_after(
        self, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        params = self._require(params, ("codes", "ranges", "metric"))
        return _query_nearest_in_range_after(
            cursor=self._cursor,
            table_name=self.sheet,
            codes=params["codes"],
            ranges=params["ranges"],
            metric=params["metric"],
            time_tolerance=params.get("time_tolerance"),
            logger=self.logger,
        )

    def query_nearest_in_range_before(
        self, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        params = self._require(params, ("codes", "ranges", "metric"))
        return _query_nearest_in_range_before(
            cursor=self._cursor,
            table_name=self.sheet,
            codes=params["codes"],
            ranges=params["ranges"],
            metric=params["metric"],
            time_tolerance=params.get("time_tolerance"),
            logger=self.logger,
        )

    def get_latest_date(self, code: str | None = None, metric: str | None = None):
        return _get_latest_date(
            cursor=self._cursor,
            table_name=self.sheet,
            code=code,
            metric=metric,
            logger=self.logger,
        )

    def get_available_dates(
        self,
        code: str,
        metric: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        return _get_available_dates(
            cursor=self._cursor,
            table_name=self.sheet,
            code=code,
            metric=metric,
            start_date=start_date,
            end_date=end_date,
            logger=self.logger,
        )

    def query_industry(
        self,
        codes: list[str],
        dates,
        scheme: str = "申万一级行业",
        *,
        exact: bool = False,
    ) -> pd.DataFrame:
        """Return point-in-time industry memberships for code/date inputs."""
        from .industry import query_industry

        return query_industry(
            self._cursor,
            codes=codes,
            dates=dates,
            scheme=scheme,
            table_name=self.sheet,
            exact=exact,
            logger=self.logger,
        )

    def get_industry_members(
        self,
        industry,
        date: str,
        scheme: str = "申万一级行业",
        *,
        by: str = "name",
        exact: bool = False,
    ) -> pd.DataFrame:
        """Return members of one industry at a point in time."""
        from .industry import get_industry_members

        return get_industry_members(
            self._cursor,
            industry=industry,
            date=date,
            scheme=scheme,
            table_name=self.sheet,
            by=by,
            exact=exact,
            logger=self.logger,
        )

    def get_index_universe(self, index_code: str, date: str) -> list[str]:
        """Return the latest index constituent snapshot available at ``date``."""
        from .universe import get_index_universe

        return get_index_universe(
            self._cursor,
            index_code=index_code,
            date=date,
            table_name=self.sheet,
            logger=self.logger,
        )

    def get_index_universe_date(self, index_code: str, date: str):
        """Return the effective timestamp of an index constituent snapshot."""
        from .universe import get_index_universe_date

        return get_index_universe_date(
            self._cursor,
            index_code=index_code,
            date=date,
            table_name=self.sheet,
            logger=self.logger,
        )

    def query_names(self, codes: list[str]) -> pd.DataFrame:
        """Resolve current names for a batch of entity codes."""
        unique_codes = [str(code) for code in dict.fromkeys(codes) if code]
        if not unique_codes:
            return pd.DataFrame(columns=["code", "name"])
        if _normalized_schema_available(self._cursor):
            self._cursor.execute(
                """
                SELECT code, current_name AS name
                FROM betalens.entity_dim
                WHERE code = ANY(%s::text[]) AND current_name <> ''
                ORDER BY code
                """,
                (unique_codes,),
            )
        else:
            query = psql.SQL(
                """
                SELECT DISTINCT ON (code) code, name
                FROM public.{}
                WHERE code = ANY(%s::text[]) AND name IS NOT NULL
                ORDER BY code, datetime DESC
                """
            ).format(psql.Identifier(self.sheet))
            self._cursor.execute(query, (unique_codes,))
        return pd.DataFrame(self._cursor.fetchall(), columns=["code", "name"])

    @staticmethod
    def _require(params: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
        params = params or {}
        if not all(key in params for key in keys):
            raise ValueError(f"必须提供参数: {list(keys)}")
        return params

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._cursor.close()
        finally:
            self._pool.release(self.conn)
            self._cursor = None
            self.conn = None


def get_absolute_trade_days(begin_date, end_date, period, use_pmc=True):
    """Return China A-share trading dates, optionally sampled by period end."""
    period_map = {"D": None, "W": "W", "M": "M", "Q": "Q", "S": "2Q", "Y": "Y"}
    period_key = str(period).upper()
    if period_key not in period_map:
        raise ValueError(f"不支持的 period: {period}")
    freq = period_map[period_key]
    if use_pmc:
        import pandas_market_calendars as mcal

        schedule = mcal.get_calendar("XSHG").schedule(
            start_date=begin_date, end_date=end_date, tz="Asia/Shanghai"
        )
        dates = pd.to_datetime(schedule.index).tz_localize(None).to_series().reset_index(drop=True)
    else:
        import akshare as ak

        frame = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(frame["trade_date"])
        mask = (dates >= pd.Timestamp(begin_date)) & (dates <= pd.Timestamp(end_date))
        dates = dates[mask].sort_values().reset_index(drop=True)
    if freq:
        dates = dates.groupby(dates.dt.to_period(freq)).last().reset_index(drop=True)
    return [value.date() for value in dates]


def trade_days_offset(begin_datetime, offset, period="D"):
    """Offset a date/time by exchange trading periods."""
    import akshare as ak

    original = pd.Timestamp(begin_datetime)
    frame = ak.tool_trade_date_hist_sina()
    days = pd.DatetimeIndex(pd.to_datetime(frame["trade_date"]).sort_values().unique())
    if str(period).upper() != "D":
        periods = pd.Series(days).groupby(pd.Series(days).dt.to_period(str(period).upper())).last()
        days = pd.DatetimeIndex(periods.to_numpy())
    insertion = int(days.searchsorted(original.normalize(), side="left"))
    target_index = insertion + int(offset)
    if target_index < 0 or target_index >= len(days):
        raise IndexError("交易日偏移超出可用日历范围")
    target = pd.Timestamp(days[target_index])
    return datetime.combine(target.date(), original.time())
