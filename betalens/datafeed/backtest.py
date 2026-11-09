"""Paired adjusted/raw daily bars for execution simulations.

Market timestamps belong to the simulator.  ``available_at`` remains the source
version timestamp and is never replaced by the opening/closing clock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import available, local_time, timestamp, VersionConflictError
from .readers import get_reader, identifier, require_columns
from .reference import calendar_days


def query_backtest_bars(source, table_name, codes, start_date, end_date=None,
                        metrics=("open", "close")):
    reader = get_reader(source)
    policy = getattr(source, "pit_policy", "research")
    metrics = list(dict.fromkeys([identifier(m) for m in metrics] + ["close"]))
    columns = {"security_code": "text", "asset_type": "text", "trade_date": "date",
               "adjustment_factor": "numeric", "available_at": "timestamp",
               "availability_basis": "text", "data_quality_status": "text"}
    columns.update({key: "numeric" for m in metrics for key in (m, "raw_" + m)})
    filters = [("security_code", "IN", list(codes)),
               ("trade_date", ">=", local_time(timestamp(start_date)).date())]
    if end_date is not None:
        filters.append(("trade_date", "<=", local_time(timestamp(end_date)).date()))
    with reader.snapshot():
        require_columns(reader, table_name, columns)
        rows = reader.rows(table_name, list(columns), filters)
        require_columns(reader, "stock_profile", {"security_code": "text", "market": "text"})
        profiles = reader.rows("stock_profile", ["security_code", "market"],
                               [("security_code", "IN", list(codes))])
    markets = {}
    for row in profiles:
        code, market = row["security_code"], row["market"]
        if code in markets and markets[code] != market:
            raise VersionConflictError("股票板块冲突: " + code)
        markets[code] = market
    records, selected = [], {}
    for row in rows:
        if not available(row, policy=policy):
            continue
        if row["asset_type"] != "stock":
            raise ValueError("整数成交目前仅支持沪深A股: " + row["security_code"])
        record = {"code": row["security_code"], "trade_date": pd.Timestamp(row["trade_date"]),
                  "market": markets.get(row["security_code"]),
                  "available_at": local_time(row["available_at"]),
                  "availability_basis": row["availability_basis"],
                  "data_quality_status": row["data_quality_status"], "pit_policy": policy}
        factor = float(row["adjustment_factor"]) if row["adjustment_factor"] is not None else np.nan
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("后复权行情缺少有效复权因子: " + record["code"])
        record["adjustment_factor"] = factor
        for metric in metrics:
            adjusted, raw = row[metric], row["raw_" + metric]
            if adjusted is None and raw is None:
                record[metric] = record["raw_" + metric] = np.nan
                continue
            if adjusted is None or raw is None:
                raise ValueError("后复权价格与原始价格必须成对: " + metric)
            adjusted, raw = float(adjusted), float(raw)
            if not (np.isfinite(adjusted) and np.isfinite(raw) and adjusted > 0 and raw > 0):
                raise ValueError("回测价格必须为有限正数: " + metric)
            if not np.isclose(adjusted, raw * factor, rtol=1e-8, atol=1e-10):
                raise ValueError("后复权价格、原始价和复权因子不一致: " + record["code"])
            record[metric], record["raw_" + metric] = adjusted, raw
        key = (record["code"], record["trade_date"])
        if key in selected:
            old = selected[key]
            same = all((pd.isna(old[k]) and pd.isna(v)) or old[k] == v for k, v in record.items())
            if not same:
                raise VersionConflictError("同一证券交易日存在冲突行情: " + str(key))
            continue
        selected[key] = record
        records.append(record)
    result_columns = ["code", "trade_date", "market", "adjustment_factor", "available_at",
                      "availability_basis", "data_quality_status", "pit_policy"]
    result_columns += [key for m in metrics for key in (m, "raw_" + m)]
    return pd.DataFrame(records, columns=result_columns).sort_values(["trade_date", "code"]).reset_index(drop=True)


def query_backtest_calendar(source, start_date, end_date):
    reader = get_reader(source)
    with reader.snapshot():
        return calendar_days(reader, timestamp(start_date), timestamp(end_date),
                             table_name="trade_calendar_versions" if source.pit_policy == "strict" else "trade_calendar",
                             pit_policy=source.pit_policy)
