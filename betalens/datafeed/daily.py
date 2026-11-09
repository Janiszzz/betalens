"""Daily market observations retain business dates and source availability."""
from __future__ import annotations

import pandas as pd

from .contracts import available, local_time, timestamp, VersionConflictError
from .query import _codes, _query_fields
from .readers import get_reader, require_columns


def query_daily_market(source, table_name, codes=None, start_date=None, end_date=None,
                       metrics=("close",), *, as_of=None, filters=None):
    reader = get_reader(source)
    policy = getattr(source, "pit_policy", "research")
    metrics = [metrics] if isinstance(metrics, str) else list(dict.fromkeys(metrics))
    if not metrics:
        raise ValueError('metrics 至少需要一个物理价格字段')
    codes = _codes(codes)
    fields, predicates = {}, []
    with reader.snapshot():
        require_columns(reader, table_name, {"trade_date": "date", "asset_type": "text"})
        for metric in metrics:
            expected, predicates = _query_fields(reader, table_name, metric, filters)
            fields.update(expected)
        fields.update(trade_date="date", asset_type="text")
        if codes is not None:
            predicates.append(("security_code", "IN", codes))
        for value, operator in ((start_date, ">="), (end_date, "<=")):
            if value is not None:
                predicates.append(("trade_date", operator, local_time(timestamp(value)).date()))
        rows = reader.rows(table_name, list(fields), predicates)
    result, seen = [], {}
    for row in rows:
        if not available(row, as_of, policy):
            continue
        for metric in metrics:
            record = dict(code=row["security_code"], trade_date=pd.Timestamp(row["trade_date"]),
                          metric=metric, value=float(row[metric]) if row[metric] is not None else float("nan"),
                          available_at=local_time(row["available_at"]), asset_type=row["asset_type"],
                          availability_basis=row["availability_basis"], data_quality_status=row["data_quality_status"],
                          pit_policy=policy)
            key = (record["code"], record["trade_date"], metric)
            if key in seen:
                old = seen[key]
                if any(not ((pd.isna(old[k]) and pd.isna(v)) or old[k] == v) for k, v in record.items()):
                    raise VersionConflictError("同一业务日期存在冲突行情: " + str(key))
                continue
            seen[key] = record
            result.append(record)
    columns = ["trade_date", "code", "metric", "value", "available_at", "asset_type",
               "availability_basis", "data_quality_status", "pit_policy"]
    return pd.DataFrame(result, columns=columns).sort_values(["trade_date", "code", "metric"]).reset_index(drop=True)


def daily_wide(frame, metric):
    selected = frame.loc[frame["metric"].eq(metric)].copy()
    if selected.empty:
        return pd.DataFrame()
    wide = selected.pivot(index="trade_date", columns="code", values="value").sort_index()
    wide.index = pd.DatetimeIndex(wide.index).normalize()
    # Scalar metadata survives copies/JSON cache publication; no second value axis.
    upper = selected.groupby("trade_date")["available_at"].max()
    wide.attrs["availability_upper"] = {str(pd.Timestamp(d).date()): pd.Timestamp(t).isoformat() for d, t in upper.items()}
    return wide
