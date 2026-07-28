"""Optional WindPy market-data source adapter."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import pandas as pd

from .files import apply_time_alignment


ASSET_FIELDS = {
    "stock": (
        ("open", "high", "low", "close", "volume", "amt", "turn"),
        ("开盘价(元)", "最高价(元)", "最低价(元)", "收盘价(元)", "成交量(股)", "成交额(元)", "换手率(%)"),
    ),
    "index": (
        ("open", "high", "low", "close", "volume", "amt"),
        ("开盘价", "最高价", "最低价", "收盘价", "成交量", "成交额"),
    ),
    "fund": (
        ("open", "high", "low", "close", "volume", "amt"),
        ("开盘价(元)", "最高价(元)", "最低价(元)", "收盘价(元)", "成交量(份)", "成交额(元)"),
    ),
    "bond": (
        ("open", "high", "low", "close", "volume", "amt"),
        ("开盘价(元)", "最高价(元)", "最低价(元)", "收盘价(元)", "成交量(手)", "成交额(元)"),
    ),
}


def _wind_client(client: Any | None) -> Any:
    if client is not None:
        return client
    try:
        from WindPy import w
    except ImportError as exc:
        raise ImportError("Wind 数据源需要本机 WindPy 和已登录的 Wind 终端") from exc
    return w


def fetch_daily_market(
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    fields: Iterable[str] | None = None,
    asset_type: str = "stock",
    apply_time_stamps: bool = True,
    logger: logging.Logger | None = None,
    *,
    field_names: Iterable[str] | None = None,
    client: Any | None = None,
) -> pd.DataFrame:
    """Fetch daily Wind data and return the standard six-column long frame."""

    log = logger or logging.getLogger("betalens_db_manager.adapters.wind")
    wind = _wind_client(client)
    started = wind.start()
    if getattr(started, "ErrorCode", 0) not in (0, None):
        raise RuntimeError(f"Wind 连接失败: {getattr(started, 'ErrorCode', None)}")

    if asset_type not in ASSET_FIELDS:
        raise ValueError(f"未知资产类型: {asset_type}")
    default_fields, default_names = ASSET_FIELDS[asset_type]
    selected_fields = tuple(fields or default_fields)
    default_mapping = dict(zip(default_fields, default_names))
    selected_names = tuple(
        field_names or (default_mapping.get(field, field) for field in selected_fields)
    )
    if len(selected_fields) != len(selected_names):
        raise ValueError("fields 与 field_names 长度必须一致")

    frames: list[pd.DataFrame] = []
    for raw_code in codes:
        code = str(raw_code).strip()
        result = wind.wsd(code, list(selected_fields), start_date, end_date, "")
        if getattr(result, "ErrorCode", -1) != 0:
            log.warning("Wind 获取 %s 失败: %s", code, getattr(result, "Data", None))
            continue
        name_result = wind.wss(code, "sec_name")
        name = code
        if getattr(name_result, "ErrorCode", -1) == 0 and getattr(name_result, "Data", None):
            try:
                name = name_result.Data[0][0] or code
            except (IndexError, TypeError):
                pass
        data: dict[str, object] = {"日期": result.Times, "code": code, "name": name}
        for index, metric_name in enumerate(selected_names):
            data[metric_name] = result.Data[index]
        frames.append(pd.DataFrame(data))

    columns = ["datetime", "code", "name", "metric", "value", "remark"]
    if not frames:
        return pd.DataFrame(columns=columns)

    wide = pd.concat(frames, ignore_index=True)
    long = wide.melt(
        id_vars=["日期", "code", "name"],
        value_vars=list(selected_names),
        var_name="metric",
        value_name="value",
    )
    if apply_time_stamps:
        long = apply_time_alignment(
            long,
            date_column="日期",
            metric_column="metric",
            inplace=True,
            logger=log,
        )
    else:
        long["日期"] = pd.to_datetime(long["日期"]).dt.normalize() + pd.Timedelta(hours=15, seconds=1)
    long = long.rename(columns={"日期": "datetime"}).dropna(subset=["value"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long["remark"] = None
    return long[columns].reset_index(drop=True)


def fetch_daily_index(
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    fields: Iterable[str] | None = None,
    apply_time_stamps: bool = True,
    logger: logging.Logger | None = None,
    *,
    field_names: Iterable[str] | None = None,
    client: Any | None = None,
) -> pd.DataFrame:
    """Fetch daily index observations."""

    return fetch_daily_market(
        codes,
        start_date,
        end_date,
        fields=fields,
        asset_type="index",
        apply_time_stamps=apply_time_stamps,
        logger=logger,
        field_names=field_names,
        client=client,
    )


def fetch_daily_fund(
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    fields: Iterable[str] | None = None,
    apply_time_stamps: bool = True,
    logger: logging.Logger | None = None,
    *,
    field_names: Iterable[str] | None = None,
    client: Any | None = None,
) -> pd.DataFrame:
    """Fetch daily fund observations."""

    return fetch_daily_market(
        codes,
        start_date,
        end_date,
        fields=fields,
        asset_type="fund",
        apply_time_stamps=apply_time_stamps,
        logger=logger,
        field_names=field_names,
        client=client,
    )


def fetch_daily_bond(
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    fields: Iterable[str] | None = None,
    apply_time_stamps: bool = True,
    logger: logging.Logger | None = None,
    *,
    field_names: Iterable[str] | None = None,
    client: Any | None = None,
) -> pd.DataFrame:
    """Fetch daily bond observations."""

    return fetch_daily_market(
        codes,
        start_date,
        end_date,
        fields=fields,
        asset_type="bond",
        apply_time_stamps=apply_time_stamps,
        logger=logger,
        field_names=field_names,
        client=client,
    )
