"""Import source adapters and database writer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2.extras
import psycopg2.extensions
from psycopg2 import sql

from betalens.datafeed.ede_processor import process_ede_file
from betalens.datafeed.excel import apply_time_alignment, read_file

from .constants import ALLOWED_TABLES, DB_COLUMNS, INSERT_ONLY, UPSERT
from .db import DatabaseClient
from .utils import dataframe_summary


psycopg2.extensions.register_adapter(dict, psycopg2.extras.Json)


NORMAL_TRADE_STATUS = "交易"


def normalize_import_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "note" in out.columns and "remark" not in out.columns:
        out = out.rename(columns={"note": "remark"})
    if "remark" not in out.columns:
        out["remark"] = None
    out = out[[c for c in DB_COLUMNS if c in out.columns]]
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    if "value" in out.columns:
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def load_ede(path: str | Path, date_from: str = "filename", default_datetime: str | None = None, logger: logging.Logger | None = None) -> pd.DataFrame:
    df, errors = process_ede_file(path, date_from=date_from, default_datetime=default_datetime, logger=logger)
    if errors:
        raise ValueError(errors)
    return normalize_import_frame(df)


def load_wind_long(path: str | Path, logger: logging.Logger | None = None) -> pd.DataFrame:
    df = read_file(path, logger=logger)
    df = df.rename(columns={"代码": "code", "简称": "name", "日期": "date"})
    required = {"code", "name", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Wind 长格式文件缺少列: {sorted(missing)}")
    value_cols = [c for c in df.columns if c not in required]
    if not value_cols:
        raise ValueError("没有可识别的指标列")
    out = pd.melt(df, id_vars=["code", "name", "date"], value_vars=value_cols, var_name="metric", value_name="value")
    out = apply_time_alignment(out, date_column="date", metric_column="metric", logger=logger)
    out = out.rename(columns={"date": "datetime"})
    out["remark"] = None
    return normalize_import_frame(out)


def load_index_universe(path: str | Path, index_code: str, index_name: str, sheet_name: str = "Sheet2", seq_col: str = "序号") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    date_cols = [c for c in df.columns if c != seq_col]
    records: list[dict[str, Any]] = []
    for col in date_cols:
        eff_dt = pd.to_datetime(col)
        codes = [str(x).strip() for x in df[col].tolist() if pd.notna(x)]
        codes = list(dict.fromkeys(codes))
        if not codes:
            continue
        records.append(
            {
                "datetime": eff_dt,
                "code": index_code,
                "name": index_name,
                "metric": "universe",
                "value": len(codes),
                "remark": {"index_code": index_code, "index_name": index_name, "constituents": codes},
            }
        )
    return normalize_import_frame(pd.DataFrame(records))


def load_trade_status(path: str | Path, sheet_name: str = "Sheet1") -> pd.DataFrame:
    names_row = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=1)
    df = pd.read_excel(path, sheet_name=sheet_name, header=1)
    date_col = df.columns[0]
    code_cols = [c for c in list(df.columns[1:]) if pd.notna(c) and not str(c).startswith("Unnamed")]
    name_map = {code: names_row.iloc[0, i + 1] for i, code in enumerate(list(df.columns[1:]))}
    dates = pd.to_datetime(df[date_col])
    records: list[dict[str, Any]] = []
    for code in code_cols:
        first_normal_done = False
        for idx, status in df[code].items():
            if pd.isna(status):
                continue
            status_text = str(status).strip()
            dt = dates.iloc[idx].normalize() + pd.Timedelta(hours=15, seconds=1)
            if status_text == NORMAL_TRADE_STATUS:
                if not first_normal_done:
                    records.append(
                        {
                            "datetime": dt,
                            "code": code,
                            "name": name_map.get(code) or code,
                            "metric": "交易状态",
                            "value": 1,
                            "remark": {"status": status_text, "first_normal": True},
                        }
                    )
                    first_normal_done = True
            else:
                records.append(
                    {
                        "datetime": dt,
                        "code": code,
                        "name": name_map.get(code) or code,
                        "metric": "交易状态",
                        "value": 0,
                        "remark": {"status": status_text},
                    }
                )
    return normalize_import_frame(pd.DataFrame(records))


def infer_import_type(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "交易状态" in name or "trade_status" in name:
        return "trade_status"
    if "成分" in name or "universe" in name or "index" in name:
        return "index_universe"
    if name.startswith("ede") or "ede" in name:
        return "ede"
    return "wind_long"


def load_import_frame(import_type: str, path: str | Path, options: dict[str, Any] | None = None, logger: logging.Logger | None = None) -> pd.DataFrame:
    options = options or {}
    if import_type == "ede":
        return load_ede(path, date_from=options.get("date_from", "filename"), default_datetime=options.get("default_datetime"), logger=logger)
    if import_type == "wind_long":
        return load_wind_long(path, logger=logger)
    if import_type == "index_universe":
        return load_index_universe(
            path,
            index_code=options.get("index_code", "000906.SH"),
            index_name=options.get("index_name", "中证800"),
            sheet_name=options.get("sheet_name", "Sheet2"),
        )
    if import_type == "trade_status":
        return load_trade_status(path, sheet_name=options.get("sheet_name", "Sheet1"))
    raise ValueError(f"未知导入类型: {import_type}")


class DatabaseWriter:
    def __init__(self, client: DatabaseClient | None = None):
        self.client = client or DatabaseClient()

    def dry_run(self, table: str, df: pd.DataFrame, conflict_sample_limit: int = 20) -> dict[str, Any]:
        table = self.client.validate_table(table)
        df = normalize_import_frame(df)
        if df.empty:
            return {"summary": dataframe_summary(df), "existing": 0, "conflict_count": 0, "conflicts": [], "new_rows_estimate": 0}
        keys = [
            (row["datetime"], row["code"], row["metric"], row["value"])
            for _, row in df[["datetime", "code", "metric", "value"]].iterrows()
        ]
        existing = 0
        conflict_count = 0
        conflicts: list[dict[str, Any]] = []
        with self.client.connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                for i in range(0, len(keys), 1000):
                    batch = keys[i : i + 1000]
                    values = [(dt, code, metric) for dt, code, metric, _value in batch]
                    query = sql.SQL(
                        """
                        SELECT v.dt, v.cd, v.mt, t.value AS db_value
                        FROM (VALUES %s) AS v(dt, cd, mt)
                        JOIN {} t ON t.datetime = v.dt AND t.code = v.cd AND t.metric = v.mt
                        """
                    ).format(sql.Identifier(table))
                    psycopg2.extras.execute_values(cur, query.as_string(cur), values, template="(%s, %s, %s)")
                    rows = cur.fetchall()
                    existing += len(rows)
                    db_values = {(row["dt"], row["cd"], row["mt"]): row["db_value"] for row in rows}
                    for dt, code, metric, new_value in batch:
                        db_value = db_values.get((dt, code, metric))
                        if db_value is not None and str(db_value) != str(new_value):
                            conflict_count += 1
                            if len(conflicts) < conflict_sample_limit:
                                conflicts.append(
                                    {
                                        "datetime": dt,
                                        "code": code,
                                        "metric": metric,
                                        "db_value": db_value,
                                        "new_value": new_value,
                                    }
                                )
        return {
            "summary": dataframe_summary(df),
            "existing": existing,
            "conflict_count": conflict_count,
            "conflicts": conflicts,
            "new_rows_estimate": max(0, int(len(df) - existing)),
        }

    def write(self, table: str, df: pd.DataFrame, mode: str = INSERT_ONLY, batch_size: int = 5000) -> dict[str, Any]:
        table = self.client.validate_table(table)
        if mode not in {INSERT_ONLY, UPSERT}:
            raise ValueError(f"未知写入模式: {mode}")
        df = normalize_import_frame(df)
        if df.empty:
            return {"inserted": 0, "skipped": 0, "total": 0}
        cols = [c for c in DB_COLUMNS if c in df.columns]
        values = [tuple(row[c] for c in cols) for _, row in df.iterrows()]

        insert_sql = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
            sql.Identifier(table),
            sql.SQL(", ").join(sql.Identifier(c) for c in cols),
        )
        if mode == INSERT_ONLY:
            insert_sql += sql.SQL(" ON CONFLICT (datetime, code, metric) DO NOTHING")
        else:
            insert_sql += sql.SQL(
                """
                ON CONFLICT (datetime, code, metric)
                DO UPDATE SET name = EXCLUDED.name,
                              value = EXCLUDED.value,
                              remark = EXCLUDED.remark
                """
            )

        affected = 0
        with self.client.connect() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(values), batch_size):
                    batch = values[i : i + batch_size]
                    psycopg2.extras.execute_values(cur, insert_sql.as_string(cur), batch, page_size=batch_size)
                    affected += cur.rowcount
                conn.commit()
        skipped = max(0, len(values) - affected) if mode == INSERT_ONLY else 0
        return {"inserted": int(affected), "skipped": int(skipped), "total": int(len(values))}
