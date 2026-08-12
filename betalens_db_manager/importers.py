"""Import source adapters and database writer."""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
import numpy as np
import psycopg2.extras
import psycopg2.extensions
from psycopg2 import sql

from .constants import DB_COLUMNS, INSERT_ONLY, UPSERT
from .db import DatabaseClient
from .import_adapters import ImportBatch, collect_import_batches, infer_adapter, load_import_batches
from .registry import DATASETS, get_dataset
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
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", format="mixed")
    if "value" in out.columns:
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
    if "remark" in out.columns:
        def parse_remark(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            if isinstance(value, dict):
                return value
            if isinstance(value, (list, int, float, bool)):
                raise ValueError("remark 必须是 JSON object")
            text = str(value).strip()
            if not text or text.lower() in {"none", "nan", "null"}:
                return None
            try:
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("remark 必须是 JSON object")
                return parsed
            except json.JSONDecodeError as exc:
                raise ValueError(f"remark 不是合法 JSON: {text[:100]}") from exc

        out["remark"] = out["remark"].map(parse_remark)
    return out


def load_ede(
    path: str | Path,
    date_from: str = "filename",
    default_datetime: str | None = None,
    logger: logging.Logger | None = None,
    *,
    code_column_names: Sequence[str] | None = None,
    name_column_names: Sequence[str] | None = None,
    date_pattern: str = r"(\d{8})",
    default_time: str = "15:30:00",
    keywords_to_remove: Sequence[str] | None = None,
    table: str = "daily_market",
    column_map: dict[str, str] | None = None,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    frame, rejected = collect_import_batches(
        load_import_batches(
            "ede",
            path,
            table=table,
            logger=logger,
            options={
                "date_from": date_from,
                "default_datetime": default_datetime,
                "code_column_names": code_column_names,
                "name_column_names": name_column_names,
                "date_pattern": date_pattern,
                "default_time": default_time,
                "keywords_to_remove": keywords_to_remove,
                "column_map": column_map,
                "chunk_size": chunk_size,
            },
        )
    )
    if not rejected.empty:
        raise ValueError(rejected[["_source_row", "_errors"]].head(20).to_dict("records"))
    return frame


def load_wind_long(
    path: str | Path,
    logger: logging.Logger | None = None,
    *,
    open_metric_names: Sequence[str] | None = None,
    open_time: str = "09:30:01",
    other_time: str = "15:00:01",
    table: str = "daily_market",
    column_map: dict[str, str] | None = None,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    frame, rejected = collect_import_batches(
        load_import_batches(
            "wind_wide",
            path,
            table=table,
            logger=logger,
            options={
                "open_metric_names": open_metric_names,
                "open_time": open_time,
                "other_time": other_time,
                "column_map": column_map,
                "chunk_size": chunk_size,
            },
        )
    )
    if not rejected.empty:
        raise ValueError(rejected[["_source_row", "_errors"]].head(20).to_dict("records"))
    return frame


def load_index_universe(path: str | Path, index_code: str, index_name: str, sheet_name: str = "Sheet2", seq_col: str = "序号") -> pd.DataFrame:
    frame, rejected = collect_import_batches(
        load_import_batches(
            "index_universe",
            path,
            table="index_universe",
            options={
                "index_code": index_code,
                "index_name": index_name,
                "seq_col": seq_col,
                "read_options": {"sheet_name": sheet_name},
            },
        )
    )
    if not rejected.empty:
        raise ValueError(rejected[["_source_row", "_errors"]].head(20).to_dict("records"))
    return frame


def load_trade_status(path: str | Path, sheet_name: str = "Sheet1") -> pd.DataFrame:
    frame, rejected = collect_import_batches(
        load_import_batches(
            "trade_status",
            path,
            table="trade_status",
            options={"layout": "matrix", "sheet_name": sheet_name},
        )
    )
    if not rejected.empty:
        raise ValueError(rejected[["_source_row", "_errors"]].head(20).to_dict("records"))
    return frame


def load_trade_calendar(path: str | Path, sheet_name: str = "Sheet1") -> pd.DataFrame:
    frame, rejected = collect_import_batches(
        load_import_batches(
            "trade_calendar",
            path,
            table="trade_calendar",
            options={"sheet_name": sheet_name},
        )
    )
    if not rejected.empty:
        raise ValueError(rejected[["_source_row", "_errors"]].head(20).to_dict("records"))
    return frame


def infer_import_type(path: str | Path) -> str:
    return infer_adapter(path)


def load_import_frame(
    import_type: str | None,
    path: str | Path,
    options: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    *,
    table: str | None = None,
) -> pd.DataFrame:
    target = table or {
        "industry": "industry",
        "index_universe": "index_universe",
        "trade_status": "trade_status",
        "trade_calendar": "trade_calendar",
    }.get(import_type or "", "daily_market")
    frame, rejected = collect_import_batches(
        load_import_batches(
            import_type,
            path,
            table=target,
            options=options,
            logger=logger,
        )
    )
    if not rejected.empty:
        raise ValueError(rejected[["_source_row", "_errors"]].head(20).to_dict("records"))
    return frame


@dataclass(frozen=True)
class DeleteRequest:
    """A deliberately constrained logical deletion request."""

    table: str
    code: str | None = None
    codes: Sequence[str] | None = None
    metric: str | None = None
    start_date: str | None = None
    end_date: str | None = None

    def normalized_codes(self) -> tuple[str, ...]:
        values = [self.code] if self.code else []
        if isinstance(self.codes, str):
            values.append(self.codes)
        else:
            values.extend(self.codes or ())
        return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))

    def validate(self) -> None:
        get_dataset(self.table, writable=True)
        if not any((self.normalized_codes(), self.metric, self.start_date, self.end_date)):
            raise ValueError("删除请求必须包含 code/codes、metric 或日期条件，禁止无条件删除")


class DatabaseWriter:
    """Transactional set-based writer for standard six-column frames."""

    def __init__(self, client: DatabaseClient | None = None):
        self.client = client or DatabaseClient()

    def dry_run(
        self,
        table: str,
        df: pd.DataFrame,
        conflict_sample_limit: int = 20,
    ) -> dict[str, Any]:
        table = self.client.validate_table(table, writable=True)
        frame = self._prepare_frame(df, table=table)
        if frame.empty:
            return {
                "summary": dataframe_summary(frame),
                "existing": 0,
                "conflict_count": 0,
                "conflicts": [],
                "new_rows_estimate": 0,
            }
        with self.client.connect() as conn:
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    if not self.client._has_new_schema(cur):
                        result = self._legacy_dry_run(cur, table, frame, conflict_sample_limit)
                    else:
                        self._copy_stage(cur, frame)
                        self._prepare_dimensions(cur, table)
                        result = self._inspect_stage(cur, table, conflict_sample_limit)
                conn.rollback()
            except Exception:
                conn.rollback()
                raise
        result["summary"] = dataframe_summary(frame)
        resolved_total = int(result.pop("resolved_total", len(frame)))
        result["new_rows_estimate"] = max(0, resolved_total - result["existing"])
        return result

    def write(
        self,
        table: str,
        df: pd.DataFrame,
        mode: str = INSERT_ONLY,
        batch_size: int = 5000,
    ) -> dict[str, Any]:
        result = self.write_batches(
            table,
            [df],
            mode=mode,
            batch_size=batch_size,
        )
        return {key: int(result[key]) for key in ("inserted", "updated", "skipped", "total")}

    def write_batches(
        self,
        table: str,
        batches: Iterable[pd.DataFrame | ImportBatch],
        mode: str = INSERT_ONLY,
        batch_size: int = 50_000,
        progress: Callable[[dict[str, Any]], None] | None = None,
        conflict_sample_limit: int = 20,
    ) -> dict[str, Any]:
        """Write bounded batches in one file-level transaction."""

        table = self.client.validate_table(table, writable=True)
        if mode not in {INSERT_ONLY, UPSERT}:
            raise ValueError(f"未知写入模式: {mode}")
        totals: dict[str, Any] = {
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "total": 0,
            "existing": 0,
            "conflict_count": 0,
            "conflicts": [],
            "batches": 0,
        }
        conn = self.client.connect()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                new_schema = self.client._has_new_schema(cur)
                if new_schema:
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtext(%s))",
                        (f"betalens-writer:{table}",),
                    )
                staged = False
                minimum_datetime = None
                maximum_datetime = None
                for batch_number, item in enumerate(batches, start=1):
                    if isinstance(item, ImportBatch):
                        if item.table != table:
                            raise ValueError(f"批次目标 {item.table} 与写入目标 {table} 不一致")
                        if not item.rejected.empty:
                            raise ValueError(f"批次仍包含 {len(item.rejected)} 行 rejected rows")
                        source_frame = item.frame
                    else:
                        source_frame = item
                    frame = self._prepare_frame(source_frame, table=table)
                    if frame.empty:
                        continue
                    self._copy_stage(
                        cur,
                        frame,
                        batch_size=batch_size,
                        create=not staged,
                    )
                    staged = True
                    totals["total"] += int(len(frame))
                    totals["batches"] += 1
                    frame_min = frame["datetime"].min()
                    frame_max = frame["datetime"].max()
                    minimum_datetime = frame_min if minimum_datetime is None else min(minimum_datetime, frame_min)
                    maximum_datetime = frame_max if maximum_datetime is None else max(maximum_datetime, frame_max)
                    if progress is not None:
                        progress(
                            {
                                "phase": "stage",
                                "batch": batch_number,
                                "rows": len(frame),
                                "total_rows": totals["total"],
                            }
                        )

                if not staged:
                    conn.commit()
                    return totals

                self._validate_staged_conflicts(cur)
                self._deduplicate_stage(cur)
                if new_schema:
                    self._prepare_dimensions(cur, table)
                    if DATASETS[table].storage in {"market", "observation"}:
                        self._ensure_observation_partitions(cur, connection=conn)
                    inspection = self._inspect_stage(cur, table, conflict_sample_limit)
                    inserted = inspection["resolved_total"] - inspection["existing"]
                    updated = inspection["conflict_count"] if mode == UPSERT else 0
                    skipped = totals["total"] - inserted - updated
                    self._write_new_schema(cur, table, mode)
                    coverage_frame = pd.DataFrame(
                        {"datetime": [minimum_datetime, maximum_datetime]}
                    )
                    self._update_coverage(cur, table, coverage_frame, inserted)
                else:
                    legacy = self._legacy_write(
                        cur,
                        table,
                        pd.DataFrame(),
                        mode,
                        batch_size,
                        stage_loaded=True,
                        total_override=totals["total"],
                    )
                    inserted = legacy["inserted"]
                    updated = legacy["updated"]
                    skipped = legacy["skipped"]
                    inspection = {
                        "existing": totals["total"] - inserted,
                        "conflict_count": updated,
                        "conflicts": [],
                    }
                totals["inserted"] = int(inserted)
                totals["updated"] = int(updated)
                totals["skipped"] = int(skipped)
                totals["existing"] = int(inspection.get("existing", 0))
                totals["conflict_count"] = int(inspection.get("conflict_count", 0))
                totals["conflicts"] = list(inspection.get("conflicts", []))[:conflict_sample_limit]
                if progress is not None:
                    progress(
                        {
                            "phase": "write",
                            "rows": totals["total"],
                            "inserted": totals["inserted"],
                            "updated": totals["updated"],
                            "skipped": totals["skipped"],
                        }
                    )
                self._drop_stage_tables(cur)
            conn.commit()
            return totals
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def delete(self, request: DeleteRequest) -> dict[str, Any]:
        request.validate()
        table = self.client.validate_table(request.table, writable=True)
        with self.client.connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if not self.client._has_new_schema(cur):
                    deleted = self._legacy_delete(cur, request)
                else:
                    deleted = self._delete_new_schema(cur, table, request)
            conn.commit()
        return {"deleted": int(deleted), "total": int(deleted)}

    def _prepare_frame(self, df: pd.DataFrame, *, table: str | None = None) -> pd.DataFrame:
        frame = normalize_import_frame(df)
        missing = [column for column in DB_COLUMNS[:5] if column not in frame.columns]
        if missing:
            raise ValueError(f"导入数据缺少列: {missing}")
        frame = frame[list(DB_COLUMNS)].copy()
        if frame[["datetime", "code", "name", "metric"]].isna().any().any():
            raise ValueError("datetime/code/name/metric 不允许为空")
        if frame["datetime"].isna().any():
            raise ValueError("存在无法解析的 datetime")
        if frame["value"].isna().any():
            raise ValueError("value 不允许为空或非数值")
        finite = np.isfinite(frame["value"].to_numpy(dtype=float, na_value=np.nan))
        if not finite.all():
            raise ValueError("value 不允许 NaN/Inf")
        frame["code"] = frame["code"].astype(str).str.strip()
        frame["name"] = frame["name"].astype(str).str.strip()
        frame["metric"] = frame["metric"].astype(str).str.strip()
        if (frame[["code", "name", "metric"]] == "").any().any():
            raise ValueError("code/name/metric 不允许为空字符串")
        if table == "trade_calendar":
            frame["code"] = frame["code"].str.upper()
            frame["name"] = frame["code"]
            frame["datetime"] = frame["datetime"].dt.normalize()
        frame["_remark_key"] = frame["remark"].map(self._stable_json)
        key = ["datetime", "code", "metric"]
        conflicts = frame.groupby(key, dropna=False).agg(
            value_count=("value", "nunique"),
            name_count=("name", "nunique"),
            remark_count=("_remark_key", "nunique"),
        )
        conflicts = conflicts[
            (conflicts["value_count"] > 1)
            | (conflicts["name_count"] > 1)
            | (conflicts["remark_count"] > 1)
        ]
        if not conflicts.empty:
            sample = [tuple(value) for value in conflicts.head(5).index.tolist()]
            raise ValueError(f"批内同键记录存在不同值: {sample}")
        name_conflicts = frame.groupby(["datetime", "code"], dropna=False)["name"].nunique()
        if (name_conflicts > 1).any():
            sample = [tuple(value) for value in name_conflicts[name_conflicts > 1].head(5).index.tolist()]
            raise ValueError(f"批内同一实体时点存在不同名称: {sample}")
        frame = frame.drop_duplicates(key, keep="last").drop(columns="_remark_key")
        return frame.reset_index(drop=True)

    @staticmethod
    def _stable_json(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "null"
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    def _copy_stage(
        self,
        cur,
        frame: pd.DataFrame,
        *,
        batch_size: int = 50_000,
        create: bool = True,
    ) -> None:
        if create:
            cur.execute(
                """
                CREATE TEMP TABLE _betalens_import_stage (
                    row_no BIGSERIAL PRIMARY KEY,
                    datetime TIMESTAMP NOT NULL,
                    code VARCHAR(32) NOT NULL,
                    name VARCHAR(200) NOT NULL,
                    metric VARCHAR(160) NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    remark JSONB
                ) ON COMMIT DROP
                """
            )
        batch_size = max(1, int(batch_size))
        copy_sql = """
        COPY _betalens_import_stage (datetime, code, name, metric, value, remark)
        FROM STDIN WITH (FORMAT CSV, NULL '\\N')
        """
        for start in range(0, len(frame), batch_size):
            stream = io.StringIO(newline="")
            writer = csv.writer(stream, lineterminator="\n")
            for row in frame.iloc[start : start + batch_size].itertuples(index=False):
                remark = (
                    "\\N"
                    if row.remark is None or (isinstance(row.remark, float) and pd.isna(row.remark))
                    else self._stable_json(row.remark)
                )
                writer.writerow(
                    [
                        pd.Timestamp(row.datetime).isoformat(sep=" "),
                        row.code,
                        row.name,
                        row.metric,
                        float(row.value),
                        remark,
                    ]
                )
            stream.seek(0)
            cur.copy_expert(copy_sql, stream)

    @staticmethod
    def _validate_staged_conflicts(cur, sample_limit: int = 5) -> None:
        cur.execute(
            """
            SELECT datetime, code, metric
            FROM _betalens_import_stage
            GROUP BY datetime, code, metric
            HAVING count(DISTINCT value) > 1
                OR count(DISTINCT name) > 1
                OR count(DISTINCT COALESCE(remark, 'null'::jsonb)) > 1
            ORDER BY datetime, code, metric
            LIMIT %s
            """,
            (max(1, int(sample_limit)),),
        )
        conflicts = [tuple(row.values()) if isinstance(row, dict) else tuple(row) for row in cur.fetchall()]
        if conflicts:
            raise ValueError(f"跨 chunk 同键记录存在不同值: {conflicts}")
        cur.execute(
            """
            SELECT datetime, code
            FROM _betalens_import_stage
            GROUP BY datetime, code
            HAVING count(DISTINCT name) > 1
            ORDER BY datetime, code
            LIMIT %s
            """,
            (max(1, int(sample_limit)),),
        )
        name_conflicts = [
            tuple(row.values()) if isinstance(row, dict) else tuple(row)
            for row in cur.fetchall()
        ]
        if name_conflicts:
            raise ValueError(f"跨 chunk 同一实体时点存在不同名称: {name_conflicts}")

    @staticmethod
    def _deduplicate_stage(cur) -> None:
        cur.execute(
            """
            DELETE FROM _betalens_import_stage duplicate
            USING _betalens_import_stage canonical
            WHERE duplicate.datetime = canonical.datetime
              AND duplicate.code = canonical.code
              AND duplicate.metric = canonical.metric
              AND duplicate.row_no > canonical.row_no
            """
        )

    @staticmethod
    def _drop_stage_tables(cur) -> None:
        for name in (
            "_betalens_index_existing",
            "_betalens_index_status",
            "_betalens_index_member_stage",
            "_betalens_industry_status",
            "_betalens_trade_status",
            "_betalens_trade_calendar",
            "_betalens_market_status",
            "_betalens_resolved_stage",
            "_betalens_import_stage",
        ):
            cur.execute(sql.SQL("DROP TABLE IF EXISTS pg_temp.{}").format(sql.Identifier(name)))

    def _prepare_dimensions(self, cur, table: str) -> None:
        spec = DATASETS[table]
        if spec.storage == "trade_calendar":
            self._validate_trade_calendar_stage(cur)
            return
        cur.execute(
            """
            SELECT DISTINCT s.code, e.entity_type AS existing_type
            FROM _betalens_import_stage s
            JOIN betalens.entity_dim e ON e.code = s.code
            WHERE e.entity_type NOT IN (%s, 'unknown')
            LIMIT 5
            """,
            (spec.entity_type,),
        )
        mismatch = cur.fetchall()
        if mismatch:
            raise ValueError(f"证券代码已属于其他实体类型: {[row['code'] for row in mismatch]}")
        cur.execute(
            """
            INSERT INTO betalens.entity_dim (code, entity_type, current_name)
            SELECT DISTINCT ON (code) code, %s, name
            FROM _betalens_import_stage ORDER BY code, datetime DESC, row_no DESC
            ON CONFLICT (code) DO UPDATE
            SET current_name = EXCLUDED.current_name,
                entity_type = CASE WHEN betalens.entity_dim.entity_type='unknown'
                                   THEN EXCLUDED.entity_type
                                   ELSE betalens.entity_dim.entity_type END,
                updated_at = now()
            """,
            (spec.entity_type,),
        )
        cur.execute(
            """
            INSERT INTO betalens.entity_name_history (entity_id, valid_from, name)
            SELECT e.entity_id, min(s.datetime), s.name
            FROM _betalens_import_stage s
            JOIN betalens.entity_dim e ON e.code = s.code
            WHERE NOT EXISTS (
                SELECT 1 FROM betalens.entity_name_history h
                WHERE h.entity_id=e.entity_id AND h.name=s.name
                  AND h.valid_from <= s.datetime
                  AND (h.valid_to IS NULL OR h.valid_to > s.datetime)
            )
            GROUP BY e.entity_id, s.name
            ON CONFLICT (entity_id, valid_from) DO UPDATE SET name = EXCLUDED.name
            """
        )
        cur.execute(
            """
            WITH affected AS (
                SELECT DISTINCT e.entity_id FROM _betalens_import_stage s
                JOIN betalens.entity_dim e ON e.code = s.code
            ), bounds AS (
                SELECT h.entity_id, h.valid_from,
                       lead(h.valid_from) OVER (PARTITION BY h.entity_id ORDER BY h.valid_from) AS valid_to
                FROM betalens.entity_name_history h JOIN affected a USING (entity_id)
            )
            UPDATE betalens.entity_name_history h SET valid_to = b.valid_to
            FROM bounds b WHERE h.entity_id=b.entity_id AND h.valid_from=b.valid_from
              AND h.valid_to IS DISTINCT FROM b.valid_to
            """
        )
        if spec.storage in {"market", "observation"}:
            self._prepare_metric_dimensions(cur, table)
        elif spec.storage == "industry":
            self._prepare_industry_dimensions(cur)
        elif spec.storage == "index_universe":
            self._prepare_index_dimensions(cur)
        elif spec.storage == "trade_status":
            self._validate_trade_status_stage(cur)

    @staticmethod
    def _validate_trade_calendar_stage(cur) -> None:
        cur.execute(
            """
            SELECT row_no
            FROM _betalens_import_stage
            WHERE metric <> '交易日'
               OR value <> 1
               OR datetime <> date_trunc('day', datetime)
               OR code <> upper(btrim(code))
            LIMIT 1
            """
        )
        if cur.fetchone():
            raise ValueError("trade_calendar 仅接受大写交易所代码、日期和 metric='交易日'/value=1")

    def _prepare_industry_dimensions(self, cur) -> None:
        cur.execute(
            """
            SELECT row_no FROM _betalens_import_stage
            WHERE COALESCE(remark->>'ind_code', NULLIF(regexp_replace(value::text, '\\.0+$', ''), '')) IS NULL
               OR COALESCE(remark->>'ind_name', remark->>'industry_name') IS NULL
            LIMIT 1
            """
        )
        if cur.fetchone():
            raise ValueError("industry 导入要求 remark.ind_name，并通过 remark.ind_code 或 value 提供行业代码")
        cur.execute(
            """
            INSERT INTO betalens.industry_scheme_dim (scheme_name)
            SELECT DISTINCT metric FROM _betalens_import_stage
            ON CONFLICT (scheme_name) DO NOTHING
            """
        )
        cur.execute(
            """
            INSERT INTO betalens.industry_dim (scheme_id, industry_code, industry_name)
            SELECT DISTINCT ON (sc.scheme_id,
                       COALESCE(s.remark->>'ind_code', regexp_replace(s.value::text, '\\.0+$', '')))
                   sc.scheme_id,
                   COALESCE(s.remark->>'ind_code', regexp_replace(s.value::text, '\\.0+$', '')),
                   COALESCE(s.remark->>'ind_name', s.remark->>'industry_name')
            FROM _betalens_import_stage s
            JOIN betalens.industry_scheme_dim sc ON sc.scheme_name=s.metric
            ORDER BY sc.scheme_id,
                     COALESCE(s.remark->>'ind_code', regexp_replace(s.value::text, '\\.0+$', '')),
                     s.datetime DESC
            ON CONFLICT (scheme_id, industry_code)
            DO UPDATE SET industry_name=EXCLUDED.industry_name
            """
        )

    def _prepare_index_dimensions(self, cur) -> None:
        cur.execute(
            """
            SELECT row_no FROM _betalens_import_stage
            WHERE metric <> 'universe'
               OR jsonb_typeof(remark->'constituents') IS DISTINCT FROM 'array'
            LIMIT 1
            """
        )
        if cur.fetchone():
            raise ValueError("index_universe 要求 metric='universe' 且 remark.constituents 为数组")
        cur.execute(
            """
            CREATE TEMP TABLE _betalens_index_member_stage ON COMMIT DROP AS
            WITH expanded AS (
                SELECT s.row_no,
                       CASE jsonb_typeof(item.value)
                           WHEN 'string' THEN item.value #>> '{}'
                           WHEN 'object' THEN COALESCE(
                               item.value->>'code', item.value->>'wind_code', item.value->>'windcode')
                       END AS code,
                       item.ordinality::integer AS ordinal,
                       CASE WHEN jsonb_typeof(item.value)='object'
                                  AND COALESCE(item.value->>'weight', '')
                                      ~ '^[+-]?[0-9]+([.][0-9]+)?$'
                            THEN (item.value->>'weight')::double precision END AS weight,
                       CASE WHEN jsonb_typeof(item.value)='object' THEN item.value END AS remark
                FROM _betalens_import_stage s
                CROSS JOIN LATERAL jsonb_array_elements(s.remark->'constituents')
                    WITH ORDINALITY item(value, ordinality)
            )
            SELECT DISTINCT ON (row_no, code) row_no, code, ordinal, weight, remark
            FROM expanded ORDER BY row_no, code, ordinal
            """
        )
        cur.execute(
            "SELECT row_no FROM _betalens_index_member_stage WHERE code IS NULL OR code='' LIMIT 1"
        )
        if cur.fetchone():
            raise ValueError("index_universe constituent 必须是代码字符串或包含 code 的对象")
        cur.execute(
            """
            SELECT s.row_no
            FROM _betalens_import_stage s
            LEFT JOIN (
                SELECT row_no, count(*)::double precision AS member_count
                FROM _betalens_index_member_stage GROUP BY row_no
            ) members ON members.row_no=s.row_no
            WHERE s.value IS DISTINCT FROM COALESCE(members.member_count, 0::double precision)
            LIMIT 1
            """
        )
        if cur.fetchone():
            raise ValueError("index_universe value 必须等于去重后的 constituent 数量")
        cur.execute(
            """
            INSERT INTO betalens.entity_dim (code, entity_type, current_name)
            SELECT DISTINCT member.code, 'stock', member.code
            FROM _betalens_index_member_stage member
            ON CONFLICT (code) DO UPDATE SET
                entity_type=CASE WHEN betalens.entity_dim.entity_type='unknown'
                                 THEN 'stock' ELSE betalens.entity_dim.entity_type END,
                updated_at=now()
            """
        )

    def _validate_trade_status_stage(self, cur) -> None:
        cur.execute(
            """
            SELECT row_no FROM _betalens_import_stage
            WHERE metric <> '交易状态' OR value NOT IN (0, 1)
            LIMIT 1
            """
        )
        if cur.fetchone():
            raise ValueError("trade_status 仅接受 metric='交易状态' 且 value 为 0 或 1")

    def _prepare_metric_dimensions(self, cur, table: str) -> None:
        cur.execute(
            """
            INSERT INTO betalens.metric_dim
                (logical_dataset, metric_name, storage_kind, availability_time)
            SELECT %s, s.metric, 'observation', min(s.datetime::time)
            FROM _betalens_import_stage s
            WHERE NOT EXISTS (
                SELECT 1 FROM betalens.metric_dim md
                WHERE md.logical_dataset=%s AND md.metric_name=s.metric
            ) AND NOT EXISTS (
                SELECT 1 FROM betalens.metric_alias ma
                WHERE ma.logical_dataset=%s AND ma.alias=s.metric
            )
            GROUP BY s.metric
            ON CONFLICT (logical_dataset, metric_name) DO NOTHING
            """,
            (table, table, table),
        )
        cur.execute(
            """
            CREATE TEMP TABLE _betalens_resolved_stage ON COMMIT DROP AS
            SELECT DISTINCT ON (e.entity_id, COALESCE(ma.metric_id, md.metric_id), s.datetime)
                   s.row_no, s.datetime, s.code, s.name, s.metric AS input_metric,
                   canonical.metric_name, s.value, s.remark, e.entity_id,
                   canonical.metric_id,
                   CASE WHEN canonical.storage_kind='core'
                                  AND s.datetime::time=canonical.availability_time
                        THEN 'core' ELSE 'observation' END AS storage_kind,
                   canonical.storage_column
            FROM _betalens_import_stage s
            JOIN betalens.entity_dim e ON e.code=s.code
            LEFT JOIN betalens.metric_alias ma
              ON ma.logical_dataset=%s AND ma.alias=s.metric
            LEFT JOIN betalens.metric_dim md
              ON md.logical_dataset=%s AND md.metric_name=s.metric
            JOIN betalens.metric_dim canonical
              ON canonical.metric_id=COALESCE(ma.metric_id, md.metric_id)
            ORDER BY e.entity_id, COALESCE(ma.metric_id, md.metric_id), s.datetime, s.row_no
            """,
            (table, table),
        )
        cur.execute(
            """
            SELECT entity_id, metric_id, datetime
            FROM (
                SELECT e.entity_id, COALESCE(ma.metric_id, md.metric_id) AS metric_id,
                       s.datetime, count(DISTINCT s.value) AS value_count,
                       count(DISTINCT s.name) AS name_count,
                       count(DISTINCT COALESCE(s.remark, 'null'::jsonb)) AS remark_count
                FROM _betalens_import_stage s
                JOIN betalens.entity_dim e ON e.code=s.code
                LEFT JOIN betalens.metric_alias ma ON ma.logical_dataset=%s AND ma.alias=s.metric
                LEFT JOIN betalens.metric_dim md ON md.logical_dataset=%s AND md.metric_name=s.metric
                GROUP BY e.entity_id, COALESCE(ma.metric_id, md.metric_id), s.datetime
            ) conflicts
            WHERE value_count > 1 OR name_count > 1 OR remark_count > 1
            LIMIT 5
            """,
            (table, table),
        )
        if cur.fetchall():
            raise ValueError("批内指标别名解析到同一物理键，但值不一致")

    def _inspect_stage(self, cur, table: str, sample_limit: int) -> dict[str, Any]:
        spec = DATASETS[table]
        if spec.storage in {"market", "observation"}:
            status_table = self._inspect_market_observation(cur)
        elif spec.storage == "industry":
            status_table = self._inspect_industry(cur)
        elif spec.storage == "index_universe":
            status_table = self._inspect_index(cur)
        elif spec.storage == "trade_calendar":
            status_table = self._inspect_trade_calendar(cur)
        else:
            status_table = self._inspect_trade_status(cur)
        identifier = sql.Identifier(status_table)
        cur.execute(
            sql.SQL(
                """
                SELECT count(*) AS resolved_total,
                       count(*) FILTER (WHERE exists) AS existing,
                       count(*) FILTER (WHERE exists AND changed) AS conflict_count
                FROM pg_temp.{}
                """
            ).format(identifier)
        )
        counts = dict(cur.fetchone())
        cur.execute(
            sql.SQL(
                """
                SELECT datetime, code, metric, db_value, new_value
                FROM pg_temp.{}
                WHERE exists AND changed
                ORDER BY row_no
                LIMIT %s
                """
            ).format(identifier),
            (max(0, int(sample_limit)),),
        )
        conflicts = [dict(row) for row in cur.fetchall()]
        return {
            "resolved_total": int(counts["resolved_total"]),
            "existing": int(counts["existing"]),
            "conflict_count": int(counts["conflict_count"]),
            "conflicts": conflicts,
        }

    def _inspect_market_observation(self, cur) -> str:
        cur.execute(
            f"""
            CREATE TEMP TABLE _betalens_market_status ON COMMIT DROP AS
            SELECT r.datetime, r.code, r.input_metric AS metric, r.value AS new_value,
                   r.row_no,
                   CASE WHEN r.storage_kind='core' THEN ({self._core_value_sql('f', 'r')})
                        ELSE o.value END AS db_value,
                   CASE WHEN r.storage_kind='core'
                        THEN ({self._core_value_sql('f', 'r')}) IS NOT NULL
                        ELSE o.entity_id IS NOT NULL END AS exists,
                   CASE WHEN r.storage_kind='core'
                        THEN ({self._core_value_sql('f', 'r')}) IS DISTINCT FROM r.value
                             OR (f.remark -> r.metric_name) IS DISTINCT FROM r.remark
                        ELSE o.value IS DISTINCT FROM r.value OR o.remark IS DISTINCT FROM r.remark
                   END AS changed
            FROM _betalens_resolved_stage r
            LEFT JOIN betalens.market_daily_fact f
              ON r.storage_kind='core' AND f.entity_id=r.entity_id
             AND f.trade_date=r.datetime::date
            LEFT JOIN betalens.observation_fact o
              ON r.storage_kind='observation' AND o.entity_id=r.entity_id
             AND o.metric_id=r.metric_id AND o.available_at=r.datetime
            ORDER BY r.row_no
            """
        )
        return "_betalens_market_status"

    @staticmethod
    def _core_value_sql(fact_alias: str, resolved_alias: str) -> str:
        return (
            f"CASE {resolved_alias}.storage_column "
            f"WHEN 'open' THEN {fact_alias}.open WHEN 'high' THEN {fact_alias}.high "
            f"WHEN 'low' THEN {fact_alias}.low WHEN 'close' THEN {fact_alias}.close "
            f"WHEN 'prev_close' THEN {fact_alias}.prev_close WHEN 'volume' THEN {fact_alias}.volume "
            f"WHEN 'amount' THEN {fact_alias}.amount WHEN 'turnover_rate' THEN {fact_alias}.turnover_rate END"
        )

    def _inspect_industry(self, cur) -> str:
        cur.execute(
            """
            CREATE TEMP TABLE _betalens_industry_status ON COMMIT DROP AS
            SELECT s.datetime, s.code, s.metric, s.value AS new_value,
                   s.row_no,
                   NULLIF(regexp_replace(old.industry_code, '[^0-9]', '', 'g'), '')::double precision AS db_value,
                   old.industry_id IS NOT NULL AS exists,
                   (old.industry_code IS DISTINCT FROM
                       COALESCE(s.remark->>'ind_code', regexp_replace(s.value::text, '\\.0+$', ''))
                   OR old.industry_name IS DISTINCT FROM
                       COALESCE(s.remark->>'ind_name', s.remark->>'industry_name')
                   OR old.remark IS DISTINCT FROM s.remark) AS changed
            FROM _betalens_import_stage s
            JOIN betalens.entity_dim e ON e.code=s.code
            JOIN betalens.industry_scheme_dim sc ON sc.scheme_name=s.metric
            LEFT JOIN LATERAL (
                SELECT im.entity_id, im.remark, d.industry_id,
                       d.industry_code, d.industry_name
                FROM betalens.industry_membership im
                JOIN betalens.industry_dim d ON d.industry_id=im.industry_id
                WHERE im.entity_id=e.entity_id AND im.valid_from=s.datetime
                  AND d.scheme_id=sc.scheme_id
                LIMIT 1
            ) old ON true
            ORDER BY s.row_no
            """
        )
        return "_betalens_industry_status"

    def _inspect_index(self, cur) -> str:
        cur.execute(
            """
            CREATE TEMP TABLE _betalens_index_status ON COMMIT DROP AS
            WITH existing AS (
                SELECT sn.snapshot_id, sn.index_entity_id, sn.effective_at,
                       sn.index_name_snapshot, sn.remark,
                       COALESCE(jsonb_agg(e.code ORDER BY c.ordinal NULLS LAST, e.code)
                           FILTER (WHERE e.code IS NOT NULL), '[]'::jsonb) AS constituents,
                       COALESCE(jsonb_agg(jsonb_build_object(
                           'code', e.code, 'weight', c.weight, 'remark', c.remark)
                           ORDER BY c.ordinal NULLS LAST, e.code)
                           FILTER (WHERE e.code IS NOT NULL), '[]'::jsonb) AS member_signature
                FROM betalens.index_snapshot sn
                LEFT JOIN betalens.index_constituent c ON c.snapshot_id=sn.snapshot_id
                LEFT JOIN betalens.entity_dim e ON e.entity_id=c.constituent_entity_id
                GROUP BY sn.snapshot_id
            ), wanted AS (
                SELECT row_no,
                       COALESCE(jsonb_agg(code ORDER BY ordinal), '[]'::jsonb) AS constituents,
                       COALESCE(jsonb_agg(jsonb_build_object(
                           'code', code, 'weight', weight, 'remark', remark)
                           ORDER BY ordinal), '[]'::jsonb) AS member_signature
                FROM _betalens_index_member_stage GROUP BY row_no
            )
            SELECT s.datetime, s.code, s.metric, s.value AS new_value, s.row_no,
                   CASE WHEN x.snapshot_id IS NULL THEN NULL
                        ELSE jsonb_array_length(x.constituents)::double precision END AS db_value,
                   x.snapshot_id IS NOT NULL AS exists,
                   (x.constituents IS DISTINCT FROM COALESCE(w.constituents, '[]'::jsonb)
                   OR x.member_signature IS DISTINCT FROM COALESCE(w.member_signature, '[]'::jsonb)
                   OR x.index_name_snapshot IS DISTINCT FROM s.name
                   OR x.remark IS DISTINCT FROM (s.remark - 'constituents')) AS changed
            FROM _betalens_import_stage s
            LEFT JOIN wanted w ON w.row_no=s.row_no
            JOIN betalens.entity_dim idx ON idx.code=s.code
            LEFT JOIN existing x
              ON x.index_entity_id=idx.entity_id AND x.effective_at=s.datetime
            ORDER BY s.row_no
            """
        )
        return "_betalens_index_status"

    def _inspect_trade_status(self, cur) -> str:
        cur.execute(
            """
            CREATE TEMP TABLE _betalens_trade_status ON COMMIT DROP AS
            SELECT s.datetime, s.code, s.metric, s.value AS new_value, s.row_no,
                   CASE WHEN s.value=1 THEN CASE WHEN e.first_trade_date IS NULL THEN NULL ELSE 1 END
                        ELSE t.status::double precision END AS db_value,
                   CASE WHEN s.value=1 THEN e.first_trade_date IS NOT NULL
                        ELSE t.entity_id IS NOT NULL END AS exists,
                   CASE WHEN s.value=1 THEN e.first_trade_date IS DISTINCT FROM s.datetime::date
                        ELSE t.status::double precision IS DISTINCT FROM s.value
                          OR t.status_text IS DISTINCT FROM COALESCE(s.remark->>'status', '异常')
                          OR t.remark IS DISTINCT FROM s.remark END AS changed
            FROM _betalens_import_stage s
            JOIN betalens.entity_dim e ON e.code=s.code
            LEFT JOIN betalens.trade_status_event t
              ON s.value=0 AND t.entity_id=e.entity_id AND t.event_date=s.datetime::date
            ORDER BY s.row_no
            """
        )
        return "_betalens_trade_status"

    def _inspect_trade_calendar(self, cur) -> str:
        cur.execute(
            """
            CREATE TEMP TABLE _betalens_trade_calendar ON COMMIT DROP AS
            SELECT s.datetime, s.code, s.metric, s.value AS new_value, s.row_no,
                   CASE WHEN calendar.trade_date IS NULL THEN NULL ELSE 1::double precision END AS db_value,
                   calendar.trade_date IS NOT NULL AS exists,
                   FALSE AS changed
            FROM _betalens_import_stage s
            LEFT JOIN betalens.trade_calendar_day calendar
              ON calendar.exchange=s.code AND calendar.trade_date=s.datetime::date
            ORDER BY s.row_no
            """
        )
        return "_betalens_trade_calendar"

    def _write_new_schema(self, cur, table: str, mode: str) -> None:
        storage = DATASETS[table].storage
        if storage in {"market", "observation"}:
            self._write_market_observation(cur, mode)
        elif storage == "industry":
            self._write_industry(cur, mode)
        elif storage == "index_universe":
            self._write_index(cur, mode)
        elif storage == "trade_calendar":
            self._write_trade_calendar(cur)
        else:
            self._write_trade_status(cur, mode)

    def _write_market_observation(self, cur, mode: str) -> None:
        current_value = self._core_value_sql("current", "r")
        mode_filter = "TRUE" if mode == UPSERT else f"({current_value}) IS NULL"
        cur.execute(
            f"""
            WITH core_stage AS (
                SELECT r.entity_id, r.datetime::date AS trade_date,
                       max(r.value) FILTER (WHERE r.storage_column='open') AS open,
                       max(r.value) FILTER (WHERE r.storage_column='high') AS high,
                       max(r.value) FILTER (WHERE r.storage_column='low') AS low,
                       max(r.value) FILTER (WHERE r.storage_column='close') AS close,
                       max(r.value) FILTER (WHERE r.storage_column='prev_close') AS prev_close,
                       max(r.value) FILTER (WHERE r.storage_column='volume') AS volume,
                       max(r.value) FILTER (WHERE r.storage_column='amount') AS amount,
                       max(r.value) FILTER (WHERE r.storage_column='turnover_rate') AS turnover_rate,
                       COALESCE(jsonb_object_agg(r.metric_name, r.remark)
                           FILTER (WHERE r.remark IS NOT NULL), '{{}}'::jsonb) AS remark
                FROM _betalens_resolved_stage r
                LEFT JOIN betalens.market_daily_fact current
                  ON current.entity_id=r.entity_id AND current.trade_date=r.datetime::date
                WHERE r.storage_kind='core' AND {mode_filter}
                GROUP BY r.entity_id, r.datetime::date
            )
            INSERT INTO betalens.market_daily_fact
                (entity_id, trade_date, open, high, low, close, prev_close,
                 volume, amount, turnover_rate, remark)
            SELECT entity_id, trade_date, open, high, low, close, prev_close,
                   volume, amount, turnover_rate, NULLIF(remark, '{{}}'::jsonb)
            FROM core_stage
            ON CONFLICT (entity_id, trade_date) DO UPDATE SET
                open=COALESCE(EXCLUDED.open, betalens.market_daily_fact.open),
                high=COALESCE(EXCLUDED.high, betalens.market_daily_fact.high),
                low=COALESCE(EXCLUDED.low, betalens.market_daily_fact.low),
                close=COALESCE(EXCLUDED.close, betalens.market_daily_fact.close),
                prev_close=COALESCE(EXCLUDED.prev_close, betalens.market_daily_fact.prev_close),
                volume=COALESCE(EXCLUDED.volume, betalens.market_daily_fact.volume),
                amount=COALESCE(EXCLUDED.amount, betalens.market_daily_fact.amount),
                turnover_rate=COALESCE(EXCLUDED.turnover_rate, betalens.market_daily_fact.turnover_rate),
                remark=COALESCE(betalens.market_daily_fact.remark, '{{}}'::jsonb)
                       || COALESCE(EXCLUDED.remark, '{{}}'::jsonb),
                updated_at=now()
            WHERE ROW(
                betalens.market_daily_fact.open, betalens.market_daily_fact.high,
                betalens.market_daily_fact.low, betalens.market_daily_fact.close,
                betalens.market_daily_fact.prev_close, betalens.market_daily_fact.volume,
                betalens.market_daily_fact.amount, betalens.market_daily_fact.turnover_rate,
                betalens.market_daily_fact.remark
            ) IS DISTINCT FROM ROW(
                COALESCE(EXCLUDED.open, betalens.market_daily_fact.open),
                COALESCE(EXCLUDED.high, betalens.market_daily_fact.high),
                COALESCE(EXCLUDED.low, betalens.market_daily_fact.low),
                COALESCE(EXCLUDED.close, betalens.market_daily_fact.close),
                COALESCE(EXCLUDED.prev_close, betalens.market_daily_fact.prev_close),
                COALESCE(EXCLUDED.volume, betalens.market_daily_fact.volume),
                COALESCE(EXCLUDED.amount, betalens.market_daily_fact.amount),
                COALESCE(EXCLUDED.turnover_rate, betalens.market_daily_fact.turnover_rate),
                COALESCE(betalens.market_daily_fact.remark, '{{}}'::jsonb)
                    || COALESCE(EXCLUDED.remark, '{{}}'::jsonb)
            )
            """
        )

        if mode == INSERT_ONLY:
            cur.execute(
                """
                INSERT INTO betalens.observation_fact
                    (available_at, entity_id, metric_id, period_end, value, remark)
                SELECT datetime, entity_id, metric_id,
                       CASE WHEN COALESCE(remark->>'period_end', remark->>'report_date', '')
                                      ~ '^\\d{4}-\\d{2}-\\d{2}$'
                            THEN COALESCE(remark->>'period_end', remark->>'report_date')::date END,
                       value, remark
                FROM _betalens_resolved_stage WHERE storage_kind='observation'
                ON CONFLICT (entity_id, metric_id, available_at) DO NOTHING
                """
            )
        else:
            cur.execute(
                """
                INSERT INTO betalens.observation_fact
                    (available_at, entity_id, metric_id, period_end, value, remark)
                SELECT datetime, entity_id, metric_id,
                       CASE WHEN COALESCE(remark->>'period_end', remark->>'report_date', '')
                                      ~ '^\\d{4}-\\d{2}-\\d{2}$'
                            THEN COALESCE(remark->>'period_end', remark->>'report_date')::date END,
                       value, remark
                FROM _betalens_resolved_stage WHERE storage_kind='observation'
                ON CONFLICT (entity_id, metric_id, available_at) DO UPDATE
                SET period_end=EXCLUDED.period_end, value=EXCLUDED.value,
                    remark=EXCLUDED.remark, updated_at=now()
                WHERE betalens.observation_fact.period_end IS DISTINCT FROM EXCLUDED.period_end
                   OR betalens.observation_fact.value IS DISTINCT FROM EXCLUDED.value
                   OR betalens.observation_fact.remark IS DISTINCT FROM EXCLUDED.remark
                """
            )

    def _write_industry(self, cur, mode: str) -> None:
        if mode == UPSERT:
            cur.execute(
                """
                DELETE FROM betalens.industry_membership im
                USING _betalens_import_stage s, _betalens_industry_status st,
                      betalens.entity_dim e,
                      betalens.industry_dim old_ind, betalens.industry_scheme_dim sc
                WHERE st.row_no=s.row_no AND (NOT st.exists OR st.changed)
                  AND e.code=s.code AND im.entity_id=e.entity_id
                  AND im.valid_from=s.datetime AND old_ind.industry_id=im.industry_id
                  AND sc.scheme_id=old_ind.scheme_id AND sc.scheme_name=s.metric
                """
            )
        cur.execute(
            """
            INSERT INTO betalens.industry_membership
                (entity_id, industry_id, valid_from, remark)
            SELECT e.entity_id, d.industry_id, s.datetime, s.remark
            FROM _betalens_import_stage s
            JOIN _betalens_industry_status st ON st.row_no=s.row_no
            JOIN betalens.entity_dim e ON e.code=s.code
            JOIN betalens.industry_scheme_dim sc ON sc.scheme_name=s.metric
            JOIN betalens.industry_dim d ON d.scheme_id=sc.scheme_id
             AND d.industry_code=COALESCE(
                 s.remark->>'ind_code', regexp_replace(s.value::text, '\\.0+$', ''))
            WHERE NOT st.exists OR (%s AND st.changed)
            ON CONFLICT (entity_id, industry_id, valid_from) DO UPDATE
            SET remark=EXCLUDED.remark
            """,
            (mode == UPSERT,),
        )
        cur.execute(
            """
            WITH affected AS (
                SELECT DISTINCT e.entity_id, sc.scheme_id
                FROM _betalens_import_stage s
                JOIN betalens.entity_dim e ON e.code=s.code
                JOIN betalens.industry_scheme_dim sc ON sc.scheme_name=s.metric
            ), bounds AS (
                SELECT im.entity_id, im.industry_id, im.valid_from,
                       lead(im.valid_from) OVER (
                           PARTITION BY im.entity_id, d.scheme_id ORDER BY im.valid_from) AS valid_to
                FROM betalens.industry_membership im
                JOIN betalens.industry_dim d ON d.industry_id=im.industry_id
                JOIN affected a ON a.entity_id=im.entity_id AND a.scheme_id=d.scheme_id
            )
            UPDATE betalens.industry_membership im SET valid_to=b.valid_to
            FROM bounds b WHERE im.entity_id=b.entity_id AND im.industry_id=b.industry_id
              AND im.valid_from=b.valid_from AND im.valid_to IS DISTINCT FROM b.valid_to
            """
        )

    def _write_index(self, cur, mode: str) -> None:
        cur.execute(
            """
            CREATE TEMP TABLE _betalens_index_existing ON COMMIT DROP AS
            SELECT st.row_no, st.exists AS existed, st.changed
            FROM _betalens_index_status st
            """
        )
        if mode == INSERT_ONLY:
            cur.execute(
                """
                INSERT INTO betalens.index_snapshot
                    (index_entity_id, effective_at, index_name_snapshot, remark)
                SELECT e.entity_id, s.datetime, s.name,
                       s.remark - 'constituents'
                FROM _betalens_import_stage s
                JOIN _betalens_index_existing x ON x.row_no=s.row_no AND NOT x.existed
                JOIN betalens.entity_dim e ON e.code=s.code
                ON CONFLICT (index_entity_id, effective_at) DO NOTHING
                """
            )
        else:
            cur.execute(
                """
                INSERT INTO betalens.index_snapshot
                    (index_entity_id, effective_at, index_name_snapshot, remark)
                SELECT e.entity_id, s.datetime, s.name, s.remark - 'constituents'
                FROM _betalens_import_stage s
                JOIN _betalens_index_existing x ON x.row_no=s.row_no
                JOIN betalens.entity_dim e ON e.code=s.code
                WHERE NOT x.existed OR x.changed
                ON CONFLICT (index_entity_id, effective_at) DO UPDATE
                SET index_name_snapshot=EXCLUDED.index_name_snapshot, remark=EXCLUDED.remark
                """
            )
            cur.execute(
                """
                DELETE FROM betalens.index_constituent c
                USING _betalens_import_stage s, _betalens_index_existing x,
                      betalens.entity_dim e,
                      betalens.index_snapshot sn
                WHERE x.row_no=s.row_no AND x.existed AND x.changed
                  AND e.code=s.code AND sn.index_entity_id=e.entity_id
                  AND sn.effective_at=s.datetime AND c.snapshot_id=sn.snapshot_id
                """
            )
        cur.execute(
            """
            INSERT INTO betalens.index_constituent
                (snapshot_id, constituent_entity_id, ordinal, weight, remark)
            SELECT sn.snapshot_id, member.entity_id, constituent.ordinal,
                   constituent.weight, constituent.remark
            FROM _betalens_import_stage s
            JOIN _betalens_index_existing x ON x.row_no=s.row_no
            JOIN betalens.entity_dim idx ON idx.code=s.code
            JOIN betalens.index_snapshot sn
              ON sn.index_entity_id=idx.entity_id AND sn.effective_at=s.datetime
            JOIN _betalens_index_member_stage constituent ON constituent.row_no=s.row_no
            JOIN betalens.entity_dim member ON member.code=constituent.code
            WHERE NOT x.existed OR (%s AND x.changed)
            ON CONFLICT (snapshot_id, constituent_entity_id) DO UPDATE
            SET ordinal=EXCLUDED.ordinal, weight=EXCLUDED.weight, remark=EXCLUDED.remark
            """,
            (mode == UPSERT,),
        )

    def _write_trade_status(self, cur, mode: str) -> None:
        if mode == INSERT_ONLY:
            cur.execute(
                """
                UPDATE betalens.entity_dim e SET first_trade_date=s.first_date,
                       updated_at=now()
                FROM (
                    SELECT source.code, min(source.datetime::date) AS first_date
                    FROM _betalens_import_stage source
                    JOIN _betalens_trade_status st ON st.row_no=source.row_no
                    WHERE source.value=1 AND NOT st.exists GROUP BY source.code
                ) s
                WHERE e.code=s.code AND e.first_trade_date IS NULL
                """
            )
            cur.execute(
                """
                INSERT INTO betalens.trade_status_event
                    (entity_id, event_date, status, status_text, remark)
                SELECT e.entity_id, s.datetime::date, s.value::smallint,
                       COALESCE(s.remark->>'status', '异常'), s.remark
                FROM _betalens_import_stage s
                JOIN _betalens_trade_status st ON st.row_no=s.row_no AND NOT st.exists
                JOIN betalens.entity_dim e ON e.code=s.code
                WHERE s.value=0
                ON CONFLICT (entity_id, event_date) DO NOTHING
                """
            )
        else:
            cur.execute(
                """
                UPDATE betalens.entity_dim e
                SET first_trade_date=LEAST(COALESCE(e.first_trade_date, s.first_date), s.first_date),
                    updated_at=now()
                FROM (
                    SELECT source.code, min(source.datetime::date) AS first_date
                    FROM _betalens_import_stage source
                    JOIN _betalens_trade_status st ON st.row_no=source.row_no
                    WHERE source.value=1 AND (NOT st.exists OR st.changed) GROUP BY source.code
                ) s WHERE e.code=s.code
                """
            )
            cur.execute(
                """
                INSERT INTO betalens.trade_status_event
                    (entity_id, event_date, status, status_text, remark)
                SELECT e.entity_id, s.datetime::date, s.value::smallint,
                       COALESCE(s.remark->>'status', '异常'), s.remark
                FROM _betalens_import_stage s
                JOIN _betalens_trade_status st
                  ON st.row_no=s.row_no AND (NOT st.exists OR st.changed)
                JOIN betalens.entity_dim e ON e.code=s.code
                WHERE s.value=0
                ON CONFLICT (entity_id, event_date) DO UPDATE
                SET status=EXCLUDED.status, status_text=EXCLUDED.status_text,
                    remark=EXCLUDED.remark
                """
            )

    @staticmethod
    def _write_trade_calendar(cur) -> None:
        cur.execute(
            """
            INSERT INTO betalens.trade_calendar_day (exchange, trade_date)
            SELECT s.code, s.datetime::date
            FROM _betalens_import_stage s
            JOIN _betalens_trade_calendar status ON status.row_no=s.row_no
            WHERE NOT status.exists
            ON CONFLICT (exchange, trade_date) DO NOTHING
            """
        )

    def _ensure_observation_partitions(self, cur, *, connection) -> None:
        cur.execute(
            """
            SELECT DISTINCT extract(year FROM datetime)::integer AS year
            FROM _betalens_resolved_stage WHERE storage_kind='observation'
            ORDER BY year
            """
        )
        years = [int(row["year"]) for row in cur.fetchall()]
        from .schema import SchemaManager

        SchemaManager(self.client.db_config).ensure_observation_partitions(years, connection=connection)

    def _update_coverage(self, cur, table: str, frame: pd.DataFrame, inserted: int) -> None:
        min_dt = pd.Timestamp(frame["datetime"].min()).to_pydatetime()
        max_dt = pd.Timestamp(frame["datetime"].max()).to_pydatetime()
        cur.execute(
            """
            INSERT INTO betalens.dataset_coverage
                (logical_dataset, min_available_at, max_available_at, row_count, metadata)
            VALUES (%s, %s, %s, %s, jsonb_build_object('last_operation', 'write'))
            ON CONFLICT (logical_dataset) DO UPDATE SET
                min_available_at=CASE
                    WHEN betalens.dataset_coverage.min_available_at IS NULL THEN EXCLUDED.min_available_at
                    ELSE LEAST(betalens.dataset_coverage.min_available_at, EXCLUDED.min_available_at) END,
                max_available_at=CASE
                    WHEN betalens.dataset_coverage.max_available_at IS NULL THEN EXCLUDED.max_available_at
                    ELSE GREATEST(betalens.dataset_coverage.max_available_at, EXCLUDED.max_available_at) END,
                row_count=betalens.dataset_coverage.row_count + EXCLUDED.row_count,
                updated_at=now(),
                metadata=betalens.dataset_coverage.metadata || EXCLUDED.metadata
            """,
            (table, min_dt, max_dt, int(inserted)),
        )

    def _legacy_dry_run(self, cur, table: str, frame: pd.DataFrame, sample_limit: int) -> dict[str, Any]:
        if not self._temp_stage_exists(cur):
            self._copy_stage(cur, frame)
        query = sql.SQL(
            """
            SELECT s.datetime, s.code, s.metric, t.value AS db_value,
                   s.value AS new_value,
                   t.datetime IS NOT NULL AS exists,
                   (t.value::double precision IS DISTINCT FROM s.value
                       OR t.name IS DISTINCT FROM s.name
                       OR t.remark IS DISTINCT FROM s.remark) AS changed
            FROM _betalens_import_stage s
            LEFT JOIN public.{} t
              ON t.datetime=s.datetime AND t.code=s.code AND t.metric=s.metric
            ORDER BY s.row_no
            """
        ).format(sql.Identifier(table))
        cur.execute(query)
        rows = [dict(row) for row in cur.fetchall()]
        existing = [row for row in rows if row["exists"]]
        changed = [row for row in existing if row["changed"]]
        return {
            "existing": len(existing),
            "conflict_count": len(changed),
            "conflicts": changed[:sample_limit],
        }

    @staticmethod
    def _temp_stage_exists(cur) -> bool:
        cur.execute("SELECT to_regclass('pg_temp._betalens_import_stage') IS NOT NULL")
        row = cur.fetchone()
        return bool(next(iter(row.values()))) if isinstance(row, dict) else bool(row and row[0])

    def _legacy_write(
        self,
        cur,
        table: str,
        frame: pd.DataFrame,
        mode: str,
        batch_size: int,
        *,
        stage_loaded: bool = False,
        total_override: int | None = None,
    ) -> dict[str, Any]:
        if not stage_loaded:
            self._copy_stage(cur, frame, batch_size=batch_size)
        inspection = self._legacy_dry_run(cur, table, frame, 20)
        total = int(total_override if total_override is not None else len(frame))
        cur.execute("SELECT count(*) AS count FROM _betalens_import_stage")
        count_row = cur.fetchone()
        staged_total = int(count_row["count"] if isinstance(count_row, dict) else count_row[0])
        inserted = staged_total - inspection["existing"]
        updated = inspection["conflict_count"] if mode == UPSERT else 0
        skipped = total - inserted - updated
        query = sql.SQL(
            """
            INSERT INTO public.{} (datetime, code, name, metric, value, remark)
            SELECT datetime, code, name, metric, value, remark
            FROM _betalens_import_stage
            ON CONFLICT (datetime, code, metric) {}
            """
        ).format(
            sql.Identifier(table),
            sql.SQL("DO NOTHING")
            if mode == INSERT_ONLY
            else sql.SQL(
                "DO UPDATE SET name=EXCLUDED.name, value=EXCLUDED.value, remark=EXCLUDED.remark "
                "WHERE {}.value::double precision IS DISTINCT FROM EXCLUDED.value "
                "OR {}.name IS DISTINCT FROM EXCLUDED.name "
                "OR {}.remark IS DISTINCT FROM EXCLUDED.remark"
            ).format(sql.Identifier(table), sql.Identifier(table), sql.Identifier(table)),
        )
        cur.execute(query)
        return {
            "inserted": int(inserted),
            "updated": int(updated),
            "skipped": int(skipped),
            "total": total,
        }

    def _delete_new_schema(self, cur, table: str, request: DeleteRequest) -> int:
        storage = DATASETS[table].storage
        if storage in {"market", "observation"}:
            return self._delete_market_observation(cur, table, request)
        if storage == "industry":
            return self._delete_industry(cur, request)
        if storage == "index_universe":
            if request.metric and request.metric != "universe":
                return 0
            conditions, params = self._delete_conditions(request, "e.code", "sn.effective_at")
            cur.execute(
                "DELETE FROM betalens.index_snapshot sn USING betalens.entity_dim e "
                "WHERE e.entity_id=sn.index_entity_id AND " + " AND ".join(conditions),
                params,
            )
            deleted = cur.rowcount
            self._mark_coverage_stale(cur, table, deleted)
            return deleted
        if storage == "trade_calendar":
            if request.metric and request.metric != "交易日":
                return 0
            conditions, params = self._delete_conditions(request, "upper(exchange)", "trade_date")
            if request.normalized_codes():
                params[0] = [value.upper() for value in params[0]]
            cur.execute(
                "DELETE FROM betalens.trade_calendar_day WHERE " + " AND ".join(conditions),
                params,
            )
            deleted = cur.rowcount
            self._mark_coverage_stale(cur, table, deleted)
            return deleted
        if request.metric and request.metric != "交易状态":
            return 0
        return self._delete_trade_status(cur, request)

    def _delete_market_observation(self, cur, table: str, request: DeleteRequest) -> int:
        metric_row = None
        if request.metric:
            cur.execute(
                """
                SELECT md.metric_id, md.storage_kind, md.storage_column
                FROM betalens.metric_alias ma
                JOIN betalens.metric_dim md ON md.metric_id=ma.metric_id
                WHERE ma.logical_dataset=%s AND ma.alias=%s
                UNION ALL
                SELECT metric_id, storage_kind, storage_column
                FROM betalens.metric_dim WHERE logical_dataset=%s AND metric_name=%s
                LIMIT 1
                """,
                (table, request.metric, table, request.metric),
            )
            metric_row = cur.fetchone()
            if metric_row is None:
                return 0
        deleted = 0
        if DATASETS[table].storage == "market" and (
            metric_row is None or metric_row["storage_kind"] == "core"
        ):
            core_request = request
            conditions, params = self._delete_conditions(core_request, "e.code", "f.trade_date")
            if metric_row is None:
                cur.execute(
                    """
                    SELECT COALESCE(sum(num_nonnulls(
                        f.open, f.high, f.low, f.close, f.prev_close,
                        f.volume, f.amount, f.turnover_rate)), 0)::bigint AS logical_rows
                    FROM betalens.market_daily_fact f
                    JOIN betalens.entity_dim e ON e.entity_id=f.entity_id
                    WHERE e.entity_type=%s AND
                    """ + " AND ".join(conditions),
                    [DATASETS[table].entity_type, *params],
                )
                logical_rows = int(cur.fetchone()["logical_rows"])
                cur.execute(
                    "DELETE FROM betalens.market_daily_fact f USING betalens.entity_dim e "
                    "WHERE e.entity_id=f.entity_id AND e.entity_type=%s AND " + " AND ".join(conditions),
                    [DATASETS[table].entity_type, *params],
                )
                deleted += logical_rows
            else:
                column = metric_row["storage_column"]
                allowed = {"open", "high", "low", "close", "prev_close", "volume", "amount", "turnover_rate"}
                if column not in allowed:
                    raise RuntimeError(f"非法核心行情列: {column}")
                query = sql.SQL(
                    "UPDATE betalens.market_daily_fact f SET {}=NULL, updated_at=now() "
                    "FROM betalens.entity_dim e WHERE e.entity_id=f.entity_id "
                    "AND e.entity_type=%s AND {} IS NOT NULL AND "
                ).format(sql.Identifier(column), sql.Identifier(column)) + sql.SQL(" AND ").join(
                    sql.SQL(value) for value in conditions
                )
                cur.execute(query, [DATASETS[table].entity_type, *params])
                deleted += cur.rowcount
        conditions, params = self._delete_conditions(request, "e.code", "o.available_at")
        metric_condition = ""
        metric_params: list[Any] = []
        if metric_row is not None:
            metric_condition = " AND o.metric_id=%s"
            metric_params.append(metric_row["metric_id"])
        cur.execute(
            """
            DELETE FROM betalens.observation_fact o
            USING betalens.entity_dim e, betalens.metric_dim md
            WHERE e.entity_id=o.entity_id AND md.metric_id=o.metric_id
              AND md.logical_dataset=%s
            """ + metric_condition + " AND " + " AND ".join(conditions),
            [table, *metric_params, *params],
        )
        deleted += cur.rowcount
        self._mark_coverage_stale(cur, table, deleted)
        return deleted

    def _delete_industry(self, cur, request: DeleteRequest) -> int:
        conditions, params = self._delete_conditions(request, "e.code", "im.valid_from")
        if request.metric:
            conditions.append("sc.scheme_name=%s")
            params.append(request.metric)
        cur.execute(
            """
            DELETE FROM betalens.industry_membership im
            USING betalens.entity_dim e, betalens.industry_dim d,
                  betalens.industry_scheme_dim sc
            WHERE e.entity_id=im.entity_id AND d.industry_id=im.industry_id
              AND sc.scheme_id=d.scheme_id AND
            """ + " AND ".join(conditions),
            params,
        )
        deleted = cur.rowcount
        self._refresh_industry_ranges(cur)
        self._mark_coverage_stale(cur, "industry", deleted)
        return deleted

    @staticmethod
    def _refresh_industry_ranges(cur) -> None:
        cur.execute(
            """
            WITH bounds AS (
                SELECT im.entity_id, im.industry_id, im.valid_from,
                       lead(im.valid_from) OVER (
                           PARTITION BY im.entity_id, d.scheme_id ORDER BY im.valid_from) AS valid_to
                FROM betalens.industry_membership im
                JOIN betalens.industry_dim d ON d.industry_id=im.industry_id
            )
            UPDATE betalens.industry_membership im SET valid_to=b.valid_to
            FROM bounds b WHERE im.entity_id=b.entity_id AND im.industry_id=b.industry_id
              AND im.valid_from=b.valid_from AND im.valid_to IS DISTINCT FROM b.valid_to
            """
        )

    def _delete_trade_status(self, cur, request: DeleteRequest) -> int:
        conditions, params = self._delete_conditions(request, "e.code", "t.event_date")
        cur.execute(
            "DELETE FROM betalens.trade_status_event t USING betalens.entity_dim e "
            "WHERE e.entity_id=t.entity_id AND " + " AND ".join(conditions),
            params,
        )
        deleted = cur.rowcount
        anchor_conditions, anchor_params = self._delete_conditions(request, "e.code", "e.first_trade_date")
        cur.execute(
            "UPDATE betalens.entity_dim e SET first_trade_date=NULL, updated_at=now() "
            "WHERE e.entity_type='stock' AND e.first_trade_date IS NOT NULL AND "
            + " AND ".join(anchor_conditions),
            anchor_params,
        )
        deleted += cur.rowcount
        self._mark_coverage_stale(cur, "trade_status", deleted)
        return deleted

    @staticmethod
    def _delete_conditions(
        request: DeleteRequest,
        code_expression: str,
        datetime_expression: str,
    ) -> tuple[list[str], list[Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        codes = request.normalized_codes()
        if codes:
            conditions.append(f"{code_expression} = ANY(%s)")
            params.append(list(codes))
        if request.start_date:
            conditions.append(f"{datetime_expression} >= %s::timestamp")
            params.append(request.start_date)
        if request.end_date:
            if DatabaseClient._is_date_only(request.end_date):
                conditions.append(f"{datetime_expression} < %s::date + interval '1 day'")
            else:
                conditions.append(f"{datetime_expression} <= %s::timestamp")
            params.append(request.end_date)
        if not conditions:
            conditions.append("TRUE")
        return conditions, params

    def _mark_coverage_stale(self, cur, table: str, deleted: int) -> None:
        if deleted <= 0:
            return
        cur.execute(
            """
            UPDATE betalens.dataset_coverage
            SET row_count=GREATEST(0, row_count-%s), updated_at=now(),
                metadata=metadata || jsonb_build_object(
                    'last_operation', 'delete', 'coverage_bounds_stale', true)
            WHERE logical_dataset=%s
            """,
            (deleted, table),
        )

    def _legacy_delete(self, cur, request: DeleteRequest) -> int:
        conditions: list[sql.SQL] = []
        params: list[Any] = []
        codes = request.normalized_codes()
        if codes:
            conditions.append(sql.SQL("code = ANY(%s)"))
            params.append(list(codes))
        if request.metric:
            conditions.append(sql.SQL("metric=%s"))
            params.append(request.metric)
        if request.start_date:
            conditions.append(sql.SQL("datetime >= %s::timestamp"))
            params.append(request.start_date)
        if request.end_date:
            if DatabaseClient._is_date_only(request.end_date):
                conditions.append(sql.SQL("datetime < %s::date + interval '1 day'"))
            else:
                conditions.append(sql.SQL("datetime <= %s::timestamp"))
            params.append(request.end_date)
        query = sql.SQL("DELETE FROM public.{} WHERE ").format(sql.Identifier(request.table))
        query += sql.SQL(" AND ").join(conditions)
        cur.execute(query, params)
        return cur.rowcount
