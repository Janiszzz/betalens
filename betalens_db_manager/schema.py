"""Schema management services for the local Betalens database."""

from __future__ import annotations

import re
from typing import Any

import psycopg2
import psycopg2.extras
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from betalens.datafeed.config import get_database_config

from .constants import ALLOWED_TABLES
from .utils import clean_database_config


TABLE_DESCRIPTIONS = {
    "daily_market": "个券行情（日频）- 开盘价最早09:30，其余价量最早15:00",
    "fundamentals": "个券基本面（日频入库，事件驱动）- 按公告时点入库",
    "macro": "宏观经济数据（事件驱动）- 区分公告时点与发生时点",
    "factors": "因子库 - 存储计算好的因子数据",
    "industry": "证券分类归属（point-in-time）- 行业/指数成分等，datetime为生效时点",
    "index_universe": "指数历史股票池（point-in-time快照）- datetime为生效时点，remark.constituents存成分股列表",
    "trade_status": "个券交易状态（稀疏存储）- 仅存异常状态与首次正常交易日，正常交易日由查询时推断",
}

COLUMN_COMMENTS = {
    "datetime": "入库实际时间（最早可交易时间）",
    "code": "证券代码（WindCode格式，如000001.SZ）",
    "name": "证券中文名称",
    "metric": "指标名称（如：收盘价(元)、成交量(股)）",
    "value": "数值",
    "remark": "备注信息（JSONB）",
}

SPECIAL_COLUMN_COMMENTS = {
    "fundamentals": {
        "datetime": "入库实际时间（最早可交易时间）",
        "remark": "备注信息，可包含理论发生时间（报告期：0331/0630/0930/1231）",
    },
    "macro": {
        "code": "指标代码（WindCode格式）",
        "name": "指标名称",
        "remark": "备注信息，可包含理论发生时间（如2024年1月GDP）",
    },
    "factors": {
        "metric": "因子名称/数据编制方式",
        "remark": "备注信息，可包含因子计算参数和元数据",
    },
    "industry": {
        "datetime": "归属关系生效时点（最早可知日）",
        "metric": "分类体系（如：申万一级行业、中信一级行业）",
        "value": "行业代码数值部分（如801780），便于索引分组",
        "remark": "备注JSONB，约定 {ind_name, ind_code, scheme}",
    },
    "index_universe": {
        "datetime": "股票池生效时点（最早可知日）",
        "code": "指数代码（WindCode格式，如000906.SH）",
        "metric": "固定为 universe（标识成分股池）",
        "value": "成分股数量（便于校验）",
        "remark": "备注JSONB，约定 {index_code, index_name, constituents}",
    },
    "trade_status": {
        "datetime": "状态生效日（停牌日或首次正常交易日），最早可知时点",
        "metric": "固定为 交易状态",
        "value": "状态编码：1=正常交易（仅首次正常交易日入库做锚点），0=异常（停牌/暂停上市等）",
        "remark": "备注JSONB，约定 {status:状态文本, first_normal:bool}",
    },
}


def validate_database_name(dbname: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9_]+$", dbname))


def create_table_sql(table_name: str) -> sql.SQL:
    return sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {} (
            datetime TIMESTAMP NOT NULL,
            code VARCHAR(20) NOT NULL,
            name VARCHAR(100) NOT NULL,
            metric VARCHAR(100) NOT NULL,
            value NUMERIC(50, 6),
            remark JSONB,
            CONSTRAINT {} UNIQUE (datetime, code, metric)
        )
        """
    ).format(
        sql.Identifier(table_name),
        sql.Identifier(f"uq_{table_name}_datetime_code_metric"),
    )


def create_index_sql(table_name: str) -> list[sql.SQL]:
    return [
        sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (code, metric, datetime)").format(
            sql.Identifier(f"idx_{table_name}_code_metric_datetime"),
            sql.Identifier(table_name),
        ),
        sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (datetime)").format(
            sql.Identifier(f"idx_{table_name}_datetime"),
            sql.Identifier(table_name),
        ),
        sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (code, datetime)").format(
            sql.Identifier(f"idx_{table_name}_code_datetime"),
            sql.Identifier(table_name),
        ),
        sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (metric)").format(
            sql.Identifier(f"idx_{table_name}_metric"),
            sql.Identifier(table_name),
        ),
    ]


class SchemaManager:
    """Non-interactive schema creation and verification service."""

    def __init__(self, db_config: dict[str, Any] | None = None):
        self.db_config = clean_database_config(db_config or get_database_config())

    def validate_table(self, table_name: str) -> str:
        if table_name not in ALLOWED_TABLES:
            raise ValueError(f"非法表名: {table_name}")
        return table_name

    def connect(self, dbname: str | None = None):
        cfg = dict(self.db_config)
        if dbname:
            cfg["dbname"] = dbname
        return psycopg2.connect(**cfg)

    def database_exists(self, dbname: str | None = None) -> bool:
        dbname = dbname or self.db_config["dbname"]
        with self.connect("postgres") as conn:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
                return cur.fetchone() is not None

    def create_database(self, dbname: str | None = None) -> None:
        dbname = dbname or self.db_config["dbname"]
        if not validate_database_name(dbname):
            raise ValueError(f"非法数据库名: {dbname}")
        with self.connect("postgres") as conn:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("CREATE DATABASE {} ENCODING 'UTF8' TEMPLATE template0").format(
                        sql.Identifier(dbname)
                    )
                )

    def table_exists(self, cur, table_name: str) -> bool:
        table_name = self.validate_table(table_name)
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = %s
            )
            """,
            (table_name,),
        )
        row = cur.fetchone()
        return row["exists"] if isinstance(row, dict) else row[0]

    def drop_table(self, cur, table_name: str) -> None:
        table_name = self.validate_table(table_name)
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table_name)))

    def create_table(self, cur, table_name: str, create_indexes: bool = True, create_comments: bool = True) -> None:
        table_name = self.validate_table(table_name)
        cur.execute(create_table_sql(table_name))
        if create_indexes:
            for stmt in create_index_sql(table_name):
                cur.execute(stmt)
        if create_comments:
            self.comment_table(cur, table_name)

    def comment_table(self, cur, table_name: str) -> None:
        table_name = self.validate_table(table_name)
        cur.execute(
            sql.SQL("COMMENT ON TABLE {} IS %s").format(sql.Identifier(table_name)),
            (TABLE_DESCRIPTIONS.get(table_name, ""),),
        )
        comments = {**COLUMN_COMMENTS, **SPECIAL_COLUMN_COMMENTS.get(table_name, {})}
        for column, comment in comments.items():
            cur.execute(
                sql.SQL("COMMENT ON COLUMN {}.{} IS %s").format(
                    sql.Identifier(table_name),
                    sql.Identifier(column),
                ),
                (comment,),
            )

    def ensure_schema(
        self,
        tables: list[str] | None = None,
        force: bool = False,
        create_database_if_missing: bool = False,
        create_indexes: bool = True,
        create_comments: bool = True,
    ) -> dict[str, Any]:
        tables = tables or list(ALLOWED_TABLES)
        if not self.database_exists():
            if not create_database_if_missing:
                raise RuntimeError("数据库不存在；请显式传入 --create-database 创建")
            self.create_database()

        result: dict[str, Any] = {"created": [], "dropped": [], "skipped": []}
        with self.connect() as conn:
            with conn.cursor() as cur:
                for table in tables:
                    table = self.validate_table(table)
                    exists = self.table_exists(cur, table)
                    if exists and force:
                        self.drop_table(cur, table)
                        result["dropped"].append(table)
                        exists = False
                    if exists:
                        result["skipped"].append(table)
                        continue
                    self.create_table(cur, table, create_indexes=create_indexes, create_comments=create_comments)
                    result["created"].append(table)
                conn.commit()
        return result

    def verify_schema(self) -> dict[str, Any]:
        result: dict[str, Any] = {"database_exists": False, "tables": {}, "errors": []}
        try:
            with self.connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    result["database_exists"] = True
                    for table_name in ALLOWED_TABLES:
                        info: dict[str, Any] = {
                            "exists": False,
                            "columns": [],
                            "indexes": [],
                            "constraints": [],
                        }
                        if self.table_exists(cur, table_name):
                            info["exists"] = True
                            cur.execute(
                                """
                                SELECT column_name, data_type, is_nullable
                                FROM information_schema.columns
                                WHERE table_schema = 'public'
                                  AND table_name = %s
                                ORDER BY ordinal_position
                                """,
                                (table_name,),
                            )
                            info["columns"] = [dict(row) for row in cur.fetchall()]
                            cur.execute(
                                """
                                SELECT indexname
                                FROM pg_indexes
                                WHERE schemaname = 'public'
                                  AND tablename = %s
                                ORDER BY indexname
                                """,
                                (table_name,),
                            )
                            info["indexes"] = [row["indexname"] for row in cur.fetchall()]
                            cur.execute(
                                """
                                SELECT conname, contype
                                FROM pg_constraint
                                WHERE conrelid = (
                                    SELECT oid FROM pg_class
                                    WHERE relname = %s
                                      AND relnamespace = 'public'::regnamespace
                                )
                                ORDER BY conname
                                """,
                                (table_name,),
                            )
                            info["constraints"] = [dict(row) for row in cur.fetchall()]
                        result["tables"][table_name] = info
        except Exception as exc:
            result["errors"].append(str(exc))
        return result
