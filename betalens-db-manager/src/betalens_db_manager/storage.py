"""Canonical PostgreSQL storage and target-independent audit operations."""
from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from datetime import date, datetime, timezone
from typing import Iterable

import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor, execute_values

from .canonical import CanonicalTable, canonical, load_canonical_catalog
from .config import Settings
from .comparison import compare as compare_rows, row_hash, row_key, typed
from .util import digest, dumps, identifier, parse_date

PG_TYPES = {
    "text": "text",
    "date": "date",
    "timestamp": "timestamp with time zone",
    "numeric": "numeric(38,12)",
    "integer": "integer",
}

def pg_type(field):
    return f"numeric({field.get('precision',38)},{field.get('scale',12)})" if field['type']=='numeric' else PG_TYPES[field['type']]

# Compatibility names used by the source and inspection modules.
META = {}


def columns(spec: CanonicalTable) -> list[str]:
    return [field["name"] for field in spec.fields]


def primary(spec: CanonicalTable) -> list[str]:
    return list(spec.keys)


def relation(settings: Settings, name: str):
    return sql.Identifier(settings.pg_schema, identifier(name))


CONTROL_DDL = {
    "meta_table": [
        ("table_name", "text", True), ("table_zh_name", "text", True), ("description", "text", True),
        ("grain", "text", True), ("domain", "text", True), ("mapping_version", "text", True),
        ("updated_at", "timestamp with time zone", True), ("schema_version","text",False), ("business_keys","text",False), ("time_field","text",False), ("temporal_rule","text",False), ("sources","text",False),
    ],
    "meta_column": [
        ("table_name", "text", True), ("column_name", "text", True), ("column_zh_name", "text", True),
        ("description", "text", True), ("unit", "text", True), ("data_type", "text", True),
        ("ordinal", "integer", True), ("mapping_version", "text", True), ("updated_at", "timestamp with time zone", True), ("date_role", "text", False), ("precision","integer",False), ("scale","integer",False), ("nullable","integer",False), ("system_generated","integer",False),
    ],
    "meta_source_mapping": [
        ("source_name", "text", True), ("source_endpoint", "text", True), ("source_field", "text", True),
        ("canonical_table", "text", True), ("canonical_column", "text", True), ("source_type", "text", True),
        ("target_type", "text", True), ("source_unit", "text", False), ("target_unit", "text", False),
        ("transform_expression", "text", False), ("date_role", "text", False), ("required", "integer", True),
        ("mapping_version", "text", True), ("description", "text", False),
    ],
    "meta_source_capability": [
        ("source_name", "text", True), ("endpoint", "text", True), ("capability", "text", True),
        ("status", "text", True), ("permission_note", "text", False), ("last_probe_at", "timestamp with time zone", False),
        ("sample_rows", "integer", False), ("response_fields", "text", False), ("error", "text", False),
    ],
    "import_task": [
        ("task_id", "text", True), ("source_name", "text", True), ("targets", "text", True),
        ("status", "text", True), ("request", "jsonb", True), ("mapping_version", "text", True),
        ("created_at", "timestamp with time zone", True), ("started_at", "timestamp with time zone", False), ("finished_at", "timestamp with time zone", False),
        ("error", "text", False),
    ],
    "import_source_archive": [
        ("task_id", "text", True), ("dataset", "text", True), ("endpoint", "text", True),
        ("request_hash", "text", True), ("archive_path", "text", True), ("response_sha256", "text", True),
        ("response_fields", "text", True), ("fetched_at", "timestamp with time zone", True), ("rows", "integer", True),
    ],
    "import_target_status": [
        ("task_id", "text", True), ("target", "text", True), ("status", "text", True),
        ("row_count", "bigint", False), ("table_count", "integer", False), ("key_hash", "text", False),
        ("field_hash", "text", False), ("started_at", "timestamp with time zone", False), ("finished_at", "timestamp with time zone", False),
        ("error", "text", False),
    ],
    "import_row_audit": [
        ("id", "bigint GENERATED ALWAYS AS IDENTITY", True), ("task_id", "text", True),
        ("target", "text", True), ("table_name", "text", True), ("row_key", "text", True),
        ("operation", "text", True), ("before_hash", "text", False), ("after_hash", "text", False),
        ("before_value", "jsonb", False), ("after_value", "jsonb", False), ("created_at", "timestamp with time zone", True),
    ],
    "import_rollback": [
        ("rollback_task_id", "text", True), ("source_task_id", "text", True), ("target", "text", True),
        ("status", "text", True), ("restored", "bigint", True), ("deleted", "bigint", True),
        ("conflicts", "bigint", True), ("created_at", "timestamp with time zone", True), ("finished_at", "timestamp with time zone", False),
        ("error", "text", False),
    ],
    "dataset_coverage": [
        ("table_name", "text", True), ("asset_type", "text", True), ("security_code", "text", True),
        ("first_event_date", "date", False), ("last_event_date", "date", False), ("rows", "bigint", True),
        ("updated_at", "timestamp with time zone", True),
    ],
}


class Postgres:
    def __init__(self, settings: Settings):
        self.settings = settings

    @contextmanager
    def connect(self, readonly: bool = False, database: str | None = None):
        conn = psycopg2.connect(**self.settings.pg(database))
        try:
            conn.set_session(readonly=readonly)
            with conn.cursor() as cur:
                cur.execute("SET TIME ZONE 'UTC'")
                cur.execute("SET lock_timeout='10s'")
                cur.execute("SET statement_timeout='180s'")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_database(self):
        conn = psycopg2.connect(**self.settings.pg("postgres"))
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (self.settings.pg_database,))
                if not cur.fetchone():
                    cur.execute(sql.SQL("CREATE DATABASE {} ENCODING 'UTF8' TEMPLATE template0").format(sql.Identifier(self.settings.pg_database)))
        finally:
            conn.close()

    def _existing_columns(self, cur) -> dict[str, dict[str, str]]:
        cur.execute("SELECT table_name,column_name,data_type FROM information_schema.columns WHERE table_schema=%s", (self.settings.pg_schema,))
        result: dict[str, dict[str, str]] = {}
        for table, column, data_type in cur.fetchall():
            result.setdefault(table, {})[column] = data_type
        return result

    def schema_plan(self) -> dict:
        tables = load_canonical_catalog()
        try:
            with self.connect(readonly=True) as conn, conn.cursor() as cur:
                existing = self._existing_columns(cur)
                missing, mismatches, missing_columns = [], [], []
                for name, spec in {**tables, **{k: None for k in CONTROL_DDL}}.items():
                    expected = ({field["name"]: pg_type(field) for field in spec.fields} if spec else
                                {field[0]: PG_TYPES.get(field[1], field[1]) for field in CONTROL_DDL[name]})
                    if name not in existing:
                        missing.append(name)
                        continue
                    for col, typ in expected.items():
                        actual = existing[name].get(col)
                        # information_schema reports numeric without precision and timestamp with time zone verbatim.
                        normalized = "numeric" if typ.startswith("numeric") else "bigint" if typ.startswith("bigint ") else typ
                        if actual is None:
                            missing_columns.append(f'{name}.{col}');continue
                        if actual != normalized and not (normalized == "bigint" and actual == "bigint"):
                            mismatches.append(f"{name}.{col}: 期望 {normalized}，实际 {actual or '缺失'}")
                return {"target": self.settings.fingerprint, "database_missing": False, "create": missing,
                        "management": [], "missing_columns":missing_columns, "mismatches": mismatches, "ready": not (missing or missing_columns or mismatches)}
        except psycopg2.OperationalError:
            with self.connect(readonly=True, database="postgres") as conn, conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (self.settings.pg_database,))
                if cur.fetchone():
                    raise
            return {"target": self.settings.fingerprint, "database_missing": True,
                    "create": list(tables) + list(CONTROL_DDL), "management": [], "mismatches": [], "ready": False}

    def _create_table(self, cur, spec: CanonicalTable):
        fields = [sql.SQL("{} {}{}").format(sql.Identifier(f["name"]), sql.SQL(pg_type(f)),
                                            sql.SQL(" NOT NULL" if not f.get("nullable", True) else "")) for f in spec.fields]
        fields.append(sql.SQL("PRIMARY KEY ({})").format(sql.SQL(",").join(sql.Identifier(k) for k in spec.keys)))
        cur.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(relation(self.settings, spec.name), sql.SQL(",").join(fields)))

    def _indexes(self,cur,spec):
        cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} ({})").format(
            sql.Identifier(spec.name + "_event_idx"), relation(self.settings, spec.name), sql.Identifier("event_date"))) if "event_date" in columns(spec) else None
        if "security_code" in columns(spec):
            cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (security_code)").format(
                sql.Identifier(spec.name + "_security_idx"), relation(self.settings, spec.name)))

    def install(self, expected: str | None = None, specs=None):
        plan = self.schema_plan()
        if expected and digest(plan) != expected:
            raise ValueError("结构计划已变化，请重新检查")
        if plan["mismatches"] and specs is None:
            raise ValueError("结构不兼容：" + "; ".join(plan["mismatches"]))
        if plan["database_missing"]:
            self.create_database()
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.settings.pg_schema)))
            for spec in (specs if specs is not None else load_canonical_catalog()).values():
                self._create_table(cur, spec)
                for field in spec.fields:
                    cur.execute(sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} {}").format(relation(self.settings,spec.name),sql.Identifier(field["name"]),sql.SQL(pg_type(field))))
                self._indexes(cur,spec)
            for name, definitions in CONTROL_DDL.items():
                fields = []
                for col, typ, required in definitions:
                    fields.append(sql.SQL("{} {}{}").format(sql.Identifier(col), sql.SQL(typ), sql.SQL(" NOT NULL" if required else "")))
                if name == "meta_table": fields.append(sql.SQL("PRIMARY KEY (table_name)"))
                elif name == "meta_column": fields.append(sql.SQL("PRIMARY KEY (table_name,column_name)"))
                elif name == "meta_source_mapping": fields.append(sql.SQL("PRIMARY KEY (source_name,source_endpoint,source_field,canonical_table,canonical_column)"))
                elif name == "meta_source_capability": fields.append(sql.SQL("PRIMARY KEY (source_name,endpoint)"))
                elif name == "import_task": fields.append(sql.SQL("PRIMARY KEY (task_id)"))
                elif name == "import_source_archive": fields.append(sql.SQL("PRIMARY KEY (task_id,dataset,request_hash)"))
                elif name == "import_target_status": fields.append(sql.SQL("PRIMARY KEY (task_id,target)"))
                elif name == "import_rollback": fields.append(sql.SQL("PRIMARY KEY (rollback_task_id,target)"))
                elif name == "import_row_audit": fields.append(sql.SQL("PRIMARY KEY (id)"))
                elif name == "dataset_coverage": fields.append(sql.SQL("UNIQUE (table_name,asset_type,security_code)"))
                cur.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(relation(self.settings, name), sql.SQL(",").join(fields)))
            cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS import_row_audit_task_idx ON {} (task_id,target,table_name,id)").format(relation(self.settings, "import_row_audit")))
            for name,definitions in CONTROL_DDL.items():
                for column,kind,_ in definitions:
                    if kind.startswith('bigint GENERATED'):continue
                    cur.execute(sql.SQL('ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} {}').format(relation(self.settings,name),sql.Identifier(column),sql.SQL(kind)))
            cur.execute(sql.SQL("CREATE UNIQUE INDEX IF NOT EXISTS import_row_audit_id_uidx ON {} (id)").format(relation(self.settings,"import_row_audit")))
            cur.execute(sql.SQL("CREATE UNIQUE INDEX IF NOT EXISTS dataset_coverage_key_uidx ON {} (table_name,asset_type,security_code)").format(relation(self.settings,"dataset_coverage")))
            self._seed_metadata(cur,specs)
        return self.schema_plan()

    def _seed_metadata(self, cur, specs=None):
        from .mapping import all_source_mappings, source_capabilities
        for spec in (specs if specs is not None else load_canonical_catalog()).values():
            cur.execute(sql.SQL("INSERT INTO {} (table_name,table_zh_name,description,grain,domain,mapping_version,updated_at) VALUES (%s,%s,%s,%s,%s,%s,now()) ON CONFLICT(table_name) DO UPDATE SET table_zh_name=excluded.table_zh_name,description=excluded.description,grain=excluded.grain,domain=excluded.domain,mapping_version=excluded.mapping_version,updated_at=now()").format(relation(self.settings, "meta_table")),
                        (spec.name, spec.zh_name, spec.description, spec.grain, spec.domain, "canonical-v2"))
            for ordinal, field in enumerate(spec.fields, 1):
                cur.execute(sql.SQL("INSERT INTO {} (table_name,column_name,column_zh_name,description,unit,data_type,ordinal,mapping_version,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now()) ON CONFLICT(table_name,column_name) DO UPDATE SET column_zh_name=excluded.column_zh_name,description=excluded.description,unit=excluded.unit,data_type=excluded.data_type,ordinal=excluded.ordinal,updated_at=now()").format(relation(self.settings, "meta_column")),
                            (spec.name, field["name"], field["zh_name"], field["description"], field["unit"], field["type"], ordinal, "canonical-v2"))
        for m in all_source_mappings():
            cur.execute(sql.SQL("INSERT INTO {} (source_name,source_endpoint,source_field,canonical_table,canonical_column,source_type,target_type,source_unit,target_unit,transform_expression,date_role,required,mapping_version,description) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING").format(relation(self.settings, "meta_source_mapping")), tuple(m.get(k) for k in ("source_name","source_endpoint","source_field","canonical_table","canonical_column","source_type","target_type","source_unit","target_unit","transform_expression","date_role","required","mapping_version","description")))
        for cap in source_capabilities():
            cur.execute(sql.SQL("INSERT INTO {} (source_name,endpoint,capability,status,permission_note) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING").format(relation(self.settings, "meta_source_capability")), tuple(cap.get(k) for k in ("source_name","endpoint","capability","status","permission_note")))

    def ensure_task(self, task_id: str, source_name: str, targets: list[str], request: dict, mapping_version: str = "canonical-v2"):
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("INSERT INTO {} (task_id,source_name,targets,status,request,mapping_version,created_at) VALUES (%s,%s,%s,%s,%s,%s,now()) ON CONFLICT(task_id) DO NOTHING").format(relation(self.settings, "import_task")), (task_id, source_name, ",".join(targets), "planned", Json(request, dumps=dumps), mapping_version))

    def update_task(self, task_id: str, status: str, error: str | None = None):
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("UPDATE {} SET status=%s,error=%s,started_at=CASE WHEN %s IN ('downloading','committing') AND started_at IS NULL THEN now() ELSE started_at END,finished_at=CASE WHEN %s IN ('completed','failed','partial','cancelled') THEN now() ELSE finished_at END WHERE task_id=%s").format(relation(self.settings, "import_task")), (status, error, status, status, task_id))

    def target_status(self, task_id: str, target: str, status: str, **values):
        values = {"row_count": values.get("row_count"), "table_count": values.get("table_count"), "key_hash": values.get("key_hash"), "field_hash": values.get("field_hash"), "error": values.get("error")}
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("INSERT INTO {} (task_id,target,status,row_count,table_count,key_hash,field_hash,started_at,finished_at,error) VALUES (%s,%s,%s,%s,%s,%s,%s,now(),CASE WHEN %s IN ('completed','failed','rolled_back') THEN now() ELSE NULL END,%s) ON CONFLICT(task_id,target) DO UPDATE SET status=excluded.status,row_count=excluded.row_count,table_count=excluded.table_count,key_hash=excluded.key_hash,field_hash=excluded.field_hash,finished_at=excluded.finished_at,error=excluded.error").format(relation(self.settings, "import_target_status")), (task_id,target,status,values["row_count"],values["table_count"],values["key_hash"],values["field_hash"],status,values["error"]))

    def archive_record(self, task_id: str, dataset: str, endpoint: str, request_hash: str, archive_path: str, response_sha256: str, response_fields: list[str], fetched_at, rows: int):
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("INSERT INTO {} (task_id,dataset,endpoint,request_hash,archive_path,response_sha256,response_fields,fetched_at,rows) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING").format(relation(self.settings, "import_source_archive")),
                        (task_id, dataset, endpoint, request_hash, archive_path, response_sha256, ",".join(response_fields), fetched_at, rows))

    def _fetch_existing(self, cur, spec: CanonicalTable, rows: list[dict]) -> dict[tuple, dict]:
        if not rows:
            return {}
        keys = [tuple(typed(spec,row).get(k) for k in spec.keys) for row in rows]
        aliases = [f"k{i}" for i in range(len(spec.keys))]
        values_sql = sql.SQL(",").join(sql.SQL("%s") for _ in keys)
        statement = sql.SQL("SELECT t.* FROM {} t JOIN (VALUES %s) AS incoming({}) ON {}").format(
            relation(self.settings, spec.name), sql.SQL(",").join(sql.Identifier(x) for x in aliases),
            sql.SQL(" AND ").join(sql.SQL("t.{}=incoming.{}").format(sql.Identifier(k), sql.Identifier(a)) for k,a in zip(spec.keys, aliases)))
        found = execute_values(cur, statement, keys, page_size=1000, fetch=True)
        result = {}
        for item in found:
            row = dict(item)
            result[tuple(row.get(k) for k in spec.keys)] = row
        return result

    def compare(self, spec, rows, mode="fill_missing"):
        with self.connect(readonly=True) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            existing = self._fetch_existing(cur, spec, rows)
            return compare_rows(spec, rows, list(existing.values()), mode)

    def commit_rows(self, task_id: str, target: str, spec: CanonicalTable, rows: list[dict], mode: str, signature: str) -> dict:
        with self.connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(relation(self.settings, spec.name)))
            comparison = self._compare_with_connection(cur, spec, rows, mode)
            if comparison["signature"] != signature:
                raise ValueError("预览后的目标数据已经变化，请重新预览")
            changes = comparison["changes"]
            if changes:
                names = columns(spec)
                values = [[change["after"].get(name) for name in names] for change in changes]
                statement = sql.SQL("INSERT INTO {} ({}) VALUES %s ON CONFLICT ({}) DO UPDATE SET {}").format(
                    relation(self.settings, spec.name), sql.SQL(",").join(sql.Identifier(n) for n in names),
                    sql.SQL(",").join(sql.Identifier(k) for k in spec.keys),
                    sql.SQL(",").join(sql.SQL("{}=EXCLUDED.{}").format(sql.Identifier(n), sql.Identifier(n)) for n in names if n not in spec.keys))
                execute_values(cur, statement, values, page_size=500)
                # Before/after is durably prepared in the orchestration ledger
                # before this transaction, then mirrored into this target.
            return comparison["counts"]

    def _record_audit(self, cur, task_id: str, target: str, spec: CanonicalTable, changes: list[dict]):
        """Write row-level audit records inside the caller's transaction."""
        if not changes:
            return
        execute_values(cur, sql.SQL("INSERT INTO {} (task_id,target,table_name,row_key,operation,before_hash,after_hash,before_value,after_value,created_at) VALUES %s").format(relation(self.settings, "import_row_audit")),
                       [(task_id, target, spec.name, change["row_key"], "inserted" if change["before"] is None else "updated", change["before_hash"], change["after_hash"], Json(change["before"], dumps=dumps) if change["before"] else None, Json(change["after"], dumps=dumps), datetime.now(timezone.utc)) for change in changes], page_size=500)

    def record_audit(self, task_id: str, target: str, spec: CanonicalTable, changes: list[dict]):
        """Record a non-PG target commit in the central PG audit ledger."""
        with self.connect() as conn, conn.cursor() as cur:
            self._record_audit(cur, task_id, target, spec, changes)

    def _compare_with_connection(self, cur, spec, rows, mode):
        existing = self._fetch_existing(cur, spec, rows)
        return compare_rows(spec, rows, list(existing.values()), mode)

    def query(self, name: str, code: str | None = None, start: str | None = None, end: str | None = None, limit: int = 100, after: str | None = None) -> dict:
        spec = canonical(name)
        conditions, params = [], []
        if code and "security_code" in columns(spec): conditions.append(sql.SQL("security_code=%s")); params.append(code)
        if start and "event_date" in columns(spec): conditions.append(sql.SQL("event_date >= %s")); params.append(parse_date(start))
        if end and "event_date" in columns(spec): conditions.append(sql.SQL("event_date <= %s")); params.append(parse_date(end))
        offset = int(after or 0)
        if offset < 0: raise ValueError("分页位置无效")
        where = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions) if conditions else sql.SQL("")
        size = max(1, min(limit, 1000)); params.extend([size + 1, offset])
        with self.connect(readonly=True) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql.SQL("SELECT * FROM {}{} ORDER BY {} LIMIT %s OFFSET %s").format(relation(self.settings, name), where, sql.SQL(",").join(sql.Identifier(k) for k in spec.keys)), params)
            rows = [dict(row) for row in cur.fetchall()]
        return {"columns": columns(spec), "rows": rows[:size], "next": str(offset + size) if len(rows) > size else None}

    def receipts(self, task: str) -> list[dict]:
        with self.connect(readonly=True) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql.SQL("SELECT * FROM {} WHERE task_id=%s ORDER BY target").format(relation(self.settings, "import_target_status")), (task,))
            return [dict(row) for row in cur.fetchall()]

    def audit(self, task: str, after: int = 0, limit: int = 100) -> list[dict]:
        with self.connect(readonly=True) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql.SQL("SELECT * FROM {} WHERE task_id=%s AND id>%s ORDER BY id LIMIT %s").format(relation(self.settings, "import_row_audit")), (task, after, max(1, min(limit, 1000))))
            return [dict(row) for row in cur.fetchall()]

    def apply_rollback(self, spec, audits, rollback_task_id, connection=None):
        changes=[]
        with (nullcontext(connection) if connection is not None else self.connect()) as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql.SQL('LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE').format(relation(self.settings,spec.name)))
            live=self._fetch_existing(cur,spec,[a['after_value'] for a in audits])
            for audit in audits:
                after=typed(spec,audit['after_value']); key=tuple(after.get(k) for k in spec.keys)
                current=live.get(key)
                if current is None or row_hash(spec,current)!=audit['after_hash']: raise ValueError('目标行已被后续任务修改，禁止回退')
                before=typed(spec,audit['before_value']) if audit['before_value'] else None
                where=sql.SQL(' AND ').join(sql.SQL('{}=%s').format(sql.Identifier(k)) for k in spec.keys)
                if before is None:
                    cur.execute(sql.SQL('DELETE FROM {} WHERE {}').format(relation(self.settings,spec.name),where),list(key))
                else:
                    names=[f['name'] for f in spec.fields if f['name'] not in spec.keys]
                    cur.execute(sql.SQL('UPDATE {} SET {} WHERE {}').format(relation(self.settings,spec.name),sql.SQL(',').join(sql.SQL('{}=%s').format(sql.Identifier(n)) for n in names),where),[before.get(n) for n in names]+list(key))
                changes.append({'before':typed(spec,current),'after':before,'row_key':audit['row_key'],'before_hash':row_hash(spec,current),'after_hash':row_hash(spec,before) if before else None})
        return changes

    def record_rollback(self, rollback_task_id: str, source_task_id: str, target: str, result: dict):
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("INSERT INTO {} (rollback_task_id,source_task_id,target,status,restored,deleted,conflicts,created_at,finished_at,error) VALUES (%s,%s,%s,%s,%s,%s,%s,now(),now(),%s)").format(relation(self.settings, "import_rollback")),
                        (rollback_task_id, source_task_id, target, result.get("status", "failed"), result.get("restored", 0), result.get("deleted", 0), result.get("conflicts", 0), result.get("error")))

    def control_write(self, table, rows):
        if table not in CONTROL_DDL: raise ValueError('未登记的管理表')
        if not rows: return
        names = [f[0] for f in CONTROL_DDL[table]]
        json_fields = {f[0] for f in CONTROL_DDL[table] if f[1] == 'jsonb'}
        with self.connect() as conn, conn.cursor() as cur:
            values = [[Json(row.get(name),dumps=dumps) if name in json_fields and row.get(name) is not None else row.get(name) for name in names] for row in rows]
            override = sql.SQL(' OVERRIDING SYSTEM VALUE') if table=='import_row_audit' else sql.SQL('')
            keys = CONTROL_KEYS[table]
            action = sql.SQL('DO UPDATE SET ') + sql.SQL(',').join(sql.SQL('{}=EXCLUDED.{}').format(sql.Identifier(n),sql.Identifier(n)) for n in names if n not in keys)
            execute_values(cur,sql.SQL('INSERT INTO {} ({}){} VALUES %s ON CONFLICT ({}) {}').format(relation(self.settings,table),sql.SQL(',').join(sql.Identifier(n) for n in names),override,sql.SQL(',').join(sql.Identifier(n) for n in keys),action),values,page_size=500)


    def refresh_coverage(self,name):
        spec=canonical(name);names=columns(spec)
        asset=sql.SQL("coalesce(asset_type,'')") if 'asset_type' in names else sql.SQL("''::text")
        code=sql.SQL("coalesce(security_code,'')") if 'security_code' in names else sql.SQL("''::text")
        dates=sql.SQL('min(event_date),max(event_date)') if 'event_date' in names else sql.SQL('NULL::date,NULL::date')
        with self.connect() as conn,conn.cursor() as cur:
            cur.execute(sql.SQL('SELECT {},{},{},count(*) FROM {} GROUP BY 1,2').format(asset,code,dates,relation(self.settings,name)))
            records=[{'table_name':name,'asset_type':row[0],'security_code':row[1],'first_event_date':row[2],'last_event_date':row[3],'rows':row[4],'updated_at':datetime.now(timezone.utc)} for row in cur.fetchall()]
            cur.execute(sql.SQL('DELETE FROM {} WHERE table_name=%s').format(relation(self.settings,'dataset_coverage')),(name,))
        if not records:records=[{'table_name':name,'asset_type':'','security_code':'','first_event_date':None,'last_event_date':None,'rows':0,'updated_at':datetime.now(timezone.utc)}]
        self.control_write('dataset_coverage',records)
        return records

CONTROL_KEYS = {
    'meta_table':('table_name',),'meta_column':('table_name','column_name'),
    'meta_source_mapping':('source_name','source_endpoint','source_field','canonical_table','canonical_column'),
    'meta_source_capability':('source_name','endpoint'),'import_task':('task_id',),
    'import_source_archive':('task_id','dataset','request_hash'),'import_target_status':('task_id','target'),
    'import_row_audit':('id',),'import_rollback':('rollback_task_id','target'),
    'dataset_coverage':('table_name','asset_type','security_code')}


def control_spec(name):
    fields=[]
    for column,kind,required in CONTROL_DDL[name]:
        mapped='timestamp' if kind=='timestamp with time zone' else 'integer' if kind=='integer' or kind.startswith('bigint') else 'date' if kind=='date' else 'text'
        fields.append({'name':column,'type':mapped,'nullable':not required,'zh_name':column,'description':column,'unit':''})
    labels={'meta_table':'数据表元信息','meta_column':'字段元信息','meta_source_mapping':'来源字段映射','meta_source_capability':'数据源能力',
            'import_task':'导入任务','import_source_archive':'源数据归档','import_target_status':'目标库导入状态','import_row_audit':'行级变更审计',
            'import_rollback':'任务回退','dataset_coverage':'数据覆盖范围'}
    return CanonicalTable(name,labels[name],labels[name]+'登记、查询与历史追溯','登记键一行',tuple(fields),CONTROL_KEYS[name],domain='management')

