"""ClickHouse target with the same canonical logical table contracts as PG."""
from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, timezone
from decimal import Decimal

import requests

from .canonical import CanonicalTable, canonical, load_canonical_catalog
from .config import Settings
from .comparison import compare as compare_rows, typed
from .storage import columns, row_hash, row_key, CONTROL_DDL, control_spec
from .util import digest, dumps, identifier, parse_date


class ClickHouse:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Write policy is deployment configuration, never a task checkbox.
        # Both explicit local modes share the same lock and recovery journal.
        self.write_coordinator = None
        mode=getattr(settings,'ch_writer_mode','disabled')
        if mode in {'local_dedicated','local_admin'}:
            from .ch_writer import LocalDedicatedWriter,LocalAdminWriter
            self.write_coordinator=(LocalAdminWriter if mode=='local_admin' else LocalDedicatedWriter)(settings)
        self._write_window = ContextVar('clickhouse_write_window_'+str(id(self)),default=False)

    @contextmanager
    def write_window(self,operation):
        if self._write_window.get():
            yield;return
        if self.write_coordinator is None:
            from .commit_protocol import NeedsReconciliation
            raise NeedsReconciliation('ClickHouse 未接入受控独占写入口；需要受控维护窗口，已阻止写入或整表回退')
        with self.write_coordinator.exclusive(self,operation):
            token=self._write_window.set(operation)
            try:yield
            finally:self._write_window.reset(token)

    def write_capability(self):
        result={'target':'clickhouse','mode':getattr(self.settings,'ch_writer_mode','disabled'),'available':False,'status':'requires_controlled_writer',
                'reason':'需部署独占写入口并限制旁路写入，当前允许查询和审计预览'}
        if self.write_coordinator is not None:
            try:
                checked=self.write_coordinator.verify(self)
                result.update(available=True,status='controlled_local_admin' if self.settings.ch_writer_mode=='local_admin' else 'controlled_local_writer',reason=None,**checked)
            except Exception as exc:
                from .security import safe_error
                result.update(status='writer_verification_failed',reason=safe_error(exc,self.settings))
        return result

    def execute(self, query: str, data: str | bytes | None = None, timeout: int = 180, *, external_tables=None) -> str:
        # Unknown/new SQL forms are conservatively coordinated. A blacklist
        # missed DELETE and administrative forms such as ATTACH/DETACH.
        first=query.lstrip().split(None,1)[0].upper() if query.strip() else ''
        mutation=first not in {'SELECT','SHOW','DESCRIBE','DESC','EXISTS','EXPLAIN'}
        if external_tables and (mutation or data is not None):
            raise ValueError('外部临时键表仅用于只读查询')
        if mutation and self.write_coordinator is not None and not self._write_window.get():
            with self.write_window('sql:'+digest(query)):
                return self.execute(query,data,timeout)
        # ClickHouse treats the HTTP body as the SQL statement unless a query
        # parameter is provided.  Inserts therefore send SQL in ``query`` and
        # JSONEachRow in the body; sending the rows as the body alone makes the
        # server parse the first JSON object as SQL.
        params = {"query": query, "wait_end_of_query": "1", "date_time_input_format": "best_effort",
                  "input_format_json_read_numbers_as_strings": "1","async_insert":"0"}
        query_id=None
        if mutation and self.write_coordinator is not None:
            query_id=self.write_coordinator.before_request(query,self._write_window.get())
            params['query_id']=query_id
        payload = data if data is not None else query
        if data is None: params.pop("query")
        body={'data':payload if isinstance(payload,bytes) else payload.encode()}
        if external_tables:
            params['query']=query
            body={'files':{}}
            for name,table in external_tables.items():
                identifier(name)
                params[name+'_format']=table['format'];params[name+'_structure']=table['structure']
                body['files'][name]=(name,table['data'],'application/octet-stream')
        from .connections import http_session
        from .resource_budget import BUDGET
        params.update(max_threads=BUDGET.ch_threads)
        if not mutation:params['max_memory_usage']=BUDGET.ch_read_memory_bytes
        response = http_session().post(self.settings.ch_url, params=params, **body,
                                 auth=(self.settings.ch_user, self.settings.ch_password), timeout=timeout)
        if response.status_code != 200:
            from .security import safe_error
            message = safe_error(response.text[:500],self.settings)
            raise RuntimeError(f"ClickHouse HTTP {response.status_code}: {message}")
        if query_id:self.write_coordinator.finished_request(query_id)
        return response.text.strip()

    def table(self, name: str) -> str:
        return f"`{identifier(self.settings.ch_database)}`.`{identifier(name)}`"

    def _type(self, field: dict, required: bool = False) -> str:
        base = {"text": "String", "date": "Date32", "timestamp": "DateTime64(6, 'UTC')",
                "numeric": "Decimal(38, 12)", "integer": "Int64"}[field["type"]]
        if field["type"]=="numeric":base=f"Decimal({field.get('precision',38)}, {field.get('scale',12)})"
        return base if required or not field.get("nullable", True) else f"Nullable({base})"

    def ddl(self, spec: CanonicalTable) -> str:
        definitions = [f"`{f['name']}` {self._type(f, f['name'] in spec.keys)}" for f in spec.fields]
        keys = ",".join(f"`{key}`" for key in spec.keys)
        partition = "toYYYYMM(`event_date`)" if any(f["name"]=="event_date" and not f.get("nullable",True) for f in spec.fields) else "tuple()"
        return f"CREATE TABLE IF NOT EXISTS {self.table(spec.name)} ({','.join(definitions)}) ENGINE=ReplacingMergeTree PARTITION BY {partition} ORDER BY ({keys})"

    def install(self, specs=None) -> dict:
        self.execute(f"CREATE DATABASE IF NOT EXISTS `{identifier(self.settings.ch_database)}`")
        for spec in (specs if specs is not None else load_canonical_catalog()).values():
            existing=self._json_rows(self.execute(f"SELECT name FROM system.columns WHERE database={self._literal(self.settings.ch_database)} AND table={self._literal(spec.name)} FORMAT JSONEachRow"))
            names={r['name'] for r in existing}
            required=[f['name'] for f in spec.fields if f['name'] not in names and (f['name'] in spec.keys or not f.get('nullable',True))]
            if names and required and int(self.execute(f'SELECT count() FROM {self.table(spec.name)} FINAL')):
                raise ValueError('已有数据的 ClickHouse 表新增必填列须先执行可审计回填：'+','.join(required))
            self.execute(self.ddl(spec))
            actions=[f"ADD COLUMN IF NOT EXISTS `{field['name']}` {self._type(field,field['name'] in spec.keys)}" for field in spec.fields]
            self.execute(f"ALTER TABLE {self.table(spec.name)} "+','.join(actions))
        for name in CONTROL_DDL:
            spec = control_spec(name)
            definitions = [f"`{f['name']}` {self._type(f, f['name'] in spec.keys)}" for f in spec.fields]
            keys = ','.join(f'`{key}`' for key in spec.keys)
            self.execute(f"CREATE TABLE IF NOT EXISTS {self.table(name)} ({','.join(definitions)}) ENGINE=ReplacingMergeTree ORDER BY ({keys})")
            actions=[f"ADD COLUMN IF NOT EXISTS `{f['name']}` {self._type(f,f['name'] in spec.keys)}" for f in spec.fields]
            self.execute(f'ALTER TABLE {self.table(name)} '+','.join(actions))
        return self.status()

    def status(self) -> dict:
        raw = self.execute(f"SELECT name FROM system.tables WHERE database='{self.settings.ch_database}' FORMAT JSONEachRow")
        names = {json.loads(line)["name"] for line in raw.splitlines() if line.strip()}
        specs={**load_canonical_catalog(),**{name:control_spec(name) for name in CONTROL_DDL}}
        missing=[name for name in specs if name not in names];missing_columns=[];mismatches=[]
        present=self._json_rows(self.execute(f"SELECT table,name,type FROM system.columns WHERE database='{self.settings.ch_database}' FORMAT JSONEachRow"))
        actual={(row['table'],row['name']):row['type'] for row in present}
        for name,spec in specs.items():
            if name in missing:continue
            for field in spec.fields:
                found=actual.get((name,field['name']));expected=self._type(field,field['name'] in spec.keys)
                if found is None:missing_columns.append(f"{name}.{field['name']}")
                elif found!=expected:
                    mismatches.append(f"{name}.{field['name']}: 期望 {expected}，实际 {found}")
        return {'ready':not(missing or missing_columns or mismatches),'missing':missing,'missing_columns':missing_columns,'mismatches':mismatches,'database':self.settings.ch_database}

    def _parse(self, value, field_type):
        if value is None: return None
        if field_type == "date": return parse_date(value).isoformat()
        if field_type == "timestamp":
            if isinstance(value, datetime): return value.isoformat()
            return str(value)
        if field_type == "numeric": return format(Decimal(str(value)), "f")
        return value

    def _json_rows(self, text: str) -> list[dict]:
        return [json.loads(line, parse_float=Decimal) for line in text.splitlines() if line.strip()]

    def _find_existing(self, spec: CanonicalTable, rows: list[dict], fields=None) -> dict[tuple, dict]:
        if not rows: return {}
        from .clickhouse_keys import find_existing
        return find_existing(self,spec,rows,fields)

    @staticmethod
    def _literal(value):
        if value is None: return "NULL"
        if isinstance(value, (date, datetime)): value = value.isoformat()
        text = str(value).replace("\\", "\\\\").replace("'", "\\'")
        return "'" + text + "'"

    def compare(self, spec, rows, mode="fill_missing"):
        existing = self._find_existing(spec, rows)
        return compare_rows(spec, rows, list(existing.values()), mode)


    def query(self, name: str, code: str | None = None, start: str | None = None, end: str | None = None, limit: int = 100, after: str | None = None) -> dict:
        spec = canonical(name); conditions = []
        offset = int(after or 0)
        if offset < 0: raise ValueError("分页位置无效")
        if code and "security_code" in columns(spec): conditions.append(f"`security_code`={self._literal(code)}")
        if start and "event_date" in columns(spec): conditions.append(f"`event_date` >= {self._literal(parse_date(start))}")
        if end and "event_date" in columns(spec): conditions.append(f"`event_date` <= {self._literal(parse_date(end))}")
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"SELECT * FROM {self.table(name)} FINAL{where} ORDER BY ({','.join('`'+k+'`' for k in spec.keys)}) LIMIT {max(1,min(limit,1000))+1} OFFSET {offset} FORMAT JSONEachRow"
        rows = [typed(spec, row) for row in self._json_rows(self.execute(query))]; size = max(1, min(limit,1000))
        return {"columns": columns(spec), "rows": rows[:size], "next": str(offset + size) if len(rows) > size else None}


    def table_digest(self,spec,name=None,partitions=None):
        fields=','.join('`'+f['name']+'`' for f in spec.fields)
        where=' WHERE _partition_id IN ('+','.join(self._literal(p) for p in sorted(partitions))+')' if partitions else ''
        rows=self._json_rows(self.execute(f'SELECT count() AS rows,groupBitXor(cityHash64(tuple({fields}))) AS content FROM {self.table(name or spec.name)} FINAL{where} FORMAT JSONEachRow'))
        return {k:str(v) for k,v in rows[0].items()}


    def control_write(self, name, rows):
        if name not in CONTROL_DDL: raise ValueError("未登记的管理表")
        spec=control_spec(name)
        json_fields={f[0] for f in CONTROL_DDL[name] if f[1]=="jsonb"}
        for offset in range(0,len(rows),500):
            lines=[]
            for row in rows[offset:offset+500]:
                lines.append(dumps({f["name"]:dumps(row.get(f["name"])) if f["name"] in json_fields and row.get(f["name"]) is not None else self._parse(row.get(f["name"]),f["type"]) for f in spec.fields}))
            self.execute(f"INSERT INTO {self.table(name)} FORMAT JSONEachRow", "\n".join(lines)+"\n")

    def refresh_coverage(self,name,codes=None):
        spec=canonical(name);names=columns(spec)
        asset="coalesce(asset_type,'')" if 'asset_type' in names else "''"
        code="coalesce(security_code,'')" if 'security_code' in names else "''"
        dates="minOrNull(event_date) AS first_event_date,maxOrNull(event_date) AS last_event_date" if 'event_date' in names else "NULL AS first_event_date,NULL AS last_event_date"
        scoped=codes is not None and 'security_code' in names
        if scoped and not codes:return []
        condition='security_code IN ('+','.join(self._literal(c) for c in sorted(codes))+')' if scoped else ''
        where=' WHERE '+condition if scoped else ''
        records=self._json_rows(self.execute(f'SELECT {asset} AS asset_type,{code} AS security_code,{dates},count() AS rows FROM {self.table(name)} FINAL{where} GROUP BY asset_type,security_code FORMAT JSONEachRow'))
        if not records and not scoped:records=[{'asset_type':'','security_code':'','first_event_date':None,'last_event_date':None,'rows':0}]
        for row in records:row.update(table_name=name,updated_at=datetime.now(timezone.utc))
        self.execute(f"ALTER TABLE {self.table('dataset_coverage')} DELETE WHERE table_name={self._literal(name)}"+(' AND '+condition if scoped else '')+" SETTINGS mutations_sync=1")
        if records:self.control_write('dataset_coverage',records)
        return records
