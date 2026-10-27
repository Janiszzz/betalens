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

    def execute(self, query: str, data: str | None = None, timeout: int = 180) -> str:
        # Unknown/new SQL forms are conservatively coordinated. A blacklist
        # missed DELETE and administrative forms such as ATTACH/DETACH.
        first=query.lstrip().split(None,1)[0].upper() if query.strip() else ''
        mutation=first not in {'SELECT','SHOW','DESCRIBE','DESC','EXISTS','EXPLAIN'}
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
        response = requests.post(self.settings.ch_url, params=params, data=payload.encode(),
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

    def _find_existing(self, spec: CanonicalTable, rows: list[dict]) -> dict[tuple, dict]:
        if not rows: return {}
        result = {}
        # Large snapshots such as stock_basic can contain tens of thousands
        # of keys.  Keep each HTTP query bounded so ClickHouse does not reject
        # the request body before SQL parsing.
        keys = [tuple(row.get(k) for k in spec.keys) for row in rows]
        for offset in range(0, len(keys), 500):
            conditions = []
            for key in keys[offset:offset + 500]:
                conditions.append("(" + " AND ".join(f"`{field}`={self._literal(value)}" for field, value in zip(spec.keys, key)) + ")")
            query = f"SELECT * FROM {self.table(spec.name)} FINAL WHERE {' OR '.join(conditions)} FORMAT JSONEachRow"
            for row in self._json_rows(self.execute(query)):
                row=typed(spec,row)
                result[tuple(row.get(k) for k in spec.keys)] = row
        return result

    @staticmethod
    def _literal(value):
        if value is None: return "NULL"
        if isinstance(value, (date, datetime)): value = value.isoformat()
        text = str(value).replace("\\", "\\\\").replace("'", "\\'")
        return "'" + text + "'"

    def compare(self, spec, rows, mode="fill_missing"):
        existing = self._find_existing(spec, rows)
        return compare_rows(spec, rows, list(existing.values()), mode)

    def commit_rows(self, task_id: str, target: str, spec: CanonicalTable, rows: list[dict], mode: str, signature: str) -> dict:
        with self.write_window('commit:'+task_id):
            return self._commit_rows(task_id,target,spec,rows,mode,signature)

    def _commit_rows(self,task_id,target,spec,rows,mode,signature):
        if isinstance(signature,dict):
            from .commit_protocol import manifest_rows,remaining_changes
            comparison=signature
            live=self._find_existing(spec,manifest_rows(comparison))
            changes=remaining_changes(spec,comparison,list(live.values()))
        else:
            comparison = self.compare(spec, rows, mode)
            if comparison["signature"] != signature: raise ValueError("预览后的 ClickHouse 数据已经变化，请重新预览")
            changes = comparison["changes"]
        if not changes: return comparison["counts"]
        from collections import defaultdict
        partitions = defaultdict(list)
        for change in changes:
            row = {name: self._parse(change["after"].get(name), next(f["type"] for f in spec.fields if f["name"] == name)) for name in columns(spec)}
            partition = str(row.get('event_date', 'all'))[:7]
            partitions[partition].append(dumps(row))
        for lines in partitions.values():
            for offset in range(0, len(lines), 5000):
                self.execute(f"INSERT INTO {self.table(spec.name)} FORMAT JSONEachRow", "\n".join(lines[offset:offset + 5000]) + "\n")
        return comparison["counts"]

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

    def apply_rollback(self, spec, audits, rollback_task_id):
        with self.write_window('rollback:'+rollback_task_id):
            return self._apply_rollback(spec,audits,rollback_task_id)

    def table_digest(self,spec,name=None):
        fields=','.join('`'+f['name']+'`' for f in spec.fields)
        rows=self._json_rows(self.execute(f'SELECT count() AS rows,groupBitXor(cityHash64(tuple({fields}))) AS content FROM {self.table(name or spec.name)} FINAL FORMAT JSONEachRow'))
        return {k:str(v) for k,v in rows[0].items()}

    def cleanup_rollback(self,rollback_task_id,state):
        for record in state.ledger_rows('ch_operation',rollback_task_id):
            if record.get('status')=='cleaned':continue
            with self.write_window('cleanup:'+rollback_task_id):
                spec=canonical(record['table_name'])
                if self.table_digest(spec)!=record.get('expected_inverse'):
                    raise ValueError('目标表已发生后续变化，保留回退备份供核验')
                table=self.table(record['temporary_table'])
                if self.execute(f'EXISTS TABLE {table}')=='1':
                    if self.table_digest(spec,record['temporary_table'])!=record.get('original_digest'):
                        raise ValueError('回退备份与检查点不符，保留备份等待核验')
                    self.execute(f'DROP TABLE {table}')
                state.ledger_put('ch_operation',dumps([rollback_task_id,spec.name]),{**record,'status':'cleaned'})

    def _apply_rollback(self, spec, audits, rollback_task_id):
        """Rebuild affected partitions in a staging table and atomically exchange.

        Unaffected partitions are copied server-side. Source timestamps and the
        prior import_task_id are restored exactly; rollback identity lives in
        the immutable audit records.
        """
        import uuid
        changes=[]; live=self._find_existing(spec,[a['after_value'] for a in audits])
        for audit in audits:
            after=audit['after_value']; key=tuple(typed(spec,after).get(k) for k in spec.keys)
            current=live.get(key)
            if current is None or row_hash(spec,current)!=audit['after_hash']: raise ValueError('目标行已被后续任务修改，禁止回退')
            before=audit['before_value']
            changes.append({'before':typed(spec,current),'after':typed(spec,before) if before else None,'row_key':audit['row_key'],
                            'before_hash':row_hash(spec,current),'after_hash':row_hash(spec,before) if before else None})
        temporary=('rollback_'+rollback_task_id[:16]+'_'+spec.name)[:63]
        temp_table=self.table(temporary)
        state=getattr(self,'state',None)
        key=dumps([rollback_task_id,spec.name])
        old=state.ledger_get('ch_operation',key) if state else None
        checkpoint=old or {'task_id':rollback_task_id,'target':'clickhouse','table_name':spec.name,'temporary_table':temporary,'status':'prepared','original_digest':self.table_digest(spec)}
        def record(status):
            checkpoint['status']=status
            if state:state.ledger_put('ch_operation',key,checkpoint)
        if self.execute(f'EXISTS TABLE {temp_table}')=='1':
            if old and self.table_digest(spec)==old.get('original_digest'):
                if old.get('expected_inverse') and self.table_digest(spec,temporary)==old['expected_inverse']:
                    self.execute(f'EXCHANGE TABLES {self.table(spec.name)} AND {temp_table}')
                    record('exchanged_backup_retained');return changes
                self.execute(f'DROP TABLE {temp_table}')  # The complete original is still intact.
            else:
                from .commit_protocol import NeedsReconciliation
                raise NeedsReconciliation('回退检查点不符，保留原表及备份等待核验：'+temporary)
        record('prepared')
        self.execute(self.ddl(spec).replace(self.table(spec.name),temp_table,1))
        try:
            if any(f['name']=='event_date' and not f.get('nullable',True) for f in spec.fields):
                partitions=sorted({str(a['after_value']['event_date'])[:7].replace('-','') for a in audits})
                condition='toYYYYMM(event_date) IN ('+','.join(partitions)+')'
            else: condition='1'
            fields=','.join('`'+f['name']+'`' for f in spec.fields)
            unaffected=self._json_rows(self.execute(f'SELECT DISTINCT _partition_id AS partition_id FROM {self.table(spec.name)} FINAL WHERE NOT ({condition}) FORMAT JSONEachRow'))
            for offset in range(0,len(unaffected),100):
                selected=','.join(self._literal(p['partition_id']) for p in unaffected[offset:offset+100])
                self.execute(f'INSERT INTO {temp_table} ({fields}) SELECT {fields} FROM {self.table(spec.name)} FINAL WHERE NOT ({condition}) AND _partition_id IN ({selected})')
            affected=self._json_rows(self.execute(f'SELECT * FROM {self.table(spec.name)} FINAL WHERE {condition} FORMAT JSONEachRow'))
            result={row_key(spec,row):row for row in affected}
            for change in changes:
                if change['after'] is None: result.pop(change['row_key'],None)
                else: result[change['row_key']]=change['after']
            all_rows=list(result.values())
            for offset in range(0,len(all_rows),1000):
                lines=[dumps({f['name']:self._parse(row.get(f['name']),f['type']) for f in spec.fields}) for row in all_rows[offset:offset+1000]]
                self.execute(f'INSERT INTO {temp_table} FORMAT JSONEachRow','\n'.join(lines)+'\n')
            fresh=self._find_existing(spec,[a['after_value'] for a in audits])
            if any(row_hash(spec,fresh.get(tuple(typed(spec,a['after_value']).get(k) for k in spec.keys),{}))!=a['after_hash'] for a in audits):
                raise ValueError('回退发布前目标行发生变化，已放弃发布')
            record('staged')
            checkpoint['expected_inverse']=self.table_digest(spec,temporary)
            record('staged')
            self.execute(f'EXCHANGE TABLES {self.table(spec.name)} AND {temp_table}')
            if self.table_digest(spec)!=checkpoint['expected_inverse']:raise ValueError('整表回退发布后内容核验失败')
            record('exchanged_backup_retained')
        except BaseException:
            record('needs_reconciliation')
            raise
        # The old table is retained until the durable inverse audit is verified.
        return changes

    def drop_legacy(self) -> dict:
        raw = self.execute(f"SELECT name FROM system.tables WHERE database='{self.settings.ch_database}' FORMAT JSONEachRow")
        names = {json.loads(line)["name"] for line in raw.splitlines() if line.strip()}
        keep = set(load_canonical_catalog())
        dropped = []
        for name in sorted(names - keep):
            self.execute(f"DROP TABLE IF EXISTS `{identifier(self.settings.ch_database)}`.`{identifier(name)}`")
            dropped.append(name)
        return {"dropped": dropped}

    def control_write(self, name, rows):
        if name not in CONTROL_DDL: raise ValueError("未登记的管理表")
        spec=control_spec(name)
        json_fields={f[0] for f in CONTROL_DDL[name] if f[1]=="jsonb"}
        for offset in range(0,len(rows),500):
            lines=[]
            for row in rows[offset:offset+500]:
                lines.append(dumps({f["name"]:dumps(row.get(f["name"])) if f["name"] in json_fields and row.get(f["name"]) is not None else self._parse(row.get(f["name"]),f["type"]) for f in spec.fields}))
            self.execute(f"INSERT INTO {self.table(name)} FORMAT JSONEachRow", "\n".join(lines)+"\n")

    def refresh_coverage(self,name):
        spec=canonical(name);names=columns(spec)
        asset="coalesce(asset_type,'')" if 'asset_type' in names else "''"
        code="coalesce(security_code,'')" if 'security_code' in names else "''"
        dates="minOrNull(event_date) AS first_event_date,maxOrNull(event_date) AS last_event_date" if 'event_date' in names else "NULL AS first_event_date,NULL AS last_event_date"
        records=self._json_rows(self.execute(f'SELECT {asset} AS asset_type,{code} AS security_code,{dates},count() AS rows FROM {self.table(name)} FINAL GROUP BY asset_type,security_code FORMAT JSONEachRow'))
        if not records:records=[{'asset_type':'','security_code':'','first_event_date':None,'last_event_date':None,'rows':0}]
        for row in records:row.update(table_name=name,updated_at=datetime.now(timezone.utc))
        self.execute(f"ALTER TABLE {self.table('dataset_coverage')} DELETE WHERE table_name={self._literal(name)} SETTINGS mutations_sync=1")
        self.control_write('dataset_coverage',records)
        return records
