"""Durable local orchestration, deliberately separate from database commits."""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from .util import dumps, now, digest


class State:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "tasks.sqlite3"
        with self.connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS ledger(table_name TEXT NOT NULL,row_key TEXT NOT NULL,value TEXT NOT NULL,PRIMARY KEY(table_name,row_key));
                CREATE TABLE IF NOT EXISTS audit_sequence(id INTEGER PRIMARY KEY);
                CREATE INDEX IF NOT EXISTS ledger_task_idx ON ledger(table_name,json_extract(value,'$.task_id'));
                CREATE INDEX IF NOT EXISTS ledger_audit_page_idx ON ledger(table_name,json_extract(value,'$.task_id'),CAST(json_extract(value,'$.id') AS INTEGER));
                CREATE INDEX IF NOT EXISTS ledger_change_table_idx ON ledger(table_name,json_extract(value,'$.target'),json_extract(value,'$.table_name'),json_extract(value,'$.created_at'));
                CREATE INDEX IF NOT EXISTS ledger_source_task_idx ON ledger(table_name,json_extract(value,'$.source_task_id'));
                CREATE TABLE IF NOT EXISTS jobs(
                    id TEXT PRIMARY KEY, created TEXT NOT NULL, updated TEXT NOT NULL,
                    status TEXT NOT NULL, target TEXT NOT NULL, plan TEXT NOT NULL,
                    preview TEXT, error TEXT, cancel INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS slices(
                    job TEXT NOT NULL, id TEXT NOT NULL, dataset TEXT NOT NULL, request TEXT NOT NULL,
                    status TEXT NOT NULL, archive TEXT, sha TEXT, rows INTEGER, result TEXT,
                    PRIMARY KEY(job,id), FOREIGN KEY(job) REFERENCES jobs(id));
                CREATE TABLE IF NOT EXISTS tasks(
                    id TEXT PRIMARY KEY, job TEXT NOT NULL, operation TEXT NOT NULL,
                    payload TEXT NOT NULL, status TEXT NOT NULL, created TEXT NOT NULL, error TEXT,
                    FOREIGN KEY(job) REFERENCES jobs(id));
                CREATE TABLE IF NOT EXISTS schedules(
                    id TEXT PRIMARY KEY, label TEXT NOT NULL, request TEXT NOT NULL,
                    hour INTEGER NOT NULL, minute INTEGER NOT NULL, enabled INTEGER NOT NULL DEFAULT 0,
                    next_run TEXT, last_job TEXT);
                CREATE TABLE IF NOT EXISTS rate_limit(id INTEGER PRIMARY KEY CHECK(id=1), next REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS request_rate(scope TEXT PRIMARY KEY, next REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS sync_outbox(
                    task TEXT NOT NULL,target TEXT NOT NULL,revision INTEGER NOT NULL DEFAULT 1,
                    acknowledged INTEGER NOT NULL DEFAULT 0,error TEXT,last_attempt TEXT,
                    PRIMARY KEY(task,target));
            """)

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def create_job(self, plan: dict, target: str) -> dict:
        job = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,NULL,NULL,0)",
                         (job, now(), now(), "planned", target, dumps(plan)))
        return self.job(job)

    def job(self, job: str) -> dict:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
            if row is None: raise ValueError("任务不存在")
            result = dict(row)
            for key in ("plan", "preview"):
                result[key] = json.loads(result[key]) if result[key] else None
            result["slices"] = [dict(r) for r in conn.execute("SELECT * FROM slices WHERE job=? ORDER BY id", (job,))]
            for item in result["slices"]:
                item["request"] = json.loads(item["request"])
                item["result"] = json.loads(item["result"]) if item["result"] else None
            result["tasks"] = [dict(r) for r in conn.execute("SELECT id,operation,status,error FROM tasks WHERE job=? ORDER BY created", (job,))]
            result['audit_sync']=[dict(r) for r in conn.execute('SELECT * FROM sync_outbox WHERE task=? ORDER BY target',(job,))]
            result['audit_sync_status']='pending' if any(r['revision']>r['acknowledged'] for r in result['audit_sync']) else 'synced'
            recorded=conn.execute("SELECT value FROM ledger WHERE table_name='publication_result' AND row_key=?",(job,)).fetchone()
            result['data_status']=json.loads(recorded[0])['status'] if recorded else result['status']
            return result

    def jobs(self, limit: int = 100, target: str | None = None, offset: int = 0) -> list[dict]:
        condition=' WHERE target=?' if target else ''
        params=[target,max(1,min(limit,1000)),max(0,offset)] if target else [max(1,min(limit,1000)),max(0,offset)]
        with self.connect() as conn:
            return [dict(r) for r in conn.execute('SELECT id,created,updated,status,error FROM jobs'+condition+' ORDER BY created DESC LIMIT ? OFFSET ?',params)]

    def get_slice(self, job: str, key: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM slices WHERE job=? AND id=?", (job,key)).fetchone()
        return dict(row) if row else None

    def update_job(self, job: str, **values):
        if not set(values) <= {"status", "preview", "error", "cancel"}: raise ValueError("无效任务字段")
        values["updated"] = now()
        if "preview" in values: values["preview"] = dumps(values["preview"])
        with self.connect() as conn:
            conn.execute("UPDATE jobs SET " + ",".join(k + "=?" for k in values) + " WHERE id=?", (*values.values(), job))

    def slice(self, job: str, key: str, name: str, request: dict, **values):
        if not set(values) <= {"status", "archive", "sha", "rows", "result"}: raise ValueError("无效分片字段")
        with self.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO slices(job,id,dataset,request,status) VALUES(?,?,?,?,?)",
                         (job, key, name, dumps(request), "pending"))
            if values:
                if "result" in values: values["result"] = dumps(values["result"])
                conn.execute("UPDATE slices SET " + ",".join(k + "=?" for k in values) + " WHERE job=? AND id=?", (*values.values(), job, key))

    def enqueue(self, job: str, operation: str, payload: dict | None = None) -> str:
        task = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM tasks WHERE job=? AND status IN ('queued','running')", (job,)).fetchone():
                raise ValueError("该任务已有等待或执行中的操作")
            conn.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,NULL)",
                         (task, job, operation, dumps(payload or {}), "queued", now()))
        return task

    def claim(self,allow_ui=True) -> dict | None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            condition="" if allow_ui else " AND json_extract(payload,'$.owner')='cli'"
            row = conn.execute("SELECT * FROM tasks WHERE status='queued'"+condition+" ORDER BY created LIMIT 1").fetchone()
            if row is None: return None
            conn.execute("UPDATE tasks SET status='running' WHERE id=?", (row["id"],))
            result = dict(row)
            result["payload"] = json.loads(result["payload"])
            return result

    def finish(self, task: str, error: str | None = None):
        with self.connect() as conn:
            conn.execute("UPDATE tasks SET status=?,error=? WHERE id=?", ("failed" if error else "completed", error, task))

    def cancel(self, job: str):
        self.update_job(job, cancel=1)
        with self.connect() as conn:
            conn.execute("UPDATE tasks SET status='cancelled' WHERE job=? AND status='queued'", (job,))

    def check_cancel(self, job: str):
        with self.connect() as conn:
            row = conn.execute("SELECT cancel FROM jobs WHERE id=?", (job,)).fetchone()
        if row and row[0]: raise InterruptedError("任务已取消；已提交分片保留")


    def ledger_put(self, table: str, key: str, value: dict):
        with self.connect() as conn:
            conn.execute('INSERT INTO ledger VALUES(?,?,?) ON CONFLICT(table_name,row_key) DO UPDATE SET value=excluded.value', (table,key,dumps(value)))
            if table in {'import_task','import_source_archive','import_target_status','import_row_audit','import_rollback','meta_source_mapping','meta_source_capability'}:
                task=value.get('task_id') or value.get('source_task_id')
                if task:
                    targets=[value['target']] if value.get('target') in {'postgres','clickhouse'} else self._job_targets(conn,task)
                    self._queue_sync(conn,task,targets)
                if table=='import_rollback':self._queue_sync(conn,value['rollback_task_id'],[value['target']])

    @staticmethod
    def _job_targets(conn,task):
        row=conn.execute('SELECT plan FROM jobs WHERE id=?',(task,)).fetchone()
        return json.loads(row[0]).get('request',{}).get('targets',[]) if row else []

    @staticmethod
    def _queue_sync(conn,task,targets):
        for target in set(targets):
            conn.execute('INSERT INTO sync_outbox(task,target) VALUES(?,?) ON CONFLICT(task,target) DO UPDATE SET revision=revision+1',(task,target))

    def request_sync(self,task,target):
        with self.connect() as conn:self._queue_sync(conn,task,[target])

    def record_result(self,task,status,error,record):
        """Persist data outcome and its control-sync obligation atomically."""
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('UPDATE jobs SET status=?,error=?,updated=? WHERE id=?',(status,error,now(),task))
            for table,value in [('publication_result',{'task_id':task,'status':status,'error':error}),('import_task',record)]:
                conn.execute('INSERT INTO ledger VALUES(?,?,?) ON CONFLICT(table_name,row_key) DO UPDATE SET value=excluded.value',(table,task,dumps(value)))
            self._queue_sync(conn,task,self._job_targets(conn,task))

    def sync_records(self,task=None):
        with self.connect() as conn:
            return [dict(r) for r in conn.execute('SELECT * FROM sync_outbox'+(' WHERE task=?' if task else '')+' ORDER BY task,target', (task,) if task else ())]

    def sync_result(self,task,target,revision,error=None):
        with self.connect() as conn:
            if error:
                conn.execute('UPDATE sync_outbox SET error=?,last_attempt=? WHERE task=? AND target=?',(error,now(),task,target))
            else:
                conn.execute('UPDATE sync_outbox SET acknowledged=max(acknowledged,?),error=CASE WHEN revision<=? THEN NULL ELSE error END,last_attempt=? WHERE task=? AND target=?',(revision,revision,now(),task,target))

    def ledger_rows(self, table: str, task: str | None = None):
        query='SELECT value FROM ledger WHERE table_name=?'
        params=[table]
        if task is not None:
            query+=" AND (json_extract(value,'$.task_id')=? OR json_extract(value,'$.source_task_id')=?)"
            params.extend([task,task])
        with self.connect() as conn:
            return [json.loads(row[0]) for row in conn.execute(query+' ORDER BY row_key',params)]

    def stage_commit(self, task: str, target: str, table: str, comparison: dict):
        key = dumps([task,target,table])
        sparse={**comparison,'changes':[{**change,'before':{k:v for k,v in change['before'].items() if v is not None} if change['before'] is not None else None,
                                             'after':{k:v for k,v in change['after'].items() if v is not None} if change['after'] is not None else None} for change in comparison['changes']]}
        sparse.pop('conflicts',None)
        record = {'task':task,'target':target,'table':table,'status':'prepared','comparison':sparse}
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            old=conn.execute("SELECT value FROM ledger WHERE table_name='table_commit' AND row_key=?",(key,)).fetchone()
            if old:
                if json.loads(old[0])['comparison']!=json.loads(dumps(sparse)):raise ValueError('持久化提交意图不可覆盖，请核验原操作')
                return json.loads(old[0])
            conn.execute('INSERT INTO ledger VALUES(?,?,?)',('table_commit',key,dumps(record)))
        return record

    def table_commits(self,task,target=None):
        with self.connect() as conn:
            query="SELECT value FROM ledger WHERE table_name='table_commit' AND json_extract(value,'$.task')=?"
            params=[task]
            if target:query+=" AND json_extract(value,'$.target')=?";params.append(target)
            return [json.loads(r[0]) for r in conn.execute(query,params)]

    def table_commit(self, task: str, target: str, table: str):
        with self.connect() as conn:
            row = conn.execute('SELECT value FROM ledger WHERE table_name=? AND row_key=?',('table_commit',dumps([task,target,table]))).fetchone()
        return json.loads(row[0]) if row else None

    def finish_commit(self, task: str, target: str, table: str, applied_keys=None):
        record=self.table_commit(task,target,table)
        if record is None:raise ValueError('缺少持久化提交记录')
        record['status']='completed'
        if applied_keys is not None:
            record['applied_keys']=sorted(applied_keys)
            record['status']='reconciled'
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('UPDATE ledger SET value=? WHERE table_name=? AND row_key=?',(dumps(record),'table_commit',dumps([task,target,table])))
            audit_id=conn.execute('SELECT coalesce(max(id),0) FROM audit_sequence').fetchone()[0]
            values=[]
            for change in record['comparison']['changes']:
                if applied_keys is not None and change['row_key'] not in applied_keys:continue
                key=dumps([task,target,table,change['row_key']])
                if conn.execute('SELECT 1 FROM ledger WHERE table_name=? AND row_key=?',('import_row_audit',key)).fetchone(): continue
                audit_id=int(digest([task,target,table,change['row_key']])[:15],16)
                value={'id':audit_id,'task_id':task,'target':target,'table_name':table,'row_key':change['row_key'],
                       'operation':'deleted' if change['after'] is None else 'inserted' if change['before'] is None else 'updated',
                       'before_hash':change['before_hash'],'after_hash':change['after_hash'],
                       'before_value':change['before'],'after_value':change['after'],'created_at':now()}
                values.append(('import_row_audit',key,dumps(value)))
            conn.executemany('INSERT INTO ledger VALUES(?,?,?)',values)
            conn.execute('INSERT OR IGNORE INTO audit_sequence VALUES(?)',(audit_id,))
            self._queue_sync(conn,task,[target])

    def audits(self, task: str, target: str | None = None, after: int = 0, limit: int | None = None):
        query="SELECT value FROM ledger WHERE table_name='import_row_audit' AND json_extract(value,'$.task_id')=? AND CAST(json_extract(value,'$.id') AS INTEGER)>?"
        params=[task,after]
        if target:
            query+=" AND json_extract(value,'$.target')=?";params.append(target)
        query+=" ORDER BY CAST(json_extract(value,'$.id') AS INTEGER)"
        if limit is not None:query+=' LIMIT ?';params.append(max(1,min(limit,1000)))
        with self.connect() as conn:rows=[json.loads(row[0]) for row in conn.execute(query,params)]
        for row in rows:row['id']=int(row['id'])
        return rows

    def table_commit_summary(self,task,target,table):
        with self.connect() as conn:
            row=conn.execute("SELECT json_extract(value,'$.status'),json_extract(value,'$.comparison.counts') FROM ledger WHERE table_name='table_commit' AND row_key=?",(dumps([task,target,table]),)).fetchone()
        return {'status':row[0],'counts':json.loads(row[1] or '{}')} if row else None

    def log(self,job,event,**values):
        # Every record is durable before it is streamed to a browser.
        from .security import redact
        values=redact(values)
        with self.connect() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS event_log(id INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT NOT NULL,created TEXT NOT NULL,event TEXT NOT NULL,value TEXT NOT NULL)')
            conn.execute('INSERT INTO event_log(job,created,event,value) VALUES(?,?,?,?)',(job,now(),event,dumps(values)))

    def logs(self,job,after=0,limit=200):
        with self.connect() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS event_log(id INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT NOT NULL,created TEXT NOT NULL,event TEXT NOT NULL,value TEXT NOT NULL)')
            return [{**dict(r),**json.loads(r['value'])} for r in conn.execute('SELECT * FROM event_log WHERE job=? AND id>? ORDER BY id LIMIT ?',(job,after,max(1,min(limit,1000))))]


    def ledger_get(self,table,key):
        with self.connect() as conn:
            row=conn.execute('SELECT value FROM ledger WHERE table_name=? AND row_key=?',(table,key)).fetchone()
        return json.loads(row[0]) if row else None
