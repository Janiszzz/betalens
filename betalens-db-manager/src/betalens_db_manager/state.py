"""Durable local orchestration, deliberately separate from database commits."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
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
                CREATE TABLE IF NOT EXISTS event_log(id INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT NOT NULL,created TEXT NOT NULL,event TEXT NOT NULL,value TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS event_log_job_id ON event_log(job,id);
                CREATE INDEX IF NOT EXISTS tasks_status_created ON tasks(status,created);
                CREATE INDEX IF NOT EXISTS tasks_job_status ON tasks(job,status);
                CREATE INDEX IF NOT EXISTS slices_job_status ON slices(job,status);
                CREATE TABLE IF NOT EXISTS validation_watermarks(table_name TEXT,target TEXT,version INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(table_name,target));
                CREATE TABLE IF NOT EXISTS plan_snapshots(hash TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS request_payloads(hash TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS job_requests(job TEXT,position INTEGER,request_hash TEXT NOT NULL,PRIMARY KEY(job,position));
                CREATE INDEX IF NOT EXISTS job_requests_hash ON job_requests(job,request_hash);
                CREATE TABLE IF NOT EXISTS pipeline_review_decisions(
                    id TEXT PRIMARY KEY,job TEXT NOT NULL,bucket TEXT NOT NULL,action TEXT NOT NULL,
                    status TEXT NOT NULL,note TEXT NOT NULL,version TEXT NOT NULL,payload TEXT NOT NULL,
                    operation_job TEXT,result TEXT,error TEXT,created TEXT NOT NULL,updated TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS pipeline_review_task ON pipeline_review_decisions(job,bucket,created);
                CREATE TABLE IF NOT EXISTS sync_outbox(
                    task TEXT NOT NULL,target TEXT NOT NULL,revision INTEGER NOT NULL DEFAULT 1,
                    acknowledged INTEGER NOT NULL DEFAULT 0,error TEXT,last_attempt TEXT,
                    PRIMARY KEY(task,target));
            """)

        self.retire_schedules()

    def retire_schedules(self):
        """One-time retirement; preserve old schedules and already running imports."""
        with self.connect() as conn:
            if conn.execute("SELECT 1 FROM ledger WHERE table_name='migration' AND row_key='retire_schedules_v1'").fetchone():return
            conn.execute('BEGIN IMMEDIATE')
            marker=conn.execute("SELECT 1 FROM ledger WHERE table_name='migration' AND row_key='retire_schedules_v1'").fetchone()
            if marker:return
            conn.execute('UPDATE schedules SET enabled=0,next_run=NULL')
            queued=conn.execute("SELECT id,job FROM tasks WHERE status='queued' AND json_extract(payload,'$.schedule') IS NOT NULL").fetchall()
            for task in queued:
                conn.execute("UPDATE tasks SET status='cancelled',error='自动更新功能已移除' WHERE id=?",(task['id'],))
                conn.execute("UPDATE jobs SET status='cancelled',error='自动更新功能已移除',updated=? WHERE id=? AND status NOT IN ('completed','rolled_back') AND NOT EXISTS(SELECT 1 FROM tasks WHERE job=jobs.id AND status='running')",(now(),task['job']))
            conn.execute("INSERT INTO ledger VALUES('migration','retire_schedules_v1',?)",(dumps({'finished_at':now(),'cancelled_queued_operations':len(queued)}),))

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=getattr(self,'busy_timeout',30))
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            with conn:yield conn
        finally:conn.close()

    def create_job(self, plan: dict, target: str) -> dict:
        job = uuid.uuid4().hex
        packed=self._pack_plan(plan)
        with self.connect() as conn:
            serialized=self._store_plan(conn,job,packed)
            conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,NULL,NULL,0)",
                         (job, now(), now(), "planned", target, serialized))
        return self.job(job)

    @staticmethod
    def _pack_plan(plan):
        if plan.get('pipeline_version')!=2:return dumps(plan),None,[]
        local={'request','requests','pipeline_parent','pipeline_bucket','request_estimate'}
        snapshot={k:v for k,v in plan.items() if k not in local}
        identity=digest(snapshot)
        summary={k:v for k,v in plan.items() if k in local-{'requests'}}
        summary.update(snapshot_ref=identity,pipeline_version=2,requests_count=len(plan.get('requests',[])),requests_present='requests' in plan)
        if 'revision' in plan:summary['revision']=plan['revision']
        requests=[(digest(r),dumps(r)) for r in plan.get('requests',[])]
        return dumps(summary),(identity,dumps(snapshot)),requests

    @staticmethod
    def _store_plan(conn,job,packed):
        summary,snapshot,requests=packed
        if snapshot is not None:
            conn.execute('INSERT OR IGNORE INTO plan_snapshots VALUES(?,?)',snapshot)
            conn.executemany('INSERT OR IGNORE INTO request_payloads VALUES(?,?)',list(dict(requests).items()))
            conn.executemany('INSERT INTO job_requests VALUES(?,?,?)',[(job,i,r[0]) for i,r in enumerate(requests)])
        return summary

    def _unpack_plan(self,conn,plan,requests=True):
        if 'snapshot_ref' not in plan:return plan
        saved=conn.execute('SELECT value FROM plan_snapshots WHERE hash=?',(plan['snapshot_ref'],)).fetchone()
        if saved is None:raise ValueError('冻结合同快照缺失')
        snapshot=json.loads(saved[0])
        if digest(snapshot)!=plan['snapshot_ref']:raise ValueError('冻结合同快照校验失败')
        result={**snapshot,**{k:v for k,v in plan.items() if k not in {'snapshot_ref','requests_count','requests_present'}}}
        if requests and plan.get('requests_present',True) and 'requests' not in result:
            rows=conn.execute('SELECT r.request_hash,p.value FROM job_requests r JOIN request_payloads p ON p.hash=r.request_hash WHERE job=? ORDER BY position',(plan['_job_id'],)).fetchall()
            result['requests']=[]
            for row in rows:
                value=json.loads(row['value'])
                if digest(value)!=row['request_hash']:raise ValueError('请求清单校验失败')
                result['requests'].append(value)
            if len(rows)!=plan['requests_count']:raise ValueError('请求清单不完整')
        result.pop('_job_id',None)
        return result

    def plan(self,job,requests=True):
        with self.connect() as conn:
            row=conn.execute('SELECT plan FROM jobs WHERE id=?',(job,)).fetchone()
            if row is None:raise ValueError('任务不存在')
            plan=json.loads(row[0])
            return self._unpack_plan(conn,{**plan,'_job_id':job},requests) if 'snapshot_ref' in plan else plan

    def job_header(self,job):
        with self.connect() as conn:
            row=conn.execute('SELECT id,status,target,cancel,created,updated,error,json_extract(plan,\'$.request\') AS request,json_extract(plan,\'$.pipeline_version\') AS pipeline_version FROM jobs WHERE id=?',(job,)).fetchone()
        if row is None:raise ValueError('任务不存在')
        return {**dict(row),'request':json.loads(row['request'] or '{}')}

    def replace_plan(self,job,plan):
        packed=self._pack_plan(plan)
        with self.connect() as conn:
            conn.execute('DELETE FROM job_requests WHERE job=?',(job,))
            conn.execute('UPDATE jobs SET plan=? WHERE id=?',(self._store_plan(conn,job,packed),job))

    def job(self, job: str) -> dict:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
            if row is None: raise ValueError("任务不存在")
            result = dict(row)
            for key in ("plan", "preview"):
                result[key] = json.loads(result[key]) if result[key] else None
            if 'snapshot_ref' in result['plan']:
                result['plan']=self._unpack_plan(conn,{**result['plan'],'_job_id':job})
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
        condition=" WHERE json_extract(plan,'$.pipeline_parent') IS NULL"+(' AND target=?' if target else '')
        params=[target,max(1,min(limit,1000)),max(0,offset)] if target else [max(1,min(limit,1000)),max(0,offset)]
        with self.connect() as conn:
            return [dict(r) for r in conn.execute('SELECT id,created,updated,status,error FROM jobs'+condition+' ORDER BY created DESC LIMIT ? OFFSET ?',params)]

    def get_slice(self, job: str, key: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM slices WHERE job=? AND id=?", (job,key)).fetchone()
        return dict(row) if row else None

    def slices(self,job,identifiers=None,dependencies=False):
        query='SELECT * FROM slices WHERE job=?';params=[job]
        if identifiers is not None:
            if not identifiers:return []
            query+=' AND id IN ('+','.join('?' for _ in identifiers)+')';params.extend(identifiers)
        if dependencies:query+=" AND (dataset IN ('stock_basic','fund_basic','index_basic','trade_cal') OR json_extract(request,'$.dependency_only')=1)"
        with self.connect() as conn:rows=conn.execute(query,params).fetchall()
        return [{**dict(r),'request':json.loads(r['request']),'result':json.loads(r['result']) if r['result'] else None} for r in rows]

    def copy_slices(self,source_job,target_job,identifiers):
        """Copy validated archive references in bounded transactions, retaining wire bytes."""
        identifiers=list(dict.fromkeys(identifiers))
        for offset in range(0,len(identifiers),128):
            group=identifiers[offset:offset+128];marks=','.join('?' for _ in group)
            with self.connect() as conn:
                conn.execute('INSERT OR REPLACE INTO slices(job,id,dataset,request,status,archive,sha,rows,result) SELECT ?,id,dataset,request,status,archive,sha,rows,result FROM slices WHERE job=? AND id IN ('+marks+')',(target_job,source_job,*group))

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
        self.ensure_execution(job)
        from .execution_policy import LegacyPipelineError
        if operation not in {'pipeline_run','pipeline_commit','pipeline_validate','pipeline_review','pipeline_rollback','pipeline_sync'}:
            raise LegacyPipelineError('旧执行操作已移除，请使用新版启动、提交、校验或回退入口')
        task = uuid.uuid4().hex
        with self.connect() as conn:
            payload=dict(payload or {})
            planned=conn.execute("SELECT json_extract(plan,'$.pipeline_version'),json_extract(plan,'$.request.run_in_background') FROM jobs WHERE id=?",(job,)).fetchone()
            if planned and planned[0]==2:payload.setdefault('background',True if planned[1] is None else bool(planned[1]))
            serialized=dumps(payload)
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM tasks WHERE job=? AND status IN ('queued','running')", (job,)).fetchone():
                raise ValueError("该任务已有等待或执行中的操作")
            conn.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,NULL)",
                         (task, job, operation, serialized, "queued", now()))
            if operation=='pipeline_run':conn.execute("UPDATE jobs SET status='queued',cancel=0,error=NULL,updated=? WHERE id=? AND status NOT IN ('completed','rolled_back','partially_rolled_back')",(now(),job))
            elif operation=='pipeline_review':conn.execute("UPDATE jobs SET status='queued',cancel=0,error=NULL,updated=? WHERE id=?",(now(),job))
            elif operation=='pipeline_rollback':conn.execute('UPDATE jobs SET cancel=0,error=NULL,updated=? WHERE id=?',(now(),job))
        return task

    def ensure_execution(self,job):
        from .execution_policy import LegacyTaskError,ensure_available
        if self.job_header(job)['pipeline_version']!=2:
            raise LegacyTaskError('旧任务仅保留查看和导出，不能继续执行、提交、回退或重试')
        ensure_available(self.root)

    def claim(self,allow_ui=True) -> dict | None:
        from .execution_policy import maintenance,operation_scope,operations_conflict
        if maintenance(self.root):return None
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            query="SELECT t.*,json_extract(j.plan,'$.request') AS request FROM tasks t JOIN jobs j ON j.id=t.job "
            active=list(conn.execute(query+"WHERE t.status='running'"))
            if len(active)>=2:return None
            def scope(row):return operation_scope(row['operation'],json.loads(row['request'] or '{}'),json.loads(row['payload']))
            occupied=[scope(row) for row in active]
            condition="" if allow_ui else " AND json_extract(t.payload,'$.background')=1"
            # Stream narrow queue rows; a conflicting head must not starve an
            # independent audit later in the queue. Old tasks are never claimed.
            for row in conn.execute(query+"WHERE t.status='queued' AND (j.cancel=0 OR t.operation='pipeline_rollback') AND json_extract(j.plan,'$.pipeline_version')=2"+condition+" ORDER BY t.created,t.id"):
                candidate=scope(row)
                if any(row['job']==r['job'] for r in active) or any(operations_conflict(candidate,other) for other in occupied):continue
                conn.execute("UPDATE tasks SET status='running' WHERE id=?", (row["id"],))
                if row['operation']=='pipeline_rollback':
                    # Explicit queued rollback is a new intent after stopping
                    # ingestion. A later cancel removes that queued operation.
                    conn.execute("UPDATE jobs SET cancel=0,status='rolling_back',error=NULL,updated=? WHERE id=?",(now(),row['job']))
                result={k:row[k] for k in row.keys() if k!='request'}
                result['payload']=json.loads(result['payload'])
                return result
            return None

    def finish(self, task: str, error: str | None = None):
        with self.connect() as conn:
            conn.execute("UPDATE tasks SET status=?,error=? WHERE id=?", ("failed" if error else "completed", error, task))

    def cancel(self, job: str):
        from .execution_policy import LegacyTaskError
        if self.job_header(job)['pipeline_version']!=2:raise LegacyTaskError('旧任务为只读归档')
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('UPDATE jobs SET cancel=1,updated=? WHERE id=?',(now(),job))
            conn.execute("UPDATE tasks SET status='cancelled' WHERE job=? AND status='queued'", (job,))
            conn.execute("UPDATE jobs SET status='cancelled' WHERE id=? AND status NOT IN ('completed','rolled_back','partially_rolled_back','audit_pending') AND NOT EXISTS(SELECT 1 FROM tasks WHERE job=jobs.id AND status='running')",(job,))

    def check_cancel(self, job: str):
        with self.connect() as conn:
            row = conn.execute("SELECT cancel FROM jobs WHERE id=?", (job,)).fetchone()
        if row and row[0]: raise InterruptedError("任务已取消；已提交分片保留")


    def ledger_put(self, table: str, key: str, value: dict):
        self.ledger_put_many([(table,key,value)])

    def ledger_put_many(self,records,task_targets=None,batch_size=128):
        """Prepare JSON and routing before short, bounded write transactions."""
        prepared=[(table,key,dumps(value),value) for table,key,value in records]
        mirrored={'import_task','import_source_archive','import_target_status','import_row_audit','import_rollback','meta_source_mapping','meta_source_capability'}
        routes=dict(task_targets or {})
        missing={v.get('task_id') or v.get('source_task_id') for t,_,_,v in prepared if t in mirrored and v.get('target') not in {'postgres','clickhouse'}}-set(routes)-{None}
        if missing:
            with self.connect() as conn:
                routes.update({task:self._job_targets(conn,task) for task in missing})
        for offset in range(0,len(prepared),batch_size):
            block=prepared[offset:offset+batch_size];sync={}
            for table,_,_,value in block:
                if table not in mirrored:continue
                task=value.get('task_id') or value.get('source_task_id')
                if task:sync.setdefault(task,set()).update([value['target']] if value.get('target') in {'postgres','clickhouse'} else routes.get(task,[]))
                if table=='import_rollback':sync.setdefault(value['rollback_task_id'],set()).add(value['target'])
            with self.connect() as conn:
                conn.executemany('INSERT INTO ledger VALUES(?,?,?) ON CONFLICT(table_name,row_key) DO UPDATE SET value=excluded.value',[(t,k,v) for t,k,v,_ in block])
                for task,targets in sync.items():self._queue_sync(conn,task,targets)

    def invalidate_validation_reports(self,tables,targets):
        """Conservative invalidation on managed writes, without scanning business data."""
        with self.connect() as conn:
            conn.executemany('INSERT INTO validation_watermarks VALUES(?,?,1) ON CONFLICT(table_name,target) DO UPDATE SET version=version+1',[(t,target) for t in tables for target in targets])
            for row in conn.execute("SELECT row_key,value FROM ledger WHERE table_name='pipeline_validation'").fetchall():
                report=json.loads(row['value']);scope=report.get('scope',{})
                if report['status']=='stale' or not set(targets).intersection(scope.get('targets',[])):continue
                if scope.get('tables') and not set(tables).intersection(scope['tables']):continue
                report.update(previous_status=report['status'],status='stale',stale_at=now(),stale_reason='所选表发生后续写入或回退，请重新校验当前快照')
                conn.execute("UPDATE ledger SET value=? WHERE table_name='pipeline_validation' AND row_key=?",(dumps(report),row['row_key']))

    def validation_watermarks(self,tables,targets):
        with self.connect() as conn:
            rows=conn.execute('SELECT * FROM validation_watermarks').fetchall()
        return {r['target']+':'+r['table_name']:r['version'] for r in rows if r['table_name'] in tables and r['target'] in targets}

    def save_validation_report(self,report,watermarks):
        scope=report['scope'];tables=scope['tables'];targets=scope['targets']
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            rows=conn.execute('SELECT * FROM validation_watermarks').fetchall()
            current={r['target']+':'+r['table_name']:r['version'] for r in rows if r['table_name'] in tables and r['target'] in targets}
            report['watermarks']=watermarks
            if current!=watermarks:report.update(previous_status=report['status'],status='stale',stale_reason='校验期间目标表发生写入或回退')
            conn.execute("INSERT INTO ledger VALUES('pipeline_validation',?,?) ON CONFLICT(table_name,row_key) DO UPDATE SET value=excluded.value",(report['id'],dumps(report)))

    @staticmethod
    def _job_targets(conn,task):
        row=conn.execute("SELECT json_extract(plan,'$.request.targets') FROM jobs WHERE id=?",(task,)).fetchone()
        return json.loads(row[0] or '[]') if row else []

    @staticmethod
    def _queue_sync(conn,task,targets):
        for target in set(targets):
            conn.execute('INSERT INTO sync_outbox(task,target) VALUES(?,?) ON CONFLICT(task,target) DO UPDATE SET revision=revision+1',(task,target))

    def request_sync(self,task,target):
        with self.connect() as conn:self._queue_sync(conn,task,[target])


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
            conn.execute('INSERT INTO event_log(job,created,event,value) VALUES(?,?,?,?)',(job,now(),event,dumps(values)))

    def logs(self,job,after=0,limit=200,*,latest=False):
        """Return a bounded chronological page, optionally taken from the tail."""
        with self.connect() as conn:
            size=max(1,min(limit,1000))
            if latest:
                rows=list(conn.execute('SELECT * FROM event_log WHERE job=? ORDER BY id DESC LIMIT ?',(job,size)))
                rows.reverse()
            else:
                rows=conn.execute('SELECT * FROM event_log WHERE job=? AND id>? ORDER BY id LIMIT ?',(job,after,size))
            return [{**dict(r),**json.loads(r['value'])} for r in rows]

    def live_status(self,job):
        """Small change notification; log streaming must not load frozen requests."""
        with self.connect() as conn:
            row=conn.execute('SELECT id,updated,status,cancel FROM jobs WHERE id=?',(job,)).fetchone()
            if row is None:raise ValueError('任务不存在')
            return {**dict(row),'tasks':[dict(r) for r in conn.execute('SELECT id,operation,status FROM tasks WHERE job=? ORDER BY created',(job,))]}


    def ledger_get(self,table,key):
        with self.connect() as conn:
            row=conn.execute('SELECT value FROM ledger WHERE table_name=? AND row_key=?',(table,key)).fetchone()
        return json.loads(row[0]) if row else None
