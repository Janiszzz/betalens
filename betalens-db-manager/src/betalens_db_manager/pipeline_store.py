"""Small durable stage records; large immutable payloads stay outside SQLite."""
import json,time,uuid
from .util import dumps,now


class PipelineStore:
    def __init__(self,state):
        self.state=state
        with state.connect() as c:
            c.executescript('''CREATE TABLE IF NOT EXISTS pipeline_units(
                job TEXT NOT NULL,id TEXT NOT NULL,kind TEXT NOT NULL,status TEXT NOT NULL,
                payload TEXT NOT NULL,result TEXT,error TEXT,lease TEXT,lease_until REAL,
                created TEXT NOT NULL,updated TEXT NOT NULL,PRIMARY KEY(job,id));
                CREATE INDEX IF NOT EXISTS pipeline_ready ON pipeline_units(kind,status,created);
                CREATE TABLE IF NOT EXISTS pipeline_receipts(
                job TEXT NOT NULL,batch TEXT NOT NULL,target TEXT NOT NULL,status TEXT NOT NULL,
                manifest TEXT NOT NULL,result TEXT,error TEXT,updated TEXT NOT NULL,
                PRIMARY KEY(job,batch,target));
                CREATE TABLE IF NOT EXISTS pipeline_sync(job TEXT,batch TEXT,target TEXT,revision INTEGER NOT NULL DEFAULT 1,
                acknowledged INTEGER NOT NULL DEFAULT 0,error TEXT,PRIMARY KEY(job,batch,target));''')

    def put(self,job,key,kind,payload):
        with self.state.connect() as c:
            c.execute('INSERT OR IGNORE INTO pipeline_units VALUES(?,?,?,?,?,NULL,NULL,NULL,NULL,?,?)',
                      (job,key,kind,'pending',dumps(payload),now(),now()))
        return self.get(job,key)

    def get(self,job,key):
        with self.state.connect() as c:r=c.execute('SELECT * FROM pipeline_units WHERE job=? AND id=?',(job,key)).fetchone()
        if r is None:return None
        return {**dict(r),'payload':json.loads(r['payload']),'result':json.loads(r['result']) if r['result'] else None}

    def units(self,job,kind=None):
        with self.state.connect() as c:
            rows=c.execute('SELECT * FROM pipeline_units WHERE job=?'+(' AND kind=?' if kind else '')+' ORDER BY created,id', (job,kind) if kind else (job,)).fetchall()
        return [{**dict(r),'payload':json.loads(r['payload']),'result':json.loads(r['result']) if r['result'] else None} for r in rows]

    def claim(self,job,key,seconds=120):
        token=uuid.uuid4().hex
        with self.state.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            r=c.execute('SELECT status,lease_until FROM pipeline_units WHERE job=? AND id=?',(job,key)).fetchone()
            if not r or r['status']=='completed' or r['status']=='running' and (r['lease_until'] or 0)>time.time():return None
            c.execute('UPDATE pipeline_units SET status=?,lease=?,lease_until=?,updated=? WHERE job=? AND id=?',('running',token,time.time()+seconds,now(),job,key))
        return token

    def finish(self,job,key,result=None,error=None,token=None):
        with self.state.connect() as c:
            where='job=? AND id=?'+(' AND lease=?' if token else '')
            c.execute('UPDATE pipeline_units SET status=?,result=?,error=?,lease=NULL,lease_until=NULL,updated=? WHERE '+where,
                      ('failed' if error else 'completed',dumps(result) if result is not None else None,error,now(),job,key,*([token] if token else [])))

    def receipt(self,job,batch,target):
        with self.state.connect() as c:r=c.execute('SELECT * FROM pipeline_receipts WHERE job=? AND batch=? AND target=?',(job,batch,target)).fetchone()
        return {**dict(r),'manifest':json.loads(r['manifest']),'result':json.loads(r['result']) if r['result'] else None} if r else None

    def prepare(self,job,batch,target,manifest):
        self.prepare_many(job,target,[(batch,manifest)])
        return self.receipt(job,batch,target)

    def prepare_many(self,job,target,entries):
        if not entries:return
        with self.state.connect() as c:
            for batch,manifest in entries:
                cursor=c.execute('INSERT OR IGNORE INTO pipeline_receipts VALUES(?,?,?,?,?,NULL,NULL,?)',(job,batch,target,'prepared',dumps(manifest),now()))
                if cursor.rowcount:c.execute('INSERT OR IGNORE INTO pipeline_sync(job,batch,target) VALUES(?,?,?)',(job,batch,target))
                else:
                    saved=c.execute('SELECT manifest FROM pipeline_receipts WHERE job=? AND batch=? AND target=?',(job,batch,target)).fetchone()
                    if json.loads(saved['manifest'])!=manifest:raise ValueError('固定批次镜像不可覆盖')

    def acknowledge(self,job,batch,target,result,error=None):
        self.acknowledge_many(job,target,[(batch,result,error)])

    def acknowledge_many(self,job,target,entries):
        if not entries:return
        with self.state.connect() as c:
            stamp=now()
            c.executemany('UPDATE pipeline_receipts SET status=?,result=?,error=?,updated=? WHERE job=? AND batch=? AND target=?',
                          [('failed' if error else 'completed',dumps(result),error,stamp,job,batch,target) for batch,result,error in entries])
            c.executemany('UPDATE pipeline_sync SET revision=revision+1 WHERE job=? AND batch=? AND target=?',[(job,batch,target) for batch,_,_ in entries])

    def receipts(self,job):
        with self.state.connect() as c:rows=c.execute('SELECT * FROM pipeline_receipts WHERE job=? ORDER BY rowid',(job,)).fetchall()
        return [{**dict(r),'manifest':json.loads(r['manifest']),'result':json.loads(r['result']) if r['result'] else None} for r in rows]

    def sync_status(self,job):
        with self.state.connect() as c:return [dict(r) for r in c.execute('SELECT * FROM pipeline_sync WHERE job=?',(job,))]

    def sync(self,service,job,target):
        """Mirror only dirty small manifests; row images remain immutable files."""
        from .storage import CONTROL_DDL,control_spec,relation
        from psycopg2 import sql
        with self.state.connect() as c:
            dirty=c.execute('''SELECT r.*,s.revision FROM pipeline_sync s JOIN pipeline_receipts r
                ON r.job=s.job AND r.batch=s.batch AND r.target=s.target
                WHERE s.job=? AND s.target=? AND s.revision>s.acknowledged ORDER BY r.rowid''',(job,target)).fetchall()
        if not dirty:return
        db=service.pg if target=='postgres' else service.ch;name='import_batch_receipt'
        if target=='postgres':
            definitions=[sql.SQL('{} {}{}').format(sql.Identifier(col),sql.SQL(kind),sql.SQL(' NOT NULL' if required else '')) for col,kind,required in CONTROL_DDL[name]]
            definitions.append(sql.SQL('PRIMARY KEY(task_id,batch_id,target)'))
            with db.connect() as conn,conn.cursor() as cur:cur.execute(sql.SQL('CREATE TABLE IF NOT EXISTS {} ({})').format(relation(service.settings,name),sql.SQL(',').join(definitions)))
        else:db.execute(db.ddl(control_spec(name)))
        for offset in range(0,len(dirty),128):
            entries=dirty[offset:offset+128];rows=[]
            for entry in entries:
                manifest=json.loads(entry['manifest'])
                rows.append({'task_id':job,'batch_id':entry['batch'],'target':target,'table_name':manifest['table'],
                    'status':entry['status'],'manifest':manifest,'result':json.loads(entry['result']) if entry['result'] else None,'error':entry['error'],'updated_at':entry['updated']})
            try:db.control_write(name,rows)
            except Exception as exc:
                with self.state.connect() as c:c.executemany('UPDATE pipeline_sync SET error=? WHERE job=? AND batch=? AND target=?',[(service.safe_error(exc),job,e['batch'],target) for e in entries])
                raise
            # A newer local revision must remain dirty if it changed in flight.
            with self.state.connect() as c:c.executemany('UPDATE pipeline_sync SET acknowledged=max(acknowledged,?),error=CASE WHEN revision<=? THEN NULL ELSE error END WHERE job=? AND batch=? AND target=?',[(e['revision'],e['revision'],job,e['batch'],target) for e in entries])

    def recover(self,job):
        with self.state.connect() as c:
            c.execute("UPDATE pipeline_units SET status='pending',lease=NULL,lease_until=NULL WHERE job=? AND status='running'",(job,))
