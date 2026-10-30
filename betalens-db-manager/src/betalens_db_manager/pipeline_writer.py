"""Batch writers with frozen intent, native value verification and exact inverses."""
import io,uuid
import pyarrow as pa
from psycopg2 import sql
from .canonical import CanonicalTable
from .comparison import typed,row_hash,row_key
from .commit_protocol import manifest_rows,remaining_changes,TargetConflict
from .pipeline_cache import schema
from .storage import columns,relation


class ApplyResult(dict):
    """Per-intent attribution stays in memory, outside durable/public JSON."""
    def __init__(self,changes,**values):
        super().__init__(verified=True,applied=len(changes),**values)
        self.applied_keys={c['row_key'] for c in changes}


def arrow_bytes(spec,rows):
    data=pa.Table.from_pylist(rows,schema=schema(spec));out=pa.BufferOutputStream()
    with pa.ipc.new_stream(out,data.schema) as stream:stream.write_table(data)
    return out.getvalue().to_pybytes()


def ch_insert(db,spec,table,rows):
    # Each block spans at most 64 monthly partitions, below the server default 100.
    groups={}
    partitioned=any(f['name']=='event_date' and not f.get('nullable',True) for f in spec.fields)
    for row in rows:
        key=str(row.get('event_date'))[:7] if partitioned else ''
        groups.setdefault(key,[]).append(row)
    keys=list(groups)
    for start in range(0,len(keys),64):
        block=[r for k in keys[start:start+64] for r in groups[k]]
        db.execute(f'INSERT INTO {table} FORMAT ArrowStream',arrow_bytes(spec,block))


def _copy(cur,name,spec,rows):
    if not rows:return
    buf=io.StringIO()
    def value(v):
        if v is None:return r'\N'
        v=v.isoformat() if hasattr(v,'isoformat') else str(v)
        return v.replace('\\','\\\\').replace('\t','\\t').replace('\n','\\n').replace('\r','\\r')
    for row in rows:buf.write('\t'.join(value(row.get(f['name'])) for f in spec.fields)+'\n')
    buf.seek(0)
    cur.copy_expert(sql.SQL('COPY {} ({}) FROM STDIN').format(sql.Identifier(name),sql.SQL(',').join(sql.Identifier(n) for n in columns(spec))),buf)


def pg_apply(db,spec,comparison):
    from psycopg2.extras import RealDictCursor
    with db.connect() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
        target=relation(db.settings,spec.name)
        cur.execute(sql.SQL('LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE').format(target))
        live=db._fetch_existing(cur,spec,manifest_rows(comparison))
        changes=remaining_changes(spec,comparison,list(live.values()))
        for deleting in (False,True):
            rows=[c['before'] if deleting else c['after'] for c in changes if (c['after'] is None)==deleting]
            if not rows:continue
            stage='pipeline_'+uuid.uuid4().hex
            cur.execute(sql.SQL('CREATE TEMP TABLE {} (LIKE {}) ON COMMIT DROP').format(sql.Identifier(stage),target))
            _copy(cur,stage,spec,rows)
            join=sql.SQL(' AND ').join(sql.SQL('t.{}=s.{}').format(sql.Identifier(k),sql.Identifier(k)) for k in spec.keys)
            if deleting:
                cur.execute(sql.SQL('DELETE FROM {} t USING {} s WHERE {}').format(target,sql.Identifier(stage),join))
                cur.execute(sql.SQL('SELECT count(*) AS n FROM {} t JOIN {} s ON {}').format(target,sql.Identifier(stage),join))
            else:
                names=sql.SQL(',').join(sql.Identifier(n) for n in columns(spec))
                updates=sql.SQL(',').join(sql.SQL('{}=EXCLUDED.{}').format(sql.Identifier(n),sql.Identifier(n)) for n in columns(spec) if n not in spec.keys)
                action=sql.SQL('DO UPDATE SET {}').format(updates) if updates.as_string(conn) else sql.SQL('DO NOTHING')
                cur.execute(sql.SQL('INSERT INTO {} ({}) SELECT {} FROM {} WHERE true ON CONFLICT ({}) {}').format(target,names,names,sql.Identifier(stage),sql.SQL(',').join(sql.Identifier(k) for k in spec.keys),action))
                match=sql.SQL(' AND ').join(sql.SQL('t.{} IS NOT DISTINCT FROM s.{}').format(sql.Identifier(n),sql.Identifier(n)) for n in columns(spec))
                cur.execute(sql.SQL('SELECT count(*) AS n FROM {} s LEFT JOIN {} t ON {} WHERE t.{} IS NULL OR NOT ({})').format(sql.Identifier(stage),target,join,sql.Identifier(spec.keys[0]),match))
            if cur.fetchone()['n']:raise ValueError(spec.name+': 批次集合核验失败')
    return ApplyResult(changes)


def _ch_check(db,spec,stage,deleted=False,table=None,rows=None):
    from .clickhouse_keys import scoped_filter
    keys=','.join('`'+k+'`' for k in spec.keys)
    target=db.table(table or spec.name)
    predicate=scoped_filter(db,spec,rows or [],stage,table=table)
    source=f'(SELECT * FROM {target} FINAL WHERE {predicate}({keys}) IN (SELECT {keys} FROM {stage}))'
    if deleted:
        n=int(db.execute(f'SELECT count() FROM {source}'))
    else:
        join=' AND '.join('t.`'+k+'`=s.`'+k+'`' for k in spec.keys)
        match=' AND '.join(f'ifNull(t.`{n}`=s.`{n}`,isNull(t.`{n}`) AND isNull(s.`{n}`))' for n in columns(spec))
        observed=db._json_rows(db.execute(f'SELECT count() AS joined,countIf(isNull(t.`{spec.keys[0]}`) OR NOT ({match})) AS bad FROM {stage} AS s LEFT JOIN {source} AS t ON {join} SETTINGS join_use_nulls=1,max_threads=4 FORMAT JSONEachRow'))[0]
        expected=int(db.execute(f'SELECT count() FROM {stage} FINAL'))
        if int(observed['joined'])!=expected:raise ValueError(spec.name+': 批次键集合不一致')
        n=int(observed['bad'])
    if n:raise ValueError(spec.name+': ClickHouse 批次集合核验失败')


def ch_apply(db,state,job,batch,spec,comparison,inverse=False):
    with db.write_window('pipeline:'+job):
        if inverse:
            checkpoint=state.ledger_get('pipeline_ch_inverse',job+':'+batch+':'+spec.name)
            if checkpoint and db.table_digest(spec,partitions=checkpoint.get('partitions'))==checkpoint.get('expected'):
                temp=db.table(checkpoint['temporary_table'])
                db.execute(f'DROP TABLE IF EXISTS {temp}')
                return {'verified':True,'applied':0,'recovered':True}
        live=list(db._find_existing(spec,manifest_rows(comparison)).values())
        changes=remaining_changes(spec,comparison,live)
        if not changes:return ApplyResult([])
        if inverse:return _ch_inverse(db,state,job,batch,spec,comparison,changes)
        stage_name=('pipeline_'+uuid.uuid4().hex)
        stage=db.table(stage_name)
        db.execute(db.ddl(spec).replace(db.table(spec.name),stage,1))
        try:
            rows=[c['after'] for c in changes]
            if any(r is None for r in rows):raise ValueError('删除必须通过可恢复逆操作执行')
            ch_insert(db,spec,stage,rows)
            names=','.join('`'+n+'`' for n in columns(spec))
            parts=db._json_rows(db.execute(f'SELECT DISTINCT _partition_id AS id FROM {stage} FINAL FORMAT JSONEachRow'))
            for part in parts:db.execute(f'INSERT INTO {db.table(spec.name)} ({names}) SELECT {names} FROM {stage} FINAL WHERE _partition_id={db._literal(part["id"])}')
            _ch_check(db,spec,stage,rows=rows)
            return ApplyResult(changes)
        finally:db.execute(f'DROP TABLE IF EXISTS {stage}')


def _ch_inverse(db,state,job,batch,spec,comparison,changes):
    """Server-side copy excludes changed keys; no Python whole-partition reconstruction."""
    key=job+':'+batch+':'+spec.name
    old=state.ledger_get('pipeline_ch_inverse',key)
    partitioned=any(f['name']=='event_date' and not f.get('nullable',True) for f in spec.fields)
    partitions={str(row['event_date']).replace('-','')[:6] for c in comparison['changes'] for row in (c['before'],c['after']) if row and row.get('event_date')} if partitioned else {'all'}
    if old:partitions=old.get('partitions')
    current_digest=lambda name=None:db.table_digest(spec,name,partitions=partitions)
    temp_name=('inverse_'+__import__('hashlib').sha256(key.encode()).hexdigest()[:20])
    temp=db.table(temp_name);target=db.table(spec.name)
    if old and current_digest()==old.get('expected'):
        if db.execute(f'EXISTS TABLE {temp}')=='1':db.execute(f'DROP TABLE {temp}')
        return {'verified':True,'applied':0,'recovered':True}
    if old and current_digest()!=old['original']:raise TargetConflict('逆操作目标已被其他任务改变')
    original=old['original'] if old else current_digest()
    stage_name='inverse_keys_'+uuid.uuid4().hex;stage=db.table(stage_name)
    db.execute(db.ddl(spec).replace(target,stage,1))
    try:
        ch_insert(db,spec,stage,[c['before'] for c in changes])
        if db.execute(f'EXISTS TABLE {temp}')=='1':db.execute(f'DROP TABLE {temp}')
        db.execute(db.ddl(spec).replace(target,temp,1))
        keys=','.join('`'+k+'`' for k in spec.keys);names=','.join('`'+n+'`' for n in columns(spec))
        # Copy one partition at a time so long histories never hit max_partitions_per_insert_block.
        parts=db._json_rows(db.execute(f'SELECT DISTINCT _partition_id AS id FROM {target} FINAL FORMAT JSONEachRow'))
        for part in parts:
            if partitions and part['id'] not in partitions:
                db.execute(f'ALTER TABLE {temp} ATTACH PARTITION ID {db._literal(part["id"])} FROM {target}')
            else:db.execute(f'INSERT INTO {temp} ({names}) SELECT {names} FROM {target} FINAL WHERE _partition_id={db._literal(part["id"])} AND ({keys}) NOT IN (SELECT {keys} FROM {stage})')
        restored=[c['after'] for c in changes if c['after'] is not None]
        if restored:ch_insert(db,spec,temp,restored)
        fresh=list(db._find_existing(spec,manifest_rows(comparison)).values())
        if len(remaining_changes(spec,comparison,fresh))!=len(changes):raise TargetConflict('逆操作发布前目标发生变化')
        checkpoint={'task_id':job,'table':spec.name,'original':original,'expected':current_digest(temp_name),'temporary_table':temp_name,'partitions':sorted(partitions) if partitions else None}
        state.ledger_put('pipeline_ch_inverse',key,checkpoint)
        db.execute(f'EXCHANGE TABLES {target} AND {temp}')
        if current_digest()!=checkpoint['expected']:raise ValueError('逆操作受影响分区摘要不符')
        db.execute(f'DROP TABLE {temp}')
        return {'verified':True,'applied':len(changes)}
    finally:db.execute(f'DROP TABLE IF EXISTS {stage}')
