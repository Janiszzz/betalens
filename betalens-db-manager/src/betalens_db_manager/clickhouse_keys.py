"""Query-scoped typed key tables; no persistent DDL or writer lock for lookups."""
from dataclasses import replace
from .comparison import typed


def partition_filter(db,spec,rows):
    # A non-key event_date may change on update. Pruning it could hide the
    # previous version (and bypass conflict detection), so never prune there.
    if 'event_date' not in spec.keys or not any(
            f['name']=='event_date' and f['type']=='date' and not f.get('nullable',True) for f in spec.fields):
        return ''
    dates=[r.get('event_date') for r in rows]
    if not dates or any(d is None for d in dates):return ''
    months=sorted({str(d).replace('-','')[:6] for d in dates})
    if any(len(m)!=6 or not m.isdigit() for m in months):raise ValueError('无效的分区日期')
    return 'toYYYYMM(`event_date`) IN ('+','.join(months)+') AND '


def scoped_filter(db,spec,rows,key_source,external_tables=None,table=None):
    direct=partition_filter(db,spec,rows)
    if direct:return direct
    if not any(f['name']=='event_date' and not f.get('nullable',True) for f in spec.fields):return ''
    # The usual daily tables key on trade_date, while physically partitioning
    # on event_date. Discover actual partitions from key columns only; never
    # assume the two dates match, which could hide an old/corrupt version.
    keys=','.join('`'+k+'`' for k in spec.keys)
    query=f'SELECT DISTINCT _partition_id AS partition FROM {db.table(table or spec.name)} WHERE ({keys}) IN (SELECT {keys} FROM {key_source}) SETTINGS max_threads=4 FORMAT JSONEachRow'
    kwargs={'external_tables':external_tables} if external_tables else {}
    parts=db._json_rows(db.execute(query,**kwargs))
    if not parts:return '0 AND '
    return '_partition_id IN ('+','.join(db._literal(p['partition']) for p in parts)+') AND '


def find_existing(db,spec,rows):
    from .pipeline_writer import arrow_bytes
    fields=tuple(f for f in spec.fields if f['name'] in spec.keys)
    key_spec=replace(spec,fields=fields)
    unique={}
    for row in rows:
        key=typed(spec,row,keys_only=True)
        if any(key[k] is None for k in spec.keys):raise ValueError(spec.name+': 查重主键不能为空')
        unique[tuple(key[k] for k in spec.keys)]=key
    keys=list(unique.values());result={};name='_betalens_keys'
    columns=','.join('`'+k+'`' for k in spec.keys)
    structure=','.join('`'+f['name']+'` '+db._type(f,True) for f in fields)
    # Bound rows and Arrow bytes independently. SQL text no longer scales with
    # key count. Each key packet serves a single, automatically cleaned query.
    pending=[keys[i:i+50000] for i in range(0,len(keys),50000)]
    while pending:
        packet=pending.pop(0);data=arrow_bytes(key_spec,packet)
        if len(data)>16*1024**2 and len(packet)>1:
            middle=len(packet)//2;pending[:0]=[packet[:middle],packet[middle:]];continue
        if len(data)>16*1024**2:raise ValueError('单个查重键超过大小上限')
        external={name:{'format':'ArrowStream','structure':structure,'data':data}}
        predicate=scoped_filter(db,spec,packet,name,external)
        if predicate=='0 AND ':continue
        selected=f'SELECT * FROM {db.table(spec.name)} FINAL WHERE {predicate}({columns}) IN (SELECT {columns} FROM {name})'
        query=f'SELECT t.* FROM ({selected}) AS t INNER JOIN {name} AS k USING ({columns}) SETTINGS max_threads=4 FORMAT JSONEachRow'
        for row in db._json_rows(db.execute(query,external_tables=external)):
            row=typed(spec,row);result[tuple(row[k] for k in spec.keys)]=row
    return result
