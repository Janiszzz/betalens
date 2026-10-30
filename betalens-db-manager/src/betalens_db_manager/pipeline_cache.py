"""Typed immutable Parquet snapshots and exact rollback images."""
import hashlib,json,os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from .util import digest,atomic_json,dumps


def schema(spec):
    types={'date':pa.date32(),'timestamp':pa.timestamp('us',tz='UTC'),'numeric':pa.decimal128(38,12),'integer':pa.int64(),'text':pa.string()}
    return pa.schema([pa.field(f['name'],pa.decimal128(f.get('precision',38),f.get('scale',12)) if f['type']=='numeric' else types[f['type']],nullable=True) for f in spec.fields])


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write_rows(path,spec,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    table=pa.Table.from_pylist([{f['name']:r.get(f['name']) for f in spec.fields} for r in rows],schema=schema(spec))
    tmp=path.with_suffix('.tmp')
    pq.write_table(table,tmp,compression='zstd',row_group_size=50000)
    with tmp.open('rb') as h:os.fsync(h.fileno())
    tmp.replace(path)
    return {'path':str(path),'sha256':file_sha(path),'rows':len(rows),'bytes':path.stat().st_size,'schema_hash':spec.hash}


def read_rows(record,spec):
    path=Path(record['path'])
    if spec.hash!=record['schema_hash'] or file_sha(path)!=record['sha256']:raise ValueError('标准化缓存或镜像校验失败')
    return pq.read_table(path,schema=schema(spec)).to_pylist()


def save_journal(root,spec,comparison,expected=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    changes=comparison['changes']
    before=write_rows(root/'before.parquet',spec,[c['before'] or {} for c in changes])
    after=write_rows(root/'after.parquet',spec,[c['after'] or {} for c in changes])
    meta={**{k:v for k,v in comparison.items() if k not in {'changes','conflicts'}},'table':spec.name,'before':before,'after':after,
          'changes':[{k:v for k,v in c.items() if k not in {'before','after'}}|{'has_before':c['before'] is not None,'has_after':c['after'] is not None} for c in changes]}
    if expected is not None:meta['expected']=write_rows(root/'expected.parquet',spec,expected)
    atomic_json(root/'manifest.json',meta)
    return {'path':str(root/'manifest.json'),'sha256':file_sha(root/'manifest.json'),'table':spec.name}


def load_journal(record,spec):
    if file_sha(record['path'])!=record['sha256']:raise ValueError('回退清单哈希不符')
    meta=json.loads(Path(record['path']).read_text());before=read_rows(meta['before'],spec);after=read_rows(meta['after'],spec)
    return {**{k:v for k,v in meta.items() if k not in {'before','after','table'}},
            'changes':[{**{k:v for k,v in c.items() if k not in {'has_before','has_after'}},
                        'before':before[i] if c['has_before'] else None,'after':after[i] if c['has_after'] else None} for i,c in enumerate(meta['changes'])]}
