"""Shared target locks, bounded key lookup and task metadata.

Business writes and recovery are implemented only by the v2 pipeline.
"""
from contextlib import contextmanager
from datetime import datetime, timezone

from .canonical import canonical, load_canonical_catalog
from .comparison import compare, row_hash, row_key, typed, content_hash
from .mapping import VERSION
from .util import digest, dumps, now


@contextmanager
def operation_lock(service, filename="publication.lock"):
    handle=(service.settings.state_dir/filename).open('a+')
    try:
        if __import__('os').name=='nt':
            import msvcrt
            handle.write('0');handle.flush();handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    except BlockingIOError:
        raise ValueError('另一个提交或回退正在执行，请稍后重试') from None
    finally: handle.close()


def existing(target,spec,rows,fields=None):
    from dataclasses import replace
    if fields is not None:
        fields=set(fields)|set(spec.keys)
        if fields-{f['name'] for f in spec.fields}:raise ValueError('未知查询列')
        fields=[f['name'] for f in spec.fields if f['name'] in fields]
    kwargs={'fields':fields} if fields is not None else {}
    if hasattr(target,'_find_existing'):
        found=list(target._find_existing(spec,rows,**kwargs).values())
    else:
        from psycopg2.extras import RealDictCursor
        with target.connect(readonly=True) as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            found=list(target._fetch_existing(cur,spec,rows,**kwargs).values())
    selected=spec if fields is None else replace(spec,fields=tuple(f for f in spec.fields if f['name'] in fields))
    return {row_key(spec,row):typed(selected,row) for row in found}


def task_record(service,job,status,error=None,persist=True):
    item=service.state.job(job)
    request=item['plan']['request']
    old=next(iter(service.state.ledger_rows('import_task',job)),{})
    row={'task_id':job,'source_name':','.join(request.get('sources',['tushare'])),'targets':','.join(request['targets']),
         'status':status,'request':request,'mapping_version':item['plan'].get('mapping_version',VERSION),'created_at':item['created'],
         'started_at':old.get('started_at') or now(),'finished_at':now() if status in {'completed','failed','partial','rolled_back'} else None,'error':error}
    if persist:service.state.ledger_put('import_task',job,row)
    return row
