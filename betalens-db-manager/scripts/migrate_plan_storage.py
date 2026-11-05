"""Rehearse lossless v2 plan deduplication in a separate SQLite copy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from betalens_db_manager.state import State
from betalens_db_manager.util import digest,atomic_json


def facts(connection):
    result={}
    tables=[r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for table in sorted(tables):
        if table in {'plan_snapshots','request_payloads','job_requests','validation_watermarks'}:continue
        if not table.replace('_','').isalnum():raise ValueError('Unexpected table name')
        columns=[r[1] for r in connection.execute('PRAGMA table_info("'+table+'")')]
        # Plans get a stronger semantic roundtrip check separately.
        if table=='jobs':columns.remove('plan')
        query='SELECT '+','.join('"'+c+'"' for c in columns)+' FROM "'+table+'" ORDER BY rowid'
        hasher=hashlib.sha256();count=0
        for row in connection.execute(query):
            hasher.update(json.dumps(tuple(row),ensure_ascii=False,separators=(',',':')).encode());hasher.update(b'\n');count+=1
        result[table]={'rows':count,'sha256':hasher.hexdigest()}
    return result


def migrate(source,destination):
    os.umask(0o077)
    destination.mkdir(mode=0o700,parents=True,exist_ok=False)
    path=destination/'tasks.sqlite3'
    with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as original,sqlite3.connect(path) as copied:
        original.backup(copied,pages=4096)
    print(json.dumps({'phase':'backup_complete','bytes':path.stat().st_size}),flush=True)
    with sqlite3.connect(path) as c:before=facts(c)
    state=State(destination)
    with state.connect() as c:
        ids=[r[0] for r in c.execute("SELECT id FROM jobs WHERE json_extract(plan,'$.pipeline_version')=2 AND json_extract(plan,'$.snapshot_ref') IS NULL ORDER BY id")]
        old_bytes=c.execute('SELECT sum(length(plan)) FROM jobs').fetchone()[0]
    hashes=hashlib.sha256();started=time.monotonic()
    for i,job in enumerate(ids,1):
        plan=state.plan(job);proof=digest(plan)
        state.replace_plan(job,plan)
        if digest(state.plan(job))!=proof:raise ValueError('Plan roundtrip failed: '+job)
        hashes.update((job+proof).encode())
        if i%100==0 or i==len(ids):print(json.dumps({'phase':'plans','done':i,'total':len(ids),'seconds':round(time.monotonic()-started)}),flush=True)
    with sqlite3.connect(path) as c:
        after=facts(c)
        if before!=after:raise ValueError('Non-plan facts changed')
        if c.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('SQLite check failed')
        new_bytes=c.execute('SELECT sum(length(plan)) FROM jobs').fetchone()[0]
        snapshots=c.execute('SELECT count(*),coalesce(sum(length(value)),0) FROM plan_snapshots').fetchone()
        requests=c.execute('SELECT count(*),coalesce(sum(length(value)),0) FROM request_payloads').fetchone()
        c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        compact=destination/'tasks-compact.sqlite3'
        c.execute('VACUUM INTO ?',(str(compact),))
    with sqlite3.connect(compact) as c:
        if c.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Compacted SQLite check failed')
        if facts(c)!=before:raise ValueError('Compacted facts changed')
    result={'migrated_jobs':len(ids),'plans_sha256':hashes.hexdigest(),'original_plan_bytes':old_bytes,
            'compact_plan_bytes':new_bytes,'snapshot_count':snapshots[0],'snapshot_bytes':snapshots[1],
            'request_count':requests[0],'request_bytes':requests[1],'before_bytes':path.stat().st_size,
            'after_bytes':compact.stat().st_size,'facts':before,'seconds':round(time.monotonic()-started)}
    atomic_json(destination/'migration-result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='facts'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();migrate(args.source.resolve(),args.destination.resolve())
