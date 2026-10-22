"""Explicit, reviewable database rebuild and initial-load commands."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from psycopg2 import sql

from .canonical import load_canonical_catalog
from .storage import CONTROL_DDL, Postgres, relation
from .util import atomic_json, identifier


def _legacy_database(settings):
    return os.environ.get("BETALENS_LEGACY_PGDATABASE", "datafeed")


def postgres_inventory(storage: Postgres, database: str | None = None) -> dict:
    """Return table/schema names and row estimates without mutating anything."""
    try:
        with storage.connect(readonly=True, database=database) as conn, conn.cursor() as cur:
            cur.execute("""SELECT n.nspname,c.relname,c.reltuples::bigint
                          FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                          WHERE c.relkind IN ('r','p','v')
                          AND n.nspname NOT IN ('pg_catalog','information_schema')
                          AND n.nspname NOT LIKE 'pg_toast%' ORDER BY n.nspname,c.relname""")
            objects=[{'schema':row[0],'name':row[1],'estimated_rows':row[2]} for row in cur.fetchall()]
            for obj in objects:
                cur.execute('SELECT column_name,data_type,is_nullable,numeric_precision,numeric_scale FROM information_schema.columns WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position',(obj['schema'],obj['name']))
                obj['columns']=[{'name':row[0],'type':row[1],'nullable':row[2],'precision':row[3],'scale':row[4]} for row in cur.fetchall()]
                cur.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(obj['schema'],obj['name'])))
                obj['rows']=cur.fetchone()[0]
                date_columns=[r['name'] for r in obj['columns'] if r['type'] in ('date','timestamp with time zone','timestamp without time zone')]
                if date_columns:
                    column=date_columns[0]
                    cur.execute(sql.SQL('SELECT min({}),max({}) FROM {}').format(sql.Identifier(column),sql.Identifier(column),sql.Identifier(obj['schema'],obj['name'])))
                    first,last=cur.fetchone();obj['coverage']={'column':column,'first':first,'last':last}
            return {'database':database or storage.settings.pg_database,'objects':objects}
    except Exception as exc:
        return {"database": database or storage.settings.pg_database, "objects": [], "error": str(exc)[:500]}


def clickhouse_inventory(replica) -> dict:
    try:
        raw = replica.execute(f"SELECT name FROM system.tables WHERE database='{replica.settings.ch_database}' FORMAT JSONEachRow")
        import json
        return {"database": replica.settings.ch_database, "tables": [json.loads(line)["name"] for line in raw.splitlines() if line.strip()]}
    except Exception as exc:
        return {"database": replica.settings.ch_database, "tables": [], "error": str(exc)[:500]}


def plan(service) -> dict:
    """Build the dry-run report used by both CLI and web operators."""
    legacy = _legacy_database(service.settings)
    return {
        "mode": "dry-run",
        "target": {"postgres": service.schema_plan()["postgres"], "clickhouse": service.ch.status()},
        "legacy_postgres": postgres_inventory(service.pg, legacy),
        "current_postgres": postgres_inventory(service.pg),
        "clickhouse": clickhouse_inventory(service.ch),
        "canonical_tables": list(load_canonical_catalog()),
        "legacy_database": legacy,
        "drop_scope": {
            "postgres_target_schema": service.settings.pg_schema,
            "postgres_legacy_schemas": os.environ.get("BETALENS_LEGACY_PGSCHEMAS", "tushare,market,betalens").split(","),
            "clickhouse_database": service.settings.ch_database,
        },
    }


def _drop_pg_schema(storage: Postgres, schema_name: str, database: str):
    if not schema_name or not schema_name.isidentifier():
        raise ValueError("非法 PostgreSQL schema 名称")
    with storage.connect(database=database) as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema_name)))


def _drop_pg_target(storage: Postgres):
    # The target schema is dropped as a unit, then install() recreates it from
    # canonical contracts.  This removes old market/tushare tables and views.
    _drop_pg_schema(storage, storage.settings.pg_schema, storage.settings.pg_database)


def rebuild(service, confirm: bool = False) -> dict:
    report = plan(service)
    if not confirm:
        return report
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report["mode"] = "confirm-drop"
    report["snapshot_id"] = stamp + "-" + uuid.uuid4().hex[:8]
    report_path = service.settings.state_dir / "rebuild" / (report["snapshot_id"] + ".json")
    atomic_json(report_path, report)

    service.pg.create_database()
    for schema_name in set([service.settings.pg_schema,'tushare','market']):
        _drop_pg_schema(service.pg,schema_name,service.settings.pg_database)
    legacy = report["legacy_database"]
    if legacy != service.settings.pg_database and not report['legacy_postgres'].get('error'):
        schemas = [name.strip() for name in os.environ.get("BETALENS_LEGACY_PGSCHEMAS", "tushare,market,betalens").split(",") if name.strip()]
        for schema_name in schemas:
            _drop_pg_schema(service.pg, schema_name, legacy)

    # Recreate the canonical PostgreSQL schema and replace every ClickHouse
    # table in the selected database.  The pre-drop inventory is retained in
    # the snapshot above for audit and recovery from external backups.
    pg_result = service.pg.install()
    service.ch.execute(f"CREATE DATABASE IF NOT EXISTS `{identifier(service.settings.ch_database)}`")
    for name in clickhouse_inventory(service.ch).get("tables", []):
        service.ch.execute(f"DROP TABLE IF EXISTS `{identifier(service.settings.ch_database)}`.`{identifier(name)}`")
    ch_result = service.ch.install()
    from .metadata import records
    for target in (service.pg,service.ch):
        for name,rows in records().items():target.control_write(name,rows)
    return {"mode": "completed", "snapshot": str(report_path), "postgres": pg_result, "clickhouse": ch_result}


def initial_load(service, config: dict) -> dict:
    """Run a configured plan through download, preview and publication."""
    job = service.plan(config)
    operation=service.state.enqueue(job['id'],'download',{'owner':'cli'})
    return {'job':job['id'],'operation':operation,'status':'queued','confirmation_required':True}



def _load_manifest(service,config):
    """Execute finite batches once; completed batches are never re-downloaded.

    The manifest itself is hashed. Re-running the same command resumes its
    current batch without enabling a scheduler or changing the requested scope.
    """
    from calendar import monthrange
    from datetime import date,timedelta
    from .util import digest,parse_date,dumps,now
    from .service import SHANGHAI
    manifest_id=digest(config)
    result=[]
    defaults=config.get('defaults',{})
    for index,entry in enumerate(config['batches']):
        request={**defaults,**entry}
        months=request.pop('batch_months',None)
        label=request.pop('label',f'batch-{index}')
        all_codes=request.pop('all_codes_for',None)
        capabilities={row['endpoint']:row for row in service.state.ledger_rows('meta_source_capability')}
        bulk=all_codes=='stock' and request.get('datasets') and all(name in {'income','balancesheet','cashflow','fina_indicator'} and capabilities.get(name+'_vip',{}).get('status')=='available' for name in request['datasets'])
        if bulk:request['datasets']=[name+'_vip' for name in request['datasets']]
        start=parse_date(request.get('start_date','2004-01-01'))
        end=parse_date(request.get('end_date',(datetime.now(SHANGHAI).date()-timedelta(days=1)).isoformat()))
        spans=[]
        if months:
            first=start
            while first<=end:
                next_index=first.year*12+(first.month-1)+int(months)
                next_date=date(next_index//12,next_index%12+1,1)
                last=min(end,next_date-timedelta(days=1))
                spans.append((first,last));first=last+timedelta(days=1)
        else:spans=[(start,end)]
        if all_codes:
            target=service.pg if 'postgres' in request.get('targets',['postgres','clickhouse']) else service.ch
            codes=[];after=None
            while True:
                page=target.query('security_dim',limit=1000,after=after)
                codes.extend(r['security_code'] for r in page['rows'] if r['asset_type']==all_codes)
                after=page['next']
                if not after:break
            if not codes:raise ValueError(f'{all_codes} 证券清单为空，需先完成基础信息批次')
            if all_codes=='stock':
                selected=set(codes);after=None
                while True:
                    aliases=target.query('security_code_history',limit=1000,after=after)
                    codes.extend(row['previous_code'] for row in aliases['rows'] if row['security_code'] in selected)
                    after=aliases['next']
                    if not after:break
                codes=sorted(set(codes))
            code_groups=[codes] if bulk else [codes[i:i+20] for i in range(0,len(codes),20)]
        else:code_groups=[request.get('codes',[])]
        for first,last in spans:
            for codes in code_groups:
                current={**request,'start_date':first.isoformat(),'end_date':last.isoformat(),'codes':codes}
                key=dumps([manifest_id,index,first,last,codes])
                with service.state.connect() as conn:
                    old=conn.execute('SELECT value FROM ledger WHERE table_name=? AND row_key=?',('initial_load_batch',key)).fetchone()
                import json
                record=json.loads(old[0]) if old else None
                if record and record['status']=='completed':continue
                job=service._job(record['task_id']) if record else service.plan(current)
                record={'task_id':job['id'],'manifest_id':manifest_id,'label':label,'start_date':first,'end_date':last,'status':'running','updated_at':now()}
                service.state.ledger_put('initial_load_batch',key,record)
                print(dumps({'task_id':job['id'],'label':label,'start_date':first,'end_date':last,'status':'running'}),flush=True)
                if job['status'] not in {'completed','rolled_back'}:
                    operation=service.state.enqueue(job['id'],'download',{'owner':'cli'})
                    record.update(status='queued',operation=operation,updated_at=now())
                    service.state.ledger_put('initial_load_batch',key,record)
                result.append({'task_id':job['id'],'label':label,'status':record['status']})
    return {'manifest_id':manifest_id,'status':'queued_for_audit','batches':result}


def load_manifest(service,config):
    from .publishing import operation_lock
    from .util import digest
    with operation_lock(service,'initial-load-'+digest(config)+'.lock'):
        return _load_manifest(service,config)
