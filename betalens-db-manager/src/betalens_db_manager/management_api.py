"""Lifecycle, mapping, task and live-session routes for the local manager."""
from __future__ import annotations
from datetime import datetime,timedelta,timezone,date
from pathlib import Path
import asyncio,json,uuid
import yaml
from fastapi import Body,WebSocket,WebSocketDisconnect
from .canonical import registry_context
from .util import digest,now,parse_date


def table_list(service):
    service.registry.refresh();service.mappings.refresh();rows=[]
    with service.state.connect() as conn:
        prior={(r[0],r[1]):r[2] for r in conn.execute("SELECT json_extract(value,'$.table_name'),json_extract(value,'$.target'),max(json_extract(value,'$.created_at')) FROM ledger WHERE table_name='import_row_audit' AND json_extract(value,'$.operation') IN ('inserted','updated') GROUP BY 1,2")}
    for record in service.state.ledger_rows('table_definition'):
        v=record['definition'];name=v['name']
        sources=sorted({b['source_id']+':'+b['endpoint'] for b in service.mappings.get(name).get('bindings',[])})
        for target in ('postgres','clickhouse'):
            row={'table':name,'zh_name':v['zh_name'],'description':v['description'],'grain':v['grain'],'keys':v['keys'],'fields':len(v['fields']),'version':v['version'],'lifecycle':record.get('status','draft'),'sources':sources,'target':target,'status':'not_deployed','rows':None,'first_date':None,'last_date':None,'last_ingested_at':None,'last_checked_at':now(),'yaml_error':service.registry.errors.get(name)}
            deployed=record.get('deployed',{}).get(target,{})
            row['deployed_version']=deployed.get('definition_version')
            try:
                actual=service.schema.inspect(name,target)
                if actual['exists']:
                    from .storage import pg_type
                    by_name={f['name']:f for f in actual['fields']}
                    def matches(field):
                        found=by_name.get(field['name'])
                        if not found:return False
                        expected=pg_type(field) if target=='postgres' else service.ch._type(field,field['name'] in v['keys'])
                        return found['physical_type']==expected or (target=='postgres' and field['type']=='numeric' and found['physical_type']=='numeric' and found.get('precision')==field.get('precision',38) and found.get('scale')==field.get('scale',12))
                    from .schema_contract import compare_structure
                    row['status']='ready' if compare_structure(v,actual,target)['ready'] else 'drifted'
                    datecol=v.get('time_field');has_ingested=any(f['name']=='ingested_at' for f in actual['fields'])
                    if target=='postgres':
                        from psycopg2 import sql
                        from .storage import relation
                        temporal=sql.SQL(',min({0}),max({0})').format(sql.Identifier(datecol)) if datecol else sql.SQL(',NULL,NULL')
                        stamp=sql.SQL(',max(ingested_at)') if has_ingested else sql.SQL(',NULL')
                        with service.pg.connect(readonly=True) as conn,conn.cursor() as cur:
                            cur.execute(sql.SQL('SELECT count(*){}{} FROM {}').format(temporal,stamp,relation(service.settings,name)))
                            row['rows'],row['first_date'],row['last_date'],row['last_ingested_at']=cur.fetchone()
                    else:
                        temporal=f',minOrNull(`{datecol}`) AS first_date,maxOrNull(`{datecol}`) AS last_date' if datecol else ',NULL AS first_date,NULL AS last_date'
                        stamp=',maxOrNull(ingested_at) AS last_ingested_at' if has_ingested else ',NULL AS last_ingested_at'
                        stats=service.ch._json_rows(service.ch.execute(f'SELECT count() AS rows{temporal}{stamp} FROM {service.ch.table(name)} FINAL FORMAT JSONEachRow'))[0]
                        row.update(stats)
            except Exception as exc:row.update(status='unavailable',error=service.safe_error(exc))
            if row['last_ingested_at'] is None and prior.get((name,target)):
                row['last_ingested_at']=prior[(name,target)];row['last_ingested_basis']='historical_audit'
            rows.append(row)
    return rows


def time_coverage(service,name,target,start,end):
    definition=service.registry.get(name);start=parse_date(start);end=parse_date(end)
    if start>end or (end-start).days>36600:raise ValueError('时间范围无效或超过一百年')
    datecol=definition.get('time_field');rule=definition.get('temporal_rule','snapshot')
    result={'table':name,'target':target,'time_field':datecol,'temporal_rule':rule,'last_checked_at':now(),'days':[],'missing_ratio':None}
    if not datecol:return result
    backend=service.pg if target=='postgres' else service.ch
    if target=='postgres':
        from psycopg2 import sql
        from .storage import relation
        with backend.connect(readonly=True) as conn,conn.cursor() as cur:
            cur.execute(sql.SQL('SELECT {},count(*) FROM {} WHERE {} BETWEEN %s AND %s GROUP BY 1 ORDER BY 1').format(sql.Identifier(datecol),relation(service.settings,name),sql.Identifier(datecol)),(start,end))
            observed={str(d):int(n) for d,n in cur.fetchall()}
    elif target=='clickhouse':
        found=backend._json_rows(backend.execute(f'SELECT `{datecol}` AS day,count() AS rows FROM {backend.table(name)} FINAL WHERE `{datecol}` BETWEEN {backend._literal(start)} AND {backend._literal(end)} GROUP BY day ORDER BY day FORMAT JSONEachRow'))
        observed={str(r['day']):int(r['rows']) for r in found}
    else:raise ValueError('目标库无效')
    expected=set();calendar_known=True
    from .service import dates
    if rule=='trading_day':
        try:
            if target=='postgres':
                with backend.connect(readonly=True) as conn,conn.cursor() as cur:
                    cur.execute(sql.SQL("SELECT trade_date,is_open FROM {} WHERE exchange='SSE' AND trade_date BETWEEN %s AND %s").format(relation(service.settings,'trade_calendar')),(start,end))
                    calendar=[{'trade_date':r[0],'is_open':r[1]} for r in cur.fetchall()]
            else:
                calendar=backend._json_rows(backend.execute(f"SELECT trade_date,is_open FROM {backend.table('trade_calendar')} FINAL WHERE exchange='SSE' AND trade_date BETWEEN {backend._literal(start)} AND {backend._literal(end)} FORMAT JSONEachRow"))
            expected={str(r['trade_date']) for r in calendar if str(r.get('is_open'))=='1'}
            calendar_known=len({str(r['trade_date']) for r in calendar})==(end-start).days+1
            result.update(calendar_exchange='SSE',calendar_days_known=len(calendar))
        except Exception:calendar_known=False
    elif rule=='calendar_day':expected={str(d) for d in dates(start,end)}
    elif rule=='quarter_end':expected={str(date(y,m,d)) for y in range(start.year,end.year+1) for m,d in ((3,31),(6,30),(9,30),(12,31)) if start<=date(y,m,d)<=end}
    for day in dates(start,end):
        label=str(day);count=observed.get(label,0);is_expected=label in expected
        result['days'].append({'date':label,'rows':count,'expected':is_expected,'status':'has_records' if count else 'missing' if is_expected else 'not_expected' if rule in {'calendar_day','trading_day','quarter_end'} and calendar_known else 'unknown'})
    missing=sum(r['status']=='missing' for r in result['days'])
    if expected and calendar_known:result['missing_ratio']=str(missing/len(expected))
    result.update(missing_days=missing,expected_days=len(expected),calendar_known=calendar_known)
    service.state.ledger_put('coverage_check',digest([name,target]),result)
    return result


def session_online(state,include_grace=True):
    from time import time
    return any(time()-s.get('seen',0)<45 and (s.get('closed_at') is None or (include_grace and time()-s['closed_at']<15)) for s in state.ledger_rows('browser_session'))


def install_routes(app,service,response):
    @app.get('/api/worker-status')
    def execution_status():
        from .worker_supervisor import worker_status
        return response(worker_status(service.settings))

    from urllib.parse import urlsplit
    @app.get('/api/table-registry')
    def registry():return response(table_list(service))

    @app.post('/api/tables')
    def save_table(payload:dict=Body(...)):
        value=yaml.safe_load(payload['yaml']) if 'yaml' in payload else payload['definition']
        return response(service.registry.save(value,payload.get('file_hash')))

    @app.get('/api/tables/{name}/definition')
    def definition(name:str):
        service.registry.refresh();return response(service.registry.document(name))

    @app.post('/api/tables/{name}/schema-plan')
    def schema_plan(name:str,payload:dict=Body(default={})):
        with registry_context(service.registry):return response(service.schema.plan(name,payload.get('targets'),payload.get('action','deploy'),payload.get('destructive',False),payload.get('rename_to')))

    @app.post('/api/tables/schema-apply')
    def schema_apply(payload:dict=Body(...)):
        with registry_context(service.registry):return response(service.schema.apply(payload['token']))

    @app.post('/api/tables/{name}/archive')
    def archive(name:str):return response(service.schema.archive(name))

    @app.get('/api/tables/{name}/mapping')
    def mapping(name:str):return response(service.mappings.document(name))

    @app.get('/api/tables/{name}/field-coverage')
    def field_coverage(name:str,job:str|None=None,target:str|None=None):
        from .field_lineage import coverage
        if target is not None and target not in {'postgres','clickhouse'}:raise ValueError('目标库无效')
        return response(coverage(service,name,job,target))

    @app.post('/api/tables/{name}/mapping')
    def save_mapping(name:str,payload:dict=Body(...)):
        value=yaml.safe_load(payload['yaml']) if 'yaml' in payload else payload['mapping']
        if value.get('table')!=name:raise ValueError('映射表名不一致')
        return response(service.mappings.save(value,payload.get('file_hash')))

    @app.post('/api/tables/{name}/mapping/validate')
    def validate_mapping(name:str,payload:dict=Body(...)):
        value=service.mappings.validate(payload['mapping'])
        if value['table']!=name:raise ValueError('映射表名不一致')
        return response({'diagnostics':service.mappings.diagnostics(value)})

    @app.get('/api/tables/{name}/coverage')
    def coverage(name:str,target:str='postgres',start:str='',end:str=''):
        return response(time_coverage(service,name,target,start,end))

    @app.get('/api/tables/unmanaged/list')
    def unmanaged():
        from .inspection import tables
        managed=set(service.registry.catalog());rows=[]
        try:rows.extend({'target':'postgres',**r} for r in tables(service.pg) if r['name'] not in managed and not r['name'].startswith(('meta_','import_')))
        except Exception:pass
        try:rows.extend({'target':'clickhouse',**r} for r in service.ch._json_rows(service.ch.execute(f"SELECT name FROM system.tables WHERE database={service.ch._literal(service.settings.ch_database)} FORMAT JSONEachRow")) if r['name'] not in managed and not r['name'].startswith(('meta_','import_')))
        except Exception:pass
        return response(rows)

    @app.get('/api/tables/{name}/introspect')
    def introspect(name:str,target:str='postgres'):return response(service.schema.inspect(name,target))

    @app.post('/api/tables/archive-backfill')
    def archive_backfill(payload:dict=Body(...)):
        from .archive_migration import plan_archive_backfill
        with registry_context(service.registry):return response(plan_archive_backfill(service,payload['tables'],payload.get('targets')))

    @app.get('/api/schema/operations')
    def operations():return response(service.state.ledger_rows('schema_operation'))

    @app.get('/api/sources')
    def sources():service.sources.refresh();return response(service.sources.public())

    @app.post('/api/sources')
    def save_source(payload:dict=Body(...)):return response(service.sources.save(payload))

    @app.get('/api/field-audit')
    def field_audit(job:str|None=None):return response(service.field_audit(job))

    @app.post('/api/tasks/compile')
    def compile(payload:dict=Body(...)):return response(service.compile(payload))

    @app.post('/api/tasks')
    def create_task(payload:dict=Body(...)):return response(service.plan({'pipeline_version':2,**payload}))

    @app.post('/api/tasks/{job}/replay')
    def replay(job:str):
        return response(service.replay_archive(job))

    @app.get('/api/tasks/{job}/logs')
    def logs(job:str,after:int=0,limit:int=200,latest:bool=False):
        return response(service.state.logs(job,after,limit,latest=latest))

    @app.get('/api/tasks/{job}/evidence')
    def evidence(job:str):
        rows=service.state.ledger_rows('task_field_evidence',job)
        return response(rows[0].get('rows',[]) if rows else [])

    @app.get('/api/tasks/{job}/cache')
    def cache(job:str,offset:int=0,limit:int=100,dataset:str|None=None):
        item=service._job(job);offset=max(0,offset);size=max(1,min(limit,1000));samples=[];cursor=0
        from .source import read_archive
        parts=[p for p in item['slices'] if p.get('archive') and p['status']!='split' and (not dataset or p['dataset']==dataset) and Path(p['archive']).exists()]
        total=sum(p['rows'] or 0 for p in parts)
        for part in parts:
            count=part['rows'] or 0
            if cursor+count<=offset:cursor+=count;continue
            data=read_archive(Path(part['archive']),part['sha']);start=max(0,offset-cursor)
            for n,values in enumerate(data['items'][start:start+size-len(samples)],start+1):samples.append({'endpoint':part['dataset'],'source_row':n,**dict(zip(data['fields'],values))})
            cursor+=count
            if len(samples)>=size:break
        return response({'rows':samples,'total':total,'offset':offset,'next':offset+size if offset+size<total else None,'cache_purged':bool(item['plan'].get('cache_purged'))})

    @app.post('/api/sessions')
    def open_session():
        import time
        sid=uuid.uuid4().hex;service.state.ledger_put('browser_session',sid,{'id':sid,'seen':time.time(),'closed_at':None})
        return response({'id':sid})

    @app.post('/api/sessions/{sid}/heartbeat')
    def heartbeat(sid:str):
        import time
        previous=service.state.ledger_get('browser_session',sid)
        if not previous or previous.get('closed_at') is not None:return response({'online':False,'closed':True})
        service.state.ledger_put('browser_session',sid,{'id':sid,'seen':time.time(),'closed_at':None});return response({'online':True})

    @app.post('/api/sessions/{sid}/close')
    def close(sid:str):
        import time
        service.state.ledger_put('browser_session',sid,{'id':sid,'seen':time.time(),'closed_at':time.time()});return response({'stopping_after_reconnect_grace':True})

    @app.websocket('/api/live/{job}')
    async def live(websocket:WebSocket,job:str,after:int=0,latest:bool=False,limit:int=500):
        from .security import public_data
        def safe(value):return public_data(value,service.settings,service.sources.values.values())
        origin=websocket.headers.get('origin')
        if origin and urlsplit(origin).netloc!=websocket.headers.get('host'):await websocket.close(code=1008);return
        await websocket.accept();after=max(0,after);last=None;initial=True
        try:
            while True:
                rows=service.state.logs(job,after,limit,latest=True) if latest else service.state.logs(job,after)
                if latest and initial:
                    # Start at the current tail, never replay a large historical backlog.
                    await websocket.send_text(__import__('betalens_db_manager.util',fromlist=['dumps']).dumps(safe({'type':'logs','records':rows})))
                    after=rows[-1]['id'] if rows else after
                    rows=[]
                for row in rows:
                    if latest and row['id']<=after:continue
                    await websocket.send_text(__import__('betalens_db_manager.util',fromlist=['dumps']).dumps(safe({'type':'log','record':row})));after=row['id']
                if latest:await websocket.send_text('{"type":"heartbeat"}')
                initial=False
                item=service.state.live_status(job) if latest else service.state.job(job)
                signature=digest(item if latest else [item['updated'],[(p['id'],p['status'],p['rows']) for p in item['slices']],item['tasks']])
                if signature!=last:
                    await websocket.send_text(__import__('betalens_db_manager.util',fromlist=['dumps']).dumps(safe({'type':'task','record':item})));last=signature
                await asyncio.sleep(1)
        except (WebSocketDisconnect,RuntimeError):pass
