"""HTTP entry points for explicit, durable business-reference management."""
from __future__ import annotations

from fastapi import Body
from .reference_contracts import TABLES, MODULE_TABLE, CONTRACT_HASH, LOGICAL_KEYS, SOURCE_MAPPINGS
from .reference_service import ReferenceWorkflow
from .util import digest, now


def schema_preview(service, targets=None):
    from .reference_storage import METADATA
    from .storage import pg_type
    targets=targets or ['postgres','clickhouse']
    if not isinstance(targets,list) or not targets or set(targets)-{'postgres','clickhouse'}:raise ValueError('目标库无效')
    specs={**TABLES,**METADATA};statements=[]
    for target in targets:
        for name,spec in specs.items():
            if target=='postgres':
                columns=[f'"{f["name"]}" {pg_type(f)}'+(' NOT NULL' if not f.get('nullable',True) or f['name'] in spec.keys else '') for f in spec.fields]
                columns.append('PRIMARY KEY ('+','.join('"'+k+'"' for k in spec.keys)+')')
                ddl=f'CREATE TABLE IF NOT EXISTS "{service.settings.pg_schema}"."{name}" ('+', '.join(columns)+')'
            else:ddl=service.ch.ddl(spec)
            statements.append({'target':target,'table':name,'zh_name':spec.zh_name,'ddl':ddl,'action':'create_if_absent'})
    result={'targets':targets,'target_fingerprint':service.settings.fingerprint,'contract_hash':CONTRACT_HASH,'statements':statements,'destructive':False}
    result['token']=digest(result)
    service.state.ledger_put('reference_schema_preview',result['token'],result)
    return result


def table_list(service):
    rows=[]
    for target in ('postgres','clickhouse'):
        try:
            store=ReferenceWorkflow(service).store(target);snapshot=store.snapshot(audit=True)
            deployed={name:store._exists(name) for name in TABLES}
            error=None
        except Exception as exc:snapshot={'rows':[],'heads':{},'events':[]};deployed={};error=service.safe_error(exc)
        for module,name in MODULE_TABLE.items():
            spec=TABLES[name];history=[r for r in snapshot['rows'] if r.get('module')==module]
            visible=[r for r in history if snapshot['heads'].get(r['scope_id'])==r['revision_id']]
            dates=[str(r['event_date']) for r in visible if r.get('event_date')]
            rows.append({'table':name,'zh_name':spec.zh_name,'description':spec.description,'grain':spec.grain,
                'target':target,'module':module,'keys':list(spec.keys),'fields':len(spec.fields),'version':spec.hash,
                'logical_keys':LOGICAL_KEYS[module],'columns':list(spec.fields),'source_mappings':SOURCE_MAPPINGS[module],
                'status':'unavailable' if error else 'ready' if deployed.get(name) else 'not_deployed',
                'error':error,'rows':len(visible),'history_rows':len(history),'trusted_scopes':sum(s['module']==module and snapshot['heads'].get(s['scope_id'])==s['revision_id'] for s in snapshot.get('scopes',[])),
                'first_date':min(dates) if dates else None,'last_date':max(dates) if dates else None,
                'last_ingested_at':max((r['ingested_at'] for r in history),default=None),
                'last_checked_at':now(),'lifecycle':'available' if deployed.get(name) else 'pending_deployment',
                'sources':['tushare'],'append_only':True,'integrity_url':'/integrity'})
    return rows


def install_routes(app,service,response):
    def workflow():return ReferenceWorkflow(service)

    @app.post('/api/integrity/check')
    def check(payload:dict=Body(...)):return response(workflow().check(payload))

    @app.get('/api/reference/tables')
    def reference_tables():return response(table_list(service))

    @app.post('/api/reference/schema/preview')
    def preview_schema(payload:dict=Body(default={})):
        return response(schema_preview(service,payload.get('targets')))

    @app.post('/api/reference/schema/apply')
    def apply_schema(payload:dict=Body(...)):
        plans=service.state.ledger_rows('reference_schema_preview')
        plan=next((p for p in plans if p['token']==payload.get('token')),None)
        if not plan or plan['target_fingerprint']!=service.settings.fingerprint or plan['contract_hash']!=CONTRACT_HASH:
            raise ValueError('结构确认已过期，请重新预览')
        if payload.get('targets') and payload['targets']!=plan['targets']:raise ValueError('目标库与预览不一致')
        results=[]
        for target in plan['targets']:
            try:result=workflow().store(target).install()
            except Exception as exc:result={'target':target,'status':'failed','reason':service.safe_error(exc)}
            results.append(result)
        operation=digest([plan['token'],now()])
        service.state.ledger_put('reference_schema_operation',operation,{'operation_id':operation,'results':results,'created_at':now(),'preview_token':plan['token']})
        return response({'operation_id':operation,'results':results})

    @app.post('/api/reference/tasks')
    def create_task(payload:dict=Body(...)):return response(workflow().plan(payload))

    @app.post('/api/reference/baselines')
    def create_baseline(payload:dict=Body(...)):
        return response(workflow().baseline(payload['request'],payload['source_task_ids']))

    @app.get('/api/reference/tasks')
    def tasks():
        with service.state.connect() as conn:
            ids=[r[0] for r in conn.execute("SELECT id FROM jobs WHERE json_extract(plan,'$.revision')='reference-v1' ORDER BY created DESC LIMIT 100")]
        return response([workflow().detail(job) for job in ids])

    @app.get('/api/reference/tasks/{job}')
    def task(job:str):return response(workflow().detail(job))

    @app.post('/api/reference/tasks/{job}/download')
    def download(job:str):
        workflow().job(job);return response({'task':service.state.enqueue(job,'reference_download')})

    @app.post('/api/reference/tasks/{job}/preview')
    def preview(job:str):
        workflow().job(job);return response({'task':service.state.enqueue(job,'reference_preview')})

    @app.post('/api/reference/tasks/{job}/commit')
    def commit(job:str,payload:dict=Body(...)):
        item=workflow().job(job)
        if not item['preview'] or item['preview']['token']!=payload.get('token'):raise ValueError('确认令牌无效')
        return response({'task':service.state.enqueue(job,'reference_commit',{'token':payload['token'],'accept_partial':bool(payload.get('accept_partial'))})})

    @app.post('/api/reference/tasks/{job}/rollback')
    def rollback(job:str,payload:dict=Body(default={})):
        workflow().job(job);return response({'task':service.state.enqueue(job,'reference_rollback',{'targets':payload.get('targets')})})

    @app.post('/api/reference/tasks/{job}/retry')
    def retry(job:str):return response(workflow().retry(job))

    @app.get('/api/reference/versions')
    def versions(target:str='postgres',audit:bool=False):return response(workflow().store(target).snapshot(audit=audit))
