"""Metadata-only workspace and compact task history endpoints."""
import json
from fastapi import Body
from .schema_bundle import SchemaBundle
from .reference_contracts import MODULE_TABLE


def task_history(service,limit=20,offset=0,status='',target='',table='',search=''):
    if target and target not in {'postgres','clickhouse'}:raise ValueError('数据库目标无效')
    limit=max(1,min(limit,100));offset=max(0,offset)
    conditions=["json_extract(plan,'$.pipeline_parent') IS NULL","coalesce(json_extract(plan,'$.revision'),'')!='reference-rollback-v1'"];params=[]
    if status:conditions.append('status=?');params.append(status)
    if target:
        conditions.append("EXISTS(SELECT 1 FROM json_each(json_extract(plan,'$.request.targets')) WHERE value=?)");params.append(target)
    if table:
        module=next((m for m,n in MODULE_TABLE.items() if n==table),table)
        conditions.append("(EXISTS(SELECT 1 FROM json_each(coalesce(json_extract(plan,'$.request.tables'),json_extract(plan,'$.request.datasets'),'[]')) WHERE value=?) OR EXISTS(SELECT 1 FROM json_each(json_extract(plan,'$.request.modules')) WHERE value=?))");params.extend([table,module])
    if search:conditions.append('id LIKE ?');params.append('%'+search+'%')
    where=' WHERE '+' AND '.join(conditions)
    projection="""id,created,updated,status,error,target AS fingerprint,json_extract(plan,'$.revision') AS kind,
        json_extract(plan,'$.pipeline_version') AS pipeline_version,
        coalesce(json_extract(plan,'$.request.tables'),json_extract(plan,'$.request.datasets'),json_extract(plan,'$.request.modules'),'[]') AS tables,
        coalesce(json_extract(plan,'$.request.targets'),'[]') AS targets,
        json_extract(plan,'$.request.start_date') AS start_date,json_extract(plan,'$.request.end_date') AS end_date"""
    with service.state.connect() as conn:
        total=conn.execute('SELECT count(*) FROM jobs'+where,params).fetchone()[0]
        rows=[dict(r) for r in conn.execute('SELECT '+projection+' FROM jobs'+where+' ORDER BY created DESC,id DESC LIMIT ? OFFSET ?',(*params,limit,offset))]
    for row in rows:
        for key in ('tables','targets'):row[key]=json.loads(row[key])
        if row['kind']=='reference-v1':row['tables']=[MODULE_TABLE.get(n,n) for n in row['tables']]
        row['target_matches']=row.pop('fingerprint')==service.settings.fingerprint
    return {'rows':rows,'total':total,'offset':offset,'limit':limit}


def task_detail(service,job):
    item=service.state.job(job);item['target_matches']=item['target']==service.settings.fingerprint
    if not item['target_matches']:return item
    if item['plan'].get('revision')=='reference-v1':
        from .reference_service import ReferenceWorkflow
        return {**ReferenceWorkflow(service).detail(job),'target_matches':True}
    if item['plan'].get('pipeline_version')==2:
        from .pipeline import Pipeline
        return {**Pipeline(service).detail(job,compact=True),'target_matches':True}
    item['rollback_summary']=service.state.ledger_rows('import_rollback',job)
    return item


def install_routes(app,service,response):
    @app.get('/api/schema-manifest')
    def manifest():return response(SchemaBundle(service).manifest())

    @app.post('/api/databases/inspect')
    def inspect(payload:dict=Body(default={})):
        if set(payload)-{'targets','tables'}:raise ValueError('存在未知检查参数')
        return response(SchemaBundle(service).inspect(payload.get('targets'),payload.get('tables')))

    @app.post('/api/schema/batch-plan')
    def plan(payload:dict=Body(...)):
        if set(payload)!={'selections'}:raise ValueError('需要 selections 表范围')
        return response(SchemaBundle(service).plan(payload['selections']))

    @app.post('/api/schema/batch-apply')
    def apply(payload:dict=Body(...)):
        if set(payload)!={'token'}:raise ValueError('需要建表预览令牌')
        return response(SchemaBundle(service).apply(payload['token']))

    @app.get('/api/task-history')
    def history(limit:int=20,offset:int=0,status:str='',target:str='',table:str='',search:str=''):
        return response(task_history(service,limit,offset,status,target,table,search))

    @app.get('/api/task-history/{job}')
    def detail(job:str):return response(task_detail(service,job))
