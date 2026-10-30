from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .canonical import canonical_metadata, load_canonical_catalog
from .catalog import load_catalog
from .inspection import coverage, sample, tables
from .mapping import all_source_mappings, source_capabilities
from .service import TushareImportService
from .util import dumps, digest


def create_app(service: TushareImportService | None = None) -> FastAPI:
    from .manager import DatabaseManagerService
    service = service or DatabaseManagerService()
    app = FastAPI(title="Betalens DB Manager", version="0.3.0")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "写操作须来自当前管理网页"}, status_code=403)
        from .canonical import registry_context
        with registry_context(getattr(service,"registry",None)):
            return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid(_, exc): return JSONResponse({"detail": service.safe_error(exc)}, status_code=409)

    @app.exception_handler(Exception)
    async def failed(_, exc): return JSONResponse({"detail": service.safe_error(exc)}, status_code=500)

    def response(value):
        from .security import public_data
        definitions=getattr(getattr(service,'sources',None),'values',{}).values()
        return Response(dumps(public_data(value,service.settings,definitions)), media_type="application/json")

    @app.get('/api/write-capabilities')
    def write_capabilities():
        return response([{'target':'postgres','available':True,'status':'transactional'},service.ch.write_capability()])

    @app.post('/api/jobs/{job}/sync-audit')
    def sync_audit(job:str):
        service._job(job)
        return response({'task':service.state.enqueue(job,'sync_audit')})

    @app.get("/api/config")
    def config(): return response(service.settings.public())

    @app.post("/api/config")
    def save_config(payload: dict = Body(...)):
        payload=dict(payload);destination=payload.pop('env_file',None)
        if destination is not None and (not isinstance(destination,str) or not destination.strip()):raise ValueError('配置文件路径不能为空')
        allowed = {"PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD", "PGSCHEMA", "CLICKHOUSE_URL", "CLICKHOUSE_DB", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD", "CLICKHOUSE_WRITER_MODE", "TUSHARE_TOKEN", "TUSHARE_RPM", "BETALENS_INFRA_STATE", "BETALENS_INFRA_ARCHIVE", "WIND_API_KEY", "WIND_USER", "BETALENS_DB_MANAGER_CONFIG", "BETALENS_DB_MANAGER_STATE"}
        allowed |= {v.get("auth",{}).get("env") for v in service.sources.values.values()} if hasattr(service,"sources") else set()
        unknown = set(payload) - allowed
        if unknown: raise ValueError("未知配置字段：" + ",".join(sorted(unknown)))
        from dataclasses import replace
        value_fields={'PGHOST':'pg_host','PGPORT':'pg_port','PGDATABASE':'pg_database','PGSCHEMA':'pg_schema','PGUSER':'pg_user','PGPASSWORD':'pg_password','CLICKHOUSE_URL':'ch_url','CLICKHOUSE_DB':'ch_database','CLICKHOUSE_USER':'ch_user','CLICKHOUSE_PASSWORD':'ch_password','CLICKHOUSE_WRITER_MODE':'ch_writer_mode','TUSHARE_TOKEN':'token','TUSHARE_RPM':'requests_per_minute'}
        candidate_values={value_fields[k]:int(v) if k in {'PGPORT','TUSHARE_RPM'} else str(v) for k,v in payload.items() if k in value_fields and v not in (None,'')}
        replace(service.settings,**candidate_values)  # Validate before publishing a private file.
        active_path=service.settings.env_file or service.settings.config_dir/'credentials.env'
        path = service.settings.write_private_env(payload,destination)
        from .config import Settings
        candidate=Settings.from_env(path)
        restart_required=path.resolve()!=active_path.resolve() or candidate.state_dir!=service.settings.state_dir or candidate.config_dir!=service.settings.config_dir
        updated=service.reload_config() if hasattr(service,'reload_config') and not restart_required else service.settings.public()
        import shlex,sys
        return response({'saved':True,'path':str(path),'active_path':str(active_path),'restart_required':restart_required,'config':updated,'pending_config':candidate.public() if restart_required else None,
                         'restart_command':shlex.join([sys.executable,'-m','betalens_db_manager','--env',str(path),'serve','--port','8010']) if restart_required else None})

    @app.post('/api/config/test')
    def test_config(payload: dict = Body(...)):
        from dataclasses import replace
        from .storage import Postgres
        from .clickhouse import ClickHouse
        fields={'PGHOST':'pg_host','PGPORT':'pg_port','PGDATABASE':'pg_database','PGSCHEMA':'pg_schema','PGUSER':'pg_user','PGPASSWORD':'pg_password',
                'CLICKHOUSE_URL':'ch_url','CLICKHOUSE_DB':'ch_database','CLICKHOUSE_USER':'ch_user','CLICKHOUSE_PASSWORD':'ch_password','CLICKHOUSE_WRITER_MODE':'ch_writer_mode','TUSHARE_TOKEN':'token','TUSHARE_RPM':'requests_per_minute'}
        candidate={fields[key]:int(value) if key in {'PGPORT','TUSHARE_RPM'} else str(value) for key,value in payload.items() if key in fields and value not in (None,'')}
        settings=replace(service.settings,**candidate);result={}
        targets=payload.get('targets',['postgres','clickhouse'])
        if not isinstance(targets,list) or not targets or set(targets)-{'postgres','clickhouse'}:raise ValueError('数据库目标无效')
        for name,target in [('postgres',Postgres(settings)),('clickhouse',ClickHouse(settings))]:
            if name not in targets:continue
            try:
                if name=='postgres':
                    with target.connect(readonly=True,database='postgres') as conn,conn.cursor() as cur:cur.execute('SELECT 1')
                else:target.execute('SELECT 1')
                result[name]={'connected':True}
            except Exception as exc:
                message=str(exc)
                for value in (settings.token,settings.pg_password,settings.ch_password):
                    if value:message=message.replace(value,'[redacted]')
                result[name]={'connected':False,'error':message[:500]}
        return response(result)

    @app.get("/api/doctor")
    def doctor(): return response(service.doctor())

    @app.get("/api/catalog")
    def catalog():
        if hasattr(service,'registry'):return response([{**service.registry.spec(r['definition']['name']).as_dict(),'lifecycle':r.get('status','draft')} for r in service.state.ledger_rows('table_definition')])
        return response([spec.as_dict() for spec in load_canonical_catalog().values()])

    @app.get("/api/source-catalog")
    def source_catalog(): return response([spec.as_dict() for spec in load_catalog().values()])

    @app.get("/api/mappings")
    def mappings():
        if hasattr(service,'registry'):
            from .mapping_store import mapping_rows
            return response(mapping_rows(service.registry))
        return response(all_source_mappings()+service.state.ledger_rows('meta_source_mapping'))

    @app.get("/api/capabilities")
    def capabilities():
        results=source_capabilities()
        probes={row['endpoint']:row for row in service.state.ledger_rows('meta_source_capability')}
        return response([{**row,**probes.get(row['endpoint'],{})} for row in results])

    @app.get("/api/metadata")
    def metadata(): return response(canonical_metadata())

    @app.get("/api/schema/plan")
    def schema_plan(): return response(service.schema_plan())

    @app.post("/api/schema/apply")
    def schema_apply(payload: dict = Body(...)): return response(service.initialize(payload["token"],payload.get("targets")))

    @app.get("/api/tables")
    def list_tables(target: str = "postgres"):
        if hasattr(service,"registry"):
            from .management_api import table_list
            return response([r for r in table_list(service) if r["target"]==target])
        if target=="postgres": return response(tables(service.pg))
        if target!="clickhouse": raise ValueError("目标库无效")
        return response(service.ch._json_rows(service.ch.execute(f"SELECT database AS schema,name,total_rows AS estimated_rows,total_bytes AS bytes FROM system.tables WHERE database='{service.settings.ch_database}' ORDER BY name FORMAT JSONEachRow")))

    @app.get("/api/tables/sample")
    def table_sample(schema: str, table: str, limit: int = 100): return response(sample(service.pg, schema, table, limit))

    @app.get("/api/coverage")
    def dataset_coverage(target: str = "postgres"):
        if target=="postgres": return response(coverage(service.pg))
        if target!="clickhouse": raise ValueError("目标库无效")
        rows=[]
        for name,spec in load_canonical_catalog().items():
            date_col="event_date" if any(f['name']=='event_date' for f in spec.fields) else None
            dates=f",min({date_col}) AS first_date,max({date_col}) AS last_date" if date_col else ""
            row=service.ch._json_rows(service.ch.execute(f"SELECT count() AS rows{dates} FROM {service.ch.table(name)} FINAL FORMAT JSONEachRow"))[0]
            rows.append({'dataset':name,**row})
        return response(rows)

    @app.post("/api/imports")
    def plan(payload: dict = Body(...)): return response(service.plan(payload))

    @app.get('/api/coverage/gaps')
    def gaps(table: str,code: str,start: str,end: str,target: str='postgres'):
        from .inspection import daily_gaps
        if target not in {'postgres','clickhouse'}:raise ValueError('目标库无效')
        return response(daily_gaps(service.pg if target=='postgres' else service.ch,table,code,start,end))

    @app.post("/api/files")
    def files(payload: dict = Body(...)):
        from .files import import_file
        return response(import_file(service, payload["dataset"], Path(payload["path"]), payload.get("column_map"), payload.get("mode", "fill_missing"), payload.get("encoding", "utf-8-sig"), payload.get("targets"),defer=True))

    @app.get('/api/initial-loads')
    def initial_loads():return response(service.state.ledger_rows('initial_load_batch'))

    @app.get("/api/jobs")
    def jobs(limit:int=100,offset:int=0): return response(service.state.jobs(limit,target=service.settings.fingerprint,offset=offset))

    @app.get("/api/jobs/{job}")
    def job_info(job: str):
        item=service._job(job)
        if item['plan'].get('pipeline_version')==2:
            from .pipeline import Pipeline
            return response(Pipeline(service).detail(job))
        return response(item)

    @app.get("/api/jobs/{job}/source-preview")
    def source_preview(job: str, dataset: str | None = None, limit: int = 100):
        from .source import read_archive
        from .service import normalize_source
        from .catalog import dataset as source_dataset
        from .mapping import all_source_mappings
        item=service._job(job); samples=[]; conversions=[];fields=[]
        mappings={}
        for mapping in all_source_mappings():mappings.setdefault((mapping['source_endpoint'],mapping['source_field']),mapping)
        for part in item['slices']:
            if part['status']=='split' or not part['archive'] or (dataset and part['dataset']!=dataset):continue
            data=read_archive(service.settings.state_dir/part['archive'],part['sha'])
            fields.extend({'dataset':part['dataset'],'field':name,'registered':(part['dataset'],name) in mappings} for name in data['fields'])
            selected={'fields':data['fields'],'items':data['items'][:max(1,min(limit,100))]}
            for n,values in enumerate(selected['items'],1): samples.append({'dataset':part['dataset'],'source_row':n,**dict(zip(data['fields'],values))})
            import json
            metadata=json.loads((service.settings.state_dir/part['archive']).with_suffix('.meta.json').read_text())
            if metadata.get('canonical_table'):
                for field in fields:
                    if field['dataset']==part['dataset']:field['registered']=field['field'] in metadata['column_map']
                from .sources import FileSourceAdapter,canonical_file_rules
                spec=load_canonical_catalog()[metadata['canonical_table']]
                target_fields={field['name']:field for field in spec.fields}
                rules=canonical_file_rules(spec,metadata['column_map'])
                for n,values in enumerate(selected['items'][:3],1):
                    raw=dict(zip(data['fields'],values))
                    try:normalized=FileSourceAdapter().normalize({'fields':data['fields'],'items':[values]},rules)[0]
                    except Exception as exc:
                        conversions.append({'dataset':part['dataset'],'source_row':n,'error':service.safe_error(exc)});continue
                    for field,rule in rules.items():
                        column=rule['column']
                        conversions.append({'dataset':part['dataset'],'source_row':n,'source_field':field,'raw_value':raw.get(field),'converted_value':normalized.get(column),
                                            'canonical_table':spec.name,'canonical_column':column,'target_unit':target_fields[column]['unit'],
                                            'multiplier':rule.get('multiplier'),'date_format':rule.get('date_format'),'timezone':rule.get('timezone')})
                continue
            try:
                rows,_=normalize_source(source_dataset(part['dataset']),selected,job,part['id'],item['created'])
                for n,row in enumerate(rows[:3]):
                    raw=dict(zip(data['fields'],selected['items'][n]))
                    for name,value in raw.items():
                        m=mappings.get((part['dataset'],name),{})
                        conversions.append({'dataset':part['dataset'],'source_row':n+1,'source_field':name,'raw_value':value,
                                            'source_unit':m.get('source_unit'),'converted_value':row.get(name),'target_unit':m.get('target_unit'),
                                            'canonical_table':m.get('canonical_table'),'canonical_column':m.get('canonical_column'),'rule':m.get('transform_expression')})
            except ValueError:pass
            if len(samples)>=limit:break
        return response({'fields':fields,'rows':samples[:limit],'conversions':conversions})

    @app.post("/api/probe")
    def probe(payload: dict = Body(...)): return response(service.probe(payload['datasets']))

    @app.post("/api/jobs/{job}/download")
    def download(job: str):
        service._job(job); return response(dict(task=service.state.enqueue(job, "download")))

    @app.post("/api/jobs/{job}/preview")
    def preview(job: str):
        service._job(job); return response(dict(task=service.state.enqueue(job, "preview")))

    @app.post("/api/jobs/{job}/commit")
    def commit(job: str, payload: dict = Body(...)):
        item = service._job(job)
        if not item["preview"] or item["preview"]["token"] != payload.get("token"): raise ValueError("预览令牌无效")
        return response(dict(task=service.state.enqueue(job, "commit", {"token": payload["token"], "accept_partial":bool(payload.get("accept_partial"))})))

    @app.post("/api/jobs/{job}/rollback")
    def rollback(job: str, payload: dict = Body(default={})):
        service._job(job);return response({"task":service.state.enqueue(job,"rollback",{"targets":payload.get("targets")})})

    @app.post("/api/jobs/{job}/resume")
    def resume(job: str):
        service._job(job); service.state.update_job(job, cancel=0); return response(dict(task=service.state.enqueue(job, "resume")))

    @app.post("/api/jobs/{job}/cancel")
    def cancel(job: str):
        service._job(job); service.state.cancel(job); return response({"cancel_requested": True})

    @app.get("/api/jobs/{job}/audit")
    def audit(job: str, after: int = 0, limit: int = 100):
        service._job(job); return response(service.audit(job,after,limit))

    @app.get("/api/jobs/{job}/archive/{slice_id}")
    def archive(job: str, slice_id: str, wire: bool = False):
        item = service._job(job); part = next((p for p in item["slices"] if p["id"] == slice_id), None)
        if not part or not part["archive"]: raise ValueError("归档不存在")
        path=service.settings.state_dir / part["archive"]
        if wire:
            import json
            metadata=json.loads(path.with_suffix(".meta.json").read_text())
            if not metadata.get("wire_archive"):raise ValueError("该归档没有独立的 HTTP 原文；Tushare 归档本身即原始响应")
            path=Path(metadata["wire_archive"])
        from .security import archive_export
        data,original,exported=archive_export(path,service.settings,getattr(getattr(service,'sources',None),'values',{}).values())
        return Response(data,media_type='application/gzip',headers={'Content-Disposition':'attachment; filename="'+path.name+'"',
            'X-Original-SHA256':original,'X-Export-SHA256':exported})

    @app.get("/api/jobs/{job}/events")
    async def events(job: str):
        service._job(job)
        async def generate():
            last = None
            for _ in range(120):
                from .security import public_data
                item = service.state.job(job); content = dumps(public_data(item,service.settings,getattr(getattr(service,'sources',None),'values',{}).values()))
                if content != last: yield "data: " + content + "\n\n"; last = content
                await asyncio.sleep(2)
        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/api/query/{dataset}")
    def query(dataset: str, code: str | None = None, start: str | None = None, end: str | None = None, limit: int = 100, after: str | None = None, target: str = "postgres"):
        from .reference_contracts import TABLES, MODULE_TABLE
        if dataset in TABLES:
            from .reference_service import ReferenceWorkflow
            from .util import parse_date
            snapshot=ReferenceWorkflow(service).store(target).snapshot()
            module=next(k for k,v in MODULE_TABLE.items() if v==dataset)
            rows=[r for r in snapshot['rows'] if r.get('module')==module]
            if code:rows=[r for r in rows if r.get('security_code')==code]
            # Industry intervals are selected by overlap, never valid_from alone.
            if module=='industry':
                if start:rows=[r for r in rows if r.get('valid_to') is None or str(r['valid_to'])>=str(parse_date(start))]
                if end:rows=[r for r in rows if r.get('valid_from') is None or str(r['valid_from'])<=str(parse_date(end))]
            else:
                if start:rows=[r for r in rows if r.get('event_date') and str(r['event_date'])>=str(parse_date(start))]
                if end:rows=[r for r in rows if r.get('event_date') and str(r['event_date'])<=str(parse_date(end))]
            offset=int(after or 0)
            if offset<0:raise ValueError('分页偏移无效')
            size=max(1,min(limit,1000))
            return response({'rows':rows[offset:offset+size],'next':str(offset+size) if offset+size<len(rows) else None,'trusted_only':True})
        if target == "clickhouse": return response(service.ch.query(dataset, code, start, end, limit, after))
        if target != "postgres": raise ValueError("目标库无效")
        return response(service.pg.query(dataset, code, start, end, limit, after))

    @app.get("/api/schedules")
    def schedules(): return JSONResponse({"detail":"自动更新功能已移除，请手动创建任务"},status_code=410)

    @app.post("/api/schedules")
    def save_schedule(payload: dict = Body(...)): return JSONResponse({"detail":"自动更新功能已移除，请手动创建任务"},status_code=410)

    @app.post("/api/replica/sync")
    def sync(): raise ValueError("PG/CH 已是同等目标，请针对原任务选择目标库后重试")

    if hasattr(service,"registry"):
        from .management_api import install_routes
        install_routes(app,service,response)
        from .workspace_api import install_routes as install_workspace
        install_workspace(app,service,response)
        from .reference_api import install_routes as install_reference_routes
        install_reference_routes(app,service,response)
    # Public task routes and legacy job records share one durable store.
    for route in list(app.routes):
        path=getattr(route,"path","")
        if path.startswith("/api/jobs"):
            app.add_api_route(path.replace("/api/jobs","/api/tasks",1),route.endpoint,methods=list(route.methods or ["GET"]))
    from .pipeline_api import install_routes as install_pipeline
    install_pipeline(app,service,response)
    web = Path(__file__).parent / "web"
    if web.is_dir() and (web / "index.html").exists():
        def manager_page():
            return FileResponse(web / "index.html", media_type="text/html", headers={"Cache-Control":"no-cache"})
        # Only known UI paths receive the application shell; missing API routes
        # and assets must retain their real 404 response.
        for path in ("/", "/tables", "/updates", "/tasks", "/tasks/review", "/mappings", "/sources", "/settings", "/schedules", "/integrity"):
            app.add_api_route(path, manager_page, methods=["GET", "HEAD"], include_in_schema=False)
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    else:
        @app.get("/")
        def missing_web(): return {"message": "前端尚未构建，请在 frontend 执行 npm ci && npm run build", "api": "/docs"}
    return app
