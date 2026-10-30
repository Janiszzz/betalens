"""Explicit start and independent post-validation routes."""
from fastapi import Body,Response
from .pipeline import Pipeline
from .pipeline_options import RULES
from .util import dumps


def install_routes(app,service,response):
    from .review_service import ReviewService

    @app.get('/api/tasks/{job}/reviews')
    def reviews(job:str,offset:int=0,limit:int=20,status:str=''):
        return response(ReviewService(service).list(job,offset,limit,status))

    @app.get('/api/tasks/{job}/reviews/{batch}')
    def review_detail(job:str,batch:str):return response(ReviewService(service).detail(job,batch))

    @app.get('/api/tasks/{job}/reviews/{batch}/export')
    def review_export(job:str,batch:str):
        result=response(ReviewService(service).export(job,batch))
        result.headers['Content-Disposition']='attachment; filename="rejected-'+batch+'.json"'
        return result

    @app.get('/api/tasks/{job}/reviews/{batch}/template')
    def review_template(job:str,batch:str):
        detail=ReviewService(service).detail(job,batch)
        value={'rows':{t['name']:[] for t in detail['tables']},'fields':{t['name']:{f['name']:{'type':f['type'],'required':not f.get('nullable',True),'key':f['name'] in t['keys']} for f in t['fields']} for t in detail['tables']},
               'instructions':'在 rows 中填写本地表字段；必须填写业务主键，系统审计字段自动生成。记录须属于原批次表和任务日期、证券范围。'}
        result=response(value);result.headers['Content-Disposition']='attachment; filename="repair-'+batch+'.json"';return result

    @app.post('/api/tasks/{job}/reviews/{batch}/preview')
    def review_preview(job:str,batch:str,payload:dict=Body(...)):
        return response(ReviewService(service).preview(job,batch,payload))

    @app.post('/api/tasks/{job}/reviews/{batch}/confirm')
    def review_confirm(job:str,batch:str,payload:dict=Body(...)):
        if set(payload)!={'token'}:raise ValueError('确认须携带本次审计预览令牌')
        return response(ReviewService(service).confirm(job,batch,payload['token']))

    @app.post('/api/tasks/{job}/reviews/{batch}/reject')
    def review_reject(job:str,batch:str,payload:dict=Body(...)):
        return response(ReviewService(service).reject(job,batch,payload))

    @app.post('/api/tasks/{job}/reviews/{batch}/retry')
    def review_retry(job:str,batch:str):return response(ReviewService(service).retry(job,batch))

    @app.post('/api/tasks/{job}/reviews/{batch}/reopen')
    def review_reopen(job:str,batch:str):return response(ReviewService(service).reopen(job,batch))

    @app.post('/api/tasks/{job}/replan')
    def replan(job:str,payload:dict=Body(default={})):
        if payload:raise ValueError('优化任务沿用原业务范围，不接受参数覆盖')
        return response(service.replan_retrieval(job))

    @app.post('/api/tasks/{job}/start')
    def start(job:str):
        item=service._job(job,True)
        if item['plan'].get('pipeline_version')!=2:raise ValueError('旧任务需创建显式的新管线任务')
        if item['plan'].get('pipeline_parent'):raise ValueError('内部转换批次只能由所属父任务启动')
        if item['status'] in {'completed','rolled_back','partially_rolled_back'}:raise ValueError('结束或回退的任务不可启动')
        return response({'task':service.state.enqueue(job,'pipeline_run',{'background':item['plan']['request']['run_in_background']})})

    @app.get('/api/tasks/{job}/pipeline')
    def detail(job:str):return response(Pipeline(service).detail(job))

    @app.post('/api/tasks/{job}/pipeline/commit')
    def commit(job:str):
        item=service._job(job,True)
        if item['plan'].get('pipeline_version')!=2 or item['plan']['request']['auto_commit']:raise ValueError('仅适用于选择手动提交的新任务')
        return response({'task':service.state.enqueue(job,'pipeline_commit',{'background':item['plan']['request']['run_in_background']})})

    @app.post('/api/tasks/{job}/validations')
    def validate(job:str,payload:dict=Body(default={})):
        item=service._job(job)
        if item['plan'].get('pipeline_version')!=2:raise ValueError('该入口仅适用于新管线任务')
        if not isinstance(payload.get('rules',list(RULES)),list) or not payload.get('rules',list(RULES)) or set(payload.get('rules',RULES))-set(RULES):raise ValueError('请选择有效校验规则')
        if set(payload)-{'rules','targets','tables','codes','start_date','end_date'}:raise ValueError('校验请求包含未知字段')
        return response({'task':service.state.enqueue(job,'pipeline_validate',{'background':True,'validation':payload})})

    @app.get('/api/tasks/{job}/validations')
    def validations(job:str):
        service._job(job);return response(sorted(service.state.ledger_rows('pipeline_validation',job),key=lambda r:r['started_at']))

    @app.get('/api/tasks/{job}/validations/{report_id}')
    def validation_report(job:str,report_id:str,offset:int=0,limit:int=100,download:bool=False):
        service._job(job);report=service.state.ledger_get('pipeline_validation',report_id)
        if not report or report['task_id']!=job:raise ValueError('校验报告不存在')
        if download:return Response(dumps(report),media_type='application/json',headers={'Content-Disposition':'attachment; filename=validation-'+report_id+'.json'})
        start=max(0,offset);size=max(1,min(1000,limit))
        return response({**report,'checks':report['checks'][start:start+size],'total_checks':len(report['checks']),'offset':start,'limit':size})
