"""Reference domains enter the same v2 stage ledger and resource budget."""
from .downloader import TaskDownloader
from .pipeline_store import PipelineStore
from .resource_budget import BUDGET,shared_slot
from .util import digest,now


def acquire_reference(service,item):
    """Acquire complete request chains with durable, restartable stage units."""
    service.state.ensure_execution(item['id'])
    if item['status'] in {'interrupted','cancelled'}:
        raise ValueError('中断任务须通过新版启动入口协调恢复')
    job=item['id'];request=item['plan']['request'];store=PipelineStore(service.state)
    service.state.update_job(job,status='downloading',error=None)
    service.source.active_job=job
    service.source.requests_per_minute=request['requests_per_minute']
    service.source.retrieval_policies=item['plan']['retrieval_policies']
    groups={digest(root)[:24]:[root] for root in item['plan']['requests']}
    errors=[]
    with TaskDownloader(service,item) as downloader:
        for key,roots,failures in downloader.run_groups(groups,request['download_workers']):
            marker='source:'+key
            store.put(job,marker,'download',{'request_ids':[TaskDownloader.key(downloader.prepared(r)[0]) for r in roots]})
            store.finish(job,marker,{'errors':failures},error='来源请求失败' if failures else None)
            errors.extend(failures)
    service.state.ledger_put('pipeline_progress',job,{'task_id':job,'download_status':'partial' if errors else 'completed','downloaded_at':now()})
    service.state.update_job(job,status='partial' if errors else 'downloaded')
    return errors


def run_reference(pipeline,item,commit=False):
    from .reference_service import ReferenceWorkflow
    service=pipeline.s;store=pipeline.store;job=item['id'];request=item['plan']['request']
    workflow=ReferenceWorkflow(service)
    service.state.update_job(job,status='running',cancel=0,error=None);store.recover(job)
    cache=store.get(job,'reference-cache')
    # A manual commit consumes its confirmed cache and original target heads;
    # it cannot silently replace the preview with a later database snapshot.
    if not cache or cache['status']!='completed':
        if commit:raise ValueError('参考域尚无通过核验的标准化缓存，请先启动任务')
        failures=acquire_reference(service,{**item,'status':'running'})
        store.put(job,'reference-cache','reference_normalize',{})
        store.claim(job,'reference-cache')
        with shared_slot(service.settings.state_dir,'cpu',BUDGET.cpu_workers,lambda:service.state.check_cancel(job)):
            preview=workflow.build_cache(job)
        rejected=preview['blocking'] or preview['rejected'] or failures
        store.finish(job,'reference-cache',{'path':preview['cache_path'],'hash':preview['cache_hash'],'rejected':preview['rejected']},error='参考范围证据不完整' if rejected else None)
        if rejected:
            service.state.update_job(job,status='partial',error='参考范围证据不完整，未发布')
            return pipeline.detail(job)
    else:preview=service.state.job(job)['preview']
    if not (commit or request['auto_commit']):
        service.state.update_job(job,status='ready_to_commit')
        return pipeline.detail(job)
    result=workflow.publish_cache(job,preview['token'])
    for record in service.state.ledger_rows('reference_result',job):
        marker='published:'+record['target']+':'+record['module']
        store.put(job,marker,'reference_publish',{'target':record['target'],'module':record['module']})
        store.finish(job,marker,record,error=record.get('reason') if record['status']!='published' else None)
    if result['status']=='completed' and request['post_check_rules']:
        from .pipeline_validation import ValidationRunner
        ValidationRunner(service).run(job,{'rules':request['post_check_rules']})
    return pipeline.detail(job)
