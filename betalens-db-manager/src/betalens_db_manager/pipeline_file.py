"""File ingestion uses the same immutable caches and target writers as HTTP."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from .target_lanes import target_lanes
from .util import dumps,now


def run_file(pipeline,item,commit=False):
    from .files import download_file,verify_file_cache
    from .pipeline import normalize_child
    service=pipeline.s;store=pipeline.store;job=item['id'];request=item['plan']['request']
    service.state.update_job(job,status='running',cancel=0,error=None)
    # Reuse a complete archived source; SHA verification remains part of every
    # normalization and consumption boundary, including a resumed attempt.
    proof=service.state.ledger_get('file_ingestion',job)
    if not proof or not proof.get('complete') or len(item['slices'])!=proof['parts']:
        download_file(service,job)
    verify_file_cache(service,service.state.job(job))
    key='file';store.put(job,key,'normalize',{})
    store.recover(job);store.claim(job,key)
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        snapshot=pool.submit(normalize_child,service.settings,job).result()
    snapshot={**snapshot,'child_id':job}
    store.finish(job,key,snapshot,error='文件存在拒收记录' if snapshot['rejected'] else None)
    errors=[]
    if snapshot['rejected']:errors.append({'stage':'normalize','rejected':snapshot['rejected']})
    auto=commit or request['auto_commit']
    if auto and not errors:
        outcomes=target_lanes(request['targets'],lambda target:pipeline._write(job,job,snapshot,target))
        for outcome in outcomes:
            if outcome.error is not None:errors.append({'target':outcome.target,'error':service.safe_error(outcome.error)})
            else:
                marker='stored:'+job+':'+outcome.target
                store.put(job,marker,'stored',{'child_id':job,'target':outcome.target});store.finish(job,marker,outcome.value)
        errors.extend(pipeline.reconcile(job))
        pipeline._refresh_targets(job,[o.target for o in outcomes if o.error is None],{t:set(snapshot['tables']) for t in request['targets']})
    status='partial' if errors else 'completed' if auto else 'ready_to_commit'
    service.state.update_job(job,status=status,error=dumps(errors) if errors else None)
    service.state.ledger_put('pipeline_result',job,{'task_id':job,'status':status,'errors':errors,'finished_at':now()})
    if status=='completed' and request['post_check_rules']:
        from .pipeline_validation import ValidationRunner
        ValidationRunner(service).run(job,{'rules':request['post_check_rules']})
    return pipeline.detail(job)
