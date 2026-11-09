"""Link a new retrieval plan, retaining only exact, verified archive checkpoints."""
from .downloader import TaskDownloader
from .publishing import operation_lock
from .response_validation import validate_response
from .retrieval import universe_spec
from .source import read_archive
from .util import digest,now
from pathlib import Path
import gzip,hashlib,json


def checkpoint_fields(part,plan,request,proof):
    """Identical contracts + acquisition proof + byte hash need no row reparse."""
    path=Path(part['archive'])
    meta=json.loads(path.with_suffix('.meta.json').read_text())
    identity=request['source_id']+':'+request['endpoint']
    trusted=(proof and proof.get('endpoint')==identity
             and proof.get('response_sha256')==part['sha']
             and proof.get('request_hash')==digest(request['parameters'])
             and proof.get('rows')==part['rows']
             and (part.get('result') or {}).get('validation_errors')==[])
    if trusted:
        checksum=hashlib.sha256()
        with gzip.open(path,'rb') as stream:
            while block:=stream.read(1024*1024):checksum.update(block)
        if checksum.hexdigest()!=part['sha']:raise ValueError('原始归档哈希不匹配')
        return proof['response_fields'].split(','),meta
    data=read_archive(path,part['sha'])
    if validate_response(plan,request,plan['source_contracts'][identity],data):
        raise ValueError('归档不符合冻结的源合同')
    return data['fields'],meta


def replan(service,job,execution_options=None):
    execution_options={} if execution_options is None else execution_options
    allowed={'normalization_workers','download_workers','history_batch_size','backlog_high_bytes','backlog_low_bytes'}
    if not isinstance(execution_options,dict) or set(execution_options)-allowed:
        raise ValueError('恢复只能调整执行资源，不能改变业务范围')
    with operation_lock(service):
        old=service._job(job,True);prior=old['plan']
        if prior.get('pipeline_version')!=2 or prior.get('pipeline_parent') or prior.get('revision')!='manager-v1':raise ValueError('仅普通表的异步主任务可优化请求')
        if any(t['status'] in {'queued','running'} for t in old['tasks']):raise ValueError('请先停止原任务，等待执行操作退出后再优化')
        request={**prior['request'],**execution_options,'parent_task_id':job}
        plan=service.compile(request)
        for field in ('canonical_contracts','mapping_version','source_instances'):
            if plan.get(field)!=prior.get(field):raise ValueError('业务结构、映射或来源身份已变化，不能直接沿用原下载范围')
        # Estimates use existing verified source checkpoints, never target-table samples.
        warm=TaskDownloader(service,{'id':job,'plan':plan,'slices':[]})
        assets={'index' if r['endpoint'].startswith('index_') else 'stock' for r in plan['requests'] if r.get('strategy')=='universe_expansion'}
        for asset in sorted(assets):
            try:warm.complete_universe(asset,fetch=False)
            except (OSError,ValueError,KeyError,TypeError):pass
        from .pipeline_plan import estimate_requests
        plan['request_estimate']=estimate_requests(service,plan)
        probe=TaskDownloader(service,{'id':job,'plan':plan,'slices':[]})
        roots=list(plan['requests'])
        assets={'index' if r['endpoint'].startswith('index_') else 'stock' for r in roots if r.get('strategy')=='universe_expansion'}
        for asset in sorted(assets):
            endpoint,filters=universe_spec(asset)
            roots.extend({'source_id':'tushare','endpoint':endpoint,'parameters':p,'strategy':'retrieval_dependency','dependency_only':True} for p in filters)
        by={p['id']:p for p in old['slices']};pending=[probe.key(probe.prepared(r)[0]) for r in roots];visited=set();selected=[];archive_records=[];reused_rows=0
        # Expanded security requests have no unexpanded physical root. Keep
        # their exact verified checkpoints as well as the original root tree.
        pending.extend(by)
        proofs={r['archive_path']:r for r in service.state.ledger_rows('import_source_archive',job) if r.get('archive_path')}
        signature=lambda p:{k:p.get(k) for k in ('mode','row_limit','code_batch_size')}
        while pending:
            key=pending.pop()
            if key in visited:continue
            visited.add(key);part=by.get(key)
            if not part or not part.get('archive') or part['status'] not in {'downloaded','paged','split'}:continue
            r=part['request'];identity=r['source_id']+':'+r['endpoint']
            if identity not in plan['source_contracts'] or digest(prior['source_contracts'].get(identity))!=digest(plan['source_contracts'][identity]):continue
            if signature(prior.get('retrieval_policies',{}).get(identity,{}))!=signature(plan.get('retrieval_policies',{}).get(identity,{})):continue
            if part['status']=='split' and len((part.get('result') or {}).get('children',[]))>plan.get('retrieval_policies',{}).get(identity,{}).get('max_security_fanout',2000):continue
            if probe.key(probe.prepared(r)[0])!=key:continue
            try:
                fields,meta=checkpoint_fields(part,plan,r,proofs.get(part['archive']))
                archive_records.append({'dataset':part['dataset'],'endpoint':identity,'request_hash':digest(r['parameters']),'archive_path':part['archive'],'response_sha256':part['sha'],'response_fields':','.join(fields),'fetched_at':meta['fetched_at'],'rows':part['rows'],'slice_id':key})
            except (OSError,EOFError,ValueError,KeyError,TypeError):continue
            selected.append(key);reused_rows+=part['rows'] or 0
            pending.extend(probe.key(c) for c in (part.get('result') or {}).get('children',[]))
        plan['retrieval_replan']={'source_task_id':job,'reused_slices':len(selected),'reused_response_rows':reused_rows,'created_at':now(),'wire_bytes_unchanged':True,'initial_requests':len(plan['requests'])}
        item=service.state.create_job(plan,service.settings.fingerprint)
        service._record_mapping_plan(item)
        service.state.copy_slices(job,item['id'],selected)
        service.state.ledger_put_many([('import_source_archive',digest([item['id'],record['slice_id']]),{'task_id':item['id'],**{k:v for k,v in record.items() if k!='slice_id'}}) for record in archive_records])
        service.state.log(item['id'],'retrieval_replanned',**plan['retrieval_replan'])
        service.state.log(job,'retrieval_replacement_created',replacement_task_id=item['id'])
        return service.state.job(item['id'])
