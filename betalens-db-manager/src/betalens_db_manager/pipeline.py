"""Explicit background pipeline: network producer, process conversion, independent target lanes."""
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
from contextlib import contextmanager
from copy import deepcopy
from datetime import date,timedelta
from pathlib import Path
import collections,json,multiprocessing,shutil,threading,time
from .canonical import registry_context,catalog_snapshot
from .comparison import row_hash,row_key,content_hash,compare,PROVENANCE
from .downloader import TaskDownloader
from .pipeline_cache import write_rows,read_rows,save_journal,load_journal
from .pipeline_options import light_normalization
from .pipeline_store import PipelineStore
from .target_lanes import target_lanes,lane_values
from .util import digest,dumps,now,parse_date,atomic_json


def cohort_key(root,history_buckets,bulk):
    p=root['parameters'];day=p.get('trade_date') or p.get('period') or p.get('nav_date')
    if root['endpoint']=='suspend_d':return str(p.get('start_date',''))[:4]+'-'+str(p.get('start_date',''))[4:6]
    if day:
        # Whole-market pages stay together for joins, bounded to one day locally.
        if bulk and (p.get('trade_date') or p.get('nav_date')):return str(parse_date(day))
        return str(day)[:4]+'-'+str(day)[4:6]
    security=str(p.get('ts_code') or p.get('index_code'))
    if p.get('ts_code') or p.get('index_code'):
        return 'history_batch_'+str(history_buckets[security]) if bulk else 'history_'+security
    return 'snapshots'


class LightState:
    """Suppress optional per-column evidence, retaining durable orchestration records."""
    def __init__(self,base):self.base=base
    def __getattr__(self,name):return getattr(self.base,name)
    def ledger_put(self,table,key,value):
        if table not in {'field_evidence','task_column_lineage','task_field_evidence'}:return self.base.ledger_put(table,key,value)


@contextmanager
def memory_guard(request,root):
    """Bound conversion heaps on Linux; a stopped child leaves all input archives intact."""
    try:import resource
    except ImportError:
        yield;return
    import os
    stop=threading.Event();limit=request['memory_budget_bytes']//request['normalization_workers']
    def watch():
        while not stop.wait(.1):
            used=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            if used>limit:
                atomic_json(root/'memory-limit.json',{'limit_bytes':limit,'peak_bytes':used,'reason':'转换进程达到内存预算，请缩小日期范围或降低并行数'})
                os._exit(70)
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    try:yield
    finally:stop.set();thread.join()


NORMALIZATION_POLICY_VERSION='normalization-v2-bang-nav-null-close-precision'


def normalize_child(settings,job):
    from .manager import DatabaseManagerService
    service=DatabaseManagerService(settings);service.state=LightState(service.state)
    item=service._job(job,True);root=service.settings.state_dir/'pipeline'/job
    cache_key=digest([NORMALIZATION_POLICY_VERSION,[[s['id'],s['sha'],s['status']] for s in item['slices']],item['plan']['mapping_version'],item['plan']['canonical_contracts'],item['plan']['processor_version'],item['plan']['request']])
    index=root/'normalized.json'
    if index.exists():
        saved=json.loads(index.read_text())
        if saved['cache_key']==cache_key:
            with registry_context(service.registry):
                for name,records in saved['tables'].items():
                    for record in records:read_rows(record,service.registry.spec(name))
            return saved
    started=time.monotonic()
    with memory_guard(item['plan']['request'],root),registry_context(service.registry),catalog_snapshot(service.registry),light_normalization():
        tables,rejects=service._build_canonical(item)
        result={'cache_key':cache_key,'normalization_policy':NORMALIZATION_POLICY_VERSION,'tables':{},'rejected':len(rejects),'reject_sample':rejects[:100]}
        for name,rows in tables.items():
            spec=service.registry.spec(name);records=[]
            # Typed Arrow batches cap bytes as well as rows without repeatedly serializing row dictionaries.
            import pyarrow as pa
            from .pipeline_cache import schema
            limit=item['plan']['request']['commit_batch_rows'];byte_limit=item['plan']['request']['commit_batch_bytes']
            for offset in range(0,len(rows),limit):
                pending=[rows[offset:offset+limit]]
                while pending:
                    block=pending.pop(0)
                    size=pa.Table.from_pylist(block,schema=schema(spec)).nbytes
                    if size>byte_limit and len(block)>1:
                        middle=len(block)//2;pending[:0]=[block[:middle],block[middle:]];continue
                    if size>byte_limit:raise ValueError('单行超过本地批次字节上限')
                    records.append(write_rows(root/name/(str(len(records))+'.parquet'),spec,block))
            result['tables'][name]=records
    result['normalization_seconds']=round(time.monotonic()-started,4)
    atomic_json(index,result);return result


@contextmanager
def target_lock(service,target):
    from .publishing import operation_lock
    with operation_lock(service,'pipeline-'+target+'.lock'):yield


class Pipeline:
    def __init__(self,service):
        self.s=service;self.store=PipelineStore(service.state)

    def detail(self,job,compact=False):
        item=self.s.state.job(job);units=self.store.units(job);receipts=self.store.receipts(job)
        item['pipeline']={'units':[{'id':u['id'],'kind':u['kind'],'status':u['status'],'error':u['error'],'result':u['result']} for u in units],
            'receipts':[{k:r[k] for k in ('batch','target','status','result','error')} for r in receipts],
            'business_validation':'not_run','options':{k:v for k,v in item['plan']['request'].items() if k in __import__('betalens_db_manager.pipeline_options',fromlist=['DEFAULTS']).DEFAULTS}}
        item['pipeline']['counts']=dict(collections.Counter(u['status'] for u in units))
        item['pipeline']['validations']=sorted(self.s.state.ledger_rows('pipeline_validation',job),key=lambda r:r['started_at'])
        item['pipeline']['business_validation']=item['pipeline']['validations'][-1]['status'] if item['pipeline']['validations'] else 'not_run'
        item['pipeline']['backlog_bytes']=self.backlog(job,item,units)
        item['pipeline']['download']=self.s.state.ledger_get('pipeline_progress',job)
        from .task_progress import summary
        item['pipeline']['progress']=summary(self.s,item,units)
        from .review_service import ReviewService
        item['pipeline']['reviews']=ReviewService(self.s).counts(job)
        item['pipeline']['audit_sync']=self.store.sync_status(job)
        item['audit_sync_status']='pending' if any(r['revision']>r['acknowledged'] for r in item['pipeline']['audit_sync']) else 'synced'
        from datetime import datetime
        elapsed=max(1,(datetime.fromisoformat(now())-datetime.fromisoformat(item['created'])).total_seconds())
        counts=collections.Counter(p['status'] for p in item['slices']);rows=sum(p['rows'] or 0 for p in item['slices'])
        done=sum(counts[k] for k in ('downloaded','paged','split','unavailable','contract_blocked','failed'))
        fixed=item['pipeline']['progress'].get('requests')
        remaining=max(0,fixed['total']-fixed['processed']) if fixed else None;rate=done/elapsed
        normal=[u for u in units if u['kind']=='normalize'];ready=sum(u['status']=='completed' for u in normal)
        seconds=sum((u['result'] or {}).get('normalization_seconds',0) for u in normal if u['status']=='completed')
        pending=sum(u['status'] in {'pending','running'} for u in normal)
        item['pipeline']['metrics']={'source_rows':rows,'requests_archived':done,'requests_per_minute_observed':round(rate*60,2),
            'network_eta_seconds_lower_bound':round(remaining/rate) if rate and remaining is not None else None,'estimate_excludes_dynamic_partitions':False,
            'local_pending_cohorts':pending,'local_eta_seconds':round(pending*seconds/ready/max(1,item['plan']['request'].get('normalization_workers',1))) if ready else None,
            'cooldowns':[r for r in self.s.state.logs(job,limit=100) if r.get('event')=='rate_limit_cooldown'][-5:]}
        if compact:
            item['pipeline']['datasets']=sorted({p['dataset'] for p in item['slices']})
            item['pipeline']['physical_slice_count']=len(item['slices'])
            item['slices']=item['slices'][:100]
            for unit in item['pipeline']['units']:
                result=unit['result'] or {}
                unit['result']={k:result[k] for k in ('child_id','normalization_seconds','rejected') if k in result}
        return item

    def audit(self,job,after=0,limit=100):
        """Expand only the requested page from immutable journals; no row JSON mirror."""
        import heapq
        candidates=[];receipts=self.store.receipts(job)
        for receipt in receipts:
            if receipt['batch'].startswith('rollback:'):continue
            from .pipeline_cache import file_sha
            record=receipt['manifest']
            if file_sha(record['path'])!=record['sha256']:raise ValueError('审计清单哈希不符')
            metadata=json.loads(Path(record['path']).read_text())
            for index,change in enumerate(metadata['changes']):
                ident=int(digest([job,receipt['target'],receipt['batch'],change['row_key']])[:15],16)
                if ident>after:candidates.append((ident,len(candidates),receipt,index))
        page=heapq.nsmallest(max(1,min(limit,1000)),candidates);loaded={};rows=[]
        with registry_context(self.s.registry):
            for ident,_,receipt,index in page:
                key=(receipt['batch'],receipt['target'])
                if key not in loaded:loaded[key]=load_journal(receipt['manifest'],self.s.registry.spec(receipt['manifest']['table']))
                c=loaded[key]['changes'][index]
                rows.append({'id':ident,'task_id':job,'target':receipt['target'],'table_name':receipt['manifest']['table'],'row_key':c['row_key'],
                    'operation':'deleted' if c['after'] is None else 'inserted' if c['before'] is None else 'updated',
                    'before_hash':c['before_hash'],'after_hash':c['after_hash'],'before_value':c['before'],'after_value':c['after'],'receipt_status':receipt['status'],'created_at':receipt['updated']})
        return {'changes':rows,'receipts':receipts,'next_after':rows[-1]['id'] if rows else after,
                'rollbacks':[r for r in receipts if r['batch'].startswith('rollback:')],
                'verification':self.s.state.ledger_rows('pipeline_reconciliation',job)}

    def reconcile(self,job):
        """Compare typed business after-images, retaining fill_missing differences."""
        import pyarrow as pa
        from .pipeline_cache import schema
        pairs=collections.defaultdict(dict);errors=[]
        for r in self.store.receipts(job):
            if not r['batch'].startswith('rollback:') and r['status']=='completed':pairs[r['batch']][r['target']]=r
        with registry_context(self.s.registry),ThreadPoolExecutor(max_workers=2,thread_name_prefix='reconcile') as pool:
            for batch,pair in pairs.items():
                if set(pair)!={'postgres','clickhouse'}:continue
                spec=self.s.registry.spec(pair['postgres']['manifest']['table'])
                names=[f['name'] for f in spec.fields if f['name'] not in PROVENANCE-{'available_at','availability_basis','data_quality_status'}]
                def read_target(target):
                    journal=load_journal(pair[target]['manifest'],spec)
                    rows=read_rows(journal['expected'],spec)
                    return pa.Table.from_pylist(rows,schema=schema(spec)).select(names).sort_by([(k,'ascending') for k in spec.keys])
                tables=lane_values(target_lanes(('postgres','clickhouse'),read_target,pool))
                result={'task_id':job,'batch':batch,'table':spec.name,'status':'passed' if tables[0].equals(tables[1]) else 'failed','checked_at':now()}
                self.s.state.ledger_put('pipeline_reconciliation',job+':'+batch,result)
                if result['status']=='failed':errors.append({'stage':'cross_target','batch':batch,'table':spec.name,'error':'双库业务值不同，保留各库原值，需显式协调'})
        return errors

    def backlog(self,job,item=None,units=None):
        targets=(item if item is not None else self.s.state.job(job))['plan']['request']['targets']
        children={v['bucket']:v['child_id'] for v in self.s.state.ledger_rows('pipeline_child',job)}
        units=units if units is not None else self.store.units(job);stored={(u['payload'].get('child_id'),u['payload'].get('target')) for u in units if u['kind']=='stored' and u['status']=='completed'}
        return sum(u['payload'].get('source_bytes',0) for u in units if u['kind']=='normalize' and not all((children.get(u['id']),t) in stored for t in targets))

    def refresh_coverage(self,job,target,name):
        import pyarrow.parquet as pq
        spec=self.s.registry.spec(name);codes=None
        if 'security_code' in {f['name'] for f in spec.fields}:
            codes=set()
            for unit in self.store.units(job,'normalize'):
                for record in (unit['result'] or {}).get('tables',{}).get(name,[]):
                    codes.update(c for c in pq.read_table(record['path'],columns=['security_code']).column(0).unique().to_pylist() if c)
        return (self.s.pg if target=='postgres' else self.s.ch).refresh_coverage(name,codes=codes)

    def _refresh_targets(self,job,targets,names):
        def refresh(target):
            with registry_context(self.s.registry),catalog_snapshot(self.s.registry):
                for name in sorted(names.get(target,())):
                    try:self.refresh_coverage(job,target,name)
                    except Exception as exc:self.s.state.log(job,'coverage_refresh_pending',target=target,table=name,error=self.s.safe_error(exc))
        lane_values(target_lanes(targets,refresh))

    def _child(self,item,key,roots):
        unit=self.store.put(item['id'],key,'normalize',{'roots':[TaskDownloader.key(r) for r in roots]})
        if unit['status']=='completed' and unit['result'] and unit['result'].get('child_id'):return unit['result']['child_id']
        # Child creation is journalled before processing; cache and receipts survive parent restarts.
        saved=self.s.state.ledger_get('pipeline_child',item['id']+':'+key)
        plan=deepcopy(item['plan']);plan['pipeline_parent']=item['id'];plan['pipeline_bucket']=key
        parts=self.s.state.job(item['id'])['slices'];selected=[]
        preparer=TaskDownloader(self.s,item)
        root_ids={TaskDownloader.key(preparer.prepared(r)[0]) for r in roots}
        dependency_ids={p['id'] for p in parts if p['dataset'] in {'stock_basic','fund_basic','index_basic','trade_cal'} or p['request'].get('dependency_only')}
        queue=list(root_ids|dependency_ids);by={p['id']:p for p in parts};selected_ids=set()
        while queue:
            ident=queue.pop();p=by.get(ident)
            if not p:continue
            if ident in selected_ids:continue
            selected_ids.add(ident);selected.append(p);queue.extend(TaskDownloader.key(c) for c in (p.get('result') or {}).get('children',[]))
        endpoint_pairs={(r['source_id'],r['endpoint']) for r in roots}
        names=[name for name,m in plan['mapping_snapshots'].items() if any((b['source_id'],b['endpoint']) in endpoint_pairs for b in m['bindings'])]
        if not names:names=list(plan['request']['tables'])
        plan['request']['tables']=names;plan['mapping_snapshots']={n:plan['mapping_snapshots'][n] for n in names};plan['canonical_contracts']={n:plan['canonical_contracts'][n] for n in names}
        plan['requests']=list(roots)+[p['request'] for p in selected if p['id'] in dependency_ids]
        if key[:4].isdigit() and len(key)==7:
            first=parse_date(key+'-01');last=(first.replace(day=28)+timedelta(days=4)).replace(day=1)-timedelta(days=1)
            plan['request'].update(start_date=str(max(first,parse_date(item['plan']['request']['start_date']))),end_date=str(min(last,parse_date(item['plan']['request']['end_date']))))
        elif key[:4].isdigit() and len(key)==10:
            plan['request'].update(start_date=key,end_date=key)
        source_bytes=sum(Path(p['archive']).stat().st_size for p in selected if p.get('archive') and Path(p['archive']).exists())
        with self.s.state.connect() as connection:connection.execute('UPDATE pipeline_units SET payload=? WHERE job=? AND id=?',(dumps({'roots':[TaskDownloader.key(r) for r in roots],'source_bytes':source_bytes}),item['id'],key))
        if saved:
            child=self.s.state.job(saved['child_id'])
            # Refresh a failed conversion's internal slice view after a successful retry.
            # No archived files, published intents or completed caches are removed.
            with self.s.state.connect() as connection:connection.execute('DELETE FROM slices WHERE job=?',(child['id'],))
        else:child=self.s.state.create_job(plan,self.s.settings.fingerprint)
        self.s.state.copy_slices(item['id'],child['id'],selected_ids)
        self.s.state.update_job(child['id'],status='downloaded')
        self.s.state.ledger_put('pipeline_child',item['id']+':'+key,{'task_id':item['id'],'child_id':child['id'],'bucket':key})
        return child['id']

    def _write(self,parent,child,snapshot,target):
        return self._write_many(parent,[(child,snapshot)],target)[0]

    def _write_many(self,parent,items,target):
        from .pipeline_commit import write_cohorts
        return write_cohorts(self,parent,items,target)

    def run(self,job,commit=False):
        try:return self._run(job,commit)
        except BaseException as exc:
            item=self.s.state.job(job)
            if item['plan'].get('pipeline_version')==2 and item['status'] not in {'completed','rolled_back','partially_rolled_back'}:
                self.s.state.update_job(job,status='interrupted' if isinstance(exc,InterruptedError) else 'paused',error=self.s.safe_error(exc))
            raise

    def _run(self,job,commit=False):
        item=self.s._job(job,True)
        if item['plan'].get('pipeline_version')!=2:raise ValueError('该任务不属于新管线')
        if item['status'] in {'rolled_back','partially_rolled_back','completed'}:raise ValueError('结束或回退的任务不可重新启动')
        if item['plan'].get('pipeline_parent'):raise ValueError('内部批次只能由父任务调度')
        if item['plan'].get('revision')=='reference-v1':return self._run_reference(item,commit)
        self.s.state.update_job(job,status='running',cancel=0,error=None);self.store.recover(job)
        req={**item['plan']['request']}
        if commit:req['auto_commit']=True
        self.s.source.active_job=job;self.s.source.requests_per_minute=req['requests_per_minute'];self.s.source.retrieval_policies=item['plan']['retrieval_policies']
        downloader=TaskDownloader(self.s,item);roots=[];initial_errors=[]
        for root in item['plan']['requests']:
            if root['strategy']=='universe_expansion':
                asset='fund_all' if root['endpoint']=='fund_nav' else 'index' if root['endpoint'].startswith('index_') else 'stock'
                try:universe=downloader.complete_universe(asset)
                except InterruptedError:raise
                except Exception as exc:
                    message=self.s.safe_error(exc);initial_errors.append({'endpoint':root['endpoint'],'stage':'universe','error':message})
                    self.s.state.slice(job,TaskDownloader.key(downloader.prepared(root)[0]),root['endpoint'],root,status='failed',result={'complete':False,'reason':message})
                    continue
                for code in universe:
                    if root['endpoint']=='index_daily' and code.endswith('.SI'):continue
                    p={**root['parameters'],('index_code' if root['endpoint']=='index_weight' else 'ts_code'):code}
                    roots.append({**root,'parameters':p,'strategy':'expanded_history'})
            else:roots.append(root)
        dependencies=[r for r in roots if r.get('strategy')=='dependency' or r['endpoint'] in {'stock_basic','fund_basic','index_basic','trade_cal'}]
        initial_errors.extend(downloader.run(dependencies,workers=req['download_workers']) or [])
        # Trading-date requests use the actual source calendar, including holidays, never a guessed weekday mask.
        open_days=set()
        from .source import read_archive
        for p in self.s.state.job(job)['slices']:
            if p['dataset']=='trade_cal' and p['status']=='downloaded':
                data=read_archive(Path(p['archive']),p['sha']);fields=data['fields']
                if {'cal_date','is_open'}<=set(fields):open_days.update(str(v[fields.index('cal_date')]) for v in data['items'] if str(v[fields.index('is_open')])=='1')
        groups=collections.OrderedDict()
        history_codes=sorted({str(r['parameters'].get('ts_code') or r['parameters'].get('index_code')) for r in roots if not (r['parameters'].get('trade_date') or r['parameters'].get('period') or r['parameters'].get('nav_date')) and (r['parameters'].get('ts_code') or r['parameters'].get('index_code'))})
        history_buckets={code:i//100 for i,code in enumerate(history_codes)}
        batch_history=item['plan'].get('cohort_strategy')=='security_batches_v1'
        self.s.state.log(job,'retrieval_expansion_plan',requests=len(roots),history_securities=len({code for group in history_codes for code in group.split(',')}),history_request_groups=len(history_codes),local_security_batches=(len(history_codes)+99)//100 if batch_history else len(history_codes),initial_requests=len(item['plan']['requests']))
        deps={TaskDownloader.key(downloader.prepared(r)[0]) for r in dependencies}
        for root in roots:
            p=root['parameters'];day=p.get('trade_date') or p.get('period') or p.get('nav_date')
            if root['endpoint'] in {'daily','daily_basic','adj_factor','bak_daily','fund_daily','fund_adj'} and p.get('trade_date') and open_days and str(day) not in open_days:continue
            key=cohort_key(root,history_buckets,batch_history)
            groups.setdefault(key,[]).append(root)
        from .task_progress import freeze
        if initial_errors:raise ValueError('来源清单不完整，无法冻结任务总量；请修复来源后恢复任务')
        freeze(self.s,item,groups,downloader)
        errors=initial_errors;futures=[];write_futures=[];failed_targets=set();guard=threading.Lock()
        from .pipeline_batching import WriteLane
        def target_work(items,target):
            if target in failed_targets:return [{'status':'blocked','target':target} for _ in items]
            try:
                results=self._write_many(job,items,target)
                for (child,_),result in zip(items,results):
                    marker='stored:'+child+':'+target
                    self.store.put(job,marker,'stored',{'child_id':child,'target':target});self.store.finish(job,marker,result)
                return results
            except BaseException:
                with guard:failed_targets.add(target)
                raise
        executors={t:WriteLane(lambda items,target=t:target_work(items,target),max_wait=req['commit_batch_max_wait_seconds'],
                               row_limit=req['commit_batch_rows'],name='write-'+t) for t in req['targets']}
        try:
            with ProcessPoolExecutor(max_workers=req['normalization_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
                def drain(wait=False):
                    for key,child,future in list(futures):
                        # Preserve cohort order even if a later conversion finishes first.
                        if not wait and not future.done():break
                        futures.remove((key,child,future))
                        try:
                            result=future.result()
                            self.store.finish(job,key,{**result,'child_id':child},error='转换拒收，等待重试' if result['rejected'] else None)
                            if result['rejected']:errors.append({'bucket':key,'error':'存在转换拒收，本转换批次未写入','rejected':result['rejected']})
                            if req['auto_commit'] and not result['rejected']:
                                for target in req['targets']:write_futures.append(executors[target].submit(child,result))
                        except Exception as exc:
                            message=self.s.safe_error(exc);errors.append({'bucket':key,'error':message});self.store.finish(job,key,error=message)
                for key,cohort in groups.items():
                    self.s.state.check_cancel(job);drain()
                    if shutil.disk_usage(self.s.settings.state_dir).free<req['min_free_bytes']:raise RuntimeError('剩余磁盘不足，保存检查点后暂停')
                    # Futures only carry IDs; archived payloads stay on disk so CPU does not serialize HTTP.
                    throttled=self.backlog(job)>req['backlog_high_bytes']
                    while throttled and self.backlog(job)>req['backlog_low_bytes']:
                        if failed_targets:raise RuntimeError('目标库失败且积压达到上限，已保存检查点，请恢复失败目标后继续')
                        if not req['auto_commit']:raise RuntimeError('待处理积压达到上限，请提交或分段执行')
                        self.s.state.check_cancel(job);drain();time.sleep(.1)
                    pending=[r for r in cohort if TaskDownloader.key(downloader.prepared(r)[0]) not in deps]
                    failures=downloader.run(pending,workers=req['download_workers']) or []
                    if failures:errors.extend(failures)
                    child=self._child(item,key,cohort);unit=self.store.get(job,key)
                    if unit['status']=='completed':
                        result=unit['result']
                        if req['auto_commit'] and not result.get('rejected'):
                            for target in req['targets']:write_futures.append(executors[target].submit(child,result))
                        continue
                    token=self.store.claim(job,key)
                    if token:futures.append((key,child,pool.submit(normalize_child,self.s.settings,child)))
                self.s.state.ledger_put('pipeline_progress',job,{'task_id':job,'download_status':'completed' if not errors else 'partial','downloaded_at':now()})
                drain(True)
            for executor in executors.values():executor.close()
            for future in write_futures:
                try:future.result()
                except Exception as exc:errors.append({'stage':'write','error':self.s.safe_error(exc)})
        except BaseException as exc:
            self.s.state.update_job(job,status='interrupted' if isinstance(exc,InterruptedError) else 'paused',error=self.s.safe_error(exc));raise
        finally:
            for executor in executors.values():executor.close()
            for executor in executors.values():executor.join()
        if req['auto_commit']:errors.extend(self.reconcile(job))
        status='partial' if errors or failed_targets else 'completed' if req['auto_commit'] else 'ready_to_commit'
        self.s.state.update_job(job,status=status,error=dumps(errors[:20]) if errors else None)
        self.s.state.ledger_put('pipeline_result',job,{'task_id':job,'status':status,'errors':errors,'finished_at':now()})
        # Refresh derived coverage once per table/target at task end, never per small network packet.
        receipts=self.store.receipts(job)
        names={target:{r['result']['table'] for r in receipts if r['target']==target and r['status']=='completed'} for target in req['targets'] if target not in failed_targets}
        self._refresh_targets(job,list(names),names)
        if status=='completed' and req['post_check_rules']:
            from .pipeline_validation import ValidationRunner
            ValidationRunner(self.s).run(job,{'rules':req['post_check_rules']})
        return self.detail(job)

    def commit(self,job):
        item=self.s._job(job,True)
        if item['plan']['request']['auto_commit']:raise ValueError('该任务已启用自动提交，请使用恢复入口')
        if item['status'] not in {'ready_to_commit','previewed','partial','paused','interrupted'}:raise ValueError('尚无可提交的标准化缓存')
        return self.run(job,commit=True)

    def _run_reference(self,item,commit=False):
        from .reference_service import ReferenceWorkflow
        workflow=ReferenceWorkflow(self.s);job=item['id'];self.s.state.update_job(job,cancel=0)
        if item['status'] in {'interrupted','cancelled','paused'}:self.s.state.update_job(job,status='planned')
        workflow.download(job);preview=workflow.preview(job)
        if preview['blocking'] or preview['rejected']:return self.detail(job)
        if item['plan']['request']['auto_commit'] or commit:workflow.commit(job,preview['token'])
        if self.s.state.job(job)['status']=='completed' and item['plan']['request']['post_check_rules']:
            from .pipeline_validation import ValidationRunner
            ValidationRunner(self.s).run(job,{'rules':item['plan']['request']['post_check_rules']})
        return self.detail(job)

    def rollback(self,job,targets=None):
        item=self.s._job(job);targets=targets or item['plan']['request']['targets']
        if not isinstance(targets,list) or not targets or set(targets)-set(item['plan']['request']['targets']):raise ValueError('回退目标无效')
        if item['plan']['revision']=='reference-v1':
            from .reference_service import ReferenceWorkflow
            return ReferenceWorkflow(self.s).rollback(job,targets)
        from .pipeline_writer import pg_apply,ch_apply
        from .comparison import row_hash
        originals=[r for r in self.store.receipts(job) if not r['batch'].startswith('rollback:')]
        def undo_target(target):
            results=[];started=time.monotonic()
            self.s.state.log(job,'pipeline_rollback_target_started',target=target,stage='rollback')
            with registry_context(self.s.registry),catalog_snapshot(self.s.registry):
                # Reverse each database's own durable order. Parallelizing batches
                # within a database would break overlapping before/after chains.
                for receipt in reversed(originals):
                    if receipt['target']!=target:continue
                    marker='rollback:'+receipt['batch'];saved=self.store.receipt(job,marker,target)
                    if saved and saved['status']=='completed':continue
                    try:
                        spec=self.s.registry.spec(receipt['manifest']['table']);original=load_journal(receipt['manifest'],spec)
                        inverse={'counts':{},'changes':[{'row_key':c['row_key'],'before':c['after'],'after':c['before'],'before_hash':c['after_hash'],'after_hash':c['before_hash']} for c in original['changes']]}
                        journal=self.s.settings.state_dir/'pipeline'/job/'inverse'/target/receipt['batch'].replace(':','_')
                        manifest=saved['manifest'] if saved else save_journal(journal,spec,inverse)
                        self.store.prepare(job,marker,target,manifest)
                        self.store.sync(self.s,job,target)
                        with target_lock(self.s,target):
                            result=pg_apply(self.s.pg,spec,inverse) if target=='postgres' else ch_apply(self.s.ch,self.s.state,job,marker,spec,inverse,inverse=True)
                            if result['applied']:self.s.state.invalidate_validation_reports([spec.name],[target])
                        self.store.acknowledge(job,marker,target,result);results.append({'target':target,'table':spec.name,**result})
                    except Exception as exc:
                        current=self.store.receipt(job,marker,target)
                        if current and current['status']!='completed':self.store.acknowledge(job,marker,target,{},self.s.safe_error(exc))
                        self.s.state.log(job,'pipeline_rollback_target_failed',target=target,batch=marker,stage='rollback',error=self.s.safe_error(exc))
                        raise
                    try:self.store.sync(self.s,job,target)
                    except Exception as exc:self.s.state.log(job,'pipeline_audit_sync_pending',target=target,error=self.s.safe_error(exc))
            self.s.state.log(job,'pipeline_rollback_target_completed',target=target,stage='rollback',elapsed_ms=round((time.monotonic()-started)*1000))
            return results
        outcomes=target_lanes(targets,undo_target)
        receipts=self.store.receipts(job)
        completed={(r['batch'].removeprefix('rollback:'),r['target']) for r in receipts if r['batch'].startswith('rollback:') and r['status']=='completed'}
        failures=[r for r in outcomes if r.error is not None]
        # Single-target retries can cumulatively finish a previously partial rollback.
        finished=not failures and all((r['batch'],r['target']) in completed for r in originals)
        if finished or completed:self.s.state.update_job(job,status='rolled_back' if finished else 'partially_rolled_back')
        names={r.target:{o['manifest']['table'] for o in originals if o['target']==r.target} for r in outcomes if r.error is None}
        self._refresh_targets(job,list(names),names)
        results=[v for r in outcomes if r.error is None for v in r.value]
        if failures:raise failures[0].error
        return {'job':job,'status':self.s.state.job(job)['status'],'results':results}
