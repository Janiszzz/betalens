"""Checkpointed source retrieval. A saturated response is evidence, never a complete leaf."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import threading
import time

from .retrieval import partition, tushare_policy,universe_spec,universe_hint_key,security_partitions
from .source import read_archive
from .source_registry import FetchContext
from .util import atomic_json, digest, dumps, now
from .util import parse_date


class TaskDownloader:
    RETRY_DELAYS=(5,30)

    def __init__(self, service, item):
        self.s, self.item, self.job = service, item, item['id']
        self.plan = item['plan']
        self.universes = {}
        self.universe_lock=threading.RLock()
        self.cache_lock=threading.RLock();self.response_cache={};self.wire_locks={}
        for part in item.get('slices',[]):
            if part.get('archive') and part['status'] in {'downloaded','paged','split'}:self.response_cache[self.wire_key(part['request'])]=part
        self.blocked_endpoints={}
        for row in service.state.ledger_rows('pipeline_source_block',self.job):self.blocked_endpoints[(row['source_id'],row['endpoint'])]=row

    @staticmethod
    def wire_key(request):return digest([request['source_id'],request['endpoint'],request['parameters']])

    @staticmethod
    def key(request):
        identity = [request['source_id'], request['endpoint'], request['parameters']]
        if request.get('dependency_only'): identity.append('retrieval_dependency')
        return digest(identity)[:24]

    def reuse_response(self,request,contract):
        with self.cache_lock:part=self.response_cache.get(self.wire_key(request))
        if not part or part['id']==self.key(request):return None
        try:
            data=read_archive(Path(part['archive']),part['sha'])
            meta=json.loads(Path(part['archive']).with_suffix('.meta.json').read_text())
        except FileNotFoundError:
            with self.cache_lock:self.response_cache.pop(self.wire_key(request),None)
            self.s.state.log(self.job,'archive_missing_retry',endpoint=request['endpoint'],request_id=part['id'])
            return None
        from .response_validation import validate_response
        if validate_response(self.plan,request,contract,data):return None
        result={**(part.get('result') or {}),'cache_reused_from':part['id']}
        result['children']=[{**request,'parameters':c['parameters']} for c in result.get('children',[])]
        self.s.state.slice(self.job,self.key(request),request['endpoint'],request,status=part['status'],archive=part['archive'],sha=part['sha'],rows=part['rows'],result=result)
        self.s.state.ledger_put('import_source_archive',digest([self.job,self.key(request)]),{'task_id':self.job,'dataset':request['endpoint'],'endpoint':request['source_id']+':'+request['endpoint'],'request_hash':digest(request['parameters']),'archive_path':part['archive'],'response_sha256':part['sha'],'response_fields':','.join(data['fields']),'fetched_at':meta['fetched_at'],'rows':part['rows']})
        self.s.state.log(self.job,'response_reused',source_id=request['source_id'],endpoint=request['endpoint'],rows=part['rows'],original_slice_id=part['id'])
        return result['children']

    def retry_wait(self,seconds):
        deadline=time.monotonic()+seconds
        while True:
            self.s.state.check_cancel(self.job)
            remaining=deadline-time.monotonic()
            if remaining<=0:return
            time.sleep(min(.2,remaining))

    def fetch_with_retry(self,request,contract,policy,page):
        """Retry exhausted transient fetches; preserve the same request identity."""
        for attempt in range(len(self.RETRY_DELAYS)+1):
            self.s.state.check_cancel(self.job)
            self.s.state.slice(self.job,self.key(request),request['endpoint'],request,status='pending',result={'retry_rounds':attempt+1,'complete':False})
            try:return self.fetch(request,contract,policy,page)
            except InterruptedError:raise
            except Exception as exc:
                kind=getattr(exc,'kind',None)
                retryable=kind in {'network_error','server_error','rate_limited'} or isinstance(exc,(ConnectionError,TimeoutError))
                message=self.s.safe_error(exc)
                if not retryable or attempt==len(self.RETRY_DELAYS):
                    self.s.state.slice(self.job,self.key(request),request['endpoint'],request,status='failed',result={'reason':message,'kind':kind,'complete':False,'retry_rounds':attempt+1})
                    raise
                seconds=self.RETRY_DELAYS[attempt]
                self.s.state.log(self.job,'source_retry_scheduled',source_id=request['source_id'],endpoint=request['endpoint'],request_id=self.key(request),kind=kind,retry_round=attempt+1,max_retry_rounds=len(self.RETRY_DELAYS),wait_seconds=seconds,error=message)
                self.retry_wait(seconds)

    def cached_children(self,old,page=None,request=None):
        if not old or not old.get('archive') or old['status'] not in {'downloaded','paged','split','contract_blocked'}:return None
        try:
            read_archive(Path(old['archive']),old['sha'])
            if not Path(old['archive']).with_suffix('.meta.json').is_file():raise FileNotFoundError()
        except FileNotFoundError:
            self.s.state.log(self.job,'archive_missing_retry',endpoint=old['dataset'],request_id=old['id'])
            return None
        result=old.get('result');result=json.loads(result) if isinstance(result,str) else result or {}
        children=result.get('children',[])
        if old['status']=='paged' and not children and page and request:
            children=[self.following(request,page,old['rows'])]
        return children

    def prepared(self, request):
        sid, endpoint = request['source_id'], request['endpoint']
        contract = self.plan['source_contracts'][sid+':'+endpoint]
        provider = self.s.sources.values[sid]['provider']
        policy = self.plan.get('retrieval_policies', {}).get(sid+':'+endpoint)
        if sid == 'tushare': policy = policy or tushare_policy(endpoint)
        page = dict(contract.get('pagination', {}))
        if policy and policy['mode'] in {'offset', 'offset_probe'}:
            page = {'offset_param': 'offset', 'limit_param': 'limit', 'size': policy['row_limit']}
        if provider == 'file': page = {'offset_param': 'offset', 'limit_param': 'limit', 'size': 5000}
        params = dict(request['parameters'])
        if sid=='tushare' and endpoint=='index_daily' and params.get('ts_code'):
            policy={**(policy or {}),'mode':'partition'};page={}
        if page:
            limit = page.get('limit_param', 'limit')
            size = int(page.get('size', 1000))
            params[limit] = min(int(params.get(limit, size)), size)
            if params[limit] <= 0: raise ValueError('分页大小须为正整数')
            params.setdefault(page.get('offset_param', 'offset'), 0)
            if int(params[page.get('offset_param', 'offset')]) < 0: raise ValueError('分页偏移须非负')
        return {**request, 'parameters': params}, contract, policy, page

    def run(self, roots=None, workers=1):
        if workers>1:return self.run_concurrent(roots,workers)
        queue = list(roots if roots is not None else self.plan['requests'])
        visited = set()
        position = 0
        while position < len(queue):
            request, contract, policy, page = self.prepared(queue[position]); position += 1
            key = self.key(request)
            if key in visited: continue
            visited.add(key)
            self.s.state.check_cancel(self.job)
            old = self.s.state.get_slice(self.job, key)
            children=self.cached_children(old,page,request)
            if children is not None:
                queue.extend(children)
                continue
            if old and old['status'] == 'unavailable': continue
            reused=self.reuse_response(request,contract)
            if reused is not None:queue.extend(reused);continue
            children = self.fetch_with_retry(request, contract, policy, page)
            queue.extend(children)
        if roots is None:
            from .response_validation import record_archive_validation
            return record_archive_validation(self.s,self.s.state.job(self.job))

    def run_concurrent(self,roots,workers):
        queue=list(roots if roots is not None else self.plan['requests']);seen=set();active={};errors=[]
        def retrieve(root):
            request,contract,policy,page=self.prepared(root);key=self.key(request)
            self.s.state.check_cancel(self.job)
            old=self.s.state.get_slice(self.job,key)
            children=self.cached_children(old,page,request)
            if children is not None:return children
            if old and old['status']=='unavailable':return []
            reused=self.reuse_response(request,contract)
            if reused is not None:return reused
            return self.fetch_with_retry(request,contract,policy,page)
        def operation(root):
            wire=self.wire_key(self.prepared(root)[0])
            with self.cache_lock:lock=self.wire_locks.setdefault(wire,threading.RLock())
            with lock:return retrieve(root)
        with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='source') as pool:
            while queue or active:
                self.s.state.check_cancel(self.job)
                while queue and len(active)<workers:
                    root=queue.pop(0);key=self.key(self.prepared(root)[0])
                    if key in seen:continue
                    seen.add(key);active[pool.submit(operation,root)]=root
                if not active:continue
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for future in done:
                    root=active.pop(future)
                    try:queue.extend(future.result() or [])
                    except InterruptedError:raise
                    except Exception as exc:
                        message=self.s.safe_error(exc);request=self.prepared(root)[0]
                        previous=self.s.state.get_slice(self.job,self.key(request))
                        result=json.loads(previous['result']) if previous and previous.get('result') else {}
                        self.s.state.slice(self.job,self.key(request),root['endpoint'],request,status='failed',result={**result,'reason':message,'kind':getattr(exc,'kind',None),'complete':False})
                        errors.append({'endpoint':root['endpoint'],'error':message})
        return errors

    @staticmethod
    def following(request, page, count):
        offset = page.get('offset_param', 'offset')
        return {**request, 'parameters': {**request['parameters'], offset: int(request['parameters'].get(offset, 0))+count}}

    def complete_universe(self,asset,fetch=True):
        with self.universe_lock:return self._complete_universe(asset,fetch)

    def _complete_universe(self, asset,fetch=True):
        if asset in self.universes: return self.universes[asset]
        endpoint,filters=universe_spec(asset)
        if 'tushare:'+endpoint not in self.plan['source_contracts']:
            raise ValueError('缺少完整证券清单合同；请创建新任务以安全拆分查询')
        roots = [{'source_id': 'tushare', 'endpoint': endpoint, 'parameters': p, 'strategy': 'retrieval_dependency', 'dependency_only': True} for p in filters]
        if fetch:self.run(roots)
        parts={p['id']:p for p in self.s.state.job(self.job)['slices']};pending=[self.key(self.prepared(r)[0]) for r in roots];visited=set();codes=set()
        while pending:
            key=pending.pop()
            if key in visited:continue
            visited.add(key);part=parts.get(key)
            if part is None:raise ValueError('证券清单缺少必需分区，不能拆分查询')
            result=part.get('result') or {};pending.extend(self.key(c) for c in result.get('children',[]))
            if part['status'] in {'split'}: continue
            if part['status'] not in {'downloaded', 'paged'} or not part['archive']:
                raise ValueError('证券清单获取不完整，不能用截断清单拆分查询')
            data = read_archive(Path(part['archive']), part['sha'])
            pos = data['fields'].index('ts_code')
            codes.update(row[pos] for row in data['items'] if row[pos])
        if not codes: raise ValueError('源端证券清单为空，无法证明分区完整性')
        self.universes[asset] = sorted(codes)
        self.s.state.ledger_put('source_universe_estimate',universe_hint_key(self.plan,asset),{'asset':asset,'count':len(codes),'non_sw_count':sum(not c.endswith('.SI') for c in codes),'checked_at':now(),'source_task_id':self.job})
        return self.universes[asset]

    def split(self, request, contract):
        endpoint, params = request['endpoint'], request['parameters']
        chunks = partition(endpoint, params, contract)
        if not chunks and 'ts_code' in contract.get('input_fields', []) and not params.get('ts_code') and endpoint not in {'stock_basic', 'fund_basic', 'index_basic'}:
            codes = self.complete_universe('fund' if endpoint.startswith('fund_') else 'stock')
            policy=self.plan.get('retrieval_policies',{}).get('tushare:'+endpoint,{})
            chunks=security_partitions(endpoint,params,codes,policy)
            ceiling=policy.get('max_security_fanout',2000)
            if len(chunks)>ceiling:raise ValueError(f'{endpoint} 将产生 {len(chunks)} 个逐证券请求，超过安全拆分阈值 {ceiling}；需使用已验证的分页、批量代码或显式证券范围')
        return [{**request, 'parameters': p} for p in chunks]

    def repeated(self, request, page, data):
        if not data['items']: return False
        offset, limit = page.get('offset_param', 'offset'), page.get('limit_param', 'limit')
        scope = {k: v for k, v in request['parameters'].items() if k not in {offset, limit}}
        hashes = {digest(dict(zip(data['fields'], values))) for values in data['items']}
        if len(hashes) != len(data['items']): return True
        for part in self.s.state.job(self.job)['slices']:
            r = part['request']
            if part['status'] not in {'downloaded', 'paged'} or not part['archive'] or r['source_id'] != request['source_id'] or r['endpoint'] != request['endpoint']: continue
            if r.get('dependency_only', False) != request.get('dependency_only', False): continue
            if {k: v for k, v in r['parameters'].items() if k not in {offset, limit}} != scope: continue
            if int(r['parameters'].get(offset, 0)) >= int(request['parameters'].get(offset, 0)): continue
            previous = read_archive(Path(part['archive']), part['sha'])
            if hashes.intersection(digest(dict(zip(previous['fields'], values))) for values in previous['items']): return True
        return False

    @staticmethod
    def scope_error(params, data, contract):
        """Verify the filters on which pagination/partition completeness depends."""
        for values in data['items']:
            row = dict(zip(data['fields'], values))
            if params.get('ts_code') and 'ts_code' in row:
                allowed = {v.strip() for v in str(params['ts_code']).split(',')}
                if row['ts_code'] not in allowed: return '源端返回的证券不符合请求过滤条件，无法证明分区完整性'
            for field in ('trade_date', 'nav_date'):
                if params.get(field) and field in row:
                    if not row[field] or parse_date(row[field]) != parse_date(params[field]): return '源端返回的日期不符合请求过滤条件'
            field = contract.get('date_field')
            if field in {'trade_date', 'nav_date', 'cal_date'} and row.get(field):
                value = parse_date(row[field])
                if params.get('start_date') and value < parse_date(params['start_date']): return '源端返回记录早于请求分区'
                if params.get('end_date') and value > parse_date(params['end_date']): return '源端返回记录晚于请求分区'
        return None

    def fetch(self, request, contract, policy, page):
        s, job = self.s, self.job
        sid, endpoint, params = request['source_id'], request['endpoint'], request['parameters']
        key = self.key(request)
        blocked=self.blocked_endpoints.get((sid,endpoint))
        if blocked:
            s.state.slice(job,key,endpoint,request,status='unavailable',result={'reason':blocked['reason'],'kind':blocked['kind'],'complete':False})
            return []
        s.state.slice(job, key, endpoint, request)
        root = (s.settings.archive_dir or s.settings.state_dir/'archive')/job/endpoint
        root.mkdir(parents=True, exist_ok=True)
        archive = root/(key+'.json.gz')
        wire_meta, error = {}, None
        if sid == 'tushare':
            try: data, sha = s.source.fetch(endpoint, params, [f['name'] for f in contract['fields']], archive)
            except Exception as exc:
                message = s.safe_error(exc)
                kind=getattr(exc,'kind',None)
                if kind in {'permission_denied','quota_exhausted'} and self.plan.get('pipeline_version')==2:
                    blocked={'task_id':job,'source_id':sid,'endpoint':endpoint,'reason':message,'kind':kind,'blocked_at':now()}
                    self.blocked_endpoints[(sid,endpoint)]=blocked
                    s.state.ledger_put('pipeline_source_block',job+':'+sid+':'+endpoint,blocked)
                if kind in {'permission_denied','quota_exhausted'} or kind is None and any(word in message.lower() for word in ('权限', '积分', 'permission')):
                    s.state.slice(job, key, endpoint, request, status='unavailable', rows=0, result={'reason': message})
                    s.state.log(job, 'source_unavailable', source_id=sid, endpoint=endpoint, error=message)
                    s.state.ledger_put('meta_source_capability', sid+':'+endpoint, {'task_id': job, 'source_name': sid, 'endpoint': endpoint, 'capability': endpoint, 'status': 'unavailable', 'error': message, 'last_probe_at': now()})
                    return []
                raise
        else:
            adapter = s.sources.adapter(sid)
            source_request = {'source_id': sid, 'endpoint': endpoint, 'parameters': params}
            adapter.plan(source_request)
            try:
                data = adapter.normalize(adapter.fetch(source_request, FetchContext(s.settings, s.state, job)), contract)
            except ValueError as exc:
                error = s.safe_error(exc); data = {'fields': ['response_error'], 'items': [[error]]}
            if getattr(adapter, 'last_wire', None) is not None:
                wire = root/(key+'.response.json.gz')
                from .security import write_private_wire
                write_private_wire(wire,adapter.last_wire)
                wire_meta = {'wire_archive': str(wire), 'wire_sha256': hashlib.sha256(adapter.last_wire).hexdigest()}
            payload = dumps({'code': 0, 'data': data}).encode(); sha = hashlib.sha256(payload).hexdigest()
            with gzip.open(archive.with_suffix('.tmp'), 'wb') as handle: handle.write(payload)
            archive.with_suffix('.tmp').replace(archive)
        stamp = now()
        atomic_json(archive.with_suffix('.meta.json'), {'fetched_at': stamp, 'source_id': sid, 'endpoint': endpoint, 'sha256': sha, 'parameters': params, **wire_meta})
        from .response_validation import validate_response,issue
        validation_errors=validate_response(self.plan,request,contract,data)
        status, result, children = 'downloaded', {'validation_errors':validation_errors}, []
        if validation_errors:
            status='contract_blocked';result['blocked']='；'.join(p['reason'] for p in validation_errors)
        if error:
            status='contract_blocked';result['blocked']=error
            validation_errors.append(issue('response_format',error))
        if sid != 'tushare' and status != 'contract_blocked':
            try:
                validation = adapter.validate_response(data, contract)
                if validation.get('missing') or validation.get('extra'): raise ValueError('源响应字段合同不匹配')
            except (ValueError,TypeError,KeyError) as exc:
                status='contract_blocked';result['blocked']=s.safe_error(exc)
                validation_errors.append(issue('adapter_validation',result['blocked']))
        count = len(data['items']) if isinstance(data,dict) and isinstance(data.get('items'),list) else 0
        cap = policy['row_limit'] if policy else contract.get('limit', 100000)
        result.update(row_limit=cap, retrieval_mode=policy['mode'] if policy else 'offset' if page else 'single', complete=False)
        if status != 'contract_blocked':
            if page:
                # Even a short page may be a hidden server cap. Only an empty page terminates the chain.
                if self.repeated(request, page, data):
                    status = 'contract_blocked'; result['blocked'] = '源接口忽略分页或分页记录重复/交叠，无法证明完整性，已阻止入库'
                    validation_errors.append(issue('pagination_overlap',result['blocked']))
                elif count:
                    status = 'paged'; children = [self.following(request, page, count)]
                else: result.update(complete=True, termination='empty_page')
            elif count >= cap:
                try: children = self.split(request, contract) if sid == 'tushare' else []
                except (ValueError, RuntimeError) as exc: result['blocked'] = s.safe_error(exc)
                if children: status = 'split'
                else:
                    status = 'contract_blocked'; result.setdefault('blocked', '最小查询范围仍达到单次上限，无法证明完整性；需补充查询合同或缩小范围')
                    validation_errors.append(issue('incomplete_partition',result['blocked']))
            else: result.update(complete=True, termination='below_documented_cap')
        result['children'] = children
        s.state.slice(job, key, endpoint, request, status=status, archive=str(archive), sha=sha, rows=count, result=result)
        if status in {'downloaded','paged','split'}:
            with self.cache_lock:self.response_cache[self.wire_key(request)]={'id':key,'request':request,'status':status,'archive':str(archive),'sha':sha,'rows':count,'result':result}
        s.state.log(job, 'response_archived', source_id=sid, endpoint=endpoint, rows=count, row_limit=cap, status=status, next_requests=len(children), sha256=sha)
        for child in children:
            child, _, _, _ = self.prepared(child)
            s.state.slice(job, self.key(child), endpoint, child)
        fields=data.get('fields',[]) if isinstance(data,dict) else []
        names=[field for field in fields if isinstance(field,str)] if isinstance(fields,list) else []
        s.state.ledger_put('import_source_archive', digest([job, key]), {'task_id': job, 'dataset': endpoint, 'endpoint': sid+':'+endpoint, 'request_hash': digest(params), 'archive_path': str(archive), 'response_sha256': sha, 'response_fields': ','.join(names), 'fetched_at': stamp, 'rows': count})
        s.state.check_cancel(job)
        return children
