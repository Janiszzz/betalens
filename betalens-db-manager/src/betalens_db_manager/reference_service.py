"""Managed reference tasks: archive -> audit -> append -> verify -> publish.

This path deliberately does not call the legacy upsert publisher. A local job
status is orchestration information, never authority to expose business data.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import json
import uuid

from .reference_contracts import TABLES, MODULE_TABLE, VERSION, CONTRACT_HASH, COMMON
from .source import read_archive
from .source_registry import source_identity, identity_matches
from .retrieval import request_rate, tushare_policy
from .util import digest, dumps, now, parse_date, atomic_json


def normalize_request(value):
    """Reject ambiguous/unbounded scopes and secret-bearing unexpected options."""
    allowed = {'modules','targets','security_codes','start_date','end_date','exchanges',
               'schemes','levels','check_times','requests_per_minute','parent_task_id','source_id'}
    from .pipeline_options import DEFAULTS,options
    allowed|=set(DEFAULTS)
    if not isinstance(value,dict) or set(value)-allowed:
        raise ValueError('完整性请求包含未登记字段')
    out = deepcopy(value)
    if out.get('pipeline_version')==2:out.update(options(out))
    for key,default,valid in (
        ('modules',list(MODULE_TABLE),set(MODULE_TABLE)),
        ('targets',['postgres','clickhouse'],{'postgres','clickhouse'}),
        ('exchanges',['SSE','SZSE','BSE'],{'SSE','SZSE','BSE'}),
        ('schemes',['SW2021','CITICS'],{'SW2021','SW2014','CITICS'}),
        ('levels',[1,2,3],{1,2,3}),
    ):
        items=out.get(key,default)
        if not isinstance(items,list) or not items or any(not isinstance(v,(str,int)) or isinstance(v,bool) or v not in valid for v in items):
            raise ValueError('无效的范围选项：'+key)
        out[key]=list(dict.fromkeys(items))
    codes=out.get('security_codes',[])
    import re
    if not isinstance(codes,list) or len(codes)>500 or any(not isinstance(c,str) or not re.fullmatch(r'\d{6}\.(SH|SZ|BJ)',c) for c in codes):
        raise ValueError('股票范围须为至多 500 个境内股票代码（含 .SH/.SZ/.BJ）')
    out['security_codes']=list(dict.fromkeys(codes))
    if set(out['modules']) & {'industry','suspension'} and not codes:
        raise ValueError('行业和停牌检查必须指定股票，不能隐式请求全市场')
    try:
        if any(not isinstance(out[k],str) or not re.fullmatch(r'(\d{8}|\d{4}-\d{2}-\d{2})',out[k]) for k in ('start_date','end_date')):
            raise ValueError()
        start,end=parse_date(out['start_date']),parse_date(out['end_date'])
    except (ValueError,KeyError,TypeError):raise ValueError('需要有效的起止日期') from None
    if start>end or (end-start).days>36600:raise ValueError('日期范围无效或超过一百年')
    if (end-start).days*max(1,len(codes))*len(out['modules'])>500000:
        raise ValueError('检查范围过大，请拆分明确范围的任务')
    out.update(start_date=start.isoformat(),end_date=end.isoformat())
    times=out.get('check_times',[])
    from datetime import time
    if not isinstance(times,list) or len(times)>96:raise ValueError('日内检查时刻须为列表，至多 96 项')
    for instant in times:
        try:
            parsed=time.fromisoformat(instant)
            if parsed.tzinfo is not None:raise ValueError()
        except (ValueError,TypeError):raise ValueError('日内时刻须为上海本地时间 HH:MM[:SS]') from None
    out['check_times']=list(dict.fromkeys(times))
    if 'requests_per_minute' in out:out['requests_per_minute']=request_rate(out['requests_per_minute'])
    out.setdefault('source_id','tushare')
    if not isinstance(out['source_id'],str):raise ValueError('来源实例无效')
    return out


class ReferenceWorkflow:
    def __init__(self, service):self.s=service

    def store(self,target):
        from .reference_storage import ReferenceStore
        if target not in {'postgres','clickhouse'}:raise ValueError('目标库无效')
        return ReferenceStore(self.s,target)

    def job(self,job,validate=True):
        item=self.s.state.job(job)
        if item['target']!=self.s.settings.fingerprint:raise ValueError('任务目标配置已变化')
        plan=item['plan']
        if plan.get('revision')!='reference-v1':raise ValueError('不是业务参考表任务')
        if validate:
            if plan['contract_hash']!=CONTRACT_HASH or plan['mapping_version']!=VERSION:
                raise ValueError('参考表结构或转换规则已变化，请创建关联任务重新审计')
            self.s.sources.refresh()
            sid=plan['request']['source_id']
            if not identity_matches(self.s.sources.values.get(sid,{}),plan['source_identity']):
                raise ValueError('查询来源身份已改变，请创建关联任务')
            for key,contract in plan['source_contracts'].items():
                source,endpoint=key.split(':',1)
                if digest(self.s.sources.endpoint(source,endpoint))!=digest(contract):
                    raise ValueError('源字段合同已变化，请重新审计')
        return item

    def plan(self,request):
        request=normalize_request(request);sid=request['source_id']
        if sid not in self.s.sources.values:raise ValueError('未登记订阅源')
        roots=[]
        def add(endpoint,parameters,dependency=False):
            root={'source_id':sid,'endpoint':endpoint,'parameters':parameters,'strategy':'reference_scope'}
            if dependency:root['dependency_only']=True
            if root not in roots:roots.append(root)
        compact=lambda value:value.replace('-','')
        start,end=request['start_date'],request['end_date']
        if set(request['modules']) & {'calendar','suspension'}:
            exchanges=set(request['exchanges']) if 'calendar' in request['modules'] else set()
            if 'suspension' in request['modules']:
                for code in request['security_codes']:
                    exchanges.add({'SH':'SSE','SZ':'SZSE','BJ':'BSE'}[code[-2:]])
            for exchange in sorted(exchanges):
                # Official trade_cal has no native BSE output; the scope retains
                # the explicit reference exchange rather than relabelling it.
                actual='SSE' if exchange=='BSE' else exchange
                add('trade_cal',{'exchange':actual,'start_date':compact(start),'end_date':compact(end)})
        if 'suspension' in request['modules']:
            for code in request['security_codes']:
                for status in ('L','D','P'):
                    add('stock_basic',{'ts_code':code,'list_status':status},True)
                day=parse_date(start)
                while day<=parse_date(end):
                    add('suspend_d',{'ts_code':code,'trade_date':day.strftime('%Y%m%d')})
                    day+=timedelta(days=1)
        if 'industry' in request['modules']:
            for scheme in request['schemes']:
                endpoint='ci_index_member' if scheme=='CITICS' else 'index_member_all'
                for code in request['security_codes']:
                    for current in ('Y','N'):add(endpoint,{'ts_code':code,'is_new':current})
                if scheme!='CITICS':
                    for level in request['levels']:add('index_classify',{'src':scheme,'level':'L'+str(level)},True)
        if request.get('pipeline_version')==2:
            from .pipeline_plan import params_for
            compact_request={**request,'codes':request['security_codes']}
            optimized=[]
            for root in roots:
                if root['endpoint'] in {'trade_cal','suspend_d','stock_basic'}:
                    for params in params_for(root['endpoint'],compact_request,{}):
                        candidate={**root,'parameters':params}
                        if candidate not in optimized:optimized.append(candidate)
                elif root not in optimized:optimized.append(root)
            roots=optimized
        # Distinct SW source revisions cannot be separated by index_member_all
        # (no src filter); evidence explicitly preserves the dictionary match.
        contracts={sid+':'+r['endpoint']:deepcopy(self.s.sources.endpoint(sid,r['endpoint'])) for r in roots}
        policies={key:tushare_policy(key.split(':',1)[1],points=5000 if request.get('pipeline_version')==2 else None) for key in contracts if sid=='tushare'}
        plan={'revision':'reference-v1','request':request,'requests':roots,'source_contracts':contracts,
              'retrieval_policies':policies,'mapping_version':VERSION,'contract_hash':CONTRACT_HASH,
              'source_identity':source_identity(self.s.sources.values[sid]),'parent_task_id':request.get('parent_task_id')}
        if request.get('pipeline_version')==2:
            plan['pipeline_version']=2
            plan['request']['requests_per_minute']=min(request.get('requests_per_minute',450),450)
        item=self.s.state.create_job(plan,self.s.settings.fingerprint)
        self.s.state.log(item['id'],'reference_task_created',request=request,actual_requests=roots)
        return item

    @contextmanager
    def attempt(self,job,operation):
        attempt=uuid.uuid4().hex
        row={'task_id':job,'attempt_id':attempt,'operation':operation,'started_at':now(),'status':'running'}
        self.s.state.ledger_put('reference_attempt',attempt,row)
        try:
            yield attempt
        except BaseException as exc:
            row.update(status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit,InterruptedError)) else 'failed',
                       error=self.s.safe_error(exc),finished_at=now())
            self.s.state.ledger_put('reference_attempt',attempt,row)
            self.s.state.update_job(job,status=row['status'],error=row['error'])
            raise
        else:
            row.update(status='completed',finished_at=now());self.s.state.ledger_put('reference_attempt',attempt,row)

    def download(self,job):
        from .downloader import TaskDownloader
        item=self.job(job)
        if item['status'] in {'completed','rolled_back'}:return item
        if item['status'] in {'interrupted','cancelled'}:
            raise ValueError('中断任务不得直接续写旧版本；请创建关联的新任务并重新审计')
        self.s.state.update_job(job,status='downloading',error=None)
        source=self.s.source;source.active_job=job
        source.requests_per_minute=item['plan']['request'].get('requests_per_minute',self.s.settings.requests_per_minute)
        source.retrieval_policies=item['plan']['retrieval_policies']
        if item['plan'].get('pipeline_version')==2:
            failures=TaskDownloader(self.s,item).run(item['plan']['requests'],workers=item['plan']['request']['download_workers'])
            self.s.state.update_job(job,status='partial' if failures else 'downloaded')
            return self.s.state.job(job)
        with self.attempt(job,'download'):
            for root in item['plan']['requests']:
                self.s.state.check_cancel(job)
                try:TaskDownloader(self.s,item).run([root])
                except InterruptedError:raise
                except Exception as exc:
                    reason=self.s.safe_error(exc)
                    prepared=TaskDownloader(self.s,item).prepared(root)[0]
                    key=TaskDownloader.key(prepared)
                    self.s.state.slice(job,key,root['endpoint'],prepared,status='failed',result={'reason':reason,'complete':False})
                    self.s.state.log(job,'reference_request_failed',endpoint=root['endpoint'],parameters=root['parameters'],error=reason)
            self.s.state.update_job(job,status='downloaded')
        return self.s.state.job(job)

    def bundles(self,item):
        from .downloader import TaskDownloader
        from .response_validation import validate_archives
        parts={p['id']:p for p in item['slices']};bundles=[]
        downloader=TaskDownloader(self.s,item)
        for root in item['plan']['requests']:
            relevant=[];visited=set()
            def visit(request):
                prepared=downloader.prepared(request)[0];key=TaskDownloader.key(prepared)
                if key in visited:return
                visited.add(key);part=parts.get(key)
                if not part:return
                result=part.get('result') or {}
                if isinstance(result,str):result=json.loads(result)
                if part['status']!='split':relevant.append(part)
                for child in result.get('children',[]):visit(child)
            visit(root)
            scoped=deepcopy(item);scoped['plan']['requests']=[root]
            # Response validation walks every supplied slice before walking
            # roots. An unrelated failed endpoint must not poison this module.
            scoped['slices']=[p for p in item['slices'] if p['id'] in visited]
            errors=validate_archives(self.s,scoped)
            unavailable=[p for p in relevant if p['status']=='unavailable']
            complete=not errors and not unavailable and bool(relevant)
            rows=[];stamps=[];hashes=[]
            for part in relevant:
                if not part.get('archive'):continue
                try:
                    data=read_archive(Path(part['archive']),part['sha'])
                    if len(set(data['fields']))!=len(data['fields']):raise ValueError('重复响应字段')
                    if any(len(values)!=len(data['fields']) for values in data['items']):raise ValueError('响应行宽与字段数不符')
                    metadata=json.loads(Path(part['archive']).with_suffix('.meta.json').read_text())
                    stamps.append(metadata['fetched_at']);hashes.append(part['sha'])
                    rows.extend(dict(zip(data['fields'],values)) for values in data['items'])
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    complete=False
                    errors.append({'reason':'归档损坏或元数据缺失：'+self.s.safe_error(exc),'code':'archive_integrity'})
            reasons=[str(e.get('reason',e)) for e in errors]
            reasons.extend((p.get('result') or {}).get('reason','源权限不足') for p in unavailable)
            reasons.extend((p.get('result') or {}).get('reason','请求失败') for p in relevant if p['status']=='failed')
            bundles.append({'endpoint':root['endpoint'],'parameters':root['parameters'],'rows':rows,
                            'source_id':root['source_id'],'source_name':self.s.sources.values[root['source_id']]['provider'],
                            'downloaded_at':max(stamps) if stamps else item['created'],'complete':complete,
                            'reason':'；'.join(reasons),'archive_sha':hashes,'retrieval_errors':errors})
        return bundles

    def preview(self,job):
        from .reference_domains import build_scopes
        item=self.job(job)
        if item['status'] in {'planned','downloading'}:raise ValueError('请先完成拉取')
        if item['status'] in {'interrupted','cancelled'}:raise ValueError('中断任务请建立关联新任务重新审计')
        bundles=self.bundles(item);scopes=build_scopes(item['plan']['request'],bundles)
        if not scopes:raise ValueError('未生成请求范围凭证')
        cache={'scopes':scopes,'bundles':bundles,'contract_hash':CONTRACT_HASH,'mapping_version':VERSION}
        cache_hash=digest(cache)
        path=self.s.settings.state_dir/'reference-cache'/job/'audit.json'
        atomic_json(path,cache)
        target_heads={};checks=[];changes=[]
        usable=[scope for scope in scopes if scope['complete']]
        for target in item['plan']['request']['targets']:
            try:
                store=self.store(target);target_heads[target]=store.heads([s['scope_id'] for s in usable])
                previous=store.snapshot()
                for scope in scopes:
                    fields=[f['name'] for f in TABLES[MODULE_TABLE[scope['module']]].fields if f['name'] not in {f['name'] for f in COMMON}]
                    before=[{k:r.get(k) for k in fields} for r in previous['rows'] if r['scope_id']==scope['scope_id']]
                    after=[{k:r.get(k) for k in fields} for r in scope['rows']]
                    content=lambda rows:sorted(dumps(row) for row in rows)
                    changes.append({'target':target,'module':scope['module'],'scope_id':scope['scope_id'],
                        'selector':scope['selector'],'complete':scope['complete'],'reasons':scope.get('reasons',[]),
                        'before_revision':target_heads[target].get(scope['scope_id']),
                        'operation':'no_business_change' if content(before)==content(after) else 'replace_scope' if target_heads[target].get(scope['scope_id']) else 'new_scope',
                        'before':before,'after':after,'rows':len(after)})
                checks.append({'target':target,'status':'ready','scope_count':len(usable)})
            except Exception as exc:
                checks.append({'target':target,'status':'unavailable','reason':self.s.safe_error(exc)})
        raw=[{'endpoint':b['endpoint'],'parameters':b['parameters'],**r} for b in bundles for r in b['rows']]
        normalized=[{'module':s['module'],'scope_id':s['scope_id'],**r} for s in scopes for r in s['rows']]
        result={'job':job,'cache_path':str(path),'cache_hash':cache_hash,'mapping_version':VERSION,
                'contract_hash':CONTRACT_HASH,'archives':[[p['id'],p['sha'],p['status']] for p in item['slices']],
                'target_heads':target_heads,'targets':item['plan']['request']['targets'],'checks':checks,
                'raw_sample':raw[:200],'normalized_sample':normalized[:200],'changes':changes,
                'source_report':[{k:b[k] for k in ('endpoint','parameters','complete','reason','archive_sha')}|{'rows':len(b['rows'])} for b in bundles],
                'rejected':sum(not s['complete'] for s in scopes),'blocking':not usable,
                'scope_count':len(scopes),'row_count':len(normalized)}
        result['token']=digest(result)
        self.s.state.update_job(job,status='blocked' if result['blocking'] else 'previewed',preview=result)
        self.s.state.ledger_put('reference_audit',job,{'task_id':job,'cache_hash':cache_hash,'scopes':scopes,'created_at':now()})
        self.s.state.log(job,'reference_audit_ready',scope_count=len(scopes),row_count=len(normalized),rejected=result['rejected'])
        return result

    def _confirmed(self,job,token):
        item=self.job(job);preview=item.get('preview')
        if not preview or preview.get('token')!=token or digest({k:v for k,v in preview.items() if k!='token'})!=token:
            raise ValueError('确认令牌无效，请重新审计')
        if item['status'] in {'interrupted','cancelled','rolled_back'}:raise ValueError('该任务不可继续写入，请创建关联新任务')
        if preview.get('blocking'):raise ValueError('没有可发布的完整请求范围')
        path=self.s.settings.state_dir/'reference-cache'/job/'audit.json'
        cache=json.loads(path.read_text())
        if digest(cache)!=preview['cache_hash']:raise ValueError('审计缓存已变化，请重新审计')
        if preview['archives']!=[[p['id'],p['sha'],p['status']] for p in item['slices']]:raise ValueError('下载归档已变化，请重新审计')
        # Re-read compressed bytes too, rather than trusting stored SHA labels.
        if digest(self.bundles(item))!=digest(cache['bundles']):raise ValueError('归档或下载证明已变化，请重新审计')
        return item,cache

    def commit(self,job,token,accept_partial=False):
        # Each target operation owns the existing publication/write lock. Do
        # not nest a second flock on the same path in this orchestration layer.
        with nullcontext():
            item,cache=self._confirmed(job,token)
            if item['preview']['rejected'] and not accept_partial:raise ValueError('存在不完整范围，需明确确认仅发布完整范围')
            self.s.state.update_job(job,status='committing',error=None)
            results=[]
            with self.attempt(job,'commit') as attempt:
                for target in item['plan']['request']['targets']:
                    for module in item['plan']['request']['modules']:
                        scopes=[s for s in cache['scopes'] if s['module']==module and s['complete']]
                        key=job+':'+target+':'+module
                        row={'task_id':job,'target':target,'module':module,'status':'not_published','scope_count':len(scopes)}
                        try:
                            self.s.state.check_cancel(job)
                            if not scopes:
                                row.update(status='incomplete',reason='没有完整请求范围；未发布')
                                results.append(row);self.s.state.ledger_put('reference_result',key,row);continue
                            saved=next((r for r in self.s.state.ledger_rows('reference_revision',job) if r['target']==target and r['module']==module),None)
                            revision=saved or {'revision_id':uuid.uuid4().hex,'import_task_id':job,'task_id':job,
                                'attempt_id':attempt,'target':target,'module':module,'mapping_version':VERSION,
                                'source_name':self.s.sources.values[item['plan']['request']['source_id']]['provider'],
                                'source_id':item['plan']['request']['source_id'],
                                'downloaded_at':max(b['downloaded_at'] for b in cache['bundles']),
                                'request':item['plan']['request'],'cache_hash':item['preview']['cache_hash']}
                            self.s.state.ledger_put('reference_revision',key,revision)
                            store=self.store(target)
                            expected={s['scope_id']:item['preview']['target_heads'].get(target,{}).get(s['scope_id']) for s in scopes}
                            # The store also performs this check under its target lock.
                            current=store.heads(list(expected))
                            if any(current.get(k)!=v and current.get(k)!=revision['revision_id'] for k,v in expected.items()):
                                raise ValueError('目标范围已有更新版本，请重新审计')
                            store.stage(revision,scopes)
                            verified=store.verify(revision['revision_id'])
                            if not verified.get('verified'):raise ValueError('目标库追加版本核验失败：'+dumps(verified))
                            published=store.publish(revision['revision_id'],expected)
                            self.s.state.invalidate_validation_reports([MODULE_TABLE[module]],[target])
                            row.update(status='published',revision_id=revision['revision_id'],verification=verified,publication=published)
                            self.s.state.log(job,'reference_module_published',target=target,module=module,revision_id=revision['revision_id'])
                        except InterruptedError:raise
                        except Exception as exc:
                            row.update(status='failed',reason=self.s.safe_error(exc))
                            self.s.state.log(job,'reference_publication_failed',target=target,module=module,error=row['reason'])
                        results.append(row);self.s.state.ledger_put('reference_result',key,row)
            status='completed' if all(r['status']=='published' for r in results) and not item['preview']['rejected'] else 'partial'
            self.s.state.ledger_put('reference_task_result',job,{'task_id':job,'status':status,'results':results,'finished_at':now()})
            self.s.state.update_job(job,status=status)
            return self.detail(job)

    def rollback(self,job,targets=None):
        item=self.job(job)
        targets=targets or item['plan']['request']['targets']
        if not isinstance(targets,list) or set(targets)-set(item['plan']['request']['targets']):raise ValueError('撤销目标库不属于原任务')
        rollback=self.s.state.create_job({'revision':'reference-rollback-v1','source_task_id':job,'request':{'targets':targets}},self.s.settings.fingerprint)
        results=[]
        with self.attempt(rollback['id'],'rollback'):
            for revision in self.s.state.ledger_rows('reference_revision',job):
                if revision['target'] not in targets:continue
                try:
                    result=self.store(revision['target']).revoke(revision['revision_id'],rollback['id'])
                    self.s.state.invalidate_validation_reports([MODULE_TABLE[revision['module']]],[revision['target']])
                except Exception as exc:result={'status':'failed','reason':self.s.safe_error(exc)}
                result.update(target=revision['target'],module=revision['module'],revision_id=revision['revision_id'])
                results.append(result)
        status='rolled_back' if results and all(r['status']=='revoked' for r in results) else 'partially_rolled_back'
        self.s.state.update_job(rollback['id'],status='completed' if status=='rolled_back' else 'partial')
        self.s.state.ledger_put('reference_rollback',rollback['id'],{'task_id':job,'rollback_task_id':rollback['id'],'results':results,'created_at':now()})
        # Partial target revoke must not mislabel remaining published targets.
        self.s.state.update_job(job,status=status if set(targets)==set(item['plan']['request']['targets']) else 'partially_rolled_back')
        return {'rollback_task_id':rollback['id'],'status':status,'results':results}

    def retry(self,job):
        item=self.job(job);request={**item['plan']['request'],'parent_task_id':job}
        return self.plan(request)

    def baseline(self,request,source_task_ids):
        """Borrow verifiable archives, never old business rows or task statuses.

        Exact requests and output contracts must match. Missing evidence remains
        an explicit reacquisition range; the caller still audits and confirms.
        Archives are copied so legacy cache retention cannot remove the proof.
        """
        from .downloader import TaskDownloader
        import shutil
        if not isinstance(source_task_ids,list) or not source_task_ids or len(source_task_ids)>100 or any(not isinstance(v,str) for v in source_task_ids):
            raise ValueError('基线任务需要 1 至 100 个源任务编号')
        originals=[self.s.state.job(v) for v in dict.fromkeys(source_task_ids)]
        item=self.plan(request);job=item['id'];downloader=TaskDownloader(self.s,item)
        report=[]
        for root in item['plan']['requests']:
            prepared,contract,_,_=downloader.prepared(root)
            selected=None
            for original in originals:
                old_identity=original['plan'].get('source_instances',{}).get(root['source_id']) or original['plan'].get('source_identity')
                if not old_identity or not identity_matches(self.s.sources.values[root['source_id']],old_identity):continue
                old_contract=original['plan'].get('source_contracts',{}).get(root['source_id']+':'+root['endpoint'])
                if old_contract is None or digest(old_contract)!=digest(contract):continue
                for part in original['slices']:
                    r=part['request']
                    if (r.get('source_id'),r.get('endpoint'),r.get('parameters'))==(prepared['source_id'],prepared['endpoint'],prepared['parameters']):
                        selected=(original,part);break
                if selected:break
            note={'endpoint':root['endpoint'],'parameters':root['parameters'],'status':'needs_reacquisition'}
            if selected:
                original,first=selected
                parts={TaskDownloader.key(p['request']):p for p in original['slices']}
                copied=set()
                def copy_part(part):
                    old_request=part['request'];adapted={**old_request}
                    if root.get('dependency_only'):adapted['dependency_only']=True
                    else:adapted.pop('dependency_only',None)
                    key=TaskDownloader.key(adapted)
                    if key in copied:return
                    copied.add(key)
                    if part['status'] not in {'downloaded','paged','split'} or not part.get('archive'):
                        raise ValueError('归档查询链未完整下载')
                    read_archive(Path(part['archive']),part['sha'])
                    metadata=json.loads(Path(part['archive']).with_suffix('.meta.json').read_text())
                    if not metadata.get('fetched_at'):raise ValueError('原归档缺少真实下载时间')
                    dest=(self.s.settings.archive_dir or self.s.settings.state_dir/'archive')/job/root['endpoint']/(key+'.json.gz')
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(part['archive'],dest)
                    # No credential-bearing raw HTTP envelope is copied into a
                    # public task payload; preserve only the verified data wire.
                    atomic_json(dest.with_suffix('.meta.json'),{'fetched_at':metadata['fetched_at'],
                        'source_id':root['source_id'],'endpoint':root['endpoint'],'sha256':part['sha'],
                        'parameters':adapted['parameters'],'baseline_source_task_id':original['id'],
                        'baseline_source_slice_id':part['id']})
                    result=deepcopy(part.get('result') or {})
                    children=[]
                    for old_child in result.get('children',[]):
                        child={**old_child}
                        if root.get('dependency_only'):child['dependency_only']=True
                        else:child.pop('dependency_only',None)
                        children.append(child)
                    result['children']=children
                    self.s.state.slice(job,key,root['endpoint'],adapted,status=part['status'],archive=str(dest),sha=part['sha'],rows=part['rows'],result=result)
                    for old_child in (part.get('result') or {}).get('children',[]):
                        child_part=parts.get(TaskDownloader.key(old_child))
                        if child_part is None:raise ValueError('原查询链缺少分页或分区')
                        copy_part(child_part)
                try:
                    copy_part(first)
                    # A second independent validation recomputes chain coverage,
                    # caps, filters, row widths and field sets from raw archives.
                    current=self.s.state.job(job)
                    evidence=next(b for b in self.bundles(current) if b['endpoint']==root['endpoint'] and b['parameters']==root['parameters'])
                    if not evidence['complete']:raise ValueError(evidence['reason'] or '范围证据不完整')
                    note.update(status='archive_verified',source_task_id=original['id'],copied_slices=len(copied))
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    note['reason']=self.s.safe_error(exc)
                    # Reacquisition must not silently reuse the rejected chain.
                    with self.s.state.connect() as conn:
                        for key in copied:conn.execute('DELETE FROM slices WHERE job=? AND id=?',(job,key))
            report.append(note)
        item['plan'].update(baseline_source_task_ids=list(dict.fromkeys(source_task_ids)),baseline_report=report)
        with self.s.state.connect() as conn:conn.execute('UPDATE jobs SET plan=? WHERE id=?',(dumps(item['plan']),job))
        self.s.state.update_job(job,status='downloaded')
        self.s.state.log(job,'reference_baseline_prepared',sources=source_task_ids,report=report,automatic_publication=False)
        return self.s.state.job(job)

    def detail(self,job):
        item=self.job(job,False)
        # State.logs already expands each JSON payload into named columns. Do
        # not also render the duplicate serialized `value` blob in the UI.
        item['logs']=[{k:v for k,v in row.items() if k!='value'} for row in self.s.state.logs(job,limit=1000)]
        attempts=self.s.state.ledger_rows('reference_attempt',job)
        # A worker killed before its finally block cannot certify its own exit.
        if item['status'] in {'interrupted','cancelled'}:
            attempts=[{**a,'status':'interrupted' if a['status']=='running' else a['status']} for a in attempts]
        item['attempts']=attempts
        item['result']=next(iter(self.s.state.ledger_rows('reference_task_result',job)),None)
        item['rollbacks']=self.s.state.ledger_rows('reference_rollback',job)
        return item

    def check(self,request):
        from .reference_domains import check_integrity,build_scopes
        request=normalize_request(request);results={}
        # Local execution history explains missing/unpublished ranges but never
        # grants visibility. Target-local publication remains the authority.
        with self.s.state.connect() as conn:
            jobs=[json.loads(r[1])|{'id':r[0],'status':r[2]} for r in conn.execute(
                "SELECT id,plan,status FROM jobs WHERE target=? AND json_extract(plan,'$.revision')='reference-v1' ORDER BY created DESC LIMIT 1000",(self.s.settings.fingerprint,))]
        histories={}
        for job in jobs:
            task_request=job['request']
            for scope in build_scopes(task_request,[]):
                for target in task_request['targets']:
                    histories.setdefault((target,scope['scope_id']),[]).append({'task_id':job['id'],'status':job['status']})
        for target in request['targets']:
            try:
                snapshot=self.store(target).snapshot(audit=True)
                result=check_integrity(request,snapshot)
                for row in result['rows']:
                    selector={k:row[k] for k in ({'calendar':('exchange','trade_date'),'industry':('security_code','scheme_code','level'),'suspension':('security_code','trade_date')}[row['module']])}
                    sid=digest({'module':row['module'],'selector':selector})
                    history=histories.get((target,sid),[])
                    if history:row['task_history']=history
                    if row.get('reason') in {'not_requested','unpublished'} and history:
                        state=history[0]['status']
                        row['reason']=('interrupted' if state in {'interrupted','cancelled'} else
                                       'request_failed' if state=='failed' else
                                       'pending_confirmation' if state in {'previewed','downloaded','blocked'} else
                                       'unpublished' if state in {'committing','partial','completed'} else 'not_requested')
                        row['latest_task_id']=history[0]['task_id']
                        audit=self.s.state.ledger_get('reference_audit',history[0]['task_id']) or {}
                        failed=next((s for s in audit.get('scopes',[]) if s['scope_id']==sid and not s['complete']),None)
                        if failed:
                            queries=failed.get('evidence',{}).get('queries',[])
                            messages='；'.join(str(q.get('reason') or '') for q in queries)
                            row.update(reason='source_permission_denied' if any(w in messages for w in ('权限','积分','permission')) else 'request_failed',
                                       unknown_reasons=failed.get('reasons',[]),request_errors=[q for q in queries if not q.get('complete')])
                from collections import Counter
                result['summary']['reasons']=dict(Counter(row['reason'] for row in result['rows']))
                results[target]=result
            except Exception as exc:
                result=check_integrity(request,{'revisions':[],'scopes':[],'events':[],'rows':[]})
                for row in result['rows']:row.update(reason='target_unavailable',target_error=self.s.safe_error(exc))
                results[target]={**result,'complete':False,'status':'unavailable','reason':self.s.safe_error(exc)}
        return {'request':request,'checked_at':now(),'results':results}
