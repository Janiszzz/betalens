"""Standalone database management facade; adapters never write business tables."""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from datetime import date,datetime,timedelta,timezone
from decimal import Decimal
from functools import wraps
from pathlib import Path
import gzip,hashlib,json,uuid
import yaml
from .service import TushareImportService,normalize_source,dates,SHANGHAI
from .canonical import canonical,load_canonical_catalog,registry_context,builtin_catalog
from .registry import TableRegistry,GENERATED
from .schema_manager import SchemaManager
from .mapping_store import MappingStore,DETAILS
from .source_registry import SourceRegistry,FetchContext,source_identity,identity_matches
from .source import read_archive
from .retrieval import request_rate,tushare_policy
from .comparison import typed,row_hash,row_key
from .config import Settings
from .util import digest,dumps,now,parse_date,atomic_json
from .processors import resolve as resolve_processor, validate_output, VERSION as PROCESSOR_VERSION
from .response_validation import validate_archives, scope_roles
from .time_scope import row_in_scope


def scoped(cls):
    # A registry is request/execution-local, so independent manager instances and
    # tests can safely use different catalogs in the same process.
    for name in dir(cls):
        if name.startswith('__'):continue
        method=getattr(cls,name)
        if not callable(method):continue
        def wrap(method):
            @wraps(method)
            def wrapped(self,*args,**kwargs):
                with registry_context(self.registry):return method(self,*args,**kwargs)
            return wrapped
        setattr(cls,name,wrap(method))
    return cls


@scoped
class DatabaseManagerService(TushareImportService):
    def __init__(self,settings=None,source=None):
        super().__init__(settings,source)
        self.registry=TableRegistry(self.settings,self.state)
        self.sources=SourceRegistry(self.settings,self.state)
        with registry_context(self.registry):self.mappings=MappingStore(self.registry,self.sources,self.state)
        self.schema=SchemaManager(self)

    def safe_error(self,exc):
        from .security import safe_error
        return safe_error(exc,self.settings,getattr(getattr(self,'sources',None),'values',{}).values())

    def initialize(self,token,targets=None):
        from .publishing import operation_lock
        with operation_lock(self):return super().initialize(token,targets)

    def reload_config(self):
        from .storage import Postgres
        from .clickhouse import ClickHouse
        from .sources import TushareSourceAdapter
        new=Settings.from_env(self.settings.env_file)
        # Existing state/archive locations remain stable unless explicitly changed.
        if new.state_dir!=self.settings.state_dir or new.config_dir!=self.settings.config_dir:raise ValueError('环境目录变更需要重启服务；当前任务和归档保留')
        self.settings=new;self.pg=Postgres(new);self.ch=ClickHouse(new)
        self.source=TushareSourceAdapter(new,self.state)
        self.sources.settings=new;self.sources.refresh()
        return new.public()

    def _job(self,job,validate=False):
        item=self.state.job(job)
        if item['target']!=self.settings.fingerprint:raise ValueError('任务属于其他数据库配置，请使用原目标或创建新任务')
        if item['plan'].get('revision')=='reference-v1':
            from .reference_service import ReferenceWorkflow
            return ReferenceWorkflow(self).job(job,validate)
        if item['plan'].get('revision')!='manager-v1':return super()._job(job,validate)
        if validate:
            if item['plan'].get('processor_version') and item['plan']['processor_version']!=PROCESSOR_VERSION:raise ValueError('生产处理器已变化，请重审归档并创建关联任务')
            self.sources.refresh()
            for sid,expected in item['plan'].get('source_instances',{}).items():
                if not identity_matches(self.sources.values.get(sid,{}),expected):raise ValueError('来源身份或查询地址已变化，请创建新任务')
            for name,version in item['plan']['canonical_contracts'].items():
                if canonical(name).hash!=version:raise ValueError('本地表定义已变化，请重新审计并创建关联任务')
            for key,contract in item['plan']['source_contracts'].items():
                sid,endpoint=key.split(':',1)
                if digest(self.sources.endpoint(sid,endpoint))!=digest(contract):raise ValueError('来源字段合同已变化，请创建关联任务')
        return item

    def _canonical_record(self,*args,**kwargs):
        row=super()._canonical_record(*args,**kwargs)
        spec=args[0] if args else kwargs['spec']
        row.update(schema_version=spec.hash,source_id=row.get('source_name') or 'tushare')
        return row

    def _map_part(self,endpoint,source_rows,job,downloaded,calendar,source_name='tushare'):
        output,factors=super()._map_part(endpoint,source_rows,job,downloaded,calendar,source_name)
        if output.get('market_daily_raw'):
            from .mapping import FIELD_MAP
            by_row={s['_row']:s for s in source_rows}
            for row in output['market_daily_raw']:
                source=by_row.get(row.get('source_row_no'),{});origins=row.setdefault('_field_origins',{})
                for field,destination in FIELD_MAP.get(endpoint,{}).items():
                    column=destination[1];value=source.get(field)
                    if row.get(column)==value:
                        origins[column]={'selected':value is not None,'source_null':value is None,'value_hash':digest(value),'origins':[source_name+':'+endpoint+':'+field]}
        item=self.state.job(job);snapshots=item['plan'].get('mapping_snapshots') or self.mappings.snapshots()
        for name,mapping in snapshots.items():
            if name not in load_canonical_catalog():continue
            bindings=[b for b in mapping.get('bindings',[]) if b['source_id']=='tushare' and b['endpoint']==endpoint and resolve_processor(name,b) is not None]
            if not bindings:continue
            spec=canonical(name)
            for binding in bindings:
                for source in source_rows:
                    raw=source.get('_raw',{});partial,evidence=self.mappings.map_row(spec,raw,binding,processor=True)
                    protected=set()
                    if name=='corporate_action':protected={'event_date','effective_date','cash_dividend','share_dividend','old_name','event_type'}
                    if name=='fundamental_periodic':protected={'announcement_date','actual_announcement_date','end_type','statement_type','version_id','rd_exp','research_and_development_expense','research_and_development_investment'}
                    partial={k:v for k,v in partial.items() if k not in protected}
                    if name in DETAILS.values():
                        if name=='industry_membership_record':partial['scheme_code']='SW2021' if endpoint=='index_member_all' else 'CITIC'
                        partial['record_version']=digest(partial)
                        event=next((source.get(k) for k in ('trade_date','end_date','start_date','nav_date','in_date') if source.get(k)),None)
                        available=None;basis='unknown'
                        if source.get('f_ann_date') or source.get('ann_date'):
                            from .service import _event_available
                            available,basis=_event_available(source.get('f_ann_date') or source.get('ann_date'),'announcement_next_calendar_day_assumed')
                        row=self._canonical_record(spec,partial,endpoint,job,downloaded,source['_row'],raw,event,available,basis,source_name)
                        row['mapping_version']=mapping.get('version',digest(mapping));output[name].append(row)
                    else:
                        # Enrich computed canonical rows by the explicit source key.
                        for row in output.get(name,[]):
                            if row.get('source_row_no')==source['_row']:
                                row.update(partial);row['mapping_version']=mapping.get('version',digest(mapping))
                    for row in output.get(name,[]):
                        if row.get('source_row_no')!=source['_row']:continue
                        origins=row.setdefault('_field_origins',{})
                        for field in evidence:
                            c=field['column']
                            if field['selected'] and row.get(c)==field['converted_value']:
                                origins[c]={'selected':True,'value_hash':digest(row[c]),'origins':[binding['source_id']+':'+endpoint+':'+field['source_field']]}
                            elif row.get(c) is None and field['raw_value'] in (None,''):
                                origins.setdefault(c,{'selected':False,'source_null':True,'origins':[]})
                        if name=='fundamental_periodic':
                            row['version_id']=digest({'source_business_version':row['version_id'],'mapped_business':{f['name']:row.get(f['name']) for f in spec.fields if f['name'] not in GENERATED|{'version_id','event_date','available_at','availability_basis'}}})
                    if evidence and (source['_row']<=20 or any(f.get('conflict') for f in evidence)):self.state.ledger_put('field_evidence',digest([job,name,endpoint,source['_row']]),{'task_id':job,'table':name,'source_row':source['_row'],'fields':evidence})
        return output,factors

    def _merge(self,rows,spec):
        result=super()._merge(rows,spec)
        by={row_key(spec,r):r for r in result}
        for original in rows:
            current=by[row_key(spec,original)];origins=current.setdefault('_field_origins',{})
            for column,proof in original.get('_field_origins',{}).items():
                if proof.get('value_hash')==digest(current.get(column)):origins[column]=proof
        return result

    def compile(self,request):
        self.registry.refresh();self.sources.refresh();self.mappings.refresh()
        from .security import validate_parameters
        validate_parameters(request.get('parameters',{}),path='request.parameters')
        rpm=request_rate(request.get('requests_per_minute',self.settings.requests_per_minute))
        names=request.get('tables') or ([request['table']] if request.get('table') else [])
        if not names:raise ValueError('请选择本地表')
        targets=request.get('targets') or ['postgres','clickhouse']
        if not targets or set(targets)-{'postgres','clickhouse'}:raise ValueError('目标库无效')
        start=parse_date(request['start_date']) if request.get('start_date') else None
        end=parse_date(request.get('end_date') or (datetime.now(SHANGHAI).date()-timedelta(days=1)))
        if start and start>end:raise ValueError('起始日期晚于结束日期')
        assets=request.get('asset_types',['stock','fund','index'])
        if not assets or set(assets)-{'stock','fund','index'}:raise ValueError('资产范围无效')
        requested_endpoints=request.get('endpoints')
        if requested_endpoints is not None and (not isinstance(requested_endpoints,list) or not requested_endpoints or not all(isinstance(e,str) for e in requested_endpoints) or len(set(requested_endpoints))!=len(requested_endpoints)):raise ValueError('endpoints 必须是非空且不重复的接口名列表')
        snapshots=self.mappings.snapshots(names);contracts={};bindings=[];warnings=[]
        for name in names:
            if name in self.registry.errors or name in self.mappings.errors:raise ValueError('当前表或映射 YAML 无效，请先修复：'+name)
            definition=self.registry.get(name)
            if self.registry.record(name).get('status')=='archived':raise ValueError('表已归档')
            if definition.get('time_field') and start is None:raise ValueError('时序表需要显式起始日期')
            selected=[b for b in snapshots[name].get('bindings',[]) if (requested_endpoints is None or b['endpoint'] in requested_endpoints) and (b['source_id']==request['source_id'] if request.get('source_id') else b['source_id'] in request['sources'] if request.get('sources') else True) and (not b.get('scope',{}).get('asset_type') or b['scope']['asset_type'] in assets)]
            if not selected:raise ValueError(name+' 尚未绑定可用查询源')
            snapshots[name]=self.mappings.validate({**snapshots[name],'bindings':selected})
            problems=[r for r in self.mappings.diagnostics(snapshots[name],selected) if r['severity']=='error']
            if problems:raise ValueError('字段映射尚不可执行：'+'；'.join(r['endpoint']+' '+r['field']+' '+r['message'] for r in problems[:8]))
            for binding in selected:
                sid,endpoint=binding['source_id'],binding['endpoint'];contract=self.sources.endpoint(sid,endpoint)
                contracts[sid+':'+endpoint]=contract
                strategy=contract.get('strategy','snapshot');params=dict(contract.get('parameters',{}));params.update(binding.get('parameters',{}));params.update(request.get('parameters',{}).get(endpoint,{}))
                validate_parameters(params,auth=self.sources.values[sid].get('auth'))
                if sid=='tushare':
                    normalized={'start_date':str(start or date(1900,1,1)),'end_date':str(end),'codes':request.get('codes',[]),'parameters':{endpoint:params}}
                    universe=[]
                    if not normalized['codes'] and strategy in {'security','index'}:
                        universe=self._universe({'slices':[],'plan':{'request':{'targets':targets}}},'fund' if endpoint.startswith('fund_') else 'index' if endpoint in {'index_daily','sw_daily','index_weight'} else 'stock')
                    requests=list(self._requests(endpoint,normalized,universe,[]))
                    if endpoint.endswith('_vip') and normalized['codes']:requests=[{**r,'ts_code':code} for r in requests for code in normalized['codes']]
                else:
                    pb={**contract.get('parameter_bindings',{}),**binding.get('request_bindings',{})}
                    if strategy!='snapshot':
                        if start:params[pb.get('start_date','start_date')]=start.strftime(pb.get('date_format',contract.get('date_format','%Y-%m-%d')))
                        params[pb.get('end_date','end_date')]=end.strftime(pb.get('date_format',contract.get('date_format','%Y-%m-%d')))
                    if strategy=='daily':
                        requests=[{**params,pb.get('trade_date','trade_date'):d.strftime(pb.get('date_format',contract.get('date_format','%Y-%m-%d')))} for d in dates(start,end)]
                    elif strategy=='snapshot':requests=[dict(params)]
                    else:requests=[params]
                    if request.get('codes'):
                        if pb.get('security_code'):requests=[{**p,pb['security_code']:code} for p in requests for code in request['codes']]
                        elif not any(f.get('column')=='security_code' for f in binding.get('fields',[])):raise ValueError('该查询未登记证券参数或标准证券字段映射，无法安全应用证券范围')
                if strategy=='snapshot' or not contract.get('date_field'):warnings.append({'table':name,'endpoint':endpoint,'note':'源查询为快照；日期限制不代表历史快照可恢复'})
                if endpoint=='namechange':warnings.append({'table':name,'endpoint':endpoint,'note':'源日期过滤为公告日期；默认拉取单证券全部名称历史，再按本地生效区间/事件日期筛选'})
                for params in requests:
                    validate_parameters(params,auth=self.sources.values[sid].get('auth'))
                    bindings.append({'table':name,'source_id':sid,'endpoint':endpoint,'parameters':params,'priority':binding.get('priority',0),'strategy':strategy})
        if requested_endpoints is not None:
            missing=set(requested_endpoints)-{b['endpoint'] for m in snapshots.values() for b in m['bindings']}
            if missing:raise ValueError('选择的接口未绑定到任务本地表：'+','.join(sorted(missing)))
        extra=[]
        # Fetch dependencies without adding unselected destination tables.
        if 'market_daily_post_adjusted' in names and any(b['source_id']=='tushare' for b in bindings):
            if 'stock' in assets:
                for d in dates(start,end):
                    for code in request.get('codes',[]) or [None]:
                        extra.append(('daily_basic',{'trade_date':d.strftime('%Y%m%d'),**({'ts_code':code} if code else {})}))
            for endpoint,asset in (('stock_basic','stock'),('fund_basic','fund'),('index_basic','index')):
                if asset not in assets:continue
                normalized_dep={'start_date':str(start or end),'end_date':str(end),'codes':request.get('codes',[]),'parameters':{}}
                extra.extend((endpoint,p) for p in self._requests(endpoint,normalized_dep,[],[]))
        endpoints={b['endpoint'] for b in bindings if b['source_id']=='tushare'}
        if endpoints & {'daily','fund_daily','index_daily','sw_daily','suspend_d'}:
            for exchange in ('SSE','SZSE','BSE'):
                extra.append(('trade_cal',{'exchange':exchange,'start_date':str(start or end).replace('-',''),'end_date':str(end).replace('-','')}))
        if 'trade_status_daily' in names and 'tushare' in {b['source_id'] for b in bindings}:
            if 'stock' in assets:
                for d in dates(start,end):
                    for code in request.get('codes',[]) or [None]:
                        if 'daily' not in endpoints:extra.append(('daily',{'trade_date':d.strftime('%Y%m%d'),**({'ts_code':code} if code else {})}))
        for endpoint,params in extra:
            contracts['tushare:'+endpoint]=self.sources.endpoint('tushare',endpoint)
            bindings.append({'table':names[0],'source_id':'tushare','endpoint':endpoint,'parameters':params,'priority':0,'strategy':'dependency'})
        bindings=list({digest([b['source_id'],b['endpoint'],b['parameters']]):b for b in bindings}.values())
        normalized={'tables':names,'source_id':request.get('source_id'),'asset_types':assets,'start_date':str(start or date(1900,1,1)),'end_date':str(end),'codes':request.get('codes',[]),'targets':targets,'mode':request.get('mode','fill_missing'),'sources':sorted({b['source_id'] for b in bindings}),'parameters':request.get('parameters',{}),'requests_per_minute':rpm}
        if requested_endpoints is not None:normalized['endpoints']=requested_endpoints
        if normalized['mode'] not in {'fill_missing','refresh_source'}:raise ValueError('写入策略无效')
        # Pin retrieval rules independently from field contracts, including complete-universe dependencies.
        for endpoint in list({b['endpoint'] for b in bindings if b['source_id']=='tushare'}):
            contract=contracts['tushare:'+endpoint]
            if 'ts_code' in contract.get('input_fields',[]) and endpoint not in {'stock_basic','fund_basic','index_basic'}:
                dimension='fund_basic' if endpoint.startswith('fund_') else 'stock_basic'
                contracts['tushare:'+dimension]=self.sources.endpoint('tushare',dimension)
        retrieval={key:{**tushare_policy(key.split(':',1)[1]),'requests_per_minute':rpm,'effective_rpm':min(rpm,tushare_policy(key.split(':',1)[1]).get('endpoint_rpm') or rpm)} for key in contracts if key.startswith('tushare:')}
        for binding in bindings:
            binding['validation_scope']=scope_roles({'mapping_snapshots':snapshots},binding,contracts[binding['source_id']+':'+binding['endpoint']])
        return {'processor_version':PROCESSOR_VERSION,'retrieval_policies':retrieval,'request':normalized,'requests':bindings,'mapping_snapshots':snapshots,'source_contracts':contracts,'contracts':{},'canonical_contracts':{n:canonical(n).hash for n in names},'mapping_version':digest(snapshots),'target':self.settings.fingerprint,'revision':'manager-v1','source_instance_snapshots':{sid:{**{k:self.sources.values[sid].get(k) for k in ('source_id','provider','label','base_url','auth')},'base_url':self.sources.values[sid].get('base_url') or ('https://api.tushare.pro' if sid=='tushare' else None)} for sid in {b['source_id'] for b in bindings}},'source_instances':{sid:source_identity(self.sources.values[sid]) for sid in {b['source_id'] for b in bindings}},'warnings':warnings,'parent_task_id':request.get('parent_task_id')}

    def plan(self,request):
        if request.get('modules'):
            from .reference_service import ReferenceWorkflow
            return ReferenceWorkflow(self).plan(request)
        if request.get('tables') or request.get('table'):
            plan=self.compile(request);item=self.state.create_job(plan,self.settings.fingerprint)
        else:
            item=super().plan(request);plan=item['plan'];plan['mapping_snapshots']=self.mappings.snapshots()
            with self.state.connect() as conn:conn.execute('UPDATE jobs SET plan=? WHERE id=?',(dumps(plan),item['id']))
            item=self.state.job(item['id'])
        self._record_mapping_plan(item)
        self.state.log(item['id'],'task_created',request=item['plan']['request'])
        return item

    def _record_mapping_plan(self,item):
        for table,mapping in item['plan'].get('mapping_snapshots',{}).items():
            for b in mapping.get('bindings',[]):
                fields={f['name']:f for f in self.sources.endpoint(b['source_id'],b['endpoint'])['fields']}
                targets={f['name']:f for f in self.registry.spec(table).fields}
                for rule in b.get('fields',[]):
                    f=fields[rule['source_field']];target=targets[rule['column']]
                    row={'task_id':item['id'],'source_name':b['source_id'],'source_endpoint':b['endpoint'],'source_field':rule['source_field'],'canonical_table':table,'canonical_column':rule['column'],'source_type':f['type'],'target_type':target['type'],'source_unit':rule.get('source_unit',''),'target_unit':target.get('unit',''),'transform_expression':rule.get('formula') or 'Decimal(value)*'+str(rule.get('multiplier','1')),'date_role':target.get('date_role',''),'required':int(rule.get('required',False)),'mapping_version':mapping.get('version',digest(mapping)),'description':f.get('description','')}
                    self.state.ledger_put('meta_source_mapping',digest([item['id'],table,b['source_id'],b['endpoint'],rule['source_field'],rule['column']]),row)

    def replay_archive(self,job):
        """Reaudit original request facts; never compile or silently fetch new requests."""
        import copy
        original=self._job(job)
        if original['plan'].get('revision')!='manager-v1':raise ValueError('旧版任务需要显式建立新的拉取任务')
        self.registry.refresh();self.sources.refresh();self.mappings.refresh()
        old=original['plan']
        for sid,version in old.get('source_instances',{}).items():
            if not identity_matches(self.sources.values.get(sid,{}),version):raise ValueError('来源身份或地址已变化，不能把旧归档标为新来源')
        contracts={}
        for key,contract in old['source_contracts'].items():
            current=self.sources.endpoint(*key.split(':',1))
            if any(current.get(k)!=contract.get(k) for k in ('path','method')):raise ValueError('查询地址或方法已变化，请创建新的拉取任务')
            contracts[key]=current
        names=old['request']['tables'];snapshots=self.mappings.snapshots(names)
        for name in names:
            if name in self.registry.errors or name in self.mappings.errors:raise ValueError('当前表或映射 YAML 无效：'+name)
            mapping=self.mappings.validate(snapshots[name])
            selected=[b for b in mapping['bindings'] if b['source_id'] in old['request'].get('sources',[]) and (not old['request'].get('endpoints') or b['endpoint'] in old['request']['endpoints']) and (not b.get('scope',{}).get('asset_type') or b['scope']['asset_type'] in old['request'].get('asset_types',['stock','fund','index']))]
            if not selected or any(b['source_id']+':'+b['endpoint'] not in contracts for b in selected):raise ValueError('新映射需要未归档的查询，请创建新的拉取任务')
            if any(p['severity']=='error' for p in self.mappings.diagnostics(mapping,selected)):raise ValueError('当前字段映射尚不可执行')
            snapshots[name]=self.mappings.validate({**mapping,'bindings':selected})
        plan=copy.deepcopy(old)
        plan.update(parent_task_id=job,archive_backfill=True,mapping_snapshots=snapshots,
            mapping_version=digest(snapshots),source_contracts=contracts,
            canonical_contracts={name:self.registry.spec(name).hash for name in names},
            retrieval_source_contracts=old.get('retrieval_source_contracts',old['source_contracts']),
            validation_mapping_snapshots=old.get('validation_mapping_snapshots',old['mapping_snapshots']),
            processor_version=PROCESSOR_VERSION)
        item=self.state.create_job(plan,self.settings.fingerprint)
        self._record_mapping_plan(item)
        for part in original['slices']:
            self.state.slice(item['id'],part['id'],part['dataset'],part['request'],status=part['status'],
                archive=part.get('archive'),sha=part.get('sha'),rows=part.get('rows'),result=part.get('result'))
            if part.get('archive'):
                data=read_archive(Path(part['archive']),part['sha'])
                meta=json.loads(Path(part['archive']).with_suffix('.meta.json').read_text())
                fields=data.get('fields',[]) if isinstance(data,dict) else []
                names=[f for f in fields if isinstance(f,str)] if isinstance(fields,list) else []
                self.state.ledger_put('import_source_archive',digest([item['id'],part['id']]),{
                    'task_id':item['id'],'dataset':part['dataset'],
                    'endpoint':part['request']['source_id']+':'+part['request']['endpoint'],
                    'request_hash':digest(part['request']['parameters']),'archive_path':part['archive'],
                    'response_sha256':part['sha'],'response_fields':','.join(names),
                    'fetched_at':meta.get('fetched_at'),'rows':part.get('rows',0)})
        copied=self.state.job(item['id']);problems=validate_archives(self,copied)
        for part in copied['slices']:
            errors=[p for p in problems if p.get('dataset')==part['dataset']]
            result={**(part.get('result') or {}),'validation_errors':errors}
            status=part['status']
            if errors:status='contract_blocked';result['blocked']='；'.join(p['reason'] for p in errors)
            elif status=='contract_blocked':status='downloaded';result.pop('blocked',None)
            self.state.slice(item['id'],part['id'],part['dataset'],part['request'],status=status,result=result)
        self.state.update_job(item['id'],status='blocked' if problems else 'downloaded')
        self.state.log(item['id'],'archive_revalidated',parent_task_id=job,validation_errors=problems)
        return self.state.job(item['id'])

    def download(self,job):
        if self.state.job(job)['plan'].get('revision')=='reference-v1':
            from .reference_service import ReferenceWorkflow
            return ReferenceWorkflow(self).download(job)
        self.source.active_job=job
        item=self._job(job,True)
        self.source.requests_per_minute=item['plan']['request'].get('requests_per_minute',self.settings.requests_per_minute)
        self.source.retrieval_policies=item['plan'].get('retrieval_policies',{})
        if item['plan'].get('archive_backfill'):
            problems=validate_archives(self,item)
            self.state.update_job(job,status='blocked' if problems else 'downloaded');return self.state.job(job)
        if item['plan'].get('file',{}).get('path'):
            from .files import download_file
            return download_file(self,job)
        if item['plan'].get('revision')!='manager-v1':return super().download(job)
        if item['status'] in {'completed','rolled_back','partially_rolled_back'}:return item
        self.state.update_job(job,status='downloading',error=None);self._task_status(job,'downloading')
        try:
            from .downloader import TaskDownloader
            validation=TaskDownloader(self,item).run()
            self.state.update_job(job,status='blocked' if validation['blocking'] else 'downloaded');return self.state.job(job)
        except Exception as exc:
            self.state.update_job(job,status='cancelled' if isinstance(exc,InterruptedError) else 'failed',error=self.safe_error(exc));raise

    def _build_canonical(self,item):
        if item['plan'].get('revision')!='manager-v1':return super()._build_canonical(item)
        for mapping in item['plan']['mapping_snapshots'].values():self.mappings.validate(mapping)
        archive_errors=validate_archives(self,item)
        if archive_errors:return {},archive_errors
        tables=defaultdict(list);rejects=[];factors={};raw_cache={};evidence=[]
        for part in item['slices']:
            if not part.get('archive') or part['status']=='split' or part['request'].get('dependency_only'):continue
            request=part['request'];sid=request['source_id'];endpoint=request['endpoint']
            data=read_archive(Path(part['archive']),part['sha']);meta=json.loads(Path(part['archive']).with_suffix('.meta.json').read_text());contract=item['plan']['source_contracts'][sid+':'+endpoint]
            known={f['name'] for f in contract['fields']};actual=data['fields']
            if len(actual)!=len(set(actual)) or set(actual)-known or set(contract.get('keys',[]))-set(actual):
                rejects.append({'dataset':endpoint,'reason':part.get('result',{}).get('blocked') or '源字段合同不匹配','blocking':True});continue
            if sid=='tushare' and any(name in builtin_catalog() for name in item['plan']['request']['tables']):
                from .catalog import dataset
                try:rows,bad=normalize_source(dataset(endpoint),data,item['id'],part['id'],meta['fetched_at'])
                except ValueError as exc:
                    rejects.append({'dataset':endpoint,'reason':str(exc),'blocking':True});continue
                rejects.extend(bad)
                codes=set(item['plan']['request'].get('codes',[]))
                if codes:rows=[r for r in rows if not r.get('ts_code') or r['ts_code'] in codes]
                mapped,found=self._map_part(endpoint,rows,item['id'],meta['fetched_at'],[],sid);factors.update(found)
                for name,values in mapped.items():
                    if name in item['plan']['request']['tables'] and name in builtin_catalog():
                        candidates=[b for b in item['plan']['mapping_snapshots'][name]['bindings'] if b['source_id']==sid and b['endpoint']==endpoint and resolve_processor(name,b) is not None]
                        if not candidates:continue
                        for row in values:
                            binding=next((b for b in candidates if not b.get('scope',{}).get('asset_type') or b['scope']['asset_type']==row.get('asset_type')),None)
                            if binding:row['_priority']=binding.get('priority',0)
                        tables[name].extend(values)
            for name,mapping in item['plan']['mapping_snapshots'].items():
                bindings=[b for b in mapping['bindings'] if b['source_id']==sid and b['endpoint']==endpoint and resolve_processor(name,b) is None]
                spec=canonical(name)
                for binding in bindings:
                    for index,values in enumerate(data['items'],1):
                        raw=dict(zip(actual,values))
                        try:
                            if len(values)!=len(actual):raise ValueError('源字段和值数量不一致')
                            partial,fields=self.mappings.map_row(spec,raw,binding)
                            if 'asset_type' in {f['name'] for f in spec.fields} and binding.get('scope',{}).get('asset_type'):partial.setdefault('asset_type',binding['scope']['asset_type'])
                            if item['plan']['request'].get('codes') and partial.get('security_code') and partial['security_code'] not in item['plan']['request']['codes']:continue
                            event=partial.get(self.registry.get(name).get('time_field'))
                            if not row_in_scope(name,partial,item['plan']['request']['start_date'],item['plan']['request']['end_date'],self.registry.get(name).get('time_field')):continue
                            row=self._canonical_record(spec,partial,endpoint,item['id'],meta['fetched_at'],index,raw,event,partial.get('available_at'),partial.get('availability_basis') or 'unknown',self.sources.values[sid]['provider'])
                            row.update(source_id=sid,mapping_version=mapping['version']);row['_priority']=binding.get('priority',0)
                            row['_binding_identity']=sid+':'+endpoint
                            row['_field_origins']={f['column']:{'selected':f['selected'],'source_null':f['raw_value'] in (None,''),'value_hash':digest(f['converted_value']),'origins':[sid+':'+endpoint+':'+f['source_field']]} for f in fields}
                            tables[name].append(row)
                            if index<=20 or any(f.get('conflict') for f in fields):evidence.extend({'task_id':item['id'],'table':name,'source_row':index,**f} for f in fields)
                        except (ValueError,TypeError,KeyError) as exc:rejects.append({'dataset':endpoint,'source_row':index,'reason':str(exc)})
        derived={name for name in {'market_daily_raw','market_daily_post_adjusted','trade_status_daily'} & set(item['plan']['request']['tables']) if any(resolve_processor(name,b) is not None for b in item['plan']['mapping_snapshots'][name]['bindings'])}
        if derived and not any(r.get('blocking') for r in rejects) and all(p['request']['source_id']=='tushare' for p in item['slices']):
            clone=json.loads(dumps(item));clone['plan']['request']['datasets']=list({p['dataset'] for p in item['slices']})
            built,bad=super()._build_canonical(clone)
            for name in derived:
                traces=(self.state.ledger_get('task_column_lineage',item['id']+':'+name) or {}).get('rows',{})
                values=built.get(name,[])
                for row in values:row['_field_origins']=traces.get(row_key(canonical(name),row),{})
                tables[name]=values+[r for r in tables.get(name,[]) if r.get('_binding_identity')]
            rejects.extend(bad)
        if tables.get('fund_nav_daily'):
            from .nav_semantics import resolve_nav_rows
            tables['fund_nav_daily'],conflicts=resolve_nav_rows(tables['fund_nav_daily']);rejects.extend(conflicts)
        normalized={}
        for name,rows in tables.items():
            spec=canonical(name);merged={};selected_fields={};column_lineage={};order=sorted(rows,key=lambda r:r.get('_priority',0));source_rows={};invalid_keys=set()
            for row in order:
                try:
                    binding_id=row.get('_binding_identity')
                    origins=row.get('_field_origins',{})
                    row=typed(spec,row)
                    for binding in item['plan']['mapping_snapshots'][name]['bindings']:
                        if resolve_processor(name,binding) is not None and binding['endpoint'] in str(row.get('source_endpoint','')).split(','):validate_output(binding,row)
                    time_field=self.registry.get(name).get('time_field')
                    if not row_in_scope(name,row,item['plan']['request']['start_date'],item['plan']['request']['end_date'],time_field):continue
                    if any(row.get(k) is None for k in spec.keys):raise ValueError('共同键或联合键为空，无法定位同一行记录')
                    key=row_key(spec,row)
                    if binding_id:
                        identity=(binding_id,key)
                        previous=source_rows.get(identity)
                        business={k:v for k,v in row.items() if k not in GENERATED}
                        if previous is not None:
                            if previous!=business:
                                invalid_keys.add(key);rejects.append({'dataset':name,'source':binding_id,'reason':'同一查询源的共同键不唯一，存在不同值，已阻止不明确的多对一合并','blocking':True})
                            else:evidence.append({'task_id':item['id'],'table':name,'event':'identical_source_row_deduplicated','source':binding_id,'row_key':key})
                            continue
                        source_rows[identity]=business
                    if key not in merged:
                        merged[key]=row
                        column_lineage[key]={c:p for c,p in origins.items() if p.get('source_null') or p.get('value_hash')==digest(row.get(c))}
                        if len(merged)<=20:selected_fields[key]={c:{'source_id':row.get('source_id'),'endpoint':row.get('source_endpoint'),'reason':'primary_or_only_source'} for c in row if row[c] is not None and c not in GENERATED}
                    else:
                        current=merged[key];contributed=False
                        for f in spec.fields:
                            c=f['name']
                            if c in GENERATED or c in {'available_at','availability_basis'}:continue
                            if current.get(c) is None and row.get(c) is not None:
                                current[c]=row[c];selected_fields.setdefault(key,{})[c]={'source_id':row.get('source_id'),'endpoint':row.get('source_endpoint'),'reason':'primary_null_fallback'}
                                if c in origins:column_lineage.setdefault(key,{})[c]=origins[c]
                                contributed=True
                            elif c not in GENERATED and current.get(c) is not None and row.get(c) is not None and current[c]!=row[c]:
                                evidence.append({'task_id':item['id'],'table':name,'column':c,'conflict':True,'selected_source':current.get('source_id'),'rejected_source':row.get('source_id'),'selected_value':current[c],'other_value':row[c]})
                        if contributed:
                            for c in ('source_name','source_id','source_endpoint'):
                                current[c]=','.join(sorted(set(filter(None,str(current.get(c) or '').split(',')+str(row.get(c) or '').split(',')))))
                            current['source_record_hash']=digest([current.get('source_record_hash'),row.get('source_record_hash')])
                            current['data_quality_status']='mixed_sources'
                            if current.get('available_at') is not None and row.get('available_at') is not None:current['available_at']=max(current['available_at'],row['available_at'])
                            else:current['available_at']=None;current['availability_basis']='composite_unknown'
                            current['downloaded_at']=max(v for v in (current.get('downloaded_at'),row.get('downloaded_at')) if v is not None)
                except (ValueError,TypeError) as exc:rejects.append({'dataset':name,'source_row':row.get('source_row_no'),'reason':str(exc)})
            for key,row in list(merged.items()):
                missing=[f['name'] for f in spec.fields if not f.get('nullable',True) and row.get(f['name']) is None]
                if key in invalid_keys or missing:
                    merged.pop(key)
                    if missing:rejects.append({'dataset':name,'reason':'按共同键合并后必需字段仍为空：'+','.join(missing),'row_key':key})
            if merged:
                normalized[name]=list(merged.values())
                for key,row in merged.items():
                    proof=column_lineage.setdefault(key,{})
                    for producer in item['plan']['mapping_snapshots'][name].get('producers',[]):
                        c=producer['column']
                        if c not in proof and row.get(c) is not None and producer['kind'] in {'computed','copy','constant','system'}:
                            proof[c]={'selected':True,'value_hash':digest(row[c]),'origins':['producer:'+producer['kind']+':'+str(producer.get('processor') or c)]}
                self.state.ledger_put('task_column_lineage',item['id']+':'+name,{'task_id':item['id'],'table':name,'rows':{key:column_lineage.get(key,{}) for key in merged}})
                evidence.extend({'task_id':item['id'],'event':'field_selected','table':name,'row_key':key,'column':column,**provenance} for key,fields in selected_fields.items() for column,provenance in fields.items())
        self.state.ledger_put('task_field_evidence',item['id'],{'task_id':item['id'],'rows':evidence})
        return normalized,rejects

    def preview(self,job):
        item=self._job(job,True)
        if item['plan'].get('revision')=='reference-v1':
            from .reference_service import ReferenceWorkflow
            return ReferenceWorkflow(self).preview(job)
        if item['plan'].get('revision')!='manager-v1':return super().preview(job)
        if item['status'] in {'planned','downloading','failed','cancelled'} or not item['slices']:raise ValueError('请先完成或恢复拉取')
        if item['status'] in {'completed','rolled_back','partially_rolled_back'}:return item['preview']
        tables,rejects=self._build_canonical(item);results=[]
        for name,rows in tables.items():
            spec=canonical(name)
            for key in item['plan']['request']['targets']:
                target=self.pg if key=='postgres' else self.ch
                from .schema_contract import compare_structure
                actual=self.schema.inspect(name,key)
                if not compare_structure(spec,actual,key)['ready']:
                    rejects.append({'dataset':name,'target':key,'reason':'目标物理结构与当前合同不一致，请先预览并执行结构迁移','code':'schema_drift','blocking':True})
                    continue
                try:comparison=target.compare(spec,rows,item['plan']['request']['mode'])
                except ValueError as exc:
                    if hasattr(exc,'conflict'):
                        rejects.append({**exc.conflict,'target':key});continue
                    raise
                results.append({'target':key,'table':name,'signature':comparison['signature'],'counts':comparison['counts'],'change_sample':comparison['changes'][:20],'conflict_sample':comparison['conflicts'][:20],'normalized_sample':rows[:20]})
        result={'job':job,'targets':item['plan']['request']['targets'],'tables':results,'rejected':len(rejects),'reject_sample':rejects[:100],'blocking':any(r.get('blocking') for r in rejects),'mapping_version':item['plan']['mapping_version'],'schema_versions':item['plan']['canonical_contracts'],'archives':[[p['id'],p['sha'],p['status']] for p in item['slices']],'target_schemas':{r['target']+':'+r['table']:digest(self.schema.inspect(r['table'],r['target'])) for r in results},'field_validation':self.field_audit(job),'source_report':[{'dataset':p['dataset'],'status':p['status'],'rows':p['rows'],'parameters':p['request']['parameters'],'row_limit':(p.get('result') or {}).get('row_limit'),'retrieval_mode':(p.get('result') or {}).get('retrieval_mode'),'termination':(p.get('result') or {}).get('termination'),'blocked':(p.get('result') or {}).get('blocked'),'dependency_only':p['request'].get('dependency_only',False)} for p in item['slices']]}
        result['token']=digest(result);self.state.update_job(job,status='blocked' if result['blocking'] else 'previewed',preview=result)
        self.state.log(job,'audit_ready',rejected=len(rejects),tables=len(results),blocking=result['blocking'])
        return result

    def commit(self,job,preview_token,accept_partial=False):
        item=self._job(job,True)
        if item['plan'].get('revision')=='reference-v1':
            from .reference_service import ReferenceWorkflow
            return ReferenceWorkflow(self).commit(job,preview_token,accept_partial)
        if (item.get('preview') or {}).get('rejected') and not accept_partial:raise ValueError('存在拒收记录，请明确确认仅写入通过检查的部分')
        if (item.get('preview') or {}).get('blocking'):raise ValueError('存在字段合同问题，必须修复后重新审计')
        self._accept_partial=accept_partial
        try:result=super().commit(job,preview_token)
        finally:self._accept_partial=False
        self.state.log(job,'commit_finished',status=result['status']);return result

    def resume(self,job):
        item=self._job(job)
        if item['plan'].get('revision')=='reference-v1':
            from .reference_service import ReferenceWorkflow
            # A resumed operation never grants visibility to an interrupted
            # revision. The new linked task needs download/audit/confirmation.
            return ReferenceWorkflow(self).retry(job)
        if item['status'] in {'completed','audit_pending','rolled_back','partially_rolled_back'}:
            from .publishing import retry_audit_sync
            retry_audit_sync(self,job);return self.state.job(job)
        self.state.update_job(job,cancel=0)
        # A resume never bypasses the audit confirmation gate.
        self.download(job);self.preview(job);return self.state.job(job)

    def rollback(self,job,targets=None):
        if self.state.job(job)['plan'].get('revision')=='reference-v1':
            from .reference_service import ReferenceWorkflow
            return ReferenceWorkflow(self).rollback(job,targets)
        return super().rollback(job,targets)

    def audit(self,job,after=0,limit=100):
        if self.state.job(job)['plan'].get('revision')=='reference-v1':
            from .reference_service import ReferenceWorkflow
            return ReferenceWorkflow(self).detail(job)
        return super().audit(job,after,limit)

    def field_audit(self,job=None):
        mappings=self.mappings.snapshots();registered_mappings=mappings;observed={};source_counts=defaultdict(lambda:defaultdict(int));stored=set();proofs=[];item=None
        contracts={(sid,endpoint):contract for sid,source in self.sources.values.items() for endpoint,contract in source.get('endpoints',{}).items()}
        if job:
            item=self._job(job)
            if item['plan'].get('file'):
                mapping=item['plan']['file']['column_map'];name=item['plan']['request']['datasets'][0]
                actual=set()
                for part in item['slices']:
                    if part.get('archive') and part['status']=='downloaded':
                        data=read_archive(Path(part['archive']),part['sha'])
                        if isinstance(data,dict) and isinstance(data.get('fields'),list) and all(len(r)==len(data['fields']) for r in data.get('items',[])):actual.update(data['fields'])
                proofs=self.state.ledger_rows('field_verification',job);result=[]
                for source,rule in mapping.items():
                    column=rule if isinstance(rule,str) else rule['column']
                    evidence=[p for p in proofs if p['table']==name and p['column']==column]
                    selected=[p for p in evidence if p.get('verified_origins',{}).get('file:'+name+':'+source,0)>0]
                    status=('verified_stored' if set(item['plan']['request']['targets'])<={p['target'] for p in selected} else 'partially_verified_stored') if selected else 'table_verified' if evidence else 'verified_response' if source in actual else 'unverified'
                    result.append({'source_id':'file','endpoint':name,'source_field':source,'source_type':'file_value','destinations':[{'table':name,'column':column}],'mapped':True,'response_verified':source in actual,'status':status,'evidence':evidence})
                return result
            mappings=item['plan'].get('mapping_snapshots') or mappings
            contracts={}
            for key,contract in item['plan'].get('source_contracts',{}).items():
                sid,endpoint=key.split(':',1) if ':' in key else ('tushare',key)
                contracts[sid,endpoint]=contract
            verification=self.state.ledger_rows('verification',job)
            stored={r['table'] for report in verification for r in report.get('tables',[])}
            proofs=self.state.ledger_rows('field_verification',job)
            for part in item['slices']:
                if not part.get('archive') or not Path(part['archive']).exists():continue
                data=read_archive(Path(part['archive']),part['sha'])
                sid=part['request'].get('source_id','tushare');observed.setdefault((sid,part['dataset']),set()).update(data['fields'])
                if part['status']=='split':continue
                for index,field in enumerate(data['fields']):
                    counts=source_counts[sid,part['dataset'],field]
                    for values in data.get('items',[]):
                        if not isinstance(values,list) or index>=len(values):continue
                        counts['source_null_count' if values[index] in (None,'') else 'source_non_null_count']+=1
        result=[]
        for (sid,endpoint),contract in contracts.items():
            for field in contract['fields']:
                destinations=[]
                for name,mapping in mappings.items():
                    for binding in mapping['bindings']:
                        if binding['source_id']==sid and binding['endpoint']==endpoint:
                            destinations.extend({'table':name,'column':rule['column'],'builtin':rule.get('builtin',False),'selected_for_write':True} for rule in binding['fields'] if rule['source_field']==field['name'])
                selected_destinations=list(destinations)
                if job:
                    known={(d['table'],d['column']) for d in destinations}
                    for name,mapping in registered_mappings.items():
                        for binding in mapping.get('bindings',[]):
                            if (binding['source_id'],binding['endpoint'])!=(sid,endpoint):continue
                            for rule in binding['fields']:
                                identity=(name,rule['column'])
                                if rule['source_field']==field['name'] and identity not in known:
                                    known.add(identity);destinations.append({'table':name,'column':rule['column'],'builtin':rule.get('builtin',False),'selected_for_write':False})
                present=field['name'] in observed.get((sid,endpoint),set())
                unavailable=bool(item and any(p['dataset']==endpoint and p['request'].get('source_id','tushare')==sid and p['status']=='unavailable' for p in item['slices']))
                origin=sid+':'+endpoint+':'+field['name']
                evidence=[p for p in proofs if any(p['table']==d['table'] and p['column']==d['column'] for d in destinations)]
                source_proofs=[p for p in evidence if p.get('verified_origins',{}).get(origin,0)>0]
                all_targets=set(item['plan']['request']['targets']) if item else set()
                proof_status='verified_stored' if all_targets<={p['target'] for p in source_proofs} else 'partially_verified_stored'
                counts=dict(source_counts[sid,endpoint,field['name']])
                source_empty=counts.get('source_null_count',0)>0 and not counts.get('source_non_null_count')
                status='source_unavailable' if unavailable else proof_status if present and source_proofs else 'source_null' if present and source_empty else 'registered_destination_not_selected' if present and destinations and not selected_destinations else 'table_verified' if present and any(d['table'] in stored for d in selected_destinations) else 'verified_response' if present and selected_destinations else 'pending_column' if not destinations else 'source_field_missing' if job and (sid,endpoint) in observed else 'unverified'
                result.append({'source_id':sid,'endpoint':endpoint,'source_field':field['name'],'source_type':field['type'],'description':field.get('description',''),'destinations':destinations,'mapped':bool(destinations),'response_verified':present,'status':status,'evidence':evidence,**counts,'document':contract.get('document')})
        for key,actual in observed.items():
            expected={f['name'] for f in contracts.get(key,{}).get('fields',[])}
            result.extend({'source_id':key[0],'endpoint':key[1],'source_field':field,'source_type':'unknown','destinations':[],'mapped':False,'response_verified':True,'status':'unregistered_blocked'} for field in sorted(actual-expected))
        return result
