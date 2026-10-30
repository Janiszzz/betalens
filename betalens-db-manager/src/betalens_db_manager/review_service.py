"""Human decisions over immutable rejected cohorts and recoverable target writes."""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone,timedelta
from pathlib import Path
import json,uuid

from .canonical import registry_context,catalog_snapshot
from .comparison import typed,row_key,compare,PROVENANCE
from .pipeline_cache import read_rows,write_rows,save_journal,load_journal,file_sha,schema
from .pipeline_store import PipelineStore
from .target_lanes import target_lanes,lane_values
from .publishing import operation_lock,existing
from .registry import GENERATED
from .time_scope import row_in_scope
from .util import digest,dumps,now


def convert_for_coverage_review(settings,job):
    from .response_validation import manual_coverage_review
    from .pipeline import normalize_child
    with manual_coverage_review(job):return normalize_child(settings,job)


def bounded_blocks(spec,rows,max_rows,max_bytes):
    import pyarrow as pa
    for offset in range(0,len(rows),max_rows):
        pending=[rows[offset:offset+max_rows]]
        while pending:
            block=pending.pop(0)
            if pa.Table.from_pylist(block,schema=schema(spec)).nbytes>max_bytes:
                if len(block)==1:raise ValueError('单行修补记录超过任务批次字节上限')
                middle=len(block)//2;pending[:0]=[block[:middle],block[middle:]];continue
            yield block


class ReviewService:
    def __init__(self,service):self.s=service;self.store=PipelineStore(service.state)

    def _job(self,job,write=False):
        item=self.s._job(job,write)
        if item['plan'].get('pipeline_version')!=2 or item['plan'].get('revision')!='manager-v1' or item['plan'].get('pipeline_parent'):
            raise ValueError('人工审计仅支持普通表异步主任务')
        if write:
            if item['status'] in {'rolled_back','partially_rolled_back'}:raise ValueError('已回退任务不能人工追加写入')
            if any(t['operation']!='pipeline_review' and t['status'] in {'queued','running'} for t in item['tasks']):
                raise ValueError('任务仍在执行，请等待任务完成或停止后再处理拒收批次')
        return item

    def _decisions(self,job):
        with self.s.state.connect() as c:
            return [{**dict(r),'payload':json.loads(r['payload']),'result':json.loads(r['result']) if r['result'] else None} for r in c.execute('SELECT * FROM pipeline_review_decisions WHERE job=? ORDER BY created,id',(job,))]

    def counts(self,job):
        with self.s.state.connect() as c:
            failed=[r[0] for r in c.execute("SELECT id FROM pipeline_units WHERE job=? AND kind='normalize' AND status='failed'",(job,))]
            decisions={r['bucket']:r['status'] for r in c.execute("SELECT bucket,status FROM pipeline_review_decisions WHERE job=? AND status!='superseded' ORDER BY created,id",(job,))}
        values=Counter(decisions.get(key,'pending') for key in failed)
        return {'total':len(failed),'pending':values['pending'],'queued':values['queued']+values['running'],
                'released':values['released'],'repaired':values['repaired'],'rejected':values['rejected'],'failed':values['failed'],
                'url':'/tasks/review?task='+job}

    def _unit(self,item,bucket):
        unit=self.store.get(item['id'],bucket)
        if not unit or unit['kind']!='normalize' or unit['status']!='failed':raise ValueError('该批次没有待人工处理的转换拒收')
        result=unit['result'] or {};link=self.s.state.ledger_get('pipeline_child',item['id']+':'+bucket) or {}
        child_id=result.get('child_id') or link.get('child_id')
        if not child_id:raise ValueError('缺少内部批次快照，请先恢复任务生成检查点')
        child=self.s.state.job(child_id)
        if child['plan'].get('pipeline_parent')!=item['id']:raise ValueError('内部批次与原任务不匹配')
        previous=next((d for d in reversed(self._decisions(item['id'])) if d['bucket']==bucket and d['status']!='superseded'),None)
        version=digest([unit['payload'],unit['result'],unit['error'],[(p['id'],p['sha'],p['status']) for p in child['slices']],item['target'],previous['id'] if previous else None])
        return unit,child,previous,version

    def _record(self,unit,decision=None):
        result=unit['result'] or {};samples=result.get('reject_sample') or []
        return {'batch':unit['id'],'status':decision['status'] if decision else 'pending','error':unit['error'],
                'rejected_rows':result.get('rejected'),'reasons':list(dict.fromkeys(s.get('reason') or s.get('code') or '未记录具体原因' for s in samples)),
                'samples':samples[:3],'cached_rows':sum(r['rows'] for records in result.get('tables',{}).values() for r in records),
                'decision':{k:decision[k] for k in ('id','action','status','note','result','error','created','updated','operation_job')} if decision else None}

    def list(self,job,offset=0,limit=20,status=''):
        item=self._job(job);decisions={d['bucket']:d for d in self._decisions(job) if d['status']!='superseded'}
        with self.s.state.connect() as c:
            keys=[r[0] for r in c.execute("SELECT id FROM pipeline_units WHERE job=? AND kind='normalize' AND status='failed' ORDER BY created,id",(job,))]
        if status:keys=[k for k in keys if (decisions.get(k) or {}).get('status','pending')==status]
        size=max(1,min(100,limit));offset=max(0,offset)
        return {'task_id':job,'rows':[self._record(self.store.get(job,key),decisions.get(key)) for key in keys[offset:offset+size]],
                'total':len(keys),'offset':offset,'limit':size,'counts':self.counts(job),'targets':item['plan']['request']['targets'],
                'target_matches':item['target']==self.s.settings.fingerprint,
                'active':any(t['status'] in {'queued','running'} for t in item['tasks'])}

    def detail(self,job,bucket):
        item=self._job(job);unit,child,decision,version=self._unit(item,bucket);result=unit['result'] or {}
        sources=[]
        for p in child['slices']:
            if not p.get('archive') or p['request'].get('dependency_only') or p['dataset'] in {'stock_basic','fund_basic','index_basic','trade_cal'}:continue
            sources.append({'slice_id':p['id'],'endpoint':p['dataset'],'parameters':p['request']['parameters'],'status':p['status'],'rows':p['rows'],
                            'sha256':p['sha'],'url':'/api/tasks/'+child['id']+'/archive/'+p['id']})
        tables=[]
        with registry_context(self.s.registry):
            for name in child['plan']['request']['tables']:
                spec=self.s.registry.spec(name);records=result.get('tables',{}).get(name,[])
                preview=[]
                if records:
                    import pyarrow.parquet as pq
                    record=records[0]
                    if record['schema_hash']!=spec.hash or file_sha(record['path'])!=record['sha256']:raise ValueError('标准化缓存哈希不符')
                    preview=pq.read_table(record['path'],schema=schema(spec)).slice(0,10).to_pylist()
                tables.append({'name':name,'zh_name':spec.zh_name,'keys':list(spec.keys),'fields':[f for f in spec.fields if f['name'] not in GENERATED],
                               'cached_rows':sum(r['rows'] for r in records),'preview':preview})
        from .response_validation import MANUAL_COVERAGE_CODES
        coverage=bool(sources) and any(s.get('code') in MANUAL_COVERAGE_CODES for s in result.get('reject_sample',[]))
        return {**self._record(unit,decision),'task_id':job,'child_id':child['id'],'version':version,'targets':item['plan']['request']['targets'],
                'samples':result.get('reject_sample') or [],'samples_are_partial':result.get('rejected',0)>len(result.get('reject_sample') or []),
                'sources':sources,'tables':tables,'coverage_acceptance_available':coverage,'can_release':(any(t['cached_rows'] for t in tables) or coverage) and decision is None,
                'release_reason':None if any(t['cached_rows'] for t in tables) or coverage else '此批次没有通过转换的可入库记录，请修补记录后预览导入，或拒绝并导出元信息',
                'scope':child['plan']['request'],'target_matches':item['target']==self.s.settings.fingerprint,
                'active':any(t['status'] in {'queued','running'} for t in item['tasks'])}

    def export(self,job,bucket):
        detail=self.detail(job,bucket);item=self._job(job)
        return {'format_version':1,'exported_at':now(),'task_id':job,'batch':bucket,'original_request':item['plan']['request'],
                'canonical_contracts':item['plan']['canonical_contracts'],'mapping_version':item['plan']['mapping_version'],
                'review':detail,'decisions':[{k:v for k,v in d.items() if k!='payload'} for d in self._decisions(job) if d['bucket']==bucket]}

    def _validate_row(self,spec,row,request):
        try:value=typed(spec,row)
        except (ValueError,TypeError,ArithmeticError) as exc:raise ValueError(spec.name+'：字段类型不符合合同：'+str(exc)) from None
        missing=[f['name'] for f in spec.fields if not f.get('nullable',True) and value.get(f['name']) in (None,'')]
        if missing:raise ValueError(spec.name+'：必需字段为空：'+','.join(missing))
        if any(value.get(k) in (None,'') for k in spec.keys):raise ValueError(spec.name+'：业务主键不能为空')
        if not row_in_scope(spec.name,value,request['start_date'],request['end_date'],self.s.registry.get(spec.name).get('time_field')):
            raise ValueError(spec.name+'：修补记录超出原任务日期范围')
        codes=set(request.get('codes') or [])
        if codes:
            key='index_code' if spec.name=='index_membership' else 'security_code' if 'security_code' in value else 'code' if 'code' in value else None
            if key is None or value.get(key) not in codes:raise ValueError(spec.name+'：修补记录超出原任务证券范围或无法确定证券字段')
        return value

    def preview(self,job,bucket,payload):
        if set(payload)-{'action','note','targets','rows','include_cached','version','accept_coverage'}:raise ValueError('人工审计包含未知参数')
        item=self._job(job,True);unit,child,decision,version=self._unit(item,bucket)
        if payload.get('version')!=version:raise ValueError('拒收批次已变化，请刷新后重新审计')
        if decision:raise ValueError('该批次已有处理决定；失败的决定请使用原决定重试')
        action=payload.get('action');raw_note=payload.get('note','')
        if not isinstance(raw_note,str):raise ValueError('人工审计说明必须为文本')
        note=raw_note.strip()
        if action not in {'release','repair'}:raise ValueError('请选择确认放行或手工修补')
        if not note or len(note)>4000:raise ValueError('请填写 1 至 4000 字的人工审计说明')
        targets=payload.get('targets',item['plan']['request']['targets'])
        if not isinstance(targets,list) or not targets or not all(isinstance(t,str) for t in targets) or len(set(targets))!=len(targets) or set(targets)-set(item['plan']['request']['targets']):raise ValueError('目标库必须属于原任务且不能重复')
        repairs=payload.get('rows',{})
        if not isinstance(repairs,dict) or set(repairs)-set(child['plan']['request']['tables']):raise ValueError('修补表必须属于原拒收批次')
        if action=='release' and repairs:raise ValueError('确认放行不接受修改记录，请选择手工修补')
        if action=='repair' and not any(repairs.values()):raise ValueError('请提供修补后的本地表记录')
        if any(not isinstance(rows,list) or not all(isinstance(r,dict) for r in rows) for rows in repairs.values()) or len(dumps(repairs).encode())>4*1024**2:
            raise ValueError('修补记录须为表名到对象数组，文件不能超过 4 MiB')
        include=payload.get('include_cached',True)
        if not isinstance(include,bool):raise ValueError('include_cached 必须为布尔值')
        accept_coverage=payload.get('accept_coverage',False)
        if not isinstance(accept_coverage,bool):raise ValueError('accept_coverage 必须为布尔值')
        source_result=unit['result'] or {}
        if accept_coverage:
            from .response_validation import MANUAL_COVERAGE_CODES
            if not any(s.get('code') in MANUAL_COVERAGE_CODES for s in source_result.get('reject_sample',[])):raise ValueError('此批次不存在可人工接受的来源覆盖异常')
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            conversion_plan=deepcopy(child['plan']);conversion_plan['pipeline_parent']=job
            conversion_plan['manual_coverage_preview']={'source_batch':bucket,'source_child':child['id'],'note':note,'created_at':now()}
            conversion=self.s.state.create_job(conversion_plan,self.s.settings.fingerprint)
            self.s.state.copy_slices(child['id'],conversion['id'],[p['id'] for p in child['slices']])
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
                try:source_result=pool.submit(convert_for_coverage_review,self.s.settings,conversion['id']).result()
                except Exception as exc:raise ValueError('人工覆盖预览转换失败：'+self.s.safe_error(exc)) from None
        operation_job=uuid.uuid4().hex;root=self.s.settings.state_dir/'review-previews'/operation_job;root.mkdir(parents=True,exist_ok=True)
        snapshot={'tables':{},'rejected':0,'review_operation_job':operation_job};journals=[];rows_summary=[];physical_schemas={};stamp=now()
        with registry_context(self.s.registry),catalog_snapshot(self.s.registry):
            for name in child['plan']['request']['tables']:
                spec=self.s.registry.spec(name);merged={}
                if include:
                    for record in source_result.get('tables',{}).get(name,[]):
                        for raw in read_rows(record,spec):
                            row=self._validate_row(spec,raw,child['plan']['request']);row['data_quality_status']='manual_release_with_exclusions' if action=='release' else 'manual_repair_with_exclusions'
                            merged[row_key(spec,row)]=row
                patch_keys=set()
                for index,patch in enumerate(repairs.get(name,[]),1):
                    if set(patch)-{f['name'] for f in spec.fields} or set(patch)&GENERATED:raise ValueError(name+'：修补不能增加未知字段或覆盖系统审计字段')
                    if any(patch.get(k) in (None,'') for k in spec.keys):raise ValueError(name+'：修补记录必须填写完整业务主键')
                    key=row_key(spec,patch);row={**merged.get(key,{}),**patch}
                    if key in patch_keys:raise ValueError(name+'：修补记录包含重复业务主键')
                    patch_keys.add(key)
                    defaults={'import_task_id':operation_job,'source_name':'manual_review','source_id':'manual_review','source_endpoint':'manual_review',
                              'source_row_no':index,'source_record_hash':digest([job,bucket,patch]),'mapping_version':item['plan']['mapping_version'],
                              'schema_version':spec.hash,'downloaded_at':datetime.fromisoformat(stamp),'data_quality_status':'manual_repair_with_exclusions'}
                    for field in spec.fields:
                        if field['name'] in defaults:row[field['name']]=defaults[field['name']]
                    time_field=self.s.registry.get(name).get('time_field')
                    if 'event_date' in {f['name'] for f in spec.fields} and row.get('event_date') is None:row['event_date']=row.get(time_field)
                    value=self._validate_row(spec,row,child['plan']['request']);merged[key]=value
                if not merged:continue
                values=list(merged.values());records=[];limit=child['plan']['request']['commit_batch_rows']
                for index,block in enumerate(bounded_blocks(spec,values,limit,child['plan']['request']['commit_batch_bytes'])):
                    record=write_rows(root/name/(str(index)+'.parquet'),spec,block);records.append(record)
                snapshot['tables'][name]=records
        if not snapshot['tables']:raise ValueError('没有满足存储合同的可入库记录；请手工修补或拒绝并导出')
        def preview_target(target):
            from .schema_contract import compare_structure
            journals=[];rows=[];structures={}
            with registry_context(self.s.registry),catalog_snapshot(self.s.registry):
                for name,records in snapshot['tables'].items():
                    spec=self.s.registry.spec(name);structure=self.s.schema.inspect(name,target)
                    if not compare_structure(spec,structure,target)['ready']:raise ValueError(name+'：目标物理结构不符合合同')
                    structures[target+':'+name]=digest(structure);db=self.s.pg if target=='postgres' else self.s.ch
                    for index,record in enumerate(records):
                        block=read_rows(record,spec);batch=operation_job+':'+name+':'+str(index)
                        before=existing(db,spec,block);comparison=compare(spec,block,before.values(),item['plan']['request']['mode'])
                        expected={**before,**{c['row_key']:c['after'] for c in comparison['changes']}}
                        journal=save_journal(root/'journals'/target/name/str(index),spec,comparison,list(expected.values()))
                        journals.append({'batch':batch,'target':target,'manifest':journal,'signature':comparison['signature']})
                        rows.append({'table':name,'target':target,'rows':record['rows'],**comparison['counts']})
            return journals,rows,structures
        for prepared,rows,structures in lane_values(target_lanes(targets,preview_target)):
            journals.extend(prepared);rows_summary.extend(rows);physical_schemas.update(structures)
        saved={'task_id':job,'bucket':bucket,'version':version,'action':action,'note':note,'targets':targets,'operation_job':operation_job,
               'snapshot':snapshot,'journals':journals,'created_at':stamp,'scope':child['plan']['request'],'canonical_contracts':item['plan']['canonical_contracts'],
               'mapping_version':item['plan']['mapping_version'],'source_instances':item['plan']['source_instances'],'fingerprint':self.s.settings.fingerprint,'physical_schemas':physical_schemas,
               'coverage_accepted':accept_coverage,'excluded_rows':source_result.get('rejected',0)}
        public={'action':action,'rows':rows_summary,'targets':targets,'expires_in_seconds':1800,'preview_id':operation_job,'content_hash':digest(saved),
                'coverage_accepted':accept_coverage,'excluded_rows':source_result.get('rejected',0),
                'notice':('本次人工接受现有归档覆盖；' if accept_coverage else '')+'只导入本次预览中满足主键、类型和范围约束的记录；原始拒收与未修补记录继续保留'}
        token=digest(public)
        self.s.state.ledger_put('pipeline_review_preview',token,saved)
        return {**public,'token':token}

    def confirm(self,job,bucket,token):
        with operation_lock(self.s,'manual-review.lock'):
            item=self._job(job,True);saved=self.s.state.ledger_get('pipeline_review_preview',token)
            if not saved or saved['task_id']!=job or saved['bucket']!=bucket:raise ValueError('预览令牌无效')
            with self.s.state.connect() as c:previous=c.execute('SELECT id,status,operation_job FROM pipeline_review_decisions WHERE id=?',(token,)).fetchone()
            if previous:
                if previous['status']=='superseded':raise ValueError('此审计预览已撤回，请重新预览')
                return dict(previous)
            unit,child,decision,version=self._unit(item,bucket)
            if decision or version!=saved['version']:raise ValueError('批次或审计决定已变化，请重新预览')
            if datetime.now(timezone.utc)-datetime.fromisoformat(saved['created_at'])>timedelta(minutes=30):raise ValueError('审计预览已过期，请重新预览')
            for key in ('canonical_contracts','mapping_version','source_instances'):
                if saved[key]!=item['plan'].get(key):raise ValueError('任务合同或来源已变化，请重新预览')
            if saved['fingerprint']!=self.s.settings.fingerprint:raise ValueError('数据库连接目标已变化，请重新预览')
            plan=deepcopy(child['plan']);plan['pipeline_parent']=job;plan['review_decision_id']=token;plan['request']['targets']=saved['targets']
            opjob=saved['operation_job'];stamp=now();task_id=uuid.uuid4().hex
            with self.s.state.connect() as c:
                c.execute('BEGIN IMMEDIATE')
                c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,NULL,NULL,0)',(opjob,stamp,stamp,'queued',self.s.settings.fingerprint,dumps(plan)))
                c.execute('INSERT INTO pipeline_review_decisions VALUES(?,?,?,?,?,?,?,?,?,NULL,NULL,?,?)',
                          (token,job,bucket,saved['action'],'queued',saved['note'],version,dumps(saved),opjob,stamp,stamp))
                c.execute('INSERT INTO tasks VALUES(?,?,?,?,?,?,NULL)',(task_id,opjob,'pipeline_review',dumps({'background':True,'decision_id':token}),'queued',stamp))
            self.s.state.log(job,'manual_review_queued',batch=bucket,decision_id=token,action=saved['action'])
            return {'id':token,'status':'queued','operation_job':opjob}

    def reject(self,job,bucket,payload):
        if set(payload)-{'version','note'}:raise ValueError('拒绝操作包含未知字段')
        with operation_lock(self.s,'manual-review.lock'):
            item=self._job(job,True);unit,child,decision,version=self._unit(item,bucket)
            if decision:raise ValueError('此批次已有审计决定')
            if payload.get('version')!=version:raise ValueError('批次已变化，请刷新后重试')
            if not isinstance(payload.get('note',''),str):raise ValueError('拒绝原因必须为文本')
            note=payload.get('note','').strip()
            if not note or len(note)>4000:raise ValueError('请填写拒绝原因，最多 4000 字')
            ident=uuid.uuid4().hex;stamp=now()
            with self.s.state.connect() as c:c.execute('INSERT INTO pipeline_review_decisions VALUES(?,?,?,?,?,?,?,?,NULL,?,NULL,?,?)',
                (ident,job,bucket,'reject','rejected',note,version,'{}',dumps({'export_url':'/api/tasks/'+job+'/reviews/'+bucket+'/export'}),stamp,stamp))
            self.s.state.log(job,'manual_review_rejected',batch=bucket,decision_id=ident,note=note)
            return {'id':ident,'status':'rejected','export_url':'/api/tasks/'+job+'/reviews/'+bucket+'/export'}

    def retry(self,job,bucket):
        item=self._job(job,True);unit,child,decision,version=self._unit(item,bucket)
        if not decision or decision['status']!='failed':raise ValueError('只有执行失败的人工决定可以重试')
        opjob=decision['operation_job']
        task=self.s.state.enqueue(opjob,'pipeline_review',{'background':True,'decision_id':decision['id']})
        with self.s.state.connect() as c:c.execute("UPDATE pipeline_review_decisions SET status='queued',error=NULL,updated=? WHERE id=?",(now(),decision['id']))
        return {'id':decision['id'],'status':'queued','task':task}

    def reopen(self,job,bucket):
        with operation_lock(self.s,'manual-review.lock'):
            item=self._job(job,True);unit,child,decision,version=self._unit(item,bucket)
            if not decision or decision['status'] not in {'failed','rejected'}:raise ValueError('仅失败或已拒绝的决定可以重新审计')
            with self.s.state.connect() as c:c.execute("UPDATE pipeline_review_decisions SET status='superseded',updated=? WHERE id=?",(now(),decision['id']))
            self.s.state.log(job,'manual_review_reopened',batch=bucket,previous_decision_id=decision['id'])
            return {'status':'pending','notice':'原决定与已经产生的写入回执继续保留'}

    def execute(self,ident):
        with self.s.state.connect() as c:r=c.execute('SELECT * FROM pipeline_review_decisions WHERE id=?',(ident,)).fetchone()
        if not r:raise ValueError('人工审计决定不存在')
        if r['status'] in {'released','repaired','rejected'}:return json.loads(r['result'])
        if r['status']=='superseded':raise ValueError('此审计决定已撤回')
        saved=json.loads(r['payload']);job=r['job'];opjob=r['operation_job']
        with self.s.state.connect() as c:c.execute("UPDATE pipeline_review_decisions SET status='running',error=NULL,updated=? WHERE id=?",(now(),ident))
        results=[];target_errors=[]
        try:
            item=self.s._job(job,True)
            if item['status'] in {'rolled_back','partially_rolled_back'}:raise ValueError('原任务已经回退，不能追加写入')
            self.s._job(opjob,True)
            from .pipeline import Pipeline,target_lock
            def execute_target(target):
                self.s.state.check_cancel(opjob)
                # Install the frozen preview intent before the writer. Its CAS
                # checks reject target changes since the human inspected it.
                with registry_context(self.s.registry),catalog_snapshot(self.s.registry),target_lock(self.s,target):
                    for entry in saved['journals']:
                        if entry['target']!=target or self.store.receipt(job,entry['batch'],target):continue
                        with registry_context(self.s.registry):
                            spec=self.s.registry.spec(entry['manifest']['table']);comparison=load_journal(entry['manifest'],spec)
                            if digest(self.s.schema.inspect(spec.name,target))!=saved['physical_schemas'][target+':'+spec.name]:raise ValueError(spec.name+'：目标物理结构在预览后发生变化')
                            stamp=datetime.now(timezone.utc)
                            for change in comparison['changes']:
                                if change['after'] is not None and 'ingested_at' in change['after']:change['after']['ingested_at']=stamp
                            expected={row_key(spec,row):row for row in read_rows(comparison['expected'],spec)}
                            expected.update({change['row_key']:change['after'] for change in comparison['changes'] if change['after'] is not None})
                            journal=save_journal(self.s.settings.state_dir/'pipeline'/job/'review-intents'/ident/target/entry['batch'].replace(':','_'),spec,comparison,list(expected.values()))
                            self.store.prepare(job,entry['batch'],target,journal)
                return Pipeline(self.s)._write(job,opjob,saved['snapshot'],target)
            outcomes=target_lanes(saved['targets'],execute_target)
            for outcome in outcomes:
                if outcome.error is not None:
                    target_errors.append({'target':outcome.target,'error':self.s.safe_error(outcome.error)})
                else:results.append(outcome.value)
                self.s.state.log(job,'manual_review_target_failed' if outcome.error else 'manual_review_target_completed',target=outcome.target,stage='write',elapsed_ms=outcome.elapsed_ms,error=self.s.safe_error(outcome.error) if outcome.error else None)
            lane_values(outcomes)
            status='released' if saved['action']=='release' else 'repaired'
            result={'status':status,'targets':results,'operation_job':opjob,'original_rejections_retained':True,
                    'coverage_accepted':saved.get('coverage_accepted',False),'excluded_rows':saved.get('excluded_rows',0)}
            with self.s.state.connect() as c:c.execute('UPDATE pipeline_review_decisions SET status=?,result=?,error=NULL,updated=? WHERE id=?',(status,dumps(result),now(),ident))
            self.s.state.update_job(opjob,status='completed');self.s.state.log(job,'manual_review_completed',decision_id=ident,batch=r['bucket'],action=saved['action'])
            return result
        except BaseException as exc:
            error=self.s.safe_error(exc)
            with self.s.state.connect() as c:c.execute("UPDATE pipeline_review_decisions SET status='failed',result=?,error=?,updated=? WHERE id=?",(dumps({'targets':results,'target_errors':target_errors}),error,now(),ident))
            self.s.state.update_job(opjob,status='partial' if results else 'failed',error=error)
            self.s.state.log(job,'manual_review_failed',decision_id=ident,batch=r['bucket'],error=error)
            raise
