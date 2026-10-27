"""Durable publication and recovery shared by the service and CLI."""
from contextlib import contextmanager
from datetime import datetime, timezone

from .canonical import canonical, load_canonical_catalog
from .comparison import compare, row_hash, row_key, typed, content_hash
from .mapping import VERSION
from .util import digest, dumps, now


@contextmanager
def operation_lock(service, filename="publication.lock"):
    handle=(service.settings.state_dir/filename).open('a+')
    try:
        if __import__('os').name=='nt':
            import msvcrt
            handle.write('0');handle.flush();handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    except BlockingIOError:
        raise ValueError('另一个提交或回退正在执行，请稍后重试') from None
    finally: handle.close()


def existing(target,spec,rows):
    if hasattr(target,'_find_existing'):
        found=list(target._find_existing(spec,rows).values())
    else:
        from psycopg2.extras import RealDictCursor
        with target.connect(readonly=True) as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            found=list(target._fetch_existing(cur,spec,rows).values())
    return {row_key(spec,row):typed(spec,row) for row in found}


def task_record(service,job,status,error=None,persist=True):
    item=service.state.job(job)
    request=item['plan']['request']
    old=next(iter(service.state.ledger_rows('import_task',job)),{})
    row={'task_id':job,'source_name':','.join(request.get('sources',['tushare'])),'targets':','.join(request['targets']),
         'status':status,'request':request,'mapping_version':item['plan'].get('mapping_version',VERSION),'created_at':item['created'],
         'started_at':old.get('started_at') or now(),'finished_at':now() if status in {'completed','failed','partial','rolled_back'} else None,'error':error}
    if persist:service.state.ledger_put('import_task',job,row)
    return row


def sync_control(service,target_name,job):
    target=service.pg if target_name=='postgres' else service.ch
    service.state.request_sync(job,target_name)
    revision=next(r['revision'] for r in service.state.sync_records(job) if r['target']==target_name)
    try:
        from contextlib import nullcontext
        window=target.write_window('audit_sync:'+job) if hasattr(target,'write_window') else nullcontext()
        with window:
            for table in ('import_task','import_source_archive','import_target_status','import_row_audit','import_rollback','meta_source_mapping','meta_source_capability'):
                if table=='import_row_audit':
                    after=0
                    while True:
                        rows=service.state.audits(job,after=after,limit=1000)
                        if not rows:break
                        target.control_write(table,rows);after=rows[-1]['id']
                    continue
                rows=service.state.ledger_rows(table,job)
                if rows: target.control_write(table,rows)
    except Exception as exc:
        service.state.sync_result(job,target_name,revision,service.safe_error(exc));raise
    service.state.sync_result(job,target_name,revision)


def retry_audit_sync(service,job):
    """Retry latest control snapshots without replaying any business operation."""
    item=service._job(job)
    related={job}
    related.update(r['rollback_task_id'] for r in service.state.ledger_rows('import_rollback',job))
    for task in sorted(related):
        targets={r['target'] for r in service.state.sync_records(task)}
        if task==job:targets.update(item['plan']['request']['targets'])
        for target in sorted(targets):
            try:sync_control(service,target,task)
            except Exception:pass  # Error is durable in sync_outbox.
    pending=[r for task in related for r in service.state.sync_records(task) if r['revision']>r['acknowledged']]
    result=service.state.ledger_get('publication_result',job)
    if result:
        service.state.update_job(job,status='audit_pending' if pending and result['status'] in {'completed','rolled_back','partially_rolled_back'} else result['status'],
                                 error='; '.join(r['target']+': '+(r['error'] or '审计待同步') for r in pending) if pending else result.get('error'))
    for inverse in service.state.ledger_rows('import_rollback',job):
        rid=inverse['rollback_task_id']
        if inverse['target']!='clickhouse' or inverse['status']!='completed' or not hasattr(service.ch,'cleanup_rollback'):continue
        if any(r['target']=='clickhouse' for r in pending):continue
        try:
            service.ch.cleanup_rollback(rid,service.state)
            service.state.ledger_put('cleanup_pending',rid,{'task_id':rid,'target':'clickhouse','status':'completed','error':None})
        except Exception as exc:
            service.state.ledger_put('cleanup_pending',rid,{'task_id':rid,'target':'clickhouse','status':'pending','error':service.safe_error(exc)})
    return {'job':job,'pending':pending,'status':service.state.job(job)['status']}


def historical_hash_matches(service,target_name,spec,row,expected):
    """Accept additive nullable migrations only when every new value is null."""
    if row is None:return expected is None
    if row_hash(spec,row)==expected:return True
    from dataclasses import replace
    for operation in service.state.ledger_rows('schema_operation'):
        plan=operation.get('plan',{})
        if plan.get('table')!=spec.name:continue
        if not any(r.get('target')==target_name and r.get('status')=='completed' for r in operation.get('results',[])):continue
        deployed=next((p for p in plan.get('targets',[]) if p['target']==target_name),{})
        names={f['name'] for f in deployed.get('actual',{}).get('fields',[])}
        if not names or not set(spec.keys)<=names:continue
        if any(row.get(f['name']) is not None for f in spec.fields if f['name'] not in names):continue
        previous=replace(spec,fields=tuple(f for f in spec.fields if f['name'] in names))
        if row_hash(previous,row)==expected:return True
    return False


def record_result(service,job,status,error=None):
    row=task_record(service,job,status,error,persist=False)
    service.state.record_result(job,status,error,row)


def publish(service,job,token):
    with operation_lock(service):
        item=service._job(job); preview=item.get('preview')
        if item['status'] in {'completed','audit_pending'}:
            retry_audit_sync(service,job);return service.state.job(job)
        if item['status'] in {'rolled_back','partially_rolled_back'}: raise ValueError('已回退任务不能再次提交，请创建新任务')
        service._job(job,True)
        if not preview or preview.get('token')!=token or preview.get('blocking') or (preview.get('rejected') and not getattr(service,'_accept_partial',False)): raise ValueError('需要有效预览，并明确确认拒收部分不入库')
        if preview.get('archives') is not None and preview['archives']!=[[p['id'],p['sha'],p['status']] for p in item['slices']]:raise ValueError('缓存归档已变化，请重新审计')
        for identity,expected in preview.get('target_schemas',{}).items():
            target_name,table=identity.split(':',1)
            if digest(service.schema.inspect(table,target_name))!=expected:raise ValueError('目标物理结构已变化，请重新审计')
        tables,rejects=service._build_canonical(item)
        if hasattr(service,'registry') and any(service.registry.record(name).get('status')=='archived' for name in tables):raise ValueError('表已归档，禁止更新；历史查询和回退仍可使用')
        if rejects and not getattr(service,'_accept_partial',False): raise ValueError('标准化出现拒收记录，请重新预览')
        service.state.update_job(job,status='committing',error=None);task_record(service,job,'committing')
        failures=[];verification=[]
        for target_name in item['plan']['request']['targets']:
            target=service.pg if target_name=='postgres' else service.ch
            if target_name=='clickhouse':target.state=service.state
            total=0;keys=[];hashes=[]
            status={'task_id':job,'target':target_name,'status':'running','started_at':now(),'finished_at':None,'error':None,'row_count':0,'table_count':len(tables),'key_hash':None,'field_hash':None}
            service.state.ledger_put('import_target_status',dumps([job,target_name]),status)
            try:
                for name,rows in tables.items():
                    service.state.check_cancel(job)
                    spec=canonical(name)
                    if hasattr(service,'schema'):
                        from .schema_contract import compare_structure
                        if not compare_structure(spec,service.schema.inspect(name,target_name),target_name)['ready']:
                            raise ValueError(name+': 物理结构未迁移到当前类型与可空性合同，已阻止提交')
                    prepared=service.state.table_commit(job,target_name,name)
                    if prepared:
                        saved=prepared['comparison']
                    else:
                        expected=next(v for v in preview['tables'] if v['target']==target_name and v['table']==name)
                        saved=target.compare(spec,rows,item['plan']['request']['mode'])
                        if saved['signature']!=expected['signature']: raise ValueError(f'{name}: 预览后的目标数据已经变化，请重新预览')
                        stamp=now()
                        for change in saved['changes']:
                            if 'ingested_at' in change['after']:change['after']['ingested_at']=stamp
                        service.state.stage_commit(job,target_name,name,saved)
                    target.commit_rows(job,target_name,spec,[c['after'] for c in saved['changes']],item['plan']['request']['mode'],saved)
                    live=existing(target,spec,rows)
                    for change in saved['changes']:
                        current=live.get(change['row_key'])
                        if current is None or row_hash(spec,current)!=change['after_hash']: raise ValueError(f'{name}: 提交后字段哈希核验失败')
                    from .field_lineage import record_readback
                    record_readback(service,item,target_name,spec,saved,live)
                    service.state.finish_commit(job,target_name,name)
                    target.refresh_coverage(name)
                    total+=saved['counts']['inserted']+saved['counts']['updated']
                    table_keys=sorted(live)
                    table_hashes=sorted((key,content_hash(spec,live[key])) for key in table_keys)
                    audit_hashes=sorted((key,row_hash(spec,live[key])) for key in table_keys)
                    keys.append([name,table_keys]); hashes.append([name,table_hashes])
                    verification.append({'target':target_name,'table':name,'row_count':len(live),'key_hash':digest(table_keys),'field_hash':digest(table_hashes),'audit_field_hash':digest(audit_hashes),
                                         'first_date':min((str(v.get('event_date')) for v in live.values() if v.get('event_date')),default=None),
                                         'last_date':max((str(v.get('event_date')) for v in live.values() if v.get('event_date')),default=None),
                                         'null_ratios':{f['name']:str(sum(v.get(f['name']) is None for v in live.values())/len(live)) for f in spec.fields} if live else {}})
                status.update(status='completed',row_count=total,key_hash=digest(keys),field_hash=digest(hashes),finished_at=now())
            except Exception as exc:
                message=service.safe_error(exc); failures.append({'target':target_name,'error':message})
                completed_tables=[service.state.table_commit(job,target_name,name) for name in tables]
                completed_tables=[row for row in completed_tables if row and row['status']=='completed']
                status.update(status='partial' if completed_tables else 'failed',row_count=sum(row['comparison']['counts'].get('inserted',0)+row['comparison']['counts'].get('updated',0) for row in completed_tables),table_count=len(completed_tables),error=message,finished_at=now())
            service.state.ledger_put('import_target_status',dumps([job,target_name]),status)
            try: sync_control(service,target_name,job)
            except Exception as exc:
                # Business outcome is independent of its control-table mirror.
                # sync_control already persisted the retryable outbox error.
                service.state.log(job,'audit_sync_pending',target=target_name,error=service.safe_error(exc))
        cross_checks=[]
        for name in tables:
            pair=[row for row in verification if row['table']==name]
            if len(pair)==2:
                same=pair[0]['row_count']==pair[1]['row_count'] and pair[0]['key_hash']==pair[1]['key_hash'] and pair[0]['field_hash']==pair[1]['field_hash']
                cross_checks.append({'table':name,'status':'matched' if same else 'inconsistent'})
                if not same:failures.append({'target':'cross_target','error':name+' PG/CH 键集合或字段值不一致'})
        service.state.ledger_put('verification',job,{'task_id':job,'tables':verification,'cross_checks':cross_checks})
        status='partial' if failures or rejects or any(p.get('status') in {'unavailable','contract_blocked'} for p in item['slices']) else 'completed'
        error='; '.join(entry['target']+': '+entry['error'] for entry in failures) or None
        record_result(service,job,status,error)
        retry_audit_sync(service,job)
        return service.state.job(job)


def compatible_audits(service,target_name,target,audits):
    """Validate old immutable hashes against pre-migration column snapshots.

    Adding nullable audit columns must not invalidate historical rollback.
    Newly populated columns or a different import_task_id still block it.
    """
    from dataclasses import replace
    operations=service.state.ledger_rows('schema_operation');result=[];cache={}
    for audit in audits:
        name=audit['table_name'];spec=canonical(name)
        if name not in cache:cache[name]=existing(target,spec,[a['after_value'] for a in audits if a['table_name']==name])
        current=cache[name].get(audit['row_key'])
        if current and row_hash(spec,current)!=audit['after_hash']:
            for operation in operations:
                plan=operation.get('plan',{})
                if plan.get('table')!=name:continue
                target_plan=next((p for p in plan.get('targets',[]) if p['target']==target_name),{})
                names={f['name'] for f in target_plan.get('actual',{}).get('fields',[])}
                if not names or not set(spec.keys)<=names:continue
                if any(current.get(f['name']) is not None for f in spec.fields if f['name'] not in names):continue
                legacy=replace(spec,fields=tuple(f for f in spec.fields if f['name'] in names))
                if row_hash(legacy,current)==audit['after_hash']:
                    audit={**audit,'after_hash':row_hash(spec,current)}
                    break
        result.append(audit)
    return result


def reconcile_audits(service,item,target_name,target):
    """Recover only observed applied keys, never manufacture a whole batch audit."""
    from .commit_protocol import NeedsReconciliation,TargetConflict
    job=item['id'];records=service.state.table_commits(job,target_name)
    audits=service.state.audits(job,target_name)
    expected={r['table']:r['counts'].get('inserted',0)+r['counts'].get('updated',0)
              for r in (item.get('preview') or {}).get('tables',[]) if r['target']==target_name}
    known={r['table'] for r in records}|{a['table_name'] for a in audits}
    if any(count and name not in known for name,count in expected.items()):
        raise NeedsReconciliation('缺少预期变化的提交清单和审计，无法证明回退范围')
    if not records and not audits and item['status'] not in {'planned','previewed','downloaded','blocked','cancelled'} and not expected:
        raise NeedsReconciliation('没有提交或零变化证据，必须先核验目标库')
    for record in records:
        spec=canonical(record['table']);changes=record['comparison']['changes']
        live=existing(target,spec,[c['after'] or c['before'] for c in changes]);applied=set()
        for change in changes:
            current=live.get(change['row_key']);value=row_hash(spec,current) if current is not None else None
            if historical_hash_matches(service,target_name,spec,current,change['after_hash']):applied.add(change['row_key'])
            elif not historical_hash_matches(service,target_name,spec,current,change['before_hash']):
                raise TargetConflict(spec.name+': 目标行已被后续任务修改，禁止回退')
        present={a['row_key'] for a in audits if a['table_name']==spec.name}
        if record['status']=='prepared' or not applied<=present:
            service.state.finish_commit(job,target_name,spec.name,applied_keys=applied)
    return service.state.audits(job,target_name)


def rollback(service,job,targets=None):
    import uuid
    from contextlib import nullcontext
    from .commit_protocol import NeedsReconciliation,TargetConflict
    with operation_lock(service):
        item=service._job(job);targets=targets or item['plan']['request']['targets']
        if not targets or set(targets)-set(item['plan']['request']['targets']):raise ValueError('只能回退任务原先选择的目标库')
        results=[]
        for target_name in targets:
            prior=[r for r in service.state.ledger_rows('import_rollback',job) if r['target']==target_name]
            completed=[r for r in prior if r['status']=='completed']
            if completed:
                results.append(completed[-1]);continue
            pending=[r for r in prior if r['status'] in {'running','failed','blocked','needs_reconciliation'}]
            rollback_id=pending[-1]['rollback_task_id'] if pending else uuid.uuid4().hex
            target=service.pg if target_name=='postgres' else service.ch
            result={'rollback_task_id':rollback_id,'source_task_id':job,'target':target_name,'status':'running',
                    'restored':0,'deleted':0,'conflicts':0,'created_at':pending[-1]['created_at'] if pending else now(),'finished_at':None,'error':None}
            service.state.ledger_put('import_rollback',dumps([rollback_id,target_name]),result)
            if target_name=='clickhouse':target.state=service.state
            conflicts=[]
            try:
                # A CH writer window is acquired before reading or reconstructing audits.
                window=target.write_window('rollback:'+rollback_id) if hasattr(target,'write_window') else nullcontext()
                with window:
                    audits=reconcile_audits(service,item,target_name,target)
                    audits=compatible_audits(service,target_name,target,audits)
                    groups={}
                    for audit in reversed(audits):groups.setdefault(audit['table_name'],[]).append(audit)
                    actions={}
                    for name,entries in groups.items():
                        spec=canonical(name);prepared=service.state.table_commit(rollback_id,target_name,name)
                        if prepared and prepared['status']=='completed':
                            changes=prepared['comparison']['changes']
                            result['restored']+=sum(c['after'] is not None for c in changes)
                            result['deleted']+=sum(c['after'] is None for c in changes)
                            continue
                        live=existing(target,spec,[a['after_value'] for a in entries]);changes=[];selected=[]
                        for audit in entries:
                            current=live.get(audit['row_key']);value=row_hash(spec,current) if current is not None else None
                            before_hash=row_hash(spec,audit['before_value']) if audit['before_value'] is not None else None
                            if value==before_hash:continue  # Already at the inverse target; no historical claim.
                            if value!=audit['after_hash']:
                                conflicts.append({'table':name,'row_key':audit['row_key'],'reason':'目标行已被后续任务修改或删除'});continue
                            selected.append(audit)
                            changes.append({'before':typed(spec,audit['after_value']),'after':typed(spec,audit['before_value']) if audit['before_value'] is not None else None,
                                            'row_key':audit['row_key'],'before_hash':audit['after_hash'],'after_hash':before_hash})
                        if prepared:
                            # The complete inverse intent is immutable, including already-applied rows.
                            changes=prepared['comparison']['changes']
                            from .commit_protocol import remaining_changes
                            remaining_changes(spec,prepared['comparison'],list(live.values()))
                            selected=[a for a in entries if any(c['row_key']==a['row_key'] for c in changes) and
                                      (row_hash(spec,live[a['row_key']]) if a['row_key'] in live else None)==a['after_hash']]
                        actions[name]=(selected,changes,prepared)
                    if conflicts:raise TargetConflict('目标行存在后续修改，未执行逆操作')
                    context=service.pg.connect() if target_name=='postgres' else nullcontext()
                    with context as connection:
                        for name,(entries,changes,prepared) in actions.items():
                            if not prepared:service.state.stage_commit(rollback_id,target_name,name,{'changes':changes,'counts':{},'signature':''})
                            spec=canonical(name)
                            if entries:
                                if target_name=='postgres':target.apply_rollback(spec,entries,rollback_id,connection)
                                else:target.apply_rollback(spec,entries,rollback_id)
                            # Re-read before declaring completion, including crash recovery.
                            if target_name=='postgres' and connection is not None:
                                from psycopg2.extras import RealDictCursor
                                with connection.cursor(cursor_factory=RealDictCursor) as cur:
                                    found=target._fetch_existing(cur,spec,[c['before'] for c in changes])
                                    live={row_key(spec,r):typed(spec,r) for r in found.values()}
                            else:live=existing(target,spec,[c['before'] for c in changes])
                            if any((row_hash(spec,live[c['row_key']]) if c['row_key'] in live else None)!=c['after_hash'] for c in changes):
                                raise NeedsReconciliation(name+': 回退后字段核验失败')
                            if target_name!='postgres':service.state.finish_commit(rollback_id,target_name,name)
                            result['restored']+=sum(c['after'] is not None for c in changes)
                            result['deleted']+=sum(c['after'] is None for c in changes)
                    if target_name=='postgres':
                        for name in actions:service.state.finish_commit(rollback_id,target_name,name)
                    for name in groups:target.refresh_coverage(name)
                    result.update(status='completed',finished_at=now())
            except Exception as exc:
                result.update(status='blocked' if isinstance(exc,TargetConflict) else 'needs_reconciliation' if isinstance(exc,NeedsReconciliation) else 'failed',error=service.safe_error(exc),finished_at=now(),conflicts=len(conflicts))
            service.state.ledger_put('import_rollback',dumps([rollback_id,target_name]),result)
            service.state.ledger_put('rollback_conflicts',rollback_id,{'task_id':rollback_id,'rows':conflicts})
            results.append(result)
        all_results=service.state.ledger_rows('import_rollback',job)
        finished={r['target'] for r in all_results if r['status']=='completed'}
        if finished==set(item['plan']['request']['targets']):record_result(service,job,'rolled_back')
        elif finished:record_result(service,job,'partially_rolled_back')
        retry_audit_sync(service,job)
        for result in results:
            pending=[r for task in (job,result['rollback_task_id']) for r in service.state.sync_records(task) if r['target']==result['target'] and r['revision']>r['acknowledged']]
            result['audit_sync_status']='pending' if pending else 'synced'
            if pending:result['sync_error']='; '.join(r['error'] or '审计待同步' for r in pending)
            elif result['status']=='completed' and result['target']=='clickhouse' and hasattr(service.ch,'cleanup_rollback'):
                try:service.ch.cleanup_rollback(result['rollback_task_id'],service.state)
                except Exception as exc:
                    message=service.safe_error(exc)
                    service.state.ledger_put('cleanup_pending',result['rollback_task_id'],{'task_id':result['rollback_task_id'],'target':'clickhouse','error':message})
                    result['cleanup_error']=message
        return {'job':job,'results':results}
