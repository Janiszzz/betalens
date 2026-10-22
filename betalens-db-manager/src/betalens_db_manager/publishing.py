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


def task_record(service,job,status,error=None):
    item=service.state.job(job)
    request=item['plan']['request']
    old=next(iter(service.state.ledger_rows('import_task',job)),{})
    row={'task_id':job,'source_name':','.join(request.get('sources',['tushare'])),'targets':','.join(request['targets']),
         'status':status,'request':request,'mapping_version':item['plan'].get('mapping_version',VERSION),'created_at':item['created'],
         'started_at':old.get('started_at') or now(),'finished_at':now() if status in {'completed','failed','partial','rolled_back'} else None,'error':error}
    service.state.ledger_put('import_task',job,row)


def sync_control(service,target_name,job):
    target=service.pg if target_name=='postgres' else service.ch
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


def publish(service,job,token):
    with operation_lock(service):
        item=service._job(job); preview=item.get('preview')
        if item['status']=='completed': return item
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
            total=0;keys=[];hashes=[]
            status={'task_id':job,'target':target_name,'status':'running','started_at':now(),'finished_at':None,'error':None,'row_count':0,'table_count':len(tables),'key_hash':None,'field_hash':None}
            service.state.ledger_put('import_target_status',dumps([job,target_name]),status)
            try:
                for name,rows in tables.items():
                    service.state.check_cancel(job)
                    spec=canonical(name)
                    prepared=service.state.table_commit(job,target_name,name)
                    if prepared:
                        saved=prepared['comparison']
                        live=existing(target,spec,[v['after'] for v in saved['changes']])
                        remaining=[]
                        for change in saved['changes']:
                            current=live.get(change['row_key']); current_hash=row_hash(spec,current) if current else None
                            if current_hash==change['after_hash']: continue
                            if current_hash!=change['before_hash']: raise ValueError(f'{name}: 预览后目标数据已经变化，需重新审计')
                            remaining.append(change['after'])
                        if remaining:
                            comparison=target.compare(spec,remaining,item['plan']['request']['mode'])
                            target.commit_rows(job,target_name,spec,remaining,item['plan']['request']['mode'],comparison['signature'])
                    else:
                        expected=next(v for v in preview['tables'] if v['target']==target_name and v['table']==name)
                        saved=target.compare(spec,rows,item['plan']['request']['mode'])
                        if saved['signature']!=expected['signature']: raise ValueError(f'{name}: 预览后的目标数据已经变化，请重新预览')
                        stamp=now()
                        for change in saved['changes']:
                            if 'ingested_at' in change['after']:change['after']['ingested_at']=stamp
                        service.state.stage_commit(job,target_name,name,saved)
                        target.commit_rows(job,target_name,spec,[c['after'] for c in saved['changes']],item['plan']['request']['mode'],target.compare(spec,[c['after'] for c in saved['changes']],item['plan']['request']['mode'])['signature'])
                    live=existing(target,spec,rows)
                    for change in saved['changes']:
                        current=live.get(change['row_key'])
                        if current is None or row_hash(spec,current)!=change['after_hash']: raise ValueError(f'{name}: 提交后字段哈希核验失败')
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
                status.update(status='failed',error='管理审计写入失败：'+service.safe_error(exc)); failures.append({'target':target_name,'error':status['error']})
                service.state.ledger_put('import_target_status',dumps([job,target_name]),status)
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
        service.state.update_job(job,status=status,error=error);task_record(service,job,status,error)
        for name in item['plan']['request']['targets']:
            try: sync_control(service,name,job)
            except Exception: pass  # The durable ledger retains the failed target for resume.
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


def rollback(service,job,targets=None):
    import uuid
    from contextlib import nullcontext
    with operation_lock(service):
        item=service._job(job);targets=targets or item['plan']['request']['targets']
        if not targets or set(targets)-set(item['plan']['request']['targets']):raise ValueError('只能回退任务原先选择的目标库')
        results=[]
        for target_name in targets:
            prior=[row for row in service.state.ledger_rows('import_rollback',job) if row['target']==target_name]
            completed=[row for row in prior if row['status']=='completed']
            if completed:results.append(completed[-1]);continue
            pending=[row for row in prior if row['status'] in {'running','failed'}]
            rollback_id=pending[-1]['rollback_task_id'] if pending else uuid.uuid4().hex
            audits=service.state.audits(job,target_name)
            if not audits:
                try:
                    cursor=0
                    while True:
                        page=service.pg.audit(job,cursor,1000);audits.extend(a for a in page if a['target']==target_name)
                        if len(page)<1000:break
                        cursor=page[-1]['id']
                except Exception:pass
            target=service.pg if target_name=='postgres' else service.ch
            audits=compatible_audits(service,target_name,target,audits)
            groups={};conflicts=[];actions={};restored=deleted=0
            for audit in reversed(audits):groups.setdefault(audit['table_name'],[]).append(audit)
            for name,entries in groups.items():
                spec=canonical(name);prepared=service.state.table_commit(rollback_id,target_name,name)
                if prepared and prepared['status']=='completed':
                    changes=prepared['comparison']['changes']
                    restored+=sum(c['after'] is not None for c in changes);deleted+=sum(c['after'] is None for c in changes)
                    continue
                live=existing(target,spec,[a['after_value'] for a in entries])
                changes=[{'before':typed(spec,a['after_value']),'after':typed(spec,a['before_value']) if a['before_value'] else None,
                          'row_key':a['row_key'],'before_hash':a['after_hash'],'after_hash':row_hash(spec,a['before_value']) if a['before_value'] else None} for a in entries]
                current_hashes={key:row_hash(spec,row) for key,row in live.items()}
                if prepared and all(current_hashes.get(c['row_key'])==c['after_hash'] for c in changes):
                    service.state.finish_commit(rollback_id,target_name,name)
                    restored+=sum(c['after'] is not None for c in changes);deleted+=sum(c['after'] is None for c in changes)
                    continue
                for change in changes:
                    if current_hashes.get(change['row_key'])!=change['before_hash']:
                        conflicts.append({'table':name,'row_key':change['row_key'],'reason':'目标行已被后续任务修改或删除'})
                actions[name]=(entries,changes)
            result={'rollback_task_id':rollback_id,'source_task_id':job,'target':target_name,'status':'blocked' if conflicts else 'running',
                    'restored':restored,'deleted':deleted,'conflicts':len(conflicts),'created_at':pending[-1]['created_at'] if pending else now(),'finished_at':None,'error':None}
            service.state.ledger_put('import_rollback',dumps([rollback_id,target_name]),result)
            if not conflicts:
                try:
                    context=service.pg.connect() if target_name=='postgres' else nullcontext()
                    with context as connection:
                        for name,(entries,changes) in actions.items():
                            service.state.stage_commit(rollback_id,target_name,name,{'changes':changes,'counts':{},'signature':''})
                            spec=canonical(name)
                            if target_name=='postgres':target.apply_rollback(spec,entries,rollback_id,connection)
                            else:
                                target.apply_rollback(spec,entries,rollback_id)
                                service.state.finish_commit(rollback_id,target_name,name)
                            restored+=sum(c['after'] is not None for c in changes);deleted+=sum(c['after'] is None for c in changes)
                    if target_name=='postgres':
                        for name in actions:service.state.finish_commit(rollback_id,target_name,name)
                    for name in groups:target.refresh_coverage(name)
                    result.update(status='completed',restored=restored,deleted=deleted,finished_at=now())
                except Exception as exc:result.update(status='failed',error=service.safe_error(exc),finished_at=now())
            else:result.update(finished_at=now(),error='目标行存在后续修改，尚未执行的逆操作已阻止')
            service.state.ledger_put('import_rollback',dumps([rollback_id,target_name]),result)
            service.state.ledger_put('rollback_conflicts',rollback_id,{'task_id':rollback_id,'rows':conflicts})
            for target_job in (job,rollback_id):
                try:sync_control(service,target_name,target_job)
                except Exception as exc:result['error']='审计镜像待重试：'+service.safe_error(exc)
            results.append(result)
        all_results=service.state.ledger_rows('import_rollback',job)
        finished={r['target'] for r in all_results if r['status']=='completed'}
        if finished==set(item['plan']['request']['targets']):service.state.update_job(job,status='rolled_back')
        elif finished:service.state.update_job(job,status='partially_rolled_back')
        task_record(service,job,service.state.job(job)['status'])
        for name in targets:
            try:sync_control(service,name,job)
            except Exception:pass
        return {'job':job,'results':results}
