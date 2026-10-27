"""User-authorized bounded acceptance on actual Betalens business tables.

Uses the manager's schema/task/download/audit/commit/readback entry points.
The temporary CH writer is database-scoped and removed in finally. Existing
private connection configuration, schedules and historical queues are unchanged.
"""
from __future__ import annotations
import argparse,copy,json,secrets,sys,uuid,ipaddress
from collections import Counter
from dataclasses import replace
from pathlib import Path
from psycopg2 import sql
from betalens_db_manager.config import Settings
from betalens_db_manager.manager import DatabaseManagerService
from betalens_db_manager.clickhouse import ClickHouse
from betalens_db_manager.storage import Postgres,relation
from betalens_db_manager.canonical import registry_context
from betalens_db_manager.comparison import row_hash,row_key,typed
from betalens_db_manager.publishing import existing,retry_audit_sync
from betalens_db_manager.private import atomic_private
from betalens_db_manager.security import public_data,remember_secrets
from betalens_db_manager.util import dumps,digest,now
from production_recipe import recipes
from production_verify_readback import proof_summary


def temporary_writer(service):
    original=service.settings;admin=ClickHouse(original)
    assert original.pg_database=='betalens' and original.ch_database=='betalens' and original.pg_schema=='betalens'
    user='manager_probe_'+uuid.uuid4().hex[:12];password=secrets.token_urlsafe(36);remember_secrets(password)
    address=ipaddress.ip_address(admin.execute('SELECT toString(address) FROM system.processes WHERE query_id=currentQueryID() FORMAT TSVRaw'))
    address=getattr(address,'ipv4_mapped',None) or address
    admin.execute(f'CREATE USER `{user}` IDENTIFIED WITH sha256_password BY {admin._literal(password)} HOST IP {admin._literal(str(address))}')
    try:
        admin.execute(f'GRANT SELECT,INSERT,ALTER,CREATE DATABASE,CREATE TABLE,DROP TABLE,TRUNCATE,OPTIMIZE ON `betalens`.* TO `{user}`')
        admin.execute(f'GRANT SELECT ON system.* TO `{user}`')
        admin.execute(f'GRANT SHOW ACCESS ON *.* TO `{user}`')
        settings=replace(original,ch_user=user,ch_password=password,ch_writer_mode='local_dedicated')
        service.settings=settings;service.ch=ClickHouse(settings);service.pg=Postgres(settings)
        capability=service.ch.write_coordinator.verify(service.ch)
    except BaseException:
        admin.execute(f'DROP USER IF EXISTS `{user}`');raise
    def cleanup():
        admin.execute(f'DROP USER IF EXISTS `{user}`')
        service.settings=original;service.ch=admin;service.pg=Postgres(original)
    return cleanup,{'user':user,'database':'betalens','host':str(address),'temporary':True,'capability':capability}


def row_count(service,target,name):
    adapter=service.pg if target=='postgres' else service.ch
    if target=='postgres':
        with adapter.connect(readonly=True) as conn,conn.cursor() as cur:
            cur.execute(sql.SQL('SELECT count(*) FROM {}').format(relation(adapter.settings,name)));return int(cur.fetchone()[0])
    return int(adapter.execute(f'SELECT count() FROM {adapter.table(name)} FINAL'))


def selected_fingerprint(service,target,name,rows):
    spec=service.registry.spec(name);adapter=service.pg if target=='postgres' else service.ch
    values=existing(adapter,spec,rows)
    return values,digest(sorted((k,row_hash(spec,r)) for k,r in values.items()))


def run_case(service,case,persist,audited_job=None):
    request=copy.deepcopy(case['request']);table=request['tables'][0]
    result={'label':case['label'],'table':table,'request':request,'started_at':now(),'stages':[]}
    job=service.replay_archive(audited_job) if audited_job else service.plan(request)
    jid=job['id'];result['task_id']=jid
    if audited_job:result['parent_task_id']=audited_job;result['archive_replay_without_network']=True
    service.state.log(jid,'production_smoke_started',label=case['label'],sample_only=True)
    persist(result)
    try:
        actual=job['plan']['request']
        if any(actual.get(k)!=request.get(k) for k in ('tables','targets','codes','start_date','end_date','mode','endpoints')):
            raise ValueError('关联归档的固定请求与验收范围不一致，停止执行')
        if not audited_job:service.download(jid)
        job=service.state.job(jid)
        result['source_slices']=[{'endpoint':p['dataset'],'parameters':p['request']['parameters'],'status':p['status'],'rows':p['rows'],'archive':p['archive'],'retrieval':p.get('result')} for p in job['slices']]
        result['stages'].append('downloaded');persist(result)
        source_rows=sum(p.get('rows') or 0 for p in job['slices'] if p['status']!='split')
        if source_rows>3000:raise ValueError('样本响应超过3000行，已停止确认，保留缓存供审计')
        tables,rejects=service._build_canonical(job)
        if rejects:raise ValueError('标准化拒收：'+dumps(rejects[:10]))
        rows=tables.get(table,[]);result['normalized_rows']=len(rows)
        if not rows:
            preview=service.preview(jid);result.update(status='no_sample',preview=preview,finished_at=now());persist(result);return result
        before={}
        for target in request['targets']:
            _,fingerprint=selected_fingerprint(service,target,table,rows)
            before[target]={'rows':row_count(service,target,table),'selected_hash':fingerprint}
        preview=service.preview(jid)
        result['preview']={'blocking':preview['blocking'],'rejected':preview['rejected'],'tables':[{k:r[k] for k in ('target','table','counts','signature')} for r in preview['tables']],'source_report':preview['source_report']}
        for target in request['targets']:
            _,fingerprint=selected_fingerprint(service,target,table,rows)
            if fingerprint!=before[target]['selected_hash'] or row_count(service,target,table)!=before[target]['rows']:raise ValueError('确认前业务数据已变化，停止验收')
        result['no_business_writes_before_confirmation']=True
        if preview['blocking'] or preview['rejected']:raise ValueError('审计未通过，禁止自动确认')
        result['stages'].append('audit_passed');persist(result)
        committed=service.commit(jid,preview['token'])
        if committed['status']=='audit_pending':
            retry_audit_sync(service,jid);committed=service.state.job(jid)
        result['commit_status']=committed['status'];result['commit_error']=committed.get('error')
        result['stages'].append('commit_finished');persist(result)
        reads=[]
        for target in request['targets']:
            spec=service.registry.spec(table);values,_=selected_fingerprint(service,target,table,rows)
            manifest=service.state.table_commit(jid,target,table)
            if not manifest:raise ValueError(target+': 缺少固定提交清单')
            expected={c['row_key']:c['after_hash'] for c in manifest['comparison']['changes']}
            expected.update({c['row_key']:c['before_hash'] for c in manifest['comparison']['checks'] if c['row_key'] not in expected})
            missing=set(expected)-set(values)
            mismatches=[k for k,r in values.items() if k in expected and row_hash(spec,r)!=expected[k]]
            after_count=row_count(service,target,table);counts=manifest['comparison']['counts']
            if missing or mismatches or after_count!=before[target]['rows']+counts['inserted']:raise ValueError(target+': 回读键集合、字段哈希或行数不一致')
            reads.append({'target':target,'before_rows':before[target]['rows'],'after_rows':after_count,'readback_rows':len(values),'expected_keys':len(expected),'missing_keys':len(missing),'field_mismatches':len(mismatches),'counts':counts,'readback_sample':list(values.values())[:2]})
        proofs=service.state.ledger_rows('field_verification',jid)
        if not proof_summary(proofs,spec,request['targets'])['complete']:
            raise ValueError('逐字段回读核验缺失、重复或不一致')
        result.update(readback=reads,verification=service.state.ledger_get('verification',jid),
            field_audit=service.field_audit(jid),field_verification=proofs,
            sample=rows[:2],status='passed' if committed['status']=='completed' else committed['status'],finished_at=now())
        result['stages'].append('readback_verified');persist(result)
        service.state.log(jid,'production_smoke_finished',label=case['label'],status=result['status'],rows=len(rows))
        return result
    except Exception as exc:
        result.update(status='failed',error=service.safe_error(exc),finished_at=now())
        service.state.log(jid,'production_smoke_failed',error=result['error']);persist(result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env',required=True,type=Path);parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--trade-date',default='20261023');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--period',default='20241231',help='无归档重放时使用的财报报告期 YYYYMMDD')
    parser.add_argument('--only',nargs='+');parser.add_argument('--include-vip',action='store_true')
    parser.add_argument('--use-configured-writer',action='store_true',help='使用已配置并通过核验的 CH 写入口，不创建临时账号')
    parser.add_argument('--audited-reports',nargs='+',type=Path,help='复用已通过下载审计任务的归档，建立关联任务，不重复下载')
    args=parser.parse_args();settings=Settings.from_env(args.env);service=DatabaseManagerService(settings)
    assert settings.pg_database==settings.ch_database==settings.pg_schema=='betalens'
    root=args.output_dir.expanduser().resolve();root.mkdir(parents=True,exist_ok=True,mode=0o700);root.chmod(0o700)
    path=root/'report.json'
    if path.exists():raise ValueError('验收报告已存在；使用新目录避免覆盖历史')
    with service.state.connect() as conn:
        if conn.execute("SELECT count(*) FROM tasks WHERE status IN ('queued','running')").fetchone()[0]:raise ValueError('存在执行/等待任务，停止生产验收')
        if conn.execute('SELECT count(*) FROM schedules WHERE enabled=1').fetchone()[0]:raise ValueError('先暂停定时更新再执行生产验收')
    available=recipes(args.trade_date,args.period)
    if args.only and set(args.only)-{c['label'] for c in available}:raise ValueError('未知验收案例')
    cases=[c for c in available if (args.include_vip or not any(e.endswith('_vip') for e in c['request']['endpoints'])) and (not args.only or c['label'] in args.only)]
    if not cases:raise ValueError('未选择任何可执行案例；VIP 案例需要 --include-vip')
    audited={}
    for source_report in args.audited_reports or []:
        for case in json.loads(source_report.read_text())['cases']:
            if case.get('status')=='download_audit_passed':audited[case['label']]=case
    if args.audited_reports:
        missing={c['label'] for c in cases}-set(audited)
        if missing:raise ValueError('这些请求没有通过下载审计的任务：'+','.join(sorted(missing)))
        cases=[{**c,'request':audited[c['label']]['request']} for c in cases]
    for case in cases:
        request=case['request'];request['mode']='fill_missing'
        if 'index_basic' in request['endpoints']:request['parameters'].pop('index_basic',None)
        if case['label'] in {'financial_income','financial_fina_indicator'}:request['codes']=['000333.SZ']
    report={'run_id':uuid.uuid4().hex,'started_at':now(),'production':True,'sample_only':True,'historical_backfill_resumed':False,
        'targets':settings.public()['postgres']|{'clickhouse_database':settings.ch_database},'cases':[], 'schema_operations':[], 'planned_cases':cases}
    def save():atomic_private(path,dumps(public_data(report,service.settings)))
    def persist(result):
        old=next((r for r in report['cases'] if r['label']==result['label']),None)
        if old is None:report['cases'].append(copy.deepcopy(result))
        else:old.clear();old.update(copy.deepcopy(result))
        save();print(dumps({k:result[k] for k in ('label','table','task_id','status','error') if k in result}|{'stage':result['stages'][-1] if result['stages'] else 'planned'}),flush=True)
    with registry_context(service.registry):
        report['schema_plans']=[service.schema.plan(name) for name in sorted(service.registry.catalog())]
        if any(t['requires_rebuild'] or t['blocked'] or t['removed'] or t['key_change'] for p in report['schema_plans'] for t in p['targets']):raise ValueError('存在非增量结构变更，必须单独处理')
        save()
        if not args.execute:
            print(dumps({'read_only_business_preview':True,'tables':len(report['schema_plans']),'cases':len(cases),'report':str(path)}));return
        if args.use_configured_writer:
            capability=service.ch.write_capability()
            if not capability['available']:raise ValueError('配置的 CH 写入口未通过核验：'+dumps(capability))
            report['configured_writer']=capability
            cleanup=None
        else:
            cleanup,writer=temporary_writer(service);report['temporary_writer']=writer
        save()
        try:
            for schema_plan in report['schema_plans']:
                table=schema_plan['table'];targets=[]
                from betalens_db_manager.schema_contract import compare_structure
                spec=service.registry.spec(table)
                for target in ('postgres','clickhouse'):
                    if not compare_structure(spec,service.schema.inspect(table,target),target)['ready']:targets.append(target)
                if not targets:continue
                counts={t:row_count(service,t,table) for t in targets if service.schema.inspect(table,t)['exists']}
                plan=service.schema.plan(table,targets);applied=service.schema.apply(plan['token'])
                report['schema_operations'].append(applied);save()
                if applied['status']!='completed':raise ValueError('结构适配失败：'+table+' '+dumps(applied['results']))
                for target,before in counts.items():
                    if row_count(service,target,table)!=before:raise ValueError('结构适配改变了历史行数：'+table)
                print(dumps({'schema':table,'status':'ready','targets':targets}),flush=True)
            for case in cases:
                try:run_case(service,case,persist,(audited.get(case['label']) or {}).get('task_id'))
                except Exception as exc:
                    # Planning/replay can fail before a new task ID exists.
                    persist({'label':case['label'],'table':case['request']['tables'][0],
                        'request':case['request'],'status':'failed','stages':[],
                        'error':service.safe_error(exc),'finished_at':now()})
            report['table_coverage']=[{'table':name,'cases':[r['label'] for r in report['cases'] if r['table']==name],
                'status':'passed' if all(any(r['label']==c['label'] and r.get('status')=='passed' for r in report['cases'])
                    for c in cases if c['request']['tables']==[name]) and any(c['request']['tables']==[name] for c in cases)
                    else 'not_verified'} for name in sorted(service.registry.catalog())]
            report['counts']=dict(Counter(r.get('status','interrupted') for r in report['cases']));report['finished_at']=now();save()
        finally:
            if cleanup:
                cleanup();report['temporary_writer_removed']=True
            else:report['configured_writer_retained']=True
            save()
    print(dumps({'report':str(path),'counts':report.get('counts'),'table_coverage':dict(Counter(r['status'] for r in report.get('table_coverage',[])))}),flush=True)
    if any(r.get('status')!='passed' for r in report['cases']) or len(report['cases'])!=len(cases):
        raise SystemExit(1)


if __name__=='__main__':main()
