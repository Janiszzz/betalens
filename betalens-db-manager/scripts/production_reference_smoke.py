"""Bounded managed reference acceptance. Existing tables are never migrated.

Defaults to a saved dry-run. --execute installs six create-only tables, downloads
three reference domains, audits, publishes, reads both targets, then appends
revocations. It deliberately leaves immutable audit and historical version rows.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from betalens_db_manager.config import Settings
from betalens_db_manager.manager import DatabaseManagerService
from betalens_db_manager.reference_service import ReferenceWorkflow
from betalens_db_manager.reference_api import schema_preview
from betalens_db_manager.util import atomic_json, now


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--env',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    settings=Settings.from_env(args.env);service=DatabaseManagerService(settings)
    workflow=ReferenceWorkflow(service);args.output.mkdir(parents=True,exist_ok=True)
    request={'modules':['calendar','industry','suspension'],'targets':['postgres','clickhouse'],
             'source_id':'tushare','security_codes':['000001.SZ'],'start_date':'2026-10-12','end_date':'2026-10-14',
             'exchanges':['SSE','SZSE','BSE'],'schemes':['SW2021','CITICS'],'levels':[1,2,3],
             'check_times':['10:00:00'],'requests_per_minute':40}
    preview=schema_preview(service)
    report={'created_at':now(),'request':request,'schema_preview':preview,'executed':False}
    atomic_json(args.output/'report.json',report)
    if not args.execute:
        print('Saved create-only DDL preview; no database writes or downloads.');return
    # Current local administrator writer is explicitly configured by the user.
    if not service.ch.write_capability()['available']:raise ValueError('CH 受控写入不可用')
    with service.state.connect() as conn:
        active=conn.execute("SELECT count(*) FROM tasks WHERE status IN ('queued','running')").fetchone()[0]
        schedules=conn.execute('SELECT count(*) FROM schedules WHERE enabled=1').fetchone()[0]
    if active or schedules:raise ValueError('存在活动任务或启用的调度；先停止，避免验收交叉写入')
    report['schema_results']=[workflow.store(target).install() for target in request['targets']]
    job=workflow.plan(request);job_id=job['id'];report['task_id']=job_id
    atomic_json(args.output/'report.json',report)
    service.download(job_id);audit=service.preview(job_id)
    atomic_json(args.output/'audit.json',audit)
    report['before_confirm']={target:workflow.store(target).snapshot() for target in request['targets']}
    if audit['blocking']:
        report.update(status='blocked',executed=True);atomic_json(args.output/'report.json',report)
        raise ValueError('所有范围均未通过完整查询核验；已保留归档及审计，未写入业务数据')
    result=service.commit(job_id,audit['token'],True)
    report['publication_result']=result['result'];report['executed']=True
    report['after_publish']={target:workflow.store(target).snapshot(audit=True) for target in request['targets']}
    report['integrity_after_publish']=workflow.check(request)
    atomic_json(args.output/'report.json',report)
    report['rollback']=service.rollback(job_id)
    report['after_revoke']={target:workflow.store(target).snapshot(audit=True) for target in request['targets']}
    report['integrity_after_revoke']=workflow.check(request)
    report['finished_at']=now()
    with service.state.connect() as conn:
        report['operational']={'active_tasks':conn.execute("SELECT count(*) FROM tasks WHERE status IN ('queued','running')").fetchone()[0],
                               'enabled_schedules':conn.execute('SELECT count(*) FROM schedules WHERE enabled=1').fetchone()[0]}
    atomic_json(args.output/'report.json',report)
    print({'task_id':job_id,'status':result['status'],'rollback':report['rollback']['status'],
           'targets':[{k:r.get(k) for k in ('target','module','status','scope_count','reason')} for r in result['result']['results']],
           'operational':report['operational']})


if __name__=='__main__':main()
