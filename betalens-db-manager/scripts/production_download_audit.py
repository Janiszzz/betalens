"""Download and audit production-managed tasks; contains no business write path.

This command never applies schema, creates accounts, commits or rolls back.
The resulting production task IDs can be confirmed later using fresh previews.
"""
import argparse,copy,json
from collections import Counter
from pathlib import Path
from betalens_db_manager.config import Settings
from betalens_db_manager.manager import DatabaseManagerService
from betalens_db_manager.private import atomic_private
from betalens_db_manager.security import public_data
from betalens_db_manager.response_validation import validate_archives
from betalens_db_manager.util import dumps,now
from production_recipe import recipes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trade-date',default='20261023')
    parser.add_argument('--period',default='20241231',help='财报样本报告期 YYYYMMDD')
    parser.add_argument('--only',nargs='+')
    parser.add_argument('--include-vip',action='store_true',help='包含独立权限的 VIP 财报样本；默认不请求')
    parser.add_argument('--parent-report',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('报告已存在，不覆盖旧任务记录')
    service=DatabaseManagerService(Settings.from_env(args.env))
    assert service.settings.pg_database==service.settings.ch_database==service.settings.pg_schema=='betalens'
    available=recipes(args.trade_date,args.period)
    if args.only and set(args.only)-{c['label'] for c in available}:
        raise ValueError('未知验收案例')
    cases=[c for c in available if (args.include_vip or not any(e.endswith('_vip') for e in c['request']['endpoints'])) and (not args.only or c['label'] in args.only)]
    if not cases:raise ValueError('未选择任何可执行案例；VIP 案例需要 --include-vip')
    parents={c['label']:c.get('task_id') for c in json.loads(args.parent_report.read_text()).get('cases',[])} if args.parent_report else {}
    report={'started_at':now(),'business_writes_performed':False,'schema_changes_performed':False,'account_changes_performed':False,'cases':[]}
    def save():atomic_private(args.output,dumps(public_data(report,service.settings)))
    save()
    for case in cases:
        request=copy.deepcopy(case['request']);request['mode']='fill_missing'
        if 'index_basic' in request['endpoints']:request['parameters'].pop('index_basic',None)
        if case['label'] in {'financial_income','financial_fina_indicator'}:request['codes']=['000333.SZ']
        if parents.get(case['label']):request['parent_task_id']=parents[case['label']]
        result={'label':case['label'],'table':request['tables'][0],'request':request,'started_at':now()}
        report['cases'].append(result);save()
        try:
            job=service.plan(request);result['task_id']=job['id'];save()
            service.state.log(job['id'],'production_acceptance_download_started',label=case['label'],business_write=False)
            service.download(job['id']);job=service.state.job(job['id'])
            result['slices']=[{'endpoint':p['dataset'],'parameters':p['request']['parameters'],'status':p['status'],'rows':p['rows'],'archive':p['archive'],'retrieval':p.get('result')} for p in job['slices']]
            result['archive_validation']=validate_archives(service,job)
            tables,rejects=service._build_canonical(job)
            result['normalized_rows']={name:len(rows) for name,rows in tables.items()}
            result['rejects']=rejects;result['normalized_sample']=tables.get(result['table'],[])[:2]
            result['field_audit']=service.field_audit(job['id'])
            result['field_status_counts']=dict(Counter(r['status'] for r in result['field_audit']))
            preview=service.preview(job['id'])
            result['preview']={'blocking':preview['blocking'],'rejected':preview['rejected'],'reject_sample':preview['reject_sample'],
                'tables':[{k:r[k] for k in ('target','table','counts')} for r in preview['tables']]}
            result['status']='source_or_mapping_failed' if rejects or result['archive_validation'] else 'download_audit_passed' if tables.get(result['table']) else 'no_sample'
            result['schema_pending']=any(r.get('code')=='schema_drift' for r in preview['reject_sample'])
        except Exception as exc:result.update(status='failed',error=service.safe_error(exc))
        result['finished_at']=now();save()
        print(dumps({k:result.get(k) for k in ('label','table','task_id','status','schema_pending','error')}),flush=True)
    report['finished_at']=now();report['counts']=dict(Counter(r['status'] for r in report['cases']));save()
    print(dumps({'report':str(args.output),'counts':report['counts']}),flush=True)


if __name__=='__main__':main()
