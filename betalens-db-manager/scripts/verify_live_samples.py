"""One-off live acceptance samples. Secrets are read only from the private env."""
import argparse
from pathlib import Path
from betalens_db_manager.config import Settings
from betalens_db_manager.service import TushareImportService
from betalens_db_manager.util import atomic_json,now,dumps


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--env',required=True)
    args=parser.parse_args()
    service=TushareImportService(Settings.from_env(args.env))
    cases=[
        ('stock',['stock_basic','daily','daily_basic','suspend_d','bse_mapping'],['000001.SZ'],'2026-10-16','2026-10-16',{}),
        ('delisted_stock',['stock_basic','daily','daily_basic','suspend_d'],['600001.SH'],'2007-10-16','2007-10-16',{}),
        ('beijing_stock',['stock_basic','daily','daily_basic','suspend_d'],['920001.BJ'],'2026-10-16','2026-10-16',{}),
        ('etf',['fund_basic','fund_daily','fund_adj'],['510300.SH'],'2026-10-16','2026-10-16',{}),
        ('index',['index_basic','index_daily'],['000300.SH'],'2026-10-16','2026-10-16',{}),
        ('industry',['index_classify','index_member_all','ci_index_member'],['000001.SZ'],'2004-01-01','2026-10-16',{}),
        ('financial',['income','balancesheet','cashflow','fina_indicator'],['000001.SZ'],'2025-01-01','2025-12-31',
         {name:{'period':'20241231'} for name in ('income','balancesheet','cashflow','fina_indicator')}),
    ]
    report={'started_at':now(),'cases':[]}
    path=service.settings.state_dir/'reports'/'live_acceptance.json'
    for label,datasets,codes,start,end,parameters in cases:
        print(dumps({'case':label,'status':'started'}),flush=True)
        case={'case':label}
        try:
            job=service.plan({'datasets':datasets,'codes':codes,'start_date':start,'end_date':end,'parameters':parameters,'targets':['postgres','clickhouse'],'mode':'refresh_source'})
            case['task_id']=job['id']
            service.download(job['id']);preview=service.preview(job['id'])
            case.update(rejected=preview['rejected'],reject_sample=preview['reject_sample'],source_report=preview['source_report'],tables=[{'target':t['target'],'table':t['table'],**t['counts']} for t in preview['tables']])
            if not preview['rejected']:
                result=service.commit(job['id'],preview['token']);case.update(status=result['status'],error=result.get('error'),audit=service.audit(job['id'],limit=1)['receipts'])
            else:case['status']='blocked'
        except Exception as exc:case.update(status='failed',error=service.safe_error(exc))
        report['cases'].append(case);atomic_json(path,report)
        print(dumps({k:case[k] for k in ('case','task_id','status','rejected','error') if k in case}),flush=True)
    report['finished_at']=now();atomic_json(path,report)
    print(dumps({'report':str(path)}),flush=True)


if __name__=='__main__':main()
