"""Bounded live Tushare acceptance in disposable peer databases; no history loader."""
from dataclasses import replace
from pathlib import Path
import argparse,tempfile,uuid
from betalens_db_manager import Settings,DatabaseManagerService
from betalens_db_manager.catalog import load_catalog
from betalens_db_manager.util import digest,atomic_json,dumps


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--endpoints',nargs='*');parser.add_argument('--env',required=True);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    root=Path(tempfile.mkdtemp(prefix='betalens-manager-live-'));suffix=uuid.uuid4().hex[:10]
    settings=replace(Settings.from_env(args.env),config_dir=root/'config',state_dir=root/'state',archive_dir=None,env_file=None,pg_database='live_check_'+suffix,ch_database='live_check_'+suffix)
    service=DatabaseManagerService(settings);results=[];jobs=[]
    day='20261016';period='20251231'
    parameters={
        'stock_basic':{'ts_code':'000001.SZ'},'trade_cal':{'exchange':'SSE','start_date':day,'end_date':day},
        'namechange':{'ts_code':'000001.SZ'},'bse_mapping':{},'daily':{'ts_code':'000001.SZ','trade_date':day},
        'daily_basic':{'ts_code':'000001.SZ','trade_date':day},'adj_factor':{'ts_code':'000001.SZ','trade_date':day},
        'dividend':{'ts_code':'000001.SZ'},'fund_basic':{'ts_code':'510300.SH'},'fund_daily':{'ts_code':'510300.SH','trade_date':day},'fund_adj':{'ts_code':'510300.SH','trade_date':day},
        'fund_nav':{'ts_code':'510300.SH','start_date':day,'end_date':day},'fund_share':{'ts_code':'510300.SH','start_date':day,'end_date':day},
        'index_basic':{'ts_code':'000300.SH'},'index_daily':{'ts_code':'000300.SH','start_date':day,'end_date':day},'index_weight':{'index_code':'000300.SH','start_date':'20261001','end_date':day},
        'index_classify':{'src':'SW2021'},'index_member_all':{'ts_code':'000001.SZ','is_new':'Y'},'ci_index_member':{'ts_code':'000001.SZ','is_new':'Y'},
        'sw_daily':{'ts_code':'801010.SI','start_date':day,'end_date':day},'suspend_d':{'ts_code':'000001.SZ','trade_date':day},
    }
    for name in ('income','balancesheet','cashflow','fina_indicator'):
        parameters[name]={'ts_code':'000001.SZ','period':period};parameters[name+'_vip']={'ts_code':'000001.SZ','period':period,'limit':100,'offset':0}
    try:
        plan=service.schema_plan();installation=service.initialize(plan['token']);assert installation['postgres']['ready'] and installation['clickhouse']['ready'],installation
        cases=[(name,params) for name,params in parameters.items()]+[('stock_basic',{'ts_code':'000005.SZ','list_status':'D'}),('stock_basic',{'ts_code':'920002.BJ','list_status':'L'})]
        if args.endpoints:cases=[(n,({"trade_date":day} if n=='suspend_d' else p)) for n,p in cases if n in args.endpoints]
        for endpoint,params in cases:
            snapshots=service.mappings.snapshots();names=[n for n,m in snapshots.items() if any(b['source_id']=='tushare' and b['endpoint']==endpoint for b in m['bindings'])]
            canonical={n:service.registry.spec(n).hash for n in names}
            plan={'revision':'manager-v1','request':{'tables':names,'targets':['postgres','clickhouse'],'sources':['tushare'],'start_date':'1900-01-01','end_date':'2026-10-20','codes':[],'mode':'fill_missing','parameters':{}},'requests':[{'table':names[0],'source_id':'tushare','endpoint':endpoint,'parameters':params,'priority':0,'strategy':'sample'}],'mapping_snapshots':{n:snapshots[n] for n in names},'source_contracts':{'tushare:'+endpoint:service.sources.endpoint('tushare',endpoint)},'canonical_contracts':canonical,'contracts':{},'mapping_version':digest(snapshots),'target':settings.fingerprint}
            job=service.state.create_job(plan,settings.fingerprint);jobs.append(job['id'])
            result={'endpoint':endpoint,'parameters':params,'task_id':job['id']}
            try:
                service.download(job['id']);preview=service.preview(job['id']);result.update(status='audited',rejected=preview['rejected'],response_rows=sum(p['rows'] or 0 for p in service.state.job(job['id'])['slices']))
                if not preview['rejected']:
                    result['write_status']=service.commit(job['id'],preview['token'])['status'];result['verification']=service.state.ledger_rows('verification',job['id'])
                result['field_audit']=[r for r in service.field_audit(job['id']) if r['endpoint']==endpoint]
            except Exception as exc:result.update(status='unavailable',error=service.safe_error(exc))
            results.append(result);print(dumps({k:v for k,v in result.items() if k not in {'field_audit','verification'}}),flush=True)
        rollbacks=[]
        for job in reversed(jobs):
            if service.state.audits(job):rollbacks.append(service.rollback(job))
        report={'results':results,'rollbacks':rollbacks,'state_dir':str(settings.state_dir),'production_data_changed':False}
        atomic_json(args.report,report)
    finally:
        import psycopg2
        from psycopg2 import sql
        service.ch.execute(f'DROP DATABASE IF EXISTS `{settings.ch_database}`')
        conn=psycopg2.connect(**settings.pg('postgres'));conn.autocommit=True
        with conn.cursor() as cur:cur.execute(sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(settings.pg_database)))
        conn.close()

if __name__=='__main__':main()
