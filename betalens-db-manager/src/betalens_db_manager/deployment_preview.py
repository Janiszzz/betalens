"""Read-only physical inventory; deployment still uses explicit schema tasks."""
from .canonical import registry_context
from .schema_contract import compare_structure
from .storage import CONTROL_DDL
from .util import now


def historical_repair_preview(service,limit=200):
    """Bounded evidence only; replay creates a separate task and confirmation."""
    limit=max(1,min(int(limit),1000))
    result={'read_only':True,'generated_at':now(),'sample_limit_per_table':limit,'sampling':'各库各行情表最新的非空指数均价记录；不是全历史盘点','changes':[],'pending':[]}
    with registry_context(service.registry):
        for target_name,target in [('postgres',service.pg),('clickhouse',service.ch)]:
            for table in ('market_daily_raw','market_daily_post_adjusted'):
                try:
                    if target_name=='postgres':
                        from psycopg2 import sql
                        from psycopg2.extras import RealDictCursor
                        from .storage import relation
                        with target.connect(readonly=True) as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
                            cur.execute("SET LOCAL statement_timeout='10s'")
                            cur.execute(sql.SQL("SELECT * FROM {} WHERE asset_type=%s AND average_price IS NOT NULL ORDER BY trade_date DESC LIMIT %s").format(relation(target.settings,table)),('index',limit))
                            samples=list(cur.fetchall())
                    else:
                        samples=target._json_rows(target.execute(f"SELECT * FROM {target.table(table)} FINAL WHERE asset_type='index' AND average_price IS NOT NULL ORDER BY trade_date DESC LIMIT {limit} SETTINGS max_execution_time=10 FORMAT JSONEachRow"))
                    for row in samples:
                        if row.get('asset_type')=='index' and row.get('average_price') is not None:
                            result['changes'].append({'target':target_name,'table':table,'key':{k:row.get(k) for k in service.registry.spec(table).keys},'column':'average_price','before':row['average_price'],'after':None,'reason':'指数成交额/成交量不是指数点位均价','import_task_id':row.get('import_task_id'),'action':'独立修正任务；需显式确认'})
                except Exception as exc:result['pending'].append({'target':target_name,'table':table,'reason':service.safe_error(exc)})
        result['pending'].extend([
            {'table':'fundamental_periodic','reason':'先预览部署两研发列；按statement_type及归档证据迁移旧rd_exp，不猜测来源；重审任务保留旧version_id关联'},
            {'table':'corporate_action','reason':'从分红及改名归档重审；缺ex_date或完整历史依据时不制造生效日期或旧名称'},
            {'table':'fund_nav_daily','reason':'从fund_nav观察归档重新解析公告版本；冲突必须先人工解决'}])
    return result


def migration_preview(service):
    rows=[]
    with registry_context(service.registry):
        for name in sorted(service.registry.catalog()):
            spec=service.registry.spec(name)
            for target_name in ('postgres','clickhouse'):
                try:
                    actual=service.schema.inspect(name,target_name)
                    difference=compare_structure(spec,actual,target_name)
                    rows.append({'table':name,'target':target_name,'exists':actual['exists'],
                        'status':'ready' if difference['ready'] else 'migration_required',
                        'actual':actual,'difference':difference})
                except Exception as exc:
                    rows.append({'table':name,'target':target_name,'status':'unavailable','error':service.safe_error(exc)})
    # These bounded administrative counters deliberately retain int32. Business
    # integers and audit identity/row counts have separate Int64 contracts.
    controls=[{'table':name,'column':column,'physical_type':kind,'reason':'bounded administrative counter; unchanged'}
              for name,fields in CONTROL_DDL.items() for column,kind,_ in fields if kind=='integer']
    return {'generated_at':now(),'read_only':True,'tables':rows,'control_int32_review':controls,
        'clickhouse_write_capability':service.ch.write_capability(),
        'execution':'Review a table schema-plan, then explicitly apply its token. No DDL or data writes performed.'}
