from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
import uuid

import pytest
from fastapi.testclient import TestClient

from betalens_db_manager.api import create_app
from betalens_db_manager.catalog import dataset, load_catalog
from betalens_db_manager.config import Settings
from betalens_db_manager.service import TushareImportService, normalize
from betalens_db_manager.source import read_archive
from betalens_db_manager.util import atomic_json, dumps


def response(name, values):
    fields = [f["name"] for f in dataset(name).fields]
    return dict(fields=fields, items=[[row.get(key) for key in fields] for row in values])


class FakeSource:
    close = Decimal("12.3456789")
    fail = False

    def fetch(self, name, params, fields, path):
        if self.fail: raise RuntimeError("transient test failure")
        if name == "trade_cal":
            year = params["start_date"][:4]
            values = [dict(exchange=params["exchange"], cal_date=year+"1016", is_open=1, pretrade_date=year+"1015")]
        elif name == "daily":
            values = [dict(ts_code=params.get("ts_code", "000001.SZ"), trade_date=params["trade_date"], open=12, high=13, low=11,
                           close=self.close, pre_close=12, change=Decimal(".3456789"), pct_chg=1,
                           vol=Decimal("123.4567"), amount=Decimal("12.3456789"))]
        elif name=='income_vip':
            values=[dict(ts_code='000001.SZ',end_date=params['period'],ann_date='20250328',f_ann_date='20250328',report_type='1',comp_type='1',revenue='100',n_income='0')]
        elif name in {"adj_factor","fund_adj"}:
            values=[dict(ts_code=params.get('ts_code','000001.SZ'),trade_date=params['trade_date'],adj_factor='2')]
        else: values = []
        data = response(name, values)
        raw = dumps(dict(code=0, data=data)).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as handle: handle.write(raw)
        sha = hashlib.sha256(raw).hexdigest()
        atomic_json(path.with_suffix(".meta.json"), dict(sha256=sha, fetched_at="2026-10-17T00:00:00+00:00"))
        return data, sha


def test_architecture_has_no_business_or_retired_imports():
    source = Path(__file__).parents[1] / "src"
    for path in source.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [n.name for n in node.names] if isinstance(node, ast.Import) else []
            assert not any(n and n.split(".")[0] in {"betalens", "betalens_db_manager", "dashboard"} for n in names), path


def test_decimal_units_and_temporal_semantics():
    data = response("daily", [dict(ts_code="000001.SZ", trade_date="20261016", vol="1.123456789", amount="2.123456789", close="12.123456789")])
    rows, rejected = normalize(dataset("daily"), data, "job", "slice", "2026-10-17T00:00:00Z")
    assert not rejected
    assert rows[0]["vol"] == Decimal("112.345678900")
    assert rows[0]["amount"] == Decimal("2123.456789000")
    assert rows[0]["_available_at"].isoformat() == "2026-10-16T16:00:00+00:00"
    assert rows[0]["_availability_basis"].startswith("next_calendar_day_assumed")


def test_revisions_and_missing_dates_are_preserved_or_rejected():
    data = response("income", [dict(ts_code="000001.SZ", end_date="20260930", ann_date="20261017", report_type="1", comp_type="1", n_income="100"),
                               dict(ts_code="000001.SZ", end_date="20260930", ann_date="20261017", report_type="1", comp_type="1", n_income="101")])
    rows, rejected = normalize(dataset("income"), data, "job", "slice", "2026-10-18T00:00:00Z", [date(2026,10,19)])
    assert len(rows) == 2 and not rejected
    assert rows[0]["_key"] == rows[1]["_key"] and rows[0]["_hash"] != rows[1]["_hash"]
    assert "historical_revision_not_proven" in rows[0]["_availability_basis"]
    bad = response("daily", [dict(ts_code="000001.SZ", close="NaN")])
    assert normalize(dataset("daily"), bad, "j", "s", "2026-10-18")[1]


def test_removed_schedules_return_gone_and_manual_cancel_remains(tmp_path):
    service = TushareImportService(Settings(state_dir=tmp_path))
    with TestClient(create_app(service)) as client:
        assert client.get('/api/schedules').status_code==410
        assert client.post('/api/schedules',json={'enabled':True}).status_code==410
    job=service.plan({'datasets':['daily'],'codes':['000001.SZ'],'start_date':'2026-10-16','end_date':'2026-10-16'})
    service.state.enqueue(job['id'],'download');service.state.cancel(job['id'])
    assert service.state.job(job['id'])['tasks'][0]['status']=='cancelled'


def test_web_can_open_before_database_and_rejects_foreign_origin(tmp_path):
    service = TushareImportService(Settings(state_dir=tmp_path, token="not-for-client"))
    with TestClient(create_app(service)) as client:
        assert client.get("/api/config").status_code == 200
        assert "not-for-client" not in client.get("/api/config").text
        assert client.get("/api/catalog").status_code == 200
        assert client.get("/api/schedules").status_code == 410
        assert client.post("/api/imports", json={"datasets": ["daily"]}, headers={"Origin": "http://foreign.invalid"}).status_code == 403
        assert client.get("/api/config", headers={"Host": "foreign.invalid"}).status_code == 400


@pytest.fixture
def live(tmp_path):
    if os.environ.get("BETALENS_INFRA_INTEGRATION") != "1": pytest.skip("set BETALENS_INFRA_INTEGRATION=1 for isolated PG/CH integration")
    base = Settings.from_env()
    suffix = uuid.uuid4().hex[:12]
    settings = replace(base, pg_database="infra_test_" + suffix, ch_database="infra_test_" + suffix, state_dir=tmp_path,config_dir=tmp_path/'config',env_file=None,archive_dir=None)
    service = TushareImportService(settings, FakeSource())
    plan = service.schema_plan()
    service.initialize(plan["token"])
    from controlled_ch import configure_writer
    cleanup=configure_writer(service)
    try: yield service
    finally:
        cleanup()
        service.ch.execute(f"DROP DATABASE IF EXISTS `{settings.ch_database}`")
        import psycopg2
        from psycopg2 import sql
        conn = psycopg2.connect(**settings.pg("postgres")); conn.autocommit = True
        with conn.cursor() as cur: cur.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(settings.pg_database)))
        conn.close()


def create_sample(service):
    job = service.plan({"datasets": ["daily"], "codes": ["000001.SZ"], "start_date": "2026-10-16", "end_date": "2026-10-16"})
    service.download(job["id"])
    preview = service.preview(job["id"])
    return job, preview


def test_empty_install_commit_replica_and_idempotency(live):
    assert live.pg.schema_plan()["ready"]
    plan = live.schema_plan()
    assert live.initialize(plan["token"])["postgres"]["ready"]
    job, preview = create_sample(live)
    result = live.commit(job["id"], preview["token"])
    assert result["status"] == "completed", result.get("error")
    row = live.pg.query("market_daily_raw")["rows"][0]
    assert row["close"] == Decimal("12.3456789") and row["volume"] == Decimal("12345.67")
    assert live.pg.receipts(job["id"])
    repeated = live.commit(job["id"], preview["token"])
    assert repeated["status"] == "completed"
    second, second_preview = create_sample(live)
    assert sum(p["counts"]["inserted"]+p["counts"]["updated"] for p in second_preview["tables"]) == 0
    assert live.commit(second["id"], second_preview["token"])["status"] == "completed"
    assert len(live.pg.query("market_daily_raw")["rows"]) == 1


def test_independent_targets_partial_recovery_and_stale_preview(live,monkeypatch):
    job,preview=create_sample(live)
    original=live.ch.commit_rows
    monkeypatch.setattr(live.ch,'commit_rows',lambda *a: (_ for _ in ()).throw(RuntimeError('CH unavailable')))
    assert live.commit(job['id'],preview['token'])['status']=='partial'
    assert live.pg.query('market_daily_raw')['rows']
    monkeypatch.setattr(live.ch,'commit_rows',original)
    assert live.resume(job['id'])['status']=='completed'
    assert live.ch.query('market_daily_raw')['rows']
    live.source.close=Decimal('12.1')
    changed=live.plan({'datasets':['daily'],'codes':['000001.SZ'],'start_date':'2026-10-16','end_date':'2026-10-16','mode':'refresh_source'})
    live.download(changed['id']); pv=live.preview(changed['id'])
    with live.pg.connect() as conn,conn.cursor() as cur:
        cur.execute(f'UPDATE "{live.settings.pg_schema}".market_daily_raw SET close=99')
    result=live.commit(changed['id'],pv['token'])
    assert result['status']=='partial' and '已经变化' in result['error']


def test_both_targets_rollback_and_later_edit_protection(live):
    job,preview=create_sample(live)
    assert live.commit(job['id'],preview['token'])['status']=='completed'
    assert {row['target'] for row in live.audit(job['id'])['changes']}=={'postgres','clickhouse'}
    result=live.rollback(job['id'])
    assert all(r['status']=='completed' for r in result['results']),result
    assert live.pg.query('market_daily_raw')['rows']==[]
    assert live.ch.query('market_daily_raw')['rows']==[]
    job,preview=create_sample(live);live.commit(job['id'],preview['token'])
    live.source.close=Decimal('13')
    second=live.plan({'datasets':['daily'],'codes':['000001.SZ'],'start_date':'2026-10-16','end_date':'2026-10-16','mode':'refresh_source'})
    live.download(second['id']);pv=live.preview(second['id']);live.commit(second['id'],pv['token'])
    result=live.rollback(job['id'])
    assert all(r['status']=='blocked' for r in result['results']),result
    assert live.pg.query('market_daily_raw')['rows'][0]['close']==Decimal('13')


def test_raw_archive_tamper_blocks_publication(live):
    job, preview = create_sample(live)
    part = next(p for p in live.state.job(job["id"])["slices"] if p["dataset"] == "daily")
    with gzip.open(live.settings.state_dir / part["archive"], "wb") as handle: handle.write(b'{}')
    with pytest.raises(ValueError, match="哈希"):
        live.commit(job["id"], preview["token"])
    assert live.pg.query("market_daily_raw")["rows"] == []


def test_fields_are_mapped_by_name_and_unknown_fields_block():
    from betalens_db_manager.service import normalize_source
    data=response('daily',[dict(ts_code='000001.SZ',trade_date='20261016',close='12.123456789',vol='2')])
    data['fields'].reverse();data['items']=[list(reversed(row)) for row in data['items']]
    rows,rejected=normalize_source(dataset('daily'),data,'job','slice','2026-10-17T00:00:00Z')
    assert not rejected and rows[0]['close']==Decimal('12.123456789') and rows[0]['vol']==200
    data['fields'].append('unregistered');data['items'][0].append('1')
    with pytest.raises(ValueError,match='未登记字段'):normalize_source(dataset('daily'),data,'j','s','2026-10-17')


def test_financial_contract_covers_all_numeric_fields():
    from betalens_db_manager.canonical import canonical,FINANCIAL_ALIASES
    fields={f['name'] for f in canonical('fundamental_periodic').fields}
    for endpoint in ('income','balancesheet','cashflow','fina_indicator'):
        for field in dataset(endpoint).fields:
            if field['type']=='numeric':assert FINANCIAL_ALIASES.get(field['name'],field['name']) in fields


def test_private_configuration_is_atomic_and_secret(tmp_path):
    from dotenv import dotenv_values
    settings=Settings(env_file=tmp_path/'private.env')
    password="quote'$\\secret"
    path=settings.write_private_env({'PGPASSWORD':password,'TUSHARE_TOKEN':'private-token'})
    assert path.stat().st_mode&0o777==0o600
    assert dotenv_values(path)['PGPASSWORD']==password
    settings.write_private_env({'PGPASSWORD':''})
    assert dotenv_values(path)['PGPASSWORD']==password
    with pytest.raises(ValueError):settings.write_private_env({'PGUSER':'injected\nTOKEN=x'})


def test_clickhouse_only_is_independent_of_postgres(live,monkeypatch):
    monkeypatch.setattr(live.pg,'connect',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('PG unavailable')))
    job=live.plan({'datasets':['daily'],'codes':['000001.SZ'],'start_date':'2026-10-16','end_date':'2026-10-16','targets':['clickhouse']})
    live.download(job['id']);preview=live.preview(job['id'])
    result=live.commit(job['id'],preview['token'])
    assert result['status']=='completed',result.get('error')
    assert live.ch.query('market_daily_raw')['rows']
    assert live.audit(job['id'])['changes']
    assert live.rollback(job['id'])['results'][0]['status']=='completed'


def test_bulk_keys_are_not_lost_after_first_page(live):
    from betalens_db_manager.canonical import canonical
    from betalens_db_manager.comparison import typed
    from betalens_db_manager.publishing import existing
    spec=canonical('security_dim')
    rows=[typed(spec,{'security_code':f'{i:06}.SZ','asset_type':'stock','name':str(i),'downloaded_at':'2026-10-17T00:00:00Z','import_task_id':'bulk','source_name':'file'}) for i in range(1250)]
    for name,target in [('postgres',live.pg),('clickhouse',live.ch)]:
        comparison=target.compare(spec,rows)
        target.commit_rows('bulk',name,spec,rows,'fill_missing',comparison['signature'])
        assert len(existing(target,spec,rows))==1250
        assert target.compare(spec,rows)['counts']['skipped']==1250


def test_explicit_canonical_file_round_trip(live,tmp_path):
    from betalens_db_manager.files import import_file
    path=tmp_path/'证券.csv';path.write_text('code,type,name\n920001.BJ,stock,北交所样本\n',encoding='utf-8')
    job=import_file(live,'security_dim',path,{'code':'security_code','type':'asset_type','name':'name'})
    pv=live.preview(job['id'])
    assert not pv['rejected'],pv['reject_sample']
    assert live.commit(job['id'],pv['token'])['status']=='completed'
    assert live.pg.query('security_dim')['rows'][0]['source_name']=='file'
    assert live.ch.query('security_dim')['rows'][0]['name']=='北交所样本'
    assert all(r['status']=='completed' for r in live.rollback(job['id'])['results'])


def test_update_rollback_restores_identical_values_in_both_targets(live):
    from betalens_db_manager.canonical import canonical
    from betalens_db_manager.comparison import row_hash
    first,preview=create_sample(live);live.commit(first['id'],preview['token'])
    before=live.pg.query('market_daily_raw')['rows'][0]
    live.source.close=Decimal('12.9')
    second=live.plan({'datasets':['daily'],'codes':['000001.SZ'],'start_date':'2026-10-16','end_date':'2026-10-16','mode':'refresh_source'})
    live.download(second['id']);pv=live.preview(second['id']);assert live.commit(second['id'],pv['token'])['status']=='completed'
    result=live.rollback(second['id']);assert all(r['status']=='completed' for r in result['results']),result
    spec=canonical('market_daily_raw')
    assert row_hash(spec,before)==row_hash(spec,live.pg.query(spec.name)['rows'][0])==row_hash(spec,live.ch.query(spec.name)['rows'][0])
    assert all(r['status']=='completed' for r in live.rollback(first['id'])['results'])


def test_recovered_rollback_does_not_repeat_published_partitions(live,monkeypatch):
    job,pv=create_sample(live);live.commit(job['id'],pv['token'])
    original=live.ch.apply_rollback;counter={'value':0}
    def fail_second(spec,audits,task):
        counter['value']+=1
        if counter['value']==2:raise RuntimeError('interrupted publication')
        return original(spec,audits,task)
    monkeypatch.setattr(live.ch,'apply_rollback',fail_second)
    result=live.rollback(job['id'],['clickhouse'])
    assert result['results'][0]['status']=='failed'
    monkeypatch.setattr(live.ch,'apply_rollback',original)
    resumed=live.rollback(job['id'],['clickhouse'])
    assert resumed['results'][0]['status']=='completed',resumed
    assert live.ch.query('market_daily_raw')['rows']==[]


def test_file_date_misalignment_blocks_publication(live,tmp_path):
    from betalens_db_manager.files import import_file
    path=tmp_path/'wrong.csv';path.write_text('asset,code,day,event,close\nstock,000001.SZ,20261016,20261015,12.12\n')
    job=import_file(live,'market_daily_raw',path,{'asset':'asset_type','code':'security_code','day':'trade_date','event':'event_date','close':'close'})
    preview=live.preview(job['id'])
    assert preview['rejected'] and any('错位' in r['reason'] for r in preview['reject_sample'])
    with pytest.raises(ValueError):live.commit(job['id'],preview['token'])


def test_market_mapping_units_are_explicit():
    from betalens_db_manager.mapping import all_source_mappings
    mappings=all_source_mappings()
    def find(endpoint,field):return next(row for row in mappings if row['source_endpoint']==endpoint and row['source_field']==field)
    assert find('daily','vol')['source_unit']=='手'
    assert find('daily','vol')['target_unit']=='股/份'
    assert find('daily_basic','total_share')['source_unit']=='万股'
    assert find('fund_share','fd_share')['source_unit']=='万份'
    assert find('ci_index_member','l3_code')['canonical_table']!='__raw_archive_only__'


def test_optional_source_columns_and_required_quote_columns():
    from betalens_db_manager.service import normalize_source
    payload=response('daily',[{'ts_code':'000001.SZ','trade_date':'20261016','close':'12'}])
    index=payload['fields'].index('ah_vol');payload['fields'].pop(index);payload['items'][0].pop(index)
    assert not normalize_source(dataset('daily'),payload,'j','s','2026-10-17')[1]
    index=payload['fields'].index('close');payload['fields'].pop(index);payload['items'][0].pop(index)
    with pytest.raises(ValueError,match='必需字段缺失'):normalize_source(dataset('daily'),payload,'j','s','2026-10-17')


def test_mixed_source_fill_retains_lineage_and_conflict_details():
    from betalens_db_manager.canonical import canonical
    from betalens_db_manager.comparison import compare
    spec=canonical('market_daily_raw')
    before={'asset_type':'stock','security_code':'000001.SZ','trade_date':'2026-10-16','close':'10','pb':None,'source_name':'wind','source_record_hash':'wind-record'}
    incoming={**before,'close':'11','pb':'2','source_name':'tushare','source_record_hash':'tushare-record'}
    result=compare(spec,[incoming],[before],'fill_missing')
    after=result['changes'][0]['after']
    assert after['close']==Decimal('10') and after['pb']==Decimal('2')
    assert after['source_name']=='tushare,wind' and after['data_quality_status']=='mixed_sources'
    assert 'close' in result['conflicts'][0]['fields']
    recovered=compare(spec,[after],[before],'fill_missing')
    assert recovered['changes'][0]['after_hash']==result['changes'][0]['after_hash']


def test_file_adapter_fetch_and_decimal_normalize(tmp_path):
    from betalens_db_manager.sources import FileSourceAdapter
    path=tmp_path/'sample.csv';path.write_text('code,quantity\n000001.SZ,1.123456789\n')
    adapter=FileSourceAdapter()
    batches=list(adapter.fetch({'path':str(path),'column_map':{'code':'security_code','quantity':{'column':'volume','multiplier':'100'}}}))
    assert adapter.validate_response(batches[0])['missing']==[]
    normalized=adapter.normalize(batches[0],{'code':'security_code','quantity':{'column':'volume','multiplier':'100'}})
    assert normalized[0]['volume']==Decimal('112.345678900')


def test_file_timestamp_requires_explicit_timezone():
    from betalens_db_manager.sources import FileSourceAdapter,canonical_file_rules
    from betalens_db_manager.canonical import canonical
    adapter=FileSourceAdapter();payload={'fields':['known'],'items':[['2026/10/16 15:00:00']]}
    mapping={'known':{'column':'available_at','date_format':'%Y/%m/%d %H:%M:%S','timezone':'Asia/Shanghai'}}
    rules=canonical_file_rules(canonical('market_daily_raw'),mapping)
    assert adapter.normalize(payload,rules)[0]['available_at'].isoformat()=='2026-10-16T07:00:00+00:00'
    rules['known'].pop('timezone')
    with pytest.raises(ValueError,match='时区'):adapter.normalize(payload,rules)


def test_file_duplicate_headers_are_rejected(tmp_path):
    from betalens_db_manager.sources import FileSourceAdapter
    path=tmp_path/'duplicate.csv';path.write_text('code,code\n000001.SZ,600000.SH\n')
    with pytest.raises(ValueError,match='重复'):list(FileSourceAdapter.rows(path))


def test_clickhouse_insert_spans_more_than_100_months(live):
    from betalens_db_manager.canonical import canonical
    from betalens_db_manager.comparison import typed
    spec=canonical('trade_calendar');rows=[]
    for index in range(121):
        day=date(2004+index//12,index%12+1,2)
        rows.append(typed(spec,{'exchange':'SSE','trade_date':day,'event_date':day,'is_open':1,'downloaded_at':'2026-10-17T00:00:00Z',
                               'import_task_id':'long-calendar','source_name':'tushare','source_endpoint':'trade_cal','source_row_no':index+1,
                               'source_record_hash':str(index),'mapping_version':'canonical-v2','data_quality_status':'accepted'}))
    comparison=live.ch.compare(spec,rows)
    result=live.ch.commit_rows('long-calendar','clickhouse',spec,rows,'fill_missing',comparison['signature'])
    assert result['inserted']==121
    assert len(live.ch.query(spec.name,limit=1000)['rows'])==121


def test_bulk_financial_endpoint_uses_identical_canonical_fields(live):
    job=live.plan({'datasets':['income_vip'],'codes':['000001.SZ'],'start_date':'2024-12-31','end_date':'2024-12-31'})
    requests=list(live._requests('income_vip',job['plan']['request'],[],[]))
    assert requests==[{'period':'20241231','offset':0,'limit':1000}]
    live.download(job['id']);preview=live.preview(job['id'])
    assert preview['rejected']==0,preview['reject_sample']
    result=live.commit(job['id'],preview['token']);assert result['status']=='completed',result.get('error')
    row=live.pg.query('fundamental_periodic')['rows'][0]
    assert row['statement_type']=='income' and row['source_endpoint']=='income_vip'
    assert row['net_income']==0 and row['revenue']==100


def test_index_source_routing_keeps_csi_and_uses_sw_contract(tmp_path):
    service=TushareImportService(Settings(state_dir=tmp_path))
    job=service.plan({'datasets':['index_daily'],'codes':['000300.SH','000001.CSI','801010.SI'],
                      'start_date':'2026-10-16','end_date':'2026-10-16'})
    request=job['plan']['request']
    assert 'sw_daily' in request['datasets'] and 'sw_daily' in job['plan']['source_contracts']
    standard=list(service._requests('index_daily',request,[],[]))
    shenwan=list(service._requests('sw_daily',request,[],[]))
    assert {r['ts_code'] for r in standard}=={'000300.SH','000001.CSI'}
    assert {r['ts_code'] for r in shenwan}=={'801010.SI'}
    class SelectedTarget:
        def query(self,*args,**kwargs):
            return {'rows':[{'security_code':'000001.SZ','asset_type':'stock'},
                            {'security_code':'000001.CSI','asset_type':'index'}],'next':None}
    class UnselectedTarget:
        def query(self,*args,**kwargs):raise AssertionError('unselected PG queried')
    service.pg=UnselectedTarget();service.ch=SelectedTarget()
    job['plan']['request']['targets']=['clickhouse']
    assert service._universe(job,'index')==['000001.CSI']


def test_daily_gap_check_respects_lifecycle_and_does_not_infer_halts():
    from betalens_db_manager.inspection import daily_gaps
    class Target:
        def query(self,name,**kwargs):
            values={'security_dim':[{'security_code':'000001.SZ','asset_type':'stock','exchange':'SZSE','list_date':'2026-10-15','delist_date':None}],
                    'trade_calendar':[{'exchange':'SZSE','trade_date':f'2026-10-{day}','is_open':1 if day!='17' else 0} for day in ('14','15','16','17')],
                    'market_daily_raw':[{'security_code':'000001.SZ','asset_type':'stock','trade_date':'2026-10-15'}]}
            return {'rows':values[name],'next':None}
    result=daily_gaps(Target(),'market_daily_raw','000001.SZ','2026-10-14','2026-10-17')
    assert result['summary'][0]['expected_rows']==2 and result['summary'][0]['missing_rows']==1
    assert str(result['gaps'][0]['trade_date'])=='2026-10-16'
    assert '不能据此认定停牌' in result['summary'][0]['note']


@pytest.mark.skipif(os.name=='nt',reason='POSIX flock restart handover')
def test_worker_waits_for_previous_worker_shutdown(tmp_path):
    import fcntl
    from threading import Timer
    from betalens_db_manager.worker import worker_lock
    service=TushareImportService(Settings(state_dir=tmp_path))
    with (tmp_path/'worker.lock').open('a+') as previous:
        fcntl.flock(previous,fcntl.LOCK_EX|fcntl.LOCK_NB)
        release=Timer(.1,lambda:fcntl.flock(previous,fcntl.LOCK_UN));release.start()
        try:
            with worker_lock(service):assert release.finished.is_set()
        finally:release.join()
