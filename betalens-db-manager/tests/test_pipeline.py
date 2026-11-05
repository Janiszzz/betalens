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
from betalens_db_manager.manager_kernel import ManagerKernel, normalize
from betalens_db_manager.manager import DatabaseManagerService
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
    service = ManagerKernel(Settings(state_dir=tmp_path))
    with TestClient(create_app(service)) as client:
        assert client.get('/api/schedules').status_code==410
        assert client.post('/api/schedules',json={'enabled':True}).status_code==410
    job=service.state.create_job({'pipeline_version':2,'request':{'targets':['postgres']}},service.settings.fingerprint)
    service.state.enqueue(job['id'],'pipeline_run');service.state.cancel(job['id'])
    assert service.state.job(job['id'])['tasks'][0]['status']=='cancelled'


def test_web_can_open_before_database_and_rejects_foreign_origin(tmp_path):
    service = ManagerKernel(Settings(state_dir=tmp_path, token="not-for-client"))
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
    service = DatabaseManagerService(settings, FakeSource())
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


def test_fields_are_mapped_by_name_and_unknown_fields_block():
    from betalens_db_manager.manager_kernel import normalize_source
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
    from betalens_db_manager.manager_kernel import normalize_source
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
    service=ManagerKernel(Settings(state_dir=tmp_path))
    with (tmp_path/'worker.lock').open('a+') as previous:
        fcntl.flock(previous,fcntl.LOCK_EX|fcntl.LOCK_NB)
        release=Timer(.1,lambda:fcntl.flock(previous,fcntl.LOCK_UN));release.start()
        try:
            with worker_lock(service):assert release.finished.is_set()
        finally:release.join()
