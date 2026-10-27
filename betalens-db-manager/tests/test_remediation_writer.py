"""F01/F02 checks against disposable PG/CH databases and temporary CH users.

The imported fixture creates uniquely named manager_test_* databases only when
BETALENS_INFRA_INTEGRATION=1.  No configured business table is touched.
"""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import secrets
import uuid

import pytest
from psycopg2 import sql

from betalens_db_manager.canonical import registry_context
from betalens_db_manager.clickhouse import ClickHouse
from betalens_db_manager.commit_protocol import NeedsReconciliation
from betalens_db_manager.comparison import typed
from betalens_db_manager.config import Settings
from betalens_db_manager.storage import relation
from test_manager import live_manager, mock_api, table_definition


def prepare_job(service, mock_api, target):
    service.registry.save(table_definition())
    with registry_context(service.registry):
        plan=service.schema.plan('sample_prices',[target])
        deployed=service.schema.apply(plan['token'])
        assert deployed['status']=='completed',deployed
    service.sources.save({'source_id':'writer_fixture','provider':'http','base_url':mock_api,
        'endpoints':{'prices':{'path':'prices','strategy':'range','date_field':'day',
            'keys':['ticker','day'],'fields':[{'name':'ticker','type':'text'},
                {'name':'day','type':'date'},{'name':'price_milli','type':'numeric'}]}}})
    service.mappings.save({'format_version':1,'table':'sample_prices','bindings':[
        {'source_id':'writer_fixture','endpoint':'prices','mapping_mode':'direct','fields':[
            {'source_field':'ticker','column':'code'},
            {'source_field':'day','column':'business_date','date_format':'%Y/%m/%d'},
            {'source_field':'price_milli','column':'price','multiplier':'0.001'}]}]},None)
    job=service.plan({'tables':['sample_prices'],'start_date':'2026-10-13',
                      'end_date':'2026-10-13','targets':[target],
                      'mode':'refresh_source'})
    service.download(job['id'])
    return job['id']


def test_pg_stage_race_preserves_external_write(live_manager,mock_api,monkeypatch):
    service=live_manager
    job=prepare_job(service,mock_api,'postgres')
    with service.pg.connect() as conn,conn.cursor() as cur:
        cur.execute(sql.SQL('INSERT INTO {} (code,business_date,price) VALUES (%s,%s,%s)').format(
            relation(service.settings,'sample_prices')),('a','2026-10-13',Decimal('10')))
    preview=service.preview(job)
    assert not preview['blocking'] and not preview['rejected'],preview
    original=service.state.stage_commit
    staged=[]
    def race(*args,**kwargs):
        result=original(*args,**kwargs)
        staged.append(args)
        with service.pg.connect() as conn,conn.cursor() as cur:
            cur.execute(sql.SQL('UPDATE {} SET price=%s WHERE code=%s').format(
                relation(service.settings,'sample_prices')),(Decimal('77'),'a'))
        return result
    monkeypatch.setattr(service.state,'stage_commit',race)
    result=service.commit(job,preview['token'])
    assert staged and result['status']=='partial',result
    with service.pg.connect(readonly=True) as conn,conn.cursor() as cur:
        cur.execute(sql.SQL('SELECT price FROM {} WHERE code=%s').format(
            relation(service.settings,'sample_prices')),('a',))
        assert cur.fetchone()[0]==Decimal('77')
    intent=service.state.table_commit(job,'postgres','sample_prices')
    assert Decimal(str(intent['comparison']['changes'][0]['before']['price']))==Decimal('10')
    assert not service.state.audits(job,'postgres')


def test_two_ch_adapters_share_lock_and_unknown_query_blocks(live_manager,tmp_path):
    service=live_manager
    first=service.ch
    second=ClickHouse(replace(service.settings,state_dir=tmp_path/'other_state'))
    assert first.write_coordinator.lock==second.write_coordinator.lock
    journal=tmp_path/'uncertain_query.json'
    first.write_coordinator.journal=journal
    second.write_coordinator.journal=journal
    with first.write_window('first:hold'):
        with pytest.raises(NeedsReconciliation,match='已有写操作'):
            with second.write_window('second:overlap'):
                pytest.fail('a second writer entered the critical section')
    with second.write_window('second:after'):
        assert second.execute('SELECT 1 FORMAT TSVRaw')=='1'
    with first.write_window('first:unknown'):
        first.write_coordinator.before_request('INSERT INTO isolated VALUES (1)','first:unknown')
    try:
        with pytest.raises(NeedsReconciliation,match='结果不确定'):
            with second.write_window('second:after_unknown'):
                pytest.fail('an unresolved query must block all subsequent writes')
    finally:
        journal.unlink(missing_ok=True)


def test_ch_read_only_account_and_rollback_preserve_other_writer(live_manager,mock_api,tmp_path):
    service=live_manager
    job=prepare_job(service,mock_api,'clickhouse')
    preview=service.preview(job)
    assert not preview['blocking'] and not preview['rejected'],preview
    result=service.commit(job,preview['token'])
    assert result['status']=='completed',result
    second=ClickHouse(replace(service.settings,state_dir=tmp_path/'other_state'))
    spec=service.registry.spec('sample_prices')
    other=typed(spec,{'code':'b','business_date':'2026-10-13','price':'55'})
    manifest=second.compare(spec,[other],'refresh_source')
    assert manifest['counts']['inserted']==1
    second.commit_rows('independent_writer','clickhouse',spec,[other],'refresh_source',manifest)
    with registry_context(service.registry):
        before={row['code']:row for row in service.ch.query('sample_prices')['rows']}
    assert set(before)=={'a','b'} and before['b']['price']==Decimal('55')

    base=Settings.from_env('/home/janiszzz/betalens-infra/tushare.private.env')
    admin=ClickHouse(replace(base,ch_database=service.settings.ch_database,ch_writer_mode='disabled'))
    reader_name='test_reader_'+uuid.uuid4().hex[:12]
    reader_password=secrets.token_urlsafe(32)
    address=admin.execute('SELECT toString(address) FROM system.processes WHERE query_id=currentQueryID() FORMAT TSVRaw')
    admin.execute(f'CREATE USER `{reader_name}` IDENTIFIED WITH sha256_password BY {admin._literal(reader_password)} HOST IP {admin._literal(address)}')
    try:
        admin.execute(f'GRANT SELECT ON `{service.settings.ch_database}`.* TO `{reader_name}`')
        reader=ClickHouse(replace(service.settings,ch_user=reader_name,ch_password=reader_password,ch_writer_mode='disabled'))
        assert reader.execute(f'SELECT count() FROM {reader.table(spec.name)} FINAL')=='2'
        with pytest.raises(RuntimeError,match='ClickHouse HTTP'):
            reader.execute(f'INSERT INTO {reader.table(spec.name)} (code,business_date,price) VALUES (\'c\',\'2026-10-13\',1)')
    finally:
        admin.execute(f'DROP USER IF EXISTS `{reader_name}`')

    rollback=service.rollback(job,['clickhouse'])
    assert rollback['results'][0]['status']=='completed',rollback
    with registry_context(service.registry):
        after={row['code']:row for row in service.ch.query('sample_prices')['rows']}
    assert set(after)=={'b'} and after['b']['price']==before['b']['price']
