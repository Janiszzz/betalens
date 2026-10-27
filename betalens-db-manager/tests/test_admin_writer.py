"""Explicit administrator mode uses the same writer protocol, never a bypass."""
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
import json
import os
from pathlib import Path
import uuid

import pytest

from betalens_db_manager.api import create_app
from betalens_db_manager.canonical import CanonicalTable
from betalens_db_manager.ch_writer import LocalAdminWriter, LocalDedicatedWriter
from betalens_db_manager.clickhouse import ClickHouse
from betalens_db_manager.commit_protocol import NeedsReconciliation
from betalens_db_manager.comparison import row_hash, row_key, typed
from betalens_db_manager.config import Settings
from betalens_db_manager.manager import DatabaseManagerService
from betalens_db_manager.state import State


def grant(user='admin', right='ALL', database=None, role=None):
    return {'user_name':user,'role_name':role,'access_type':right,'database':database,'table':None,'is_partial_revoke':False}


class FakeAdapter:
    _literal=staticmethod(ClickHouse._literal)
    _json_rows=staticmethod(lambda text:[json.loads(line) for line in text.splitlines() if line])

    def __init__(self,settings,grants=None,engine='ReplacingMergeTree',readonly='0',current=None):
        self.settings=settings;self.grants=grants if grants is not None else [grant(settings.ch_user)]
        self.engine=engine;self.readonly=readonly;self.current=current or settings.ch_user
        self.events=[];self.active='0'

    def execute(self,query):
        if 'currentUser()' in query:return self.current
        if 'FROM system.grants' in query:return '\n'.join(json.dumps(row) for row in self.grants)
        if "name='readonly'" in query:return self.readonly
        if 'FROM system.tables' in query:return json.dumps({'name':'example','engine':self.engine})
        if 'FROM system.processes' in query:return self.active
        if 'FROM system.query_log' in query:return '\n'.join(json.dumps(row) for row in self.events)
        raise AssertionError('unexpected query: '+query)


def admin_settings(tmp_path,**values):
    return Settings(config_dir=tmp_path/'config',state_dir=tmp_path/'state',ch_database='test_admin_'+uuid.uuid4().hex[:12],
                    ch_user='admin',ch_password='unit-secret-value',ch_writer_mode='local_admin',**values)


def test_admin_mode_is_explicit_and_disabled_and_dedicated_semantics_remain(tmp_path,monkeypatch):
    settings=admin_settings(tmp_path)
    admin=ClickHouse(settings)
    assert isinstance(admin.write_coordinator,LocalAdminWriter)
    disabled=ClickHouse(replace(settings,ch_writer_mode='disabled'))
    assert disabled.write_coordinator is None and not disabled.write_capability()['available']
    monkeypatch.setattr('requests.post',lambda *args,**kwargs:pytest.fail('disabled business write reached HTTP'))
    with pytest.raises(NeedsReconciliation,match='未接入'):
        with disabled.write_window('commit:blocked'):pass
    dedicated=LocalDedicatedWriter(replace(settings,ch_writer_mode='local_dedicated'))
    with pytest.raises(NeedsReconciliation,match='不能是 ClickHouse 管理员'):
        dedicated.verify(FakeAdapter(settings))
    with pytest.raises(ValueError,match='显式配置'):
        LocalAdminWriter(replace(settings,ch_writer_mode='disabled'))


def test_admin_mode_rejects_remote_and_invalid_modes_and_environment_loads(tmp_path):
    with pytest.raises(ValueError,match='仅支持本机'):
        admin_settings(tmp_path,ch_url='http://192.0.2.1:8123')
    with pytest.raises(ValueError,match='写入模式无效'):
        replace(admin_settings(tmp_path),ch_writer_mode='admin_bypass')
    env=tmp_path/'mode.env';env.write_text('CLICKHOUSE_WRITER_MODE=local_admin\nCLICKHOUSE_URL=http://127.0.0.1:8123\n')
    assert Settings.from_env(env).ch_writer_mode=='local_admin'


def test_admin_capability_checks_actual_identity_and_reports_boundary(tmp_path):
    settings=admin_settings(tmp_path);adapter=FakeAdapter(settings)
    gate=LocalAdminWriter(settings)
    verified=gate.verify(adapter)
    assert verified['administrator'] and verified['writer']=='admin'
    assert verified['external_writes']=='must_be_stopped_during_manager_writes'
    ch=ClickHouse(settings);ch.execute=adapter.execute
    capability=ch.write_capability()
    assert capability['mode']=='local_admin' and capability['status']=='controlled_local_admin'
    assert capability['available'] and 'unit-secret-value' not in json.dumps(capability)


@pytest.mark.parametrize('case', ['identity','password','ordinary','readonly','engine','other_user','other_role'])
def test_admin_verification_fails_closed_for_invalid_inputs(tmp_path,case):
    settings=admin_settings(tmp_path)
    if case=='password':settings=replace(settings,ch_password='')
    adapter=FakeAdapter(settings)
    if case=='identity':adapter.current='someone_else'
    if case=='ordinary':adapter.grants=[grant('admin','INSERT',settings.ch_database)]
    if case=='readonly':adapter.readonly='1'
    if case=='engine':adapter.engine='Distributed'
    if case=='other_user':adapter.grants.append(grant('external_writer','INSERT',settings.ch_database))
    if case=='other_role':adapter.grants.append(grant(None,'INSERT',settings.ch_database,role='writer_role'))
    with pytest.raises(NeedsReconciliation):LocalAdminWriter(settings).verify(adapter)


def test_unrelated_database_writers_do_not_block_admin_mode(tmp_path):
    settings=admin_settings(tmp_path)
    adapter=FakeAdapter(settings,grants=[grant('admin'),grant('browser_writer','INSERT','browser_test_unrelated'),grant('old_writer','ALTER','manager_test_old')])
    assert LocalAdminWriter(settings).verify(adapter)['administrator']


def test_admin_and_dedicated_share_lock_and_pending_query_recovery(tmp_path):
    settings=admin_settings(tmp_path)
    gate=LocalAdminWriter(settings);dedicated=LocalDedicatedWriter(replace(settings,ch_writer_mode='local_dedicated',state_dir=tmp_path/'other_state'))
    assert gate.lock==dedicated.lock and gate.journal==dedicated.journal
    adapter=FakeAdapter(settings)
    try:
        with gate.exclusive(adapter,'commit:original'):
            with pytest.raises(NeedsReconciliation,match='已有写操作'):
                with dedicated.exclusive(adapter,'commit:overlap'):pass
            query_id=gate.before_request('INSERT INTO example VALUES (1)','commit:original')
        with pytest.raises(NeedsReconciliation,match='结果不确定'):
            with gate.exclusive(adapter,'commit:original'):pass
        adapter.events=[{'type':'QueryFinish'}]
        with pytest.raises(NeedsReconciliation,match='先恢复原操作'):
            with gate.exclusive(adapter,'commit:unrelated'):pass
        with gate.exclusive(adapter,'commit:original'):pass
        journal=json.loads(gate.journal.read_text())
        assert journal['query_id']==query_id and not journal['inflight'] and journal['reconciled_at']
        assert 'INSERT INTO' not in gate.journal.read_text()
    finally:
        gate.journal.unlink(missing_ok=True);gate.lock.unlink(missing_ok=True)


@pytest.mark.parametrize('query',['INSERT INTO example VALUES (1)','DELETE FROM example WHERE id=1','EXCHANGE TABLES example AND backup'])
def test_admin_mutations_still_enter_gate_before_http(tmp_path,monkeypatch,query):
    ch=ClickHouse(admin_settings(tmp_path))
    @contextmanager
    def reject(operation):
        raise NeedsReconciliation('admin gate checked')
        yield
    monkeypatch.setattr(ch,'write_window',reject)
    monkeypatch.setattr('requests.post',lambda *args,**kwargs:pytest.fail('administrator gate was bypassed'))
    with pytest.raises(NeedsReconciliation,match='admin gate checked'):ch.execute(query)


def test_api_exposes_selected_mode_and_verified_capability_without_secret(tmp_path):
    settings=admin_settings(tmp_path);service=DatabaseManagerService(settings)
    service.ch.execute=FakeAdapter(settings).execute
    app=create_app(service)
    def handler(path,method='GET'):
        return next(route.endpoint for route in app.routes if getattr(route,'path',None)==path and method in getattr(route,'methods',set()))
    config=handler('/api/config')()
    assert config.status_code==200 and json.loads(config.body)['clickhouse_writer_mode']=='local_admin'
    cap=handler('/api/write-capabilities')()
    assert cap.status_code==200
    row=next(row for row in json.loads(cap.body) if row['target']=='clickhouse')
    assert row['available'] and row['administrator'] and row['mode']=='local_admin'
    assert settings.ch_password.encode() not in cap.body+config.body
    with pytest.raises(ValueError,match='写入模式无效'):
        handler('/api/config/test','POST')({'CLICKHOUSE_WRITER_MODE':'admin_bypass'})


@pytest.mark.skipif(os.environ.get('BETALENS_INFRA_INTEGRATION')!='1',reason='requires explicit disposable CH integration')
@pytest.mark.parametrize('monthly_history',[False,True])
def test_isolated_admin_mode_commit_lock_readback_and_table_exchange_rollback(tmp_path,monthly_history):
    base=Settings.from_env(os.environ.get('BETALENS_INFRA_ENV','/home/janiszzz/betalens-infra/tushare.private.env'))
    name='manager_test_admin_'+uuid.uuid4().hex[:12]
    assert name.startswith('manager_test_admin_')
    settings=replace(base,ch_database=name,ch_writer_mode='local_admin',state_dir=tmp_path/'state',config_dir=tmp_path/'config')
    ch=ClickHouse(settings);ch.state=State(settings.state_dir)
    second=ClickHouse(replace(settings,state_dir=tmp_path/'other'))
    cleanup=ClickHouse(replace(settings,ch_writer_mode='disabled'))
    spec=CanonicalTable('admin_probe','管理员模式测试','仅隔离库','代码一行',(
        {'name':'code','type':'text','nullable':False},
        {'name':'event_date','type':'date','nullable':False},
        {'name':'price','type':'numeric','precision':28,'scale':8,'nullable':True},
        {'name':'import_task_id','type':'text','nullable':True}),('code',))
    try:
        assert ch.write_capability()['available']
        ch.execute(f'CREATE DATABASE `{name}`')
        ch.execute(ch.ddl(spec))
        with ch.write_window('commit:hold'):
            with pytest.raises(NeedsReconciliation,match='已有写操作'):
                with second.write_window('commit:overlap'):pass
        first=typed(spec,{'code':'a','event_date':'2026-01-01','price':'12.34567890','import_task_id':'first'})
        original=ch.compare(spec,[first],'refresh_source')
        assert ch.commit_rows('first','clickhouse',spec,[first],'refresh_source',original)['inserted']==1
        actual=ch._find_existing(spec,[first])[('a',)]
        assert actual['price']==Decimal('12.34567890')
        other=typed(spec,{'code':'b','event_date':'2026-02-01','price':'5','import_task_id':'second'})
        second.commit_rows('second','clickhouse',spec,[other],'refresh_source',second.compare(spec,[other],'refresh_source'))
        history=[]
        if monthly_history:
            history=[typed(spec,{'code':f'history_{i}','event_date':f'{1990+i//12}-{i%12+1:02d}-01','price':'3','import_task_id':'history'}) for i in range(101)]
            ch.commit_rows('history','clickhouse',spec,history,'refresh_source',ch.compare(spec,history,'refresh_source'))
        # An earlier ALTER can leave physical column order unlike the contract.
        ch.execute(f'ALTER TABLE {ch.table(spec.name)} MODIFY COLUMN price Nullable(Decimal(28,8)) AFTER import_task_id')
        audit={'after_value':actual,'before_value':None,'after_hash':row_hash(spec,actual),'row_key':row_key(spec,actual)}
        inverse=ch.apply_rollback(spec,[audit],'inverse_admin_probe')
        assert len(inverse)==1
        assert ch.execute(f"SELECT code FROM {ch.table(spec.name)} FINAL WHERE code IN ('a','b') ORDER BY code FORMAT TSVRaw")=='b'
        assert int(ch.execute(f'SELECT count() FROM {ch.table(spec.name)} FINAL'))==1+len(history)
        checkpoint=ch.state.ledger_rows('ch_operation')[0]
        assert checkpoint['status']=='exchanged_backup_retained'
        journal=json.loads(ch.write_coordinator.journal.read_text())
        assert journal['operation']=='rollback:inverse_admin_probe' and not journal['inflight']
        assert ch.write_capability()['administrator']
    finally:
        cleanup.execute(f'DROP DATABASE IF EXISTS `{name}`')
        ch.write_coordinator.journal.unlink(missing_ok=True)
        ch.write_coordinator.lock.unlink(missing_ok=True)
