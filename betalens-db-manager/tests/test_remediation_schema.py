"""F06/F07 regression checks without touching configured business databases."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from betalens_db_manager.canonical import CanonicalTable
from betalens_db_manager.comparison import compare, row_hash, typed
from betalens_db_manager.integer_contract import INT64_MAX, INT64_MIN, validate_int64
from betalens_db_manager.schema_contract import compare_structure
from betalens_db_manager.schema_manager import SchemaManager
from betalens_db_manager.storage import pg_type
from betalens_db_manager.util import dumps
from test_manager import live_manager  # isolated databases; skipped unless explicitly enabled


def example_spec(*, required=False, extra=None):
    fields=[{'name':'code','type':'text','nullable':False},
            {'name':'amount','type':'integer','nullable':not required}]
    if extra:fields.append(extra)
    return CanonicalTable('schema_probe','结构测试','结构测试','代码一行',tuple(fields),('code',))


@pytest.mark.parametrize('value', [str(INT64_MIN), str(INT64_MAX), '2147483648', '-2147483649'])
def test_integer_contract_accepts_exact_int64_and_serializes_large_values(value):
    number=validate_int64(value,'amount')
    assert number==int(value)
    assert typed(example_spec(),{'code':'A','amount':value})['amount']==number
    if abs(number)>9007199254740991:
        assert dumps({'amount':number})=='{"amount":"'+value+'"}'


@pytest.mark.parametrize('value', [str(INT64_MIN-1),str(INT64_MAX+1),'NaN','Infinity','1.2',True,float(2**53)])
def test_integer_contract_rejects_invalid_or_imprecise_input(value):
    with pytest.raises(ValueError):validate_int64(value,'amount')


def test_comparison_checks_include_skipped_keys_without_changing_signature():
    spec=example_spec()
    row={'code':'A','amount':2}
    result=compare(spec,[row],[row],'fill_missing')
    assert result['counts']['skipped']==1 and result['changes']==[]
    assert result['checks']==[{'row_key':result['checks'][0]['row_key'],
                               'key':{'code':'A'},'before_hash':row_hash(spec,row)}]


def test_pg_and_ch_structure_comparison_detects_integer_nullable_and_keys():
    spec=example_spec(required=True)
    pg={'exists':True,'keys':['code'],'fields':[
        {'name':'code','physical_type':'text','nullable':False},
        {'name':'amount','physical_type':'integer','nullable':True}]}
    diff=compare_structure(spec,pg,'postgres')
    assert pg_type(spec.fields[1])=='bigint'
    assert diff['changed']==[{'column':'amount','from':'integer','to':'bigint'}]
    assert diff['nullable_changes'][0]['column']=='amount' and not diff['ready']
    pg['fields'][1].update(physical_type='bigint',nullable=False)
    assert compare_structure(spec,pg,'postgres')['ready']
    ch={'exists':True,'keys':['code'],'fields':[
        {'name':'code','physical_type':'String'},
        {'name':'amount','physical_type':'Nullable(Int32)'}]}
    diff=compare_structure(spec,ch,'clickhouse')
    assert diff['changed'] and diff['nullable_changes'] and not diff['ready']
    ch['fields'][1]['physical_type']='Int64'
    assert compare_structure(spec,ch,'clickhouse')['ready']
    ch['keys']=['amount']
    assert compare_structure(spec,ch,'clickhouse')['key_change']


def test_pg_numeric_precision_is_part_of_contract():
    spec=example_spec(extra={'name':'price','type':'numeric','precision':28,'scale':8})
    actual={'exists':True,'keys':['code'],'fields':[
        {'name':'code','physical_type':'text','nullable':False},
        {'name':'amount','physical_type':'bigint','nullable':True},
        {'name':'price','physical_type':'numeric','precision':38,'scale':12,'nullable':True}]}
    assert compare_structure(spec,actual,'postgres')['changed'][0]['column']=='price'
    actual['fields'][2].update(precision=28,scale=8)
    assert compare_structure(spec,actual,'postgres')['ready']


class FakeState:
    def __init__(self):self.ledger={}
    def ledger_put(self,table,key,value):self.ledger.setdefault(table,{})[key]=value
    def ledger_rows(self,table):return list(self.ledger.get(table,{}).values())
    def jobs(self,limit=1000):return []


def fake_manager(spec):
    state=FakeState()
    definition={'name':spec.name,'version':'fixture-v1'}
    registry=SimpleNamespace(refresh=lambda:None,errors={},spec=lambda name:spec,
                             get=lambda name:definition,record=lambda name:{'definition':definition})
    pg=SimpleNamespace(settings=SimpleNamespace(pg_schema='isolated'))
    ch=SimpleNamespace(table=lambda name:f'`isolated`.`{name}`',
                       _type=lambda field,key=False:('Int64' if key or not field.get('nullable',True) else 'Nullable(Int64)'),
                       ddl=lambda table:'CREATE TABLE')
    service=SimpleNamespace(registry=registry,state=state,pg=pg,ch=ch,
                            safe_error=lambda exc:str(exc))
    return SchemaManager(service)


@pytest.mark.parametrize('target', ['postgres','clickhouse'])
def test_plan_blocks_required_column_on_populated_table(target,monkeypatch):
    spec=example_spec(extra={'name':'required_new','type':'integer','nullable':False})
    manager=fake_manager(spec)
    actual={'exists':True,'keys':['code'],'fields':[
        {'name':'code','physical_type':'text' if target=='postgres' else 'String','nullable':False},
        {'name':'amount','physical_type':'bigint' if target=='postgres' else 'Nullable(Int64)','nullable':True}]}
    monkeypatch.setattr(manager,'inspect',lambda name,key:actual)
    monkeypatch.setattr(manager,'_row_count',lambda name,key:3)
    plan=manager.plan(spec.name,[target])
    item=plan['targets'][0]
    assert item['added']==['required_new'] and item['row_count']==3 and item['blocked']
    with pytest.raises(ValueError,match='可审计回填'):
        manager._apply(plan['token'])
    monkeypatch.setattr(manager,'_row_count',lambda name,key:0)
    empty=manager.plan(spec.name,[target])['targets'][0]
    assert not empty['blocked'] and 'ADD COLUMN' in empty['ddl'][0]
    assert ('NOT NULL' if target=='postgres' else 'Int64') in empty['ddl'][0]


@pytest.mark.parametrize('target', ['postgres','clickhouse'])
def test_plan_counts_nulls_before_tightening(target,monkeypatch):
    spec=example_spec(required=True)
    manager=fake_manager(spec)
    actual={'exists':True,'keys':['code'],'fields':[
        {'name':'code','physical_type':'text' if target=='postgres' else 'String','nullable':False},
        {'name':'amount','physical_type':'bigint' if target=='postgres' else 'Nullable(Int64)','nullable':True}]}
    monkeypatch.setattr(manager,'inspect',lambda name,key:actual)
    monkeypatch.setattr(manager,'_row_count',lambda name,key:4)
    monkeypatch.setattr(manager,'_null_count',lambda name,key,column:2)
    item=manager.plan(spec.name,[target])['targets'][0]
    assert item['null_counts']=={'amount':2} and item['blocked']
    with pytest.raises(ValueError,match='历史空值'):
        manager._apply(manager.state.ledger_rows('schema_plan')[0]['token'])
    monkeypatch.setattr(manager,'_null_count',lambda name,key,column:0)
    safe=manager.plan(spec.name,[target])['targets'][0]
    assert not safe['blocked'] and not safe['requires_rebuild']
    assert ('SET NOT NULL' if target=='postgres' else 'MODIFY COLUMN') in safe['ddl'][0]


def test_deploy_verification_cannot_mark_still_drifted_target_ready(monkeypatch):
    spec=example_spec()
    manager=fake_manager(spec)
    actual={'exists':True,'keys':['code'],'fields':[
        {'name':'code','physical_type':'text','nullable':False},
        {'name':'amount','physical_type':'bigint','nullable':False}]}
    monkeypatch.setattr(manager,'inspect',lambda name,key:actual)
    monkeypatch.setattr(manager,'_alter_nullable',lambda *args:None)
    manager.service.pg.install=lambda **kwargs:None
    plan=manager.plan(spec.name,['postgres'])
    assert 'DROP NOT NULL' in plan['targets'][0]['ddl'][0]
    report=manager._apply(plan['token'])
    assert report['status']=='partial'
    assert report['results'][0]['status']=='failed'
    saved=manager.state.ledger['table_definition'][spec.name]
    assert saved['deployed']['postgres']['status']=='drifted'


def test_clickhouse_type_rebuild_requires_write_isolation(monkeypatch):
    spec=example_spec()
    manager=fake_manager(spec)
    actual={'exists':True,'keys':['code'],'fields':[
        {'name':'code','physical_type':'String'},
        {'name':'amount','physical_type':'Nullable(Int32)'}]}
    monkeypatch.setattr(manager,'inspect',lambda name,key:actual)
    plan=manager.plan(spec.name,['clickhouse'],destructive=True)
    assert plan['targets'][0]['requires_rebuild']
    with pytest.raises(ValueError,match='受控写入维护窗口'):
        manager._apply(plan['token'])


def test_optional_isolated_pg_int64_and_nullable_migration(live_manager):
    """The fixture creates a unique disposable database and is opt-in only."""
    from psycopg2 import sql
    from betalens_db_manager.canonical import registry_context
    from betalens_db_manager.storage import relation

    service=live_manager
    definition={'name':'schema_probe','zh_name':'结构测试','description':'仅隔离测试',
                'grain':'代码一行','keys':['code'],'fields':[
                    {'name':'code','type':'text'}, {'name':'amount','type':'integer'}]}
    service.registry.save(definition)
    with registry_context(service.registry):
        plan=service.schema.plan('schema_probe',['postgres'])
        assert service.schema.apply(plan['token'])['status']=='completed'
        actual=service.schema.inspect('schema_probe','postgres')
        assert compare_structure(service.registry.spec('schema_probe'),actual,'postgres')['ready']
        assert next(f for f in actual['fields'] if f['name']=='amount')['physical_type']=='bigint'
        with service.pg.connect() as conn,conn.cursor() as cur:
            cur.execute(sql.SQL('INSERT INTO {} (code,amount) VALUES (%s,%s),(%s,%s)').format(
                relation(service.settings,'schema_probe')),
                ('max',INT64_MAX,'null',None))
        doc=service.registry.document('schema_probe');updated=doc['definition']
        next(f for f in updated['fields'] if f['name']=='amount')['nullable']=False
        service.registry.save(updated,doc['file_hash'])
        tightening=service.schema.plan('schema_probe',['postgres'])['targets'][0]
        assert tightening['null_counts']=={'amount':1} and tightening['blocked']
        updated=service.registry.document('schema_probe')['definition']
        updated['fields'].append({'name':'required_new','type':'integer','nullable':False})
        service.registry.save(updated)
        adding=service.schema.plan('schema_probe',['postgres'])['targets'][0]
        assert any('required_new' in message for message in adding['blocked'])


def test_optional_int64_boundary_values_round_trip_in_both_targets(live_manager):
    from betalens_db_manager.canonical import registry_context
    from betalens_db_manager.comparison import typed
    service=live_manager
    definition={'name':'integer_round_trip','zh_name':'整数往返','description':'精确Int64边界','grain':'代码一行',
        'keys':['code'],'fields':[{'name':'code','type':'text'},{'name':'number','type':'integer'}]}
    service.registry.save(definition)
    with registry_context(service.registry):
        assert service.schema.apply(service.schema.plan('integer_round_trip')['token'])['status']=='completed'
        spec=service.registry.spec('integer_round_trip')
        job=service.state.create_job({'request':{'targets':['postgres','clickhouse']}},'isolated')['id']
        expected={'max':INT64_MAX,'min':-(2**63),'above_js':9007199254740993}
        rows=[typed(spec,{'code':code,'number':str(value),'import_task_id':job}) for code,value in expected.items()]
        for name,target in [('postgres',service.pg),('clickhouse',service.ch)]:
            manifest=target.compare(spec,rows,'refresh_source')
            service.state.stage_commit(job,name,spec.name,manifest)
            target.commit_rows(job,name,spec,rows,'refresh_source',manifest)
            service.state.finish_commit(job,name,spec.name)
            assert {r['code']:r['number'] for r in target.query(spec.name)['rows']}==expected
