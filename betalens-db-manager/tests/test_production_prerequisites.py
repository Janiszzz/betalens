from psycopg2 import sql
import pytest
from betalens_db_manager.canonical import registry_context
from test_manager import manager,live_manager


def test_endpoint_selection_is_frozen_without_editing_user_yaml(tmp_path):
    service=manager(tmp_path);path=service.mappings.path('fundamental_periodic');before=path.read_bytes()
    plan=service.compile(dict(tables=['fundamental_periodic'],endpoints=['income'],codes=['000001.SZ'],asset_types=['stock'],start_date='2024-12-31',end_date='2024-12-31'))
    assert plan['request']['endpoints']==['income']
    assert {b['endpoint'] for b in plan['mapping_snapshots']['fundamental_periodic']['bindings']}=={'income'}
    assert {r['endpoint'] for r in plan['requests']}=={'income'}
    assert before==path.read_bytes()
    with pytest.raises(ValueError):service.compile(dict(tables=['fundamental_periodic'],endpoints=['income','does_not_exist'],codes=['000001.SZ'],start_date='2024-12-31',end_date='2024-12-31'))


def test_stock_basic_custom_status_does_not_drop_security_filter(tmp_path):
    service=manager(tmp_path)
    request=dict(tables=['stock_profile'],endpoints=['stock_basic'],codes=['000001.SZ'],asset_types=['stock'],parameters={'stock_basic':{'list_status':'L'}},start_date='2026-10-23',end_date='2026-10-23')
    plan=service.compile(request)
    assert [r['parameters'] for r in plan['requests']]==[{'ts_code':'000001.SZ','list_status':'L'}]
    request['parameters']['stock_basic']['ts_code']='000002.SZ'
    with pytest.raises(ValueError,match='证券范围'):service.compile(request)


@pytest.mark.parametrize('table,endpoint', [('fundamental_periodic','fina_indicator'),('industry_scheme_dim','ci_index_member')])
def test_single_interface_declares_generated_business_keys(tmp_path,table,endpoint):
    service=manager(tmp_path)
    plan=service.compile(dict(tables=[table],endpoints=[endpoint],codes=['000001.SZ'],asset_types=['stock'],start_date='2004-01-01',end_date='2026-10-23',parameters={endpoint:{'period':'20241231'} if endpoint=='fina_indicator' else {'is_new':'Y'}}))
    assert plan['requests'] and {b['endpoint'] for b in plan['mapping_snapshots'][table]['bindings']}=={endpoint}


def test_archive_reaudit_keeps_selected_endpoints_and_updates_processor_version(tmp_path):
    from test_mapping_lineage_flow import cache_requests
    from betalens_db_manager.util import dumps
    from betalens_db_manager.processors import VERSION
    service=manager(tmp_path)
    original=service.plan(dict(tables=['fundamental_periodic'],endpoints=['income'],codes=['000001.SZ'],asset_types=['stock'],start_date='2024-12-31',end_date='2024-12-31'))
    cache_requests(service,original,lambda endpoint,params:[dict(ts_code='000001.SZ',ann_date='20250328',end_date='20241231',report_type='1',comp_type='1',revenue='100')])
    old=service.state.job(original['id'])['plan'];old['processor_version']='registered-processors-v2'
    with service.state.connect() as conn:conn.execute('UPDATE jobs SET plan=? WHERE id=?',(dumps(old),original['id']))
    child=service.replay_archive(original['id'])
    assert child['plan']['parent_task_id']==original['id']
    assert child['plan']['processor_version']==VERSION
    assert {b['endpoint'] for b in child['plan']['mapping_snapshots']['fundamental_periodic']['bindings']}=={'income'}
    assert child['status']=='downloaded'
    rows,rejects=service._build_canonical(child)
    assert len(rows['fundamental_periodic'])==1 and not rejects


def test_financial_period_is_checked_as_exact_business_date():
    from betalens_db_manager.response_validation import validate_response
    from betalens_db_manager.catalog import dataset
    contract=dataset('income').as_dict();fields=[f['name'] for f in contract['fields']]
    values={'ts_code':'000001.SZ','end_date':'20240930','ann_date':'20241020','report_type':'1','comp_type':'1'}
    request=dict(source_id='tushare',endpoint='income',parameters={'ts_code':'000001.SZ','period':'20241231'})
    errors=validate_response({'request':{'codes':['000001.SZ']},'mapping_snapshots':{}},request,contract,{'fields':fields,'items':[[values.get(f) for f in fields]]})
    assert any(e['code']=='date_scope' for e in errors)


def test_pg_safe_integer_widening_keeps_oid_values_indexes_and_grants(live_manager):
    service=live_manager
    service.registry.save(dict(name='integer_widen_probe',zh_name='整数扩容测试',description='保留原表的整数类型扩容',grain='代码一行',keys=['code'],fields=[{'name':'code','type':'text'},{'name':'quantity','type':'integer'}]))
    with registry_context(service.registry):
        first=service.schema.apply(service.schema.plan('integer_widen_probe',['postgres'])['token'])
        assert first['status']=='completed',first
        with service.pg.connect() as conn,conn.cursor() as cur:
            cur.execute(sql.SQL('ALTER TABLE {}.integer_widen_probe ALTER COLUMN quantity TYPE integer').format(sql.Identifier(service.settings.pg_schema)))
            cur.execute(sql.SQL('INSERT INTO {}.integer_widen_probe(code,quantity) VALUES(%s,%s)').format(sql.Identifier(service.settings.pg_schema)),('a',2147483647))
            cur.execute('SELECT oid,relacl FROM pg_class WHERE oid=%s::regclass',(service.settings.pg_schema+'.integer_widen_probe',));before=cur.fetchone()
        plan=service.schema.plan('integer_widen_probe',['postgres'])
        assert not plan['destructive'] and not plan['targets'][0]['requires_rebuild']
        assert plan['targets'][0]['safe_widenings']==[{'column':'quantity','from':'integer','to':'bigint'}]
        result=service.schema.apply(plan['token']);assert result['status']=='completed',result
        assert 'retained_backup' not in result['results'][0]
        with service.pg.connect(readonly=True) as conn,conn.cursor() as cur:
            cur.execute('SELECT oid,relacl FROM pg_class WHERE oid=%s::regclass',(service.settings.pg_schema+'.integer_widen_probe',));assert cur.fetchone()==before
            cur.execute(sql.SQL('SELECT quantity,pg_typeof(quantity)::text FROM {}.integer_widen_probe WHERE code=%s').format(sql.Identifier(service.settings.pg_schema)),('a',));assert cur.fetchone()==(2147483647,'bigint')
