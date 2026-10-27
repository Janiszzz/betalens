from types import SimpleNamespace
from betalens_db_manager.deployment_preview import migration_preview
from betalens_db_manager.canonical import CanonicalTable


def test_read_only_deployment_inventory_has_no_ddl_or_data_side_effects():
    spec=CanonicalTable('inventory_probe','盘点测试','说明','代码一行',
        ({'name':'code','type':'text','nullable':False},{'name':'number','type':'integer','nullable':True}),('code',),'custom')
    pg={'exists':True,'keys':['code'],'fields':[{'name':'code','physical_type':'text','nullable':False},{'name':'number','physical_type':'integer','nullable':True}]}
    ch={'exists':True,'keys':['code'],'fields':[{'name':'code','physical_type':'String'},{'name':'number','physical_type':'Nullable(Int64)'}]}
    registry=SimpleNamespace(catalog=lambda:{spec.name:spec},spec=lambda name:spec)
    service=SimpleNamespace(registry=registry,schema=SimpleNamespace(inspect=lambda name,target:pg if target=='postgres' else ch),
        ch=SimpleNamespace(write_capability=lambda:{'available':False}),safe_error=str)
    report=migration_preview(service)
    assert report['read_only']
    assert [r['status'] for r in report['tables']]==['migration_required','ready']
    assert report['tables'][0]['difference']['changed']==[{'column':'number','from':'integer','to':'bigint'}]
