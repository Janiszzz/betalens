from decimal import Decimal
from types import SimpleNamespace
import pytest
from betalens_db_manager.field_lineage import coverage,record_readback,validate_producers
from betalens_db_manager.comparison import compare,row_key
from betalens_db_manager.util import digest
from test_manager import manager,table_definition


def test_every_builtin_column_has_explicit_producer_and_custom_gap_is_visible(tmp_path):
    service=manager(tmp_path)
    for name in service.registry.catalog():
        report=coverage(service,name)
        assert all(row['producers'] for row in report['columns']),name
        assert all(not row['evidence'] for row in report['columns'])
    service.registry.save(table_definition())
    assert any(row['column']=='price' and row['status']=='unmapped' for row in coverage(service,'sample_prices')['columns'])
    for name,column in [('market_daily_raw','adjustment_factor'),('security_code_history','valid_from'),('market_daily_raw','pcf')]:
        row=next(r for r in coverage(service,name)['columns'] if r['column']==column)
        assert row['status'] in {'not_applicable','unavailable'}


def test_producer_injection_and_audit_overwrite_are_rejected(tmp_path):
    service=manager(tmp_path);spec=service.registry.spec('market_daily_raw')
    for rule in [dict(column='close',kind='computed',processor='eval',inputs=[]),dict(column='import_task_id',kind='constant',value='forged'),dict(column='close',kind='computed',processor='daily_average',expression='__import__("os")')]:
        with pytest.raises(ValueError):validate_producers([rule],spec,service.sources)


def test_removed_links_do_not_leave_stale_direct_coverage(tmp_path):
    service=manager(tmp_path);doc=service.mappings.document('etf_profile');value=doc['mapping']
    assert any(p['kind']=='direct' and p['column']=='csname' for p in value['producers'])
    value['bindings'][0]['fields']=[r for r in value['bindings'][0]['fields'] if r['column']!='csname']
    service.mappings.save(value,doc['file_hash'])
    row=next(c for c in coverage(service,'etf_profile')['columns'] if c['column']=='csname')
    assert not row['producers'] and row['status']=='unmapped'


def test_readback_proof_is_per_column_target_and_selected_business_key(tmp_path):
    service=manager(tmp_path);service.registry.save(table_definition());spec=service.registry.spec('sample_prices')
    item={'id':'proof-task','plan':{'mapping_snapshots':{'sample_prices':{'producers':[]}}}}
    before={'code':'a','business_date':'2026-10-13','price':'12'}
    incoming={'code':'a','business_date':'2026-10-13','price':'0','source_id':'first'}
    zero={'code':'b','business_date':'2026-10-13','price':'0'}
    empty={'code':'c','business_date':'2026-10-13','price':None}
    rows=[incoming,zero,empty];change=compare(spec,rows,[before],'fill_missing')
    traces={row_key(spec,r):{'price':{'selected':r['price'] is not None,'source_null':r['price'] is None,'value_hash':digest(Decimal(r['price'])) if r['price'] is not None else digest(None),'origins':['first:prices:price']}} for r in rows}
    service.state.ledger_put('task_column_lineage','proof-task:sample_prices',{'rows':traces})
    # fill_missing preserves a=12; it must not prove incoming source price=0.
    from betalens_db_manager.comparison import typed
    live={row_key(spec,before):typed(spec,before)}
    live.update({c['row_key']:typed(spec,c['after']) for c in change['changes']})
    record_readback(service,item,'postgres',spec,change,live)
    proof=next(p for p in service.state.ledger_rows('field_verification') if p['column']=='price')
    assert proof['target']=='postgres' and proof['verified_count']==1
    assert proof['preserved_count']==1 and proof['source_null_count']==1
    assert proof['verified_origins']=={'first:prices:price':1}
    assert not any(p['target']=='clickhouse' for p in service.state.ledger_rows('field_verification'))
    # Corrupted readback cannot inherit PG proof.
    wrong={key:{**row,'price':Decimal('999')} for key,row in live.items()}
    record_readback(service,item,'clickhouse',spec,change,wrong)
    proof=next(p for p in service.state.ledger_rows('field_verification') if p['column']=='price' and p['target']=='clickhouse')
    assert proof['status']=='mismatch' and proof['mismatch_count']>=1


def test_file_proof_requires_real_archive_normalization_and_target_readback(tmp_path,monkeypatch):
    from betalens_db_manager.files import import_file
    from betalens_db_manager.comparison import typed
    service=manager(tmp_path);service.registry.save(table_definition())
    monkeypatch.setattr(service,'_calendar',lambda:[])
    for target in (service.pg,service.ch):monkeypatch.setattr(target,'query',lambda *a,**k:{'rows':[],'next':None})
    source=tmp_path/'prices.csv';source.write_text('ticker,day,milli\na,20261013,12345.6789\n')
    mapping={'ticker':'code','day':{'column':'business_date','date_format':'%Y%m%d'},'milli':{'column':'price','multiplier':'0.001'}}
    job=import_file(service,'sample_prices',source,mapping,targets=['postgres'])
    rows,rejects=service._build_canonical(job);assert not rejects
    spec=service.registry.spec('sample_prices');row=rows['sample_prices'][0]
    assert row['price']==Decimal('12.34567890')
    before=service.field_audit(job['id']);assert all(r['status']=='verified_response' for r in before)
    diff=compare(spec,[row],[],'fill_missing')
    record_readback(service,job,'postgres',spec,diff,{row_key(spec,row):typed(spec,row)})
    after=service.field_audit(job['id'])
    assert next(r for r in after if r['source_field']=='milli')['status']=='verified_stored'
    assert {p['target'] for p in next(r for r in after if r['source_field']=='milli')['evidence']}=={'postgres'}


def test_global_classification_does_not_inherit_stock_scope_but_explicit_index_filter_is_checked():
    from betalens_db_manager.response_validation import validate_response
    from betalens_db_manager.catalog import dataset
    contract=dataset('index_classify').as_dict()
    request={'source_id':'tushare','endpoint':'index_classify','parameters':{'level':'L1','src':'SW2021'}}
    fields=[f['name'] for f in contract['fields']]
    values={'index_code':'801010.SI','industry_name':'农林牧渔','level':'L1','src':'SW2021'}
    data={'fields':fields,'items':[[values.get(f) for f in fields]]}
    plan={'request':{'codes':['000001.SZ']},'mapping_snapshots':{}}
    assert not validate_response(plan,request,contract,data)
    request['parameters']['index_code']='801020.SI'
    assert any(p['code']=='security_scope' for p in validate_response(plan,request,contract,data))
