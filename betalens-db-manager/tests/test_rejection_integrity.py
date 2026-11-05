from copy import deepcopy
from datetime import date

import pytest

from betalens_db_manager.canonical import canonical,registry_context
from betalens_db_manager.comparison import typed
from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.industry_codes import normalize_parents
from betalens_db_manager.pipeline import Pipeline
from betalens_db_manager.response_validation import validate_archives
from betalens_db_manager.task_progress import freeze,summary
from test_manager import manager
from test_retrieval import task


@pytest.mark.parametrize('code',['HSSCITI.HI','IBOV.OTH','SPX.GI','000300.SH','Au99.99.SGE','spx.GI'])
def test_etf_benchmark_preserves_qualified_foreign_identifier(code):
    row={'security_code':'159010.SZ','index_code':code}
    assert typed(canonical('etf_profile'),row)['index_code']==code
    with pytest.raises(ValueError):typed(canonical('etf_profile'),{**row,'security_code':'00700.HK'})
    if not code.endswith('.SH'):
        with pytest.raises(ValueError):typed(canonical('index_membership'),{'index_code':code})


@pytest.mark.parametrize('code',['SPX',"SPX.GI';DROP",'../SPX.GI','SPX GI','SPX.gi','A'*65+'.HI',''])
def test_etf_benchmark_still_rejects_unqualified_or_damaged_code(code):
    with pytest.raises(ValueError):typed(canonical('etf_profile'),{'security_code':'159010.SZ','index_code':code})


def industry_rows():
    rows = [
        {'scheme_code':'SW2021','industry_code':'801730.SI','classification_code':'630000','level':'L1',
         'parent_code':'0','industry_name':'电力设备','source_endpoint':'index_classify','source_record_hash':'root-wire'},
        {'scheme_code':'SW2021','industry_code':'801735.SI','classification_code':'630500','level':'L2',
         'parent_code':'630000','industry_name':'电池','source_endpoint':'index_classify','source_record_hash':'child-wire'},
        {'scheme_code':'SW2021','industry_code':'801735.SI','level':'L2','parent_code':'801730.SI',
         'industry_name':'电池','source_endpoint':'index_member_all','source_record_hash':'member-wire'},
    ]
    for row in rows:row['source_id']='tushare'
    return rows


def test_industry_parent_uses_unique_archived_alias_and_retains_source_hash(tmp_path):
    rows=industry_rows();before=deepcopy(rows)
    rows[1]['_field_origins']={}
    normalize_parents(rows)
    assert rows[0]['parent_code'] is None
    assert rows[1]['parent_code']==rows[2]['parent_code']=='801730.SI'
    assert rows[1]['classification_code']=='630500'
    assert [r['source_record_hash'] for r in rows]==[r['source_record_hash'] for r in before]
    assert rows[1]['_field_origins']['parent_code']['origins']==[
        'tushare:index_classify:parent_code','tushare:index_classify:industry_code','tushare:index_classify:index_code']
    service=manager(tmp_path)
    with registry_context(service.registry):merged=service._merge(rows,canonical('industry_dim'))
    assert all(r.get('data_quality_status')!='conflicting_source_rows' for r in merged)
    # Actual name differences remain conflicts after namespace normalization.
    rows[2]['industry_name']='另一个行业'
    with registry_context(service.registry):merged=service._merge(rows,canonical('industry_dim'))
    assert any(r.get('data_quality_status')=='conflicting_source_rows' for r in merged)


@pytest.mark.parametrize('damage',['ambiguous','missing','different_scheme','wrong_level','self_parent'])
def test_industry_parent_does_not_guess_missing_or_ambiguous_relationships(damage):
    rows=industry_rows()
    if damage=='ambiguous':rows.append({**rows[0],'industry_code':'801999.SI'})
    elif damage=='missing':rows[1]['parent_code']='999999'
    elif damage=='different_scheme':rows[0]['scheme_code']='SW2014'
    elif damage=='wrong_level':rows[0]['level']='L3';rows[0]['parent_code']=None
    else:rows[1]['parent_code']='630500'
    with pytest.raises(ValueError):normalize_parents(rows)


@pytest.mark.parametrize('override',[{'source_id':'custom'},{'_binding_identity':'tushare:index_classify'}])
def test_industry_normalization_preserves_custom_source_and_explicit_direct_mapping(override):
    rows=[{**r,**override} for r in industry_rows()];before=deepcopy(rows)
    normalize_parents(rows)
    assert rows==before


def test_prepared_split_children_are_copied_with_all_pages_and_empty_end(tmp_path):
    rows=[{'index_code':'000013.CSI','con_code':code,'trade_date':'20261020','weight':'1'}
          for code in ('000001.SZ','00700.HK','BABA')]
    def response(_endpoint,params):
        values=[r for r in rows if params['start_date']<=r['trade_date']<=params['end_date']]
        start=params.get('offset',0)
        return values[start:start+params.get('limit',2)]
    service,source,item=task(tmp_path,'index_weight',{'index_code':'000013.CSI','start_date':'20261020','end_date':'20261021'},response)
    plan=item['plan'];plan['retrieval_policies']['tushare:index_weight']['single_day_offset_probe']=True
    plan['mapping_snapshots']={'index_membership':service.mappings.get('index_membership')}
    plan['canonical_contracts']={'index_membership':service.registry.spec('index_membership').hash}
    plan['request'].update(tables=['index_membership'],start_date='2026-10-20',end_date='2026-10-21')
    service.state.replace_plan(item['id'],plan)
    pipeline=Pipeline(service)
    with TaskDownloader(service,item) as downloader:
        freeze(service,item,{'history_batch_0':plan['requests']},downloader)
        downloader.run()
    parent=service.state.job(item['id'])
    assert not validate_archives(service,parent)
    child_id=pipeline._child(parent,'history_batch_0',plan['requests'])
    child=service.state.job(child_id)
    assert not validate_archives(service,child)
    assert {p['id'] for p in child['slices']}=={p['id'] for p in parent['slices']}
    progress=summary(service,parent,pipeline.store.units(parent['id']))
    assert progress['requests']['processed']==progress['requests']['total']==1
    # The terminal empty page is necessary evidence, even with valid data pages.
    terminal=next(p for p in child['slices'] if p['request']['parameters'].get('offset')==3)
    with service.state.connect() as c:c.execute('delete from slices where job=? and id=?',(child_id,terminal['id']))
    assert any(e['code']=='missing_request' for e in validate_archives(service,service.state.job(child_id)))


def test_readonly_progress_survives_removal_of_historical_source(tmp_path):
    rows=[{'ts_code':'000001.SZ','trade_date':'20261020','adj_factor':'1'}]
    service,_,item=task(tmp_path,'fund_adj',{'ts_code':'000001.SZ','trade_date':'20261020'},
        lambda _endpoint,params:rows[params.get('offset',0):],mode='offset')
    pipeline=Pipeline(service)
    with TaskDownloader(service,item) as downloader:
        freeze(service,item,{'one':item['plan']['requests']},downloader)
        downloader.run()
    service.sources.values.pop('tushare')
    progress=summary(service,service.state.job(item['id']),pipeline.store.units(item['id']))
    assert progress['requests']['processed']==1
