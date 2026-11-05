from datetime import date
from decimal import Decimal

import pytest

from betalens_db_manager.canonical import canonical
from betalens_db_manager.comparison import compare
from betalens_db_manager.commit_protocol import NeedsReconciliation,remaining_changes
from betalens_db_manager.normalization_context import normalization_inputs
from betalens_db_manager.pipeline_options import light_normalization
from test_manager import cached_part,manager,table_definition


def test_derived_tables_reuse_one_source_conversion_and_keep_adjustment(tmp_path,monkeypatch):
    import betalens_db_manager.manager as module
    service=manager(tmp_path)
    monkeypatch.setattr(service,'_calendar',lambda:[])
    job=service.plan({'tables':['market_daily_raw','market_daily_post_adjusted','trade_status_daily'],
        'source_id':'tushare','endpoints':['daily','adj_factor'],'asset_types':['stock'],
        'codes':['000001.SZ'],'start_date':'2026-10-20','end_date':'2026-10-20'})
    examples={
        'daily':{'ts_code':'000001.SZ','trade_date':'20261020','open':'10','high':'12','low':'9','close':'11','pre_close':'10','vol':'2','amount':'1'},
        'adj_factor':{'ts_code':'000001.SZ','trade_date':'20261020','adj_factor':'2'},
    }
    for endpoint,values in examples.items():
        fields=[f['name'] for f in service.sources.endpoint('tushare',endpoint)['fields']]
        cached_part(service,job,endpoint,{'fields':fields,'items':[[values.get(f) for f in fields]]})
    calls=[];original=module.normalize_source
    def normalize(spec,*args,**kwargs):
        calls.append(spec.name)
        return original(spec,*args,**kwargs)
    monkeypatch.setattr(module,'normalize_source',normalize)
    # Calling the old second pass would re-read and map every source again.
    monkeypatch.setattr('betalens_db_manager.manager_kernel.normalize_source',lambda *_args,**_kwargs:pytest.fail('second source conversion'))
    with light_normalization():tables,rejects=service._build_canonical(service.state.job(job['id']))
    assert not rejects
    assert len(calls)==sum(bool(p['archive']) and p['status']!='split' for p in service.state.slices(job['id']))
    assert tables['market_daily_raw'][0]['close']==Decimal('11')
    assert tables['market_daily_post_adjusted'][0]['close']==Decimal('22')
    assert tables['trade_status_daily'][0]['is_tradable'] is None  # Empty calendar cannot prove tradability.


def test_write_preconditions_validate_each_live_row_once_without_stale_hashes(monkeypatch):
    import betalens_db_manager.commit_protocol as module
    spec=canonical('market_daily_raw')
    old={'asset_type':'stock','security_code':'000001.SZ','trade_date':date(2026,10,20),'close':Decimal('10')}
    incoming={**old,'close':Decimal('11')}
    intent=compare(spec,[incoming],[old],'refresh_source')
    calls=[];original=module.typed
    def normalize(*args):calls.append(1);return original(*args)
    monkeypatch.setattr(module,'typed',normalize)
    assert remaining_changes(spec,intent,[old])==intent['changes']
    assert len(calls)==1
    assert remaining_changes(spec,intent,[intent['changes'][0]['after']])==[]
    assert len(calls)==2
    old['close']=Decimal('12')
    with pytest.raises(NeedsReconciliation):remaining_changes(spec,intent,[old])


def test_compiled_mapping_is_scoped_to_one_frozen_conversion(tmp_path,monkeypatch):
    import betalens_db_manager.mapping_store as module
    service=manager(tmp_path);service.registry.save(table_definition());spec=service.registry.spec('sample_prices')
    binding={'source_id':'tushare','endpoint':'daily','fields':[{'source_field':'close','column':'price','multiplier':'1000'}]}
    calls=[];original=module.validate_binding
    def validate(*args):calls.append(1);return original(*args)
    monkeypatch.setattr(module,'validate_binding',validate)
    with normalization_inputs({'id':'one'}):
        for _ in range(10):
            row,evidence=service.mappings.map_row(spec,{'close':'0.001'},binding,collect_evidence=False)
            assert row['price']==Decimal('1') and not evidence
    assert len(calls)==1
    binding['fields'][0]['multiplier']='2000'
    with normalization_inputs({'id':'two'}):
        assert service.mappings.map_row(spec,{'close':'0.001'},binding)[0]['price']==Decimal('2')
    assert len(calls)==2


@pytest.mark.parametrize('has_version',[True,False])
def test_mapping_version_is_resolved_once_for_a_batch(tmp_path,monkeypatch,has_version):
    import betalens_db_manager.manager as module
    service=manager(tmp_path)
    mapping=service.mappings.get('fund_nav_observation')
    expected=mapping['version']
    if not has_version:
        mapping=dict(mapping);mapping.pop('version');expected=module.digest(mapping)
    item={'id':'test','plan':{'mapping_snapshots':{'fund_nav_observation':mapping}}}
    rows=[{'_row':i,'_raw':{'ts_code':'000001.OF','nav_date':'20261020','unit_nav':'1.25'},
           'ts_code':'000001.OF','nav_date':date(2026,10,20)} for i in range(1,4)]
    calls=[];original=module.digest
    def digest(value):
        if value is mapping:calls.append(1)
        return original(value)
    monkeypatch.setattr(module,'digest',digest)
    with normalization_inputs(item),light_normalization():
        result,_=service._map_part('fund_nav',rows,'test','2026-10-21T00:00:00+00:00',[])
    assert [r['mapping_version'] for r in result['fund_nav_observation']]==[expected]*3
    assert len(calls)==(0 if has_version else 1)


def test_light_conversion_skips_only_empty_derived_overlay(tmp_path,monkeypatch):
    service=manager(tmp_path)
    mappings={name:service.mappings.get(name) for name in ('market_daily_raw','market_daily_post_adjusted','trade_status_daily')}
    item={'id':'test','plan':{'mapping_snapshots':mappings}}
    row={'_row':1,'_raw':{'ts_code':'000001.SZ','trade_date':'20261020','close':'10'},
         'ts_code':'000001.SZ','trade_date':date(2026,10,20),'close':Decimal('10')}
    calls=[];original=service.mappings.map_row
    def map_row(spec,*args,**kwargs):
        calls.append(spec.name);return original(spec,*args,**kwargs)
    monkeypatch.setattr(service.mappings,'map_row',map_row)
    with normalization_inputs(item),light_normalization():
        result,_=service._map_part('daily',[row],'test','2026-10-21T00:00:00+00:00',[])
    assert calls==['market_daily_raw']
    assert result['market_daily_raw'][0]['close']==Decimal('10')
    calls.clear()
    with normalization_inputs(item):service._map_part('daily',[row],'test','2026-10-21T00:00:00+00:00',[])
    assert set(calls)==set(mappings)  # Full evidence replay is preserved.


def test_fast_date_parser_validates_dates_and_bounds_cache():
    from datetime import datetime,timezone
    from betalens_db_manager.util import parse_date,_parse_date_text
    assert parse_date('20240229')==parse_date('2024-02-29')==date(2024,2,29)
    assert parse_date(date(2024,2,29))==date(2024,2,29)
    assert parse_date(datetime(2024,2,29,23,tzinfo=timezone.utc))==date(2024,2,29)
    for text in ('20230229','20261301','20260001','00000101','２０２４０２２９','invalid'):
        with pytest.raises(ValueError):parse_date(text)
    assert _parse_date_text.cache_info().maxsize==8192
