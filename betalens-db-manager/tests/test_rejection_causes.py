from datetime import date
from decimal import Decimal

import pytest

from betalens_db_manager.canonical import canonical,registry_context
from betalens_db_manager.catalog import dataset
from betalens_db_manager.comparison import row_key,typed
from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.manager_kernel import normalize_source
from betalens_db_manager.pipeline_options import light_normalization
from betalens_db_manager.response_validation import validate_archives
from betalens_db_manager.retrieval import tushare_policy
from test_manager import cached_part,manager
from test_retrieval import task


@pytest.mark.parametrize('code',['00700.HK','03690.HK','BABA','PDD','BRK.B','RDS-A','000001.SZ','6758','035720','5PD','900250','000001'])
def test_index_constituents_preserve_native_identifiers(code):
    spec=canonical('index_membership')
    row={'index_code':'000501HKD08.CSI','security_code':code,'trade_date':date(2026,10,30),'weight':Decimal('7.121')}
    assert typed(spec,row)['security_code']==code
    if not code.endswith('.SZ'):
        with pytest.raises(ValueError):typed(canonical('market_daily_raw'),{**row,'asset_type':'stock'})


@pytest.mark.parametrize('code',['!000001','00700 HK',"BABA';DROP",'../BABA','', 'A'*65])
def test_index_constituents_reject_damaged_or_ambiguous_identifiers(code):
    with pytest.raises(ValueError):typed(canonical('index_membership'),{'security_code':code})


def test_missing_dividend_announcement_is_an_observation_not_a_missing_key(tmp_path):
    service=manager(tmp_path)
    job=service.plan({'tables':['dividend_event','corporate_action'],'source_id':'tushare','endpoints':['dividend'],
        'codes':['000062.SZ'],'start_date':'2026-01-01','end_date':'2026-12-31'})
    fields=[f['name'] for f in service.sources.endpoint('tushare','dividend')['fields']]
    raw={'ts_code':'000062.SZ','end_date':'20260630','ann_date':None,'div_proc':'股东提议','cash_div_tax':'0.3','stk_div':'0'}
    payload={'fields':fields,'items':[[raw.get(f) for f in fields]]}
    rows,rejected=normalize_source(dataset('dividend'),payload,job['id'],'part','2026-11-04T00:00:00Z')
    assert not rejected and rows[0]['ann_date'] is None
    cached_part(service,job,'dividend',payload)
    with light_normalization(),registry_context(service.registry):
        tables,rejected=service._build_canonical(service.state.job(job['id']))
    assert not rejected
    observation=tables['dividend_event'][0]
    assert observation['ann_date'] is None and observation['available_at'] is None
    assert observation['record_version'] and not tables.get('corporate_action')
    raw['ts_code']=None
    bad={'fields':fields,'items':[[raw.get(f) for f in fields]]}
    assert normalize_source(dataset('dividend'),bad,'job','part','2026-11-04T00:00:00Z')[1]


@pytest.mark.parametrize('ignore_offset',[False,True])
def test_saturated_index_weight_day_requires_advancing_pages_and_empty_end(tmp_path,ignore_offset):
    records=[{'index_code':'000300.SH','con_code':code,'trade_date':'20261020','weight':'1'} for code in ['000001.SZ','00700.HK','BABA']]
    def response(endpoint,params):
        offset=0 if ignore_offset else params.get('offset',0)
        return records[offset:offset+2]
    service,source,item=task(tmp_path,'index_weight',{'index_code':'000300.SH','start_date':'20261020','end_date':'20261020'},response)
    item['plan']['retrieval_policies']['tushare:index_weight']['single_day_offset_probe']=True
    service.state.replace_plan(item['id'],item['plan'])
    with TaskDownloader(service,item) as downloader:downloader.run()
    result=service.state.job(item['id']);parts=result['slices']
    if ignore_offset:
        assert any(p['status']=='contract_blocked' for p in parts)
        assert validate_archives(service,result)
    else:
        assert not validate_archives(service,result)
        assert [params['offset'] for _,params in source.calls]==[0,2,3]
        assert sum(p['rows'] for p in parts)==3


@pytest.mark.parametrize('endpoint',['index_member_all','ci_index_member'])
def test_complete_industry_universe_can_exceed_old_2000_guard(tmp_path,endpoint,monkeypatch):
    service,source,item=task(tmp_path,endpoint,{'is_new':'Y'},lambda *_:[])
    policy=tushare_policy(endpoint,5000)
    assert 6000<=policy['max_security_fanout']<=12000
    item['plan']['retrieval_policies']['tushare:'+endpoint]=policy
    with TaskDownloader(service,item) as downloader:
        codes=[f'{i:06d}.SZ' for i in range(6000)]
        monkeypatch.setattr(downloader,'complete_universe',lambda *_:codes)
        roots=downloader.split(item['plan']['requests'][0],item['plan']['source_contracts']['tushare:'+endpoint])
    assert len(roots)==len(codes)
    assert {r['parameters']['ts_code'] for r in roots}==set(codes)
    assert all(r['parameters']['is_new']=='Y' for r in roots)
