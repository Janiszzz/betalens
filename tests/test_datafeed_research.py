"""Research assumptions are explicit and never widen the strict contract."""
from copy import deepcopy
from datetime import date
import json

import pandas as pd
import pytest

from betalens.datafeed import Datafeed, PITUnavailableError, VersionConflictError, query_trade_risks
from betalens.datafeed.config import load_subscription_config
from betalens.datafeed.contracts import timestamp
from betalens.datafeed import query
from betalens.datafeed.industry import query_industry
from betalens.datafeed.reference import calendar_days
from betalens.datafeed.universe import get_index_universe
from datafeed_support import MemoryReader, market_reader, quote, evidence, NAME_TYPES


def research_reader():
    reader = market_reader([quote(at="2024-01-03 00:00", availability_basis="next_calendar_day_assumed")])
    reader.pit_policy = "research"
    reader.types['security_name_history'] = NAME_TYPES
    reader.tables['security_name_history'] = [dict(security_code='A',name='*ST Example',
        start_date=date(2020,1,1),end_date=None,**evidence('2020-01-02',availability_basis='announcement_next_calendar_day_assumed'))]
    return reader


def test_research_accepts_assumptions_without_changing_strict_or_shanghai_boundaries():
    reader = research_reader()
    times = ['2024-01-02 23:59:59', '2024-01-03', '2024-01-02T16:00:00Z']
    result = query.query_nearest_before(reader,'market_daily_raw',['A'],times,'close')
    assert result.close.iloc[0] != result.close.iloc[0]
    assert result.close.iloc[1:].tolist() == [5,5]
    assert result.name.iloc[1] == '*ST Example'
    strict = query.query_nearest_before(reader,'market_daily_raw',['A'],times,'close',pit_policy='strict')
    assert strict.close.isna().all()
    assert reader.pit_policy == 'research'
    result = query.query_time_range(reader,'market_daily_raw',['A'],metric='close')
    assert result.remark.iloc[0]['availability_basis'] == 'next_calendar_day_assumed'


@pytest.mark.parametrize('basis', ['unknown','downloaded_at','ingested_at','source_snapshot',None])
def test_research_does_not_accept_arbitrary_availability_labels(basis):
    reader = research_reader()
    reader.tables['market_daily_raw'][0]['availability_basis'] = basis
    assert query.query_nearest_before(reader,'market_daily_raw',['A'],['2025-01-01'],'close').close.isna().all()


def test_after_remains_a_future_observation_and_raw_post_tables_stay_independent():
    reader = research_reader()
    after = query.query_nearest_after(reader,'market_daily_post_adjusted',['A'],['2024-01-02'],'close')
    assert after.close.tolist() == [10]
    assert after.datetime.iloc[0] > after.input_ts.iloc[0]
    reader.tables['market_daily_post_adjusted'][0]['close'] = None
    assert query.query_nearest_before(reader,'market_daily_post_adjusted',['A'],['2024-01-04'],'close').close.isna().all()
    assert query.query_nearest_before(reader,'market_daily_raw',['A'],['2024-01-04'],'close').close.tolist() == [5]


def test_subscription_policy_and_explicit_constructor_override(tmp_path, monkeypatch):
    path=tmp_path/'subscription.json'
    path.write_text(json.dumps({'pit_policy':'strict','postgres':{'host':'localhost','dbname':'test','user':'reader'}}))
    assert load_subscription_config(config_file=path)['pit_policy'] == 'strict'
    monkeypatch.setattr('betalens.datafeed.core.PostgresReader', MemoryConnection)
    with Datafeed('market_daily_raw',config_file=path,pit_policy='research') as data:
        assert data.pit_policy == 'research'
    with pytest.raises(ValueError,match='pit_policy'):
        Datafeed('market_daily_raw',config_file=path,pit_policy='loose')


class MemoryConnection:
    def __init__(self, config):
        self.config=config
    def close(self):
        pass


def industry_reader():
    reader=research_reader()
    availability={'available_at':'timestamp','availability_basis':'text','data_quality_status':'text'}
    reader.types['industry_scheme_dim']={'scheme_code':'text','event_date':'date',**availability}
    reader.tables['industry_scheme_dim']=[dict(scheme_code=scheme,event_date=day,**evidence(at,availability_basis='manual_publication_date_assumed')) for scheme,day,at in
        [('SW2014',date(2014,1,1),'2013-12-10'),('SW2021',date(2021,7,30),'2021-08-01'),('CITICS',date(2019,12,1),'2020-01-03')]]
    reader.types['industry_dim']={'scheme_code':'text','industry_code':'text','industry_name':'text','level':'text'}
    reader.tables['industry_dim']=[dict(scheme_code=scheme,industry_code=f'{n}{level}',industry_name=f'{scheme}-{level}',level=f'L{level}')
        for n,scheme in enumerate(['SW2014','SW2021','CITICS'],1) for level in [1,2,3]]
    reader.types['security_industry_membership']={'security_code':'text','scheme_code':'text','industry_code':'text','valid_from':'date','valid_to':'date',**availability}
    reader.tables['security_industry_membership']=[dict(security_code='A',scheme_code=r['scheme_code'],industry_code=r['industry_code'],valid_from=date(2014,1,1),valid_to=None,
        **evidence('2014-01-02',availability_basis='membership_effective_date_assumed;historical_disclosure_unproven')) for r in reader.tables['industry_dim']]
    return reader


def test_research_scheme_selection_is_common_across_levels_and_duplicate_inputs():
    reader=industry_reader()
    for level in [1,2,3]:
        result=query_industry(reader,['A','X','A'],['2021-07-31','2021-08-01'],level=level)
        assert result.code.tolist() == ['A','A','X','X','A','A']
        assert [r['scheme_code'] for r in result.remark] == ['SW2014','SW2021']*3
        assert result.ind_name.iloc[:2].tolist() == [f'SW2014-{level}',f'SW2021-{level}']
        assert result.ind_code.iloc[2:4].isna().all()
    assert query_industry(reader,['A'],['2024-01-01'],'CITICS').ind_name.tolist() == ['CITICS-1']
    with pytest.raises(PITUnavailableError):
        query_industry(reader,['A'],['2013-01-01'])
    with pytest.raises(PITUnavailableError):
        query_industry(reader,['A'],['2024-01-01'],pit_policy='strict')


def test_industry_conflicts_and_left_closed_right_open_boundaries():
    reader=industry_reader()
    row=reader.tables['security_industry_membership'][3]
    row['valid_to']=date(2024,1,1)
    assert query_industry(reader,['A'],['2024-01-01']).ind_code.isna().all()
    row['valid_to']=None
    reader.tables['security_industry_membership'].append(dict(row,valid_from=date(2023,1,1)))
    with pytest.raises(VersionConflictError):
        query_industry(reader,['A'],['2024-01-01'])


def test_calendar_static_assumption_still_requires_every_natural_day():
    reader=research_reader()
    reader.types['trade_calendar']={'exchange':'text','trade_date':'date','is_open':'integer'}
    reader.tables['trade_calendar']=[{'exchange':'SSE','trade_date':date(2024,1,d),'is_open':int(d!=1)} for d in [1,2,3]]
    assert calendar_days(reader,'2024-01-01','2024-01-03',as_of='2023-01-01',table_name='trade_calendar',pit_policy='research') == [date(2024,1,2),date(2024,1,3)]
    with pytest.raises(PITUnavailableError):
        calendar_days(reader,'2024-01-01','2024-01-04',table_name='trade_calendar',pit_policy='research')


def status_reader():
    reader=research_reader()
    availability={'available_at':'timestamp','availability_basis':'text','data_quality_status':'text'}
    reader.types['security_dim']={'security_code':'text','asset_type':'text','list_date':'date','delist_date':'date'}
    reader.tables['security_dim']=[dict(security_code=c,asset_type='stock',list_date=date(2020,1,1),delist_date=None) for c in ['A','B','C']]
    reader.types['trade_status_daily']={'security_code':'text','asset_type':'text','trade_date':'date','is_tradable':'integer','status_source':'text',**availability}
    reader.tables['trade_status_daily']=[]
    reader.types['suspension_event']={'security_code':'text','trade_date':'date','suspend_type':'text','suspend_timing':'text',**availability}
    reader.tables['suspension_event']=[dict(security_code='A',trade_date=date(2024,1,2),suspend_type='S',suspend_timing='09:30-10:00',
        **evidence('2024-01-02',availability_basis='suspension_effective_date_assumed'))]
    return reader


def test_risks_report_st_separately_from_suspension_and_keep_unknown():
    reader=status_reader()
    result=query_trade_risks(reader,'trade_status_daily',['A','B'],['2024-01-02 09:30','2024-01-02 10:00'])
    assert result.is_st.tolist()[:1] == [1]
    assert result.is_st.iloc[[1,3]].isna().all()
    assert result.is_suspended.iloc[[0,2]].tolist() == [1,0]
    assert result.is_suspended.iloc[[1,3]].isna().all()
    status=query.query_trade_status(reader,'trade_status_daily',['B'],['2024-01-02'])
    assert status.value.isna().all()
    assert str(status.value.dtype) == 'Int8'


def test_malformed_suspension_does_not_fall_back_to_daily_normal():
    reader=status_reader()
    reader.tables['suspension_event'][0]['suspend_timing']='unparseable'
    reader.tables['trade_status_daily']=[dict(security_code='A',asset_type='stock',trade_date=date(2024,1,2),is_tradable=1,status_source='quote',**evidence('2024-01-02'))]
    assert query_trade_risks(reader,'trade_status_daily',['A'],['2024-01-02 11:00']).is_suspended.isna().all()
    empty=query_trade_risks(reader,'trade_status_daily',[],[])
    assert empty.columns.tolist() == ['code','datetime','is_listed','is_st','is_suspended','name','remark']


def test_index_uses_last_available_complete_snapshot_and_never_future_snapshot():
    reader=research_reader()
    reader.types['index_membership']={'index_code':'text','security_code':'text','trade_date':'date',
        'snapshot_complete':'integer','snapshot_member_count':'integer','available_at':'timestamp','availability_basis':'text','data_quality_status':'text'}
    reader.tables['index_membership']=[dict(index_code='I',security_code='A',trade_date=date(2024,1,d),snapshot_complete=int(d==2),snapshot_member_count=1,
        **evidence(f'2024-01-{d+1:02}',availability_basis='next_calendar_day_assumed')) for d in [2,3]]
    assert get_index_universe(reader,'I','2024-01-04') == ['A']
    with pytest.raises(PITUnavailableError):
        get_index_universe(reader,'I','2024-01-02')
    with pytest.raises(PITUnavailableError):
        get_index_universe(reader,'I','2024-01-04',pit_policy='strict')


def test_multiple_intraday_intervals_and_unlisted_vs_delisted():
    reader=status_reader()
    reader.tables['suspension_event'][0]['suspend_timing']='9:30-9:40,10:30-10:40'
    result=query_trade_risks(reader,codes=['A'],dates=['2024-01-02 09:30','2024-01-02 09:40','2024-01-02 10:30','2024-01-02 10:40'])
    assert result.is_suspended.tolist() == [1,0,1,0]
    reader.tables['security_dim'][0]['list_date']=date(2025,1,1)
    assert query.query_trade_status(reader,codes=['A'],dates=['2024-01-02']).value.tolist() == [-1]
    reader.tables['security_dim'][0].update(list_date=date(2020,1,1),delist_date=date(2023,1,1))
    assert query.query_trade_status(reader,codes=['A'],dates=['2024-01-02']).value.isna().all()


def test_industry_reverse_does_not_treat_retired_history_as_unknown_current_membership():
    from betalens.datafeed.industry import get_industry_members
    reader=industry_reader()
    retired=dict(reader.tables['security_industry_membership'][3],security_code='RETIRED',valid_to=date(2020,1,1))
    reader.tables['security_industry_membership'].append(retired)
    result=get_industry_members(reader,'SW2021-1','2024-01-01')
    assert result.code.tolist() == ['A']


@pytest.mark.parametrize('policy',[None,[],True])
def test_invalid_config_policy_is_not_silently_treated_as_default(tmp_path,policy):
    path=tmp_path/'invalid.json'
    path.write_text(json.dumps({'pit_policy':policy,'postgres':{'host':'localhost','dbname':'test','user':'reader'}}))
    with pytest.raises(ValueError,match='pit_policy'):
        load_subscription_config(config_file=path)
