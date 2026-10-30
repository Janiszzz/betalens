"""Prove bulk retrieval preserves scope and does not explode into per-code work."""
from pathlib import Path

import pytest

from test_manager import manager
from test_retrieval import task
from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.pipeline_plan import params_for
from betalens_db_manager.response_validation import validate_archives, validate_response
from betalens_db_manager.retrieval import partition, security_partitions, tushare_policy
from betalens_db_manager.source import read_archive
from betalens_db_manager.util import digest


def test_32570_fund_shares_use_18_pages_and_no_security_expansion(tmp_path):
    rows=[{'ts_code':f'{i:06d}.SH','trade_date':'20260930','fd_share':'1'} for i in range(32570)]
    def response(endpoint,p):
        assert endpoint=='fund_share'
        return rows[p['offset']:p['offset']+p['limit']]
    s,source,item=task(tmp_path,'fund_share',{'trade_date':'20260930'},response,cap=2000,mode='offset_probe')
    TaskDownloader(s,item).run()
    actual=s.state.job(item['id'])
    assert len(source.calls)==18
    assert all('ts_code' not in p for _,p in source.calls)
    assert sum(p['rows'] for p in actual['slices'])==32570
    assert len([p for p in actual['slices'] if p['rows']==0])==1
    assert validate_archives(s,actual)==[]
    assert all(p['status'] in {'downloaded','paged'} for p in actual['slices'])


def test_offset_ignored_blocks_instead_of_fanning_out(tmp_path):
    s,source,item=task(tmp_path,'fund_share',{'trade_date':'20260930'},lambda *_:[{'ts_code':'510300.SH','trade_date':'20260930'}],cap=2000,mode='offset_probe')
    TaskDownloader(s,item).run()
    assert len(source.calls)==2
    assert any(p['status']=='contract_blocked' for p in s.state.job(item['id'])['slices'])


def test_month_plans_use_bulk_days_and_known_multi_code_ranges(tmp_path):
    request={'start_date':'2026-09-28','end_date':'2026-10-27','codes':[]}
    nav=list(params_for('fund_nav',request,{}));indices=list(params_for('index_daily',request,{}))
    assert len(nav)==60 and len(indices)==30
    assert {p['market'] for p in nav}=={'E','O'}
    assert all('ts_code' not in p for p in nav+indices)
    codes=[f'{i:06d}.SZ' for i in range(1000)]
    for endpoint in ('daily','adj_factor','fund_share'):
        chunks=list(params_for(endpoint,{**request,'codes':codes},{}))
        assert 1<len(chunks)<1000
        assert [c for p in chunks for c in p['ts_code'].split(',')]==codes
        assert all(p['start_date']=='20260928' and p['end_date']=='20261027' for p in chunks)
        assert max(len(p['ts_code'].split(','))*30 for p in chunks)<tushare_policy(endpoint,5000)['row_limit']
    singles=list(params_for('daily_basic',{**request,'codes':codes[:2]},{}))
    assert [p['ts_code'] for p in singles]==codes[:2]
    s=manager(tmp_path)
    plan=s.compile({'pipeline_version':2,'tables':['fund_nav_daily'],'targets':['postgres'],'asset_types':['fund'],**request})
    assert not any(r.get('strategy')=='universe_expansion' for r in plan['requests'] if r['endpoint']=='fund_nav')


def test_code_splitting_halves_and_legacy_policy_remains_singleton():
    assert partition('daily',{'trade_date':'20260930','ts_code':'a,b,c,d'},{'input_fields':['ts_code']})==[
        {'trade_date':'20260930','ts_code':'a,b'},{'trade_date':'20260930','ts_code':'c,d'}]
    assert [p['ts_code'] for p in security_partitions('daily',{'trade_date':'20260930'},['a','b','c'],{'row_limit':6000})]==['a','b','c']


def test_local_cohorts_bound_bulk_days_and_keep_legacy_checkpoint_names():
    from betalens_db_manager.pipeline import cohort_key
    daily={'endpoint':'daily','parameters':{'trade_date':'20260930'}}
    nav={'endpoint':'fund_nav','parameters':{'nav_date':'20260930','market':'O'}}
    assert cohort_key(daily,{},True)==cohort_key(nav,{},True)=='2026-09-30'
    assert cohort_key(daily,{},False)=='2026-09'
    codes=[str(i) for i in range(32570)];buckets={code:i//100 for i,code in enumerate(codes)}
    roots=[{'endpoint':'namechange','parameters':{'ts_code':c}} for c in codes]
    assert len({cohort_key(r,buckets,True) for r in roots})==326
    assert cohort_key(roots[0],buckets,False)=='history_0'


def test_cached_fund_universes_do_not_cross_market_boundaries(tmp_path):
    def response(endpoint,p):
        assert endpoint=='fund_basic'
        if p['offset'] or p['status']!='L':return []
        return [{'ts_code':'510300.SH' if p['market']=='E' else '000001.OF','market':p['market']}]
    s,source,item=task(tmp_path,'fund_basic',{},response,cap=2000,mode='offset_probe')
    downloader=TaskDownloader(s,item)
    assert downloader.complete_universe('fund_all')==['000001.OF','510300.SH']
    before=len(source.calls)
    assert downloader.complete_universe('fund')==['510300.SH']
    assert len(source.calls)==before


def test_dependency_and_business_roles_share_wire_cache_under_concurrency(tmp_path):
    s,source,item=task(tmp_path,'fund_basic',{'market':'E','status':'L'},lambda _,p:[] if p['offset'] else [{'ts_code':'510300.SH','market':'E'}],cap=2000,mode='offset_probe')
    root=item['plan']['requests'][0]
    dependency={**root,'dependency_only':True,'strategy':'retrieval_dependency'}
    TaskDownloader(s,item).run([root,dependency],workers=4)
    parts=s.state.job(item['id'])['slices']
    assert len(source.calls)==2 and len(parts)==4
    assert len({p['archive'] for p in parts})==2
    assert sum(bool(p['request'].get('dependency_only')) for p in parts)==2
    assert all(read_archive(Path(p['archive']),p['sha']) is not None for p in parts)


def test_unverified_per_security_fanout_has_hard_guard(tmp_path,monkeypatch):
    s,source,item=task(tmp_path,'daily_basic',{'trade_date':'20260930'},lambda *_:[{'ts_code':'a','trade_date':'20260930'},{'ts_code':'b','trade_date':'20260930'}])
    downloader=TaskDownloader(s,item)
    monkeypatch.setattr(downloader,'complete_universe',lambda _:list(map(str,range(2101))))
    downloader.run()
    parts=s.state.job(item['id'])['slices']
    assert len(source.calls)==1 and len(parts)==1
    assert parts[0]['status']=='contract_blocked' and '2101' in parts[0]['result']['blocked']


def test_fund_response_market_is_verified(tmp_path):
    s,_,item=task(tmp_path,'fund_basic',{'market':'E'},lambda *_:[])
    contract=item['plan']['source_contracts']['tushare:fund_basic']
    fields=[f['name'] for f in contract['fields']]
    row={'ts_code':'000001.OF','market':'O'}
    errors=validate_response(item['plan'],item['plan']['requests'][0],contract,{'fields':fields,'items':[[row.get(f) for f in fields]]})
    assert any(e['code']=='market_scope' for e in errors)


def test_off_exchange_nav_may_use_a_listed_lof_code(tmp_path):
    s,source,item=task(tmp_path,'fund_nav',{'nav_date':'20260930','market':'O'},lambda _,p:[] if p['offset'] else [{'ts_code':'168002.SZ','nav_date':'20260930','ann_date':'20260930','unit_nav':'2.601'}],cap=10000,mode='offset_probe')
    TaskDownloader(s,item).run()
    assert len(source.calls)==2
    assert validate_archives(s,s.state.job(item['id']))==[]


def test_replan_reuses_verified_archives_without_fetch_or_mutating_old_plan(tmp_path,monkeypatch):
    s=manager(tmp_path)
    request={'pipeline_version':2,'tables':['security_dim'],'targets':['postgres'],'asset_types':['stock'],'start_date':'2026-09-28','end_date':'2026-10-27','codes':[]}
    old=s.plan(request)
    from test_retrieval import ArchivedSource
    s.source=ArchivedSource(lambda *_:[])
    TaskDownloader(s,old).run()
    frozen=digest(s.state.job(old['id'])['plan'])
    calls=len(s.source.calls)
    new=s.replan_retrieval(old['id'])
    assert len(s.source.calls)==calls
    assert new['status']=='planned' and not new['tasks']
    assert new['plan']['request']==old['plan']['request']
    assert new['plan']['parent_task_id']==old['id']
    assert new['plan']['retrieval_replan']['reused_slices']==len(old['plan']['requests'])
    assert {p['archive'] for p in new['slices']}=={p['archive'] for p in s.state.job(old['id'])['slices']}
    evidence=s.state.ledger_rows('import_source_archive',new['id'])
    assert len(evidence)==len(new['slices']) and all(r['fetched_at'] for r in evidence)
    assert digest(s.state.job(old['id'])['plan'])==frozen
    TaskDownloader(s,new).run()
    assert len(s.source.calls)==calls
    s.state.enqueue(old['id'],'pipeline_run',{'background':True})
    with pytest.raises(ValueError,match='先停止'):s.replan_retrieval(old['id'])


def test_replan_discards_corrupt_cache_and_changed_source(tmp_path):
    s=manager(tmp_path)
    from test_retrieval import ArchivedSource
    s.source=ArchivedSource(lambda *_:[])
    old=s.plan({'pipeline_version':2,'tables':['security_dim'],'targets':['postgres'],'asset_types':['stock'],'start_date':'2026-09-28','end_date':'2026-10-27'})
    TaskDownloader(s,old).run()
    part=s.state.job(old['id'])['slices'][0];Path(part['archive']).write_bytes(b'corrupt')
    new=s.replan_retrieval(old['id'])
    assert part['id'] not in {p['id'] for p in new['slices']}
    original=s.compile
    def changed(request):
        plan=original(request);plan['source_instances']={'tushare':'different'};return plan
    s.compile=changed
    with pytest.raises(ValueError,match='来源身份'):s.replan_retrieval(old['id'])
