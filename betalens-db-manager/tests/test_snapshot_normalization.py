"""Bounded historical chunks preserve source closure and canonical values."""
from collections import defaultdict
from copy import deepcopy

import pytest

from betalens_db_manager.canonical import registry_context
from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.pipeline import _snapshot_normalization_items
from betalens_db_manager.pipeline_options import light_normalization
from betalens_db_manager.response_validation import validate_archives
from test_manager import manager
from test_retrieval import ArchivedSource


def archived_snapshot(tmp_path,mode='partition'):
    days=['20261020','20261021','20261022','20261023']
    def response(endpoint,params):
        if endpoint=='sw_daily':
            rows=[{'ts_code':'801010.SI','trade_date':day,'name':'农林牧渔',
                   'open':'100','high':'101','low':'99','close':'100.5',
                   'pre_close':'100','change':'0.5','pct_change':'0.5',
                   'vol':'10','volume':'10','amount':'100'}
                  for day in days if params.get('start_date',days[0])<=day<=params.get('end_date',days[-1])]
            if mode=='offset':return rows[params.get('offset',0):params.get('offset',0)+2]
            return rows
        if endpoint=='index_basic' and params.get('market')=='SSE' and not params.get('offset',0):
            return [{'ts_code':'801010.SI','name':'农林牧渔','fullname':'申万农林牧渔',
                     'market':'SSE','publisher':'申万','category':'行业指数','base_date':'20000101',
                     'base_point':'100','list_date':'20000101'}]
        if endpoint=='trade_cal':return [{'exchange':params['exchange'],'cal_date':day,'is_open':'1'} for day in days]
        return []
    service=manager(tmp_path);service.source=ArchivedSource(response)
    item=service.plan({'tables':['market_daily_raw','market_daily_post_adjusted','security_dim','index_profile','trade_calendar'],
        'source_id':'tushare','asset_types':['index'],'endpoints':['sw_daily','index_basic','trade_cal'],
        'start_date':'2026-10-20','end_date':'2026-10-23',
        'parameters':{'sw_daily':{'ts_code':'801010.SI'}},'auto_commit':False,'min_free_bytes':0})
    plan=item['plan'];plan['retrieval_policies']['tushare:sw_daily'].update(row_limit=2,mode=mode)
    service.state.replace_plan(item['id'],plan)
    with TaskDownloader(service,service.state.job(item['id'])) as downloader:
        report=downloader.run()
        assert not report['blocking'],report
    return service,service.state.job(item['id'])


def sorted_rows(tables):
    return {name:sorted(rows,key=lambda row:str(sorted(row.items()))) for name,rows in tables.items() if rows}


def test_split_snapshot_preserves_all_values_provenance_and_context_once(tmp_path):
    service,item=archived_snapshot(tmp_path);before=deepcopy(item)
    with registry_context(service.registry),light_normalization():
        expected,bad=service._build_canonical(item)
        assert not bad,bad
        chunks=list(_snapshot_normalization_items(service,item,max_rows=2))
        combined=defaultdict(list)
        assert len(chunks)==3  # One shared snapshot context and two price branches.
        for chunk in chunks:
            assert not validate_archives(service,chunk)
            assert sum(p['rows'] or 0 for p in chunk['slices'] if p['dataset']=='sw_daily' and p['status']!='split')<=2
            rows,rejected=service._build_canonical(chunk)
            assert not rejected,rejected
            for name,values in rows.items():combined[name].extend(values)
        assert sorted_rows(combined)==sorted_rows(expected)
    assert item==before
    assert len(combined['market_daily_raw'])==len(combined['market_daily_post_adjusted'])==4
    assert len(combined['security_dim'])==1


def test_chunking_never_accepts_a_missing_archive_branch(tmp_path):
    service,item=archived_snapshot(tmp_path)
    part=next(p for p in item['slices'] if p['dataset']=='sw_daily' and p['status']=='downloaded')
    item['slices'].remove(part)
    chunks=list(_snapshot_normalization_items(service,item,max_rows=2))
    assert chunks==[item]
    with registry_context(service.registry),light_normalization():
        rows,rejected=service._build_canonical(chunks[0])
    assert not rows and rejected


def test_complete_pagination_including_empty_terminal_page_stays_together(tmp_path):
    service,item=archived_snapshot(tmp_path,mode='offset')
    chunks=list(_snapshot_normalization_items(service,item,max_rows=4))
    assert len(chunks)==2
    price_chunk=chunks[-1]
    assert sorted(p['rows'] for p in price_chunk['slices'] if p['dataset']=='sw_daily')==[0,2,2]
    assert not validate_archives(service,price_chunk)
    with pytest.raises(ValueError,match='分页链超过'):
        list(_snapshot_normalization_items(service,item,max_rows=2))
