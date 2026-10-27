"""Archived partition proofs must remain complete during offline revalidation."""
from copy import deepcopy
import gzip
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from betalens_db_manager.catalog import load_catalog
from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.response_validation import validate_archives, record_archive_validation
from betalens_db_manager.retrieval import tushare_policy
from betalens_db_manager.state import State
from betalens_db_manager.util import dumps


class ArchivedSource:
    def __init__(self,response):self.response=response
    def fetch(self,endpoint,params,fields,path):
        rows=self.response(endpoint,params)
        data={'fields':fields,'items':[[row.get(field) for field in fields] for row in rows]}
        payload=dumps({'code':0,'data':data}).encode()
        path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wb') as handle:handle.write(payload)
        return data,hashlib.sha256(payload).hexdigest()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**k:pytest.fail('No network in response validation tests'))


def downloaded(tmp_path,endpoint,params,response,cap=2,mode='partition'):
    state=State(tmp_path/'state')
    service=SimpleNamespace(state=state,source=ArchivedSource(response),safe_error=str,
        settings=SimpleNamespace(state_dir=tmp_path/'state',archive_dir=None),
        sources=SimpleNamespace(values={'tushare':{'provider':'tushare'}}))
    request={'source_id':'tushare','endpoint':endpoint,'parameters':params,'table':'offline'}
    plan={'requests':[request],'request':{'targets':['postgres'],'tables':[]},'mapping_snapshots':{},
        'source_contracts':{'tushare:'+name:spec.as_dict() for name,spec in load_catalog().items()},
        'retrieval_policies':{'tushare:'+endpoint:{**tushare_policy(endpoint),'row_limit':cap,'mode':mode}}}
    item=state.create_job(plan,'isolated-test-target')
    TaskDownloader(service,item).run()
    service.source.fetch=lambda *a,**k:pytest.fail('Archive validation must never fetch')
    return service,state.job(item['id'])


def update_part(service,item,part,**changes):
    service.state.slice(item['id'],part['id'],part['dataset'],part['request'],**changes)
    return service.state.job(item['id'])


def parent(item,endpoint):
    return next(p for p in item['slices'] if p['dataset']==endpoint and p['status']=='split')


def error_codes(service,item):return {e['code'] for e in validate_archives(service,item)}


def date_rows(endpoint,params):
    return [{'ts_code':'a','trade_date':day,'close':'1'} for day in ('20261013','20261014','20261015','20261016')
        if params['start_date']<=day<=params['end_date']][:2]


def test_complete_date_tree_is_accepted_without_fetch(tmp_path):
    service,item=downloaded(tmp_path,'daily',{'ts_code':'a','start_date':'20261013','end_date':'20261016'},date_rows)
    assert not validate_archives(service,item)


@pytest.mark.parametrize('change',['missing','overlap','duplicate'])
def test_date_children_must_exactly_cover_parent_even_when_all_archives_exist(tmp_path,change):
    service,item=downloaded(tmp_path,'daily',{'ts_code':'a','start_date':'20261013','end_date':'20261016'},date_rows)
    part=parent(item,'daily');result=deepcopy(part['result'])
    if change=='missing':result['children'].pop()
    elif change=='duplicate':result['children'][1]=deepcopy(result['children'][0])
    else:result['children'][1]['parameters']['start_date']=result['children'][0]['parameters']['start_date']
    item=update_part(service,item,part,result=result)
    assert 'partition_coverage' in error_codes(service,item)


def test_enumeration_split_cannot_omit_empty_category(tmp_path):
    def response(endpoint,params):
        return [{'ts_code':'a'},{'ts_code':'b'}] if 'list_status' not in params else [{'ts_code':'a'}] if params['list_status']=='L' else []
    service,item=downloaded(tmp_path,'stock_basic',{},response)
    assert not validate_archives(service,item)
    part=parent(item,'stock_basic');result=deepcopy(part['result'])
    result['children']=[r for r in result['children'] if r['parameters']['list_status']!='D']
    item=update_part(service,item,part,result=result)
    assert 'partition_coverage' in error_codes(service,item)


def test_comma_securities_split_cannot_repeat_one_security_instead_of_another(tmp_path):
    def response(endpoint,params):
        return [{'ts_code':code,'trade_date':'20261013'} for code in params['ts_code'].split(',')][:2]
    service,item=downloaded(tmp_path,'daily',{'ts_code':'a,b,c','trade_date':'20261013'},response)
    assert not validate_archives(service,item)
    part=parent(item,'daily');result=deepcopy(part['result'])
    result['children'][-1]=deepcopy(result['children'][0])
    item=update_part(service,item,part,result=result)
    assert 'partition_coverage' in error_codes(service,item)


def universe_rows(endpoint,params):
    if endpoint in {'stock_basic','fund_basic'}:
        status=params.get('list_status',params.get('status'))
        return [{'ts_code':code} for code in ('a','b','c')] if status=='L' and params.get('offset',0)==0 else []
    codes=[params['ts_code']] if 'ts_code' in params else ['a','b']
    return [{'ts_code':code,'trade_date':'20261013'} for code in codes]


@pytest.mark.parametrize('endpoint',['daily','fund_daily'])
def test_complete_universe_is_reconstructed_only_from_archived_dependency_roots(tmp_path,endpoint):
    service,item=downloaded(tmp_path,endpoint,{'trade_date':'20261013'},universe_rows)
    assert not validate_archives(service,item)


def test_universe_split_requires_security_absent_from_saturated_parent(tmp_path):
    service,item=downloaded(tmp_path,'daily',{'trade_date':'20261013'},universe_rows)
    part=parent(item,'daily');result=deepcopy(part['result'])
    result['children']=[r for r in result['children'] if r['parameters']['ts_code']!='c']
    item=update_part(service,item,part,result=result)
    assert 'partition_coverage' in error_codes(service,item)


@pytest.mark.parametrize('damage',['missing','unavailable','all_missing'])
def test_universe_requires_every_status_root_even_if_that_root_was_empty(tmp_path,damage):
    service,item=downloaded(tmp_path,'daily',{'trade_date':'20261013'},universe_rows)
    dependencies=[p for p in item['slices'] if p['request'].get('dependency_only')]
    part=next(p for p in dependencies if p['request']['parameters'].get('list_status')=='D')
    if damage=='unavailable':
        item=update_part(service,item,part,status='unavailable')
    else:
        ids=[p['id'] for p in dependencies] if damage=='all_missing' else [part['id']]
        with service.state.connect() as conn:
            conn.executemany('DELETE FROM slices WHERE job=? AND id=?',[(item['id'],key) for key in ids])
        item=service.state.job(item['id'])
    assert error_codes(service,item)&{'missing_request','unavailable_dependency'}


@pytest.mark.parametrize('data,code',[
    (None,'response_format'),
    ({'fields':[{'unhashable':1}],'items':[[1]]},'duplicate_fields'),
    ({'fields':[['unhashable']],'items':[[1]]},'duplicate_fields'),
    ({'fields':['ts_code'],'items':[None]},'row_width'),
    ({'fields':['ts_code','unknown'],'items':[['a',{}]]},'unknown_fields'),
    ({'fields':['ts_code'],'items':[[{}]]},'security_scope'),
])
def test_invalid_response_stops_before_graph_and_overlap_iteration(tmp_path,data,code):
    service,item=downloaded(tmp_path,'stock_basic',{'ts_code':'a'},lambda *_:[])
    part=item['slices'][0];payload=dumps({'code':0,'data':data}).encode()
    with gzip.open(part['archive'],'wb') as handle:handle.write(payload)
    item=update_part(service,item,part,sha=hashlib.sha256(payload).hexdigest())
    assert code in error_codes(service,item)


def test_structured_validation_report_is_shared_by_ledger_and_log(tmp_path):
    service,item=downloaded(tmp_path,'daily',{'ts_code':'a','start_date':'20261013','end_date':'20261016'},date_rows)
    part=parent(item,'daily');result=deepcopy(part['result']);result['children'].pop()
    item=update_part(service,item,part,result=result)
    report=record_archive_validation(service,item)
    assert report['blocking'] and any(e['code']=='partition_coverage' for e in report['errors'])
    assert service.state.ledger_get('task_response_validation',item['id'])==report
    log=service.state.logs(item['id'])[-1]
    assert log['event']=='archive_validation' and log['validation_errors']==report['errors']
