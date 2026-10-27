"""Download completion and replay use the same frozen response validation facts."""
from copy import deepcopy
import gzip
import hashlib
from pathlib import Path

import pytest

from test_manager import manager,table_definition
from test_remediation_response import downloaded,universe_rows
from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.source import read_archive
from betalens_db_manager.source_registry import SubscriptionAdapter
from betalens_db_manager.util import dumps


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**k:pytest.fail('These tests cannot use the network'))


@pytest.mark.parametrize('data,code',[
    (None,'response_format'),([], 'response_format'),
    ({'fields':None,'items':[]},'response_format'),
    ({'fields':['ts_code'],'items':{}},'response_format'),
    ({'fields':[{'field':'ts_code'}],'items':[['a']]},'duplicate_fields'),
    ({'fields':[['ts_code']],'items':[['a']]},'duplicate_fields'),
    ({'fields':['ts_code','ts_code'],'items':[['a','a']]},'duplicate_fields'),
    ({'fields':['ts_code'],'items':[None]},'row_width'),
    ({'fields':['ts_code'],'items':[['a','extra']]},'row_width'),
    ({'fields':['ts_code','unregistered'],'items':[['a',{}]]},'unknown_fields'),
    ({'fields':['ts_code'],'items':[[{}]]},'security_scope'),
])
def test_fetch_archives_malformed_response_then_reports_same_structured_error(tmp_path,data,code):
    service,base=downloaded(tmp_path,'stock_basic',{'ts_code':'a'},lambda *_:[])
    item=service.state.create_job(base['plan'],'isolated-test-target')
    def fetch(endpoint,params,fields,path):
        raw=dumps({'code':0,'data':data}).encode()
        path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wb') as handle:handle.write(raw)
        return data,hashlib.sha256(raw).hexdigest()
    service.source.fetch=fetch
    report=TaskDownloader(service,item).run()
    saved=service.state.job(item['id']);part=saved['slices'][0]
    assert part['status']=='contract_blocked' and Path(part['archive']).exists()
    assert read_archive(Path(part['archive']),part['sha'])==data
    assert code in {e['code'] for e in part['result']['validation_errors']}
    assert report['blocking'] and code in {e['code'] for e in report['errors']}
    assert service.state.ledger_get('task_response_validation',item['id'])==report
    logs=[r for r in service.state.logs(item['id']) if r['event']=='archive_validation']
    assert len(logs)==1 and logs[0]['validation_errors']==report['errors']


def test_nested_universe_download_records_validation_only_after_outer_run(tmp_path):
    service,item=downloaded(tmp_path,'daily',{'trade_date':'20261013'},universe_rows)
    assert len([p for p in item['slices'] if p['request'].get('dependency_only')])==5
    logs=service.state.logs(item['id'])
    validation=[r for r in logs if r['event']=='archive_validation']
    assert len(validation)==1 and not validation[0]['blocking'] and validation[0]['validation_errors']==[]
    assert validation[0]['id']>max(r['id'] for r in logs if r['event']=='response_archived')


def vendor_manager(tmp_path,*,wrong_security=False,pagination=False):
    service=manager(tmp_path);service.registry.save(table_definition())
    contract={'strategy':'range','security_field':'actual_code','date_field':'day',
        'parameter_bindings':{'security_code':'requested_code'},'keys':['actual_code','day'],
        'fields':[{'name':'actual_code','type':'text'},{'name':'pretend_code','type':'text'},
            {'name':'day','type':'date'},{'name':'price','type':'numeric'}]}
    if pagination:contract['pagination']={'offset_param':'offset','limit_param':'limit','size':2}
    definition={'source_id':'vendor','provider':'http','base_url':'https://example.invalid','endpoints':{'prices':contract}}
    service.sources.save(definition)
    rules=[{'source_field':a,'column':b} for a,b in [('actual_code','code'),('day','business_date'),('price','price')]]
    mapping={'format_version':2,'table':'sample_prices','bindings':[{'source_id':'vendor','endpoint':'prices','mapping_mode':'direct',
        'fields':rules,'join_keys':rules[:2],'ignored_fields':{'pretend_code':'not the requested security'}}]}
    service.mappings.save(mapping,None)
    calls=[]
    class FakeAdapter(SubscriptionAdapter):
        def _fetch(self,endpoint,params,*args):
            calls.append(deepcopy(params))
            return {'fields':['actual_code','pretend_code','day','price'],
                'items':[['B' if wrong_security else 'A','A','2026-10-13','12.5']] if params.get('offset',0)==0 else []}
    service.sources.register('http',lambda value:FakeAdapter(value))
    job=service.plan({'tables':['sample_prices'],'targets':['postgres'],'start_date':'2026-10-13','end_date':'2026-10-13','codes':['A']})
    service.download(job['id'])
    return service,service.state.job(job['id']),definition,mapping,calls


@pytest.mark.parametrize('legacy_scope',[False,True])
def test_replay_cannot_reassign_security_role_to_hide_wrong_original_response(tmp_path,legacy_scope):
    service,original,definition,mapping,calls=vendor_manager(tmp_path,wrong_security=True)
    assert original['status']=='blocked'
    if legacy_scope:
        # Before validation_scope existed, the original contract and mapping were
        # still frozen. A current contract must not replace their security role.
        plan=deepcopy(original['plan'])
        for request in plan['requests']:request.pop('validation_scope',None)
        with service.state.connect() as conn:
            conn.execute('UPDATE jobs SET plan=? WHERE id=?',(dumps(plan),original['id']))
            for part in original['slices']:
                request=deepcopy(part['request']);request.pop('validation_scope',None)
                conn.execute('UPDATE slices SET request=? WHERE job=? AND id=?',(dumps(request),original['id'],part['id']))
        original=service.state.job(original['id'])
    definition['endpoints']['prices']['security_field']='pretend_code'
    definition['endpoints']['prices']['keys']=['pretend_code','day']
    service.sources.save(definition)
    binding=mapping['bindings'][0];binding['fields'][0]['source_field']='pretend_code'
    binding['join_keys'][0]['source_field']='pretend_code';binding['ignored_fields']={'actual_code':'current alias differs'}
    service.mappings.save(mapping,service.mappings.document('sample_prices')['file_hash'])
    count=len(calls)
    replay=service.replay_archive(original['id'])
    assert len(calls)==count and replay['plan']['requests']==original['plan']['requests']
    assert replay['status']=='blocked'
    rows,errors=service._build_canonical(replay)
    assert not rows and any(e['code']=='security_scope' for e in errors)


def test_replay_uses_original_pagination_when_current_contract_removes_it(tmp_path):
    service,original,definition,mapping,calls=vendor_manager(tmp_path,pagination=True)
    assert original['status']=='downloaded' and len(calls)==2
    terminal=next(p for p in original['slices'] if p['rows']==0)
    with service.state.connect() as conn:
        conn.execute('DELETE FROM slices WHERE job=? AND id=?',(original['id'],terminal['id']))
    definition['endpoints']['prices'].pop('pagination')
    service.sources.save(definition)
    replay=service.replay_archive(original['id'])
    assert len(calls)==2 and replay['status']=='blocked'
    assert replay['plan']['retrieval_source_contracts']['vendor:prices']['pagination']['size']==2
    rows,errors=service._build_canonical(replay)
    assert not rows and any(e['code']=='missing_request' for e in errors)
