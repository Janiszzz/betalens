"""Auditing replay, mapping execution, request parameters, and credential leaks."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest

from test_manager import manager,table_definition
from test_retrieval import ArchivedSource
from betalens_db_manager.api import create_app
from betalens_db_manager.security import validate_parameters,request_auth,public_data,archive_export,write_private_wire
from betalens_db_manager.source_registry import HTTPAdapter
from betalens_db_manager.util import digest,dumps


def daily_mapping(s):
    s.registry.save(table_definition())
    rules=[{'source_field':a,'column':b} for a,b in [('ts_code','code'),('trade_date','business_date'),('close','price')]]
    fields=s.sources.endpoint('tushare','daily')['fields']
    value={'format_version':2,'table':'sample_prices','bindings':[{'source_id':'tushare','endpoint':'daily','mapping_mode':'direct','join_keys':rules[:2],'fields':rules,
        'ignored_fields':{f['name']:'not used in this test' for f in fields if f['name'] not in {'ts_code','trade_date','close'}}}]}
    s.mappings.save(value,None)
    return value


def test_wrong_security_remains_blocked_after_replay_without_new_fetch(tmp_path):
    s=manager(tmp_path);daily_mapping(s)
    source=ArchivedSource(lambda endpoint,params:[{'ts_code':'WRONG.SZ','trade_date':'20261013','close':'12.5'}] if endpoint=='daily' else [])
    s.source=source
    job=s.plan({'tables':['sample_prices'],'start_date':'2026-10-13','end_date':'2026-10-13','codes':['EXPECTED.SZ'],'targets':['postgres']})
    s.download(job['id']);calls=len(source.calls)
    replay=s.replay_archive(job['id'])
    assert len(source.calls)==calls and replay['status']=='blocked'
    rows,rejects=s._build_canonical(replay)
    assert not rows and any(r.get('code')=='security_scope' for r in rejects)
    assert replay['plan']['requests']==job['plan']['requests']


def test_custom_builtin_cannot_silently_skip_required_value(tmp_path):
    s=manager(tmp_path);m=daily_mapping(s)
    m['bindings'][0]['fields'][2].update(builtin=True,formula='native_index_value',required=True)
    with pytest.raises(ValueError,match='内置处理器'):s.mappings.validate(m)
    with pytest.raises(ValueError,match='内置处理器'):
        s.mappings.map_row(s.registry.spec('sample_prices'),{'close':'12.5'},m['bindings'][0])


def test_snapshot_keeps_three_level_parameters_and_freezes_them(tmp_path):
    s=manager(tmp_path);s.registry.save(table_definition('snapshot_prices'))
    source={'source_id':'vendor','provider':'http','base_url':'https://example.invalid','endpoints':{'prices':{'strategy':'snapshot','keys':['code','day'],
        'parameters':{'exchange':'SSE','contract_only':0},'fields':[{'name':'code','type':'text'},{'name':'day','type':'date'},{'name':'price','type':'numeric'}]}}}
    s.sources.save(source)
    rules=[{'source_field':a,'column':b} for a,b in [('code','code'),('day','business_date'),('price','price')]]
    s.mappings.save({'format_version':2,'table':'snapshot_prices','bindings':[{'source_id':'vendor','endpoint':'prices','parameters':{'exchange':'BSE','binding_only':False},'fields':rules,'join_keys':rules[:2]}]},None)
    request={'tables':['snapshot_prices'],'start_date':'2026-10-13','end_date':'2026-10-13','parameters':{'prices':{'exchange':'SZSE','limit':7}}}
    original=deepcopy(request);plan=s.compile(request)
    assert request==original and plan['requests'][0]['parameters']=={'exchange':'SZSE','contract_only':0,'binding_only':False,'limit':7}
    job=s.plan(request);source['endpoints']['prices']['parameters']['exchange']='CHANGED';s.sources.save(source)
    assert s.state.job(job['id'])['plan']['requests']==plan['requests']


@pytest.mark.parametrize('key',['token','api_key','apiKey','apikey','access_key','Authorization'])
def test_credential_aliases_and_nested_lists_cannot_enter_parameters(key):
    with pytest.raises(ValueError):validate_parameters({'filters':[{'nested':{key:'SYNTHETIC_CREDENTIAL'}}]})
    validate_parameters({'market':'SZSE','enabled':False,'count':0})


@pytest.mark.parametrize('auth,method',[
    ({'env':'AUDIT_API_SECRET','header':'X-Key'},'GET'),
    ({'env':'AUDIT_API_SECRET','location':'query','name':'apiKey'},'GET'),
    ({'env':'AUDIT_API_SECRET','location':'body','name':'auth.key'},'POST')])
def test_auth_is_resolved_at_request_boundary_and_outputs_are_redacted(tmp_path,monkeypatch,auth,method):
    secret='SYNTHETIC_BOUNDARY_CREDENTIAL';monkeypatch.setenv('AUDIT_API_SECRET',secret)
    s=manager(tmp_path);params={'market':'SZSE'}
    headers,query,body=request_auth(auth,s.settings,params,method)
    assert secret in dumps([headers,query,body]) and secret not in dumps(params)
    s.state.log('test','http_error',parameters={auth.get('name',auth.get('header')):secret},message='upstream echo '+secret)
    assert secret not in dumps(s.state.logs('test'))
    assert secret not in dumps(public_data({'parameters':{'apiKey':secret},'message':'upstream '+secret},s.settings))
    with pytest.raises(ValueError):validate_parameters({'ordinary_name':secret})


def test_public_api_preserves_confirmation_token_and_redacts_legacy_task(tmp_path):
    s=manager(tmp_path);secret='SYNTHETIC_LEGACY_CREDENTIAL'
    job=s.state.create_job({'contracts':{},'request':{'parameters':{'apiKey':secret},'targets':['postgres']}},s.settings.fingerprint)
    app=create_app(s)
    route=next(r for r in app.routes if r.path=='/api/jobs/{job}' and 'GET' in r.methods)
    body=json.loads(route.endpoint(job['id']).body)
    assert secret not in dumps(body) and body['plan']['request']['parameters']['apiKey']=='[redacted]'
    preview={'tables':[],'mapping_version':'version'};preview['token']=digest(preview)
    assert public_data(preview)['token']==preview['token']


def test_raw_wire_stays_private_and_download_has_separate_redacted_hash(tmp_path):
    secret='SYNTHETIC_WIRE_CREDENTIAL'
    data=json.dumps({'apiKey':secret,'safe':1}).encode();path=tmp_path/'private-wire.gz'
    write_private_wire(path,data)
    assert path.stat().st_mode&0o777==0o600
    exported,original_hash,export_hash=archive_export(path)
    import gzip,hashlib
    assert secret not in gzip.decompress(exported).decode()
    assert gzip.decompress(path.read_bytes())==data
    assert original_hash==hashlib.sha256(data).hexdigest() and export_hash!=original_hash
