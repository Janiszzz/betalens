"""Credential-boundary regressions; synthetic values and in-process HTTP only."""
import asyncio
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace
from urllib.parse import quote

import pytest
import requests
from starlette.websockets import WebSocketDisconnect

from test_manager import manager
from betalens_db_manager import security
from betalens_db_manager.api import create_app
from betalens_db_manager.source_registry import HTTPAdapter
from betalens_db_manager.util import digest, dumps


@pytest.fixture(autouse=True)
def isolated_secret_memory(monkeypatch):
    monkeypatch.setattr(security, '_secrets', set())
    monkeypatch.setattr(security, '_resolved_names', set())


def adapter_fixture(auth, method='GET'):
    definition = {'source_id': 'synthetic_vendor', 'provider': 'http', 'base_url': 'https://source.invalid',
                  'auth': auth, 'endpoints': {'prices': {'method': method, 'fields': [{'name': 'code', 'type': 'text'}]}}}
    logs = []
    state = SimpleNamespace(check_cancel=lambda job: None, log=lambda job, event, **values: logs.append({'event': event, **values}))
    return HTTPAdapter(definition), SimpleNamespace(env_file=None, timeout=1), state, logs


def response(status, payload, *, url='https://source.invalid/prices'):
    value = requests.Response()
    value.status_code = status
    value._content = json.dumps(payload).encode()
    value.url = url
    return value


@pytest.mark.parametrize('auth,method', [
    ({'env': 'EDGE_CREDENTIAL', 'header': 'X-Api-Key', 'prefix': 'Token '}, 'GET'),
    ({'env': 'EDGE_CREDENTIAL', 'query': 'access_key'}, 'GET'),
    ({'env': 'EDGE_CREDENTIAL', 'query': 'opaque'}, 'POST'),
    ({'env': 'EDGE_CREDENTIAL', 'body': 'credentials.opaque'}, 'POST'),
])
def test_http_only_receives_injected_credentials(monkeypatch, auth, method):
    secret = 'SYNTHETIC_EDGE_BOUNDARY_VALUE'
    monkeypatch.setenv('EDGE_CREDENTIAL', secret)
    adapter, settings, state, logs = adapter_fixture(auth, method)
    params = {'market': 'SZSE', 'limit': 0, 'enabled': False}
    original = deepcopy(params)
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return response(200, [{'code': '000001.SZ'}])
    monkeypatch.setattr(requests, 'request', request)
    result = adapter._fetch('prices', params, settings, state, 'synthetic-job')
    assert result == {'fields': ['code'], 'items': [['000001.SZ']]}
    assert len(calls) == 1 and calls[0][1]['allow_redirects'] is False
    assert secret in dumps(calls) and secret not in dumps(logs)
    assert params == original and secret not in dumps(adapter.definition)


def test_redirect_never_follows_or_echoes_location(monkeypatch):
    secret = 'SYNTHETIC_REDIRECT_VALUE'
    monkeypatch.setenv('EDGE_CREDENTIAL', secret)
    adapter, settings, state, logs = adapter_fixture({'env': 'EDGE_CREDENTIAL', 'query': 'opaque'})
    redirected = response(302, {'detail': 'redirect'})
    redirected.headers['Location'] = 'https://foreign.invalid/?opaque=' + secret
    calls = []
    monkeypatch.setattr(requests, 'request', lambda *a, **k: (calls.append(k) or redirected))
    with pytest.raises(ValueError, match='重定向') as failure:
        adapter._fetch('prices', {}, settings, state, 'synthetic-job')
    assert len(calls) == 1 and calls[0]['allow_redirects'] is False
    assert secret not in str(failure.value) and secret not in dumps(logs)


def test_http_error_query_echo_is_redacted(monkeypatch):
    secret = 'SYNTHETIC + QUERY / ERROR'
    monkeypatch.setenv('EDGE_CREDENTIAL', secret)
    adapter, settings, state, logs = adapter_fixture({'env': 'EDGE_CREDENTIAL', 'query': 'opaque'})
    failed = response(401, {'message': secret}, url='https://source.invalid/prices?opaque=' + quote(secret, safe=''))
    monkeypatch.setattr(requests, 'request', lambda *a, **k: failed)
    with pytest.raises(ValueError) as failure:
        adapter._fetch('prices', {}, settings, state, 'synthetic-job')
    assert secret not in str(failure.value) and quote(secret, safe='') not in str(failure.value)
    assert secret not in dumps(logs)
    assert adapter.last_wire == failed.content


@pytest.mark.parametrize('shape', ['object', 'columnar', 'object_key'])
def test_source_echo_cannot_enter_business_archive(monkeypatch, shape):
    secret = 'SYNTHETIC_SOURCE_ECHO_VALUE'
    monkeypatch.setenv('EDGE_CREDENTIAL', secret)
    adapter, settings, state, logs = adapter_fixture({'env': 'EDGE_CREDENTIAL', 'header': 'X-Key'})
    payload = [{'code': secret}] if shape == 'object' else {'fields': ['code'], 'items': [[secret]]}
    if shape == 'object_key':payload = [{secret: 'echoed as a field name'}]
    echoed = response(200, payload)
    monkeypatch.setattr(requests, 'request', lambda *a, **k: echoed)
    with pytest.raises(ValueError, match='鉴权回显') as failure:
        adapter._fetch('prices', {}, settings, state, 'synthetic-job')
    assert secret not in str(failure.value) and secret not in dumps(logs)
    assert adapter.last_wire == echoed.content


def test_short_secret_masks_exact_echo_without_corrupting_business_text():
    security.remember_secrets('a', '123')
    original = {'code': '000123.SZ', 'name': 'market data', 'id': 'a123b', 'enabled': True}
    assert security.redact(original) == original
    security.validate_parameters(original)
    assert security.redact({'value': 'a', 'message': 'received 123', 'apiKey': 'another secret'}) == {
        'value': '[redacted]', 'message': 'received [redacted]', 'apiKey': '[redacted]'}
    with pytest.raises(ValueError):security.validate_parameters({'ordinary': 'a'})


def test_known_secret_check_handles_encoding_without_returning_values():
    secret = 'SYNTHETIC + QUOTED / VALUE'
    security.remember_secrets(secret)
    assert security.contains_known_secret('echo ' + secret) is True
    assert security.contains_known_secret(quote(secret, safe='')) is True
    assert security.contains_known_secret('ordinary') is False
    with pytest.raises(ValueError):security.validate_parameters({'ordinary': quote(secret, safe='')})


@pytest.mark.parametrize('auth', [False, [], '', {'env': ''}])
def test_invalid_auth_descriptors_are_rejected(auth):
    with pytest.raises(ValueError):security.auth_reference(auth)


def test_sensitive_configured_suffix_is_not_a_secret_bypass():
    assert security.public_data({'token_configured': True}) == {'token_configured': True}
    assert security.public_data({'apiKey_configured': 'SYNTHETIC_UNREGISTERED_VALUE'}) == {'apiKey_configured': '[redacted]'}
    with pytest.raises(ValueError):security.validate_parameters({'apiKey_configured': 'SYNTHETIC_UNREGISTERED_VALUE'})


def test_unknown_authorization_text_and_invalid_auth_descriptor_are_masked():
    secret = 'SYNTHETIC_LEGACY_UNKNOWN_VALUE'
    assert secret not in security.safe_error('Authorization: Bearer ' + secret)
    assert secret not in security.safe_error('HTTP header Bearer ' + secret)
    assert security.public_data({'auth': {'env': 'EDGE_CREDENTIAL', 'literal': secret}})['auth'] == '[redacted]'
    assert security.public_data({'auth': {'env': 'EDGE_CREDENTIAL', 'header': 'X-Key'}})['auth']['env'] == 'EDGE_CREDENTIAL'


def test_declared_generic_auth_name_masks_request_but_preserves_task_identity():
    definitions = [{'source_id': 'vendor', 'auth': {'env': 'EDGE_CREDENTIAL', 'query': 'id'}}]
    preview = {'tables': [], 'job': 'job-123'}
    preview['token'] = digest(preview)
    record = {'id': 'job-123', 'preview': preview,
              'plan': {'requests': [{'source_id': 'vendor', 'parameters': {'id': 'SYNTHETIC_OLD_VALUE', 'limit': 7}}]}}
    safe = security.public_data(record, definitions=definitions)
    assert safe['id'] == record['id'] and safe['preview']['token'] == preview['token']
    assert safe['plan']['requests'][0]['parameters'] == {'id': '[redacted]', 'limit': 7}


def test_confirmation_token_is_verified_before_redacting_nested_values():
    preview = {'tables': [], 'requests': [{'parameters': {'apiKey': 'SYNTHETIC_OLD_VALUE'}}]}
    preview['token'] = digest(preview)
    safe = security.public_data({'preview': preview})['preview']
    assert safe['token'] == preview['token']
    assert safe['requests'][0]['parameters']['apiKey'] == '[redacted]'
    invalid = {**preview, 'token': 'unverified-confirmation'}
    assert security.public_data(invalid)['token'] == '[redacted]'


def legacy_app(tmp_path):
    service = manager(tmp_path)
    secret = 'SYNTHETIC_LEGACY_STREAM_VALUE'
    job = service.state.create_job({'contracts': {}, 'request': {'targets': ['postgres'], 'parameters': {'apiKey': secret}}}, service.settings.fingerprint)
    preview = {'tables': [], 'job': job['id']}
    preview['token'] = digest(preview)
    service.state.update_job(job['id'], preview=preview)
    return service, create_app(service), job['id'], secret, preview['token']


def test_legacy_task_rest_and_sse_outputs_are_redacted(tmp_path):
    service, app, job, secret, token = legacy_app(tmp_path)
    rest = next(route for route in app.routes if route.path == '/api/jobs/{job}')
    decoded = json.loads(rest.endpoint(job).body)
    assert secret not in dumps(decoded) and decoded['preview']['token'] == token
    events = next(route for route in app.routes if route.path == '/api/jobs/{job}/events')

    async def first_event():
        result = await events.endpoint(job)
        try:return await result.body_iterator.__anext__()
        finally:await result.body_iterator.aclose()
    event = asyncio.run(first_event())
    assert secret not in event and token in event and job in event


def test_legacy_websocket_logs_and_task_outputs_are_redacted(tmp_path, monkeypatch):
    service, app, job, secret, token = legacy_app(tmp_path)
    monkeypatch.setattr(service.state, 'logs', lambda *a: [{'id': 1, 'parameters': {'access_key': secret}}])
    live = next(route for route in app.routes if route.path == '/api/live/{job}')

    class Socket:
        headers = {}
        def __init__(self):self.messages = []
        async def accept(self):pass
        async def send_text(self, message):
            self.messages.append(message)
            if len(self.messages) == 2:raise WebSocketDisconnect()
    socket = Socket()
    asyncio.run(live.endpoint(socket, job))
    assert len(socket.messages) == 2 and secret not in ''.join(socket.messages)
    assert token in socket.messages[1] and job in socket.messages[1]


def test_generic_auth_fields_are_redacted_in_archive_export(tmp_path):
    raw = json.dumps({'data': [{'id': 'SYNTHETIC_UNREGISTERED_VALUE', 'market': 'SZSE'}]}).encode()
    path = tmp_path/'archive.gz'
    path.write_bytes(gzip.compress(raw))
    definitions = [{'source_id': 'vendor', 'auth': {'env': 'EDGE_CREDENTIAL', 'query': 'id'}}]
    output, original, exported = security.archive_export(path, definitions=definitions)
    assert json.loads(gzip.decompress(output))['data'][0] == {'id': '[redacted]', 'market': 'SZSE'}
    assert gzip.decompress(path.read_bytes()) == raw and original != exported


def test_redacted_archive_preserves_exact_decimal_values(tmp_path):
    path = tmp_path/'decimal.gz'
    path.write_bytes(gzip.compress(b'{"price":1.123456789123456789,"apiKey":"SYNTHETIC_DECIMAL_SECRET"}'))
    output, _, _ = security.archive_export(path)
    decoded = json.loads(gzip.decompress(output))
    assert decoded['price'] == '1.123456789123456789'
    assert decoded['apiKey'] == '[redacted]'
