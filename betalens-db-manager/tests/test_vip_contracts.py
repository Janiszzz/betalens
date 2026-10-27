"""VIP field/filter parity is documented; pagination needs archived evidence."""
import gzip
import hashlib
from types import SimpleNamespace

import pytest

from betalens_db_manager.catalog import load_catalog
from betalens_db_manager.config import Settings
from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.response_validation import validate_response
from betalens_db_manager.retrieval import tushare_policy
from betalens_db_manager.state import State
from betalens_db_manager.util import dumps


BASES = ('income', 'balancesheet', 'cashflow', 'fina_indicator')


@pytest.mark.parametrize('base', BASES)
def test_vip_keeps_full_base_fields_but_does_not_claim_published_page_limit(base):
    ordinary, vip = load_catalog()[base], load_catalog()[base+'_vip']
    assert vip.fields == ordinary.fields
    assert vip.keys == ordinary.keys
    assert vip.date_field == ordinary.date_field == 'end_date'
    assert vip.document == ordinary.document
    assert set(ordinary.input_fields) <= set(vip.input_fields)
    assert {'ts_code', 'period', 'offset', 'limit'} <= set(vip.input_fields)
    assert vip.limit == 1000 and vip.strategy == 'periodic'
    policy = tushare_policy(base+'_vip')
    assert policy['mode'] == 'offset_probe'
    assert policy['basis'] == 'runtime_probe_required'
    assert policy['limit_basis'] == 'conservative_guard_not_published'
    assert policy['row_limit'] == 1000
    # The ordinary financial-indicator 100-row limit is not a VIP guarantee.
    if base == 'fina_indicator':
        assert ordinary.limit == 100


def source_response(contract, code='000333.SZ', period='20241231', announcement='20250328'):
    names = [field['name'] for field in contract['fields']][::-1]
    values = {'ts_code': code, 'end_date': period, 'ann_date': announcement,
              'f_ann_date': announcement, 'report_type': '1', 'comp_type': '1'}
    return {'fields': names, 'items': [[values.get(name) for name in names]]}


@pytest.mark.parametrize('base', BASES)
def test_vip_ignoring_single_security_or_period_is_blocked_before_local_filter(base):
    endpoint = base+'_vip';contract = load_catalog()[endpoint].as_dict()
    plan = {'request': {'codes': ['000333.SZ']}}
    request = {'source_id': 'tushare', 'endpoint': endpoint, 'strategy': 'periodic',
               'parameters': {'ts_code': '000333.SZ', 'period': '20241231', 'limit': 1, 'offset': 0}}
    assert validate_response(plan, request, contract, source_response(contract)) == []
    wrong_code = validate_response(plan, request, contract, source_response(contract, code='600519.SH'))
    assert 'security_scope' in {error['code'] for error in wrong_code}
    wrong_period = validate_response(plan, request, contract, source_response(contract, period='20240930'))
    assert 'date_scope' in {error['code'] for error in wrong_period}
    # Explicit all-market requests do not pretend that a single-security filter
    # was supplied. They still enforce the requested report period.
    market = {**request, 'parameters': {'period': '20241231', 'limit': 1, 'offset': 0}}
    assert validate_response({'request': {'codes': []}}, market, contract,
                             source_response(contract, code='600519.SH')) == []


@pytest.mark.parametrize('base', BASES)
def test_vip_short_pages_require_advancing_offsets_and_empty_terminal_archive(tmp_path, base):
    endpoint = base+'_vip';contract = load_catalog()[endpoint].as_dict()
    settings = Settings(state_dir=tmp_path/'state', config_dir=tmp_path/'config')
    state = State(settings.state_dir)

    class ArchivedSource:
        def __init__(self):
            self.offsets = []

        def fetch(self, name, params, fields, archive):
            assert name == endpoint
            assert set(fields) == {field['name'] for field in contract['fields']}
            assert params['limit'] == 1
            offset = params['offset'];self.offsets.append(offset)
            response = source_response(contract, announcement='20250328' if offset == 0 else '20250401')
            if offset >= 2:
                response['items'] = []
            payload = dumps({'code': 0, 'data': response}).encode()
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(gzip.compress(payload))
            return response, hashlib.sha256(payload).hexdigest()

    source = ArchivedSource()
    service = SimpleNamespace(settings=settings, state=state, source=source, safe_error=str,
                              sources=SimpleNamespace(values={'tushare': {'provider': 'tushare'}}))
    request = {'source_id': 'tushare', 'endpoint': endpoint, 'strategy': 'periodic',
               'parameters': {'ts_code': '000333.SZ', 'period': '20241231', 'limit': 1}}
    plan = {'request': {'codes': ['000333.SZ'], 'tables': [], 'targets': ['postgres']},
            'requests': [request], 'source_contracts': {'tushare:'+endpoint: contract},
            'retrieval_policies': {'tushare:'+endpoint: tushare_policy(endpoint)}}
    item = state.create_job(plan, settings.fingerprint)
    TaskDownloader(service, item).run()
    assert source.offsets == [0, 1, 2]
    slices = state.job(item['id'])['slices']
    assert len(slices) == 3
    assert [part['rows'] for part in slices if part['status'] == 'downloaded'] == [0]
    assert sum(part['rows'] for part in slices if part['status'] == 'paged') == 2
    assert not any(part['status'] == 'contract_blocked' for part in slices)
    TaskDownloader(service, state.job(item['id'])).run()
    assert source.offsets == [0, 1, 2]


@pytest.mark.parametrize('suffix', ['', '_vip'])
@pytest.mark.parametrize('base', BASES)
def test_financial_windows_use_official_date_dimension_and_not_exact_upper_bound(base, suffix):
    endpoint=base+suffix;contract=load_catalog()[endpoint].as_dict()
    params={'ts_code':'000333.SZ'}
    if base=='fina_indicator':
        # A report inside the interval may precede its end bound, and its later
        # announcement date is not what this interface's window constrains.
        params.update(start_date='20240101',end_date='20241231')
        response=source_response(contract,period='20240930',announcement='20250328')
        wrong=source_response(contract,period='20231231',announcement='20240328')
    else:
        params.update(period='20241231',start_date='20250301',end_date='20250331')
        response=source_response(contract,announcement='20250328')
        wrong=source_response(contract,announcement='20250401')
    request={'source_id':'tushare','endpoint':endpoint,'parameters':params}
    plan={'request':{'codes':['000333.SZ']}}
    assert validate_response(plan,request,contract,response)==[]
    assert 'date_scope' in {e['code'] for e in validate_response(plan,request,contract,wrong)}


@pytest.mark.parametrize('base', BASES)
def test_exact_period_and_announcement_date_are_both_enforced(base):
    endpoint=base+'_vip';contract=load_catalog()[endpoint].as_dict()
    request={'source_id':'tushare','endpoint':endpoint,
             'parameters':{'ts_code':'000333.SZ','period':'20241231','ann_date':'20250328'}}
    plan={'request':{'codes':['000333.SZ']}}
    assert validate_response(plan,request,contract,source_response(contract))==[]
    for response in (source_response(contract,announcement='20250329'),
                     source_response(contract,period='20240930')):
        assert 'date_scope' in {e['code'] for e in validate_response(plan,request,contract,response)}


@pytest.mark.parametrize('suffix', ['', '_vip'])
@pytest.mark.parametrize('base', ['income','cashflow'])
def test_actual_announcement_exact_filter_uses_f_ann_date_when_documented(base, suffix):
    endpoint=base+suffix;contract=load_catalog()[endpoint].as_dict()
    request={'source_id':'tushare','endpoint':endpoint,'parameters':{
        'ts_code':'000333.SZ','period':'20241231','ann_date':'20250328','f_ann_date':'20250330'}}
    response=source_response(contract)
    position=response['fields'].index('f_ann_date');response['items'][0][position]='20250330'
    assert validate_response({'request':{'codes':['000333.SZ']}},request,contract,response)==[]
    response['items'][0][position]='20250331'
    assert 'date_scope' in {e['code'] for e in validate_response({},request,contract,response)}


@pytest.mark.parametrize('suffix', ['', '_vip'])
@pytest.mark.parametrize('base', ['balancesheet','fina_indicator'])
def test_undocumented_actual_announcement_filter_cannot_be_silently_accepted(base, suffix):
    endpoint=base+suffix;contract=load_catalog()[endpoint].as_dict()
    request={'source_id':'tushare','endpoint':endpoint,'parameters':{'period':'20241231','f_ann_date':'20250328'}}
    assert 'f_ann_date' not in contract['input_fields']
    errors=validate_response({},request,contract,source_response(contract))
    assert 'unsupported_date_filter' in {error['code'] for error in errors}


@pytest.mark.parametrize('base', BASES)
def test_financial_filter_cannot_be_proven_from_missing_or_null_date_field(base):
    endpoint=base+'_vip';contract=load_catalog()[endpoint].as_dict()
    date_field='end_date' if base=='fina_indicator' else 'ann_date'
    request={'source_id':'tushare','endpoint':endpoint,'parameters':{
        'ts_code':'000333.SZ','start_date':'20240101' if base=='fina_indicator' else '20250301',
        'end_date':'20241231' if base=='fina_indicator' else '20250331'}}
    response=source_response(contract)
    position=response['fields'].index(date_field)
    response['items'][0][position]=None
    assert 'date_format' in {e['code'] for e in validate_response({},request,contract,response)}
    # Use an optional date in this contract variant to prove that active
    # filters require their date even independently of business-key checks.
    contract={**contract,'keys':[key for key in contract['keys'] if key!=date_field]}
    response['fields'].pop(position);response['items'][0].pop(position)
    assert 'missing_date_fields' in {e['code'] for e in validate_response({},request,contract,response)}


def test_invalid_financial_request_window_is_rejected_even_for_empty_response():
    endpoint='income_vip';contract=load_catalog()[endpoint].as_dict()
    request={'source_id':'tushare','endpoint':endpoint,'parameters':{'start_date':'20250331','end_date':'20250301'}}
    response=source_response(contract);response['items']=[]
    assert 'date_request' in {e['code'] for e in validate_response({},request,contract,response)}
