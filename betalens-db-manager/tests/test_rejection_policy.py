from decimal import Decimal
from pathlib import Path
import json

import pytest

from betalens_db_manager.canonical import canonical, registry_context
from betalens_db_manager.catalog import dataset
from betalens_db_manager.comparison import row_key, typed
from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.pipeline import normalize_child
from betalens_db_manager.pipeline_options import light_normalization
from betalens_db_manager.response_validation import validate_archives, validate_response
from betalens_db_manager.service import TushareImportService, normalize_source
from betalens_db_manager.source import SourceError
from betalens_db_manager.util import digest
from test_manager import cached_part, manager, live_manager
from test_pipeline_v2 import pipeline_http, prepare
from test_retrieval import task


def test_bang_codes_preserve_distinct_source_identity_and_suffix_validation():
    spec = canonical('fund_nav_daily')
    historical = {'security_code': '110011!1.OF', 'nav_date': '2011-02-09'}
    current = {**historical, 'security_code': '110011.OF'}
    assert typed(spec, historical)['security_code'] == '110011!1.OF'
    assert row_key(spec, historical) != row_key(spec, current)
    for code in ['150021!1.SZ', '166007!1.SZ', '206001!1.OF']:
        assert typed(spec, {**historical, 'security_code': code})['security_code'] == code
    for code in ['110011!1', '110011!1.OTHER', '110011 !1.OF', "110011';.OF"]:
        with pytest.raises(ValueError):
            typed(spec, {**historical, 'security_code': code})


def quote(source, close, **values):
    return dict(asset_type='stock', security_code='000901.SZ', trade_date='2009-07-14',
                source_endpoint=source, data_quality_status='accepted', close=Decimal(close), **values)


@pytest.mark.parametrize('reverse', [False, True])
def test_cross_feed_close_rounding_selects_daily_independent_of_order(reverse):
    service = TushareImportService.__new__(TushareImportService)
    rows = [quote('daily', '13.94'), quote('daily_basic', '13.9389')]
    if reverse:
        rows.reverse()
    result = service._merge(rows, canonical('market_daily_raw'))[0]
    assert result['close'] == Decimal('13.94')
    assert result['data_quality_status'] == 'accepted'
    assert set(result['source_endpoint'].split(',')) == {'daily', 'daily_basic'}


@pytest.mark.parametrize('rows', [
    [quote('daily', '13.94'), quote('daily_basic', '13.93')],
    [quote('daily', '13.94'), quote('daily', '13.9389')],
    [quote('daily', '13.94', volume=Decimal('10')), quote('daily_basic', '13.9389', volume=Decimal('11'))],
])
def test_close_tolerance_does_not_hide_real_or_unrelated_conflicts(rows):
    service = TushareImportService.__new__(TushareImportService)
    assert service._merge(rows, canonical('market_daily_raw'))[0]['data_quality_status'] == 'conflicting_source_rows'


@pytest.mark.parametrize('omit_field', [False, True])
def test_undated_nav_reaches_both_local_tables_with_unknown_availability(tmp_path, omit_field):
    service = manager(tmp_path)
    job = service.plan({'tables': ['fund_nav_daily', 'fund_nav_observation'], 'source_id': 'tushare',
                        'endpoints': ['fund_nav'], 'asset_types': ['fund'], 'codes': ['519505.OF'],
                        'start_date': '2005-01-04', 'end_date': '2005-01-04'})
    contract = service.sources.endpoint('tushare', 'fund_nav')
    fields = [f['name'] for f in contract['fields'] if not omit_field or f['name'] != 'ann_date']
    raw = {'ts_code': '519505.OF', 'ann_date': None, 'nav_date': '20050104', 'unit_nav': '1'}
    data = {'fields': fields, 'items': [[raw.get(f) for f in fields]]}
    request = next(r for r in job['plan']['requests'] if r['endpoint'] == 'fund_nav')
    assert not validate_response(job['plan'], request, contract, data)
    cached_part(service, job, 'fund_nav', data)
    with light_normalization(), registry_context(service.registry):
        tables, rejects = service._build_canonical(service.state.job(job['id']))
    assert not rejects
    for name, column in [('fund_nav_daily', 'announcement_date'), ('fund_nav_observation', 'ann_date')]:
        assert len(tables[name]) == 1
        assert tables[name][0][column] is None
        assert tables[name][0]['available_at'] is None
    rows, errors = normalize_source(dataset('fund_nav'), data, job['id'], 'slice', '2026-10-29T00:00:00Z')
    assert len(rows) == 1 and not errors
    # A missing NAV date still prevents an unidentified record from importing.
    invalid = {'fields': fields, 'items': [[None if f == 'nav_date' else raw.get(f) for f in fields]]}
    assert normalize_source(dataset('fund_nav'), invalid, job['id'], 'slice', '2026-10-29T00:00:00Z')[1]


@pytest.mark.parametrize('workers', [1, 4])
def test_transient_failure_retries_same_request_then_reuses_archive(tmp_path, monkeypatch, workers):
    count = 0
    def response(endpoint, params):
        nonlocal count
        count += 1
        if count < 3:
            raise SourceError('temporary connection failure', kind='network_error')
        return [{'ts_code': '000001.SZ', 'trade_date': '20260930'}]
    service, source, item = task(tmp_path, 'daily', {'trade_date': '20260930'}, response, cap=6000)
    waits = []
    monkeypatch.setattr(TaskDownloader, 'retry_wait', lambda _, seconds: waits.append(seconds))
    downloader = TaskDownloader(service, item)
    assert not downloader.run(item['plan']['requests'], workers=workers)
    assert count == 3 and waits == [5, 30]
    assert len(service.state.job(item['id'])['slices']) == 1
    assert not validate_archives(service, service.state.job(item['id']))
    downloader.run(item['plan']['requests'], workers=workers)
    assert count == 3


def test_exhausted_network_failure_is_bounded_and_cancellation_interrupts_wait(tmp_path, monkeypatch):
    def unavailable(*_):
        raise SourceError('offline', kind='network_error')
    service, source, item = task(tmp_path, 'daily', {'trade_date': '20260930'}, unavailable, cap=6000)
    monkeypatch.setattr(TaskDownloader, 'retry_wait', lambda *_: None)
    errors = TaskDownloader(service, item).run(item['plan']['requests'], workers=2)
    assert len(source.calls) == 3 and len(errors) == 1
    part = service.state.job(item['id'])['slices'][0]
    assert part['status'] == 'failed' and part['result']['retry_rounds'] == 3
    assert part['result']['kind'] == 'network_error' and not part['archive']
    def cancel(downloader, _):
        service.state.cancel(item['id'])
        service.state.check_cancel(item['id'])
    monkeypatch.setattr(TaskDownloader, 'retry_wait', cancel)
    with pytest.raises(InterruptedError):
        TaskDownloader(service, service.state.job(item['id'])).run(item['plan']['requests'], workers=2)
    assert len(source.calls) == 4


@pytest.mark.parametrize('kind,message',[('permission_denied','权限不足'),('quota_exhausted','配额耗尽')])
def test_permission_and_contract_errors_are_not_network_retries(tmp_path, monkeypatch, kind, message):
    def denied(*_):
        raise SourceError(message, kind=kind)
    service, source, item = task(tmp_path, 'daily', {'trade_date': '20260930'}, denied, cap=6000)
    monkeypatch.setattr(TaskDownloader, 'retry_wait', lambda *_: pytest.fail('权限错误不得重试'))
    TaskDownloader(service, item).run(item['plan']['requests'], workers=2)
    assert len(source.calls) == 1
    assert service.state.job(item['id'])['slices'][0]['status'] == 'unavailable'


def test_missing_archive_refetches_only_missing_page(tmp_path):
    def response(_, params):
        return [] if params['offset'] else [{'ts_code': '510300.SH', 'trade_date': '20260930'}]
    service, source, item = task(tmp_path, 'fund_share', {'trade_date': '20260930'}, response, cap=2000, mode='offset_probe')
    TaskDownloader(service, item).run(item['plan']['requests'], workers=2)
    parts = service.state.job(item['id'])['slices']
    first = next(p for p in parts if p['rows'])
    Path(first['archive']).unlink()
    before = len(source.calls)
    TaskDownloader(service, service.state.job(item['id'])).run(item['plan']['requests'], workers=2)
    assert len(source.calls) == before + 1
    assert not validate_archives(service, service.state.job(item['id']))


def test_policy_change_invalidates_old_rejected_conversion_cache(tmp_path, monkeypatch):
    service = manager(tmp_path)
    job = service.plan({'pipeline_version': 2, 'tables': ['fund_nav_daily'], 'source_id': 'tushare', 'asset_types': ['fund'],
                        'codes': ['519505.OF'], 'start_date': '2005-01-04', 'end_date': '2005-01-04'})
    calls = []
    from betalens_db_manager.manager import DatabaseManagerService
    monkeypatch.setattr(DatabaseManagerService, '_build_canonical', lambda *_: calls.append(1) or ({}, []))
    result = normalize_child(service.settings, job['id'])
    path = service.settings.state_dir / 'pipeline' / job['id'] / 'normalized.json'
    old_key = digest([[], job['plan']['mapping_version'], job['plan']['canonical_contracts'],
                      job['plan']['processor_version'], job['plan']['request']])
    path.write_text(json.dumps({**result, 'cache_key': old_key, 'rejected': 1, 'reject_sample': [{'reason': 'old rule'}]}))
    assert normalize_child(service.settings, job['id'])['rejected'] == 0
    assert len(calls) == 2
    assert normalize_child(service.settings, job['id'])['rejected'] == 0
    assert len(calls) == 2


def test_real_network_retry_completes_same_task_in_both_databases(live_manager, pipeline_http, monkeypatch):
    from betalens_db_manager.pipeline import Pipeline
    from betalens_db_manager.source_registry import HTTPAdapter
    service = live_manager
    assert service.settings.pg_database == service.settings.ch_database
    assert service.settings.pg_database.startswith('manager_test_')
    assert service.settings.env_file is None
    request = prepare(service, pipeline_http)
    with registry_context(service.registry):
        service.initialize(service.schema_plan()['token'])
        service.schema.apply(service.schema.plan('pipeline_quotes')['token'])
    fetch = HTTPAdapter._fetch
    failed = []
    def transient(*args, **kwargs):
        if not failed:
            failed.append(True)
            raise SourceError('temporary transport failure', kind='network_error')
        return fetch(*args, **kwargs)
    waits = []
    monkeypatch.setattr(HTTPAdapter, '_fetch', transient)
    monkeypatch.setattr(TaskDownloader, 'retry_wait', lambda _, seconds: waits.append(seconds))
    job = service.plan(request)
    assert Pipeline(service).run(job['id'])['status'] == 'completed'
    assert waits == [5]
    with registry_context(service.registry):
        assert len(service.pg.query('pipeline_quotes')['rows']) == 2
        assert len(service.ch.query('pipeline_quotes')['rows']) == 2
