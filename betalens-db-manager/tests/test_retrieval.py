from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest

from betalens_db_manager.catalog import load_catalog
from betalens_db_manager.config import Settings
from betalens_db_manager.manager import DatabaseManagerService
from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.retrieval import request_rate, tushare_policy, partition
from betalens_db_manager.source import TushareSource, read_archive
from betalens_db_manager.state import State
from betalens_db_manager.util import dumps


class Clock:
    def __init__(self): self.value = 1000.
    def time(self): return self.value
    def sleep(self, seconds): self.value += seconds


@pytest.fixture
def clock(monkeypatch):
    import betalens_db_manager.source as module
    value = Clock()
    monkeypatch.setattr(module.time, 'time', value.time)
    monkeypatch.setattr(module.time, 'monotonic', value.time)
    monkeypatch.setattr(module.time, 'sleep', value.sleep)
    return value


def test_shared_rate_task_override_fixed_endpoint_cap_and_cancellation(tmp_path, clock):
    settings = Settings(state_dir=tmp_path, requests_per_minute=200)
    state = State(settings.state_dir)
    a, b = TushareSource(settings, state), TushareSource(settings, state)
    assert a._rate('stock_basic') == 50
    b._rate('stock_basic'); assert clock.value == pytest.approx(1001.2)
    a.requests_per_minute = 20
    a._rate('daily'); before = clock.value
    b._rate('daily'); assert clock.value-before == pytest.approx(3.)
    task = state.create_job({}, settings.fingerprint)
    a.active_job = task['id']; state.cancel(task['id'])
    with pytest.raises(InterruptedError): a._rate('daily')


def test_http_429_and_minute_quota_waits_count_retries(tmp_path, clock, monkeypatch):
    import betalens_db_manager.source as module
    settings = Settings(state_dir=tmp_path, token='secret-never-log', requests_per_minute=500)
    source = TushareSource(settings, State(tmp_path))
    task = source.state.create_job({}, settings.fingerprint); source.active_job = task['id']
    class Reply:
        status_code = 200
        headers = {}
        content = b'{"code":0,"data":{"fields":["ts_code"],"items":[]}}'
    calls = []
    def post(*args, **kwargs):
        calls.append(clock.value)
        r = Reply()
        if len(calls) == 1: r.status_code = 429; r.headers = {'Retry-After': '2'}
        if len(calls) == 2: r.content = '{"code":-2002,"msg":"每分钟最多访问50次"}'.encode()
        return r
    monkeypatch.setattr(module.requests, 'post', post)
    source.fetch('daily', {}, ['ts_code'], tmp_path/'page.gz')
    assert calls == pytest.approx([1000, 1002, 1062])
    logs = source.state.logs(task['id'])
    assert len([r for r in logs if r['event'] == 'request_started']) == 3
    assert 'secret-never-log' not in dumps(logs)


def test_policy_covers_31_contracts_and_partitions_are_disjoint():
    assert len(load_catalog()) == 31
    assert tushare_policy('fund_adj')['mode'] == 'offset'
    assert tushare_policy('fund_daily')['row_limit'] == 5000
    assert tushare_policy('bse_mapping')['row_limit'] == 1000
    assert tushare_policy('index_basic')['row_limit'] == 8000
    chunks = partition('daily', {'start_date': '20261013', 'end_date': '20261016'}, load_catalog()['daily'].as_dict())
    assert chunks == [{'start_date': '20261013', 'end_date': '20261014'}, {'start_date': '20261015', 'end_date': '20261016'}]
    for value in (0, -1, '1.5', 1.5, True, 'unknown', 100001):
        with pytest.raises(ValueError): request_rate(value)


class ArchivedSource:
    def __init__(self, response): self.response, self.calls = response, []
    def fetch(self, endpoint, params, fields, path):
        self.calls.append((endpoint, dict(params)))
        rows = self.response(endpoint, params)
        data = {'fields': fields[::-1], 'items': [[row.get(f) for f in fields[::-1]] for row in rows]}
        payload = dumps({'code': 0, 'data': data}).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wb') as handle: handle.write(payload)
        return data, hashlib.sha256(payload).hexdigest()


def task(tmp_path, endpoint, params, response, cap=2, mode='partition'):
    source = ArchivedSource(response)
    s = DatabaseManagerService(Settings(state_dir=tmp_path/'state', config_dir=tmp_path/'config'), source)
    request = {'source_id': 'tushare', 'endpoint': endpoint, 'parameters': params, 'table': 'test'}
    policy = {**tushare_policy(endpoint), 'row_limit': cap, 'mode': mode}
    plan = {'requests': [request], 'request': {'targets': ['postgres'], 'tables': []}, 'source_contracts': {'tushare:'+n: d.as_dict() for n, d in load_catalog().items()}, 'retrieval_policies': {'tushare:'+endpoint: policy}}
    item = s.state.create_job(plan, s.settings.fingerprint)
    return s, source, item


def test_offset_short_pages_until_empty_resume_and_no_duplicate_requests(tmp_path):
    def response(endpoint, params):
        # Server enforces one row although client asks for two.
        offset = params['offset']
        return [{'ts_code': str(offset), 'trade_date': '20261013', 'adj_factor': '1'}] if offset < 3 else []
    s, source, item = task(tmp_path, 'fund_adj', {'trade_date': '20261013'}, response, mode='offset')
    TaskDownloader(s, item).run()
    assert [p['offset'] for _, p in source.calls] == [0, 1, 2, 3]
    assert [p['rows'] for p in s.state.job(item['id'])['slices']].count(0) == 1
    TaskDownloader(s, s.state.job(item['id'])).run()
    assert len(source.calls) == 4


@pytest.mark.parametrize('overlap', [False, True])
def test_repeated_short_or_overlapping_page_blocks(tmp_path, overlap):
    def response(endpoint, params):
        codes = ['a', 'b'] if params['offset'] == 0 and overlap else ['b']
        return [{'ts_code': code, 'trade_date': '20261013', 'adj_factor': '1'} for code in codes]
    s, source, item = task(tmp_path, 'fund_adj', {}, response, cap=100, mode='offset')
    TaskDownloader(s, item).run()
    blocked = [p for p in s.state.job(item['id'])['slices'] if p['status'] == 'contract_blocked']
    assert len(blocked) == 1 and len(source.calls) == 2


def test_date_split_discards_parent_and_saturated_single_code_blocks(tmp_path):
    def response(endpoint, params):
        values = [{'ts_code': 'a', 'trade_date': day, 'close': '1'} for day in ('20261013', '20261014', '20261015')]
        return [v for v in values if params['start_date'] <= v['trade_date'] <= params['end_date']][:2]
    s, source, item = task(tmp_path, 'daily', {'ts_code': 'a', 'start_date': '20261013', 'end_date': '20261015'}, response)
    TaskDownloader(s, item).run()
    parts = s.state.job(item['id'])['slices']
    assert sum(p['rows'] for p in parts if p['status'] == 'downloaded') == 3
    assert any(p['status'] == 'split' for p in parts)
    rows = []
    for p in parts:
        if p['status'] == 'downloaded': rows.extend(read_archive(Path(p['archive']), p['sha'])['items'])
    assert len(rows) == 3
    s2, source2, item2 = task(tmp_path/'blocked', 'daily', {'ts_code': 'a', 'trade_date': '20261013'}, lambda *_: [{'ts_code': 'a'}, {'ts_code': 'b'}])
    TaskDownloader(s2, item2).run()
    assert s2.state.job(item2['id'])['slices'][0]['status'] == 'contract_blocked'


def test_saturated_market_uses_source_complete_universe_not_partial_database(tmp_path):
    def response(endpoint, params):
        if endpoint == 'stock_basic':
            return [{'ts_code': code} for code in ('a', 'b', 'c')] if params['list_status'] == 'L' else []
        codes = [params['ts_code']] if 'ts_code' in params else ['a', 'b']
        return [{'ts_code': code, 'trade_date': '20261013'} for code in codes]
    s, source, item = task(tmp_path, 'daily', {'trade_date': '20261013'}, response)
    TaskDownloader(s, item).run()
    parts = s.state.job(item['id'])['slices']
    assert {p['request']['parameters']['ts_code'] for p in parts if p['dataset'] == 'daily' and p['status'] == 'downloaded'} == {'a', 'b', 'c'}
    assert len([p for p in parts if p['request'].get('dependency_only')]) == 5
    count = len(source.calls)
    TaskDownloader(s, s.state.job(item['id'])).run()
    assert len(source.calls) == count


def test_interrupted_page_chain_recovers_from_last_successful_page(tmp_path):
    failed = False
    def response(endpoint, params):
        nonlocal failed
        if params['offset'] == 1 and not failed:
            failed = True
            raise RuntimeError('temporary failure')
        return [{'ts_code': str(params['offset']), 'trade_date': '20261013'}] if params['offset'] < 2 else []
    s, source, item = task(tmp_path, 'fund_adj', {}, response, mode='offset')
    with pytest.raises(RuntimeError): TaskDownloader(s, item).run()
    TaskDownloader(s, s.state.job(item['id'])).run()
    assert [p['offset'] for _, p in source.calls] == [0, 1, 1, 2]


def test_compile_pins_rate_and_policy_without_accessing_network(tmp_path):
    s = DatabaseManagerService(Settings(state_dir=tmp_path/'state', config_dir=tmp_path/'config'))
    request = {'tables': ['security_dim'], 'asset_types': ['stock'], 'codes': ['000001.SZ'], 'requests_per_minute': 200}
    item = s.plan(request)
    assert item['plan']['request']['requests_per_minute'] == 200
    assert item['plan']['retrieval_policies']['tushare:stock_basic']['effective_rpm'] == 50
    other = DatabaseManagerService(s.settings).state.job(item['id'])
    assert other['plan'] == item['plan']
    with pytest.raises(ValueError): s.compile({**request, 'requests_per_minute': 0})


def test_source_ignoring_security_or_date_filter_cannot_pass_completeness(tmp_path):
    s, _, item = task(tmp_path, 'daily', {'ts_code': 'a', 'trade_date': '20261013'}, lambda *_: [{'ts_code': 'b', 'trade_date': '20261013'}])
    TaskDownloader(s, item).run()
    assert s.state.job(item['id'])['slices'][0]['status'] == 'contract_blocked'
    s, _, item = task(tmp_path/'date', 'fund_adj', {'trade_date': '20261013'}, lambda *_: [{'ts_code': 'a', 'trade_date': '20261014'}], mode='offset')
    TaskDownloader(s, item).run()
    assert s.state.job(item['id'])['slices'][0]['status'] == 'contract_blocked'
