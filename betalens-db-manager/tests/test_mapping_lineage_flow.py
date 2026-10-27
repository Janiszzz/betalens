"""Mapping and column evidence acceptance using only disposable databases."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest

from betalens_db_manager.canonical import canonical, registry_context
from betalens_db_manager.comparison import content_hash
from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.util import atomic_json, dumps, now
from test_manager import live_manager, manager


def cache_requests(service, job, responses):
    """Persist exact named-field archives for every fixed planned request."""
    downloader = TaskDownloader(service, job)
    queue = list(job['plan']['requests'])
    for original in queue:
        assert len(queue) < 100, 'test archive callback must return an empty pagination terminator'
        request, contract, _, page = downloader.prepared(original)
        endpoint, params = request['endpoint'], request['parameters']
        source_rows = responses(endpoint, params)
        fields = [f['name'] for f in contract['fields']]
        data = {'fields': fields, 'items': [[row.get(f) for f in fields] for row in source_rows]}
        key = downloader.key(request)
        path = service.settings.state_dir / 'archive' / job['id'] / (key + '.json.gz')
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = dumps({'code': 0, 'data': data}).encode()
        with gzip.open(path, 'wb') as handle:
            handle.write(raw)
        atomic_json(path.with_suffix('.meta.json'), {'fetched_at': '2026-10-24T00:00:00Z', 'source_name': 'tushare'})
        children = [downloader.following(request, page, len(source_rows))] if page and source_rows else []
        queue.extend(children)
        service.state.slice(job['id'], key, endpoint, request, status='paged' if children else 'downloaded', archive=str(path),
                            sha=hashlib.sha256(raw).hexdigest(), rows=len(source_rows), result={'children': children})
    service.state.update_job(job['id'], status='downloaded')
    return service.state.job(job['id'])


def quotes(endpoint, params):
    if params.get('offset'):
        return []
    code = params.get('ts_code', '000001.SZ')
    if endpoint == 'stock_basic':
        return [dict(ts_code=code, name='普通股票', exchange='SZSE', list_date='20000101', list_status='L')] if params.get('list_status') == 'L' else []
    if endpoint == 'index_basic':
        return [dict(ts_code=code, name='境内指数', market='SSE', list_date='20000101')]
    if endpoint in {'daily', 'index_daily'}:
        return [dict(ts_code=code, trade_date='20261016', open='11', high='13', low='10', close='12',
                     pre_close='11', change='1', pct_chg='9.09', vol='2', amount='2.4')]
    if endpoint == 'daily_basic':
        return [dict(ts_code=code, trade_date='20261016', close='12', pe='10', pb='2',
                     total_share='1.5', float_share='1', free_share='0.5', total_mv='3', circ_mv='2')]
    if endpoint == 'adj_factor':
        return [dict(ts_code=code, trade_date='20261016', adj_factor='2')]
    if endpoint == 'trade_cal':
        return [dict(exchange=params['exchange'], cal_date='20261016', is_open='1', pretrade_date='20261015')]
    return []


def test_post_adjusted_single_table_has_same_inputs_and_values_as_combined_update(tmp_path):
    service = manager(tmp_path)
    request = dict(start_date='2026-10-16', end_date='2026-10-16', codes=['000001.SZ'], asset_types=['stock'], targets=['postgres'])
    single = service.plan({**request, 'tables': ['market_daily_post_adjusted']})
    endpoints = {r['endpoint'] for r in single['plan']['requests']}
    assert {'daily', 'daily_basic', 'adj_factor', 'stock_basic', 'trade_cal'} <= endpoints
    assert single['plan']['request']['tables'] == ['market_daily_post_adjusted']
    one, rejected = service._build_canonical(cache_requests(service, single, quotes))
    assert not rejected and set(one) == {'market_daily_post_adjusted'}
    both = service.plan({**request, 'tables': ['market_daily_raw', 'market_daily_post_adjusted']})
    combined, rejected = service._build_canonical(cache_requests(service, both, quotes))
    assert not rejected
    selected = one['market_daily_post_adjusted'][0]
    assert content_hash(canonical('market_daily_post_adjusted'), selected) == content_hash(canonical('market_daily_post_adjusted'), combined['market_daily_post_adjusted'][0])
    assert selected['volume'] == Decimal('200') and selected['amount'] == Decimal('2400')
    assert selected['total_shares'] == Decimal('15000') and selected['pe'] == Decimal('10')
    assert selected['raw_close'] == Decimal('12') and selected['close'] == selected['average_price'] == Decimal('24')
    assert selected['security_name'] == '普通股票'


def test_index_average_is_unavailable_after_archive_normalization(tmp_path):
    service = manager(tmp_path)
    job = service.plan(dict(tables=['market_daily_post_adjusted'], start_date='2026-10-16', end_date='2026-10-16',
                            codes=['000300.SH'], asset_types=['index'], targets=['postgres']))
    built, rejected = service._build_canonical(cache_requests(service, job, quotes))
    assert not rejected
    row = built['market_daily_post_adjusted'][0]
    assert row['adjustment_method'] == 'native_index_value'
    assert row['close'] == Decimal('12') and row['average_price'] is None


@pytest.fixture
def lineage_api():
    payloads = {'primary': [None, None, '5000', None], 'backup': [None, None, None, None]}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            endpoint = self.path.split('?', 1)[0].strip('/')
            numeric = payloads[endpoint]
            # Reorder response fields deliberately; no positional contracts.
            value = dict(fields=['price_milli', 'day', 'preserved_milli', 'ticker', 'null_milli', 'fallback_milli'],
                         items=[[numeric[0], '2026-10-16', numeric[2], 'a', numeric[1], numeric[3]]])
            data = json.dumps(value).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(data)
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        yield 'http://127.0.0.1:' + str(server.server_port), payloads
    finally:
        server.shutdown(); server.server_close()


def configure_lineage(service, url):
    service.registry.save(dict(name='lineage_prices', zh_name='字段证据行情', description='精确来源证据', grain='代码和日期一行',
        keys=['code', 'business_date'], time_field='business_date', temporal_rule='calendar_day', fields=[
            {'name': 'code', 'type': 'text'}, {'name': 'business_date', 'type': 'date'},
            *[{'name': c, 'type': 'numeric'} for c in ('price', 'null_metric', 'preserved_metric', 'fallback_metric')]]))
    fields = [{'name': 'ticker', 'type': 'text'}, {'name': 'day', 'type': 'date'},
              *[{'name': c, 'type': 'numeric'} for c in ('price_milli', 'null_milli', 'preserved_milli', 'fallback_milli')]]
    service.sources.save(dict(source_id='lineage_vendor', provider='http', base_url=url, endpoints={
        endpoint: dict(path=endpoint, strategy='range', keys=['ticker', 'day'], date_field='day', fields=fields)
        for endpoint in ('primary', 'backup')}))
    rules = [{'source_field': 'ticker', 'column': 'code'}, {'source_field': 'day', 'column': 'business_date'},
             *[{'source_field': source, 'column': dest, 'multiplier': '0.001'} for source, dest in (
                 ('price_milli', 'price'), ('null_milli', 'null_metric'), ('preserved_milli', 'preserved_metric'), ('fallback_milli', 'fallback_metric'))]]
    service.mappings.save(dict(format_version=1, table='lineage_prices', bindings=[
        dict(source_id='lineage_vendor', endpoint=endpoint, priority=priority, fields=rules)
        for priority, endpoint in enumerate(('primary', 'backup'))]), None)
    with registry_context(service.registry):
        result = service.schema.apply(service.schema.plan('lineage_prices')['token'])
    assert result['status'] == 'completed', result


def query(service, target, name):
    with registry_context(service.registry):
        return (service.pg if target == 'postgres' else service.ch).query(name)['rows']


def test_dual_target_column_proof_distinguishes_zero_null_fallback_and_retained_values(live_manager, lineage_api):
    service = live_manager
    url, payloads = lineage_api
    configure_lineage(service, url)
    request = dict(tables=['lineage_prices'], start_date='2026-10-16', end_date='2026-10-16')
    seed = service.plan(request); service.download(seed['id']); preview = service.preview(seed['id'])
    assert not preview['blocking'] and not preview['rejected'], preview
    assert not query(service, 'postgres', 'lineage_prices') and not query(service, 'clickhouse', 'lineage_prices')
    result = service.commit(seed['id'], preview['token']); assert result['status'] == 'completed', result.get('error')
    payloads['primary'] = ['0', None, '99000', None]
    payloads['backup'] = ['20000', None, '88000', '7000']
    job = service.plan(request); service.download(job['id']); preview = service.preview(job['id'])
    assert not preview['blocking'] and not preview['rejected'], preview
    for target in ('postgres', 'clickhouse'):
        before = query(service, target, 'lineage_prices')[0]
        assert before['price'] is None and before['preserved_metric'] == Decimal('5')
    traces = service.state.ledger_get('task_column_lineage', job['id'] + ':lineage_prices')['rows']
    trace = next(iter(traces.values()))
    assert trace['price']['selected'] and trace['price']['origins'] == ['lineage_vendor:primary:price_milli']
    assert trace['fallback_metric']['origins'] == ['lineage_vendor:backup:fallback_milli']
    result = service.commit(job['id'], preview['token']); assert result['status'] == 'completed', result.get('error')
    proofs = service.state.ledger_rows('field_verification', job['id'])
    for target in ('postgres', 'clickhouse'):
        current = query(service, target, 'lineage_prices')[0]
        assert current['price'] == Decimal('0') and current['null_metric'] is None
        assert current['fallback_metric'] == Decimal('7') and current['preserved_metric'] == Decimal('5')
        columns = {p['column']: p for p in proofs if p['target'] == target}
        assert columns['price']['status'] == 'verified_stored' and columns['price']['verified_count'] == 1
        assert columns['price']['verified_origins'] == {'lineage_vendor:primary:price_milli': 1}
        assert columns['fallback_metric']['verified_origins'] == {'lineage_vendor:backup:fallback_milli': 1}
        assert columns['null_metric']['status'] != 'verified_stored'
        assert columns['preserved_metric']['status'] != 'verified_stored' and columns['preserved_metric']['preserved_count'] == 1
    fields = service.field_audit(job['id'])
    selected = next(f for f in fields if f['endpoint'] == 'primary' and f['source_field'] == 'price_milli')
    unselected = next(f for f in fields if f['endpoint'] == 'backup' and f['source_field'] == 'price_milli')
    source_null = next(f for f in fields if f['endpoint'] == 'primary' and f['source_field'] == 'null_milli')
    assert selected['status'] == 'verified_stored' and {p['target'] for p in selected['evidence']} == {'postgres', 'clickhouse'}
    assert unselected['status'] != 'verified_stored' and source_null['status'] != 'verified_stored'
    assert any(log['event'] == 'http_response' for log in service.state.logs(job['id']))
    rolled = service.rollback(job['id']); assert all(r['status'] == 'completed' for r in rolled['results']), rolled
    for target in ('postgres', 'clickhouse'):
        restored = query(service, target, 'lineage_prices')[0]
        assert restored['price'] is None and restored['preserved_metric'] == Decimal('5')
    assert all(r['status'] == 'completed' for r in service.rollback(seed['id'])['results'])
    assert not query(service, 'postgres', 'lineage_prices') and not query(service, 'clickhouse', 'lineage_prices')


def supplement_archive(service, targets=None):
    sources = {}
    for endpoint in ('etf_basic', 'bak_daily'):
        contract = service.sources.endpoint('tushare', endpoint)
        source = {f['name']: '1.2500' if f['type'] == 'numeric' else '20200101' if f['type'] == 'date' else '字段_' + f['name'] for f in contract['fields']}
        source['ts_code'] = '510300.SH' if endpoint == 'etf_basic' else '000001.SZ'
        if endpoint == 'etf_basic': source['index_code'] = '000300.SH'
        else: source.update(trade_date='20261016', total_share='15.125', float_share='10.5')
        sources[endpoint] = source
    def response(endpoint, params):
        raw = sources.get(endpoint)
        return [raw] if raw and not params.get('offset') and params.get('ts_code', raw['ts_code']) == raw['ts_code'] else []
    job = service.plan(dict(tables=['etf_profile', 'stock_quote_supplement_daily'], codes=['510300.SH', '000001.SZ'],
                            start_date='2026-10-16', end_date='2026-10-16', targets=targets or ['postgres']))
    return cache_requests(service, job, response), sources


def test_etf_and_supplement_direct_contract_preserves_all_fields_and_native_units(tmp_path):
    service = manager(tmp_path)
    job, sources = supplement_archive(service)
    built, rejected = service._build_canonical(job)
    assert not rejected and set(built) == {'etf_profile', 'stock_quote_supplement_daily'}
    for endpoint, table in (('etf_basic', 'etf_profile'), ('bak_daily', 'stock_quote_supplement_daily')):
        actual = built[table][0]
        contract = service.sources.endpoint('tushare', endpoint)
        for field in contract['fields']:
            column = 'security_code' if field['name'] == 'ts_code' else field['name']
            value = sources[endpoint][field['name']]
            expected = Decimal(value) if field['type'] == 'numeric' else date.fromisoformat(value[:4] + '-' + value[4:6] + '-' + value[6:8]) if field['type'] == 'date' else value
            assert actual[column] == expected, (endpoint, column)
    assert built['stock_quote_supplement_daily'][0]['total_share'] == Decimal('15.125')
    assert 'market_daily_raw' not in built  # Unit-uncertain supplement never auto-projects.


def test_new_direct_tables_store_typed_values_in_both_targets_and_rollback(live_manager):
    service = live_manager
    tables = ['etf_profile', 'stock_quote_supplement_daily']
    with registry_context(service.registry):
        for name in tables:
            result = service.schema.apply(service.schema.plan(name)['token'])
            assert result['status'] == 'completed', result
    job, _ = supplement_archive(service, targets=['postgres', 'clickhouse'])
    preview = service.preview(job['id'])
    assert not preview['blocking'] and not preview['rejected'], preview
    for target in ('postgres', 'clickhouse'):
        assert all(not query(service, target, name) for name in tables)
    result = service.commit(job['id'], preview['token'])
    assert result['status'] == 'completed', result.get('error')
    for target in ('postgres', 'clickhouse'):
        etf = query(service, target, 'etf_profile')[0]
        supplement = query(service, target, 'stock_quote_supplement_daily')[0]
        assert etf['security_code'] == '510300.SH' and etf['index_code'] == '000300.SH'
        assert etf['mgt_fee'] == Decimal('1.25') and etf['setup_date'] == date(2020, 1, 1)
        assert supplement['total_share'] == Decimal('15.125') and supplement['float_share'] == Decimal('10.5')
        assert supplement['avg_price'] == Decimal('1.25') and supplement['trade_date'] == date(2026, 10, 16)
        assert not service.schema.inspect('market_daily_raw', target)['exists']
    proofs = service.state.ledger_rows('field_verification', job['id'])
    assert {p['target'] for p in proofs if p['table'] == 'stock_quote_supplement_daily' and p['column'] == 'total_share' and p['status'] == 'verified_stored'} == {'postgres', 'clickhouse'}
    rolled = service.rollback(job['id'])
    assert all(r['status'] == 'completed' for r in rolled['results']), rolled
    for target in ('postgres', 'clickhouse'):
        assert all(not query(service, target, name) for name in tables)
