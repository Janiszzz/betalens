"""Replay verified, real Tushare probe archives in disposable PG/CH targets.

No source data is invented and no download is issued by this script.  Every
fixed request is backed by its original sanitized probe response, including
actual empty pagination terminators.  The test fixture alone loads credentials
and creates/drops manager_test_* databases and a temporary CH writer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

from betalens_db_manager.canonical import canonical, registry_context
from betalens_db_manager.comparison import typed
from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.processors import resolve
from betalens_db_manager.response_validation import scope_roles, validate_archives
from betalens_db_manager.security import public_data
from betalens_db_manager.service import normalize_source
from betalens_db_manager.util import atomic_json, dumps, now, parse_date
from test_manager import live_manager


DESTINATIONS = {
    'stock_basic': 'stock_profile', 'fund_basic': 'fund_profile', 'index_basic': 'index_profile',
    'daily': 'market_daily_raw', 'daily_basic': 'market_daily_raw', 'fund_daily': 'market_daily_raw',
    'index_daily': 'market_daily_raw', 'sw_daily': 'market_daily_raw', 'trade_cal': 'trade_calendar',
    'adj_factor': 'corporate_action', 'fund_adj': 'corporate_action', 'fund_share': 'fund_shares_daily',
    'fund_nav': 'fund_nav_observation', 'suspend_d': 'suspension_event', 'bse_mapping': 'security_code_history',
    'index_classify': 'industry_dim', 'index_member_all': 'industry_membership_record',
    'ci_index_member': 'industry_membership_record', 'index_weight': 'index_membership',
    'income': 'fundamental_periodic', 'balancesheet': 'fundamental_periodic', 'cashflow': 'fundamental_periodic',
    'fina_indicator': 'fundamental_periodic', 'dividend': 'dividend_event', 'namechange': 'security_name_history',
    'etf_basic': 'etf_profile', 'bak_daily': 'stock_quote_supplement_daily',
}


def load_cases(paths):
    result = []
    for report in paths:
        document = json.loads(report.read_text())
        for case in document['cases']:
            case = {**case, 'probe_report': str(report)}
            if case.get('archive'):
                raw = gzip.decompress(Path(case['archive']).read_bytes())
                if hashlib.sha256(raw).hexdigest() != case['export_sha256']:
                    raise ValueError('Probe export hash mismatch: ' + case['endpoint'])
                wrapper = json.loads(raw)
                if wrapper['endpoint'] != case['endpoint'] or wrapper['parameters'] != case['parameters']:
                    raise ValueError('Probe request identity mismatch')
                case['_data'], case['_downloaded_at'] = wrapper['data'], wrapper['downloaded_at']
            result.append(case)
    return result


def normalize_case(service, case):
    endpoint = case['endpoint']
    result = {k: case.get(k) for k in ('sample', 'endpoint', 'parameters', 'archive', 'status', 'rows')}
    if '_data' not in case:
        return {**result, 'validation_status': 'source_unavailable'}
    from betalens_db_manager.catalog import dataset
    rows, rejected = normalize_source(dataset(endpoint), case['_data'], 'real-probe', 'probe', case['_downloaded_at'])
    if rejected:
        return {**result, 'validation_status': 'rejected', 'rejects': [{'source_row': r['source_row'], 'reason': r['reason']} for r in rejected]}
    table = DESTINATIONS[endpoint]
    mapping = service.mappings.get(table)
    binding = next(b for b in mapping['bindings'] if b['source_id'] == 'tushare' and b['endpoint'] == endpoint)
    spec = service.registry.spec(table)
    mapped_fields = {r['source_field'] for r in binding['fields']}
    missing = set(case['_data']['fields']) - mapped_fields
    # industry membership has complete raw detail storage, while corporate
    # action adjustment sources explicitly map all three source fields.
    verified = 0
    with registry_context(service.registry):
        for source in rows:
            partial, evidence = service.mappings.map_row(spec, source['_raw'], binding, processor=resolve(table, binding) is not None)
            converted = typed(spec, partial)
            for field in evidence:
                if field.get('builtin'):
                    continue
                if converted[field['column']] != field['converted_value']:
                    raise ValueError(endpoint + ': typed value mismatch: ' + field['source_field'])
                verified += 1
    return {**result, 'validation_status': 'no_sample' if not rows else 'typed_complete' if not missing else 'pending_destination',
            'canonical_table': table, 'normalized_rows': len(rows), 'typed_field_checks': verified, 'missing_destinations': sorted(missing)}


def groups(cases):
    def matching(sample=None, endpoints=None, report=None):
        return [c for c in cases if '_data' in c and (sample is None or c['sample'] == sample)
                and (endpoints is None or c['endpoint'] in endpoints) and (report is None or report in c['probe_report'])]
    calendar = matching('calendar')
    stocks = []
    for label, code, day in [('ordinary_stock', '000001.SZ', '2026-10-16'), ('delisted_stock', '600001.SH', '2007-10-16'), ('beijing_stock', '920001.BJ', '2026-10-16')]:
        roots = matching(label, {'stock_basic', 'daily', 'daily_basic', 'adj_factor'}, 'probe-report.json')
        if day == '2026-10-16': roots += calendar
        stocks.append(dict(label=label, tables=['stock_profile', 'security_dim', 'market_daily_raw', 'market_daily_post_adjusted'],
                           cases=roots, codes=[code], assets=['stock'], start=day, end=day))
    pagination = matching(report='probe-pagination.json')
    etf = matching('etf', {'etf_basic', 'fund_daily', 'fund_share'})
    etf += [c for c in pagination if c['endpoint'] == 'fund_basic'] + matching('etf_nav') + calendar
    etf += matching(report='probe-fund-adj-pages.json')
    index = matching('index', {'index_daily'}) + [c for c in pagination if c['endpoint'] == 'index_basic'] + calendar
    return stocks + [
        dict(label='etf', tables=['fund_profile', 'etf_profile', 'security_dim', 'market_daily_raw', 'market_daily_post_adjusted', 'fund_shares_daily', 'fund_nav_daily', 'fund_nav_observation'], cases=etf, codes=['510300.SH'], assets=['fund'], start='2026-10-16', end='2026-10-16'),
        dict(label='index', tables=['index_profile', 'security_dim', 'market_daily_raw', 'market_daily_post_adjusted'], cases=index, codes=['000300.SH'], assets=['index'], start='2026-10-16', end='2026-10-16'),
        dict(label='sw_index', tables=['market_daily_raw', 'market_daily_post_adjusted'], cases=matching('sw_index') + calendar, codes=['801010.SI'], assets=['index'], start='2026-10-16', end='2026-10-16'),
        dict(label='industry', tables=['industry_scheme_dim', 'industry_dim', 'industry_membership_record', 'security_industry_membership'], cases=matching(endpoints={'index_classify', 'index_member_all', 'ci_index_member'}), codes=['000001.SZ'], assets=['stock'], start='1990-01-01', end='2026-10-24'),
        dict(label='financial', tables=['fundamental_periodic'], cases=matching('financial'), codes=['000001.SZ'], assets=['stock'], start='2024-12-31', end='2024-12-31'),
        dict(label='dividend_detail', tables=['dividend_event', 'corporate_action'], cases=matching('dividend'), codes=['000001.SZ'], assets=['stock'], start='1990-01-01', end='2026-10-24'),
        dict(label='name_history', tables=['security_name_history', 'corporate_action'], cases=matching('name_history'), codes=['000001.SZ'], assets=['stock'], start='1990-01-01', end='2026-10-24'),
        dict(label='suspension_detail', tables=['suspension_event', 'trade_status_daily'], cases=matching('suspension_market_sample') + calendar, codes=[], assets=['stock'], start='2026-10-16', end='2026-10-16'),
        dict(label='beijing_code_history', tables=['security_code_history'], cases=matching('beijing_code_history'), codes=['920001.BJ'], assets=['stock'], start='1990-01-01', end='2026-10-24'),
        dict(label='supplement', tables=['stock_quote_supplement_daily'], cases=matching('supplementary_quote') + [c for c in pagination if c['endpoint'] == 'bak_daily'], codes=['000001.SZ'], assets=['stock'], start='2026-10-16', end='2026-10-16'),
    ]


def fixed_task(service, group, originals):
    endpoints = {c['endpoint'] for c in group['cases']}
    for name in group['tables']:
        document = service.mappings.document(name)
        value = copy.deepcopy(originals[name])
        value['bindings'] = [b for b in value['bindings'] if b['endpoint'] in endpoints]
        service.mappings.save(value, document['file_hash'])
    request = dict(tables=group['tables'], codes=group['codes'], asset_types=group['assets'], start_date=group['start'], end_date=group['end'], targets=['postgres', 'clickhouse'], mode='refresh_source')
    # Compile current mappings/contracts, then bind this archive replay to the
    # exact requests actually issued by the probe.  No unrequested API filters
    # or fabricated dependency responses are attached.
    plan = service.compile(request)
    plan['requests'] = []
    plan['archive_acceptance'] = {'sample_only': True, 'request_basis': 'exact_original_probe_requests', 'no_source_download': True}
    for case in group['cases']:
        endpoint = case['endpoint']; contract = service.sources.endpoint('tushare', endpoint)
        r = dict(table=group['tables'][0], source_id='tushare', endpoint=endpoint, parameters=case['parameters'], strategy='dependency' if endpoint == 'trade_cal' else contract['strategy'], priority=0)
        r['validation_scope'] = scope_roles(plan, r, contract)
        plan['source_contracts']['tushare:' + endpoint] = contract
        plan['requests'].append(r)
    plan['requests'] = list({TaskDownloader.key(r): r for r in plan['requests']}.values())
    item = service.state.create_job(plan, service.settings.fingerprint)
    service._record_mapping_plan(item)
    downloader = TaskDownloader(service, item)
    lookup = {TaskDownloader.key(r): case for r, case in zip([dict(source_id='tushare', endpoint=c['endpoint'], parameters=c['parameters']) for c in group['cases']], group['cases'])}
    for request in plan['requests']:
        prepared, contract, _, page = downloader.prepared(request)
        if prepared['parameters'] != request['parameters']:
            raise ValueError('Original probe lacks explicit current paging parameters: ' + request['endpoint'])
        case = lookup[TaskDownloader.key(request)]
        data = case['_data']; raw = dumps({'code': 0, 'data': data}).encode()
        key = downloader.key(request)
        target = service.settings.state_dir / 'archive' / item['id'] / (key + '.json.gz')
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(target, 'wb') as handle: handle.write(raw)
        atomic_json(target.with_suffix('.meta.json'), {'fetched_at': case['_downloaded_at'], 'source_name': 'tushare', 'original_probe_archive': case['archive'], 'original_export_sha256': case['export_sha256']})
        children = [downloader.following(request, page, len(data['items']))] if page and data['items'] else []
        service.state.slice(item['id'], key, request['endpoint'], request, status='paged' if children else 'downloaded', archive=str(target), sha=hashlib.sha256(raw).hexdigest(), rows=len(data['items']), result={'children': children, 'real_probe_archive': case['archive']})
    service.state.update_job(item['id'], status='downloaded')
    errors = validate_archives(service, service.state.job(item['id']))
    if errors: raise ValueError('Real archive graph rejected: ' + dumps(errors))
    return service.state.job(item['id'])


def run_group(service, group, originals):
    result = {'case': group['label'], 'tables': group['tables'], 'source_archives': [c['archive'] for c in group['cases']]}
    if not group['cases']: return {**result, 'status': 'no_sample'}
    for name in group['tables']:
        with registry_context(service.registry):
            applied = service.schema.apply(service.schema.plan(name)['token'])
        if applied['status'] != 'completed': raise ValueError('Isolated schema failed: ' + name)
    item = fixed_task(service, group, originals)
    result['task_id'] = item['id']
    preview = service.preview(item['id'])
    result['rejected'], result['reject_sample'] = preview['rejected'], preview['reject_sample']
    result['preview_tables'] = [{k: r[k] for k in ('target', 'table', 'counts')} for r in preview['tables']]
    if preview['blocking'] or preview['rejected']: return {**result, 'status': 'blocked'}
    for name in group['tables']:
        with registry_context(service.registry):
            if service.pg.query(name)['rows'] or service.ch.query(name)['rows']:
                raise ValueError('Unexpected pre-confirmation business rows')
    result['no_business_writes_before_confirmation'] = True
    committed = service.commit(item['id'], preview['token'])
    result['status'], result['commit_error'] = committed['status'], committed.get('error')
    result['verification'] = service.state.ledger_get('verification', item['id'])
    proofs = service.state.ledger_rows('field_verification', item['id'])
    result['field_proof_counts'] = {target: dict(Counter(p['status'] for p in proofs if p['target'] == target)) for target in ('postgres', 'clickhouse')}
    result['field_proof_rows'] = len(proofs)
    if committed['status'] not in {'completed', 'audit_pending'}: return result
    if any(p['status'] == 'mismatch' for p in proofs): raise ValueError('Field readback mismatch')
    if group['label'] == 'dividend_detail':
        expected = {}
        for case in group['cases']:
            data = case['_data']
            for values in data['items']:
                source = dict(zip(data['fields'], values))
                if source.get('div_proc') not in {'实施', '已实施'} or not source.get('ex_date'): continue
                key = source['ts_code'], parse_date(source['ex_date'])
                pair = tuple(None if source.get(f) in (None, '') else str(source[f]) for f in ('cash_div', 'stk_bo_rate'))
                expected.setdefault(key, set()).add(pair)
        summary_checks = {}
        from decimal import Decimal
        for target_name, database in (('postgres', service.pg), ('clickhouse', service.ch)):
            with registry_context(service.registry): rows = database.query('corporate_action', limit=1000)['rows']
            for row in rows:
                key = row['security_code'], row['event_date']
                if row['effective_date'] != row['event_date'] or key not in expected: raise ValueError('Dividend event did not use actual ex_date')
                actual = row['cash_dividend'], row['share_dividend']
                choices = {tuple(None if v is None else Decimal(v) for v in pair) for pair in expected[key]}
                if actual not in choices: raise ValueError('Dividend cash_div/stk_bo_rate semantics mismatch')
            summary_checks[target_name] = len(rows)
        result['dividend_summary_checks'] = summary_checks
        result['dividend_summary_source_fields'] = ['ex_date', 'cash_div', 'stk_bo_rate']
    rollback = service.rollback(item['id'])
    result['rollback'] = rollback
    for name in group['tables']:
        with registry_context(service.registry):
            if service.pg.query(name)['rows'] or service.ch.query(name)['rows']:
                raise ValueError('Rows remain after isolated task rollback')
    result['rollback_verified_empty'] = True
    if not all(r['status'] == 'completed' for r in rollback['results']): raise ValueError('Rollback did not complete')
    result['status'] = 'passed'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, default=[Path('/home/janiszzz/betalens-infra/mapping-audit-20261024') / name for name in ('probe-report.json', 'probe-followup.json', 'probe-pagination.json', 'probe-fund-adj-pages.json')])
    parser.add_argument('--output', type=Path, default=Path('/tmp/betalens-real-mapping-acceptance.json'))
    parser.add_argument('--only', nargs='+', help='Limit isolated replay groups; raw field checks still cover all reports')
    args = parser.parse_args()
    os.environ['BETALENS_INFRA_INTEGRATION'] = '1'
    root = Path('/home/janiszzz/.cache/betalens-real-mapping-acceptance'); root.mkdir(parents=True, exist_ok=True, mode=0o700)
    report = {'started_at': now(), 'source_reports': list(map(str, args.reports)), 'production_business_changes': False, 'no_background_worker': True, 'sample_only': True, 'normalization': [], 'cases': []}
    with tempfile.TemporaryDirectory(dir=root, prefix='run-') as directory:
        fixture = live_manager.__wrapped__(Path(directory))
        service = next(fixture)
        assert service.settings.pg_database.startswith('manager_test_') and service.settings.ch_database.startswith('manager_test_')
        report['isolated_targets'] = {'postgres': service.settings.pg_database, 'clickhouse': service.settings.ch_database}
        try:
            cases = load_cases(args.reports)
            for case in cases:
                try: checked = normalize_case(service, case)
                except Exception as exc: checked = {'sample': case['sample'], 'endpoint': case['endpoint'], 'validation_status': 'failed', 'error': service.safe_error(exc)}
                report['normalization'].append(checked)
            originals = service.mappings.snapshots()
            selected_groups = [g for g in groups(cases) if not args.only or g['label'] in args.only]
            if args.only and set(args.only) - {g['label'] for g in selected_groups}: raise ValueError('Unknown acceptance group')
            report['compiled_plan_execution'] = False
            report['replay_scope'] = 'Exact original probe requests; controlled archive replay of a sample subset, not execution of the full compiler-generated plan.'
            report['source_response_count'] = len(cases)
            report['source_report_response_counts'] = {str(path): sum(c['probe_report'] == str(path) for c in cases) for path in args.reports}
            for group in selected_groups:
                print(dumps({'case': group['label'], 'status': 'started'}), flush=True)
                try: result = run_group(service, group, originals)
                except Exception as exc: result = {'case': group['label'], 'status': 'failed', 'error': service.safe_error(exc)}
                report['cases'].append(result)
                atomic_json(args.output, public_data(report, service.settings))
                print(dumps({k: result[k] for k in ('case', 'status', 'task_id', 'error') if k in result}), flush=True)
            report['counts'] = dict(Counter(c['status'] for c in report['cases']))
            report['normalization_counts'] = dict(Counter(c['validation_status'] for c in report['normalization']))
            report['finished_at'] = now()
            report['notes'] = ['All inputs are actual sanitized source responses; no empty responses are invented.', 'Task scopes freeze the exact original probe requests; samples do not prove market-wide completeness.', 'Uncertain source values, permissions, missing pages and rejected rows are reported, never fabricated.', 'Disposable databases and temporary writer user are removed by the integration fixture.']
            atomic_json(args.output, public_data(report, service.settings)); args.output.chmod(0o600)
        finally:
            try: next(fixture)
            except StopIteration:
                report['isolated_fixture_cleanup_completed'] = True
                atomic_json(args.output, public_data(report, service.settings))
    print(dumps({'report': str(args.output), 'counts': report.get('counts'), 'normalization_counts': report.get('normalization_counts')}), flush=True)
    if any(c['status'] in {'failed', 'blocked'} for c in report['cases']) or any(c['validation_status'] in {'failed', 'rejected', 'pending_destination', 'source_unavailable'} for c in report['normalization']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
