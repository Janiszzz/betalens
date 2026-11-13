"""Compare deterministic legacy/compact coverage; never connects to a database."""
import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'betalens-db-manager' / 'src'))

from betalens_db_manager.reference_domains import build_scopes as daily_scopes, check_integrity
from betalens_db_manager.suspension_ranges import build_scopes
from betalens_db_manager.util import dumps


def benchmark(securities=50, days=365):
    first = date(2024, 1, 1)
    dates = [first + timedelta(days=i) for i in range(days)]
    codes = [f'{i:06}.SZ' for i in range(1, securities + 1)]
    request = {'modules': ['suspension'], 'security_codes': codes,
               'start_date': str(first), 'end_date': str(dates[-1]), 'check_times': ['10:00']}
    def bundle(endpoint, rows, **parameters):
        return {'endpoint': endpoint, 'parameters': parameters, 'rows': rows, 'complete': True,
                'downloaded_at': '2026-11-10T00:00:00Z', 'archive_sha': 'deterministic-proof'}
    bundles = [bundle('suspend_d', [
        {'ts_code': code, 'trade_date': day.strftime('%Y%m%d'), 'suspend_type': 'S', 'suspend_timing': None}
        for i, code in enumerate(codes) for n, day in enumerate(dates) if i % 7 == 0 and n % 41 == 0 and day.weekday() < 5],
        start_date=first.strftime('%Y%m%d'), end_date=dates[-1].strftime('%Y%m%d')),
        bundle('stock_basic', [{'ts_code': code, 'list_date': '20000101', 'list_status': 'L'} for code in codes]),
        bundle('trade_cal', [{'exchange': 'SZSE', 'cal_date': day.strftime('%Y%m%d'),
                              'is_open': int(day.weekday() < 5),
                              'pretrade_date': (day - timedelta(days=1)).strftime('%Y%m%d')} for day in dates],
               exchange='SZSE', start_date=first.strftime('%Y%m%d'), end_date=dates[-1].strftime('%Y%m%d'))]
    started = time.perf_counter();legacy = daily_scopes(request, bundles)
    legacy_seconds = time.perf_counter() - started
    started = time.perf_counter();compact, pool = build_scopes(request, bundles)
    compact_seconds = time.perf_counter() - started
    def snapshot(scopes, pool=None):
        revision = {'revision_id': 'r', 'ingestion_seq': 1}
        if pool is not None:revision['evidence_pool'] = pool
        return {'revisions': [revision], 'scopes': [dict(s, revision_id='r') for s in scopes],
                'events': [{'revision_id': 'r', 'event_type': 'publish'}]}
    snapshots = {'legacy': snapshot(legacy), 'compact': snapshot(compact, pool)}
    report = {'kind': 'synthetic_python_benchmark', 'securities': securities, 'calendar_days': days,
              'database_io_included': False, 'production_modified': False,
              'legacy': {'scopes': len(legacy), 'json_bytes': len(dumps(legacy).encode()), 'build_seconds': legacy_seconds},
              'compact': {'scopes': len(compact), 'json_bytes': len(dumps({'scopes': compact, 'evidence_pool': pool}).encode()), 'build_seconds': compact_seconds},
              'workloads': {}}
    cases = {'single_security_day': {**request, 'security_codes': codes[:1], 'start_date': str(first), 'end_date': str(first)},
             'market_day': {**request, 'start_date': str(first), 'end_date': str(first)},
             'history_matrix': request}
    for name, query in cases.items():
        values = {}
        timing = {}
        for mode, state in snapshots.items():
            started = time.perf_counter();result = check_integrity(query, state)
            timing[mode + '_seconds'] = time.perf_counter() - started
            values[mode] = [(r['security_code'], r['trade_date'], r['reason'], r.get('is_suspended')) for r in result['rows']]
        assert values['legacy'] == values['compact'], name
        report['workloads'][name] = {**timing, 'answers_equal': True, 'cells': len(values['legacy'])}
    report['scope_reduction_ratio'] = len(legacy) / len(compact)
    report['json_reduction_ratio'] = report['legacy']['json_bytes'] / report['compact']['json_bytes']
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--securities', type=int, default=50)
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.securities <= 500 or not 1 <= args.days <= 3650:
        parser.error('securities must be 1..500; days must be 1..3650')
    result = benchmark(args.securities, args.days)
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:args.output.write_text(content)
    print(content)
