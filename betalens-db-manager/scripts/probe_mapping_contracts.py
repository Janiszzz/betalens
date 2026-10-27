"""Small full-field Tushare probes; never connect to PostgreSQL or ClickHouse.

Example::

    .venv/bin/python scripts/probe_mapping_contracts.py \
        --env /private/credentials.env --output /private/probe/report.json \
        --trade-date 20261023 --only etf_basic bak_daily

The env file is read only. Runtime state is temporary; only a sanitized report
and sanitized source archives remain. A short response is sample evidence,
not a claim of market-wide or historical completeness.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

from betalens_db_manager.catalog import load_catalog
from betalens_db_manager.config import Settings
from betalens_db_manager.retrieval import request_rate, tushare_policy
from betalens_db_manager.security import public_data, safe_error
from betalens_db_manager.source import TushareSource
from betalens_db_manager.state import State
from betalens_db_manager.util import atomic_json, dumps, now, parse_date


def sample_cases(trade_date: str, period: str) -> list[dict]:
    cases = []

    def add(label, endpoint, **parameters):
        cases.append({'sample': label, 'endpoint': endpoint, 'parameters': parameters})

    for label, code, day, status in (
        ('ordinary_stock', '000001.SZ', trade_date, 'L'),
        ('delisted_stock', '600001.SH', '20071016', 'D'),
        ('beijing_stock', '920001.BJ', trade_date, 'L'),
    ):
        add(label, 'stock_basic', ts_code=code, list_status=status)
        for endpoint in ('daily', 'daily_basic', 'adj_factor', 'suspend_d'):
            add(label, endpoint, ts_code=code, trade_date=day)
    add('calendar', 'trade_cal', exchange='SSE', start_date=trade_date, end_date=trade_date)
    add('beijing_code_history', 'bse_mapping', n_code='920001.BJ')
    add('etf', 'fund_basic', ts_code='510300.SH', market='E')
    add('etf', 'etf_basic', ts_code='510300.SH')
    for endpoint in ('fund_daily', 'fund_adj', 'fund_share'):
        add('etf', endpoint, ts_code='510300.SH', trade_date=trade_date)
    add('etf_nav', 'fund_nav', ts_code='510300.SH', start_date=trade_date, end_date=trade_date)
    add('index', 'index_basic', ts_code='000300.SH')
    add('index', 'index_daily', ts_code='000300.SH', start_date=trade_date, end_date=trade_date)
    add('index_membership', 'index_weight', index_code='000300.SH', start_date=trade_date[:6]+'01', end_date=trade_date)
    add('suspension_market_sample','suspend_d',trade_date=trade_date)
    add('sw_classification', 'index_classify', src='SW2021', level='L1')
    add('sw_membership_current', 'index_member_all', ts_code='000001.SZ', is_new='Y')
    add('sw_membership_history', 'index_member_all', ts_code='000001.SZ', is_new='N')
    add('sw_index', 'sw_daily', ts_code='801010.SI', start_date=trade_date, end_date=trade_date)
    add('citic_membership_current', 'ci_index_member', ts_code='000001.SZ', is_new='Y')
    add('citic_membership_history', 'ci_index_member', ts_code='000001.SZ', is_new='N')
    for endpoint in ('income', 'balancesheet', 'cashflow', 'fina_indicator'):
        add('financial', endpoint, ts_code='000001.SZ', period=period)
    add('dividend', 'dividend', ts_code='000001.SZ')
    add('name_history', 'namechange', ts_code='000001.SZ')
    add('supplementary_quote', 'bak_daily', ts_code='000001.SZ', trade_date=trade_date, offset=0, limit=7000)
    return cases


def probe_case(source, settings, case, directory: Path, sequence: int):
    contract = load_catalog()[case['endpoint']]
    fields = [f['name'] for f in contract.fields]
    result = {**case, 'document': contract.document, 'contract_hash': contract.hash,
              'requested_fields': fields, 'started_at': now(), 'sample_only': True}
    wire = directory / f'{sequence:03d}-{case["sample"]}-{case["endpoint"]}.json.gz'
    raw_path = settings.state_dir / 'raw' / wire.name
    before_logs = len(source.state.logs(source.active_job, limit=10000))
    try:
        response, original_hash = source.fetch(case['endpoint'], case['parameters'], fields, raw_path)
        safe_response = public_data(response, settings)
        archive = {'source_id': 'tushare', 'endpoint': case['endpoint'],
                   'parameters': case['parameters'], 'downloaded_at': now(),
                   'data': safe_response, 'redacted': safe_response != response}
        encoded = dumps(archive).encode()
        # The fetch used a temporary path; only this sanitized export survives.
        wire.write_bytes(gzip.compress(encoded, mtime=0))
        wire.chmod(0o600)
        actual = response.get('fields', [])
        rows = response.get('items', [])
        malformed = (not isinstance(actual, list) or not all(isinstance(f, str) for f in actual)
                     or not isinstance(rows, list))
        if malformed:
            result.update(status='contract_blocked', reason='响应字段或记录结构无效')
        else:
            missing, unexpected = sorted(set(fields)-set(actual)), sorted(set(actual)-set(fields))
            duplicate = len(actual) != len(set(actual))
            invalid_rows = sum(not isinstance(row, list) or len(row) != len(actual) for row in rows)
            cap = tushare_policy(case['endpoint'])['row_limit']
            status = ('contract_blocked' if missing or unexpected or duplicate or invalid_rows else
                      'sample_limit_reached' if len(rows) >= cap else
                      'no_sample' if not rows else 'sample_verified')
            result.update(status=status, returned_fields=actual, missing_fields=missing,
                          unexpected_fields=unexpected, duplicate_fields=duplicate,
                          malformed_rows=invalid_rows, rows=len(rows),
                          declared_row_limit=cap, source_limit_reached=len(rows) >= cap,
                          fields=[{'source_field': field['name'], 'default_display': field.get('default_display'),
                                   'returned': field['name'] in actual, 'source_type': field.get('source_type'),
                                   'logical_type': field['type'], 'source_unit': field.get('source_unit', ''),
                                   'stored_unit': field.get('unit', ''), 'multiplier': field.get('multiplier', '1'),
                                   'non_null_samples': sum(isinstance(row, list) and len(row) == len(actual)
                                       and row[actual.index(field['name'])] is not None for row in rows)
                                       if field['name'] in actual else 0}
                                  for field in contract.fields],
                          sample_rows=safe_response.get('items', [])[:3])
        result.update(source_response_sha256=original_hash, export_sha256=hashlib.sha256(encoded).hexdigest(),
                      archive=str(wire), finished_at=now())
    except Exception as exc:
        message = safe_error(exc, settings)
        permission = any(word in message.lower() for word in ('权限', '积分', 'permission', '无权'))
        result.update(status='permission_denied' if permission else 'failed', error=message, finished_at=now())
        wire.unlink(missing_ok=True)
    finally:
        # The metadata has no raw response, but remains transient with rate state.
        raw_path.unlink(missing_ok=True)
        raw_path.with_suffix('.meta.json').unlink(missing_ok=True)
        raw_path.with_suffix('.tmp').unlink(missing_ok=True)
    result['http_logs'] = public_data(source.state.logs(source.active_job, limit=10000)[before_logs:], settings)
    return public_data(result, settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='sanitized JSON report path')
    parser.add_argument('--trade-date', default='20261016', help='YYYYMMDD sample trading day')
    parser.add_argument('--period', default='20241231', help='YYYYMMDD financial period')
    parser.add_argument('--rpm', type=request_rate, help='requests per minute; defaults to private env setting')
    parser.add_argument('--only', nargs='+', choices=sorted(load_catalog()), help='probe only these endpoints')
    args = parser.parse_args()
    if not args.env.is_file():
        parser.error('--env must identify an existing private environment file')
    day, period = parse_date(args.trade_date).strftime('%Y%m%d'), parse_date(args.period).strftime('%Y%m%d')
    output = args.output.expanduser().resolve()
    if output == args.env.expanduser().resolve():
        parser.error('--output must not overwrite the private environment file')
    if output.exists():
        parser.error('--output already exists; choose a new report path')
    archive_dir = output.parent / (output.stem+'-archives')
    if archive_dir.exists():
        parser.error('archive directory already exists; choose a new report path')
    os.umask(0o077)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive_dir.mkdir(mode=0o700)
    loaded = Settings.from_env(args.env)
    if not loaded.token:
        parser.error('TUSHARE_TOKEN is not configured in the selected environment')
    with tempfile.TemporaryDirectory(prefix='betalens-contract-probe-') as temporary:
        settings = replace(loaded, state_dir=Path(temporary),
                           requests_per_minute=args.rpm or loaded.requests_per_minute)
        source = TushareSource(settings, State(settings.state_dir))
        task = source.state.create_job({'kind': 'readonly_contract_probe'}, settings.fingerprint)
        source.active_job = task['id']
        cases = [case for case in sample_cases(day, period) if not args.only or case['endpoint'] in args.only]
        report = {'started_at': now(), 'task_id': task['id'], 'sample_only': True,
                  'database_connections_used': False, 'requests_per_minute': settings.requests_per_minute,
                  'trade_date': day, 'period': period, 'cases': []}
        for sequence, case in enumerate(cases, 1):
            result = probe_case(source, settings, case, archive_dir, sequence)
            report['cases'].append(result)
            atomic_json(output, public_data(report, settings))
            print(dumps({key: result[key] for key in ('sample', 'endpoint', 'status', 'rows') if key in result}), flush=True)
        report['finished_at'] = now()
        report['counts'] = {status: sum(case['status'] == status for case in report['cases'])
                            for status in sorted({case['status'] for case in report['cases']})}
        report['notes'] = ['Full field lists are explicit, including non-default output fields.',
                           'No empty, denied, truncated or failed sample counts as stored-data verification.',
                           'No business database connection, schema change or write was performed.',
                           'VIP full-market paging requires a separate bounded retrieval task.']
        atomic_json(output, public_data(report, settings))
        output.chmod(0o600)
    print(dumps({'report': str(output), 'archives': str(archive_dir)}), flush=True)


if __name__ == '__main__':
    main()
