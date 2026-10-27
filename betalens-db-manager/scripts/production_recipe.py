"""Generate bounded real-source acceptance requests without running them.

This helper never creates a manager, loads credentials or opens a database.
The caller must run each request through plan -> download -> preview -> commit
-> read-back. No cached or invented source response is accepted by this recipe.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from betalens_db_manager.catalog import load_catalog
from betalens_db_manager.canonical import builtin_catalog
from betalens_db_manager.retrieval import request_rate
from betalens_db_manager.safe_yaml import safe_load
from betalens_db_manager.util import atomic_json, now, parse_date


def recipes(trade_date='20261016', period='20241231', targets=None, rpm=40):
    day = parse_date(trade_date)
    report_period = parse_date(period)
    target_names = list(targets or ['postgres', 'clickhouse'])
    if not target_names or set(target_names)-{'postgres', 'clickhouse'}:
        raise ValueError('目标库只能包含 postgres/clickhouse')
    rpm = request_rate(rpm)
    current_day = day.isoformat()
    event_start = '2004-01-01'
    output = []

    def add(label, table, endpoints, asset='stock', codes=None, start=None, end=None,
            parameters=None, notes=None):
        request = {'tables': [table], 'source_id': 'tushare', 'endpoints': list(endpoints),
                   'asset_types': [asset], 'codes': list(codes if codes is not None else ['000001.SZ']),
                   'start_date': start or current_day, 'end_date': end or current_day,
                   'targets': target_names, 'requests_per_minute': rpm,
                   'mode': 'refresh_source', 'parameters': parameters or {}}
        output.append({'label': label, 'request': request, 'expected_tables': [table],
                       'expected_endpoints': list(endpoints), 'sample_only': True,
                       'execution': 'plan_download_preview_confirm_commit_readback', 'notes': list(notes or [])})

    # Unified dimensions and the complete per-source profile tables.
    for table in ('security_dim', 'stock_profile'):
        add(table+'_stock', table, ['stock_basic'], parameters={'stock_basic': {'list_status': 'L'}})
    for table in ('security_dim', 'fund_profile'):
        add(table+'_fund', table, ['fund_basic'], asset='fund', codes=['510300.SH'],
            parameters={'fund_basic': {'market': 'E'}})
    for table in ('security_dim', 'index_profile'):
        add(table+'_index', table, ['index_basic'], asset='index', codes=['000300.SH'],
            parameters={'index_basic': {'market': 'CSI'}})
    add('etf_profile', 'etf_profile', ['etf_basic'], asset='fund', codes=['510300.SH'])
    add('trade_calendar', 'trade_calendar', ['trade_cal'], codes=[],
        parameters={'trade_cal': {'exchange': 'SSE'}})

    # Separate asset tasks prevent a stock code accidentally querying all ETFs
    # or an SW index code accidentally probing an ordinary index endpoint.
    for asset, code, quote, factor in (
        ('stock', '000001.SZ', 'daily', 'adj_factor'),
        ('fund', '510300.SH', 'fund_daily', 'fund_adj'),
        ('index', '000300.SH', 'index_daily', None),
        ('index', '801010.SI', 'sw_daily', None),
    ):
        suffix = 'sw_index' if quote == 'sw_daily' else asset
        raw_endpoints = [quote] + (['daily_basic'] if asset == 'stock' else [])
        add('raw_'+suffix, 'market_daily_raw', raw_endpoints, asset=asset, codes=[code])
        add('post_adjusted_'+suffix, 'market_daily_post_adjusted',
            [quote] + ([factor] if factor else []), asset=asset, codes=[code],
            notes=['Compiler must add daily metrics/name/calendar dependencies without writing unselected tables.'])
        add('trade_status_'+suffix, 'trade_status_daily',
            [quote] + (['suspend_d'] if asset == 'stock' else []), asset=asset, codes=[code],
            notes=['Fund/index status remains derived; a quote does not prove authoritative tradability.'])
    add('stock_quote_supplement_daily', 'stock_quote_supplement_daily', ['bak_daily'],
        notes=['Undocumented units remain source values; no projection into unified prices.'])

    # Industry versions are explicit; unavailable CITIC is a capability result.
    for table in ('industry_scheme_dim', 'industry_dim'):
        add(table+'_sw', table, ['index_classify'], codes=[],
            parameters={'index_classify': {'src': 'SW2021', 'level': 'L1'}})
    for table in ('security_industry_membership', 'industry_membership_record'):
        for endpoint, scheme in (('index_member_all', 'sw'), ('ci_index_member', 'citic')):
            add(table+'_'+scheme, table, [endpoint], start=event_start,
                parameters={endpoint: {'is_new': 'Y'}},
                notes=['Permission denied/no returned rows must remain unverified, never fabricated.'])
    for table in ('industry_scheme_dim', 'industry_dim'):
        add(table+'_citic', table, ['ci_index_member'], start=event_start,
            parameters={'ci_index_member': {'is_new': 'Y'}},
            notes=['CITIC may require additional permission.'])
    add('index_membership', 'index_membership', ['index_weight'], asset='index', codes=['399300.SZ'],
        start=date(day.year, day.month, 1).isoformat(),
        notes=['Use the month through sample day because weights need not exist on every trading day.'])

    # Ordinary and VIP contracts are tested independently; do not blend two
    # endpoints with identical business content into an ambiguous merge.
    for endpoint in ('income', 'balancesheet', 'cashflow', 'fina_indicator'):
        for source in (endpoint, endpoint+'_vip'):
            add('financial_'+source, 'fundamental_periodic', [source],
                codes=['000333.SZ'] if endpoint in {'income','fina_indicator'} else ['000001.SZ'],
                start=date(report_period.year, 1, 1).isoformat(), end=report_period.isoformat(),
                parameters={source: {'period': report_period.strftime('%Y%m%d'), **({'limit':1} if source.endswith('_vip') else {})}},
                notes=['VIP is separately permission-gated; ignoring ts_code must block rather than download all-market data.']
                if source.endswith('_vip') else None)

    # Full history for event observations is bounded to this single security.
    for table, endpoint in (
        ('dividend_event', 'dividend'), ('security_name_history', 'namechange'),
        ('corporate_action', 'dividend'), ('corporate_action', 'namechange'),
        ('corporate_action', 'adj_factor'),
    ):
        add(table+'_'+endpoint, table, [endpoint], start=event_start if endpoint != 'adj_factor' else current_day,
            notes=['An empty event history is no sample evidence; report separately.'])
    add('suspension_event', 'suspension_event', ['suspend_d'], codes=[],
        notes=['One market day; only actual suspension events are imported. Empty is an explicit no-sample result.'])
    for table in ('fund_nav_daily', 'fund_nav_observation'):
        add(table, table, ['fund_nav'], asset='fund', codes=['510300.SH'],
            parameters={'fund_nav': {'start_date': day.strftime('%Y%m%d'), 'end_date': day.strftime('%Y%m%d')}},
            notes=['No NAV on requested date is no sample evidence; never manufacture a NAV.'])
    add('fund_shares_daily', 'fund_shares_daily', ['fund_share'], asset='fund', codes=['510300.SH'])
    add('security_code_history', 'security_code_history', ['bse_mapping'], codes=['920001.BJ'], start=event_start,
        parameters={'bse_mapping': {'n_code': '920001.BJ'}},
        notes=['Listing date must not become code-change effective date.'])
    return output


def inspect_config(config_dir: Path, cases: list[dict]):
    """Read authoritative YAML only; no registry construction or publication."""
    tables, mappings = {}, {}
    for kind, destination in (('tables', tables), ('mappings', mappings)):
        for path in sorted((config_dir/kind).glob('*.yaml')):
            destination[path.stem] = safe_load(path.read_text())
    coverage = []
    for name in sorted(tables):
        selected = [case for case in cases if name in case['expected_tables']]
        available = {(binding['source_id'], binding['endpoint'])
                     for binding in mappings.get(name, {}).get('bindings', [])}
        coverage.append({'table': name, 'cases': [case['label'] for case in selected],
                         'targets': ['postgres', 'clickhouse'], 'mapping_found': name in mappings,
                         'unbound_requested_endpoints': sorted({endpoint for case in selected
                             for endpoint in case['expected_endpoints'] if ('tushare', endpoint) not in available}),
                         'status': 'recipe_available' if selected else 'no_recipe'})
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path, default=Path.home()/'.config/betalens/db-manager')
    parser.add_argument('--trade-date', default='20261016')
    parser.add_argument('--period', default='20241231')
    parser.add_argument('--rpm', type=request_rate, default=40)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cases = recipes(args.trade_date, args.period, rpm=args.rpm)
    all_tables = set(builtin_catalog())
    all_endpoints = set(load_catalog())
    covered_tables = {table for case in cases for table in case['expected_tables']}
    covered_endpoints = {endpoint for case in cases for endpoint in case['expected_endpoints']}
    report = {'generated_at': now(), 'execution_performed': False,
              'database_connections_used': False, 'cases': cases,
              'tables': inspect_config(args.config_dir, cases),
              'missing_builtin_tables': sorted(all_tables-covered_tables),
              'missing_source_endpoints': sorted(all_endpoints-covered_endpoints),
              'notes': ['Requests are plans only; caller must fetch through the real managed downloader.',
                        'Confirmation is produced after preview; no confirmation hash is synthesized.',
                        'Schema changes and account changes are separate operations.']}
    atomic_json(args.output, report)


if __name__ == '__main__':
    main()
