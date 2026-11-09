"""Read-only production acceptance probe; never creates sample data or changes stored PIT evidence.

Set BETALENS_DATAFEED_TEST_CONFIG (subscription JSON) or
BETALENS_DATAFEED_TEST_ENV (explicit manager env), then run this file with
--output PATH. The optional --task reads manager SQLite in mode=ro.
"""
from pathlib import Path
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from betalens.datafeed import Datafeed, get_absolute_trade_days, trade_periods_offset
from betalens.datafeed.contracts import DatafeedError, json_value, timestamp

TABLES = ['market_daily_raw', 'market_daily_post_adjusted', 'fundamental_periodic', 'security_dim', 'security_name_history', 'stock_industry_membership_versions', 'trade_calendar_versions', 'stock_suspension_intervals_versions', 'reference_revision', 'reference_scope', 'reference_publication_event', 'index_membership', 'security_industry_membership', 'industry_membership_record', 'trade_calendar', 'trade_status_daily', 'suspension_event', 'meta_column']

def manager_env():
    path = os.environ.get('BETALENS_DATAFEED_TEST_ENV')
    if not path:
        raise ValueError('Explicit BETALENS_DATAFEED_TEST_CONFIG or BETALENS_DATAFEED_TEST_ENV required')
    from dotenv import dotenv_values
    return dotenv_values(path, interpolate=False)

def configs():
    path = os.environ.get('BETALENS_DATAFEED_TEST_CONFIG')
    if path:
        config = json.loads(Path(path).read_text())
        return {backend: config[backend] for backend in ('postgres', 'clickhouse')}
    e = manager_env()
    return {'postgres': {'host': e['PGHOST'], 'port': int(e.get('PGPORT',5432)), 'dbname': e['PGDATABASE'], 'schema': e.get('PGSCHEMA','betalens'), 'user': e['PGUSER'], 'password': e.get('PGPASSWORD',''), 'statement_timeout_ms':60000},
        'clickhouse': {'url': e['CLICKHOUSE_URL'], 'database': e['CLICKHOUSE_DB'], 'user': e['CLICKHOUSE_USER'], 'password': e.get('CLICKHOUSE_PASSWORD',''), 'statement_timeout_ms':60000}}

def task_info(task_id):
    e = manager_env()
    root = Path(e.get('BETALENS_DB_MANAGER_STATE') or e.get('BETALENS_INFRA_STATE') or Path(os.environ.get('BETALENS_DATAFEED_TEST_ENV')).parent / 'tushare')
    with sqlite3.connect(f'file:{root / "tasks.sqlite3"}?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        task = dict(conn.execute('SELECT id,created,updated,status,plan FROM jobs WHERE id=?',(task_id,)).fetchone())
        plan = json.loads(task.pop('plan'))
        if plan.get('snapshot_ref'):
            saved = conn.execute('SELECT value FROM plan_snapshots WHERE hash=?',(plan['snapshot_ref'],)).fetchone()
            plan = {**json.loads(saved['value']), **plan}
        task['request'] = plan.get('request')
        task['slices'] = [dict(r) for r in conn.execute('SELECT dataset,status,count(*) slices,sum(rows) rows FROM slices WHERE job=? GROUP BY dataset,status',(task_id,))]
    return task

def relation(reader,table):
    q = '"' if reader.backend=='postgres' else '`'
    return f'{q}{reader.namespace}{q}.{q}{table}{q}' + (' FINAL' if reader.backend=='clickhouse' else '')

def inventory(backend, config):
    result = {}
    with Datafeed('security_dim',config,backend=backend) as data:
        r = data._reader
        with r.snapshot():
            if backend=='clickhouse':
                result['browser_test_databases'] = r.execute("SELECT name FROM system.databases WHERE startsWith(name,'browser_test_') ORDER BY name")
                result['browser_test_tables'] = r.execute("SELECT database,name FROM system.tables WHERE startsWith(name,'browser_test_') OR startsWith(database,'browser_test_') ORDER BY database,name")
                result['production_tables'] = r.execute('SELECT name FROM system.tables WHERE database={db:String} ORDER BY name',{'db':r.namespace})
            else:
                result['production_tables'] = r.execute('SELECT table_name AS name FROM information_schema.tables WHERE table_schema=%(schema)s ORDER BY table_name',{'schema':r.namespace})
            result['catalog_columns'] = {row['name']: dict(r.columns(row['name'])) for row in result['production_tables']}
        for table in TABLES:
            with r.snapshot():
                columns = r.columns(table)
                stats={'columns':columns}
                if not columns:
                    result[table]=stats
                    continue
                expressions=['count(*) AS rows']
                for col in ['available_at','event_date','trade_date','period_end','start_date']:
                    if col in columns:
                        expressions.extend([f'min({col}) AS {col}_min',f'max({col}) AS {col}_max'])
                if 'available_at' in columns:
                    expressions.append('sum(CASE WHEN available_at IS NOT NULL THEN 1 ELSE 0 END) AS available_at_nonnull')
                if {'available_at','availability_basis','data_quality_status'}<=columns.keys():
                    expressions.append("sum(CASE WHEN available_at IS NOT NULL AND availability_basis IN ('source_version_publication_time','verified_historical_version') AND data_quality_status IN ('accepted','verified') THEN 1 ELSE 0 END) AS proven_rows")
                stats['counts']=r.execute('SELECT '+','.join(expressions)+' FROM '+relation(r,table))[0]
                # Empty ClickHouse aggregates return default dates; they are not coverage.
                if int(stats['counts']['rows']) == 0:
                    stats['counts'] = {key: (0 if key in {'rows', 'available_at_nonnull', 'proven_rows'} else None)
                                       for key in stats['counts']}
                group=[c for c in ['asset_type','availability_basis','data_quality_status'] if c in columns]
                if group:
                    stats['groups']=r.execute('SELECT '+','.join(group)+',count(*) AS rows FROM '+relation(r,table)+' GROUP BY '+','.join(group)+' ORDER BY '+','.join(group))
                if table in ['fundamental_periodic','reference_publication_event','reference_revision']:
                    dims={'fundamental_periodic':['statement_type','report_type'], 'reference_publication_event':['event_type'], 'reference_revision':['module']}[table]
                    stats['dimensions']=r.execute('SELECT '+','.join(dims)+',count(*) AS rows FROM '+relation(r,table)+' GROUP BY '+','.join(dims)+' ORDER BY '+','.join(dims))
                result[table]=stats
    return result

def cases(stock_codes, pit_policy="research"):
    """Explicit public calls, including missing-data and invalid-parameter contracts."""
    table = 'market_daily_post_adjusted'
    codes = ['000001.SZ', '600000.SH', '__missing__', '000001.SZ']
    dates = ['2026-10-13 15:00', '2026-11-04 00:00']
    nearest = {'codes': codes, 'datetimes': dates, 'metric': 'close', 'time_tolerance': 72,
               'filters': {'asset_type': 'stock'}}
    result = []
    for price_table in ['market_daily_raw', table]:
        for asset, code in [('stock', '000001.SZ'), ('fund', '510300.SH'), ('index', '000300.SH'), ('bond', '110059.SH')]:
            result.append((price_table + '/' + asset, price_table, 'query_time_range',
                           [[code], '2026-10-05', '2026-11-04', 'close'], {'filters': {'asset_type': asset}}, None))
    for metric in ['open', 'high', 'low', 'volume', 'amount', 'pe_ttm', 'total_market_value']:
        result.append(('field/' + metric, table, 'query_time_range', [codes, '2026-10-05', '2026-11-04', metric],
                       {'filters': {'asset_type': 'stock'}, 'as_of': '2026-11-04'}, None))
    for direction in ['before', 'after']:
        result.append(('nearest/' + direction, table, 'query_nearest_' + direction, [nearest], {}, None))
        p = {k: v for k, v in nearest.items() if k != 'datetimes'}
        p['ranges'] = [('2026-10-12', '2026-10-15'), ('2026-11-02', '2026-11-04')]
        result.append(('nearest/range_' + direction, table, 'query_nearest_in_range_' + direction, [p], {}, None))
    result.extend([
        ('nearest/utc', table, 'query_nearest_before', [{**nearest, 'datetimes': ['2026-10-13T07:00:00Z', '2026-11-03T16:00:00Z']}], {}, None),
        ('nearest/bulk5000', table, 'query_nearest_before', [{**nearest, 'codes': stock_codes[:5000], 'datetimes': [dates[-1]]}], {}, None),
        ('available_dates', table, 'get_available_dates', [codes[0], 'close'], {'filters': {'asset_type': 'stock'}}, None),
        ('latest_date', table, 'get_latest_date', [codes[0], 'close'], {'filters': {'asset_type': 'stock'}}, None),
        ('latest_date/all', table, 'get_latest_date', [], {'metric': 'close', 'filters': {'asset_type': 'stock'}}, None),
        ('names/current', 'security_dim', 'query_names', [codes + ['510300.SH', '000300.SH']], {}, None),
        ('names/current_all_stocks', 'security_dim', 'query_names', [stock_codes], {}, None),
        ('names/historical', 'security_dim', 'query_names', [codes], {'as_of': '2026-10-13 15:00'}, None),
        ('status', ('stock_suspension_intervals_versions' if pit_policy=='strict' else 'trade_status_daily'), 'query_trade_status', [{'codes': codes, 'dates': dates}], {}, None),
        ('status/all_stocks', ('stock_suspension_intervals_versions' if pit_policy=='strict' else 'trade_status_daily'), 'query_trade_status', [{'dates': [dates[0]]}], {}, None),
        ('empty/range', table, 'query_time_range', [[]], {'metric': 'close'}, None),
        ('empty/nearest', table, 'query_nearest_before', [{**nearest, 'codes': []}], {}, None),
        ('empty/names', 'security_dim', 'query_names', [[]], {}, None),
        ('empty/status', ('stock_suspension_intervals_versions' if pit_policy=='strict' else 'trade_status_daily'), 'query_trade_status', [{'codes': [], 'dates': dates}], {}, None),
        ('empty/industry', ('stock_industry_membership_versions' if pit_policy=='strict' else 'security_industry_membership'), 'query_industry', [[], dates], {}, None),
        ('empty/index_panel', 'index_membership', 'get_index_universe_panel', ['000300.SH', []], {}, None),
    ])
    for statement, metric in [('income', 'revenue'), ('balancesheet', 'total_assets'), ('cashflow', 'operating_cashflow'), ('fina_indicator', 'roe')]:
        filters = {'statement_type': statement, 'report_type': 'unspecified' if statement == 'fina_indicator' else '1'}
        result.append(('finance/' + statement, 'fundamental_periodic', 'query_time_range',
                       [codes, None, None, metric], {'filters': filters}, None))
    for scheme in [None, 'SW2014', 'SW2021', 'CITICS']:
        for level in (1, 2, 3):
            result.append((f'industry/{scheme}/{level}', ('stock_industry_membership_versions' if pit_policy=='strict' else 'security_industry_membership'), 'query_industry',
                           [codes, dates, scheme], {'level': level}, None))
    result.append(('industry/members', ('stock_industry_membership_versions' if pit_policy=='strict' else 'security_industry_membership'), 'get_industry_members',
                   ['银行', dates[0], 'SW2021'], {'level': 1}, None))
    for exchange in ['SSE', 'SZSE', 'BSE']:
        for period in ['D', 'W', 'M', 'Q', 'H', 'Y']:
            result.append((f'calendar/{exchange}/{period}', None, 'get_absolute_trade_days',
                           ['2026-10-12', '2026-10-14', period, exchange], {'as_of': '2026-11-04'}, None))
    for period in ['D', 'W', 'M', 'Q', 'H', 'Y']:
        result.append(('offset/' + period, None, 'trade_periods_offset',
                       ['2026-10-12 10:00', 1, period], {'as_of': '2026-11-04'}, None))
    for method, date in [('get_index_universe', dates[0]), ('get_index_universe_date', dates[0]),
                         ('get_index_universe_panel', ['2026-10-13', '2026-11-03'])]:
        result.append(('index/' + method, 'index_membership', method, ['000300.SH', date], {}, None))
    for method, observation in [('get_index_universe','2026-10-31'),('get_index_universe_date','2026-10-31'),('get_index_universe_panel',['2026-10-31','2026-11-03'])]:
        result.append(('index/known/'+method,'index_membership',method,['000300.SH',observation],{},None))
    result.append(('risks', 'stock_suspension_intervals_versions' if pit_policy=='strict' else 'trade_status_daily',
                   'query_trade_risks',[codes,dates],{},None))
    result.extend([
        ('reject/metric', table, 'query_time_range', [], {'metric': 'nonexistent_metric'}, 'SchemaError'),
        ('reject/finance_dimensions', 'fundamental_periodic', 'query_time_range', [], {'metric': 'revenue'}, 'ValueError'),
        ('reject/SHSE', None, 'get_absolute_trade_days', ['2026-10-12', '2026-10-14', 'D', 'SHSE'], {}, 'ValueError'),
        ('reject/H2', None, 'get_absolute_trade_days', ['2026-10-12', '2026-10-14', 'H2'], {}, 'ValueError'),
        ('reject/scalar_codes', table, 'query_time_range', ['000001.SZ'], {'metric': 'close'}, 'ValueError'),
        ('reject/numeric_time', table, 'query_nearest_before', [{**nearest, 'datetimes': [20261103]}], {}, 'ValueError'),
    ])
    return result



def report_value(value):
    """Serialize report containers without changing the actual API return value."""
    if isinstance(value, dict):
        return {str(key): report_value(item) for key,item in value.items()}
    if isinstance(value, (set,list,tuple)):
        items=sorted(value) if isinstance(value,set) else value
        return [report_value(item) for item in items]
    return value


def call_case(backend, config, case, pit_policy="research"):
    name, table, method, args, kwargs, expected_error = case
    started = perf_counter()
    try:
        if table is None:
            fn = get_absolute_trade_days if method == 'get_absolute_trade_days' else trade_periods_offset
            value = fn(*args, **kwargs, backend=backend, db_config=config, pit_policy=pit_policy,
                       table_name="trade_calendar_versions" if pit_policy=="strict" else "trade_calendar")
        else:
            with Datafeed(table, config, backend=backend, pit_policy=pit_policy) as data:
                value = getattr(data, method)(*args, **kwargs)
    except Exception as exc:
        kind = type(exc).__name__
        message = str(exc) if isinstance(exc, (DatafeedError, ValueError, IndexError)) else 'Database/transport error (details omitted)'
        status = 'error'
        if kind == expected_error:
            status = 'expected_rejection'
        elif kind == 'PITUnavailableError':
            status = 'evidence_blocked'
        elif kind == 'IndexError' and method == 'trade_periods_offset':
            status = 'data_range_blocked'
        return {'name': name, 'status': status, 'error': kind, 'message': message,
                'seconds': round(perf_counter() - started, 4)}, None
    report = {'name': name, 'seconds': round(perf_counter() - started, 4)}
    if expected_error:
        report['status'] = 'error_expected_rejection'
        return report, value
    if isinstance(value, pd.DataFrame):
        data_column = next((c for c in ['value', 'close', 'industry_value', 'name'] if c in value), None)
        known = int(value[data_column].notna().sum()) if data_column else 0
        report.update(status='values' if known else 'empty_or_unknown', rows=len(value), known=known,
                      columns=list(value), dtypes={c: str(t) for c, t in value.dtypes.items()},
                      sample=json.loads(value.head(2).to_json(orient='records', date_format='iso', force_ascii=False)))
        if method=='query_trade_risks':
            report['known_flags']={column:int(value[column].notna().sum()) for column in ['is_listed','is_st','is_suspended']}
    else:
        report.update(status='values' if value else 'empty_or_unknown', value=report_value(value))
    return report, value


def native_sample(backend, config):
    """Compare stored values and UTC instants without making them PIT-visible."""
    result = {}
    with Datafeed('market_daily_raw', config, backend=backend) as data:
        r = data._reader
        with r.snapshot():
            result['readonly'] = r.execute("SELECT current_setting('transaction_read_only') AS readonly, current_setting('transaction_isolation') AS isolation") if backend == 'postgres' else r.execute("SELECT getSetting('readonly') AS readonly")
            fields = ['asset_type', 'security_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount',
                      'adjustment_factor', 'adjustment_method', 'available_at', 'availability_basis', 'data_quality_status']
            for table in ['market_daily_raw', 'market_daily_post_adjusted']:
                rows = r.rows(table, fields, [('security_code', 'IN', ['000001.SZ', '600000.SH', '510300.SH', '000300.SH'])],
                              [('asset_type', 'ASC'), ('security_code', 'ASC'), ('trade_date', 'ASC')])
                for row in rows:
                    row['available_at'] = timestamp(row['available_at'], stored=True).isoformat() if row['available_at'] else None
                result[table] = rows
            securities = r.rows('security_dim', ['security_code'], [('asset_type', '=', 'stock')], [('security_code', 'ASC')])
            result['stock_codes'] = [row['security_code'] for row in securities]
    return result


def compare_outputs(left, right):
    if isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
        return
    assert left == right


def compare_backends(outputs, reports):
    comparison = {}
    for name in reports['postgres']:
        pg, ch = reports['postgres'][name], reports['clickhouse'][name]
        if pg.get('error') or ch.get('error'):
            comparison[name] = 'same_error' if (pg.get('error'), pg.get('status')) == (ch.get('error'), ch.get('status')) else 'mismatch'
            continue
        try:
            compare_outputs(outputs['postgres'][name], outputs['clickhouse'][name])
            comparison[name] = 'equal'
        except (AssertionError, TypeError, ValueError):
            comparison[name] = 'mismatch'
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task')
    parser.add_argument('--pit-policy',choices=['research','strict'],default='research')
    args = parser.parse_args()
    out = {'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
           'scope': 'read-only real databases; no fixtures, DDL, imports or evidence relabelling', 'pit_policy':args.pit_policy}
    if args.task:
        out['task'] = task_info(args.task)
    cfg = configs()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {key: executor.submit(inventory, key, value) for key, value in cfg.items()}
        for key, future in futures.items():
            out[key] = future.result()
    print('Inventory complete', flush=True)
    samples = {backend: native_sample(backend, config) for backend, config in cfg.items()}
    stock_codes = samples['postgres'].pop('stock_codes')
    assert stock_codes == samples['clickhouse'].pop('stock_codes')
    out['readonly'] = {backend: sample.pop('readonly') for backend, sample in samples.items()}
    out['native_samples'] = samples
    reports, outputs = {}, {}
    for backend, config in cfg.items():
        reports[backend], outputs[backend] = {}, {}
        for case in cases(stock_codes,args.pit_policy):
            report, value = call_case(backend, config, case,args.pit_policy)
            reports[backend][case[0]], outputs[backend][case[0]] = report, value
        print(backend + ': ' + str(len(reports[backend])) + ' public API probes complete', flush=True)
    out['api'] = reports
    out['cross_backend'] = compare_backends(outputs, reports)
    out['checks'] = {'utc_equals_shanghai': {}, 'nearest_duplicate_inputs': {}}
    for backend in cfg:
        try:
            compare_outputs(outputs[backend]['nearest/before'], outputs[backend]['nearest/utc'])
            out['checks']['utc_equals_shanghai'][backend] = True
        except (AssertionError, TypeError, ValueError):
            out['checks']['utc_equals_shanghai'][backend] = False
        nearest = outputs[backend]['nearest/before']
        out['checks']['nearest_duplicate_inputs'][backend] = isinstance(nearest, pd.DataFrame) and nearest.code.tolist() == [code for code in ['000001.SZ', '600000.SH', '__missing__', '000001.SZ'] for _ in range(2)]
    out['checks']['native_samples_equal'] = samples['postgres'] == samples['clickhouse']
    pg_catalog, ch_catalog = out['postgres']['catalog_columns'], out['clickhouse']['catalog_columns']
    out['schema_type_differences'] = {table: {column: [pg_catalog.get(table, {}).get(column), ch_catalog.get(table, {}).get(column)]
        for column in sorted(pg_catalog.get(table, {}).keys() | ch_catalog.get(table, {}).keys())
        if pg_catalog.get(table, {}).get(column) != ch_catalog.get(table, {}).get(column)}
        for table in sorted(pg_catalog.keys() | ch_catalog.keys()) if pg_catalog.get(table) != ch_catalog.get(table)}
    out['checks']['catalog_column_names_equal'] = {table: sorted(fields) for table, fields in pg_catalog.items()} == {table: sorted(fields) for table, fields in ch_catalog.items()}
    out['checks']['query_schema_logical_types_equal'] = all(out['postgres'][table]['columns'] == out['clickhouse'][table]['columns'] for table in TABLES)
    required = ['market_daily_raw/stock', 'market_daily_raw/fund', 'market_daily_raw/index',
                'market_daily_post_adjusted/stock', 'market_daily_post_adjusted/fund',
                'finance/income', 'finance/balancesheet', 'finance/cashflow', 'finance/fina_indicator',
                'names/current_all_stocks', 'names/historical', 'status/all_stocks',
                'industry/None/1', 'industry/None/2', 'industry/None/3', 'calendar/SSE/D',
                'calendar/SZSE/D', 'calendar/BSE/D', 'offset/D', 'index/get_index_universe']
    out['business_readiness'] = {backend: {name: reports[backend][name]['status'] == 'values' for name in required} for backend in cfg}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=json_value) + '\n')
    print(json.dumps({'checks': out['checks'], 'comparison': {status: list(out['cross_backend'].values()).count(status) for status in set(out['cross_backend'].values())}}, ensure_ascii=False))
    failed = any(value == 'mismatch' for value in out['cross_backend'].values()) or any(report['status'].startswith('error') for rows in reports.values() for report in rows.values())
    failed = failed or not all(out['checks'][key] for key in ['native_samples_equal', 'query_schema_logical_types_equal', 'catalog_column_names_equal'])
    failed = failed or not all(out['checks']['utc_equals_shanghai'].values()) or not all(out['checks']['nearest_duplicate_inputs'].values())
    ready = all(known for backend in out['business_readiness'].values() for known in backend.values())
    # 0: representative business queries available; 1: unexpected failure;
    # 2: API probe completed but real data/evidence still blocks acceptance.
    raise SystemExit(1 if failed else 0 if ready else 2)

if __name__=='__main__':
    main()
