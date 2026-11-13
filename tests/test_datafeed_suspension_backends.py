"""Datafeed acceptance in disposable databases using the existing local accounts."""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import pandas as pd
import pytest


@pytest.mark.skipif(os.environ.get('BETALENS_SUSPENSION_INTEGRATION') != '1', reason='requires disposable PG/CH databases')
def test_compact_storage_and_datafeed_on_both_backends(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'betalens-db-manager' / 'src'))
    from betalens_db_manager.config import Settings
    from betalens_db_manager.storage import Postgres
    from betalens_db_manager.clickhouse import ClickHouse
    from betalens_db_manager.reference_domains import build_scopes as legacy_scopes, check_integrity
    from betalens_db_manager.reference_storage import ReferenceStore
    from betalens_db_manager.suspension_ranges import build_scopes, coverage_heads
    from betalens_db_manager.util import digest
    from betalens_db_manager.suspension_migration import retirement_sql
    from betalens.datafeed import Datafeed
    from betalens.datafeed.pool import close_all_pools
    import psycopg2
    from psycopg2 import sql

    base = Settings.from_env('/home/janiszzz/betalens-infra/tushare.private.env')
    database = 'manager_test_suspension_' + uuid.uuid4().hex[:10]
    settings = replace(base, pg_database=database, ch_database=database,
                       config_dir=tmp_path / 'config', state_dir=tmp_path / 'state',
                       archive_dir=None, env_file=None)
    assert database.startswith('manager_test_suspension_') and database not in (base.pg_database, base.ch_database)
    settings.state_dir.mkdir()
    service = SimpleNamespace(settings=settings, pg=Postgres(settings), ch=ClickHouse(settings))
    request = {'modules': ['suspension'], 'security_codes': ['000001.SZ'],
               'start_date': '2024-01-02', 'end_date': '2024-01-04', 'check_times': ['10:00']}
    def bundle(endpoint, rows, parameters):
        return {'endpoint': endpoint, 'rows': rows, 'parameters': parameters, 'complete': True,
                'downloaded_at': '2026-11-10T00:00:00+00:00', 'archive_sha': ['test-proof']}
    bundles = [
        bundle('suspend_d', [{'ts_code': '000001.SZ', 'trade_date': '20240103', 'suspend_type': 'S', 'suspend_timing': None}],
               {'ts_code': '000001.SZ', 'start_date': '20240102', 'end_date': '20240104'}),
        bundle('stock_basic', [{'ts_code': '000001.SZ', 'list_date': '19910403', 'list_status': 'L'}], {'ts_code': '000001.SZ'}),
        bundle('trade_cal', [{'exchange': 'SZSE', 'cal_date': day, 'is_open': 1, 'pretrade_date': previous}
                            for day, previous in [('20240102', '20231229'), ('20240103', '20240102'), ('20240104', '20240103')]],
               {'exchange': 'SZSE', 'start_date': '20240102', 'end_date': '20240104'}),
    ]
    def revision(rid, **extra):
        return {'revision_id': rid, 'import_task_id': 'acceptance', 'attempt_id': 'attempt-' + rid,
                'module': 'suspension', 'mapping_version': 'reference-domains-v1', 'source_id': 'mock',
                'source_name': 'mock', 'downloaded_at': '2026-11-10T00:00:00+00:00', 'request': request, **extra}
    frames = {}
    report = {'database': database, 'targets': {}, 'production_modified': False}
    try:
        service.pg.create_database()
        service.ch.execute(f'CREATE DATABASE `{database}`')
        for target in ('postgres', 'clickhouse'):
            store = ReferenceStore(service, target)
            store.install()
            old = legacy_scopes(request, bundles)
            store.stage(revision('legacy'), old)
            store.publish('legacy', {s['scope_id']: None for s in old})
            scopes, pool = build_scopes(request, bundles)
            prior = store.snapshot()
            overlap_hash = digest(coverage_heads(prior['scopes'], prior['heads'], scopes))
            store.stage(revision('compact', evidence_pool=pool, expected_coverage_hash=overlap_hash), scopes)
            assert store.verify('compact')['verified']
            store.publish('compact', {s['scope_id']: None for s in scopes})
            assert [r['is_suspended'] for r in check_integrity(request, store.snapshot(audit=True))['rows']] == [False, True, False]
            config = ({'host': settings.pg_host, 'port': settings.pg_port, 'dbname': database,
                       'schema': settings.pg_schema, 'user': settings.pg_user, 'password': settings.pg_password}
                      if target == 'postgres' else
                      {'url': settings.ch_url, 'database': database, 'user': settings.ch_user, 'password': settings.ch_password})
            dates = ['2024-01-02 10:00', '2024-01-03 10:00', '2024-01-04 10:00']
            with Datafeed('stock_suspension_intervals_versions', config, backend=target, pit_policy='research') as data:
                result = data.query_trade_status({'codes': ['000001.SZ'], 'dates': dates})
                matrix = data.query_trade_status_matrix(['000001.SZ'], dates)
                assert result.value.tolist() == [1, 0, 1] == matrix['000001.SZ'].tolist()
                frames[target] = result
                middle = {**request, 'start_date': '2024-01-03', 'end_date': '2024-01-03'}
                empty = [bundle('suspend_d', [], {'ts_code': '000001.SZ', 'trade_date': '20240103'}), *bundles[1:]]
                replacement, evidence = build_scopes(middle, empty)
                prior = store.snapshot()
                store.stage(revision('empty', evidence_pool=evidence,
                    expected_coverage_hash=digest(coverage_heads(prior['scopes'], prior['heads'], replacement))), replacement)
                store.publish('empty', {s['scope_id']: None for s in replacement})
                assert data.query_trade_status_matrix(['000001.SZ'], dates)['000001.SZ'].tolist() == [1, 1, 1]
                assert not store.snapshot()['rows']
                store.revoke('empty', 'rollback-empty')
                assert data.query_trade_status_matrix(['000001.SZ'], dates)['000001.SZ'].tolist() == [1, 0, 1]
            with Datafeed('stock_suspension_intervals_versions', config, backend=target, pit_policy='strict') as strict:
                assert strict.query_trade_status_matrix(['000001.SZ'], dates).isna().all().all()
            migration = retirement_sql(settings)
            if target == 'postgres':
                with service.pg.connect() as connection, connection.cursor() as cursor:
                    cursor.execute(sql.SQL('CREATE TABLE {} (security_code text PRIMARY KEY,status_source text,is_tradable bigint)').format(sql.Identifier(settings.pg_schema, 'trade_status_daily')))
                    cursor.execute(sql.SQL('INSERT INTO {} VALUES (%s,%s,%s),(%s,%s,%s),(%s,%s,%s)').format(sql.Identifier(settings.pg_schema, 'trade_status_daily')),
                                   ('A','derived_from_daily_quote',1,'B','tushare.suspend_d',0,'C','security_lifecycle',0))
                for phase in ('stage', 'cutover', 'rollback'):
                    with service.pg.connect() as connection, connection.cursor() as cursor:
                        for statement in migration[target][phase]:cursor.execute(statement)
                        cursor.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(settings.pg_schema, 'trade_status_daily')))
                        assert cursor.fetchone()[0] == (1 if phase == 'cutover' else 3)
            else:
                table=service.ch.table('trade_status_daily')
                service.ch.execute(f'CREATE TABLE {table} (security_code String,status_source String,is_tradable Int8) ENGINE=ReplacingMergeTree ORDER BY security_code')
                service.ch.execute(f"INSERT INTO {table} VALUES ('A','derived_from_daily_quote',1),('B','tushare.suspend_d',0),('C','security_lifecycle',0)")
                for phase in ('stage', 'cutover', 'rollback'):
                    for statement in migration[target][phase]:service.ch.execute(statement)
                    assert int(service.ch.execute(f'SELECT count() FROM {table}')) == (1 if phase == 'cutover' else 3)
            report['targets'][target] = {'verified': True, 'values': [1, 0, 1], 'empty_revision': True,
                                          'revoke_restored': True, 'strict_preserves_unknown': True,
                                          'retirement_cutover_and_rollback': True}
        pd.testing.assert_frame_equal(frames['postgres'], frames['clickhouse'])
        report['cross_target_equal'] = True
    finally:
        close_all_pools()
        service.ch.execute(f'DROP DATABASE IF EXISTS `{database}`')
        connection = psycopg2.connect(**settings.pg('postgres'))
        try:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute(sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(database)))
        finally:connection.close()
        report['temporary_databases_removed'] = True
        if os.environ.get('BETALENS_SUSPENSION_REPORT'):
            Path(os.environ['BETALENS_SUSPENSION_REPORT']).write_text(json.dumps(report, ensure_ascii=False, indent=2))
