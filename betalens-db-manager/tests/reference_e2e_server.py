"""Real browser fixture: disposable targets and an actual local HTTP source.

This fixture never writes to production tables. It retains only a sanitized
acceptance report after removing its exact database, CH writer, and env file.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import json
import os
import subprocess
import signal
import sys
import tempfile
import threading
import uuid

import uvicorn
from betalens_db_manager.api import create_app
from betalens_db_manager.catalog import load_catalog
from betalens_db_manager.config import Settings
from betalens_db_manager.manager import DatabaseManagerService
from betalens_db_manager.reference_service import ReferenceWorkflow
from betalens_db_manager.util import atomic_json, dumps, parse_date
from controlled_ch import configure_writer


def main():
    suffix = uuid.uuid4().hex[:10]
    artifact_root = Path(os.environ.get('BETALENS_REFERENCE_ARTIFACTS', str(Path.home() / '.cache' / 'betalens-reference-browser'))).expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = Path(tempfile.mkdtemp(prefix='fixture-', dir=artifact_root))
    root.chmod(0o700)
    base = Settings.from_env(os.environ.get('BETALENS_TEST_ENV', '/home/janiszzz/betalens-infra/tushare.private.env'))
    database = 'browser_test_' + suffix
    settings = replace(base, pg_database=database, ch_database=database, state_dir=root/'state',
                       config_dir=root/'config', archive_dir=root/'archive', env_file=root/'credentials.env', token='')
    service = DatabaseManagerService(settings)
    cleanup_writer = None
    child = None
    mock = None
    source_requests = []
    seeds = []
    verification = []
    report = {'fixture': 'reference-e2e-v1', 'database': database, 'production_writes': False}
    try:
        # The PG database itself is fixture setup; all version-table DDL still
        # comes from the user's preview/confirm action in the browser.
        service.pg.create_database()
        cleanup_writer = configure_writer(service)
        settings = service.settings
        endpoints = {}
        for endpoint in ('trade_cal', 'stock_basic', 'suspend_d', 'index_classify', 'index_member_all', 'ci_index_member'):
            spec = deepcopy(load_catalog()[endpoint].as_dict())
            spec.update(path='/' + endpoint, method='GET', response_path='',
                        pagination={'offset_param': 'offset', 'limit_param': 'limit', 'size': 1})
            endpoints[endpoint] = spec

        def snapshot():
            result = {}
            for target in ('postgres', 'clickhouse'):
                store = ReferenceWorkflow(service).store(target)
                trusted = store.snapshot()
                history = store.snapshot(audit=True)
                result[target] = {'trusted_rows': len(trusted['rows']), 'history_rows': len(history['rows']),
                                  'events': len(history['events']), 'heads': trusted['heads'],
                                  'trusted_dates': sorted(str(row['trade_date']) for row in trusted['rows'] if row.get('trade_date')),
                                  'history_dates': sorted(str(row['trade_date']) for row in history['rows'] if row.get('trade_date')),
                                  'verification': history['verification'], 'invalid_revision_ids': history['invalid_revision_ids'],
                                  'actual_versions': history}
            return result

        class Handler(BaseHTTPRequestHandler):
            def send(self, value, code=200):
                raw = dumps(value).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                if self.path != '/fixture/seed':
                    self.send({'error': 'unknown fixture operation'}, 404); return
                try:
                    if not seeds:
                        workflow = ReferenceWorkflow(service)
                        for day in ('2026-10-12', '2026-10-14'):
                            item = workflow.plan({'modules': ['calendar'], 'targets': ['postgres', 'clickhouse'],
                                'start_date': day, 'end_date': day, 'exchanges': ['SSE'],
                                'security_codes': [], 'source_id': 'reference_http'})
                            service.download(item['id'])
                            preview = service.preview(item['id'])
                            assert not preview['blocking'], preview
                            committed = service.commit(item['id'], preview['token'])
                            assert committed['status'] == 'completed', committed
                            seeds.append(item['id'])
                    result = {'seed_tasks': seeds, 'targets': snapshot()}
                    verification.append({'operation': 'seed', **result})
                    self.send(result)
                except Exception as exc:
                    self.send({'error': service.safe_error(exc)}, 500)

            def do_GET(self):
                parsed = urlsplit(self.path)
                if parsed.path == '/fixture/state':
                    try:
                        result = {'targets': snapshot(), 'source_requests': list(source_requests),
                                  'seed_tasks': seeds, 'database': database}
                        verification.append({'operation': 'readback', **result})
                        self.send(result)
                    except Exception as exc:
                        self.send({'error': service.safe_error(exc)}, 500)
                    return
                if parsed.path == '/fixture/ready':
                    self.send({'ready': True, 'production_writes': False}); return
                endpoint = parsed.path.strip('/')
                if endpoint not in endpoints:
                    self.send({'error': 'unknown endpoint'}, 404); return
                params = {key: values[0] for key, values in parse_qs(parsed.query).items()}
                source_requests.append({'endpoint': endpoint, 'parameters': params})
                rows = []
                if endpoint == 'trade_cal':
                    day = parse_date(params['start_date'])
                    while day <= parse_date(params['end_date']):
                        previous = day - timedelta(days=1)
                        while previous.weekday() >= 5:
                            previous -= timedelta(days=1)
                        rows.append({'exchange': params['exchange'], 'cal_date': day.strftime('%Y%m%d'),
                                     'is_open': int(day.weekday() < 5), 'pretrade_date': previous.strftime('%Y%m%d')})
                        day += timedelta(days=1)
                elif endpoint == 'stock_basic' and params.get('list_status') == 'L':
                    rows = [{'ts_code': params['ts_code'], 'name': '样本股票', 'list_status': 'L',
                             'list_date': '19910403', 'exchange': 'SZSE'}]
                elif endpoint in {'index_member_all', 'ci_index_member'} and params.get('is_new') == 'Y':
                    rows = [{'ts_code': params['ts_code'], 'name': '样本股票', 'is_new': 'Y', 'in_date': '20100101',
                             'l1_code': '801010.SI', 'l1_name': '农林牧渔', 'l2_code': '801011.SI',
                             'l2_name': '种植业', 'l3_code': '850111.SI', 'l3_name': '粮食种植'}]
                elif endpoint == 'index_classify':
                    level = int(params['level'][1:]); code = {1: '801010.SI', 2: '801011.SI', 3: '850111.SI'}[level]
                    rows = [{'index_code': code, 'industry_name': '样本行业', 'level': params['level'], 'src': params['src']}]
                offset = int(params.get('offset', 0)); size = int(params.get('limit', 100000))
                rows = rows[offset:offset + size]
                # Deliberately reorder every output field. All fields, including
                # optional nulls, are present so mapping can only use names.
                fields = list(reversed([field['name'] for field in endpoints[endpoint]['fields']]))
                self.send({'fields': fields, 'items': [[row.get(field) for field in fields] for row in rows]})

            def log_message(self, *_):
                pass

        mock = ThreadingHTTPServer(('127.0.0.1', 18086), Handler)
        threading.Thread(target=mock.serve_forever, daemon=True).start()
        service.sources.save({'source_id': 'reference_http', 'provider': 'http', 'label': '参考表本地 HTTP 验收源',
                              'base_url': 'http://127.0.0.1:18086', 'endpoints': endpoints})
        assert settings.env_file.parent.resolve() == root.resolve()
        settings.write_private_env({'PGHOST': settings.pg_host, 'PGPORT': str(settings.pg_port),
            'PGDATABASE': settings.pg_database, 'PGSCHEMA': settings.pg_schema, 'PGUSER': settings.pg_user,
            'PGPASSWORD': settings.pg_password, 'CLICKHOUSE_URL': settings.ch_url, 'CLICKHOUSE_DB': settings.ch_database,
            'CLICKHOUSE_USER': settings.ch_user, 'CLICKHOUSE_PASSWORD': settings.ch_password,
            'CLICKHOUSE_WRITER_MODE': settings.ch_writer_mode, 'TUSHARE_TOKEN': '', 'TUSHARE_RPM': '10000',
            'BETALENS_INFRA_STATE': str(settings.state_dir), 'BETALENS_DB_MANAGER_STATE': str(settings.state_dir),
            'BETALENS_INFRA_ARCHIVE': str(settings.archive_dir), 'BETALENS_DB_MANAGER_CONFIG': str(settings.config_dir)})
        child = subprocess.Popen([sys.executable, '-m', 'betalens_db_manager', '--env', str(settings.env_file), 'worker'])
        report['fixture_root'] = str(root)
        # Uvicorn restores and re-raises SIGTERM after graceful shutdown.
        # A Python exit handler keeps that re-raise inside this finally block.
        def exit_fixture(*_):
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, exit_fixture)
        signal.signal(signal.SIGINT, exit_fixture)
        uvicorn.run(create_app(service), host='127.0.0.1', port=18084, log_level='warning')
    finally:
        if child is not None:
            child.terminate()
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=10)
        if mock is not None:
            mock.shutdown(); mock.server_close()
        report.update(seed_tasks=seeds, source_requests=source_requests, verification=verification)
        if cleanup_writer is not None:
            cleanup_writer()
            report['writer_removed'] = True
        # Identifier came solely from this invocation's random disposable name.
        assert database.startswith('browser_test_') and database.endswith(suffix)
        service.ch.execute(f'DROP DATABASE IF EXISTS `{database}`')
        import psycopg2
        from psycopg2 import sql
        connection = psycopg2.connect(**service.settings.pg('postgres')); connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(database)))
        finally:
            connection.close()
        report['databases_removed'] = True
        settings.env_file.unlink(missing_ok=True)
        report['private_env_removed'] = True
        atomic_json(root/'fixture-report.json', report)
        print('Reference browser fixture sanitized report:', root/'fixture-report.json', flush=True)


if __name__ == '__main__':
    main()
