"""Independently verify a production smoke report with read-only connections.

No manager is constructed: its initialization publishes local metadata. SQLite
is opened mode=ro, PG connections enforce readonly, and the CH adapter rejects
SQL other than SELECT/SHOW/DESCRIBE/EXISTS. Only --output is written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from urllib.parse import quote

from psycopg2 import sql

from betalens_db_manager.canonical import CanonicalTable
from betalens_db_manager.catalog import load_catalog
from betalens_db_manager.clickhouse import ClickHouse
from betalens_db_manager.comparison import content_hash, row_hash, row_key
from betalens_db_manager.config import Settings
from betalens_db_manager.private import atomic_private
from betalens_db_manager.publishing import existing
from betalens_db_manager.registry import GENERATED
from betalens_db_manager.safe_yaml import safe_load
from betalens_db_manager.schema_contract import compare_structure
from betalens_db_manager.schema_manager import SchemaManager
from betalens_db_manager.security import public_data, safe_error
from betalens_db_manager.source import read_archive
from betalens_db_manager.state import State
from betalens_db_manager.storage import Postgres, relation
from betalens_db_manager.util import dumps, now


class ReadOnlyState(State):
    def __init__(self, root):
        self.root = root
        self.path = root/'tasks.sqlite3'
        if not self.path.is_file():
            raise ValueError('任务状态数据库不存在')

    def connect(self):
        connection = sqlite3.connect('file:'+quote(str(self.path.resolve()))+'?mode=ro', uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA query_only=ON')
        return connection


class ReadOnlyPostgres(Postgres):
    def connect(self, readonly=True, database=None):
        return super().connect(readonly=True, database=database)


class ReadOnlyClickHouse(ClickHouse):
    def __init__(self, settings):
        # Avoid constructing a writer coordinator (it creates/chmods lock dirs).
        super().__init__(replace(settings, ch_writer_mode='disabled'))

    def execute(self, query, data=None, timeout=180):
        first = query.lstrip().split(None, 1)[0].upper() if query.strip() else ''
        if data is not None or first not in {'SELECT', 'SHOW', 'DESCRIBE', 'DESC', 'EXISTS', 'EXPLAIN'}:
            raise ValueError('独立核验连接禁止执行写语句')
        return super().execute(query, data=None, timeout=timeout)


class ReadOnlyRegistry:
    def __init__(self, config_dir):
        self.definitions = {}
        for path in sorted((config_dir/'tables').glob('*.yaml')):
            value = safe_load(path.read_text())
            if value['name'] != path.stem:
                raise ValueError('表定义文件名与表名不一致：'+path.stem)
            self.definitions[path.stem] = value

    def spec(self, name):
        value = self.definitions[name]
        return CanonicalTable(value['name'], value['zh_name'], value['description'], value['grain'],
                              tuple(value['fields']), tuple(value['keys']), value.get('domain', 'custom'))


def manifest_expectations(spec, manifest):
    """Return every checked key's expected value, rejecting incomplete intents."""
    comparison = manifest['comparison']
    changes = comparison.get('changes', [])
    checks = comparison.get('checks', [])
    change_keys = [change['row_key'] for change in changes]
    check_keys = [check['row_key'] for check in checks]
    if len(change_keys) != len(set(change_keys)) or len(check_keys) != len(set(check_keys)):
        raise ValueError('固定提交清单包含重复业务键')
    if not set(change_keys) <= set(check_keys):
        raise ValueError('变更业务键不在固定检查清单中')
    by_key = {change['row_key']: change for change in changes}
    expected = {}
    for check in checks:
        key = check['row_key']
        if row_key(spec, check['key']) != key:
            raise ValueError('提交清单业务键哈希错误')
        change = by_key.get(key)
        if change:
            if row_hash(spec, change['after']) != change['after_hash']:
                raise ValueError('提交清单 after 内容与哈希不一致')
            if change['before_hash'] != check['before_hash']:
                raise ValueError('提交清单 before 检查不一致')
            expected[key] = {'key': check['key'], 'expected_hash': change['after_hash'], 'operation':
                             'insert' if change.get('before') is None else 'update'}
        else:
            if not check.get('before_hash'):
                raise ValueError('跳过记录缺少原值哈希，不能证明已存在')
            expected[key] = {'key': check['key'], 'expected_hash': check['before_hash'], 'operation': 'skip'}
    counts = comparison['counts']
    actual_counts = Counter(value['operation'] for value in expected.values())
    if (counts['inserted'] != actual_counts['insert'] or counts['updated'] != actual_counts['update']
            or counts['skipped'] != actual_counts['skip']):
        raise ValueError('固定清单业务键数与新增/更新/跳过计数不一致')
    return expected


def proof_summary(proofs, spec, targets):
    identities = [(p.get('target'), p.get('table'), p.get('column')) for p in proofs]
    counts = Counter(identities)
    expected = {(target, spec.name, field['name']) for target in targets for field in spec.fields}
    missing = sorted(expected-set(counts))
    duplicate = [list(key) for key, count in counts.items() if count > 1]
    mismatch = [list(key) for key, proof in zip(identities, proofs)
                if proof.get('status') == 'mismatch' or proof.get('mismatch_count', 0)]
    unexpected = sorted(set(counts)-expected)
    invalid = []
    for proof in proofs:
        verified, written = proof.get('verified_count', 0), proof.get('written_count', 0)
        applicable, preserved = proof.get('applicable_count', 0), proof.get('preserved_count', 0)
        if (verified > written or written+preserved > applicable
                or any(value < 0 for name, value in proof.items() if name.endswith('_count') and isinstance(value, int))):
            invalid.append([proof.get('target'), proof.get('column')])
    totals = []
    for target in targets:
        selected = [p for p in proofs if p.get('target') == target and p.get('column') not in GENERATED
                    and not next((f.get('system') for f in spec.fields if f['name'] == p.get('column')), False)]
        metrics = Counter()
        for proof in selected:
            for key, value in proof.items():
                if key.endswith('_count') and isinstance(value, int):
                    metrics[key] += value
        totals.append({'target': target, 'business_columns': len(selected),
                       'columns_with_new_source_values_verified': sum(p.get('verified_count', 0) > 0 for p in selected),
                       'columns_with_preserved_values': sum(p.get('preserved_count', 0) > 0 for p in selected),
                       'columns_without_new_source_proof': sum(not p.get('verified_count', 0) for p in selected),
                       'metrics': dict(metrics), 'status_counts': dict(Counter(p['status'] for p in selected))})
    return {'complete': not (missing or duplicate or mismatch or unexpected or invalid),
            'expected_field_target_pairs': len(expected), 'actual_records': len(proofs),
            'missing': [list(key) for key in missing], 'duplicates': duplicate, 'mismatches': mismatch,
            'unexpected': [list(key) for key in unexpected], 'invalid_counters': invalid, 'business_totals': totals,
            'columns': proofs}


def source_summary(item, proofs):
    """Separate source presence from selected writes and preserved old values."""
    sources = defaultdict(lambda: {'rows': 0, 'nonnull': Counter(), 'null': Counter(), 'statuses': set(), 'returned': set()})
    for part in item['slices']:
        endpoint = part['dataset']
        sources[endpoint]['statuses'].add(part['status'])
        if not part.get('archive') or part['status'] not in {'downloaded', 'paged'}:
            continue
        data = read_archive(Path(part['archive']), part['sha'])
        source = sources[endpoint]
        source['returned'].update(data['fields'])
        source['rows'] += len(data['items'])
        for values in data['items']:
            if not isinstance(values, list) or len(values) != len(data['fields']):
                raise ValueError(endpoint+': 归档字段/行长度不一致')
            for name, value in zip(data['fields'], values):
                source['null' if value in (None, '') else 'nonnull'][name] += 1
    mappings = item['plan'].get('mapping_snapshots', {})
    output = []
    for endpoint, source in sorted(sources.items()):
        for key, contract in item['plan'].get('source_contracts', {}).items():
            if key != 'tushare:'+endpoint:
                continue
            field_results = []
            for field in contract['fields']:
                name = field['name'];origin = 'tushare:'+endpoint+':'+name
                destinations = {(table, rule['column']) for table, mapping in mappings.items()
                    for binding in mapping.get('bindings', []) if binding['source_id'] == 'tushare' and binding['endpoint'] == endpoint
                    for rule in binding.get('fields', []) if rule['source_field'] == name}
                destinations.update((p['table'], p['column']) for p in proofs if p.get('verified_origins', {}).get(origin))
                target_evidence = []
                for target in item['plan']['request']['targets']:
                    selected = [p for p in proofs if p['target'] == target and (p['table'], p['column']) in destinations]
                    verified = sum(p.get('verified_origins', {}).get(origin, 0) for p in selected)
                    preserved = sum(p.get('preserved_count', 0) for p in selected)
                    status = ('verified_new_source_values' if verified else
                              'source_null' if source['null'][name] and not source['nonnull'][name] else
                              'not_projected_by_this_task' if not destinations else
                              'existing_values_retained' if preserved else 'no_new_source_proof')
                    target_evidence.append({'target': target, 'status': status,
                        'verified_origin_values': verified, 'preserved_destination_values': preserved})
                field_results.append({'source_field': name, 'returned': name in source['returned'],
                    'source_nonnull_count': source['nonnull'][name], 'source_null_count': source['null'][name],
                    'destinations': [{'table': table, 'column': column} for table, column in sorted(destinations)],
                    'targets': target_evidence})
            output.append({'endpoint': endpoint, 'rows': source['rows'], 'slice_statuses': sorted(source['statuses']),
                           'fields': field_results})
    return output


def read_count(service, target, name):
    adapter = service.pg if target == 'postgres' else service.ch
    if target == 'postgres':
        with adapter.connect() as connection, connection.cursor() as cursor:
            cursor.execute(sql.SQL('SELECT count(*) FROM {}').format(relation(adapter.settings, name)))
            return int(cursor.fetchone()[0])
    return int(adapter.execute(f'SELECT count() FROM {adapter.table(name)} FINAL'))


def table_case_coverage(planned_cases, verified_cases):
    """All planned scopes must pass; one successful asset cannot mask another."""
    expected = defaultdict(list)
    for case in planned_cases:
        for name in case['request']['tables']:
            expected[name].append(case['label'])
    by_label = defaultdict(list)
    for case in verified_cases:
        by_label[case['label']].append(case)
    output = []
    for name, labels in sorted(expected.items()):
        missing = [label for label in labels if label not in by_label]
        failed = [label for label in labels if label in by_label
                  and (len(by_label[label]) != 1 or by_label[label][0]['status'] != 'verified')]
        output.append({'table': name, 'planned_cases': labels, 'missing_cases': missing,
                       'not_verified_cases': failed, 'status': 'verified' if not missing and not failed else 'not_verified'})
    return output


def verify(service, smoke):
    report = {'started_at': now(), 'read_only': True, 'cases': [], 'tables': [], 'issues': [],
              'business_writes_performed': False, 'metadata_or_task_writes_performed': False}
    final = defaultdict(dict);net_insertions = Counter();baselines = {};case_rows = []
    actual_endpoints = defaultdict(list);attempted = set()
    for index, case in enumerate(smoke.get('cases', [])):
        result = {'label': case['label'], 'reported_status': case.get('status'), 'task_id': case.get('task_id'),
                  'table': case.get('table'), 'issues': [], 'targets': []}
        report['cases'].append(result)
        try:
            if not case.get('task_id'):
                raise ValueError('案例没有真实任务ID')
            item = service.state.job(case['task_id'])
            if item['target'] != service.settings.fingerprint:
                raise ValueError('任务目标与当前数据库配置不一致')
            request = item['plan']['request'];result['actual_request'] = request
            for role in ('tables', 'targets', 'codes', 'start_date', 'end_date', 'mode', 'endpoints'):
                if case.get('request', {}).get(role) != request.get(role):
                    result['issues'].append('reported_request_differs_from_actual:'+role)
            result['job_status'] = item['status'];result['data_status'] = item['data_status']
            result['audit_sync_status'] = item['audit_sync_status']
            for part in item['slices']:
                attempted.add(part['dataset'])
                if part.get('archive') and part['status'] in {'downloaded', 'paged'} and (part.get('rows') or 0) > 0:
                    actual_endpoints[part['dataset']].append(case['label'])
            name = result['table'];spec = service.registry.spec(name)
            if item['plan'].get('canonical_contracts', {}).get(name) != spec.hash:
                raise ValueError('任务结构版本与当前结构定义不一致，不能混用哈希')
            completed_targets = []
            for target in request['targets']:
                manifest = service.state.table_commit(item['id'], target, name)
                if manifest is None or manifest.get('status') != 'completed':
                    result['issues'].append(target+':missing_completed_manifest')
                    continue
                expectations = manifest_expectations(spec, manifest)
                if not expectations:
                    result['issues'].append(target+':empty_manifest')
                for key, value in expectations.items():
                    final[target, name][key] = {**value, 'last_task_id': item['id'], 'last_case': case['label']}
                counts = manifest['comparison']['counts']
                net_insertions[target, name] += counts['inserted']
                recorded = next((r for r in case.get('readback', []) if r['target'] == target), None)
                if (target, name) not in baselines:
                    baselines[target, name] = recorded['before_rows'] if recorded else None
                result['targets'].append({'target': target, 'manifest_status': manifest['status'],
                    'checked_keys': len(expectations), 'counts': counts,
                    'written_rows': counts['inserted']+counts['updated'], 'retained_rows': counts['skipped']})
                completed_targets.append(target)
            proofs = service.state.ledger_rows('field_verification', item['id'])
            result['field_proofs'] = proof_summary(proofs, spec, request['targets'])
            if not result['field_proofs']['complete']:
                result['issues'].append('missing_duplicate_or_invalid_field_proofs')
            result['source_fields'] = source_summary(item, proofs)
            if item['status'] != 'completed' or item['data_status'] != 'completed' or item['audit_sync_status'] != 'synced':
                result['issues'].append('job_or_audit_not_complete')
            if case.get('status') != 'passed':
                result['issues'].append('smoke_case_not_passed')
            result['status'] = 'verified' if not result['issues'] else 'not_verified'
        except Exception as exc:
            result['status'] = 'not_verified';result['issues'].append(service.safe_error(exc))
    live_tables = {}
    for name in sorted(service.registry.definitions):
        spec = service.registry.spec(name)
        for target in ('postgres', 'clickhouse'):
            result = {'table': name, 'target': target, 'issues': []}
            report['tables'].append(result)
            try:
                physical = service.schema.inspect(name, target)
                structure = compare_structure(spec, physical, target)
                result['schema_ready'] = structure['ready']
                if not structure['ready']:
                    result['issues'].append('schema_not_ready')
                    result['schema_difference'] = structure
                    continue
                expectations = final.get((target, name), {})
                keys = list(expectations)
                adapter = service.pg if target == 'postgres' else service.ch
                live = {}
                for offset in range(0, len(keys), 500):
                    live.update(existing(adapter, spec, [expectations[key]['key'] for key in keys[offset:offset+500]]))
                missing = sorted(set(keys)-set(live))
                extra = sorted(set(live)-set(keys))
                mismatches = [dict(row_key=key, last_task_id=expectations[key]['last_task_id'],
                    last_case=expectations[key]['last_case'], expected_hash=expectations[key]['expected_hash'],
                    actual_hash=row_hash(spec, live[key])) for key in keys if key in live
                    and row_hash(spec, live[key]) != expectations[key]['expected_hash']]
                result.update(expected_keys=len(keys), readback_keys=len(live), missing_keys=missing,
                              unexpected_keys=extra, hash_mismatches=mismatches)
                if missing or extra or mismatches:
                    result['issues'].append('final_key_or_field_hash_mismatch')
                if not keys:
                    result['issues'].append('no_committed_sample')
                count = read_count(service, target, name)
                baseline = baselines.get((target, name))
                expected_count = baseline+net_insertions[target, name] if baseline is not None else None
                result.update(current_rows=count, baseline_rows=baseline, net_inserted_rows=net_insertions[target, name],
                              expected_rows=expected_count)
                if expected_count is None:
                    result['issues'].append('missing_initial_count_baseline')
                elif count != expected_count:
                    result['issues'].append('row_count_net_change_mismatch')
                live_tables[target, name] = live
            except Exception as exc:
                result['issues'].append(service.safe_error(exc))
            finally:
                result['status'] = 'verified' if not result['issues'] else 'not_verified'
    report['cross_target'] = []
    for name in sorted(service.registry.definitions):
        pg, ch = live_tables.get(('postgres', name), {}), live_tables.get(('clickhouse', name), {})
        same_keys = bool(pg) and set(pg) == set(ch)
        mismatch = [key for key in set(pg)&set(ch) if content_hash(service.registry.spec(name), pg[key]) != content_hash(service.registry.spec(name), ch[key])]
        report['cross_target'].append({'table': name, 'same_keys': same_keys, 'business_value_mismatch_keys': mismatch,
                                       'status': 'verified' if same_keys and not mismatch else 'not_verified'})
    expected_labels = {case['label'] for case in smoke.get('planned_cases', [])}
    actual_labels = {case['label'] for case in smoke.get('cases', [])}
    report['missing_cases'] = sorted(expected_labels-actual_labels)
    report['table_case_coverage'] = table_case_coverage(smoke.get('planned_cases', []), report['cases'])
    duplicated_labels = [label for label, count in Counter(c['label'] for c in report['cases']).items() if count > 1]
    if duplicated_labels:
        report['issues'].append({'duplicate_case_labels': duplicated_labels})
    report['endpoint_coverage'] = [{'endpoint': endpoint,
        'status': 'nonempty_archived_response' if endpoint in actual_endpoints else 'attempted_without_nonempty_sample' if endpoint in attempted else 'not_run',
        'cases': sorted(set(actual_endpoints.get(endpoint, [])))} for endpoint in sorted(load_catalog())]
    report['untested_vip_endpoints'] = [name for name in load_catalog() if name.endswith('_vip') and name not in attempted]
    report['summary'] = {'cases': len(report['cases']), 'case_statuses': dict(Counter(c['status'] for c in report['cases'])),
        'logical_tables': len(service.registry.definitions), 'target_table_statuses': dict(Counter(t['status'] for t in report['tables'])),
        'registered_endpoints': len(load_catalog()), 'attempted_endpoints': len(attempted),
        'endpoints_with_nonempty_archives': len(actual_endpoints)}
    report['notes'] = [
        'verified_stored proves at least one newly written, selected, non-null source value for that column; inspect counts for partial coverage.',
        'preserved_count/table_verified proves retained target values only; it does not prove new source values were written.',
        'Source NULL, not applicable, unselected projection and unknown source provenance remain distinct.',
        'Initial no-write evidence is limited to snapshots actually recorded by the smoke run; it cannot be reconstructed for earlier download stages.',
        'Final hashes follow the last completed manifest per target/table/key, so later cases may legitimately replace earlier task provenance.',
        'Row-count checks cover business tables only; metadata and schema operation records are outside this count.',
    ]
    report['status'] = 'verified' if (report['cases'] and not report['missing_cases'] and not report['issues']
        and all(c['status'] == 'verified' for c in report['cases'])
        and all(t['status'] == 'verified' for t in report['table_case_coverage'])
        and all(t['status'] == 'verified' for t in report['tables'])
        and all(t['status'] == 'verified' for t in report['cross_target'])) else 'not_fully_verified'
    report['finished_at'] = now()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() or output in {args.env.resolve(), args.report.resolve()}:
        parser.error('--output must be a new path distinct from input files')
    settings = Settings.from_env(args.env)
    registry = ReadOnlyRegistry(settings.config_dir)
    service = SimpleNamespace(settings=settings, registry=registry, state=ReadOnlyState(settings.state_dir),
        pg=ReadOnlyPostgres(settings), ch=ReadOnlyClickHouse(settings), safe_error=lambda exc: safe_error(exc, settings))
    service.schema = SchemaManager(service)
    report = verify(service, json.loads(args.report.read_text()))
    atomic_private(output, dumps(public_data(report, settings)))
    print(dumps({'report': str(output), 'status': report['status'], 'summary': report['summary'],
                 'untested_vip_endpoints': report['untested_vip_endpoints']}), flush=True)
    if report['status'] != 'verified':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
