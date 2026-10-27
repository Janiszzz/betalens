"""Recompute incorrect scope labels without changing source or business data.

The original smoke report and every previous field proof are retained. Target
connections enforce read-only access. Only local task evidence and --output
are written, after validating the immutable commit manifest against live rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import json
import uuid

from betalens_db_manager.comparison import row_hash
from betalens_db_manager.config import Settings
from betalens_db_manager.field_lineage import record_readback
from betalens_db_manager.private import atomic_private
from betalens_db_manager.publishing import existing, operation_lock
from betalens_db_manager.security import public_data
from betalens_db_manager.state import State
from betalens_db_manager.util import digest, dumps, now
from production_verify_readback import (ReadOnlyRegistry, ReadOnlyPostgres,
                                      ReadOnlyClickHouse, manifest_expectations,
                                      proof_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('使用新的输出路径，禁止覆盖原审计修复记录')
    original = json.loads(args.report.read_text())
    if not original.get('finished_at') or any(c.get('status') != 'passed' for c in original['cases']):
        raise ValueError('只允许复核已完成的生产小样本任务')
    settings = Settings.from_env(args.env)
    if (settings.pg_database, settings.pg_schema, settings.ch_database) != ('betalens',)*3:
        raise ValueError('修复范围仅限已批准的 betalens 生产样本')
    service = SimpleNamespace(settings=settings, state=State(settings.state_dir),
                              registry=ReadOnlyRegistry(settings.config_dir))
    adapters = {'postgres': ReadOnlyPostgres(settings), 'clickhouse': ReadOnlyClickHouse(settings)}
    result = {'revision_id': uuid.uuid4().hex, 'started_at': now(),
              'reason': '修复无 asset_type 物理列时的字段生产范围判定；实际数据不变',
              'original_report': str(args.report), 'business_data_written': False,
              'target_connections_read_only': True, 'cases': []}

    def save():
        atomic_private(args.output, dumps(public_data(result, settings)))

    save()
    with operation_lock(service):
        with service.state.connect() as conn:
            if conn.execute("SELECT count(*) FROM tasks WHERE status IN ('queued','running')").fetchone()[0]:
                raise ValueError('有正在执行的数据任务，停止证据修复')
            if conn.execute('SELECT count(*) FROM schedules WHERE enabled=1').fetchone()[0]:
                raise ValueError('自动更新必须保持停止')
        for case in original['cases']:
            task = service.state.job(case['task_id'])
            spec = service.registry.spec(case['table'])
            if task['plan']['canonical_contracts'].get(spec.name) != spec.hash:
                raise ValueError('原任务结构版本与当前表定义不一致')
            before = service.state.ledger_rows('field_verification', task['id'])
            entry = {'task_id': task['id'], 'table': spec.name, 'label': case['label'],
                     'before': before, 'targets': []}
            result['cases'].append(entry)
            # Durable backup precedes every update, including an interrupted run.
            for proof in before:
                service.state.ledger_put('field_verification_revision',
                    digest([result['revision_id'], task['id'], proof['target'], spec.name, proof['column']]),
                    {'task_id': task['id'], 'revision_id': result['revision_id'],
                     'original': proof, 'created_at': now(), 'reason': result['reason']})
            save()
            for target in task['plan']['request']['targets']:
                manifest = service.state.table_commit(task['id'], target, spec.name)
                if manifest is None:
                    raise ValueError('缺少固定提交清单')
                expected = manifest_expectations(spec, manifest)
                live = existing(adapters[target], spec, [value['key'] for value in expected.values()])
                if set(live) != set(expected) or any(row_hash(spec, live[key]) != value['expected_hash']
                                                   for key, value in expected.items()):
                    raise ValueError('目标数据已被后续任务更改，不得为旧任务重写核验证据：'+task['id'])
                record_readback(service, task, target, spec, manifest['comparison'], live)
                entry['targets'].append({'target': target, 'rows': len(live), 'manifest_matched': True})
            after = service.state.ledger_rows('field_verification', task['id'])
            entry['after'] = after
            check = proof_summary(after, spec, task['plan']['request']['targets'])
            if not check['complete']:
                raise ValueError('重算后的字段证据不完整或存在不一致：'+task['id'])
            entry['status'] = 'corrected'
            service.state.log(task['id'], 'field_verification_scope_corrected',
                              revision_id=result['revision_id'], reason=result['reason'],
                              prior_proofs_preserved=True, business_data_written=False)
            save()
            print(dumps({'label': case['label'], 'status': entry['status']}), flush=True)
        result['finished_at'] = now()
        result['status'] = 'completed'
        service.state.ledger_put('field_verification_repair', result['revision_id'],
            {k: v for k, v in result.items() if k != 'cases'} | {'tasks': [c['task_id'] for c in result['cases']]})
        save()
    print(dumps({'status': result['status'], 'cases': len(result['cases']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
