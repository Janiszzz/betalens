"""Read-only canonical coverage and PG/CH parity report for this migration."""
import argparse
import hashlib
from pathlib import Path

from betalens_db_manager.canonical import load_canonical_catalog
from betalens_db_manager.comparison import row_hash, row_key
from betalens_db_manager.config import Settings
from betalens_db_manager.publishing import operation_lock
from betalens_db_manager.service import TushareImportService
from betalens_db_manager.storage import CONTROL_DDL, control_spec
from betalens_db_manager.util import atomic_json, now


def scan(target, spec):
    keys, fields = [], []
    count, first, last, after = 0, None, None, None
    nulls = {field['name']: 0 for field in spec.fields}
    while True:
        page = target.query(spec.name, limit=1000, after=after)
        for row in page['rows']:
            count += 1
            key = row_key(spec, row)
            keys.append(key)
            fields.append((key, row_hash(spec, row)))
            event = row.get('event_date')
            if event:
                event = str(event)
                first = min(first, event) if first else event
                last = max(last, event) if last else event
            for name in nulls:
                nulls[name] += row.get(name) is None
        after = page['next']
        if not after:
            break
    # Database collations can order security codes differently. Hash sorted
    # canonical keys, independent of each engine's pagination order.
    return {'rows': count, 'first_event_date': first, 'last_event_date': last,
            'key_hash': hashlib.sha256(''.join(sorted(keys)).encode()).hexdigest(),
            'field_hash': hashlib.sha256(''.join(key+value for key,value in sorted(fields)).encode()).hexdigest(),
            'null_counts': nulls}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    service = TushareImportService(Settings.from_env(args.env))
    report = {'generated_at': now(), 'tables': [], 'schema': [], 'capabilities': [],
              'historical_batches': service.state.ledger_rows('initial_load_batch')}
    # Keep the two snapshots stable while the download worker can still fetch.
    with operation_lock(service):
        specs = {**load_canonical_catalog(), **{name: control_spec(name) for name in CONTROL_DDL}}
        with service.pg.connect(readonly=True) as conn, conn.cursor() as cur:
            cur.execute('SELECT table_name,column_name FROM information_schema.columns WHERE table_schema=%s', (service.settings.pg_schema,))
            pg_columns = {}
            for name, column in cur.fetchall():
                pg_columns.setdefault(name, set()).add(column)
        ch_columns = {}
        for row in service.ch._json_rows(service.ch.execute(f"SELECT table,name FROM system.columns WHERE database='{service.settings.ch_database}' FORMAT JSONEachRow")):
            ch_columns.setdefault(row['table'], set()).add(row['name'])
        for name, spec in specs.items():
            expected = {field['name'] for field in spec.fields}
            report['schema'].append({'table': name, 'column_count': len(expected),
                                     'postgres_exact_columns': pg_columns.get(name) == expected,
                                     'clickhouse_exact_columns': ch_columns.get(name) == expected})
        report['schema_status'] = service.schema_plan()
        for name, spec in load_canonical_catalog().items():
            pg, ch = scan(service.pg, spec), scan(service.ch, spec)
            same = pg == ch
            report['tables'].append({'table': name, 'zh_name': spec.zh_name,
                                     'status': 'matched' if same else 'different', 'postgres': pg, 'clickhouse': ch})
            print(name, pg['rows'], ch['rows'], 'matched' if same else 'DIFFERENT', flush=True)
        report['capabilities'] = service.state.ledger_rows('meta_source_capability')
        report['unfilled_tables'] = [row['table'] for row in report['tables'] if not row['postgres']['rows']]
        report['full_history_completed'] = False
        report['notes'] = [
            '这是一致性快照；样本和已完成批次通过，不代表 2004 年以来的全量历史已完成。',
            '基金/指数没有等价于股票停复牌的权威状态接口；已生成状态标记为派生。',
            'Tushare daily_basic 不提供每日 pcf；本地列保持空值。',
            'Wind 尚未配置 SDK 和字段合同，明确不可用。',
        ]
        atomic_json(Path(args.output), report)


if __name__ == '__main__':
    main()
