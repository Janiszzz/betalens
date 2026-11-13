"""Read current storage and pending writers, and write a reviewable cutover plan."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'betalens-db-manager' / 'src'))

from betalens_db_manager.config import Settings
from betalens_db_manager.storage import Postgres
from betalens_db_manager.clickhouse import ClickHouse
from betalens_db_manager.suspension_migration import retirement_sql
from betalens_db_manager.util import digest, now


def preview(settings):
    import psycopg2
    plan = {'captured_at': now(), 'target_fingerprint': settings.fingerprint,
            'mutates_database': False, 'sql': retirement_sql(settings), 'storage': {}}
    with psycopg2.connect(**settings.pg()) as connection:
        connection.set_session(readonly=True)
        with connection.cursor() as cursor:
            cursor.execute("SELECT relname,reltuples::bigint,pg_total_relation_size(oid) FROM pg_class "
                "WHERE relnamespace=(SELECT oid FROM pg_namespace WHERE nspname=%s) "
                "AND relname IN ('trade_status_daily','reference_scope','stock_suspension_intervals_versions')", (settings.pg_schema,))
            plan['storage']['postgres'] = [{'table': row[0], 'estimated_rows': row[1], 'bytes': row[2]} for row in cursor]
    ch = ClickHouse(settings)
    plan['storage']['clickhouse'] = ch._json_rows(ch.execute(
        "SELECT table,sum(rows) AS physical_rows,sum(bytes_on_disk) AS bytes FROM system.parts "
        "WHERE active AND database=" + ch._literal(settings.ch_database) +
        " AND table IN ('trade_status_daily','reference_scope','stock_suspension_intervals_versions') GROUP BY table FORMAT JSONEachRow"))
    plan['clickhouse_database_engine'] = ch.execute('SELECT engine FROM system.databases WHERE name=' + ch._literal(settings.ch_database) + ' FORMAT TSVRaw')
    path = settings.state_dir / 'tasks.sqlite3'
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
        plan['active_operations'] = [{'task_id': row[0], 'job_id': row[1], 'operation': row[2], 'status': row[3]}
            for row in connection.execute("SELECT t.id,t.job,t.operation,j.status FROM tasks t LEFT JOIN jobs j ON j.id=t.job WHERE t.status='running'")]
    plan['cutover_blocked'] = bool(plan['active_operations']) or plan['clickhouse_database_engine'] != 'Atomic'
    plan['token'] = digest(plan)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = preview(Settings.from_env(args.env))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    sql_path = args.output.with_suffix('.sql')
    sql_path.write_text('\n\n'.join('-- ' + target + ' / ' + phase + '\n' + ';\n'.join(statements) + ';'
                       for target in ('postgres', 'clickhouse')
                       for phase, statements in report['sql'][target].items()))
    print(json.dumps({'report': str(args.output), 'sql': str(sql_path), 'cutover_blocked': report['cutover_blocked'],
                      'active_operations': report['active_operations'], 'token': report['token']}, ensure_ascii=False))
