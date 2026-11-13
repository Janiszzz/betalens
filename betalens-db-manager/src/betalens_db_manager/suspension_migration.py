"""Read-only retirement preview. SQL retains the original table for rollback.

Run the stage and cutover only after stopping old writers and validating an
archive. Reference manifests are append-only and are never rewritten here.
"""
from .util import identifier

DERIVED_SOURCES = ('derived_from_daily_quote', 'security_lifecycle')


def retirement_sql(settings, suffix='sparse_v2'):
    identifier(suffix)
    schema = identifier(settings.pg_schema)
    database = identifier(settings.ch_database)
    candidate = 'trade_status_daily_' + suffix + '_candidate'
    archive = 'trade_status_daily_' + suffix + '_legacy'
    predicate = "status_source NOT IN ('derived_from_daily_quote','security_lifecycle')"
    pg = lambda table: '"' + schema + '"."' + table + '"'
    ch = lambda table: '`' + database + '`.`' + table + '`'
    return {
        'postgres': {
            'stage': [f'CREATE TABLE {pg(candidate)} (LIKE {pg("trade_status_daily")} INCLUDING ALL)',
                      f'INSERT INTO {pg(candidate)} SELECT * FROM {pg("trade_status_daily")} WHERE {predicate}'],
            'cutover': ['BEGIN', f'LOCK TABLE {pg("trade_status_daily")} IN ACCESS EXCLUSIVE MODE',
                        f'ALTER TABLE {pg("trade_status_daily")} RENAME TO "{archive}"',
                        f'ALTER TABLE {pg(candidate)} RENAME TO "trade_status_daily"', 'COMMIT'],
            'rollback': ['BEGIN', f'ALTER TABLE {pg("trade_status_daily")} RENAME TO "{candidate}"',
                         f'ALTER TABLE {pg(archive)} RENAME TO "trade_status_daily"', 'COMMIT'],
        },
        'clickhouse': {
            'stage': [f'CREATE TABLE {ch(candidate)} AS {ch("trade_status_daily")}',
                      f'INSERT INTO {ch(candidate)} SELECT * FROM {ch("trade_status_daily")} FINAL WHERE {predicate}'],
            'cutover': [f'EXCHANGE TABLES {ch("trade_status_daily")} AND {ch(candidate)}',
                        f'RENAME TABLE {ch(candidate)} TO {ch(archive)}'],
            'rollback': [f'EXCHANGE TABLES {ch("trade_status_daily")} AND {ch(archive)}'],
        },
        'archive_table': archive,
        'drops_old_data': False,
        'preconditions': [
            'No old task or worker may write trade_status_daily during staging and cutover.',
            'Deploy and verify sparse-v2 readers before cutover; old tasks must be re-audited to use sparse-v2.',
            'Compare retained observations and business-query results before cutover.',
            'ClickHouse must use an Atomic database; candidate and archive table names must not exist.',
            'Cutover retains the old table and does not reclaim its disk space.',
            'Physical reclamation requires a verified external archive and a separate explicit retirement.',
            'Reference revisions/scopes/publication events remain immutable; rebuild compact revisions from verified archives.',
        ],
    }
