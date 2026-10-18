"""Exact, auditable wide-market import into PostgreSQL and ClickHouse.

Explicit source contract and staging only; no automatic schema migrations.
Run via ``python -m betalens_db_manager.wide_market --help``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, localcontext
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import sql
import requests

from .adapters.files import read_file, apply_time_alignment
from .registry import CORE_MARKET_METRICS

SOURCES = {
    'etf': ('ETF非货币260819', 'etf_daily', 'hfq'),
    'stock_raw': ('全A不复权20260902', 'stock_daily_unadjusted', 'none'),
    'stock_hfq': ('全A后复权截止20260723', 'stock_daily_hfq', 'hfq'),
    'stock_delisted_hfq': ('全A摘牌', 'stock_daily_hfq', 'hfq'),
    'sw_index': ('申万一级行业指数260818', 'index_daily', 'index'),
    'csi_index': ('主要中证指数260819', 'index_daily', 'index'),
}
EXPECTED = {'etf_daily': 1405439, 'stock_daily_unadjusted': 18613980,
            'stock_daily_hfq': 18453249, 'index_daily': 246114}
SENTINEL = Decimal('3.40282346638529E38')
NULL = '\\N'
SEP = '\x1f'


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def log(event, **kwargs):
    print(json.dumps({'event': event, **kwargs}, ensure_ascii=False), flush=True)


def b_share(code):
    return bool(re.fullmatch(r'(?:200\d{3}\.SZ|900\d{3}\.SH)', code))


def decimal4(values):
    """Normalize decimal text without ever passing through float64."""
    result = values.copy()
    nonempty = result.ne('')
    ordinary = result.str.fullmatch(r'-?\d+(?:\.\d{1,4})?')
    for idx in result[nonempty & ~ordinary].index:
        with localcontext() as ctx:
            ctx.prec = 80
            value = Decimal(result.at[idx])
            if not value.is_finite() or value != value.quantize(Decimal('.0001')):
                raise ValueError(f'Value cannot be represented exactly at scale 4: {result.at[idx]}')
            result.at[idx] = format(value, '.4f')
    populated = result.ne('')
    if not populated.any():
        return result.replace('', NULL)
    parts = result[populated].str.partition('.')
    whole = parts[0].str.replace(r'^(-?)0+(?=\d)', r'\1', regex=True)
    frac = parts[2].str.pad(4, side='right', fillchar='0')
    result.loc[populated] = whole + '.' + frac
    result = result.replace({'': NULL, '-0.0000': '0.0000'})
    return result


def normalize(frame, columns, file_id):
    """Return a wide frame and original-value audit; never melt market rows."""
    frame = frame.copy()
    empty_cols = [c for c in frame if c.startswith('Unnamed:')]
    if any(frame[c].ne('').any() for c in empty_cols):
        raise ValueError('Nonempty unnamed column')
    frame = frame.drop(columns=empty_cols)
    if list(frame.columns) != columns:
        raise ValueError(f'Unexpected source columns: {list(frame.columns)}')
    if frame[['代码', '日期']].duplicated().any():
        raise ValueError('Duplicate code/date')
    for col in columns:
        if frame[col].str.contains(r'[\t\r\n\\\x1f]', regex=True).any():
            raise ValueError(f'Unsafe TSV/canonical separator in {col}')
    source_rows = pd.Series(range(1, len(frame)+1), index=frame.index).astype(str)
    audit = []
    suspect = pd.Series(False, index=frame.index)
    if '市盈率' in frame:
        # This precise vendor token is the only value the user approved replacing.
        suspect = frame['市盈率'].eq('3.40282346638529E38')
        for idx in frame.index[suspect]:
            audit.append({'source_file_id': file_id, 'source_row': int(source_rows.at[idx]),
                          'code': frame.at[idx, '代码'], 'date': frame.at[idx, '日期'],
                          'column': '市盈率', 'original_value': frame.at[idx, '市盈率'],
                          'action': 'null_vendor_sentinel'})
        frame.loc[suspect, '市盈率'] = ''
    for col in columns[3:]:
        frame[col] = decimal4(frame[col])
        if frame[col].ne(NULL).any():
            if frame.loc[frame[col].ne(NULL), col].str.replace('-', '', regex=False).str.split('.').str[0].str.len().max() > 16:
                raise ValueError(f'Numeric(20,4) overflow: {col}')
    # Call the existing helper only on the two timing groups, preserving wide shape.
    open_names = [m for m, spec in CORE_MARKET_METRICS.items() if spec.available_time == '09:30:01']
    dates = pd.DataFrame({'日期': frame['日期'], 'metric': '开盘价(元)'})
    frame['open_time'] = apply_time_alignment(dates, open_metric_names=open_names)['日期'].dt.strftime('%Y-%m-%d %H:%M:%S')
    dates['metric'] = '收盘价(元)'
    frame['close_time'] = apply_time_alignment(dates, open_metric_names=open_names)['日期'].dt.strftime('%Y-%m-%d %H:%M:%S')
    frame['source_file_id'] = file_id
    frame['source_row'] = source_rows
    if '市盈率' in frame:
        frame['pe_suspect'] = suspect.astype(int).astype(str)
    frame = frame.sort_values(['代码', '日期'], kind='stable')
    return frame, audit


def prepare_file(task):
    record, source_root, run_dir, contract = task
    logging.disable(logging.CRITICAL)
    source = record['source']
    folder, table, adjustment = SOURCES[source]
    path = Path(source_root) / folder / record['file']
    payload = path.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != record['sha256']:
        raise ValueError(f'Source changed after audit: {path}')
    file_id = hashlib.sha256((source + '/' + record['file']).encode()).hexdigest()[:32]
    stage_dir = Path(run_dir) / 'prepared' / table
    stage_dir.mkdir(parents=True, exist_ok=True)
    info = stage_dir / (file_id + '.json')
    if info.exists():
        prior = json.loads(info.read_text())
        if prior['sha256'] != checksum:
            raise ValueError('Prepared source checksum mismatch')
        return prior
    code = path.stem
    disposition = 'excluded_b_share' if b_share(code) else 'superseded_same_values' if source == 'stock_hfq' and code == '920305.BJ' else 'accepted'
    base = {'source_file_id': file_id, 'source': source, 'table': table, 'path': str(path),
            'sha256': checksum, 'encoding': 'gb18030', 'bytes': len(payload), 'source_rows': record['rows'],
            'code': code, 'adjustment': adjustment, 'disposition': disposition, 'accepted_rows': 0,
            'pe_nulls': 0, 'null_counts': {}, 'first_date': record['min_date'], 'last_date': record['max_date']}
    if disposition != 'accepted':
        save(info, base)
        return base
    frame = read_file(path, encoding='gb18030', dtype=str, keep_default_na=False)
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError('Source changed while being read')
    if len(frame) != record['rows'] or not frame['代码'].eq(code).all():
        raise ValueError('Source identity or row count changed')
    columns = [c['name'] for c in contract['tables'][table]['columns']]
    frame, changes = normalize(frame, columns, file_id)
    tsv = stage_dir / (file_id + '.tsv')
    frame.to_csv(tsv, sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE, lineterminator='\n')
    digest_path = stage_dir / (file_id + '.hashes')
    with digest_path.open('wb') as output:
        for row in frame.itertuples(index=False, name=None):
            output.write(hashlib.md5(SEP.join(row).encode()).hexdigest().encode() + b'\n')
    if changes:
        with (stage_dir / (file_id + '.audit.jsonl')).open('w', encoding='utf-8') as output:
            for change in changes:
                output.write(json.dumps(change, ensure_ascii=False) + '\n')
    base.update(accepted_rows=len(frame), pe_nulls=len(changes), tsv=str(tsv), hashes=str(digest_path),
                tsv_sha256=hashlib.sha256(tsv.read_bytes()).hexdigest(),
                null_counts={c: int(frame[c].eq(NULL).sum()) for c in columns[3:]})
    save(info, base)
    return base


class Importer:
    def __init__(self, run_dir, contract_path, env_path):
        self.root = Path(run_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = json.loads(Path(contract_path).read_text())
        self.tag = 'wide_20261015'
        self.schema = '_load_' + self.tag
        secrets = dict(line.strip().split('=', 1) for line in Path(env_path).read_text().splitlines() if '=' in line and not line.startswith('#'))
        self.pg_config = {'host': '127.0.0.1', 'dbname': 'datafeed', 'user': secrets['POSTGRES_USER'], 'password': secrets['POSTGRES_PASSWORD']}
        self.http = requests.Session()
        self.http.auth = (secrets['CLICKHOUSE_USER'], secrets['CLICKHOUSE_PASSWORD'])
        self.ch_database = secrets['CLICKHOUSE_DB']
        self.state_path = self.root / 'state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {'status': 'new', 'pg_batches': [], 'ch_batches': []}
        contract_sha = hashlib.sha256(json.dumps(self.contract,sort_keys=True).encode()).hexdigest()
        if self.state.get('contract_sha256',contract_sha) != contract_sha:
            raise ValueError('The run contract changed; use a new run directory')
        self.state['contract_sha256'] = contract_sha
        self.pg = None

    def connect(self):
        self.pg = psycopg2.connect(**self.pg_config)
        with self.pg, self.pg.cursor() as cur:
            cur.execute("SET TIME ZONE 'Asia/Shanghai'")

    def ch(self, query, data=None, stream=False):
        resp = self.http.post('http://127.0.0.1:8123/', params={'query': query}, data=data, timeout=(10, 3600), stream=stream)
        if resp.status_code != 200:
            raise RuntimeError(resp.text[:2000])
        return resp if stream else resp.text

    def cht(self, table, final=False):
        return f'`{self.ch_database}`.`{table if final else self.schema + "_" + table}`'

    def pgt(self, table, final=False):
        return f'"{"market" if final else self.schema}"."{table}"'

    def checkpoint(self):
        save(self.state_path, self.state)

    def prepare(self, audit_path, source_root):
        records = [json.loads(line) for line in Path(audit_path).read_text().splitlines()]
        tasks = [(r, source_root, str(self.root), self.contract) for r in records]
        manifest = []
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i, result in enumerate(pool.map(prepare_file, tasks, chunksize=8), 1):
                manifest.append(result)
                if i % 300 == 0 or i == len(tasks):
                    log('prepare', files=i, total=len(tasks))
        save(self.root / 'manifest.json', manifest)
        aggregate = {}
        for table, expected in EXPECTED.items():
            files = sorted((r for r in manifest if r['table'] == table and r['disposition'] == 'accepted'), key=lambda r:r['code'])
            if len({r['code'] for r in files}) != len(files):
                raise ValueError(f'Unresolved duplicate code in {table}')
            total = sum(r['accepted_rows'] for r in files)
            if total != expected:
                raise ValueError(f'Unexpected rows for {table}: {total} vs {expected}')
            digest = hashlib.sha256()
            for record in files:
                with Path(record['hashes']).open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024*1024), b''):
                        digest.update(chunk)
            aggregate[table] = {'rows': total, 'codes': len(files), 'sha256_of_ordered_row_md5': digest.hexdigest(),
                'pe_nulls': sum(r['pe_nulls'] for r in files),
                'null_counts': {c['name']: sum(r['null_counts'].get(c['name'],0) for r in files) for c in self.contract['tables'][table]['columns'][3:]}}
        save(self.root / 'expected.json', aggregate)
        self.state['status'] = 'prepared'
        self.checkpoint()
        log('prepared', summary=aggregate)

    def columns(self, table):
        spec = self.contract['tables'][table]
        return [c['name'] for c in spec['columns'] + spec['added_columns']]

    def create_stage(self):
        self.connect()
        with self.pg, self.pg.cursor() as cur:
            cur.execute('SELECT to_regnamespace(%s)', (self.schema,))
            exists = cur.fetchone()[0] is not None
            if not exists:
                cur.execute(f'CREATE SCHEMA "{self.schema}"')
            for table, spec in self.contract['tables'].items():
                columns = []
                ch_columns = []
                for col in spec['columns'] + spec['added_columns']:
                    name = col['name']
                    pg_type, ch_type = col['postgres_type'], col['clickhouse_type']
                    columns.append(f'"{name}" {pg_type}')
                    ch_columns.append(f'`{name}` {ch_type}')
                columns.append('PRIMARY KEY ("代码", "日期")')
                cur.execute(f'CREATE TABLE IF NOT EXISTS {self.pgt(table)} ({", ".join(columns)}) PARTITION BY RANGE ("日期")')
                for year in range(1990, 2027):
                    cur.execute(f'CREATE TABLE IF NOT EXISTS {self.pgt(table + "_" + str(year))} PARTITION OF {self.pgt(table)} FOR VALUES FROM (\'{year}-01-01\') TO (\'{year+1}-01-01\')')
                self.ch(f'CREATE TABLE IF NOT EXISTS {self.cht(table)} ({", ".join(ch_columns)}) ENGINE=MergeTree PARTITION BY toYear(`日期`) ORDER BY (`代码`, `日期`)')
        log('staging_created')

    def batches(self):
        manifest = json.loads((self.root / 'manifest.json').read_text())
        for table in self.contract['tables']:
            files = sorted((r for r in manifest if r['table'] == table and r['disposition'] == 'accepted'), key=lambda r:r['code'])
            batch, rows, num = [], 0, 0
            for record in files:
                batch.append(record)
                rows += record['accepted_rows']
                if rows >= 250000:
                    yield table, f'{table}_{num:04}', batch
                    batch, rows, num = [], 0, num+1
            if batch:
                yield table, f'{table}_{num:04}', batch

    def load(self):
        self.create_stage()
        # Durable per-batch records make retries safe even after an uncertain HTTP outcome.
        with self.pg, self.pg.cursor() as cur:
            cur.execute(f'CREATE TABLE IF NOT EXISTS "{self.schema}".load_checkpoint (batch text PRIMARY KEY, rows bigint NOT NULL)')
        total = 0
        for table, batch_id, files in self.batches():
            rows = sum(r['accepted_rows'] for r in files)
            ids = [r['source_file_id'] for r in files]
            with self.pg.cursor() as cur:
                cur.execute(f'SELECT rows FROM "{self.schema}".load_checkpoint WHERE batch=%s', (batch_id,))
                committed = cur.fetchone()
            pg_done = committed is not None
            ch_done = batch_id in self.state['ch_batches']
            if pg_done and ch_done:
                total += rows
                continue
            payload_path = self.root / 'batch.tsv'
            with payload_path.open('wb') as output:
                for record in files:
                    path = Path(record['tsv'])
                    data = path.read_bytes()
                    if hashlib.sha256(data).hexdigest() != record['tsv_sha256']:
                        raise ValueError('Prepared data modified')
                    output.write(data)
            if not pg_done:
                with self.pg, self.pg.cursor() as cur, payload_path.open('r', encoding='utf-8') as handle:
                    cur.copy_expert(f'COPY {self.pgt(table)} FROM STDIN WITH (FORMAT text, ENCODING \'UTF8\')', handle)
                    cur.execute(f'INSERT INTO "{self.schema}".load_checkpoint VALUES (%s,%s)', (batch_id, rows))
                self.state['pg_batches'].append(batch_id)
                self.checkpoint()
            if not ch_done:
                id_sql = ','.join("'" + i + "'" for i in ids)
                existing = int(self.ch(f'SELECT count() FROM {self.cht(table)} WHERE source_file_id IN ({id_sql})'))
                if existing:
                    self.ch(f'ALTER TABLE {self.cht(table)} DELETE WHERE source_file_id IN ({id_sql}) SETTINGS mutations_sync=2')
                with payload_path.open('rb') as handle:
                    self.ch(f'INSERT INTO {self.cht(table)} FORMAT TabSeparated', data=handle)
                actual = int(self.ch(f'SELECT count() FROM {self.cht(table)} WHERE source_file_id IN ({id_sql})'))
                if actual != rows:
                    raise ValueError(f'ClickHouse batch incomplete: {actual} vs {rows}')
                self.state['ch_batches'].append(batch_id)
                self.checkpoint()
            total += rows
            log('loaded', batch=batch_id, rows=rows, cumulative_rows=total)
        self.state['status'] = 'loaded'
        self.checkpoint()

    def hash_expressions(self, table):
        pg, ch = [], []
        names = self.columns(table)
        metrics = {c['name'] for c in self.contract['tables'][table]['columns'][3:]}
        for col in names:
            p, h = f'"{col}"', f'`{col}`'
            if col in metrics:
                pg.append(f"coalesce({p}::text, E'\\\\N')")
                ch.append(f"ifNull(toDecimalString({h}, 4), '\\\\N')")
            elif col in ('open_time', 'close_time'):
                pg.append(f"to_char({p} AT TIME ZONE 'Asia/Shanghai', 'YYYY-MM-DD HH24:MI:SS')")
                ch.append(f"formatDateTime({h}, '%Y-%m-%d %H:%i:%S', 'Asia/Shanghai')")
            elif col == 'pe_suspect':
                pg.append(f'{p}::int::text')
                ch.append(f'toString(toUInt8({h}))')
            else:
                pg.append(f'{p}::text')
                ch.append(f'toString({h})')
        return f'md5(concat_ws(chr(31), {", ".join(pg)}))', f'lower(hex(MD5(arrayStringConcat([{", ".join(ch)}], char(31)))))'

    def verify(self, final=False):
        if self.pg is None:
            self.connect()
        expected = json.loads((self.root / 'expected.json').read_text())
        report = {}
        for table, exp in expected.items():
            log('verify_start', table=table, final=final)
            ph, ch = self.hash_expressions(table)
            digest = hashlib.sha256()
            class Sink:
                def write(self, data):
                    digest.update(data.encode() if isinstance(data,str) else data)
            with self.pg.cursor() as cur:
                cur.execute("SET work_mem='128MB'")
                cur.copy_expert(f'COPY (SELECT {ph} FROM {self.pgt(table,final)} ORDER BY "代码", "日期") TO STDOUT', Sink())
            pg_hash = digest.hexdigest()
            digest = hashlib.sha256()
            resp = self.ch(f'SELECT {ch} FROM {self.cht(table,final)} ORDER BY `代码`, `日期` SETTINGS max_threads=4, max_bytes_before_external_sort=1000000000 FORMAT TabSeparated', stream=True)
            for chunk in resp.iter_content(1024*1024):
                digest.update(chunk)
            resp.close()
            ch_hash = digest.hexdigest()
            with self.pg.cursor() as cur:
                cur.execute(f'SELECT count(*), count(DISTINCT "代码") FROM {self.pgt(table,final)}')
                pg_count, pg_codes = cur.fetchone()
            ch_stats = json.loads(self.ch(f'SELECT count() rows, uniqExact(`代码`) codes, uniqExact(tuple(`代码`,`日期`)) keys FROM {self.cht(table,final)} FORMAT JSON'))['data'][0]
            ok = pg_hash == ch_hash == exp['sha256_of_ordered_row_md5'] and pg_count == int(ch_stats['rows']) == int(ch_stats['keys']) == exp['rows'] and pg_codes == int(ch_stats['codes']) == exp['codes']
            report[table] = {'ok': ok, 'rows': pg_count, 'codes': pg_codes, 'postgres_sha256': pg_hash,
                             'clickhouse_sha256': ch_hash, 'source_sha256': exp['sha256_of_ordered_row_md5'], 'pe_nulls': exp['pe_nulls']}
            save(self.root / ('verification_final.json' if final else 'verification.json'), report)
            log('verified', table=table, **report[table])
            if not ok:
                raise ValueError(f'Full-field verification failed: {table}')
        self.state['status'] = 'published' if final else 'verified'
        self.checkpoint()

    def write_metadata(self):
        manifest = json.loads((self.root / 'manifest.json').read_text())
        audit = []
        for path in sorted((self.root / 'prepared').rglob('*.audit.jsonl')):
            audit.extend(json.loads(line) for line in path.read_text().splitlines())
        old = next(r for r in manifest if r['source']=='stock_hfq' and r['code']=='920305.BJ')
        new = next(r for r in manifest if r['source']=='stock_delisted_hfq' and r['code']=='920305.BJ')
        if hashlib.sha256(Path(old['path']).read_bytes()).hexdigest() != old['sha256']:
            raise ValueError('Overlap source changed before audit was recorded')
        source_frame = read_file(old['path'], encoding='gb18030', dtype=str, keep_default_na=False)
        for i, row in source_frame.iterrows():
            audit.append({'source_file_id':old['source_file_id'], 'source_row':i+1, 'code':row['代码'], 'date':row['日期'],
                'column':'简称', 'original_value':row['简称'], 'action':'prefer_delisted_source',
                'selected_source_file_id':new['source_file_id'], 'selected_value':'云创退(退市)'})
        objects = {
            'source_files': [(r['source_file_id'], json.dumps(r, ensure_ascii=False)) for r in manifest],
            'value_audit': [(f'{r["source_file_id"]}:{r["source_row"]}:{r["column"]}', json.dumps(r,ensure_ascii=False)) for r in audit],
            'field_contract': [(t, json.dumps(v,ensure_ascii=False)) for t,v in self.contract['tables'].items()],
            'import_batch': [(self.tag, json.dumps({'status':'verified', 'decisions':self.contract['decisions'], 'verification':json.loads((self.root/'verification.json').read_text())},ensure_ascii=False))]
        }
        with self.pg, self.pg.cursor() as cur:
            for table, rows in objects.items():
                cur.execute(f'CREATE TABLE IF NOT EXISTS {self.pgt(table)} (id text PRIMARY KEY, detail jsonb NOT NULL)')
                cur.execute(f'TRUNCATE {self.pgt(table)}')
                buf = io.StringIO()
                writer = csv.writer(buf, lineterminator='\n')
                writer.writerows(rows)
                buf.seek(0)
                cur.copy_expert(f'COPY {self.pgt(table)} FROM STDIN WITH CSV', buf)
                self.ch(f'CREATE TABLE IF NOT EXISTS {self.cht(table)} (id String, detail String) ENGINE=MergeTree ORDER BY id')
                self.ch(f'TRUNCATE TABLE {self.cht(table)}')
                payload = ''.join(json.dumps({'id':i,'detail':d},ensure_ascii=False)+'\n' for i,d in rows).encode()
                self.ch(f'INSERT INTO {self.cht(table)} FORMAT JSONEachRow', payload)
                actual = int(self.ch(f'SELECT count() FROM {self.cht(table)}'))
                if actual != len(rows):
                    raise ValueError('Metadata count mismatch')
        save(self.root/'audit_summary.json', {'source_files':len(manifest), 'excluded_b_share_files':sum(r['disposition']=='excluded_b_share' for r in manifest),
              'excluded_b_share_rows':sum(r['source_rows'] for r in manifest if r['disposition']=='excluded_b_share'),
              'sentinel_to_null':sum(r['pe_nulls'] for r in manifest), 'overlap_rows':len(source_frame), 'audit_records':len(audit)})
        save(self.root/'excluded_files.json', [r for r in manifest if r['disposition']!='accepted'])
        log('metadata_written', records={k:len(v) for k,v in objects.items()})

    def publish(self):
        verified = json.loads((self.root/'verification.json').read_text())
        if set(verified)!=set(EXPECTED) or not all(r['ok'] for r in verified.values()):
            raise ValueError('Not verified')
        if self.pg is None:
            self.connect()
        names = list(EXPECTED)+['source_files','value_audit','field_contract','import_batch']
        for table in names:
            final_exists = int(self.ch(f'EXISTS TABLE {self.cht(table,True)}'))
            stage_exists = int(self.ch(f'EXISTS TABLE {self.cht(table)}'))
            if final_exists and stage_exists:
                raise ValueError(f'Refusing to overwrite existing final table: {table}')
            if stage_exists:
                self.ch(f'RENAME TABLE {self.cht(table)} TO {self.cht(table,True)}')
            elif not final_exists:
                raise ValueError(f'Missing table during publish: {table}')
        with self.pg, self.pg.cursor() as cur:
            cur.execute('SELECT to_regnamespace(%s),to_regnamespace(%s)', (self.schema,'market'))
            staging, final = cur.fetchone()
            if staging and final:
                raise ValueError('Refusing to overwrite existing market schema')
            if staging:
                cur.execute(f'ALTER SCHEMA "{self.schema}" RENAME TO market')
            elif not final:
                raise ValueError('Missing PostgreSQL stage')
            cur.execute("UPDATE market.import_batch SET detail=jsonb_set(detail,'{status}','\"published\"') WHERE id=%s",(self.tag,))
        self.ch(f"ALTER TABLE {self.cht('import_batch',True)} UPDATE detail=replaceOne(detail, '\"status\": \"verified\"', '\"status\": \"published\"') WHERE id='{self.tag}' SETTINGS mutations_sync=2")
        self.state['status']='published'
        self.checkpoint()
        log('published', postgres='datafeed.market', clickhouse=self.ch_database)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','load','verify','metadata','publish','verify-final'])
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--env-file', required=True)
    parser.add_argument('--audit')
    parser.add_argument('--source-root')
    args=parser.parse_args()
    import fcntl
    Path(args.run_dir).mkdir(parents=True,exist_ok=True)
    lock=Path(args.run_dir,'import.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    importer=Importer(args.run_dir,args.contract,args.env_file)
    if importer.state['status']=='published' and args.command!='verify-final':
        log('already_published',run_dir=args.run_dir)
        return
    if args.command=='prepare': importer.prepare(args.audit,args.source_root)
    elif args.command=='load': importer.load()
    elif args.command=='verify': importer.verify()
    elif args.command=='verify-final': importer.verify(final=True)
    elif args.command=='metadata':
        importer.connect()
        importer.write_metadata()
    elif args.command=='publish': importer.publish()


if __name__=='__main__':
    main()
