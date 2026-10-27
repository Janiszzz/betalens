"""Target-local immutable reference revisions and publication credentials.

Nothing in this module uses an orchestration job's status to grant visibility.
Rows are inserted first; a separately verified, append-only publication event
is the sole visibility gate.  The physical keys always retain the revision.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json

from psycopg2 import sql
from psycopg2.extras import RealDictCursor, execute_values

from .canonical import CanonicalTable, _f
from .reference_contracts import MODULE_TABLE, TABLES, VERSION
from .storage import pg_type
from .util import digest, dumps, identifier, now, parse_date


def _meta(name, fields, keys):
    return CanonicalTable(name, name, '不可变业务参考版本及发布凭证', '版本或发布事件',
                          tuple(_f(n, t, n, nullable=nullable) for n, t, nullable in fields),
                          tuple(keys), 'reference_control')


METADATA = {
    'reference_revision': _meta('reference_revision', [
        ('revision_id', 'text', False), ('import_task_id', 'text', False),
        ('attempt_id', 'text', False), ('module', 'text', False),
        ('ingestion_seq', 'integer', False), ('ingested_at', 'timestamp', False),
        ('manifest_hash', 'text', False), ('payload', 'text', False),
    ], ['revision_id']),
    'reference_scope': _meta('reference_scope', [
        ('revision_id', 'text', False), ('scope_id', 'text', False),
        ('module', 'text', False), ('selector_hash', 'text', False),
        ('row_count', 'integer', False), ('content_hash', 'text', False),
        ('key_hash', 'text', False), ('complete', 'integer', False),
        ('payload', 'text', False),
    ], ['revision_id', 'scope_id']),
    'reference_publication_event': _meta('reference_publication_event', [
        ('event_id', 'text', False), ('event_seq', 'integer', False),
        ('revision_id', 'text', False), ('event_type', 'text', False),
        ('rollback_task_id', 'text', True), ('created_at', 'timestamp', False),
        ('payload', 'text', False),
    ], ['event_id']),
}


def _timestamp(value):
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00').replace(' ', 'T'))
    if result.tzinfo is None:
        # Storage is never the place to guess the timezone of an API timestamp.
        raise ValueError('版本时间必须包含时区')
    return result.astimezone(timezone.utc).isoformat(timespec='microseconds')


def _typed(field, value):
    if value is None:
        if not field.get('nullable', True):
            raise ValueError('版本字段不能为空: ' + field['name'])
        return None
    typ = field['type']
    if typ == 'timestamp':
        return _timestamp(value)
    if typ == 'date':
        return parse_date(value).isoformat()
    if typ == 'integer':
        if isinstance(value, bool):
            value = int(value)
        number = Decimal(str(value))
        if not number.is_finite() or number != number.to_integral_value() or not -(2**63) <= number < 2**63:
            raise ValueError('无效的整数版本字段: ' + field['name'])
        return int(number)
    if typ == 'numeric':
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            raise ValueError('无效数值: ' + field['name']) from None
        if not number.is_finite():
            raise ValueError('数值必须有限: ' + field['name'])
        # Fixed decimals from both backends hash identically, including zeros.
        text = format(number, 'f')
        return (text.rstrip('0').rstrip('.') if '.' in text else text) if number else '0'
    return str(value)


def _normal(spec, row):
    names = {f['name'] for f in spec.fields}
    unexpected = set(row) - names
    if unexpected:
        raise ValueError(spec.name + ': 未登记字段 ' + ','.join(sorted(unexpected)))
    return {f['name']: _typed(f, row.get(f['name'])) for f in spec.fields}


def _ordered(rows):
    return sorted(rows, key=lambda row: (row['scope_id'], row['record_id']))


def _content(rows):
    return digest(_ordered(rows))


def _keys(rows):
    return digest(sorted(row['record_id'] for row in rows))


def publication_heads(revisions, scopes, events):
    """Filter credentials before ordering: revocation reveals the prior snapshot."""
    published = {e['revision_id'] for e in events if e['event_type'] == 'publish'}
    revoked = {e['revision_id'] for e in events if e['event_type'] == 'revoke'}
    live = {r['revision_id']: r for r in revisions if r['revision_id'] in published - revoked}
    result = {}
    for scope in scopes:
        revision_id = scope['revision_id']
        if revision_id not in live:
            continue
        existing = result.get(scope['scope_id'])
        if existing is None or int(live[revision_id]['ingestion_seq']) > int(live[existing]['ingestion_seq']):
            result[scope['scope_id']] = revision_id
    return result


class ReferenceStore:
    """Append-only PG/CH storage; all decisions are made using target-local facts."""

    def __init__(self, service, target):
        if target not in {'postgres', 'clickhouse'}:
            raise ValueError('未知目标库: ' + str(target))
        self.service = service
        self.target = target
        self.db = service.pg if target == 'postgres' else service.ch
        self.settings = service.settings

    def _relation(self, name):
        return sql.Identifier(self.settings.pg_schema, identifier(name))

    @contextmanager
    def _write(self, operation):
        if self.target == 'postgres':
            with self.db.connect() as conn:
                with conn.cursor() as cur:
                    key = int.from_bytes(hashlib.sha256(('betalens-reference:' + self.settings.pg_schema).encode()).digest()[:8], 'big', signed=True)
                    cur.execute('SELECT pg_advisory_xact_lock(%s)', (key,))
                yield conn
        else:
            from .publishing import operation_lock
            with operation_lock(self.service), self.db.write_window('reference:' + operation):
                yield None

    def install(self):
        """Create-only installation; never silently alter an existing contract."""
        specs = {**TABLES, **METADATA}
        with self._write('install') as conn:
            if self.target == 'postgres':
                with conn.cursor() as cur:
                    cur.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(self.settings.pg_schema)))
                    for spec in specs.values():
                        definitions = [sql.SQL('{} {}{}').format(sql.Identifier(f['name']), sql.SQL(pg_type(f)),
                                       sql.SQL(' NOT NULL' if f['name'] in spec.keys or not f.get('nullable', True) else ''))
                                       for f in spec.fields]
                        definitions.append(sql.SQL('PRIMARY KEY ({})').format(sql.SQL(',').join(map(sql.Identifier, spec.keys))))
                        cur.execute(sql.SQL('CREATE TABLE IF NOT EXISTS {} ({})').format(self._relation(spec.name), sql.SQL(',').join(definitions)))
            else:
                self.db.execute(f'CREATE DATABASE IF NOT EXISTS `{identifier(self.settings.ch_database)}`')
                for spec in specs.values():
                    self.db.execute(self.db.ddl(spec))
            self._assert_structure(conn)
        return {'target': self.target, 'tables': list(specs), 'status': 'installed'}

    def _assert_structure(self, conn=None, module=None):
        """A mismatched physical key must never collapse historical revisions."""
        from .schema_contract import compare_structure
        specs = {**(TABLES if module is None else {MODULE_TABLE[module]: TABLES[MODULE_TABLE[module]]}), **METADATA}
        if self.target == 'postgres' and conn is None:
            with self.db.connect(readonly=True) as read:
                return self._assert_structure(read, module)
        problems = []
        for spec in specs.values():
            if self.target == 'postgres':
                with conn.cursor() as cur:
                    cur.execute('SELECT column_name,data_type,is_nullable,numeric_precision,numeric_scale FROM information_schema.columns '
                                'WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position',
                                (self.settings.pg_schema, spec.name))
                    fields = [dict(name=r[0], physical_type=r[1], nullable=r[2] == 'YES', precision=r[3], scale=r[4]) for r in cur.fetchall()]
                    cur.execute("SELECT kcu.column_name FROM information_schema.table_constraints tc JOIN information_schema.key_column_usage kcu "
                                "USING(constraint_catalog,constraint_schema,constraint_name) WHERE tc.table_schema=%s AND tc.table_name=%s "
                                "AND tc.constraint_type='PRIMARY KEY' ORDER BY kcu.ordinal_position", (self.settings.pg_schema, spec.name))
                    keys = [r[0] for r in cur.fetchall()]
            else:
                where = f'database={self.db._literal(self.settings.ch_database)} AND table={self.db._literal(spec.name)}'
                fields = self.db._json_rows(self.db.execute('SELECT name,type AS physical_type FROM system.columns WHERE ' + where + ' ORDER BY position FORMAT JSONEachRow'))
                properties = self.db._json_rows(self.db.execute('SELECT sorting_key,engine FROM system.tables WHERE '
                    f'database={self.db._literal(self.settings.ch_database)} AND name={self.db._literal(spec.name)} FORMAT JSONEachRow'))
                sorting = properties[0]['sorting_key'] if properties else ''
                keys = [key.strip().strip('`') for key in sorting.removeprefix('tuple(').removesuffix(')').split(',') if key.strip()]
                if properties and properties[0]['engine'] != 'ReplacingMergeTree':
                    problems.append(spec.name + ': 历史版本表引擎必须为 ReplacingMergeTree')
            difference = compare_structure(spec, {'fields': fields, 'keys': keys, 'exists': bool(fields)}, self.target)
            if not difference['ready']:
                problems.append(spec.name + ': ' + dumps(difference))
        if problems:
            raise ValueError('参考版本物理结构漂移，已阻止操作: ' + '; '.join(problems))

    def _exists(self, name, conn=None):
        if self.target == 'postgres':
            if conn is None:
                with self.db.connect(readonly=True) as read:
                    return self._exists(name, read)
            with conn.cursor() as cur:
                cur.execute('SELECT 1 FROM information_schema.tables WHERE table_schema=%s AND table_name=%s',
                            (self.settings.pg_schema, name))
                return cur.fetchone() is not None
        return self.db.execute('EXISTS TABLE ' + self.db.table(name)) == '1'

    def _rows(self, spec, conn=None, revision_id=None):
        if not self._exists(spec.name, conn):
            return []
        if self.target == 'postgres':
            if conn is None:
                with self.db.connect(readonly=True) as read:
                    return self._rows(spec, read, revision_id)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = sql.SQL('SELECT {} FROM {}').format(sql.SQL(',').join(sql.Identifier(f['name']) for f in spec.fields), self._relation(spec.name))
                if revision_id is not None:
                    query += sql.SQL(' WHERE revision_id=%s')
                cur.execute(query, (revision_id,) if revision_id is not None else None)
                rows = list(cur.fetchall())
        else:
            where = ' WHERE revision_id=' + self.db._literal(revision_id) if revision_id is not None else ''
            names = ','.join('`' + f['name'] + '`' for f in spec.fields)
            rows = self.db._json_rows(self.db.execute(f'SELECT {names} FROM {self.db.table(spec.name)} FINAL{where} FORMAT JSONEachRow'))
            # JSONEachRow omits the timezone suffix even for DateTime64(...,
            # 'UTC'). The column contract supplies UTC on *read*, whereas a
            # caller-supplied naive timestamp is still rejected on write.
            for row in rows:
                for field in spec.fields:
                    value = row.get(field['name'])
                    if field['type'] == 'timestamp' and value is not None:
                        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
                        if parsed.tzinfo is None:
                            row[field['name']] = parsed.replace(tzinfo=timezone.utc)
        return [_normal(spec, row) for row in rows]

    def _insert(self, spec, rows, conn=None):
        if not rows:
            return
        normalized = [_normal(spec, row) for row in rows]
        if self.target == 'postgres':
            names = [f['name'] for f in spec.fields]
            query = sql.SQL('INSERT INTO {} ({}) VALUES %s').format(self._relation(spec.name), sql.SQL(',').join(map(sql.Identifier, names)))
            with conn.cursor() as cur:
                execute_values(cur, query.as_string(conn), [[row[n] for n in names] for row in normalized], page_size=1000)
        else:
            for offset in range(0, len(normalized), 1000):
                self.db.execute(f'INSERT INTO {self.db.table(spec.name)} FORMAT JSONEachRow',
                                '\n'.join(dumps(row) for row in normalized[offset:offset + 1000]) + '\n')

    def _control(self, conn=None):
        def payloads(name):
            spec = METADATA[name]
            result = []
            for row in self._rows(spec, conn):
                payload = json.loads(row['payload'])
                expected = _normal(spec, {**{f['name']: payload.get(f['name']) for f in spec.fields if f['name'] != 'payload'},
                                         'payload': row['payload']})
                if row != expected:
                    raise ValueError('目标库不可变清单字段与载荷不一致: ' + name)
                result.append(payload)
            return result
        # A CH publication is the final insert. Read the event cut first so a
        # concurrently staged/published revision cannot appear as an event
        # without the earlier immutable manifest and data. PG readers also use
        # one repeatable-read transaction across all tables and verification.
        events = payloads('reference_publication_event')
        revisions = payloads('reference_revision')
        scopes = payloads('reference_scope')
        revisions.sort(key=lambda row: int(row['ingestion_seq']))
        events.sort(key=lambda row: int(row['event_seq']))
        return revisions, scopes, events

    @contextmanager
    def _read(self):
        if self.target == 'postgres':
            with self.db.connect(readonly=True) as conn:
                with conn.cursor() as cur:
                    cur.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
                yield conn
        else:
            yield None

    def heads(self, scope_ids):
        ids = list(scope_ids)
        with self._read() as conn:
            revisions, scopes, events = self._control(conn)
            current, _ = self._valid_heads(revisions, scopes, events, conn)
        return {scope_id: current.get(scope_id) for scope_id in ids}

    def _valid_heads(self, revisions, scopes, events, conn=None):
        published = {e['revision_id'] for e in events if e['event_type'] == 'publish'}
        revoked = {e['revision_id'] for e in events if e['event_type'] == 'revoke'}
        verification = [self._verify(r['revision_id'], conn, control=(revisions, scopes, events))
                        for r in revisions if r['revision_id'] in published - revoked]
        by_id = {r['revision_id']: r for r in revisions}
        for result in verification:
            revision = by_id[result['revision_id']]
            credentials = [e for e in events if e['revision_id'] == revision['revision_id'] and e['event_type'] == 'publish']
            if any(e.get('manifest_hash') != revision['manifest_hash'] or e.get('scope_ids') != revision['scope_ids'] for e in credentials):
                result['verified'] = False
                result['errors'].append('发布凭证与修订清单不一致')
        invalid = {v['revision_id'] for v in verification if not v['verified']}
        current = publication_heads([r for r in revisions if r['revision_id'] not in invalid], scopes, events)
        return current, verification

    def _input_hash(self, revision, scopes):
        # The caller must not supply target-generated values to bypass immutable
        # retry checks. Those values are assigned once in the target manifest.
        supplied = {k: v for k, v in revision.items() if k not in {'ingestion_seq', 'ingested_at', 'manifest_hash', 'input_hash'}}
        if supplied.get('downloaded_at'):
            supplied['downloaded_at'] = _timestamp(supplied['downloaded_at'])
        return digest({'revision': supplied, 'scopes': sorted(scopes, key=lambda s: s['scope_id'])})

    def _prepare(self, revision, scopes, sequence, ingested):
        revision = json.loads(dumps(revision))
        scopes = json.loads(dumps(scopes))
        module = revision.get('module')
        if module not in MODULE_TABLE:
            raise ValueError('未知业务模块: ' + str(module))
        for key in ('revision_id', 'import_task_id', 'attempt_id', 'mapping_version', 'source_id', 'downloaded_at'):
            if not revision.get(key):
                raise ValueError('缺少修订字段: ' + key)
        revision['downloaded_at'] = _timestamp(revision['downloaded_at'])
        if not scopes:
            raise ValueError('修订必须有明确覆盖范围；空响应也需要范围凭证')
        if len({s['scope_id'] for s in scopes}) != len(scopes):
            raise ValueError('修订覆盖范围重复')
        spec = TABLES[MODULE_TABLE[module]]
        prepared_scopes, all_rows = [], []
        for scope in scopes:
            if scope.get('module', module) != module:
                raise ValueError('一个修订只能包含一个独立业务模块')
            selector = scope.get('selector')
            if not isinstance(selector, dict) or not selector:
                raise ValueError('覆盖范围必须包含显式 selector')
            rows = []
            for source in scope.get('rows', []):
                row = dict(source)
                row.update(revision_id=revision['revision_id'], scope_id=scope['scope_id'],
                           import_task_id=revision['import_task_id'], attempt_id=revision['attempt_id'],
                           ingestion_seq=sequence, ingested_at=ingested)
                row.setdefault('record_id', digest(source))
                for key in ('downloaded_at', 'mapping_version', 'source_id'):
                    row.setdefault(key, revision[key])
                row.setdefault('source_name', revision.get('source_name', 'tushare'))
                row.setdefault('schema_version', VERSION)
                row.setdefault('data_quality_status', 'unverified')
                rows.append(_normal(spec, row))
            record_ids = [r['record_id'] for r in rows]
            if len(record_ids) != len(set(record_ids)):
                raise ValueError('范围内业务记录编号重复: ' + scope['scope_id'])
            payload = {k: v for k, v in scope.items() if k != 'rows'}
            payload.update(revision_id=revision['revision_id'], module=module,
                           row_count=len(rows), key_hash=_keys(rows), content_hash=_content(rows),
                           complete=bool(scope.get('complete', False)), reasons=scope.get('reasons', []),
                           evidence=scope.get('evidence', {}), dependencies=scope.get('dependencies', []),
                           selector_hash=digest(selector))
            prepared_scopes.append(payload)
            all_rows.extend(rows)
        manifest = dict(revision, ingestion_seq=sequence, ingested_at=ingested,
                        scope_ids=sorted(s['scope_id'] for s in prepared_scopes),
                        scope_count=len(prepared_scopes), row_count=len(all_rows),
                        input_hash=self._input_hash(revision, scopes))
        manifest['manifest_hash'] = digest({'revision': manifest, 'scopes': sorted(prepared_scopes, key=lambda s: s['scope_id'])})
        return manifest, prepared_scopes, _ordered(all_rows)

    def stage(self, revision, scopes):
        """Append rows without publishing; exact retry can finish uncertain inserts."""
        with self._write('stage:' + str(revision.get('revision_id'))) as conn:
            if revision.get('module') not in MODULE_TABLE:
                raise ValueError('未知业务模块: ' + str(revision.get('module')))
            self._assert_structure(conn, revision['module'])
            revisions, stored_scopes, events = self._control(conn)
            old = next((r for r in revisions if r['revision_id'] == revision['revision_id']), None)
            if old is not None:
                if old['input_hash'] != self._input_hash(revision, scopes):
                    raise ValueError('修订编号已绑定不同输入，必须创建新修订')
                sequence, ingested = int(old['ingestion_seq']), old['ingested_at']
            else:
                sequence = max((int(r['ingestion_seq']) for r in revisions), default=0) + 1
                ingested = _timestamp(now())
            manifest, scope_docs, rows = self._prepare(revision, scopes, sequence, ingested)
            for scope in scope_docs:
                if any(s['scope_id'] == scope['scope_id'] and (s['module'] != scope['module'] or s['selector_hash'] != scope['selector_hash'])
                       for s in stored_scopes):
                    raise ValueError('覆盖范围编号已绑定不同的业务范围')
            if old is not None and old != manifest:
                raise ValueError('修订清单与已保存内容不一致')
            if old is None:
                self._insert(METADATA['reference_revision'], [{
                    **{k: manifest[k] for k in ('revision_id', 'import_task_id', 'attempt_id', 'module', 'ingestion_seq', 'ingested_at', 'manifest_hash')},
                    'payload': dumps(manifest),
                }], conn)
            existing_scopes = {s['scope_id']: s for s in stored_scopes if s['revision_id'] == manifest['revision_id']}
            for scope in scope_docs:
                present = existing_scopes.get(scope['scope_id'])
                if present is not None and present != scope:
                    raise ValueError('不可变覆盖凭证发生变化')
                if present is None:
                    self._insert(METADATA['reference_scope'], [{
                        **{k: scope[k] for k in ('revision_id', 'scope_id', 'module', 'selector_hash', 'row_count', 'content_hash', 'key_hash')},
                        'complete': int(scope['complete']), 'payload': dumps(scope),
                    }], conn)
            spec = TABLES[MODULE_TABLE[manifest['module']]]
            existing = {(r['scope_id'], r['record_id']): r for r in self._rows(spec, conn, manifest['revision_id'])}
            incoming = {(r['scope_id'], r['record_id']): r for r in rows}
            if set(existing) - set(incoming):
                raise ValueError('目标库修订出现清单以外的记录')
            for key, row in existing.items():
                if row != incoming[key]:
                    raise ValueError('目标库存在内容不同的同版本记录')
            self._insert(spec, [row for key, row in incoming.items() if key not in existing], conn)
        return manifest

    def _verify(self, revision_id, conn=None, control=None):
        revisions, scopes, events = control if control is not None else self._control(conn)
        revision = next((r for r in revisions if r['revision_id'] == revision_id), None)
        if revision is None:
            raise ValueError('修订不存在: ' + revision_id)
        scope_docs = [s for s in scopes if s['revision_id'] == revision_id]
        spec = TABLES[MODULE_TABLE[revision['module']]]
        rows = self._rows(spec, conn, revision_id)
        errors, results = [], []
        if sorted(s['scope_id'] for s in scope_docs) != revision['scope_ids']:
            errors.append('覆盖范围清单不完整')
        manifest = {k: v for k, v in revision.items() if k != 'manifest_hash'}
        if digest({'revision': manifest, 'scopes': sorted(scope_docs, key=lambda s: s['scope_id'])}) != revision['manifest_hash']:
            errors.append('修订清单哈希不匹配')
        for scope in scope_docs:
            actual = [r for r in rows if r['scope_id'] == scope['scope_id']]
            actual_keys, actual_hash = _keys(actual), _content(actual)
            good = len(actual) == int(scope['row_count']) and actual_keys == scope['key_hash'] and actual_hash == scope['content_hash']
            results.append({'scope_id': scope['scope_id'], 'verified': good,
                            'expected_rows': scope['row_count'], 'actual_rows': len(actual),
                            'key_hash': actual_keys, 'content_hash': actual_hash})
            if not good:
                errors.append('范围记录、键或内容核验不通过: ' + scope['scope_id'])
        if len(rows) != int(revision['row_count']) or any(r['scope_id'] not in revision['scope_ids'] for r in rows):
            errors.append('修订记录总数或范围不匹配')
        return {'target': self.target, 'revision_id': revision_id, 'verified': not errors,
                'row_count': len(rows), 'scope_count': len(scope_docs), 'scope_results': results, 'errors': errors}

    def verify(self, revision_id):
        with self._read() as conn:
            return self._verify(revision_id, conn)

    def _append_event(self, revision, events, kind, rollback_task_id, conn=None):
        sequence = max((int(e['event_seq']) for e in events), default=0) + 1
        event_id = digest([revision['revision_id'], kind, rollback_task_id])
        event = {'event_id': event_id, 'event_seq': sequence, 'revision_id': revision['revision_id'],
                 'event_type': kind, 'rollback_task_id': rollback_task_id, 'created_at': _timestamp(now()),
                 'ingestion_seq': revision['ingestion_seq'], 'scope_ids': revision['scope_ids'],
                 'manifest_hash': revision['manifest_hash']}
        self._insert(METADATA['reference_publication_event'], [{
            **{k: event[k] for k in ('event_id', 'event_seq', 'revision_id', 'event_type', 'rollback_task_id', 'created_at')},
            'payload': dumps(event),
        }], conn)
        return event

    def publish(self, revision_id, expected_heads):
        with self._write('publish:' + revision_id) as conn:
            revisions, scopes, events = self._control(conn)
            revision = next((r for r in revisions if r['revision_id'] == revision_id), None)
            if revision is None:
                raise ValueError('修订不存在: ' + revision_id)
            self._assert_structure(conn, revision['module'])
            previous = next((e for e in events if e['revision_id'] == revision_id and e['event_type'] == 'publish'), None)
            if any(e['revision_id'] == revision_id and e['event_type'] == 'revoke' for e in events):
                raise ValueError('已撤销修订不能重新发布，必须创建新修订')
            if previous is not None:
                verified = self._verify(revision_id, conn, control=(revisions, scopes, events))
                if not verified['verified'] or previous.get('manifest_hash') != revision['manifest_hash'] or previous.get('scope_ids') != revision['scope_ids']:
                    raise ValueError('已发布版本的目标库核验失败，不能以重试掩盖不一致')
                return dict(previous, target=self.target, status='published', published_at=previous['created_at'])
            if set(expected_heads) != set(revision['scope_ids']):
                raise ValueError('确认必须覆盖修订的全部范围版本头')
            current, existing_verification = self._valid_heads(revisions, scopes, events, conn)
            if any(current.get(scope_id) != expected_heads[scope_id] for scope_id in revision['scope_ids']):
                raise ValueError('审计后范围版本头已变化，请重新审计')
            valid_revisions = {v['revision_id'] for v in existing_verification if v['verified']}
            for scope in (s for s in scopes if s['revision_id'] == revision_id):
                for dependency in scope.get('dependencies', []):
                    if isinstance(dependency, str):
                        dependency = {'revision_id': dependency}
                    if not isinstance(dependency, dict) or not dependency.get('revision_id'):
                        continue
                    if dependency.get('target') not in (None, self.target):
                        raise ValueError('依赖凭证必须位于当前目标库')
                    if dependency['revision_id'] not in valid_revisions:
                        raise ValueError('依赖版本未可信发布或已撤销，请重新审计')
                    if dependency.get('scope_id') and current.get(dependency['scope_id']) != dependency['revision_id']:
                        raise ValueError('依赖范围版本头已变化，请重新审计')
            verified = self._verify(revision_id, conn)
            if not verified['verified']:
                raise ValueError('目标库核验失败: ' + '; '.join(verified['errors']))
            event = self._append_event(revision, events, 'publish', None, conn)
        return dict(event, target=self.target, status='published', published_at=event['created_at'])

    def revoke(self, revision_id, rollback_task_id):
        if not rollback_task_id:
            raise ValueError('撤销必须包含独立回退任务编号')
        with self._write('revoke:' + rollback_task_id) as conn:
            revisions, scopes, events = self._control(conn)
            revision = next((r for r in revisions if r['revision_id'] == revision_id), None)
            if revision is None:
                raise ValueError('修订不存在: ' + revision_id)
            if not any(e['revision_id'] == revision_id and e['event_type'] == 'publish' for e in events):
                raise ValueError('修订从未发布，不能伪造撤销发布凭证')
            previous = next((e for e in events if e['revision_id'] == revision_id and e['event_type'] == 'revoke'), None)
            if previous is not None:
                return dict(previous, target=self.target, status='revoked')
            event = self._append_event(revision, events, 'revoke', rollback_task_id, conn)
        return dict(event, target=self.target, status='revoked')

    def snapshot(self, audit=False):
        with self._read() as conn:
            return self._snapshot(audit, conn)

    def _snapshot(self, audit=False, conn=None):
        revisions, scopes, events = self._control(conn)
        current, verification = self._valid_heads(revisions, scopes, events, conn)
        invalid = {v['revision_id'] for v in verification if not v['verified']}
        if not audit:
            scopes = [s for s in scopes if current.get(s['scope_id']) == s['revision_id']]
            visible_ids = {s['revision_id'] for s in scopes}
            revisions = [r for r in revisions if r['revision_id'] in visible_ids]
        selected = {(s['revision_id'], s['scope_id']) for s in scopes}
        rows = []
        for module, name in MODULE_TABLE.items():
            if not any(r['module'] == module for r in revisions):
                continue
            rows.extend(dict(row, module=module) for row in self._rows(TABLES[name], conn)
                        if audit or (row['revision_id'], row['scope_id']) in selected)
        return {'target': self.target, 'revisions': revisions, 'scopes': scopes,
                'rows': sorted(rows, key=lambda r: (int(r['ingestion_seq']), r['scope_id'], r['record_id'])),
                'events': events, 'heads': current, 'verification': verification,
                'invalid_revision_ids': sorted(invalid)}
