from copy import deepcopy
import json

import pandas as pd
import pytest

from betalens.datafeed import query
from betalens.datafeed.contracts import digest, timestamp, VersionConflictError
from betalens.datafeed.reference import load_reference
from datafeed_support import reference_reader, append_revision, evidence
from test_datafeed import suspension_scope

TABLE = 'stock_suspension_intervals_versions'


def compact(first='2024-01-02', last='2024-01-05', rows=None, availability=True):
    calendar = {day.date().isoformat(): {'complete': True, 'rows': [{'is_open': 1, **evidence()}]}
                for day in pd.date_range(first, pd.Timestamp(last) - pd.Timedelta(days=1))}
    lifecycle = {'complete': True, 'list_date': '2020-01-01', **evidence()}
    pool = {digest(value): value for value in (calendar, lifecycle, [])}
    selector = {'security_code': 'A', 'start_date': first, 'end_date': last}
    scope = {'scope_id': digest({'module': 'suspension', 'selector': selector}),
             'selector': selector, 'format': 'suspension-ranges-v2', 'complete': True,
             'rows': rows or [], 'evidence': {'event_query_complete': True,
                 'calendar_ref': digest(calendar), 'lifecycle_ref': digest(lifecycle),
                 'queries_ref': digest([])}}
    if availability:
        scope['evidence']['availability'] = evidence()
    return scope, pool


def event(day='2024-01-03', **extra):
    return {'security_code': 'A', 'trade_date': day, 'suspend_type': 'S', 'parse_status': 'parsed',
            'start_at': timestamp(day + ' 09:30').isoformat(),
            'end_at': timestamp(day + ' 11:00').isoformat(), **evidence(), **extra}


def test_compact_scope_restores_daily_and_intraday_status_with_one_batched_read():
    scope, pool = compact(rows=[event()])
    reader = reference_reader('suspension', [('range', [scope], 'publish', pool)])
    result = query.query_trade_status(reader, TABLE, ['A'],
        ['2024-01-02 10:00', '2024-01-03 10:00', '2024-01-03 12:00', '2024-01-05 10:00'])
    assert result.value.iloc[:3].tolist() == [1, 0, 1]
    assert pd.isna(result.value.iloc[3])
    assert reader.calls.count(TABLE) == 1


def test_latest_empty_partial_range_hides_old_event_and_revoke_restores_it():
    old, pool = compact(rows=[event()])
    middle, shared = compact('2024-01-03', '2024-01-04')
    reader = reference_reader('suspension', [('old', [old], 'publish', pool),
        ('new', [middle], 'publish', shared)])
    assert query.query_trade_status(reader, TABLE, ['A'], ['2024-01-03 10:00']).value.tolist() == [1]
    revoked = reference_reader('suspension', [('old', [old], 'publish', pool),
        ('new', [middle], 'revoke', shared)])
    assert query.query_trade_status(revoked, TABLE, ['A'], ['2024-01-03 10:00']).value.tolist() == [0]
    legacy = suspension_scope([event(day='2024-01-02')])
    append_revision(reader, 'suspension', 'daily', 3, [legacy])
    assert query.query_trade_status(reader, TABLE, ['A'], ['2024-01-02 10:00']).value.tolist() == [0]


def test_research_uses_trusted_history_without_claiming_strict_availability():
    scope, pool = compact(availability=False)
    reader = reference_reader('suspension', [('range', [scope], 'publish', pool)])
    assert query.query_trade_status(reader, TABLE, ['A'], ['2024-01-02'], pit_policy='strict').value.isna().all()
    assert query.query_trade_status(reader, TABLE, ['A'], ['2024-01-02'], pit_policy='research').value.tolist() == [1]


def test_matrix_keeps_unknown_and_cache_is_invalidated_by_revoke_or_content_changes():
    old, pool = compact(rows=[event()])
    middle, shared = compact('2024-01-03', '2024-01-04')
    reader = reference_reader('suspension', [('old', [old], 'publish', pool),
        ('new', [middle], 'publish', shared)])
    reader._status_matrix_cache = None
    def matrix():
        return query._strict_trade_status(reader, TABLE, ['A', 'X'], ['2024-01-03 10:00'], matrix=True)
    first = matrix()
    assert first.A.tolist() == [1] and first.X.isna().all()
    assert str(first.A.dtype) == 'Int8'
    first.loc[:, 'A'] = 0
    assert matrix().A.tolist() == [1]  # Caller edits cannot poison the cache.
    reader.tables['reference_publication_event'] = [row for row in reader.tables['reference_publication_event'] if row['revision_id'] != 'new']
    assert matrix().A.tolist() == [0]
    reader.tables[TABLE][0]['suspend_type'] = 'R'
    with pytest.raises(VersionConflictError):matrix()


def test_missing_or_tampered_evidence_never_becomes_normal():
    scope, pool = compact()
    broken = deepcopy(pool)
    broken.pop(scope['evidence']['calendar_ref'])
    reader = reference_reader('suspension', [('range', [scope], 'publish', broken)])
    with pytest.raises(VersionConflictError, match='共享证据缺失'):
        query.query_trade_status(reader, TABLE, ['A'], ['2024-01-02'])


def test_tampered_request_cannot_hide_a_published_revision_from_filtering():
    scope, pool = compact()
    reader = reference_reader('suspension', [('range', [scope], 'publish', pool)])
    original = json.loads(reader.tables['reference_revision'][0]['payload'])
    event = json.loads(reader.tables['reference_publication_event'][0]['payload'])
    event['revision_hash'] = digest(original)
    reader.tables['reference_publication_event'][0]['payload'] = json.dumps(event)
    original['request'] = {'security_codes': ['OTHER'], 'start_date': '2099-01-01'}
    reader.tables['reference_revision'][0]['payload'] = json.dumps(original)
    with pytest.raises(VersionConflictError, match='版本头哈希'):
        query.query_trade_status(reader, TABLE, ['A'], ['2024-01-02'])
