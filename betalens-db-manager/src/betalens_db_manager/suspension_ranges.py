"""Versioned suspension coverage, independent of physical daily status rows.

Ranges prove which dates were queried, never that an order can execute. Shared
source evidence belongs to the immutable revision manifest. Legacy daily scopes
remain readable and compete with ranges by target-local ingestion sequence.
"""
from collections import defaultdict
from datetime import date, timedelta

from .util import digest

FORMAT = 'suspension-ranges-v2'


def bounds(selector):
    day = selector.get('trade_date')
    if day:
        first = date.fromisoformat(str(day)[:10])
        return first.isoformat(), (first + timedelta(days=1)).isoformat()
    return selector['start_date'], selector['end_date']


def overlaps(left, right):
    if left.get('security_code') != right.get('security_code'):
        return False
    a, b = bounds(left)
    c, d = bounds(right)
    return a < d and c < b


def contains(selector, code, day):
    if selector.get('security_code') != code:
        return False
    first, last = bounds(selector)
    return first <= day < last


def coverage_heads(scopes, heads, wanted):
    by_code = defaultdict(list)
    for scope in wanted:
        if scope['module'] == 'suspension':
            by_code[scope['selector']['security_code']].append(scope['selector'])
    return {scope['scope_id']: heads[scope['scope_id']] for scope in scopes
            if scope['module'] == 'suspension' and scope['scope_id'] in heads
            and any(overlaps(scope['selector'], selector)
                    for selector in by_code.get(scope['selector']['security_code'], []))}


def resolve(pool, reference):
    value = pool.get(reference)
    if value is None:
        raise ValueError('停牌共享证据缺失')
    return value


def day_scope(scope, day, pool):
    """Hydrate after manifest verification; never hash this derived day view."""
    if scope.get('format') != FORMAT:
        return scope
    evidence = dict(scope['evidence'])
    calendar = resolve(pool, evidence.pop('calendar_ref'))
    evidence['calendar'] = calendar.get(day)
    evidence['lifecycle'] = resolve(pool, evidence.pop('lifecycle_ref'))
    evidence['queries'] = resolve(pool, evidence.pop('queries_ref'))
    return {**scope, 'selector': {'security_code': scope['selector']['security_code'],
                                'trade_date': day}, 'evidence': evidence,
            'rows': [r for r in scope.get('rows', []) if str(r['trade_date'])[:10] == day]}


def referenced_pool(scopes, pool):
    references = {s['evidence'][key] for s in scopes if s.get('format') == FORMAT
                  for key in ('calendar_ref', 'lifecycle_ref', 'queries_ref')}
    selected = {ref: resolve(pool, ref) for ref in sorted(references)}
    if any(digest(value) != ref for ref, value in selected.items()):
        raise ValueError('停牌共享证据哈希不匹配')
    return selected


class SuspensionIndex:
    """Index physical heads once, then resolve only a security's range heads."""
    def __init__(self, heads):
        self.daily = {}
        self.ranges = defaultdict(list)
        verified = set()
        for head in heads:
            _, scope, revision, _ = head
            if scope.get('module') != 'suspension':
                continue
            selector = scope['selector']
            if selector.get('trade_date'):
                self.daily[(selector['security_code'], selector['trade_date'])] = head
            else:
                if scope.get('format') != FORMAT:
                    raise ValueError('停牌范围格式未登记')
                if revision.get('revision_id') not in verified:
                    if any(digest(value) != ref for ref, value in revision.get('evidence_pool', {}).items()):
                        raise ValueError('停牌共享证据哈希不匹配')
                    verified.add(revision.get('revision_id'))
                self.ranges[selector['security_code']].append(head)

    def head(self, code, day):
        candidates = [h for h in self.ranges.get(code, []) if contains(h[1]['selector'], code, day)]
        legacy = self.daily.get((code, day))
        if legacy:
            candidates.append(legacy)
        if not candidates:
            return None
        rank, scope, revision, rows = max(candidates, key=lambda h: h[0])
        view = day_scope({**scope, 'rows': rows}, day, revision.get('evidence_pool', {}))
        return rank, view, revision, view['rows']


def build_scopes(request, bundles):
    """Build at most one range per code/month/query-proof segment.

    Calendar evidence is normalized once per exchange/date and event rows once
    per source record, rather than scanning every raw response per stock-day.
    Incomplete responses remain separate incomplete ranges, including empties.
    """
    from .reference_domains import (_days, _covers, _proof, _lifecycle,
        _calendar_scope, _suspension_rows, _scope)

    days = _days(request)
    pool = {}

    def intern(value):
        ref = digest(value)
        pool.setdefault(ref, value)
        return ref

    exchanges = {code[-2:]: {'SH': 'SSE', 'SZ': 'SZSE', 'BJ': 'BSE'}[code[-2:]]
                 for code in request['security_codes']}
    calendars = {}
    for suffix, exchange in exchanges.items():
        calendar = {}
        for day in days:
            scope = _calendar_scope(exchange, day, bundles)
            calendar[day.isoformat()] = {k: scope[k] for k in ('complete', 'scope_id', 'rows', 'evidence')}
        calendars[suffix] = intern(calendar)

    event_bundles = [b for b in bundles if b.get('endpoint') == 'suspend_d']
    events = defaultdict(list)
    for bundle in event_bundles:
        for number, row in enumerate(bundle.get('rows', []), 1):
            events[(row.get('ts_code'), str(row.get('trade_date')).replace('-', '')[:8])].append((bundle, number, row))

    scopes = []
    proof_cache = {}
    for code in request['security_codes']:
        lifecycle_ref = intern(_lifecycle(code, bundles))
        segment = None
        for day in days:
            candidates = [b for b in event_bundles if _covers(b, code=code, day=day)]
            complete_bundles = [b for b in candidates if b.get('complete')]
            filters = {str((b.get('parameters') or {}).get('suspend_type') or '').upper() for b in complete_bundles}
            complete = bool(complete_bundles) and ('' in filters or {'S', 'R'} <= filters) and all(b.get('complete') for b in candidates)
            proof_key = tuple(id(b) for b in candidates)
            if proof_key not in proof_cache:
                proof_cache[proof_key] = intern(_proof(candidates))
            queries_ref = proof_cache[proof_key]
            signature = (day.year, day.month, queries_ref, complete)
            if segment is None or segment[0] != signature:
                selector = {'security_code': code, 'start_date': day.isoformat(),
                            'end_date': (day + timedelta(days=1)).isoformat()}
                evidence = {'event_query_complete': complete, 'calendar_ref': calendars[code[-2:]],
                    'lifecycle_ref': lifecycle_ref, 'queries_ref': queries_ref, 'timezone': 'Asia/Shanghai',
                    'interval_convention': 'left_closed_right_open',
                    'source_intraday_endpoint_inclusivity': 'unknown',
                    'dependency_basis': 'self_contained_archived_source_queries'}
                scope = _scope('suspension', selector, [], complete,
                    [] if complete else ['query_incomplete' if candidates else 'not_requested'], evidence)
                scope['format'] = FORMAT
                scopes.append(scope)
                segment = (signature, scope)
            scope = segment[1]
            scope['selector']['end_date'] = (day + timedelta(days=1)).isoformat()
            selected = [(b, n, r) for b, n, r in events.get((code, day.strftime('%Y%m%d')), [])
                        if any(b is candidate for candidate in candidates)]
            scope['rows'].extend(_suspension_rows(code, day, selected))
        for scope in scopes:
            if scope['selector']['security_code'] == code:
                scope['scope_id'] = digest({'module': 'suspension', 'selector': scope['selector']})
                if scope['complete'] and not scope['rows']:
                    scope['reasons'] = ['source_no_record']
    return scopes, pool
