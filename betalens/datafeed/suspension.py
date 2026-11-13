"""Resolve legacy daily scopes and compact coverage without importing DB Manager."""
from collections import defaultdict

from .contracts import VersionConflictError, digest, proven, local_time

FORMAT = 'suspension-ranges-v2'


class SuspensionScopes:
    def __init__(self, scopes):
        self.by_code = defaultdict(list)
        self.rows = {}
        verified = set()
        for scope in scopes:
            if not scope['selector'].get('trade_date') and scope.get('format') != FORMAT:
                raise VersionConflictError('停牌范围格式未登记')
            self.by_code[scope['selector']['security_code']].append(scope)
            if scope.get('format') != FORMAT:
                continue
            rid = scope['revision_id']
            pool = scope.get('_evidence_pool', {})
            if rid not in verified:
                if any(digest(value) != key for key, value in pool.items()):
                    raise VersionConflictError('停牌共享证据哈希不匹配')
                verified.add(rid)
            grouped = defaultdict(list)
            for row in scope['rows']:
                grouped[str(row['trade_date'])[:10]].append(row)
            self.rows[(rid, scope['scope_id'])] = grouped

    def at(self, code, at, policy='strict'):
        day = local_time(at).date().isoformat()
        candidates = []
        for scope in self.by_code.get(code, []):
            selector = scope['selector']
            if selector.get('trade_date'):
                if selector['trade_date'] != day:
                    continue
            elif not selector['start_date'] <= day < selector['end_date']:
                continue
            if not scope.get('complete'):
                continue
            if policy == 'strict' and (not proven(scope.get('evidence', {}).get('availability', {}), at)
                                      or not all(proven(row, at) for row in scope['rows'])):
                continue
            candidates.append(scope)
        if not candidates:
            return None
        latest = max(candidates, key=lambda s: int(s['ingestion_seq']))
        if any(s['ingestion_seq'] == latest['ingestion_seq'] and s['revision_id'] != latest['revision_id'] for s in candidates):
            raise VersionConflictError('同一停牌日期的版本顺序冲突')
        if latest.get('format') != FORMAT:
            return latest
        evidence = dict(latest['evidence'])
        pool = latest['_evidence_pool']
        try:
            evidence['calendar'] = pool[evidence.pop('calendar_ref')].get(day)
            evidence['lifecycle'] = pool[evidence.pop('lifecycle_ref')]
            evidence['queries'] = pool[evidence.pop('queries_ref')]
        except KeyError as exc:
            raise VersionConflictError('停牌共享证据缺失') from exc
        return {**latest, 'selector': {'security_code': code, 'trade_date': day},
                'evidence': evidence, 'rows': self.rows[(latest['revision_id'], latest['scope_id'])].get(day, [])}
