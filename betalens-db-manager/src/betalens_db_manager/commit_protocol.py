"""Immutable write preconditions, shared by fresh commits and recovery."""
from .comparison import typed, typed_row_hash, typed_row_key


class NeedsReconciliation(ValueError):
    """A durable intent cannot be safely interpreted as a completed operation."""


class TargetConflict(NeedsReconciliation):
    """Known later changes must be preserved; no inverse write is permitted."""


def manifest_rows(manifest):
    checks=manifest.get('checks') or [
        {'key':c['after'] or c['before'],'row_key':c['row_key'],'before_hash':c['before_hash']}
        for c in manifest['changes']]
    return [c['key'] for c in checks]


def remaining_changes(spec,manifest,existing):
    """Check the original before images; never grant a new baseline.

    A matching after image is a replay of this exact intent (provenance is part
    of the hash), so it requires no second write. Skipped keys remain guarded.
    """
    live={}
    for row in existing:
        normalized=typed(spec,row)
        live[typed_row_key(spec,normalized)]=typed_row_hash(normalized)
    changes={c['row_key']:c for c in manifest['changes']}
    checks=manifest.get('checks') or [
        {'row_key':c['row_key'],'before_hash':c['before_hash']} for c in manifest['changes']]
    for check in checks:
        actual=live.get(check['row_key'])
        change=changes.get(check['row_key'])
        if actual==check['before_hash']:continue
        if change and actual==change['after_hash']:continue
        raise NeedsReconciliation(f'{spec.name}: 目标数据已变化，固定提交前提不再成立，请重新审计')
    return [c for c in manifest['changes'] if
            live.get(c['row_key'])!=c['after_hash']]
