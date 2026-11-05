"""Resolve industry parents only through the archived, scheme-local catalog."""
from .util import digest

POLICY = 'industry-parent-index-namespace-v1'


def normalize_parents(rows):
    aliases = {}
    levels = {}
    for row in rows:
        if row.get('source_id') != 'tushare' or row.get('_binding_identity'):continue
        scheme, code = row.get('scheme_code'), row.get('industry_code')
        if not scheme or not code:continue
        levels.setdefault((scheme, code), set()).add(row.get('level'))
        classification = row.get('classification_code')
        if classification:
            aliases.setdefault((scheme, classification), set()).add(code)
    for row in rows:
        if row.get('source_id') != 'tushare' or row.get('_binding_identity') or row.get('source_endpoint') != 'index_classify':continue
        scheme, parent, level = row.get('scheme_code'), row.get('parent_code'), row.get('level')
        if parent in (None, '', '0'):
            if parent == '0' and level != 'L1':
                raise ValueError('行业父级为0但当前行业不是L1')
            normalized = None
        else:
            candidates = set(aliases.get((scheme, parent), ()))
            if (scheme, parent) in levels:candidates.add(parent)
            if len(candidates) != 1:
                raise ValueError(f'行业父代码缺少同体系唯一映射：{scheme}/{parent}')
            normalized = next(iter(candidates))
            expected = {'L2':'L1', 'L3':'L2'}.get(level)
            if normalized == row.get('industry_code') or expected is None or levels.get((scheme, normalized)) != {expected}:
                raise ValueError(f'行业父子层级不一致：{scheme}/{row.get("industry_code")} → {normalized}')
        if parent == normalized:continue
        row['parent_code'] = normalized
        row['mapping_version'] = digest([row.get('mapping_version'), POLICY])
        if '_field_origins' in row:
            row['_field_origins']['parent_code'] = {
                'selected':normalized is not None, 'source_null':parent in (None, ''),
                'value_hash':digest(normalized),
                'origins':['tushare:index_classify:'+name for name in ('parent_code','industry_code','index_code')]}
