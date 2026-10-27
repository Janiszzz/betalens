"""Bindings to the shipped executors, never expressions supplied by a mapping."""
from functools import lru_cache

VERSION = 'registered-processors-v3'
DETAIL_ENDPOINTS = {'stock_basic':'stock_profile','fund_basic':'fund_profile','index_basic':'index_profile',
    'dividend':'dividend_event','suspend_d':'suspension_event','namechange':'security_name_history',
    'fund_nav':'fund_nav_observation','index_member_all':'industry_membership_record','ci_index_member':'industry_membership_record'}


@lru_cache(maxsize=1)
def registry():
    from .canonical import builtin_catalog, registry_context
    from .mapping import all_source_mappings
    result = {}
    with registry_context(None):
        for rule in all_source_mappings():
            table, endpoint = rule['canonical_table'], rule['source_endpoint']
            if endpoint in {'etf_basic','bak_daily'}:continue  # These use the public direct mapper.
            if table not in builtin_catalog():
                continue
            entry = result.setdefault((table, endpoint), {'inputs':set(), 'outputs':set(), 'formulas':set()})
            source, column = rule['source_field'], rule['canonical_column']
            entry['inputs'].add(source); entry['outputs'].add(column)
            formula = rule.get('transform_expression')
            if table in {'industry_dim','security_industry_membership'} and source.startswith(('l1_','l2_','l3_')):
                formula = 'expand_industry_levels'
            if formula in {'native_index_value','Decimal(raw_price)*Decimal(adjustment_factor)','expand_industry_levels'}:
                entry['formulas'].add((source, column, formula))
        for endpoint, table in DETAIL_ENDPOINTS.items():
            result.setdefault((table, endpoint), {'inputs':set(), 'outputs':set(), 'formulas':set()})
        for endpoint in ('index_classify','index_member_all','ci_index_member'):
            result.setdefault(('industry_scheme_dim',endpoint), {'inputs':set(), 'outputs':{'scheme_code'}, 'formulas':set()})
        for endpoint in ('daily','fund_daily','index_daily','sw_daily'):
            result.setdefault(('trade_status_daily',endpoint), {'inputs':{'ts_code','trade_date'}, 'outputs':{'security_code','trade_date'}, 'formulas':set()})
    return result


def resolve(table, binding):
    """Missing mode in old, recognized seed bindings means the shipped executor."""
    entry = registry().get((table, binding.get('endpoint'))) if binding.get('source_id') == 'tushare' else None
    mode = binding.get('mapping_mode', 'builtin' if entry else 'direct')
    if mode == 'builtin' and entry is None:
        raise ValueError('没有已注册内置处理器：'+table+':'+str(binding.get('endpoint')))
    return entry if mode == 'builtin' else None


def validate_binding(table, binding):
    entry = resolve(table, binding)
    for rule in binding.get('fields', []):
        if rule.get('builtin'):
            identity = (rule['source_field'], rule['column'], rule.get('formula'))
            if entry is None or identity not in entry['formulas']:
                raise ValueError('字段没有匹配的内置处理器：'+rule['source_field']+' → '+rule['column'])
        elif rule.get('formula'):
            raise ValueError('公式只能用于已注册内置处理器')
    return entry


def validate_output(binding, row):
    """A missing factor is explicit unavailable data, not a successful conversion."""
    for rule in binding.get('fields', []):
        if not rule.get('builtin'):
            continue
        column, formula = rule['column'], rule['formula']
        if column not in row:
            raise ValueError('内置处理器未产生声明输出：'+column)
        required = rule.get('required', False)
        raw = row.get('raw_'+column)
        expected = raw is not None and (formula == 'native_index_value' or
            formula == 'Decimal(raw_price)*Decimal(adjustment_factor)' and row.get('adjustment_factor') is not None)
        if (required or expected) and row[column] is None:
            raise ValueError('内置处理器输出为空：'+column)
