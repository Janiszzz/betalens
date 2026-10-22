"""Versioned retrieval rules, separate from immutable field/mapping contracts."""
from __future__ import annotations

from datetime import timedelta

from .catalog import load_catalog
from .util import parse_date


def request_rate(value):
    if isinstance(value, bool) or not str(value).isdigit() or not 1 <= int(value) <= 100000:
        raise ValueError('每分钟请求次数须为 1 至 100000 的整数；实际权限由数据源决定')
    return int(value)


def tushare_policy(endpoint):
    spec = load_catalog()[endpoint]
    cap = {'bse_mapping': 1000, 'fund_daily': 5000, 'index_basic': 8000}.get(endpoint, spec.limit)
    mode = 'offset' if endpoint == 'fund_adj' else 'offset_probe' if endpoint.endswith('_vip') or endpoint in {'fund_basic', 'index_basic'} else 'partition'
    return {'version': '2026-10-21.1', 'endpoint': endpoint, 'row_limit': cap,
            'mode': mode, 'endpoint_rpm': 50 if endpoint == 'stock_basic' else None,
            'basis': 'documented_offset' if mode == 'offset' else 'runtime_verified_offset' if mode == 'offset_probe' else 'documented_filters',
            'limit_basis': 'conservative_guard_not_published' if endpoint == 'fund_nav' else 'documented_or_verified_cap',
            'document': spec.document}


def partition(endpoint, params, contract):
    """Return exhaustive disjoint filters, never derive partitions from a truncated page."""
    allowed = set(contract.get('input_fields', []))
    if {'start_date', 'end_date'} <= allowed and params.get('start_date') and params.get('end_date'):
        start, end = parse_date(params['start_date']), parse_date(params['end_date'])
        if start < end:
            middle = start + (end-start)//2
            return [{**params, 'start_date': start.strftime('%Y%m%d'), 'end_date': middle.strftime('%Y%m%d')},
                    {**params, 'start_date': (middle+timedelta(days=1)).strftime('%Y%m%d'), 'end_date': end.strftime('%Y%m%d')}]
    if isinstance(params.get('ts_code'), str) and ',' in params['ts_code']:
        return [{**params, 'ts_code': code.strip()} for code in dict.fromkeys(params['ts_code'].split(',')) if code.strip()]
    filters = {
        'stock_basic': [('list_status', ['L', 'D', 'P', 'G', 'UN']), ('exchange', ['SSE', 'SZSE', 'BSE'])],
        'fund_basic': [('market', ['E', 'O']), ('status', ['L', 'D', 'I'])],
        'index_basic': [('market', ['MSCI', 'CSI', 'SSE', 'SZSE', 'CICC', 'SW', 'OTH', 'CNI'])],
        'index_classify': [('level', ['L1', 'L2', 'L3'])],
    }
    for field, values in filters.get(endpoint, []):
        if not params.get(field):
            return [{**params, field: value} for value in values]
    return []
