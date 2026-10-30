"""Versioned retrieval rules, separate from immutable field/mapping contracts."""
from __future__ import annotations

from datetime import timedelta

from .catalog import load_catalog
from .util import parse_date

CODE_BATCHES={'daily':1000,'adj_factor':1000,'fund_share':1000}


def security_partitions(endpoint,params,codes,policy):
    """Pack verified multi-code filters below the packet guard; keep scope intact."""
    codes=list(dict.fromkeys(codes));size=policy.get('code_batch_size',1)
    if params.get('trade_date') or params.get('nav_date'):days=1
    elif params.get('start_date') and params.get('end_date'):days=max(1,(parse_date(params['end_date'])-parse_date(params['start_date'])).days+1)
    else:days=policy.get('row_limit',1)
    size=max(1,min(size,max(1,(policy.get('row_limit',1)-1)//days)))
    return [{**params,'ts_code':','.join(codes[i:i+size])} for i in range(0,len(codes),size)]


def universe_spec(asset):
    endpoint='index_basic' if asset=='index' else 'fund_basic' if asset in {'fund','fund_all'} else 'stock_basic'
    filters=([{'market':m} for m in ('SSE','SZSE','CSI','SW','CICC','OTH','MSCI','CNI')] if asset=='index' else
             [{'market':m,'status':s} for m in (('E','O') if asset=='fund_all' else ('E',)) for s in ('L','D','I')] if asset in {'fund','fund_all'} else
             [{'list_status':s} for s in ('L','D','P','G','UN')])
    return endpoint,filters


def universe_hint_key(plan,asset):
    from .util import digest
    endpoint,_=universe_spec(asset)
    return digest([plan.get('source_instances',{}).get('tushare'),asset,plan.get('source_contracts',{}).get('tushare:'+endpoint)])


def request_rate(value):
    if isinstance(value, bool) or not str(value).isdigit() or not 1 <= int(value) <= 100000:
        raise ValueError('每分钟请求次数须为 1 至 100000 的整数；实际权限由数据源决定')
    return int(value)


def tushare_policy(endpoint, points=None):
    spec = load_catalog()[endpoint]
    cap = {'bse_mapping': 1000, 'fund_daily': 5000, 'index_basic': 8000}.get(endpoint, spec.limit)
    mode = 'offset' if endpoint in {'fund_adj', 'bak_daily'} else 'offset_probe' if endpoint.endswith('_vip') or endpoint in {'fund_basic', 'index_basic'} else 'partition'
    vip_probe = endpoint.endswith('_vip')
    result = {'version': '2026-10-25.1', 'endpoint': endpoint, 'row_limit': cap,
            'mode': mode, 'endpoint_rpm': 50 if endpoint == 'stock_basic' else None,
            'basis': 'runtime_probe_required' if vip_probe else 'documented_offset' if mode == 'offset' else 'runtime_verified_offset' if mode == 'offset_probe' else 'documented_filters',
            'limit_basis': 'conservative_guard_not_published' if endpoint == 'fund_nav' or vip_probe else 'documented_or_verified_cap',
            'document': spec.document}
    if points is not None:
        result.update(version='2026-10-28.1',points=5000,rate_strategy='adaptive',endpoint_rpm=50 if endpoint in {'stock_basic','fund_daily','etf_basic'} else 500,
                      max_security_fanout=2000,
                      limit_basis='protective_guard' if endpoint in {'trade_cal','adj_factor','namechange','index_classify','income','balancesheet','cashflow','fund_nav'} or vip_probe else result['limit_basis'])
        if vip_probe:result.update(row_limit=6000,limit_basis='runtime_page_size_probe_not_official_cap')
        if endpoint in {'fund_share','fund_nav','index_daily'}:
            result.update(mode='offset_probe',row_limit={'fund_nav':10000,'fund_share':2000,'index_daily':8000}[endpoint],
                          basis='runtime_advancing_pages_verified_20261028',limit_basis='requested_page_size_not_official_cap' if endpoint=='fund_nav' else result['limit_basis'])
        if endpoint=='index_daily':result['scope_basis']='runtime_global_trade_date_verified_20261028'
        if endpoint in CODE_BATCHES:result.update(code_batch_size=CODE_BATCHES[endpoint],batch_basis='documented_or_runtime_verified_multi_code_20261028')
    return result


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
        codes=[code.strip() for code in dict.fromkeys(params['ts_code'].split(',')) if code.strip()]
        middle=(len(codes)+1)//2
        return [{**params,'ts_code':','.join(group)} for group in (codes[:middle],codes[middle:]) if group]
    filters = {
        'stock_basic': [('list_status', ['L', 'D', 'P', 'G', 'UN']), ('exchange', ['SSE', 'SZSE', 'BSE'])],
        'fund_basic': [('market', ['E', 'O']), ('status', ['L', 'D', 'I'])],
        'index_basic': [('market', ['MSCI', 'CSI', 'SSE', 'SZSE', 'CICC', 'SW', 'OTH', 'CNI'])],
        'etf_basic': [('list_status', ['L', 'D', 'P']), ('exchange', ['SH', 'SZ'])],
        'index_classify': [('level', ['L1', 'L2', 'L3'])],
    }
    for field, values in filters.get(endpoint, []):
        if not params.get(field):
            return [{**params, field: value} for value in values]
    return []
