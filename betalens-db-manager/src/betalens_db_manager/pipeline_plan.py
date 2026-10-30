"""Bounded request planning, independent of target database contents."""
from copy import deepcopy
from datetime import date,timedelta
from .pipeline_options import options,DEFAULTS
from .retrieval import tushare_policy,universe_hint_key,security_partitions,CODE_BATCHES
from .util import parse_date,digest

DAILY={'daily','daily_basic','adj_factor','fund_daily','fund_adj','fund_share','bak_daily'}
FINANCIAL={'income','balancesheet','cashflow','fina_indicator'}


def windows(start,end,days):
    while start<=end:
        stop=min(end,start+timedelta(days=days-1));yield start,stop;start=stop+timedelta(days=1)


def params_for(endpoint,request,custom):
    a,b=parse_date(request['start_date']),parse_date(request['end_date'])
    codes=request.get('codes',[])
    period=lambda x:x.strftime('%Y%m%d')
    if endpoint=='stock_basic':
        for status in ([custom['list_status']] if custom.get('list_status') else ['L','D','P','G','UN']):yield {'list_status':status,**custom}
    elif endpoint=='trade_cal':
        for exchange in ([custom['exchange']] if custom.get('exchange') else ['SSE','SZSE']):
            if exchange=='BSE':exchange='SSE'
            for lo,hi in windows(a,b,5400):yield {**custom,'exchange':exchange,'start_date':period(lo),'end_date':period(hi)}
    elif endpoint=='fund_basic':
        for market in ([custom['market']] if custom.get('market') else ['E','O']):
            for status in ([custom['status']] if custom.get('status') else ['L','D','I']):yield {**custom,'market':market,'status':status}
    elif endpoint=='index_basic':
        for market in ([custom['market']] if custom.get('market') else ['SSE','SZSE','CSI','SW','CICC','OTH','MSCI','CNI']):yield {**custom,'market':market}
    elif endpoint=='etf_basic':
        for status in ([custom['list_status']] if custom.get('list_status') else ['L','D','P']):yield {**custom,'list_status':status}
    elif endpoint in {'index_member_all','ci_index_member'}:
        for flag in ([custom['is_new']] if custom.get('is_new') else ['Y','N']):
            for code in codes or [None]:yield {**custom,'is_new':flag,**({'ts_code':code} if code else {})}
    elif endpoint=='index_classify':yield {'src':'SW2021',**custom}
    elif endpoint=='fund_nav':
        if codes or custom.get('ts_code'):
            for code in codes or [custom['ts_code']]:yield {**custom,'ts_code':code,'start_date':period(a),'end_date':period(b)}
        else:
            for lo,_ in windows(a,b,1):
                for market in ([custom['market']] if custom.get('market') else ['E','O']):yield {**custom,'nav_date':period(lo),'market':market}
    elif endpoint in CODE_BATCHES and codes:
        yield from security_partitions(endpoint,{**custom,'start_date':period(a),'end_date':period(b)},codes,tushare_policy(endpoint,5000))
    elif endpoint=='index_daily' and not codes and not custom.get('ts_code'):
        for lo,_ in windows(a,b,1):yield {**custom,'trade_date':period(lo)}
    elif endpoint in DAILY:
        if codes:
            for code in codes:yield {**custom,'ts_code':code,'start_date':period(a),'end_date':period(b)}
        else:
            for lo,_ in windows(a,b,1):yield {**custom,'trade_date':period(lo)}
    elif endpoint=='suspend_d':
        lo=a
        while lo<=b:
            hi=min(b,(lo.replace(day=28)+timedelta(days=4)).replace(day=1)-timedelta(days=1))
            yield {**custom,'start_date':period(lo),'end_date':period(hi)}
            lo=hi+timedelta(days=1)
    elif endpoint.endswith('_vip'):
        for year in range(a.year,b.year+1):
            for month,day in ((3,31),(6,30),(9,30),(12,31)):
                d=date(year,month,day)
                if a<=d<=b:yield {'period':period(d),'offset':0,'limit':6000,**custom}
    elif endpoint in FINANCIAL:
        for code in codes or [None]:
            yield {**custom,**({'ts_code':code} if code else {}),'start_date':period(a),'end_date':period(b)}
    elif endpoint in {'index_daily','sw_daily','index_weight','fund_nav','namechange','dividend'}:
        for code in codes or [None]:
            if endpoint in {'index_daily','sw_daily'} and code and (code.endswith('.SI')!=(endpoint=='sw_daily')):continue
            p={**custom,**({('index_code' if endpoint=='index_weight' else 'ts_code'):code} if code else {})}
            if endpoint not in {'namechange','dividend'}:p.update(start_date=period(a),end_date=period(b))
            yield p
    else:yield dict(custom)


def compile_plan(service,request):
    opts=options(request)
    # Compile one tiny seed to freeze mappings/contracts without an N securities × D days expansion.
    seed={k:v for k,v in request.items() if k not in DEFAULTS}
    seed['end_date']=seed.get('start_date') or seed.get('end_date')
    original_end=request.get('end_date')
    if not original_end:
        from .service import SHANGHAI
        from datetime import datetime
        original_end=str(datetime.now(SHANGHAI).date()-timedelta(days=1))
    from .pipeline_options import PLANNING_SEED
    token=PLANNING_SEED.set(True)
    try:plan=service.compile(seed)
    finally:PLANNING_SEED.reset(token)
    normalized=plan['request'];normalized.update(start_date=request.get('start_date',normalized['start_date']),end_date=original_end,codes=list(request.get('codes',[])),**opts)
    normalized['requests_per_minute']=min(int(request.get('requests_per_minute',450)),450)
    endpoints={b['endpoint'] for m in plan['mapping_snapshots'].values() for b in m['bindings']}
    # Full-market finance uses the explicitly mapped 5000-point batch interface.
    if not normalized['codes'] and not request.get('endpoints'):
        for endpoint in FINANCIAL:
            if endpoint+'_vip' in endpoints:endpoints.discard(endpoint)
    roots=[];seen=set();templates=list(plan['requests'])
    present={(r['source_id'],r['endpoint']) for r in templates}
    # A one-day seed may have no fiscal quarter; selected bindings must still
    # generate their real full-range requests from the frozen contracts.
    from .response_validation import scope_roles
    for name,mapping in plan['mapping_snapshots'].items():
        for binding in mapping['bindings']:
            sid,endpoint=binding['source_id'],binding['endpoint']
            if (sid,endpoint) in present:continue
            contract=plan['source_contracts'][sid+':'+endpoint]
            root={'table':name,'source_id':sid,'endpoint':endpoint,'parameters':{},'strategy':contract.get('strategy','snapshot'),'priority':binding.get('priority',0)}
            root['validation_scope']=scope_roles(plan,root,contract);templates.append(root);present.add((sid,endpoint))
    for root in templates:
        sid,endpoint=root['source_id'],root['endpoint']
        if sid=='tushare' and not root.get('strategy')=='dependency' and endpoint not in endpoints:continue
        if endpoint=='daily_basic' and endpoint not in endpoints:continue
        if sid=='tushare':
            contract=plan['source_contracts'][sid+':'+endpoint]
            binding=next((b for m in plan['mapping_snapshots'].values() for b in m['bindings'] if b['source_id']==sid and b['endpoint']==endpoint),{})
            custom={**contract.get('parameters',{}),**binding.get('parameters',{}),**request.get('parameters',{}).get(endpoint,{})}
            for p in params_for(endpoint,normalized,custom):
                r={**root,'parameters':p}
                if not normalized['codes'] and not (p.get('ts_code') or p.get('index_code')) and not p.get('trade_date') and endpoint in FINANCIAL|{'index_daily','index_weight','namechange','dividend'}:
                    r['strategy']='universe_expansion'
                key=digest([sid,endpoint,p])
                if key not in seen:roots.append(r);seen.add(key)
        else:
            r=deepcopy(root);contract=plan['source_contracts'][sid+':'+endpoint];pb=contract.get('parameter_bindings',{})
            for k in ('start_date','end_date'):
                key=pb.get(k,k)
                if key in r['parameters']:r['parameters'][key]=parse_date(normalized[k]).strftime(pb.get('date_format',contract.get('date_format','%Y-%m-%d')))
            key=digest([sid,endpoint,r['parameters']])
            if key not in seen:roots.append(r);seen.add(key)
    # Freeze complete universe contracts for expansions, including off-exchange funds and indices.
    for endpoint in {r['endpoint'] for r in roots if r['source_id']=='tushare' and r['strategy']=='universe_expansion'}:
        dim='fund_basic' if endpoint.startswith('fund_') else 'index_basic' if endpoint.startswith('index_') else 'stock_basic'
        plan['source_contracts']['tushare:'+dim]=deepcopy(service.sources.endpoint('tushare',dim))
    plan['requests']=roots;plan['pipeline_version']=2;plan['cohort_strategy']='security_batches_v1'
    plan['retrieval_policies']={k:tushare_policy(k.split(':',1)[1],points=5000) for k in plan['source_contracts'] if k.startswith('tushare:')}
    for policy in plan['retrieval_policies'].values():policy['effective_rpm']=min(normalized['requests_per_minute'],policy['endpoint_rpm'])
    plan['request_estimate']=estimate_requests(service,plan)
    return plan


def estimate_requests(service,plan):
    roots=plan['requests'];normalized=plan['request']
    dynamic=sorted({r['endpoint'] for r in roots if r['strategy']=='universe_expansion'})
    expansions=[]
    from datetime import datetime,timezone
    for endpoint in dynamic:
        asset='index' if endpoint.startswith('index_') else 'stock'
        hint=service.state.ledger_get('source_universe_estimate',universe_hint_key(plan,asset))
        try:fresh=hint and 0<=(datetime.now(timezone.utc)-datetime.fromisoformat(hint['checked_at'])).total_seconds()<86400
        except (ValueError,KeyError):fresh=False
        count=(hint.get('non_sw_count') if endpoint=='index_daily' else hint.get('count')) if fresh else None
        expansions.append({'endpoint':endpoint,'securities':count,'estimated_requests':count,'basis':'已验证来源清单的近一日参考值' if fresh else '等待来源证券清单，不能把入口数当作总次数'})
    estimated=(len(roots)-len(dynamic)+sum(e['estimated_requests'] for e in expansions)) if all(e['estimated_requests'] is not None for e in expansions) else None
    return {'initial_requests':len(roots),'includes_dynamic_partitions':False,'account_points':5000,'account_rpm':normalized['requests_per_minute'],
        'estimated_requests_before_paging':estimated,'total_requests':None,'total_is_lower_bound':True,'security_expansion_endpoints':dynamic,'security_expansions':expansions,
        'warnings':[{'endpoint':n,'reason':'该接口的完整历史查询仍需按证券展开；初始请求数不是总次数，运行前会登记实际证券数量。'} for n in dynamic]}
