"""Declarative column producers and conservative, target-specific storage proof.

Rules describe shipped code; they never evaluate expressions from YAML.
"""
from collections import Counter
from functools import lru_cache
from .util import digest
from .registry import GENERATED

VERSION = 'column-producers-v1'
PROOF_VERSION = 'column-verification-v2-scope-context'
KINDS = {'direct','constant','copy','computed','system','not_applicable','unavailable'}
PROCESSORS = {'business_version','post_adjustment','daily_average','asset_scope','industry_scheme',
              'industry_membership','trade_status','event_summary','announcement_time','record_hash'}


def producers_for(spec, mapping):
    """Generate metadata only for known production paths, preserving custom rules."""
    from .canonical import builtin_catalog
    result=[];table=spec.name;builtin=table in builtin_catalog()
    for binding in mapping.get('bindings',[]):
        for rule in binding.get('fields',[]):
            result.append(dict(column=rule['column'],kind='computed' if rule.get('builtin') else 'direct',
                source_id=binding['source_id'],endpoint=binding['endpoint'],source_field=rule['source_field'],
                inputs=[binding['source_id']+':'+binding['endpoint']+'.'+rule['source_field']],
                scope={**binding.get('scope',{}),**({'statement_type':binding['endpoint'].removesuffix('_vip')} if table=='fundamental_periodic' else {})},processor='post_adjustment' if rule.get('builtin') and table=='market_daily_post_adjusted' else 'industry_membership' if rule.get('builtin') else None,
                multiplier=str(rule.get('multiplier','1')),reason=rule.get('formula',''),version=VERSION))
    covered={p['column'] for p in result}
    def add(column,kind,reason,inputs=(),processor=None,scope=None,**extra):
        result.append(dict(column=column,kind=kind,reason=reason,inputs=list(inputs),processor=processor,scope=scope or {},version=VERSION,**extra));covered.add(column)
    for f in spec.fields:
        c=f['name']
        if f.get('system') or c in GENERATED:
            add(c,'system','由任务、目标库及时间证据生成；未知可得时间保留空值')
        elif builtin and c=='scheme_code':add(c,'computed','按查询接口和显式分类版本生成；中信固定CITIC，申万固定SW2021',['source_endpoint'],'industry_scheme')
        elif table=='fundamental_periodic' and c=='report_type':add(c,'constant','财务指标接口不提供报告类型，使用显式unspecified',['source_endpoint'],scope={'statement_type':'fina_indicator'},value='unspecified')
        elif table=='market_daily_post_adjusted' and c=='adjustment_factor':add(c,'not_applicable','指数保留原生点位，不使用股票/基金复权因子',scope={'asset_type':'index'})
        elif table in {'market_daily_raw','market_daily_post_adjusted'} and c=='security_name':add(c,'copy','优先源行情名称，否则取证券档案名称；档案快照不证明历史名称',['security_dim.name','stock_basic.name','fund_basic.name','index_basic.name'])
        elif c in covered:continue
        elif not builtin:continue
        elif c=='pcf':add(c,'unavailable','已登记接口没有市现率；ETF申赎PCF不等于市现率')
        elif table=='security_code_history' and c=='valid_from':add(c,'unavailable','bse_mapping.list_date为上市日，不能证明代码切换生效日')
        elif c=='rd_exp':add(c,'unavailable','旧同名不同义字段已弃用；历史迁移需按报表类型和归档证明')
        elif c in {'record_version','version_id'}:add(c,'computed','完整规范化业务记录的内容哈希', ['source_response.*'], 'business_version')
        elif c=='statement_type':add(c,'computed','普通/VIP接口归一为相同报表类型',['source_endpoint'],'asset_scope')
        elif c=='asset_type':add(c,'computed','按查询合同的资产范围生成',['binding.scope.asset_type'],'asset_scope')
        elif table=='industry_scheme_dim':add(c,'computed','按SW2021/CITIC分别生成分类名称和说明',['scheme_code'],'industry_scheme')
        elif table=='security_industry_membership' and c=='membership_status':add(c,'computed','依据纳入、剔除日期派生成分状态',['in_date','out_date'],'industry_membership')
        elif table=='market_daily_raw' and (c.startswith('raw_') or c.startswith('adjustment_')):add(c,'not_applicable','原始行情不使用后复权辅助列')
        elif table=='market_daily_post_adjusted':
            if c=='average_price':add(c,'computed','原始均价乘复权因子；指数不适用',['market_daily_raw.average_price','adj_factor','fund_adj'],'post_adjustment',scope={'asset_type':['stock','fund']})
            elif c.startswith('adjustment_'):add(c,'computed','原生指数值或raw_price × adjustment_factor；缺因子不伪造',['market_daily_raw','adj_factor','fund_adj'],'post_adjustment')
            elif c.startswith('raw_'):add(c,'copy','保留原始行情以支持重算',['market_daily_raw.'+c.removeprefix('raw_')])
            else:add(c,'copy','继承本次原始行情和每日指标；成交量额不复权',['market_daily_raw.'+c,'daily_basic'])
        elif c=='average_price' and table=='market_daily_raw':add(c,'computed','成交额/成交量；指数不适用，零成交量保留空值',['amount','volume'],'daily_average',scope={'asset_type':['stock','fund']})
        elif table=='trade_status_daily':add(c,'computed','股票按交易日历/停复牌/上市边界；基金指数按行情存在性派生',['trade_cal','suspend_d','stock_basic','daily_quote'],'trade_status')
        elif table=='corporate_action':
            if c=='old_name':add(c,'unavailable','只在完整、无歧义的紧邻名称区间证明后生成；当前不推断')
            else:add(c,'computed','已实施分红以ex_date为生效日；改名以start_date；送股比例stk_bo_rate',['dividend','namechange','adj_factor','fund_adj'],'event_summary')
        else:add(c,'unavailable','当前已登记生产路径未提供此列；需补充来源或明确业务规则')
    return result


def validate_producers(producers,spec,sources):
    if not isinstance(producers,list):raise ValueError('producers 必须为列表')
    columns={f['name'] for f in spec.fields};seen=set()
    allowed={'column','kind','reason','inputs','processor','scope','source_id','endpoint','source_field','multiplier','value','version'}
    for p in producers:
        if not isinstance(p,dict) or set(p)-allowed:raise ValueError('生产规则含不支持字段；禁止任意表达式')
        if p.get('column') not in columns or p.get('kind') not in KINDS:raise ValueError('生产规则列名或类型无效')
        if p['column'] in GENERATED and p['kind']!='system':raise ValueError('审计字段只能由系统生成')
        if p.get('processor') is not None and p['processor'] not in PROCESSORS:raise ValueError('未登记列生产处理器')
        if p['kind']=='computed' and not p.get('processor'):raise ValueError('计算规则需要已登记处理器')
        if not isinstance(p.get('inputs',[]),list) or not all(isinstance(i,str) for i in p.get('inputs',[])):raise ValueError('生产依赖必须是文本列表')
        if p['kind'] in {'not_applicable','unavailable'} and not str(p.get('reason','')).strip():raise ValueError('留空规则必须说明原因')
        if p['kind']=='direct':
            contract=sources.endpoint(p.get('source_id'),p.get('endpoint'))
            if p.get('source_field') not in {f['name'] for f in contract['fields']}:raise ValueError('生产规则引用未登记源字段')
        identity=digest(p)
        if identity in seen:raise ValueError('生产规则重复')
        seen.add(identity)
    return producers


def reconcile_direct_producers(spec,mapping):
    """Direct declarations follow editable links, including source removal."""
    declared=mapping.get('producers',[])
    direct=[p for p in producers_for(spec,mapping) if p['kind']=='direct']
    return [p for p in declared if p['kind']!='direct']+direct


def applicability(producers,row,asset_types=None):
    """Evaluate declared scopes without inventing fields absent from a row.

    A scope is an AND of conditions; producers are alternatives. The task's
    pinned asset set can answer only an absent asset_type, never replace an
    actual row value or fill another unknown dimension such as statement_type.
    """
    active=[p for p in producers if p['kind'] not in {'unavailable','not_applicable'}]
    if not active:
        return 'not_applicable' if producers and all(p['kind']=='not_applicable' for p in producers) else 'unavailable'
    assets=set(asset_types) if isinstance(asset_types,(list,tuple,set,frozenset)) and asset_types else None
    unknown=False
    for p in active:
        scope=p.get('scope',{})
        conditions=[]
        for key,values in scope.items():
            allowed=values if isinstance(values,list) else [values]
            if key in row:
                conditions.append(row[key] in allowed)
            elif key=='asset_type' and assets is not None:
                conditions.append(bool(assets.intersection(allowed)))
            else:
                conditions.append(None)
        if False in conditions:continue
        if None in conditions:unknown=True;continue
        return 'applicable'
    return 'unknown' if unknown else 'not_applicable'


def applies(producers,row,asset_types=None):
    """Compatibility predicate; unknown scope never grants a storage proof."""
    return applicability(producers,row,asset_types)=='applicable'


@lru_cache(maxsize=1)
def financial_column_scopes():
    """Known source semantics survive endpoint filtering for audit only.

    A task may remove an unselected endpoint's direct producer from its frozen
    mapping. Its unrelated financial columns remain in the physical wide table.
    Deriving statement applicability here never adds bindings, values or traces.
    """
    from .catalog import load_catalog
    from .mapping import FIELD_MAP
    result={}
    for endpoint in ('income','balancesheet','cashflow','fina_indicator'):
        fields={f['name'] for f in load_catalog()[endpoint].fields}
        for source,(table,column) in FIELD_MAP.get(endpoint,{}).items():
            if source in fields and table=='fundamental_periodic':
                result.setdefault(column,set()).add(endpoint)
    return result


def coverage(service,name,job=None,target=None):
    spec=service.registry.spec(name);mapping=service.mappings.get(name)
    if job:
        item=service._job(job);mapping=item['plan'].get('mapping_snapshots',{}).get(name,mapping)
    producers=mapping.get('producers',[]);rows=[]
    proof=service.state.ledger_rows('field_verification',job) if job else service.state.ledger_rows('field_verification')
    for f in spec.fields:
        ps=[p for p in producers if p['column']==f['name']]
        status='configured' if any(p['kind'] not in {'not_applicable','unavailable'} for p in ps) else 'not_applicable' if ps and all(p['kind']=='not_applicable' for p in ps) else 'unavailable' if ps else 'unmapped'
        evidence=[p for p in proof if p['table']==name and p['column']==f['name'] and (not target or p['target']==target)]
        rows.append({**{k:f.get(k) for k in ('type','unit','zh_name','system')},'column':f['name'],'producers':ps,'status':status,'evidence':evidence})
    return {'table':name,'columns':rows,'summary':dict(Counter(r['status'] for r in rows)),'producer_version':VERSION}


def record_readback(service,item,target,spec,comparison,live):
    """Only changed, selected, non-null values can prove a source column written."""
    from .comparison import row_key
    mapping=item['plan'].get('mapping_snapshots',{}).get(spec.name,{})
    task_assets=item['plan'].get('request',{}).get('asset_types')
    source_ids={b.get('source_id') for b in mapping.get('bindings',[])}
    financial_scopes=(financial_column_scopes() if spec.name=='fundamental_periodic' and source_ids=={'tushare'} else {})
    changes={c['row_key']:c for c in comparison['changes']}
    traces=service.state.ledger_get('task_column_lineage',item['id']+':'+spec.name) or {'rows':{}}
    for f in spec.fields:
        column=f['name'];stats=Counter();origins=Counter()
        producers=[p for p in mapping.get('producers',[]) if p['column']==column]
        for key,current in live.items():
            # No declarations retains the legacy/custom trace-based behavior.
            # Known financial semantics can still disprove applicability after
            # an unselected endpoint's direct declaration was filtered away.
            statements=financial_scopes.get(column)
            if statements is not None and current.get('statement_type') in {'income','balancesheet','cashflow','fina_indicator'} and current['statement_type'] not in statements:
                stats['not_applicable_count']+=1;continue
            scope=applicability(producers,current,task_assets) if producers else 'applicable'
            if scope=='unknown':stats['unknown_scope_count']+=1;continue
            if scope=='unavailable':stats['unavailable_count']+=1;continue
            if scope=='not_applicable':stats['not_applicable_count']+=1;continue
            stats['applicable_count']+=1
            if current.get(column) is None:stats['null_count']+=1
            else:stats['non_null_count']+=1
            change=changes.get(key);trace=traces['rows'].get(key,{}).get(column)
            if trace and trace.get('source_null'):stats['source_null_count']+=1
            if trace and trace.get('selected'):stats['selected_count']+=1
            if not change:stats['preserved_count']+=1;continue
            if change.get('before') is not None and change['before'].get(column)==change['after'].get(column):
                stats['preserved_count']+=1;continue
            stats['written_count']+=1
            if not trace or not trace.get('selected'):continue
            if current.get(column) is not None and digest(current[column])==trace.get('value_hash'):
                stats['verified_count']+=1
                for origin in trace.get('origins',[]):origins[origin]+=1
            else:stats['mismatch_count']+=1
        status='mismatch' if stats['mismatch_count'] else 'verified_stored' if stats['verified_count'] else 'unknown_scope' if stats['unknown_scope_count'] else 'source_null' if stats['source_null_count'] else 'unavailable' if stats['unavailable_count'] and not stats['applicable_count'] else 'not_applicable' if stats['not_applicable_count'] and not stats['applicable_count'] else 'table_verified'
        record=dict(task_id=item['id'],target=target,table=spec.name,column=column,status=status,**dict(stats),verified_origins=dict(origins),
                    proof_version=PROOF_VERSION,scope_context={'task_asset_types':list(task_assets) if isinstance(task_assets,(list,tuple,set,frozenset)) else None,'row_field_precedence':True})
        service.state.ledger_put('field_verification',digest([item['id'],target,spec.name,column]),record)
