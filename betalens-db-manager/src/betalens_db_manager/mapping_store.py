"""Authoritative YAML mappings, safe transforms, and field-level provenance."""
from __future__ import annotations
from decimal import Decimal, localcontext
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import copy
from . import safe_yaml as yaml
from .mapping import all_source_mappings
from .canonical import FINANCIAL_ALIASES
from .registry import GENERATED
from .private import atomic_private
from .util import digest, parse_date, now
from .processors import validate_binding, resolve

DETAILS={'stock_basic':'stock_profile','fund_basic':'fund_profile','index_basic':'index_profile','dividend':'dividend_event','suspend_d':'suspension_event','namechange':'security_name_history','fund_nav':'fund_nav_observation','index_member_all':'industry_membership_record','ci_index_member':'industry_membership_record'}


class MappingStore:
    def __init__(self,registry,sources,state):
        self.registry,self.sources,self.state=registry,sources,state
        self.root=registry.settings.config_dir/'mappings';self.root.mkdir(parents=True,exist_ok=True)
        self.errors={};self.seed();self.refresh()

    def seed(self):
        groups={name:[] for name in self.registry.catalog()}
        for r in all_source_mappings():
            if r['canonical_table'] not in groups or r['canonical_column'] in GENERATED:continue
            field={'source_field':r['source_field'],'column':r['canonical_column'],'multiplier':'1','source_unit':r.get('source_unit',''),'required':False}
            expression=r.get('transform_expression','')
            if expression.startswith('Decimal(value)*'):field['multiplier']=expression.split('*',1)[1]
            elif expression not in {'identity',''}:field.update(builtin=True,formula=expression)
            if r['canonical_table'] in {'industry_dim','security_industry_membership'} and r['source_field'].startswith(('l1_','l2_','l3_')):field.update(builtin=True,formula='expand_industry_levels')
            if r['source_endpoint'].removesuffix('_vip') in {'income','balancesheet','cashflow','fina_indicator'} and r['source_field']=='f_ann_date':field['column']='actual_announcement_date'
            groups[r['canonical_table']].append((r['source_endpoint'],field))
        for endpoint,table in DETAILS.items():
            for f in self.sources.endpoint('tushare',endpoint)['fields']:
                groups[table].append((endpoint,{'source_field':f['name'],'column':'security_code' if f['name']=='ts_code' else f['name'],'multiplier':str(f.get('multiplier','1')),'source_unit':f.get('source_unit',f.get('unit','')),'required':f['name']=='ts_code'}))
        groups['industry_scheme_dim'].append(('index_classify',{'source_field':'src','column':'scheme_code'}))
        for endpoint in ('daily','fund_daily','index_daily','sw_daily'):
            groups['trade_status_daily'].extend((endpoint,{'source_field':source,'column':column}) for source,column in (('ts_code','security_code'),('trade_date','trade_date')))
            groups['market_daily_post_adjusted'].extend((endpoint,{'source_field':source,'column':column,'required':True}) for source,column in (('ts_code','security_code'),('trade_date','trade_date')))
        for name,fields in groups.items():
            if self.path(name).exists():continue
            by={}
            for endpoint,f in fields:
                by.setdefault(endpoint,{})[(f['source_field'],f['column'])]=f
            if name=='industry_scheme_dim':by.setdefault('ci_index_member',{})
            bindings=[]
            for endpoint,rules in by.items():
                contract=self.sources.endpoint('tushare',endpoint)
                asset='fund' if endpoint.startswith('fund_') else 'index' if endpoint in {'index_basic','index_daily','sw_daily','index_weight'} else 'stock' if endpoint in {'stock_basic','daily','daily_basic','adj_factor','suspend_d','income','balancesheet','cashflow','fina_indicator'} or endpoint.endswith('_vip') else None
                bindings.append({'source_id':'tushare','endpoint':endpoint,'priority':0,'scope':{'asset_type':asset} if asset else {},'strategy':contract.get('strategy','snapshot'),'fields':list(rules.values()),**({'mapping_mode':'direct'} if endpoint in {'etf_basic','bak_daily'} else {})})
            if name=='industry_scheme_dim':
                for b in bindings:b['processor']='industry_scheme_definition'
            # Finance description fields and industry catalog flags were previously archive-only.
            if name=='fundamental_periodic':
                for b in bindings:
                    if any(f['name']=='end_type' for f in self.sources.endpoint('tushare',b['endpoint'])['fields']) and not any(r['source_field']=='end_type' for r in b['fields']):b['fields'].append({'source_field':'end_type','column':'end_type'})
            if name=='industry_dim':
                for b in bindings:
                    if b['endpoint']=='index_classify':
                        known={f['source_field'] for f in b['fields']}
                        for f in self.sources.endpoint('tushare','index_classify')['fields']:
                            if f['name'] not in known:b['fields'].append({'source_field':f['name'],'column':'classification_code' if f['name']=='industry_code' else f['name']})
            value={'format_version':1,'table':name,'bindings':bindings,'ignored_fields':[]}
            from .field_lineage import producers_for
            value['producer_version']=1;value['producers']=producers_for(self.registry.spec(name),value)
            atomic_private(self.path(name),yaml.safe_dump(value,allow_unicode=True,sort_keys=False))

    def path(self,name):return self.root/(self.registry.get(name)['name']+'.yaml')

    def validate(self,value):
        if not isinstance(value,dict) or value.get('format_version') not in (1,2):raise ValueError('映射 format_version 必须为 1 或 2')
        value=copy.deepcopy(value);spec=self.registry.spec(value['table']);columns={f['name']:f for f in spec.fields}
        if 'producers' in value:
            from .field_lineage import validate_producers
            validate_producers(value['producers'],spec,self.sources)
        if value.get('overlap_policy','priority') not in {'reject','priority'}:raise ValueError('重复列策略只能是 reject 或 priority')
        bindings=value.get('bindings',[])
        if not isinstance(bindings,list) or not all(isinstance(b,dict) for b in bindings):raise ValueError('bindings 必须是查询绑定对象列表')
        queries=set()
        for b in bindings:
            query=(b['source_id'],b['endpoint'])
            if query in queries:raise ValueError('查询源重复：'+':'.join(query))
            queries.add(query)
            if b.get('mapping_mode','direct') not in {'direct','builtin'}:raise ValueError('映射模式无效')
            if type(b.get('priority',0)) is not int:raise ValueError('来源优先级必须是整数')
            rb=b.get('request_bindings',{})
            if not isinstance(rb,dict) or set(rb)-{'start_date','end_date','trade_date','security_code','date_format'}:raise ValueError('请求参数绑定只接受日期、证券和日期格式角色')
            if b['source_id']=='tushare' and rb:raise ValueError('Tushare 日期参数由官方合同确定，请使用接口参数筛选')
            scope=b.get('scope',{})
            if set(scope)-{'asset_type'} or (scope.get('asset_type') and scope['asset_type'] not in {'stock','fund','index'}):raise ValueError('资产范围只能是 stock/fund/index')
            contract=self.sources.endpoint(b['source_id'],b['endpoint']);fields={f['name']:f for f in contract['fields']}
            from .security import validate_parameters
            validate_parameters(b.get('parameters',{}),path='binding.parameters',auth=self.sources.values[b['source_id']].get('auth'))
            if not isinstance(b.get('fields',[]),list) or not all(isinstance(f,dict) for f in b.get('fields',[])):raise ValueError('映射 fields 必须是规则对象列表')
            ignored=b.get('ignored_fields',{})
            if not isinstance(ignored,dict) or any(k not in fields or not isinstance(v,str) or not v.strip() for k,v in ignored.items()):raise ValueError('忽略字段必须登记源字段和非空原因')
            seen=set();seen_sources=set()
            for rule in b.get('fields',[]):
                source,column=rule['source_field'],rule['column']
                if source not in fields:raise ValueError('未登记源字段：'+source)
                if column not in columns:raise ValueError('未登记本地字段：'+column)
                if column in GENERATED:raise ValueError('系统审计字段禁止映射覆盖：'+column)
                if source in ignored:raise ValueError('同一源字段不能同时映射和忽略：'+source)
                if value['format_version']==2 and not rule.get('builtin'):
                    if source in seen_sources:raise ValueError('一一映射中源字段重复：'+source)
                    if any(r['column']==column and not r.get('builtin') for r in b['fields'] if r is not rule):raise ValueError('一一映射中本地字段重复：'+column)
                seen_sources.add(source)
                identity=(column,rule.get('priority',0))
                if identity in seen and not rule.get('builtin'):raise ValueError('多个源字段不能同时覆盖同一本地列；请设置不同主备优先级')
                seen.add(identity)
                if rule.get('builtin'):
                    if rule.get('formula') not in {'native_index_value','Decimal(raw_price)*Decimal(adjustment_factor)','expand_industry_levels'}:raise ValueError('未登记内置计算')
                if set(rule)-{'source_field','column','multiplier','source_unit','required','priority','date_format','timezone','enum','builtin','formula'}:raise ValueError('映射含未支持转换；禁止执行任意表达式')
                if rule.get('enum') is not None and not isinstance(rule['enum'],dict):raise ValueError('枚举转换必须是源值到本地值的对象')
                factor=Decimal(str(rule.get('multiplier','1')))
                if not factor.is_finite():raise ValueError('单位乘数必须有限')
                if columns[column]['type']!='numeric' and factor!=1:raise ValueError('单位转换仅适用于数值字段')
                if rule.get('timezone'):ZoneInfo(rule['timezone'])
            validate_binding(value['table'],b)
            joins=b.get('join_keys')
            if joins is not None:
                if not isinstance(joins,list) or not all(isinstance(k,dict) for k in joins):raise ValueError('共同键必须是 source_field / column 对象列表')
                local_keys=set();source_keys=set()
                for k in joins:
                    if set(k)!={'source_field','column'} or k['column'] not in spec.keys or k['source_field'] not in fields:raise ValueError('共同键须连接已登记源字段和本地业务主键')
                    if k['column'] in local_keys or k['source_field'] in source_keys:raise ValueError('共同键重复')
                    if not any(r['source_field']==k['source_field'] and r['column']==k['column'] and not r.get('builtin') for r in b['fields']):raise ValueError('共同键必须与字段连线一致')
                    local_keys.add(k['column']);source_keys.add(k['source_field'])
        if 'producers' in value:
            from .field_lineage import reconcile_direct_producers
            value['producers']=reconcile_direct_producers(spec,value)
        value.pop('version',None);value['version']=digest(value)
        return value

    def refresh(self):
        self.errors={}
        for path in self.root.glob('*.yaml'):
            try:
                v=self.validate(yaml.safe_load(path.read_text()))
                if v['table']!=path.stem:raise ValueError('映射表名与文件名不同')
                self.state.ledger_put('mapping_active',v['table'],v)
                self.state.ledger_put('mapping_version',v['version'],v)
            except (ValueError,TypeError,KeyError,yaml.YAMLError,AttributeError) as exc:self.errors[path.stem]=str(exc)
        return self.errors

    def get(self,name):
        return self.state.ledger_get('mapping_active',name) or {'format_version':1,'table':name,'bindings':[],'ignored_fields':[]}

    def document(self,name):
        from .canonical import builtin_catalog
        self.refresh();path=self.path(name)
        return {'builtin_table':name in builtin_catalog(),'protected_columns':sorted(GENERATED),'mapping':self.get(name),'diagnostics':self.diagnostics(self.get(name)),'yaml':path.read_text() if path.exists() else yaml.safe_dump(self.get(name),allow_unicode=True),'file_hash':digest(path.read_text()) if path.exists() else None,'path':str(path),'error':self.errors.get(name)}

    def diagnostics(self,mapping,bindings=None):
        """Drafts may be incomplete; execution requires an unambiguous row identity."""
        from .canonical import builtin_catalog
        spec=self.registry.spec(mapping['table']);selected=mapping.get('bindings',[]) if bindings is None else bindings
        issues=[];destinations={};covered=set();strict=mapping.get('format_version')==2
        def issue(b,code,message,field='',severity='error'):
            issues.append(dict(severity=severity,code=code,source_id=b.get('source_id',''),endpoint=b.get('endpoint',''),field=field,message=message))
        for b in selected:
            direct=resolve(mapping['table'],b) is None
            rules=b.get('fields',[]);mapped={r['source_field'] for r in rules};columns={r['column'] for r in rules};covered|=columns
            if b.get('scope',{}).get('asset_type'):covered.add('asset_type')
            if not direct:continue  # Registered built-in processors also produce keys/derived columns.
            keys=b.get('join_keys')
            if keys is None and not strict:keys=[r for r in rules if r['column'] in spec.keys]
            present={k['column'] for k in (keys or [])}
            for key in spec.keys:
                if key not in present:issue(b,'missing_join_key','请为每个查询源配置完整业务键；联合键不能缺项',key)
            contract=self.sources.endpoint(b['source_id'],b['endpoint'])
            for f in contract['fields']:
                if f['name'] not in mapped and f['name'] not in b.get('ignored_fields',{}):
                    issue(b,'unmapped_source_field','源字段尚未映射；请连接或明确记录忽略原因',f['name'], 'error' if strict or b.get('mapping_mode')=='direct' else 'warning')
            for column in columns-set(spec.keys):destinations.setdefault(column,[]).append(b)
        for column,owners in destinations.items():
            for i,b in enumerate(owners):
                overlap=[other for other in owners[:i] if not (b.get('scope',{}).get('asset_type') and other.get('scope',{}).get('asset_type') and b['scope']['asset_type']!=other['scope']['asset_type'])]
                if overlap and mapping.get('overlap_policy','priority')=='reject':issue(b,'duplicate_target','多个查询源写入同一业务列；请移除重复连线或显式选择主备策略',column)
                elif overlap and (strict or b.get('mapping_mode')=='direct') and any(o.get('priority',0)==b.get('priority',0) for o in overlap):issue(b,'ambiguous_priority','重复业务列必须配置不同的来源优先级',column)
        if selected:
            covered|={p['column'] for p in mapping.get('producers',[]) if p['kind'] in {'constant','copy','computed','system','not_applicable','unavailable'}}
            for f in spec.fields:
                if f['name'] not in GENERATED and not f.get('system') and f['name'] not in covered:
                    issue({},'unmapped_target','本地字段尚无来源',f['name'],'error' if f['name'] in spec.keys or not f.get('nullable',True) else 'warning')
        return issues

    def save(self,value,expected):
        v=self.validate(value);path=self.path(v['table']);actual=digest(path.read_text()) if path.exists() else None
        if expected!=actual:raise ValueError('映射 YAML 已被修改，请重新加载')
        atomic_private(path,yaml.safe_dump(v,allow_unicode=True,sort_keys=False));self.refresh()
        return self.document(v['table'])

    def snapshots(self,names=None):
        self.refresh();selected=names or list(self.registry.catalog())
        return {n:self.get(n) for n in selected}

    @staticmethod
    def convert(value,rule,field):
        if value in (None,''):return None
        if rule.get('enum') is not None:
            from .util import encode
            def enum_key(item):
                if isinstance(item,bool):return str(item).lower()
                if isinstance(item,(Decimal,int,float)):return encode(Decimal(str(item)))
                return str(item)
            values={enum_key(k):v for k,v in rule['enum'].items()}
            key=enum_key(value)
            if key not in values:raise ValueError('枚举值未登记')
            value=values[key]
            if value in (None,''):return None
        if field['type']=='numeric':
            with localcontext() as ctx:
                ctx.prec=80;value=Decimal(str(value))*Decimal(str(rule.get('multiplier','1')))
                if not value.is_finite():raise ValueError('非有限数值')
                return value
        if field['type']=='date':
            return datetime.strptime(str(value),rule['date_format']).date() if rule.get('date_format') else parse_date(value)
        if field['type']=='timestamp':
            value=datetime.strptime(str(value),rule['date_format']) if rule.get('date_format') else datetime.fromisoformat(str(value).replace('Z','+00:00'))
            if value.tzinfo is None:
                if not rule.get('timezone'):raise ValueError('无时区时间需要显式 timezone')
                value=value.replace(tzinfo=ZoneInfo(rule['timezone']))
            return value.astimezone(timezone.utc)
        if field['type']=='integer':
            from .integer_contract import validate_int64
            return validate_int64(value,field['name'])
        return str(value)

    def map_row(self,spec,raw,binding,*,processor=False):
        registered=validate_binding(spec.name,binding)
        fields={f['name']:f for f in spec.fields};result={};evidence=[]
        for rule in sorted(binding.get('fields',[]),key=lambda r:r.get('priority',0)):
            if rule.get('builtin'):
                if not processor or registered is None:raise ValueError('内置映射必须由注册处理器执行')
                if rule.get('required') and raw.get(rule['source_field']) in (None,''):raise ValueError('必需源字段为空：'+rule['source_field'])
                continue
            source,column=rule['source_field'],rule['column']
            if rule.get('required') and raw.get(source) in (None,''):raise ValueError('必需源字段为空：'+source)
            try:converted=self.convert(raw.get(source),rule,fields[column])
            except (ValueError,TypeError,ArithmeticError) as exc:raise ValueError(source+': 转换失败 '+str(exc)) from None
            if rule.get('required') and converted is None:raise ValueError('必需源字段转换后为空：'+source)
            prior=result.get(column)
            if prior is None:result[column]=converted
            evidence.append({'column':column,'source_field':source,'raw_value':raw.get(source),'converted_value':converted,'target_unit':fields[column].get('unit',''),'source_id':binding['source_id'],'endpoint':binding['endpoint'],'selected':prior is None and converted is not None,'conflict':prior is not None and converted is not None and prior!=converted})
        return result,evidence


def mapping_rows(registry):
    sources={s['source_id']:s for s in registry.state.ledger_rows('source_definition')}
    result=[];registered=registry.catalog()
    for mapping in registry.state.ledger_rows('mapping_active'):
        table=mapping['table']
        if table not in registered:continue
        definition=registry.get(table);columns={f['name']:f for f in definition['fields']}
        for binding in mapping.get('bindings',[]):
            fields={f['name']:f for f in sources.get(binding['source_id'],{}).get('endpoints',{}).get(binding['endpoint'],{}).get('fields',[])}
            for rule in binding['fields']:
                field=fields.get(rule['source_field'],{});column=columns[rule['column']]
                expression=rule.get('formula') if rule.get('builtin') else 'Decimal(value)*'+str(rule.get('multiplier','1')) if column['type']=='numeric' else 'date_format='+rule['date_format'] if rule.get('date_format') else 'identity'
                result.append({'source_name':binding['source_id'],'source_endpoint':binding['endpoint'],'source_field':rule['source_field'],'canonical_table':table,'canonical_column':rule['column'],'source_type':field.get('type','text'),'target_type':column['type'],'source_unit':rule.get('source_unit',field.get('source_unit','')),'target_unit':column.get('unit',''),'transform_expression':expression,'date_role':'event_date' if rule['column']==definition.get('time_field') else 'announcement_date' if 'announcement' in rule['column'] else '', 'required':int(rule.get('required',False)),'mapping_version':mapping['version'],'description':field.get('description',''),'priority':binding.get('priority',0),'asset_type':binding.get('scope',{}).get('asset_type')})
    return result
