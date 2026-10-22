"""Versioned, editable table definitions shared by every database operation."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import copy
import yaml
from .canonical import CanonicalTable, builtin_catalog, COMMON_DAILY
from .private import atomic_private
from .util import digest, identifier, now

GENERATED = {'import_task_id','source_name','source_id','source_endpoint','source_row_no','source_record_hash','mapping_version','schema_version','data_quality_status','downloaded_at','ingested_at'}
SYSTEM = {f['name']:dict(f,nullable=True,system=True) for f in COMMON_DAILY}
SYSTEM.update({name:dict(name=name,type=typ,zh_name=label,description=label,unit='',nullable=True,system=True) for name,typ,label in [('ingested_at','timestamp','实际入库时间'),('schema_version','text','表结构版本'),('source_id','text','订阅源实例')]})


def definition(spec):
    value=spec.as_dict();value['format_version']=1
    value['fields']=[copy.deepcopy(f) for f in spec.fields]
    value['keys']=list(spec.keys)
    value['time_field']=next((f for f in ('trade_date','period_end','nav_date','valid_from','event_date') if any(x['name']==f for x in spec.fields)),None)
    value['temporal_rule']='trading_day' if spec.name.startswith(('market_daily','trade_status')) else 'calendar_day' if spec.name=='trade_calendar' else 'quarter_end' if spec.name=='fundamental_periodic' else 'observations'
    if spec.name.endswith('_profile'):value['time_field']=None
    if not value['time_field']:value['temporal_rule']='snapshot'
    return value


class TableRegistry:
    def __init__(self, settings, state):
        self.settings,self.state=settings,state
        self.root=settings.config_dir/'tables';self.root.mkdir(parents=True,exist_ok=True)
        self.errors={}
        for name,spec in builtin_catalog().items():
            if not self.path(name).exists():
                value=definition(spec)
                value['fields']=[f for f in value['fields'] if f['name'] not in SYSTEM]+[dict(SYSTEM.get(f['name'],f),**f,system=True) for f in value['fields'] if f['name'] in SYSTEM]
                value=self.validate(value)
                atomic_private(self.path(name),yaml.safe_dump(value,allow_unicode=True,sort_keys=False))
        self.refresh()

    def path(self,name):return self.root/(identifier(name)+'.yaml')

    def validate(self,value):
        if not isinstance(value,dict):raise ValueError('表定义必须是对象')
        value=copy.deepcopy(value);name=identifier(value.get('name'))
        if name in {'meta_table','meta_column','import_task','import_row_audit','import_target_status','dataset_coverage'} or name.startswith(('meta_','import_')):raise ValueError('表名属于管理元数据保留名称')
        for key in ('zh_name','description','grain'):
            if not isinstance(value.get(key),str) or not value[key].strip():raise ValueError('需要填写文本 '+key)
        fields=value.get('fields',[])
        if not isinstance(fields,list) or not fields or not all(isinstance(f,dict) for f in fields):raise ValueError('业务字段必须是非空对象列表')
        known=set()
        for f in fields:
            col=identifier(f.get('name'))
            if col in known:raise ValueError('字段重复：'+col)
            known.add(col)
            if f.get('type') not in {'text','numeric','integer','date','timestamp'}:raise ValueError(col+': 不支持的逻辑类型')
            if f.get('type')=='numeric':
                p,s=f.get('precision',38),f.get('scale',12)
                if type(p) is not int or type(s) is not int or not 1<=p<=38 or not 0<=s<=p:raise ValueError(col+': Decimal 精度无效')
                f.update(precision=p,scale=s)
            if 'nullable' in f and not isinstance(f['nullable'],bool):raise ValueError(col+': nullable 必须为布尔值')
            f.setdefault('nullable',True);f.setdefault('unit','');f.setdefault('zh_name',col);f.setdefault('description',f['zh_name'])
            if col in SYSTEM:
                if f['type']!=SYSTEM[col]['type']:raise ValueError('系统字段类型不能修改：'+col)
                f['system']=True
        for name,f in SYSTEM.items():
            if name not in known:fields.append(copy.deepcopy(f));known.add(name)
        keys=value.get('keys',[])
        if not isinstance(keys,list) or not all(isinstance(k,str) for k in keys):raise ValueError('业务主键必须是字段名列表')
        if not keys or len(keys)!=len(set(keys)) or set(keys)-known or set(keys)&GENERATED:raise ValueError('需要非重复的业务主键，且主键不能使用系统审计字段')
        for f in fields:
            if f['name'] in keys:f['nullable']=False
        if value.get('time_field'):
            f=next((f for f in fields if f['name']==value['time_field']),None)
            if not f or f['type']!='date':raise ValueError('业务日期字段必须存在并使用 date 类型')
        value.setdefault('temporal_rule','snapshot' if not value.get('time_field') else 'observations')
        if value['temporal_rule'] not in {'snapshot','observations','calendar_day','trading_day','quarter_end'}:raise ValueError('时间规则无效')
        value.setdefault('domain','custom');value['format_version']=1
        value.pop('version',None);value['version']=digest(value)
        return value

    def refresh(self):
        self.errors={}
        for path in self.root.glob('*.yaml'):
            try:
                value=self.validate(yaml.safe_load(path.read_text()))
                if value['name']!=path.stem:raise ValueError('表名必须与 YAML 文件名一致')
                old=self.record(path.stem)
                status=old.get('status','draft')
                if status!='archived' and any(d.get('definition_version')!=value['version'] for d in old.get('deployed',{}).values()):status='pending'
                self.state.ledger_put('table_definition',path.stem,{**old,'definition':value,'updated_at':now(),'status':status})
                self.state.ledger_put('table_definition_version',value['version'],value)
            except (ValueError,yaml.YAMLError,TypeError,KeyError,AttributeError) as exc:self.errors[path.stem]=str(exc)
        return self.errors

    def record(self,name):
        return self.state.ledger_get('table_definition',name) or {}

    def get(self,name):
        r=self.record(name)
        if not r:raise ValueError('未登记本地表：'+name)
        return r['definition']

    def save(self,value,expected=None):
        value=self.validate(value);path=self.path(value['name'])
        actual=digest(path.read_text()) if path.exists() else None
        if expected is not None and expected!=actual:raise ValueError('YAML 已被其他编辑修改，请重新加载')
        atomic_private(path,yaml.safe_dump(value,allow_unicode=True,sort_keys=False));self.refresh()
        return self.document(value['name'])

    def document(self,name):
        value=self.get(name);path=self.path(name)
        return {'definition':value,'yaml':path.read_text(),'file_hash':digest(path.read_text()),'path':str(path),'error':self.errors.get(name)}

    def spec(self,name):
        v=self.get(name)
        return CanonicalTable(v['name'],v['zh_name'],v['description'],v['grain'],tuple(v['fields']),tuple(v['keys']),v['domain'])

    def catalog(self):
        result={}
        for r in self.state.ledger_rows('table_definition'):
            if r.get('status')=='archived':continue
            v=r['definition'];result[v['name']]=CanonicalTable(v['name'],v['zh_name'],v['description'],v['grain'],tuple(v['fields']),tuple(v['keys']),v['domain'])
        return result
