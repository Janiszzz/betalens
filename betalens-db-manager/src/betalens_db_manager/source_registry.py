"""Provider-independent subscription contracts and real HTTP/file adapters."""
from __future__ import annotations
import os
import time
from pathlib import Path
from decimal import Decimal
import json
import requests
import yaml
from dotenv import dotenv_values
from .catalog import load_catalog
from .private import atomic_private
from .util import identifier, digest, now


class SourceRegistry:
    def __init__(self,settings,state):
        self.settings,self.state=settings,state
        self.path=settings.config_dir/'sources.yaml'
        self.providers={'tushare':lambda definition:None,'file':lambda definition:None,'wind':lambda definition:None,'http':HTTPAdapter}
        from importlib.metadata import entry_points
        for entry in entry_points(group='betalens_db_manager.sources'):
            self.register(entry.name,entry.load())
        self.errors=[]
        if not self.path.exists():atomic_private(self.path,yaml.safe_dump({'format_version':1,'sources':[]},sort_keys=False))
        self.refresh()

    def register(self,kind,factory):self.providers[kind]=factory

    def refresh(self):
        values={'tushare':{'source_id':'tushare','provider':'tushare','label':'Tushare','endpoints':{n:s.as_dict() for n,s in load_catalog().items()}},'file':{'source_id':'file','provider':'file','label':'本地文件','endpoints':{}},'wind':{'source_id':'wind','provider':'wind','label':'Wind','endpoints':{},'status':'not_configured'}}
        try:
            document=yaml.safe_load(self.path.read_text())
            for source in document.get('sources',[]):
                self.validate(source);values[source['source_id']]=source
            self.errors=[]
            for source in values.values():self.state.ledger_put('source_definition',source['source_id'],source)
        except (ValueError,KeyError,TypeError,yaml.YAMLError,AttributeError) as exc:
            self.errors=[str(exc)];values={s['source_id']:s for s in self.state.ledger_rows('source_definition')} or values
        self.values=values
        return values

    def validate(self,source):
        if set(source)-{'source_id','provider','label','description','base_url','auth','endpoints'}:raise ValueError('来源包含未登记配置项；鉴权须使用 auth.env 引用')
        identifier(source['source_id'])
        if source['provider'] not in self.providers:raise ValueError('来源适配器类型未注册')
        if source['source_id'] in {'tushare','wind'}:raise ValueError('内置来源类型保留；新增实例请使用其他 source_id')
        if source['provider']=='http':
            from urllib.parse import urlsplit
            url=urlsplit(source.get('base_url',''))
            if url.scheme not in {'http','https'} or not url.hostname or url.username or url.password or url.query:raise ValueError('HTTP 来源地址必须不含凭据和查询参数')
        auth=source.get('auth',{})
        if set(auth)-{'env','header','prefix'}:raise ValueError('鉴权仅允许环境变量引用')
        if auth.get('env') and not __import__('re').fullmatch(r'[A-Z][A-Z0-9_]*',auth['env']):raise ValueError('鉴权环境变量名称无效')
        for endpoint,spec in source.get('endpoints',{}).items():
            if any(__import__('re').search(r'token|password|secret|authorization|api_key',str(k),__import__('re').I) for k in spec.get('parameters',{})):raise ValueError('查询配置不能包含鉴权值；请引用环境变量')
            identifier(endpoint)
            if '?' in spec.get('path','') or '#' in spec.get('path',''):raise ValueError('查询路径不能携带查询参数或凭据')
            names=[]
            for f in spec['fields']:
                name=f.get('name')
                if not isinstance(name,str) or not name.strip() or len(name)>1024 or any(c in name for c in ('\n','\r','\x00')):raise ValueError('源字段名称为空或含控制字符')
                names.append(name)
                if f.get('type') not in {'text','numeric','integer','date','timestamp'}:raise ValueError('源字段类型无效')
            if len(names)!=len(set(names)):raise ValueError('源字段重复')
            if not spec.get('keys') or set(spec['keys'])-set(names):raise ValueError('查询需要登记业务键字段')
            if spec.get('method','GET') not in {'GET','POST'}:raise ValueError('HTTP 方法须为 GET/POST')
        return source

    def save(self,source):
        self.validate(source);document=yaml.safe_load(self.path.read_text())
        document['sources']=[s for s in document.get('sources',[]) if s['source_id']!=source['source_id']]+[source]
        atomic_private(self.path,yaml.safe_dump(document,allow_unicode=True,sort_keys=False));self.refresh()
        return source

    def endpoint(self,source_id,endpoint):
        try:return self.values[source_id]['endpoints'][endpoint]
        except KeyError:raise ValueError('未登记查询源：'+source_id+'.'+endpoint) from None

    def public(self):
        env={**os.environ,**(dotenv_values(self.settings.env_file,interpolate=False) if self.settings.env_file and self.settings.env_file.exists() else {})}
        return [{**s,'configured':bool(self.settings.token) if sid=='tushare' else False if sid=='wind' else bool(env.get(s.get('auth',{}).get('env'))) if s.get('auth',{}).get('env') else True} for sid,s in self.values.items()]

    def adapter(self,source_id):
        source=self.values[source_id]
        if source['provider']=='wind':raise ValueError('Wind 未配置 SDK 和字段合同')
        if source['provider']=='file':return FileAdapter(source)
        if source['provider']=='tushare':raise ValueError('Tushare 使用完整分页适配器')
        adapter=self.providers[source['provider']](source)
        return adapter


from dataclasses import dataclass

@dataclass(frozen=True)
class FetchContext:
    settings: object
    state: object
    task_id: str

class SubscriptionAdapter:
    def __init__(self,definition):self.definition=definition
    def capabilities(self):return self.definition['endpoints']
    def plan(self,request):
        if request['endpoint'] not in self.definition['endpoints']:raise ValueError('查询未登记')
        return request
    def fetch(self,request,context):
        return self._fetch(request['endpoint'],request.get('parameters',{}),context.settings,context.state,context.task_id)
    def validate_response(self,response,contract):
        fields=response.get('fields',[])
        if len(fields)!=len(set(fields)):raise ValueError('源字段重复')
        return {'missing':sorted(set(contract.get('keys',[]))-set(fields)),'extra':sorted(set(fields)-{f['name'] for f in contract['fields']})}
    def normalize(self,response,mapping):
        if not isinstance(response,dict) or not isinstance(response.get('fields'),list) or not isinstance(response.get('items'),list):raise ValueError('响应不是 fields/items 或对象行列表；请检查 response_path')
        return response

class HTTPAdapter(SubscriptionAdapter):
    def __init__(self,definition):self.definition=definition
    def capabilities(self):return self.definition['endpoints']
    def plan(self,request):return request
    def validate_response(self,response,contract):
        fields=response.get('fields',[])
        if len(fields)!=len(set(fields)):raise ValueError('源字段重复')
        if set(fields)-{f['name'] for f in contract['fields']}:raise ValueError('返回未登记字段')
        return response
    def normalize(self,response,mapping):
        if not isinstance(response,dict) or not isinstance(response.get('fields'),list) or not isinstance(response.get('items'),list):raise ValueError('响应不是 fields/items 或对象行列表；请检查 response_path')
        return response
    def _fetch(self,endpoint,params,settings,state,job):
        spec=self.definition['endpoints'][endpoint];auth=self.definition.get('auth',{});headers={}
        if auth.get('env'):
            env={**os.environ,**(dotenv_values(settings.env_file,interpolate=False) if settings.env_file and settings.env_file.exists() else {})}
            secret=env.get(auth['env'])
            if not secret:raise ValueError('来源鉴权环境变量未配置：'+auth['env'])
            headers[auth.get('header','Authorization')]=auth.get('prefix','')+secret
        url=self.definition['base_url'].rstrip('/')+'/'+spec.get('path',endpoint).lstrip('/')
        method=spec.get('method','GET');started=time.monotonic()
        for attempt in range(4):
            state.check_cancel(job)
            state.log(job,'request_started',source_id=self.definition['source_id'],endpoint=endpoint,parameters=params,attempt=attempt+1)
            try:
                response=requests.request(method,url,params=params if method=='GET' else None,json=params if method=='POST' else None,headers=headers,timeout=settings.timeout,allow_redirects=False)
                state.log(job,'http_response',source_id=self.definition['source_id'],endpoint=endpoint,http_status=response.status_code,elapsed_ms=int((time.monotonic()-started)*1000))
                if response.status_code==429 or response.status_code>=500:
                    if attempt<3:
                        for _ in range(2**attempt*5):state.check_cancel(job);time.sleep(.2)
                        continue
                response.raise_for_status()
                self.last_wire=response.content
                value=json.loads(response.content,parse_float=Decimal,parse_constant=str)
                for segment in spec.get('response_path','').split('.'):
                    if segment:value=value[segment]
                if isinstance(value,list):
                    if not all(isinstance(row,dict) for row in value):raise ValueError('响应列表必须由字段对象组成')
                    fields=list(dict.fromkeys(field for row in value for field in row)) if value else [f['name'] for f in spec['fields']]
                    return {'fields':fields,'items':[[r.get(f) for f in fields] for r in value]}
                return value
            except (requests.Timeout,requests.ConnectionError):
                if attempt==3:raise ValueError('来源网络超时或连接失败') from None
                state.log(job,'network_retry',endpoint=endpoint,attempt=attempt+1)
        raise ValueError('请求失败')


class FileAdapter(SubscriptionAdapter):
    def __init__(self,definition):self.definition=definition
    def _fetch(self,endpoint,params,settings,state,job):
        from .sources import FileSourceAdapter
        spec=self.definition['endpoints'][endpoint];fields=[f['name'] for f in spec['fields']];items=[]
        from itertools import islice
        import hashlib
        from pathlib import Path
        path=Path(params.get('path') or spec.get('path'))
        hasher=hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b''):hasher.update(chunk)
        sha=hasher.hexdigest()
        # Compare every page to the first archived file fingerprint.
        for part in state.job(job)['slices']:
            if part.get('archive') and part['dataset']==endpoint:
                from .source import read_archive
                old=read_archive(Path(part['archive']),part['sha']).get('source_sha256')
                if old and old!=sha:raise ValueError('分页期间源文件发生变化，请创建新任务')
                if old:break
        iterator=FileSourceAdapter.rows(path)
        offset=int(params.get('offset',0));limit=int(params.get('limit',5000))
        for row in islice(iterator,offset,offset+limit):
            state.check_cancel(job)
            if set(row)-set(fields):raise ValueError('文件存在未登记字段')
            items.append([row.get(f) for f in fields])
        return {'fields':fields,'items':items,'source_sha256':sha}


def source_identity(definition):
    """Subscription identity excludes editable field and parser contracts."""
    return 'identity-v1:'+digest({k:definition.get(k) for k in ('source_id','provider','base_url','auth')})


def identity_matches(definition,expected):
    return (source_identity(definition) if expected.startswith('identity-v1:') else digest(definition))==expected
