"""Versioned, create-only structure bundles. Inspection never reads business rows."""
from contextlib import nullcontext
from copy import copy,deepcopy
import re,uuid
from psycopg2 import sql
from .canonical import registry_context
from .private import atomic_private
from . import safe_yaml as yaml
from .schema_contract import expected_type
from .storage import CONTROL_DDL,CONTROL_KEYS,control_spec,pg_type,relation
from .reference_contracts import TABLES as REFERENCE_TABLES
from .reference_storage import METADATA as REFERENCE_METADATA
from .util import digest,identifier,now

TARGETS=('postgres','clickhouse')


def compact(value):return re.sub(r'[\s`\"]','',value).lower()


class SchemaBundle:
    def __init__(self,service):self.s=service

    def entries(self):
        self.s.registry.refresh();result={}
        with registry_context(self.s.registry):
            definitions=[(name,self.s.registry.spec(name),'business','registry') for name in self.s.registry.catalog() if self.s.registry.record(name).get('status')!='archived']
        definitions += [(name,control_spec(name),'audit','control') for name in CONTROL_DDL]
        definitions += [(name,spec,'business','reference') for name,spec in REFERENCE_TABLES.items()]
        definitions += [(name,spec,'audit','reference_metadata') for name,spec in REFERENCE_METADATA.items()]
        for name,spec,role,owner in definitions:
            fields=[]
            for f in spec.fields:
                field=deepcopy(f);physical={t:expected_type(f,t,f['name'] in spec.keys) for t in TARGETS}
                if owner=='control':physical['postgres']=next(kind for col,kind,_ in CONTROL_DDL[name] if col==f['name'])
                field['physical_types']=physical;fields.append(field)
            indexes=[]
            if owner=='registry':
                for col,suffix in (('event_date','event'),('security_code','security')):
                    if any(f['name']==col for f in fields):indexes.append({'target':'postgres','name':name+'_'+suffix+'_idx','columns':[col],'unique':False})
            if name=='import_row_audit':indexes.append({'target':'postgres','name':'import_row_audit_task_idx','columns':['task_id','target','table_name','id'],'unique':False})
            dependencies=list(CONTROL_DDL) if owner=='registry' else list(REFERENCE_METADATA) if owner=='reference' else []
            entry={'name':name,'zh_name':spec.zh_name,'description':spec.description,'grain':spec.grain,'role':role,'owner':owner,
                   'fields':fields,'keys':list(spec.keys),'indexes':indexes,'dependencies':dependencies,
                   'definition_source':str(self.s.registry.path(name)) if owner=='registry' else owner,
                   'importable':owner=='registry','definition_error':self.s.registry.errors.get(name),
                   'clickhouse_engine':'ReplacingMergeTree','partition_key':'toYYYYMM(event_date)' if any(f['name']=='event_date' and not f.get('nullable',True) for f in fields) else 'tuple()'}
            entry['version']=digest(entry);result[name]=entry
        return result

    def manifest(self):
        entries=self.entries();document={'format_version':1,'tables':list(entries.values())};document['version']=digest(document)
        path=self.s.settings.config_dir/'schema-manifest.yaml'
        content=yaml.safe_dump(document,allow_unicode=True,sort_keys=False)
        if not path.exists() or path.read_text()!=content:atomic_private(path,content)
        return {**document,'path':str(path)}

    def _inspect_pg(self):
        db=self.s.pg
        with db.connect(readonly=True,database='postgres') as conn,conn.cursor() as cur:
            cur.execute('SELECT 1 FROM pg_database WHERE datname=%s',(db.settings.pg_database,))
            if not cur.fetchone():return {'database_exists':False,'tables':{}}
        with db.connect(readonly=True) as conn,conn.cursor() as cur:
            fields=db._existing_columns(cur);keys=db._existing_keys(cur)
            cur.execute('SELECT table_name,column_name,is_identity FROM information_schema.columns WHERE table_schema=%s',(db.settings.pg_schema,))
            identity={(t,c):i=='YES' for t,c,i in cur.fetchall()}
            cur.execute('SELECT tablename,indexname,indexdef FROM pg_indexes WHERE schemaname=%s',(db.settings.pg_schema,))
            indexes={}
            for table,name,definition in cur.fetchall():indexes.setdefault(table,[]).append({'name':name,'definition':definition})
            cur.execute("SELECT tc.table_name,tc.constraint_name,kcu.column_name FROM information_schema.table_constraints tc JOIN information_schema.key_column_usage kcu USING(constraint_catalog,constraint_schema,constraint_name) WHERE tc.table_schema=%s AND tc.constraint_type='UNIQUE' ORDER BY tc.table_name,tc.constraint_name,kcu.ordinal_position",(db.settings.pg_schema,))
            unique={}
            for table,constraint,column in cur.fetchall():unique.setdefault(table,{}).setdefault(constraint,[]).append(column)
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema=%s AND table_type='BASE TABLE'",(db.settings.pg_schema,))
            names=[r[0] for r in cur.fetchall()]
        return {'database_exists':True,'tables':{n:{'exists':True,'fields':[{**f,'identity':identity.get((n,k),False)} for k,f in sorted(fields.get(n,{}).items())],'keys':keys.get(n,[]),'unique_keys':list(unique.get(n,{}).values()),'indexes':sorted(indexes.get(n,[]),key=lambda i:i['name'])} for n in names}}

    def _inspect_ch(self):
        db=self.s.ch;database=db._literal(db.settings.ch_database)
        if db.execute(f'EXISTS DATABASE `{identifier(db.settings.ch_database)}`')!='1':return {'database_exists':False,'tables':{}}
        tables=db._json_rows(db.execute(f'SELECT name,engine,sorting_key,partition_key FROM system.tables WHERE database={database} FORMAT JSONEachRow'))
        columns=db._json_rows(db.execute(f'SELECT table,name,type AS physical_type FROM system.columns WHERE database={database} ORDER BY table,position FORMAT JSONEachRow'))
        fields={}
        for row in columns:fields.setdefault(row['table'],[]).append({'name':row['name'],'physical_type':row['physical_type']})
        def keys(value):
            value=value.removeprefix('tuple(').removesuffix(')')
            return [k.strip().strip('`') for k in value.split(',') if k.strip()]
        return {'database_exists':True,'tables':{r['name']:{'exists':True,'fields':fields.get(r['name'],[]),'keys':keys(r['sorting_key']),'indexes':[], 'engine':r['engine'],'partition_key':r['partition_key']} for r in tables}}

    @staticmethod
    def differences(entry,actual,target):
        if not actual.get('exists'):return []
        changes=[];fields={f['name']:f for f in actual['fields']}
        for f in entry['fields']:
            old=fields.get(f['name']);expected=f['physical_types'][target]
            if not old:changes.append({'column':f['name'],'reason':'缺少字段','expected':expected});continue
            found=old['physical_type']
            if target=='postgres':
                if found=='numeric':found=f'numeric({old.get("precision")},{old.get("scale")})'
                if 'GENERATED' in expected:
                    if not old.get('identity'):changes.append({'column':f['name'],'reason':'缺少 Identity 属性'})
                    expected=expected.split(' GENERATED',1)[0]
                nullable=f['name'] not in entry['keys'] and f.get('nullable',True)
                if old.get('nullable')!=nullable:changes.append({'column':f['name'],'reason':'可空性不同'})
            if compact(found)!=compact(expected):changes.append({'column':f['name'],'reason':'类型不同','expected':expected,'actual':found})
        for extra in set(fields)-{f['name'] for f in entry['fields']}:changes.append({'column':extra,'reason':'额外字段'})
        equivalent_control_key=target=='postgres' and entry['owner']=='control' and entry['keys'] in actual.get('unique_keys',[])
        if actual.get('keys')!=entry['keys'] and not equivalent_control_key:changes.append({'reason':'主键或排序键不同','expected':entry['keys'],'actual':actual.get('keys')})
        if target=='postgres':
            indexes={i['name']:i['definition'] for i in actual.get('indexes',[])}
            for index in entry['indexes']:
                definition=indexes.get(index['name'],'')
                expected='('+','.join(index['columns'])+')'
                if not definition or compact(expected) not in compact(definition):changes.append({'reason':'缺少或不匹配的索引','index':index['name']})
        else:
            if actual.get('engine')!=entry['clickhouse_engine']:changes.append({'reason':'存储引擎不同'})
            if compact(actual.get('partition_key') or 'tuple()')!=compact(entry['partition_key']):changes.append({'reason':'分区键不同'})
        return changes

    def inspect(self,targets=None,tables=None):
        targets=list(TARGETS) if targets is None else targets
        if not isinstance(targets,list) or not targets or set(targets)-set(TARGETS):raise ValueError('数据库目标无效')
        if tables is not None and (not isinstance(tables,list) or not all(isinstance(n,str) for n in tables)):raise ValueError('表范围必须为列表')
        entries=self.entries();result={}
        for target in targets:
            try:
                snapshot=self._inspect_pg() if target=='postgres' else self._inspect_ch()
                names=sorted(set(tables) if tables is not None else set(entries)|set(snapshot['tables']))
                rows=[]
                for name in names:
                    identifier(name);actual=snapshot['tables'].get(name,{'exists':False,'fields':[],'keys':[],'indexes':[]});entry=entries.get(name)
                    changes=self.differences(entry,actual,target) if entry else []
                    rows.append({'name':name,'target':target,'registered':bool(entry),'status':'missing' if not actual['exists'] else 'drifted' if changes else 'ready' if entry else 'unregistered','actual':actual,'differences':changes})
                result[target]={'connected':True,'database_exists':snapshot['database_exists'],'checked_at':now(),'tables':rows}
            except Exception as exc:result[target]={'connected':False,'checked_at':now(),'error':self.s.safe_error(exc),'tables':[]}
        return result

    def plan(self,selections):
        manifest=self.manifest();entries={e['name']:e for e in manifest['tables']}
        if not isinstance(selections,dict) or not selections or set(selections)-set(TARGETS):raise ValueError('需要按数据库指定表范围')
        selected={};expanded={}
        for target in TARGETS:
            if target not in selections:continue
            names=selections[target]
            if not isinstance(names,list) or not names or not all(isinstance(n,str) and n in entries for n in names):raise ValueError('只能选择已登记的表，表范围不能为空')
            selected[target]=sorted(set(names));pending=list(names);scope=set()
            while pending:
                name=pending.pop()
                if name in scope:continue
                if entries[name].get('definition_error'):raise ValueError(name+': 表定义无效')
                scope.add(name);pending.extend(entries[name]['dependencies'])
            expanded[target]=sorted(scope,key=lambda n:(entries[n]['role']!='audit',n))
        inspected=self.inspect(list(expanded),sorted({n for names in expanded.values() for n in names}));actions=[]
        for target,names in expanded.items():
            snapshot=inspected[target];by={r['name']:r for r in snapshot['tables']}
            for name in names:
                row=by.get(name,{'status':'unavailable','actual':None,'differences':[]})
                actions.append({**row,'name':name,'target':target,'automatic_dependency':name not in selected[target],
                    'action':'create' if row['status']=='missing' else 'skip' if row['status']=='ready' else 'blocked',
                    'error':snapshot.get('error')})
        result={'manifest_version':manifest['version'],'target_fingerprint':self.s.settings.fingerprint,'selections':selected,
                'expanded':expanded,'actions':actions,'database_exists':{t:r.get('database_exists') for t,r in inspected.items()},'created_at':now()}
        result['snapshot_hash']=digest({k:v for k,v in result.items() if k!='created_at'})
        result['token']=digest(result)
        self.s.state.ledger_put('schema_bundle_plan',result['token'],result);return result

    def _create_pg(self,entry):
        with self.s.pg.connect() as conn,conn.cursor() as cur:
            cur.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(self.s.settings.pg_schema)))
            fields=[sql.SQL('{} {}{}').format(sql.Identifier(f['name']),sql.SQL(f['physical_types']['postgres']),sql.SQL(' NOT NULL' if f['name'] in entry['keys'] or not f.get('nullable',True) else '')) for f in entry['fields']]
            fields.append(sql.SQL('PRIMARY KEY ({})').format(sql.SQL(',').join(map(sql.Identifier,entry['keys']))))
            cur.execute(sql.SQL('CREATE TABLE {} ({})').format(relation(self.s.settings,entry['name']),sql.SQL(',').join(fields)))
            for index in entry['indexes']:
                cur.execute(sql.SQL('CREATE {}INDEX {} ON {} ({})').format(sql.SQL('UNIQUE ' if index['unique'] else ''),sql.Identifier(index['name']),relation(self.s.settings,entry['name']),sql.SQL(',').join(map(sql.Identifier,index['columns']))))

    def _create_ch(self,entry):
        definitions=[f"`{f['name']}` {f['physical_types']['clickhouse']}" for f in entry['fields']]
        partition='toYYYYMM(`event_date`)' if entry['partition_key']!='tuple()' else 'tuple()'
        keys=','.join('`'+k+'`' for k in entry['keys'])
        self.s.ch.execute(f"CREATE TABLE {self.s.ch.table(entry['name'])} ({','.join(definitions)}) ENGINE=ReplacingMergeTree PARTITION BY {partition} ORDER BY ({keys})")

    def apply(self,token):
        # A concurrent config save must never redirect an approved batch midway.
        from .storage import Postgres
        from .clickhouse import ClickHouse
        frozen=copy(self);frozen.s=copy(self.s)
        settings=frozen.s.settings
        frozen.s.pg=Postgres(settings);frozen.s.ch=ClickHouse(settings)
        return frozen._apply(token)

    def _apply(self,token):
        from .publishing import operation_lock
        with operation_lock(self.s):
            plan=self.s.state.ledger_get('schema_bundle_plan',token)
            if not plan:raise ValueError('建表预览不存在，请重新预览')
            if plan['target_fingerprint']!=self.s.settings.fingerprint or plan['manifest_version']!=self.manifest()['version']:raise ValueError('配置或定义已变化，请重新预览')
            completed=self.s.state.ledger_get('schema_bundle_result',token)
            if completed and completed['status']=='completed':return completed
            fresh=self.plan(plan['selections'])
            if fresh['snapshot_hash']!=plan['snapshot_hash']:raise ValueError('结构状态已变化，请重新预览')
            with self.s.state.connect() as conn:
                active=conn.execute("SELECT 1 FROM tasks WHERE status='running' AND operation NOT IN ('pipeline_validate','preview','reference_preview') LIMIT 1").fetchone()
            if active:raise ValueError('有数据操作正在执行，请结束后再建表')
            entries=self.entries();results=[];operation=uuid.uuid4().hex
            record={'operation_id':operation,'token':token,'status':'running','started_at':now(),'results':results}
            self.s.state.ledger_put('schema_bundle_result',token,record)
            for target,names in plan['expanded'].items():
                try:
                    window=self.s.ch.write_window('schema-bundle:'+operation) if target=='clickhouse' else nullcontext()
                    with window:
                        actions={a['name']:a for a in plan['actions'] if a['target']==target};bad=set()
                        if all(actions[n].get('error') for n in names):raise ValueError(actions[names[0]]['error'])
                        if not plan['database_exists'][target]:
                            if target=='postgres':self.s.pg.create_database()
                            else:self.s.ch.execute(f'CREATE DATABASE IF NOT EXISTS `{identifier(self.s.settings.ch_database)}`')
                        for name in names:
                            action=actions[name];entry=entries[name];row={'target':target,'name':name}
                            try:
                                if action['action']=='blocked':
                                    row['differences']=action.get('differences',[])
                                    raise ValueError(action.get('error') or '已有结构不兼容，请查看结构差异')
                                if bad.intersection(entry['dependencies']):raise ValueError('必需依赖创建失败：'+','.join(sorted(bad.intersection(entry['dependencies']))))
                                if action['action']=='skip':row['status']='existing'
                                else:
                                    if target=='postgres':self._create_pg(entry)
                                    else:self._create_ch(entry)
                                    checked=self.inspect([target],[name])[target]
                                    if not checked['connected'] or checked['tables'][0]['status']!='ready':raise ValueError('创建后结构核验未通过')
                                    row['status']='created'
                            except Exception as exc:row.update(status='failed',error=self.s.safe_error(exc));bad.add(name)
                            results.append(row)
                            self.s.state.ledger_put('schema_bundle_result',token,record)
                except Exception as exc:
                    existing={r['name'] for r in results if r['target']==target}
                    results.extend({'target':target,'name':n,'status':'failed','error':self.s.safe_error(exc)} for n in names if n not in existing)
            record.update(status='completed' if all(r['status'] in {'created','existing'} for r in results) else 'partial',finished_at=now())
            self.s.state.ledger_put('schema_bundle_result',token,record);return record
