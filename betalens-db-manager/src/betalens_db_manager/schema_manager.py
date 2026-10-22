"""Explicit, resumable table lifecycle operations for peer database targets."""
from __future__ import annotations
import uuid
from psycopg2 import sql
from .storage import pg_type, relation, CONTROL_DDL
from .util import digest, identifier, now


class SchemaManager:
    def __init__(self,service):self.service=service;self.registry=service.registry;self.state=service.state

    def inspect(self,name,target):
        target=self.service.pg if target=='postgres' else self.service.ch
        if hasattr(target,'_existing_columns'):
            with target.connect(readonly=True) as conn,conn.cursor() as cur:
                cur.execute('SELECT column_name,data_type,is_nullable,numeric_precision,numeric_scale FROM information_schema.columns WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position',(target.settings.pg_schema,name))
                fields=[dict(name=r[0],physical_type=r[1],nullable=r[2]=='YES',precision=r[3],scale=r[4]) for r in cur.fetchall()]
                cur.execute('SELECT kcu.column_name FROM information_schema.table_constraints tc JOIN information_schema.key_column_usage kcu USING(constraint_catalog,constraint_schema,constraint_name) WHERE tc.table_schema=%s AND tc.table_name=%s AND tc.constraint_type=\'PRIMARY KEY\' ORDER BY kcu.ordinal_position',(target.settings.pg_schema,name))
                keys=[r[0] for r in cur.fetchall()]
        else:
            fields=target._json_rows(target.execute(f"SELECT name,type AS physical_type,is_in_primary_key FROM system.columns WHERE database={target._literal(target.settings.ch_database)} AND table={target._literal(name)} ORDER BY position FORMAT JSONEachRow"))
            sorting=target.execute(f"SELECT sorting_key FROM system.tables WHERE database={target._literal(target.settings.ch_database)} AND name={target._literal(name)} FORMAT TSVRaw")
            keys=[k.strip().strip('`') for k in sorting.removeprefix('tuple(').removesuffix(')').split(',') if k.strip()]
        return {'fields':fields,'keys':keys,'exists':bool(fields)}

    def plan(self,name,targets=None,action='deploy',destructive=False,rename_to=None):
        self.registry.refresh()
        if name in self.registry.errors:raise ValueError('表定义 YAML 无效：'+self.registry.errors[name])
        spec=self.registry.spec(name);definition=self.registry.get(name)
        targets=targets or ['postgres','clickhouse']
        if not targets or set(targets)-{'postgres','clickhouse'}:raise ValueError('目标库无效')
        if action not in {'deploy','drop','rename'}:raise ValueError('结构操作无效')
        if action=='rename':
            identifier(rename_to)
            if self.registry.record(rename_to):raise ValueError('新表名已登记')
        result={'rename_to':rename_to,'table':name,'version':definition['version'],'action':action,'targets':[], 'destructive':destructive}
        for key in targets:
            target=self.service.pg if key=='postgres' else self.service.ch
            try:actual=self.inspect(name,key)
            except Exception as exc:
                # Distinguish an absent database from a failed connection.
                if key=='postgres':
                    with target.connect(readonly=True,database='postgres') as conn,conn.cursor() as cur:
                        cur.execute('SELECT 1 FROM pg_database WHERE datname=%s',(target.settings.pg_database,))
                        if cur.fetchone():raise
                else:
                    if target.execute(f"EXISTS DATABASE `{identifier(target.settings.ch_database)}`")=='1':raise
                actual={'fields':[],'keys':[],'exists':False}
            current={f['name']:f for f in actual['fields']};added=[];changed=[]
            for f in spec.fields:
                old=current.get(f['name'])
                if not old:added.append(f);continue
                expected=pg_type(f) if key=='postgres' else target._type(f,f['name'] in spec.keys)
                matches=(old['physical_type']==expected or (f['type']=='numeric' and old['physical_type']=='numeric' and old.get('precision')==f.get('precision',38) and old.get('scale')==f.get('scale',12)))
                if not matches:changed.append({'column':f['name'],'from':old['physical_type'],'to':expected})
            removed=[c for c in current if c not in {f['name'] for f in spec.fields}]
            key_change=actual['exists'] and actual['keys']!=list(spec.keys)
            requires_rebuild=bool(changed or removed or key_change)
            if key=='postgres':
                q=lambda x:'"'+x.replace('"','""')+'"'
                table=q(target.settings.pg_schema)+'.'+q(name)
                ddl=['CREATE TABLE '+table+' ('+', '.join(q(f['name'])+' '+pg_type(f)+(' NOT NULL' if f['name'] in spec.keys or not f.get('nullable',True) else '') for f in spec.fields)+', PRIMARY KEY ('+','.join(q(k) for k in spec.keys)+'))'] if not actual['exists'] else ['ALTER TABLE '+table+' ADD COLUMN '+q(f['name'])+' '+pg_type(f) for f in added]
            else:
                table=target.table(name)
                ddl=[target.ddl(spec)] if not actual['exists'] else [f"ALTER TABLE {table} ADD COLUMN `{f['name']}` {target._type(f,f['name'] in spec.keys)}" for f in added]
            if action=='rename':
                destination=self.inspect(rename_to,key)
                if actual['exists'] and destination['exists']:raise ValueError('目标表名已经存在：'+rename_to)
                ddl=[('ALTER TABLE '+table+' RENAME TO '+q(rename_to)) if key=='postgres' else f'RENAME TABLE {table} TO {target.table(rename_to)}']
            elif action=='drop':ddl=['DROP TABLE '+table]
            elif requires_rebuild:ddl+=['REBUILD WITH RETAINED BACKUP: '+table]
            result['targets'].append({'target':key,'actual':actual,'added':[f['name'] for f in added],'changed':changed,'removed':removed,'key_change':key_change,'requires_rebuild':requires_rebuild,'ddl':ddl})
        result['token']=digest(result);self.state.ledger_put('schema_plan',result['token'],result)
        return result

    def apply(self,token):
        from .publishing import operation_lock
        with operation_lock(self.service):return self._apply(token)

    def _apply(self,token):
        plan=next((r for r in self.state.ledger_rows('schema_plan') if r['token']==token),None)
        if not plan:raise ValueError('结构计划不存在')
        fresh=self.plan(plan['table'],[r['target'] for r in plan['targets']],plan['action'],plan['destructive'],plan.get('rename_to'))
        if fresh['token']!=token:raise ValueError('结构或 YAML 已变化，请重新预览')
        name=plan['table'];spec=self.registry.spec(name)
        if any(r['requires_rebuild'] for r in plan['targets']) and not plan['destructive']:raise ValueError('修改类型、主键或删除字段需要明确选择破坏性迁移并重新预览')
        if plan['action']=='drop' and self.registry.record(name).get('status')!='archived':raise ValueError('请先归档表再执行物理删除')
        active=[j for j in self.state.jobs(limit=1000) if j['status'] in {'downloading','committing'}]
        if active:raise ValueError('存在执行中的数据任务，结构迁移暂不可执行')
        task_id=uuid.uuid4().hex;results=[]
        self.state.ledger_put('schema_operation',task_id,{'task_id':task_id,'plan':plan,'status':'running','created_at':now(),'results':[]})
        for item in plan['targets']:
            key=item['target'];target=self.service.pg if key=='postgres' else self.service.ch
            result={'target':key,'status':'running'}
            try:
                if plan['action']=='rename':
                    if self.inspect(name,key)['exists']:
                        if key=='postgres':
                            with target.connect() as conn,conn.cursor() as cur:cur.execute(sql.SQL('ALTER TABLE {} RENAME TO {}').format(relation(target.settings,name),sql.Identifier(plan['rename_to'])))
                        else:target.execute(f'RENAME TABLE {target.table(name)} TO {target.table(plan["rename_to"])}')
                elif plan['action']=='drop':
                    if key=='postgres':
                        with target.connect() as conn,conn.cursor() as cur:cur.execute(sql.SQL('DROP TABLE IF EXISTS {}').format(relation(target.settings,name)))
                    else:target.execute(f'DROP TABLE IF EXISTS {target.table(name)}')
                elif item['requires_rebuild']:
                    self._rebuild(spec,target,key,task_id,result)
                else:target.install(specs={name:spec})
                if plan['action']=='deploy':self._metadata(target,spec)
                result.update(status='completed',finished_at=now())
            except Exception as exc:result.update(status='failed',error=self.service.safe_error(exc),finished_at=now())
            results.append(result)
            self.state.ledger_put('schema_operation',task_id,{'task_id':task_id,'plan':plan,'created_at':now(),'status':'running','results':results})
        if plan['action']=='rename' and all(r['status']=='completed' for r in results):
            value=dict(self.registry.get(name));value['name']=plan['rename_to'];self.registry.save(value)
            mapping=dict(self.service.mappings.get(name));mapping['table']=plan['rename_to'];self.service.mappings.save(mapping,None)
            self.archive(name)
            name=plan['rename_to'];spec=self.registry.spec(name)
            for target in (self.service.pg if r['target']=='postgres' else self.service.ch for r in results):self._metadata(target,spec)
        record=self.registry.record(name);deployed=record.get('deployed',{})
        for r in results:
            if r['status']=='completed':deployed[r['target']]={'version':spec.hash,'definition_version':plan['version'],'status':'dropped' if plan['action']=='drop' else 'ready','at':now()}
        status='archived' if plan['action']=='drop' else 'active' if all(r['status']=='completed' for r in results) else 'pending'
        self.state.ledger_put('table_definition',name,{**record,'deployed':deployed,'status':status})
        report={'task_id':task_id,'plan':plan,'status':'completed' if all(r['status']=='completed' for r in results) else 'partial','results':results,'finished_at':now()}
        self.state.ledger_put('schema_operation',task_id,report)
        return report

    def _metadata(self,target,spec):
        from .metadata import records
        metadata=records()
        target.control_write('meta_table',[r for r in metadata['meta_table'] if r['table_name']==spec.name])
        target.control_write('meta_column',[r for r in metadata['meta_column'] if r['table_name']==spec.name])
        target.control_write('meta_source_mapping',[r for r in metadata['meta_source_mapping'] if r['canonical_table']==spec.name])

    def _rebuild(self,spec,target,key,task_id,result):
        from dataclasses import replace
        backup=('backup_'+task_id[:8]+'_'+spec.name)[:63];temporary=('migrate_'+task_id[:8]+'_'+spec.name)[:63]
        old=self.inspect(spec.name,key);shared=[f['name'] for f in spec.fields if f['name'] in {x['name'] for x in old['fields']}]
        required=[f['name'] for f in spec.fields if (f['name'] in spec.keys or not f.get('nullable',True)) and f['name'] not in shared]
        if required:raise ValueError('新必需列没有历史值，需先安排可审计补数：'+','.join(required))
        if key=='postgres':
            with target.connect() as conn,conn.cursor() as cur:
                cur.execute(sql.SQL('LOCK TABLE {} IN ACCESS EXCLUSIVE MODE').format(relation(target.settings,spec.name)))
                target._create_table(cur,replace(spec,name=temporary))
                cols=sql.SQL(',').join(sql.Identifier(n) for n in shared)
                cur.execute(sql.SQL('INSERT INTO {} ({}) SELECT {} FROM {}').format(relation(target.settings,temporary),cols,cols,relation(target.settings,spec.name)))
                cur.execute(sql.SQL('ALTER TABLE {} RENAME TO {}').format(relation(target.settings,spec.name),sql.Identifier(backup)))
                cur.execute(sql.SQL('ALTER TABLE {} RENAME TO {}').format(relation(target.settings,temporary),sql.Identifier(spec.name)))
        else:
            target.execute(target.ddl(replace(spec,name=temporary)))
            cols=','.join('`'+n+'`' for n in shared)
            target.execute(f'INSERT INTO {target.table(temporary)} ({cols}) SELECT {cols} FROM {target.table(spec.name)} FINAL')
            count=target.execute(f'SELECT count() FROM {target.table(temporary)} FINAL');before=target.execute(f'SELECT count() FROM {target.table(spec.name)} FINAL')
            if count!=before:raise ValueError('迁移前后行数不一致，尚未发布')
            target.execute(f'EXCHANGE TABLES {target.table(spec.name)} AND {target.table(temporary)}')
            target.execute(f'RENAME TABLE {target.table(temporary)} TO {target.table(backup)}')
        result['retained_backup']=backup

    def archive(self,name):
        value=self.registry.record(name)
        if not value:raise ValueError('表不存在')
        self.state.ledger_put('table_definition',name,{**value,'status':'archived','updated_at':now()})
        return {'table':name,'status':'archived'}
