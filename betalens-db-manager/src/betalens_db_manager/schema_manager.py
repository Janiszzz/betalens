"""Explicit, resumable table lifecycle operations for peer database targets."""
from __future__ import annotations
import uuid
from psycopg2 import sql
from .schema_contract import compare_structure
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

    def _row_count(self,name,key):
        target=self.service.pg if key=='postgres' else self.service.ch
        if key=='postgres':
            with target.connect(readonly=True) as conn,conn.cursor() as cur:
                cur.execute(sql.SQL('SELECT count(*) FROM {}').format(relation(target.settings,name)))
                return int(cur.fetchone()[0])
        return int(target.execute(f'SELECT count() FROM {target.table(name)} FINAL'))

    def _null_count(self,name,key,column):
        target=self.service.pg if key=='postgres' else self.service.ch
        identifier(column)
        if key=='postgres':
            with target.connect(readonly=True) as conn,conn.cursor() as cur:
                cur.execute(sql.SQL('SELECT count(*) FROM {} WHERE {} IS NULL').format(
                    relation(target.settings,name),sql.Identifier(column)))
                return int(cur.fetchone()[0])
        return int(target.execute(f'SELECT countIf(isNull(`{column}`)) FROM {target.table(name)} FINAL'))

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
            difference=compare_structure(spec,actual,key)
            added=difference['added'];changed=difference['changed'];removed=difference['removed']
            nullable_changes=difference['nullable_changes'];key_change=difference['key_change']
            requires_rebuild=difference['requires_rebuild']
            required_added=[f['name'] for f in added if f['name'] in spec.keys or not f.get('nullable',True)]
            tightening=[item['column'] for item in nullable_changes if item['to'] is False]
            row_count=self._row_count(name,key) if actual['exists'] and (required_added or tightening) else None
            null_counts={column:self._null_count(name,key,column) for column in tightening} if actual['exists'] else {}
            blocked=[]
            if row_count and required_added:
                blocked.append('已有数据的表新增必填列须先执行可审计回填：'+','.join(required_added))
            for column,count in null_counts.items():
                if count:
                    blocked.append(f'{column} 含 {count} 个历史空值，须先审计回填')
            if key=='clickhouse' and actual['exists'] and requires_rebuild and getattr(target,'write_coordinator',None) is None:
                blocked.append('ClickHouse 重建须先建立可信的受控写入维护窗口')
            if key=='postgres':
                q=lambda x:'"'+x.replace('"','""')+'"'
                table=q(target.settings.pg_schema)+'.'+q(name)
                ddl=['CREATE TABLE '+table+' ('+', '.join(q(f['name'])+' '+pg_type(f)+(' NOT NULL' if f['name'] in spec.keys or not f.get('nullable',True) else '') for f in spec.fields)+', PRIMARY KEY ('+','.join(q(k) for k in spec.keys)+'))'] if not actual['exists'] else []
                if actual['exists'] and not requires_rebuild:
                    ddl.extend('ALTER TABLE '+table+' ADD COLUMN '+q(f['name'])+' '+pg_type(f)+(' NOT NULL' if f['name'] in spec.keys or not f.get('nullable',True) else '') for f in added)
                    ddl.extend('ALTER TABLE '+table+' ALTER COLUMN '+q(item['column'])+' TYPE bigint USING '+q(item['column'])+'::bigint' for item in difference['safe_widenings'])
                    ddl.extend('ALTER TABLE '+table+' ALTER COLUMN '+q(item['column'])+(' DROP NOT NULL' if item['to'] else ' SET NOT NULL') for item in nullable_changes)
            else:
                table=target.table(name)
                ddl=[target.ddl(spec)] if not actual['exists'] else []
                if actual['exists'] and not requires_rebuild:
                    ddl.extend(f"ALTER TABLE {table} ADD COLUMN `{f['name']}` {target._type(f,f['name'] in spec.keys)}" for f in added)
                    by_name={f['name']:f for f in spec.fields}
                    ddl.extend(f"ALTER TABLE {table} MODIFY COLUMN `{item['column']}` {target._type(by_name[item['column']],item['column'] in spec.keys)}" for item in nullable_changes)
            if action=='rename':
                destination=self.inspect(rename_to,key)
                if actual['exists'] and destination['exists']:raise ValueError('目标表名已经存在：'+rename_to)
                ddl=[('ALTER TABLE '+table+' RENAME TO '+q(rename_to)) if key=='postgres' else f'RENAME TABLE {table} TO {target.table(rename_to)}']
            elif action=='drop':ddl=['DROP TABLE '+table]
            elif requires_rebuild:ddl+=['REBUILD WITH RETAINED BACKUP: '+table]
            result['targets'].append({'target':key,'actual':actual,'added':[f['name'] for f in added],
                'changed':changed,'nullable_changes':nullable_changes,'removed':removed,
                'safe_widenings':difference['safe_widenings'],
                'key_change':key_change,'requires_rebuild':requires_rebuild,
                'row_count':row_count,'null_counts':null_counts,'blocked':blocked,'ddl':ddl})
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
        if plan['action']=='deploy':
            blocked=[f"{item['target']}: {message}" for item in plan['targets'] for message in item['blocked']]
            if blocked:raise ValueError('结构迁移已阻断：'+'; '.join(blocked))
            if any(r['requires_rebuild'] for r in plan['targets']) and not plan['destructive']:
                raise ValueError('修改类型、主键或删除字段需要明确选择破坏性迁移并重新预览')
        if plan['action']=='rename' and any(not compare_structure(spec,item['actual'],item['target'])['ready'] for item in plan['targets']):
            raise ValueError('目标表结构已漂移，请先完成结构修复再重命名')
        if plan['action']=='drop' and self.registry.record(name).get('status')!='archived':raise ValueError('请先归档表再执行物理删除')
        active=[j for j in self.state.jobs(limit=1000) if j['status'] in {'downloading','committing'}]
        if active:raise ValueError('存在执行中的数据任务，结构迁移暂不可执行')
        task_id=uuid.uuid4().hex;results=[]
        self.state.ledger_put('schema_operation',task_id,{'task_id':task_id,'plan':plan,'status':'running','created_at':now(),'results':[]})
        for item in plan['targets']:
            key=item['target'];target=self.service.pg if key=='postgres' else self.service.ch
            result={'target':key,'status':'running'}
            try:
                from contextlib import ExitStack
                with ExitStack() as windows:
                    if key=='clickhouse' and getattr(target,'write_coordinator',None) is not None:
                        windows.enter_context(target.write_window('schema:'+task_id))
                    if item['actual']['exists'] and self.inspect(name,key)!=item['actual']:
                        raise ValueError('结构在执行前发生变化，请重新预览')
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
                    else:
                        if key=='postgres' and item.get('safe_widenings'):
                            with target.connect() as conn,conn.cursor() as cur:
                                cur.execute("SET LOCAL lock_timeout='5s'")
                                cur.execute("SET LOCAL statement_timeout='120s'")
                                for widening in item['safe_widenings']:
                                    column=sql.Identifier(widening['column'])
                                    cur.execute(sql.SQL('ALTER TABLE {} ALTER COLUMN {} TYPE bigint USING {}::bigint').format(relation(target.settings,name),column,column))
                        target.install(specs={name:spec})
                        self._alter_nullable(spec,target,key,item['nullable_changes'])
                    self._verify_result(plan,spec,key)
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
            elif plan['action']=='deploy':deployed[r['target']]={'version':spec.hash,'definition_version':plan['version'],'status':'drifted','at':now(),'error':r.get('error')}
        status='archived' if plan['action']=='drop' else 'active' if all(r['status']=='completed' for r in results) else 'pending'
        self.state.ledger_put('table_definition',name,{**record,'deployed':deployed,'status':status})
        report={'task_id':task_id,'plan':plan,'status':'completed' if all(r['status']=='completed' for r in results) else 'partial','results':results,'finished_at':now()}
        self.state.ledger_put('schema_operation',task_id,report)
        return report

    def _alter_nullable(self,spec,target,key,changes):
        fields={field['name']:field for field in spec.fields}
        for item in changes:
            column=item['column']
            if key=='postgres':
                with target.connect() as conn,conn.cursor() as cur:
                    action=sql.SQL('DROP NOT NULL' if item['to'] else 'SET NOT NULL')
                    cur.execute(sql.SQL('ALTER TABLE {} ALTER COLUMN {} {}').format(
                        relation(target.settings,spec.name),sql.Identifier(column),action))
            else:
                typ=target._type(fields[column],column in spec.keys)
                target.execute(f'ALTER TABLE {target.table(spec.name)} MODIFY COLUMN `{column}` {typ}')

    def _verify_result(self,plan,spec,key):
        name=plan['table'];action=plan['action']
        if action=='drop':
            if self.inspect(name,key)['exists']:
                raise ValueError('删除后目标表仍存在')
            return
        actual=self.inspect(plan['rename_to'] if action=='rename' else name,key)
        difference=compare_structure(spec,actual,key)
        if not difference['ready']:
            raise ValueError('结构操作后目标库仍存在类型、可空性或主键漂移：'+str(difference))

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
        old_count=self._row_count(spec.name,key)
        if required and old_count:raise ValueError('新必需列没有历史值，需先安排可审计补数：'+','.join(required))
        if key=='postgres':
            with target.connect() as conn,conn.cursor() as cur:
                cur.execute(sql.SQL('LOCK TABLE {} IN ACCESS EXCLUSIVE MODE').format(relation(target.settings,spec.name)))
                cur.execute(sql.SQL('SELECT count(*) FROM {}').format(relation(target.settings,spec.name)))
                locked_count=int(cur.fetchone()[0])
                if required and locked_count:
                    raise ValueError('新必需列没有历史值，需先安排可审计补数：'+','.join(required))
                target._create_table(cur,replace(spec,name=temporary))
                if shared and locked_count:
                    cols=sql.SQL(',').join(sql.Identifier(n) for n in shared)
                    cur.execute(sql.SQL('INSERT INTO {} ({}) SELECT {} FROM {}').format(relation(target.settings,temporary),cols,cols,relation(target.settings,spec.name)))
                elif locked_count:
                    raise ValueError('结构迁移没有可复制字段')
                target._indexes(cur,replace(spec,name=temporary))
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
