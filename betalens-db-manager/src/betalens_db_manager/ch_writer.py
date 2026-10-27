"""Conservative single-host writer gates for explicit ClickHouse write modes.

The account and its password belong solely to the manager's local write entry
points. Database administrators remain trusted maintenance operators; their
out-of-band writes must be stopped during maintenance. Remote/role-based writer
deployments require their own enforced coordinator and are rejected here.

``local_admin`` is an explicit alternative for local installations retaining
their existing administrator. It shares the same lock and recovery journal; it
does not claim that an administrator credential prevents external direct SQL.
"""
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit
import ipaddress
import os
import tempfile
import uuid

from .commit_protocol import NeedsReconciliation
from .util import atomic_json,digest,now


def is_write(right):
    return right in {'ALL','WRITE','INSERT','DELETE','UPDATE','TRUNCATE','OPTIMIZE','ATTACH','DETACH','REPLACE','RENAME','EXCHANGE','RESTORE'} or right.startswith(('ALTER','CREATE','DROP','UNDROP','SYSTEM','KILL','MOVE'))


class LocalDedicatedWriter:
    def __init__(self,settings):
        host=urlsplit(settings.ch_url).hostname
        if host not in {'localhost','127.0.0.1','::1'}:raise ValueError('本机专用写入口只支持本机 ClickHouse')
        if os.name!='posix':raise ValueError('本机受控写入口暂仅支持 WSL/Linux')
        # TMPDIR, state directories and localhost spelling cannot split one
        # database's coordination domain across independent manager processes.
        root=Path('/tmp')/('betalens-ch-writer-'+str(os.getuid()))
        if root.is_symlink():raise ValueError('写入锁目录不能是符号链接')
        root.mkdir(mode=0o700,exist_ok=True);root.chmod(0o700)
        if hasattr(os,'getuid') and root.stat().st_uid!=os.getuid():raise ValueError('写入锁目录所有者不符')
        url=urlsplit(settings.ch_url)
        name=digest([url.scheme,'local',url.port or (443 if url.scheme=='https' else 80),url.path.rstrip('/'),settings.ch_database])
        self.lock=root/(name+'.lock');self.journal=root/(name+'.json')

    def verify(self,adapter):
        s=adapter.settings
        current=adapter.execute('SELECT currentUser() FORMAT TSVRaw')
        if current!=s.ch_user or not s.ch_password:raise NeedsReconciliation('必须配置具有独立凭据的专用写账号')
        grants=adapter._json_rows(adapter.execute('SELECT user_name,role_name,access_type,database,table,is_partial_revoke FROM system.grants FORMAT JSONEachRow'))
        by_user={}
        for g in grants:
            if not g.get('is_partial_revoke') and g.get('user_name'):by_user.setdefault(g['user_name'],set()).add(g['access_type'])
        admins={u for u,rights in by_user.items() if 'ALL' in rights or 'ACCESS MANAGEMENT' in rights or {'CREATE USER','ALTER USER','ROLE ADMIN'}<=rights}
        if current in admins:raise NeedsReconciliation('管理器写账号不能是 ClickHouse 管理员，请先部署专用账号')
        own=[g for g in grants if g.get('user_name')==current and not g.get('is_partial_revoke')]
        if not any(g['access_type'] in {'INSERT','WRITE','ALL'} and g.get('database')==s.ch_database for g in own):raise NeedsReconciliation('专用账号缺少目标库写权限')
        for g in grants:
            if g.get('is_partial_revoke') or not is_write(g['access_type']):continue
            if g.get('user_name')==current:
                if g.get('database')!=s.ch_database:raise NeedsReconciliation('专用账号写权限必须仅限当前目标库')
            elif g.get('database') in (None,'*',s.ch_database) and g.get('user_name') not in admins:
                raise NeedsReconciliation('存在其他普通账号或角色的旁路写权限，已阻止维护操作')
        roles=adapter._json_rows(adapter.execute('SELECT user_name,role_name,granted_role_name FROM system.role_grants FORMAT JSONEachRow'))
        if any(r.get('user_name')==current for r in roles):raise NeedsReconciliation('第一版专用写账号不支持角色继承')
        users=adapter._json_rows(adapter.execute('SELECT name,host_ip,host_names,host_names_regexp,host_names_like FROM system.users FORMAT JSONEachRow'))
        owner=next((u for u in users if u['name']==current),None)
        if not owner:raise NeedsReconciliation('无法读取专用写账号连接限制')
        hosts=owner.get('host_ip',[])
        observed=ipaddress.ip_address(adapter.execute('SELECT toString(address) FROM system.processes WHERE query_id=currentQueryID() FORMAT TSVRaw'))
        observed=getattr(observed,'ipv4_mapped',None) or observed
        if not hosts and not owner.get('host_names'):raise NeedsReconciliation('专用写账号必须限定本机连接')
        for host in hosts:
            network=ipaddress.ip_network(host,strict=False)
            loopback=ipaddress.ip_network('127.0.0.0/8' if network.version==4 else '::1/128')
            single_client=network.prefixlen==network.max_prefixlen and network.network_address==observed
            if not network.subnet_of(loopback) and not single_client:raise NeedsReconciliation('专用写账号必须限定本机或容器转发的单一客户端地址')
        if any(h!='localhost' for h in owner.get('host_names',[])) or owner.get('host_names_regexp') or owner.get('host_names_like'):
            raise NeedsReconciliation('专用写账号包含不受支持的主机匹配规则')
        engines=adapter._json_rows(adapter.execute(f'SELECT name,engine FROM system.tables WHERE database={adapter._literal(s.ch_database)} FORMAT JSONEachRow'))
        if any(r['engine']!='ReplacingMergeTree' for r in engines):raise NeedsReconciliation('受控写入口仅支持本机同步 ReplacingMergeTree 表')
        return {'writer':current,'trusted_administrators':sorted(admins),'database':s.ch_database}

    @contextmanager
    def exclusive(self,adapter,operation):
        import fcntl,json
        flags=os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0)
        with os.fdopen(os.open(self.lock,flags,0o600),'a+') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise NeedsReconciliation('同一 ClickHouse 目标已有写操作，请稍后重试') from None
            self.verify(adapter)
            prior=json.loads(self.journal.read_text()) if self.journal.exists() else {}
            if prior.get('inflight'):
                query_id=prior['query_id']
                active=int(adapter.execute(f'SELECT count() FROM system.processes WHERE query_id={adapter._literal(query_id)}'))
                events=adapter._json_rows(adapter.execute(f'SELECT type FROM system.query_log WHERE query_id={adapter._literal(query_id)} AND type IN (\'QueryFinish\',\'ExceptionBeforeStart\',\'ExceptionWhileProcessing\') ORDER BY event_time_microseconds DESC LIMIT 1 FORMAT JSONEachRow'))
                if active or not events:raise NeedsReconciliation('前次写入结果不确定，需等待服务器终止记录后恢复原任务：'+query_id)
                recover=prior.get('operation')==operation
                if not recover and operation.startswith('rollback:') and getattr(adapter,'state',None):
                    rid=operation.split(':',1)[1]
                    recover=any(r.get('rollback_task_id')==rid and prior.get('operation')=='commit:'+r['source_task_id']
                                for r in adapter.state.ledger_rows('import_rollback'))
                if not recover:raise NeedsReconciliation('前次写请求已终止；必须先恢复原操作并核验发布结果，禁止其他任务写入')
                atomic_json(self.journal,{**prior,'inflight':False,'reconciled_at':now()})
            try:yield
            finally:fcntl.flock(handle,fcntl.LOCK_UN)

    def before_request(self,query,operation):
        query_id='betalens_'+uuid.uuid4().hex
        atomic_json(self.journal,{'query_id':query_id,'query_hash':digest(query),'operation':operation,'inflight':True,'started_at':now()})
        self.journal.chmod(0o600)
        return query_id

    def finished_request(self,query_id):
        import json
        record=json.loads(self.journal.read_text())
        if record.get('query_id')==query_id:atomic_json(self.journal,{**record,'inflight':False,'finished_at':now()})


class LocalAdminWriter(LocalDedicatedWriter):
    """Opt-in administrator credentials with the identical local write protocol."""

    def __init__(self,settings):
        if settings.ch_writer_mode!='local_admin':
            raise ValueError('管理员受控写入必须显式配置 CLICKHOUSE_WRITER_MODE=local_admin')
        super().__init__(settings)

    def verify(self,adapter):
        s=adapter.settings
        current=adapter.execute('SELECT currentUser() FORMAT TSVRaw')
        if current!=s.ch_user or not s.ch_password:
            raise NeedsReconciliation('本机管理员写入必须使用已配置密码且身份一致的管理员账号')
        grants=adapter._json_rows(adapter.execute('SELECT user_name,role_name,access_type,database,table,is_partial_revoke FROM system.grants FORMAT JSONEachRow'))
        by_user={}
        for grant in grants:
            if not grant.get('is_partial_revoke') and grant.get('user_name'):
                by_user.setdefault(grant['user_name'],set()).add(grant['access_type'])
        admins={user for user,rights in by_user.items() if 'ALL' in rights or 'ACCESS MANAGEMENT' in rights or {'CREATE USER','ALTER USER','ROLE ADMIN'}<=rights}
        if current not in admins:
            raise NeedsReconciliation('local_admin 仅接受具有直接管理员授权的现有账号；普通写账号请使用 local_dedicated')
        if not any(grant.get('user_name')==current and not grant.get('is_partial_revoke') and grant['access_type'] in {'INSERT','WRITE','ALL'} and grant.get('database') in (None,'*',s.ch_database) for grant in grants):
            raise NeedsReconciliation('管理员账号缺少当前目标库的写权限')
        for grant in grants:
            if grant.get('is_partial_revoke') or not is_write(grant['access_type']):continue
            if grant.get('user_name')==current or grant.get('database') not in (None,'*',s.ch_database):continue
            if grant.get('user_name') not in admins:
                raise NeedsReconciliation('存在其他普通账号或角色对当前目标库的旁路写权限，已阻止管理员维护操作')
        readonly=adapter.execute("SELECT value FROM system.settings WHERE name='readonly' FORMAT TSVRaw")
        if readonly!='0':
            raise NeedsReconciliation('管理员连接处于只读设置，不能声明写入能力')
        engines=adapter._json_rows(adapter.execute(f'SELECT name,engine FROM system.tables WHERE database={adapter._literal(s.ch_database)} FORMAT JSONEachRow'))
        if any(row['engine']!='ReplacingMergeTree' for row in engines):
            raise NeedsReconciliation('受控管理员写入口仅支持本机同步 ReplacingMergeTree 表')
        return {'writer':current,'trusted_administrators':sorted(admins),'database':s.ch_database,
                'administrator':True,'coordination_scope':'same_local_os_user_and_target',
                'external_writes':'must_be_stopped_during_manager_writes'}
