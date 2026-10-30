from __future__ import annotations
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,threading,uuid,os
import pytest,yaml
from fastapi.testclient import TestClient
from betalens_db_manager.manager import DatabaseManagerService
from betalens_db_manager.config import Settings
from betalens_db_manager.api import create_app
from betalens_db_manager.canonical import registry_context
from betalens_db_manager.private import private_path


def manager(tmp_path,**kwargs):return DatabaseManagerService(Settings(config_dir=tmp_path/'config',state_dir=tmp_path/'state',**kwargs))

def table_definition(name='sample_prices'):
    return {'name':name,'zh_name':'示例行情','description':'按证券和日期记录精确价格','grain':'证券和日期一行','keys':['code','business_date'],'time_field':'business_date','temporal_rule':'calendar_day','fields':[{'name':'code','type':'text'},{'name':'business_date','type':'date'},{'name':'price','type':'numeric','precision':28,'scale':8,'unit':'元'}]}


def test_definitions_persist_and_generated_columns_are_protected(tmp_path):
    s=manager(tmp_path);doc=s.registry.save(table_definition())
    names={f['name'] for f in doc['definition']['fields']}
    assert {'ingested_at','source_id','schema_version','import_task_id'}<=names
    s2=manager(tmp_path);assert s2.registry.get('sample_prices')['keys']==['code','business_date']
    d=table_definition();d['keys']=['import_task_id']
    with pytest.raises(ValueError):s.registry.save(d)
    d=table_definition();d['fields'].append({'name':'price','type':'text'})
    with pytest.raises(ValueError):s.registry.save(d)


def test_all_registered_source_fields_have_typed_destinations(tmp_path):
    s=manager(tmp_path)
    assert not s.mappings.errors
    rows=s.field_audit()
    assert len({r['endpoint'] for r in rows})==31
    assert all(r['mapped'] and r['status']=='unverified' for r in rows)
    finance=s.mappings.get('fundamental_periodic')
    rules=next(b['fields'] for b in finance['bindings'] if b['endpoint']=='income')
    assert next(r for r in rules if r['source_field']=='ann_date')['column']=='announcement_date'
    assert next(r for r in rules if r['source_field']=='f_ann_date')['column']=='actual_announcement_date'


def test_yaml_external_invalid_and_optimistic_edits(tmp_path):
    s=manager(tmp_path);s.registry.save(table_definition())
    d=s.mappings.document('sample_prices')
    mapping={'format_version':1,'table':'sample_prices','bindings':[{'source_id':'tushare','endpoint':'daily','fields':[{'source_field':'close','column':'price','multiplier':'1000'}]}]}
    saved=s.mappings.save(mapping,d['file_hash'])
    path=Path(saved['path']);path.write_text('bindings: [broken')
    assert 'sample_prices' in s.mappings.refresh()
    assert s.mappings.get('sample_prices')['version']==saved['mapping']['version']
    with pytest.raises(ValueError):s.mappings.save(mapping,saved['file_hash'])
    mapping['bindings'][0]['fields'][0]['column']='ingested_at'
    with pytest.raises(ValueError):s.mappings.validate(mapping)


def test_credentials_outside_other_git_worktree_and_reload(tmp_path,monkeypatch):
    worktree=tmp_path/'different_repo';worktree.mkdir();(worktree/'.git').write_text('gitdir: /tmp/somewhere')
    with pytest.raises(ValueError):private_path(worktree/'ignored/credentials.env')
    s=manager(tmp_path);path=s.settings.write_private_env({'TUSHARE_TOKEN':'private-value','PGPORT':'5433'})
    assert path.stat().st_mode&0o777==0o600
    monkeypatch.setenv('TUSHARE_TOKEN','stale-value')
    settings=Settings.from_env(path)
    assert settings.token=='private-value' and settings.pg_port==5433
    settings.write_private_env({'TUSHARE_TOKEN':''})
    assert Settings.from_env(path).token=='private-value'


def test_decimal_priority_zero_and_invalid_value(tmp_path):
    s=manager(tmp_path);s.registry.save(table_definition());spec=s.registry.spec('sample_prices')
    binding={'source_id':'tushare','endpoint':'daily','fields':[{'source_field':'close','column':'price','priority':0},{'source_field':'open','column':'price','priority':1}]}
    row,evidence=s.mappings.map_row(spec,{'close':'0','open':'2'},binding)
    assert row['price']==Decimal(0) and evidence[1]['conflict']
    with pytest.raises(ValueError):s.mappings.map_row(spec,{'close':'NaN','open':'2'},binding)


def test_manager_routes_and_durable_task_request(tmp_path):
    s=manager(tmp_path)
    with TestClient(create_app(s)) as client:
        r=client.post('/api/tables',json={'definition':table_definition()});assert r.status_code==200,r.text
        assert 'ingested_at' in client.get('/api/tables/sample_prices/definition').text
        assert client.get('/api/sources').status_code==200
        assert client.get('/api/field-audit').status_code==200
        sid=client.post('/api/sessions').json()['id']
        assert client.post('/api/sessions/'+sid+'/heartbeat').status_code==200
        assert client.post('/api/sessions/'+sid+'/close').status_code==200
        assert client.post('/api/tables',json={'definition':table_definition()},headers={'Origin':'http://evil.invalid'}).status_code==403


@pytest.fixture
def live_manager(tmp_path):
    if os.environ.get('BETALENS_INFRA_INTEGRATION')!='1':pytest.skip('requires isolated PostgreSQL and ClickHouse')
    base=Settings.from_env('/home/janiszzz/betalens-infra/tushare.private.env')
    suffix=uuid.uuid4().hex[:10]
    settings=replace(base,config_dir=tmp_path/'config',state_dir=tmp_path/'state',archive_dir=None,env_file=None,pg_database='manager_test_'+suffix,ch_database='manager_test_'+suffix)
    assert settings.pg_database==settings.ch_database=='manager_test_'+suffix
    assert settings.pg_database!='betalens' and settings.env_file is None
    s=DatabaseManagerService(settings)
    from controlled_ch import configure_writer
    cleanup=configure_writer(s)
    try:yield s
    finally:
        cleanup()
        import psycopg2
        from psycopg2 import sql
        try:s.ch.execute(f'DROP DATABASE IF EXISTS `{settings.ch_database}`')
        finally:
            c=psycopg2.connect(**settings.pg('postgres'));c.autocommit=True
            with c.cursor() as cur:cur.execute(sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(settings.pg_database)))
            c.close()


@pytest.fixture
def mock_api():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Deliberately reordered fields and exact decimal strings.
            if self.path.startswith('/volumes'):
                value={'fields':['instrument','quantity_lots','observed'],'items':[['a','2','20261013']]}
            elif self.path.startswith('/duplicate'):
                value={'fields':['price_milli','day','ticker'],'items':[['12000','2026/10/13','a'],['13000','2026/10/13','a']]}
            elif self.path.startswith('/paged'):
                from urllib.parse import parse_qs,urlsplit
                offset=int(parse_qs(urlsplit(self.path).query).get('offset',['0'])[0]);values=[['12345.678901','2026/10/13',code] for code in ('a','b','c')]
                value={'fields':['price_milli','day','ticker'],'items':values[offset:offset+2]}
            else:value={'fields':['price_milli','day','ticker'],'items':[['12345.678901','2026/10/13','a']]}
            raw=json.dumps(value).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield 'http://127.0.0.1:'+str(server.server_port)
    server.shutdown();server.server_close()


def test_real_http_two_targets_audit_idempotency_rollback(live_manager,mock_api):
    s=live_manager;s.registry.save(table_definition())
    with registry_context(s.registry):
        plan=s.schema.plan('sample_prices');result=s.schema.apply(plan['token'])
        assert result['status']=='completed',result
    source={'source_id':'mock_vendor','provider':'http','label':'示例订阅源','base_url':mock_api,'endpoints':{'prices':{'method':'GET','strategy':'range','date_field':'day','keys':['ticker','day'],'fields':[{'name':'ticker','type':'text'},{'name':'day','type':'date'},{'name':'price_milli','type':'numeric'}]}}}
    s.sources.save(source)
    mapping={'format_version':1,'table':'sample_prices','bindings':[{'source_id':'mock_vendor','endpoint':'prices','fields':[{'source_field':'ticker','column':'code'},{'source_field':'day','column':'business_date','date_format':'%Y/%m/%d'},{'source_field':'price_milli','column':'price','multiplier':'0.001'}]}]}
    s.mappings.save(mapping,None)
    job=s.plan({'tables':['sample_prices'],'start_date':'2026-10-13','end_date':'2026-10-13'})
    s.download(job['id']);preview=s.preview(job['id']);assert not preview['rejected'],preview
    with registry_context(s.registry):assert s.pg.query('sample_prices')['rows']==[]
    result=s.commit(job['id'],preview['token']);assert result['status']=='completed',result['error']
    with registry_context(s.registry):
        pg=s.pg.query('sample_prices')['rows'][0];ch=s.ch.query('sample_prices')['rows'][0]
    assert pg['price']==ch['price']==Decimal('12.34567890')
    assert pg['ingested_at'] and ch['ingested_at'] and pg['import_task_id']==job['id']
    assert any(r['event']=='http_response' for r in s.state.logs(job['id']))
    j2=s.plan({'tables':['sample_prices'],'start_date':'2026-10-13','end_date':'2026-10-13'})
    s.download(j2['id']);p2=s.preview(j2['id']);assert all(r['counts']['skipped']==1 for r in p2['tables'])
    assert s.commit(j2['id'],p2['token'])['status']=='completed'
    assert all(r['status']=='completed' for r in s.rollback(job['id'])['results'])
    with registry_context(s.registry):assert not s.pg.query('sample_prices')['rows'] and not s.ch.query('sample_prices')['rows']


def test_peer_schema_add_column_rename_and_drop(live_manager):
    s=live_manager;s.registry.save(table_definition())
    with registry_context(s.registry):
        result=s.schema.apply(s.schema.plan('sample_prices')['token']);assert result['status']=='completed'
        old=s.registry.document('sample_prices');v=old['definition'];v['fields'].append({'name':'note','type':'text','zh_name':'说明'})
        s.registry.save(v,old['file_hash'])
        p=s.schema.plan('sample_prices');assert all('note' in t['added'] and not t['requires_rebuild'] for t in p['targets'])
        assert s.schema.apply(p['token'])['status']=='completed'
        p=s.schema.plan('sample_prices',action='rename',rename_to='renamed_prices');assert s.schema.apply(p['token'])['status']=='completed'
        assert 'renamed_prices' in s.registry.catalog() and 'sample_prices' not in s.registry.catalog()
        p=s.schema.plan('renamed_prices',action='drop')
        with pytest.raises(ValueError):s.schema.apply(p['token'])
        s.schema.archive('renamed_prices')
        assert s.pg.query('renamed_prices')['rows']==[] and s.ch.query('renamed_prices')['rows']==[]
        with pytest.raises(ValueError):s.compile({'tables':['renamed_prices'],'start_date':'2026-10-13','end_date':'2026-10-13'})
        assert s.schema.apply(s.schema.plan('renamed_prices',action='drop')['token'])['status']=='completed'


def test_sessions_close_grace_and_multiple_pages(tmp_path,monkeypatch):
    import time
    from betalens_db_manager.management_api import session_online
    s=manager(tmp_path);clock=1000;monkeypatch.setattr(time,'time',lambda:clock)
    s.state.ledger_put('browser_session','a',{'id':'a','seen':1000,'closed_at':1000})
    assert session_online(s.state)
    clock=1016;assert not session_online(s.state)
    s.state.ledger_put('browser_session','b',{'id':'b','seen':1016,'closed_at':None})
    assert session_online(s.state)
    clock=1062;assert not session_online(s.state)


def test_readonly_transform_cannot_run_code_and_all_field_defaults(tmp_path):
    s=manager(tmp_path);s.registry.save(table_definition())
    with pytest.raises(ValueError):s.mappings.validate({'format_version':1,'table':'sample_prices','bindings':[{'source_id':'tushare','endpoint':'daily','fields':[{'source_field':'close','column':'price','expression':'__import__("os").system("echo unsafe")'}]}]})
    contract=s.sources.endpoint('tushare','stock_basic')
    assert next(f for f in contract['fields'] if f['name']=='fullname')['default_display'] is False


def test_partial_acceptance_and_target_snapshot_staleness(live_manager,mock_api):
    s=live_manager;s.registry.save(table_definition())
    with registry_context(s.registry):assert s.schema.apply(s.schema.plan('sample_prices')['token'])['status']=='completed'
    source={'source_id':'vendor','provider':'http','label':'来源','base_url':mock_api,'endpoints':{'prices':{'strategy':'range','keys':['ticker','day'],'date_field':'day','fields':[{'name':'ticker','type':'text'},{'name':'day','type':'date'},{'name':'price_milli','type':'numeric'}]}}}
    s.sources.save(source)
    m={'format_version':1,'table':'sample_prices','bindings':[{'source_id':'vendor','endpoint':'prices','fields':[{'source_field':'ticker','column':'code'},{'source_field':'day','column':'business_date','date_format':'%Y/%m/%d'},{'source_field':'price_milli','column':'price'}]}]}
    s.mappings.save(m,None)
    j=s.plan({'tables':['sample_prices'],'start_date':'2026-10-13','end_date':'2026-10-13','targets':['postgres']})
    s.download(j['id'])
    part=s.state.job(j['id'])['slices'][0]
    from betalens_db_manager.source import read_archive
    from betalens_db_manager.util import dumps
    import gzip,hashlib
    # A value conversion failure can be partially accepted; an invalid date
    # would invalidate the response's request-scope/completeness evidence.
    data=read_archive(Path(part['archive']),part['sha']);data['items'].append(['bad','2026/10/13','b'])
    raw=dumps({'code':0,'data':data}).encode()
    with gzip.open(part['archive'],'wb') as f:f.write(raw)
    s.state.slice(j['id'],part['id'],'prices',part['request'],sha=hashlib.sha256(raw).hexdigest())
    preview=s.preview(j['id']);assert preview['rejected']==1
    with pytest.raises(ValueError):s.commit(j['id'],preview['token'])
    result=s.commit(j['id'],preview['token'],accept_partial=True);assert result['status']=='partial'
    with registry_context(s.registry):assert len(s.pg.query('sample_prices')['rows'])==1 and not s.ch.query('sample_prices')['rows']


def test_cleanup_keeps_archives_referenced_by_pending_migration(tmp_path):
    s=manager(tmp_path)
    from betalens_db_manager.worker import clean_cache
    from betalens_db_manager.util import dumps
    plan={'request':{'targets':['postgres']},'contracts':{}}
    original=s.state.create_job(plan,s.settings.fingerprint);consumer=s.state.create_job(plan,s.settings.fingerprint)
    path=tmp_path/'archive.json.gz';path.write_bytes(b'archive')
    for j in [original,consumer]:s.state.slice(j['id'],'part','source',{},status='downloaded',archive=str(path),sha='hash',rows=1)
    with s.state.connect() as conn:conn.execute("UPDATE jobs SET status='completed',updated='2000-01-01T00:00:00+00:00' WHERE id=?",(original['id'],))
    clean_cache(s);assert path.exists()
    with s.state.connect() as conn:conn.execute("UPDATE jobs SET status='completed',updated='2000-01-01T00:00:00+00:00' WHERE id=?",(consumer['id'],))
    clean_cache(s);assert not path.exists()
    assert s.state.job(original['id'])['plan']['cache_purged']


def test_http_paging_and_repeated_page_block(live_manager,mock_api):
    s=live_manager;s.registry.save(table_definition())
    with registry_context(s.registry):assert s.schema.apply(s.schema.plan('sample_prices')['token'])['status']=='completed'
    def register(sid,path,size):
        source={'source_id':sid,'provider':'http','base_url':mock_api,'endpoints':{'prices':{'path':path,'strategy':'range','keys':['ticker','day'],'date_field':'day','pagination':{'size':size},'fields':[{'name':'ticker','type':'text'},{'name':'day','type':'date'},{'name':'price_milli','type':'numeric'}]}}}
        s.sources.save(source)
        doc=s.mappings.document('sample_prices');value=doc['mapping']
        value['bindings'].append({'source_id':sid,'endpoint':'prices','fields':[{'source_field':'ticker','column':'code'},{'source_field':'day','column':'business_date','date_format':'%Y/%m/%d'},{'source_field':'price_milli','column':'price'}]})
        s.mappings.save(value,doc['file_hash'])
    register('paged','paged',2)
    request={'tables':['sample_prices'],'source_id':'paged','start_date':'2026-10-13','end_date':'2026-10-13','targets':['postgres']}
    job=s.plan(request);s.download(job['id']);parts=s.state.job(job['id'])['slices']
    assert sorted(p['rows'] for p in parts)==[0,1,2]
    preview=s.preview(job['id']);assert not preview['blocking'] and preview['tables'][0]['counts']['inserted']==3
    register('repeating','repeat',1)
    j=s.plan({**request,'source_id':'repeating'});s.download(j['id']);p=s.preview(j['id']);assert p['blocking']
    assert len(s.state.job(j['id'])['slices'])==2
    with pytest.raises(ValueError):s.commit(j['id'],p['token'],accept_partial=True)


def test_peer_content_hash_is_independent_of_import_history(tmp_path):
    from betalens_db_manager.comparison import row_hash,content_hash
    s=manager(tmp_path);s.registry.save(table_definition());spec=s.registry.spec('sample_prices')
    a={'code':'a','business_date':'2026-10-13','price':'12.34567890','import_task_id':'pg_task','source_id':'vendor','mapping_version':'v1'}
    b={**a,'import_task_id':'ch_task','mapping_version':'v2','downloaded_at':'2026-10-14T00:00:00Z'}
    assert row_hash(spec,a)!=row_hash(spec,b)
    assert content_hash(spec,a)==content_hash(spec,b)
    b['available_at']='2026-10-14T00:00:00Z'
    assert content_hash(spec,a)!=content_hash(spec,b)


def test_financial_publication_dates_remain_distinct(tmp_path):
    from betalens_db_manager.catalog import dataset
    from betalens_db_manager.service import normalize_source
    s=manager(tmp_path)
    job=s.plan({'datasets':['income'],'codes':['000001.SZ'],'start_date':'2025-12-31','end_date':'2025-12-31'})
    fields=[f['name'] for f in dataset('income').fields]
    raw={'ts_code':'000001.SZ','end_date':'20251231','ann_date':'20260301','f_ann_date':'20260305','report_type':'1','comp_type':'1','end_type':'4','revenue':'100'}
    rows,rejected=normalize_source(dataset('income'),{'fields':fields,'items':[[raw.get(f) for f in fields]]},job['id'],'p','2026-10-20T00:00:00Z')
    assert not rejected
    result,_=s._map_part('income',rows,job['id'],'2026-10-20T00:00:00Z',[])
    row=result['fundamental_periodic'][0]
    assert row['announcement_date']==date(2026,3,1)
    assert row['actual_announcement_date']==date(2026,3,5)
    assert row['end_type']=='4'


def test_table_schedule_creation_is_removed(tmp_path):
    s=manager(tmp_path)
    with TestClient(create_app(s)) as client:
        assert client.post('/api/schedules',json={'label':'旧计划','request':{'tables':['market_daily_raw']},'enabled':True}).status_code==410
    assert not s.state.jobs()


def test_invalid_web_config_does_not_overwrite_private_file(tmp_path):
    s=manager(tmp_path);path=s.settings.write_private_env({'TUSHARE_TOKEN':'retained-private-credential'})
    before=path.read_bytes()
    with TestClient(create_app(s)) as client:
        result=client.post('/api/config',json={'PGPORT':'bad-port','TUSHARE_TOKEN':'replacement'})
        assert result.status_code==409
    assert path.read_bytes()==before


def test_http_object_rows_preserve_late_optional_fields(tmp_path,monkeypatch):
    from betalens_db_manager.source_registry import HTTPAdapter,FetchContext
    import requests
    s=manager(tmp_path)
    job=s.state.create_job({'request':{'targets':['postgres']},'contracts':{}},s.settings.fingerprint)
    payload=[{'ticker':'a','price':'1'},{'ticker':'b','price':'2','late_field':'preserved'}]
    class Response:
        status_code=200
        content=json.dumps(payload).encode()
        def raise_for_status(self):pass
    monkeypatch.setattr(requests,'request',lambda *args,**kwargs:Response())
    contract={'fields':[{'name':'ticker','type':'text'},{'name':'price','type':'numeric'}]}
    adapter=HTTPAdapter({'source_id':'sample','base_url':'http://example.test','endpoints':{'prices':contract}})
    data=adapter.fetch({'endpoint':'prices','parameters':{}},FetchContext(s.settings,s.state,job['id']))
    assert 'late_field' in data['fields']
    assert data['items'][0][data['fields'].index('late_field')] is None
    assert data['items'][1][data['fields'].index('late_field')]=='preserved'
    with pytest.raises(ValueError):adapter.validate_response(data,contract)


def test_manual_tables_do_not_inherit_stock_only_business_rules(tmp_path):
    from betalens_db_manager.comparison import typed
    s=manager(tmp_path)
    value={'name':'custom_futures','zh_name':'手工期货数据','description':'非股票来源示例','grain':'合约一行','keys':['security_code'],'fields':[{'name':'security_code','type':'text'},{'name':'asset_type','type':'text'}]}
    s.registry.save(value)
    row=typed(s.registry.spec('custom_futures'),{'security_code':'IF2610.CFX','asset_type':'future'})
    assert row['security_code']=='IF2610.CFX' and row['asset_type']=='future'


def test_source_fields_keep_uppercase_and_chinese_names(tmp_path):
    s=manager(tmp_path);s.registry.save(table_definition())
    s.sources.save({'source_id':'local_file','provider':'file','label':'中文文件','endpoints':{'prices':{'path':'/not-read.csv','keys':['CODE','交易日期'],'fields':[{'name':'CODE','type':'text'},{'name':'交易日期','type':'date'},{'name':'收盘价(元)','type':'numeric'}]}}})
    binding={'source_id':'local_file','endpoint':'prices','fields':[{'source_field':'CODE','column':'code'},{'source_field':'交易日期','column':'business_date','date_format':'%Y年%m月%d日'},{'source_field':'收盘价(元)','column':'price'}]}
    s.mappings.validate({'format_version':1,'table':'sample_prices','bindings':[binding]})
    row,_=s.mappings.map_row(s.registry.spec('sample_prices'),{'CODE':'000001.SZ','交易日期':'2026年10月13日','收盘价(元)':'12.30'},binding)
    assert row['code']=='000001.SZ' and row['business_date']==date(2026,10,13) and row['price']==Decimal('12.30')


def test_register_new_field_then_replay_keeps_cached_response(live_manager,mock_api,monkeypatch):
    from betalens_db_manager.source import read_archive
    from betalens_db_manager.source_registry import HTTPAdapter
    from betalens_db_manager.util import dumps
    import gzip,hashlib
    s=live_manager;definition=table_definition();definition['fields'].append({'name':'extra_metric','type':'numeric'})
    s.registry.save(definition)
    with registry_context(s.registry):assert s.schema.apply(s.schema.plan('sample_prices')['token'])['status']=='completed'
    contract={'method':'GET','strategy':'range','keys':['ticker','day'],'date_field':'day','fields':[{'name':'ticker','type':'text'},{'name':'day','type':'date'},{'name':'price_milli','type':'numeric'}]}
    source={'source_id':'cached_vendor','provider':'http','base_url':mock_api,'endpoints':{'prices':contract}}
    s.sources.save(source)
    mapping={'format_version':1,'table':'sample_prices','bindings':[{'source_id':'cached_vendor','endpoint':'prices','fields':[{'source_field':'ticker','column':'code'},{'source_field':'day','column':'business_date','date_format':'%Y/%m/%d'},{'source_field':'price_milli','column':'price'}]}]}
    s.mappings.save(mapping,None)
    job=s.plan({'tables':['sample_prices'],'source_id':'cached_vendor','start_date':'2026-10-13','end_date':'2026-10-13'});s.download(job['id'])
    part=s.state.job(job['id'])['slices'][0];data=read_archive(Path(part['archive']),part['sha']);data['fields'].append('extra_value')
    for row in data['items']:row.append('17')
    raw=dumps({'code':0,'data':data}).encode()
    with gzip.open(part['archive'],'wb') as handle:handle.write(raw)
    s.state.slice(job['id'],part['id'],'prices',part['request'],sha=hashlib.sha256(raw).hexdigest())
    preview=s.preview(job['id']);assert preview['blocking']
    assert any(r['source_field']=='extra_value' and r['status']=='unregistered_blocked' for r in preview['field_validation'])
    contract['fields'].append({'name':'extra_value','type':'numeric'});s.sources.save(source)
    mapping['bindings'][0]['fields'].append({'source_field':'extra_value','column':'extra_metric'})
    s.mappings.save(mapping,s.mappings.document('sample_prices')['file_hash'])
    with TestClient(create_app(s)) as client:
        response=client.post('/api/tasks/'+job['id']+'/replay',json={});assert response.status_code==200,response.text
        linked=response.json()
    monkeypatch.setattr(HTTPAdapter,'_fetch',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('缓存重审不得再次联网')))
    s.download(linked['id']);preview=s.preview(linked['id']);assert not preview['blocking'] and not preview['rejected']
    assert preview['tables'][0]['normalized_sample'][0]['extra_metric']==Decimal('17')
    assert s.state.ledger_rows('import_source_archive',linked['id'])


def test_enum_conversion_keeps_zero_and_explicit_missing_values(tmp_path):
    s=manager(tmp_path);s.registry.save(table_definition());field={'name':'price','type':'numeric'}
    assert s.mappings.convert('N/A',{'enum':{'N/A':None}},field) is None
    assert s.mappings.convert(False,{'enum':{'true':'1','false':'0'}},field)==Decimal(0)
    assert s.mappings.convert(Decimal('1.0'),{'enum':{'1':'2'}},field)==Decimal(2)
    with pytest.raises(ValueError):s.mappings.map_row(s.registry.spec('sample_prices'),{'close':'N/A'},{'source_id':'tushare','endpoint':'daily','fields':[{'source_field':'close','column':'price','enum':{'N/A':None},'required':True}]})


def join_mapping(s):
    definition=table_definition('joined_quotes')
    definition['fields'][2]['nullable']=False
    definition['fields'].append({'name':'volume','type':'numeric','nullable':False})
    s.registry.save(definition)
    bindings=[]
    for endpoint,fields in [('prices',[('ticker','code'),('day','business_date'),('price_milli','price')]),('volumes',[('instrument','code'),('observed','business_date'),('quantity_lots','volume')])]:
        bindings.append({'source_id':'join_vendor','endpoint':endpoint,'mapping_mode':'direct','priority':len(bindings),
            'join_keys':[{'source_field':source,'column':column} for source,column in fields[:2]],
            'fields':[dict(source_field=source,column=column,**({'date_format':'%Y/%m/%d' if endpoint=='prices' else '%Y%m%d'} if column=='business_date' else {'multiplier':'0.001' if column=='price' else '100'} if column in {'price','volume'} else {})) for source,column in fields]})
    return {'format_version':2,'table':'joined_quotes','overlap_policy':'reject','bindings':bindings}


def register_join_source(s,url):
    s.sources.save({'source_id':'join_vendor','provider':'http','base_url':url,'endpoints':{
        'prices':{'path':'prices','strategy':'range','date_field':'day','keys':['ticker','day'],'fields':[{'name':'ticker','type':'text'},{'name':'day','type':'date'},{'name':'price_milli','type':'numeric'}]},
        'volumes':{'path':'volumes','strategy':'range','date_field':'observed','keys':['instrument','observed'],'fields':[{'name':'instrument','type':'text'},{'name':'observed','type':'date'},{'name':'quantity_lots','type':'numeric'}]}}})


def test_mapping_graph_rejects_duplicate_queries_and_connections(tmp_path):
    import copy
    s=manager(tmp_path);register_join_source(s,'http://127.0.0.1:1');m=join_mapping(s)
    assert not [i for i in s.mappings.diagnostics(s.mappings.validate(m)) if i['severity']=='error']
    bad=copy.deepcopy(m);bad['bindings'].append(bad['bindings'][0])
    with pytest.raises(ValueError,match='查询源重复'):s.mappings.validate(bad)
    bad=copy.deepcopy(m);bad['bindings'][0]['fields'].append({'source_field':'price_milli','column':'volume'})
    with pytest.raises(ValueError,match='源字段重复'):s.mappings.validate(bad)
    bad=copy.deepcopy(m);bad['bindings'][0]['join_keys'].append(bad['bindings'][0]['join_keys'][0])
    with pytest.raises(ValueError,match='共同键重复'):s.mappings.validate(bad)
    bad=copy.deepcopy(m);bad['bindings'][0]['join_keys'].pop()
    assert any(i['code']=='missing_join_key' for i in s.mappings.diagnostics(s.mappings.validate(bad)))


def test_mapping_graph_drafts_report_omissions_and_explicit_overlap(tmp_path):
    import copy
    s=manager(tmp_path);register_join_source(s,'http://127.0.0.1:1');m=join_mapping(s)
    incomplete=copy.deepcopy(m);incomplete['bindings'][0]['fields'].pop()
    saved=s.mappings.save(incomplete,None)
    assert any(i['code']=='unmapped_source_field' for i in saved['diagnostics'])
    with pytest.raises(ValueError,match='字段映射尚不可执行'):s.plan({'tables':['joined_quotes'],'start_date':'2026-10-13','end_date':'2026-10-13'})
    m['bindings'][1]['fields'][-1]['column']='price'
    assert any(i['code']=='duplicate_target' for i in s.mappings.diagnostics(s.mappings.validate(m)))
    m['overlap_policy']='priority';m['bindings'][1]['priority']=0
    assert any(i['code']=='ambiguous_priority' for i in s.mappings.diagnostics(s.mappings.validate(m)))
    m['bindings'][1]['ignored_fields']={'quantity_lots':'业务无关'}
    with pytest.raises(ValueError,match='同时映射和忽略'):s.mappings.validate(m)


def test_composite_key_merge_checks_required_fields_after_join(live_manager,mock_api):
    s=live_manager;register_join_source(s,mock_api);m=join_mapping(s);s.mappings.save(m,None)
    with registry_context(s.registry):assert s.schema.apply(s.schema.plan('joined_quotes')['token'])['status']=='completed'
    job=s.plan({'tables':['joined_quotes'],'start_date':'2026-10-13','end_date':'2026-10-13'})
    s.download(job['id']);p=s.preview(job['id']);assert p['rejected']==0,p
    assert all(t['normalized_sample'][0]['price']==Decimal('12.34567890') and t['normalized_sample'][0]['volume']==Decimal(200) for t in p['tables'])
    with registry_context(s.registry):assert s.pg.query('joined_quotes')['rows']==s.ch.query('joined_quotes')['rows']==[]
    assert s.commit(job['id'],p['token'])['status']=='completed'
    with registry_context(s.registry):
        for db in (s.pg,s.ch):
            row=db.query('joined_quotes')['rows'][0]
            assert row['code']=='a' and row['business_date']==date(2026,10,13) and row['volume']==200
            assert row['data_quality_status']=='mixed_sources'
    evidence=s.state.ledger_get('task_field_evidence',job['id'])['rows']
    assert any(e.get('column')=='volume' and e.get('endpoint')=='volumes' for e in evidence)
    assert all(r['status']=='completed' for r in s.rollback(job['id'])['results'])


def test_same_source_nonunique_join_key_blocks_instead_of_choosing_first(live_manager,mock_api):
    s=live_manager;register_join_source(s,mock_api);m=join_mapping(s)
    source=s.sources.values['join_vendor'];source['endpoints']['prices']['path']='duplicate';s.sources.save(source)
    s.mappings.save(m,None)
    with registry_context(s.registry):assert s.schema.apply(s.schema.plan('joined_quotes')['token'])['status']=='completed'
    job=s.plan({'tables':['joined_quotes'],'start_date':'2026-10-13','end_date':'2026-10-13'})
    s.download(job['id']);p=s.preview(job['id']);assert p['blocking'] and p['rejected']>0
    assert any('共同键不唯一' in r['reason'] for r in p['reject_sample'])
    assert all(not t['normalized_sample'] for t in p['tables'])


def cached_part(s,job,endpoint,data):
    import gzip,hashlib
    from betalens_db_manager.util import dumps,digest,atomic_json,now
    request=next(r for r in job['plan']['requests'] if r['endpoint']==endpoint)
    key=digest([request['source_id'],endpoint,request['parameters']])[:24]
    path=s.settings.state_dir/'archive'/job['id']/(key+'.json.gz');path.parent.mkdir(parents=True,exist_ok=True)
    raw=dumps({'code':0,'data':data}).encode()
    with gzip.open(path,'wb') as f:f.write(raw)
    sha=hashlib.sha256(raw).hexdigest();atomic_json(path.with_suffix('.meta.json'),{'fetched_at':now()})
    s.state.slice(job['id'],key,endpoint,request,status='downloaded',archive=str(path),sha=sha,rows=len(data['items']))
    # The real downloader also archives all planned calendar dependencies.
    for dependency in job['plan']['requests']:
        if dependency.get('strategy')!='dependency':continue
        dependency_key=digest([dependency['source_id'],dependency['endpoint'],dependency['parameters']])[:24]
        if s.state.get_slice(job['id'],dependency_key):continue
        empty={'fields':[f['name'] for f in job['plan']['source_contracts'][dependency['source_id']+':'+dependency['endpoint']]['fields']],'items':[]}
        payload=dumps({'code':0,'data':empty}).encode();target=path.parent/(dependency_key+'.json.gz')
        with gzip.open(target,'wb') as f:f.write(payload)
        atomic_json(target.with_suffix('.meta.json'),{'fetched_at':now()})
        s.state.slice(job['id'],dependency_key,dependency['endpoint'],dependency,status='downloaded',archive=str(target),sha=hashlib.sha256(payload).hexdigest(),rows=0)


def test_joint_key_does_not_join_different_dates_by_code_only(tmp_path):
    s=manager(tmp_path);register_join_source(s,'http://127.0.0.1:1');m=join_mapping(s);s.mappings.save(m,None)
    job=s.plan({'tables':['joined_quotes'],'start_date':'2026-10-13','end_date':'2026-10-14'})
    cached_part(s,job,'prices',{'fields':['ticker','day','price_milli'],'items':[['a','2026/10/13','12000']]})
    cached_part(s,job,'volumes',{'fields':['instrument','observed','quantity_lots'],'items':[['a','20261014','2']]})
    result,rejects=s._build_canonical(s.state.job(job['id']))
    assert not result and len(rejects)==2
    assert all('合并后必需字段仍为空' in r['reason'] for r in rejects)


def test_tushare_queries_can_populate_manual_tables_without_legacy_processors(tmp_path):
    s=manager(tmp_path);definition=table_definition();definition['fields'].append({'name':'volume','type':'numeric'})
    s.registry.save(definition)
    rules=[{'source_field':'ts_code','column':'code'},{'source_field':'trade_date','column':'business_date'},{'source_field':'close','column':'price'},{'source_field':'vol','column':'volume','multiplier':'100'}]
    contract=s.sources.endpoint('tushare','daily')
    binding={'source_id':'tushare','endpoint':'daily','mapping_mode':'direct','join_keys':[{'source_field':r['source_field'],'column':r['column']} for r in rules[:2]],'fields':rules,'ignored_fields':{f['name']:'本表仅记录收盘价与成交量' for f in contract['fields'] if f['name'] not in {r['source_field'] for r in rules}}}
    s.mappings.save({'format_version':2,'table':'sample_prices','bindings':[binding]},None)
    job=s.plan({'tables':['sample_prices'],'start_date':'2026-10-13','end_date':'2026-10-13','codes':['000001.SZ']})
    raw={'ts_code':'000001.SZ','trade_date':'20261013','close':'12.3456789','vol':'2'}
    fields=[f['name'] for f in contract['fields']];cached_part(s,job,'daily',{'fields':fields,'items':[[raw.get(f) for f in fields]]})
    result,rejects=s._build_canonical(s.state.job(job['id']))
    assert not rejects
    row=result['sample_prices'][0]
    assert row['price']==Decimal('12.34567890') and row['volume']==Decimal(200) and row['source_id']=='tushare'


@pytest.mark.skipif(os.name!='posix' or not Path('/proc').exists(),reason='Linux process ownership check')
def test_serving_sigterm_stops_owned_worker(tmp_path):
    import socket,subprocess,sys,time,requests,signal
    s=Settings(config_dir=tmp_path/'config',state_dir=tmp_path/'state',env_file=tmp_path/'credentials.env')
    s.write_private_env({'BETALENS_DB_MANAGER_CONFIG':str(s.config_dir),'BETALENS_DB_MANAGER_STATE':str(s.state_dir)})
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    child=subprocess.Popen([sys.executable,'-m','betalens_db_manager','--env',str(s.env_file),'serve','--port',str(port)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    workers=[]
    try:
        for _ in range(60):
            try:
                health=requests.get(f'http://127.0.0.1:{port}/api/worker-status',timeout=.2).json()
                if health.get('available'):break
            except requests.RequestException:pass
            time.sleep(.1)
        else:raise AssertionError('server did not start')
        workers=[health['heartbeat']['pid']]
        assert workers
        child.terminate();assert child.wait(timeout=12)==0
        assert all(not Path(f'/proc/{pid}').exists() for pid in workers)
    finally:
        if child.poll() is None:child.kill();child.wait()
        for pid in workers:
            if Path(f'/proc/{pid}').exists():os.kill(pid,signal.SIGKILL)


def test_ui_paths_support_reload_without_catching_missing_api(tmp_path):
    s=manager(tmp_path)
    with TestClient(create_app(s)) as client:
        for path in ('/','/tables','/tasks','/mappings?table=dividend_event','/sources','/settings','/schedules'):
            response=client.get(path)
            assert response.status_code==200,path
            assert 'text/html' in response.headers['content-type']
            assert response.headers['cache-control']=='no-cache'
            assert '<div id="root">' in response.text
            assert client.head(path).status_code==200
        for path in ('/api/not-a-route','/assets/missing.js','/mappings/not-a-page'):
            response=client.get(path)
            assert response.status_code==404
            assert '<div id="root">' not in response.text
