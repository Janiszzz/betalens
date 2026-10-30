"""Isolated real PG/CH + HTTP fixture for browser acceptance. Never uses production tables."""
from dataclasses import replace
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs
import json,threading,tempfile,uuid
import uvicorn
from betalens_db_manager.config import Settings
from betalens_db_manager.manager import DatabaseManagerService
from betalens_db_manager.api import create_app
from betalens_db_manager.canonical import registry_context
from controlled_ch import configure_writer

def main():
    root=Path(tempfile.mkdtemp(prefix='betalens-manager-browser-'));suffix=uuid.uuid4().hex[:10]
    base=Settings.from_env(__import__('os').environ.get('BETALENS_TEST_ENV','/home/janiszzz/betalens-infra/tushare.private.env'))
    settings=replace(base,pg_database='browser_test_'+suffix,ch_database='browser_test_'+suffix,state_dir=root/'state',config_dir=root/'config',archive_dir=None,env_file=root/'credentials.env')
    s=DatabaseManagerService(settings)
    cleanup_writer=configure_writer(s);settings=s.settings
    # Only this newly created disposable fixture's environment file may be written.
    assert settings.env_file.resolve().parent==root.resolve()
    settings.write_private_env({'PGHOST':settings.pg_host,'PGPORT':str(settings.pg_port),'PGDATABASE':settings.pg_database,'PGSCHEMA':settings.pg_schema,'PGUSER':settings.pg_user,'PGPASSWORD':settings.pg_password,'CLICKHOUSE_URL':settings.ch_url,'CLICKHOUSE_DB':settings.ch_database,'CLICKHOUSE_USER':settings.ch_user,'CLICKHOUSE_PASSWORD':settings.ch_password,'CLICKHOUSE_WRITER_MODE':settings.ch_writer_mode,'BETALENS_INFRA_STATE':str(settings.state_dir),'BETALENS_DB_MANAGER_CONFIG':str(settings.config_dir)})
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=urlsplit(self.path);p=parse_qs(path.query)
            if path.path=='/seed':
                try:
                    for day in ('2026-10-12','2026-10-14'):
                        task=s.plan({'tables':['browser_prices'],'start_date':day,'end_date':day});s.download(task['id']);preview=s.preview(task['id']);result=s.commit(task['id'],preview['token'])
                        assert result['status']=='completed',result.get('error')
                    value={'seeded':True}
                except Exception as exc:value={'error':str(exc)}
            elif path.path=='/seed-review-coverage':
                assert settings.pg_database.startswith('browser_test_') and settings.ch_database.startswith('browser_test_')
                from test_manual_review import seed_missing_archive,initialize
                task,_=seed_missing_archive(s);initialize(s);value={'id':task['id']}
            elif path.path=='/assert-empty':
                with registry_context(s.registry):value={'pg':len(s.pg.query('browser_prices',start='2026-10-13',end='2026-10-13')['rows']),'ch':len(s.ch.query('browser_prices',start='2026-10-13',end='2026-10-13')['rows'])}
            elif path.path=='/audit-sync-fault':
                from psycopg2 import sql
                assert settings.pg_database.startswith('browser_test_')
                with s.pg.connect() as conn,conn.cursor() as cur:
                    cur.execute(sql.SQL('DROP TRIGGER IF EXISTS test_audit_fault ON {}.import_task').format(sql.Identifier(settings.pg_schema)))
                    if p.get('enabled')==['true']:
                        cur.execute(sql.SQL("CREATE OR REPLACE FUNCTION {}.test_audit_fault() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.status='completed' THEN RAISE EXCEPTION 'test final audit mirror fault'; END IF; RETURN NEW; END; $$").format(sql.Identifier(settings.pg_schema)))
                        cur.execute(sql.SQL('CREATE TRIGGER test_audit_fault BEFORE INSERT OR UPDATE ON {}.import_task FOR EACH ROW EXECUTE FUNCTION {}.test_audit_fault()').format(sql.Identifier(settings.pg_schema),sql.Identifier(settings.pg_schema)))
                value={'enabled':p.get('enabled')==['true']}
            else:
                day=p.get('start_date',['2026-10-13'])[0]
                import time;time.sleep(55 if path.path=='/slow' else .4)
                if path.path=='/review-prices':value={'fields':['price_milli','day','ticker'],'items':[['12345.678901',day.replace('-','/'),'a'],['9000',day.replace('-','/'),None]]}
                elif path.path=='/volumes':value={'fields':['instrument','quantity_lots','observed','note'],'items':[['a','2',day.replace('-',''),'explicitly ignored']]}
                elif path.path=='/alternate':value=[{'symbol':'a','observed':day.replace('-','/'),'close_yuan':'12.34567890'}]
                else:value={'fields':['price_milli','day','ticker'],'items':[['12345.678901',day.replace('-','/'),'a']]}
            raw=json.dumps(value).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
        def log_message(self,*args):pass
    mock=ThreadingHTTPServer(('127.0.0.1',18082),Handler);threading.Thread(target=mock.serve_forever,daemon=True).start()
    import subprocess,sys
    child=subprocess.Popen([sys.executable,'-m','betalens_db_manager','--env',str(settings.env_file),'worker'])
    try:
        uvicorn.run(create_app(s),host='127.0.0.1',port=18081,log_level='warning')
    finally:
        child.terminate();child.wait(timeout=15)
        mock.shutdown();mock.server_close()
        import psycopg2
        from psycopg2 import sql
        cleanup_writer()
        s.ch.execute(f'DROP DATABASE IF EXISTS `{settings.ch_database}`')
        c=psycopg2.connect(**settings.pg('postgres'));c.autocommit=True
        with c.cursor() as cur:cur.execute(sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(settings.pg_database)))
        c.close()


if __name__=='__main__':main()
