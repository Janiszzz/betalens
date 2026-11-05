"""The retired execution engine cannot be reached through any public surface."""
import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from betalens_db_manager.api import create_app
from betalens_db_manager.cli import main
from betalens_db_manager.execution_policy import LegacyPipelineError,LegacyTaskError
from betalens_db_manager.manager import DatabaseManagerService
from betalens_db_manager.manager_kernel import ManagerKernel
from betalens_db_manager.pipeline import Pipeline
from betalens_db_manager.pipeline_validation import ValidationRunner
from test_manager import manager
from test_pipeline_v2 import prepare


def test_only_v2_execution_code_and_queue_operations_remain(tmp_path):
    from betalens_db_manager import publishing
    from betalens_db_manager.storage import Postgres
    from betalens_db_manager.clickhouse import ClickHouse
    from betalens_db_manager.state import State
    assert not Path(__file__).parents[1].joinpath('src/betalens_db_manager/service.py').exists()
    for kind in (ManagerKernel,DatabaseManagerService):
        assert not any(hasattr(kind,name) for name in ('download','preview','commit','resume','rollback','_download_one','_requests','_make_plan'))
    assert not any(hasattr(publishing,n) for n in ('publish','rollback','retry_audit_sync'))
    for kind in (Postgres,ClickHouse):assert not any(hasattr(kind,n) for n in ('commit_rows','apply_rollback','cleanup_rollback'))
    assert not any(hasattr(State,n) for n in ('stage_commit','finish_commit'))
    s=manager(tmp_path);job=s.plan(prepare(s,'http://127.0.0.1:1'))
    operations=('download','update','file_download','preview','commit','resume','rollback','sync_audit','reference_download','reference_preview','reference_commit','reference_rollback')
    for operation in operations:
        with pytest.raises(LegacyPipelineError):s.state.enqueue(job['id'],operation)
    assert not s.state.job(job['id'])['tasks']


def test_removed_cli_commands_and_http_routes_do_not_dispatch(tmp_path):
    for name in ('download','preview','resume','load'):
        with pytest.raises(SystemExit) as exc:main([name,'retired'])
        assert exc.value.code==2
    s=manager(tmp_path);job=s.plan(prepare(s,'http://127.0.0.1:1'))
    app=create_app(s)
    paths={r.path for r in app.routes}
    for operation in ('download','preview','commit','resume'):
        for prefix in ('/api/jobs/','/api/tasks/','/api/reference/tasks/'):
            assert prefix+'{job}/'+operation not in paths
    assert '/api/tables/archive-backfill' not in paths
    with TestClient(app) as client:
        for operation in ('download','preview','commit','resume'):
            assert client.post('/api/tasks/'+job['id']+'/'+operation,json={}).status_code in {404,405}
    assert not s.state.job(job['id'])['tasks']


def test_archived_task_cannot_enter_direct_v2_execution(tmp_path):
    s=manager(tmp_path);job=s.state.create_job({'request':{'targets':['postgres']},'revision':'2'},s.settings.fingerprint)
    for operation in (lambda:Pipeline(s).run(job['id']),lambda:Pipeline(s).commit(job['id']),
                      lambda:Pipeline(s).rollback(job['id']),lambda:ValidationRunner(s).run(job['id'],{})):
        with pytest.raises(LegacyTaskError):operation()
    assert s.state.job(job['id'])['status']=='planned'


def test_no_scripts_or_modules_import_the_retired_engine():
    root=Path(__file__).parents[1]
    for folder in ('src','scripts'):
        for path in (root/folder).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom):
                    assert node.module not in {'service','betalens_db_manager.service'},path
                    assert not any(v.name in {'TushareImportService','retry_audit_sync','publish','stage_commit','finish_commit'} for v in node.names),path
