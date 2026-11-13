from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from betalens_db_manager.api import create_app
from betalens_db_manager.config import Settings
from betalens_db_manager.download_cache import cleanup_download_cache
from betalens_db_manager.pipeline_store import PipelineStore
from betalens_db_manager.state import State
from betalens_db_manager.worker import clean_cache
from test_manager import manager


@pytest.fixture
def service(tmp_path):
    settings = Settings(state_dir=tmp_path / 'state', config_dir=tmp_path / 'config',
                        archive_dir=tmp_path / 'downloads')
    return SimpleNamespace(settings=settings, state=State(settings.state_dir))


def task(service, status='completed', parent=None, revision='manager-v1'):
    plan = {'pipeline_version': 2, 'revision': revision,
            'request': {'targets': ['postgres'], 'tables': [], 'run_in_background': True}}
    if parent:
        plan['pipeline_parent'] = parent
    job = service.state.create_job(plan, service.settings.fingerprint)['id']
    service.state.update_job(job, status=status)
    return job


def payload(service, job, name='sample'):
    root = (service.settings.archive_dir or service.settings.state_dir / 'archive') / job / 'daily'
    root.mkdir(parents=True, exist_ok=True)
    path = root / (name + '.json.gz')
    path.write_bytes(b'downloaded payload')
    path.with_suffix('.meta.json').write_text('{}')
    path.with_name(name + '.response.json.gz').write_bytes(b'wire payload')
    service.state.slice(job, name, 'daily', {}, status='downloaded', archive=str(path), sha='hash', rows=1)
    return path


def test_terminal_families_clean_only_downloads_and_keep_audit(service):
    root = task(service)
    child = task(service, 'downloaded', root)
    raw = payload(service, root)
    child_raw = payload(service, child)
    typed = service.settings.state_dir / 'pipeline' / child / 'typed.parquet'
    journal = service.settings.state_dir / 'pipeline' / root / 'intents' / 'before.parquet'
    audit = service.settings.state_dir / 'reference-cache' / root / 'audit.json'
    for path in (typed, journal, audit):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'durable evidence')
    pending = task(service, 'interrupted')
    completed_child = task(service, 'completed', pending)
    protected = payload(service, completed_child)
    report = cleanup_download_cache(service)
    assert report['files'] == 6 and report['tasks'] == 1
    assert not raw.exists() and not child_raw.exists() and protected.exists()
    assert all(p.read_bytes() == b'durable evidence' for p in (typed, journal, audit))
    assert service.state.job(child)['download_cache_cleanup']['status'] == 'cleaned'
    assert service.state.job(root)['status'] == 'completed'
    assert cleanup_download_cache(service)['files'] == 0


@pytest.mark.parametrize('status', ['planned', 'running', 'ready_to_commit', 'partial', 'paused',
                                   'interrupted', 'cancelled', 'failed', 'partially_rolled_back'])
def test_unfinished_and_recoverable_tasks_are_retained(service, status):
    job = task(service, status)
    raw = payload(service, job)
    assert cleanup_download_cache(service)['files'] == 0
    assert raw.exists() and service.state.job(job)['download_cache_cleanup'] is None


def test_shared_payload_survives_until_last_consumer_finishes(service):
    owner = task(service)
    raw = payload(service, owner)
    consumer = task(service, 'ready_to_commit')
    service.state.slice(consumer, 'shared', 'daily', {}, archive=str(raw), status='downloaded')
    first = cleanup_download_cache(service)
    assert first['retained_files'] == 3 and first['files'] == 0
    assert raw.exists() and raw.with_suffix('.meta.json').exists()
    assert raw.with_name('sample.response.json.gz').exists()
    service.state.update_job(consumer, status='completed')
    assert cleanup_download_cache(service)['files'] == 3
    assert not raw.exists()


def test_active_operations_and_rejected_cohorts_keep_their_evidence(service):
    active = task(service)
    raw = payload(service, active)
    operation = service.state.enqueue(active, 'pipeline_validate')
    assert cleanup_download_cache(service)['files'] == 0 and raw.exists()
    service.state.finish(operation)
    store = PipelineStore(service.state)
    store.put(active, 'rejects', 'normalize', {})
    store.finish(active, 'rejects', {'rejected': 1}, error='needs review')
    assert cleanup_download_cache(service)['files'] == 0 and raw.exists()
    store.finish(active, 'rejects', {'rejected': 0})
    assert cleanup_download_cache(service)['files'] == 3


def test_reference_and_legacy_completed_downloads_are_also_cleaned(service):
    reference = task(service, revision='reference-v1')
    legacy = service.state.create_job({'revision': '2', 'request': {}}, service.settings.fingerprint)['id']
    service.state.update_job(legacy, status='rolled_back')
    paths = [payload(service, j) for j in (reference, legacy)]
    assert cleanup_download_cache(service)['files'] == 6
    assert not any(p.exists() for p in paths)


def test_dry_run_reports_bytes_and_preserves_files_and_ledger(service):
    job = task(service)
    raw = payload(service, job)
    report = cleanup_download_cache(service, dry_run=True)
    assert report['files'] == 3 and report['bytes'] > 0
    assert raw.exists() and service.state.job(job)['download_cache_cleanup'] is None


def test_symlinks_never_delete_external_files(service, tmp_path):
    job = task(service)
    raw = payload(service, job)
    external = tmp_path / 'keep'
    external.mkdir()
    (external / 'secret').write_text('keep')
    (raw.parent / 'link').symlink_to(external, target_is_directory=True)
    (raw.parent / 'file-link').symlink_to(external / 'secret')
    report = cleanup_download_cache(service)
    assert (external / 'secret').read_text() == 'keep'
    assert (raw.parent / 'file-link').is_symlink() and report['retained_files'] == 2


def test_unlink_failure_is_retryable_without_changing_job_status(service, monkeypatch):
    job = task(service)
    raw = payload(service, job)
    unlink = Path.unlink

    def fail(path, *args, **kwargs):
        if path == raw:
            raise PermissionError('temporarily locked')
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', fail)
        report = clean_cache(service)
    assert report['errors'] and raw.exists()
    assert service.state.job(job)['status'] == 'completed'
    assert service.state.job(job)['download_cache_cleanup']['status'] == 'failed'
    clean_cache(service)
    assert not raw.exists()
    assert service.state.job(job)['download_cache_cleanup']['status'] == 'cleaned'


def test_cache_api_marks_purged_download_and_preserves_slice_provenance(tmp_path):
    s = manager(tmp_path)
    job = s.plan({'tables': ['market_daily_raw'], 'start_date': '2026-10-20', 'end_date': '2026-10-20'})['id']
    raw = payload(s, job)
    s.state.update_job(job, status='completed')
    cleanup_download_cache(s)
    with TestClient(create_app(s)) as client:
        response = client.get('/api/tasks/' + job + '/cache')
        assert response.status_code == 200, response.text
        assert response.json()['cache_purged'] and response.json()['rows'] == []
        response = client.get('/api/tasks/' + job + '/archive/sample')
        assert response.status_code == 410, response.text
    part = s.state.job(job)['slices'][0]
    assert part['archive'] == str(raw) and part['sha'] == 'hash' and part['rows'] == 1


def test_dependencies_report_cleaned_source_as_unknown(service):
    from betalens_db_manager.pipeline_validation import ValidationRunner
    job = task(service)
    raw = payload(service, job)
    service.state.slice(job, 'calendar', 'trade_cal', {}, archive=str(raw), status='downloaded')
    cleanup_download_cache(service)
    dependencies = ValidationRunner(service)._dependencies(service.state.job(job))
    assert dependencies['cache_purged'] and dependencies['calendar'] == {}


def test_execution_cleans_only_after_operation_has_finished(tmp_path, monkeypatch):
    from betalens_db_manager.pipeline import Pipeline
    from betalens_db_manager.worker import execute_task
    s = manager(tmp_path)
    job = task(s, 'planned')
    raw = payload(s, job)
    s.state.enqueue(job, 'pipeline_run')
    operation = s.state.claim()

    def run(pipeline, identity):
        assert raw.exists()
        assert s.state.job(identity)['tasks'][0]['status'] == 'running'
        s.state.update_job(identity, status='completed')

    monkeypatch.setattr(Pipeline, 'run', run)
    monkeypatch.setattr('betalens_db_manager.manager.DatabaseManagerService', lambda _: s)
    execute_task(s.settings, operation)
    assert not raw.exists()
    assert s.state.job(job)['tasks'][0]['status'] == 'completed'
    assert s.state.ledger_rows('execution_attempt', job)[0]['status'] == 'completed'


def test_readback_still_checks_target_after_source_cleanup(tmp_path, monkeypatch):
    from datetime import date
    from decimal import Decimal
    from betalens_db_manager.canonical import registry_context
    from betalens_db_manager.comparison import row_key, row_hash, typed
    from betalens_db_manager.files import import_file
    from betalens_db_manager.pipeline_cache import write_rows, save_journal
    from betalens_db_manager.pipeline_validation import ValidationRunner
    from test_manager import table_definition
    s = manager(tmp_path)
    s.registry.save(table_definition('cache_quotes'))
    source = tmp_path / 'prices.csv'
    source.write_text('ticker,day,price\na,2026-10-20,2\n')
    item = import_file(s, 'cache_quotes', source,
                       {'ticker': 'code', 'day': 'business_date', 'price': 'price'}, targets=['postgres'])
    job = item['id']
    with registry_context(s.registry):
        spec = s.registry.spec('cache_quotes')
        row = typed(spec, {'code': 'a', 'business_date': date(2026, 10, 20), 'price': Decimal('2')})
        record = write_rows(s.settings.state_dir / 'pipeline' / job / 'data.parquet', spec, [row])
        manifest = save_journal(s.settings.state_dir / 'pipeline' / job / 'intents', spec,
                                {'changes': [{'row_key': row_key(spec, row), 'before': None, 'after': row,
                                              'before_hash': None, 'after_hash': row_hash(spec, row)}]}, [row])
    store = PipelineStore(s.state)
    store.put(job, 'file', 'normalize', {})
    store.finish(job, 'file', {'child_id': job, 'tables': {'cache_quotes': [record]}})
    store.prepare(job, job + ':cache_quotes:0', 'postgres', manifest)
    store.acknowledge(job, job + ':cache_quotes:0', 'postgres', {})
    s.state.update_job(job, status='completed')
    cleanup_download_cache(s)
    assert source.exists()  # Only the managed copy is disposable.
    monkeypatch.setattr(s, '_build_canonical', lambda *_: pytest.fail('cleaned source must not be replayed'))
    live = dict(row)
    monkeypatch.setattr('betalens_db_manager.pipeline_validation.existing',
                        lambda *args, **kwargs: {row_key(spec, live): live})
    report = ValidationRunner(s).run(job, {'rules': ['readback']})
    assert report['status'] == 'unknown'
    assert any(c.get('readback_verified') == 1 for c in report['checks'])
    assert any(c.get('stage') == 'source_replay' and c.get('unknown') for c in report['checks'])
    live['price'] = Decimal('3')
    report = ValidationRunner(s).run(job, {'rules': ['readback']})
    assert report['status'] == 'failed'
    assert any('stored_value_changed' in sample['problems'] for sample in report['samples'])
