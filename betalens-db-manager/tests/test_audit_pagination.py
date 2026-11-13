from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from test_manager import manager
from betalens_db_manager.api import create_app
from betalens_db_manager.canonical import canonical, registry_context
from betalens_db_manager.pipeline import Pipeline
from betalens_db_manager.pipeline_cache import save_journal


def journal_task(service, root):
    job = service.state.create_job({'pipeline_version': 2, 'contracts': {}, 'request': {'targets': ['postgres', 'clickhouse']}}, service.settings.fingerprint)
    pipeline = Pipeline(service)
    with registry_context(service.registry):
        spec = canonical('market_daily_raw')
        from betalens_db_manager.comparison import compare
        rows = [{'asset_type': 'stock', 'security_code': f'{i:06d}.SZ', 'trade_date': date(2026, 10, 20), 'close': Decimal(i)} for i in range(5)]
        manifest = save_journal(root / 'journal', spec, compare(spec, rows, [], 'fill_missing'))
    for target in ('postgres', 'clickhouse'):
        pipeline.store.prepare(job['id'], 'batch', target, manifest)
        pipeline.store.acknowledge(job['id'], 'batch', target, {'verified': True})
    # Rollback receipts summarize the original journal and must not double the total.
    pipeline.store.prepare(job['id'], 'rollback:batch', 'postgres', manifest)
    return job['id']


def test_journal_pages_support_direct_jumps_and_cursor_compatibility(tmp_path, monkeypatch):
    service = manager(tmp_path)
    job = journal_task(service, tmp_path)
    pipeline = Pipeline(service)
    import betalens_db_manager.pipeline as module
    original = module.load_journal
    loaded = []

    def record_load(*args):
        loaded.append(args[0]['path'])
        return original(*args)

    monkeypatch.setattr(module, 'load_journal', record_load)
    first = service.audit(job, offset=0, limit=3)
    last = service.audit(job, offset=9, limit=3)
    assert first['total'] == last['total'] == 10
    assert first['offset'] == 0 and last['offset'] == 9
    assert len(first['changes']) == 3 and len(last['changes']) == 1
    assert last['next_after'] is None
    assert loaded  # Row images are expanded only for journals on the requested page.
    loaded.clear()
    beyond = service.audit(job, offset=12, limit=3)
    assert beyond['changes'] == [] and beyond['total'] == 10
    assert beyond['next_after'] is None and loaded == []
    pages = [pipeline.audit(job, offset=n, limit=3) for n in (0, 3, 6, 9)]
    ids = [row['id'] for page in pages for row in page['changes']]
    assert ids == sorted(set(ids)) and len(ids) == 10
    second = pipeline.audit(job, after=first['next_after'], limit=3)
    assert second['changes'] == pages[1]['changes']
    assert pipeline.audit(job, after=ids[-1], limit=3)['changes'] == []


def test_audit_api_returns_page_metadata_and_validates_offset(tmp_path):
    service = manager(tmp_path)
    job = journal_task(service, tmp_path)
    with TestClient(create_app(service)) as client:
        for prefix in ('tasks', 'jobs'):
            result = client.get(f'/api/{prefix}/{job}/audit?offset=8&limit=2')
            assert result.status_code == 200, result.text
            page = result.json()
            assert page['total'] == 10 and page['offset'] == 8 and page['limit'] == 2
            assert len(page['changes']) == 2 and page['next_after'] is None
        assert client.get(f'/api/tasks/{job}/audit?offset=-1').status_code == 422
        assert client.get(f'/api/tasks/{job}/audit?offset=1.5').status_code == 422
    empty = service.state.create_job({'pipeline_version': 2}, service.settings.fingerprint)
    result = service.audit(empty['id'], offset=0)
    assert result['total'] == 0 and result['changes'] == [] and result['next_after'] is None


def test_legacy_audit_pages_count_and_sort_sparse_ids(tmp_path, monkeypatch):
    service = manager(tmp_path)
    job = service.state.create_job({'request': {'targets': []}}, service.settings.fingerprint)['id']
    for ident in (91, 2, 17, 43, 8):
        service.state.ledger_put('import_row_audit', str(ident), {'task_id': job, 'id': ident})
    service.state.ledger_put('import_row_audit', 'unrelated', {'task_id': 'other', 'id': 99})
    monkeypatch.setattr(service.pg, 'audit', lambda *a, **kw: pytest.fail('Local audit pages must not read PostgreSQL'))
    first = service.audit(job, offset=0, limit=2)
    second = service.audit(job, offset=2, limit=2)
    last = service.audit(job, offset=4, limit=2)
    assert first['total'] == second['total'] == last['total'] == 5
    assert [r['id'] for r in first['changes']] == [2, 8]
    assert [r['id'] for r in second['changes']] == [17, 43]
    assert [r['id'] for r in last['changes']] == [91] and last['next_after'] is None
    assert service.audit(job, after=8, limit=2)['changes'] == second['changes']
    assert service.audit(job, offset=6, limit=2)['changes'] == []


def test_legacy_audit_can_page_postgres_fallback(tmp_path, monkeypatch):
    service = manager(tmp_path)
    job = service.state.create_job({'request': {'targets': []}}, service.settings.fingerprint)['id']
    rows = [{'id': n} for n in range(1, 6)]
    monkeypatch.setattr(service.pg, 'audit', lambda job, after, limit, offset=0: [r for r in rows if r['id'] > after][offset:offset + limit])
    monkeypatch.setattr(service.pg, 'audit_count', lambda job, after=0: sum(r['id'] > after for r in rows))
    monkeypatch.setattr(service.pg, 'receipts', lambda job: [])
    last = service.audit(job, offset=4, limit=2)
    assert last['total'] == 5 and last['changes'] == [{'id': 5}] and last['next_after'] is None
