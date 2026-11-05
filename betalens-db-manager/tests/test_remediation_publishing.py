"""Failure injection at commit/rollback boundaries; no production connections."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import pytest

from betalens_db_manager.canonical import canonical
from betalens_db_manager.clickhouse import ClickHouse
from betalens_db_manager.commit_protocol import manifest_rows, remaining_changes, NeedsReconciliation
from betalens_db_manager.comparison import compare, row_hash, row_key, typed
from betalens_db_manager.state import State
from betalens_db_manager.config import Settings

SPEC=canonical('security_dim')


def test_clickhouse_gate_identity_cannot_change_with_tmpdir_or_loopback_alias(tmp_path,monkeypatch):
    from betalens_db_manager.ch_writer import LocalDedicatedWriter
    monkeypatch.setenv('TMPDIR',str(tmp_path))
    a=LocalDedicatedWriter(Settings(ch_url='http://localhost:8123',ch_database='sample'))
    b=LocalDedicatedWriter(Settings(ch_url='http://127.0.0.1:8123/',ch_database='sample'))
    assert a.lock==b.lock and a.lock.parent.parent.name=='tmp'


def test_finished_unknown_query_still_requires_original_operation_recovery(tmp_path,monkeypatch):
    from betalens_db_manager.ch_writer import LocalDedicatedWriter
    from betalens_db_manager.util import atomic_json
    import uuid
    gate=LocalDedicatedWriter(Settings(ch_url='http://localhost:8123',ch_database='test_'+uuid.uuid4().hex))
    monkeypatch.setattr(gate,'verify',lambda adapter:None)
    adapter=SimpleNamespace(execute=lambda query:'0',_json_rows=lambda value:[{'type':'QueryFinish'}],_literal=lambda value:repr(value))
    atomic_json(gate.journal,{'inflight':True,'query_id':'past','operation':'commit:original'})
    try:
        with pytest.raises(NeedsReconciliation,match='先恢复原操作'):
            with gate.exclusive(adapter,'commit:unrelated'):pytest.fail('must remain blocked')
        with gate.exclusive(adapter,'commit:original'):pass
    finally:
        gate.journal.unlink(missing_ok=True);gate.lock.unlink(missing_ok=True)


@pytest.mark.parametrize('query',['DELETE FROM example WHERE code=1','DETACH TABLE example','ATTACH TABLE example','WITH 1 AS x INSERT INTO example SELECT x'])
def test_unlisted_clickhouse_mutations_cannot_bypass_writer_gate(tmp_path,monkeypatch,query):
    from contextlib import contextmanager
    ch=ClickHouse(Settings(config_dir=tmp_path/'config',state_dir=tmp_path/'state',ch_writer_mode='local_dedicated'))
    @contextmanager
    def blocked(operation):
        raise NeedsReconciliation('gate checked')
        yield
    monkeypatch.setattr(ch,'write_window',blocked)
    monkeypatch.setattr('requests.post',lambda *a,**k:pytest.fail('SQL reached server before writer verification'))
    with pytest.raises(NeedsReconciliation,match='gate checked'):ch.execute(query)
