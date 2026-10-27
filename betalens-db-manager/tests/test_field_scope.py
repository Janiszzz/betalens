from decimal import Decimal
from types import SimpleNamespace

import pytest

from betalens_db_manager.canonical import CanonicalTable, builtin_catalog
from betalens_db_manager.comparison import row_key
from betalens_db_manager.field_lineage import PROOF_VERSION, applicability, record_readback
from betalens_db_manager.util import digest


class MemoryState:
    def __init__(self, traces):
        self.traces = traces
        self.proofs = {}

    def ledger_get(self, table, key):
        assert table == 'task_column_lineage'
        return {'rows': self.traces}

    def ledger_put(self, table, key, value):
        assert table == 'field_verification'
        self.proofs[value['column']] = value


def producer(column, scope, endpoint='income'):
    return {'column': column, 'kind': 'direct', 'source_id': 'tushare',
            'endpoint': endpoint, 'source_field': 'rd_exp', 'scope': scope}


def proof(spec, row, column, producers, asset_types, *, endpoint='income'):
    key = row_key(spec, row)
    trace = {'selected': True, 'value_hash': digest(row[column]),
             'origins': ['tushare:'+endpoint+':rd_exp']}
    state = MemoryState({key: {column: trace}})
    item = {'id': 'task', 'plan': {'request': {'asset_types': asset_types},
        'mapping_snapshots': {spec.name: {'bindings': [{'source_id': 'tushare', 'endpoint': endpoint}],
                                        'producers': producers}}}}
    comparison = {'changes': [{'row_key': key, 'before': None, 'after': row}]}
    record_readback(SimpleNamespace(state=state), item, 'postgres', spec, comparison, {key: row})
    return state.proofs


@pytest.mark.parametrize('statement,column,other', [
    ('income', 'research_and_development_expense', 'research_and_development_investment'),
    ('fina_indicator', 'research_and_development_investment', 'research_and_development_expense'),
])
def test_non_null_financial_field_proven_with_task_asset_and_other_statement_not_applicable(statement, column, other):
    spec = builtin_catalog()['fundamental_periodic']
    row = {'security_code': '000333.SZ', 'period_end': '2024-12-31', 'statement_type': statement,
           'report_type': '1', 'version_id': 'version', column: Decimal('16232771000')}
    # Exactly the selected endpoint's producer survives the frozen snapshot.
    producers = [producer(column, {'asset_type': 'stock', 'statement_type': statement}, statement)]
    records = proof(spec, row, column, producers, ['stock'], endpoint=statement)
    assert records[column]['status'] == 'verified_stored'
    assert records[column]['verified_count'] == 1
    assert records[column]['proof_version'] == PROOF_VERSION
    assert records[other]['status'] == 'not_applicable'
    assert records[other].get('verified_count', 0) == 0


def test_profile_without_physical_asset_type_uses_pinned_task_asset_set():
    spec = CanonicalTable('stock_profile', '股票', '股票档案', '证券一行', (
        {'name': 'security_code', 'type': 'text'}, {'name': 'name', 'type': 'text'}), ('security_code',))
    row = {'security_code': '000333.SZ', 'name': '美的集团'}
    records = proof(spec, row, 'name', [producer('name', {'asset_type': 'stock'}, 'stock_basic')],
                    ['stock'], endpoint='stock_basic')
    assert records['name']['status'] == 'verified_stored'


def test_actual_row_asset_type_and_statement_type_override_mixed_task():
    declared = [producer('value', {'asset_type': 'stock'})]
    assert applicability(declared, {'asset_type': 'fund'}, ['stock', 'fund']) == 'not_applicable'
    assert applicability(declared, {'asset_type': 'stock'}, ['fund']) == 'applicable'
    assert applicability(declared, {}, ['stock', 'fund']) == 'applicable'
    statement = [producer('value', {'asset_type': 'stock', 'statement_type': 'income'})]
    assert applicability(statement, {'statement_type': 'fina_indicator'}, ['stock']) == 'not_applicable'
    assert applicability(statement, {}, ['stock']) == 'unknown'


def test_unknown_scope_has_no_write_proof_and_is_not_marked_not_applicable():
    spec = CanonicalTable('sample', '示例', '示例', '一行', (
        {'name': 'code', 'type': 'text'}, {'name': 'value', 'type': 'numeric'}), ('code',))
    row = {'code': 'one', 'value': Decimal('12')}
    records = proof(spec, row, 'value', [producer('value', {'asset_type': 'stock'})], None)
    assert records['value']['status'] == 'unknown_scope'
    assert records['value']['unknown_scope_count'] == 1
    assert records['value'].get('not_applicable_count', 0) == 0
    assert records['value'].get('verified_count', 0) == 0
    # Existing custom mappings with no declarations retain trace verification.
    undeclared = proof(spec, row, 'value', [], None)
    assert undeclared['value']['status'] == 'verified_stored'


def test_unavailable_columns_remain_unavailable_instead_of_statement_na():
    spec = builtin_catalog()['fundamental_periodic']
    row = {'security_code': '000333.SZ', 'period_end': '2024-12-31', 'statement_type': 'income',
           'report_type': '1', 'version_id': 'version', 'rd_exp': None}
    declared = [{'column': 'rd_exp', 'kind': 'unavailable', 'reason': 'deprecated'}]
    records = proof(spec, row, 'rd_exp', declared, ['stock'])
    assert records['rd_exp']['status'] == 'unavailable'
    assert records['rd_exp']['unavailable_count'] == 1
    assert records['rd_exp'].get('not_applicable_count', 0) == 0
