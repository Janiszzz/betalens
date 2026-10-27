import gzip
import hashlib
import importlib.util
from pathlib import Path

from betalens_db_manager.canonical import CanonicalTable
from betalens_db_manager.util import dumps


def verifier_module():
    path = Path(__file__).parents[1]/'scripts/production_verify_readback.py'
    specification = importlib.util.spec_from_file_location('production_verify_readback', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_one_passing_asset_cannot_hide_missing_or_failed_table_case():
    module = verifier_module()
    planned = [{'label': label, 'request': {'tables': ['market_daily_raw']}}
               for label in ('raw_stock', 'raw_index')]
    partial = module.table_case_coverage(planned, [{'label': 'raw_stock', 'status': 'verified'}])[0]
    assert partial['status'] == 'not_verified'
    assert partial['missing_cases'] == ['raw_index']
    failed = module.table_case_coverage(planned, [{'label': 'raw_stock', 'status': 'verified'},
                                                 {'label': 'raw_index', 'status': 'not_verified'}])[0]
    assert failed['status'] == 'not_verified'
    assert failed['not_verified_cases'] == ['raw_index']


def test_missing_target_column_proof_is_not_a_complete_audit():
    module = verifier_module()
    spec = CanonicalTable('sample', '示例', '示例', '一行', (
        {'name': 'code', 'type': 'text'}, {'name': 'price', 'type': 'numeric'}), ('code',))
    proofs = [{'target': target, 'table': 'sample', 'column': column, 'status': 'table_verified',
               'applicable_count': 1, 'preserved_count': 1}
              for target in ('postgres', 'clickhouse') for column in ('code', 'price')]
    assert module.proof_summary(proofs, spec, ['postgres', 'clickhouse'])['complete']
    incomplete = module.proof_summary(proofs[:-1], spec, ['postgres', 'clickhouse'])
    assert not incomplete['complete']
    assert incomplete['missing'] == [['clickhouse', 'sample', 'price']]


def test_fill_missing_retained_value_does_not_prove_new_source_written(tmp_path):
    module = verifier_module()
    raw = dumps({'data': {'fields': ['open'], 'items': [['12.4']]}}).encode()
    archive = tmp_path/'response.json.gz';archive.write_bytes(gzip.compress(raw))
    item = {'slices': [{'dataset': 'daily', 'status': 'downloaded', 'archive': str(archive),
                       'sha': hashlib.sha256(raw).hexdigest()}],
            'plan': {'request': {'targets': ['postgres', 'clickhouse']},
                     'source_contracts': {'tushare:daily': {'fields': [{'name': 'open'}]}},
                     'mapping_snapshots': {'sample': {'bindings': [{'source_id': 'tushare', 'endpoint': 'daily',
                         'fields': [{'source_field': 'open', 'column': 'price'}]}]}}}}
    proofs = [{'target': target, 'table': 'sample', 'column': 'price', 'status': 'table_verified',
               'preserved_count': 1, 'verified_count': 0, 'verified_origins': {}}
              for target in ('postgres', 'clickhouse')]
    field = module.source_summary(item, proofs)[0]['fields'][0]
    assert field['source_nonnull_count'] == 1
    assert {target['status'] for target in field['targets']} == {'existing_values_retained'}
    assert all(not target['verified_origin_values'] for target in field['targets'])
