import importlib.util
from pathlib import Path

from betalens_db_manager.canonical import builtin_catalog
from betalens_db_manager.catalog import load_catalog


def recipe_module():
    path = Path(__file__).parents[1]/'scripts/production_recipe.py'
    spec = importlib.util.spec_from_file_location('production_recipe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recipes_cover_all_registered_tables_and_endpoints_with_explicit_targets():
    cases = recipe_module().recipes()
    assert {table for case in cases for table in case['expected_tables']} == set(builtin_catalog())
    assert {endpoint for case in cases for endpoint in case['expected_endpoints']} == set(load_catalog())
    assert len({case['label'] for case in cases}) == len(cases)
    for case in cases:
        request = case['request']
        assert request['targets'] == ['postgres', 'clickhouse']
        assert len(request['tables']) == 1
        assert request['endpoints'] == case['expected_endpoints']
        for endpoint, parameters in request['parameters'].items():
            assert set(parameters) <= set(load_catalog()[endpoint].input_fields)
        assert not {'token', 'password', 'confirmation'} & set(request)


def test_recipe_separates_assets_financial_versions_and_event_dates():
    cases = {case['label']: case['request'] for case in recipe_module().recipes()}
    assert cases['raw_fund']['codes'] == ['510300.SH']
    assert cases['raw_sw_index']['endpoints'] == ['sw_daily']
    for endpoint in ('income', 'balancesheet', 'cashflow', 'fina_indicator'):
        for name in (endpoint, endpoint+'_vip'):
            request = cases['financial_'+name]
            assert request['endpoints'] == [name]
            assert request['parameters'][name]['period'] == '20241231'
    assert cases['corporate_action_dividend']['start_date'] < cases['corporate_action_dividend']['end_date']


def test_config_inspection_is_read_only_and_flags_unbound_query(tmp_path):
    (tmp_path/'tables').mkdir()
    (tmp_path/'mappings').mkdir()
    table = tmp_path/'tables/etf_profile.yaml'
    mapping = tmp_path/'mappings/etf_profile.yaml'
    table.write_text('name: etf_profile\nfields: []\n')
    mapping.write_text('table: etf_profile\nbindings: []\n')
    before = {str(path): path.read_bytes() for path in tmp_path.rglob('*.yaml')}
    module = recipe_module()
    result = module.inspect_config(tmp_path, module.recipes())
    assert result[0]['unbound_requested_endpoints'] == ['etf_basic']
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob('*.yaml')}
