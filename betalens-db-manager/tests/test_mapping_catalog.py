"""Official field contracts must not imply undocumented financial semantics."""
from decimal import Decimal

import pytest

from betalens_db_manager.canonical import builtin_catalog, financial_column
from betalens_db_manager.catalog import load_catalog
from betalens_db_manager.mapping import FIELD_MAP, all_source_mappings
from betalens_db_manager.retrieval import partition, tushare_policy


ETF_FIELDS = {
    'ts_code', 'csname', 'extname', 'cname', 'index_code', 'index_name',
    'setup_date', 'list_date', 'list_status', 'exchange', 'mgr_name',
    'custod_name', 'mgt_fee', 'etf_type',
}
BAK_FIELDS = {
    'ts_code', 'trade_date', 'name', 'pct_change', 'close', 'change', 'open',
    'high', 'low', 'pre_close', 'vol_ratio', 'turn_over', 'swing', 'vol',
    'amount', 'selling', 'buying', 'total_share', 'float_share', 'pe',
    'industry', 'area', 'float_mv', 'total_mv', 'avg_price', 'strength',
    'activity', 'avg_turnover', 'attack', 'interval_3', 'interval_6',
}


@pytest.mark.parametrize('endpoint,table,names,keys', [
    ('etf_basic', 'etf_profile', ETF_FIELDS, ('security_code',)),
    ('bak_daily', 'stock_quote_supplement_daily', BAK_FIELDS, ('security_code', 'trade_date')),
])
def test_complete_contract_to_local_table(endpoint, table, names, keys):
    source = load_catalog()[endpoint]
    target = builtin_catalog()[table]
    assert {f['name'] for f in source.fields} == names
    columns = {f['name']: f for f in target.fields}
    assert target.keys == keys
    mappings = [r for r in all_source_mappings() if r['source_endpoint'] == endpoint]
    assert len(mappings) == len(names)
    assert {r['source_field'] for r in mappings} == names
    for mapping in mappings:
        column = 'security_code' if mapping['source_field'] == 'ts_code' else mapping['source_field']
        assert mapping['canonical_table'] == table
        assert mapping['canonical_column'] == column
        assert columns[column]['type'] == mapping['source_type']
        assert mapping['transform_expression'] == 'identity'


def test_documented_units_are_kept_in_supplement_instead_of_guessed():
    fields = {f['name']: f for f in load_catalog()['bak_daily'].fields}
    for name in ('vol', 'amount', 'total_mv', 'float_mv', 'avg_price'):
        assert fields[name]['unit'] == fields[name]['source_unit'] == ''
        assert '未明确单位' in fields[name]['description']
        assert Decimal(fields[name]['multiplier']) == 1
        assert FIELD_MAP['bak_daily'][name] == ('stock_quote_supplement_daily', name)
    assert fields['selling']['source_unit'] == fields['buying']['source_unit'] == '手'
    assert fields['total_share']['source_unit'] == fields['float_share']['source_unit'] == '亿股'
    assert fields['strength']['source_unit'] == '%'
    # No source field, including avg_price, silently enriches unified prices.
    assert {table for table, _ in FIELD_MAP['bak_daily'].values()} == {'stock_quote_supplement_daily'}
    fee = next(f for f in load_catalog()['etf_basic'].fields if f['name'] == 'mgt_fee')
    assert fee['unit'] == '' and fee['multiplier'] == '1'


def test_financial_same_name_different_meaning_is_not_collapsed():
    fields = {f['name']: f for f in builtin_catalog()['fundamental_periodic'].fields}
    for endpoint, column, description in (
        ('income', 'research_and_development_expense', '研发费用'),
        ('fina_indicator', 'research_and_development_investment', '研发投入合计'),
    ):
        for name in (endpoint, endpoint + '_vip'):
            assert financial_column(name, 'rd_exp') == column
            assert FIELD_MAP[name]['rd_exp'] == ('fundamental_periodic', column)
            assert description in fields[column]['description']
            actual = [r for r in all_source_mappings()
                      if r['source_endpoint'] == name and r['source_field'] == 'rd_exp']
            assert len(actual) == 1 and actual[0]['canonical_column'] == column
    assert 'deprecated' in fields['rd_exp']['description']
    assert not any(r['canonical_column'] == 'rd_exp' for r in all_source_mappings())


def test_business_descriptions_do_not_masquerade_as_units():
    for source in load_catalog().values():
        for field in source.fields:
            assert not field['unit'] or field['unit'] != field['description']
    units = {f['name']: (f['source_unit'], f['unit'], f['multiplier'])
             for f in load_catalog()['daily_basic'].fields}
    assert units['total_share'] == ('万股', '股', '10000')
    assert units['total_mv'] == ('万元', '元', '10000')
    assert units['turnover_rate'] == ('%', '%', '1')
    assert FIELD_MAP['income']['ann_date'] != FIELD_MAP['income']['f_ann_date']


def test_event_metadata_uses_actual_event_dates_and_distinct_dividend_concepts():
    rows = [r for r in all_source_mappings() if r['canonical_table'] == 'corporate_action']
    for endpoint, source_date in (('namechange', 'start_date'), ('dividend', 'ex_date')):
        dates = {r['canonical_column']: r['source_field'] for r in rows
                 if r['source_endpoint'] == endpoint and r['canonical_column'] in {'event_date', 'effective_date'}}
        assert dates == {'event_date': source_date, 'effective_date': source_date}
    assert FIELD_MAP['dividend']['stk_bo_rate'] == ('corporate_action', 'share_dividend')
    assert 'stk_div' not in FIELD_MAP['dividend']
    assert FIELD_MAP['dividend']['cash_div'] == ('corporate_action', 'cash_dividend')


def test_retrieval_new_endpoints_have_documented_bounds_and_safe_splits():
    bak = tushare_policy('bak_daily')
    assert bak['row_limit'] == 7000 and bak['mode'] == 'offset'
    assert bak['basis'] == 'documented_offset'
    etf = tushare_policy('etf_basic')
    assert etf['row_limit'] == 5000 and etf['mode'] == 'partition'
    assert load_catalog()['etf_basic'].date_field is None
    assert load_catalog()['bak_daily'].date_field == 'trade_date'
    source = load_catalog()['etf_basic'].as_dict()
    statuses = partition('etf_basic', {}, source)
    assert statuses == [{'list_status': s} for s in ('L', 'D', 'P')]
    assert partition('etf_basic', {'list_status': 'D'}, source) == [
        {'list_status': 'D', 'exchange': exchange} for exchange in ('SH', 'SZ')]
    assert partition('etf_basic', {'list_status': 'D', 'exchange': 'SH'}, source) == []
