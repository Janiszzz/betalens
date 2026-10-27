from datetime import date
from decimal import Decimal

import pytest

from betalens_db_manager.business_semantics import (
    corporate_action_partial, daily_average_price, derive_previous_names, financial_record,
)
from betalens_db_manager.catalog import dataset
from betalens_db_manager.mapping import FIELD_MAP
from betalens_db_manager.service import TushareImportService, normalize_source


def financial(endpoint="income", **updates):
    source = dict(ts_code="000001.SZ", end_date="20251231", ann_date="20260301",
                  f_ann_date="20260305", end_type="4", report_type="1", comp_type="1",
                  update_flag="0", revenue="100.000", rd_exp="23.500")
    source.update(updates)
    return financial_record(endpoint, source, FIELD_MAP[endpoint.removesuffix("_vip")])


@pytest.mark.parametrize("field,value", [
    ("ann_date", "20260302"), ("f_ann_date", "20260306"), ("end_type", "3"),
    ("update_flag", "1"), ("revenue", "101"), ("rd_exp", "23.51"),
])
def test_financial_version_contains_every_revision_field(field, value):
    before, after = financial(), financial(**{field: value})
    assert before["version_id"] != after["version_id"]


def test_financial_versions_are_normalized_and_independent_of_transport():
    original = financial()
    repeated = financial("income_vip", revenue=Decimal("100"), ann_date=date(2026, 3, 1),
                         _downloaded_at="later", _row=99, source_endpoint="income_vip", import_task_id="other")
    assert original["version_id"] == repeated["version_id"]
    assert repeated["statement_type"] == "income"
    assert original["announcement_date"] == date(2026, 3, 1)
    assert original["actual_announcement_date"] == date(2026, 3, 5)
    assert original["end_type"] == "4"


def test_financial_research_measures_are_not_conflated():
    income, indicator = financial(), financial("fina_indicator")
    assert income["research_and_development_expense"] == Decimal("23.5")
    assert indicator["research_and_development_investment"] == Decimal("23.5")
    assert "research_and_development_investment" not in income
    assert "research_and_development_expense" not in indicator
    assert "rd_exp" not in income and "rd_exp" not in indicator


def map_rows(endpoint, raw_rows):
    spec = dataset(endpoint)
    fields = [field["name"] for field in spec.fields]
    rows, bad = normalize_source(spec, {"fields": fields, "items": [[row.get(f) for f in fields] for row in raw_rows]},
                                 "task", "slice", "2026-10-24T00:00:00Z")
    assert not bad
    service = object.__new__(TushareImportService)
    return service._map_part(endpoint, rows, "task", "2026-10-24T00:00:00Z", [])[0]


def test_service_financial_version_and_evidence_survive_full_mapping():
    raw = dict(ts_code="000001.SZ", end_date="20251231", ann_date="20260301", f_ann_date="20260305",
               end_type="4", report_type="1", comp_type="1", update_flag="0", rd_exp="23.5")
    ordinary = map_rows("income", [raw])["fundamental_periodic"][0]
    vip = map_rows("income_vip", [raw])["fundamental_periodic"][0]
    assert ordinary["version_id"] == vip["version_id"]
    assert ordinary["research_and_development_expense"] == Decimal("23.5")
    assert ordinary.get("rd_exp") is None
    assert ordinary["source_endpoint"] == "income" and vip["source_endpoint"] == "income_vip"
    assert ordinary["source_record_hash"] and ordinary["source_row_no"] == 1
    assert ordinary["import_task_id"] == "task"


def test_manager_vip_overlay_runs_once_and_preserves_version_identity(tmp_path):
    from betalens_db_manager.config import Settings
    from betalens_db_manager.manager import DatabaseManagerService
    manager = DatabaseManagerService(Settings(config_dir=tmp_path / 'config', state_dir=tmp_path / 'state'))
    task = manager.plan({'datasets': ['income', 'income_vip'], 'codes': ['000001.SZ'],
                         'start_date': '2025-12-31', 'end_date': '2025-12-31', 'targets': ['postgres']})
    raw = dict(ts_code='000001.SZ', end_date='20251231', ann_date='20260301', f_ann_date='20260305',
               end_type='4', report_type='1', comp_type='1', update_flag='0', rd_exp='23.5')
    fields = [field['name'] for field in dataset('income').fields]
    source, rejected = normalize_source(dataset('income'), {'fields': fields, 'items': [[raw.get(f) for f in fields]]},
                                        task['id'], 'slice', '2026-10-24T00:00:00Z')
    assert not rejected
    ordinary = manager._map_part('income', source, task['id'], '2026-10-24T00:00:00Z', [])[0]['fundamental_periodic'][0]
    vip = manager._map_part('income_vip', source, task['id'], '2026-10-24T00:00:00Z', [])[0]['fundamental_periodic'][0]
    assert ordinary['version_id'] == vip['version_id']
    assert ordinary['source_endpoint'] == 'income' and vip['source_endpoint'] == 'income_vip'


def test_actual_dividend_uses_ex_date_and_share_bonus_only():
    raw = dict(ts_code="000001.SZ", end_date="20251231", ann_date="20260301", div_proc="实施",
               ex_date="20260618", cash_div=Decimal("0.20"), stk_div=Decimal("0.5"),
               stk_bo_rate=Decimal("0.2"), stk_co_rate=Decimal("0.3"))
    actual = corporate_action_partial("dividend", raw)
    assert actual["event_date"] == actual["effective_date"] == date(2026, 6, 18)
    assert actual["cash_dividend"] == Decimal("0.20")
    assert actual["share_dividend"] == Decimal("0.2")
    assert "end_date" not in actual


@pytest.mark.parametrize("updates", [{"div_proc": "预案"}, {"ex_date": None}, {"div_proc": None}])
def test_proposed_or_undated_dividend_does_not_become_an_actual_event(updates):
    raw = dict(ts_code="000001.SZ", end_date="20251231", ann_date="20260301", div_proc="实施", ex_date="20260618")
    raw.update(updates)
    assert corporate_action_partial("dividend", raw) is None
    if raw.get("div_proc") is not None:
        assert not map_rows("dividend", [raw]).get("corporate_action")


def test_service_dividend_event_and_implementation_announcement_are_distinct():
    raw = dict(ts_code="000001.SZ", end_date="20251231", ann_date="20260301", div_proc="实施",
               ex_date="20260618", imp_ann_date="20260610", stk_bo_rate="0.2", stk_co_rate="0.3", stk_div="0.5")
    actual = map_rows("dividend", [raw])["corporate_action"][0]
    assert actual["event_date"] == actual["effective_date"] == date(2026, 6, 18)
    assert actual["available_at"].date() == date(2026, 6, 10)  # Shanghai next midnight expressed in UTC
    assert actual["availability_basis"].startswith("implementation_announcement_")
    assert actual["share_dividend"] == Decimal("0.2")


def test_name_event_uses_start_and_does_not_trust_undocumented_old_name():
    raw = dict(ts_code="000001.SZ", name="新名称", old_name="不能证实", start_date="20260618", end_date="20260930")
    actual = corporate_action_partial("namechange", raw)
    assert actual["event_date"] == actual["effective_date"] == date(2026, 6, 18)
    assert actual["new_name"] == "新名称" and actual["old_name"] is None
    assert corporate_action_partial("namechange", {**raw, "start_date": None}) is None
    assert map_rows("namechange", [raw])["corporate_action"][0]["effective_date"] == date(2026, 6, 18)


def history_row(name, start, end=None):
    return dict(ts_code="000001.SZ", name=name, start_date=start, end_date=end)


def test_old_names_require_complete_unambiguous_contiguous_history():
    rows = [history_row("旧名称", "20200101", "20260617"), history_row("新名称", "20260618")]
    assert not derive_previous_names(rows)
    assert derive_previous_names(list(reversed(rows)), complete_history=True) == {("000001.SZ", date(2026, 6, 18)): "旧名称"}
    assert not derive_previous_names([rows[0], {**rows[1], "start_date": "20260619"}], complete_history=True)
    assert not derive_previous_names([rows[0], {**rows[0], "name": "有歧义"}, rows[1]], complete_history=True)
    assert not derive_previous_names([history_row("重叠", "20190101"), *rows], complete_history=True)


def test_index_turnover_does_not_become_index_point_average():
    assert daily_average_price("index", Decimal("100000000"), Decimal("200")) is None
    assert daily_average_price("stock", Decimal("1200"), Decimal("100")) == Decimal("12")
    assert daily_average_price("fund", "1200.00000001", "100") == Decimal("12.0000000001")
    assert daily_average_price("stock", "1200", "0") is None


def test_bse_mapping_never_guesses_code_transition_from_listing_date():
    row = map_rows("bse_mapping", [dict(o_code="430047.BJ", n_code="920047.BJ", name="代码映射", list_date="20211115")])["security_code_history"][0]
    assert row["list_date"] == date(2021, 11, 15)
    assert row["valid_from"] is None
