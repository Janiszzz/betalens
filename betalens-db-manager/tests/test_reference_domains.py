from copy import deepcopy

import pytest

from betalens_db_manager.reference_contracts import TABLES, MODULE_TABLE
from betalens_db_manager.reference_domains import build_scopes, check_integrity


def request(modules=None, **kw):
    return {"modules": modules or ["calendar"], "security_codes": ["000001.SZ"],
            "start_date": "2026-10-19", "end_date": "2026-10-21", "exchanges": ["SZSE"],
            "schemes": ["SW2021"], "levels": [1], **kw}


def bundle(endpoint, rows, **parameters):
    return {"endpoint": endpoint, "parameters": parameters, "rows": rows, "complete": True,
            "downloaded_at": "2026-10-22T01:00:00+00:00", "archive_sha": "raw-proof"}


def calendar(days=("20261019", "20261020", "20261021"), exchange="SZSE", opened="1"):
    return bundle("trade_cal", [{"exchange": exchange, "cal_date": day, "is_open": opened,
                                 "pretrade_date": "20261016"} for day in days],
                  exchange=exchange, start_date="20261019", end_date="20261021")


def lifecycle(code="000001.SZ"):
    return bundle("stock_basic", [{"ts_code": code, "list_date": "19910403", "delist_date": None,
                                    "list_status": "L"}], ts_code=code, list_status="L")


def membership(industry="801780.SI", **kw):
    return {"ts_code": "000001.SZ", "l1_code": industry, "l1_name": "银行", "l2_code": "801782.SI",
            "l2_name": "银行Ⅱ", "l3_code": "850951.SI", "l3_name": "股份制银行", "in_date": "20000101",
            "out_date": None, "is_new": "Y", **kw}


def industries(rows=None):
    rows = [membership()] if rows is None else rows
    dictionary = [{"index_code": r["l1_code"], "industry_name": r.get("l1_name"), "src": "SW2021"} for r in rows]
    return [bundle("index_member_all", [r for r in rows if r["is_new"] == "Y"], ts_code="000001.SZ", is_new="Y"),
            bundle("index_member_all", [r for r in rows if r["is_new"] == "N"], ts_code="000001.SZ", is_new="N"),
            bundle("index_classify", dictionary, src="SW2021", level="L1")]


def suspended(timing=None, event="S", day="20261020"):
    return bundle("suspend_d", [{"ts_code": "000001.SZ", "trade_date": day, "suspend_type": event,
                                  "suspend_timing": timing}], ts_code="000001.SZ", trade_date=day)


def snapshot(scopes, revision="r1", seq=1, publish=True):
    return {"revisions": [{"revision_id": revision, "ingestion_seq": seq, "import_task_id": "task-"+revision,
                           "ingested_at": "2026-10-22T01:02:00Z"}],
            "scopes": [{**s, "revision_id": revision} for s in scopes],
            "rows": [{**r, "revision_id": revision, "scope_id": s["scope_id"]} for s in scopes for r in s["rows"]],
            "events": [{"revision_id": revision, "event_type": "publish", "event_seq": seq}] if publish else []}


def combine(*snapshots):
    return {key: [item for shot in snapshots for item in shot[key]] for key in ("revisions", "scopes", "rows", "events")}


def test_calendar_missing_middle_day_cannot_be_hidden_by_other_exchange():
    req = request()
    scopes = build_scopes(req, [calendar(("20261019", "20261021")), calendar(exchange="SSE")])
    result = check_integrity(req, snapshot(scopes))
    assert [r["status"] for r in result["rows"]] == ["complete", "unknown", "complete"]
    assert result["rows"][1]["reason"] == "calendar_date_missing"
    assert not result["complete"]


def test_calendar_two_adjacent_tasks_keep_middle_gap_then_fill_it():
    first = request(end_date="2026-10-19")
    last = request(start_date="2026-10-21")
    joined = combine(snapshot(build_scopes(first, [calendar()]), "r1"), snapshot(build_scopes(last, [calendar()]), "r2", 2))
    assert check_integrity(request(), joined)["rows"][1]["reason"] == "not_requested"
    middle = request(start_date="2026-10-20", end_date="2026-10-20")
    joined = combine(joined, snapshot(build_scopes(middle, [calendar()]), "r3", 3))
    assert check_integrity(request(), joined)["complete"]


def test_bse_explicitly_references_sse_and_retains_source_exchange():
    req = request(exchanges=["BSE"])
    result = check_integrity(req, snapshot(build_scopes(req, [calendar(exchange="SSE")])))
    assert result["complete"]
    assert all(r["data"]["exchange"] == "BSE" and r["data"]["reference_exchange"] == "SSE" for r in result["rows"])
    assert all("reference_calendar" in r["data"]["calendar_basis"] for r in result["rows"])


def test_is_open_filtered_calendar_cannot_prove_closed_dates():
    b = calendar()
    b["parameters"]["is_open"] = "1"
    scopes = build_scopes(request(), [b])
    assert not any(s["complete"] for s in scopes)


def test_industry_full_history_requires_current_and_historical_queries():
    req = request(["industry"])
    scope = build_scopes(req, industries()[:1])[0]
    assert not scope["complete"]
    assert check_integrity(req, snapshot([scope]))["rows"][0]["reason"] == "history_query_incomplete"


def test_industry_query_matches_interval_overlap_not_in_date_window():
    req = request(["industry"])
    result = check_integrity(req, snapshot(build_scopes(req, industries())))
    assert result["complete"]
    assert result["rows"][0]["data"]["valid_from"] == "2000-01-01"
    assert result["rows"][0]["data"]["available_at"] is None


def test_unpublished_and_interrupted_new_versions_never_hide_trusted_old():
    req = request(["industry"])
    old = snapshot(build_scopes(req, industries()), "old", 1)
    staged = snapshot(build_scopes(req, industries([membership("new", l1_name=None)])), "staged", 2, publish=False)
    staged["revisions"][0]["status"] = "interrupted"
    result = check_integrity(req, combine(old, staged))
    assert result["complete"] and result["rows"][0]["data"]["industry_code"] == "801780.SI"
    only_staged = check_integrity(req, staged)
    assert only_staged["rows"][0]["reason"] == "interrupted"


def test_whole_industry_snapshot_replaces_old_keys_without_filling_new_nulls():
    req = request(["industry"])
    old = snapshot(build_scopes(req, industries()), "old", 1)
    revised = snapshot(build_scopes(req, industries([membership("new", l1_name=None, in_date="20261020")])), "new", 2)
    result = check_integrity(req, combine(old, revised))
    assert not result["complete"]
    assert result["rows"][0]["reason"] == "industry_history_gap"
    assert result["rows"][1]["data"][0]["industry_name"] is None
    assert "801780.SI" not in str(result)


def test_revocation_restores_preceding_snapshot_and_keeps_audit_history():
    req = request(["industry"], audit=True)
    old = snapshot(build_scopes(req, industries()), "old", 1)
    revised = snapshot(build_scopes(req, industries([membership("new")])), "new", 2)
    shot = combine(old, revised)
    shot["events"].append({"revision_id": "new", "event_type": "revoke", "event_seq": 3})
    result = check_integrity(req, shot)
    assert result["complete"] and result["rows"][0]["revision_id"] == "old"
    assert len(result["versions"]) == 2
    assert result["versions"][1]["revoked"]


def test_industry_overlap_gap_and_unproven_end_boundary_are_unknown_or_conflict():
    req = request(["industry"])
    rows = [membership("first", out_date="20261020", is_new="N"), membership("second", in_date="20261019")]
    result = check_integrity(req, snapshot(build_scopes(req, industries(rows))))
    assert result["rows"][0]["status"] == "conflict"
    assert result["rows"][1]["reason"] == "industry_boundary_or_meaning_unknown"
    assert result["rows"][2]["data"]["industry_code"] == "second"


def test_industry_different_stock_never_fills_requested_stock_gap():
    req = request(["industry"], security_codes=["000002.SZ"])
    result = check_integrity(req, snapshot(build_scopes(req, industries())))
    assert not result["complete"]


def test_suspension_no_event_requires_stock_and_calendar_evidence():
    req = request(["suspension"])
    empty = bundle("suspend_d", [], ts_code="000001.SZ", start_date="20261019", end_date="20261021")
    for evidence, reason in (([], "calendar_unknown"), ([calendar()], "lifecycle_unknown")):
        result = check_integrity(req, snapshot(build_scopes(req, [empty, *evidence])))
        assert result["rows"][0]["reason"] == reason
    result = check_integrity(req, snapshot(build_scopes(req, [empty, calendar(), lifecycle()])))
    assert result["complete"]
    assert all(r["negative_evidence"] and r["is_suspended"] is False for r in result["rows"])


def test_suspension_s_only_query_does_not_prove_absence_of_resumption():
    req = request(["suspension"])
    b = bundle("suspend_d", [], ts_code="000001.SZ", start_date="20261019", end_date="20261021", suspend_type="S")
    assert all(not s["complete"] for s in build_scopes(req, [b, calendar(), lifecycle()]))


def test_full_day_suspension_utc_storage_and_different_stocks():
    req = request(["suspension"], start_date="2026-10-20", end_date="2026-10-20", check_times=["00:00", "09:30", "23:59:59"])
    scope = build_scopes(req, [suspended(), calendar(), lifecycle()])[0]
    assert scope["rows"][0]["start_at"] == "2026-10-19T16:00:00+00:00"
    assert scope["rows"][0]["end_at"] == "2026-10-20T16:00:00+00:00"
    result = check_integrity(req, snapshot([scope]))
    assert result["complete"] and all(r["is_suspended"] for r in result["rows"])


def test_multiple_intraday_intervals_preserved_with_exact_boundaries_unknown():
    req = request(["suspension"], start_date="2026-10-20", end_date="2026-10-20",
                  check_times=["09:30", "09:45", "10:00", "10:01", "13:15", "14:01"])
    scope = build_scopes(req, [suspended("09:30-10:00;13:00-14:00"), calendar(), lifecycle()])[0]
    assert len(scope["rows"]) == 2
    result = check_integrity(req, snapshot([scope]))
    assert [r["reason"] for r in result["rows"]] == ["source_time_boundary_unknown", "suspended", "source_time_boundary_unknown",
                                                         "outside_reported_intervals", "suspended", "outside_reported_intervals"]


@pytest.mark.parametrize("timing", ["09:30-", "9:99-10:00", "14:00-09:30", "不明", "09:30-10:00/13:00-14:00"])
def test_bad_timing_is_retained_as_unknown(timing):
    req = request(["suspension"], start_date="2026-10-20", end_date="2026-10-20", check_times=["09:45"])
    scope = build_scopes(req, [suspended(timing), calendar(), lifecycle()])[0]
    assert scope["complete"] and scope["rows"][0]["raw_timing"] == timing
    result = check_integrity(req, snapshot([scope]))
    assert result["rows"][0]["reason"] == "suspension_time_unknown"
    assert "is_suspended" not in result["rows"][0]


def test_resumption_without_time_and_overlapping_events_do_not_guess():
    req = request(["suspension"], start_date="2026-10-20", end_date="2026-10-20", check_times=["09:45"])
    result = check_integrity(req, snapshot(build_scopes(req, [suspended(event="R"), calendar(), lifecycle()])))
    assert result["rows"][0]["reason"] == "suspension_time_unknown"
    result = check_integrity(req, snapshot(build_scopes(req, [suspended("09:30-10:00,09:45-11:00"), calendar(), lifecycle()])))
    assert result["rows"][0]["status"] == "conflict"


def test_closed_date_is_not_open_and_unlisted_is_distinct_from_normal():
    req = request(["suspension"], start_date="2026-10-20", end_date="2026-10-20")
    empty = bundle("suspend_d", [], trade_date="20261020", ts_code="000001.SZ")
    closed = check_integrity(req, snapshot(build_scopes(req, [empty, calendar(opened="0"), lifecycle()])))
    assert closed["rows"][0]["reason"] == "not_open"
    life = lifecycle()
    life["rows"][0]["list_date"] = "20261021"
    unlisted = check_integrity(req, snapshot(build_scopes(req, [empty, calendar(), life])))
    assert unlisted["rows"][0]["reason"] == "not_listed"


def test_dependency_revoke_or_new_head_invalidates_derived_evidence():
    req = request(["calendar"], end_date="2026-10-19")
    scopes = build_scopes(req, [calendar()])
    base = snapshot(scopes, "base", 1)
    dependent = snapshot(deepcopy(scopes), "dependent", 2)
    dependent["scopes"][0]["dependencies"] = [{"revision_id": "base"}]
    shot = combine(base, dependent)
    shot["events"].append({"revision_id": "base", "event_type": "revoke", "event_seq": 3})
    assert check_integrity(req, shot)["rows"][0]["reason"] == "dependency_revoked"


def test_all_business_rows_fit_contract_and_inputs_are_not_mutated():
    req = request(["calendar", "industry", "suspension"])
    bundles = [calendar(), lifecycle(), suspended(), *industries()]
    original = deepcopy(bundles)
    scopes = build_scopes(req, bundles)
    for scope in scopes:
        names = {f["name"] for f in TABLES[MODULE_TABLE[scope["module"]]].fields}
        for row in scope["rows"]:
            assert set(row) <= names
    assert bundles == original


def test_invalid_request_dates_levels_and_injected_module_are_rejected():
    for req in [request(start_date="2026-02-30"), request(end_date="2020-01-01"),
                request(levels=[True]), request(modules=["calendar; drop table"]), request(exchanges=["bogus"]),
                request(security_codes=[123]), request(security_codes=["000001.SZ; DROP TABLE"] )]:
        with pytest.raises(ValueError):
            build_scopes(req, [])


def test_sw_classification_version_needs_dictionary_evidence():
    req = request(["industry"])
    result = check_integrity(req, snapshot(build_scopes(req, industries()[:2])))
    assert not result["complete"]
    assert result["rows"][0]["reason"] == "industry_boundary_or_meaning_unknown"


def test_invalid_target_readback_excludes_revision_and_warns_when_older_served():
    req = request(["industry"])
    old = snapshot(build_scopes(req, industries()), "old", 1)
    latest = snapshot(build_scopes(req, industries([membership("new")])), "new", 2)
    latest["verification"] = [{"revision_id": "new", "verified": False}]
    result = check_integrity(req, latest)
    assert result["rows"][0]["reason"] == "target_verification_failed"
    combined = combine(old, latest)
    combined["verification"] = latest["verification"]
    result = check_integrity(req, combined)
    assert result["rows"][0]["revision_id"] == "old"
    assert result["rows"][0]["warnings"][0]["revision_id"] == "new"


def test_publication_time_is_read_from_event_and_orphan_publish_never_trusted():
    req = request(end_date="2026-10-19")
    shot = snapshot(build_scopes(req, [calendar()]))
    shot["events"][0]["created_at"] = "2026-10-22T01:05:00Z"
    assert check_integrity(req, shot)["rows"][0]["published_at"] == "2026-10-22T01:05:00Z"
    shot["revisions"] = []
    assert not check_integrity(req, shot)["complete"]


def test_self_contained_calendar_proof_is_invalidated_by_new_contradictory_trusted_calendar():
    req = request(["suspension"], start_date="2026-10-20", end_date="2026-10-20")
    empty = bundle("suspend_d", [], trade_date="20261020", ts_code="000001.SZ")
    shot = snapshot(build_scopes(req, [empty, calendar(), lifecycle()]), "suspension", 1)
    assert check_integrity(req, shot)["complete"]
    cal_req = request(start_date="2026-10-20", end_date="2026-10-20")
    same_calendar = snapshot(build_scopes(cal_req, [calendar()]), "calendar-old", 2)
    assert check_integrity(req, combine(shot, same_calendar))["complete"]
    changed_calendar = snapshot(build_scopes(cal_req, [calendar(opened="0")]), "calendar-new", 3)
    result = check_integrity(req, combine(shot, same_calendar, changed_calendar))
    assert result["rows"][0]["reason"] == "dependency_changed"


def test_calendar_malformed_previous_date_is_unknown_and_failed_dictionary_is_not_proof():
    b = calendar()
    b["rows"][0]["pretrade_date"] = "20261020"
    assert check_integrity(request(), snapshot(build_scopes(request(), [b])))["rows"][0]["reason"] == "calendar_field_meaning_unknown"
    req = request(["industry"])
    bundles = industries()
    bundles[-1]["complete"] = False
    assert not check_integrity(req, snapshot(build_scopes(req, bundles)))["complete"]
