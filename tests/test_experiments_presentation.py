"""Factor catalogs and live/archive rendering share the same data contract."""
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from betalens.analyst import Analyst, presentation as shared
from betalens.experiments import ExperimentStore
from betalens.experiments.api import create_app
from betalens.experiments.catalog import factor_catalog
from betalens.experiments.snapshot import Snapshot
from test_experiments import bt, config


def test_catalog_chooses_visible_source_and_filters_before_selection(tmp_path, monkeypatch):
    with ExperimentStore(tmp_path) as store:
        f = store.register_factor(key="f", name="中文因子", category="量价")
        raw = Snapshot(tables={"nav": pd.Series([1., 1.1])}, metrics={"夏普比率": 1.5, "年化收益": .12, "最大回撤": .08})
        first = store._save_snapshot(raw, factor_id=f, config=config(20))
        daily = config(60); daily["run"]["rebal_freq"] = "D"
        store._save_snapshot(Snapshot(metrics={"sharpe_median": 9., "ann_ret_median": .8}), factor_id=f, config=daily, kind="mining")
        latest = store._save_snapshot(Snapshot(metrics={"sharpe_median": 3., "ann_ret_median": .2}), factor_id=f, config=config(20), kind="mining")
        derived = store.derive_factor(f, key="twenty", name="二十日", locks={"factor_spec.compute_kwargs.window": 20})
        monkeypatch.setattr(store, "read_table", lambda *a, **k: pytest.fail("Catalog loaded an original table"))
        response = factor_catalog(store)
        assert all(row["source"]["experiment_id"] == first for row in response["items"])
        assert all(row["metrics"]["sharpe"] == 1.5 for row in response["items"])
        assert factor_catalog(store, basis="latest", entry_type="root")["items"][0]["source"]["experiment_id"] == latest
        assert factor_catalog(store, basis="latest", frequency="D")["items"][0]["metrics"]["sharpe"] == 9.
        assert factor_catalog(store, metrics={"sharpe": {"gte": 2}})["total"] == 0
        assert factor_catalog(store, search="二十日")["items"][0]["id"] == derived
        assert factor_catalog(store, entry_type="derived")["total"] == 1


def test_catalog_sorts_missing_values_last_both_directions(tmp_path):
    with ExperimentStore(tmp_path) as store:
        for key, value in (("a", 3.), ("b", None), ("c", 0.)):
            factor = store.register_factor(key=key, name=key)
            if value is not None:
                store._save_snapshot(Snapshot(metrics={"夏普比率": value}), factor_id=factor)
        for descending, keys in ((True, ["a", "c", "b"]), (False, ["c", "a", "b"])):
            assert [r["key"] for r in factor_catalog(store, sort_by="sharpe", descending=descending)["items"]] == keys


def test_catalog_separates_rank_ic_from_pearson_ic(tmp_path):
    with ExperimentStore(tmp_path) as store:
        factor = store.register_factor(key="f", name="因子")
        store._save_snapshot(Snapshot(metrics={"rank_ic_median": .08, "ic_ir_median": .7}), factor_id=factor, kind="mining")
        row = factor_catalog(store)["items"][0]
        assert row["metrics"]["ic"] is None and row["metrics"]["rank_ic"] == .08
        assert row["metric_basis"] == "窗口聚合指标"


def rich_run(backtest):
    backtest.daily_pnl_total = backtest.daily_pnl.sum(axis=1)
    days = backtest.daily_weight.index
    values = pd.DataFrame({"信号日": days, "股票代码": "000001.SZ", "因子值": range(len(days)), "分组": [1, 2] * 4})
    backtest.factor_values = values
    backtest.research_group_nav = pd.DataFrame({"G1": [1 + i * .01 for i in range(len(days))],
                                               "G2": [1 - i * .005 for i in range(len(days))]}, index=days)
    analyst = Analyst.from_backtest(backtest)
    analyst._name_map = {"000001.SZ": "归档名称"}
    backtest.trade_cycles = analyst.an.trade_cycles.copy()
    return SimpleNamespace(backtest=backtest, analyst=analyst, factor_values=values, chart_data=None,
                           pit_validation=None, neutralize_stats=None, profiling=None)


def test_archive_result_matches_dashboard_without_market_access(tmp_path, bt, monkeypatch):
    run = rich_run(bt)
    with ExperimentStore(tmp_path) as store:
        factor = store.register_factor(key="f", name="共享展示", category="量价")
        cfg = config(); cfg["run"]["n_quantiles"] = 2
        identity = store.save_run(run, factor_id=factor, config=cfg)
        import betalens.analyst.naming as naming
        monkeypatch.setattr(naming, "get_name_map", lambda *a, **k: pytest.fail("Offline view queried a market DB"))
        with shared.archived_names(run.analyst._name_map):
            expected_charts = {**shared.build_chart_data(bt, run.factor_values),
                **shared.build_generated_chart_data(bt, run.factor_values, 2),
                "profiling": shared.build_factor_profile_payload(run.factor_values)}
            expected_positions = shared.build_position_table(bt)
            expected_metrics = shared.build_metrics(run.analyst, bt)
            expected_timing = shared.build_timing_payload(bt, run.factor_values)
        with TestClient(create_app(tmp_path)) as client:
            response = client.get(f"/api/experiments/{identity}/result")
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["metrics"] == expected_metrics
            assert payload["charts"] == expected_charts
            assert payload["timing"] == expected_timing
            assert payload["tables"]["positions"]["total"] == len(expected_positions)
            assert payload["availability"]["raw"]
            page = client.get(f"/api/experiments/{identity}/result-table/positions", params={"page": 2, "size": 3}).json()
            assert page["rows"] == expected_positions[3:6]
            profile = client.get(f"/api/experiments/{identity}/profiling", params={"date_from": "2025-01-03", "date_to": "2025-01-05"}).json()
            assert profile["available"] and profile["summary"]["count"] == 3
            assert client.get(f"/api/experiments/{identity}/result-table/positions", params={"date_from": "2025-01-03", "date_to": "2025-01-03"}).json()["total"] == 2
            for kind in ("config", "dump", "report", "html"):
                exported = client.get(f"/api/experiments/{identity}/result-download/{kind}")
                assert exported.status_code == 200, exported.text[:200]
        assert shared._ARCHIVED_NAMES.get() is None


def test_history_pagination_and_metrics_only_result(tmp_path):
    with ExperimentStore(tmp_path) as store:
        factor = store.register_factor(key="f", name="因子")
        for n in range(3):
            identity = store._save_snapshot(Snapshot(metrics={"sharpe_median": n}), factor_id=factor, config=config(20), kind="mining")
        with TestClient(create_app(tmp_path)) as client:
            history = client.get(f"/api/factors/{factor}/history", params={"page_size": 2}).json()
            assert history["total"] == 3 and len(history["items"]) == 2
            assert history["items"][0]["id"] == identity
            result = client.get(f"/api/experiments/{identity}/result")
            assert result.status_code == 200, result.text
            assert not result.json()["availability"]["raw"]
            assert result.json()["charts"]["nav"] == []
            assert client.post(f"/api/experiments/{identity}/result").status_code == 405
            assert client.get("/api/catalog", params={"metrics": '{"bogus":{"gte":1}}'}).status_code == 400


def test_scope_selection_does_not_mix_independent_windows(tmp_path):
    with ExperimentStore(tmp_path) as store:
        factor = store.register_factor(key="f", name="窗口")
        index = pd.date_range("2025-01-01", periods=2)
        identity = store._save_snapshot(Snapshot(tables={"scopes/a/nav": pd.Series([1., 1.1], index=index),
            "scopes/b/nav": pd.Series([1., .9], index=index)}, metadata={"available_scopes": ["a", "b"]}), factor_id=factor)
        with TestClient(create_app(tmp_path)) as client:
            assert client.get(f"/api/experiments/{identity}/result", params={"scope": "a"}).json()["charts"]["nav"][-1]["nav"] == 1.1
            assert client.get(f"/api/experiments/{identity}/result", params={"scope": "b"}).json()["charts"]["nav"][-1]["nav"] == .9
            assert client.get(f"/api/experiments/{identity}/result", params={"scope": "missing"}).status_code == 400
