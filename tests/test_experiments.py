"""Behavioral contracts for the durable archive and its read-only viewer."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import io
import json
import multiprocessing as mp
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from betalens.analyst import Analyst
from betalens.experiments import ArchiveError, ConflictError, ExperimentInput, ExperimentStore, MissingArtifactError
from betalens.experiments.api import create_app
from betalens.experiments.snapshot import Snapshot
from betalens.factor.mining_audit import FactorMiningResult, MiningResult


@pytest.fixture
def bt():
    days = pd.date_range("2025-01-02", periods=8, name="datetime")
    returns = pd.Series([.01, -.02, .03, .005, -.008, .02, .01, -.003], index=days)
    amounts = (1 + returns).cumprod() * 100_000
    nav = pd.concat([pd.Series([1.], index=pd.DatetimeIndex([pd.Timestamp("2025-01-01")], name="datetime")), amounts / 100_000]).rename("nav")
    weights = pd.DataFrame({"000001.SZ": .8, "cash": .2}, index=days)
    values = weights.mul(amounts, axis=0)
    logs = pd.DataFrame(columns=["datetime", "code", "direction", "shares", "signed_shares", "value", "price", "equity_before", "fee"])
    prices = pd.DataFrame({"000001.SZ": np.linspace(10., 11., len(days)), "cash": 1.}, index=days)
    return SimpleNamespace(nav=nav, daily_returns=returns, daily_amount=amounts,
        weight=weights, actual_weight=weights.copy(), daily_weight=weights.copy(),
        daily_position_value=values, daily_pnl=weights.mul(amounts.diff().fillna(1000), axis=0),
        position=values / prices, rebalance_log=logs, cost_price=prices,
        synthetic_settlements=pd.DataFrame(), risk_events=pd.DataFrame(),
        symbol="测试组合", metric="close", initial_amount=100_000, ftc=0., ptc=0.,
        start=days[0], end=days[-1], calculation_version="backtest-v2-drift-risk", data_sources={"prices": "database"})


@pytest.fixture
def archive(tmp_path):
    with ExperimentStore(tmp_path / "experiments", chunk_rows=3) as store:
        factor = store.register_factor(key="momentum", name="动量因子", category="量价")
        yield store, factor


def config(window=20):
    return {"factor_spec": {"compute_kwargs": {"window": window}}, "run": {"rebal_freq": "W"}}


def test_native_roundtrip_and_chunk_projection(archive):
    store, factor = archive
    index = pd.date_range("2025-01-01", periods=9, freq="h", tz="Asia/Shanghai", name="时间")
    index = index + pd.Timedelta(nanoseconds=123)
    frame = pd.DataFrame({"数值": pd.array(range(9), dtype="Int64"), "中文": pd.array(["甲", None, "乙"] * 3, dtype="string")}, index=index)
    frame.attrs = {"index_role": "execution_events", "timestamp": index[0], "shape": (3, 3)}
    multi = pd.DataFrame({"值": [1., np.nan]}, index=pd.MultiIndex.from_tuples([(index[0], "甲"), (index[1], "乙")], names=["时间", "证券"]))
    empty = pd.DataFrame({"代码": pd.Series(dtype="string"), "数量": pd.Series(dtype="Int64")})
    series = pd.Series(np.arange(9.), index=index, name=("净值", "组合"))
    experiment = store._save_snapshot(Snapshot(tables={"frame": frame, "multi": multi, "empty": empty, "nav": series}), factor_id=factor)
    pd.testing.assert_frame_equal(frame, store.read_table(experiment, "frame"))
    pd.testing.assert_frame_equal(multi, store.read_table(experiment, "multi"))
    pd.testing.assert_frame_equal(empty, store.read_table(experiment, "empty"))
    pd.testing.assert_series_equal(series, store.read_table(experiment, "nav"))
    pd.testing.assert_frame_equal(frame.iloc[2:7][["数值"]], store.read_table(experiment, "frame", offset=2, limit=5, columns=["数值"]))
    pd.testing.assert_frame_equal(frame.iloc[2:4], store.read_table(experiment, "frame", start=index[2].isoformat(), end=index[3].isoformat()))
    assert store.read_table(experiment, "frame", offset=100, limit=10).empty


def test_nested_scientific_artifacts_and_table_attrs(archive):
    store, factor = archive
    frame = pd.DataFrame({"weight": [1., .5]})
    attached = pd.DataFrame({"因子值": [3., 4.]})
    frame.attrs = {"factor_values": attached, "array": np.array([[1., np.nan], [2., 3.]])}
    extras = {"profiling": {"table": attached, "image": b"PNG bytes", "plot": (1, "甲"), "at": pd.Timestamp("2025-01-01")}}
    e = store._save_snapshot(Snapshot(tables={"weights": frame}, extras=extras), factor_id=factor)
    restored = store.read_table(e, "weights")
    pd.testing.assert_frame_equal(restored.attrs["factor_values"], attached)
    np.testing.assert_equal(restored.attrs["array"], frame.attrs["array"])
    result = store.read_extras(e)["profiling"]
    pd.testing.assert_frame_equal(result["table"], attached)
    assert result["image"] == b"PNG bytes" and result["plot"] == (1, "甲")
    assert result["at"] == pd.Timestamp("2025-01-01")
    assert "PNG bytes" not in json.dumps(store.get(e))


def test_idempotency_conflict_and_factor_identity(archive):
    store, factor = archive
    snap = Snapshot(metrics={"sharpe": 1.})
    first = store._save_snapshot(snap, factor_id=factor, source_id="run-1")
    assert store._save_snapshot(snap, factor_id=factor, source_id="run-1") == first
    with pytest.raises(ConflictError):
        store._save_snapshot(Snapshot(metrics={"sharpe": 2.}), factor_id=factor, source_id="run-1")
    assert store.query()["total"] == 1
    assert store.register_factor(key="momentum", name="动量因子", category="量价") == factor
    with pytest.raises(ConflictError):
        store.register_factor(key="momentum", name="另一个因子")


def test_flat_derived_membership_is_retroactive_and_does_not_duplicate(archive):
    store, factor = archive
    a = store._save_snapshot(Snapshot(), factor_id=factor, config=config(20))
    store._save_snapshot(Snapshot(), factor_id=factor, config=config(60))
    d = store.derive_factor(factor, key="factor_20", name="二十日动量", locks={"factor_spec.compute_kwargs.window": 20})
    weekly = store.derive_factor(factor, key="weekly", name="周频", locks={"run.rebal_freq": "W"})
    assert store.query(factor_id=d)["items"][0]["id"] == a
    assert store.query(factor_id=weekly)["total"] == 2
    assert store.query()["total"] == 2
    assert store.query(factor_id=d, config={"factor_spec.compute_kwargs.window": 60})["total"] == 0
    with pytest.raises(ArchiveError):
        store.derive_factor(d, key="nested", name="不允许递归", locks={})
    with pytest.raises(ArchiveError):
        store._save_snapshot(Snapshot(), factor_id=d, config=config(60))
    store.update_factor(d, name="中文新名称", description="研究说明")
    assert store.get_factor(d)["name"] == "中文新名称"
    with pytest.raises(ArchiveError):
        store.update_factor(d, locks={})


def test_parent_locks_cannot_be_overridden_even_through_nested_paths(tmp_path):
    with ExperimentStore(tmp_path) as store:
        root = store.register_factor(key="f", name="固定母因子", locks={"a": {"window": 20}})
        with pytest.raises(ConflictError):
            store.derive_factor(root, key="bad", name="冲突", locks={"a.window": 60})
        store.derive_factor(root, key="ok", name="兼容", locks={"a.window": 20})


def test_summary_queries_do_not_read_any_artifact(archive, monkeypatch):
    store, factor = archive
    store._save_snapshot(Snapshot(tables={"nav": pd.Series([1., 2.])}, metrics={"sharpe": 1.2}), factor_id=factor, parameters={"window": 20}, tags=["精选"])
    monkeypatch.setattr(store, "read_table", lambda *a, **k: pytest.fail("Summary query loaded original data"))
    result = store.query(factor_id=factor, parameters={"window": 20}, metrics={"sharpe": {"gte": 1}}, tags=["精选"], has_raw=True)
    assert result["total"] == 1
    assert store.query(metrics={"missing": {"gt": 0}})["total"] == 0
    assert store.query(search="精选")["total"] == 1


def test_validation_failure_cannot_publish_half_an_experiment(archive):
    store, factor = archive
    with pytest.raises(ArchiveError):
        store._save_snapshot(Snapshot(tables={"good": pd.Series([1.]), "bad": object()}), factor_id=factor)
    assert store.query()["total"] == 0


def test_capacity_failure_rolls_back_record_index_and_source(tmp_path):
    with ExperimentStore(tmp_path, map_size=1024 * 1024) as store:
        factor = store.register_factor(key="f", name="容量测试")
        noise = pd.DataFrame(np.random.default_rng(2).normal(size=(50_000, 8)))
        with pytest.raises(ArchiveError, match="capacity"):
            store._save_snapshot(Snapshot(tables={"large": noise}), factor_id=factor, source_id="too-big")
        assert store.query(factor_id=factor)["total"] == 0
        assert store._save_snapshot(Snapshot(), factor_id=factor, source_id="too-big")


def test_backtest_run_and_analyst_roundtrip_are_offline(archive, bt, monkeypatch):
    store, factor = archive
    analyst = Analyst.from_backtest(bt, risk_free_rate=.025, annualizer=240, window=3, name="中文组合")
    analyst._name_map = {"000001.SZ": "平安银行"}
    original = analyst.an.summary()
    run = SimpleNamespace(backtest=bt, analyst=analyst, factor_values=pd.DataFrame({"值": [3.]}), profiling={"覆盖率": .95})
    ids = [store.save_backtest(bt, analyst=analyst, factor_id=factor, config=config()),
           store.save_analyst(analyst, factor_id=factor), store.save_run(run, factor_id=factor)]
    import betalens.analyst.naming as naming
    import betalens.datafeed as datafeed
    monkeypatch.setattr(naming, "get_name_map", lambda *_: pytest.fail("Queried market DB"))
    monkeypatch.setattr(datafeed, "Datafeed", lambda *_: pytest.fail("Opened market DB"))
    for identity in ids:
        restored = store.to_analyst(identity)
        pd.testing.assert_series_equal(restored.an.nav, analyst.an.nav)
        pd.testing.assert_series_equal(restored.an.returns, analyst.an.returns)
        assert restored.name == "中文组合"
        assert restored.an.risk_free_rate == .025
        for key, value in original.items():
            result = restored.an.summary()[key]
            if isinstance(value, (int, float, np.number)):
                np.testing.assert_allclose(result, value, equal_nan=True)
            else:
                assert result == value
        restored.top_holdings_df()
        restored.interactive_plots()
    changed = store.to_analyst(ids[0], risk_free_rate=0.)
    assert changed.an.risk_free_rate == 0.
    assert store.get(ids[0])["metadata"]["analysis"]["risk_free_rate"] == .025
    assert store.read_extras(ids[-1]) == {"profiling": {"覆盖率": .95}}


def test_missing_raw_data_are_explicit(archive):
    store, factor = archive
    e = store._save_snapshot(Snapshot(metrics={"sharpe": 1}), factor_id=factor)
    with pytest.raises(MissingArtifactError):
        store.read_table(e, "nav")
    with pytest.raises(MissingArtifactError):
        store.to_analyst(e)


def test_batch_partial_success_retry_and_cross_batch_isolation(archive, bt):
    store, factor = archive
    values = [ExperimentInput(bt, factor, config(), source_id="a"),
              ExperimentInput(object(), factor, source_id="broken"),
              ExperimentInput(bt, factor, config(60), source_id="b")]
    batch = store.save_batch(values, source_id="parallel-1")
    assert batch["status"] == "partial" and len(batch["experiment_ids"]) == 2 and len(batch["failures"]) == 1
    again = store.save_batch(values, source_id="parallel-1")
    assert again["id"] == batch["id"] and len(again["experiment_ids"]) == 2
    values[1] = ExperimentInput(bt, factor, config(120), source_id="broken")
    fixed = store.save_batch(values, source_id="parallel-1")
    assert fixed["status"] == "complete" and len(fixed["experiment_ids"]) == 3
    store.save_batch({"a": bt}, source_id="parallel-2", factor_id=factor, config=config())
    assert store.query()["total"] == 4
    assert store.query(batch_id=batch["id"])["total"] == 3


def test_generator_failure_is_reported_after_successes(archive):
    store, factor = archive
    def results():
        yield ExperimentInput(Snapshot(), factor)
        raise RuntimeError("upstream result iterator failed")
    batch = store.save_batch(results())
    assert batch["status"] == "partial"
    assert "iterator" == batch["failures"][0]["source_id"]


def test_batch_coalesces_identical_effective_parameters_and_rejects_changed_data(archive, bt):
    store, factor = archive
    batch = store.save_batch({"a": ExperimentInput(bt, factor, config()),
                              "b": ExperimentInput(bt, factor, config())})
    assert batch["status"] == "complete" and len(batch["experiment_ids"]) == 1
    changed = SimpleNamespace(**vars(bt))
    changed.nav = bt.nav * 2
    bad = store.save_batch({"a": ExperimentInput(bt, factor, config()),
                           "b": ExperimentInput(changed, factor, config())})
    assert bad["status"] == "partial" and len(bad["experiment_ids"]) == 1
    assert "ConflictError" in bad["failures"][0]["error"]


def _writer(path, factor, index):
    with ExperimentStore(path) as store:
        identity = store._save_snapshot(Snapshot(tables={"nav": pd.Series([1., 1.1])}, metrics={"n": index}), factor_id=factor, source_id=f"worker-{index}")
        return identity, store.query()["total"]


def test_process_writers_and_readers_and_shared_handles(archive):
    store, factor = archive
    with ExperimentStore(store.path, readonly=True) as reader:
        assert reader._shared is store._shared
        with ThreadPoolExecutor(max_workers=2) as threads:
            assert list(threads.map(lambda _: reader.query()["total"], range(4))) == [0] * 4
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
            results = list(pool.map(_writer, [str(store.path)] * 4, [factor] * 4, range(4)))
        assert len({identity for identity, _ in results}) == 4
        assert reader.query()["total"] == 4
        with pytest.raises(PermissionError):
            reader.register_factor(key="bad", name="只读")
    assert store.query()["total"] == 4


def mining_result(tmp_path, *, captured=False, capture_error=False):
    rows = pd.DataFrame([
        {"candidate_id": c, "params_json": json.dumps({"window": w}), "window_id": f"252/21/{i}",
         "window_start": "2025-01-01", "window_end": "2025-01-09", "sharpe": score, "ann_ret": .1, "error": None}
        for c, w, score in (("c1", 20, 1.), ("c2", 60, 2.)) for i in range(2)])
    summary = pd.DataFrame([{"candidate_id": c, "params_json": json.dumps({"window": w}), "objective": score, "sharpe_median": score} for c, w, score in (("c1", 20, 1.), ("c2", 60, 2.))])
    snapshot = Snapshot(tables={"nav": pd.Series([1., 1.1], index=pd.date_range("2025-01-01", periods=2))}, metrics={"夏普比率": 3.})
    run = FactorMiningResult("动量", "run-1", tmp_path, "complete", stage_window_results={"coarse": rows, "fine": rows.copy()},
        stage_summaries={"coarse": summary, "fine": summary.copy()}, selected_candidates=summary.iloc[1:].copy(),
        archive_context={"factor_class": "量价", "strategy_type": "cross_sectional"},
        candidate_configs={"c1": config(20), "c2": config(60)},
        selected_results={"c2": {"scopes": {"total": snapshot}, "config": config(60)}} if captured else {},
        capture_errors={"c2": "capture failed"} if capture_error else {})
    return MiningResult("launch-1", (run,))


def test_mining_groups_candidates_and_preserves_source_and_capture_metrics(tmp_path):
    with ExperimentStore(tmp_path / "db") as store:
        result = mining_result(tmp_path, captured=True)
        batch = store.save_mining(result)
        assert batch["status"] == "complete" and len(batch["experiment_ids"]) == 2
        docs = store.query(kind="mining")["items"]
        selected = next(d for d in docs if d["metadata"]["selected"])
        assert selected["has_raw"] and selected["metrics"]["sharpe_median"] == 2.
        assert selected["metadata"]["scope_metrics"]["total"]["夏普比率"] == 3.
        assert len(store.read_table(selected["id"], "mining/window_results")) == 4
        assert len(store.save_mining(result)["experiment_ids"]) == 2
        assert store.query()["total"] == 2
        root = selected["factor_id"]
        derived = store.derive_factor(root, key="sixty", name="六十日版本", locks={"factor_spec.compute_kwargs.window": 60})
        assert store.query(factor_id=derived)["total"] == 1


def test_mining_capture_failure_preserves_metrics(tmp_path):
    with ExperimentStore(tmp_path / "db") as store:
        batch = store.save_mining(mining_result(tmp_path, capture_error=True))
        assert batch["status"] == "partial"
        assert store.query()["total"] == 2
        selected = next(d for d in store.query()["items"] if d["metadata"]["selected"])
        assert not selected["has_raw"] and selected["capture_error"] == "capture failed"


def test_http_readonly_filters_tables_downloads_and_compare(archive, bt):
    store, factor = archive
    first = store.save_backtest(bt, factor_id=factor, config=config(), parameters={"window": 20})
    second = store._save_snapshot(Snapshot(metrics={"sharpe": 2.}), factor_id=factor)
    with TestClient(create_app(store.path)) as client:
        assert client.get("/api/health").json()["readonly"] is True
        assert client.get("/api/factors", params={"search": "动量"}).json()[0]["name"] == "动量因子"
        records = client.get("/api/experiments", params={"factor_id": factor, "parameters": '{"window":20}'}).json()
        assert records["total"] == 1
        assert client.get("/api/experiments", params={"metrics": "[1]"}).status_code == 400
        assert client.get(f"/api/experiments/{second}/charts").status_code == 404
        charts = client.get(f"/api/experiments/{first}/charts")
        assert charts.status_code == 200 and len(charts.json()["nav"]) == len(bt.nav)
        table = client.get(f"/api/experiments/{first}/table", params={"name": "daily_weight", "page": 2, "page_size": 3}).json()
        assert len(table["rows"]) == 3 and table["total"] == 8
        for fmt in ("csv", "parquet"):
            response = client.get(f"/api/experiments/{first}/download", params={"name": "nav", "format": fmt})
            assert response.status_code == 200
            if fmt == "parquet":
                assert len(pd.read_parquet(io.BytesIO(response.content))) == len(bt.nav)
        compare = client.get("/api/compare", params={"ids": first + "," + second}).json()
        assert len(compare) == 2 and compare[0]["charts"] and compare[1]["charts"] is None
        for method in ("post", "put", "patch", "delete"):
            assert getattr(client, method)(f"/api/experiments/{first}").status_code == 405


def test_heatmap_aggregation_and_scheme_filters(tmp_path):
    with ExperimentStore(tmp_path / "db") as store:
        store.save_mining(mining_result(tmp_path))
        with TestClient(create_app(store.path)) as client:
            heatmap = client.get("/api/mining/heatmap", params={"x": "window", "stage": "coarse", "window_scheme": "252/21"}).json()
            assert heatmap["x"] == [20, 60]
            assert heatmap["z"] == [[1., 2.]]
            assert heatmap["counts"] == [[2, 2]]
            assert client.get("/api/mining/heatmap", params={"x": "unknown"}).status_code == 400


def test_imports_do_not_load_database_or_web_dependencies():
    code = "import sys; import betalens; baseline=set(sys.modules); import betalens.experiments; assert not ({'lmdb','pyarrow','fastapi'} & (set(sys.modules)-baseline))"
    subprocess.run([sys.executable, "-c", code], check=True)
