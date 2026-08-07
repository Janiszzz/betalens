from __future__ import annotations

from types import SimpleNamespace
import pickle
import sys

import numpy as np
import pandas as pd

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
ALPHA_ROOT = REPO_ROOT / "betalens-factor" / "alpha101"
for path in (REPO_ROOT, REPO_ROOT / "betalens-factor", ALPHA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha101_formulas import (  # noqa: E402
    compute_alpha,
    default_formula_params,
    get_definition,
    required_history_bars_for_alpha,
    resolve_formula_params,
)
from alpha101_parameters import candidate_values, formula_param_candidates, formula_param_gid  # noqa: E402
from betalens.factor import mining  # noqa: E402


def _panel() -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2024-01-01", periods=8)
    columns = ["A", "B"]
    base = pd.DataFrame(np.arange(16, dtype=float).reshape(8, 2) + 10, index=index, columns=columns)
    return {"close_wide": base, "open_wide": base - 0.2, "high_wide": base + 0.4, "low_wide": base - 0.4}


def test_formula_defaults_and_overrides_are_validated() -> None:
    inputs = _panel()
    inputs["returns_wide"] = inputs["close_wide"].pct_change(fill_method=None).fillna(0) * 100
    baseline = compute_alpha(1, **inputs)
    defaults = compute_alpha(1, formula_params=default_formula_params(1), **inputs)
    pd.testing.assert_frame_equal(baseline, defaults)

    changed = dict(default_formula_params(1))
    changed["window_1"] = 40
    assert required_history_bars_for_alpha(1, changed) > required_history_bars_for_alpha(1)
    try:
        resolve_formula_params(1, {"unknown": 1})
    except KeyError:
        pass
    else:
        raise AssertionError("unknown parameters must fail")


def test_candidate_generation_is_bounded_and_deterministic() -> None:
    first = formula_param_candidates(65, max_candidates=8)
    second = formula_param_candidates(65, max_candidates=8)
    assert first == second
    assert 1 <= len(first) <= 8
    assert first[0] == default_formula_params(65)
    assert formula_param_gid(65, first[0]) != formula_param_gid(65, first[1])
    window_spec = get_definition(65).parameters["window_1"]
    values = candidate_values(window_spec)
    assert len(values) >= 5
    assert values == sorted(values)
    assert values[-1] >= values[0] * 4


def test_paired_rolling_windows_are_contiguous(monkeypatch) -> None:
    days = list(pd.bdate_range("2024-01-01", periods=12).date)
    monkeypatch.setattr("betalens.datafeed.get_absolute_trade_days", lambda *args, **kwargs: days)

    pairs = mining.gen_rolling_train_test_windows(
        "2024-01-01", "2024-01-31", train_len=4, test_len=2, step=3
    )

    assert pairs[0] == ("2024-01-01", "2024-01-04", "2024-01-05", "2024-01-08")
    assert pairs[1][0] == "2024-01-04"
    assert pairs[1][2] == "2024-01-10"


def test_align_daily_wides_uses_latest_timestamp_per_day() -> None:
    days = pd.bdate_range("2024-01-01", periods=2)
    open_index = pd.DatetimeIndex([days[0] + pd.Timedelta(hours=9, minutes=30), days[1] + pd.Timedelta(hours=9, minutes=30)])
    close_index = pd.DatetimeIndex([days[0] + pd.Timedelta(hours=15), days[1] + pd.Timedelta(hours=15)])
    opens = pd.DataFrame([[1.0], [2.0]], index=open_index, columns=["A"])
    closes = pd.DataFrame([[1.1], [2.1]], index=close_index, columns=["A"])

    aligned = mining.align_daily_wides({"open": opens, "close": closes})

    expected = close_index
    pd.testing.assert_index_equal(aligned["open"].index, expected)
    pd.testing.assert_index_equal(aligned["close"].index, expected)
    assert aligned["open"].iloc[:, 0].tolist() == [1.0, 2.0]


def test_core_cache_masks_market_and_industry_inputs(monkeypatch, tmp_path) -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    columns = ["A", "B"]
    market = pd.DataFrame(np.arange(8, dtype=float).reshape(4, 2), index=index, columns=columns)
    industry = pd.DataFrame([["x", "y"]] * 4, index=index, columns=columns)
    pit_days = [ts.date() for ts in index]
    spec = SimpleNamespace(
        inputs={"close_wide": "收盘价(元)"},
        industry_inputs={"sector_wide": "申万一级行业"},
        index_code="000906.SH",
        mask_inputs_by_pit=True,
        table_name="daily_market",
        industry_scheme="申万一级行业",
        use_industry=True,
        backtest_metric="收盘价(元)",
    )
    config = SimpleNamespace(
        cache_dir=tmp_path,
        output_dir=tmp_path,
        rebuild_cache=True,
        max_warmup_days=2,
        universe=None,
        factor_module="alpha101_mining",
        spec_factory="make_mining_spec",
    )

    monkeypatch.setattr(
        "betalens.datafeed.get_absolute_trade_days",
        lambda *args, **kwargs: pit_days,
    )
    monkeypatch.setattr(mining, "_load_spec", lambda *args, **kwargs: spec)
    monkeypatch.setattr(
        mining,
        "build_pit_universe",
        lambda days, index_code: {day: {"A"} for day in days},
    )
    monkeypatch.setattr(mining, "fetch_daily_wide", lambda *args, **kwargs: market.copy())
    monkeypatch.setattr(mining, "fetch_industry_wide", lambda *args, **kwargs: industry.copy())

    paths = mining.build_cache_for_config(config, [("2024-01-01", "2024-01-04")], {})
    payload = pickle.loads(paths.data.read_bytes())
    assert payload["inputs"]["close_wide"]["B"].isna().all()
    assert payload["inputs"]["sector_wide"]["B"].isna().all()
    assert payload["price"]["B"].notna().all()
    assert "申万一级行业" in payload["industry_by_scheme"]

    config.rebuild_cache = False
    monkeypatch.setattr(mining, "fetch_daily_wide", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    mining.build_cache_for_config(config, [("2024-01-01", "2024-01-04")], {})
