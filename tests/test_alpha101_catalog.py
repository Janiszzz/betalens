from __future__ import annotations

import importlib.util
import inspect
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_DIR = REPO_ROOT / "betalens-factor" / "alpha101"
for _path in (REPO_ROOT, REPO_ROOT / "betalens-factor", CLASS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from alpha101_formulas import ALPHA_DEFINITIONS, compute_alpha, default_compute_kwargs  # noqa: E402
from factor_template_alpha101 import (  # noqa: E402
    clean_inf,
    decay_linear,
    indneutralize,
    product,
    rank,
    scale,
    signed_power,
    ts_argmax,
    ts_argmin,
    ts_rank,
    window,
)


def test_window_rounding_uses_floor_plus_half() -> None:
    assert window(3.49) == 3
    assert window(3.5) == 4
    assert window(0.1) == 1


def test_cross_sectional_rank_ties_and_nan() -> None:
    frame = pd.DataFrame([[1.0, 1.0, 3.0, np.nan]], columns=list("abcd"))

    actual = rank(frame).iloc[0]

    assert actual["a"] == pytest.approx(0.5)
    assert actual["b"] == pytest.approx(0.5)
    assert actual["c"] == pytest.approx(1.0)
    assert pd.isna(actual["d"])


def test_time_series_operators_require_full_windows() -> None:
    frame = pd.DataFrame({"a": [1.0, 3.0, 2.0, 4.0]})

    ranked = ts_rank(frame, 3)
    maxima = ts_argmax(frame, 3)
    minima = ts_argmin(frame, 3)
    products = product(frame, 3)
    decayed = decay_linear(frame, 3)

    assert ranked.iloc[:2].isna().all().all()
    assert ranked.iloc[2, 0] == pytest.approx(2 / 3)
    assert maxima.iloc[2, 0] == 2.0
    assert minima.iloc[2, 0] == 1.0
    assert products.iloc[2, 0] == 6.0
    assert decayed.iloc[2, 0] == pytest.approx(13 / 6)


def test_scale_signed_power_and_inf_cleanup() -> None:
    frame = pd.DataFrame([[1.0, -3.0], [0.0, 0.0]], columns=["a", "b"])

    scaled = scale(frame, 2.0)
    powered = signed_power(frame, 0.5)
    cleaned = clean_inf(pd.DataFrame([[np.inf, -np.inf, 1.0]]))

    assert scaled.iloc[0].tolist() == pytest.approx([0.5, -1.5])
    assert scaled.iloc[1].isna().all()
    assert powered.iloc[0].tolist() == pytest.approx([1.0, -np.sqrt(3.0)])
    assert cleaned.iloc[0, :2].isna().all()


def test_industry_neutralization_is_daily_and_excludes_missing_groups() -> None:
    index = pd.date_range("2024-01-01", periods=2)
    values = pd.DataFrame(
        [[1.0, 3.0, 9.0, 7.0], [2.0, 4.0, 8.0, np.nan]],
        index=index,
        columns=list("abcd"),
    )
    groups = pd.DataFrame(
        [["x", "x", "y", None], ["x", "x", "y", "y"]],
        index=index,
        columns=values.columns,
    )

    actual = indneutralize(values, groups)

    assert actual.loc[index[0], ["a", "b", "c"]].tolist() == pytest.approx([-1.0, 1.0, 0.0])
    assert pd.isna(actual.loc[index[0], "d"])
    assert actual.loc[index[1], ["a", "b", "c"]].tolist() == pytest.approx([-1.0, 1.0, 0.0])


@pytest.fixture(scope="module")
def synthetic_alpha_inputs() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(20260730)
    index = pd.bdate_range("2022-01-03", periods=650)
    columns = [f"S{i:04d}" for i in range(40)]
    close = pd.DataFrame(
        30.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.018, (650, 40)), axis=0)),
        index=index,
        columns=columns,
    )
    open_ = close * (1 + pd.DataFrame(rng.normal(0, 0.006, close.shape), index=index, columns=columns))
    high = np.maximum(open_, close) * (1 + pd.DataFrame(rng.uniform(0.001, 0.02, close.shape), index=index, columns=columns))
    low = np.minimum(open_, close) * (1 - pd.DataFrame(rng.uniform(0.001, 0.02, close.shape), index=index, columns=columns))
    volume = pd.DataFrame(rng.lognormal(16, 0.5, close.shape), index=index, columns=columns)
    vwap = (open_ + high + low + close) / 4
    returns = close.pct_change(fill_method=None).fillna(0.0) * 100.0
    amount = volume * vwap
    cap = close * pd.DataFrame(rng.lognormal(18, 0.15, close.shape), index=index, columns=columns)
    sector_labels = np.array([f"sector-{i % 5}" for i in range(40)], dtype=object)
    industry_labels = np.array([f"industry-{i % 8}" for i in range(40)], dtype=object)
    subindustry_labels = np.array([f"subindustry-{i % 10}" for i in range(40)], dtype=object)

    def labels(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(np.tile(values, (len(index), 1)), index=index, columns=columns)

    return {
        "open_wide": open_,
        "close_wide": close,
        "high_wide": high,
        "low_wide": low,
        "volume_wide": volume,
        "vwap_wide": vwap,
        "returns_wide": returns,
        "amount_wide": amount,
        "cap_wide": cap,
        "sector_wide": labels(sector_labels),
        "industry_wide": labels(industry_labels),
        "subindustry_wide": labels(subindustry_labels),
    }


def test_all_101_formulas_on_synthetic_panel_without_mutation(synthetic_alpha_inputs) -> None:
    originals = {name: frame.copy(deep=True) for name, frame in synthetic_alpha_inputs.items()}

    for number, definition in ALPHA_DEFINITIONS.items():
        names = [*definition.inputs, *definition.industry_inputs]
        result = compute_alpha(number, **{name: synthetic_alpha_inputs[name] for name in names})

        assert result.shape == (650, 40), definition.name
        assert not np.isinf(result.to_numpy(dtype=float)).any(), definition.name
        tail_start = min(definition.required_history_bars, len(result) - 1)
        assert result.iloc[tail_start:].notna().any().any(), definition.name

    for name, original in originals.items():
        pd.testing.assert_frame_equal(synthetic_alpha_inputs[name], original)


def test_all_numeric_formula_positions_are_flat_and_semantically_named() -> None:
    parameters = [spec for definition in ALPHA_DEFINITIONS.values() for spec in definition.parameters.values()]

    assert len(parameters) >= 487
    assert all(not re.match(r"^(window|lag|coefficient|threshold)_\d+$", spec.name) for spec in parameters)
    assert all(
        default_compute_kwargs(definition.number)
        == {name: spec.default for name, spec in definition.parameters.items()}
        for definition in ALPHA_DEFINITIONS.values()
    )


def _load_module(path: Path, ordinal: int):
    module_spec = importlib.util.spec_from_file_location(f"_alpha101_catalog_{ordinal}", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_generated_scripts_and_yamls_match_registry_and_import_cleanly() -> None:
    seen = set()
    ordinal = 0
    for number, definition in ALPHA_DEFINITIONS.items():
        factor_dir = CLASS_DIR / definition.name
        for timing in (False, True):
            suffix = "_timing" if timing else ""
            name = f"{definition.name}{suffix}"
            script = factor_dir / f"factor_{name}.py"
            config_path = factor_dir / f"factor_{name}.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            module = _load_module(script, ordinal)
            ordinal += 1
            compute = getattr(module, f"compute_alpha{number}{suffix}")
            declared = [*definition.inputs, *definition.industry_inputs]
            signature = inspect.signature(compute)

            assert declared == [
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            ]
            keyword_only = [
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            ]
            expected_keyword_only = list(definition.parameters)
            if timing:
                expected_keyword_only.extend(["stock_code", "signal_weight"])
            assert keyword_only == expected_keyword_only
            assert module.spec.name == name
            assert module.spec.required_history_bars == definition.required_history_bars
            assert module.spec.industry_inputs == dict(definition.industry_inputs)
            assert config["meta"]["name"] == name
            assert config["factor_spec"]["inputs"] == dict(definition.inputs)
            assert config["factor_spec"]["industry_inputs"] == dict(definition.industry_inputs)
            assert config["factor_spec"]["mask_inputs_by_pit"] is True
            assert config["factor_spec"]["required_history_bars"] == definition.required_history_bars
            assert "formula_params" not in config["factor_spec"]["compute_kwargs"]
            assert {
                name: config["factor_spec"]["compute_kwargs"][name]
                for name in definition.parameters
            } == default_compute_kwargs(number)
            search_space = config["mining"]["search_space"]
            assert list(search_space) == list(definition.parameters)
            assert sum(len(values) > 1 for values in search_space.values()) <= 3
            assert np.prod([len(values) for values in search_space.values()], dtype=int) <= 256
            if timing:
                signal = config["factor_spec"]["compute_kwargs"]["signal_weight"]
                assert signal == {
                    "method": "rolling_z",
                    "window": 120,
                    "sigma": 1.0,
                    "operator": "gt",
                    "side": "short",
                    "max_weight": 1.0,
                }
            seen.add(name)

    assert len(seen) == 202


def test_generated_catalog_is_current() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLASS_DIR / "tools" / "generate_catalog.py"), "--check"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
