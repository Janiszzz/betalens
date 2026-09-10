from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import betalens.factor.mining as mining
from betalens.factor.config import ConfigError, factor_spec_options


FACTOR_ROOT = Path(__file__).parents[1] / "betalens-factor"
if str(FACTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(FACTOR_ROOT))

import factor_template  # noqa: E402


def _config(*, use_industry: bool, **preprocessing):
    return {
        "factor_spec": {
            "inputs": {"close_wide": "收盘价(元)"},
            "compute_kwargs": {},
            "direction": "positive",
            "table_name": "daily_market",
            "index_code": "000906.SH",
            "use_industry": use_industry,
            "use_mktcap": False,
            "industry_scheme": "申万一级行业",
            "backtest_metric": "收盘价(元)",
            **preprocessing,
        },
        "weight": {
            "mode": "classic-long-short",
            "long_groups": None,
            "short_groups": None,
        },
    }


def test_factor_spec_preprocessing_defaults_preserve_legacy_behavior():
    enabled = factor_spec_options(_config(use_industry=True), "factor.yaml")
    disabled = factor_spec_options(_config(use_industry=False), "factor.yaml")

    assert enabled["winsorize_method"] == "mad"
    assert enabled["standardize_method"] == "zscore"
    assert enabled["neutralize"] is True
    assert disabled["winsorize_method"] == "none"
    assert disabled["standardize_method"] == "none"
    assert disabled["neutralize"] is False


def test_factor_spec_preprocessing_options_are_independent_and_validated():
    options = factor_spec_options(
        _config(
            use_industry=True,
            winsorize_method="none",
            winsorize_n=1,
            standardize_method="rank",
            neutralize=False,
        ),
        "factor.yaml",
    )

    assert options["winsorize_method"] == "none"
    assert options["winsorize_n"] == 1.0
    assert options["standardize_method"] == "rank"
    assert options["neutralize"] is False

    with pytest.raises(ConfigError, match="winsorize_method"):
        factor_spec_options(
            _config(use_industry=True, winsorize_method="invalid"),
            "factor.yaml",
        )


def test_ordinary_pipeline_can_disable_winsorization_only():
    factor_template._ensure_runtime()
    timestamp = pd.Timestamp("2024-01-02 15:00:00")
    raw = pd.DataFrame(
        {
            "input_ts": timestamp,
            "code": [f"S{i}" for i in range(10)],
            "F": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0],
        }
    )
    pipeline = factor_template.FactorPipeline(
        factor_template.FactorSpec(name="F", inputs={}, compute=lambda: None)
    )

    processed, stats = pipeline._preprocess_with_stats(
        raw,
        "F",
        industry_scheme=None,
        mktcap_col=None,
        preprocessing={
            "winsorize_method": "none",
            "winsorize_n": 3.0,
            "standardize_method": "zscore",
            "neutralize": False,
        },
        verbose=False,
    )

    expected = (raw["F"] - raw["F"].mean()) / raw["F"].std()
    pd.testing.assert_series_equal(processed["F"], expected, check_names=False)
    assert stats is None


def test_mining_pipeline_honors_preprocessing_options():
    timestamp = pd.Timestamp("2024-01-02 15:00:00")
    values = pd.DataFrame(
        {
            "input_ts": timestamp,
            "code": [f"S{i}" for i in range(10)],
            "F": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0],
        }
    )
    data = mining.MiningData(
        inputs={},
        price=pd.DataFrame(),
        execution_price=pd.DataFrame(),
        trade_status=None,
        pit=None,
        universe=list(values["code"]),
    )
    factor = SimpleNamespace(
        name="F",
        use_industry=False,
        use_mktcap=False,
        winsorize_method="percentile",
        winsorize_n=10.0,
        standardize_method="none",
        neutralize=False,
    )

    processed = mining._preprocess(values, factor, [timestamp.date()], data)

    assert processed["F"].max() < values["F"].max()
    assert processed["F"].min() > values["F"].min()


def test_mining_config_overrides_factor_preprocessing():
    module = types.ModuleType("_preprocessing_override_factor")

    def make_mining_spec(_params):
        factor = SimpleNamespace(
            name="F",
            use_industry=True,
            use_mktcap=False,
            winsorize_method=None,
            winsorize_n=3.0,
            standardize_method=None,
            neutralize=None,
        )
        return mining.MiningSpec(factor_spec=factor)

    module.make_mining_spec = make_mining_spec
    sys.modules[module.__name__] = module
    try:
        spec = mining._import_spec(
            module.__name__,
            {},
            {
                "winsorize_method": "none",
                "standardize_method": "rank",
                "neutralize": False,
            },
        )
    finally:
        sys.modules.pop(module.__name__, None)

    assert spec.factor_spec.winsorize_method == "none"
    assert spec.factor_spec.standardize_method == "rank"
    assert spec.factor_spec.neutralize is False
