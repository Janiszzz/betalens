"""Mining hooks that adapt all Alpha101 formulas to betalens.factor.mining."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from alpha101_formulas import (
    INDUSTRY_INPUTS,
    MARKET_INPUTS,
    compute_alpha,
    get_definition,
    required_history_bars_for_alpha,
)
from alpha101_parameters import formula_param_gid
from factor_template_alpha101 import FactorPipeline, FactorSpec


_CLASS_DIR = Path(__file__).resolve().parent
_ALL_MARKET_INPUTS = {argument: metric for argument, metric in MARKET_INPUTS.values()}
_ALL_INDUSTRY_INPUTS = {argument: scheme for argument, scheme in INDUSTRY_INPUTS.values()}


def _require(params: Mapping[str, Any], key: str) -> Any:
    if key not in params:
        raise KeyError(f"mining params missing required key: {key}")
    return params[key]


def _alpha_id(params: Mapping[str, Any]) -> int:
    value = _require(params, "alpha_id")
    return get_definition(int(value)).number


def compute_alpha_mining(**kwargs):
    alpha_id = kwargs.pop("alpha_id")
    formula_params = kwargs.pop("formula_params", None)
    return compute_alpha(alpha_id, formula_params=formula_params, **kwargs)


def make_mining_spec(params: Mapping[str, Any]) -> FactorSpec:
    """Declare the class-wide cache contract and one candidate's formula settings."""
    alpha_id = _alpha_id(params)
    formula_params = dict(_require(params, "formula_params"))
    return FactorSpec(
        name=get_definition(alpha_id).name,
        inputs=dict(_ALL_MARKET_INPUTS),
        industry_inputs=dict(_ALL_INDUSTRY_INPUTS),
        compute=compute_alpha_mining,
        compute_kwargs={"alpha_id": alpha_id, "formula_params": formula_params},
        strategy_type="cross_section",
        required_history_bars=required_history_bars_for_alpha(alpha_id, formula_params),
        mask_inputs_by_pit=True,
        direction="positive",
        table_name="daily_market",
        index_code="000906.SH",
        use_industry=True,
        use_mktcap=False,
        industry_scheme="申万一级行业",
        backtest_metric="开盘价(元)",
        weight_mode="classic-long-short",
    )


def mining_gid(params: Mapping[str, Any]) -> str:
    alpha_id = _alpha_id(params)
    n_quantiles = int(_require(params, "n_quantiles"))
    return f"{formula_param_gid(alpha_id, _require(params, 'formula_params'))}_q{n_quantiles}"


def mining_warmup_days(params: Mapping[str, Any]) -> int:
    alpha_id = _alpha_id(params)
    bars = required_history_bars_for_alpha(alpha_id, _require(params, "formula_params"))
    return max(30, int(bars) * 2 + 30)


def _validation_spec(params: Mapping[str, Any], rank: int) -> FactorSpec:
    alpha_id = _alpha_id(params)
    definition = get_definition(alpha_id)
    formula_params = dict(_require(params, "formula_params"))
    return FactorSpec(
        name=f"{definition.name}_valid{rank}",
        inputs=dict(definition.inputs),
        industry_inputs=dict(definition.industry_inputs),
        compute=compute_alpha_mining,
        compute_kwargs={"alpha_id": alpha_id, "formula_params": formula_params},
        strategy_type="cross_section",
        required_history_bars=required_history_bars_for_alpha(alpha_id, formula_params),
        mask_inputs_by_pit=True,
        direction="positive",
        table_name="daily_market",
        index_code="000906.SH",
        use_industry=True,
        use_mktcap=False,
        industry_scheme="申万一级行业",
        backtest_metric="开盘价(元)",
        weight_mode="classic-long-short",
    )


def mining_valid_report(params, rank, output_dir, start_date, end_date):
    """Run the selected candidate through the normal exact pipeline and persist its config."""
    alpha_id = _alpha_id(params)
    target_dir = Path(output_dir) / f"exact_rank_{rank}_{get_definition(alpha_id).name}"
    target_dir.mkdir(parents=True, exist_ok=True)
    factor_config = _CLASS_DIR / get_definition(alpha_id).name / f"factor_{get_definition(alpha_id).name}.yaml"
    config = yaml.safe_load(factor_config.read_text(encoding="utf-8"))
    config["factor_spec"]["compute_kwargs"]["formula_params"] = dict(_require(params, "formula_params"))
    config["run"].update({
        "start_date": str(start_date),
        "end_date": str(end_date),
        "rebal_freq": "W",
        "n_quantiles": int(_require(params, "n_quantiles")),
        "include_profiling": False,
        "dump_excel": False,
        "output_dir": str(target_dir),
    })
    (target_dir / "best_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return FactorPipeline(_validation_spec(params, int(rank))).run(
        str(start_date),
        str(end_date),
        warmup_days=mining_warmup_days(params),
        rebal_freq="W",
        n_quantiles=int(_require(params, "n_quantiles")),
        output_dir=str(target_dir),
        include_profiling=False,
        dump_excel=False,
        verbose=False,
    )


__all__ = [
    "compute_alpha_mining",
    "make_mining_spec",
    "mining_gid",
    "mining_valid_report",
    "mining_warmup_days",
]
