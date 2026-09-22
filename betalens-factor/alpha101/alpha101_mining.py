"""将 Alpha101 公式适配到统一 mining 执行接口。

本模块不决定参数空间；参数边界来自 ``parameter_space.yaml``，或在配置使用
``factors: all`` 时由 :mod:`alpha101_parameters` 生成。这里仅根据一个普通参数
字典构造 ``MiningSpec``，声明该 Alpha 的输入字段、公式参数、预热长度和
``precomputed`` 执行模式。
"""
from __future__ import annotations

from typing import Any, Mapping

from alpha101_formulas import (
    compute_alpha,
    get_definition,
    required_history_bars_for_alpha,
)
from factor_template_alpha101 import FactorSpec
from betalens.factor.mining import MiningSpec


def _require(params: Mapping[str, Any], key: str) -> Any:
    if key not in params:
        raise KeyError(f"mining params missing required key: {key}")
    return params[key]


def _alpha_id(params: Mapping[str, Any]) -> int:
    value = _require(params, "alpha_id")
    return get_definition(int(value)).number


def _formula_kwargs(params: Mapping[str, Any]) -> dict[str, Any]:
    definition = get_definition(_alpha_id(params))
    missing = [name for name in definition.parameters if name not in params]
    if missing and str(params.get("strategy_type", "")).lower() != "timing":
        raise KeyError(f"mining params missing formula keys: {', '.join(missing)}")
    return {
        name: params.get(name, definition.parameters[name].default)
        for name in definition.parameters
    }


def compute_alpha_mining(**kwargs):
    alpha_id = kwargs.pop("alpha_id")
    # Timing-only controls are consumed by the signal hook, not the Alpha
    # formula itself.
    for key in ("stock_code", "target_codes", "signal_weight"):
        kwargs.pop(key, None)
    return compute_alpha(alpha_id, **kwargs)


def make_mining_spec(params: Mapping[str, Any]) -> MiningSpec:
    """根据一个候选参数字典声明所选 Alpha 的计算与缓存输入。"""
    alpha_id = _alpha_id(params)
    definition = get_definition(alpha_id)
    formula_kwargs = _formula_kwargs(params)
    factor_spec = FactorSpec(
        name=definition.name,
        inputs=dict(definition.inputs),
        industry_inputs=dict(definition.industry_inputs),
        compute=compute_alpha_mining,
        compute_kwargs={"alpha_id": alpha_id, **formula_kwargs},
        strategy_type="cross_section",
        required_history_bars=required_history_bars_for_alpha(alpha_id, formula_kwargs),
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
    return MiningSpec(
        factor_spec=factor_spec,
        execution_mode="precomputed",
        warmup_days=lambda values: mining_warmup_days(values),
    )


def mining_warmup_days(params: Mapping[str, Any]) -> int:
    """按公式所需历史 bars 推导自然日预热长度，并保证至少 30 天。"""
    alpha_id = _alpha_id(params)
    bars = required_history_bars_for_alpha(alpha_id, _formula_kwargs(params))
    return max(30, int(bars) * 2 + 30)


__all__ = [
    "compute_alpha_mining",
    "make_mining_spec",
    "mining_warmup_days",
]
