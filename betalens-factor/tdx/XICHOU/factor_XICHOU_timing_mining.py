"""Mining adapter for the XICHOU single-index timing strategy.

The adapter deliberately keeps formula computation separate from signal
generation.  ``run_mining`` supplies candidate parameters to
``make_mining_spec``; the shared timing hook then turns the factor series into
target weights for the configured index code.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

_FACTOR_DIR = Path(__file__).resolve().parent
_CLASS_DIR = _FACTOR_DIR.parent
_FACTOR_ROOT = _CLASS_DIR.parent
_REPO_ROOT = _FACTOR_ROOT.parent
for _path in (_REPO_ROOT, _FACTOR_ROOT, _CLASS_DIR, _FACTOR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from betalens.factor.mining import MiningSpec
from betalens.factor.signal import standard_timing_weight_hook
from factor_template_tdx import FactorSpec
from factor_XICHOU import compute_xichou


def compute_xichou_timing_mining(
    low_wide,
    high_wide=None,
    *,
    sma_n: int = 3,
    ema_n: int = 3,
    llv_n: int = 15,
    **_kwargs: Any,
):
    """Compute XICHOU from daily-index low prices.

    ``high_wide`` is loaded for event-history signal methods but is not part of
    the XICHOU formula itself.  Extra timing controls are accepted and ignored
    so candidate dictionaries can be passed through without formula leakage.
    """
    del high_wide
    return compute_xichou(low_wide, int(sma_n), int(ema_n), int(llv_n))


def _int_param(params: Mapping[str, Any], key: str, default: int) -> int:
    value = params.get(key, default)
    return int(value) if value is not None else int(default)


def make_mining_spec(params: Mapping[str, Any]) -> MiningSpec:
    """Build a timing ``MiningSpec`` for one daily-index target."""
    formula_kwargs = {
        "sma_n": _int_param(params, "sma_n", 3),
        "ema_n": _int_param(params, "ema_n", 3),
        "llv_n": _int_param(params, "llv_n", 15),
    }
    target_codes = params.get("target_codes") or params.get("stock_code") or ("000300.SH",)
    signal_weight = {
        "method": params.get("signal_method", "event_history"),
        "operator": params.get("trigger_operator", "auto"),
        "side": params.get("signal_side", "long"),
        "max_weight": float(params.get("max_weight", 1.0)),
        # Keep the historical bare name accepted by this adapter while the
        # mining engine also understands the ``signal_max_gross_exposure``
        # alias used by generic timing configurations.
        "max_gross_exposure": float(params.get("max_gross_exposure", 1.0)),
    }
    if params.get("signal_window") is not None:
        signal_weight["window"] = int(params["signal_window"])
    if params.get("signal_sigma") is not None:
        signal_weight["sigma"] = float(params["signal_sigma"])
    factor_kwargs = {
        **formula_kwargs,
        "target_codes": target_codes,
        "stock_code": target_codes,
        "signal_weight": signal_weight,
        "trigger_threshold": float(params.get("trigger_threshold", 0.1)),
        "history_window": _int_param(params, "history_window", 5),
        "duration_quantile": float(params.get("duration_quantile", 0.8)),
        "exit_wait_quantile": float(params.get("exit_wait_quantile", 0.5)),
        "min_history_events": _int_param(params, "min_history_events", 3),
        "default_exit_wait_days": _int_param(params, "default_exit_wait_days", 5),
        "max_weight": float(params.get("max_weight", 1.0)),
    }
    if params.get("state_enabled") is not None:
        factor_kwargs["signal_weight"]["state"] = {
            "enabled": bool(params.get("state_enabled")),
            "mode": params.get("state_mode", "level"),
            "entry_threshold": params.get("state_entry_threshold"),
            "exit_threshold": params.get("state_exit_threshold"),
            "min_hold_days": _int_param(params, "state_min_hold_days", 0),
            "max_hold_days": params.get("state_max_hold_days"),
            "cooldown_days": _int_param(params, "state_cooldown_days", 0),
            "stop_loss": params.get("state_stop_loss"),
            "take_profit": params.get("state_take_profit"),
        }

    factor_spec = FactorSpec(
        name="XICHOU_timing",
        inputs={"low_wide": "最低价(元)", "high_wide": "最高价(元)"},
        compute=compute_xichou_timing_mining,
        direction="positive",
        compute_kwargs=factor_kwargs,
        table_name="daily_index",
        index_code=None,
        use_industry=False,
        use_mktcap=False,
        industry_scheme="申万一级行业",
        backtest_metric="收盘价(元)",
        weight_mode="freeplay",
        long_groups=None,
        short_groups=None,
    )
    # FactorSpec in the TDX template predates strategy_type; attach it
    # dynamically so the shared mining executor selects the timing branch.
    setattr(factor_spec, "strategy_type", "timing")
    return MiningSpec(
        factor_spec=factor_spec,
        execution_mode="precomputed",
        warmup_days=mining_warmup_days,
        weight_hook=standard_timing_weight_hook,
    )


def mining_warmup_days(params: Mapping[str, Any]) -> int:
    windows = [
        _int_param(params, "sma_n", 3),
        _int_param(params, "ema_n", 3),
        _int_param(params, "llv_n", 15),
        _int_param(params, "signal_window", 120),
        _int_param(params, "history_window", 5) * 2,
    ]
    return max(90, max(windows) * 3 + 60)


__all__ = [
    "compute_xichou_timing_mining",
    "make_mining_spec",
    "mining_warmup_days",
]
