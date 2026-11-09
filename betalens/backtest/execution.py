"""Stock-order quantity constraints; amounts are never rounded into shares."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ShareRule:
    minimum: int
    increment: int

    def truncate(self, quantity: float) -> int:
        quantity = math.floor(abs(quantity) + 1e-9)
        if quantity < self.minimum:
            return 0
        return self.minimum + ((quantity - self.minimum) // self.increment) * self.increment


def share_rule(market: str) -> ShareRule:
    if market in {"主板", "创业板"}:
        return ShareRule(100, 100)
    if market == "科创板":
        return ShareRule(200, 1)
    raise ValueError(f"未支持或缺失的股票板块: {market!r}；目前支持主板、创业板、科创板")


def round_order(delta: float, holding: float, rule: ShareRule) -> tuple[int, float]:
    """Return signed integer trade and signed synthetic fractional settlement.

    Closing odd shares is allowed in one piece.  Less than one equivalent share
    can only be redeemed on full closure, separately from an exchange order.
    Increasing/opening an exposure always obeys the board minimum/increment.
    """
    if abs(delta) < 1e-9:
        return 0, 0.0
    sign = 1 if delta > 0 else -1
    reducing = holding * delta < 0
    if reducing and abs(delta) >= abs(holding) - 1e-8:
        whole = math.floor(abs(holding) + 1e-9)
        fraction = max(0.0, abs(holding) - whole)
        opening = max(0.0, abs(delta) - abs(holding))
        return sign * (whole + rule.truncate(opening)), sign * fraction
    if reducing and rule.increment > 1:
        # Existing odd shares may be sold once, alone or together with whole lots.
        odd = math.floor(abs(holding) + 1e-9) % rule.increment
        limit = math.floor(abs(delta) + 1e-9)
        normal = rule.truncate(delta)
        with_odd = odd + ((limit - odd) // rule.increment) * rule.increment if odd and limit >= odd else 0
        return sign * max(normal, with_odd), 0.0
    return sign * rule.truncate(delta), 0.0
