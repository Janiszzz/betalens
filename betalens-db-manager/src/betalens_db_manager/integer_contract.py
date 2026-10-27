"""The common integer contract for canonical business fields."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
JS_SAFE_INTEGER = (1 << 53) - 1


def validate_int64(value, name: str = "integer") -> int:
    """Return an exact signed Int64 or reject values that would lose information."""
    if isinstance(value, bool):
        raise ValueError(f"{name}: 需要整数")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"{name}: 需要整数") from None
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{name}: 需要有限整数")
    if number < INT64_MIN or number > INT64_MAX:
        raise ValueError(f"{name}: 超出有符号 64 位整数范围")
    if isinstance(value, float) and abs(number) > JS_SAFE_INTEGER:
        raise ValueError(f"{name}: 大整数请使用字符串以避免精度损失")
    return int(number)
