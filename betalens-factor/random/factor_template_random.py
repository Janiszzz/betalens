"""Shared pipeline exports for random benchmark factors."""
from __future__ import annotations

import sys
from pathlib import Path

_FACTOR_ROOT = Path(__file__).resolve().parent.parent
if str(_FACTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_FACTOR_ROOT))

from factor_template import FactorPipeline, FactorSpec, RunResult  # noqa: E402

__all__ = ["FactorPipeline", "FactorSpec", "RunResult"]
