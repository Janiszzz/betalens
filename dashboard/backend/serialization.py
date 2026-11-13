"""Compatibility alias for the shared live/archive presentation module.

Keep changes in betalens.analyst.presentation so both dashboards stay aligned.
The module alias also preserves existing monkeypatch/customization behavior.
"""
import sys

from betalens.analyst import presentation as _presentation

sys.modules[__name__] = _presentation
