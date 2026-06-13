"""通达信(TDX)因子页 — 薄封装，逻辑见 betalens-factor/factor_page.py"""
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_DIR.parent))

from factor_page import render_factor_class  # noqa: E402


def render():
    render_factor_class(_DIR, "tdx")
