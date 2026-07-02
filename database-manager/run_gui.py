#!/usr/bin/env python3
"""Launch the standalone Betalens Database Manager GUI."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    try:
        from betalens_db_manager.gui import main as gui_main
    except RuntimeError as exc:
        print(exc)
        return 1
    return gui_main(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

