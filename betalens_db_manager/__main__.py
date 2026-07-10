"""Run the Betalens Database Manager desktop GUI."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    try:
        from .gui import main as gui_main
    except RuntimeError as exc:
        print(f"无法启动 Betalens Database Manager: {exc}", file=sys.stderr)
        return 1
    return gui_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
