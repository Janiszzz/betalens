#!/usr/bin/env python3
"""Import index universe snapshots through Betalens Database Manager."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from betalens_db_manager.cli import run_file_import_cli


if __name__ == "__main__":
    raise SystemExit(run_file_import_cli(default_import_type="index_universe", default_table="index_universe"))
