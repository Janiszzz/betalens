#!/usr/bin/env python3
"""Create or verify the Betalens PostgreSQL schema without interactive input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from betalens_db_manager.constants import ALLOWED_TABLES
from betalens_db_manager.schema import SchemaManager
from betalens_db_manager.utils import json_default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="创建或验证 Betalens datafeed 数据库结构")
    parser.add_argument("--verify-only", action="store_true", help="仅验证 schema")
    parser.add_argument("--create-database", action="store_true", help="数据库不存在时创建数据库")
    parser.add_argument("--force", action="store_true", help="删除并重建指定表，必须同时传 --yes")
    parser.add_argument("--yes", action="store_true", help="确认执行 --force 这类破坏性操作")
    parser.add_argument("--no-indexes", action="store_true", help="不创建索引")
    parser.add_argument("--no-comments", action="store_true", help="不添加表和列注释")
    parser.add_argument("--tables", nargs="+", choices=ALLOWED_TABLES, help="仅处理指定表")
    args = parser.parse_args(argv)

    if args.force and not args.yes:
        parser.error("--force 会删除表，必须显式传入 --yes")

    manager = SchemaManager()
    if args.verify_only:
        print(json.dumps(manager.verify_schema(), ensure_ascii=False, default=json_default, indent=2))
        return 0

    result = manager.ensure_schema(
        tables=args.tables,
        force=args.force,
        create_database_if_missing=args.create_database,
        create_indexes=not args.no_indexes,
        create_comments=not args.no_comments,
    )
    print(json.dumps(result, ensure_ascii=False, default=json_default, indent=2))
    print(json.dumps(manager.verify_schema(), ensure_ascii=False, default=json_default, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
