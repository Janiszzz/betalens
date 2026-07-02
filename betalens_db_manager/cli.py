"""Command line helpers for database-manager scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .constants import ALLOWED_TABLES, IMPORT_MODES, IMPORT_TYPES, INSERT_ONLY
from .jobs import ImportJobRunner
from .utils import json_default


DATA_SUFFIXES = {".csv", ".xls", ".xlsx"}


def iter_data_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for file in sorted(path.iterdir()):
        if file.is_file() and file.suffix.lower() in DATA_SUFFIXES:
            yield file


def run_file_import_cli(
    argv: list[str] | None = None,
    default_import_type: str = "auto",
    default_table: str = "daily_market",
) -> int:
    parser = argparse.ArgumentParser(description="导入本地文件到 Betalens 数据库")
    parser.add_argument("path", nargs="?", default=".", help="文件或目录路径，默认当前目录")
    parser.add_argument("--table", default=default_table, choices=ALLOWED_TABLES, help="目标表")
    parser.add_argument("--import-type", default=default_import_type, choices=["auto", *IMPORT_TYPES], help="导入类型")
    parser.add_argument("--mode", default=INSERT_ONLY, choices=IMPORT_MODES, help="写入模式")
    parser.add_argument("--dry-run", action="store_true", help="只预览和校验，不写入数据库")
    parser.add_argument("--allow-unsafe-metrics", action="store_true", help="允许 Unnamed:* 指标")
    parser.add_argument("--allow-nan-values", action="store_true", help="允许 NaN/Inf value")
    parser.add_argument("--sheet", help="Excel 工作表名，用于指数成分或交易状态导入")
    parser.add_argument("--index-code", default="000906.SH", help="指数成分导入的指数代码")
    parser.add_argument("--index-name", default="中证800", help="指数成分导入的指数名称")
    parser.add_argument("--date-from", default="filename", help="EDE 日期来源，默认从文件名提取")
    parser.add_argument("--default-datetime", help="EDE 文件名无日期时使用的默认 datetime")
    args = parser.parse_args(argv)

    source = Path(args.path).expanduser().resolve()
    if not source.exists():
        parser.error(f"路径不存在: {source}")

    files = list(iter_data_files(source))
    if not files:
        parser.error(f"没有找到可导入文件: {source}")

    runner = ImportJobRunner()
    options = {
        "date_from": args.date_from,
        "default_datetime": args.default_datetime,
        "index_code": args.index_code,
        "index_name": args.index_name,
    }
    if args.sheet:
        options["sheet_name"] = args.sheet
    options = {key: value for key, value in options.items() if value is not None}
    failures = 0
    for file in files:
        import_type = None if args.import_type == "auto" else args.import_type
        print(f"\n==> {file}")
        if args.dry_run:
            result = runner.preview(file, import_type, options=options)
            print(json.dumps(result, ensure_ascii=False, default=json_default, indent=2))
            if not result.get("validation", {}).get("ok", False):
                failures += 1
            continue

        record = runner.run(
            file,
            table=args.table,
            import_type=import_type,
            mode=args.mode,
            options=options,
            allow_unsafe_metrics=args.allow_unsafe_metrics,
            allow_nan_values=args.allow_nan_values,
        )
        print(json.dumps(record, ensure_ascii=False, default=json_default, indent=2))
        if record.get("status") != "completed":
            failures += 1

    return 1 if failures else 0
