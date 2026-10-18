"""CLI and desktop entry point for Betalens Database Manager."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .manager import DatabaseManager
from .schema import SchemaManager, migration_disabled
from .utils import json_default


def _add_database_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", help="已保存的非密码连接 profile")
    parser.add_argument("--database", dest="dbname", help="目标数据库名")
    parser.add_argument("--user", help="PostgreSQL 用户")
    parser.add_argument("--password", help="仅在当前进程使用；建议改用 BETALENS_DB_PASSWORD")
    parser.add_argument("--host", help="PostgreSQL 主机")
    parser.add_argument("--port", help="PostgreSQL 端口")


def _database_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: getattr(args, key)
        for key in ("dbname", "user", "password", "host", "port")
        if getattr(args, key, None) is not None
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m betalens_db_manager")
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="只读检查现有表并预检可选 Manifest")
    _add_database_arguments(plan_parser)
    plan_parser.add_argument("--manifest", type=Path)

    init_parser = subparsers.add_parser("init", help="已停用：不再自动建库或迁移")
    _add_database_arguments(init_parser)
    init_parser.add_argument("--yes", action="store_true")
    init_parser.add_argument("--manifest", type=Path)

    import_parser = subparsers.add_parser("import", help="预检并执行批量导入清单")
    _add_database_arguments(import_parser)
    import_parser.add_argument("--manifest", type=Path, required=True, help="version: 1 清单路径")
    import_parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    import_parser.add_argument("--no-resume", dest="resume", action="store_false")
    import_parser.add_argument("--yes", action="store_true", help="确认 upsert 导入")
    import_parser.add_argument("--report", type=Path, help="JSON 导入报告路径")

    verify_parser = subparsers.add_parser("verify", help="已停用：请改用 inspect")
    _add_database_arguments(verify_parser)
    verify_parser.add_argument("--deep", action="store_true")

    inspect_parser = subparsers.add_parser("inspect", help="只读检查现有表、字段类型和视图查询")
    _add_database_arguments(inspect_parser)
    return parser


def _make_manager(args: argparse.Namespace) -> DatabaseManager:
    return DatabaseManager(
        _database_overrides(args),
        profile=getattr(args, "profile", None),
    )


def _json_print(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))


def _has_upsert(manifest_plan: Mapping[str, Any] | None) -> bool:
    return bool(
        manifest_plan
        and any(entry.get("mode") == "upsert" for entry in manifest_plan.get("entries", []))
    )


def _confirm_upsert() -> bool:
    try:
        answer = input("Manifest 包含 upsert，已有不同值会被更新。确认? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _run_plan(args: argparse.Namespace) -> int:
    manager = _make_manager(args)
    plan = manager.plan(manifest=args.manifest)
    _json_print(plan)
    return 0 if plan.get("ok") else 1


def _run_init(args: argparse.Namespace) -> int:
    migration_disabled()


def _run_import_command(args: argparse.Namespace) -> int:
    manager = _make_manager(args)
    plan = manager.plan_manifest(args.manifest)
    if _has_upsert(plan) and not args.yes and not _confirm_upsert():
        print("已取消。")
        return 2
    report = manager.run_manifest(
        plan,
        resume=args.resume,
        report_path=args.report,
    )
    _json_print(report)
    return 0 if report.get("status") == "completed" else 1


def _run_verify(args: argparse.Namespace) -> int:
    migration_disabled()


def _load_manifest(path: Path, manager: SchemaManager | DatabaseManager | None = None) -> list[dict[str, Any]]:
    """Compatibility helper returning fully expanded, validated entries."""

    facade = (
        manager
        if isinstance(manager, DatabaseManager)
        else DatabaseManager(getattr(manager, "db_config", None))
    )
    return list(facade.plan_manifest(path).get("entries", []))


def _run_manifest(path: Path, manager: SchemaManager | DatabaseManager) -> dict[str, Any]:
    """Compatibility wrapper used by older integration tests and callers."""

    facade = manager if isinstance(manager, DatabaseManager) else DatabaseManager(manager.db_config)
    report = facade.run_manifest(path, resume=False)
    result = dict(report.get("result") or {})
    result.setdefault("status", report.get("status"))
    result.setdefault("path", str(path.expanduser().resolve()))
    result.setdefault("report_path", report.get("report_path"))
    return result


def main(argv: list[str] | None = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    if not actual_argv:
        try:
            from .gui import main as gui_main
        except (ImportError, RuntimeError) as exc:
            print(f"无法启动 Betalens Database Manager: {exc}", file=sys.stderr)
            return 1
        return gui_main([sys.argv[0]])

    parser = _build_parser()
    args = parser.parse_args(actual_argv)
    try:
        if args.command == "inspect":
            result = _make_manager(args).inspect_schema()
            _json_print(result)
            return 0 if result["ok"] else 1
        if args.command == "plan":
            return _run_plan(args)
        if args.command == "init":
            return _run_init(args)
        if args.command == "import":
            return _run_import_command(args)
        if args.command == "verify":
            return _run_verify(args)
        parser.print_help()
        return 2
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
