"""CLI and desktop entry point for Betalens Database Manager."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .manager import DatabaseManager
from .schema import SchemaManager
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


def _add_schema_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-create-database",
        action="store_true",
        help="目标数据库不存在时失败",
    )
    parser.add_argument("--no-migrate-legacy", action="store_true", help="只安装基础 schema")
    parser.add_argument("--no-compat-views", action="store_true", help="暂不切换 public 兼容视图")
    parser.add_argument(
        "--year",
        dest="observation_years",
        type=int,
        action="append",
        help="额外创建 observation 年度分区，可重复",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m betalens_db_manager")
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="只读预览 schema 与可选 Manifest")
    _add_database_arguments(plan_parser)
    _add_schema_options(plan_parser)
    plan_parser.add_argument("--manifest", type=Path, help="同时预检批量导入清单")

    init_parser = subparsers.add_parser("init", help="初始化或升级本地数据库")
    _add_database_arguments(init_parser)
    _add_schema_options(init_parser)
    init_parser.add_argument("--yes", action="store_true", help="确认 schema 变更及 upsert")
    init_parser.add_argument("--manifest", type=Path, help="schema 完成后执行导入清单")
    init_parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    init_parser.add_argument("--no-resume", dest="resume", action="store_false")
    init_parser.add_argument("--report", type=Path, help="统一 JSON 运行报告路径")

    import_parser = subparsers.add_parser("import", help="预检并执行批量导入清单")
    _add_database_arguments(import_parser)
    import_parser.add_argument("--manifest", type=Path, required=True, help="version: 1 清单路径")
    import_parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    import_parser.add_argument("--no-resume", dest="resume", action="store_false")
    import_parser.add_argument("--yes", action="store_true", help="确认 upsert 导入")
    import_parser.add_argument("--report", type=Path, help="JSON 导入报告路径")

    verify_parser = subparsers.add_parser("verify", help="严格校验本地数据库 schema")
    _add_database_arguments(verify_parser)
    verify_parser.add_argument("--deep", action="store_true", help="执行完整语义及覆盖核验")
    verify_parser.add_argument("--no-compat-views", action="store_true")
    verify_parser.add_argument("--report", type=Path, help="JSON 核验报告路径")
    return parser


def _target_version(args: argparse.Namespace) -> int | None:
    if getattr(args, "no_migrate_legacy", False):
        return 5
    if getattr(args, "no_compat_views", False):
        return 6
    return None


def _make_manager(args: argparse.Namespace) -> DatabaseManager:
    return DatabaseManager(
        _database_overrides(args),
        profile=getattr(args, "profile", None),
    )


def _json_print(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))


def _confirm_schema(plan: Mapping[str, Any]) -> bool:
    print(f"数据库: {plan['database']}")
    print(f"当前版本: {plan['current_version']} -> 目标版本: {plan['target_version']}")
    pending = plan.get("pending") or []
    if pending:
        print("待执行 migration:")
        for item in pending:
            print(f"  {item['version']:04d}_{item['name']}")
    else:
        print("schema 已是目标版本；仍会补齐分区并重新核验。")
    try:
        answer = input("继续? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


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
    plan = manager.plan(
        target_version=_target_version(args),
        observation_years=args.observation_years,
        manifest=args.manifest,
        create_compat_views=not args.no_compat_views and not args.no_migrate_legacy,
    )
    _json_print(plan)
    return 0


def _run_init(args: argparse.Namespace) -> int:
    manager = _make_manager(args)
    plan = manager.plan(
        target_version=_target_version(args),
        observation_years=args.observation_years,
        manifest=args.manifest,
        create_compat_views=not args.no_compat_views and not args.no_migrate_legacy,
    )
    if not args.yes and not _confirm_schema(plan):
        print("已取消。")
        return 2
    if _has_upsert(plan.get("manifest")) and not args.yes and not _confirm_upsert():
        print("已取消。")
        return 2
    report = manager.bootstrap(
        create_database_if_missing=not args.no_create_database,
        migrate_legacy=not args.no_migrate_legacy,
        create_compat_views=not args.no_compat_views and not args.no_migrate_legacy,
        verify=True,
        observation_years=args.observation_years,
        report_path=args.report,
        manifest=plan.get("manifest"),
        resume=args.resume,
    )
    _json_print(report)
    return 0 if report.get("status") == "completed" else 1


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
    manager = _make_manager(args)
    report = manager.verify(
        deep=args.deep,
        require_compat_views=not args.no_compat_views,
        report_path=args.report,
    )
    _json_print(report)
    return 0 if report.get("status") == "completed" else 1


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
