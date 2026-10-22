from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import Settings
from .manager import DatabaseManagerService
from .util import dumps


def main(argv=None):
    parser = argparse.ArgumentParser(description="Betalens DB Manager — 独立数据接入与维护工具")
    parser.add_argument("--env", type=Path, help="私有环境文件；不依赖研究框架配置")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="检查配置、数据库及磁盘")
    schema = commands.add_parser("schema", help="显式结构计划或建库建表")
    schema.add_argument("action", choices=["plan", "apply"])
    schema.add_argument("--token", help="schema plan 输出的结构计划令牌")
    plan = commands.add_parser("plan", help="创建数据导入计划，不自动下载")
    plan.add_argument("request", type=Path, help="JSON/YAML 请求文件")
    for name in ["download", "preview", "commit", "resume", "cancel", "job"]:
        command = commands.add_parser(name)
        command.add_argument("job")
        if name == "commit": command.add_argument("--token", required=True)
    commands.add_parser("jobs")
    probe = commands.add_parser("probe"); probe.add_argument("datasets", nargs="+")
    query = commands.add_parser("query"); query.add_argument("dataset")
    query.add_argument("--code"); query.add_argument("--start"); query.add_argument("--end"); query.add_argument("--after")
    query.add_argument("--limit", type=int, default=100)
    query.add_argument("--target",choices=["postgres","clickhouse"],default="postgres")
    file = commands.add_parser("import-file")
    file.add_argument("dataset"); file.add_argument("path", type=Path)
    file.add_argument("--column-map", type=Path); file.add_argument("--encoding", default="utf-8-sig")
    file.add_argument("--mode", choices=["fill_missing", "refresh_source"], default="fill_missing")
    rollback=commands.add_parser('rollback',help='按任务回退到 before 值')
    rollback.add_argument('job');rollback.add_argument('--targets',nargs='+',choices=['postgres','clickhouse'])
    rebuild = commands.add_parser("rebuild", help="预览或确认删除旧表并重建统一 canonical 结构")
    rebuild.add_argument("--dry-run", action="store_true", help="只输出盘点，不执行删除（默认行为）")
    rebuild.add_argument("--confirm-drop", action="store_true", help="确认删除目标库旧 schema/table")
    load = commands.add_parser("load", help="执行 initial_load YAML 的下载、预览和导入")
    load.add_argument("--config", type=Path, required=True)
    services = commands.add_parser("services", help="通过独立模板建立或检查本地数据库容器")
    services.add_argument("action", choices=["up", "status"])
    worker = commands.add_parser("worker"); worker.add_argument("--once", action="store_true")
    serve = commands.add_parser("serve"); serve.add_argument("--port", type=int, default=8010)
    serve.add_argument("--no-worker", action="store_true", help="外部已管理 worker 时使用")
    args = parser.parse_args(argv)
    settings = Settings.from_env(args.env)
    service = DatabaseManagerService(settings)
    try:
        result = None
        if args.command == "doctor": result = service.doctor()
        elif args.command == "schema":
            result = service.schema_plan() if args.action == "plan" else service.initialize(args.token or "")
        elif args.command == "plan":
            import yaml
            result = service.plan(yaml.safe_load(args.request.read_text(encoding="utf-8")))
        elif args.command == "download": result = {"task":service.state.enqueue(args.job,"download",{"owner":"cli"})}
        elif args.command == "preview": result = service.preview(args.job)
        elif args.command == "commit": result = {"task":service.state.enqueue(args.job,"commit",{"token":args.token,"owner":"cli"})}
        elif args.command == "resume": result = {"task":service.state.enqueue(args.job,"resume",{"owner":"cli"})}
        elif args.command == "cancel": service.state.cancel(args.job); result = {"cancel_requested": True}
        elif args.command == "job": result = service._job(args.job)
        elif args.command == "jobs": result = service.state.jobs()
        elif args.command == "probe": result = service.probe(args.datasets)
        elif args.command == "query": result = (service.pg if args.target=='postgres' else service.ch).query(args.dataset,args.code,args.start,args.end,args.limit,args.after)
        elif args.command=='rollback':result={'task':service.state.enqueue(args.job,'rollback',{'targets':args.targets,'owner':'cli'})}
        elif args.command == "rebuild":
            from .migration import rebuild as rebuild_databases
            result = rebuild_databases(service, args.confirm_drop)
        elif args.command == "load":
            import yaml
            from .migration import initial_load,load_manifest
            config=yaml.safe_load(args.config.read_text(encoding="utf-8"))
            result=load_manifest(service,config) if "batches" in config else initial_load(service,config)
        elif args.command == "services":
            template = Path(__file__).parent / "templates/compose.yaml"
            environment = os.environ.copy()
            environment.update(PGUSER=settings.pg_user, PGPASSWORD=settings.pg_password, PGDATABASE=settings.pg_database,
                               PGPORT=str(settings.pg_port), CLICKHOUSE_DB=settings.ch_database,
                               CLICKHOUSE_USER=settings.ch_user, CLICKHOUSE_PASSWORD=settings.ch_password)
            command = ["docker", "compose", "--project-name", "betalens-infra", "-f", str(template)]
            command.extend(["up", "-d", "--wait"] if args.action == "up" else ["ps"])
            return subprocess.run(command, env=environment).returncode
        elif args.command == "import-file":
            from .files import import_file
            mapping = json.loads(args.column_map.read_text()) if args.column_map else None
            result = import_file(service, args.dataset, args.path, mapping, args.mode, args.encoding,defer=True)
        elif args.command == "worker":
            from .worker import work
            work(service, args.once)
        elif args.command == "serve":
            import uvicorn
            import signal
            from .api import create_app
            child = None
            if not args.no_worker:
                command = [sys.executable, "-m", "betalens_db_manager"]
                if args.env: command.extend(["--env", str(args.env.resolve())])
                command.append("worker")
                log = (settings.state_dir / "worker.log").open("a")
                child = subprocess.Popen(command, stdout=log, stderr=log)
                log.close()
            # Uvicorn replays SIGTERM after shutdown. Convert that replay to a
            # Python exit so our finally block also stops the owned worker.
            previous_term = signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
            try: uvicorn.run(create_app(service), host="127.0.0.1", port=args.port, log_level="warning", timeout_graceful_shutdown=10)
            finally:
                if child and child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=70)
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait(timeout=5)
                signal.signal(signal.SIGTERM, previous_term)
        if result is not None: print(dumps(result))
        return 0
    except Exception as exc:
        print(dumps({"error": service.safe_error(exc)}), file=sys.stderr)
        return 1

