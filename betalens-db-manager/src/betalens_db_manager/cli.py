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
    monitor = commands.add_parser('monitor',help='独立只读任务监控，不启动 worker 或初始化状态')
    monitor.add_argument('--port',type=int,default=8011)
    commands.add_parser("doctor", help="检查配置、数据库及磁盘")
    commands.add_parser('migration-preview',help='只读盘点两库物理类型、可空性与待迁移字段，不执行DDL')
    repair=commands.add_parser('mapping-repair',help='预览或发布逐列生产规则和已核对的映射修复；不执行DDL或补写')
    repair.add_argument('action',choices=['preview','apply'])
    repair.add_argument('--plan',type=Path,help='预览输出文件或要确认的计划文件')
    repair.add_argument('--token',help='预览内容令牌；发布必须提供')
    commands.add_parser('historical-repair-preview',help='只读生成历史错误值及待迁移列的修正建议')
    credential_scan=commands.add_parser("credentials-audit", help="只读盘点当前配置和任务中的旧凭据位置及迁移建议")
    credential_scan.add_argument('--max-values',type=int,default=2000000)
    credential_scan.add_argument('--max-seconds',type=float,default=60)
    schema = commands.add_parser("schema", help="显式结构计划或建库建表")
    schema.add_argument("action", choices=["plan", "apply"])
    schema.add_argument("--token", help="schema plan 输出的结构计划令牌")
    plan = commands.add_parser("plan", help="创建数据导入计划，不自动下载")
    plan.add_argument("request", type=Path, help="JSON/YAML 请求文件")
    for name in ["download", "preview", "commit", "resume", "cancel", "job", "sync-audit", "run"]:
        command = commands.add_parser(name)
        command.add_argument("job")
        if name == "commit": command.add_argument("--token", required=True)
    validator=commands.add_parser('validate');validator.add_argument('job');validator.add_argument('--rules',nargs='+',choices=['values','relations','readback'],default=['values','relations'])
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
    if args.command == "credentials-audit":
        # The normal service constructor seeds YAML and SQLite ledger records.
        # This command must not initialize it or print configuration exceptions,
        # which can themselves contain credential values.
        try:
            from .credential_audit import audit_credentials
            print(dumps(audit_credentials(Settings.from_env(args.env),max_values=args.max_values,max_seconds=args.max_seconds)))
            return 0
        except Exception:
            print(dumps({"error": "凭据审计初始化失败；请检查配置路径和权限"}), file=sys.stderr)
            return 1
    settings = Settings.from_env(args.env)
    if args.command=='monitor':
        import uvicorn
        from .monitor import create_monitor_app
        uvicorn.run(create_monitor_app(settings),host='127.0.0.1',port=args.port,log_level='warning')
        return 0
    service = None
    try:
        if args.command=='worker':
            from .worker import retry_control
            service=retry_control('initialize',lambda:DatabaseManagerService(settings),settings)
        else:service=DatabaseManagerService(settings)
        result = None
        if args.command == "doctor": result = service.doctor()
        elif args.command == 'migration-preview':
            from .deployment_preview import migration_preview
            result=migration_preview(service)
        elif args.command=='mapping-repair':
            from .config_migration import preview,apply
            if args.action=='preview':
                result=preview(service)
                if args.plan:
                    from .private import atomic_private
                    atomic_private(args.plan,dumps(result))
                    result={k:v for k,v in result.items() if k!='files'}|{'files':[{'path':v['path'],'changes':v['changes']} for v in result['files']],'plan':str(args.plan)}
            else:
                if not args.plan or not args.token:raise ValueError('发布需要--plan和--token')
                result=apply(service,json.loads(args.plan.read_text()),args.token)
        elif args.command=='historical-repair-preview':
            from .deployment_preview import historical_repair_preview
            result=historical_repair_preview(service)
        elif args.command == "schema":
            result = service.schema_plan() if args.action == "plan" else service.initialize(args.token or "")
        elif args.command == "plan":
            import yaml
            result = service.plan(yaml.safe_load(args.request.read_text(encoding="utf-8")))
        elif args.command == "download": result = {"task":service.state.enqueue(args.job,"download",{"owner":"cli"})}
        elif args.command == "preview": result = service.preview(args.job)
        elif args.command == "commit": result = {"task":service.state.enqueue(args.job,"commit",{"token":args.token,"owner":"cli"})}
        elif args.command=='run':
            item=service._job(args.job)
            if item['plan'].get('pipeline_version')!=2:raise ValueError('run 仅适用于新管线任务')
            result={'task':service.state.enqueue(args.job,'pipeline_run',{'owner':'cli','background':True})}
        elif args.command=='validate':result={'task':service.state.enqueue(args.job,'pipeline_validate',{'owner':'cli','background':True,'validation':{'rules':args.rules}})}
        elif args.command == "resume": result = {"task":service.state.enqueue(args.job,"resume",{"owner":"cli"})}
        elif args.command == "sync-audit":result={'task':service.state.enqueue(args.job,'sync_audit',{'owner':'cli'})}
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
            supervisor = None
            if not args.no_worker:
                from .worker_supervisor import WorkerSupervisor
                supervisor=WorkerSupervisor(settings);supervisor.start()
            # Uvicorn replays SIGTERM after shutdown. Convert that replay to a
            # Python exit so our finally block also stops the owned worker.
            previous_term = signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
            try: uvicorn.run(create_app(service), host="127.0.0.1", port=args.port, log_level="warning", timeout_graceful_shutdown=10)
            finally:
                if supervisor:supervisor.close()
                signal.signal(signal.SIGTERM, previous_term)
        if result is not None:
            from .security import public_data
            print(dumps(public_data(result,service.settings,service.sources.values.values())))
        return 0
    except Exception as exc:
        from .security import safe_error
        from .util import now
        print(dumps({"at":now(),"command":args.command,"exception":type(exc).__name__,"error":service.safe_error(exc) if service else safe_error(exc,settings)}), file=sys.stderr)
        return 1

