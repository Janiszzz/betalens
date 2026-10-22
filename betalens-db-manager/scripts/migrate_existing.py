"""Additive, inventoried upgrade. Existing rows and task IDs are never rewritten."""
import argparse,uuid
from pathlib import Path
from betalens_db_manager import DatabaseManagerService,Settings
from betalens_db_manager.canonical import registry_context
from betalens_db_manager.management_api import table_list
from betalens_db_manager.util import atomic_json,now,dumps

parser=argparse.ArgumentParser();parser.add_argument('--env',required=True);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
s=DatabaseManagerService(Settings.from_env(args.env));report={'created_at':now(),'mode':'apply' if args.apply else 'dry_run','before':table_list(s),'plans':[],'results':[]}
with registry_context(s.registry):
    for name in s.registry.catalog():
        plan=s.schema.plan(name)
        if any(r['requires_rebuild'] for r in plan['targets']):raise ValueError(name+' requires explicit destructive migration; no automatic changes performed')
        report['plans'].append(plan)
    report_path=s.settings.state_dir/'reports'/('db-manager-additive-upgrade-'+uuid.uuid4().hex[:12]+'.json')
    atomic_json(report_path,report)
    if args.apply:
        for plan in report['plans']:
            result=s.schema.apply(plan['token']);report['results'].append(result);atomic_json(report_path,report)
            if result['status']!='completed':raise ValueError('migration incomplete; see '+str(report_path))
    report['after']=table_list(s);atomic_json(report_path,report)
atomic_json(s.settings.state_dir/'reports'/'db-manager-latest-upgrade.json',{'report':str(report_path),'created_at':report['created_at']})
print(dumps({'report':str(report_path),'tables':len(report['plans']),'applied':len(report['results'])}))
