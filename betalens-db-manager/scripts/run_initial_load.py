"""One-off orchestration; all data work is delegated to the installed component."""
import argparse
import yaml
from betalens_db_manager.config import Settings
from betalens_db_manager.manager import DatabaseManagerService
from betalens_db_manager.util import dumps
parser=argparse.ArgumentParser(description='登记初始导入；拉取后需在网页审计确认')
parser.add_argument('request');parser.add_argument('--env',required=True)
args=parser.parse_args()
service=DatabaseManagerService(Settings.from_env(args.env))
with open(args.request,encoding='utf-8') as handle:request=yaml.safe_load(handle)
job=service.plan(request)
operation=service.state.enqueue(job['id'],'download',{'owner':'cli'})
print(dumps({'job':job['id'],'operation':operation,'confirmation_required':True}))
