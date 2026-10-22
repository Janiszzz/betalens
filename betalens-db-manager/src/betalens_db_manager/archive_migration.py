"""Create inspectable field-backfill tasks from existing source archives."""
from pathlib import Path
import json
from .util import digest,dumps


def plan_archive_backfill(service,names,targets=None):
    targets=targets or ['postgres','clickhouse']
    if set(targets)-{'postgres','clickhouse'}:raise ValueError('目标库无效')
    mappings=service.mappings.snapshots(names)
    endpoints={b['endpoint'] for m in mappings.values() for b in m['bindings'] if b['source_id']=='tushare'}
    selected={};missing=set(endpoints)
    with service.state.connect() as conn:
        owners=[r[0] for r in conn.execute('SELECT id FROM jobs WHERE target=? ORDER BY created DESC',(service.settings.fingerprint,))]
    for owner in owners:
        item=service.state.job(owner)
        if item['plan'].get('archive_backfill'):continue
        found={}
        for p in item['slices']:
            if p['dataset'] not in missing or p['status'] not in {'downloaded','paged','committed'} or not p.get('archive') or not Path(p['archive']).exists():continue
            meta=Path(p['archive']).with_suffix('.meta.json')
            if meta.exists() and json.loads(meta.read_text()).get('canonical_table'):continue
            found.setdefault(p['dataset'],[]).append((owner,p))
        selected.update(found);missing-=set(found)
        if not missing:break
    if not selected:raise ValueError('没有可复用的源归档；请创建新的 API 拉取任务')
    from .source import read_archive
    from .util import parse_date
    observed=[]
    for endpoint,parts in selected.items():
        field=service.sources.endpoint('tushare',endpoint).get('date_field')
        if not field:continue
        for owner,part in parts:
            data=read_archive(Path(part['archive']),part['sha'])
            if field in data['fields']:
                position=data['fields'].index(field)
                for row in data['items']:
                    if row[position]:
                        try:observed.append(parse_date(row[position]))
                        except ValueError:pass
    scope_start=str(min(observed)) if observed else service.state.job(owners[0])['created'][:10]
    scope_end=str(max(observed)) if observed else scope_start
    plan={'revision':'manager-v1','archive_backfill':True,'parent_task_ids':sorted({o for parts in selected.values() for o,p in parts}),
          'request':{'tables':names,'targets':targets,'sources':['tushare'],'start_date':scope_start,'end_date':scope_end,'codes':[],'parameters':{},'mode':'refresh_source' if 'fundamental_periodic' in names else 'fill_missing'},
          'requests':[],'contracts':{},'source_contracts':{'tushare:'+e:service.sources.endpoint('tushare',e) for e in selected},
          'canonical_contracts':{n:service.registry.spec(n).hash for n in names},'mapping_snapshots':mappings,'mapping_version':digest(mappings),'target':service.settings.fingerprint,
          'warnings':[{'endpoint':e,'note':'未找到归档，需要重新拉取'} for e in sorted(missing)]+([{'table':'fundamental_periodic','note':'旧版公告日期可能使用实际公告日期；本次按源 ann_date / f_ann_date 分列，并审计更正'}] if 'fundamental_periodic' in names else [])}
    job=service.state.create_job(plan,service.settings.fingerprint)
    for endpoint,parts in selected.items():
        for owner,part in parts:
            raw=part['request'];params=raw.get('parameters',raw)
            request={'table':names[0],'source_id':'tushare','endpoint':endpoint,'parameters':params,'reused_from_task':owner}
            key=digest([owner,part['id']])[:24]
            service.state.slice(job['id'],key,endpoint,request,status='downloaded',archive=part['archive'],sha=part['sha'],rows=part['rows'])
    service.state.update_job(job['id'],status='downloaded')
    service.state.log(job['id'],'archive_backfill_planned',source_tasks=plan['parent_task_ids'],endpoints=sorted(selected),missing_endpoints=sorted(missing),confirmation_required=True)
    return service.state.job(job['id'])
