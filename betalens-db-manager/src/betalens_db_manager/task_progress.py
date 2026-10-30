"""Immutable work denominators; pagination is transport work below a logical root."""
from .downloader import TaskDownloader
from .util import digest,now,dumps


def freeze(service,item,groups,downloader):
    roots=list(dict.fromkeys(TaskDownloader.key(downloader.prepared(r)[0]) for cohort in groups.values() for r in cohort))
    buckets=list(groups)
    value={'task_id':item['id'],'frozen_at':now(),'basis':'logical_requests','request_ids':roots,'batch_ids':buckets,
           'requests_total':len(roots),'batches_total':len(buckets),'version':digest([roots,buckets])}
    previous=service.state.ledger_get('pipeline_work_plan',item['id'])
    if previous:
        if previous['basis']!='historical_archives' and previous['version']!=value['version']:raise ValueError('冻结的任务总量与本次执行范围不同，请建立关联任务')
        value=previous
    else:service.state.ledger_put('pipeline_work_plan',item['id'],value)
    records=[(key,{'roots':[TaskDownloader.key(downloader.prepared(r)[0]) for r in cohort]}) for key,cohort in groups.items()]
    for offset in range(0,len(records),128):
        with service.state.connect() as c:
            c.executemany('INSERT OR IGNORE INTO pipeline_units(job,id,kind,status,payload,created,updated) VALUES(?,?,\'normalize\',\'pending\',?,?,?)',
                          [(item['id'],key,dumps(payload),now(),now()) for key,payload in records[offset:offset+128]])
    return value


def summary(service,item,units):
    job=item['id'];plan=service.state.ledger_get('pipeline_work_plan',job)
    if plan is None and item['status'] in {'completed','partial','rolled_back','partially_rolled_back'} and not any(t['operation']=='pipeline_run' and t['status'] in {'queued','running'} for t in item['tasks']):
        # Old finished tasks have no expansion journal. Their final archived work
        # is an exact, immutable denominator, explicitly labelled as historical.
        roots=[p['id'] for p in item['slices']];buckets=[u['id'] for u in units if u['kind']=='normalize']
        plan={'task_id':job,'frozen_at':now(),'basis':'historical_archives','request_ids':roots,'batch_ids':buckets,
              'requests_total':len(roots),'batches_total':len(buckets),'version':digest([roots,buckets])}
        service.state.ledger_put('pipeline_work_plan',job,plan)
    if plan is None:return {'phase':'planning','requests':None,'batches':None,'physical_requests':len(item['slices'])}
    by={p['id']:p for p in item['slices']};memo={};visiting=set()
    def done(key):
        if key in memo:return memo[key]
        part=by.get(key)
        if not part or part['status']=='pending':return False,False
        if key in visiting:return True,True
        if part['status'] in {'failed','unavailable','contract_blocked'}:return True,True
        visiting.add(key);children=(part.get('result') or {}).get('children',[])
        states=[done(TaskDownloader.key(c)) for c in children]
        visiting.discard(key)
        value=(all(s[0] for s in states),any(s[1] for s in states))
        if part['status'] in {'paged','split'} and not children:value=(False,False)
        memo[key]=value;return value
    states=[done(k) for k in plan['request_ids']]
    if plan['basis']=='historical_archives':states=[(p['status']!='pending',p['status'] in {'failed','unavailable','contract_blocked'}) for p in (by[k] for k in plan['request_ids'])]
    normal={u['id']:u for u in units if u['kind']=='normalize'};selected=[normal.get(k,{'status':'pending'}) for k in plan['batch_ids']]
    stored={(u['payload'].get('child_id'),u['payload'].get('target')) for u in units if u['kind']=='stored' and u['status']=='completed'}
    rejected=sum(u['status']=='failed' for u in selected)
    writes={}
    for target in item['plan']['request']['targets']:
        written=sum(((u.get('result') or {}).get('child_id'),target) in stored for u in selected)
        writes[target]={'total':plan['batches_total'],'processed':written+rejected,'written':written,'rejected':rejected}
    root_ids=set(plan['request_ids'])
    return {'phase':'frozen','basis':plan['basis'],'frozen_at':plan['frozen_at'],'version':plan['version'],
            'requests':{'total':plan['requests_total'],'processed':sum(s[0] for s in states),'failed':sum(s[0] and s[1] for s in states)},
            'batches':{'total':plan['batches_total'],'processed':sum(u['status'] in {'completed','failed'} for u in selected),
                       'passed':sum(u['status']=='completed' for u in selected),'rejected':sum(u['status']=='failed' for u in selected)},
            'writes':writes,'physical_requests':len(item['slices']),'additional_requests':sum(p['id'] not in root_ids for p in item['slices'])}
