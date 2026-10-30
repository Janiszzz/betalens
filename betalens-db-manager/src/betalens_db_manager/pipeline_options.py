"""Frozen, validated options for explicitly started v2 ingestion."""
from contextlib import contextmanager
from contextvars import ContextVar

LIGHT = ContextVar('pipeline_light', default=False)
DOMAIN_CHECKS = ContextVar('pipeline_domain_checks',default=True)
PLANNING_SEED = ContextVar('pipeline_planning_seed',default=False)
RULES = ('values', 'relations', 'readback')
DEFAULTS = dict(pipeline_version=2, auto_commit=True, run_in_background=True,
                validation_profile='light', post_check_rules=[], download_workers=8,
                normalization_workers=2, commit_batch_rows=50000,
                commit_batch_bytes=64*1024*1024, commit_batch_max_wait_seconds=10,
                memory_budget_bytes=4*1024**3, backlog_high_bytes=20*1024**3,
                backlog_low_bytes=10*1024**3, min_free_bytes=10*1024**3)


def options(request):
    value={k:request.get(k,v) for k,v in DEFAULTS.items()}
    if value['pipeline_version']!=2:raise ValueError('不支持的管线版本')
    for k in ('auto_commit','run_in_background'):
        if not isinstance(value[k],bool):raise ValueError(k+' 必须为布尔值')
    if value['validation_profile'] not in ('light','strict'):raise ValueError('未知校验预设')
    rules=value['post_check_rules']
    if not isinstance(rules,list) or any(r not in RULES for r in rules):raise ValueError('未知事后校验规则')
    value['post_check_rules']=list(dict.fromkeys(rules))
    if value['validation_profile']=='strict':value['post_check_rules']=list(RULES)
    for key,low,high in [('download_workers',1,16),('normalization_workers',1,4),('commit_batch_rows',1,200000),
                          ('commit_batch_bytes',1024,256*1024**2),('commit_batch_max_wait_seconds',1,60),
                          ('memory_budget_bytes',64*1024**2,32*1024**3),('backlog_high_bytes',1024,1024**4),
                          ('backlog_low_bytes',0,1024**4),('min_free_bytes',0,1024**4)]:
        v=value[key]
        if isinstance(v,bool) or not isinstance(v,int) or not low<=v<=high:raise ValueError(key+' 超出允许范围')
    if value['backlog_low_bytes']>=value['backlog_high_bytes']:raise ValueError('积压恢复阈值必须小于暂停阈值')
    return value


@contextmanager
def light_normalization():
    token=LIGHT.set(True)
    try:yield
    finally:LIGHT.reset(token)


@contextmanager
def source_replay():
    """Collect complete lineage without implicitly enabling L1 domain rules."""
    token=DOMAIN_CHECKS.set(False)
    try:yield
    finally:DOMAIN_CHECKS.reset(token)
