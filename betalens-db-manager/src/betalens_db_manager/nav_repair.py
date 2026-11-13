"""Build a guarded intent for explicitly approved, exact NAV replacements."""
from .comparison import typed, row_hash, row_key
from .nav_semantics import nav_publication_decision, NavPublicationConflict
from .util import digest


def approved_nav_replacement(spec, pairs, *, ingested_at):
    """Keep complete old/new images; never relax ordinary publication rules."""
    if spec.name!='fund_nav_daily':raise ValueError('仅支持基金日净值的明确版本修订')
    changes=[];checks=[];seen=set()
    for pair in pairs:
        before=typed(spec,pair['before']);after=typed(spec,pair['incoming'])
        key=row_key(spec,before)
        if key!=row_key(spec,after):raise ValueError('净值修订不能改变业务主键')
        if key in seen:raise ValueError('净值修订包含重复主键')
        seen.add(key)
        if before.get('announcement_date') is None or before['announcement_date']!=after.get('announcement_date'):
            raise ValueError('净值修订必须明确对应同一已知公告日期')
        try:nav_publication_decision(before,after)
        except NavPublicationConflict:pass
        else:raise ValueError('该净值没有需要明确选择的同公告日版本冲突')
        if 'ingested_at' in after:after['ingested_at']=ingested_at
        before_hash=row_hash(spec,before);after_hash=row_hash(spec,after)
        checks.append({'key':{name:before[name] for name in spec.keys},'row_key':key,'before_hash':before_hash})
        changes.append({'before':before,'after':after,'row_key':key,
                        'before_hash':before_hash,'after_hash':after_hash})
    if not changes:raise ValueError('净值修订不能为空')
    return {'changes':changes,'checks':checks,'conflicts':[],
            'counts':{'inserted':0,'updated':len(changes),'skipped':0,'conflicts':len(changes)},
            'signature':digest({'snapshots':[[c['row_key'],c['before_hash']] for c in checks],
                                'changes':[[c['row_key'],c['before_hash'],c['after_hash']] for c in changes]})}
