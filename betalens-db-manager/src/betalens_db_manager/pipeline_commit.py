"""Coalesce physical writes while preserving every original durable receipt."""
from datetime import datetime
from time import monotonic
import pyarrow as pa
from .canonical import registry_context,catalog_snapshot
from .comparison import compare,row_key,typed
from .commit_protocol import manifest_rows
from .pipeline_cache import read_rows,save_journal,load_journal,schema
from .util import now,digest


def combined_intent(entries):
    """Only disjoint keys may share an apply; overlap is a FIFO barrier."""
    result={'checks':[],'changes':[]};seen=set()
    for entry in entries:
        comparison=entry['comparison']
        checks=comparison.get('checks') or [
            {'key':c['after'] or c['before'],'row_key':c['row_key'],'before_hash':c['before_hash']}
            for c in comparison['changes']]
        keys={c['row_key'] for c in checks}
        if seen & keys:raise ValueError('重叠键不能合并提交')
        seen.update(keys);result['checks'].extend(checks);result['changes'].extend(comparison['changes'])
    return result


def write_cohorts(pipeline,parent,items,target):
    from .manager import DatabaseManagerService
    from .pipeline import target_lock
    from .pipeline_writer import pg_apply,ch_apply
    from .publishing import existing
    from .schema_contract import compare_structure
    service=DatabaseManagerService(pipeline.s.settings);store=pipeline.store
    db=service.pg if target=='postgres' else service.ch
    with registry_context(service.registry),catalog_snapshot(service.registry),target_lock(service,target):
        request=service._job(parent,True)['plan']['request']
        for child,_ in items:service._job(child,True)
        row_limit=request['commit_batch_rows'];byte_limit=request['commit_batch_bytes']
        names=list(dict.fromkeys(name for _,snapshot in items for name in snapshot['tables']))
        prepared=[];dirty_names=set();last_sync=monotonic();pending_ack=0

        def sync_results(force=False):
            nonlocal last_sync,pending_ack
            if not force and pending_ack<128 and monotonic()-last_sync<request['commit_batch_max_wait_seconds']:return
            try:store.sync(service,parent,target)
            except Exception as exc:service.state.log(parent,'pipeline_audit_sync_pending',target=target,error=service.safe_error(exc))
            last_sync=monotonic();pending_ack=0

        def apply_group(spec,entries):
            nonlocal pending_ack
            if not entries:return
            started=monotonic();timings={}
            for entry in entries:service.state.check_cancel(entry['cancel_job'])
            fresh=[e for e in entries if not e.get('comparison')]
            before=existing(db,spec,[r for e in fresh for r in e['rows']]) if fresh else {}
            timings['lookup_ms']=round((monotonic()-started)*1000);tick=monotonic()
            for entry in fresh:
                identities=[row_key(spec,r) for r in entry['rows']]
                selected={key:before[key] for key in identities if key in before}
                comparison=compare(spec,entry['rows'],selected.values(),request['mode'])
                stamp=datetime.fromisoformat(now())
                for change in comparison['changes']:
                    if 'ingested_at' in change['after']:change['after']['ingested_at']=stamp
                expected={**selected,**{c['row_key']:c['after'] for c in comparison['changes']}}
                journal=save_journal(service.settings.state_dir/'pipeline'/parent/'journals'/target/entry['batch'].replace(':','_'),spec,comparison,list(expected.values()))
                entry['comparison']=comparison;prepared.append((entry['batch'],journal))
            store.prepare_many(parent,target,prepared);prepared.clear()
            # Every original intent and mirror is durable before the first
            # combined business write. Recovery retains the original hashes.
            store.sync(service,parent,target)
            timings['prepare_ms']=round((monotonic()-tick)*1000);tick=monotonic()
            combined=combined_intent(entries)
            try:
                result=pg_apply(db,spec,combined) if target=='postgres' else ch_apply(db,service.state,parent,'group:'+digest([e['batch'] for e in entries]),spec,combined)
                if result['applied']:dirty_names.add(spec.name)
                # The writer reports which original intents were replayed so
                # receipts remain exact even after a crash midway through ACKs.
                applied=result.applied_keys;acks=[]
                for entry in entries:
                    comparison=entry['comparison']
                    own={c['row_key'] for c in comparison['changes']}
                    value={**result,'applied':len(own & applied),'counts':comparison['counts'],
                           'table':spec.name,'child_id':entry['child'],'schema_hash':spec.hash}
                    acks.append((entry['batch'],value,None))
                store.acknowledge_many(parent,target,acks);pending_ack+=len(acks)
            except BaseException as exc:
                store.acknowledge_many(parent,target,[(e['batch'],{},service.safe_error(exc)) for e in entries]);raise
            timings['apply_ms']=round((monotonic()-tick)*1000)
            service.state.log(parent,'pipeline_write_group_completed',target=target,table=spec.name,
                              batches=len(entries),rows=sum(len(e['rows']) for e in entries),
                              applied=result['applied'],elapsed_ms=round((monotonic()-started)*1000),**timings)
            sync_results()

        try:
            for name in names:
                spec=service.registry.spec(name);group=[];keys=set();count=0;size=0;checked=False
                for child,snapshot in items:
                    for index,record in enumerate(snapshot['tables'].get(name,[])):
                        batch=child+':'+name+':'+str(index);receipt=store.receipt(parent,batch,target)
                        if receipt and receipt['status']=='completed':continue
                        if not checked:
                            if not compare_structure(spec,service.schema.inspect(name,target),target)['ready']:raise ValueError(name+': 目标物理结构不符合合同')
                            checked=True
                        comparison=load_journal(receipt['manifest'],spec) if receipt else None
                        rows=[typed(spec,r,keys_only=True) for r in manifest_rows(comparison)] if comparison else read_rows(record,spec)
                        identities={tuple(r.get(k) for k in spec.keys) for r in rows}
                        images=[c['after'] or c['before'] for c in comparison['changes']] if comparison else rows
                        used=pa.Table.from_pylist(images,schema=schema(spec)).nbytes
                        if group and (count+len(rows)>row_limit or size+used>byte_limit or keys & identities):
                            apply_group(spec,group);group=[];keys=set();count=0;size=0
                        group.append({'batch':batch,'child':child,'rows':rows,'comparison':comparison,
                                      'cancel_job':snapshot.get('review_operation_job',parent)})
                        keys.update(identities);count+=len(rows);size+=used
                apply_group(spec,group)
        finally:
            if dirty_names:service.state.invalidate_validation_reports(sorted(dirty_names),[target])
            sync_results(force=True)
        for child,_ in items:service.state.log(parent,'pipeline_target_batch_completed',target=target,child_id=child)
    return [{'target':target,'child_id':child,'status':'completed'} for child,_ in items]
