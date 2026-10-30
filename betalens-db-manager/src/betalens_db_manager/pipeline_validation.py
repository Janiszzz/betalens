"""Read-only, independently queued business and storage checks."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from .target_lanes import target_lanes
from datetime import date
from decimal import Decimal
import uuid
from .canonical import registry_context,catalog_snapshot
from .comparison import row_key,content_hash,row_hash
from .pipeline_cache import read_rows,load_journal
from .pipeline_store import PipelineStore
from .publishing import existing
from .util import digest,now,parse_date
from .pipeline_options import RULES

VERSION='business-checks-2026-10-27.1'


def values(row):
    problems=[]
    if all(row.get(k) is not None for k in ('high','low')) and row['high']<row['low']:problems.append('high_below_low')
    for key in ('open','close'):
        if row.get(key) is not None and row.get('high') is not None and row[key]>row['high']:problems.append(key+'_above_high')
        if row.get(key) is not None and row.get('low') is not None and row[key]<row['low']:problems.append(key+'_below_low')
    for key in ('volume','amount'):
        if row.get(key) is not None and row[key]<0:problems.append('negative_'+key)
    if row.get('adjustment_factor') is not None and row['adjustment_factor']<=0:problems.append('invalid_factor')
    if row.get('valid_to') and row.get('valid_from') and row['valid_to']<row['valid_from']:problems.append('invalid_interval')
    return problems


def relations(row):
    checked=[];unknown=[];problems=[]
    if all(row.get(k) is not None for k in ('raw_close','adjustment_factor','close')):
        expected=row['raw_close']*row['adjustment_factor'];tolerance=max(Decimal('0.000000000001'),abs(expected)*Decimal('0.000001'))
        checked.append('post_adjustment')
        if abs(row['close']-expected)>tolerance:problems.append('post_adjustment_mismatch')
    if row.get('total_assets') is not None:
        # Missing equity/precision is unknown, not a manufactured accounting success.
        equity=next((row[k] for k in ('total_equity','equity_including_minority','total_hldr_eqy_inc_min_int') if row.get(k) is not None),None)
        if row.get('total_liabilities') is None or equity is None:unknown.append('accounting_fields_missing')
        else:
            expected=row['total_liabilities']+equity
            precision=row.get('disclosure_precision')
            if precision is None:unknown.append('accounting_precision_unknown')
            else:
                checked.append('balance_equation')
                if abs(row['total_assets']-expected)>max(Decimal(str(precision)),abs(expected)*Decimal('0.000001')):problems.append('balance_equation_mismatch')
    if row.get('available_at') is not None and row.get('event_date'):
        basis=str(row.get('availability_basis') or '')
        if any(x in basis.lower() for x in ('unknown','assumed','not_proven','download')) or not basis:unknown.append('historical_availability_unproven')
        else:checked.append('availability_evidence')
        event=row.get('announcement_date') or row['event_date']
        from .service import SHANGHAI
        if row['available_at'].astimezone(SHANGHAI).date()<event:problems.append('available_before_event')
    return checked,unknown,problems


class ValidationRunner:
    def __init__(self,service):self.s=service;self.store=PipelineStore(service.state)

    def run(self,job,payload):
        allowed={'rules','targets','tables','codes','start_date','end_date'}
        if set(payload)-allowed:raise ValueError('校验请求包含未知字段')
        rules=payload.get('rules',list(RULES))
        if not isinstance(rules,list) or not rules or set(rules)-set(RULES):raise ValueError('请选择有效校验规则')
        parent=self.s._job(job);request=parent['plan']['request']
        targets=payload.get('targets',request['targets'])
        if not isinstance(targets,list) or not targets or set(targets)-set(request['targets']):raise ValueError('校验目标不属于原任务')
        available_tables=request.get('tables',[])
        if parent['plan']['revision']=='reference-v1':
            from .reference_contracts import MODULE_TABLE
            available_tables=[MODULE_TABLE[m] for m in request['modules']]
        tables=payload.get('tables',available_tables)
        if not isinstance(tables,list) or set(tables)-set(available_tables):raise ValueError('校验表不属于原任务')
        codes=payload.get('codes',[])
        if not isinstance(codes,list) or any(not isinstance(c,str) for c in codes):raise ValueError('证券范围必须为列表')
        a=parse_date(payload.get('start_date',request['start_date']));b=parse_date(payload.get('end_date',request['end_date']))
        if a>b or a<parse_date(request['start_date']) or b>parse_date(request['end_date']):raise ValueError('校验范围超出原任务')
        report={'id':uuid.uuid4().hex,'task_id':job,'rule_version':VERSION,'scope':{**payload,'targets':targets,'tables':tables,'start_date':str(a),'end_date':str(b)},'started_at':now(),'checks':[],'samples':[],'fingerprints':{},'status':'running'}
        if parent['status'] in {'running','downloading','committing'}:raise ValueError('请等待写入结束后运行事后校验')
        if parent['plan']['revision']=='reference-v1':return self._reference(parent,rules,targets,report)
        dependencies=self._dependencies(parent) if 'relations' in rules else {}
        with registry_context(self.s.registry),catalog_snapshot(self.s.registry),ThreadPoolExecutor(max_workers=2,thread_name_prefix='validate') as pool:
            for unit in self.store.units(job,'normalize'):
                if unit['status']!='completed':continue
                snapshot=unit['result'];child=snapshot['child_id'];child_item=None
                if 'readback' in rules:
                    # Expensive source/column lineage is rebuilt only on explicit L3 requests.
                    child_item=self.s._job(child,True)
                    from .pipeline_options import source_replay
                    with source_replay():_,rejected=self.s._build_canonical(child_item)
                    if rejected:report['checks'].append({'stage':'source_replay','child_id':child,'failed':len(rejected),'samples':rejected[:20]})
                for name,records in snapshot['tables'].items():
                    if name not in tables:continue
                    spec=self.s.registry.spec(name);time_field=self.s.registry.get(name).get('time_field')
                    for record_index,record in enumerate(records):
                        expected=read_rows(record,spec)
                        expected=[r for r in expected if (not codes or r.get('security_code',r.get('code')) in codes) and (not time_field or not r.get(time_field) or a<=r[time_field]<=b)]
                        if not expected:continue
                        outcomes=target_lanes(targets,lambda target:self._target_check(job,child,name,record_index,spec,expected,rules,dependencies,child_item,target),pool)
                        for outcome in outcomes:
                            if outcome.error is not None:
                                report['checks'].append({'stage':'target_check','table':name,'target':outcome.target,'failed':1,'error':self.s.safe_error(outcome.error)})
                                continue
                            part=outcome.value
                            report['checks'].extend(part['checks']);report['samples'].extend(part['samples']);report['fingerprints'].update(part['fingerprints'])
                            if part.get('field_audit'):report.setdefault('field_audit',[]).extend(part['field_audit'])
        failed=any(r.get('failed') or r.get('missing') for r in report['checks'])
        unknown=not report['checks'] or any(r.get('unknown') or r.get('readback_preserved_or_unwritten') for r in report['checks'])
        applicable=any(c.get('values_checked') or c.get('relations_checked') or c.get('readback_verified') for c in report['checks'])
        report.update(status='failed' if failed else 'unknown' if unknown else 'passed' if applicable else 'not_applicable',finished_at=now())
        self.s.state.ledger_put('pipeline_validation',report['id'],report);return report

    def _target_check(self,job,child,name,record_index,spec,expected,rules,dependencies,child_item,target):
        result={'checks':[],'samples':[],'fingerprints':{}}
        db=self.s.pg if target=='postgres' else self.s.ch
        live=existing(db,spec,expected);counts=Counter();sample=[]
        receipt=self.store.receipt(job,child+':'+name+':'+str(record_index),target)
        journal=load_journal(receipt['manifest'],spec) if receipt else None
        intended={row_key(spec,r):r for r in read_rows(journal['expected'],spec)} if journal and journal.get('expected') else {}
        fingerprint=digest(sorted((k,content_hash(spec,r)) for k,r in live.items()))
        result['fingerprints'][target+':'+child+':'+name+':'+str(record_index)]=fingerprint
        for row in expected:
            key=row_key(spec,row);current=live.get(key)
            if current is None:counts['missing']+=1;continue
            problems=[]
            if 'values' in rules:
                problems.extend(values(current))
                counts['values_checked' if any(current.get(k) is not None for k in ('open','high','low','close','volume','amount','adjustment_factor','valid_from','valid_to')) else 'not_applicable']+=1
            if 'relations' in rules:
                checked,unknown,bad=relations(current);counts['relations_checked']+=len(checked);counts['unknown']+=len(unknown);problems.extend(bad)
                checked,unknown,bad=self._cross_relations(current,dependencies);counts['relations_checked']+=len(checked);counts['unknown']+=len(unknown);problems.extend(bad)
                if not checked and not unknown:counts['not_applicable']+=1
            if 'readback' in rules:
                if key not in intended:counts['readback_preserved_or_unwritten']+=1
                elif row_hash(spec,current)!=row_hash(spec,intended[key]):problems.append('stored_value_changed')
                else:counts['readback_verified']+=1
            if problems:
                counts['failed']+=1
                if len(sample)<20:sample.append({'row_key':key,'security_code':current.get('security_code',current.get('code')),'problems':problems})
        result['checks'].append({'table':name,'target':target,'rows':len(expected),**dict(counts)})
        result['samples'].extend(sample)
        if 'readback' in rules and journal:
            from .field_lineage import record_readback
            record_readback(self.s,child_item,target,spec,journal,live)
            fields=[r for r in self.s.state.ledger_rows('field_verification',child) if r['table']==name and r['target']==target]
            result.setdefault('field_audit',[]).append({'table':name,'target':target,'batch':record_index,'child_id':child,'columns':fields})
            proof_errors=sum(r.get('mismatch_count',0) for r in fields)
            from .registry import GENERATED
            # System columns are proved by exact manifest readback, not source-field lineage.
            system=GENERATED|{'event_date','available_at','availability_basis'}
            unproven=sum(max(0,r.get('written_count',0)-r.get('verified_count',0)) for r in fields if r['column'] not in system and r['status'] not in {'not_applicable','unavailable','source_null'})
            if proof_errors or unproven:result['checks'].append({'stage':'field_proof','table':name,'target':target,'failed':proof_errors,'unknown':unproven})
        return result

    def _dependencies(self,parent):
        from .source import read_archive
        result={'calendar':{},'lifecycle':{},'balance':{}}
        for part in parent['slices']:
            if part['dataset'] not in {'trade_cal','stock_basic','balancesheet','balancesheet_vip'} or part['status'] not in {'downloaded','paged'}:continue
            data=read_archive(part['archive'],part['sha'])
            for values in data['items']:
                r=dict(zip(data['fields'],values))
                if part['dataset']=='trade_cal':result['calendar'][(r.get('exchange'),parse_date(r['cal_date']))]=str(r['is_open'])=='1'
                elif part['dataset']=='stock_basic':result['lifecycle'][r['ts_code']]=r
                else:result['balance'][(r.get('ts_code'),str(r.get('end_date')),str(r.get('f_ann_date') or r.get('ann_date')),str(r.get('report_type')))]=r
        return result

    @staticmethod
    def _cross_relations(row,deps):
        checked=[];unknown=[];bad=[];day=row.get('trade_date');code=row.get('security_code')
        if day and code and any(row.get(k) is not None for k in ('close','volume')):
            exchange={'SH':'SSE','SZ':'SZSE','BJ':'SSE'}.get(code.rsplit('.',1)[-1]);opened=deps['calendar'].get((exchange,day))
            if opened is None:unknown.append('calendar_missing')
            else:
                checked.append('calendar')
                if not opened:bad.append('quote_on_closed_day')
            if row.get('asset_type')=='stock':
                life=deps['lifecycle'].get(code)
                if not life:unknown.append('lifecycle_missing')
                else:
                    checked.append('lifecycle')
                    if life.get('list_date') and day<parse_date(life['list_date']):bad.append('quote_before_listing')
                    if life.get('delist_date') and day>parse_date(life['delist_date']):bad.append('quote_after_delisting')
        if row.get('total_assets') is not None and row.get('total_liabilities') is not None:
            key=(code,str(row.get('period_end')),str(row.get('announcement_date')),str(row.get('report_type')));raw=deps['balance'].get(key)
            if not raw or raw.get('total_hldr_eqy_inc_min_int') is None:unknown.append('accounting_equity_source_missing')
            else:
                # Original lexical precision is not guaranteed by JSON integer rendering.
                precision=raw.get('disclosure_precision')
                if precision is None:unknown.append('accounting_precision_unknown')
                else:
                    checked.append('balance_equation')
                    expected=row['total_liabilities']+Decimal(str(raw['total_hldr_eqy_inc_min_int']))
                    if abs(row['total_assets']-expected)>max(Decimal(str(precision)),abs(expected)*Decimal('0.000001')):bad.append('balance_equation_mismatch')
        return checked,unknown,bad

    def _reference(self,parent,rules,targets,report):
        from .reference_service import ReferenceWorkflow
        from .reference_contracts import MODULE_TABLE
        scope=report['scope'];a=parse_date(scope['start_date']);b=parse_date(scope['end_date']);codes=scope.get('codes',[])
        for target in targets:
            store=ReferenceWorkflow(self.s).store(target);snapshot=store.snapshot(audit=True)
            for revision in snapshot['revisions']:
                if revision['import_task_id']!=parent['id']:continue
                if MODULE_TABLE[revision['module']] not in scope['tables']:continue
                verification=store.verify(revision['revision_id'])
                rows=[r for r in snapshot['rows'] if r['revision_id']==revision['revision_id'] and (not codes or r.get('security_code') in codes) and (revision['module']=='industry' or a<=parse_date(r['trade_date'])<=b)]
                active=all(snapshot['heads'].get(s)==revision['revision_id'] for s in revision['scope_ids'])
                counts=Counter();groups=defaultdict(list)
                for row in rows:
                    bad=[]
                    if 'values' in rules:
                        bad.extend(values(row));counts['values_checked']+=1
                        if row.get('is_open') not in (None,0,1):bad.append('invalid_calendar_flag')
                        if row.get('start_at') and row.get('end_at') and row['end_at']<=row['start_at']:bad.append('invalid_suspension_interval')
                    if 'relations' in rules:
                        checked,unknown,problems=relations(row);counts['relations_checked']+=len(checked);counts['unknown']+=len(unknown);bad.extend(problems)
                        if row.get('previous_trade_date'):
                            counts['relations_checked']+=1
                            if row['previous_trade_date']>=row['trade_date']:bad.append('invalid_previous_trade_day')
                        if revision['module']=='industry':groups[(row['security_code'],row['scheme_code'],row['level'])].append(row)
                        if revision['module']=='suspension' and row.get('parse_status') not in {'parsed','ok','full_day'}:counts['unknown']+=1
                    if bad:
                        counts['failed']+=1
                        if len(report['samples'])<100:report['samples'].append({'target':target,'record_id':row['record_id'],'problems':bad})
                for grouped in groups.values():
                    ordered=sorted(grouped,key=lambda r:r.get('valid_from') or date.min)
                    for left,right in zip(ordered,ordered[1:]):
                        if not left.get('valid_from') or not right.get('valid_from'):counts['unknown']+=1;continue
                        counts['relations_checked']+=1
                        if left.get('valid_to') is None or left['valid_to']>right['valid_from']:
                            counts['failed']+=1
                            if len(report['samples'])<100:report['samples'].append({'target':target,'problems':['industry_interval_overlap'],'record_id':right['record_id']})
                report['fingerprints'][target+':'+revision['revision_id']]=digest([revision['manifest_hash'],active])
                report['checks'].append({'target':target,'table':MODULE_TABLE[revision['module']],'revision_id':revision['revision_id'],'rows':len(rows),'verified':verification['verified'],'publication_current':active,'rules':rules,**dict(counts),'errors':verification['errors']})
        failed=any(not c['verified'] or c.get('failed') for c in report['checks'])
        unknown=not report['checks'] or any(not c['publication_current'] or c.get('unknown') for c in report['checks'])
        report.update(status='failed' if failed else 'unknown' if unknown else 'passed',finished_at=now())
        self.s.state.ledger_put('pipeline_validation',report['id'],report);return report
