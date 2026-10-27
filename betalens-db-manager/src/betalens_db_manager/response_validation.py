"""Shared checks for downloaded and replayed responses and their retrieval graph."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .source import read_archive
from .util import digest, parse_date, now

VERSION = 'response-validation-v2-financial-dates'


def issue(code, message, **details):
    return {'code':code, 'reason':message, 'blocking':True, **details}


def scope_roles(plan, request, contract):
    """Use original mappings on replay, including custom business key names."""
    sid, endpoint = request['source_id'], request['endpoint']
    fields = {f['name']:f for f in contract['fields']}
    result = {'security':[], 'dates':[], 'parameters':dict(contract.get('parameter_bindings', {}))}
    code = contract.get('security_field')
    if not code and sid == 'tushare':
        code = 'ts_code' if 'ts_code' in fields else 'index_code' if 'index_code' in fields else None
    if code:result['security'].append(code)
    if contract.get('filter_date_field') or contract.get('date_field'):result['dates'].append({'field':contract.get('filter_date_field') or contract['date_field']})
    mappings = plan.get('validation_mapping_snapshots', plan.get('mapping_snapshots', {}))
    for mapping in mappings.values():
        for binding in mapping.get('bindings', []):
            if (binding['source_id'], binding['endpoint']) != (sid, endpoint):continue
            result['parameters'].update(binding.get('request_bindings', {}))
            rules = binding.get('fields', [])
            for rule in rules:
                if rule['column'] == 'security_code':result['security'].append(rule['source_field'])
                if any(d['field'] == rule['source_field'] for d in result['dates']):
                    for d in result['dates']:
                        if d['field'] == rule['source_field']:d['date_format'] = rule.get('date_format')
            if not result['security']:
                keys = binding.get('join_keys') or [r for r in rules if r['source_field'] in contract.get('keys', [])]
                candidates = [r['source_field'] for r in keys if fields.get(r['source_field'], {}).get('type') == 'text']
                if len(set(candidates)) == 1:result['security'].extend(candidates)
            if not result['dates']:
                candidates = [r for r in rules if r['source_field'] in contract.get('keys', []) and fields[r['source_field']]['type'] == 'date']
                if len(candidates) == 1:result['dates'].append({'field':candidates[0]['source_field'], 'date_format':candidates[0].get('date_format')})
    result['security'] = list(dict.fromkeys(result['security']))
    return result


def financial_date_constraints(request, contract):
    """Official financial filters use different date dimensions by endpoint.

    Income/balance/cash windows filter announcement dates. Indicator windows
    filter reporting periods. All four use period for an exact report end.
    """
    base=request['endpoint'].removesuffix('_vip')
    if request['source_id']!='tushare' or base not in {'income','balancesheet','cashflow','fina_indicator'}:
        return None,[]
    params=request.get('parameters',{});allowed=set(contract.get('input_fields',[]))
    constraints={};errors=[]
    def provided(name):return params.get(name) not in (None,'')
    def supported(name):
        if name not in allowed:
            errors.append(issue('unsupported_date_filter','接口未登记此日期过滤参数，不能证明源端应用了该筛选',parameter=name))
            return False
        return True
    for parameter,field in (('period','end_date'),('ann_date','ann_date'),('f_ann_date','f_ann_date')):
        if provided(parameter) and supported(parameter):
            constraints.setdefault(field,{})['exact']=params[parameter]
    window_field='end_date' if base=='fina_indicator' else 'ann_date'
    for parameter,bound in (('start_date','start'),('end_date','end')):
        if provided(parameter) and supported(parameter):
            constraints.setdefault(window_field,{})[bound]=params[parameter]
    for field,bounds in constraints.items():
        try:
            parsed={name:parse_date(value) for name,value in bounds.items()}
            if parsed.get('start') and parsed.get('end') and parsed['start']>parsed['end']:
                errors.append(issue('date_request','请求日期窗口起点晚于终点',field=field))
        except (ValueError,TypeError):
            errors.append(issue('date_request','请求日期参数不符合日期合同',field=field))
    return constraints,errors


def validate_response(plan, request, contract, data):
    errors = []
    if not isinstance(data, dict) or not isinstance(data.get('fields'), list) or not isinstance(data.get('items'), list):
        return [issue('response_format', '响应必须是 fields/items 对象')]
    actual, items = data['fields'], data['items']
    if not all(isinstance(f,str) for f in actual) or len(actual) != len(set(actual)):
        return [issue('duplicate_fields', '源响应字段名无效或重复')]
    known = {f['name'] for f in contract['fields']}
    required = set(contract.get('keys', [])) | {f['name'] for f in contract['fields'] if f.get('required')}
    for mapping in plan.get('mapping_snapshots', {}).values():
        for binding in mapping.get('bindings', []):
            if (binding['source_id'],binding['endpoint']) == (request['source_id'],request['endpoint']):
                required.update(r['source_field'] for r in binding.get('fields', []) if r.get('required'))
    if set(actual)-known:errors.append(issue('unknown_fields', '源响应包含未登记字段', fields=sorted(set(actual)-known)))
    if required-set(actual):errors.append(issue('missing_fields', '源响应缺少必需字段', fields=sorted(required-set(actual))))
    if any(not isinstance(row, (list,tuple)) or len(row) != len(actual) for row in items):
        errors.append(issue('row_width', '源字段和值数量不一致'))
    if errors:return errors
    original_contract=plan.get('retrieval_source_contracts',{}).get(request['source_id']+':'+request['endpoint'],contract)
    financial_dates,date_errors=financial_date_constraints(request,original_contract)
    errors.extend(date_errors)
    if financial_dates is not None:
        missing_dates=set(financial_dates)-set(actual)
        if missing_dates:errors.append(issue('missing_date_fields','响应缺少用于核对请求日期范围的字段',fields=sorted(missing_dates)))
    if errors:return errors
    roles = request.get('validation_scope') or scope_roles(plan, request, original_contract)
    if contract.get('filter_date_field'):roles={**roles,'dates':[{'field':contract['filter_date_field']}]}
    if request['source_id']=='tushare' and request['endpoint']=='index_weight':roles={**roles,'security':['index_code']}
    params, pb = request.get('parameters', {}), roles['parameters']
    security_param = pb.get('security_code', 'ts_code' if request['source_id']=='tushare' else 'security_code')
    code_filter = params.get(security_param)
    # index_weight's index code is the requested object, not each constituent.
    if code_filter is None and 'index_code' in params:code_filter = params['index_code']
    codes = set(str(code_filter).split(',')) if code_filter is not None else set()
    global_query=request['source_id']=='tushare' and request['endpoint'] in {'index_classify','trade_cal'}
    if not codes and not global_query and not request.get('dependency_only') and request.get('strategy') != 'dependency':
        codes = set(plan.get('request', {}).get('codes', []))
    if codes and not roles['security']:
        errors.append(issue('scope_role_missing', '请求包含证券范围，但无法确定响应证券字段'))
    def day(value, fmt=None):
        return datetime.strptime(str(value), fmt).date() if fmt else parse_date(value)
    request_format = pb.get('date_format', contract.get('date_format'))
    for index, values in enumerate(items, 1):
        row = dict(zip(actual, values))
        if codes and any(not isinstance(row.get(field),str) or row[field] not in codes for field in roles['security']):
            errors.append(issue('security_scope', '源端返回的证券不符合请求过滤条件', source_row=index));break
        if financial_dates is not None:
            for field,bounds in financial_dates.items():
                try:
                    value=day(row[field])
                    if ('exact' in bounds and value!=day(bounds['exact'])
                            or 'start' in bounds and value<day(bounds['start'])
                            or 'end' in bounds and value>day(bounds['end'])):
                        errors.append(issue('date_scope','财报响应日期不符合该接口的请求范围',field=field,source_row=index));break
                except (ValueError,TypeError,KeyError):
                    errors.append(issue('date_format','财报响应缺少可验证的日期值或日期格式无效',field=field,source_row=index));break
            if errors:break
            continue
        for date_role in roles['dates']:
            field = date_role['field']
            if field not in row:continue
            exact = params.get(pb.get('trade_date','trade_date'))
            if exact is None and field not in {pb.get('start_date','start_date'),pb.get('end_date','end_date')}:exact = params.get(field)
            if exact is None and request['source_id']=='tushare' and request['endpoint'].removesuffix('_vip') in {'income','balancesheet','cashflow','fina_indicator'} and field=='end_date':exact=params.get('period')
            start, end = params.get(pb.get('start_date','start_date')), params.get(pb.get('end_date','end_date'))
            if exact is None and start is None and end is None:continue
            try:
                value = day(row[field], date_role.get('date_format'))
                if exact is not None and value != day(exact,request_format) or start is not None and value < day(start,request_format) or end is not None and value > day(end,request_format):
                    errors.append(issue('date_scope', '源端返回的日期不符合请求分区', source_row=index));break
            except (ValueError, TypeError):
                errors.append(issue('date_format', '源端日期格式不符合请求合同', source_row=index));break
        if errors:break
    return errors


def validate_archives(service, item):
    """Reconstruct completeness from fixed requests and archived children, never new queries."""
    from .downloader import TaskDownloader
    from .retrieval import partition
    plan = item['plan']; downloader = TaskDownloader(service,item)
    parts, data_by_key, errors = {}, {}, []
    frozen = plan.get('retrieval_source_contracts', plan['source_contracts'])
    def valid_request(request):
        return (isinstance(request,dict) and isinstance(request.get('source_id'),str)
            and isinstance(request.get('endpoint'),str) and isinstance(request.get('parameters'),dict))
    for part in item['slices']:
        if not valid_request(part.get('request')):
            errors.append(issue('request_format','归档请求结构无效',dataset=part.get('dataset')));continue
        key=TaskDownloader.key(part['request'])
        if key in parts:
            errors.append(issue('duplicate_request','同一固定请求存在多个归档分片',dataset=part.get('dataset')))
        parts[key]=part
    for key, part in parts.items():
        request = part['request']; identity = request['source_id']+':'+request['endpoint']
        if part['status'] == 'unavailable':continue
        if identity not in plan['source_contracts'] or identity not in frozen:
            errors.append(issue('missing_contract','归档请求缺少固定来源合同',dataset=part['dataset']));continue
        if not part.get('archive'):
            errors.append(issue('missing_archive','请求分片尚无完整归档',dataset=part['dataset']));continue
        try:data = read_archive(Path(part['archive']),part['sha'])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(issue('archive_integrity',str(exc),dataset=part['dataset']));continue
        problems = validate_response(plan,request,plan['source_contracts'][identity],data)
        if not problems and request['source_id'] != 'tushare':
            try:
                checked = service.sources.adapter(request['source_id']).validate_response(data,plan['source_contracts'][identity])
                if not isinstance(checked,dict):raise ValueError('适配器校验结果必须为对象')
                if checked.get('missing') or checked.get('extra'):raise ValueError('源响应字段合同不匹配')
            except (ValueError, TypeError, KeyError) as exc:problems.append(issue('adapter_validation',str(exc)))
        errors.extend({**p,'dataset':part['dataset'],'request_key':key} for p in problems)
        # Invalid field names/row widths must never reach dict(zip(...)) below.
        if not problems:data_by_key[key] = data

    def prepared_request(request):
        if not valid_request(request):raise ValueError('请求结构无效')
        original = downloader.plan
        downloader.plan = {**plan,'source_contracts':frozen}
        try:return downloader.prepared(request)
        finally:downloader.plan = original

    seen, visiting, memo, edges, universes = set(), set(), {}, {}, {}

    def dependency_universe(asset):
        """Recover only leaves reachable from every required archived universe root."""
        if asset in universes:return universes[asset]
        endpoint='fund_basic' if asset=='fund' else 'stock_basic'
        filters=([{'market':'E','status':status} for status in ('L','D','I')] if asset=='fund'
            else [{'list_status':status} for status in ('L','D','P','G','UN')])
        roots=[{'source_id':'tushare','endpoint':endpoint,'parameters':params,
            'strategy':'retrieval_dependency','dependency_only':True} for params in filters]
        problems=[];root_keys=[]
        for root in roots:
            problems.extend(visit(root,require_available=True))
            try:root_keys.append(TaskDownloader.key(prepared_request(root)[0]))
            except (ValueError,TypeError,KeyError):pass  # visit recorded the contract error.
        codes=set();pending=list(root_keys);visited=set()
        while pending:
            key=pending.pop()
            if key in visited:continue
            visited.add(key);pending.extend(edges.get(key,[]))
            part=parts.get(key);data=data_by_key.get(key)
            if part is None or data is None or part['status']=='split':continue
            if 'ts_code' not in data['fields']:
                problems.append(issue('universe_keys','证券清单归档缺少 ts_code',dataset=endpoint));continue
            pos=data['fields'].index('ts_code')
            for row in data['items']:
                if not isinstance(row[pos],str) or not row[pos]:
                    problems.append(issue('universe_keys','证券清单归档含无效证券代码',dataset=endpoint));break
                codes.add(row[pos])
        if not codes:problems.append(issue('incomplete_universe','归档无法证明完整证券清单，必须重新拉取',dataset=endpoint))
        universes[asset]=(sorted(codes),problems)
        return universes[asset]

    def visit(request,require_available=False):
        try:prepared, contract, policy, page = prepared_request(request)
        except (ValueError,TypeError,KeyError) as exc:
            return [issue('retrieval_contract','无法按固定合同验证请求：'+str(exc),dataset=request.get('endpoint') if isinstance(request,dict) else None)]
        key = TaskDownloader.key(prepared)
        if key in visiting:return [issue('retrieval_cycle','分页或拆分请求存在循环',dataset=request['endpoint'])]
        memo_key=(key,require_available)
        if memo_key in memo:return memo[memo_key]
        seen.add(key);visiting.add(key)
        part = parts.get(key); data = data_by_key.get(key); problems = []
        try:
            if part is None:
                problems.append(issue('missing_request','固定请求缺少归档分片',dataset=request['endpoint']))
            elif part['status']=='unavailable':
                if require_available:problems.append(issue('unavailable_dependency','证券清单分区不可用，无法证明完整性',dataset=request['endpoint']))
            elif data is not None:
                result=part.get('result') or {}
                if not isinstance(result,dict):raise ValueError('分页证据必须为对象')
                children=result.get('children') or []
                if not isinstance(children,list) or not all(valid_request(c) for c in children):raise ValueError('分页后继必须是固定请求对象列表')
                count=len(data['items']);cap=(policy or {}).get('row_limit',contract.get('limit',100000))
                if type(cap) is not int or cap<=0:raise ValueError('响应上限必须为正整数')
                expected=[]
                if page:
                    if count:
                        expected=[downloader.following(prepared,page,count)]
                        if children and [TaskDownloader.key(c) for c in children] != [TaskDownloader.key(expected[0])]:
                            problems.append(issue('pagination_chain','分页后继与原始请求不一致',dataset=request['endpoint']))
                    elif children:problems.append(issue('pagination_chain','空页终点不能包含后继',dataset=request['endpoint']))
                elif count>=cap or part['status']=='split':
                    chunks=partition(request['endpoint'],prepared['parameters'],contract) if request['source_id']=='tushare' else []
                    if not chunks and request['source_id']=='tushare' and 'ts_code' in contract.get('input_fields',[]) and not prepared['parameters'].get('ts_code') and request['endpoint'] not in {'stock_basic','fund_basic','index_basic'}:
                        codes,universe_errors=dependency_universe('fund' if request['endpoint'].startswith('fund_') else 'stock')
                        problems.extend(universe_errors)
                        if not universe_errors:chunks=[{**prepared['parameters'],'ts_code':code} for code in codes]
                    expected=[{**prepared,'parameters':params} for params in chunks]
                    if not expected:problems.append(issue('incomplete_partition','响应达到上限，缺少完整拆分证据，必须重新拉取',dataset=request['endpoint']))
                    elif sorted(TaskDownloader.key(c) for c in children)!=sorted(TaskDownloader.key(c) for c in expected):
                        problems.append(issue('partition_coverage','归档子分区未完整且唯一地覆盖父请求',dataset=request['endpoint']))
                elif children:problems.append(issue('retrieval_chain','非分页叶节点包含未知后继',dataset=request['endpoint']))
                edges[key]=[TaskDownloader.key(prepared_request(c)[0]) for c in expected]
                for child in expected:problems.extend(visit(child,require_available=require_available))
        except (ValueError,TypeError,KeyError) as exc:
            problems.append(issue('retrieval_evidence','归档分页证据无效：'+str(exc),dataset=request['endpoint']))
        finally:visiting.discard(key)
        memo[memo_key]=problems
        return problems
    for request in plan.get('requests', []):errors.extend(visit(request))
    # Repeated pages cannot prove completeness even when a terminal empty page exists.
    groups = {}
    for key, part in parts.items():
        if key not in seen or key not in data_by_key:continue
        request = part['request']; identity = request['source_id']+':'+request['endpoint']
        try:_, _, _, page = prepared_request(request)
        except (ValueError,TypeError,KeyError):continue
        if not page:continue
        data = data_by_key[key]
        group = digest([identity,request.get('dependency_only',False),{k:v for k,v in request['parameters'].items() if k not in {page.get('offset_param','offset'),page.get('limit_param','limit')}}])
        prior = groups.setdefault(group,set())
        for row in data['items']:
            fingerprint = digest(dict(zip(data['fields'],row)))
            if fingerprint in prior:errors.append(issue('pagination_overlap','分页记录重复或交叠，无法证明完整性',dataset=request['endpoint']));break
            prior.add(fingerprint)
    return list({digest(error):error for error in errors}.values())


def record_archive_validation(service,item):
    """Persist the same structured outcome for download, replay and audit callers."""
    errors=validate_archives(service,item)
    report={'task_id':item['id'],'version':VERSION,'blocking':bool(errors),'errors':errors,'validated_at':now()}
    service.state.ledger_put('task_response_validation',item['id'],report)
    service.state.log(item['id'],'archive_validation',validation_version=VERSION,blocking=report['blocking'],validation_errors=errors)
    return report
