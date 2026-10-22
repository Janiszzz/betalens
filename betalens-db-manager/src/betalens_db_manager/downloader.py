"""Checkpointed source retrieval. A saturated response is evidence, never a complete leaf."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from .retrieval import partition, tushare_policy
from .source import read_archive
from .source_registry import FetchContext
from .util import atomic_json, digest, dumps, now
from .util import parse_date


class TaskDownloader:
    def __init__(self, service, item):
        self.s, self.item, self.job = service, item, item['id']
        self.plan = item['plan']
        self.universes = {}

    @staticmethod
    def key(request):
        identity = [request['source_id'], request['endpoint'], request['parameters']]
        if request.get('dependency_only'): identity.append('retrieval_dependency')
        return digest(identity)[:24]

    def prepared(self, request):
        sid, endpoint = request['source_id'], request['endpoint']
        contract = self.plan['source_contracts'][sid+':'+endpoint]
        provider = self.s.sources.values[sid]['provider']
        policy = self.plan.get('retrieval_policies', {}).get(sid+':'+endpoint)
        if sid == 'tushare': policy = policy or tushare_policy(endpoint)
        page = dict(contract.get('pagination', {}))
        if policy and policy['mode'] in {'offset', 'offset_probe'}:
            page = {'offset_param': 'offset', 'limit_param': 'limit', 'size': policy['row_limit']}
        if provider == 'file': page = {'offset_param': 'offset', 'limit_param': 'limit', 'size': 5000}
        params = dict(request['parameters'])
        if page:
            limit = page.get('limit_param', 'limit')
            size = int(page.get('size', 1000))
            params[limit] = min(int(params.get(limit, size)), size)
            if params[limit] <= 0: raise ValueError('分页大小须为正整数')
            params.setdefault(page.get('offset_param', 'offset'), 0)
            if int(params[page.get('offset_param', 'offset')]) < 0: raise ValueError('分页偏移须非负')
        return {**request, 'parameters': params}, contract, policy, page

    def run(self, roots=None):
        queue = list(roots if roots is not None else self.plan['requests'])
        visited = set()
        position = 0
        while position < len(queue):
            request, contract, policy, page = self.prepared(queue[position]); position += 1
            key = self.key(request)
            if key in visited: continue
            visited.add(key)
            self.s.state.check_cancel(self.job)
            old = self.s.state.get_slice(self.job, key)
            if old and old.get('archive') and old['status'] in {'downloaded', 'paged', 'split', 'contract_blocked'}:
                read_archive(Path(old['archive']), old['sha'])
                result = json.loads(old['result']) if old['result'] else {}
                children = result.get('children', [])
                # Compatibility with pre-policy checkpoints.
                if old['status'] == 'paged' and not children:
                    children = [self.following(request, page, old['rows'])]
                queue.extend(children)
                continue
            if old and old['status'] == 'unavailable': continue
            children = self.fetch(request, contract, policy, page)
            queue.extend(children)

    @staticmethod
    def following(request, page, count):
        offset = page.get('offset_param', 'offset')
        return {**request, 'parameters': {**request['parameters'], offset: int(request['parameters'].get(offset, 0))+count}}

    def complete_universe(self, asset):
        if asset in self.universes: return self.universes[asset]
        endpoint = 'fund_basic' if asset == 'fund' else 'stock_basic'
        if 'tushare:'+endpoint not in self.plan['source_contracts']:
            raise ValueError('缺少完整证券清单合同；请创建新任务以安全拆分查询')
        filters = [{'market': 'E', 'status': status} for status in ('L', 'D', 'I')] if asset == 'fund' else [{'list_status': status} for status in ('L', 'D', 'P', 'G', 'UN')]
        roots = [{'source_id': 'tushare', 'endpoint': endpoint, 'parameters': p, 'strategy': 'retrieval_dependency', 'dependency_only': True} for p in filters]
        self.run(roots)
        codes = set()
        for part in self.s.state.job(self.job)['slices']:
            if part['dataset'] != endpoint or not part['request'].get('dependency_only'): continue
            if part['status'] in {'split'}: continue
            if part['status'] not in {'downloaded', 'paged'} or not part['archive']:
                raise ValueError('证券清单获取不完整，不能用截断清单拆分查询')
            data = read_archive(Path(part['archive']), part['sha'])
            pos = data['fields'].index('ts_code')
            codes.update(row[pos] for row in data['items'] if row[pos])
        if not codes: raise ValueError('源端证券清单为空，无法证明分区完整性')
        self.universes[asset] = sorted(codes)
        return self.universes[asset]

    def split(self, request, contract):
        endpoint, params = request['endpoint'], request['parameters']
        chunks = partition(endpoint, params, contract)
        if not chunks and 'ts_code' in contract.get('input_fields', []) and not params.get('ts_code') and endpoint not in {'stock_basic', 'fund_basic', 'index_basic'}:
            codes = self.complete_universe('fund' if endpoint.startswith('fund_') else 'stock')
            chunks = [{**params, 'ts_code': code} for code in codes]
        return [{**request, 'parameters': p} for p in chunks]

    def repeated(self, request, page, data):
        if not data['items']: return False
        offset, limit = page.get('offset_param', 'offset'), page.get('limit_param', 'limit')
        scope = {k: v for k, v in request['parameters'].items() if k not in {offset, limit}}
        hashes = {digest(dict(zip(data['fields'], values))) for values in data['items']}
        if len(hashes) != len(data['items']): return True
        for part in self.s.state.job(self.job)['slices']:
            r = part['request']
            if part['status'] not in {'downloaded', 'paged'} or not part['archive'] or r['source_id'] != request['source_id'] or r['endpoint'] != request['endpoint']: continue
            if r.get('dependency_only', False) != request.get('dependency_only', False): continue
            if {k: v for k, v in r['parameters'].items() if k not in {offset, limit}} != scope: continue
            if int(r['parameters'].get(offset, 0)) >= int(request['parameters'].get(offset, 0)): continue
            previous = read_archive(Path(part['archive']), part['sha'])
            if hashes.intersection(digest(dict(zip(previous['fields'], values))) for values in previous['items']): return True
        return False

    @staticmethod
    def scope_error(params, data, contract):
        """Verify the filters on which pagination/partition completeness depends."""
        for values in data['items']:
            row = dict(zip(data['fields'], values))
            if params.get('ts_code') and 'ts_code' in row:
                allowed = {v.strip() for v in str(params['ts_code']).split(',')}
                if row['ts_code'] not in allowed: return '源端返回的证券不符合请求过滤条件，无法证明分区完整性'
            for field in ('trade_date', 'nav_date'):
                if params.get(field) and field in row:
                    if not row[field] or parse_date(row[field]) != parse_date(params[field]): return '源端返回的日期不符合请求过滤条件'
            field = contract.get('date_field')
            if field in {'trade_date', 'nav_date', 'cal_date'} and row.get(field):
                value = parse_date(row[field])
                if params.get('start_date') and value < parse_date(params['start_date']): return '源端返回记录早于请求分区'
                if params.get('end_date') and value > parse_date(params['end_date']): return '源端返回记录晚于请求分区'
        return None

    def fetch(self, request, contract, policy, page):
        s, job = self.s, self.job
        sid, endpoint, params = request['source_id'], request['endpoint'], request['parameters']
        key = self.key(request)
        s.state.slice(job, key, endpoint, request)
        root = (s.settings.archive_dir or s.settings.state_dir/'archive')/job/endpoint
        root.mkdir(parents=True, exist_ok=True)
        archive = root/(key+'.json.gz')
        wire_meta, error = {}, None
        if sid == 'tushare':
            try: data, sha = s.source.fetch(endpoint, params, [f['name'] for f in contract['fields']], archive)
            except Exception as exc:
                message = s.safe_error(exc)
                if any(word in message.lower() for word in ('权限', '积分', 'permission')):
                    s.state.slice(job, key, endpoint, request, status='unavailable', rows=0, result={'reason': message})
                    s.state.log(job, 'source_unavailable', source_id=sid, endpoint=endpoint, error=message)
                    s.state.ledger_put('meta_source_capability', sid+':'+endpoint, {'task_id': job, 'source_name': sid, 'endpoint': endpoint, 'capability': endpoint, 'status': 'unavailable', 'error': message, 'last_probe_at': now()})
                    return []
                raise
        else:
            adapter = s.sources.adapter(sid)
            source_request = {'source_id': sid, 'endpoint': endpoint, 'parameters': params}
            adapter.plan(source_request)
            try:
                data = adapter.normalize(adapter.fetch(source_request, FetchContext(s.settings, s.state, job)), contract)
            except ValueError as exc:
                error = s.safe_error(exc); data = {'fields': ['response_error'], 'items': [[error]]}
            if getattr(adapter, 'last_wire', None) is not None:
                wire = root/(key+'.response.json.gz')
                with gzip.open(wire, 'wb') as handle: handle.write(adapter.last_wire)
                wire_meta = {'wire_archive': str(wire), 'wire_sha256': hashlib.sha256(adapter.last_wire).hexdigest()}
            payload = dumps({'code': 0, 'data': data}).encode(); sha = hashlib.sha256(payload).hexdigest()
            with gzip.open(archive.with_suffix('.tmp'), 'wb') as handle: handle.write(payload)
            archive.with_suffix('.tmp').replace(archive)
        stamp = now()
        atomic_json(archive.with_suffix('.meta.json'), {'fetched_at': stamp, 'source_id': sid, 'endpoint': endpoint, 'sha256': sha, 'parameters': params, **wire_meta})
        status, result, children = 'downloaded', {}, []
        expected = {f['name'] for f in contract['fields']}
        if error or len(data['fields']) != len(set(data['fields'])) or set(data['fields'])-expected or set(contract.get('keys', []))-set(data['fields']) or any(len(values) != len(data['fields']) for values in data['items']):
            status = 'contract_blocked'; result['blocked'] = error or '源响应字段合同不匹配'
        if sid == 'tushare' and status != 'contract_blocked':
            try: error = self.scope_error(params, data, contract)
            except (ValueError, TypeError): error = '源端日期格式不符合请求合同'
            if error: status = 'contract_blocked'; result['blocked'] = error
        if sid != 'tushare' and status != 'contract_blocked':
            try:
                validation = adapter.validate_response(data, contract)
                if validation.get('missing') or validation.get('extra'): raise ValueError('源响应字段合同不匹配')
            except ValueError as exc: status = 'contract_blocked'; result['blocked'] = str(exc)
        count = len(data['items'])
        cap = policy['row_limit'] if policy else contract.get('limit', 100000)
        result.update(row_limit=cap, retrieval_mode=policy['mode'] if policy else 'offset' if page else 'single', complete=False)
        if status != 'contract_blocked':
            if page:
                # Even a short page may be a hidden server cap. Only an empty page terminates the chain.
                if self.repeated(request, page, data):
                    status = 'contract_blocked'; result['blocked'] = '源接口忽略分页或分页记录重复/交叠，无法证明完整性，已阻止入库'
                elif count:
                    status = 'paged'; children = [self.following(request, page, count)]
                else: result.update(complete=True, termination='empty_page')
            elif count >= cap:
                try: children = self.split(request, contract) if sid == 'tushare' else []
                except (ValueError, RuntimeError) as exc: result['blocked'] = s.safe_error(exc)
                if children: status = 'split'
                else:
                    status = 'contract_blocked'; result.setdefault('blocked', '最小查询范围仍达到单次上限，无法证明完整性；需补充查询合同或缩小范围')
            else: result.update(complete=True, termination='below_documented_cap')
        result['children'] = children
        s.state.slice(job, key, endpoint, request, status=status, archive=str(archive), sha=sha, rows=count, result=result)
        s.state.log(job, 'response_archived', source_id=sid, endpoint=endpoint, rows=count, row_limit=cap, status=status, next_requests=len(children), sha256=sha)
        for child in children:
            child, _, _, _ = self.prepared(child)
            s.state.slice(job, self.key(child), endpoint, child)
        s.state.ledger_put('import_source_archive', digest([job, key]), {'task_id': job, 'dataset': endpoint, 'endpoint': sid+':'+endpoint, 'request_hash': digest(params), 'archive_path': str(archive), 'response_sha256': sha, 'response_fields': ','.join(data['fields']), 'fetched_at': stamp, 'rows': count})
        s.state.check_cancel(job)
        return children
