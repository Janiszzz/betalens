from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from zoneinfo import ZoneInfo

from .canonical import CanonicalTable, canonical, load_canonical_catalog, FINANCIAL_ALIASES
from .comparison import typed, row_key, PROVENANCE
from .catalog import Dataset, dataset, load_catalog
from .config import Settings
from .mapping import FIELD_MAP, VERSION
from .clickhouse import ClickHouse
from .source import read_archive
from .sources import TushareSourceAdapter
from .state import State
from .storage import Postgres, columns, relation, row_hash
from .util import atomic_json, digest, dumps, now, parse_date
from .business_semantics import corporate_action_partial, daily_average_price, financial_record
from .time_scope import row_in_scope

SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_DATASETS = [
    "stock_basic", "fund_basic", "index_basic", "trade_cal", "daily", "daily_basic", "adj_factor",
    "fund_daily", "fund_adj", "index_daily", "suspend_d", "index_classify", "index_member_all", "ci_index_member",
]


def high_precision(function):
    from functools import wraps
    @wraps(function)
    def wrapped(*args,**kwargs):
        with localcontext() as context:
            context.prec=80
            return function(*args,**kwargs)
    return wrapped


def dates(start: date, end: date):
    while start <= end:
        yield start
        start += timedelta(days=1)


def _datetime(value):
    if value is None: return None
    if isinstance(value, datetime): return value
    text = str(value).replace("Z", "+00:00")
    result = datetime.fromisoformat(text)
    return result if result.tzinfo else result.replace(tzinfo=timezone.utc)


def _event_available(event_date: date | None, basis: str = "next_calendar_day_assumed"):
    if not event_date: return None, "unknown"
    return datetime.combine(event_date + timedelta(days=1), time.min, SHANGHAI).astimezone(timezone.utc), basis


def normalize_source(spec: Dataset, data: dict, job: str, slice_id: str, downloaded: str, calendar: list[date] | None = None):
    """Validate a Tushare response and convert values without positional guessing."""
    expected = {field["name"] for field in spec.fields}
    actual = set(data.get("fields", []))
    missing, extra = expected - actual, actual - expected
    required = set(spec.keys) - {"out_date", "f_ann_date", "suspend_timing"}
    required |= {'adj_factor'} if spec.name in {'adj_factor','fund_adj'} else {'open','high','low','close','vol','amount'} if spec.name in {'daily','fund_daily','index_daily'} else set()
    if len(data.get("fields", [])) != len(actual): raise ValueError(f"{spec.name} 返回重复字段名")
    if missing & required:
        raise ValueError(f"{spec.name} 返回必需字段缺失：{','.join(sorted(missing & required))}")
    if extra:
        raise ValueError(f"{spec.name} 返回未登记字段：{','.join(sorted(extra))}")
    rows, rejected, seen = [], [], set()
    for index, values in enumerate(data.get("items", []), 1):
        raw = dict(zip(data["fields"], values))
        try:
            if len(values) != len(data["fields"]): raise ValueError("字段数量不一致")
            row = {}
            for field in spec.fields:
                key, value = field["name"], raw.get(field["name"])
                if value is None or value == "": value = None
                elif field["type"] == "date": value = parse_date(value)
                elif field["type"] == "numeric":
                    with localcontext() as context:
                        context.prec = 80
                        value = Decimal(str(value)) * Decimal(str(field.get("multiplier", "1")))
                    if not value.is_finite(): raise ValueError(f"{key}: 非有限数值")
                else: value = str(value)
                row[key] = value
            if any(row.get(key) is None for key in set(spec.keys)-{'out_date','f_ann_date','suspend_timing'}): raise ValueError("源数据业务主键为空")
            key = digest([row.get(k) for k in spec.keys])
            payload_hash = digest(row)
            identity = (key, payload_hash)
            if identity in seen: continue
            seen.add(identity)
            row['_raw']=raw
            row["_job"] = job; row["_slice"] = slice_id; row["_row"] = index
            row["_downloaded_at"] = downloaded; row["_payload_hash"] = payload_hash
            rows.append(row)
        except (ValueError, InvalidOperation, TypeError, KeyError) as exc:
            rejected.append({"source_row": index, "reason": str(exc), "raw": raw})
    return rows, rejected


def normalize(spec: Dataset, data: dict, job: str, slice_id: str, observed: str, calendar=None):
    """Backward-compatible helper used by small adapter tests."""
    rows, rejected = normalize_source(spec, data, job, slice_id, observed, calendar)
    for row in rows:
        partition = row.get(spec.date_field) if spec.date_field else date(1900, 1, 1)
        if partition is None: partition = date(1900, 1, 1)
        row["_date"] = partition
        row["_key"] = digest([row.get(key) for key in spec.keys])
        row["_hash"] = row.pop("_payload_hash")
        row["_observed_at"] = _datetime(observed)
        available, basis = _event_available(partition)
        announcement = row.get("f_ann_date") or row.get("ann_date")
        if announcement:
            available, basis = _event_available(announcement, "announcement_next_exchange_open_assumed;historical_revision_not_proven")
        row["_available_at"] = available
        row["_availability_basis"] = basis
    return rows, rejected


class TushareImportService:
    def __init__(self, settings: Settings | None = None, source=None):
        self.settings = settings or Settings.from_env()
        self.state = State(self.settings.state_dir)
        self.pg = Postgres(self.settings)
        self.ch = ClickHouse(self.settings)
        self.source = source or TushareSourceAdapter(self.settings, self.state)

    def safe_error(self, exc: Exception) -> str:
        message = str(exc)
        for secret in [self.settings.token, self.settings.pg_password, self.settings.ch_password]:
            if secret: message = message.replace(secret, "[redacted]")
        return message[:2000]

    def _task_status(self, task_id: str, status: str, error: str | None = None):
        from .publishing import task_record
        task_record(self,task_id,status,error)

    def doctor(self) -> dict:
        result = dict(config=self.settings.public(), postgres={}, clickhouse={}, archive={})
        for key, function in [("postgres", self.pg.schema_plan), ("clickhouse", self.ch.status)]:
            try: result[key] = function()
            except Exception as exc: result[key] = dict(ready=False, error=self.safe_error(exc))
        import shutil
        try: result["archive"] = dict(ready=True, free_bytes=shutil.disk_usage(self.settings.state_dir).free)
        except OSError as exc: result["archive"] = dict(ready=False, error=self.safe_error(exc))
        return result

    def schema_plan(self):
        try: pg=self.pg.schema_plan()
        except Exception as exc: pg={'ready':False,'error':self.safe_error(exc)}
        try: ch = self.ch.status()
        except Exception as exc: ch = {"ready": False, "error": self.safe_error(exc)}
        plan = {"postgres": pg, "clickhouse": ch, "contracts": {name: spec.hash for name, spec in load_canonical_catalog().items()}, "mapping_version": VERSION}
        plan["token"] = digest(plan)
        return plan

    def initialize(self, token: str, targets: list[str] | None = None):
        current=self.schema_plan()
        if current['token']!=token:raise ValueError('结构计划已变化，请重新检查')
        targets=targets or ['postgres','clickhouse'];result={}
        if set(targets)-{'postgres','clickhouse'}:raise ValueError('目标库无效')
        from .metadata import records
        for target_name in targets:
            target=self.pg if target_name=='postgres' else self.ch
            try:
                result[target_name]=target.install(digest(current['postgres'])) if target_name=='postgres' else target.install()
                for name,rows in records().items():target.control_write(name,rows)
            except Exception as exc:result[target_name]={'ready':False,'error':self.safe_error(exc)}
        atomic_json(self.settings.state_dir/'installation.json',result)
        return result

    def _make_plan(self, request: dict) -> dict:
        allowed = {"datasets", "start_date", "end_date", "codes", "mode", "parameters", "targets", "sources", "strict_capabilities"}
        unknown = set(request) - allowed
        if unknown: raise ValueError("未知请求字段：" + ",".join(sorted(unknown)))
        selected = list(request.get("datasets") or DEFAULT_DATASETS)
        for name in selected: dataset(name)
        start = parse_date(request.get("start_date", "2004-01-01"))
        today = datetime.now(SHANGHAI).date()
        end = parse_date(request.get("end_date", (today - timedelta(days=1)).isoformat()))
        if start > end or start.year < 1900 or end > today: raise ValueError("日期范围无效")
        codes = sorted(set(request.get("codes", [])))
        import re
        if any(not re.fullmatch(r"[A-Z0-9_]+\.(SH|SZ|BJ|SI|CI|CSI|CNI|OF|WI|IB)", code) for code in codes): raise ValueError("证券代码格式无效")
        mode = request.get("mode", "fill_missing")
        if mode not in {"fill_missing", "refresh_source"}: raise ValueError("不支持的冲突模式")
        targets = request.get("targets", ["postgres", "clickhouse"])
        if set(targets) - {"postgres", "clickhouse"} or not targets: raise ValueError("targets 只能选择 postgres/clickhouse")
        sources = request.get("sources", ["tushare"])
        if sources != ["tushare"]: raise ValueError("当前可执行来源为 tushare；Wind 和文件接入使用同一合同接口")
        parameters = request.get("parameters", {})
        for name, values in parameters.items():
            if name not in selected or not isinstance(values, dict) or set(values) - set(dataset(name).input_fields):
                raise ValueError(f"{name} 的接口参数不在合同中")
        for quote,factor in (("daily","adj_factor"),("fund_daily","fund_adj")):
            if quote in selected and factor not in selected: selected.append(factor)
        if 'index_daily' in selected and (not codes or any(code.endswith('.SI') for code in codes)) and 'sw_daily' not in selected:
            selected.append('sw_daily')
        if "index_member_all" in selected and "index_classify" not in selected: selected.insert(0,"index_classify")
        if "ci_index_member" in selected and not codes and "stock_basic" not in selected: selected.insert(0,"stock_basic")
        if (any(dataset(n).strategy == "daily" for n in selected) or any(n in {'index_daily','sw_daily'} for n in selected)) and "trade_cal" not in selected:
            selected.insert(0, "trade_cal")
        if any(n in {"daily", "daily_basic", "adj_factor", "suspend_d"} for n in selected) and "stock_basic" not in selected and not codes:
            selected.insert(0, "stock_basic")
        selected = list(dict.fromkeys(selected))
        plan = dict(request={"datasets": selected, "start_date": start.isoformat(), "end_date": end.isoformat(), "codes": codes,
                             "mode": mode, "parameters": parameters, "targets": list(dict.fromkeys(targets)), "sources": sources,
                             "strict_capabilities": bool(request.get("strict_capabilities", False))},
                    contracts={n: dataset(n).hash for n in selected}, source_contracts={n:dataset(n).as_dict() for n in selected}, target=self.settings.fingerprint,
                    mapping_version=VERSION, canonical_contracts={name:spec.hash for name,spec in load_canonical_catalog().items()}, warnings=["空响应保留审计，不证明证券无数据", "基金/指数交易状态可能为派生状态"],
                    estimated_daily_requests=sum(1 for n in selected if dataset(n).strategy == "daily") * ((end-start).days+1), revision="2")
        return plan

    def plan(self,request:dict)->dict:
        plan=self._make_plan(request)
        item=self.state.create_job(plan,self.settings.fingerprint)
        self._task_status(item['id'],'planned')
        return item

    def _job(self, job: str, validate: bool = False) -> dict:
        item = self.state.job(job)
        if item["target"] != self.settings.fingerprint: raise ValueError("任务属于其他数据库配置")
        for name, value in item["plan"]["contracts"].items():
            spec=canonical(name) if item['plan']['request'].get('sources')==['file'] and name in load_canonical_catalog() else dataset(name)
            if validate and spec.hash != value: raise ValueError("源字段合同已变化，请重新创建计划；原归档和审计仍可查询")
        return item

    def _calendar(self) -> list[date]:
        try:
            with self.pg.connect(readonly=True) as conn, conn.cursor() as cur:
                cur.execute(f'SELECT trade_date FROM "{self.settings.pg_schema}"."trade_calendar" WHERE exchange=\'SSE\' AND is_open=1 ORDER BY trade_date')
                return [row[0] for row in cur.fetchall()]
        except Exception: return []

    def _universe(self, job: dict, asset_type: str | None = None) -> list[str]:
        codes = set()
        endpoints={'stock':'stock_basic','fund':'fund_basic','index':'index_basic'}
        for part in job["slices"]:
            if part["dataset"] in {"stock_basic", "fund_basic", "index_basic"} and part["status"] in {"downloaded","paged"}:
                if asset_type and part['dataset']!=endpoints[asset_type]:continue
                data = read_archive(self.settings.state_dir / part["archive"], part["sha"])
                if "ts_code" in data["fields"]:
                    pos = data["fields"].index("ts_code"); codes.update(row[pos] for row in data["items"] if row[pos])
        if not codes:
            for name in job['plan']['request']['targets']:
                target=self.pg if name=='postgres' else self.ch
                try:
                    after=None
                    while True:
                        page=target.query('security_dim',limit=1000,after=after)
                        codes.update(row['security_code'] for row in page['rows'] if not asset_type or row['asset_type']==asset_type)
                        after=page['next']
                        if not after:break
                except Exception:continue
                if codes:break
        if not codes: raise ValueError("没有证券清单，请先下载基础信息或显式指定 codes")
        return sorted(codes)

    def _job_calendar(self, item: dict) -> list[date]:
        result = set(self._calendar())
        for part in item["slices"]:
            if part["dataset"] != "trade_cal" or part["status"] == "split" or not part["archive"]: continue
            data = read_archive(self.settings.state_dir / part["archive"], part["sha"])
            for values in data["items"]:
                row = dict(zip(data["fields"], values))
                if row.get("exchange") == "SSE" and str(row.get("is_open")) == "1": result.add(parse_date(row["cal_date"]))
        return sorted(result)

    def _requests(self, name: str, request: dict, universe: list[str], calendar: list[date]):
        spec = dataset(name); start, end = parse_date(request["start_date"]), parse_date(request["end_date"])
        custom = request.get("parameters", {}).get(name, {})
        if request.get('codes'):
            parameter='index_code' if spec.strategy=='index' else 'ts_code'
            if custom.get(parameter) and set(str(custom[parameter]).split(','))!=set(request['codes']):raise ValueError('接口证券参数与任务证券范围不一致：'+name)
        if spec.strategy == "stock_basic":
            if request["codes"]:
                for code in request["codes"]:
                    for status in ([custom['list_status']] if 'list_status' in custom else ('L','D','P')):yield {**custom,'ts_code':code,'list_status':status}
            else:
                for status in ([custom["list_status"]] if "list_status" in custom else ("L","D","P")):
                    yield {"list_status":status,**custom}
        elif spec.strategy == "calendar":
            for exchange in ["SSE", "SZSE", "BSE"]:
                for year in range(start.year, end.year + 1): yield {"exchange": exchange, "start_date": max(start,date(year,1,1)).strftime("%Y%m%d"), "end_date": min(end,date(year,12,31)).strftime("%Y%m%d"), **custom}
        elif spec.strategy == "snapshot":
            if name=='etf_basic':
                if request['codes']:
                    for code in request['codes']:yield {'ts_code':code,**custom}
                else:yield dict(custom)
            elif name in {"fund_basic","index_basic"}:
                if request["codes"]:
                    for code in request["codes"]: yield {"ts_code":code,**custom}
                elif name=="fund_basic":
                    for status in ([custom['status']] if 'status' in custom else ['L','D','I']): yield {"market":"E","status":status,**custom}
                else:
                    for market in ([custom['market']] if 'market' in custom else ['SSE','SZSE','CSI','SW','CICC','OTH']): yield {"market":market,**custom}
            elif name in {"index_member_all","ci_index_member"}:
                for current in ([str(custom['is_new'])] if 'is_new' in custom else ['Y','N']):
                    if request['codes']:
                        for code in request['codes']: yield {"ts_code":code,"is_new":current,**custom}
                    else: yield {"is_new":current,**custom}
            else: yield {**({"src":"SW2021"} if name=="index_classify" else {}),**custom}
        elif spec.strategy=='periodic':
            periods=[custom['period']] if custom.get('period') else [date(year,month,day).strftime('%Y%m%d') for year in range(start.year,end.year+1) for month,day in ((3,31),(6,30),(9,30),(12,31)) if start<=date(year,month,day)<=end]
            for period in periods:yield {'period':period,'offset':0,'limit':spec.limit,**custom}
        elif spec.strategy == "daily":
            days = [d for d in calendar if start <= d <= end] if calendar else list(dates(start, end))
            for day in days:
                if request["codes"]:
                    for code in request["codes"]: yield {"trade_date": day.strftime("%Y%m%d"), "ts_code": code, **custom}
                else: yield {"trade_date": day.strftime("%Y%m%d"), **custom}
        else:
            codes = request["codes"]
            if name in {'index_daily','sw_daily'}:
                codes=[code for code in (codes or universe) if code.endswith('.SI') == (name=='sw_daily')]
                if not codes:return
            if not codes and name in {"income", "balancesheet", "cashflow", "fina_indicator", "dividend", "namechange"}: codes = universe
            if not codes and "ts_code" in custom: codes = [custom["ts_code"]]
            if not codes and name == "index_weight" and "index_code" in custom: codes = [custom["index_code"]]
            if not codes and name == "ci_index_member": yield {**custom}; return
            if not codes: raise ValueError(f"{name} 需要显式证券代码；不会自动下载全部基金或指数")
            if name=='namechange':
                # API dates filter announcements, while local rows describe
                # effective name intervals. Full single-security history is
                # bounded and avoids missing earlier announcements.
                for code in codes:yield {'ts_code':code,**custom}
                return
            for code in codes:
                if name in {'income','balancesheet','cashflow','fina_indicator'}:
                    if custom.get('period'):
                        yield {'ts_code':code,**custom}
                    else:
                        for year in range(start.year,end.year+1):
                            for month,day in ((3,31),(6,30),(9,30),(12,31)):
                                period=date(year,month,day)
                                if start<=period<=end:yield {'ts_code':code,'period':period.strftime('%Y%m%d'),**custom}
                    continue
                for year in range(start.year, end.year + 1):
                    params = {"index_code" if spec.strategy == "index" else "ts_code": code}
                    if "start_date" in spec.input_fields:
                        params.update(start_date=max(start, date(year,1,1)).strftime("%Y%m%d"), end_date=min(end,date(year,12,31)).strftime("%Y%m%d"))
                    elif "period" in spec.input_fields:
                        for month, day in [(3,31),(6,30),(9,30),(12,31)]:
                            period = date(year, month, day)
                            if start <= period <= end: yield {**params, "period": period.strftime("%Y%m%d"), **custom}
                        continue
                    else:
                        if year != start.year: continue
                    yield {**params, **custom}

    def _download_one(self, job: str, name: str, params: dict, universe: list[str]):
        self.state.check_cancel(job); spec = dataset(name); slice_id = digest([name, params])[:24]
        existing = self.state.get_slice(job, slice_id)
        if existing and existing['status']=='paged':
            read_archive(self.settings.state_dir/existing['archive'],existing['sha'])
            self._download_one(job,name,{**params,'offset':int(params.get('offset',0))+existing['rows']},universe)
            return
        if existing and existing["status"] in {"downloaded", "committed"}:
            read_archive(self.settings.state_dir / existing["archive"], existing["sha"]); return
        self.state.slice(job, slice_id, name, params)
        archive = (self.settings.archive_dir or self.settings.state_dir/"archive") / job / name / (slice_id + ".json.gz")
        fields = [field["name"] for field in spec.fields]
        data, sha = self.source.fetch(name, params, fields, self.settings.state_dir / archive)
        metadata=json.loads((self.settings.state_dir/archive).with_suffix('.meta.json').read_text())
        try:
            normalize_source(spec, {'fields': data['fields'], 'items': []}, job, slice_id, metadata['fetched_at'])
            capability_status, capability_error = 'available', None
        except ValueError as exc:
            capability_status, capability_error = 'contract_mismatch', str(exc)
        self.state.ledger_put('meta_source_capability', name, {
            'task_id': job, 'source_name': 'tushare', 'endpoint': name, 'capability': name,
            'status': capability_status, 'permission_note': '返回字段已实测；空响应不证明历史覆盖',
            'last_probe_at': metadata['fetched_at'], 'sample_rows': len(data['items']),
            'response_fields': ','.join(data['fields']), 'error': capability_error,
        })
        self.state.ledger_put('import_source_archive',dumps([job,name,digest(params)]),
                             {'task_id':job,'dataset':name,'endpoint':name,'request_hash':digest(params),'archive_path':str(archive),
                              'response_sha256':sha,'response_fields':','.join(data.get('fields',[])),
                              'fetched_at':metadata['fetched_at'],'rows':len(data.get('items',[]))})
        if (name in {'index_basic','fund_basic'} or name.endswith('_vip')) and len(data['items'])>=spec.limit and 'ts_code' not in params:
            # Test common API pagination rather than assuming it is honored.
            # Every page remains a distinct, auditable archive.
            if 'offset' not in params:
                self.state.slice(job,slice_id,name,params,status='split',archive=str(archive),sha=sha,rows=len(data['items']))
                self._download_one(job,name,{**params,'offset':0,'limit':spec.limit},universe)
                return
            self.state.slice(job,slice_id,name,params,status='paged',archive=str(archive),sha=sha,rows=len(data['items']))
            previous=set()
            current={tuple(str(v) for v in values) for values in data['items']}
            for part in self.state.job(job)['slices']:
                if part['dataset']!=name or part['id']==slice_id or part['status'] not in {'downloaded','paged'}:continue
                if part['request'].get('market')!=params.get('market') or part['request'].get('status')!=params.get('status') or part['request'].get('period')!=params.get('period'):continue
                payload=read_archive(self.settings.state_dir/part['archive'],part['sha'])
                previous.update(tuple(str(v) for v in values) for values in payload['items'])
            if current and current<=previous:raise ValueError(name+' 未按 offset 分页，已阻止将截断快照标为完整')
            self._download_one(job,name,{**params,'offset':int(params['offset'])+len(data['items'])},universe)
            return
        if len(data["items"]) >= spec.limit:
            self.state.slice(job, slice_id, name, params, status="split", archive=str(archive), sha=sha, rows=len(data["items"]))
            if name in {"index_member_all","ci_index_member"} and "ts_code" not in params:
                if name=='ci_index_member' and not any(key in params for key in ('l1_code','l2_code','l3_code')):
                    source_rows=[dict(zip(data['fields'],values)) for values in data['items']]
                    groups=sorted({r['l1_code'] for r in source_rows if r.get('l1_code')})
                    if groups:
                        for group in groups:self._download_one(job,name,{**params,'l1_code':group},universe)
                        # A truncated discovery response may omit entire groups.
                        # Explicitly query every security not covered by a child
                        # partition before declaring the snapshot downloaded.
                        covered=set()
                        for part in self.state.job(job)['slices']:
                            if part['dataset']!=name or part['status']!='downloaded' or part['request'].get('is_new')!=params.get('is_new'):continue
                            payload=read_archive(self.settings.state_dir/part['archive'],part['sha'])
                            if 'ts_code' in payload['fields']:
                                pos=payload['fields'].index('ts_code');covered.update(values[pos] for values in payload['items'])
                        if not universe:universe=self._universe(self.state.job(job))
                        for code in sorted(set(universe)-covered):self._download_one(job,name,{**params,'ts_code':code},universe)
                        return
                if name=="index_member_all" and not any(key in params for key in ('l1_code','l2_code','l3_code')):
                    classes=[]
                    for part in self.state.job(job)['slices']:
                        if part['dataset']=='index_classify' and part['status']=='downloaded':
                            payload=read_archive(self.settings.state_dir/part['archive'],part['sha'])
                            classes += [dict(zip(payload['fields'],values)) for values in payload['items']]
                    groups=sorted({r['index_code'] for r in classes if r.get('level')=='L1'})
                    if groups:
                        for group in groups: self._download_one(job,name,{**params,'l1_code':group},universe)
                        return
                if not universe:
                    try: universe=self._universe(self.state.job(job))
                    except ValueError: pass
                if universe:
                    for code in universe: self._download_one(job,name,{**params,'ts_code':code},universe)
                    return
            if name=='stock_basic' and 'exchange' not in params and 'ts_code' not in params:
                for exchange in ('SSE','SZSE','BSE'): self._download_one(job,name,{**params,'exchange':exchange},universe)
                return
            if "trade_date" in params and "ts_code" not in params and universe:
                for code in universe: self._download_one(job, name, {**params, "ts_code": code}, universe)
                return
            if "start_date" in params and params["start_date"] < params["end_date"]:
                start, end = parse_date(params["start_date"]), parse_date(params["end_date"]); mid = start + (end-start)//2
                self._download_one(job,name,{**params,"end_date":mid.strftime("%Y%m%d")},universe); self._download_one(job,name,{**params,"start_date":(mid+timedelta(days=1)).strftime("%Y%m%d")},universe); return
            raise ValueError(f"{name} 达到返回上限，须缩小参数范围")
        self.state.slice(job, slice_id, name, params, status="downloaded", archive=str(archive), sha=sha, rows=len(data["items"]))

    def download(self, job: str) -> dict:
        item = self._job(job,True)
        if item["status"] in {"committed", "completed", "rolled_back"}: return item
        self.state.update_job(job, status="downloading", error=None)
        self._task_status(job, "downloading")
        request = item["plan"]["request"]; calendar, universe = self._calendar(), request["codes"]
        try:
            for name in request["datasets"]:
                if name in {'index_daily','sw_daily'} and not request['codes']:
                    universe=self._universe(self.state.job(job),'index')
                if name not in {"stock_basic", "fund_basic", "index_basic", "trade_cal", "index_classify", "index_member_all", "ci_index_member"} and not universe:
                    try: universe = self._universe(self.state.job(job))
                    except ValueError:
                        if dataset(name).strategy not in {"daily", "snapshot","periodic"}: raise
                for params in self._requests(name, request, universe, calendar):
                    try: self._download_one(job,name,params,universe)
                    except Exception as exc:
                        if name.removesuffix('_vip') not in {"ci_index_member","income","balancesheet","cashflow","fina_indicator"} or not any(word in str(exc) for word in ("权限","积分","permission")): raise
                        self.state.slice(job,digest([name,params])[:24],name,params,status="unavailable",result={"reason":self.safe_error(exc)})
                        break
                if name == "trade_cal": calendar = self._job_calendar(self.state.job(job))
                if name in {"stock_basic", "fund_basic", "index_basic"}:
                    try: universe = self._universe(self.state.job(job))
                    except ValueError: pass
            self.state.update_job(job, status="downloaded"); self._task_status(job, "downloaded"); return self.state.job(job)
        except Exception as exc:
            status = "cancelled" if isinstance(exc, InterruptedError) else "failed"
            self.state.update_job(job, status=status, error=self.safe_error(exc)); self._task_status(job, status, self.safe_error(exc)); raise

    def _canonical_record(self, spec: CanonicalTable, partial: dict, endpoint: str, job: str, downloaded: str, row_no: int, raw: dict, event_date: date | None, available_at=None, availability_basis="unknown", source_name="tushare"):
        row = {field["name"]: None for field in spec.fields}; row.update(partial)
        if "event_date" in row: row["event_date"] = event_date
        if "available_at" in row: row["available_at"] = available_at
        if "downloaded_at" in row: row["downloaded_at"] = _datetime(downloaded)
        if "availability_basis" in row: row["availability_basis"] = availability_basis
        if "import_task_id" in row: row["import_task_id"] = job
        if "source_name" in row: row["source_name"] = source_name
        if "source_endpoint" in row: row["source_endpoint"] = endpoint
        if "source_row_no" in row: row["source_row_no"] = row_no
        if "source_record_hash" in row: row["source_record_hash"] = digest(raw)
        if "mapping_version" in row: row["mapping_version"] = VERSION
        if "data_quality_status" in row: row["data_quality_status"] = "accepted"
        return row

    def _map_part(self, endpoint: str, source_rows: list[dict], job: str, downloaded: str, calendar: list[date], source_name: str = "tushare"):
        if endpoint.endswith('_vip'):
            # Dispatch only the base projection here.  Calling self._map_part
            # would re-enter DatabaseManagerService and apply the YAML overlay
            # and version derivation twice for VIP rows.
            output,factors=TushareImportService._map_part(self,endpoint.removesuffix('_vip'),source_rows,job,downloaded,calendar,source_name)
            for rows in output.values():
                for row in rows:row['source_endpoint']=endpoint
            return output,factors
        output = defaultdict(list); factors = {}
        asset = "stock" if endpoint in {"daily", "daily_basic", "adj_factor", "suspend_d"} else "fund" if endpoint in {"fund_daily", "fund_adj"} else "index" if endpoint in {"index_daily", "index_weight", "sw_daily"} else None
        for source in source_rows:
            if source.get('ts_code') and str(source['ts_code']).split('.')[-1] not in {'SH','SZ','BJ','SI','CI','CSI','CNI','OF','WI','IB'}:continue
            if endpoint in {'stock_basic','daily','daily_basic','adj_factor','suspend_d','income','balancesheet','cashflow','fina_indicator'}:
                code=str(source.get('ts_code',''))
                if (code.startswith('900') and code.endswith('.SH')) or (code.startswith('200') and code.endswith('.SZ')):continue
            raw = source.get('_raw') or {key: value for key, value in source.items() if not key.startswith("_")}; row_no = source["_row"]
            event_date = next((source.get(key) for key in ("trade_date", "cal_date", "end_date", "in_date", "start_date", "nav_date") if source.get(key)), None)
            if not isinstance(event_date, date): event_date = parse_date(event_date) if event_date else None
            available, basis = _event_available(event_date)
            if source.get("ann_date") or source.get("f_ann_date"):
                announcement = source.get("f_ann_date") or source.get("ann_date"); available, basis = _event_available(announcement, "announcement_next_calendar_day_assumed;historical_revision_not_proven")
            if endpoint in {'income','balancesheet','cashflow','fina_indicator'} and not (source.get('ann_date') or source.get('f_ann_date')):
                available,basis=None,'unknown_announcement_date'
            if endpoint in {'index_member_all','ci_index_member'}:
                basis='membership_effective_date_assumed;historical_disclosure_unproven'
            if endpoint in {"daily", "fund_daily", "index_daily", "daily_basic", "sw_daily"}:
                target = canonical("market_daily_raw"); partial = {"asset_type": asset, "security_code": source.get("ts_code"), "trade_date": source.get("trade_date")}
                mapping = FIELD_MAP.get(endpoint, {})
                for field, value in source.items():
                    if field.startswith("_") or field not in mapping: continue
                    target_column = mapping[field][1]
                    if target_column in target.keys and target_column not in {"trade_date", "security_code"}: continue
                    partial[target_column] = value
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, event_date, available, basis, source_name)); continue
            if endpoint in {"adj_factor", "fund_adj"}:
                code, trade = source.get("ts_code"), source.get("trade_date"); factors[(asset, code, trade)] = source.get("adj_factor")
                target = canonical("corporate_action"); partial = {"security_code": code, "event_type": "adjustment_factor", "event_date": trade, "effective_date": trade, "adjustment_factor": source.get("adj_factor")}
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, trade, available, basis, source_name)); continue
            if endpoint in {"stock_basic", "fund_basic", "index_basic"}:
                target = canonical("security_dim"); partial = {"security_code": source.get("ts_code"), "asset_type": "stock" if endpoint == "stock_basic" else "fund" if endpoint == "fund_basic" else "index", "name": source.get("name"), "exchange": source.get("exchange") or {"SH":"SSE","SZ":"SZSE","BJ":"BSE"}.get(str(source.get("ts_code","")).split(".")[-1],source.get("market")), "list_date": source.get("list_date"), "delist_date": source.get("delist_date") or source.get("exp_date"), "status": source.get("list_status") or source.get("status")}
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, None, None, "source_snapshot", source_name)); continue
            if endpoint == "trade_cal":
                target = canonical("trade_calendar"); partial = {"exchange": source.get("exchange"), "trade_date": source.get("cal_date"), "is_open": int(source.get("is_open")) if source.get("is_open") is not None else None, "previous_trade_date": source.get("pretrade_date")}
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, source.get("cal_date"), available, basis, source_name)); continue
            if endpoint == "suspend_d":
                target = canonical("trade_status_daily"); is_tradable = None if source.get("suspend_timing") else 0 if source.get("suspend_type") == "S" else 1; partial = {"asset_type": "stock", "security_code": source.get("ts_code"), "trade_date": source.get("trade_date"), "is_tradable": is_tradable, "status_code": source.get("suspend_type"), "status_name": "盘中停复牌，日频状态待核验" if is_tradable is None else "停牌" if is_tradable == 0 else "复牌", "status_source": "tushare.suspend_d", "confidence": "authoritative_partial_day" if source.get("suspend_timing") else "authoritative"}
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, event_date, available, basis, source_name)); continue
            if endpoint == "index_weight":
                target = canonical("index_membership"); partial = {"index_code": source.get("index_code"), "security_code": source.get("con_code"), "trade_date": source.get("trade_date"), "weight": source.get("weight")}
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, event_date, available, basis, source_name)); continue
            if endpoint == "index_classify":
                scheme=source.get('src') or 'SW2021'
                scheme_spec=canonical('industry_scheme_dim')
                output[scheme_spec.name].append(self._canonical_record(scheme_spec,{'scheme_code':scheme,'scheme_name':'申万'+str(scheme).removeprefix('SW'),'description':'申万行业分类体系'},endpoint,job,downloaded,row_no,raw,None,None,'source_snapshot',source_name))
                target = canonical("industry_dim"); partial = {"scheme_code": scheme, "industry_code": source.get("index_code"), "industry_name": source.get("industry_name"), "parent_code": source.get("parent_code"), "level": source.get("level")}
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, None, None, "source_snapshot", source_name)); continue
            if endpoint in {"index_member_all", "ci_index_member"}:
                scheme = "SW2021" if endpoint == "index_member_all" else "CITIC"
                scheme_spec = canonical("industry_scheme_dim")
                output[scheme_spec.name].append(self._canonical_record(scheme_spec, {"scheme_code":scheme,"scheme_name":"申万2021" if scheme=="SW2021" else "中信","description":"申万行业分类体系" if scheme=='SW2021' else "中信行业分类体系"}, endpoint,job,downloaded,row_no,raw,None,None,"source_snapshot",source_name))
                for level in (1,2,3):
                    code, name = source.get(f"l{level}_code"), source.get(f"l{level}_name")
                    if not code: continue
                    dim = canonical("industry_dim")
                    dim_values = {"scheme_code":scheme,"industry_code":code,"industry_name":name or code,"parent_code":source.get(f"l{level-1}_code") if level>1 else None,"level":f"L{level}"}
                    output[dim.name].append(self._canonical_record(dim,dim_values,endpoint,job,downloaded,row_no,raw,None,None,"source_snapshot",source_name))
                    target = canonical("security_industry_membership")
                    partial = {"scheme_code":scheme,"security_code":source.get("ts_code"),"industry_code":code,
                               "valid_from":source.get("in_date"),"valid_to":source.get("out_date"),
                               "membership_status":"current" if source.get("is_new") in {"Y","1"} else "historical"}
                    output[target.name].append(self._canonical_record(target,partial,endpoint,job,downloaded,row_no,raw,event_date,available,basis,source_name))
                continue
            if endpoint in {"income", "balancesheet", "cashflow", "fina_indicator"}:
                target = canonical("fundamental_periodic")
                partial = financial_record(endpoint, source, FIELD_MAP[endpoint])
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, event_date, available, basis, source_name)); continue
            if endpoint in {"namechange", "dividend"}:
                partial = corporate_action_partial(endpoint, source)
                if partial is None: continue
                target = canonical("corporate_action")
                if endpoint == 'dividend' and source.get('imp_ann_date'):
                    available,basis = _event_available(parse_date(source['imp_ann_date']), 'implementation_announcement_next_calendar_day_assumed;historical_revision_not_proven')
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, partial['event_date'], available, basis, source_name)); continue
            if endpoint in {'fund_nav','fund_share','bse_mapping'}:
                mapping=FIELD_MAP[endpoint];target=canonical(next(iter(mapping.values()))[0])
                partial={dest[1]:source.get(field) for field,dest in mapping.items()}
                output[target.name].append(self._canonical_record(target,partial,endpoint,job,downloaded,row_no,raw,event_date,available,basis,source_name));continue
            if endpoint not in FIELD_MAP: raise ValueError(f'{endpoint} 尚无可入库字段映射；原始数据已归档')
        return output, factors

    def _merge(self, rows: list[dict], spec: CanonicalTable):
        result = {}
        for row in rows:
            key = tuple(row.get(k) for k in spec.keys)
            if key not in result: result[key] = dict(row); continue
            current = result[key]
            for name in columns(spec):
                if name not in PROVENANCE and name not in spec.keys and current.get(name) is not None and row.get(name) is not None and current[name]!=row[name]:
                    current['data_quality_status']='conflicting_source_rows'
                if current.get(name) is None and row.get(name) is not None: current[name] = row[name]
            current["source_endpoint"] = ",".join(sorted(set(filter(None, str(current.get("source_endpoint", "")).split(",") + [str(row.get("source_endpoint", ""))]))))
            current["source_record_hash"] = digest([current.get("source_record_hash"), row.get("source_record_hash")])
        return list(result.values())

    @high_precision
    def _build_canonical(self, item: dict):
        tables = defaultdict(list); factors = {}; rejects = []; calendar = self._job_calendar(item); download_stamp=item["created"]
        for part in item["slices"]:
            if part["status"] == "split" or not part["archive"]: continue
            data = read_archive(self.settings.state_dir / part["archive"], part["sha"])
            metadata = json.loads((self.settings.state_dir / part["archive"]).with_suffix(".meta.json").read_text())
            download_stamp=max(download_stamp,metadata['fetched_at'])
            if metadata.get('canonical_table'):
                from .sources import FileSourceAdapter,canonical_file_rules
                target=canonical(metadata['canonical_table'])
                rules=canonical_file_rules(target,metadata['column_map'])
                for index,values in enumerate(data['items'],1):
                    original=dict(zip(data['fields'],values))
                    try:
                        partial = FileSourceAdapter().normalize({'fields':data['fields'],'items':[values]}, rules)[0]
                        event=partial.get('event_date') or partial.get('trade_date') or partial.get('period_end') or partial.get('valid_from') or partial.get('nav_date')
                        event=parse_date(event) if event else None
                        row=self._canonical_record(target,partial,metadata.get('source_endpoint','file'),item['id'],metadata['fetched_at'],index+int(metadata.get('source_row_start',1))-1,original,event,partial.get('available_at'),partial.get('availability_basis') or 'unknown_file_availability','file')
                        normalized=typed(target,row)
                        row['_field_origins']={rule['column']:{'selected':normalized.get(rule['column']) is not None,'source_null':original.get(source) in (None,''),'value_hash':digest(normalized.get(rule['column'])),'origins':['file:'+target.name+':'+source]} for source,rule in rules.items()}
                        tables[target.name].append(row)
                    except (ValueError, InvalidOperation) as exc:
                        rejects.append({'dataset':target.name,'source_row':index,'reason':str(exc),'raw':original})
                continue
            try: source_rows, rejected = normalize_source(dataset(part["dataset"]), data, item["id"], part["id"], metadata["fetched_at"], calendar)
            except ValueError as exc:
                rejects.append({"dataset": part["dataset"], "reason": str(exc)}); continue
            rejects.extend({"dataset": part["dataset"], **entry} for entry in rejected)
            source_name = metadata.get("source_name") or ("file" if metadata.get("endpoint") == "file" else "tushare")
            if part['dataset'].endswith('_vip') and item['plan']['request']['codes']:
                scope=set(item['plan']['request']['codes']);source_rows=[r for r in source_rows if r.get('ts_code') in scope]
            mapped, found_factors = self._map_part(part["dataset"], source_rows, item["id"], metadata["fetched_at"], calendar, source_name)
            for name, rows in mapped.items(): tables[name].extend(rows)
            factors.update(found_factors)
        # Merge daily and daily_basic into a single raw wide row.
        if tables.get('fund_nav_daily'):
            from .nav_semantics import resolve_nav_rows
            tables['fund_nav_daily'],conflicts=resolve_nav_rows(tables['fund_nav_daily']);rejects.extend(conflicts)
        for name, spec in load_canonical_catalog().items():
            if name in tables: tables[name] = self._merge(tables[name], spec)
        raw_rows = tables.get("market_daily_raw", [])
        for row in raw_rows:
            row['average_price'] = daily_average_price(row.get('asset_type'), row.get('amount'), row.get('volume'))
        securities=list(tables.get('security_dim',[]))
        if not securities:
            for target_name in item['plan']['request']['targets']:
                target=self.pg if target_name=='postgres' else self.ch
                try:
                    page=None
                    while True:
                        result=target.query('security_dim',limit=1000,after=page)
                        securities.extend(result['rows']);page=result['next']
                        if not page:break
                    if securities:break
                except Exception:pass
        names = {(row.get("asset_type"), row.get("security_code")): row.get("name") for row in securities if row.get("name")}
        for raw in raw_rows:
            raw["security_name"] = raw.get("security_name") or names.get((raw.get("asset_type"), raw.get("security_code")))
        adjusted = []
        for raw in raw_rows:
            key = (raw.get("asset_type"), raw.get("security_code"), raw.get("trade_date")); factor = factors.get(key)
            row = dict(raw)
            for price in ("open","high","low","close","previous_close","change"):
                row["raw_"+price] = raw.get(price)
            row["adjustment_version"] = VERSION
            if raw.get("asset_type") == "index": method = "native_index_value"
            elif factor is not None: method = "tushare_adj_factor_post_adjusted"
            else:
                row.update(adjustment_factor=None,adjustment_method="unavailable_missing_factor",adjustment_formula="unavailable",data_quality_status="unavailable_missing_factor")
                for price in ("open","high","low","close","previous_close","change","average_price"): row[price] = None
                adjusted.append(row); continue
            if factor is not None:
                for field in ("open", "high", "low", "close", "previous_close", "change"):
                    if row.get(field) is not None: row[field] = Decimal(str(row[field])) * Decimal(str(factor))
            if row.get("average_price") is not None and factor is not None: row["average_price"] *= Decimal(str(factor))
            row["adjustment_factor"] = factor; row["adjustment_method"] = method; row["adjustment_formula"] = "raw_price * adjustment_factor" if factor is not None else "native_index_value"
            row["source_endpoint"] = ','.join(sorted(set(str(raw.get('source_endpoint') or '').split(',')+(['fund_adj' if raw.get('asset_type')=='fund' else 'adj_factor'] if factor is not None else [])))); row["source_record_hash"] = digest([raw.get("source_record_hash"), factor])
            adjusted.append(row)
        if adjusted: tables["market_daily_post_adjusted"] = self._merge(adjusted, canonical("market_daily_post_adjusted"))
        # Explicit stop/resume events and listing boundaries take precedence.
        status_spec=canonical('trade_status_daily')
        explicit={(r['asset_type'],r['security_code'],r['trade_date']):r for r in tables.get(status_spec.name,[])}
        open_dates=set(calendar)
        for raw in raw_rows:
            key=(raw['asset_type'],raw['security_code'],raw['trade_date'])
            if key in explicit:continue
            status={field:raw.get(field) for field in columns(status_spec)}
            has_trade=raw.get('volume') is not None and raw['volume']>0 and raw['trade_date'] in open_dates
            status.update(is_tradable=1 if has_trade else None,status_code='QUOTE_PRESENT' if has_trade else 'UNKNOWN_QUOTE_ONLY',
                          status_name='行情与成交量推断可交易' if has_trade else '行情存在但可交易性未确认',status_source='derived_from_daily_quote',
                          confidence='derived',source_endpoint='trade_cal+daily',source_record_hash=digest([raw.get('source_record_hash'),'derived_status']))
            explicit[key]=status
        selected=set(item['plan']['request']['datasets'])
        assets={asset for endpoint,asset in (('daily','stock'),('fund_daily','fund'),('index_daily','index'),('sw_daily','index'),('suspend_d','stock')) if endpoint in selected}
        scope=set(item['plan']['request'].get('codes',[]));start=parse_date(item['plan']['request']['start_date']);end=parse_date(item['plan']['request']['end_date'])
        for security in securities:
            asset=security.get('asset_type');code=security.get('security_code')
            if asset not in assets or (scope and code not in scope):continue
            for day in calendar:
                if not start<=day<=end:continue
                key=(asset,code,day)
                listed=security.get('list_date');delisted=security.get('delist_date')
                before_listing=listed and day<parse_date(listed);after_delisting=delisted and day>parse_date(delisted)
                if key in explicit and not (before_listing or after_delisting):continue
                value=0 if before_listing or after_delisting else None
                status_code='NOT_LISTED' if before_listing else 'DELISTED' if after_delisting else 'UNKNOWN_MISSING_QUOTE'
                status_name='尚未上市' if before_listing else '已退市' if after_delisting else '源行情缺失，不能认定停牌'
                available,basis=_event_available(day)
                explicit[key]=self._canonical_record(status_spec,{'asset_type':asset,'security_code':code,'trade_date':day,'is_tradable':value,
                                      'status_code':status_code,'status_name':status_name,'status_source':'security_lifecycle' if value==0 else 'derived_from_daily_quote',
                                      'confidence':'boundary' if value==0 else 'unknown'},'trade_cal+security_dim',item['id'],download_stamp,0,security,day,available,basis,'derived')
        tables[status_spec.name]=list(explicit.values())
        for name, spec in load_canonical_catalog().items():
            if name in tables: tables[name] = self._merge(tables[name], spec)
        validated = {}
        for name, rows in tables.items():
            spec = canonical(name); valid = [];lineage={}
            for row in rows:
                try:
                    if row.get('data_quality_status')=='conflicting_source_rows':raise ValueError('同一业务键的源数据字段互相冲突')
                    origins=row.get('_field_origins',{})
                    row = typed(spec,row)
                    scope_field=next((c for c in ('trade_date','period_end','nav_date','valid_from','event_date') if c in row),None)
                    if not row_in_scope(name,row,item['plan']['request']['start_date'],item['plan']['request']['end_date'],scope_field):continue
                    for field in spec.fields:
                        if (not field.get("nullable",True) or field["name"] in spec.keys) and row.get(field["name"]) is None:
                            raise ValueError("本地必需字段为空："+field["name"])
                    date_field=next((name for name in ('trade_date','period_end','valid_from') if name in row and row[name] is not None),None)
                    if date_field and 'event_date' in row and row['event_date']!=row[date_field]:raise ValueError('事件日期与业务日期错位')
                    if 'high' in row and row.get('high') is not None and row.get('low') is not None and row['high']<row['low']:raise ValueError('最高价低于最低价')
                    if row.get('volume') is not None and row['volume']<0:raise ValueError('成交量不能为负')
                    if row.get('valid_to') and row.get('valid_from') and row['valid_to']<row['valid_from']:raise ValueError('剔除日期早于纳入日期')
                    valid.append(row)
                    from .comparison import row_key
                    lineage[row_key(spec,row)]={c:p for c,p in origins.items() if p.get('source_null') or p.get('value_hash')==digest(row.get(c))}
                except (ValueError,TypeError) as exc: rejects.append({"dataset":name,"reason":str(exc),"source_row":row.get("source_row_no")})
            if valid:
                validated[name] = valid
                self.state.ledger_put('task_column_lineage',item['id']+':'+name,{'task_id':item['id'],'table':name,'rows':lineage})
        return validated, rejects

    def preview(self, job: str) -> dict:
        item = self._job(job)
        if item['status'] in {'completed','rolled_back','partially_rolled_back'}:
            if item.get('preview'):return item['preview']
            raise ValueError('已完成任务不能再次预览写入，请创建新任务')
        self._job(job,True)
        if item['plan'].get('mapping_version',VERSION)!=VERSION:raise ValueError('映射版本已变化，请使用归档建立新任务')
        for name,value in item['plan'].get('canonical_contracts',{}).items():
            if canonical(name).hash!=value:raise ValueError('本地字段合同已变化，请建立新任务')
        if any(part["status"] == "pending" for part in item["slices"]) or not item["slices"]: raise ValueError("下载尚未完整结束")
        if item["status"] in {"failed", "downloading", "planned", "cancelled"}: raise ValueError("请完成或恢复下载后再预览")
        tables, rejects = self._build_canonical(item); target_results = []; table_results = []
        source_report=[{'dataset':part['dataset'],'status':part['status'],'rows':part['rows'] or 0,'reason':(part.get('result') or {}).get('reason') or ('空响应不证明历史覆盖完整' if not part['rows'] else None),'parameters':part['request'],'archive':part.get('archive')}
                       for part in item['slices'] if part['status']!='split']
        if item['plan']['request'].get('strict_capabilities'):
            rejects.extend({'dataset':part['dataset'],'reason':'源接口不可用'} for part in source_report if part['status']=='unavailable')
        for table_name, rows in tables.items():
            spec = canonical(table_name)
            for target in item["plan"]["request"]["targets"]:
                comparator = self.pg if target == "postgres" else self.ch
                comparison = comparator.compare(spec, rows, item["plan"]["request"]["mode"])
                table_results.append({"target": target, "table": table_name, "signature": comparison["signature"], "counts": comparison["counts"], "change_sample": comparison["changes"][:20], "conflict_sample":comparison.get('conflicts',[])[:20], "normalized_sample": rows[:10]})
        field_validation=[]
        for part in item['slices']:
            if not part['archive'] or part['status']=='split':continue
            data=read_archive(self.settings.state_dir/part['archive'],part['sha'])
            if item['plan']['request'].get('sources')==['file']:continue
            expected={field['name'] for field in dataset(part['dataset']).fields};actual=set(data['fields'])
            for field in sorted(expected|actual):field_validation.append({'dataset':part['dataset'],'field':field,'present':field in actual,'registered':field in expected,
                                                                 'status':'unregistered_blocked' if field not in expected else 'missing_optional' if field not in actual else 'matched'})
        preview = {"field_validation":field_validation, "job": job, "targets": item["plan"]["request"]["targets"], "mapping_version": VERSION, "rejected": len(rejects), "reject_sample": rejects[:100], "tables": table_results, "source_report":source_report}
        preview["token"] = digest(preview); self.state.update_job(job, status="blocked" if rejects else "previewed", preview=preview)
        return preview

    def commit(self, job: str, preview_token: str) -> dict:
        from .publishing import publish
        return publish(self,job,preview_token)

    def rollback(self, job: str, targets: list[str] | None = None) -> dict:
        from .publishing import rollback
        return rollback(self,job,targets)

    def audit(self,job,after=0,limit=100):
        rows=self.state.audits(job,after=after,limit=limit)
        receipts=self.state.ledger_rows('import_target_status',job)
        if not rows and not receipts:
            try: return {'changes':self.pg.audit(job,after,limit),'receipts':self.pg.receipts(job),'rollbacks':[]}
            except Exception: pass
        table_status=[]
        item=self.state.job(job)
        for target in item['plan']['request']['targets']:
            for spec in load_canonical_catalog().values():
                record=self.state.table_commit_summary(job,target,spec.name)
                if record:table_status.append({'target':target,'table':spec.name,'status':record['status'],**record.get('counts',{})})
        return {'tables':table_status,'changes':[r for r in rows if r['id']>after][:max(1,min(limit,1000))],
                'receipts':receipts,'rollbacks':self.state.ledger_rows('import_rollback',job),
                'audit_sync':[r for task in {job}|{v['rollback_task_id'] for v in self.state.ledger_rows('import_rollback',job)} for r in self.state.sync_records(task)],
                'cleanup':[r for task in {v['rollback_task_id'] for v in self.state.ledger_rows('import_rollback',job)} for r in self.state.ledger_rows('cleanup_pending',task)],
                'verification':self.state.ledger_rows('verification',job),
                'field_verification':self.state.ledger_rows('field_verification',job)}

    def resume(self, job: str) -> dict:
        item = self._job(job)
        if item['status'] in {'completed','audit_pending','rolled_back','partially_rolled_back'}:
            from .publishing import retry_audit_sync
            retry_audit_sync(self,job);return self.state.job(job)
        self.state.update_job(job,cancel=0)
        if item["status"] in {"partial", "committing"} and item.get("preview"):
            return self.commit(job, item["preview"]["token"])
        self.download(job); self.preview(job); return self.state.job(job)

    def probe(self, names: list[str]) -> dict:
        import uuid
        result={};root=self.settings.state_dir/'probes'/uuid.uuid4().hex
        root.mkdir(parents=True)
        recent=datetime.now(SHANGHAI).date()-timedelta(days=1)
        while recent.weekday()>=5:recent-=timedelta(days=1)
        for name in names:
            spec=dataset(name)
            if name in {'daily','daily_basic','adj_factor','suspend_d','fund_daily','fund_adj','index_daily'}:
                code='510300.SH' if name.startswith('fund_') else '000300.SH' if name=='index_daily' else '000001.SZ'
                params={'trade_date':recent.strftime('%Y%m%d'),'ts_code':code}
            elif name in {'index_member_all','ci_index_member'}:params={'ts_code':'000001.SZ','is_new':'Y'}
            elif name.removesuffix('_vip') in {'income','balancesheet','cashflow','fina_indicator'}:params={'ts_code':'000001.SZ','period':f'{recent.year-1}1231'}
            elif name=='trade_cal':params={'exchange':'SSE','start_date':recent.strftime('%Y%m%d'),'end_date':recent.strftime('%Y%m%d')}
            elif name=='fund_basic':params={'ts_code':'510300.SH'}
            elif name=='index_basic':params={'ts_code':'000300.SH'}
            elif name=='stock_basic':params={'ts_code':'000001.SZ'}
            else:params={}
            try:
                data,sha=self.source.fetch(name,params,[f['name'] for f in spec.fields],root/(name+'.json.gz'))
                expected={f['name'] for f in spec.fields};actual=set(data['fields'])
                normalize_source(spec,data,'probe','probe',now())
                result[name]={'available':True,'sample_rows':len(data['items']),'response_fields':data['fields'],
                              'missing_fields':sorted(expected-actual),'extra_fields':sorted(actual-expected),'sha256':sha,
                              'sample_evidence':'non_empty' if data['items'] else 'empty_response_only','archive':str(root/(name+'.json.gz'))}
            except Exception as exc:result[name]={'available':False,'error':self.safe_error(exc)}
            value=result[name]
            record={'source_name':'tushare','endpoint':name,'capability':name,'status':'available' if value['available'] else 'unavailable',
                    'permission_note':'空响应仅验证字段合同' if value.get('sample_rows')==0 else None,'last_probe_at':now(),
                    'sample_rows':value.get('sample_rows'),'response_fields':','.join(value.get('response_fields',[])),'error':value.get('error')}
            self.state.ledger_put('meta_source_capability',name,record)
            for target in (self.pg,self.ch):
                try:target.control_write('meta_source_capability',[record])
                except Exception:pass
        return result
