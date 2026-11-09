"""Shared normalization, connection setup and historical audit readers.

Task execution belongs exclusively to Pipeline; this module cannot start,
commit, resume or roll back an import.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from zoneinfo import ZoneInfo

from .canonical import CanonicalTable, canonical, load_canonical_catalog, FINANCIAL_ALIASES
from .comparison import typed, row_key, PROVENANCE, quote_close_precision_matches
from .catalog import Dataset, dataset, load_catalog, optional_source_fields
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
    required = set(spec.keys) - {"out_date", "f_ann_date", "suspend_timing"} - optional_source_fields(spec.name)
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
            if any(row.get(key) is None for key in set(spec.keys)-{'out_date','f_ann_date','suspend_timing'}-optional_source_fields(spec.name)): raise ValueError("源数据业务主键为空")
            key = tuple(row.get(k) for k in spec.keys)
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


class ManagerKernel:
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


    def _job_calendar(self, item: dict) -> list[date]:
        result = set(self._calendar())
        for part in item["slices"]:
            if part["dataset"] != "trade_cal" or part["status"] == "split" or not part["archive"]: continue
            data = read_archive(self.settings.state_dir / part["archive"], part["sha"])
            for values in data["items"]:
                row = dict(zip(data["fields"], values))
                if row.get("exchange") == "SSE" and str(row.get("is_open")) == "1": result.add(parse_date(row["cal_date"]))
        return sorted(result)




    def _canonical_record(self, spec: CanonicalTable, partial: dict, endpoint: str, job: str, downloaded: str, row_no: int, raw: dict, event_date: date | None, available_at=None, availability_basis="unknown", source_name="tushare", *, source_hash=None):
        row = {field["name"]: None for field in spec.fields}; row.update(partial)
        if "event_date" in row: row["event_date"] = event_date
        if "available_at" in row: row["available_at"] = available_at
        if "downloaded_at" in row: row["downloaded_at"] = _datetime(downloaded)
        if "availability_basis" in row: row["availability_basis"] = availability_basis
        if "import_task_id" in row: row["import_task_id"] = job
        if "source_name" in row: row["source_name"] = source_name
        if "source_endpoint" in row: row["source_endpoint"] = endpoint
        if "source_row_no" in row: row["source_row_no"] = row_no
        if "source_record_hash" in row: row["source_record_hash"] = source_hash if source_hash is not None else digest(raw)
        if "mapping_version" in row: row["mapping_version"] = VERSION
        if "data_quality_status" in row: row["data_quality_status"] = "accepted"
        return row

    def _map_part(self, endpoint: str, source_rows: list[dict], job: str, downloaded: str, calendar: list[date], source_name: str = "tushare"):
        if endpoint.endswith('_vip'):
            # Dispatch only the base projection here.  Calling self._map_part
            # would re-enter DatabaseManagerService and apply the YAML overlay
            # and version derivation twice for VIP rows.
            output,factors=ManagerKernel._map_part(self,endpoint.removesuffix('_vip'),source_rows,job,downloaded,calendar,source_name)
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
            source_hash=digest(raw)
            event_date = next((source.get(key) for key in ("trade_date", "cal_date", "end_date", "in_date", "start_date", "nav_date") if source.get(key)), None)
            if not isinstance(event_date, date): event_date = parse_date(event_date) if event_date else None
            available, basis = _event_available(event_date)
            if source.get("ann_date") or source.get("f_ann_date"):
                announcement = source.get("f_ann_date") or source.get("ann_date"); available, basis = _event_available(announcement, "announcement_next_calendar_day_assumed;historical_revision_not_proven")
            if endpoint in {'income','balancesheet','cashflow','fina_indicator','fund_nav'} and not (source.get('ann_date') or source.get('f_ann_date')):
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
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, event_date, available, basis, source_name,source_hash=source_hash)); continue
            if endpoint in {"adj_factor", "fund_adj"}:
                code, trade = source.get("ts_code"), source.get("trade_date"); factors[(asset, code, trade)] = source.get("adj_factor")
                target = canonical("corporate_action"); partial = {"security_code": code, "event_type": "adjustment_factor", "event_date": trade, "effective_date": trade, "adjustment_factor": source.get("adj_factor")}
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, trade, available, basis, source_name,source_hash=source_hash)); continue
            if endpoint in {"stock_basic", "fund_basic", "index_basic"}:
                target = canonical("security_dim"); partial = {"security_code": source.get("ts_code"), "asset_type": "stock" if endpoint == "stock_basic" else "fund" if endpoint == "fund_basic" else "index", "name": source.get("name"), "exchange": source.get("exchange") or {"SH":"SSE","SZ":"SZSE","BJ":"BSE"}.get(str(source.get("ts_code","")).split(".")[-1],source.get("market")), "list_date": source.get("list_date"), "delist_date": source.get("delist_date") or source.get("exp_date"), "status": source.get("list_status") or source.get("status")}
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, None, None, "source_snapshot", source_name,source_hash=source_hash)); continue
            if endpoint == "trade_cal":
                target = canonical("trade_calendar"); partial = {"exchange": source.get("exchange"), "trade_date": source.get("cal_date"), "is_open": int(source.get("is_open")) if source.get("is_open") is not None else None, "previous_trade_date": source.get("pretrade_date")}
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, source.get("cal_date"), available, basis, source_name,source_hash=source_hash)); continue
            if endpoint == "suspend_d":
                target = canonical("trade_status_daily"); is_tradable = None if source.get("suspend_timing") else 0 if source.get("suspend_type") == "S" else 1; partial = {"asset_type": "stock", "security_code": source.get("ts_code"), "trade_date": source.get("trade_date"), "is_tradable": is_tradable, "status_code": source.get("suspend_type"), "status_name": "盘中停复牌，日频状态待核验" if is_tradable is None else "停牌" if is_tradable == 0 else "复牌", "status_source": "tushare.suspend_d", "confidence": "authoritative_partial_day" if source.get("suspend_timing") else "authoritative"}
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, event_date, available, basis, source_name,source_hash=source_hash)); continue
            if endpoint == "index_weight":
                target = canonical("index_membership"); partial = {"index_code": source.get("index_code"), "security_code": source.get("con_code"), "trade_date": source.get("trade_date"), "weight": source.get("weight")}
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, event_date, available, basis, source_name,source_hash=source_hash)); continue
            if endpoint == "index_classify":
                scheme=source.get('src') or 'SW2021'
                scheme_spec=canonical('industry_scheme_dim')
                output[scheme_spec.name].append(self._canonical_record(scheme_spec,{'scheme_code':scheme,'scheme_name':'申万'+str(scheme).removeprefix('SW'),'description':'申万行业分类体系'},endpoint,job,downloaded,row_no,raw,None,None,'source_snapshot',source_name,source_hash=source_hash))
                target = canonical("industry_dim"); partial = {"scheme_code": scheme, "industry_code": source.get("index_code"), "industry_name": source.get("industry_name"), "parent_code": source.get("parent_code"), "level": source.get("level")}
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, None, None, "source_snapshot", source_name,source_hash=source_hash)); continue
            if endpoint in {"index_member_all", "ci_index_member"}:
                scheme = "SW2021" if endpoint == "index_member_all" else "CITICS"
                scheme_spec = canonical("industry_scheme_dim")
                output[scheme_spec.name].append(self._canonical_record(scheme_spec, {"scheme_code":scheme,"scheme_name":"申万2021" if scheme=="SW2021" else "中信","description":"申万行业分类体系" if scheme=='SW2021' else "中信行业分类体系"}, endpoint,job,downloaded,row_no,raw,None,None,"source_snapshot",source_name,source_hash=source_hash))
                for level in (1,2,3):
                    code, name = source.get(f"l{level}_code"), source.get(f"l{level}_name")
                    if not code: continue
                    dim = canonical("industry_dim")
                    dim_values = {"scheme_code":scheme,"industry_code":code,"industry_name":name or code,"parent_code":source.get(f"l{level-1}_code") if level>1 else None,"level":f"L{level}"}
                    output[dim.name].append(self._canonical_record(dim,dim_values,endpoint,job,downloaded,row_no,raw,None,None,"source_snapshot",source_name,source_hash=source_hash))
                    target = canonical("security_industry_membership")
                    partial = {"scheme_code":scheme,"security_code":source.get("ts_code"),"industry_code":code,
                               "valid_from":source.get("in_date"),"valid_to":source.get("out_date"),
                               "membership_status":"current" if source.get("is_new") in {"Y","1"} else "historical"}
                    output[target.name].append(self._canonical_record(target,partial,endpoint,job,downloaded,row_no,raw,event_date,available,basis,source_name,source_hash=source_hash))
                continue
            if endpoint in {"income", "balancesheet", "cashflow", "fina_indicator"}:
                target = canonical("fundamental_periodic")
                partial = financial_record(endpoint, source, FIELD_MAP[endpoint])
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, event_date, available, basis, source_name,source_hash=source_hash)); continue
            if endpoint in {"namechange", "dividend"}:
                partial = corporate_action_partial(endpoint, source)
                if partial is None: continue
                target = canonical("corporate_action")
                if endpoint == 'dividend' and source.get('imp_ann_date'):
                    available,basis = _event_available(parse_date(source['imp_ann_date']), 'implementation_announcement_next_calendar_day_assumed;historical_revision_not_proven')
                output[target.name].append(self._canonical_record(target, partial, endpoint, job, downloaded, row_no, raw, partial['event_date'], available, basis, source_name,source_hash=source_hash)); continue
            if endpoint in {'fund_nav','fund_share','bse_mapping'}:
                mapping=FIELD_MAP[endpoint];target=canonical(next(iter(mapping.values()))[0])
                partial={dest[1]:source.get(field) for field,dest in mapping.items()}
                output[target.name].append(self._canonical_record(target,partial,endpoint,job,downloaded,row_no,raw,event_date,available,basis,source_name,source_hash=source_hash));continue
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
                    if quote_close_precision_matches(spec,name,current,row):
                        # The price feed wins regardless of archive ordering.
                        if 'daily' in str(row.get('source_endpoint') or '').split(','):current[name]=row[name]
                    else:current['data_quality_status']='conflicting_source_rows'
                if current.get(name) is None and row.get(name) is not None: current[name] = row[name]
            current["source_endpoint"] = ",".join(sorted(set(filter(None, str(current.get("source_endpoint", "")).split(",") + [str(row.get("source_endpoint", ""))]))))
            current["source_record_hash"] = digest([current.get("source_record_hash"), row.get("source_record_hash")])
        return list(result.values())

    @high_precision
    def _build_canonical(self, item: dict):
        from .pipeline_options import LIGHT,DOMAIN_CHECKS
        light=LIGHT.get() or not DOMAIN_CHECKS.get()
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
        return self._assemble_canonical(item,tables,factors,rejects,calendar,download_stamp)

    @high_precision
    def _assemble_canonical(self,item,tables,factors,rejects,calendar,download_stamp):
        from .pipeline_options import LIGHT,DOMAIN_CHECKS
        light=LIGHT.get() or not DOMAIN_CHECKS.get()
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
                    if not light and 'high' in row and row.get('high') is not None and row.get('low') is not None and row['high']<row['low']:raise ValueError('最高价低于最低价')
                    if not light and row.get('volume') is not None and row['volume']<0:raise ValueError('成交量不能为负')
                    if not light and row.get('valid_to') and row.get('valid_from') and row['valid_to']<row['valid_from']:raise ValueError('剔除日期早于纳入日期')
                    valid.append(row)
                    from .comparison import row_key
                    if not LIGHT.get():
                        lineage[row_key(spec,row)]={c:p for c,p in origins.items() if p.get('source_null') or p.get('value_hash')==digest(row.get(c))}
                except (ValueError,TypeError) as exc: rejects.append({"dataset":name,"reason":str(exc),"source_row":row.get("source_row_no")})
            if valid:
                validated[name] = valid
                if not LIGHT.get():self.state.ledger_put('task_column_lineage',item['id']+':'+name,{'task_id':item['id'],'table':name,'rows':lineage})
        return validated, rejects




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
