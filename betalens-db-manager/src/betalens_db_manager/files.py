"""Explicit file-to-contract ingestion; source values use the documented source units."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from .catalog import dataset
from .canonical import canonical, load_canonical_catalog
from .comparison import PROVENANCE
from .util import atomic_json, digest, dumps, now


def import_file(service, *args, **kwargs):
    from .canonical import registry_context
    with registry_context(getattr(service,'registry',None)):
        return _import_file(service,*args,**kwargs)


def _import_file(service, name: str, path: Path, column_map: dict | None = None, mode="fill_missing", encoding="utf-8-sig", targets: list[str] | None = None, defer: bool = False, job_id: str | None = None) -> dict:
    from .sources import FileSourceAdapter
    FileSourceAdapter().plan({'path':str(path),'column_map':column_map})
    canonical_file=name in load_canonical_catalog()
    spec=canonical(name) if canonical_file else dataset(name)
    if not column_map: raise ValueError("文件导入必须提供显式列映射")
    path = path.expanduser().resolve(strict=True)
    if not path.is_file(): raise ValueError("输入必须是文件")
    mapping = column_map or {}
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = dict(request=dict(datasets=[name], start_date="1900-01-01", end_date="2200-01-01", codes=[], mode=mode, parameters={}, targets=targets or ["postgres", "clickhouse"], sources=["file"]),
                contracts={name: spec.hash}, target=service.settings.fingerprint,
                file=dict(name=path.name, sha256=source_hash, column_map=mapping, encoding=encoding), revision="1")
    plan['file']['path']=str(path)
    if job_id:
        item=service._job(job_id,True)
        if item['plan']['file']['sha256']!=source_hash:raise ValueError('源文件已变化，请创建新任务')
    else:item = service.state.create_job(plan, service.settings.fingerprint)
    job = item["id"]
    service._task_status(job,'planned')
    if defer:
        service.state.enqueue(job,'file_download')
        return service.state.job(job)
    from .mapping import VERSION
    for field,rule in mapping.items():
        column=rule if isinstance(rule,str) else rule['column']
        target_field=next((f for f in spec.fields if f['name']==column),None)
        if not target_field:raise ValueError('列映射含未知目标字段')
        record={'task_id':job,'source_name':'file','source_endpoint':'file:'+name+':'+source_hash[:12],'source_field':field,
                'canonical_table':name if canonical_file else '__via_source_contract__','canonical_column':column,
                'source_type':'text','target_type':target_field['type'],'source_unit':rule.get('source_unit','') if isinstance(rule,dict) else '',
                'target_unit':target_field.get('unit',''),'transform_expression':'Decimal(value)*'+str(rule.get('multiplier','1')) if isinstance(rule,dict) else 'identity',
                'date_role':'event_date' if column in {'event_date','trade_date','period_end','nav_date'} else '',
                'required':int(not target_field.get('nullable',True) or column in spec.keys),'mapping_version':VERSION,'description':'用户显式文件映射'}
        service.state.ledger_put('meta_source_mapping',digest([job,field]),record)
    root = (service.settings.archive_dir or service.settings.state_dir/"archive") / job / name
    root.mkdir(parents=True)
    stored = root / ("source" + "".join(path.suffixes))
    shutil.copyfile(path, stored)
    if hashlib.sha256(stored.read_bytes()).hexdigest() != source_hash: raise ValueError("复制期间源文件发生变化")
    fields = list(mapping) if canonical_file else [f["name"] for f in spec.fields]
    source_names={key:value if isinstance(value,str) else value.get('column') for key,value in mapping.items()}
    if set(source_names.values()) - {f['name'] for f in spec.fields}: raise ValueError("列映射含未知目标字段")

    def records():
        yield from FileSourceAdapter.rows(stored,encoding)

    batch, counter, observed = [], 0, now()
    def flush():
        nonlocal batch, counter
        key = f"file-{counter:08d}"
        archive = root / (key + ".json.gz")
        raw = dumps(dict(code=0, data=dict(fields=fields, items=batch))).encode()
        with gzip.open(archive, "wb") as handle: handle.write(raw)
        sha = hashlib.sha256(raw).hexdigest()
        atomic_json(archive.with_suffix(".meta.json"), dict(source_name="file", endpoint="file", source_endpoint='file:'+name+':'+source_hash[:12], canonical_table=name if canonical_file else None, column_map=mapping, fetched_at=observed, sha256=sha, source_sha256=source_hash, source_row_start=counter*5000+1))
        service.state.slice(job, key, name, {"file": path.name, "part": counter}, status="downloaded",
                            archive=str(archive), sha=sha, rows=len(batch))
        service.state.ledger_put('import_source_archive',digest([job,key]),{'task_id':job,'dataset':name,'endpoint':'file:'+name+':'+source_hash[:12],
                                 'request_hash':digest([source_hash,counter]),'archive_path':str(archive),'response_sha256':sha,'response_fields':','.join(fields),'fetched_at':observed,'rows':len(batch)})
        counter += 1; batch = []
    try:
        for record in records():
            unknown=set(record)-set(mapping)
            if unknown:raise ValueError("源文件含未映射列："+','.join(sorted(str(k) for k in unknown)))
            mapped = {source_names[k]:v for k,v in record.items()}
            if not canonical_file and set(mapped)-set(fields):raise ValueError("源文件含未映射列")
            batch.append([record.get(key) for key in fields] if canonical_file else [mapped.get(key) for key in fields])
            if len(batch) >= 5000: flush()
        if batch or counter == 0: flush()
        service.state.update_job(job, status="downloaded")
    except Exception as exc:
        service.state.update_job(job, status="failed", error=service.safe_error(exc))
        raise
    return service.state.job(job)


def download_file(service,job):
    item=service._job(job,True);v=item['plan']['file'];request=item['plan']['request']
    return import_file(service,request['datasets'][0],Path(v['path']),v['column_map'],request['mode'],v.get('encoding','utf-8-sig'),request['targets'],job_id=job)
