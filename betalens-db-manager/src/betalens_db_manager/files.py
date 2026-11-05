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
    if not canonical_file:
        from .execution_policy import LegacyPipelineError
        raise LegacyPipelineError('文件任务须选择已登记的本地表；旧来源表执行入口已移除')
    spec=canonical(name) if canonical_file else dataset(name)
    if not column_map: raise ValueError("文件导入必须提供显式列映射")
    path = path.expanduser().resolve(strict=True)
    if not path.is_file(): raise ValueError("输入必须是文件")
    mapping = column_map or {}
    from .pipeline_cache import file_sha
    source_hash = file_sha(path)
    plan = dict(request=dict(datasets=[name], start_date="1900-01-01", end_date="2200-01-01", codes=[], mode=mode, parameters={}, targets=targets or ["postgres", "clickhouse"], sources=["file"]),
                contracts={name: spec.hash}, target=service.settings.fingerprint,
                file=dict(name=path.name, sha256=source_hash, column_map=mapping, encoding=encoding), revision="1")
    plan['file']['path']=str(path)
    from .pipeline_options import options
    from .mapping import VERSION
    from .processors import VERSION as PROCESSOR_VERSION
    plan['pipeline_version']=2;plan['revision']='file-v2'
    plan['request'].update(options({'auto_commit':False}),tables=[name],requests_per_minute=service.settings.requests_per_minute)
    plan.update(canonical_contracts={name:spec.hash},mapping_version=VERSION,processor_version=PROCESSOR_VERSION,
                mapping_snapshots={},source_contracts={},retrieval_policies={},requests=[])
    if job_id:
        item=service._job(job_id,True)
        if item['plan']['file']['sha256']!=source_hash:raise ValueError('源文件已变化，请创建新任务')
    else:item = service.state.create_job(plan, service.settings.fingerprint)
    job = item["id"]
    service._task_status(job,'planned')
    if defer:
        service.state.enqueue(job,'pipeline_run')
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
    root.mkdir(parents=True,exist_ok=True)
    stored = root / ("source" + "".join(path.suffixes))
    shutil.copyfile(path, stored)
    if file_sha(stored) != source_hash: raise ValueError("复制期间源文件发生变化")
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
        service.state.ledger_put('file_ingestion',job,{'task_id':job,'complete':True,'parts':counter,'source_sha256':source_hash,'source_path':str(stored)})
        service.state.update_job(job, status="downloaded")
    except Exception as exc:
        service.state.update_job(job, status="failed", error=service.safe_error(exc))
        raise
    return service.state.job(job)


def download_file(service,job):
    item=service._job(job,True);v=item['plan']['file'];request=item['plan']['request']
    return import_file(service,request['datasets'][0],Path(v['path']),v['column_map'],request['mode'],v.get('encoding','utf-8-sig'),request['targets'],job_id=job)


def verify_file_cache(service,item):
    """Verify immutable file evidence even when a normalized cache is reusable."""
    from .pipeline_cache import file_sha
    from .source import read_archive
    proof=service.state.ledger_get('file_ingestion',item['id']) or {}
    parts=item['slices'];definition=item['plan']['file']
    if not proof.get('complete') or len(parts)!=proof.get('parts') or proof.get('source_sha256')!=definition['sha256']:
        raise ValueError('文件归档完成凭证不完整')
    if sorted(p['id'] for p in parts)!=[f'file-{i:08d}' for i in range(proof['parts'])]:raise ValueError('文件分片编号不连续')
    source=Path(proof['source_path']) if proof.get('source_path') else Path(parts[0]['archive']).parent/('source'+''.join(Path(definition['name']).suffixes))
    if file_sha(source)!=definition['sha256']:raise ValueError('源文件归档哈希不符')
    for part in parts:
        service.state.check_cancel(item['id'])
        data=read_archive(Path(part['archive']),part['sha'])
        meta=json.loads(Path(part['archive']).with_suffix('.meta.json').read_text())
        if part['status']!='downloaded' or meta.get('sha256')!=part['sha'] or meta.get('source_sha256')!=definition['sha256']:
            raise ValueError('文件分片证据不一致')
        if data.get('fields')!=list(definition['column_map']) or len(data.get('items',[]))!=part['rows'] or any(len(r)!=len(data['fields']) for r in data['items']):
            raise ValueError('文件分片字段或行宽不一致')
