"""Shared typed comparison for every canonical target and audit hash."""
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext

from .util import digest

PROVENANCE = {'ingested_at','schema_version','source_id','downloaded_at','available_at','availability_basis','import_task_id','source_name',
              'source_endpoint','source_row_no','source_record_hash','mapping_version','data_quality_status'}


def typed(spec, row):
    result = {}
    for field in spec.fields:
        name, kind = field['name'], field['type']
        value = row.get(name)
        if value is None:
            result[name] = None
        elif kind == 'date':
            result[name] = value if isinstance(value, date) and not isinstance(value, datetime) else date.fromisoformat(str(value)[:10])
        elif kind == 'timestamp':
            value = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z','+00:00'))
            result[name] = (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
        elif kind == 'numeric':
            with localcontext() as context:
                context.prec = 80
                number = Decimal(str(value))
                precision=field.get('precision',38); scale=field.get('scale',12)
                if not number.is_finite() or abs(number) >= Decimal(10)**(precision-scale):
                    raise ValueError(f'{name}: 超出 Decimal(38,12) 范围')
                result[name] = number.quantize(Decimal(10)**(-scale))
        elif kind == 'integer':
            number = Decimal(str(value))
            if number != int(number): raise ValueError(f'{name}: 需要整数')
            result[name] = int(number)
        else:
            result[name] = str(value)
    from .canonical import builtin_catalog
    if spec.name not in builtin_catalog():return result
    import re
    for name in ('security_code','previous_code','index_code'):
        value=result.get(name)
        if value is not None and not re.fullmatch(r'[A-Z0-9_]+\.(SH|SZ|BJ|SI|CI|CSI|CNI|OF|WI|IB)',value):
            raise ValueError(f'{name}: 需要带境内市场后缀的标准证券代码，不能丢失前导零')
    if result.get('asset_type') is not None and result['asset_type'] not in {'stock','fund','index'}:
        raise ValueError('asset_type 不是已登记的资产类型')
    return result


def row_hash(spec, row):
    return digest({k:v for k,v in typed(spec,row).items() if k!="ingested_at"})


def row_key(spec, row):
    normalized = typed(spec, row)
    return digest([normalized.get(key) for key in spec.keys])


def compare(spec, rows, existing, mode):
    if mode not in {'fill_missing','refresh_source'}: raise ValueError('不支持的写入模式')
    old = {tuple(typed(spec, value).get(key) for key in spec.keys): typed(spec, value) for value in existing}
    changes, snapshots, conflicts = [], [], []
    counts = {'inserted':0,'updated':0,'skipped':0,'conflicts':0}
    business = [f['name'] for f in spec.fields if f['name'] not in PROVENANCE]
    for incoming in rows:
        incoming = typed(spec, incoming)
        key = tuple(incoming.get(k) for k in spec.keys)
        if any(value is None for value in key): raise ValueError(f'{spec.name}: 主键不能为空 {spec.keys}')
        before = old.get(key)
        snapshots.append([row_key(spec,incoming), row_hash(spec,before) if before else None])
        after = dict(incoming)
        if before is None:
            counts['inserted'] += 1
        else:
            conflict_fields=[k for k in business if incoming.get(k) is not None and before.get(k) is not None and incoming[k]!=before[k]]
            if conflict_fields:
                counts['conflicts']+=1
                conflicts.append({'row_key':row_key(spec,incoming),'before':before,'incoming':incoming,'fields':conflict_fields})
            for name in business:
                if (mode == 'fill_missing' and before.get(name) is not None) or (incoming.get(name) is None and before.get(name) is not None):
                    after[name] = before[name]
            if all(after.get(k) == before.get(k) for k in business):
                counts['skipped'] += 1
                continue
            counts['updated'] += 1
            if before.get('source_name') and not set(before['source_name'].split(',')).issubset(set(str(incoming.get('source_name','')).split(','))):
                preserved=[name for name in business if before.get(name) is not None and after.get(name)==before[name]]
                if preserved:
                    after['source_name']=','.join(sorted(set(before['source_name'].split(',')+str(incoming.get('source_name','')).split(','))))
                    if 'source_endpoint' in after:after['source_endpoint']=','.join(sorted(set(str(before.get('source_endpoint','')).split(',')+str(incoming.get('source_endpoint','')).split(','))))
                    if 'source_record_hash' in after:after['source_record_hash']=digest([before.get('source_record_hash'),incoming.get('source_record_hash'),preserved])
                    if 'data_quality_status' in after:after['data_quality_status']='mixed_sources'
        changes.append({'before':before,'after':after,'row_key':row_key(spec,after),
                        'before_hash':row_hash(spec,before) if before else None,'after_hash':row_hash(spec,after)})
    return {'counts':counts,'changes':changes,'conflicts':conflicts,'signature':digest({'snapshots':snapshots,'changes':[[c['row_key'],c['before_hash'],c['after_hash']] for c in changes]})}


def content_hash(spec,row):
    """Peer reconciliation compares business values and availability semantics.

    Import IDs and download/ingestion provenance can legitimately differ after
    a PG-only task followed by a CH-only task. Rollback still uses row_hash.
    """
    ignored=PROVENANCE-{'available_at','availability_basis','data_quality_status'}
    return digest({k:v for k,v in typed(spec,row).items() if k not in ignored})
