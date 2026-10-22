"""Source adapter registry used by the canonical import service."""
from __future__ import annotations

from .mapping import source_capabilities


class SourceAdapter:
    name = "source"

    def capabilities(self):
        return [row for row in source_capabilities() if row["source_name"] == self.name]

    def plan(self, request: dict) -> dict:
        raise NotImplementedError

    def fetch(self, request):
        raise NotImplementedError

    def normalize(self, response: dict, mapping):
        from decimal import Decimal,localcontext
        if callable(mapping):return mapping(response)
        targets = [rule if isinstance(rule, str) else rule['column'] for rule in mapping.values()]
        if len(targets) != len(set(targets)):
            raise ValueError('多个源字段不能覆盖同一本地字段')
        unknown=set(response['fields'])-set(mapping)
        if unknown:raise ValueError('源字段尚未映射：'+','.join(sorted(unknown)))
        result=[]
        for values in response['items']:
            if len(values)!=len(response['fields']):raise ValueError('字段与数值数量不一致')
            row={}
            for field,value in zip(response['fields'],values):
                rule=mapping[field];column=rule if isinstance(rule,str) else rule['column']
                if value == '': value = None
                if isinstance(rule,dict) and value not in (None,'') and rule.get('multiplier'):
                    with localcontext() as context:
                        context.prec=80;value=Decimal(str(value))*Decimal(str(rule['multiplier']))
                if isinstance(rule, dict) and value is not None and rule.get('target_type') in {'date', 'timestamp'}:
                    from datetime import datetime, timezone
                    from zoneinfo import ZoneInfo
                    from .util import parse_date
                    kind = rule['target_type']
                    if rule.get('date_format'):
                        parsed = datetime.strptime(str(value), rule['date_format'])
                    elif kind == 'date':
                        row[column] = parse_date(value)
                        continue
                    else:
                        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
                    if kind == 'date':
                        value = parsed.date()
                    else:
                        if parsed.tzinfo is None:
                            if not rule.get('timezone'):
                                raise ValueError(column + ': 无时区时间须显式指定 timezone，避免八小时时间错位')
                            try:
                                parsed = parsed.replace(tzinfo=ZoneInfo(rule['timezone']))
                            except (KeyError, ValueError):
                                raise ValueError(column + ': timezone 无效') from None
                        value = parsed.astimezone(timezone.utc)
                row[column]=value
            result.append(row)
        return result

    def validate_response(self, response: dict, expected_fields: list[str] | None = None) -> dict:
        expected_fields=expected_fields or getattr(self,'expected_fields',None)
        if expected_fields is None:raise ValueError('尚未登记源字段合同')
        actual = list(response.get("fields", []))
        if len(actual) != len(set(actual)):
            raise ValueError('源字段名重复，不能安全映射')
        expected = list(expected_fields)
        missing = [field for field in expected if field not in actual]
        extra = [field for field in actual if field not in expected]
        if missing:
            raise ValueError(f"{self.name}.{response.get('endpoint', '')} 返回字段缺失：{','.join(missing)}")
        return {"fields": actual, "missing": missing, "extra": extra, "exact": not extra and actual == expected}


class WindSourceAdapter(SourceAdapter):
    name = "wind"

    def __init__(self, configured: bool = False):
        self.configured = configured

    def plan(self, request: dict) -> dict:
        if not self.configured:
            raise ValueError("Wind API 未配置；请配置 Wind SDK 后再选择该来源")
        raise NotImplementedError("Wind adapter 需要按当前 Wind SDK 版本配置字段合同")

    def fetch(self,request):
        return self.plan(request)


class FileSourceAdapter(SourceAdapter):
    name = "file"

    def plan(self, request: dict) -> dict:
        if not request.get("path") or not request.get("column_map"):
            raise ValueError("文件来源必须提供路径和显式 column_map")
        targets = [rule if isinstance(rule, str) else rule['column'] for rule in request['column_map'].values()]
        if len(targets) != len(set(targets)):
            raise ValueError('多个源字段不能覆盖同一本地字段')
        generated = {'import_task_id', 'downloaded_at', 'source_name', 'source_endpoint', 'source_row_no', 'source_record_hash', 'mapping_version', 'data_quality_status'}
        if set(targets) & generated:
            raise ValueError('任务、下载时间和来源审计字段由系统生成，请勿映射覆盖')
        self.expected_fields = list(request["column_map"])
        return request

    @staticmethod
    def rows(path, encoding="utf-8-sig"):
        import csv
        import gzip
        from pathlib import Path

        path = Path(path)
        if path.suffix.lower() in {".csv", ".gz"}:
            opener = gzip.open if path.suffix.lower() == ".gz" else open
            with opener(path, "rt", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                headers = reader.fieldnames or []
                if len(headers) != len(set(headers)) or any(not value for value in headers):
                    raise ValueError('文件字段名重复或为空')
                yield from reader
        elif path.suffix.lower() == ".xlsx":
            from openpyxl import load_workbook
            workbook = load_workbook(path, read_only=True, data_only=True)
            try:
                iterator = workbook.active.values
                headers = next(iterator)
                if len(headers) != len(set(headers)) or any(not value for value in headers):
                    raise ValueError('文件字段名重复或为空')
                for values in iterator:
                    yield dict(zip(headers, values))
            finally:
                workbook.close()
        elif path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(path).iter_batches(batch_size=5000):
                yield from batch.to_pylist()
        else:
            raise ValueError("支持 CSV/CSV.GZ/XLSX/Parquet")

    def fetch(self, request):
        self.plan(request)
        fields, items = None, []
        for row in self.rows(request["path"], request.get("encoding", "utf-8-sig")):
            if fields is None:
                fields = list(row)
            if set(row) != set(fields):
                raise ValueError("文件列集合在读取期间发生变化")
            items.append([row.get(field) for field in fields])
            if len(items) >= 5000:
                yield {"fields": fields, "items": items}
                items = []
        if items or fields is None:
            yield {"fields": fields or self.expected_fields, "items": items}



from .source import TushareSource

class TushareSourceAdapter(TushareSource, SourceAdapter):
    name = "tushare"

    def fetch(self, request, params=None, fields=None, archive=None):
        from pathlib import Path
        if isinstance(request,dict):
            self.plan(request)
            return super().fetch(request['endpoint'],request.get('params',{}),request['fields'],Path(request['archive']))
        return super().fetch(request,params,fields,archive)

    def plan(self, request):
        from .catalog import dataset
        for name in request.get("datasets",[]):dataset(name)
        if request.get('endpoint'):
            spec=dataset(request['endpoint']);self.expected_fields=[f['name'] for f in spec.fields]
        return request


def canonical_file_rules(spec, mapping):
    types={field['name']:field['type'] for field in spec.fields}
    result={}
    for field,rule in mapping.items():
        rule={'column':rule} if isinstance(rule,str) else dict(rule)
        rule['target_type']=types[rule['column']]
        if rule['target_type']!='numeric':rule.pop('multiplier',None)
        result[field]=rule
    return result
