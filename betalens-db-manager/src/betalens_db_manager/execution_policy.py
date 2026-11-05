"""Explicit maintenance and archive-only execution boundaries."""
import json


class MaintenanceError(ValueError):
    status_code=503
    code='maintenance_active'


class LegacyTaskError(ValueError):
    status_code=410
    code='legacy_task_read_only'


class LegacyPipelineError(ValueError):
    status_code=422
    code='legacy_pipeline_removed'


def maintenance(root):
    path=root/'maintenance.json'
    if not path.exists():return False
    # A damaged maintenance marker must not accidentally enable execution.
    try:return bool(json.loads(path.read_text()).get('enabled',True))
    except (ValueError,OSError):return True


def ensure_available(root):
    if maintenance(root):raise MaintenanceError('管理器正在维护，暂不接受执行操作；历史记录仍可查看')


def operation_scope(operation,request,payload):
    """Conservative table/target fence; unspecified scopes conflict with all.

    Validation never overlaps a writer for the same table and database, even
    for apparently disjoint dates (derived rules can read neighbouring dates).
    """
    from .reference_contracts import MODULE_TABLE
    validation=operation=='pipeline_validate'
    selected=payload.get('validation',{}) if validation else {}
    tables=set(request.get('tables',[]))|{MODULE_TABLE.get(m,m) for m in request.get('modules',[])}
    if selected.get('tables'):tables&=set(selected['tables'])
    targets=set(request.get('targets',[]))
    if selected.get('targets'):targets&=set(selected['targets'])
    return validation,tables or {'*'},targets or {'*'}


def operations_conflict(left,right):
    if left[0]==right[0]:return True  # One ingest and one validation at most.
    def overlaps(a,b):return '*' in a or '*' in b or bool(a&b)
    return overlaps(left[1],right[1]) and overlaps(left[2],right[2])
