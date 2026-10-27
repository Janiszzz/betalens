"""Preview and explicitly publish additive local YAML repairs without database DDL.

The preview is read-only. Publishing re-derives the plan, checks every source file,
then backs up originals before changing any definitions. User-defined rules win
over generated metadata; uncertain custom semantics are reported for manual review.
"""
from __future__ import annotations

import copy
from pathlib import Path
import uuid

import yaml

from .canonical import builtin_catalog, financial_column
from .field_lineage import producers_for
from .private import atomic_private, restrict
from .publishing import operation_lock
from .util import digest, identifier, now

VERSION = 'mapping-repair-20261024-v1'
RESEARCH_COLUMNS = ('research_and_development_expense', 'research_and_development_investment')
_UNIT_TOKENS = {'元', '千元', '万元', '亿元', '股', '万股', '亿股', '份', '万份', '手', '点', '%', '倍', '元/股', '元/份', '天', '年', '个', '无量纲', '股（基金为份）'}


def _root(service):
    root = service.settings.config_dir.expanduser().absolute()
    for part in (root, *root.parents):
        if part.is_symlink():
            raise ValueError('配置迁移拒绝符号链接目录：' + str(part))
    return root


def _path(root, path):
    path = Path(path).absolute()
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError('配置迁移路径越出用户配置目录') from None
    if '..' in relative.parts or path.resolve() != path:
        raise ValueError('配置迁移拒绝路径穿越或符号链接')
    for part in (path, *path.parents):
        if part == root.parent:
            break
        if part.is_symlink():
            raise ValueError('配置迁移拒绝符号链接：' + str(part))
    return path


def _dump(value):
    return yaml.safe_dump(value, allow_unicode=True, sort_keys=False)


def _mapping_repairs(service, value, changes, warnings):
    value = copy.deepcopy(value)
    for binding in value.get('bindings', []):
        if binding.get('source_id') != 'tushare':
            continue
        endpoint = binding.get('endpoint', '')
        try:
            contract = {field['name']: field for field in service.sources.endpoint('tushare', endpoint)['fields']}
        except (KeyError, ValueError):
            warnings.append(value['table'] + ': 未登记查询 ' + endpoint + '，保留其自定义规则')
            continue
        rules = binding.get('fields', [])
        if value['table'] == 'market_daily_post_adjusted' and endpoint in {'daily', 'fund_daily', 'index_daily', 'sw_daily'}:
            for source, column in [('ts_code', 'security_code'), ('trade_date', 'trade_date')]:
                destinations = [rule for rule in rules if rule.get('column') == column]
                if not destinations:
                    field = contract.get(source, {})
                    rules.append({'source_field': source, 'column': column, 'multiplier': '1', 'required': True, 'source_unit': field.get('source_unit', field.get('unit', ''))})
                    changes.append(endpoint + '.' + source + ' → ' + column + ': 补齐行情业务键')
                elif any(rule.get('source_field') != source for rule in destinations):
                    warnings.append(endpoint + ': ' + column + ' 有自定义业务键，保留并要求人工复核')
        for rule in rules:
            source, column = rule.get('source_field'), rule.get('column')
            if value['table'] == 'fundamental_periodic' and source == 'rd_exp':
                destination = financial_column(endpoint, source)
                if column == 'rd_exp' and destination != column:
                    rule['column'] = destination
                    changes.append(endpoint + '.rd_exp: rd_exp → ' + destination)
                elif destination != column:
                    warnings.append(endpoint + '.rd_exp: 保留用户自定义目标 ' + str(column))
            if value['table'] == 'corporate_action':
                replacement = None
                if endpoint == 'namechange' and source == 'end_date' and column == 'effective_date':
                    replacement = 'start_date'
                elif endpoint == 'dividend' and source == 'end_date' and column in {'event_date', 'effective_date'}:
                    replacement = 'ex_date'
                elif endpoint == 'dividend' and source in {'stk_div', 'stk_co_rate'} and column == 'share_dividend':
                    replacement = 'stk_bo_rate'
                if replacement:
                    rule['source_field'] = replacement
                    changes.append(endpoint + '.' + str(source) + ' → ' + replacement + ': ' + str(column))
                    # Old annotations belong to the former field and must not follow it.
                    field = contract.get(replacement, {})
                    rule['source_unit'] = field.get('source_unit', field.get('unit', ''))
            field = contract.get(rule.get('source_field'), {})
            if rule.get('source_unit') and rule['source_unit'] == field.get('description'):
                replacement = field.get('source_unit', field.get('unit', ''))
                if replacement != rule['source_unit']:
                    rule['source_unit'] = replacement
                    changes.append(endpoint + '.' + str(rule.get('source_field')) + ': 分离字段说明与计量单位')
        if value['table'] == 'corporate_action' and endpoint == 'dividend':
            for source, column in [('cash_div', 'cash_dividend'), ('stk_bo_rate', 'share_dividend')]:
                existing = [rule for rule in rules if rule.get('column') == column]
                if not existing:
                    field = contract.get(source, {})
                    rules.append({'source_field': source, 'column': column, 'multiplier': '1', 'required': False, 'source_unit': field.get('source_unit', field.get('unit', ''))})
                    changes.append(endpoint + '.' + source + ' → ' + column)
                elif any(rule.get('source_field') != source for rule in existing):
                    warnings.append(endpoint + ': ' + column + ' 有自定义来源，保留并要求人工复核')
    return value


def preview(service):
    """Return deterministic, content-addressed changes without changing files/state."""
    root = _root(service)
    files, warnings, errors, definitions = [], [], [], {}
    baseline = builtin_catalog()
    for folder in ('tables', 'mappings'):
        _path(root, root / folder)
    for path in sorted((root / 'tables').glob('*.yaml')):
        path = _path(root, path)
        try:
            original = path.read_text(encoding='utf-8')
            value = yaml.safe_load(original)
            if not isinstance(value, dict) or value.get('name') != identifier(path.stem):
                raise ValueError('文件名与表名必须相同')
            value = copy.deepcopy(value);changes = []
            builtin_fields = {field['name']: field for field in baseline[value['name']].fields} if value['name'] in baseline else {}
            for field in value['fields']:
                if value['name'] == 'corporate_action' and field['name'] == 'cash_dividend':
                    if field.get('unit') == '元':
                        field['unit'] = '元/股'
                        changes.append('cash_dividend: 明确每股现金分红单位为元/股')
                    elif field.get('unit') not in {'元/股', '', None}:
                        warnings.append('corporate_action.cash_dividend: 保留自定义单位 ' + str(field['unit']) + '，需人工确认每股语义')
                        continue
                if field['name'] in builtin_fields and field.get('unit') and field['unit'] == field.get('description') and field['unit'] not in _UNIT_TOKENS:
                    replacement = builtin_fields[field['name']].get('unit', '')
                    # A repeated explanation is not evidence of a physical unit.
                    if replacement == field['unit']:
                        replacement = ''
                    field['unit'] = replacement
                    changes.append(field['name'] + ': 清理误作计量单位的字段说明')
            if value['name'] == 'fundamental_periodic':
                builtin = {field['name']: field for field in baseline[value['name']].fields}
                known = {field['name'] for field in value['fields']}
                for column in RESEARCH_COLUMNS:
                    if column not in known:
                        value['fields'].append(copy.deepcopy(builtin[column]))
                        changes.append('新增可空研发语义列：' + column)
                for field in value['fields']:
                    if field['name'] == 'rd_exp':
                        for key in ('zh_name', 'description'):
                            if field.get(key) != builtin['rd_exp'].get(key):
                                field[key] = builtin['rd_exp'][key]
                                changes.append('标记旧 rd_exp 已弃用：' + key)
            validated = service.registry.validate(value)
            definitions[value['name']] = validated
            if changes:
                files.append({'path': str(path), 'before_hash': digest(original), 'after_yaml': _dump(validated), 'changes': changes})
        except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            errors.append(path.name + ': ' + str(exc))
    registry = copy.copy(service.registry)
    registry.get = lambda name: definitions[name] if name in definitions else service.registry.get(name)
    mappings = copy.copy(service.mappings);mappings.registry = registry
    for path in sorted((root / 'mappings').glob('*.yaml')):
        path = _path(root, path)
        try:
            original = path.read_text(encoding='utf-8');old = yaml.safe_load(original)
            if not isinstance(old, dict) or old.get('table') != identifier(path.stem):
                raise ValueError('文件名与映射表名必须相同')
            changes = [];value = _mapping_repairs(service, old, changes, warnings)
            spec = registry.spec(value['table'])
            # Preserve hand-written declarations verbatim. Previously generated
            # declarations are refreshed only when they exactly match old output.
            old_spec = service.registry.spec(value['table'])
            generated_old = {digest(producer) for producer in producers_for(old_spec, old)}
            custom = [copy.deepcopy(producer) for producer in old.get('producers', []) if digest(producer) not in generated_old]
            custom_columns = {producer['column'] for producer in custom}
            generated = [producer for producer in producers_for(spec, value) if producer['column'] not in custom_columns]
            value['producer_version'] = 1;value['producers'] = custom + generated
            if old.get('producers') != value['producers'] or old.get('producer_version') != 1:
                changes.append('登记逐列生产者、输入依赖和留空原因（保留自定义声明）')
            validated = mappings.validate(value)
            # Validation reconciles direct metadata with current binding rules
            # and may reorder those generated entries. Compare canonical values
            # rather than the intermediate generator order, or a second preview
            # would propose the same harmless ordering change indefinitely.
            try:
                old_validated = mappings.validate(old)
            except (ValueError, KeyError, TypeError):
                old_validated = None
            if changes and validated != old_validated:
                files.append({'path': str(path), 'before_hash': digest(original), 'after_yaml': _dump(validated), 'changes': changes})
        except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            errors.append(path.name + ': ' + str(exc))
    result = {'version': VERSION, 'config_dir': str(root), 'files': files, 'warnings': warnings, 'errors': errors, 'blocked': bool(errors)}
    result['token'] = digest(result)
    return result


def apply(service, plan, token):
    """Publish only the freshly derived, explicitly confirmed plan; never do DDL."""
    with operation_lock(service):
        current = preview(service)
        if not token or token != plan.get('token') or token != current['token']:
            raise ValueError('迁移预览已过期或内容被修改，请重新生成预览')
        if {key: value for key, value in plan.items() if key != 'token'} != {key: value for key, value in current.items() if key != 'token'}:
            raise ValueError('迁移预览内容不匹配，禁止执行')
        if current['blocked']:
            raise ValueError('迁移存在无效配置，先修复全部错误后重新预览')
        root = _root(service)
        for item in current['files']:
            path = _path(root, item['path'])
            if digest(path.read_text(encoding='utf-8')) != item['before_hash']:
                raise ValueError('配置文件已被修改，请重新预览：' + path.name)
        if not current['files']:
            return {'status': 'unchanged', 'files': [], 'warnings': current['warnings']}
        migration_id = uuid.uuid4().hex
        backup_root = _path(root, root / 'migration-backups')
        backup_root.mkdir(exist_ok=True, mode=0o700);restrict(backup_root, True)
        backup = _path(root, root / 'migration-backups' / migration_id)
        backup.mkdir(parents=True, mode=0o700);restrict(backup, True)
        record = {'id': migration_id, 'version': VERSION, 'token': token, 'created_at': now(), 'status': 'prepared', 'backup_dir': str(backup), 'files': [], 'warnings': current['warnings']}
        originals = {}
        for item in current['files']:
            path = _path(root, item['path']);text = path.read_text(encoding='utf-8')
            if digest(text) != item['before_hash']:
                raise ValueError('备份前配置文件已变化，请重新预览：' + path.name)
            relative = path.relative_to(root);destination = _path(root, backup / relative)
            atomic_private(destination, text)
            originals[str(path)] = text
            record['files'].append({'path': str(path), 'backup': str(destination), 'before_hash': item['before_hash'], 'after_hash': digest(item['after_yaml']), 'status': 'pending'})
        atomic_private(backup / 'manifest.yaml', _dump(record))
        service.state.ledger_put('config_migration', migration_id, record)
        try:
            for item, checkpoint in zip(current['files'], record['files']):
                path = _path(root, item['path'])
                if digest(path.read_text(encoding='utf-8')) != item['before_hash']:
                    raise ValueError('发布前配置文件已变化：' + path.name)
                atomic_private(path, item['after_yaml'])
                checkpoint['status'] = 'published'
                atomic_private(backup / 'manifest.yaml', _dump(record))
                service.state.ledger_put('config_migration', migration_id, record)
            service.registry.refresh();service.mappings.refresh()
            if service.registry.errors or service.mappings.errors:
                raise ValueError('已发布配置未通过重新加载校验')
            record['status'] = 'completed';record['finished_at'] = now()
        except Exception as exc:
            record['status'] = 'partial';record['error'] = str(exc);record['finished_at'] = now()
            # Do not overwrite an external edit while undoing a partial publish.
            for checkpoint in record['files']:
                if checkpoint['status'] != 'published':
                    continue
                path = _path(root, checkpoint['path'])
                if digest(path.read_text(encoding='utf-8')) == checkpoint['after_hash']:
                    atomic_private(path, originals[str(path)]);checkpoint['status'] = 'restored'
                else:
                    checkpoint['status'] = 'restore_conflict'
            service.registry.refresh();service.mappings.refresh()
            raise
        finally:
            atomic_private(backup / 'manifest.yaml', _dump(record))
            service.state.ledger_put('config_migration', migration_id, record)
        return record
