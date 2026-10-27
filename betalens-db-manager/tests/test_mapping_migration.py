from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from betalens_db_manager.config import Settings
from betalens_db_manager.manager import DatabaseManagerService
from betalens_db_manager import config_migration
from betalens_db_manager.util import digest


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding='utf-8')


def legacy_manager(tmp_path):
    service = DatabaseManagerService(Settings(config_dir=tmp_path / 'config', state_dir=tmp_path / 'state'))
    path = service.registry.path('fundamental_periodic')
    definition = yaml.safe_load(path.read_text())
    definition['fields'] = [field for field in definition['fields'] if field['name'] not in config_migration.RESEARCH_COLUMNS]
    next(field for field in definition['fields'] if field['name'] == 'rd_exp').update(zh_name='原有同名研发字段', description='旧说明')
    definition['fields'].append({'name': 'custom_finance', 'type': 'text', 'zh_name': '保留自定义', 'description': '用户字段', 'nullable': True})
    write_yaml(path, definition)
    path = service.mappings.path('fundamental_periodic')
    mapping = yaml.safe_load(path.read_text())
    mapping.pop('producers', None);mapping.pop('producer_version', None)
    for binding in mapping['bindings']:
        for rule in binding['fields']:
            if rule['source_field'] == 'rd_exp':
                rule['column'] = 'rd_exp'
            field = next(field for field in service.sources.endpoint('tushare', binding['endpoint'])['fields'] if field['name'] == rule['source_field'])
            rule['source_unit'] = field['description']
    write_yaml(path, mapping)
    path = service.mappings.path('market_daily_raw')
    mapping = yaml.safe_load(path.read_text())
    for binding in mapping['bindings']:
        if binding['endpoint'] == 'daily':
            for rule in binding['fields']:
                if rule['source_field'] == 'vol':
                    rule['source_unit'] = '成交量 （手）'
    write_yaml(path, mapping)
    path = service.mappings.path('corporate_action')
    mapping = yaml.safe_load(path.read_text());mapping.pop('producers', None);mapping.pop('producer_version', None)
    for binding in mapping['bindings']:
        if binding['endpoint'] == 'namechange':
            for rule in binding['fields']:
                if rule['column'] == 'effective_date':
                    rule['source_field'] = 'end_date'
        if binding['endpoint'] == 'dividend':
            binding['fields'] = [rule for rule in binding['fields'] if rule['column'] not in {'cash_dividend', 'share_dividend'}]
            for rule in binding['fields']:
                if rule['column'] in {'event_date', 'effective_date'}:
                    rule['source_field'] = 'end_date'
    write_yaml(path, mapping)
    service.registry.refresh();service.mappings.refresh()
    return service


def snapshot(service):
    return {str(path): path.read_bytes() for path in service.settings.config_dir.rglob('*') if path.is_file()}


def test_preview_is_read_only_preserves_custom_and_repairs_semantics(tmp_path):
    service = legacy_manager(tmp_path);before = snapshot(service)
    before_ledger = service.state.ledger_rows('config_migration')
    plan = config_migration.preview(service)
    assert not plan['blocked'], plan['errors']
    assert before == snapshot(service)
    assert before_ledger == service.state.ledger_rows('config_migration')
    files = {Path(item['path']).stem + ':' + Path(item['path']).parent.name: yaml.safe_load(item['after_yaml']) for item in plan['files']}
    table = files['fundamental_periodic:tables']
    assert set(config_migration.RESEARCH_COLUMNS) <= {field['name'] for field in table['fields']}
    assert next(field for field in table['fields'] if field['name'] == 'custom_finance')['description'] == '用户字段'
    assert 'deprecated' in next(field for field in table['fields'] if field['name'] == 'rd_exp')['description']
    finance = files['fundamental_periodic:mappings']
    for binding in finance['bindings']:
        rd = next((rule for rule in binding['fields'] if rule['source_field'] == 'rd_exp'), None)
        if rd:
            expected = 'research_and_development_expense' if binding['endpoint'].removesuffix('_vip') == 'income' else 'research_and_development_investment'
            assert rd['column'] == expected
    corporate = files['corporate_action:mappings']
    for binding in corporate['bindings']:
        if binding['endpoint'] == 'namechange':
            assert next(rule for rule in binding['fields'] if rule['column'] == 'effective_date')['source_field'] == 'start_date'
        if binding['endpoint'] == 'dividend':
            rules = {rule['column']: rule['source_field'] for rule in binding['fields']}
            assert rules['event_date'] == 'ex_date'
            assert rules['cash_dividend'] == 'cash_div' and rules['share_dividend'] == 'stk_bo_rate'
    assert all(rule.get('source_unit') != 'TS代码' for binding in finance['bindings'] for rule in binding['fields'] if rule['source_field'] == 'ts_code')
    raw = files['market_daily_raw:mappings']
    volume = next(rule for binding in raw['bindings'] if binding['endpoint'] == 'daily' for rule in binding['fields'] if rule['source_field'] == 'vol')
    assert volume['source_unit'] == '手' and str(volume['multiplier']) == '100'


def test_apply_private_backups_registry_versions_and_idempotence(tmp_path):
    service = legacy_manager(tmp_path);before = snapshot(service)
    plan = config_migration.preview(service)
    result = config_migration.apply(service, plan, plan['token'])
    assert result['status'] == 'completed'
    assert not service.registry.errors and not service.mappings.errors
    assert service.state.ledger_get('config_migration', result['id'])['status'] == 'completed'
    for item in result['files']:
        assert Path(item['backup']).read_bytes() == before[item['path']]
        assert Path(item['backup']).stat().st_mode & 0o777 == 0o600
        assert Path(item['path']).stat().st_mode & 0o777 == 0o600
        assert digest(Path(item['path']).read_text()) == item['after_hash']
    assert Path(result['backup_dir']).stat().st_mode & 0o777 == 0o700
    new = config_migration.preview(service)
    assert new['files'] == [], new
    assert config_migration.apply(service, new, new['token'])['status'] == 'unchanged'


def test_stale_or_tampered_preview_changes_no_files(tmp_path):
    service = legacy_manager(tmp_path);plan = config_migration.preview(service)
    altered = deepcopy(plan);altered['files'][0]['after_yaml'] = 'name: malicious\n'
    before = snapshot(service)
    with pytest.raises(ValueError, match='内容不匹配'):
        config_migration.apply(service, altered, plan['token'])
    assert snapshot(service) == before
    path = Path(plan['files'][-1]['path']);path.write_text(path.read_text() + '\n# external edit\n')
    before = snapshot(service)
    with pytest.raises(ValueError, match='预览已过期'):
        config_migration.apply(service, plan, plan['token'])
    assert snapshot(service) == before
    assert not (service.settings.config_dir / 'migration-backups').exists()


def test_custom_producer_and_mapping_extras_are_preserved(tmp_path):
    service = legacy_manager(tmp_path)
    path = service.mappings.path('fundamental_periodic');value = yaml.safe_load(path.read_text())
    custom = {'column': 'custom_finance', 'kind': 'unavailable', 'reason': '用户声明待接入', 'inputs': [], 'scope': {}}
    value['producers'] = [custom];value['custom_note'] = '保留外部备注'
    write_yaml(path, value)
    plan = config_migration.preview(service)
    updated = yaml.safe_load(next(item['after_yaml'] for item in plan['files'] if item['path'] == str(path)))
    assert custom in updated['producers']
    assert [producer for producer in updated['producers'] if producer['column'] == 'custom_finance'] == [custom]
    assert updated['custom_note'] == '保留外部备注'


@pytest.mark.parametrize('folder', ['tables', 'mappings', 'migration-backups'])
def test_symlink_directory_is_rejected_before_backup(tmp_path, folder):
    service = legacy_manager(tmp_path)
    target = tmp_path / 'outside';target.mkdir()
    path = service.settings.config_dir / folder
    if path.exists():
        path.rename(tmp_path / ('original_' + folder))
    path.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='符号链接'):
        plan = config_migration.preview(service)
        config_migration.apply(service, plan, plan['token'])
    assert list(target.iterdir()) == []


def test_invalid_yaml_blocks_apply_without_backup(tmp_path):
    service = legacy_manager(tmp_path)
    service.mappings.path('corporate_action').write_text('bindings: [broken')
    plan = config_migration.preview(service)
    assert plan['blocked'] and plan['errors']
    with pytest.raises(ValueError, match='无效配置'):
        config_migration.apply(service, plan, plan['token'])
    assert not (service.settings.config_dir / 'migration-backups').exists()


def test_publish_failure_restores_only_unchanged_outputs(tmp_path, monkeypatch):
    service = legacy_manager(tmp_path);plan = config_migration.preview(service);before = snapshot(service)
    original_atomic = config_migration.atomic_private
    target = Path(plan['files'][1]['path']);failed = False
    def fail_once(path, content, **kwargs):
        nonlocal failed
        if Path(path) == target and not failed:
            failed = True
            raise OSError('injected local write failure')
        return original_atomic(path, content, **kwargs)
    monkeypatch.setattr(config_migration, 'atomic_private', fail_once)
    with pytest.raises(OSError, match='injected'):
        config_migration.apply(service, plan, plan['token'])
    for path, content in before.items():
        assert Path(path).read_bytes() == content
    record = service.state.ledger_rows('config_migration')[-1]
    assert record['status'] == 'partial'
    assert record['files'][0]['status'] == 'restored'


def test_post_adjusted_index_only_compile_after_missing_keys_migrated(tmp_path):
    service = legacy_manager(tmp_path)
    path = service.mappings.path('market_daily_post_adjusted')
    mapping = yaml.safe_load(path.read_text())
    for binding in mapping['bindings']:
        if binding['endpoint'] in {'daily', 'fund_daily', 'index_daily', 'sw_daily'}:
            binding['fields'] = [rule for rule in binding['fields'] if rule['column'] not in {'security_code', 'trade_date'}]
    mapping.pop('producers', None);mapping.pop('producer_version', None)
    write_yaml(path, mapping)
    plan = config_migration.preview(service)
    assert not plan['blocked'], plan['errors']
    config_migration.apply(service, plan, plan['token'])
    migrated = service.mappings.get('market_daily_post_adjusted')
    for binding in migrated['bindings']:
        if binding['endpoint'] in {'daily', 'fund_daily', 'index_daily', 'sw_daily'}:
            keys = {(rule['source_field'], rule['column']) for rule in binding['fields']}
            assert {('ts_code', 'security_code'), ('trade_date', 'trade_date')} <= keys
    compiled = service.compile(dict(tables=['market_daily_post_adjusted'], start_date='2026-10-16', end_date='2026-10-16',
                                    codes=['000300.SH'], asset_types=['index'], targets=['postgres']))
    assert compiled['request']['tables'] == ['market_daily_post_adjusted']
    assert 'index_daily' in {request['endpoint'] for request in compiled['requests']}


def test_builtin_description_is_not_a_unit_and_custom_units_are_preserved(tmp_path):
    service = legacy_manager(tmp_path)
    path = service.registry.path('fundamental_periodic')
    definition = yaml.safe_load(path.read_text())
    field = next(field for field in definition['fields'] if field['name'] == 'total_revenue')
    field['unit'] = field['description']
    custom = next(field for field in definition['fields'] if field['name'] == 'custom_finance')
    custom['unit'] = custom['description']
    write_yaml(path, definition)
    plan = config_migration.preview(service)
    updated = yaml.safe_load(next(item['after_yaml'] for item in plan['files'] if item['path'] == str(path)))
    repaired = next(field for field in updated['fields'] if field['name'] == 'total_revenue')
    assert repaired['unit'] != repaired['description']
    assert next(field for field in updated['fields'] if field['name'] == 'custom_finance')['unit'] == custom['unit']


@pytest.mark.parametrize('unit,expected,warning', [('元', '元/股', False), ('元/股', '元/股', False), ('自定义每股单位', '自定义每股单位', True)])
def test_cash_dividend_per_share_unit_migration_preserves_custom(tmp_path, unit, expected, warning):
    service = legacy_manager(tmp_path)
    path = service.registry.path('corporate_action');definition = yaml.safe_load(path.read_text())
    field = next(field for field in definition['fields'] if field['name'] == 'cash_dividend')
    field['unit'] = unit
    write_yaml(path, definition)
    plan = config_migration.preview(service)
    replacement = next((item for item in plan['files'] if item['path'] == str(path)), None)
    updated = yaml.safe_load(replacement['after_yaml']) if replacement else definition
    assert next(field for field in updated['fields'] if field['name'] == 'cash_dividend')['unit'] == expected
    assert any('cash_dividend' in message and '自定义单位' in message for message in plan['warnings']) is warning
