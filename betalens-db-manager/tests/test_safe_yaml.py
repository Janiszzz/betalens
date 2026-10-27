from datetime import date

import pytest
import yaml

from betalens_db_manager import safe_yaml


@pytest.mark.parametrize('python_fallback', [False, True])
def test_roundtrip_preserves_configuration_types_and_explicit_order(monkeypatch, python_fallback):
    if python_fallback:
        monkeypatch.setattr(safe_yaml, '_SafeLoader', yaml.SafeLoader)
        monkeypatch.setattr(safe_yaml, '_SafeDumper', yaml.SafeDumper)
    value = {'format_version': 3, '中文': '元/股', 'nullable': False, 'zero': 0,
             'multiplier': '10000', 'date': date(2026, 10, 24), 'rows': [{'empty': None}]}
    encoded = safe_yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
    assert encoded.startswith('format_version: 3\n中文: 元/股\n')
    assert safe_yaml.safe_load(encoded) == value
    assert yaml.safe_load(encoded) == value


@pytest.mark.parametrize('python_fallback', [False, True])
@pytest.mark.parametrize('document', [
    '!!python/object/apply:builtins.eval ["1 + 1"]',
    '!!python/object/new:builtins.object {}',
    '!!python/name:builtins.eval',
    'binding: !unregistered {name: daily}',
])
def test_unsafe_or_unregistered_tags_are_rejected(monkeypatch, python_fallback, document):
    if python_fallback:
        monkeypatch.setattr(safe_yaml, '_SafeLoader', yaml.SafeLoader)
    with pytest.raises(safe_yaml.YAMLError):
        safe_yaml.safe_load(document)


def test_yaml_module_itself_is_not_monkeypatched():
    assert yaml.safe_load is not safe_yaml.safe_load
    assert yaml.safe_dump is not safe_yaml.safe_dump
    assert safe_yaml._SafeLoader in (yaml.SafeLoader, getattr(yaml, 'CSafeLoader', None))


@pytest.mark.parametrize('kind', ['table', 'mapping'])
def test_external_unsafe_yaml_keeps_last_valid_published_version(tmp_path, kind):
    from betalens_db_manager.config import Settings
    from betalens_db_manager.manager import DatabaseManagerService
    service = DatabaseManagerService(Settings(config_dir=tmp_path/'config', state_dir=tmp_path/'state'))
    store = service.registry if kind == 'table' else service.mappings
    previous = store.get('security_dim')['version']
    store.path('security_dim').write_text('!!python/object/apply:builtins.eval ["1 + 1"]')
    assert 'security_dim' in store.refresh()
    assert store.get('security_dim')['version'] == previous
