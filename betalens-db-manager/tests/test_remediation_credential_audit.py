"""Legacy credential inventory must reveal locations without copying secrets."""
from __future__ import annotations

import json
import sqlite3

import yaml

from betalens_db_manager import cli
from betalens_db_manager.config import Settings
from betalens_db_manager.credential_audit import audit_credentials
from betalens_db_manager.util import digest


def _settings(tmp_path, **kwargs):
    return Settings(config_dir=tmp_path / "config", state_dir=tmp_path / "state", **kwargs)


def test_inventory_reports_config_and_task_fields_without_values_or_mutation(tmp_path):
    resolved = "resolved-secret-9af730"
    legacy = "legacy-credential-8b2041"
    settings = _settings(tmp_path, token=resolved, env_file=tmp_path / "private.env")
    settings.config_dir.mkdir()
    mappings = settings.config_dir / "mappings"
    mappings.mkdir()
    settings.state_dir.mkdir()
    (tmp_path / "private.env").write_text("AUDIT_KEY=private-env-secret\n")
    source = settings.config_dir / "sources.yaml"
    source.write_text(yaml.safe_dump({"sources": [{
        "source_id": "wire", "provider": "http",
        "auth": {"env": "AUDIT_KEY", "location": "query", "name": "id"},
        "endpoints": {"daily": {"parameters": {"nested": [{"apiKey": legacy}]}}},
    }]}))
    mapping = mappings / "daily.yaml"
    mapping.write_text(yaml.safe_dump({"bindings": [{
        "source_id": "wire", "parameters": {"id": legacy,
                                               "note": "prefix " + resolved + " suffix"},
    }]}))
    archive = settings.state_dir / "archive" / "historical.json"
    archive.parent.mkdir()
    archive.write_text("archive-only-secret-is-out-of-scope")
    state = settings.state_dir / "tasks.sqlite3"
    preview = {"x": 1, "token": digest({"x": 1})}
    with sqlite3.connect(state) as conn:
        conn.executescript("""
            CREATE TABLE jobs(plan TEXT, preview TEXT, error TEXT);
            CREATE TABLE tasks(payload TEXT, error TEXT);
            CREATE TABLE ledger(table_name TEXT, value TEXT);
        """)
        conn.execute("INSERT INTO jobs VALUES(?,?,?)", (
            json.dumps({"source_id": "wire", "parameters": {"id": legacy,
                        "comment": "echo " + resolved}}), json.dumps(preview), None))
        conn.execute("INSERT INTO tasks VALUES(?,?)", (
            json.dumps({"headers": {"Authorization": legacy}}), None))
        conn.execute("INSERT INTO ledger VALUES(?,?)", (
            "source_definition", json.dumps({"source_id": "wire", "auth": {
                "env": "AUDIT_KEY", "location": "query", "name": "id"}})))
    before = {path: path.read_bytes() for path in (source, mapping, state, archive)}

    report = audit_credentials(settings)

    assert report["status"] == "findings"
    locations = {row["file"] + "#" + row["field"]: row for row in report["findings"]}
    assert "sources.yaml#/sources/0/endpoints/daily/parameters/nested/0/apiKey" in locations
    assert "mappings/daily.yaml#/bindings/0/parameters/id" in locations
    assert "mappings/daily.yaml#/bindings/0/parameters/note" in locations
    assert any(row["field"].endswith("/jobs/1/plan/parameters/id") for row in report["findings"])
    assert any(row["field"].endswith("/tasks/1/payload/headers/Authorization") for row in report["findings"])
    assert not any("/preview/token" in row["field"] for row in report["findings"])
    assert all(row["file"] != archive.name for row in report["findings"])
    output = json.dumps(report)
    for secret in (resolved, legacy, "private-env-secret", "archive-only-secret"):
        assert secret not in output
    assert {path: path.read_bytes() for path in before} == before


def test_inventory_skips_valid_auth_references_and_symlinks(tmp_path):
    settings = _settings(tmp_path)
    settings.config_dir.mkdir()
    mappings = settings.config_dir / "mappings"
    mappings.mkdir()
    (settings.config_dir / "sources.yaml").write_text(yaml.safe_dump({
        "sources": [{"source_id": "wire", "auth": {"env": "AUDIT_KEY",
                     "location": "header", "name": "X-Api-Key", "prefix": "Bearer "},
                     "token_configured": True}]}))
    outside = tmp_path / "outside.yaml"
    outside.write_text("apiKey: do-not-follow-this-secret\n")
    (mappings / "outside.yaml").symlink_to(outside)

    report = audit_credentials(settings)

    assert report["findings"] == []
    assert report["warnings"] == [{"file": "mappings/outside.yaml", "kind": "symlink_skipped"}]
    assert "do-not-follow-this-secret" not in json.dumps(report)


def test_cli_audit_does_not_construct_service_or_leak_initialization_errors(tmp_path, monkeypatch, capsys):
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls, path=None: settings))

    def forbidden_service(_settings):
        raise AssertionError("service initialization would write config and ledger")

    monkeypatch.setattr(cli, "DatabaseManagerService", forbidden_service)
    assert cli.main(["credentials-audit"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "clean"
    assert not settings.config_dir.exists()
    assert not settings.state_dir.exists()

    def bad_settings(cls, path=None):
        raise ValueError("unprintable-secret-value")

    monkeypatch.setattr(cli.Settings, "from_env", classmethod(bad_settings))
    assert cli.main(["credentials-audit"]) == 1
    assert "unprintable-secret-value" not in capsys.readouterr().err


def test_inventory_budget_never_reports_an_incomplete_scan_as_clean(tmp_path):
    settings=_settings(tmp_path);settings.config_dir.mkdir()
    (settings.config_dir/'sources.yaml').write_text('sources:\n- source_id: example\n  parameters:\n    apiKey: hidden-value\n')
    report=audit_credentials(settings,max_values=1)
    assert report['status']=='incomplete'
    assert any(r['kind']=='scan_budget_exceeded' for r in report['warnings'])
    assert 'hidden-value' not in json.dumps(report)
