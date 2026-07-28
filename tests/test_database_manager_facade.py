from __future__ import annotations

import json
from argparse import Namespace

from betalens_db_manager import DatabaseManager
from betalens_db_manager import __main__ as cli
from betalens_db_manager.job_store import JobStore
from betalens_db_manager.profiles import ConnectionResolver, ProfileStore


class _SchemaSuccess:
    def plan_migration(self, target_version=None):
        return {
            "database": "facade_unit",
            "database_exists": True,
            "current_version": 9,
            "target_version": target_version or 9,
            "latest_version": 9,
            "applied": [],
            "pending": [],
        }

    def bootstrap_local(self, **kwargs):
        return {
            "status": "completed",
            "schema_version": 9,
            "warnings": [],
            "verification": {"ok": True},
            "report_path": str(kwargs["report_path"]),
        }

    def verify_schema(self, **kwargs):
        return {"ok": True, "database_exists": True, "errors": [], "warnings": []}


class _SchemaFailure(_SchemaSuccess):
    def bootstrap_local(self, **kwargs):
        raise RuntimeError("migration failed before commit")


def _manager(tmp_path):
    profiles = ProfileStore(tmp_path / "profiles.json")
    manager = DatabaseManager(
        {
            "dbname": "facade_unit",
            "user": "tester",
            "password": "",
            "host": "localhost",
            "port": "5432",
        },
        resolver=ConnectionResolver(profiles),
        profile_store=profiles,
        job_store=JobStore(tmp_path / "jobs.sqlite3", tmp_path / "logs"),
    )
    return manager


def test_facade_persists_report_when_schema_fails_before_commit(tmp_path):
    manager = _manager(tmp_path)
    manager.schema = _SchemaFailure()
    report_path = tmp_path / "run.json"

    report = manager.bootstrap(report_path=report_path)

    assert report["status"] == "failed_before_commit"
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["error"]
    assert manager.list_jobs()[0]["status"] == "failed_before_commit"


def test_facade_combines_schema_manifest_items_and_final_verify(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    manager.schema = _SchemaSuccess()
    manifest_plan = {
        "version": 1,
        "path": str(tmp_path / "imports.yaml"),
        "token": "token",
        "target_database": "facade_unit",
        "schema_version": 9,
        "on_error": "continue",
        "entries": [],
    }
    monkeypatch.setattr(
        manager,
        "_execute_manifest",
        lambda *args, **kwargs: {
            "status": "completed_with_errors",
            "items": [
                {
                    "item_id": "one",
                    "position": 1,
                    "path": "one.csv",
                    "target": "daily_market",
                    "adapter": "standard_long",
                    "mode": "insert_only",
                    "sha256": "abc",
                    "status": "failed",
                    "record": {"status": "failed", "error": "bad file"},
                }
            ],
        },
    )

    report = manager.bootstrap(
        report_path=tmp_path / "combined.json",
        manifest=manifest_plan,
    )

    assert report["status"] == "completed_with_errors"
    assert report["schema"]["schema_version"] == 9
    assert report["verification"]["ok"]
    assert manager.job_store.list_items(report["run_id"])[0]["status"] == "failed"


class _CliManager:
    def plan_manifest(self, path):
        return {"entries": [], "path": str(path)}

    def run_manifest(self, *args, **kwargs):
        return {"status": "completed_with_errors", "report_path": "report.json"}


def test_cli_returns_nonzero_for_completed_with_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_make_manager", lambda args: _CliManager())
    args = Namespace(
        manifest=tmp_path / "imports.yaml",
        yes=True,
        resume=True,
        report=None,
    )

    assert cli._run_import_command(args) == 1

