from __future__ import annotations

import json

from betalens_db_manager.job_store import JobStore
from betalens_db_manager.profiles import (
    ConnectionProfile,
    ConnectionResolver,
    ProfileStore,
)
from betalens_db_manager.records import ImportRecordStore


def test_profile_store_never_serializes_password(tmp_path):
    path = tmp_path / "profiles.json"
    store = ProfileStore(path)
    profile = ConnectionProfile.from_mapping(
        "local",
        {
            "host": "127.0.0.1",
            "port": 5433,
            "database": "unit_test",
            "user": "tester",
            "password": "must-not-be-saved",
        },
    )

    store.save(profile)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["active"] == "local"
    assert payload["profiles"][0]["dbname"] == "unit_test"
    assert "password" not in path.read_text(encoding="utf-8")
    assert store.get() == profile


def test_connection_resolver_priority_and_sources(tmp_path, monkeypatch):
    store = ProfileStore(tmp_path / "profiles.json")
    store.save(ConnectionProfile("local", host="profile-host", dbname="profile-db"))
    monkeypatch.setenv("BETALENS_DB_HOST", "environment-host")
    monkeypatch.setenv("BETALENS_DB_PASSWORD", "session-secret")

    resolved = ConnectionResolver(store).resolve(
        {"host": "cli-host", "dbname": "cli-db"},
        profile="local",
    )

    assert resolved.config["host"] == "cli-host"
    assert resolved.config["dbname"] == "cli-db"
    assert resolved.config["password"] == "session-secret"
    assert resolved.sources["host"] == "runtime_override"
    assert resolved.sources["password"] == "environment:BETALENS_DB_PASSWORD"
    assert "session-secret" not in json.dumps(resolved.as_dict())


def test_job_store_tracks_runs_items_checkpoints_and_cancellation(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3", tmp_path / "logs")
    job = store.create_job("manifest", target_database="unit_test", schema_version=9)
    store.start_job(job["job_id"])
    item = store.create_item(
        job["job_id"],
        item_key="market",
        source_path="market.csv",
        source_hash="abc",
        target="daily_market",
        adapter="standard_long",
        position=1,
    )
    store.update_item(item["item_id"], status="completed", inserted_rows=10)
    store.save(
        "checkpoint-token",
        {
            "target_database": "unit_test",
            "schema_version": 9,
            "item_id": "market",
            "source_hash": "abc",
            "status": "completed",
        },
    )
    store.request_cancel(job["job_id"])
    completed = store.finish_job(job["job_id"], "completed", result={"inserted": 10})

    assert completed["status"] == "completed"
    assert completed["cancel_requested"] is True
    assert completed["result"] == {"inserted": 10}
    assert store.list_items(job["job_id"])[0]["inserted_rows"] == 10
    assert store.load("checkpoint-token")["source_hash"] == "abc"


def test_import_record_store_is_a_sqlite_compatibility_wrapper(tmp_path):
    store = ImportRecordStore(tmp_path / "records.jsonl", tmp_path / "logs")
    store.append({"job_id": "one", "status": "completed", "inserted_rows": 2})
    store.append({"job_id": "two", "status": "failed", "error": "boom"})

    records = store.read_all()

    assert [record["job_id"] for record in records] == ["one", "two"]
    assert records[1]["error"] == "boom"
    assert (tmp_path / "records.sqlite3").exists()

