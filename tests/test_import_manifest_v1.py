from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from betalens_db_manager.import_manifest import ManifestRunner
from betalens_db_manager.job_store import JobStore


def _write_long(path: Path, code: str, *, remark: str = "{}") -> None:
    pd.DataFrame(
        {
            "datetime": ["2024-01-02 15:00:01"],
            "code": [code],
            "name": [code],
            "metric": ["收盘价(元)"],
            "value": [10.0],
            "remark": [remark],
        }
    ).to_csv(path, index=False)


class _FakeClient:
    db_config = {"dbname": "manifest_unit"}


class _FakeJobRunner:
    client = _FakeClient()
    store = None

    def __init__(self):
        self.calls: list[str] = []

    def run(self, path, **kwargs):
        self.calls.append(Path(path).name)
        failed = Path(path).name.startswith("bad")
        return {
            "job_id": Path(path).stem,
            "source_file": str(path),
            "status": "failed" if failed else "completed",
            **({"error": "test failure"} if failed else {"inserted_rows": 1}),
        }


def test_manifest_v1_expands_glob_stably_and_merges_defaults(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_long(data / "b.csv", "000002.SZ")
    _write_long(data / "a.csv", "000001.SZ")
    manifest = tmp_path / "imports.yaml"
    manifest.write_text(
        """
version: 1
defaults:
  mode: insert_only
  options:
    chunk_rows: 123
imports:
  - id: market
    path: data/*.csv
    target: daily_market
    adapter: standard_long
""".lstrip(),
        encoding="utf-8",
    )

    plan = ManifestRunner(
        job_runner=_FakeJobRunner(),
        checkpoint_store=JobStore(tmp_path / "jobs.sqlite3", tmp_path / "logs"),
        target_database="manifest_unit",
    ).preflight(manifest)

    assert [entry.path.name for entry in plan.entries] == ["a.csv", "b.csv"]
    assert [entry.item_id for entry in plan.entries] == ["market:0001", "market:0002"]
    assert all(entry.options["chunk_size"] == 123 for entry in plan.entries)
    assert plan.target_database == "manifest_unit"
    assert all(preview["valid_rows"] == 1 for preview in plan.previews)


def test_manifest_rejects_duplicate_files_and_implicit_adapter(tmp_path):
    source = tmp_path / "market.csv"
    _write_long(source, "000001.SZ")
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        """
version: 1
imports:
  - id: one
    path: market.csv
    target: daily_market
    adapter: standard_long
  - id: two
    path: '*.csv'
    target: daily_market
    adapter: standard_long
""".lstrip(),
        encoding="utf-8",
    )
    runner = ManifestRunner(job_runner=_FakeJobRunner(), target_database="manifest_unit")

    with pytest.raises(ValueError, match="重复安排"):
        runner.preflight(duplicate)

    implicit = tmp_path / "implicit.yaml"
    implicit.write_text(
        """
version: 1
imports:
  - id: one
    path: market.csv
    target: daily_market
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="必须显式包含 adapter"):
        runner.preflight(implicit)


def test_manifest_rejected_rows_are_structured(tmp_path):
    source = tmp_path / "bad_remark.csv"
    _write_long(source, "000001.SZ", remark="not-json")
    manifest = tmp_path / "imports.yaml"
    manifest.write_text(
        """
version: 1
imports:
  - id: rejected
    path: bad_remark.csv
    target: daily_market
    adapter: standard_long
    on_rejected: skip
""".lstrip(),
        encoding="utf-8",
    )

    plan = ManifestRunner(job_runner=_FakeJobRunner(), target_database="manifest_unit").preflight(manifest)
    rejected = plan.previews[0]["rejected_preview"][0]

    assert set(rejected) == {"source_file", "source_row", "field", "raw_value", "reason"}
    assert rejected["field"] == "remark"
    assert rejected["source_row"] == 1


def test_manifest_continues_failures_and_resumes_completed_files(tmp_path):
    _write_long(tmp_path / "bad.csv", "000001.SZ")
    _write_long(tmp_path / "good.csv", "000002.SZ")
    manifest = tmp_path / "imports.yaml"
    manifest.write_text(
        """
version: 1
imports:
  - id: bad
    path: bad.csv
    target: daily_market
    adapter: standard_long
  - id: good
    path: good.csv
    target: daily_market
    adapter: standard_long
""".lstrip(),
        encoding="utf-8",
    )
    fake = _FakeJobRunner()
    store = JobStore(tmp_path / "jobs.sqlite3", tmp_path / "logs")
    runner = ManifestRunner(
        job_runner=fake,
        checkpoint_store=store,
        target_database="manifest_unit",
    )
    plan = runner.preflight(manifest)

    first = runner.run(plan, resume=True)
    second = runner.run(plan, resume=True)

    assert first["status"] == "completed_with_errors"
    assert first["completed"] == 1
    assert first["failed"] == 1
    assert second["status"] == "completed_with_errors"
    assert second["resumed"] == 1
    assert fake.calls == ["bad.csv", "good.csv", "bad.csv"]


def test_manifest_detects_source_change_after_preview(tmp_path):
    source = tmp_path / "market.csv"
    _write_long(source, "000001.SZ")
    manifest = tmp_path / "imports.yaml"
    manifest.write_text(
        """
version: 1
imports:
  - id: market
    path: market.csv
    target: daily_market
    adapter: standard_long
""".lstrip(),
        encoding="utf-8",
    )
    runner = ManifestRunner(job_runner=_FakeJobRunner(), target_database="manifest_unit")
    plan = runner.preflight(manifest)
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="预检后文件"):
        runner.run(plan)

