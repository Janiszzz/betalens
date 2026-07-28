from __future__ import annotations

from pathlib import Path

import pytest

from betalens_db_manager.gui_controller import (
    BusyOperationError,
    GuiController,
    OperationRegistry,
)
from betalens_db_manager.jobs import ImportJobRunner
from betalens_db_manager.utils import file_sha256


class _FakeManager:
    def __init__(self):
        self.effective_config = {
            "dbname": "gui_unit",
            "user": "tester",
            "password": "",
            "host": "localhost",
            "port": "5432",
        }
        self.created: list[str] = []

    def ensure_dataset(self, table: str):
        self.created.append(table)
        return {"status": "completed", "requested_dataset": table}

    def probe_connection(self):
        return {"status": "online", "database": "gui_unit"}


class _FakeClient:
    def table_overview(self):
        return [
            {
                "table_name": "daily_market",
                "estimated_rows": 2,
                "date_range": {"min_dt": "2024-01-01", "max_dt": "2024-01-02"},
                "warnings": [],
            }
        ]

    def table_schema(self, table: str):
        return {"logical_table": table, "physical_tables": ["market_daily_fact"]}

    def query_table(self, request):
        return request

    def execute_readonly_sql(self, statement: str, *, limit: int):
        return {"statement": statement, "limit": limit}

    def diagnose_data(self, table: str, *, sample_limit: int):
        return [{"table": table, "limit": sample_limit}]


class _FakeRunner:
    def __init__(self):
        self.runs: list[str] = []
        self.preview_kwargs: list[dict] = []
        self.hash_calls = 0

    def preview(self, path, *, table, import_type, mode, options, **_kwargs):
        self.preview_kwargs.append(dict(_kwargs))
        self.hash_calls += 1
        digest = file_sha256(path)
        return {
            "source_sha256": digest,
            "preview_token": ImportJobRunner.preview_token(
                digest, table, import_type, mode, dict(options)
            ),
            "summary": {"rows": 1, "codes": 1, "metrics": 1, "rejected_rows": 0},
            "validation": {"ok": True, "errors": []},
        }

    def run(self, path, **_kwargs):
        try:
            self.hash_calls += 1
            digest = file_sha256(path)
        except OSError as exc:
            return {"path": str(path), "status": "failed", "error": f"检查后无法读取文件: {exc}"}
        if digest != _kwargs.get("expected_sha256"):
            return {"path": str(path), "status": "failed", "error": "文件在检查后发生变化"}
        self.runs.append(Path(path).name)
        if Path(path).name.startswith("bad"):
            raise RuntimeError("bad source")
        return {"path": str(path), "status": "completed", "inserted_rows": 1}


def _controller():
    manager = _FakeManager()
    runner = _FakeRunner()
    controller = GuiController(manager=manager, client=_FakeClient(), runner=runner)
    controller.connection_state = "online"
    return controller, manager, runner


def test_operation_registry_prevents_duplicate_writes():
    registry = OperationRegistry()
    with registry.claim("database-write"):
        assert registry.active()
        with pytest.raises(BusyOperationError):
            with registry.claim("database-write"):
                pass
    assert not registry.active()


def test_controller_starts_offline_without_probe():
    manager = _FakeManager()
    controller = GuiController(manager=manager, client=_FakeClient(), runner=_FakeRunner())

    assert controller.connection_state == "offline"
    assert controller.connection_details == {}
    assert manager.created == []
    assert not hasattr(controller, "plan_schema")


def test_table_catalog_marks_only_reported_dataset_as_created():
    controller, _, _ = _controller()

    rows = {item["table"]: item for item in controller.table_catalog()}

    assert rows["daily_market"]["state"] == "已建立"
    assert rows["fundamentals"]["state"] == "尚未建立"


def test_create_selected_table_uses_contract_facade():
    controller, manager, _ = _controller()

    report = controller.create_selected_table("daily_market")

    assert manager.created == ["daily_market"]
    assert report["requested_dataset"] == "daily_market"
    assert controller.connection_state == "online"


def test_folder_discovery_is_recursive_supported_and_stable(tmp_path):
    root = tmp_path / "incoming"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "z.xlsx").write_text("placeholder", encoding="utf-8")
    (nested / "a.csv").write_text("datetime,code,name,metric,value\n", encoding="utf-8")
    (nested / "ignore.txt").write_text("no", encoding="utf-8")
    (nested / "b.CSV.GZ").write_text("not actually gzip", encoding="utf-8")

    files = GuiController.discover_files([root])

    assert [item.name for item in files] == ["a.csv", "b.CSV.GZ", "z.xlsx"]


def test_preflight_and_folder_import_continue_after_one_failed_file(tmp_path):
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    good.write_text("datetime,code,name,metric,value\n", encoding="utf-8")
    bad.write_text("datetime,code,name,metric,value\n", encoding="utf-8")
    controller, _, runner = _controller()

    plan = controller.preflight_import(
        [tmp_path],
        table="daily_market",
        adapter="standard_long",
    )
    report = controller.run_import_plan(plan)

    assert len(plan.items) == 2
    assert report["status"] == "completed_with_errors"
    assert report["completed_files"] == 1
    assert report["failed_files"] == 1
    assert runner.runs == ["bad.csv", "good.csv"]
    assert all(call["inspect_database"] is False for call in runner.preview_kwargs)
    assert runner.hash_calls == 4


def test_changed_file_is_rejected_after_preflight(tmp_path):
    source = tmp_path / "market.csv"
    source.write_text("before", encoding="utf-8")
    controller, _, runner = _controller()
    plan = controller.preflight_import(
        [source],
        table="daily_market",
        adapter="standard_long",
    )
    source.write_text("after", encoding="utf-8")

    report = controller.run_import_plan(plan)

    assert report["status"] == "completed_with_errors"
    assert report["failed_files"] == 1
    assert runner.runs == []
    assert runner.hash_calls == 2


def test_deleted_file_is_reported_without_stopping_the_folder_plan(tmp_path):
    source = tmp_path / "market.csv"
    source.write_text("before", encoding="utf-8")
    controller, _, runner = _controller()
    plan = controller.preflight_import(
        [source],
        table="daily_market",
        adapter="standard_long",
    )
    source.unlink()

    report = controller.run_import_plan(plan)

    assert report["status"] == "completed_with_errors"
    assert "无法读取文件" in report["items"][0]["error"]
    assert runner.runs == []


def test_preflight_keeps_rejected_row_preview_for_the_gui_dialog(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("placeholder", encoding="utf-8")

    class RejectedRunner(_FakeRunner):
        def preview(self, path, **kwargs):
            result = super().preview(path, **kwargs)
            result["summary"] = {"rows": 1, "rejected_rows": 1, "codes": 1, "metrics": 1}
            result["validation"] = {"ok": False, "errors": ["源文件包含 rejected rows: 1"]}
            result["rejected_preview"] = [
                {
                    "source_file": str(path),
                    "source_row": 2,
                    "field": "value",
                    "metric": "收盘价(元)",
                    "raw_value": "not-a-number",
                    "reason": "value 不是有限数值",
                }
            ]
            return result

    controller = GuiController(
        manager=_FakeManager(), client=_FakeClient(), runner=RejectedRunner()
    )
    controller.connection_state = "online"

    plan = controller.preflight_import(
        [source], table="daily_market", adapter="auto"
    )

    assert plan.items[0].rejected_preview[0]["field"] == "value"
    assert plan.items[0].rejected_preview[0]["reason"] == "value 不是有限数值"
