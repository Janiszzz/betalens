from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_gui_starts_offline_with_four_beginner_pages(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    from PySide6.QtWidgets import QApplication
    from betalens_db_manager.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.windowTitle() == "Betalens Database Manager"
        assert not window.windowIcon().isNull()
        assert window.tabs.count() == 4
        assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
            "数据表",
            "文件导入",
            "查询与诊断",
            "联网更新",
        ]
        assert window.controller.connection_state == "offline"
        assert window.import_page.adapter.currentData() == "auto"
        assert window.import_page.details_button.text() == "查看问题行"
        assert window.import_page.export_failed_button.text() == "导出未通过文件"
        assert not window.import_page.export_failed_button.isEnabled()
        assert window.online_update_page.layout() is None
        assert not hasattr(window, "dashboard")
    finally:
        window.close()
        app.processEvents()


def test_import_page_exports_failed_file_paths(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    from PySide6.QtWidgets import QApplication
    from betalens_db_manager.gui import MainWindow
    from betalens_db_manager.gui_controller import FileImportItem, FileImportPlan
    import betalens_db_manager.gui_app as gui_app

    output = tmp_path / "failed-files.txt"
    accepted = tmp_path / "accepted.csv"
    rejected = tmp_path / "rejected.csv"
    fake_dialog = type(
        "FakeFileDialog",
        (),
        {"getSaveFileName": staticmethod(lambda *_args: (str(output), "文本文件 (*.txt)"))},
    )
    monkeypatch.setattr(gui_app, "QFileDialog", fake_dialog)

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.import_page.plan = FileImportPlan(
            table="daily_market",
            adapter="auto",
            mode="insert_only",
            options={},
            items=(
                FileImportItem(
                    path=accepted,
                    source_sha256="accepted",
                    preview_token="accepted",
                    validation={"ok": True},
                ),
                FileImportItem(
                    path=rejected,
                    source_sha256=None,
                    preview_token=None,
                    validation={"ok": False, "errors": ["字段不匹配"]},
                ),
            ),
            fingerprint="test",
            source_label=str(tmp_path),
        )
        window.import_page._update_actions()

        assert window.import_page.export_failed_button.isEnabled()
        window.import_page.export_failed_files()

        assert output.read_text(encoding="utf-8") == f"{rejected}\n"
        assert window.import_page.status.text() == "已导出 1 个未通过检查的文件"
    finally:
        window.close()
        app.processEvents()


def test_gui_registers_available_cjk_system_font(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    font_path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc"
    if not font_path.is_file():
        pytest.skip("Microsoft YaHei is not installed")

    from PySide6.QtWidgets import QApplication
    from betalens_db_manager.gui_app import _configure_cjk_font

    app = QApplication.instance() or QApplication([])
    _configure_cjk_font(app)

    assert app.font().family() in {"Microsoft YaHei UI", "Microsoft YaHei"}
