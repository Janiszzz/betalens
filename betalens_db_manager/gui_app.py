"""Four-page beginner desktop application for Betalens Database Manager.

The GUI deliberately exposes only the tasks a first-time local user needs:
inspect/create datasets, import files, inspect data, and a reserved update
page.  Schema migrations, legacy tables and task bookkeeping stay inside the
service layer instead of becoming another interface users must learn.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .constants import DEFAULT_LIMIT, INSERT_ONLY, UPSERT
from .db import QueryRequest
from .gui_controller import ConnectionDraft, FileImportPlan, GuiController
from .import_adapters import ADAPTERS
from .registry import DATASETS
from .utils import json_default


ADAPTER_LABELS = {
    "auto": "自动识别（推荐）",
    "standard_long": "标准六列长表",
    "wind_wide": "Wind 宽表",
    "ede": "EDE 宽表",
    "industry": "行业归属",
    "index_universe": "指数成分",
    "trade_status": "交易状态",
}


def _import_qt():
    try:
        from PySide6.QtCore import (
            QAbstractTableModel,
            QModelIndex,
            QObject,
            QRunnable,
            Qt,
            QThreadPool,
            Signal,
            Slot,
        )
        from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon
        from PySide6.QtWidgets import (
            QApplication,
            QComboBox,
            QDialog,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QProgressBar,
            QPushButton,
            QSpinBox,
            QSplitter,
            QStackedWidget,
            QTableView,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("缺少 PySide6，请先安装: pip install -e '.[gui]'") from exc
    return locals()


qt = _import_qt()
QAbstractTableModel = qt["QAbstractTableModel"]
QModelIndex = qt["QModelIndex"]
QObject = qt["QObject"]
QRunnable = qt["QRunnable"]
Qt = qt["Qt"]
QThreadPool = qt["QThreadPool"]
Signal = qt["Signal"]
Slot = qt["Slot"]
QColor = qt["QColor"]
QFont = qt["QFont"]
QFontDatabase = qt["QFontDatabase"]
QIcon = qt["QIcon"]
QApplication = qt["QApplication"]
QComboBox = qt["QComboBox"]
QDialog = qt["QDialog"]
QFileDialog = qt["QFileDialog"]
QFormLayout = qt["QFormLayout"]
QGridLayout = qt["QGridLayout"]
QGroupBox = qt["QGroupBox"]
QHBoxLayout = qt["QHBoxLayout"]
QLabel = qt["QLabel"]
QLineEdit = qt["QLineEdit"]
QMainWindow = qt["QMainWindow"]
QMessageBox = qt["QMessageBox"]
QPlainTextEdit = qt["QPlainTextEdit"]
QProgressBar = qt["QProgressBar"]
QPushButton = qt["QPushButton"]
QSpinBox = qt["QSpinBox"]
QSplitter = qt["QSplitter"]
QStackedWidget = qt["QStackedWidget"]
QTableView = qt["QTableView"]
QTabWidget = qt["QTabWidget"]
QVBoxLayout = qt["QVBoxLayout"]
QWidget = qt["QWidget"]


_CJK_FONT_FAMILIES = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Noto Sans SC",
    "Source Han Sans CN",
    "SimHei",
    "SimSun",
)
_APPLICATION_ICON_PATH = Path(__file__).resolve().parent / "assets" / "betalens_db_manager.ico"


def _application_icon() -> QIcon:
    return QIcon(str(_APPLICATION_ICON_PATH)) if _APPLICATION_ICON_PATH.is_file() else QIcon()


def _configure_windows_app_id() -> None:
    """Give the Windows taskbar a stable identity for this application icon."""

    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("com.betalens.dbmanager")
    except (AttributeError, OSError):
        pass


def _configure_cjk_font(app: QApplication) -> None:
    """Register a system CJK font before creating widgets when Qt has none."""

    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for filename in ("msyh.ttc", "NotoSansSC-VF.ttf", "SourceHanSansCN-Normal.otf"):
        path = font_dir / filename
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))

    available = set(QFontDatabase.families())
    family = next((name for name in _CJK_FONT_FAMILIES if name in available), None)
    if family is None:
        return
    font = QFont(app.font())
    font.setFamily(family)
    app.setFont(font)


class PandasTableModel(QAbstractTableModel):
    """Small read-only DataFrame model used by all four pages."""

    def __init__(self, data: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None):
        super().__init__()
        self._df = pd.DataFrame()
        self.set_data(data)

    @property
    def frame(self) -> pd.DataFrame:
        return self._df.copy()

    def value(self, row: int, column: str, default: Any = None) -> Any:
        if 0 <= row < len(self._df) and column in self._df.columns:
            return self._df.iloc[row][column]
        return default

    def set_data(self, data: pd.DataFrame | Sequence[Mapping[str, Any]] | None) -> None:
        self.beginResetModel()
        if isinstance(data, pd.DataFrame):
            self._df = data.copy().reset_index(drop=True)
        elif data is None:
            self._df = pd.DataFrame()
        else:
            self._df = pd.DataFrame(list(data)).reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):  # noqa: N802 - Qt API
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.BackgroundRole:
            text = self._display(value)
            if "需处理" in text or "失败" in text or "警告" in text:
                return QColor("#fff0f0")
            return None
        if role != Qt.DisplayRole:
            return None
        return self._display(value)

    @staticmethod
    def _display(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, default=json_default)
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802 - Qt API
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section]) if section < len(self._df.columns) else ""
        return str(section + 1)


class RejectedRowsDialog(QDialog):
    """Show the bounded rejected-row sample produced during file preflight."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("文件问题行")
        self.resize(1080, 520)
        self.model = PandasTableModel(rows)
        self.table = QTableView()
        _set_table(self.table, self.model)
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"预检问题样本：{len(rows)} 行"))
        layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(object)


class Worker(QRunnable):
    def __init__(self, operation: Callable[[], Any]):
        super().__init__()
        self.operation = operation
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.finished.emit(self.operation())
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class AsyncPage(QWidget):
    """Shared background execution with duplicate-action protection."""

    def __init__(self, pool: QThreadPool, parent=None):
        super().__init__(parent)
        self.pool = pool
        self._workers: dict[str, Worker] = {}

    def start_task(
        self,
        key: str,
        operation: Callable[[Callable[[dict[str, Any]], None]], Any],
        on_finished: Callable[[Any], None],
        on_error: Callable[[str], None] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> bool:
        if key in self._workers:
            return False
        holder: dict[str, Worker] = {}

        def invoke():
            return operation(holder["worker"].signals.progress.emit)

        worker = Worker(invoke)
        holder["worker"] = worker
        self._workers[key] = worker

        def complete(result, task_key=key):
            self._workers.pop(task_key, None)
            self._task_state_changed()
            on_finished(result)

        def failed(message, task_key=key):
            self._workers.pop(task_key, None)
            self._task_state_changed()
            if on_error is None:
                self.show_error(message)
            else:
                on_error(message)

        worker.signals.finished.connect(complete)
        worker.signals.error.connect(failed)
        if on_progress is not None:
            worker.signals.progress.connect(on_progress)
        self.pool.start(worker)
        self._task_state_changed()
        return True

    def _task_state_changed(self) -> None:
        """Hook for pages that need to disable duplicate buttons."""

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "操作失败", message)


def _set_table(table: QTableView, model: PandasTableModel) -> None:
    table.setModel(model)
    table.setAlternatingRowColors(True)
    table.setSortingEnabled(False)
    table.resizeColumnsToContents()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=json_default)


def _progress_update(bar: QProgressBar, status: QLabel, payload: Mapping[str, Any]) -> None:
    message = str(payload.get("message") or payload.get("phase") or "正在处理")
    status.setText(message)
    current = payload.get("current")
    total = payload.get("total")
    if current is not None and total:
        bar.setRange(0, 100)
        bar.setValue(max(0, min(100, int(100 * int(current) / int(total)))))
    else:
        bar.setRange(0, 0)


class ConnectionBar(AsyncPage):
    """Single, password-in-memory connection bar shared by the window."""

    connection_changed = Signal(object)

    def __init__(self, controller: GuiController, pool: QThreadPool, parent=None):
        super().__init__(pool, parent)
        self.controller = controller
        draft = controller.connection_draft
        self.host = QLineEdit(draft.host)
        self.port = QLineEdit(draft.port)
        self.database = QLineEdit(draft.dbname)
        self.user = QLineEdit(draft.user)
        self.password = QLineEdit(draft.password or "")
        self.password.setEchoMode(QLineEdit.Password)
        self.connect_button = QPushButton("连接")
        self.status = QLabel("尚未连接")

        layout = QGridLayout(self)
        layout.addWidget(QLabel("主机"), 0, 0)
        layout.addWidget(self.host, 0, 1)
        layout.addWidget(QLabel("端口"), 0, 2)
        layout.addWidget(self.port, 0, 3)
        layout.addWidget(QLabel("数据库"), 0, 4)
        layout.addWidget(self.database, 0, 5)
        layout.addWidget(QLabel("用户"), 0, 6)
        layout.addWidget(self.user, 0, 7)
        layout.addWidget(QLabel("密码"), 0, 8)
        layout.addWidget(self.password, 0, 9)
        layout.addWidget(self.connect_button, 0, 10)
        layout.addWidget(self.status, 1, 0, 1, 11)
        self.connect_button.clicked.connect(self.connect)

    def _draft(self) -> ConnectionDraft:
        return ConnectionDraft(
            host=self.host.text(),
            port=self.port.text(),
            dbname=self.database.text(),
            user=self.user.text(),
            password=self.password.text() or None,
        )

    def connect(self) -> None:
        try:
            draft = self._draft()
            draft.as_config()
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.connect_button.setEnabled(False)
        self.status.setText("正在连接...")
        self.start_task(
            "connect",
            lambda _progress: self.controller.connect(draft),
            self._connected,
            self._connect_error,
        )

    def _connected(self, result: Mapping[str, Any]) -> None:
        self.connect_button.setEnabled(True)
        self.set_result(result)
        self.connection_changed.emit(dict(result))

    def _connect_error(self, message: str) -> None:
        self.connect_button.setEnabled(True)
        self.status.setText("连接失败")
        self.show_error(message)

    def set_result(self, result: Mapping[str, Any]) -> None:
        state = str(result.get("status") or result.get("state") or "offline")
        if state == "online":
            text = f"已连接到 {result.get('database') or self.database.text()}"
        elif state == "database_missing":
            text = "服务器可达，但此数据库尚未建立。请到“数据表”页选择一个表并建立。"
        elif state == "offline":
            text = "尚未连接"
        else:
            text = f"无法连接: {result.get('error') or '请检查主机、端口、用户和密码'}"
        self.status.setText(text)


class TableCatalogPage(AsyncPage):
    """Logical dataset list and safe one-click contract bootstrap."""

    schema_changed = Signal(object)

    def __init__(self, controller: GuiController, pool: QThreadPool, parent=None):
        super().__init__(pool, parent)
        self.controller = controller
        self.connection_state = "offline"
        self.catalog_model = PandasTableModel()
        self.catalog = QTableView()
        _set_table(self.catalog, self.catalog_model)
        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        self.refresh_button = QPushButton("刷新表清单")
        self.create_button = QPushButton("建立所选表")
        self.status = QLabel("未连接；仍可查看 Betalens 的表契约")

        bar = QHBoxLayout()
        bar.addWidget(self.refresh_button)
        bar.addWidget(self.create_button)
        bar.addWidget(self.status, 1)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.catalog)
        split.addWidget(self.metadata)
        split.setSizes([420, 260])
        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(split)

        self.catalog.selectionModel().selectionChanged.connect(self._selection_changed)
        self.refresh_button.clicked.connect(self.refresh)
        self.create_button.clicked.connect(self.create_selected)
        self._set_catalog(self.controller.table_catalog())
        self._update_actions()

    def set_connection_state(self, state: str) -> None:
        self.connection_state = state
        self._update_actions()
        self.refresh()

    def _update_actions(self) -> None:
        can_create = self.connection_state in {"online", "database_missing"}
        creating = "create-table" in self._workers
        self.create_button.setEnabled(
            can_create and self._selected_table() is not None and not creating
        )
        self.refresh_button.setEnabled("catalog" not in self._workers and not creating)

    def _task_state_changed(self) -> None:
        self._update_actions()

    def _selected_table(self) -> str | None:
        indexes = self.catalog.selectionModel().selectedRows()
        if not indexes:
            return None
        value = self.catalog_model.value(indexes[0].row(), "table")
        return str(value) if value else None

    def _set_catalog(self, rows: Sequence[Mapping[str, Any]]) -> None:
        selected = self._selected_table()
        self.catalog_model.set_data(rows)
        self.catalog.resizeColumnsToContents()
        if self.catalog_model.rowCount() == 0:
            self.metadata.clear()
            return
        target_row = 0
        if selected:
            for row in range(self.catalog_model.rowCount()):
                if self.catalog_model.value(row, "table") == selected:
                    target_row = row
                    break
        self.catalog.selectRow(target_row)
        self._show_metadata(str(self.catalog_model.value(target_row, "table")))

    def refresh(self) -> None:
        if self.connection_state != "online":
            self._set_catalog(self.controller.table_catalog())
            self.status.setText(
                "数据库尚未建立，可选择任意逻辑表并建立完整 Betalens 结构"
                if self.connection_state == "database_missing"
                else "未连接；这里显示的是 Betalens 表契约"
            )
            self._update_actions()
            return
        self.refresh_button.setEnabled(False)
        self.status.setText("正在读取表清单...")
        self.start_task(
            "catalog",
            lambda _progress: self.controller.table_catalog(),
            self._catalog_loaded,
            self._catalog_error,
        )

    def _catalog_loaded(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._set_catalog(rows)
        self.status.setText(f"已读取 {len(rows)} 个逻辑表")
        self._update_actions()

    def _catalog_error(self, message: str) -> None:
        self.status.setText("读取表清单失败")
        self._update_actions()
        self.show_error(message)

    def _selection_changed(self, *_args) -> None:
        table = self._selected_table()
        self._update_actions()
        if table:
            self._show_metadata(table)

    def _show_metadata(self, table: str) -> None:
        if self.connection_state != "online":
            self.metadata.setPlainText(_json_text(self.controller.table_metadata(table)))
            return
        self.metadata.setPlainText("正在读取表元信息...")
        self.start_task(
            "metadata",
            lambda _progress: self.controller.table_metadata(table),
            lambda payload: self.metadata.setPlainText(_json_text(payload)),
            lambda message: self.metadata.setPlainText(message),
        )

    def create_selected(self) -> None:
        table = self._selected_table()
        if not table:
            QMessageBox.warning(self, "请选择表", "请先在列表中选择要建立的逻辑表")
            return
        answer = QMessageBox.question(
            self,
            "建立数据库结构",
            f"将按 Betalens 契约补齐“{table}”需要的数据库结构。\n"
            "首次执行会建立完整基础结构，不会删除已有数据。继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.create_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.status.setText("正在建立数据库结构...")
        self.start_task(
            "create-table",
            lambda _progress: self.controller.create_selected_table(table),
            self._created,
            self._create_error,
        )

    def _created(self, report: Mapping[str, Any]) -> None:
        warnings = [str(item) for item in report.get("warnings", [])]
        self.status.setText(
            "数据库结构已建立；历史数据审计有警告" if warnings else "数据库结构已建立并完成核验"
        )
        connection = dict(self.controller.connection_details)
        if not connection:
            connection = {"status": self.controller.connection_state}
        connection["bootstrap_warnings"] = warnings
        self.schema_changed.emit(connection)

    def _create_error(self, message: str) -> None:
        self.status.setText("建立结构失败；未删除已有数据")
        self._update_actions()
        self.show_error(message)


class FileImportPage(AsyncPage):
    """Single file and recursive folder import through one preflight flow."""

    def __init__(self, controller: GuiController, pool: QThreadPool, parent=None):
        super().__init__(pool, parent)
        self.controller = controller
        self.connection_state = "offline"
        self.source_paths: list[Path] = []
        self.files: list[Path] = []
        self.plan: FileImportPlan | None = None

        self.target = QComboBox()
        for name, spec in DATASETS.items():
            if spec.writable:
                self.target.addItem(name, name)
        self.adapter = QComboBox()
        self.mode = QComboBox()
        self.mode.addItem("仅插入（跳过已有记录）", INSERT_ONLY)
        self.mode.addItem("更新已有记录", UPSERT)
        self.options = QPlainTextEdit("{}")
        self.options.setMaximumHeight(70)
        self.advanced = QGroupBox("高级设置（可留空）")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        advanced_layout = QVBoxLayout(self.advanced)
        advanced_layout.addWidget(QLabel("导入选项 JSON，例如列映射：{\"column_map\": {\"日期\": \"datetime\"}}"))
        advanced_layout.addWidget(self.options)

        self.choose_files_button = QPushButton("选择文件")
        self.choose_folder_button = QPushButton("选择文件夹")
        self.check_button = QPushButton("检查文件")
        self.details_button = QPushButton("查看问题行")
        self.export_failed_button = QPushButton("导出未通过文件")
        self.import_button = QPushButton("导入可用文件")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status = QLabel("先连接数据库，再选择一个文件或文件夹")
        self.file_model = PandasTableModel()
        self.file_table = QTableView()
        _set_table(self.file_table, self.file_model)

        form = QFormLayout()
        form.addRow("导入到", self.target)
        form.addRow("文件类型", self.adapter)
        form.addRow("写入方式", self.mode)
        controls = QHBoxLayout()
        controls.addWidget(self.choose_files_button)
        controls.addWidget(self.choose_folder_button)
        controls.addWidget(self.check_button)
        controls.addWidget(self.details_button)
        controls.addWidget(self.export_failed_button)
        controls.addWidget(self.import_button)
        controls.addWidget(self.status, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.advanced)
        layout.addLayout(controls)
        layout.addWidget(self.progress)
        layout.addWidget(self.file_table)

        self.target.currentIndexChanged.connect(self._target_changed)
        self.adapter.currentIndexChanged.connect(self.invalidate_plan)
        self.mode.currentIndexChanged.connect(self.invalidate_plan)
        self.options.textChanged.connect(self.invalidate_plan)
        self.choose_files_button.clicked.connect(self.choose_files)
        self.choose_folder_button.clicked.connect(self.choose_folder)
        self.check_button.clicked.connect(self.preflight)
        self.details_button.clicked.connect(self.show_rejected_rows)
        self.export_failed_button.clicked.connect(self.export_failed_files)
        self.import_button.clicked.connect(self.run_import)
        self._refresh_adapters()
        self._update_actions()

    def set_connection_state(self, state: str) -> None:
        self.connection_state = state
        self._update_actions()
        if state != "online":
            self.status.setText("请先连接并建立数据库结构")

    def _target_changed(self, *_args) -> None:
        self._refresh_adapters()
        self.invalidate_plan()

    def _refresh_adapters(self) -> None:
        target = str(self.target.currentData() or self.target.currentText())
        previous = str(self.adapter.currentData() or "")
        names = [
            name
            for name in ADAPTERS.names(include_aliases=False)
            if target in ADAPTERS.resolve(name).allowed_targets
        ]
        self.adapter.blockSignals(True)
        self.adapter.clear()
        for name in names:
            self.adapter.addItem(ADAPTER_LABELS.get(name, name), name)
        preferred = target if target in names else "auto"
        selected = previous if previous in names else preferred
        index = self.adapter.findData(selected)
        if index >= 0:
            self.adapter.setCurrentIndex(index)
        self.adapter.blockSignals(False)

    def _update_actions(self) -> None:
        online = self.connection_state == "online"
        scanning = "scan" in self._workers
        checking = "preflight" in self._workers
        importing = "import" in self._workers
        self.choose_files_button.setEnabled(not scanning and not checking and not importing)
        self.choose_folder_button.setEnabled(not scanning and not checking and not importing)
        self.check_button.setEnabled(online and bool(self.files) and not scanning and not checking and not importing)
        self.details_button.setEnabled(
            self._has_rejection_details() and not scanning and not checking and not importing
        )
        self.export_failed_button.setEnabled(
            bool(self._failed_file_paths()) and not scanning and not checking and not importing
        )
        self.import_button.setEnabled(
            online
            and self.plan is not None
            and bool(self.plan.ready_items)
            and not scanning
            and not checking
            and not importing
        )

    def _task_state_changed(self) -> None:
        self._update_actions()

    def _has_rejection_details(self) -> bool:
        return bool(
            self.plan
            and any(
                item.rejected_preview
                or item.error
                or item.validation.get("errors")
                for item in self.plan.items
            )
        )

    def _failed_file_paths(self) -> tuple[Path, ...]:
        if self.plan is None:
            return ()
        return tuple(item.path for item in self.plan.items if not item.ready)

    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择一个或多个文件",
            "",
            "数据文件 (*.csv *.csv.gz *.xls *.xlsx *.parquet *.pq);;所有文件 (*)",
        )
        if paths:
            self._set_sources([Path(path) for path in paths])

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择要递归导入的文件夹")
        if path:
            self._set_sources([Path(path)])

    def _set_sources(self, paths: Sequence[Path]) -> None:
        self.source_paths = [path.expanduser() for path in paths]
        self.files = []
        self.invalidate_plan()
        self.file_model.set_data([])
        self.status.setText("正在递归查找可导入文件...")
        self._update_actions()
        self.start_task(
            "scan",
            lambda _progress: self.controller.discover_files(self.source_paths),
            self._scanned,
            self._scan_error,
        )

    def _scanned(self, files: Sequence[Path]) -> None:
        self.files = list(files)
        self.file_model.set_data(
            [
                {"文件": path.name, "完整路径": str(path), "状态": "待检查"}
                for path in self.files
            ]
        )
        self.file_table.resizeColumnsToContents()
        self.status.setText(f"已找到 {len(self.files)} 个可导入文件")
        self._update_actions()

    def _scan_error(self, message: str) -> None:
        self.status.setText("扫描文件失败")
        self._update_actions()
        self.show_error(message)

    def invalidate_plan(self, *_args) -> None:
        self.plan = None
        self._update_actions()

    def _options(self) -> dict[str, Any]:
        if not self.advanced.isChecked():
            return {}
        text = self.options.toPlainText().strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"高级设置不是合法 JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError("高级设置必须是 JSON object")
        return payload

    def _current_spec(self) -> tuple[str, str, str, dict[str, Any]]:
        table = str(self.target.currentData() or self.target.currentText())
        adapter = str(self.adapter.currentData() or self.adapter.currentText())
        mode = str(self.mode.currentData() or INSERT_ONLY)
        return table, adapter, mode, self._options()

    def preflight(self) -> None:
        if not self.files:
            QMessageBox.warning(self, "没有文件", "请选择文件或文件夹")
            return
        try:
            table, adapter, mode, options = self._current_spec()
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.plan = None
        self.progress.setRange(0, 0)
        self.status.setText("正在检查文件格式、字段和数据...")
        self._update_actions()
        self.start_task(
            "preflight",
            lambda progress: self.controller.preflight_import(
                self.files,
                table=table,
                adapter=adapter,
                mode=mode,
                options=options,
                progress=progress,
            ),
            self._preflight_done,
            self._preflight_error,
            self._progress,
        )

    def _preflight_done(self, plan: FileImportPlan) -> None:
        self.plan = plan
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.file_model.set_data([item.as_dict() for item in plan.items])
        self.file_table.resizeColumnsToContents()
        self.status.setText(f"检查完成：{len(plan.ready_items)}/{len(plan.items)} 个文件可以导入")
        self._update_actions()

    def _preflight_error(self, message: str) -> None:
        self.progress.setRange(0, 100)
        self.status.setText("文件检查失败")
        self._update_actions()
        self.show_error(message)

    def show_rejected_rows(self) -> None:
        if self.plan is None:
            return
        rows: list[dict[str, Any]] = []
        for item in self.plan.items:
            for rejected in item.rejected_preview:
                rows.append(
                    {
                        "文件": Path(str(rejected.get("source_file") or item.path)).name,
                        "行号": rejected.get("source_row", rejected.get("_source_row", "")),
                        "字段": rejected.get("field", ""),
                        "指标": rejected.get("metric", ""),
                        "原始值": rejected.get("raw_value", ""),
                        "原因": rejected.get("reason", rejected.get("_errors", "")),
                    }
                )
            errors = [str(error) for error in item.validation.get("errors", [])]
            if item.error:
                errors.append(item.error)
            for error in dict.fromkeys(errors):
                rows.append(
                    {
                        "文件": item.path.name,
                        "行号": "",
                        "字段": "",
                        "指标": "",
                        "原始值": "",
                        "原因": error,
                    }
                )
        if rows:
            RejectedRowsDialog(rows, self).exec()

    def export_failed_files(self) -> None:
        paths = self._failed_file_paths()
        if not paths:
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出未通过检查的文件",
            "未通过检查文件列表.txt",
            "文本文件 (*.txt)",
        )
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".txt")
        try:
            target.write_text("\n".join(str(item) for item in paths) + "\n", encoding="utf-8")
            self.status.setText(f"已导出 {len(paths)} 个未通过检查的文件")
        except OSError as exc:
            self.show_error(str(exc))

    def run_import(self) -> None:
        if self.plan is None or not self.plan.ready_items:
            QMessageBox.warning(self, "需要检查", "请先检查文件，并确认至少一个文件可以导入")
            return
        if self.plan.mode == UPSERT:
            answer = QMessageBox.question(
                self,
                "确认更新已有记录",
                "更新模式会覆盖同一键的不同数值。确认继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.progress.setRange(0, 0)
        self.status.setText("正在导入文件...")
        self._update_actions()
        self.start_task(
            "import",
            lambda progress: self.controller.run_import_plan(self.plan, progress=progress),
            self._import_done,
            self._import_error,
            self._progress,
        )

    def _import_done(self, report: Mapping[str, Any]) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        rows: list[dict[str, Any]] = []
        for item in report.get("items", []):
            rows.append(
                {
                    "文件": Path(str(item.get("path") or item.get("source_file") or "")).name,
                    "状态": item.get("status"),
                    "新增": item.get("inserted_rows", item.get("inserted")),
                    "更新": item.get("updated_rows", item.get("updated")),
                    "跳过": item.get("skipped_rows", item.get("skipped")),
                    "拒绝": item.get("rejected_rows", item.get("rejected")),
                    "错误": item.get("error", ""),
                }
            )
        self.file_model.set_data(rows)
        self.file_table.resizeColumnsToContents()
        status = str(report.get("status", "completed"))
        self.status.setText(
            "导入完成" if status == "completed" else "导入完成，但有文件未成功处理"
        )
        self.plan = None
        self._update_actions()

    def _import_error(self, message: str) -> None:
        self.progress.setRange(0, 100)
        self.status.setText("导入失败")
        self._update_actions()
        self.show_error(message)

    def _progress(self, payload: Mapping[str, Any]) -> None:
        _progress_update(self.progress, self.status, payload)


class ExplorerPage(AsyncPage):
    """Filtered logical queries, restricted SQL, and data-quality checks."""

    def __init__(self, controller: GuiController, pool: QThreadPool, parent=None):
        super().__init__(pool, parent)
        self.controller = controller
        self.connection_state = "offline"
        self._page_tokens: list[str | None] = [None]
        self._page_index = 0
        self._current_frame = pd.DataFrame()

        self.mode = QComboBox()
        self.mode.addItem("筛选查询", "filter")
        self.mode.addItem("SQL 查询", "sql")
        self.mode.addItem("脏数据诊断", "diagnose")
        self.stack = QStackedWidget()
        self.filter_page = self._build_filter_page()
        self.sql_page = self._build_sql_page()
        self.diagnose_page = self._build_diagnose_page()
        self.stack.addWidget(self.filter_page)
        self.stack.addWidget(self.sql_page)
        self.stack.addWidget(self.diagnose_page)
        self.result_model = PandasTableModel()
        self.result_table = QTableView()
        _set_table(self.result_table, self.result_model)
        self.export_button = QPushButton("导出当前结果")
        self.export_button.setEnabled(False)
        self.status = QLabel("请先连接数据库")

        header = QHBoxLayout()
        header.addWidget(QLabel("功能"))
        header.addWidget(self.mode)
        header.addWidget(self.export_button)
        header.addWidget(self.status, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.stack)
        layout.addWidget(self.result_table, 1)

        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.export_button.clicked.connect(self.export_current)
        self._set_online_controls()

    def _build_filter_page(self) -> QWidget:
        page = QWidget()
        self.filter_table = QComboBox()
        self.filter_table.addItems(list(DATASETS))
        self.filter_code = QLineEdit()
        self.filter_metric = QLineEdit()
        self.filter_start = QLineEdit()
        self.filter_end = QLineEdit()
        self.filter_limit = QSpinBox()
        self.filter_limit.setRange(1, 5000)
        self.filter_limit.setValue(DEFAULT_LIMIT)
        self.filter_button = QPushButton("查询")
        self.previous_button = QPushButton("上一页")
        self.next_button = QPushButton("下一页")
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        form = QFormLayout()
        form.addRow("数据表", self.filter_table)
        form.addRow("代码（可选）", self.filter_code)
        form.addRow("指标（可选）", self.filter_metric)
        form.addRow("开始时间（可选）", self.filter_start)
        form.addRow("结束时间（可选）", self.filter_end)
        form.addRow("每页行数", self.filter_limit)
        buttons = QHBoxLayout()
        buttons.addWidget(self.filter_button)
        buttons.addWidget(self.previous_button)
        buttons.addWidget(self.next_button)
        layout = QVBoxLayout(page)
        layout.addLayout(form)
        layout.addLayout(buttons)
        self.filter_button.clicked.connect(self.start_filter_query)
        self.previous_button.clicked.connect(self.previous_page)
        self.next_button.clicked.connect(self.next_page)
        return page

    def _build_sql_page(self) -> QWidget:
        page = QWidget()
        self.sql_input = QPlainTextEdit()
        self.sql_input.setPlaceholderText("SELECT * FROM public.daily_market WHERE code = '000001.SZ'")
        self.sql_limit = QSpinBox()
        self.sql_limit.setRange(1, 5000)
        self.sql_limit.setValue(DEFAULT_LIMIT)
        self.sql_button = QPushButton("执行只读 SQL")
        controls = QHBoxLayout()
        controls.addWidget(QLabel("最多返回"))
        controls.addWidget(self.sql_limit)
        controls.addWidget(QLabel("行"))
        controls.addWidget(self.sql_button)
        layout = QVBoxLayout(page)
        layout.addWidget(self.sql_input)
        layout.addLayout(controls)
        self.sql_button.clicked.connect(self.run_sql)
        return page

    def _build_diagnose_page(self) -> QWidget:
        page = QWidget()
        self.diagnose_table = QComboBox()
        self.diagnose_table.addItems(list(DATASETS))
        self.diagnose_limit = QSpinBox()
        self.diagnose_limit.setRange(1, 100)
        self.diagnose_limit.setValue(10)
        self.diagnose_button = QPushButton("检查脏数据")
        form = QFormLayout()
        form.addRow("数据表", self.diagnose_table)
        form.addRow("每项样本数", self.diagnose_limit)
        form.addRow("", self.diagnose_button)
        layout = QVBoxLayout(page)
        layout.addLayout(form)
        self.diagnose_button.clicked.connect(self.run_diagnosis)
        return page

    def set_connection_state(self, state: str) -> None:
        self.connection_state = state
        self._set_online_controls()
        if state != "online":
            self.status.setText("请先连接并建立数据库结构")

    def _set_online_controls(self) -> None:
        online = self.connection_state == "online"
        busy = bool(self._workers)
        self.filter_button.setEnabled(online and not busy)
        self.sql_button.setEnabled(online and not busy)
        self.diagnose_button.setEnabled(online and not busy)
        self.previous_button.setEnabled(online and not busy and self._page_index > 0)
        self.next_button.setEnabled(online and not busy and not self._current_frame.empty)
        self.export_button.setEnabled(not self._current_frame.empty and not busy)

    def _task_state_changed(self) -> None:
        self._set_online_controls()

    def _mode_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self._current_frame = pd.DataFrame()
        self.result_model.set_data([])
        self.export_button.setEnabled(False)

    def _filter_request(self, page_token: str | None) -> QueryRequest:
        return QueryRequest(
            table=self.filter_table.currentText(),
            code=self.filter_code.text().strip() or None,
            metric=self.filter_metric.text().strip() or None,
            start_date=self.filter_start.text().strip() or None,
            end_date=self.filter_end.text().strip() or None,
            limit=self.filter_limit.value(),
            page_token=page_token,
        )

    def start_filter_query(self) -> None:
        self._page_tokens = [None]
        self._page_index = 0
        self._fetch_filter_page()

    def _fetch_filter_page(self) -> None:
        request = self._filter_request(self._page_tokens[self._page_index])
        self.status.setText("正在查询...")
        self._set_online_controls()
        self.start_task(
            "filter-query",
            lambda _progress: self.controller.query(request),
            lambda frame: self._filter_done(frame, request),
            self._query_error,
        )

    def _filter_done(self, frame: pd.DataFrame, request: QueryRequest) -> None:
        self._set_result(frame)
        self.status.setText(f"第 {self._page_index + 1} 页，返回 {len(frame)} 行")
        self.previous_button.setEnabled(self.connection_state == "online" and self._page_index > 0)
        self.next_button.setEnabled(
            self.connection_state == "online" and len(frame) == request.limit and not frame.empty
        )

    def next_page(self) -> None:
        if self._current_frame.empty:
            return
        token = self.controller.client.make_page_token(self._current_frame.iloc[-1])
        self._page_tokens = self._page_tokens[: self._page_index + 1]
        self._page_tokens.append(token)
        self._page_index += 1
        self._fetch_filter_page()

    def previous_page(self) -> None:
        if self._page_index <= 0:
            return
        self._page_index -= 1
        self._fetch_filter_page()

    def run_sql(self) -> None:
        statement = self.sql_input.toPlainText()
        self.status.setText("正在执行只读 SQL...")
        self._set_online_controls()
        self.start_task(
            "sql-query",
            lambda _progress: self.controller.execute_sql(statement, limit=self.sql_limit.value()),
            self._sql_done,
            self._query_error,
        )

    def _sql_done(self, frame: pd.DataFrame) -> None:
        self._set_result(frame)
        self.status.setText(f"SQL 返回 {len(frame)} 行")

    def run_diagnosis(self) -> None:
        table = self.diagnose_table.currentText()
        self.status.setText("正在检查数据...")
        self._set_online_controls()
        self.start_task(
            "diagnosis",
            lambda _progress: self.controller.diagnose_dirty_data(
                table, sample_limit=self.diagnose_limit.value()
            ),
            self._diagnosis_done,
            self._query_error,
        )

    def _diagnosis_done(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._set_result(pd.DataFrame(list(rows)))
        self.status.setText(f"检查完成，返回 {len(rows)} 项结果")

    def _set_result(self, frame: pd.DataFrame) -> None:
        self._current_frame = frame.copy()
        self.result_model.set_data(frame)
        self.result_table.resizeColumnsToContents()
        self._set_online_controls()

    def _query_error(self, message: str) -> None:
        self.status.setText("查询或诊断失败")
        self._set_online_controls()
        self.show_error(message)

    def export_current(self) -> None:
        if self._current_frame.empty:
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出当前结果",
            "",
            "CSV (*.csv);;Excel (*.xlsx);;Parquet (*.parquet)",
        )
        if not path:
            return
        target = Path(path)
        try:
            if target.suffix.lower() == ".xlsx" or "Excel" in selected_filter:
                self._current_frame.to_excel(target, index=False)
            elif target.suffix.lower() in {".parquet", ".pq"} or "Parquet" in selected_filter:
                self._current_frame.to_parquet(target, index=False)
            else:
                self._current_frame.to_csv(target, index=False, encoding="utf-8-sig")
            self.status.setText(f"已导出 {len(self._current_frame)} 行")
        except Exception as exc:
            self.show_error(str(exc))


class OnlineUpdatePage(QWidget):
    """Reserved empty page for a future connected-data update workflow."""

    pass


class MainWindow(QMainWindow):
    """The complete desktop app: exactly four beginner-oriented pages."""

    def __init__(self, controller: GuiController | None = None):
        super().__init__()
        self.setWindowTitle("Betalens Database Manager")
        self.setWindowIcon(_application_icon())
        self.resize(1280, 820)
        self.pool = QThreadPool.globalInstance()
        self.controller = controller or GuiController()

        self.connection_bar = ConnectionBar(self.controller, self.pool)
        self.table_page = TableCatalogPage(self.controller, self.pool)
        self.import_page = FileImportPage(self.controller, self.pool)
        self.query_page = ExplorerPage(self.controller, self.pool)
        self.online_update_page = OnlineUpdatePage()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.table_page, "数据表")
        self.tabs.addTab(self.import_page, "文件导入")
        self.tabs.addTab(self.query_page, "查询与诊断")
        self.tabs.addTab(self.online_update_page, "联网更新")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.connection_bar)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("尚未连接数据库")

        self.connection_bar.connection_changed.connect(self._connection_changed)
        self.table_page.schema_changed.connect(self._schema_changed)
        self._connection_changed({"status": "offline"})

    def _connection_changed(self, result: Mapping[str, Any]) -> None:
        state = str(result.get("status") or result.get("state") or "offline")
        self.connection_bar.set_result(result)
        self.table_page.set_connection_state(state)
        self.import_page.set_connection_state(state)
        self.query_page.set_connection_state(state)
        if state == "online":
            warnings = list(result.get("bootstrap_warnings") or [])
            self.statusBar().showMessage(
                "数据库结构已建立；历史数据审计有警告，详见运行报告"
                if warnings
                else "已连接数据库"
            )
        elif state == "database_missing":
            self.statusBar().showMessage("数据库尚未建立；请在“数据表”页建立所选表")
        else:
            self.statusBar().showMessage("数据库未连接")

    def _schema_changed(self, result: Mapping[str, Any]) -> None:
        self._connection_changed(result)


def main(argv: list[str] | None = None) -> int:
    _configure_windows_app_id()
    app = QApplication(argv or sys.argv)
    _configure_cjk_font(app)
    app.setWindowIcon(_application_icon())
    window = MainWindow()
    window.show()
    return app.exec()


__all__ = [
    "ConnectionBar",
    "ExplorerPage",
    "FileImportPage",
    "MainWindow",
    "OnlineUpdatePage",
    "PandasTableModel",
    "TableCatalogPage",
    "main",
]
