"""PySide6 desktop GUI for Betalens Database Manager."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .constants import ALLOWED_TABLES, DEFAULT_LIMIT, IMPORT_MODES, IMPORT_TYPES, INSERT_ONLY
from .db import DatabaseClient, QueryRequest
from .importers import infer_import_type
from .jobs import ImportJobRunner
from .records import ImportRecordStore


def _import_qt():
    try:
        from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QRunnable, Qt, QThreadPool, Signal, Slot
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QPlainTextEdit,
            QSpinBox,
            QTableView,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("缺少 PySide6，请先安装: pip install PySide6>=6.7") from exc
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
QApplication = qt["QApplication"]
QCheckBox = qt["QCheckBox"]
QComboBox = qt["QComboBox"]
QFileDialog = qt["QFileDialog"]
QFormLayout = qt["QFormLayout"]
QGridLayout = qt["QGridLayout"]
QGroupBox = qt["QGroupBox"]
QHBoxLayout = qt["QHBoxLayout"]
QLabel = qt["QLabel"]
QLineEdit = qt["QLineEdit"]
QMainWindow = qt["QMainWindow"]
QMessageBox = qt["QMessageBox"]
QPushButton = qt["QPushButton"]
QPlainTextEdit = qt["QPlainTextEdit"]
QSpinBox = qt["QSpinBox"]
QTableView = qt["QTableView"]
QTabWidget = qt["QTabWidget"]
QTextEdit = qt["QTextEdit"]
QVBoxLayout = qt["QVBoxLayout"]
QWidget = qt["QWidget"]


class PandasTableModel(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame | list[dict[str, Any]] | None = None):
        super().__init__()
        if isinstance(data, list):
            data = pd.DataFrame(data)
        self._df = data if data is not None else pd.DataFrame()

    def set_data(self, data: pd.DataFrame | list[dict[str, Any]] | None):
        self.beginResetModel()
        if isinstance(data, list):
            data = pd.DataFrame(data)
        self._df = data if data is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        value = self._df.iat[index.row(), index.column()]
        if value is None or (pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False):
            return ""
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section]) if section < len(self._df.columns) else ""
        return str(section + 1)


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)


class Worker(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class OverviewTab(QWidget):
    def __init__(self, client: DatabaseClient, pool: QThreadPool):
        super().__init__()
        self.client = client
        self.pool = pool
        self.model = PandasTableModel()
        self.status = QLabel("未连接")
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSortingEnabled(True)
        refresh = QPushButton("刷新总览")
        refresh.clicked.connect(self.refresh)
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(refresh)
        top.addWidget(self.status, 1)
        layout.addLayout(top)
        layout.addWidget(self.table)
        self.refresh()

    def refresh(self):
        self.status.setText("正在读取数据库元数据...")
        worker = Worker(self._load)
        worker.signals.finished.connect(self._loaded)
        worker.signals.error.connect(self._error)
        self.pool.start(worker)

    def _load(self):
        connection = self.client.test_connection()
        overview = self.client.table_overview()
        rows = []
        for row in overview:
            dr = row.get("date_range") or {}
            rows.append(
                {
                    "table": row.get("table_name"),
                    "estimated_rows": row.get("estimated_rows"),
                    "size": row.get("total_size"),
                    "min_dt": dr.get("min_dt"),
                    "max_dt": dr.get("max_dt"),
                    "warnings": "; ".join(row.get("warnings") or []),
                    "comment": row.get("table_comment"),
                }
            )
        return {"connection": connection, "rows": rows}

    def _loaded(self, result):
        conn = result.get("connection") or {}
        self.status.setText(f"已连接: {conn.get('current_database')} / {conn.get('current_user')}")
        self.model.set_data(result.get("rows", []))
        self.table.resizeColumnsToContents()

    def _error(self, message):
        self.status.setText("读取失败")
        QMessageBox.critical(self, "数据库总览失败", message)


class QueryTab(QWidget):
    def __init__(self, client: DatabaseClient, pool: QThreadPool):
        super().__init__()
        self.client = client
        self.pool = pool
        self.model = PandasTableModel()
        self.table_box = QComboBox()
        self.table_box.addItems(ALLOWED_TABLES)
        self.code = QLineEdit()
        self.metric = QLineEdit()
        self.start_date = QLineEdit()
        self.end_date = QLineEdit()
        self.limit = QSpinBox()
        self.limit.setRange(1, 5000)
        self.limit.setValue(DEFAULT_LIMIT)
        self.status = QLabel("准备查询")
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSortingEnabled(True)
        query_btn = QPushButton("查询")
        query_btn.clicked.connect(self.query)

        form = QFormLayout()
        form.addRow("表", self.table_box)
        form.addRow("code", self.code)
        form.addRow("metric", self.metric)
        form.addRow("开始日期", self.start_date)
        form.addRow("结束日期", self.end_date)
        form.addRow("limit", self.limit)
        controls = QGroupBox("查询条件")
        controls.setLayout(form)
        layout = QVBoxLayout(self)
        layout.addWidget(controls)
        bar = QHBoxLayout()
        bar.addWidget(query_btn)
        bar.addWidget(self.status, 1)
        layout.addLayout(bar)
        layout.addWidget(self.table)

    def query(self):
        request = QueryRequest(
            table=self.table_box.currentText(),
            code=self.code.text().strip() or None,
            metric=self.metric.text().strip() or None,
            start_date=self.start_date.text().strip() or None,
            end_date=self.end_date.text().strip() or None,
            limit=self.limit.value(),
        )
        self.status.setText("查询中...")
        worker = Worker(self.client.query_table, request)
        worker.signals.finished.connect(self._loaded)
        worker.signals.error.connect(self._error)
        self.pool.start(worker)

    def _loaded(self, df):
        self.model.set_data(df)
        self.table.resizeColumnsToContents()
        self.status.setText(f"返回 {len(df)} 行")

    def _error(self, message):
        self.status.setText("查询失败")
        QMessageBox.critical(self, "查询失败", message)


class ImportTab(QWidget):
    def __init__(self, runner: ImportJobRunner, pool: QThreadPool):
        super().__init__()
        self.runner = runner
        self.pool = pool
        self.selected_file = QLineEdit()
        self.import_type = QComboBox()
        self.import_type.addItems(IMPORT_TYPES)
        self.target_table = QComboBox()
        self.target_table.addItems(ALLOWED_TABLES)
        self.mode = QComboBox()
        self.mode.addItems(IMPORT_MODES)
        self.mode.setCurrentText(INSERT_ONLY)
        self.allow_unsafe = QCheckBox("允许 Unnamed:* 指标")
        self.allow_nan = QCheckBox("允许 NaN/Inf value")
        self.preview_model = PandasTableModel()
        self.preview_table = QTableView()
        self.preview_table.setModel(self.preview_model)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.status = QLabel("请选择文件")

        choose = QPushButton("选择文件")
        choose.clicked.connect(self.choose_file)
        preview = QPushButton("预览 / 校验")
        preview.clicked.connect(self.preview)
        run = QPushButton("确认导入")
        run.clicked.connect(self.run_import)

        form = QGridLayout()
        form.addWidget(QLabel("文件"), 0, 0)
        form.addWidget(self.selected_file, 0, 1)
        form.addWidget(choose, 0, 2)
        form.addWidget(QLabel("导入类型"), 1, 0)
        form.addWidget(self.import_type, 1, 1)
        form.addWidget(QLabel("目标表"), 2, 0)
        form.addWidget(self.target_table, 2, 1)
        form.addWidget(QLabel("写入模式"), 3, 0)
        form.addWidget(self.mode, 3, 1)
        form.addWidget(self.allow_unsafe, 4, 1)
        form.addWidget(self.allow_nan, 5, 1)

        btns = QHBoxLayout()
        btns.addWidget(preview)
        btns.addWidget(run)
        btns.addWidget(self.status, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(btns)
        layout.addWidget(QLabel("预览"))
        layout.addWidget(self.preview_table)
        layout.addWidget(QLabel("任务日志"))
        layout.addWidget(self.log)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择导入文件", "", "Data Files (*.xlsx *.xls *.csv);;All Files (*)")
        if not path:
            return
        self.selected_file.setText(path)
        guessed = infer_import_type(path)
        self.import_type.setCurrentText(guessed)
        if guessed == "index_universe":
            self.target_table.setCurrentText("index_universe")
        elif guessed == "trade_status":
            self.target_table.setCurrentText("trade_status")
        else:
            self.target_table.setCurrentText("daily_market")

    def preview(self):
        path = self.selected_file.text().strip()
        if not path:
            QMessageBox.warning(self, "缺少文件", "请先选择文件")
            return
        self.status.setText("预览中...")
        worker = Worker(self.runner.preview, path, self.import_type.currentText())
        worker.signals.finished.connect(self._preview_loaded)
        worker.signals.error.connect(self._error)
        self.pool.start(worker)

    def _preview_loaded(self, result):
        self.preview_model.set_data(result.get("preview", []))
        self.preview_table.resizeColumnsToContents()
        validation = result.get("validation", {})
        errors = validation.get("errors") or []
        warnings = validation.get("warnings") or []
        self.status.setText(f"预览完成: {result.get('summary', {}).get('rows', 0)} 行")
        self.log.appendPlainText("预览完成")
        if errors:
            self.log.appendPlainText("错误: " + "; ".join(errors))
        if warnings:
            self.log.appendPlainText("警告: " + "; ".join(warnings))

    def run_import(self):
        path = self.selected_file.text().strip()
        if not path:
            QMessageBox.warning(self, "缺少文件", "请先选择文件")
            return
        if self.mode.currentText() != INSERT_ONLY:
            ok = QMessageBox.warning(
                self,
                "确认 upsert",
                "upsert 会覆盖已存在 key 的 name/value/remark。确认继续?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ok != QMessageBox.Yes:
                return
        self.status.setText("导入中...")
        self.log.clear()

        worker_ref: dict[str, Worker] = {}

        def progress(line: str):
            worker_ref["worker"].signals.progress.emit(line)

        worker = Worker(
            self.runner.run,
            path,
            self.target_table.currentText(),
            self.import_type.currentText(),
            self.mode.currentText(),
            None,
            self.allow_unsafe.isChecked(),
            self.allow_nan.isChecked(),
            progress,
        )
        worker_ref["worker"] = worker
        worker.signals.progress.connect(self.log.appendPlainText)
        worker.signals.finished.connect(self._run_finished)
        worker.signals.error.connect(self._error)
        self.pool.start(worker)

    def _run_finished(self, record):
        self.status.setText(f"导入{record.get('status')}: {record.get('job_id')}")
        self.log.appendPlainText(str(record))

    def _error(self, message):
        self.status.setText("操作失败")
        QMessageBox.critical(self, "导入失败", message)


class RecordsTab(QWidget):
    def __init__(self, store: ImportRecordStore):
        super().__init__()
        self.store = store
        self.model = PandasTableModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        refresh = QPushButton("刷新记录")
        refresh.clicked.connect(self.refresh)
        open_log = QPushButton("打开选中日志")
        open_log.clicked.connect(self.open_selected_log)
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(refresh)
        bar.addWidget(open_log)
        layout.addLayout(bar)
        layout.addWidget(self.table)
        layout.addWidget(QLabel("日志内容"))
        layout.addWidget(self.log)
        self.refresh()

    def refresh(self):
        rows = list(reversed(self.store.read_all()))
        self.model.set_data(rows)
        self.table.resizeColumnsToContents()

    def open_selected_log(self):
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            return
        row = indexes[0].row()
        data = self.model._df.iloc[row].to_dict()
        path = data.get("log_path")
        if not path or not Path(path).exists():
            self.log.setPlainText("日志文件不存在")
            return
        self.log.setPlainText(Path(path).read_text(encoding="utf-8", errors="replace"))


class SettingsTab(QWidget):
    def __init__(self, client: DatabaseClient):
        super().__init__()
        cfg = client.db_config
        text = QTextEdit()
        text.setReadOnly(True)
        safe_cfg = {k: ("***" if k == "password" else v) for k, v in cfg.items()}
        text.setPlainText(
            "当前数据库配置\n"
            f"{safe_cfg}\n\n"
            "API 数据源\n"
            "联网 API 更新为预留能力，第一版未启用。\n"
            "后续会通过 DataProvider(source_kind='api') 接入。"
        )
        layout = QVBoxLayout(self)
        layout.addWidget(text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Betalens Database Manager")
        self.resize(1180, 760)
        self.pool = QThreadPool.globalInstance()
        self.client = DatabaseClient()
        self.store = ImportRecordStore()
        self.runner = ImportJobRunner(self.client, self.store)

        tabs = QTabWidget()
        tabs.addTab(OverviewTab(self.client, self.pool), "数据库总览")
        tabs.addTab(QueryTab(self.client, self.pool), "数据表格")
        tabs.addTab(ImportTab(self.runner, self.pool), "文件导入")
        tabs.addTab(RecordsTab(self.store), "导入记录")
        tabs.addTab(SettingsTab(self.client), "设置/API预留")
        self.setCentralWidget(tabs)


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
