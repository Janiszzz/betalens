from __future__ import annotations

import sys
import warnings

import pandas as pd

from betalens.datafeed.core import Datafeed
from betalens_db_manager.importers import load_ede, normalize_import_frame
from betalens_db_manager.records import ImportRecordStore
from betalens_db_manager.utils import clean_database_config
from betalens_db_manager.validators import validate_import_frame


def test_importing_datafeed_does_not_import_pyside():
    import betalens.datafeed  # noqa: F401

    assert "PySide6" not in sys.modules


def test_ede_importer_outputs_remark_not_note(tmp_path):
    path = tmp_path / "sample_ede.xlsx"
    pd.DataFrame(
        {
            "证券代码": ["000001.SZ"],
            "证券简称": ["平安银行"],
            "收盘价(元) [交易日期] 最新 [单位] 元": [10.25],
        }
    ).to_excel(path, index=False)

    df = load_ede(path, default_datetime="2026-07-01 15:00:01")

    assert "remark" in df.columns
    assert "note" not in df.columns
    assert isinstance(df.loc[0, "remark"], dict)
    assert df.loc[0, "metric"] == "收盘价(元)"


def test_validator_rejects_unsafe_metrics_and_nan():
    df = pd.DataFrame(
        {
            "datetime": ["2026-07-01 15:00:01", "2026-07-01 15:00:01"],
            "code": ["000001.SZ", "000001.SZ"],
            "name": ["平安银行", "平安银行"],
            "metric": ["Unnamed: 24", "收盘价(元)"],
            "value": [1.0, float("nan")],
            "remark": [None, None],
        }
    )

    report = validate_import_frame(df)

    assert not report.ok
    assert any("Unnamed" in err for err in report.errors)
    assert any("NaN" in err for err in report.errors)


def test_record_store_roundtrip(tmp_path):
    store = ImportRecordStore(tmp_path / "records.jsonl", tmp_path / "jobs")
    store.append({"job_id": "job-1", "status": "completed", "inserted_rows": 3})
    store.append({"job_id": "job-2", "status": "failed", "error": "boom"})

    rows = store.read_all()

    assert [row["job_id"] for row in rows] == ["job-1", "job-2"]
    assert store.job_log_path("job-1").name == "job-1.log"


def test_normalize_import_frame_renames_note_to_remark():
    df = pd.DataFrame(
        {
            "datetime": ["2026-07-01 15:00:01"],
            "code": ["000001.SZ"],
            "name": ["平安银行"],
            "metric": ["收盘价(元)"],
            "value": ["10.1"],
            "note": [{"source": "test"}],
        }
    )

    out = normalize_import_frame(df)

    assert "remark" in out.columns
    assert "note" not in out.columns
    assert out.loc[0, "value"] == 10.1


def test_datafeed_management_warning_helper():
    obj = Datafeed.__new__(Datafeed)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj._warn_manager_deprecated("insert_ede_file")

    assert caught
    assert issubclass(caught[0].category, DeprecationWarning)
    assert "betalens_db_manager" in str(caught[0].message)


def test_clean_database_config_drops_comments():
    cfg = clean_database_config({"_comment": "x", "dbname": "datafeed", "user": "postgres", "host": "localhost"})

    assert "_comment" not in cfg
    assert cfg["dbname"] == "datafeed"
