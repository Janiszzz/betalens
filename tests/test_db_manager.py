from __future__ import annotations

import sys

import pandas as pd
import betalens_db_manager.import_adapters as import_adapters

from betalens_db_manager.adapters import (
    apply_time_alignment,
    fetch_daily_market,
    parse_metric_column,
)
from betalens_db_manager.adapters.files import _detect_csv_encoding
from betalens_db_manager.importers import load_ede, normalize_import_frame
from betalens_db_manager.import_adapters import collect_import_batches, infer_adapter, load_import_batches
from betalens_db_manager.jobs import ImportJobRunner
from betalens_db_manager.records import ImportRecordStore
from betalens_db_manager.utils import clean_database_config
from betalens_db_manager.validators import validate_import_frame


def _track_auto_source_reads(monkeypatch):
    original = import_adapters.iter_file_chunks
    calls = 0

    def tracked(*args, **kwargs):
        nonlocal calls
        calls += 1
        yield from original(*args, **kwargs)

    monkeypatch.setattr(import_adapters, "iter_file_chunks", tracked)
    return lambda: calls


def test_importing_datafeed_does_not_import_pyside():
    already_loaded = "PySide6" in sys.modules
    import betalens.datafeed  # noqa: F401

    assert ("PySide6" in sys.modules) is already_loaded


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


def test_auto_adapter_preprocesses_nonstandard_ede_wide_file(tmp_path, monkeypatch):
    path = tmp_path / "raw_export.csv"
    pd.DataFrame(
        {
            "证券代码": ["000001.SZ"],
            "证券简称": ["平安银行"],
            "收盘价(元) [20240102] 最新 [单位] 元": [10.25],
        }
    ).to_csv(path, index=False)
    reads = _track_auto_source_reads(monkeypatch)

    frame, rejected = collect_import_batches(
        load_import_batches("auto", path, table="daily_market")
    )

    assert rejected.empty
    assert frame[["code", "name", "metric", "value"]].to_dict("records") == [
        {"code": "000001.SZ", "name": "平安银行", "metric": "收盘价(元)", "value": 10.25}
    ]
    assert frame.loc[0, "datetime"] == pd.Timestamp("2024-01-02 15:00:01")
    assert isinstance(frame.loc[0, "remark"], dict)
    assert infer_adapter(path) == "auto"
    assert reads() == 1


def test_auto_adapter_reuses_first_chunk_for_standard_long(tmp_path, monkeypatch):
    path = tmp_path / "standard.csv"
    pd.DataFrame(
        {
            "datetime": ["2024-01-02 15:00:01"],
            "code": ["000001.SZ"],
            "name": ["平安银行"],
            "metric": ["收盘价(元)"],
            "value": [10.25],
        }
    ).to_csv(path, index=False)
    reads = _track_auto_source_reads(monkeypatch)
    frame, rejected = collect_import_batches(load_import_batches("auto", path, table="daily_market"))

    assert reads() == 1
    assert rejected.empty
    assert len(frame) == 1


def test_standard_long_skips_na_value_marker_without_rejecting_file(tmp_path):
    path = tmp_path / "market.csv"
    pd.DataFrame(
        {
            "datetime": ["2026-07-01 15:00:01", "2026-07-01 15:00:01"],
            "code": ["000001.SZ", "000002.SZ"],
            "name": ["平安银行", "万科A"],
            "metric": ["收盘价(元)", "收盘价(元)"],
            "value": [10.25, "N/A"],
        }
    ).to_csv(path, index=False)

    batches = list(load_import_batches("standard_long", path, table="daily_market"))
    batch = batches[0]
    preview = ImportJobRunner().preview(
        path,
        import_type="standard_long",
        table="daily_market",
        inspect_database=False,
    )

    assert batch.source_rows == 2
    assert batch.rejected.empty
    assert batch.frame[["code", "value"]].to_dict("records") == [{"code": "000001.SZ", "value": 10.25}]
    assert preview["validation"]["ok"]
    assert preview["summary"]["rows"] == 1
    assert preview["summary"]["rejected_rows"] == 0


def test_standard_long_keeps_non_missing_invalid_values_as_rejected(tmp_path):
    path = tmp_path / "market.csv"
    pd.DataFrame(
        {
            "datetime": ["2026-07-01 15:00:01", "2026-07-01 15:00:01"],
            "code": ["000001.SZ", "000002.SZ"],
            "name": ["平安银行", "万科A"],
            "metric": ["收盘价(元)", "收盘价(元)"],
            "value": ["N/A", "not-a-number"],
        }
    ).to_csv(path, index=False)

    batch = next(load_import_batches("standard_long", path, table="daily_market"))

    assert batch.frame.empty
    assert batch.rejected["reason"].tolist() == ["value 不是有限数值"]
    assert batch.rejected["raw_value"].tolist() == ["not-a-number"]
    assert batch.rejected["source_row"].tolist() == [2]


def test_auto_adapter_ignores_empty_trailing_wide_export_column(tmp_path, monkeypatch):
    path = tmp_path / "daily_wide.csv"
    pd.DataFrame(
        {
            "代码": ["000002.SZ", "000002.SZ"],
            "简称": ["万科A", "万科A"],
            "日期": ["2026-02-02", "2026-02-03"],
            "收盘价(元)": [8.5, 8.6],
            "Unnamed: 24": [None, None],
        }
    ).to_csv(path, index=False)
    reads = _track_auto_source_reads(monkeypatch)

    frame, rejected = collect_import_batches(
        load_import_batches("auto", path, table="daily_market")
    )

    assert rejected.empty
    assert frame[["metric", "value"]].to_dict("records") == [
        {"metric": "收盘价(元)", "value": 8.5},
        {"metric": "收盘价(元)", "value": 8.6},
    ]
    assert reads() == 1


def test_encoding_detection_handles_multibyte_character_at_sample_boundary(tmp_path):
    path = tmp_path / "boundary.csv"
    sample_size = 1024 * 1024
    header = "代码,简称,日期\n".encode("gb18030")
    filler = b"x" * (sample_size - len(header) - 1)
    path.write_bytes(header + filler + b"\xb9\xfa\n")

    assert _detect_csv_encoding(path) == "gb18030"


def test_ede_adapter_uses_date_column_for_daily_wide_file(tmp_path):
    path = tmp_path / "000001.SZ.CSV"
    pd.DataFrame(
        {
            "代码": ["000001.SZ", "000001.SZ"],
            "简称": ["平安银行", "平安银行"],
            "日期": ["2026-07-01", "2026-07-02"],
            "开盘价(元)": [10.0, 10.1],
            "收盘价(元)": [10.2, 10.3],
        }
    ).to_csv(path, index=False, encoding="gb18030")

    auto, auto_rejected = collect_import_batches(
        load_import_batches("auto", path, table="daily_market")
    )
    ede, ede_rejected = collect_import_batches(
        load_import_batches("ede", path, table="daily_market")
    )

    assert auto_rejected.empty
    assert ede_rejected.empty
    assert ede[["datetime", "code", "name", "metric", "value"]].equals(
        auto[["datetime", "code", "name", "metric", "value"]]
    )
    assert len(ede) == 4
    assert ede["datetime"].dt.normalize().tolist() == [
        pd.Timestamp("2026-07-01"),
        pd.Timestamp("2026-07-02"),
        pd.Timestamp("2026-07-01"),
        pd.Timestamp("2026-07-02"),
    ]


def test_source_adapters_are_manager_owned():
    assert parse_metric_column.__module__.startswith("betalens_db_manager.adapters")


def test_time_alignment_handles_existing_timestamps():
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01", "2026-07-01"]),
            "metric": ["开盘价(元)", "收盘价(元)"],
        }
    )

    aligned = apply_time_alignment(source, date_column="date")

    assert aligned.loc[0, "date"] == pd.Timestamp("2026-07-01 09:30:01")
    assert aligned.loc[1, "date"] == pd.Timestamp("2026-07-01 15:00:01")


def test_wind_adapter_can_use_injected_client():
    class Response:
        def __init__(self, *, data=None, times=None):
            self.ErrorCode = 0
            self.Data = data or []
            self.Times = times or []

    class FakeWind:
        def start(self):
            return Response()

        def wsd(self, code, fields, start_date, end_date, options):
            assert code == "000001.SZ"
            assert fields == ["open", "close"]
            return Response(
                data=[[10.0], [10.5]],
                times=[pd.Timestamp("2026-07-01")],
            )

        def wss(self, code, field):
            return Response(data=[["平安银行"]])

    frame = fetch_daily_market(
        ["000001.SZ"],
        "2026-07-01",
        "2026-07-01",
        fields=["open", "close"],
        client=FakeWind(),
    )

    assert list(frame.columns) == ["datetime", "code", "name", "metric", "value", "remark"]
    assert list(frame["datetime"]) == [
        pd.Timestamp("2026-07-01 09:30:01"),
        pd.Timestamp("2026-07-01 15:00:01"),
    ]
    assert frame["value"].tolist() == [10.0, 10.5]


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


def test_clean_database_config_drops_comments():
    cfg = clean_database_config({"_comment": "x", "dbname": "datafeed", "user": "postgres", "host": "localhost"})

    assert "_comment" not in cfg
    assert cfg["dbname"] == "datafeed"
