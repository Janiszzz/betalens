from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from .serialization import build_chart_data, build_position_table, read_table_page, write_table_parquet


class TablePagingTests(unittest.TestCase):
    def test_write_and_read_page(self) -> None:
        rows = [
            {"code": "000001.SZ", "direction": "buy", "amount": 10.0},
            {"code": "000002.SZ", "direction": "sell", "amount": 20.0},
            {"code": "000003.SZ", "direction": "buy", "amount": 30.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trades.parquet"
            meta = write_table_parquet(rows, path)
            page = read_table_page(path, page=2, size=1)

        self.assertEqual(meta, {"total": 3, "columns": ["code", "direction", "amount"]})
        self.assertEqual(page["total"], 3)
        self.assertEqual(page["pages"], 3)
        self.assertEqual(page["rows"], [{"code": "000002.SZ", "direction": "sell", "amount": 20.0}])

    def test_filter_query_and_clean_values(self) -> None:
        rows = [
            {"code": "000001.SZ", "direction": "buy", "amount": np.inf},
            {"code": "000002.SZ", "direction": "sell", "amount": np.nan},
            {"code": "600000.SH", "direction": "buy", "amount": 30.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trades.parquet"
            write_table_parquet(rows, path)
            page = read_table_page(
                path,
                page=1,
                size=10,
                query="000001",
                filters={"direction": "buy"},
            )

        self.assertEqual(page["total"], 1)
        self.assertEqual(page["rows"], [{"code": "000001.SZ", "direction": "buy", "amount": None}])

    def test_missing_table_returns_empty_page(self) -> None:
        page = read_table_page(Path("not-exists.parquet"), page=0, size=0)

        self.assertEqual(page, {"rows": [], "total": 0, "page": 1, "size": 1, "pages": 0})

    def test_unknown_filter_column_returns_empty_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trades.parquet"
            write_table_parquet([{"code": "000001.SZ"}], path)
            page = read_table_page(path, filters={"missing": "x"})

        self.assertEqual(page["total"], 0)
        self.assertEqual(page["rows"], [])

    def test_date_range_filter(self) -> None:
        rows = [
            {"date": "2024-01-01", "code": "000001.SZ"},
            {"date": "2024-01-02", "code": "000002.SZ"},
            {"date": "2024-01-03", "code": "000003.SZ"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "positions.parquet"
            write_table_parquet(rows, path)
            page = read_table_page(path, date_from="2024-01-02", date_to="2024-01-02")

        self.assertEqual(page["total"], 1)
        self.assertEqual(page["rows"], [{"date": "2024-01-02", "code": "000002.SZ"}])

    def test_position_weight_records_skip_zero_holdings(self) -> None:
        class FakeBacktest:
            nav = pd.Series([1.0, 1.01], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
            daily_pnl_total = pd.Series([0.0, 1.0], index=nav.index)
            daily_amount = pd.Series([100.0, 100.0], index=nav.index)
            daily_position_value = pd.DataFrame(
                {
                    "000001.SZ": [0.0, 30.0],
                    "000002.SZ": [40.0, 0.0],
                    "cash": [60.0, 70.0],
                },
                index=nav.index,
            )

        records = build_chart_data(FakeBacktest())["positionWeight"]

        self.assertNotIn({"date": "2024-01-01", "code": "000001.SZ", "weight": 0.0}, records)
        self.assertNotIn({"date": "2024-01-02", "code": "000002.SZ", "weight": 0.0}, records)
        self.assertTrue(
            any(r["date"] == "2024-01-01" and r["code"] == "000002.SZ" and r["weight"] == 0.4 and r["name"] for r in records)
        )
        self.assertTrue(
            any(r["date"] == "2024-01-02" and r["code"] == "000001.SZ" and r["weight"] == 0.3 and r["name"] for r in records)
        )
        self.assertIn({"date": "2024-01-02", "code": "现金", "name": "现金", "weight": 0.7}, records)

    def test_rebalance_holdings_include_factor_values(self) -> None:
        class FakeBacktest:
            nav = pd.Series([1.0], index=pd.to_datetime(["2024-01-01"]))
            daily_pnl_total = pd.Series([0.0], index=nav.index)
            daily_amount = pd.Series([100.0], index=nav.index)
            daily_position_value = pd.DataFrame(index=nav.index)
            actual_weight = pd.DataFrame(
                {"000001.SZ": [0.6], "000002.SZ": [0.0], "cash": [0.4]},
                index=pd.to_datetime(["2024-01-02 15:10:01"]),
            )

        factor_values = pd.DataFrame(
            {
                "信号日": pd.to_datetime(["2024-01-02 15:00:01", "2024-01-02 15:00:01"]),
                "股票代码": ["000001.SZ", "000002.SZ"],
                "因子值": [1.25, -0.5],
                "分组": [9, 1],
            }
        )

        records = build_chart_data(FakeBacktest(), factor_values)["rebalanceHoldings"]

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["code"], "000001.SZ")
        self.assertEqual(records[0]["factorValue"], 1.25)
        self.assertEqual(records[0]["group"], 9)
        self.assertEqual(records[0]["signalDate"], "2024-01-02")

    def test_position_table_skips_zero_quantity_rows(self) -> None:
        class FakeBacktest:
            idx = pd.to_datetime(["2024-01-01"])
            position = pd.DataFrame({"000001.SZ": [0.0], "000002.SZ": [100.0]}, index=idx)
            daily_position_value = pd.DataFrame({"000001.SZ": [0.0], "000002.SZ": [2000.0]}, index=idx)
            daily_pnl = pd.DataFrame({"000001.SZ": [0.0], "000002.SZ": [20.0]}, index=idx)
            cost_price = pd.DataFrame({"000001.SZ": [10.0], "000002.SZ": [20.0]}, index=idx)
            daily_amount = pd.Series([2000.0], index=idx)

        rows = build_position_table(FakeBacktest())

        self.assertEqual([row["代码"] for row in rows], ["000002.SZ"])
        self.assertEqual(rows[0]["数量"], 100.0)


if __name__ == "__main__":
    unittest.main()
