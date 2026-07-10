from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import serialization as serialization_module
from .serialization import build_chart_data, build_position_table, build_timing_payload, read_table_page, write_table_parquet


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
                {"000001.SZ": [0.6], "000002.SZ": [-0.4], "cash": [0.8]},
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

        original_name_map = serialization_module._name_map_for_codes
        serialization_module._name_map_for_codes = lambda codes: {
            "000001.SZ": "平安银行",
            "000002.SZ": "万科A",
        }
        try:
            records = build_chart_data(FakeBacktest(), factor_values)["rebalanceHoldings"]
        finally:
            serialization_module._name_map_for_codes = original_name_map

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["code"], "000001.SZ")
        self.assertEqual(records[0]["name"], "平安银行(000001.SZ)")
        self.assertEqual(records[0]["factorValue"], 1.25)
        self.assertEqual(records[0]["group"], 9)
        self.assertEqual(records[0]["signalDate"], "2024-01-02")
        self.assertEqual(records[1]["code"], "000002.SZ")
        self.assertEqual(records[1]["name"], "万科A(000002.SZ)")
        self.assertEqual(records[1]["side"], "short")
        self.assertEqual(records[1]["weight"], -0.4)
        self.assertEqual(records[1]["group"], 1)

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

    def test_timing_payload_summarizes_trades_and_position(self) -> None:
        class FakeBacktest:
            idx = pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                ]
            )
            nav = pd.Series([1.00, 1.02, 1.04, 1.03, 1.00, 0.98, 0.99], index=idx)
            actual_weight = pd.DataFrame(
                {
                    "000001.SZ": [0.0, 0.8, 0.8, 0.0, 0.5, 0.5, 0.0],
                    "cash": [1.0, 0.2, 0.2, 1.0, 0.5, 0.5, 1.0],
                },
                index=idx,
            )
            daily_pnl_total = pd.Series([0.0, 20.0, 20.0, -10.0, -30.0, -20.0, 10.0], index=idx)
            daily_amount = pd.Series([100.0, 102.0, 104.0, 103.0, 100.0, 98.0, 99.0], index=idx)
            cost_price = pd.DataFrame({"000001.SZ": [10.0, 10.2, 10.4, 10.3, 10.0, 9.8, 9.9]}, index=idx)

        payload = build_timing_payload(FakeBacktest())
        metrics = {row["label"]: row["value"] for row in payload["metrics"]}

        self.assertEqual(metrics["交易次数"], 2)
        self.assertAlmostEqual(metrics["交易胜率"], 0.5)
        self.assertAlmostEqual(metrics["平均仓位"], 2.6 / 7)
        self.assertAlmostEqual(metrics["开仓占比"], 4 / 7)
        self.assertGreater(metrics["赔率"], 0)
        self.assertEqual(len(payload["tables"]["tradeSegments"]), 2)
        self.assertEqual(payload["tables"]["tradeSegments"][0]["startDate"], "2024-01-02")
        self.assertEqual(payload["tables"]["tradeSegments"][0]["endDate"], "2024-01-03")
        self.assertTrue(payload["charts"]["navPrice"])
        self.assertTrue(payload["charts"]["position"])
        self.assertTrue(payload["charts"]["tradeReturns"])

    def test_timing_payload_consumes_factor_values_with_timing_fields(self) -> None:
        class FakeBacktest:
            idx = pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                ]
            )
            nav = pd.Series([1.00, 1.02, 1.04, 1.03, 1.00, 0.98, 0.99, 1.01], index=idx)
            actual_weight = pd.DataFrame(
                {
                    "000001.SZ": [0.0, 0.8, 0.8, 0.0, 0.5, 0.5, 0.0, 1.0],
                    "cash": [1.0, 0.2, 0.2, 1.0, 0.5, 0.5, 1.0, 0.0],
                },
                index=idx,
            )
            daily_pnl_total = pd.Series([0.0, 20.0, 20.0, -10.0, -30.0, -20.0, 10.0, 25.0], index=idx)
            daily_amount = pd.Series([100.0, 102.0, 104.0, 103.0, 100.0, 98.0, 99.0, 101.0], index=idx)
            cost_price = pd.DataFrame({"000001.SZ": [10.0, 10.2, 10.4, 10.3, 10.0, 9.8, 9.9, 10.1]}, index=idx)

        factor_values = pd.DataFrame(
            {
                "信号日": pd.to_datetime(
                    [
                        "2024-01-01 00:10:00",
                        "2024-01-02 00:10:00",
                        "2024-01-03 00:10:00",
                        "2024-01-04 00:10:00",
                        "2024-01-05 00:10:00",
                        "2024-01-06 00:10:00",
                        "2024-01-07 00:10:00",
                        "2024-01-08 00:10:00",
                    ]
                ),
                "股票代码": ["000001.SZ"] * 8,
                "因子值": [0.1, 0.5, 0.2, 0.7, 0.3, 0.1, 0.6, 0.4],
                "滚动均值": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
                "滚动标准差": [0.01, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05],
                "历史阈值": [0.06, 0.12, 0.18, 0.23, 0.29, 0.34, 0.40, 0.45],
                "分组": [0, 1, 0, 1, 0, 0, 1, 0],
                "是否触发": [False, True, False, True, False, False, True, False],
                "目标仓位": [0.0, 0.8, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            }
        )

        payload = build_timing_payload(FakeBacktest(), factor_values)
        metrics = {row["label"]: row["value"] for row in payload["metrics"]}

        self.assertTrue(payload["charts"]["predictionScatter"])
        self.assertTrue(payload["tables"]["prediction"])
        self.assertIsNotNone(metrics["主预测周期 IC"])
        self.assertIsNotNone(metrics["Beta"])

    def test_timing_payload_aligns_intraday_indexes_to_daily_records(self) -> None:
        class FakeBacktest:
            nav_idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00", "2024-01-03 00:00"])
            weight_idx = pd.to_datetime(["2024-01-01 00:10", "2024-01-02 00:10", "2024-01-03 00:10"])
            price_idx = pd.to_datetime(["2024-01-01 15:00", "2024-01-02 15:00", "2024-01-03 15:00"])
            nav = pd.Series([1.00, 1.01, 1.02], index=nav_idx)
            actual_weight = pd.DataFrame(
                {"000001.SZ": [0.0, 0.8, 0.8], "cash": [1.0, 0.2, 0.2]},
                index=weight_idx,
            )
            daily_pnl_total = pd.Series([0.0, 1.0, 1.0], index=nav_idx)
            daily_amount = pd.Series([100.0, 101.0, 102.0], index=nav_idx)
            cost_price = pd.DataFrame({"000001.SZ": [10.0, 10.1, 10.2]}, index=price_idx)

        records = build_timing_payload(FakeBacktest())["charts"]["navPrice"]

        self.assertEqual([row["date"] for row in records], ["2024-01-01", "2024-01-02", "2024-01-03"])
        self.assertEqual([row["nav"] for row in records], [1.0, 1.01, 1.02])
        self.assertEqual([row["price"] for row in records], [10.0, 10.1, 10.2])
        self.assertEqual([row["position"] for row in records], [0.0, 0.8, 0.8])

    def test_timing_prediction_falls_back_without_factor_values(self) -> None:
        class FakeBacktest:
            idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
            nav = pd.Series([1.0, 1.01, 1.02], index=idx)
            actual_weight = pd.DataFrame({"000001.SZ": [0.0, 1.0, 0.0], "cash": [1.0, 0.0, 1.0]}, index=idx)
            daily_pnl_total = pd.Series([0.0, 1.0, 1.0], index=idx)
            daily_amount = pd.Series([100.0, 101.0, 102.0], index=idx)
            cost_price = pd.DataFrame({"000001.SZ": [10.0, 10.1, 10.2]}, index=idx)

        payload = build_timing_payload(FakeBacktest(), pd.DataFrame())

        self.assertEqual(payload["charts"]["predictionScatter"], [])
        self.assertEqual(payload["tables"]["prediction"], [])
        self.assertIn("tradeSegments", payload["tables"])


if __name__ == "__main__":
    unittest.main()
