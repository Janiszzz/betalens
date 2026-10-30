"""Repeatable, opt-in nearest benchmark in disposable PG/CH namespaces.

Run with the same test connection environment as test_datafeed_integration.
Prints JSON measurements only; never source records or credentials. First-run
latency is reported, but shared server caches are deliberately not flushed.
"""
from datetime import date
import json
from statistics import median
import time
import tracemalloc
from types import MethodType

import numpy as np
import pandas as pd

from betalens.datafeed import Datafeed
from betalens.datafeed.contracts import json_value, timestamp
from betalens.datafeed.query import _nearest
from betalens.datafeed.reference import calendar_days
from datafeed_support import MARKET_TYPES, FIN_TYPES, quote, evidence, reference_reader, calendar_scope
from test_datafeed_integration import isolated_databases


def capture_execute(reader, sql, params=None):
    result = reader._benchmark_execute(sql, params)
    reader._benchmark_queries.append((sql, params))
    reader._benchmark_rows += len(result)
    reader._benchmark_bytes += len(json.dumps(result, default=json_value, ensure_ascii=False).encode())
    return result


def capture_post(session, *args, **kwargs):
    response = session._benchmark_post(*args, **kwargs)
    summary = json.loads(response.headers.get("X-ClickHouse-Summary", "{}"))
    session._benchmark_read_rows += int(summary.get("read_rows", 0))
    session._benchmark_read_bytes += int(summary.get("read_bytes", 0))
    return response


def plan_scans(plan):
    count = 0
    if "Scan" in plan.get("Node Type", ""):
        count += (plan.get("Actual Rows", 0) + plan.get("Rows Removed by Filter", 0)) * plan.get("Actual Loops", 1)
    return count + sum(plan_scans(child) for child in plan.get("Plans", []))


def transfer_stats(data, args, engine, filters):
    reader = data._reader
    reader._benchmark_execute = reader.execute
    reader._benchmark_queries = []
    reader._benchmark_rows = reader._benchmark_bytes = 0
    reader.execute = MethodType(capture_execute, reader)
    if data.backend == "clickhouse":
        reader.session._benchmark_post = reader.session.post
        reader.session._benchmark_read_rows = reader.session._benchmark_read_bytes = 0
        reader.session.post = MethodType(capture_post, reader.session)
    try:
        _nearest(data, *args, engine=engine, filters=filters)
    finally:
        del reader.execute
        if data.backend == "clickhouse":
            del reader.session.post
    scans = 0
    if data.backend == "postgres":
        for sql, params in reader._benchmark_queries:
            reader.cursor.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, params)
            scans += plan_scans(reader.cursor.fetchone()["QUERY PLAN"][0]["Plan"])
        reader.conn.rollback()
    else:
        scans = reader.session._benchmark_read_rows
    return {"queries": len(reader._benchmark_queries), "transferred_rows": reader._benchmark_rows,
            "decoded_json_bytes": reader._benchmark_bytes, "scan_rows": scans}


def measure(data, args, engine, filters):
    start = time.perf_counter()
    result = _nearest(data, *args, engine=engine, filters=filters)
    first = time.perf_counter() - start
    _nearest(data, *args, engine=engine, filters=filters)  # Warm-up.
    times = []
    for _ in range(5):
        start = time.perf_counter()
        _nearest(data, *args, engine=engine, filters=filters)
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    _nearest(data, *args, engine=engine, filters=filters)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics = {"first_ms": first * 1000, "median_ms": median(times) * 1000,
               "p95_ms": float(np.percentile(times, 95)) * 1000, "python_peak_mb": peak / 1024**2,
               **transfer_stats(data, args, engine, filters)}
    return result, metrics


def main():
    days = pd.bdate_range("2024-01-02", periods=20)
    codes = [f"{n:06}.SZ" for n in range(5000)]
    rows = [quote(code, str(day.date()) + " 15:00", 10 + n / 1000) for code in codes for n, day in enumerate(days)]
    financial = [{"security_code": code, "period_end": date(2023, 12, 31), "event_date": date(2023, 12, 31),
        "statement_type": "income", "report_type": "1", "version_id": str(n), "revenue": 100 + n,
        **evidence(str(day.date()) + " 15:00")} for code in codes[:200] for n, day in enumerate(days)]
    cal = reference_reader("calendar", [("r", [calendar_scope(str(d.date())) for d in pd.date_range(days[0], days[-1])], "publish")])
    tables = {"market_daily_post_adjusted": rows, "fundamental_periodic": financial, **cal.tables}
    types = {"market_daily_post_adjusted": MARKET_TYPES, "fundamental_periodic": FIN_TYPES, **cal.types}
    scenarios = {
        "single_code_many_dates": ("market_daily_post_adjusted", codes[:1], [str(d.date()) + " 16:00" for d in days], "close", "before", None),
        "market_single_date": ("market_daily_post_adjusted", codes, [str(days[-1].date()) + " 16:00"], "close", "before", None),
        "market_four_dates": ("market_daily_post_adjusted", codes, [str(d.date()) + " 16:00" for d in days[::5]], "close", "before", None),
        "financial_revisions": ("fundamental_periodic", codes[:200], [str(d.date()) + " 16:00" for d in days[::5]], "revenue", "before", None),
    }
    with isolated_databases(tables, types) as (configs, _, _):
        for backend, config in configs.items():
            for name, args in scenarios.items():
                filters = {"statement_type": "income", "report_type": "1"} if args[0] == "fundamental_periodic" else {"asset_type": "stock"}
                with Datafeed(args[0], config, backend=backend) as data:
                    python_result, python_metrics = measure(data, args, "python", filters)
                    sql_result, sql_metrics = measure(data, args, "sql", filters)
                    pd.testing.assert_frame_equal(python_result, sql_result)
                    print(json.dumps({"backend": backend, "scenario": name, "equal": True,
                                      "python": python_metrics, "sql": sql_metrics}), flush=True)
            with Datafeed("trade_calendar_versions", config, backend=backend) as data:
                elapsed = []
                for _ in range(6):
                    start = time.perf_counter()
                    with data._reader.snapshot():
                        calendar_days(data._reader, days[0], days[-1], as_of="2024-02-01")
                    elapsed.append(time.perf_counter() - start)
                print(json.dumps({"backend": backend, "scenario": "verified_reference_snapshot",
                                  "median_ms": median(elapsed[1:]) * 1000,
                                  "note": "manifest/content verification remains Python; not replaced by an unverified SQL max"}), flush=True)


if __name__ == "__main__":
    main()
