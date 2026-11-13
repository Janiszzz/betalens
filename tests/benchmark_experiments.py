"""Run explicitly on ext4: python tests/benchmark_experiments.py --directory outputs/benchmarks.

Generates isolated disposable data, reports serial encoding/commit/read timings,
and uses the real archive format. It never connects to market databases.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from betalens.experiments import ExperimentStore
from betalens.experiments.codec import encode_table
from betalens.experiments.snapshot import Snapshot


def write_worker(path, factor, worker):
    started = time.perf_counter()
    with ExperimentStore(path) as store:
        for i in range(10):
            store._save_snapshot(Snapshot(metrics={"score": worker * 10 + i}), factor_id=factor, source_id=f"{worker}:{i}")
    return time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    output = {}
    frame = pd.DataFrame(np.random.default_rng(7).standard_normal((2520, 500)),
        index=pd.date_range("2015-01-01", periods=2520), columns=[f"{i:06d}.SZ" for i in range(500)])
    with tempfile.TemporaryDirectory(prefix="experiments-benchmark-", dir=directory) as path:
        with ExperimentStore(path) as store:
            factor = store.register_factor(key="benchmark", name="性能验证")
            started = time.perf_counter(); descriptor, blocks = encode_table(frame)
            output["encode_ms"] = (time.perf_counter() - started) * 1000
            output["table_bytes"] = descriptor["bytes"]
            started = time.perf_counter()
            experiment = store._save_snapshot(Snapshot(tables={"weights": frame}), factor_id=factor)
            output["encode_and_durable_commit_ms"] = (time.perf_counter() - started) * 1000
            started = time.perf_counter(); restored = store.read_table(experiment, "weights")
            output["read_and_decode_ms"] = (time.perf_counter() - started) * 1000
            pd.testing.assert_frame_equal(frame, restored)
            for i in range(1000):
                store._save_snapshot(Snapshot(metrics={"score": i}), factor_id=factor, source_id=f"summary:{i}")
            timings = []
            for _ in range(5):
                started = time.perf_counter(); result = store.query(factor_id=factor, page_size=50)
                timings.append((time.perf_counter() - started) * 1000)
            output["summary_1000_median_ms"] = float(np.median(timings))
            started = time.perf_counter()
            with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
                output["writer_seconds"] = list(pool.map(write_worker, [path] * 2, [factor] * 2, range(2)))
            output["two_process_20_commits_ms_including_spawn"] = (time.perf_counter() - started) * 1000
            output["final_record_count"] = store.query()["total"]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
