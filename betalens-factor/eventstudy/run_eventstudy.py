"""
事件研究运行脚本。

参数统一从同目录的 eventstudy_params.json 读取；前端 dashboard 也读取同一份配置。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from betalens.datafeed import Datafeed
from betalens.eventstudy.eventstudy import EventStudy


PARAMS_FILE = PROJECT_ROOT / "eventstudy_params.json"
FALLBACK_PARAMS: dict[str, Any] = {
    "event_file": "1.春节假期.xlsx",
    "code": "000906.SH",
    "benchmark_code": "",
    "metric": "收盘价",
    "table_name": "daily_market",
    "mode": "flexible",
    "window_before": 20,
    "window_after": 20,
    "holding_start_offset": 0,
    "market_close_hour": 15,
    "holding_days": "1,2,3,4,5",
    "holding_months": "1,3,6,9,12",
    "save_results": False
}


def load_params() -> dict[str, Any]:
    params = dict(FALLBACK_PARAMS)
    if PARAMS_FILE.exists():
        loaded = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{PARAMS_FILE.name} 必须是 JSON object")
        params.update(loaded)
    return params


def parse_codes(value: Any) -> str | list[str]:
    if isinstance(value, list):
        codes = [str(item).strip() for item in value if str(item).strip()]
    else:
        codes = [item.strip() for item in str(value or "").replace("\n", ",").replace(";", ",").split(",") if item.strip()]
    if not codes:
        raise ValueError("参数 code 至少需要一个标的代码")
    return codes[0] if len(codes) == 1 else codes


def parse_int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value or "").replace("，", ",").split(",")
    return [int(str(item).strip()) for item in raw if str(item).strip()]


def build_holding_periods(params: dict[str, Any]) -> dict[str, list[int]] | None:
    if str(params.get("mode", "flexible")) != "fixed":
        return None
    return {
        "days": parse_int_list(params.get("holding_days", "1,2,3,4,5")),
        "months": parse_int_list(params.get("holding_months", "1,3,6,9,12")),
    }


def read_events(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"事件文件不存在: {path}")
    if path.suffix.lower() == ".csv":
        events_df = pd.read_csv(path)
    else:
        events_df = pd.read_excel(path)
    if "date" not in events_df.columns:
        raise ValueError("事件文件缺少 date 列")
    if "event" not in events_df.columns:
        events_df["event"] = 1
    events_df["date"] = pd.to_datetime(events_df["date"], errors="coerce")
    events_df = events_df.dropna(subset=["date"]).sort_values("date")
    events_df["event"] = pd.to_numeric(events_df["event"], errors="coerce").fillna(0).astype(int)
    events = events_df.set_index("date")["event"]
    events = events[events == 1]
    if events.empty:
        raise ValueError("事件文件中没有 event=1 的记录")
    return events


def main() -> int:
    params = load_params()
    event_path = PROJECT_ROOT / str(params.get("event_file") or FALLBACK_PARAMS["event_file"])
    events = read_events(event_path)

    code = parse_codes(params.get("code"))
    benchmark_code = str(params.get("benchmark_code") or "").strip() or None
    metric = str(params.get("metric") or FALLBACK_PARAMS["metric"])
    table_name = str(params.get("table_name") or FALLBACK_PARAMS["table_name"])
    mode = str(params.get("mode") or FALLBACK_PARAMS["mode"])
    window_before = int(params.get("window_before") or FALLBACK_PARAMS["window_before"])
    window_after = int(params.get("window_after") or FALLBACK_PARAMS["window_after"])
    holding_start_offset = int(params.get("holding_start_offset") or FALLBACK_PARAMS["holding_start_offset"])
    market_close_hour = int(params.get("market_close_hour") or FALLBACK_PARAMS["market_close_hour"])
    save_results = bool(params.get("save_results", False))

    print("[OK] 已读取参数:", PARAMS_FILE)
    print(f"[OK] 已读取事件序列: {int(events.sum())} 个事件")
    print("事件研究参数:")
    print(f"  - 标的代码: {code}")
    print(f"  - 基准代码: {benchmark_code or '-'}")
    print(f"  - 价格指标: {metric}")
    print(f"  - 数据表: {table_name}")
    print(f"  - 模式: {mode}")
    print(f"  - 窗口: -{window_before} / +{window_after}")

    datafeed = Datafeed(table_name)
    try:
        study = EventStudy(datafeed)
        result = study.analyze(
            events=events,
            code=code,
            benchmark_code=benchmark_code,
            window_before=window_before,
            window_after=window_after,
            metric=metric,
            mode=mode,
            holding_periods=build_holding_periods(params),
            holding_start_offset=holding_start_offset,
            market_close_hour=market_close_hour,
        )
    finally:
        datafeed.close()

    if "error" in result:
        print(f"[ERROR] 分析失败: {result['error']}")
        return 1

    daily_stats = result["daily_stats"]
    cumulative_stats = result["cumulative_stats"]
    print(f"[OK] 成功分析 {result['event_count']} 个事件")
    print("\n【每日平均收益率统计】")
    print(daily_stats.to_string())
    print("\n【累积收益率统计】")
    print(cumulative_stats.to_string())

    if save_results:
        output_file = PROJECT_ROOT / "eventstudy_results.xlsx"
        with pd.ExcelWriter(output_file) as writer:
            daily_stats.to_excel(writer, sheet_name="daily_stats")
            cumulative_stats.to_excel(writer, sheet_name="cumulative_stats")
            result["returns_matrix"].to_excel(writer, sheet_name="returns_matrix")
        print(f"\n[OK] 详细结果已保存到: {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
