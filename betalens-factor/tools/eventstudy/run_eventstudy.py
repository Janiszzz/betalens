"""Run event study from a single YAML parameter file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FACTOR_ROOT = PROJECT_ROOT.parents[1]
REPO_ROOT = FACTOR_ROOT.parent
for _path in (REPO_ROOT,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from betalens.datafeed import Datafeed  # noqa: E402
from betalens.eventstudy.eventstudy import EventStudy  # noqa: E402
from betalens.factor.config import load_yaml_config, resolve_path, section  # noqa: E402


PARAMS_FILE = PROJECT_ROOT / "eventstudy.yaml"


def load_params(config_path: str | Path = PARAMS_FILE) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    loaded = load_yaml_config(path, required_sections=("eventstudy",))
    return section(loaded, "eventstudy", context=str(path)), path


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
    if str(params["mode"]) != "fixed":
        return None
    return {
        "days": parse_int_list(params["holding_days"]),
        "months": parse_int_list(params["holding_months"]),
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


def _sheet_name(prefix: str, code: str, used: set[str]) -> str:
    """Create a unique Excel-safe sheet name for per-code comparison output."""
    cleaned = "".join("_" if char in '[]:*?/\\\\' else char for char in code)
    base = f"{prefix}_{cleaned}"[:31]
    candidate = base
    suffix = 1
    while candidate in used:
        ending = f"_{suffix}"
        candidate = f"{base[:31 - len(ending)]}{ending}"
        suffix += 1
    used.add(candidate)
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run event study from YAML.")
    parser.add_argument("--config", default=str(PARAMS_FILE), help="YAML parameter file")
    args = parser.parse_args()
    params, config_path = load_params(args.config)
    event_path = resolve_path(str(params["event_file"]), PROJECT_ROOT)
    events = read_events(event_path)

    code = parse_codes(params["code"])
    benchmark_code = str(params["benchmark_code"] or "").strip() or None
    metric = str(params["metric"])
    table_name = str(params["table_name"])
    mode = str(params["mode"])
    multi_asset_mode = str(params.get("multi_asset_mode", "aggregate"))
    window_before = int(params["window_before"])
    window_after = int(params["window_after"])
    holding_start_offset = int(params["holding_start_offset"])
    market_close_hour = int(params["market_close_hour"])
    save_results = bool(params["save_results"])

    print("[OK] 已读取参数:", config_path)
    print(f"[OK] 已读取事件序列: {int(events.sum())} 个事件")
    print("事件研究参数:")
    print(f"  - 标的代码: {code}")
    print(f"  - 基准代码: {benchmark_code or '-'}")
    print(f"  - 价格指标: {metric}")
    print(f"  - 数据表: {table_name}")
    print(f"  - 模式: {mode}")
    print(f"  - 多标的处理: {multi_asset_mode}")
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
            multi_asset_mode=multi_asset_mode,
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
        output_file = resolve_path(params["output"], config_path.parent)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_file) as writer:
            daily_stats.to_excel(writer, sheet_name="daily_stats")
            cumulative_stats.to_excel(writer, sheet_name="cumulative_stats")
            result["returns_matrix"].to_excel(writer, sheet_name="returns_matrix")
            comparison = result.get("comparison")
            if comparison:
                summaries = []
                used_sheets = {"daily_stats", "cumulative_stats", "returns_matrix"}
                for code, item in comparison["by_code"].items():
                    daily = item["daily_stats"]
                    cumulative = item["cumulative_stats"]
                    summaries.append(
                        {
                            "code": code,
                            "event_count": item["event_count"],
                            "coverage": item["coverage"],
                            "day0_mean": daily.loc[0, "mean"] if 0 in daily.index else None,
                            "final_mean": cumulative.iloc[-1]["mean"] if not cumulative.empty else None,
                        }
                    )
                    daily.to_excel(writer, sheet_name=_sheet_name("daily", code, used_sheets))
                    cumulative.to_excel(writer, sheet_name=_sheet_name("cum", code, used_sheets))
                pd.DataFrame(summaries).to_excel(writer, sheet_name="comparison_summary", index=False)
                pd.DataFrame(comparison["events"]).to_excel(writer, sheet_name="comparison_events", index=False)
        print(f"\n[OK] 详细结果已保存到: {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
