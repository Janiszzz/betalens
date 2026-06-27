#%%
"""DISP 参数挖掘 —— 扫描 rolling 窗口 window，目标函数=夏普比率

用法:
    cd .../LiqDemand/DISP && python sweep_window.py
    # 或自定义窗口网格：  python sweep_window.py 60 120 180 252 378 504

要点:
    - 复用 factor_DISP.spec，仅替换 compute（min_periods 随 window 自适应，
      避免小窗口被写死的 min_periods=120 卡成全 NaN）与 compute_kwargs。
    - warmup_days 随 window 放大（取数需覆盖整个 rolling 窗口 + 余量）。
    - 关掉 profiling/dump 提速；逐窗口跑 PreTOM 择时版，记录夏普等指标。
    - 取数耗时主要在 query_time_range；每窗口独立取数（窗口不同→预热区间不同）。
"""
import sys
import io
import json
import logging
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

logging.getLogger("IndexUniverseQuery").setLevel(logging.WARNING)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
from factor_DISP import spec as base_spec          # noqa: E402  复用因子声明
from factor_template_liqdemand import LiqDemandPipeline, clean_inf  # noqa: E402

START, END = "2024-01-01", "2025-12-31"
OBJECTIVE = "夏普比率"                                # 目标函数指标名


def make_compute(window: int):
    """窗口自适应的 dispensability 算子：min_periods 取 window 的一半（下限 20）。"""
    mp = max(20, window // 2)

    def compute_disp(close_wide, window=window):
        ratio = close_wide / close_wide.rolling(window, min_periods=mp).max()
        return clean_inf(-ratio)

    return compute_disp


def eval_window(window: int) -> dict:
    """单个 window 跑一次 PreTOM 择时回测，返回关键指标。"""
    spec = dataclasses.replace(
        base_spec,
        compute=make_compute(window),
        compute_kwargs={"window": window},
    )
    # 预热天数覆盖整个 rolling 窗口：window 交易日 ≈ window*1.5 自然日，再加 60 天余量
    warmup = int(window * 1.5) + 60
    r = LiqDemandPipeline(spec).run(
        START, END,
        warmup_days=warmup, pretom_only=True,
        n_quantiles=20, output_dir=str(_DIR),
        include_profiling=False, dump_excel=False, verbose=False,
    )
    s = r.analyst.report(to_excel=None, to_html=None)
    pick = ["年化收益", "年化波动率", "夏普比率", "最大回撤", "卡玛比率", "单边换手率(年化)"]
    out = {"window": window, "warmup_days": warmup}
    out.update({k: (round(float(s[k]), 4) if k in s else None) for k in pick})
    return out


def main():
    grid = [int(a) for a in sys.argv[1:]] or [int(a) for a in range(20, 505, 5)] 
    print(f"扫描 window 网格: {grid}  目标={OBJECTIVE}  区间={START}~{END}\n")

    rows = []
    for w in grid:
        try:
            res = eval_window(w)
        except Exception as e:                       # 单点失败不中断整个扫描
            print(f"  window={w} 失败: {e}")
            continue
        rows.append(res)
        print(f"  window={w:>4}  夏普={res['夏普比率']}  年化={res['年化收益']}  "
              f"回撤={res['最大回撤']}  换手={res['单边换手率(年化)']}")

    if not rows:
        print("无有效结果"); return

    df = pd.DataFrame(rows).sort_values(OBJECTIVE, ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    print("\n=== 按夏普降序 ===")
    print(df.to_string(index=False))
    print(f"\n最优 window = {int(best['window'])}  ({OBJECTIVE}={best[OBJECTIVE]})")

    out_path = _DIR / "sweep_window_result.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"结果已存: {out_path}")


if __name__ == "__main__":
    main()
