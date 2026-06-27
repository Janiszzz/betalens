#%%
"""DISP 因子 Walk-Forward 滚动参数挖掘框架（多进程）

在 DISP 因子（距 52 周高点的 dispensability）+ PreTOM 择时基础上，做三段式
walk-forward 样本外参数挖掘，目标函数=夏普比率：

  TRAIN(2010-2019) → 多滚动方案 × 4维grid 滚动回测，按"单窗口夺魁次数"取候选
  TEST (2020-2022) → 候选集换不同滚动方案重测，按夺魁次数取前3名
  VALID(2023-2025) → 前3名各在整段跑一次完整回测 + Analyst 报告

全程用精确 BacktestBase（整数手/停牌/查库取价），多进程并行。
取数(收盘价宽表 + PIT 成分股)只在主进程做一次，dump pickle，子进程经 initializer 共享。

用法:
    cd .../LiqDemand/DISP
    python walkforward.py --scale smoke    # 分钟级跑通验证(默认从这里开始)
    python walkforward.py --scale medium   # ~1 小时
    python walkforward.py --scale full     # 数小时，完整 168 组
    python walkforward.py --scale full --rebuild-cache   # 强制重取数据

Windows spawn 注意：run_one/_init_worker 为模块级函数(可 pickle)；大对象走磁盘
pickle + initializer 加载到子进程全局；主流程在 __main__ 保护下。
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

_DIR = Path(__file__).resolve().parent          # LiqDemand/DISP
_CLASS_DIR = _DIR.parent                          # LiqDemand
_ROOT = _CLASS_DIR.parent                         # betalens-factor (factor_template.py 所在)
for _p in (str(_CLASS_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_OUT = _DIR / "_walkforward"
_CACHE = _OUT / "_cache"

# ============================================================
# 1. 配置区
# ============================================================
INDEX_CODE = "000906.SH"
OBJECTIVE = "sharpe"                               # 目标函数（夺魁=单窗口该指标第一）

TRAIN = ("2010-01-01", "2019-12-31")
TEST = ("2020-01-01", "2022-12-31")
VALID = ("2023-01-01", "2025-12-31")

# 取数预热：最大 window 756 交易日 ≈ 1134 自然日，再留余量
MAX_WINDOW_NATURAL_DAYS = 1200

# train / test 各自的滚动方案 (窗长_交易日, 步长_交易日)
ROLL_SCHEMES_TRAIN = [(252, 63), (252, 126), (504, 63), (504, 126)]
ROLL_SCHEMES_TEST = [(252, 21), (378, 63), (504, 126)]

PERCENTILE = (0.50, 0.75)                          # 夺魁次数入选百分位带 [P50, P75]
N_WORKERS = min(10, max(1, (os.cpu_count() or 4) - 2))

# 4 维 grid（按 scale 裁剪）
GRID_FULL = dict(
    window=[60, 120, 180, 252, 378, 504, 756],
    pretom=[(9, 4), (7, 4), (11, 4), (11, 2)],
    pretom_only=[True, False],
    n_quantiles=[5, 10, 20],
)
GRID_MEDIUM = dict(
    window=[120, 180, 252, 378, 504],
    pretom=[(9, 4), (7, 4)],
    pretom_only=[True, False],
    n_quantiles=[10, 20],
)
GRID_SMOKE = dict(
    window=[180, 252],
    pretom=[(9, 4)],
    pretom_only=[True],
    n_quantiles=[20],
)


def get_config(scale: str):
    """返回 (grid, train_schemes, test_schemes, max_windows_per_scheme)。

    smoke 档额外限制每滚动方案只取首尾 2 个窗口，分钟级跑通。
    """
    if scale == "smoke":
        return GRID_SMOKE, [(504, 252)], [(504, 252)], 2
    if scale == "medium":
        return GRID_MEDIUM, ROLL_SCHEMES_TRAIN[:2], ROLL_SCHEMES_TEST[:2], None
    return GRID_FULL, ROLL_SCHEMES_TRAIN, ROLL_SCHEMES_TEST, None


# ============================================================
# 2. 取数缓存层（主进程做一次）
# ============================================================

def build_cache(rebuild: bool = False):
    """一次性取全区间收盘价宽表 + PIT 成分股映射，dump 到 _CACHE。

    取数区间：TRAIN 起前推 MAX_WINDOW 天（预热最大 rolling）~ VALID 末。
    PIT universe：覆盖三集所有交易日（信号日精度足够，回测/分组按信号日过滤）。
    """
    _CACHE.mkdir(parents=True, exist_ok=True)
    cw_path = _CACHE / "close_wide.pkl"
    pit_path = _CACHE / "pit_universe.pkl"
    if cw_path.exists() and pit_path.exists() and not rebuild:
        print(f"[cache] 命中: {cw_path.name}, {pit_path.name}")
        return

    from factor_template import fetch_daily_wide, build_pit_universe
    from betalens.datafeed import get_absolute_trade_days

    fetch_start = (pd.Timestamp(TRAIN[0]) - pd.Timedelta(days=MAX_WINDOW_NATURAL_DAYS)).strftime("%Y-%m-%d")
    fetch_end = VALID[1]
    print(f"[cache] 取数区间(含预热): {fetch_start} ~ {fetch_end}")

    # 全部交易日 → 信号日(=每个交易日，回测时再按滚动窗口切)，用于 PIT 构建
    all_days = sorted(get_absolute_trade_days(fetch_start, fetch_end, "D", use_pmc=False))
    # PIT 只需覆盖三集内可能的信号日；预热期不调仓，故 PIT 从 TRAIN 起即可
    pit_days = [d for d in all_days if d >= pd.Timestamp(TRAIN[0]).date()]
    t0 = time.time()
    pit = build_pit_universe(pit_days, INDEX_CODE)
    universe = sorted({c for codes in pit.values() for c in codes})
    print(f"[cache] PIT universe: {len(universe)} 只, {len(pit)} 信号日, {time.time()-t0:.1f}s")

    t1 = time.time()
    cw = fetch_daily_wide("收盘价(元)", universe=universe,
                          start_date=fetch_start, end_date=fetch_end)
    print(f"[cache] 收盘价宽表: {cw.shape}, {time.time()-t1:.1f}s")

    with open(cw_path, "wb") as f:
        pickle.dump(cw, f)
    with open(pit_path, "wb") as f:
        pickle.dump(pit, f)
    print(f"[cache] 已落盘: {_CACHE}")


# ============================================================
# 3. 子进程：initializer 加载缓存到全局 + run_one 单次回测
# ============================================================
_CW = None        # 全区间收盘价宽表（子进程全局）
_PIT = None       # PIT 成分股映射


def _init_worker(cw_path: str, pit_path: str):
    """子进程初始化：把大对象加载到模块全局，避免每任务重复 IO/传参。"""
    global _CW, _PIT
    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)
    with open(cw_path, "rb") as f:
        _CW = pickle.load(f)
    with open(pit_path, "rb") as f:
        _PIT = pickle.load(f)


def _metrics_from_nav(nav: pd.Series) -> dict:
    """从净值序列算关键绩效指标（挖掘阶段够用，不依赖 Analyst）。"""
    r = nav.pct_change().dropna()
    if len(r) < 2 or r.std() == 0:
        return dict(sharpe=0.0, ann_ret=0.0, ann_vol=0.0, mdd=0.0, calmar=0.0, n_days=len(r))
    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = (1 + r).cumprod().cummax()
    mdd = float((1 - (1 + r).cumprod() / cummax).max())
    calmar = ann_ret / mdd if mdd > 0 else 0.0
    return dict(sharpe=round(float(sharpe), 4), ann_ret=round(float(ann_ret), 4),
                ann_vol=round(float(ann_vol), 4), mdd=round(mdd, 4),
                calmar=round(float(calmar), 4), n_days=len(r))


def _vector_backtest(weights: pd.DataFrame) -> pd.Series:
    """轻量向量化回测：权重持有到下一调仓日，用缓存收盘价算净值（纯内存、无成本/整数手）。

    作为精确 BacktestBase 的快速代理 —— 口径对齐其「持有到下一调仓日」逻辑：
    第 i 个调仓日权重 w_i 在 [日期_i, 日期_{i+1}) 区间产生组合日收益 = Σ_j w_ij · r_j。
    差异仅在交易成本、整数手撮合、成交时点（向量化用次日收盘）。返回 nav(净值序列)。
    """
    cw = _CW
    codes = [c for c in weights.columns if c != "cash"]
    reb_days = pd.DatetimeIndex(sorted({ts.normalize() for ts in weights.index}))
    # 收盘价限制到调仓首日~末日+持有缓冲，算个股日收益
    px = cw.loc[(cw.index >= reb_days[0]) & (cw.index <= reb_days[-1] + pd.Timedelta(days=40)), codes]
    ret = px.pct_change()
    # 把每个交易日映射到"最近一次已生效的调仓日权重"（前向填充权重）
    w = weights.copy(); w.index = w.index.normalize()
    w = w[codes].reindex(px.index, method="ffill").shift(1)   # shift(1): 次日起持有，防前视
    port_ret = (w * ret).sum(axis=1).dropna()
    if port_ret.empty:
        return pd.Series([1.0])
    return (1 + port_ret).cumprod()


def run_one(task: dict) -> dict:
    """单次回测：切片缓存→算 DISP→分组→权重→PreTOM 掩码→回测→指标。

    task 含: window, pretom_lo, pretom_hi, pretom_only, n_q, win_start, win_end,
            scheme(滚动方案标识), phase(train/test), engine('exact'|'vector')。
    engine='exact' 用精确 BacktestBase(整数手/停牌/查库取价，默认)；
    engine='vector' 用 _vector_backtest(纯内存，秒级，作快速代理)。
    返回扁平 dict（含全部参数 + 指标 + 窗口期标识）；失败记 error 字段。
    """
    from factor_template_liqdemand import get_pretom_dates, clean_inf
    from factor_template import wide_to_prequery, filter_long_by_pit_universe
    from betalens.datafeed import get_absolute_trade_days
    from betalens.factor.factor import single_characteristic, get_single_factor_weight

    out = {k: task[k] for k in
           ("window", "pretom_lo", "pretom_hi", "pretom_only", "n_q",
            "win_start", "win_end", "scheme", "phase")}
    out["gid"] = task["gid"]
    engine = task.get("engine", "exact")
    try:
        win = task["window"]
        start, end = task["win_start"], task["win_end"]
        # 取数预热：窗口起前推 window*1.5 天，但调仓/回测仅 [start,end]
        fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=int(win * 1.5) + 60)).strftime("%Y-%m-%d")

        reb = get_absolute_trade_days(start, end, "D", use_pmc=False)
        alld = sorted(get_absolute_trade_days(fetch_start, end, "D", use_pmc=False))
        idx = {d: i for i, d in enumerate(alld)}
        sig = [alld[idx[r] - 1] for r in reb if idx.get(r, 0) and idx[r] > 0]
        if not sig:
            out["error"] = "no signal dates"; return out

        # 从全局缓存切片（不查库）
        cw = _CW.loc[(_CW.index >= pd.Timestamp(fetch_start)) & (_CW.index <= pd.Timestamp(end) + pd.Timedelta(days=1))]
        mp = max(20, win // 2)
        fac = clean_inf(-(cw / cw.rolling(win, min_periods=mp).max()))

        pq = wide_to_prequery(fac, "DISP", sig)
        pq = filter_long_by_pit_universe(pq, _PIT)
        if pq.empty:
            out["error"] = "empty prequery"; return out

        lab = single_characteristic(pq, "DISP", {"DISP": task["n_q"]})
        # direction=negative → 做多低分组[0]（可处置=高因子值=做空，故多头取最小分组）
        w = get_single_factor_weight(lab, {"factor_key": "DISP", "mode": "freeplay",
                                           "long": [0], "short": []})
        w.index = w.index + pd.Timedelta(minutes=10)

        if task["pretom_only"]:
            pre = get_pretom_dates(start, end, lo=task["pretom_lo"], hi=task["pretom_hi"])
            keep = np.array([t.date() in pre for t in w.index])
            w = w.loc[keep]
            if w.empty:
                out["error"] = "empty after pretom mask"; return out

        if engine == "vector":
            nav = _vector_backtest(w)
        else:
            from betalens.backtest import BacktestBase
            bt = BacktestBase(w, metric="收盘价(元)", symbol="DISP", amount=1e8,
                              time_tolerance=24 * 11, verbose=False)
            nav = bt.nav
        out.update(_metrics_from_nav(nav))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# ============================================================
# 4. 编排层
# ============================================================

def gen_windows(start: str, end: str, win_len: int, step: int, cap=None) -> list[tuple[str, str]]:
    """在 [start,end] 内按窗长(交易日)/步长(交易日)滚动切窗口，返回 [(s,e),...]。"""
    from betalens.datafeed import get_absolute_trade_days
    days = sorted(get_absolute_trade_days(start, end, "D", use_pmc=False))
    wins = []
    i = 0
    while i + win_len <= len(days):
        s = days[i]; e = days[i + win_len - 1]
        wins.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
        i += step
    if cap and len(wins) > cap:                    # smoke: 取首尾各一段
        wins = [wins[0], wins[-1]] if cap == 2 else wins[:cap]
    return wins


def grid_combos(grid: dict) -> list[dict]:
    """4 维 grid → 参数组合列表，每组带稳定 gid（参数串）。"""
    combos = []
    for win, pre, po, nq in itertools.product(
            grid["window"], grid["pretom"], grid["pretom_only"], grid["n_quantiles"]):
        gid = f"w{win}_p{pre[0]}-{pre[1]}_{'PT' if po else 'DLY'}_q{nq}"
        combos.append(dict(window=win, pretom_lo=pre[0], pretom_hi=pre[1],
                           pretom_only=po, n_q=nq, gid=gid))
    return combos


def build_tasks(phase: str, span: tuple, schemes: list, grid: dict, cap, engine: str = "exact") -> list[dict]:
    """笛卡尔积：滚动方案 × 各方案窗口 × grid组合 → 任务列表。"""
    combos = grid_combos(grid)
    tasks = []
    for (win_len, step) in schemes:
        wins = gen_windows(span[0], span[1], win_len, step, cap=cap)
        scheme = f"{win_len}/{step}"
        for (ws, we) in wins:
            for c in combos:
                t = dict(c); t.update(phase=phase, scheme=scheme, win_start=ws,
                                      win_end=we, engine=engine)
                tasks.append(t)
    return tasks


def run_parallel(tasks: list[dict]) -> pd.DataFrame:
    """多进程执行 run_one，返回结果长表。"""
    cw_path = str(_CACHE / "close_wide.pkl")
    pit_path = str(_CACHE / "pit_universe.pkl")
    rows, done, total = [], 0, len(tasks)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init_worker,
                             initargs=(cw_path, pit_path)) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 50 == 0 or done == total:
                el = time.time() - t0
                print(f"  进度 {done}/{total}  用时 {el:.0f}s  预计剩余 {el/done*(total-done):.0f}s")
    return pd.DataFrame(rows)


def tally_champions(df: pd.DataFrame) -> pd.DataFrame:
    """每个滚动窗口按 OBJECTIVE 取夺魁 grid，统计各 grid 夺魁次数 + 夺魁窗口期。"""
    ok = df[df.get("error").isna()] if "error" in df.columns else df
    ok = ok[ok["sharpe"].notna()]
    champs = []
    # 每个 (scheme, win_start, win_end) 是一个独立窗口
    for (sch, ws, we), g in ok.groupby(["scheme", "win_start", "win_end"]):
        best = g.loc[g[OBJECTIVE].idxmax()]
        champs.append(dict(scheme=sch, win_start=ws, win_end=we,
                           gid=best["gid"], sharpe=best[OBJECTIVE]))
    champ_df = pd.DataFrame(champs)
    if champ_df.empty:
        return champ_df
    tally = (champ_df.groupby("gid")
             .agg(wins_count=("gid", "size"),
                  avg_champ_sharpe=("sharpe", "mean"),
                  champ_windows=("win_start", lambda s: list(s)))
             .reset_index()
             .sort_values(["wins_count", "avg_champ_sharpe"], ascending=False)
             .reset_index(drop=True))
    return tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["smoke", "medium", "full"], default="medium")
    ap.add_argument("--engine", choices=["exact", "vector"], default="exact",
                    help="exact=精确BacktestBase(默认,口径准); vector=轻量向量化(秒级,作快速代理)")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    grid, train_sch, test_sch, cap = get_config(args.scale)
    n_combos = len(grid_combos(grid))
    print(f"=== Walk-Forward 参数挖掘 (scale={args.scale}, engine={args.engine}, workers={N_WORKERS}) ===")
    print(f"grid 组合数={n_combos}  train方案={train_sch}  test方案={test_sch}\n")

    # 取数缓存（主进程一次）
    build_cache(rebuild=args.rebuild_cache)

    # ---------- TRAIN ----------
    print(f"\n[TRAIN] {TRAIN[0]}~{TRAIN[1]}")
    train_tasks = build_tasks("train", TRAIN, train_sch, grid, cap, engine=args.engine)
    print(f"  任务数={len(train_tasks)} (≈{len(train_tasks)//max(n_combos,1)}窗口 × {n_combos}grid)")
    train_df = run_parallel(train_tasks)
    train_df.to_csv(_OUT / "train_results.csv", index=False, encoding="utf-8-sig")
    train_tally = tally_champions(train_df)
    train_tally.to_csv(_OUT / "train_champions.csv", index=False, encoding="utf-8-sig")

    if train_tally.empty:
        print("  [TRAIN] 无有效结果，检查数据/参数"); return
    # 入选: 夺魁次数落在 [P50, P75] 百分位带（主门槛 P50）
    counts = train_tally["wins_count"].values
    p50, p75 = np.percentile(counts, [PERCENTILE[0] * 100, PERCENTILE[1] * 100])
    cand = train_tally[train_tally["wins_count"] >= p50]
    cand.to_csv(_OUT / "train_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"  夺魁次数 P50={p50:.1f} P75={p75:.1f}  候选 grid 数={len(cand)} (wins_count>=P50)")
    print(train_tally.head(10).to_string(index=False))

    cand_gids = set(cand["gid"])
    cand_combos = [c for c in grid_combos(grid) if c["gid"] in cand_gids]

    # ---------- TEST ----------
    print(f"\n[TEST] {TEST[0]}~{TEST[1]}  (候选 {len(cand_combos)} grid 换滚动方案重测)")
    test_grid = {  # 仅候选 grid，用其参数枚举构造任务
        "window": sorted({c["window"] for c in cand_combos}),
        "pretom": sorted({(c["pretom_lo"], c["pretom_hi"]) for c in cand_combos}),
        "pretom_only": sorted({c["pretom_only"] for c in cand_combos}),
        "n_quantiles": sorted({c["n_q"] for c in cand_combos}),
    }
    # 用 build_tasks 全枚举后过滤到候选 gid（保证只测候选）
    test_tasks_all = build_tasks("test", TEST, test_sch, test_grid, cap, engine=args.engine)
    test_tasks = [t for t in test_tasks_all if t["gid"] in cand_gids]
    print(f"  任务数={len(test_tasks)}")
    test_df = run_parallel(test_tasks)
    test_df.to_csv(_OUT / "test_results.csv", index=False, encoding="utf-8-sig")
    test_tally = tally_champions(test_df)
    test_tally.to_csv(_OUT / "test_champions.csv", index=False, encoding="utf-8-sig")
    if test_tally.empty:
        print("  [TEST] 无有效结果"); return
    top3 = test_tally.head(3)
    print(f"  TEST 前3名:")
    print(top3.to_string(index=False))

    # ---------- VALID ----------
    print(f"\n[VALID] {VALID[0]}~{VALID[1]}  前3名各整段跑一次完整回测")
    gid2combo = {c["gid"]: c for c in grid_combos(grid)}
    valid_rows = []
    for rank, gid in enumerate(top3["gid"].tolist(), 1):
        c = gid2combo[gid]
        task = dict(c, phase="valid", scheme="full",
                    win_start=VALID[0], win_end=VALID[1], engine=args.engine)
        _init_worker(str(_CACHE / "close_wide.pkl"), str(_CACHE / "pit_universe.pkl"))
        res = run_one(task)
        res["rank"] = rank
        valid_rows.append(res)
        print(f"  #{rank} {gid}: sharpe={res.get('sharpe')} ann={res.get('ann_ret')} "
              f"mdd={res.get('mdd')} err={res.get('error')}")
        # 前3名再用 Analyst 出完整报告（含 html）
        _gen_valid_report(c, rank)
    pd.DataFrame(valid_rows).to_csv(_OUT / "valid_results.csv", index=False, encoding="utf-8-sig")

    best = valid_rows[0]
    print(f"\n=== 最优参数(test前1) gid={best['gid']}  valid夏普={best.get('sharpe')} ===")
    print(f"产物目录: {_OUT}")


def _gen_valid_report(combo: dict, rank: int):
    """valid 阶段对单个 grid 用完整 LiqDemandPipeline 出 Analyst 报告(html/xlsx)。"""
    import dataclasses
    from factor_DISP import spec as base_spec
    from factor_template_liqdemand import LiqDemandPipeline, clean_inf

    win = combo["window"]; mp = max(20, win // 2)

    def compute_disp(close_wide, window=win):
        ratio = close_wide / close_wide.rolling(window, min_periods=mp).max()
        return clean_inf(-ratio)

    spec = dataclasses.replace(base_spec, name=f"DISP_valid{rank}",
                               compute=compute_disp, compute_kwargs={"window": win})
    try:
        LiqDemandPipeline(spec).run(
            VALID[0], VALID[1],
            warmup_days=int(win * 1.5) + 60,
            pretom_only=combo["pretom_only"],
            pretom_lo=combo["pretom_lo"], pretom_hi=combo["pretom_hi"],
            n_quantiles=combo["n_q"], output_dir=str(_OUT),
            include_profiling=False, dump_excel=False, verbose=False)
    except Exception as e:
        print(f"    [valid报告] #{rank} 失败: {e}")


if __name__ == "__main__":
    main()
