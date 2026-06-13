"""申万行业分类长表 → 宽表 / 变更表 / 完整性报告。

输入:
    SwClass/StockClassifyUse_stock_20260601.xls   (4 列长表)
    SwClass/2014to2021.xlsx                        (行业代码↔名称对照, 双 sheet)

输出: SwClass/output/
    intervals.parquet            派生区间长表
    industry_dict.parquet        code6 ↔ 名称 union 字典
    wide_l1_code.parquet         全交易日 × 股票, 值=2 位 L1 代码
    wide_l1_name.parquet         同上, 值=L1 中文名
    wide_l2_code.parquet         4 位 L2 代码
    wide_l3_code.parquet         6 位 L3 代码
    wide_sparse_l3_code.parquet  仅变更日稀疏表
    changes.parquet / changes.csv 变更明细
    validation_report.csv        完整性检查
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LONG_XLS = ROOT / "StockClassifyUse_stock_20260601.xls"
DICT_XLSX = ROOT / "2014to2021.xlsx"
OUT_DIR = ROOT / "output"

START_DATE = pd.Timestamp("2005-01-04")
END_DATE = pd.Timestamp("2026-05-27")

VER_2014 = pd.Timestamp("2014-02-21")
VER_2021 = pd.Timestamp("2021-07-30")
VER_WIN = pd.Timedelta(days=7)


def load_long_table(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    df = pd.DataFrame({
        "code": raw["股票代码"].astype("Int64").astype(str).str.zfill(6),
        "enter_date": pd.to_datetime(raw["计入日期"]),
        "ind_code": raw["行业代码"].astype("Int64").astype(str).str.zfill(6),
        "update_ts": pd.to_datetime(raw["更新日期"]),
    })
    df = df.sort_values(["code", "enter_date"]).reset_index(drop=True)
    return df


def _extract_dict_from_block(df: pd.DataFrame, l1_col: str, l2_col: str,
                             l3_col: str, code_col: str
                             ) -> tuple[dict, dict, dict]:
    """从一组 (一级名, 二级名, 三级名, 代码) 列里抽出三级字典。

    每行恰好有 1 个名称非 NaN, 由代码末尾零数判断层级。
    """
    l1, l2, l3 = {}, {}, {}
    for _, row in df.iterrows():
        raw_code = row[code_col]
        if pd.isna(raw_code):
            continue
        try:
            code6 = str(int(float(raw_code))).zfill(6)
        except (ValueError, TypeError):
            continue
        n1 = row[l1_col] if not pd.isna(row[l1_col]) else None
        n2 = row[l2_col] if not pd.isna(row[l2_col]) else None
        n3 = row[l3_col] if not pd.isna(row[l3_col]) else None
        if n1:
            l1[code6[:2]] = str(n1).strip()
        if n2:
            l2[code6[:4]] = str(n2).strip()
        if n3:
            l3[code6] = str(n3).strip()
    return l1, l2, l3


def load_industry_dict(path: Path):
    """返回 (dict_2021, dict_2014, dict_union, l1_names, l2_names, l3_names)。

    前三个字典: {code6: {'l1_code','l1_name','l2_code','l2_name','l3_name'}}
    后三个: 跨版本合并的 L1/L2/L3 名称扁平字典 (2021 优先, 2014 补缺)。
    """
    xl = pd.ExcelFile(path)
    sn1, sn2 = xl.sheet_names[:2]

    df1 = pd.read_excel(path, sheet_name=sn1, header=1)
    l1_2014_a, l2_2014_a, l3_2014_a = _extract_dict_from_block(
        df1, "一级行业", "二级行业", "三级行业", "行业代码")
    l1_2021_a, l2_2021_a, l3_2021_a = _extract_dict_from_block(
        df1, "一级行业.1", "二级行业.1", "三级行业.1", "行业代码.1")

    df2 = pd.read_excel(path, sheet_name=sn2, header=1)
    l1_2014_b, l2_2014_b, l3_2014_b = _extract_dict_from_block(
        df2, "旧版一级行业", "旧版二级行业", "旧版三级行业", "行业代码")
    l1_2021_b, l2_2021_b, l3_2021_b = _extract_dict_from_block(
        df2, "新版一级行业", "新版二级行业", "新版三级行业", "行业代码.1")

    l1_2014 = {**l1_2014_b, **l1_2014_a}
    l2_2014 = {**l2_2014_b, **l2_2014_a}
    l3_2014 = {**l3_2014_b, **l3_2014_a}
    l1_2021 = {**l1_2021_b, **l1_2021_a}
    l2_2021 = {**l2_2021_b, **l2_2021_a}
    l3_2021 = {**l3_2021_b, **l3_2021_a}

    def _build(l1, l2, l3) -> dict:
        codes = set(l3.keys())
        for k in l2:
            codes.add(k + "00")
        for k in l1:
            codes.add(k + "0000")
        out = {}
        for c in codes:
            out[c] = {
                "l1_code": c[:2],
                "l1_name": l1.get(c[:2]),
                "l2_code": c[:4],
                "l2_name": l2.get(c[:4]),
                "l3_name": l3.get(c),
            }
        return out

    d2014 = _build(l1_2014, l2_2014, l3_2014)
    d2021 = _build(l1_2021, l2_2021, l3_2021)
    union = {**d2014, **d2021}
    for c, v in d2014.items():
        if c in union:
            for k, val in v.items():
                if union[c].get(k) is None and val is not None:
                    union[c][k] = val

    l1_names = {**l1_2014, **l1_2021}
    l2_names = {**l2_2014, **l2_2021}
    l3_names = {**l3_2014, **l3_2021}
    return d2021, d2014, union, l1_names, l2_names, l3_names


def derive_intervals(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["exit_date"] = df.groupby("code")["enter_date"].shift(-1)
    df["l1_code"] = df["ind_code"].str[:2]
    df["l2_code"] = df["ind_code"].str[:4]
    df["l3_code"] = df["ind_code"]
    return df[["code", "enter_date", "exit_date",
               "l1_code", "l2_code", "l3_code", "ind_code", "update_ts"]]


def validate(intervals: pd.DataFrame, union: dict) -> pd.DataFrame:
    issues = []

    # 1. 单调递增
    for code, grp in intervals.groupby("code"):
        if not grp["enter_date"].is_monotonic_increasing:
            issues.append((code, "non_monotonic_enter",
                           f"first={grp['enter_date'].min()}"))

    # 2. 重复 (code, enter_date)
    dup = intervals[intervals.duplicated(["code", "enter_date"], keep=False)]
    for code, grp in dup.groupby("code"):
        issues.append((code, "duplicate_enter_date",
                       f"{len(grp)} rows on same date"))

    # 3. 多个开口区间
    open_cnt = intervals[intervals["exit_date"].isna()].groupby("code").size()
    for code, n in open_cnt.items():
        if n != 1:
            issues.append((code, "multiple_open_intervals", f"count={n}"))

    # 4. 行业代码不在字典
    bad = intervals[~intervals["ind_code"].isin(union.keys())]
    legacy = bad[bad["enter_date"] < VER_2014]
    real_bad = bad[bad["enter_date"] >= VER_2014]
    if len(legacy):
        issues.append(("__INFO__", "legacy_pre_2014_unknown_code",
                       f"{len(legacy)} rows, "
                       f"{legacy['ind_code'].nunique()} codes — "
                       f"申万 2014 版前的废弃分类, 字典不覆盖"))
    for _, r in real_bad.iterrows():
        issues.append((r["code"], "unknown_industry_code",
                       f"ind={r['ind_code']} at {r['enter_date'].date()}"))

    # 5. 股票代码格式
    bad_code = intervals[~intervals["code"].str.fullmatch(r"\d{6}")]
    for code in bad_code["code"].unique():
        issues.append((code, "bad_code_format", "not 6-digit"))

    # info: 字典覆盖率 + 返回现象 + 单记录股票
    cov = intervals["ind_code"].isin(union.keys()).mean()
    issues.append(("__INFO__", "dict_coverage", f"{cov:.4%}"))
    rec_cnt = intervals.groupby("code").size()
    issues.append(("__INFO__", "single_record_stocks",
                   f"{(rec_cnt == 1).sum()} stocks"))
    ret_codes = intervals.groupby("code")["ind_code"].apply(
        lambda s: s.duplicated().any())
    issues.append(("__INFO__", "stocks_with_industry_return",
                   f"{ret_codes.sum()} stocks"))
    issues.append(("__INFO__", "total_unique_stocks",
                   f"{intervals['code'].nunique()}"))
    issues.append(("__INFO__", "total_intervals", f"{len(intervals)}"))

    return pd.DataFrame(issues, columns=["code", "issue_type", "detail"])


def get_trade_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    try:
        from betalens.datafeed import get_absolute_trade_days
        days = get_absolute_trade_days(start.strftime("%Y-%m-%d"),
                                       end.strftime("%Y-%m-%d"),
                                       "D", use_pmc=False)
        idx = pd.DatetimeIndex(pd.to_datetime(list(days))).sort_values()
        if len(idx) > 100:
            return idx
    except Exception as e:
        print(f"  [warn] betalens 交易日历不可用 ({e}), 退化为 bdate_range")
    return pd.bdate_range(start, end)


def build_wide(intervals: pd.DataFrame, trade_days: pd.DatetimeIndex,
               value_col: str) -> pd.DataFrame:
    """全交易日 × 股票 宽表; 区间 [enter, exit) 内填 value_col。"""
    codes = sorted(intervals["code"].unique())
    code_idx = {c: i for i, c in enumerate(codes)}
    n_rows, n_cols = len(trade_days), len(codes)

    arr = np.full((n_rows, n_cols), -1, dtype=np.int32)
    vals = pd.Series(intervals[value_col]).dropna().unique()
    cats = pd.Index(sorted(map(str, vals)))
    cat_idx = {v: i for i, v in enumerate(cats)}

    enters = intervals["enter_date"].values
    exits = intervals["exit_date"].values
    td_vals = trade_days.values

    for code, enter, exit_, val in zip(intervals["code"].values,
                                       enters, exits,
                                       intervals[value_col].values):
        if pd.isna(val):
            continue
        col = code_idx[code]
        r0 = np.searchsorted(td_vals, enter, side="left")
        r1 = (np.searchsorted(td_vals, exit_, side="left")
              if pd.notna(exit_) else n_rows)
        if r1 > r0:
            arr[r0:r1, col] = cat_idx[str(val)]

    df = pd.DataFrame(
        {c: pd.Categorical.from_codes(arr[:, code_idx[c]], cats)
         for c in codes},
        index=trade_days,
    )
    df.index.name = "datetime"
    return df


def build_sparse_wide(intervals: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = intervals.pivot_table(index="enter_date", columns="code",
                               values=value_col, aggfunc="first")
    df = df.sort_index()
    df.index.name = "datetime"
    return df


def _classify_change(change_date: pd.Timestamp,
                     prev_l3: str | None,
                     new_l3: str,
                     prev_l1: str | None,
                     new_l1: str,
                     history: set[str]) -> str:
    if prev_l3 is None:
        return "initial"
    if abs(change_date - VER_2021) <= VER_WIN:
        return "version_2021_switch"
    if abs(change_date - VER_2014) <= VER_WIN:
        return "version_2014_switch"
    if new_l3 in history:
        return "return"
    if new_l1 == prev_l1:
        return "same_l1_reclass"
    return "cross_l1_reclass"


def _version_tag(d: pd.Timestamp) -> str:
    return "2021v" if d >= VER_2021 else ("2014v" if d >= VER_2014 else "pre2014")


def build_changes(intervals: pd.DataFrame, union: dict,
                  l1_names: dict, l3_names: dict) -> pd.DataFrame:
    rows = []
    for code, grp in intervals.groupby("code"):
        grp = grp.sort_values("enter_date").reset_index(drop=True)
        history: set[str] = set()
        prev_l3 = prev_l1 = None
        prev_l3_name = prev_l1_name = None
        for _, r in grp.iterrows():
            new_l3 = r["l3_code"]
            new_l1 = r["l1_code"]
            info = union.get(new_l3, {})
            new_l3_name = info.get("l3_name") or l3_names.get(new_l3)
            new_l1_name = info.get("l1_name") or l1_names.get(new_l1)
            ctype = _classify_change(r["enter_date"], prev_l3, new_l3,
                                     prev_l1, new_l1, history)
            rows.append({
                "code": code,
                "change_date": r["enter_date"],
                "prev_l3_code": prev_l3, "prev_l3_name": prev_l3_name,
                "prev_l1_code": prev_l1, "prev_l1_name": prev_l1_name,
                "new_l3_code": new_l3, "new_l3_name": new_l3_name,
                "new_l1_code": new_l1, "new_l1_name": new_l1_name,
                "change_type": ctype,
                "version_tag": _version_tag(r["enter_date"]),
                "update_ts": r["update_ts"],
            })
            history.add(new_l3)
            prev_l3, prev_l3_name = new_l3, new_l3_name
            prev_l1, prev_l1_name = new_l1, new_l1_name
    return pd.DataFrame(rows)


def industry_dict_to_df(d2021: dict, d2014: dict, union: dict) -> pd.DataFrame:
    rows = []
    for code, info in union.items():
        if code in d2021 and code in d2014:
            src = "both"
        elif code in d2021:
            src = "2021v"
        else:
            src = "2014v"
        rows.append({"ind_code": code, **info, "source_version": src})
    return pd.DataFrame(rows).sort_values("ind_code").reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/7] 读长表...")
    long_df = load_long_table(LONG_XLS)
    print(f"  rows={len(long_df)}, codes={long_df['code'].nunique()}")

    print("[2/7] 读行业字典...")
    d2021, d2014, union, l1_names, l2_names, l3_names = load_industry_dict(
        DICT_XLSX)
    print(f"  2021={len(d2021)}, 2014={len(d2014)}, union={len(union)}, "
          f"L1={len(l1_names)}, L2={len(l2_names)}, L3={len(l3_names)}")

    print("[3/7] 派生区间...")
    intervals = derive_intervals(long_df)
    intervals.to_parquet(OUT_DIR / "intervals.parquet", index=False)

    print("[4/7] 完整性检查...")
    report = validate(intervals, union)
    report.to_csv(OUT_DIR / "validation_report.csv",
                  index=False, encoding="utf-8-sig")
    err = report[~report["code"].eq("__INFO__")]
    print(f"  issues={len(err)}, info rows={len(report) - len(err)}")

    print("[5/7] 行业字典 + 变更明细...")
    dict_df = industry_dict_to_df(d2021, d2014, union)
    dict_df.to_parquet(OUT_DIR / "industry_dict.parquet", index=False)
    changes = build_changes(intervals, union, l1_names, l3_names)
    changes.to_parquet(OUT_DIR / "changes.parquet", index=False)
    changes.to_csv(OUT_DIR / "changes.csv",
                   index=False, encoding="utf-8-sig")
    print("  change_type 分布:")
    print(changes["change_type"].value_counts().to_string())

    print("[6/7] 构造 L1 名称映射...")
    intervals_named = intervals.copy()
    intervals_named["l1_name"] = intervals_named["l1_code"].map(l1_names)
    miss = intervals_named["l1_name"].isna()
    miss_l1 = sorted(intervals_named.loc[miss, "l1_code"].unique())
    print(f"  L1 名称未命中: {miss.sum()}/{len(intervals_named)} "
          f"(L1 代码: {miss_l1}, 均为 2014 版前废弃分类)")

    print("[7/7] 全交易日宽表...")
    trade_days = get_trade_days(START_DATE, END_DATE)
    print(f"  trade_days={len(trade_days)} ({trade_days[0].date()} ~ "
          f"{trade_days[-1].date()})")

    for value_col, fname in [
        ("l1_code", "wide_l1_code.parquet"),
        ("l2_code", "wide_l2_code.parquet"),
        ("l3_code", "wide_l3_code.parquet"),
    ]:
        wide = build_wide(intervals, trade_days, value_col)
        wide.to_parquet(OUT_DIR / fname)
        print(f"  {fname}: {wide.shape}")

    wide_l1_name = build_wide(intervals_named, trade_days, "l1_name")
    wide_l1_name.to_parquet(OUT_DIR / "wide_l1_name.parquet")
    print(f"  wide_l1_name.parquet: {wide_l1_name.shape}")

    sparse = build_sparse_wide(intervals, "l3_code")
    sparse.to_parquet(OUT_DIR / "wide_sparse_l3_code.parquet")
    print(f"  wide_sparse_l3_code.parquet: {sparse.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
