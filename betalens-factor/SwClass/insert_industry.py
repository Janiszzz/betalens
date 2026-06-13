#%%By Janis 260602
"""申万个股行业分类长表 → industry 表（point-in-time 行业归属）。

输入: SwClass/StockClassifyUse_stock_20260601.xls (4 列长表)
      SwClass/2014to2021.xlsx                     (行业代码↔名称, 含 2014/2021 双版本)
      SwClass/最新个股申万行业分类(完整版-截至7月末).xlsx (wind代码↔公司简称, 补 name)

适配:
  1. 6 位行业代码按每两位拆 L1/L2/L3, 各自一条 metric '申万{级}行业（{版本}）'。
  2. 6 位证券代码 → wind 格式 (.SH/.SZ/.BJ)。

写入: industry 表 (datetime, code, name, metric, value, remark JSONB)
      按 (datetime, code, metric) 去重, 可重复运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BETALENS_ROOT = r"c:/Users/Janis/OneDrive/betalens"
if BETALENS_ROOT not in sys.path:
    sys.path.insert(0, BETALENS_ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_industry_wide as B  # 复用 load_long_table / load_industry_dict / 版本常量

from betalens.datafeed import Datafeed, incremental_insert

NAME_XLSX = ROOT / "最新个股申万行业分类(完整版-截至7月末).xlsx"

LEVELS = [
    ("一级", 2),  # L1: 前 2 位
    ("二级", 4),  # L2: 前 4 位
    ("三级", 6),  # L3: 全 6 位
]
LEVEL_NAME_KEY = {"一级": "l1_name", "二级": "l2_name", "三级": "l3_name"}


def to_wind(code6: str) -> str:
    """6 位数字证券代码 → wind 格式。6→.SH, 0/3→.SZ, 4/8/9→.BJ(北交所)。"""
    h = code6[0]
    if h == "6":
        return code6 + ".SH"
    if h in "03":
        return code6 + ".SZ"
    return code6 + ".BJ"


def era_of(dt: pd.Timestamp) -> str:
    """计入日期 → 版本族。"""
    if dt >= B.VER_2021:
        return "2021"
    if dt >= B.VER_2014:
        return "2014"
    return "旧版"


def load_name_map(path: Path) -> dict:
    """读最新分类表, 构造 {wind_code: 公司简称}。表头 GBK, 按位置取列。"""
    raw = pd.read_excel(path)
    # 列序: 市场 / 行业代码 / 股票代码(wind) / 公司简称 / 新版一/二/三级
    code_col, name_col = raw.columns[2], raw.columns[3]
    sub = raw[[code_col, name_col]].dropna(subset=[code_col])
    out = {}
    for c, n in zip(sub[code_col].astype(str), sub[name_col]):
        if pd.notna(n):
            out[c] = str(n).strip()
    return out


def build_records(long_df: pd.DataFrame, d2014: dict, d2021: dict,
                  name_map: dict) -> pd.DataFrame:
    """长表 → 长格式记录 (每行 ×3 级)。"""
    rows = []
    for code6, enter, ind6, _upd in zip(
        long_df["code"], long_df["enter_date"],
        long_df["ind_code"], long_df["update_ts"],
    ):
        wind = to_wind(code6)
        era = era_of(enter)
        dic = d2014 if era == "2014" else d2021 if era == "2021" else {}
        info = dic.get(ind6, {})
        sec_name = name_map.get(wind, wind)  # 缺失用 wind 代码占位 (NOT NULL)
        for lvl, n in LEVELS:
            metric = f"申万{lvl}行业（{era}）"
            sub_code = ind6[:n]
            ind_name = info.get(LEVEL_NAME_KEY[lvl])
            rows.append({
                "datetime": enter,
                "code": wind,
                "name": sec_name,
                "metric": metric,
                "value": int(sub_code),
                "remark": {
                    "ind_name": ind_name,
                    "ind_code": sub_code,
                    "scheme": metric,
                },
            })
    df = pd.DataFrame(rows, columns=[
        "datetime", "code", "name", "metric", "value", "remark"])
    return df


def main():
    print("[1/4] 读长表 + 行业字典 + 名称表...")
    long_df = B.load_long_table(B.LONG_XLS)
    d2021, d2014, *_ = B.load_industry_dict(B.DICT_XLSX)
    name_map = load_name_map(NAME_XLSX)
    print(f"  长表 rows={len(long_df)}, stocks={long_df['code'].nunique()}; "
          f"d2014={len(d2014)}, d2021={len(d2021)}, names={len(name_map)}")

    print("[2/4] 构造长格式记录 (每行×3级)...")
    records = build_records(long_df, d2014, d2021, name_map)
    print(f"  records={len(records)}")
    print(records["metric"].value_counts().to_string())

    print("[3/4] 连接数据库 + 注册 JSONB 适配器...")
    import psycopg2.extensions
    import psycopg2.extras
    psycopg2.extensions.register_adapter(dict, psycopg2.extras.Json)
    feed = Datafeed("industry")
    try:
        print("[4/4] 增量入库 industry 表...")
        new_rows, skipped = incremental_insert(
            feed.cursor, feed.conn, records, table="industry")
        print(f"  新增 {new_rows} 行, 跳过 {skipped} 行 (已存在)")
    finally:
        feed.close()
    print("Done.")


if __name__ == "__main__":
    main()
