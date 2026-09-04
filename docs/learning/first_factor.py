"""离线教学：使用 Betalens 的真实分组函数；所有股票和收益均为虚构。

在仓库根目录运行：python docs/learning/first_factor.py
不连接数据库，不调用回测引擎，不写入研究数据。
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from betalens.factor.factor import single_characteristic, get_single_factor_weight


def main():
    # DataFrame 可以理解为一张 Excel 表。这里每一行是一只虚构股票。
    market = pd.DataFrame({
        "code": ["A", "B", "C", "D", "E", "F"],
        "open": [10.0] * 6,
        "close": [9.2, 9.6, 9.9, 10.1, 10.4, 10.8],
        "high": [11.0] * 6,
        "low": [9.0] * 6,
    })
    # ALPHA101 的公式；值越大，收盘相对于开盘越强。
    market["score"] = (
        (market["close"] - market["open"])
        / (market["high"] - market["low"] + 0.001)
    )
    market["input_ts"] = pd.Timestamp("2024-01-04 15:00:00")
    print("\n1. 虚构行情与因子值：")
    print(market[["code", "open", "close", "high", "low", "score"]].to_string(index=False))

    # 三组的标签是 0、1、2：从低分到高分，每组两只。
    labeled = single_characteristic(
        market, "score", {"score": 3}, grouping_mode="equal_count"
    )
    print("\n2. Betalens 分组结果：")
    print(labeled.to_string())

    weights = get_single_factor_weight(labeled, {
        "factor_key": "score", "mode": "classic-long-short",
    })
    # 函数可能省略从未持有的股票列；教学展示时补回零权重。
    weights = weights.reindex(columns=market["code"].tolist(), fill_value=0.0)
    # 多头 +1、空头 -1，股票净权重为零；现金按净资产口径补足。
    weights["cash"] = 1.0 - weights.sum(axis=1)
    print("\n3. Betalens 目标权重：")
    print(weights.to_string())

    # 后续持有期收益只用于事后评价，绝不参与前面的打分和选股。
    future_returns = pd.Series({
        "A": -0.02, "B": -0.01, "C": 0.00,
        "D": 0.01, "E": 0.03, "F": 0.04, "cash": 0.0,
    })
    contributions = weights.iloc[0].mul(future_returns)
    result = pd.DataFrame({
        "weight": weights.iloc[0],
        "future_return": future_returns,
        "contribution": contributions,
    })
    print("\n4. 虚构持有期收益与贡献：")
    print(result.to_string())
    print(f"\n教学组合收益：{contributions.sum():.2%}，净值从 1 变为 {1 + contributions.sum():.3f}")
    print("这是单期权重乘收益的教学算术；未模拟成交、整数手、停牌和费用。")
    print("练习：把最后的未来收益改成负数重跑，观察高分组也可能亏损。")


if __name__ == "__main__":
    main()
