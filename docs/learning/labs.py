"""Betalens 课程离线实验。所有数据均为虚构，不连接数据库，不写文件。

在仓库根目录运行：python docs/learning/labs.py 02
列出实验：python docs/learning/labs.py --list
运行全部：python docs/learning/labs.py all
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd


def lab02():
    """表格、筛选与宽长转换。"""
    long = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
        "code": ["A", "B", "A", "B"], "close": [10.0, 20.0, 11.0, 19.0],
    })
    print("长表：\n", long.to_string(index=False))
    print("只看 A：\n", long.loc[long["code"] == "A"].to_string(index=False))
    wide = long.pivot(index="date", columns="code", values="close")
    print("宽表：\n", wide)
    print("宽表变回长表：\n", wide.rename_axis(columns="code").stack().rename("close").reset_index())
    assert wide.loc["2024-01-03", "A"] == 11.0


def lab03():
    """收益、复利与未来收益的标签。"""
    prices = pd.Series([100.0, 110.0, 99.0], index=pd.date_range("2024-01-02", periods=3), name="price")
    realized = prices.pct_change(fill_method=None)
    forward = prices.shift(-1) / prices - 1
    print(pd.DataFrame({"price": prices, "已发生收益": realized, "下一期收益标签": forward}))
    print("两期复利：", (1.1 * 0.9 - 1), "；不是 10% - 10% = 0")
    print("shift(-1) 在这里仅生成事后评价标签，不能作为当期选股输入。")
    assert np.isclose((1 + realized.dropna()).prod() - 1, -0.01)
    assert pd.isna(forward.iloc[-1])


def lab08():
    """缺失值、重复键和单位。"""
    raw = pd.DataFrame({"date": ["2024-01-02"] * 4, "code": ["A", "A", "B", "C"],
                        "close": [10.0, 10.1, np.nan, 0.0]})
    print("原始记录：\n", raw)
    print("冲突键：\n", raw[raw.duplicated(["date", "code"], keep=False)])
    print("空值或非正价格：\n", raw[raw["close"].isna() | raw["close"].le(0)])
    print("同日 3 只候选中，A 的两条记录冲突，B 缺失，C 非正，不能直接形成有效价格面板。")
    print("假定核对来源后明确 A=10.1、B=20，C 仍缺失：")
    verified = pd.Series({"A": 10.1, "B": 20.0, "C": np.nan})
    print("覆盖率：", verified.notna().mean(), "；这是人工核验后的教学修正，不是自动补值。")
    print("涨跌幅字段 3 若表示 3%，用于收益乘法前应除以 100：", 3 / 100)
    assert np.isclose(verified.notna().mean(), 2 / 3)


def lab09():
    """自己编写动量公式。"""
    prices = pd.DataFrame({"A": [10., 11., 12., 13., 14.], "B": [20., 19., 18., 17., 16.]},
                          index=pd.bdate_range("2024-01-02", periods=5))
    def momentum(close_wide, window):
        return close_wide / close_wide.shift(window) - 1
    scores = momentum(prices, window=2)
    print("虚构价格：\n", prices, "\n两行回看动量：\n", scores)
    print("本例索引为教学工作日序列，不能替代项目的交易所日历。")
    assert scores.iloc[:2].isna().all().all()
    assert np.isclose(scores.iloc[2]["A"], 0.2)


def lab10():
    """使用真实预处理函数。"""
    from betalens.factor.preprocessing import winsorize_factor, standardize_factor, neutralize_factor
    x = pd.Series([1., 2., 3., 4., 100.], index=list("ABCDE"))
    clipped = winsorize_factor(x, method="mad", n=3)
    print("去极值与标准化：\n", pd.DataFrame({"原值": x, "截尾": clipped,
                                         "zscore": standardize_factor(clipped)}))
    scores = pd.Series([1., 2., 3., 4., 5., 6., 11., 12., 13., 14., 15., 16.])
    industry = pd.Series(["行业甲"] * 6 + ["行业乙"] * 6)
    standardized = standardize_factor(scores)
    residual, diagnostics = neutralize_factor(standardized, industry_labels=industry, return_stats=True)
    print("行业中性化：\n", pd.DataFrame({"industry": industry, "before": standardized, "after": residual}))
    print("诊断：", diagnostics, "\n残差行业均值：\n", residual.groupby(industry).mean())
    assert clipped.iloc[-1] == 6.0
    assert not diagnostics["skipped"]
    assert np.allclose(residual.groupby(industry).mean(), 0, atol=1e-10)


def lab11():
    """同分值分组和显式多空选择。"""
    from betalens.factor.factor import single_characteristic, get_single_factor_weight
    raw = pd.DataFrame({"input_ts": [pd.Timestamp("2024-01-04 15:00")] * 6,
                        "code": list("ABCDEF"), "score": [0, 0, 0, 0, 1, 1]})
    for mode in ["equal_count", "value"]:
        groups = single_characteristic(raw, "score", {"score": 3}, grouping_mode=mode)
        print(f"\n{mode} 分组：\n", groups)
        weights = get_single_factor_weight(groups, {"factor_key": "score", "mode": "freeplay",
                                                    "long": ["max"], "short": [], "grouping_mode": mode})
        print("只做多最高组：\n", weights)
        assert np.isclose(weights.sum(axis=1).iloc[0], 1)
    try:
        single_characteristic(raw, "score", {"score": 10})
    except ValueError as error:
        print("\n预期的学习错误（已捕获）：", error)
    else:
        raise AssertionError("六只股票不应能生成十个严格等额组")


def lab13():
    """真实 BacktestBase，使用内存价格，关闭外部交易状态查询。"""
    from betalens.backtest import BacktestBase
    dates = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.DataFrame({"DEMO": [100., 110., 90., 120., 130.]}, index=dates)
    weights = pd.DataFrame({"DEMO": 1., "cash": 0.}, index=dates)
    bt = BacktestBase(weights, symbol="COURSE_BUY_HOLD", amount=100000.,
                      check_trade_status=False, verbose=False, lot_size=100,
                      preloaded_cost_price=prices, preloaded_close_price=prices)
    print("引擎净值：\n", bt.nav, "\n持股数量：\n", bt.position,
          "\n交易审计：\n", bt.rebalance_log)
    print("首次 10 万 / 100 元 = 1000 股，后续目标仍为全仓，持股数量不变。")
    print("另一个取整例：1 万预算 / 30 元股价 => 300 股，剩余 1000 元。")
    assert np.allclose(bt.nav.to_numpy(), [1., 1.1, .9, 1.2, 1.3])
    assert np.allclose(bt.position["DEMO"], 1000.)
    assert len(bt.rebalance_log) == 1


def lab15():
    """调用真实绩效指标，区分回撤幅度和负号表示。"""
    # 只加载纯函数源码，避免 Analyst 包初始化额外要求 prettytable 等报告依赖。
    # 正式安装完整依赖后，常规用法是 from betalens.analyst.metrics import ...。
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "betalens" / "analyst" / "metrics.py"
    spec = importlib.util.spec_from_file_location("course_metrics", path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    total_return = metrics.total_return
    max_drawdown = metrics.max_drawdown
    annualized_return = metrics.annualized_return
    sharpe_ratio = metrics.sharpe_ratio
    nav = pd.Series([1., 1.2, .9, 1.1], index=pd.bdate_range("2024-01-02", periods=4))
    returns = nav.pct_change(fill_method=None).dropna()
    print("净值：\n", nav)
    print("累计收益：", total_return(nav), "最大回撤幅度：", max_drawdown(nav))
    print("当前实现的年化收益：", annualized_return(nav), "年化夏普：", sharpe_ratio(returns))
    print("仅四个净值点，年化数字不适合评价长期表现。")
    print("当前 annualized_return 的指数分母为 len(nav)，复核时注意与收益区间数的区别。")
    assert np.isclose(total_return(nav), .1)
    assert np.isclose(max_drawdown(nav), .25)


def lab16():
    """真实 Rank IC：正确顺序、反向顺序、样本不足。"""
    from betalens.factor.stats import calc_ic
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    scores = pd.DataFrame([range(1, 7), range(1, 7)], index=dates, columns=list("ABCDEF"))
    forward_returns = pd.DataFrame([[.01, .02, .03, .04, .05, .06],
                                   [.06, .05, .04, .03, .02, .01]], index=dates, columns=scores.columns)
    ic = calc_ic(scores, forward_returns, method="spearman")
    print("分数：\n", scores, "\n已经按信号日对齐的虚构未来收益：\n", forward_returns,
          "\nRank IC：\n", ic)
    sparse = calc_ic(scores.iloc[:, :4], forward_returns.iloc[:, :4])
    print("只剩四只有效股票：\n", sparse)
    assert np.allclose(ic, [1., -1.])
    assert sparse.isna().all()


def lab21():
    """教学择时阈值，历史窗口不包括当前观测。"""
    values = pd.Series([1., 2., 1., 2., 1., 2., 8., 1., 2.],
                       index=pd.bdate_range("2024-01-02", periods=9))
    history = values.shift(1)
    upper = history.rolling(4, min_periods=4).mean() + history.rolling(4, min_periods=4).std()
    triggered = values.gt(upper) & upper.notna()
    target = triggered.astype(float) * -1
    print(pd.DataFrame({"因子值": values, "此前4个值的上轨": upper,
                        "触发": triggered, "信号目标仓位": target}))
    print("本例只解释信号，没有模拟执行。项目择时版须另核对执行时点、有效观测规则和默认窗口。")
    assert target.iloc[6] == -1
    assert target.iloc[:4].eq(0).all()


def lab22():
    """事件聚合：先平均再复利与先复利再平均不同。"""
    returns = pd.DataFrame({"事件甲": [.1, -.1], "事件乙": [-.1, .1]}, index=[1, 2])
    each_cumulative = (1 + returns).prod() - 1
    average_then_compound = (1 + returns.mean(axis=1)).prod() - 1
    print("两个虚构事件的事件后日收益：\n", returns)
    print("每次事件累计收益：\n", each_cumulative)
    print("先逐事件复利，再平均：", each_cumulative.mean())
    print("先逐日平均，再复利：", average_then_compound)
    print("这只是聚合算术，不调用 EventStudy，也不代表其所有图表都使用同一种口径。")
    assert np.allclose(each_cumulative, -.01)
    assert np.isclose(average_then_compound, 0)


def lab23():
    """从纯噪声候选中挑冠军，观察选择偏差。"""
    rng = np.random.default_rng(20260904)
    returns = pd.DataFrame(rng.normal(0, .01, size=(120, 100)),
                           columns=[f"candidate_{i:03d}" for i in range(100)])
    train, test = returns.iloc[:60], returns.iloc[60:]
    winner = train.mean().idxmax()
    table = pd.DataFrame({"训练均值": train.mean(), "留出期均值": test.mean()})
    print("按训练期排序的前五个纯噪声候选：\n", table.sort_values("训练均值", ascending=False).head())
    print("训练期选出的冠军：", winner, "\n", table.loc[winner])
    print("所有候选真实生成均值都是零。留出结果不保证为负，但训练期冠军优势没有被生成机制保证。")
    print("不能查看留出表现后再换冠军，然后仍把这段数据称为独立测试集。")
    assert train.index.intersection(test.index).empty


LABS = {"02": lab02, "03": lab03, "08": lab08, "09": lab09, "10": lab10,
        "11": lab11, "13": lab13, "15": lab15, "16": lab16,
        "21": lab21, "22": lab22, "23": lab23}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lesson", nargs="?", choices=[*LABS, "all"])
    parser.add_argument("--list", action="store_true", help="显示可运行的离线课次")
    args = parser.parse_args()
    if args.list or not args.lesson:
        for lesson, function in LABS.items():
            print(f"第 {lesson} 课：{function.__doc__}")
        return
    selected = LABS if args.lesson == "all" else {args.lesson: LABS[args.lesson]}
    for lesson, function in selected.items():
        print(f"\n{'=' * 16} 第 {lesson} 课 {'=' * 16}")
        function()
        print(f"第 {lesson} 课：基准演算核对通过。")


if __name__ == "__main__":
    main()
