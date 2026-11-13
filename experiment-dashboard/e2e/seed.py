"""Create isolated UI fixtures: python experiment-dashboard/e2e/seed.py."""
from pathlib import Path
import argparse
import runpy
import shutil

from betalens.experiments import ExperimentInput, ExperimentStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Recreate only the isolated experiments-e2e fixture database")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    path = root / "outputs/experiments-e2e"
    if args.reset and path.exists():
        shutil.rmtree(path)
    fixtures = runpy.run_path(str(root / "tests/test_experiments.py"))
    backtest = fixtures["bt"].__wrapped__()
    import pandas as pd
    from betalens.analyst import Analyst
    days = backtest.daily_weight.index
    backtest.daily_pnl_total = backtest.daily_pnl.sum(axis=1)
    backtest.factor_values = pd.DataFrame({"信号日": days, "股票代码": "000001.SZ", "因子值": range(len(days)), "分组": [1, 2] * 4})
    backtest.research_group_nav = pd.DataFrame({"G1": [1 + i * .01 for i in range(len(days))], "G2": [1 - i * .005 for i in range(len(days))]}, index=days)
    analyst = Analyst.from_backtest(backtest)
    analyst._name_map = {"000001.SZ": "平安银行"}
    with ExperimentStore(path) as store:
        factor = store.register_factor(key="momentum", name="动量因子", category="量价研究", description="比较不同观察窗口下的价格动量，记录参数探索与组合表现。")
        store.derive_factor(factor, key="factor_momentum_20", name="20日动量", locks={"factor_spec.compute_kwargs.window": 20}, description="固定二十日观察窗口，保留不同评价区间与批次的研究结果。")
        other = store.register_factor(key="timing", name="趋势择时", category="择时研究", strategy_type="timing", description="以趋势信号调整风险敞口，观察净值与持仓变化。")
        store.save_batch([ExperimentInput(backtest, factor, fixtures["config"](60), {"window": 60}, source_id="momentum-60"),
                          ExperimentInput(backtest, other, fixtures["config"](), {"window": 20}, source_id="timing-20")], source_id="ui-parallel")
        store.save_mining(fixtures["mining_result"](root / "outputs", captured=True), factor_ids={"动量": factor})
        store.save_backtest(backtest, analyst=analyst, factor_id=factor, config=fixtures["config"](), parameters={"window": 20}, source_id="ui-manual", tags=["精选"])
    print(root / "outputs/experiments-e2e")


if __name__ == "__main__":
    main()
