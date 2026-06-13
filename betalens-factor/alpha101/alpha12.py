#%%
"""
Alpha#12 因子计算与回测管线

公式: Alpha#12 = sign(delta(volume, 1)) * (-1 * delta(close, 1))
来源: WorldQuant 101 Formulaic Alphas (Kakushadze, 2016)

逻辑: 成交量放大时做空涨价股（反转），成交量萎缩时做多涨价股
因子方向: 正向
"""

import akshare as ak
import pandas as pd
import numpy as np

from betalens.datafeed import Datafeed, get_absolute_trade_days
from betalens.factor.factor import single_characteristic, get_single_factor_weight
from betalens.factor.preprocessing import preprocess_factor
from betalens.factor.stats import calc_ic, summarize_ic, group_return_summary
from betalens.backtest import BacktestBase
from betalens.analyst import PortfolioAnalyzer


DB_TABLE = "daily_market"
METRIC_CLOSE  = "收盘价(元)"
METRIC_VOLUME = "成交量(股)"
FACTOR_NAME   = "alpha12"

def _get_csi1000_universe(date: str = "2024-01-01") -> list | None:
    """获取中证1000在指定日期的历史成分股代码列表。

    直接调用 index_stock_cons_weight_csindex 获取某日成分快照；若当日非
    交易日则向后滚动查找至多 7 天。

    Args:
        date: 日期字符串 "YYYY-MM-DD"，默认 2024-01-01。
    """
    _ = date  # 接口仅返回最新快照，不支持历史日期
    try:
        df = ak.index_stock_cons_weight_csindex(symbol="000852")
    except Exception:
        return None
    if df is None or df.empty:
        return None
    code_col = next((c for c in df.columns
                     if "代码" in c and "指数" not in c), None)
    if not code_col:
        return None
    codes = df[code_col].astype(str).str.zfill(6).tolist()
    return [c + (".SH" if c.startswith(("60", "68", "9")) else ".SZ")
            for c in codes]

# 默认选股池（None = 全市场）；可在调用时覆盖
DEFAULT_UNIVERSE: list | None = _get_csi1000_universe()
pd.DataFrame(DEFAULT_UNIVERSE).to_excel("chi.xlsx")

# ───────────────────────────────────────────────
# 数据提取
# ───────────────────────────────────────────────

def fetch_daily_wide(metric, universe=None, start_date=None, end_date=None,
                     table_name=DB_TABLE):
    data = Datafeed(table_name)
    try:
        df = data.query_time_range(
            codes=universe, start_date=start_date,
            end_date=end_date, metric=metric
        )
        df.to_excel("search.xlsx")
    finally:
        data.close()

    if df.empty:
        return pd.DataFrame()

    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['datetime'])
    wide = df.pivot_table(index='datetime', columns='code', values='value')
    return wide.sort_index()


def fetch_industry_map(date_list, table_name="fundamentals"):
    data = Datafeed(table_name)
    frames = []
    try:
        for date in date_list:
            dt_str = f"{date} 10:00:00"
            df = data.query_time_range(
                start_date=dt_str, end_date=dt_str, metric="行业"
            )
            if not df.empty:
                df = df[['datetime', 'code', 'value']].copy()
                df.columns = ['input_ts', 'code', 'industry']
                frames.append(df)
    finally:
        data.close()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['input_ts', 'code', 'industry'])


def fetch_log_mktcap_map(date_list, table_name="fundamentals"):
    data = Datafeed(table_name)
    frames = []
    try:
        for date in date_list:
            dt_str = f"{date} 10:00:00"
            df = data.query_time_range(
                start_date=dt_str, end_date=dt_str, metric="log_mktcap"
            )
            if not df.empty:
                df = df[['datetime', 'code', 'value']].copy()
                df.columns = ['input_ts', 'code', 'log_mktcap']
                df['log_mktcap'] = pd.to_numeric(df['log_mktcap'], errors='coerce')
                frames.append(df)
    finally:
        data.close()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['input_ts', 'code', 'log_mktcap'])


# ───────────────────────────────────────────────
# 因子计算
# ───────────────────────────────────────────────

def compute_alpha12(close_wide, volume_wide):
    """
    Alpha#12 = sign(delta(volume, 1)) * (-1 * delta(close, 1))
    """
    delta_vol   = volume_wide.diff(1)
    delta_close = close_wide.diff(1)
    factor = np.sign(delta_vol) * (-1 * delta_close)
    return factor.replace([np.inf, -np.inf], np.nan)


# ───────────────────────────────────────────────
# 格式转换
# ───────────────────────────────────────────────

def wide_to_prequery(wide_df, metric_name, signal_dates,
                     industry_map=None, mktcap_map=None):
    """
    signal_dates: 因子计算日（调仓日前一交易日），与 wide_df.index 对齐。
    wide_df.index 是带时间的 Timestamp，signal_dates 是 date 对象列表，
    用 .date() 比较避免时间部分不匹配。
    """
    date_set = set(signal_dates)
    mask = wide_df.index.map(lambda ts: ts.date() in date_set)
    wide_df = wide_df.loc[mask]

    long = wide_df.stack().reset_index()
    long.columns = ['input_ts', 'code', metric_name]
    long['input_ts'] = pd.to_datetime(long['input_ts'])
    long['datetime'] = long['input_ts']
    long['diff_hours'] = 0.0

    if industry_map is not None and not industry_map.empty:
        industry_map = industry_map.copy()
        industry_map['input_ts'] = pd.to_datetime(industry_map['input_ts'])
        long = long.merge(industry_map, on=['input_ts', 'code'], how='left')

    if mktcap_map is not None and not mktcap_map.empty:
        mktcap_map = mktcap_map.copy()
        mktcap_map['input_ts'] = pd.to_datetime(mktcap_map['input_ts'])
        long = long.merge(mktcap_map, on=['input_ts', 'code'], how='left')

    return long


# ───────────────────────────────────────────────
# 完整管线
# ───────────────────────────────────────────────

def run_alpha12_pipeline(
    close_wide, volume_wide, rebalance_dates,
    all_trade_days,
    universe=None,
    industry_map=None, mktcap_map=None,
    n_quantiles=20, initial_amount=1e8,
    save_weights_csv=None,
):
    """
    Alpha#12 完整管线: 计算 → 预处理 → 分组 → 权重 → 回测 → 统计

    时序逻辑：
        signal_date（调仓日前一交易日收盘）→ 计算因子
        rebalance_date（调仓日开盘）→ 按因子值调仓
        收益率 = rebalance_date 收盘 → 下一 rebalance_date 收盘

    Args:
        all_trade_days: 完整交易日列表（date对象），用于查找前一交易日
    """
    # 0. 构建调仓日 → 信号日映射（调仓日的前一个交易日）
    trade_day_set = sorted(all_trade_days)
    trade_day_idx = {d: i for i, d in enumerate(trade_day_set)}
    signal_dates = []
    valid_rebal  = []
    for rd in rebalance_dates:
        i = trade_day_idx.get(rd)
        if i is not None and i > 0:
            signal_dates.append(trade_day_set[i - 1])
            valid_rebal.append(rd)
    rebalance_dates = valid_rebal
    """ 
    # 1. 限制选股池
    if universe is not None:
        universe = list(universe)
        cols = [c for c in close_wide.columns if c in universe]
        close_wide  = close_wide[cols]
        volume_wide = volume_wide[cols]
    """   
    # 2. 计算全量因子（日频），再按信号日切片
    factor_wide = compute_alpha12(close_wide, volume_wide)
    factor_wide.to_excel("factor_wide.xlsx")
    # 3. 格式转换：因子值取信号日（调仓日前一天收盘）
    industry_col = 'industry'   if (industry_map is not None and not industry_map.empty) else None
    mktcap_col   = 'log_mktcap' if (mktcap_map   is not None and not mktcap_map.empty)   else None

    prequery_data = wide_to_prequery(
        factor_wide, FACTOR_NAME, signal_dates,
        industry_map=industry_map, mktcap_map=mktcap_map,
    )
    """ 

    # 4. 预处理
    cleaned = preprocess_factor(
        prequery_data, FACTOR_NAME,
        winsorize_method='std', winsorize_n=3,
        standardize_method='zscore',
        industry_col=industry_col,
        log_mktcap_col=mktcap_col,
    )
    """    
    cleaned = prequery_data
# 5. 分组排序
    labeled = single_characteristic(
        cleaned, FACTOR_NAME, {FACTOR_NAME: n_quantiles}
    )
    labeled.to_excel("labeled.xlsx")
    # 6. 多空权重（正向因子：高分组做多）
    weights = get_single_factor_weight(labeled, {
        'factor_key': FACTOR_NAME,
        'mode': 'freeplay',
        'long': [19],
        'short': [],
    })
    weights.index = weights.index + pd.Timedelta(minutes=10)
    #weights = weights.shift(1).dropna()
    #weights['cash'] = 1.0
    weights.to_excel("weights.xlsx")
    # 7. 回测（bt 内部自动查价格、算调仓日间收益率）
    bt = BacktestBase(weights, metric="开盘价(元)", symbol=FACTOR_NAME, amount=initial_amount, time_tolerance=24*11)

    # 8. 绩效评估
    nav = bt.nav.dropna()
    nav.to_excel("nav.xlsx")

    analyzer = PortfolioAnalyzer(
        nav,
        risk_free_rate=0.00,   # 年化无风险利率
        annualizer=252,        # 日频
        window=20,             # 滚动窗口（约 1 个月）
    )

    print("\n" + "=" * 50)
    print(f"  {FACTOR_NAME} 绩效概览")
    print("=" * 50)
    print(f"回测区间      {nav.index[0].date()} ~ {nav.index[-1].date()} "
          f"({len(nav)} 日)")
    print(f"初始资金      {initial_amount:,.0f}")
    print(f"期末净值      {nav.iloc[-1]:.4f}")
    print(f"累计收益      {analyzer.total_return():.2%}")
    print(f"年化收益      {analyzer.annualized_return():.2%}")
    print(f"年化波动      {analyzer.annualized_volatility():.2%}")
    print(f"夏普比率      {analyzer.sharpe_ratio():.2f}")
    print(f"最大回撤      {analyzer.max_drawdown():.2%}")
    try:
        print(f"卡玛比率      {analyzer.calmar_ratio():.2f}")
    except Exception as e:
        print(f"卡玛比率      N/A ({e})")

    # 9. 报告（分年度 + 指定时段；需要基准请传入 benchmark_analyzer）
    from betalens.analyst.analyst import ReportExporter
    exporter = ReportExporter(analyzer)
    exporter.generate_annual_report(excel_path=f"{FACTOR_NAME}_annual.xlsx")
    exporter.generate_custom_report(
        start_date=nav.index[0].strftime('%Y-%m-%d'),
        end_date=nav.index[-1].strftime('%Y-%m-%d'),
        excel_path=f"{FACTOR_NAME}_custom.xlsx",
    ) 
    bt.dump_to_excel('alpha12_dump.xlsx')
    return bt, analyzer


# ───────────────────────────────────────────────
# 入口
# ───────────────────────────────────────────────

if __name__ == '__main__':
    START_DATE = "2024-01-01"
    END_DATE   = "2025-12-31"
    REBAL_FREQ = "D"

    # 选股池：传入代码列表即可限定范围，None = 全市场
    UNIVERSE = DEFAULT_UNIVERSE  # 例: ['000001.SZ', '000002.SZ', ...]

    rebalance_dates = get_absolute_trade_days(START_DATE, END_DATE, REBAL_FREQ, use_pmc=False)
    all_trade_days  = get_absolute_trade_days(START_DATE, END_DATE, "D",        use_pmc=False)
    print(f"调仓日数量: {len(rebalance_dates)}")

    print("提取行情数据...")
    close_wide  = fetch_daily_wide(METRIC_CLOSE,  universe=UNIVERSE,
                                   start_date=START_DATE, end_date=END_DATE)
    close_wide.to_excel("close_wide.xlsx")
    volume_wide = fetch_daily_wide(METRIC_VOLUME, universe=UNIVERSE,
                                   start_date=START_DATE, end_date=END_DATE)
    print(f"  close : {close_wide.shape}")
    print(f"  volume: {volume_wide.shape}")

    print("提取行业和市值数据...")
    #industry_map = fetch_industry_map(rebalance_dates)
    #mktcap_map   = fetch_log_mktcap_map(rebalance_dates)

    result = run_alpha12_pipeline(
        close_wide, volume_wide, rebalance_dates,
        all_trade_days=all_trade_days,
        universe=UNIVERSE,
        n_quantiles=20,
        save_weights_csv="alpha12_weights.csv",  # None = 不保存
    )
