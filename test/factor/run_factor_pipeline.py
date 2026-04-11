4
"""
因子端到端验证脚本
===================
完整流程：
  数据库 → 因子预处理 → 10组回测 → IC/Fama-MacBeth 检验 → 可视化 → Excel 导出

输出文件（SAVE_RESULTS=True 时）：
  output/{METRIC}_{date}.xlsx，含6个sheet：
    labeled_pool, weights, group_returns, nav, stats, factor_stats

参照 test/eventstudy/run_eventstudy.py 的代码风格。
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免弹窗
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
from io import BytesIO
import openpyxl
from openpyxl.drawing.image import Image as XLImage

from betalens.datafeed import Datafeed
from betalens.backtest.backtest import BacktestBase
from betalens.analyst.analyst import PortfolioAnalyzer, ReportExporter
from betalens.factor.factor import (
    get_tradable_pool,
    pre_query_characteristic_data,
    single_characteristic,
    get_single_factor_weight,
)
from betalens.factor.preprocessing import preprocess_factor
from betalens.factor.stats import calc_ic, summarize_ic, fama_macbeth, group_return_summary


# ══════════════════════════════════════════════
#  配置区（按实际情况修改）
# ══════════════════════════════════════════════
METRIC = "ROE"                         # 要检验的因子（数据库中的列名）
N_GROUPS = 10                          # 分组数
SAVE_RESULTS = True                    # True → 保存 Excel 和图片

# 调仓日期列表（月末调仓示例，需按实际改）
DATE_LIST = pd.date_range("2020-01-31", "2024-12-31", freq="ME").tolist()

# 数据库表名（按实际改）
TABLE_FUNDAMENTAL = "fundamental_data"  # 财务指标表
TABLE_MARKET = "daily_market"           # 日行情表
CLOSE_METRIC = "收盘价"                  # 收盘价列名（daily_market 中）

# 因子预处理参数
WINSORIZE_METHOD = 'mad'
WINSORIZE_N = 3.0
STANDARDIZE_METHOD = 'zscore'
INDUSTRY_COL = None                    # 若 pre_queried_data 含行业列，填列名
LOG_MKTCAP_COL = None                  # 若含 log市值列，填列名

# 多空权重参数
LONG_LABELS = [N_GROUPS]               # 最高组做多
SHORT_LABELS = [1]                     # 最低组做空
INITIAL_AMOUNT = 1_000_000

# 输出路径
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_PATH = OUTPUT_DIR / f"{METRIC}_{datetime.today().strftime('%Y%m%d')}.xlsx"
# ══════════════════════════════════════════════


def _fig_to_image(fig) -> XLImage:
    """将 matplotlib Figure 转为 openpyxl Image 对象（嵌入 Excel 用）。"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = XLImage(buf)
    return img


def plot_group_returns(group_returns: pd.DataFrame, metric: str) -> plt.Figure:
    """图1：各分组累积收益柱状图（持仓期收益均值）。"""
    group_cols = [c for c in group_returns.columns if c.startswith('G')]
    means = group_returns[group_cols].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in means]
    ax.bar(means.index, means.values * 100, color=colors)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('分组')
    ax.set_ylabel('平均持仓期收益率 (%)')
    ax.set_title(f'{metric} 因子分组平均收益（G1=最低，G{len(group_cols)}=最高）')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f%%'))
    plt.tight_layout()
    return fig


def plot_nav_curves(group_returns: pd.DataFrame, nav_series: pd.Series, metric: str) -> plt.Figure:
    """图2：G1、G10、多空组合净值曲线。"""
    n = len([c for c in group_returns.columns if c.startswith('G')])
    g1_nav = (1 + group_returns['G1'].fillna(0)).cumprod()
    gn_nav = (1 + group_returns[f'G{n}'].fillna(0)).cumprod()
    ls_nav = (1 + group_returns['long_short'].fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(g1_nav.index, g1_nav.values, label='G1（最低组）', color='#d62728')
    ax.plot(gn_nav.index, gn_nav.values, label=f'G{n}（最高组）', color='#2ca02c')
    ax.plot(ls_nav.index, ls_nav.values, label='多空组合', color='#1f77b4', linewidth=2)
    if nav_series is not None:
        ax.plot(nav_series.index, nav_series / nav_series.iloc[0],
                label='多空组合（BacktestBase）', color='#ff7f0e', linestyle='--', linewidth=1.5)
    ax.axhline(1, color='gray', linewidth=0.8, linestyle=':')
    ax.set_ylabel('净值')
    ax.set_title(f'{metric} 因子分组净值曲线')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_ic_series(ic_series: pd.Series, metric: str) -> plt.Figure:
    """图3：IC 时序图 + IC 均值虚线。"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ic_mean = ic_series.mean()
    ax.bar(ic_series.index, ic_series.values,
           color=['#2ca02c' if v > 0 else '#d62728' for v in ic_series.fillna(0)],
           width=15, alpha=0.8)
    ax.axhline(ic_mean, color='navy', linewidth=1.5, linestyle='--',
               label=f'IC 均值 = {ic_mean:.4f}')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('IC')
    ax.set_title(f'{metric} IC 时序图（Spearman Rank IC）')
    ax.legend()
    plt.tight_layout()
    return fig


def build_return_data(
    df_market: Datafeed,
    date_list: list,
    close_metric: str,
) -> pd.DataFrame:
    """
    构建持仓期收益率宽表。

    查询每个调仓日及下一调仓日的收盘价，计算持仓期收益率。

    Returns:
        宽表 DataFrame，index=input_ts（调仓日），columns=code，值=持仓期收益率
    """
    start = min(date_list)
    end = max(date_list)

    print(f"[INFO] 查询日线收盘价：{start.date()} ~ {end.date()}，指标={close_metric}")
    price_long = df_market.query_time_range(
        start=start, end=end, metrics=[close_metric]
    )
    if price_long is None or len(price_long) == 0:
        raise ValueError(f"日线数据为空，请检查表名 {df_market.table_name} 和指标 {close_metric}")

    # 转宽表：index=datetime, columns=code
    price_long['datetime'] = pd.to_datetime(price_long['datetime'])
    price_wide = price_long.pivot_table(
        index='datetime', columns='code', values=close_metric, aggfunc='last'
    )
    price_wide = price_wide.sort_index()

    # 对每个调仓日，找最近收盘价，计算到下一调仓日的持仓期收益
    date_list_sorted = sorted(date_list)
    returns = {}
    for i, ts in enumerate(date_list_sorted[:-1]):
        ts_next = date_list_sorted[i + 1]
        # 找 ts 及 ts_next 最近的交易日价格
        before_ts = price_wide.index[price_wide.index <= pd.Timestamp(ts)]
        before_next = price_wide.index[price_wide.index <= pd.Timestamp(ts_next)]
        if len(before_ts) == 0 or len(before_next) == 0:
            continue
        p0 = price_wide.loc[before_ts[-1]]
        p1 = price_wide.loc[before_next[-1]]
        ret = (p1 - p0) / p0.replace(0, np.nan)
        returns[pd.Timestamp(ts)] = ret

    return_data = pd.DataFrame(returns).T
    return_data.index.name = 'input_ts'
    print(f"[OK] 持仓期收益构建完成：{len(return_data)} 期，{return_data.shape[1]} 只股票")
    return return_data


def run_factor_pipeline(
    metric: str = METRIC,
    date_list: list = None,
    n_groups: int = N_GROUPS,
    save_results: bool = SAVE_RESULTS,
    output_path: Path = OUTPUT_PATH,
):
    """因子端到端验证主函数。"""
    if date_list is None:
        date_list = DATE_LIST

    print(f"\n{'='*60}")
    print(f"  因子验证流水线：metric={metric}, n_groups={n_groups}")
    print(f"  时间范围：{min(date_list).date()} ~ {max(date_list).date()}")
    print(f"{'='*60}\n")

    # ── 1. 初始化数据连接 ──────────────────────────────
    df_fund = Datafeed(TABLE_FUNDAMENTAL)
    df_market = Datafeed(TABLE_MARKET)

    # ── 2. 获取可交易池 ────────────────────────────────
    print("[STEP 1] 获取可交易股票池...")
    date_ranges, code_ranges = get_tradable_pool(date_list)
    print(f"[OK] 可交易池：{len(date_ranges)} 个调仓日，平均每期 {np.mean([len(c) for c in code_ranges]):.0f} 只")

    # ── 3. 查询因子数据 ────────────────────────────────
    print(f"\n[STEP 2] 查询因子数据：{metric}...")
    pre_queried_data = pre_query_characteristic_data(
        date_list=date_list,
        metric=metric,
        date_ranges=date_ranges,
        code_ranges=code_ranges,
        table_name=TABLE_FUNDAMENTAL,
    )
    print(f"[OK] 原始因子数据：{len(pre_queried_data)} 行，"
          f"{pre_queried_data['code'].nunique()} 只股票，"
          f"{pre_queried_data[metric].isna().mean():.1%} 空值率")

    # ── 4. 因子预处理 ──────────────────────────────────
    print(f"\n[STEP 3] 因子预处理（去极值={WINSORIZE_METHOD}，标准化={STANDARDIZE_METHOD}）...")
    cleaned_data = preprocess_factor(
        pre_queried_data=pre_queried_data,
        metric=metric,
        winsorize_method=WINSORIZE_METHOD,
        winsorize_n=WINSORIZE_N,
        standardize_method=STANDARDIZE_METHOD,
        industry_col=INDUSTRY_COL,
        log_mktcap_col=LOG_MKTCAP_COL,
    )
    print(f"[OK] 预处理后：{len(cleaned_data)} 行，"
          f"因子值范围 [{cleaned_data[metric].min():.3f}, {cleaned_data[metric].max():.3f}]")

    # ── 5. 分组打标签 ──────────────────────────────────
    print(f"\n[STEP 4] 单因子分组打标签（{n_groups} 组）...")
    labeled_pool = single_characteristic(
        pre_queried_data=cleaned_data,
        metric=metric,
        quantiles={metric: n_groups},
    )
    label_col = f"{metric}_label"
    label_counts = labeled_pool[label_col].value_counts().sort_index()
    print(f"[OK] 标签分布：\n{label_counts.to_string()}")

    # ── 6. 构建持仓期收益数据 ──────────────────────────
    print(f"\n[STEP 5] 构建持仓期收益数据...")
    return_data = build_return_data(df_market, date_list, CLOSE_METRIC)

    # 构建因子宽表（用于 IC 和 Fama-MacBeth）
    factor_wide = cleaned_data.pivot_table(
        index='input_ts', columns='code', values=metric, aggfunc='last'
    )
    factor_wide.index = pd.to_datetime(factor_wide.index)

    # ── 7. 分组收益统计 ────────────────────────────────
    print(f"\n[STEP 6] 计算各组持仓期收益...")
    group_returns = group_return_summary(labeled_pool, return_data, metric)
    g_cols = [c for c in group_returns.columns if c.startswith('G')]
    group_means = group_returns[g_cols].mean()
    print(f"[OK] 分组收益均值：")
    print(group_means.to_string())
    print(f"     多空组合均值：{group_returns['long_short'].mean():.4f}")

    # ── 8. 多空权重 + 回测 ─────────────────────────────
    print(f"\n[STEP 7] 生成多空权重并回测...")
    params = {
        'factor_key': metric,
        'mode': 'freeplay',
        'long': LONG_LABELS,
        'short': SHORT_LABELS,
    }
    weights = get_single_factor_weight(labeled_pool, params)
    print(f"[OK] 权重矩阵：{weights.shape[0]} 期 × {weights.shape[1]} 只股票")

    bt = BacktestBase(weight=weights, symbol=metric, amount=INITIAL_AMOUNT)
    nav_series = bt.nav
    pa = PortfolioAnalyzer(nav_series)
    stats_dict = {
        '累计收益率': f"{pa.total_return():.2%}",
        '年化收益率': f"{pa.annualized_return():.2%}",
        '年化波动率': f"{pa.annualized_volatility():.2%}",
        '夏普比率': f"{pa.sharpe_ratio():.3f}",
        '最大回撤': f"{pa.max_drawdown():.2%}",
        'Calmar比率': f"{pa.calmar_ratio():.3f}",
    }
    print("[OK] 多空组合绩效：")
    for k, v in stats_dict.items():
        print(f"     {k}: {v}")

    # ── 9. IC 分析 ─────────────────────────────────────
    print(f"\n[STEP 8] IC 分析...")
    ic_series = calc_ic(factor_wide, return_data, method='spearman')
    ic_summary = summarize_ic(ic_series)
    print("[OK] IC 统计：")
    for k, v in ic_summary.items():
        if isinstance(v, float):
            print(f"     {k}: {v:.4f}")

    # ── 10. Fama-MacBeth 回归 ──────────────────────────
    print(f"\n[STEP 9] Fama-MacBeth 截面回归...")
    fm_result = fama_macbeth({metric: factor_wide}, return_data)
    print("[OK] Fama-MacBeth 结果：")
    print(fm_result.to_string())

    # 合并 factor_stats
    ic_df = pd.DataFrame([ic_summary], index=['IC统计'])
    fm_df = fm_result.reset_index().rename(columns={'factor': '因子'}).set_index('因子')
    factor_stats = pd.concat([ic_df.T.rename(columns={'IC统计': 'IC检验'}), fm_df.T], axis=1)

    # ── 11. 可视化 ─────────────────────────────────────
    print(f"\n[STEP 10] 生成可视化图表...")
    fig1 = plot_group_returns(group_returns, metric)
    fig2 = plot_nav_curves(group_returns, nav_series, metric)
    fig3 = plot_ic_series(ic_series, metric)
    print("[OK] 3张图表已生成")

    # ── 12. 导出 Excel ─────────────────────────────────
    if save_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n[STEP 11] 导出 Excel → {output_path}")

        # labeled_pool：重置 MultiIndex 方便阅读
        labeled_pool_export = labeled_pool.reset_index()

        # weights：转 long 格式更易读（可选）
        weights_export = weights.copy()

        # nav：多空组合净值
        nav_export = nav_series.rename('nav').to_frame()

        # stats
        stats_export = pd.DataFrame(
            list(stats_dict.items()), columns=['指标', '值']
        ).set_index('指标')

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            labeled_pool_export.to_excel(writer, sheet_name='labeled_pool', index=False)
            weights_export.to_excel(writer, sheet_name='weights')
            group_returns.to_excel(writer, sheet_name='group_returns')
            nav_export.to_excel(writer, sheet_name='nav')
            stats_export.to_excel(writer, sheet_name='stats')
            factor_stats.to_excel(writer, sheet_name='factor_stats')

        # 嵌入图片
        wb = openpyxl.load_workbook(output_path)
        img1 = _fig_to_image(fig1)
        img2 = _fig_to_image(fig2)
        img3 = _fig_to_image(fig3)

        # 图1 → group_returns sheet
        ws = wb['group_returns']
        img1.anchor = f'A{len(group_returns) + 5}'
        ws.add_image(img1)

        # 图2 → nav sheet
        ws = wb['nav']
        img2.anchor = f'A{len(nav_export) + 5}'
        ws.add_image(img2)

        # 图3 → factor_stats sheet
        ws = wb['factor_stats']
        img3.anchor = f'A{len(factor_stats) + 5}'
        ws.add_image(img3)

        wb.save(output_path)
        print(f"[OK] Excel 已保存（6 sheets + 3 图表嵌入）：{output_path}")

    plt.close('all')
    print(f"\n{'='*60}")
    print(f"  验证流水线完成！")
    print(f"{'='*60}\n")

    return {
        'labeled_pool': labeled_pool,
        'weights': weights,
        'group_returns': group_returns,
        'nav': nav_series,
        'ic': ic_series,
        'ic_summary': ic_summary,
        'fm_result': fm_result,
    }


if __name__ == "__main__":
    run_factor_pipeline()
