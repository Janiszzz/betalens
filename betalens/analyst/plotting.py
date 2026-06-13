"""
绘图层

两套并行，各司其职：
- matplotlib：静态 PNG bytes，用于 CLI 预览 / Excel 嵌入 / 报告图片导出
  （复刻 factor/stats.py 的 _fig_to_bytes 范式，simhei 中文字体）
- plotly：交互图 Figure，用于 dashboard 与独立 HTML 报告

所有接收持仓/标的的函数都支持 name_map，把 code 显示为「中文名(代码)」。
"""
import io as _io
import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['simhei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from . import metrics as M
from .naming import label


# ── matplotlib → PNG bytes ──────────────────────────────────────────────────

def _fig_to_bytes(fig) -> bytes:
    buf = _io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_nav(nav: pd.Series, benchmark: pd.Series = None,
             title: str = '净值曲线') -> bytes:
    """净值曲线图（可叠加基准）"""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(nav.index, nav.values, color='#1f77b4', linewidth=1.5, label='组合')
    if benchmark is not None:
        b = benchmark.reindex(nav.index).ffill()
        b = b / b.iloc[0] * nav.iloc[0]
        ax.plot(b.index, b.values, color='#ff7f0e', linewidth=1.2,
                linestyle='--', label='基准')
    ax.set_title(title, fontsize=13)
    ax.set_ylabel('净值')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _fig_to_bytes(fig)


def plot_drawdown(nav: pd.Series, title: str = '回撤曲线') -> bytes:
    """回撤面积图"""
    dd = M._drawdown_series(nav) * -100
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(dd.index, dd.values, 0, color='#d62728', alpha=0.4)
    ax.plot(dd.index, dd.values, color='#d62728', linewidth=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel('回撤 (%)')
    ax.grid(alpha=0.3)
    return _fig_to_bytes(fig)


def plot_rolling_metric(series: pd.Series, title: str, ylabel: str,
                        color: str = '#2ca02c') -> bytes:
    """通用滚动指标折线图（滚动胜率/夏普/回撤等）"""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(series.index, series.values, color=color, linewidth=1)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return _fig_to_bytes(fig)


def plot_contribution_bar(contrib: pd.DataFrame, name_map: dict = None,
                          title: str = '收益贡献 Top') -> bytes:
    """收益贡献柱状图（横向，按 pnl 排序，中文名标签）"""
    name_map = name_map or {}
    labels = [label(c, name_map) for c in contrib.index]
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in contrib['pnl']]
    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.4)))
    ax.barh(labels, contrib['pnl'].values, color=colors)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('累计损益')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    return _fig_to_bytes(fig)


def plot_weight_concentration(hhi: pd.Series, count: pd.Series,
                              weight: "pd.DataFrame | None" = None,
                              name_map: "dict | None" = None,
                              top: int = 8, max_codes: int = 15, max_periods: int = 12,
                              title: str = '权重堆积与持仓数') -> bytes:
    """HHI 集中度 + 持仓数双轴图；传入 weight 时在下方附时序持仓权重表一览"""
    name_map = name_map or {}

    def _draw_dual(ax1):
        ax1.plot(hhi.index, hhi.values, color='#9467bd', linewidth=1, label='HHI 集中度')
        ax1.set_ylabel('HHI', color='#9467bd')
        ax1.tick_params(axis='y', labelcolor='#9467bd')
        ax2 = ax1.twinx()
        ax2.plot(count.index, count.values, color='#8c564b', linewidth=1,
                 alpha=0.6, label='持仓数')
        ax2.set_ylabel('持仓数', color='#8c564b')
        ax2.tick_params(axis='y', labelcolor='#8c564b')
        ax1.set_title(title, fontsize=13)
        ax1.grid(alpha=0.3)

    if weight is None:
        fig, ax1 = plt.subplots(figsize=(11, 3.5))
        _draw_dual(ax1)
        return _fig_to_bytes(fig)

    # 时序持仓表：选标（每期前 top 大的并集，按累计权重排序截断），采样日期
    w = weight.sort_index().fillna(0.0)
    if 'cash' in w.columns:
        w = w.drop(columns='cash')
    selected = set()
    for _, row in w.iterrows():
        nz = row[row > 0]
        if len(nz):
            selected.update(nz.nlargest(top).index)
    order = w[list(selected)].sum().sort_values(ascending=False).index.tolist()[:max_codes]
    if len(w) > max_periods:
        pos = np.unique(np.linspace(0, len(w) - 1, max_periods).round().astype(int))
        w = w.iloc[pos]
    mat = w[order].T  # 行=标的，列=日期

    dates = [d.strftime('%y/%m/%d') if hasattr(d, 'strftime') else str(d) for d in mat.columns]
    rows = [label(c, name_map) for c in order]
    vmax = mat.values.max() or 1.0
    greens = plt.get_cmap('Greens')
    cell_text = [[f'{v:.0%}' if v > 0 else '' for v in mat.loc[c].values] for c in order]
    cell_colours = [[greens(0.15 + 0.6 * v / vmax) if v > 0 else 'white'
                     for v in mat.loc[c].values] for c in order]

    table_h = max(1.5, 0.32 * len(order) + 0.6)
    fig = plt.figure(figsize=(11, 3.5 + table_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.5, table_h], hspace=0.35)
    _draw_dual(fig.add_subplot(gs[0]))
    axt = fig.add_subplot(gs[1])
    axt.axis('off')
    axt.set_title('时序持仓权重一览', fontsize=11, pad=4)
    tbl = axt.table(cellText=cell_text, cellColours=cell_colours,
                    rowLabels=rows, colLabels=dates, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.2)
    return _fig_to_bytes(fig)


def plot_monthly_heatmap(table: pd.DataFrame, title: str = '月度收益热力表') -> bytes:
    """月度收益热力图"""
    if table.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, '数据不足', ha='center', va='center')
        ax.axis('off')
        return _fig_to_bytes(fig)
    data = table.drop(columns=['全年'], errors='ignore')
    fig, ax = plt.subplots(figsize=(11, max(2.5, len(data) * 0.5)))
    im = ax.imshow(data.values, cmap='RdYlGn', aspect='auto',
                   vmin=-np.nanmax(np.abs(data.values)),
                   vmax=np.nanmax(np.abs(data.values)))
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([f'{m}月' for m in data.columns])
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            v = data.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.1%}', ha='center', va='center', fontsize=7)
    ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.6)
    return _fig_to_bytes(fig)


# ── plotly → 交互 Figure ────────────────────────────────────────────────────

def _import_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "交互图需要 plotly，请先安装: pip install plotly"
        ) from e


_PLOTLY_FONT = dict(family="SimHei, Microsoft YaHei, sans-serif")


def ip_nav(nav: pd.Series, benchmark: pd.Series = None,
           title: str = '净值曲线') -> "object":
    """交互净值曲线（plotly）"""
    go = _import_plotly()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name='组合',
                             line=dict(color='#1f77b4', width=2)))
    if benchmark is not None:
        b = benchmark.reindex(nav.index).ffill()
        b = b / b.iloc[0] * nav.iloc[0]
        fig.add_trace(go.Scatter(x=b.index, y=b.values, name='基准',
                                 line=dict(color='#ff7f0e', width=1.5, dash='dash')))
    fig.update_layout(title=title, font=_PLOTLY_FONT, hovermode='x unified',
                      yaxis_title='净值', height=420)
    return fig


def ip_drawdown(nav: pd.Series, title: str = '回撤曲线') -> "object":
    go = _import_plotly()
    dd = M._drawdown_series(nav) * -100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill='tozeroy',
                             line=dict(color='#d62728', width=1), name='回撤'))
    fig.update_layout(title=title, font=_PLOTLY_FONT, hovermode='x unified',
                      yaxis_title='回撤 (%)', height=340)
    return fig


def ip_rolling(series: pd.Series, title: str, ylabel: str,
               color: str = '#2ca02c') -> "object":
    go = _import_plotly()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values,
                             line=dict(color=color, width=1.2), name=ylabel))
    fig.update_layout(title=title, font=_PLOTLY_FONT, hovermode='x unified',
                      yaxis_title=ylabel, height=340)
    return fig


def ip_contribution(contrib: pd.DataFrame, name_map: dict = None,
                    title: str = '收益贡献 Top') -> "object":
    go = _import_plotly()
    name_map = name_map or {}
    labels = [label(c, name_map) for c in contrib.index][::-1]
    vals = contrib['pnl'].values[::-1]
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation='h',
                           marker_color=colors))
    fig.update_layout(title=title, font=_PLOTLY_FONT,
                      xaxis_title='累计损益',
                      height=max(340, len(labels) * 26))
    return fig


def ip_weight_area(daily_position_value: pd.DataFrame, name_map: dict = None,
                   top: int = 10, max_codes: int = 25,
                   title: str = '持仓权重堆积') -> "object":
    """
    持仓权重堆积面积图。

    选标逻辑（按标的，并集法）：取每个时点权重前 ``top`` 大标的的并集作为
    显示标的——低换手时就是那固定十几只全显示；高换手时各期轮动的主力都能入选。
    并集仍超过 ``max_codes`` 时，按各标的的峰值单期权重保留最重要的，其余与
    未入选标的一并归入「其他」，避免色块过多显示不过来。
    """
    go = _import_plotly()
    name_map = name_map or {}
    dpv = daily_position_value.copy()
    weights = dpv.div(dpv.sum(axis=1), axis=0).fillna(0.0)
    stock_cols = [c for c in weights.columns if c != 'cash']
    stock_w = weights[stock_cols]

    # 每日前 top 大（非零）标的的并集
    selected = set()
    for _, row in stock_w.iterrows():
        nz = row[row > 0]
        if len(nz):
            selected.update(nz.nlargest(top).index)
    selected = list(selected)

    # 防爆：并集过大时按峰值单期权重保留最重要的 max_codes 只
    if len(selected) > max_codes:
        peak = stock_w[selected].max().sort_values(ascending=False)
        selected = list(peak.index[:max_codes])

    # 按累计市值排序，让堆叠顺序稳定
    order = stock_w[selected].sum().sort_values(ascending=False).index.tolist() \
        if selected else []
    others = [c for c in stock_cols if c not in selected]

    plot_df = weights[order].copy()
    if others:
        plot_df['其他'] = weights[others].sum(axis=1)
    if 'cash' in weights.columns:
        plot_df['现金'] = weights['cash']
    fig = go.Figure()
    for col in plot_df.columns:
        disp = col if col in ('其他', '现金') else label(col, name_map)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col],
                                 stackgroup='one', name=disp, mode='lines'))
    fig.update_layout(title=title, font=_PLOTLY_FONT, hovermode='x unified',
                      yaxis_title='权重', yaxis_range=[0, 1], height=420)
    return fig


def ip_monthly_heatmap(table: pd.DataFrame, title: str = '月度收益热力表') -> "object":
    go = _import_plotly()
    if table.empty:
        return go.Figure()
    data = table.drop(columns=['全年'], errors='ignore')
    fig = go.Figure(go.Heatmap(
        z=data.values, x=[f'{m}月' for m in data.columns],
        y=[str(y) for y in data.index],
        colorscale='RdYlGn', zmid=0,
        text=[[f'{v:.1%}' if not pd.isna(v) else '' for v in row] for row in data.values],
        texttemplate='%{text}', textfont={'size': 9},
    ))
    fig.update_layout(title=title, font=_PLOTLY_FONT, height=max(300, len(data) * 40))
    return fig


def fig_to_html_div(fig) -> str:
    """plotly Figure → 可嵌入的 HTML div 片段（首图带 plotly.js CDN）"""
    return fig.to_html(full_html=False, include_plotlyjs='cdn')
