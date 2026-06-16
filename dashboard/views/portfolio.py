"""
策略评价看板页 — 4 个子页面：收益概述 / 交易详情 / 每日持仓&收益 / 日志输出
"""
import contextlib
import io
import tempfile
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from betalens.analyst import Analyst
from betalens.analyst import metrics as M


_PCT_KEYS = {
    '累计收益', '年化收益', '年化波动率', '最大回撤', '痛苦指数', '下行偏差',
    'VaR(95%)', 'CVaR(95%)', '单边换手率(年化)', '平均单边换手', 'Alpha',
    '跟踪误差', '相对基准胜率', '前5集中度(均值)',
}

# 收益概述优先展示的指标顺序（与用户示例对齐）
_OVERVIEW_KEYS = [
    '累计收益', '年化收益', '超额收益', '基准收益',
    'Alpha', 'Beta', '夏普比率', '胜率', '盈亏比',
    '最大回撤', '索提诺比率', '日均超额收益', '超额收益最大回撤',
    '超额收益夏普比率', '日胜率', '盈利次数', '亏损次数',
    '信息比率', '年化波动率', '基准波动率', '最大回撤区间',
]

_SUB_PAGES = ["收益概述", "交易详情", "每日持仓&收益", "日志输出"]


def render():
    # ── 侧边栏：上传区 + 子导航 ───────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        if "pf_analyst_ready" not in st.session_state:
            _sidebar_upload()
        else:
            sub = st.radio(
                "", _SUB_PAGES,
                label_visibility="collapsed",
                key="pf_sub",
            )
            st.divider()
            if st.button("← 重新上传", key="pf_reset"):
                for k in list(st.session_state.keys()):
                    if k.startswith("pf_"):
                        del st.session_state[k]
                st.rerun()

    if "pf_analyst_ready" not in st.session_state:
        st.info("请在左侧上传回测导出的 Excel 文件并运行评价。")
        return

    sub = st.session_state.get("pf_sub", "收益概述")
    if sub == "收益概述":
        _page_overview()
    elif sub == "交易详情":
        _page_trades()
    elif sub == "每日持仓&收益":
        _page_positions()
    else:
        _page_logs()


# ── 侧边栏上传区 ──────────────────────────────────────────────────────────────

def _sidebar_upload():
    file = st.file_uploader("回测结果 Excel", type=["xlsx"], key="pf_file")
    bench_file = st.file_uploader("基准净值（可选）", type=["xlsx"], key="pf_bench")
    name = st.text_input("组合名称", value="组合", key="pf_name")
    if file and st.button("▶ 运行评价", key="pf_run", type="primary"):
        _run(file, bench_file, name)
        st.rerun()


# ── 运行评价，捕获日志 ────────────────────────────────────────────────────────

def _run(file, bench_file, name):
    buf = io.StringIO()
    with st.spinner("正在解析与计算..."):
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
            tmp.write(file.getbuffer())
            tmp.close()

            benchmark = None
            if bench_file is not None:
                bdf = pd.read_excel(bench_file, sheet_name="nav", index_col=0)
                bdf.index = pd.to_datetime(bdf.index)
                benchmark = bdf.iloc[:, 0]

            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                analyst = Analyst.from_excel(tmp.name, benchmark=benchmark, name=name)
                st.session_state["pf_summary"] = analyst.an.summary_grouped()
                st.session_state["pf_ifigs"] = analyst.interactive_plots()
                st.session_state["pf_top"] = analyst.top_holdings_df()
                st.session_state["pf_contrib"] = analyst.contribution_df()
                st.session_state["pf_trade"] = analyst.trade_pnl_df()
                st.session_state["pf_monthly"] = analyst.monthly_table()

            # 原始 Excel sheets（用于交易详情和持仓页）
            st.session_state["pf_excel_path"] = tmp.name
            _cache_raw_sheets(tmp.name)

            # 导出报告
            xls_path = tmp.name + ".report.xlsx"
            analyst.to_excel(xls_path)
            with open(xls_path, "rb") as f:
                st.session_state["pf_excel_report"] = f.read()
            html_path = tmp.name + ".report.html"
            analyst.to_html(html_path)
            with open(html_path, "rb") as f:
                st.session_state["pf_html"] = f.read()
            for p in (xls_path, html_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

            st.session_state["pf_log"] = buf.getvalue()
            st.session_state["pf_analyst_ready"] = True
            st.session_state["pf_sub"] = "收益概述"
        except Exception as e:
            st.error(f"评价失败: {e}")
            st.session_state["pf_log"] = buf.getvalue()


def _cache_raw_sheets(path: str):
    xl = pd.ExcelFile(path)
    sheets = xl.sheet_names
    if "rebalance_log" in sheets:
        st.session_state["pf_rebalance"] = pd.read_excel(path, sheet_name="rebalance_log")
    if "daily_position_value" in sheets:
        st.session_state["pf_pos_value"] = pd.read_excel(path, sheet_name="daily_position_value", index_col=0)
    if "daily_pnl" in sheets:
        st.session_state["pf_daily_pnl"] = pd.read_excel(path, sheet_name="daily_pnl", index_col=0)
    if "nav" in sheets:
        st.session_state["pf_nav"] = pd.read_excel(path, sheet_name="nav", index_col=0)


# ── 子页面 1：收益概述 ────────────────────────────────────────────────────────

def _page_overview():
    summary = st.session_state["pf_summary"]
    flat = {k: v for items in summary.values() for k, v in items.items()}

    # 指标卡（按优先列表顺序，最多4列一行）
    display_keys = [k for k in _OVERVIEW_KEYS if k in flat]
    display_keys += [k for k in flat if k not in _OVERVIEW_KEYS]

    # 分行展示，每行7个
    N = 7
    rows = [display_keys[i:i+N] for i in range(0, len(display_keys), N)]
    for row_keys in rows:
        cols = st.columns(len(row_keys))
        for col, k in zip(cols, row_keys):
            v = flat[k]
            if isinstance(v, float):
                disp = f"{v:.2%}" if k in _PCT_KEYS else f"{v:.3f}"
            else:
                disp = str(v)
            col.metric(k, disp)

    st.divider()

    # 时间范围滑动条（基于 nav）
    nav_df = st.session_state.get("pf_nav")
    date_range = None
    if nav_df is not None and not nav_df.empty:
        nav_df.index = pd.to_datetime(nav_df.index)
        min_d, max_d = nav_df.index.min().date(), nav_df.index.max().date()
        date_range = st.slider(
            "时间区间", min_value=min_d, max_value=max_d,
            value=(min_d, max_d), key="pf_date_range",
            format="YYYY/MM/DD",
        )

    # 净值曲线
    ifigs = st.session_state["pf_ifigs"]
    if '净值曲线' in ifigs:
        fig = ifigs['净值曲线']
        if date_range:
            fig = _filter_fig_by_date(fig, date_range)
        st.plotly_chart(fig, use_container_width=True)

    # 每日盈亏柱状图
    pnl_df = st.session_state.get("pf_daily_pnl")
    if pnl_df is not None and not pnl_df.empty:
        pnl_df.index = pd.to_datetime(pnl_df.index)
        # 取第一列或名为 pnl/profit 的列作为每日盈亏
        pnl_col = None
        for c in pnl_df.columns:
            if any(kw in str(c).lower() for kw in ["pnl", "profit", "盈亏", "损益"]):
                pnl_col = c
                break
        if pnl_col is None and len(pnl_df.columns) > 0:
            pnl_col = pnl_df.columns[0]
        if pnl_col:
            pnl_series = pnl_df[pnl_col]
            if date_range:
                pnl_series = pnl_series[
                    (pnl_series.index.date >= date_range[0]) &
                    (pnl_series.index.date <= date_range[1])
                ]
            colors = ["#ef4444" if v < 0 else "#22c55e" for v in pnl_series.values]
            bar_fig = go.Figure(go.Bar(
                x=pnl_series.index, y=pnl_series.values,
                marker_color=colors,
                name="每日盈亏",
            ))
            bar_fig.update_layout(
                title="每日盈亏", height=250,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(bar_fig, use_container_width=True)

    # 下载按钮
    st.divider()
    d1, d2 = st.columns(2)
    if "pf_excel_report" in st.session_state:
        d1.download_button("📥 下载 Excel 报告", data=st.session_state["pf_excel_report"],
                           file_name="portfolio_report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if "pf_html" in st.session_state:
        d2.download_button("📥 下载 HTML 报告", data=st.session_state["pf_html"],
                           file_name="portfolio_report.html", mime="text/html")


def _filter_fig_by_date(fig, date_range):
    """按日期范围裁剪 plotly figure 的 x 轴"""
    try:
        fig.update_xaxes(range=[str(date_range[0]), str(date_range[1])])
    except Exception:
        pass
    return fig


# ── 子页面 2：交易详情 ────────────────────────────────────────────────────────

def _page_trades():
    trade = st.session_state.get("pf_rebalance")
    if trade is None:
        # 尝试从原始 Excel 读
        path = st.session_state.get("pf_excel_path")
        if path:
            try:
                trade = pd.read_excel(path, sheet_name="rebalance_log")
                st.session_state["pf_rebalance"] = trade
            except Exception:
                pass
    if trade is None or (hasattr(trade, "empty") and trade.empty):
        st.info("无交易记录（需 rebalance_log sheet）")
        return

    # 搜索过滤
    search = st.text_input("搜索（代码/品种）", key="pf_trade_search", placeholder="输入关键字过滤")
    df = trade.copy()
    if search:
        mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False)).any(axis=1)
        df = df[mask]
    st.caption(f"共 {len(df)} 条记录")
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── 子页面 3：每日持仓&收益 ───────────────────────────────────────────────────

def _page_positions():
    pos_df = st.session_state.get("pf_pos_value")
    pnl_df = st.session_state.get("pf_daily_pnl")

    if pos_df is None and pnl_df is None:
        st.info("无持仓数据（需 daily_position_value 或 daily_pnl sheet）")
        return

    # 合并可用数据
    if pos_df is not None:
        pos_df.index = pd.to_datetime(pos_df.index)
    if pnl_df is not None:
        pnl_df.index = pd.to_datetime(pnl_df.index)

    # 日期列表
    dates = sorted(set(
        (pos_df.index.tolist() if pos_df is not None else []) +
        (pnl_df.index.tolist() if pnl_df is not None else [])
    ), reverse=True)

    if not dates:
        st.info("无日期数据")
        return

    col1, col2 = st.columns([2, 3])
    with col1:
        sel_date = st.selectbox(
            "选择日期", dates,
            format_func=lambda d: d.strftime("%Y/%m/%d"),
            key="pf_pos_date",
        )
    with col2:
        search = st.text_input("搜索品种", key="pf_pos_search", placeholder="输入代码或名称过滤")

    # 展示当日持仓
    frames = []
    if pos_df is not None and sel_date in pos_df.index:
        row = pos_df.loc[sel_date]
        if isinstance(row, pd.Series):
            row = row.to_frame().T
        frames.append(("持仓价值", row))
    if pnl_df is not None and sel_date in pnl_df.index:
        row = pnl_df.loc[sel_date]
        if isinstance(row, pd.Series):
            row = row.to_frame().T
        frames.append(("每日盈亏", row))

    if not frames:
        st.info(f"{sel_date.strftime('%Y/%m/%d')} 无数据")
        return

    for label, df in frames:
        st.caption(label)
        show = df.copy()
        if search:
            mask = show.astype(str).apply(lambda c: c.str.contains(search, case=False)).any(axis=1)
            show = show[mask]
        st.dataframe(show, use_container_width=True)


# ── 子页面 4：日志输出 ────────────────────────────────────────────────────────

def _page_logs():
    log = st.session_state.get("pf_log", "")
    if not log:
        st.info("暂无日志（运行评价时会自动捕获控制台输出）")
        return
    st.code(log, language=None)
