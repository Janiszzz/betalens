"""
策略评价看板页

数据来源：上传 bt.dump_to_excel() 导出的 xlsx（含 nav/weight/daily_position_value 等）。
展示：指标卡 + plotly 交互图 + 明细表 tabs + Excel/HTML 下载。
"""
import io
import tempfile
import os
import streamlit as st
import pandas as pd

from betalens.analyst import Analyst
from betalens.analyst import metrics as M


_PCT_KEYS = {
    '累计收益', '年化收益', '年化波动率', '最大回撤', '痛苦指数', '下行偏差',
    'VaR(95%)', 'CVaR(95%)', '单边换手率(年化)', '平均单边换手', 'Alpha',
    '跟踪误差', '相对基准胜率', '前5集中度(均值)',
}


def render():
    st.header("策略评价")
    st.caption("上传回测导出的 Excel（bt.dump_to_excel），一键生成绩效报告与交互图表")

    file = st.file_uploader(
        "上传回测结果（Excel）", type=["xlsx"], key="pf_file",
        help="需包含 nav sheet；含 weight/daily_position_value/daily_pnl 时可算持仓与归因指标",
    )

    bench_file = st.file_uploader(
        "上传基准净值（可选，Excel，需 nav sheet）", type=["xlsx"], key="pf_bench",
    )

    name = st.text_input("组合名称", value="组合", key="pf_name")

    if file is None:
        st.info("请上传回测导出的 Excel 文件。")
        return

    if st.button("▶ 运行评价", key="pf_run", type="primary"):
        _run(file, bench_file, name)

    if "pf_analyst_path" not in st.session_state:
        return

    _show_results()


def _run(file, bench_file, name):
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

            analyst = Analyst.from_excel(tmp.name, benchmark=benchmark, name=name)
            # 缓存计算结果到 session（Analyst 含连接对象不可序列化，存中间产物）
            st.session_state["pf_summary"] = analyst.an.summary_grouped()
            st.session_state["pf_ifigs"] = analyst.interactive_plots()
            st.session_state["pf_top"] = analyst.top_holdings_df()
            st.session_state["pf_contrib"] = analyst.contribution_df()
            st.session_state["pf_trade"] = analyst.trade_pnl_df()
            st.session_state["pf_monthly"] = analyst.monthly_table()
            # 导出字节
            xls_path = tmp.name + ".report.xlsx"
            analyst.to_excel(xls_path)
            with open(xls_path, "rb") as f:
                st.session_state["pf_excel"] = f.read()
            html_path = tmp.name + ".report.html"
            analyst.to_html(html_path)
            with open(html_path, "rb") as f:
                st.session_state["pf_html"] = f.read()
            st.session_state["pf_analyst_path"] = tmp.name
            for p in (xls_path, html_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        except Exception as e:
            st.error(f"评价失败: {e}")
            return


def _show_results():
    summary = st.session_state["pf_summary"]

    # ── 指标卡 ───────────────────────────────────────────────────────────
    st.subheader("核心指标")
    key_metrics = []
    flat = {k: v for items in summary.values() for k, v in items.items()}
    for k in ['累计收益', '年化收益', '夏普比率', '最大回撤', '卡玛比率', '索提诺比率']:
        if k in flat:
            v = flat[k]
            disp = f"{v:.2%}" if k in _PCT_KEYS else f"{v:.3f}"
            key_metrics.append((k, disp))
    cols = st.columns(len(key_metrics) or 1)
    for col, (k, disp) in zip(cols, key_metrics):
        col.metric(k, disp)

    st.divider()

    # ── 交互图 ───────────────────────────────────────────────────────────
    ifigs = st.session_state["pf_ifigs"]
    if '净值曲线' in ifigs:
        st.plotly_chart(ifigs['净值曲线'], use_container_width=True)
    c1, c2 = st.columns(2)
    if '回撤曲线' in ifigs:
        c1.plotly_chart(ifigs['回撤曲线'], use_container_width=True)
    if '滚动胜率' in ifigs:
        c2.plotly_chart(ifigs['滚动胜率'], use_container_width=True)
    c3, c4 = st.columns(2)
    if '滚动夏普' in ifigs:
        c3.plotly_chart(ifigs['滚动夏普'], use_container_width=True)
    if '权重堆积' in ifigs:
        c4.plotly_chart(ifigs['权重堆积'], use_container_width=True)
    if '收益贡献' in ifigs:
        st.plotly_chart(ifigs['收益贡献'], use_container_width=True)
    if '月度收益' in ifigs:
        st.plotly_chart(ifigs['月度收益'], use_container_width=True)

    st.divider()

    # ── 明细表 ───────────────────────────────────────────────────────────
    st.subheader("详细数据")
    tabs = st.tabs(["指标汇总", "最频繁持仓", "收益贡献", "逐笔盈亏", "月度收益"])

    with tabs[0]:
        for group, items in summary.items():
            if not items:
                continue
            st.caption(group)
            rows = [{'指标': k,
                     '数值': (f"{v:.2%}" if k in _PCT_KEYS else f"{v:.4f}")
                     if isinstance(v, float) else v}
                    for k, v in items.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tabs[1]:
        top = st.session_state.get("pf_top")
        if top is not None and not top.empty:
            st.dataframe(top, use_container_width=True)
        else:
            st.info("无持仓数据（上传含 weight sheet 的回测结果可用）")

    with tabs[2]:
        contrib = st.session_state.get("pf_contrib")
        if contrib is not None and not contrib.empty:
            st.dataframe(contrib, use_container_width=True)
        else:
            st.info("无损益数据（需 daily_pnl sheet）")

    with tabs[3]:
        trade = st.session_state.get("pf_trade")
        if trade is not None and not trade.empty:
            st.dataframe(trade, use_container_width=True)
        else:
            st.info("无调仓记录（需 rebalance_log sheet）")

    with tabs[4]:
        monthly = st.session_state.get("pf_monthly")
        if monthly is not None and not monthly.empty:
            st.dataframe(monthly.style.format("{:.2%}", na_rep="-"),
                         use_container_width=True)
        else:
            st.info("数据不足以生成月度收益表")

    # ── 下载 ─────────────────────────────────────────────────────────────
    st.divider()
    d1, d2 = st.columns(2)
    if "pf_excel" in st.session_state:
        d1.download_button(
            "📥 下载 Excel 报告", data=st.session_state["pf_excel"],
            file_name="portfolio_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if "pf_html" in st.session_state:
        d2.download_button(
            "📥 下载 HTML 报告（交互图）", data=st.session_state["pf_html"],
            file_name="portfolio_report.html", mime="text/html",
        )
