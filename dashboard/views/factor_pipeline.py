"""
因子流水线页

直接调用 test/factor/run_factor_pipeline.py 的端到端流水线：
数据库 → 因子预处理 → 分组 → 多空回测 → IC/Fama-MacBeth 检验。

参数面板配置因子与回测细节，一键运行后展示绩效、图表与明细表，
并复用 Analyst 生成完整策略评价报告，支持 Excel/HTML 下载。
"""
import importlib.util
import tempfile
import os
from pathlib import Path

import streamlit as st
import pandas as pd

from betalens.analyst import Analyst


# ── 动态加载 test/factor/run_factor_pipeline.py（test 非包，按路径导入）──────
_PIPELINE_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "test" / "factor" / "run_factor_pipeline.py"
)


@st.cache_resource
def _load_pipeline():
    spec = importlib.util.spec_from_file_location("run_factor_pipeline", _PIPELINE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_factor_pipeline


def render():
    st.header("因子流水线")
    st.caption("从数据库取因子 → 预处理 → 分组多空回测 → IC/Fama-MacBeth 检验，一键端到端")

    if not _PIPELINE_FILE.exists():
        st.error(f"未找到流水线脚本：{_PIPELINE_FILE}")
        return

    # ── 参数面板 ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        metric = st.text_input("因子名称（数据库列名）", value="ROE", key="fp_metric")
        n_groups = st.number_input("分组数", value=10, min_value=2, max_value=50, key="fp_ng")
    with c2:
        start = st.date_input("起始日期", value=pd.Timestamp("2020-01-31"), key="fp_start")
        end = st.date_input("结束日期", value=pd.Timestamp("2024-12-31"), key="fp_end")
    with c3:
        freq = st.selectbox("调仓频率", ["ME", "QE", "W", "YE"],
                            format_func=lambda f: {"ME": "月末", "QE": "季末",
                                                   "W": "每周", "YE": "年末"}[f],
                            key="fp_freq")
        initial_amount = st.number_input("初始资金", value=1_000_000, step=100_000, key="fp_amt")

    with st.expander("⚙️ 高级参数", expanded=False):
        a1, a2, a3 = st.columns(3)
        with a1:
            table_fundamental = st.text_input("财务指标表", value="fundamental_data", key="fp_tf")
            table_market = st.text_input("日行情表", value="daily_market", key="fp_tm")
            close_metric = st.text_input("收盘价列名", value="收盘价", key="fp_close")
        with a2:
            winsorize_method = st.selectbox("去极值方法", ["mad", "quantile", "sigma"], key="fp_win")
            winsorize_n = st.number_input("去极值倍数", value=3.0, step=0.5, key="fp_winn")
            standardize_method = st.selectbox("标准化方法", ["zscore", "rank", "minmax"], key="fp_std")
        with a3:
            long_label = st.number_input("做多分组（最高组）", value=int(n_groups),
                                         min_value=1, max_value=int(n_groups), key="fp_long")
            short_label = st.number_input("做空分组（最低组）", value=1,
                                          min_value=1, max_value=int(n_groups), key="fp_short")
            save_results = st.checkbox("同时落盘 Excel（脚本默认输出目录）", value=False, key="fp_save")

    name = st.text_input("组合名称", value=f"{metric} 多空", key="fp_name")

    if st.button("▶ 运行流水线", key="fp_run", type="primary"):
        _run(
            metric=metric, n_groups=int(n_groups), start=start, end=end, freq=freq,
            initial_amount=float(initial_amount), table_fundamental=table_fundamental,
            table_market=table_market, close_metric=close_metric,
            winsorize_method=winsorize_method, winsorize_n=float(winsorize_n),
            standardize_method=standardize_method, long_label=int(long_label),
            short_label=int(short_label), save_results=save_results, name=name,
        )

    if "fp_results" not in st.session_state:
        st.info("配置参数后点击「运行流水线」。该过程会连接数据库并执行回测，耗时取决于股票池与期数。")
        return

    _show_results()


def _run(*, metric, n_groups, start, end, freq, initial_amount, table_fundamental,
         table_market, close_metric, winsorize_method, winsorize_n, standardize_method,
         long_label, short_label, save_results, name):
    date_list = pd.date_range(str(start), str(end), freq=freq).tolist()
    if len(date_list) < 2:
        st.error("日期范围内调仓日不足 2 个，请放宽范围或调整频率。")
        return

    run_factor_pipeline = _load_pipeline()
    with st.spinner(f"正在运行因子流水线（{len(date_list)} 个调仓日）..."):
        try:
            results = run_factor_pipeline(
                metric=metric,
                date_list=date_list,
                n_groups=n_groups,
                table_fundamental=table_fundamental,
                table_market=table_market,
                close_metric=close_metric,
                winsorize_method=winsorize_method,
                winsorize_n=winsorize_n,
                standardize_method=standardize_method,
                long_labels=[long_label],
                short_labels=[short_label],
                initial_amount=initial_amount,
                save_results=save_results,
            )
        except Exception as e:
            st.error(f"流水线运行失败: {e}")
            return

        # 用 Analyst 生成完整策略评价（复用 portfolio 页的指标体系）
        try:
            analyst = Analyst.from_backtest(results['bt'], name=name)
            results['summary_grouped'] = analyst.an.summary_grouped()
            results['monthly'] = analyst.monthly_table()
            xls_path = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
            analyst.to_excel(xls_path)
            with open(xls_path, "rb") as f:
                results['excel'] = f.read()
            html_path = xls_path + ".html"
            analyst.to_html(html_path)
            with open(html_path, "rb") as f:
                results['html'] = f.read()
            for p in (xls_path, html_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        except Exception as e:
            st.warning(f"Analyst 评价生成失败（仅展示流水线原始结果）: {e}")

        # bt 含数据库连接对象，不可序列化进 session，移除后再缓存
        results.pop('bt', None)
        st.session_state["fp_results"] = results
        st.session_state["fp_name"] = name


def _show_results():
    r = st.session_state["fp_results"]

    # ── 绩效指标卡 ───────────────────────────────────────────────────────
    st.subheader("多空组合绩效")
    stats = r.get('stats', {})
    if stats:
        cols = st.columns(len(stats))
        for col, (k, v) in zip(cols, stats.items()):
            col.metric(k, v)

    ic_summary = r.get('ic_summary', {})
    if ic_summary:
        st.caption("IC 检验")
        ic_show = {k: (f"{v:.4f}" if isinstance(v, float) else v)
                   for k, v in ic_summary.items()}
        st.dataframe(pd.DataFrame([ic_show]), use_container_width=True, hide_index=True)

    st.divider()

    # ── 图表 ─────────────────────────────────────────────────────────────
    figs = r.get('figs', {})
    if '净值曲线' in figs:
        st.image(figs['净值曲线'], use_container_width=True)
    g1, g2 = st.columns(2)
    if '分组收益' in figs:
        g1.image(figs['分组收益'], use_container_width=True)
    if 'IC时序' in figs:
        g2.image(figs['IC时序'], use_container_width=True)

    st.divider()

    # ── 明细表 ───────────────────────────────────────────────────────────
    st.subheader("详细数据")
    tabs = st.tabs(["策略指标", "分组收益", "Fama-MacBeth", "因子统计", "月度收益"])

    with tabs[0]:
        sg = r.get('summary_grouped')
        if sg:
            for group, items in sg.items():
                if not items:
                    continue
                st.caption(group)
                rows = [{'指标': k, '数值': f"{v:.4f}" if isinstance(v, float) else v}
                        for k, v in items.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("无 Analyst 评价结果")

    with tabs[1]:
        gr = r.get('group_returns')
        if gr is not None and not gr.empty:
            g_cols = [c for c in gr.columns if str(c).startswith('G')]
            st.write("各组平均持仓期收益：")
            st.dataframe(gr[g_cols].mean().to_frame('平均收益').T, use_container_width=True)
            st.write("分组收益时序（前若干期）：")
            st.dataframe(gr.head(50), use_container_width=True)

    with tabs[2]:
        fm = r.get('fm_result')
        if fm is not None and not fm.empty:
            st.dataframe(fm, use_container_width=True)

    with tabs[3]:
        fs = r.get('factor_stats')
        if fs is not None and not fs.empty:
            st.dataframe(fs, use_container_width=True)

    with tabs[4]:
        monthly = r.get('monthly')
        if monthly is not None and not monthly.empty:
            st.dataframe(monthly.style.format("{:.2%}", na_rep="-"),
                         use_container_width=True)
        else:
            st.info("数据不足以生成月度收益表")

    # ── 下载 ─────────────────────────────────────────────────────────────
    st.divider()
    d1, d2 = st.columns(2)
    if "excel" in r:
        d1.download_button(
            "📥 下载 Excel 报告", data=r["excel"],
            file_name=f"factor_{r.get('metric', 'report')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if "html" in r:
        d2.download_button(
            "📥 下载 HTML 报告（交互图）", data=r["html"],
            file_name=f"factor_{r.get('metric', 'report')}.html", mime="text/html",
        )
