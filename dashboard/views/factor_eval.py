import io
import streamlit as st
import pandas as pd
from betalens.factor.stats import (
    run_timing_evaluation,
    export_timing_report,
    plot_factor_timeseries,
    plot_rolling_ic,
    plot_signal_avg_return,
    plot_win_rate_comparison,
    plot_ic_by_period,
    plot_return_distribution,
    plot_factor_vs_return,
    plot_composite_score,
    run_cross_section_evaluation,
    calc_ic,
    summarize_ic,
    plot_group_cumulative_returns,
)


def render():
    with st.sidebar:
        st.divider()
        sub = st.radio("", ["择时因子评价", "截面因子评价"],
                       label_visibility="collapsed", key="fe_sub")

    if sub == "择时因子评价":
        _render_timing_tab()
    else:
        _render_cross_section_tab()


# ── 择时因子评价 ────────────────────────────────────────────────────────────

def _render_timing_tab():
    file = st.file_uploader(
        "上传因子数据（CSV/Excel）", type=["csv", "xlsx", "xls"],
        key="fe_timing_file",
        help="需包含: 日期列、因子值列、收益率列",
    )

    if file is None:
        st.info("请上传包含 **日期、因子值、收益率** 三列的 CSV 或 Excel 文件。")
        return

    # 读取数据
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.write("**数据预览（前5行）：**")
    st.dataframe(df.head(), use_container_width=True)

    # 列映射
    cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1:
        dt_col = st.selectbox("日期列", cols, key="fe_t_dt")
    with c2:
        fac_col = st.selectbox("因子值列", cols, index=min(1, len(cols) - 1), key="fe_t_fac")
    with c3:
        ret_col = st.selectbox("收益率列", cols, index=min(2, len(cols) - 1), key="fe_t_ret")

    # 参数
    with st.expander("⚙️ 参数设置", expanded=False):
        p1, p2, p3 = st.columns(3)
        with p1:
            periods = st.multiselect("预测周期", [5, 10, 20, 40, 60], default=[5, 10, 20], key="fe_t_periods")
            method = st.radio("IC 方法", ["spearman", "pearson"], key="fe_t_method")
        with p2:
            sigma = st.number_input("σ 阈值", value=1.0, step=0.1, key="fe_t_sigma")
            ma_window = st.number_input("均线窗口", value=250, step=10, key="fe_t_ma")
        with p3:
            rolling_window = st.number_input("滚动 IC 窗口", value=60, step=5, key="fe_t_rw")
            is_ratio = st.slider("样本内比例", 0.5, 0.9, 0.7, 0.05, key="fe_t_is")

    factor_name = st.text_input("因子名称（可选）", value=file.name.split('.')[0], key="fe_t_name")

    if st.button("▶ 运行择时因子评价", key="fe_t_run", type="primary"):
        try:
            df[dt_col] = pd.to_datetime(df[dt_col])
            factor = df.set_index(dt_col)[fac_col].sort_index().dropna()
            returns = df.set_index(dt_col)[ret_col].sort_index().dropna()
        except Exception as e:
            st.error(f"数据解析失败: {e}")
            return

        with st.spinner("正在计算..."):
            results = run_timing_evaluation(
                factor=factor,
                returns=returns,
                periods=periods if periods else [5, 10, 20],
                method=method,
                rolling_window=rolling_window,
                sigma=sigma,
                ma_window=ma_window,
                is_ratio=is_ratio,
                factor_name=factor_name,
            )
        st.session_state["fe_timing_results"] = results
        st.session_state["fe_timing_returns"] = returns

    # 展示结果
    if "fe_timing_results" not in st.session_state:
        return

    results = st.session_state["fe_timing_results"]
    main_period = results['main_period']
    ic_results = results['ic_results']
    signal_tests = results['signal_tests']
    score = results['score']

    st.success(f"评价完成 — 综合评分: **{score['综合评分']:.2f}**（等级 **{score['评级']}**）")
    st.divider()

    # Row 1: 因子时序 + 滚动 IC
    r1c1, r1c2 = st.columns([1, 1])
    with r1c1:
        img = plot_factor_timeseries(
            results['factor_std'],
            signal=results['signals'].get('阈值法'),
            sigma=1.0,
            title=f'因子值（预处理后）',
        )
        st.image(img, use_container_width=True)
    with r1c2:
        if main_period in ic_results and len(ic_results[main_period]['ic_series']) > 0:
            img = plot_rolling_ic(
                ic_results[main_period]['ic_series'],
                period=main_period,
            )
            st.image(img, use_container_width=True)

    # Row 2: 信号均收益 + 胜率对比 + IC by period
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        if len(signal_tests) > 0:
            img = plot_signal_avg_return(signal_tests, method_name='阈值法')
            st.image(img, use_container_width=True)
    with r2c2:
        if len(signal_tests) > 0:
            img = plot_win_rate_comparison(signal_tests)
            st.image(img, use_container_width=True)
    with r2c3:
        img = plot_ic_by_period(ic_results)
        st.image(img, use_container_width=True)

    # Row 3: 收益分布 + 散点图 + 综合评分
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        returns_saved = st.session_state.get("fe_timing_returns")
        if returns_saved is not None:
            img = plot_return_distribution(
                returns_saved,
                results['signals']['阈值法'],
                period=main_period,
            )
            st.image(img, use_container_width=True)
    with r3c2:
        returns_saved = st.session_state.get("fe_timing_returns")
        if returns_saved is not None:
            img = plot_factor_vs_return(
                results['factor_std'],
                returns_saved,
                period=main_period,
            )
            st.image(img, use_container_width=True)
        else:
            st.info("因子 vs 收益散点图：请重新运行评价")
    with r3c3:
        img = plot_composite_score(score)
        st.image(img, use_container_width=True)

    st.divider()

    # 数据表
    st.subheader("详细数据")
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "综合概览", "IC 相关性分析", "信号检验结果", "回归分析", "稳健性检验", "回测绩效汇总"
    ])

    with t1:
        ic_stats = ic_results.get(main_period, {}).get('stats', {})
        overview = {
            '因子名称': results['factor_name'],
            '预测周期': f'{main_period}期',
            'IC方法': results['method'].upper(),
            'IC均值': f"{ic_stats.get('IC均值', 0):.4f}",
            'ICIR': f"{ic_stats.get('ICIR', 0):.4f}",
            'IC正值占比': f"{ic_stats.get('胜率(IC>0)', 0):.2%}",
            '综合评分': f"{score['综合评分']:.4f}",
            '评级': score['评级'],
        }
        st.dataframe(pd.DataFrame([overview]), use_container_width=True)

    with t2:
        rows = []
        for period, data in ic_results.items():
            s = data['stats']
            rows.append({
                '预测周期': f'{period}日',
                'IC均值': f"{s.get('IC均值', 0):.4f}",
                'IC标准差': f"{s.get('IC_std', 0):.4f}",
                'ICIR': f"{s.get('ICIR', 0):.4f}",
                'IC>0占比': f"{s.get('胜率(IC>0)', 0):.2%}",
                'T统计量': f"{s.get('t统计量', 0):.4f}",
                'P值': f"{s.get('p值', 0):.4f}",
                '是否显著': '✓ 显著' if s.get('p值', 1) < 0.05 else '✗ 不显著',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with t3:
        if len(signal_tests) > 0:
            st.dataframe(signal_tests, use_container_width=True)

    with t4:
        reg = results['regression']
        st.dataframe(pd.DataFrame([reg]), use_container_width=True)

    with t5:
        rob = results['robustness']
        st.dataframe(pd.DataFrame([rob]), use_container_width=True)

    with t6:
        bt = results['backtest']
        st.dataframe(pd.DataFrame([bt]), use_container_width=True)

    # 导出
    st.divider()
    excel_bytes = export_timing_report(results)
    st.download_button(
        "📥 下载 Excel 报告",
        data=excel_bytes,
        file_name=f"timing_report_{results['factor_name']}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── 截面因子评价 ────────────────────────────────────────────────────────────

def _render_cross_section_tab():
    file = st.file_uploader(
        "上传面板数据（CSV/Excel）", type=["csv", "xlsx", "xls"],
        key="fe_cs_file",
        help="需包含: 日期列、代码列、因子值列、收益率列",
    )

    if file is None:
        st.info("请上传包含 **日期、代码、因子值、收益率** 四列的面板数据。")
        return

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.write("**数据预览（前5行）：**")
    st.dataframe(df.head(), use_container_width=True)

    cols = df.columns.tolist()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dt_col = st.selectbox("日期列", cols, key="fe_cs_dt")
    with c2:
        code_col = st.selectbox("代码列", cols, index=min(1, len(cols) - 1), key="fe_cs_code")
    with c3:
        fac_col = st.selectbox("因子值列", cols, index=min(2, len(cols) - 1), key="fe_cs_fac")
    with c4:
        ret_col = st.selectbox("收益率列", cols, index=min(3, len(cols) - 1), key="fe_cs_ret")

    with st.expander("⚙️ 参数设置", expanded=False):
        p1, p2, p3 = st.columns(3)
        with p1:
            method = st.radio("IC 方法", ["spearman", "pearson"], key="fe_cs_method")
        with p2:
            n_groups = st.number_input("分组数", value=5, min_value=2, max_value=20, key="fe_cs_ng")
        with p3:
            periods = st.multiselect("预测周期", [1, 5, 10, 20], default=[5], key="fe_cs_periods")

    factor_name = st.text_input("因子名称（可选）", value=file.name.split('.')[0], key="fe_cs_name")

    if st.button("▶ 运行截面因子评价", key="fe_cs_run", type="primary"):
        try:
            df[dt_col] = pd.to_datetime(df[dt_col])
            factor_wide = df.pivot_table(index=dt_col, columns=code_col, values=fac_col)
            return_wide = df.pivot_table(index=dt_col, columns=code_col, values=ret_col)
        except Exception as e:
            st.error(f"数据解析失败: {e}")
            return

        with st.spinner("正在计算..."):
            results = run_cross_section_evaluation(
                factor_data=factor_wide,
                return_data=return_wide,
                periods=periods if periods else [5],
                method=method,
                n_groups=n_groups,
                factor_name=factor_name,
            )
        st.session_state["fe_cs_results"] = results

    if "fe_cs_results" not in st.session_state:
        return

    results = st.session_state["fe_cs_results"]
    ic_all = results['ic_all']
    main_period = results['periods'][0]

    st.divider()

    # 图表
    c1, c2 = st.columns(2)
    with c1:
        if main_period in ic_all and len(ic_all[main_period]['ic_series']) > 0:
            img = plot_rolling_ic(
                ic_all[main_period]['ic_series'],
                period=main_period,
                title=f'截面 IC 序列（{main_period}日）',
            )
            st.image(img, use_container_width=True)
    with c2:
        img = plot_ic_by_period(ic_all)
        st.image(img, use_container_width=True)

    if len(results['group_returns']) > 0:
        img = plot_group_cumulative_returns(results['group_returns'])
        st.image(img, use_container_width=True)

    # 数据表
    st.subheader("详细数据")
    t1, t2, t3, t4 = st.tabs(["IC 分析", "Fama-MacBeth", "分组统计", "单调性检验"])

    with t1:
        rows = []
        for period, data in ic_all.items():
            s = data['stats']
            rows.append({
                '预测周期': f'{period}日',
                'IC均值': f"{s.get('IC均值', 0):.4f}",
                'ICIR': f"{s.get('ICIR', 0):.4f}",
                'IC>0占比': f"{s.get('胜率(IC>0)', 0):.2%}",
                'T统计量': f"{s.get('t统计量', 0):.4f}",
                '是否显著': '✓ 显著' if s.get('p值', 1) < 0.05 else '✗ 不显著',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with t2:
        st.dataframe(results['fm_results'], use_container_width=True)

    with t3:
        gr = results['group_returns']
        if len(gr) > 0:
            st.write("各组平均收益：")
            st.dataframe(gr.mean().to_frame('平均收益').T, use_container_width=True)
            st.write("各组累积收益：")
            st.dataframe(gr.sum().to_frame('累积收益').T, use_container_width=True)

    with t4:
        mono = results.get('mono_test', {})
        if mono:
            st.dataframe(pd.DataFrame([{
                '是否单调': mono.get('是否单调', ''),
                '方向': mono.get('方向', ''),
                'Spearman相关系数': f"{mono.get('Spearman相关系数', 0):.4f}",
                'P值': f"{mono.get('P值', 0):.4f}",
            }]), use_container_width=True)
            if '各组均值' in mono:
                st.bar_chart(pd.Series(mono['各组均值']))
