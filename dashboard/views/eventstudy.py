import io
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['simhei']
matplotlib.rcParams['axes.unicode_minus'] = False
import streamlit as st
import pandas as pd
from betalens import Datafeed
from betalens.eventstudy.eventstudy import EventStudy

# ── 步骤定义 ────────────────────────────────────────────────────────────────
STEPS = ["① 上传事件文件", "② 设置标的代码", "③ 设置回测参数", "④ 查看结果"]


def _step_indicator(current: int):
    """左侧步骤进度显示"""
    for i, label in enumerate(STEPS):
        if i < current:
            st.markdown(f"~~{label}~~ ✅")
        elif i == current:
            st.markdown(f"**{label}** ◀")
        else:
            st.markdown(f"{label}")


def _capture_plot(plot_fn, **kwargs) -> bytes:
    """将 EventStudy plot 函数的输出捕获为 PNG bytes"""
    buf = io.BytesIO()
    plot_fn(**kwargs, save_path=buf)
    buf.seek(0)
    return buf.read()


def render():
    st.header("事件研究分析")

    # ── 布局：左侧步骤导航 + 右侧内容 ───────────────────────────────────────
    left, right = st.columns([1, 4])

    # ── 初始化 session state ─────────────────────────────────────────────────
    for key, default in [
        ("es_events", None),
        ("es_step", 0),
        ("es_result", None),
        ("es_params", {}),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    step = st.session_state["es_step"]

    with left:
        st.subheader("步骤")
        _step_indicator(step)

    with right:
        # ── 步骤 0：上传事件文件 ─────────────────────────────────────────────
        if step == 0:
            st.subheader(STEPS[0])
            file = st.file_uploader("上传事件文件（xlsx 或 csv）", type=["xlsx", "csv"])
            if file:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                st.write("预览（前5行）：")
                st.dataframe(df.head())
                date_col = st.selectbox("选择日期列", df.columns.tolist())
                event_col = st.selectbox("选择事件列（0/1）", df.columns.tolist(),
                                         index=min(1, len(df.columns) - 1))
                if st.button("下一步 →"):
                    df[date_col] = pd.to_datetime(df[date_col])
                    events = df.set_index(date_col)[event_col]
                    st.session_state["es_events"] = events
                    st.session_state["es_step"] = 1
                    st.rerun()

        # ── 步骤 1：设置标的代码 ─────────────────────────────────────────────
        elif step == 1:
            st.subheader(STEPS[1])
            st.info("单个代码直接输入；多个代码用逗号分隔（多标的均值模式）")
            codes_input = st.text_input("标的代码", value=st.session_state["es_params"].get("codes_raw", "868008.WI"))
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← 上一步"):
                    st.session_state["es_step"] = 0
                    st.rerun()
            with col2:
                if st.button("下一步 →"):
                    raw = [c.strip() for c in codes_input.split(",") if c.strip()]
                    st.session_state["es_params"]["codes_raw"] = codes_input
                    st.session_state["es_params"]["codes"] = raw[0] if len(raw) == 1 else raw
                    st.session_state["es_step"] = 2
                    st.rerun()

        # ── 步骤 2：设置回测参数 ─────────────────────────────────────────────
        elif step == 2:
            st.subheader(STEPS[2])
            p = st.session_state["es_params"]
            metric = st.text_input("价格指标名", value=p.get("metric", "收盘价"))
            col1, col2 = st.columns(2)
            with col1:
                w_before = st.slider("事件前窗口（天）", 1, 60, p.get("window_before", 20))
            with col2:
                w_after = st.slider("事件后窗口（天）", 1, 60, p.get("window_after", 20))
            benchmark = st.text_input("基准代码（留空=绝对收益）", value=p.get("benchmark_code", ""))
            offset = st.number_input("持有起点偏移（0=事件发生日）", value=p.get("holding_start_offset", 0), step=1)
            table = st.text_input("数据表名", value=p.get("table", "daily_market"))

            col1, col2 = st.columns(2)
            with col1:
                if st.button("← 上一步"):
                    st.session_state["es_step"] = 1
                    st.rerun()
            with col2:
                if st.button("▶ 运行事件研究"):
                    st.session_state["es_params"].update({
                        "metric": metric,
                        "window_before": w_before,
                        "window_after": w_after,
                        "benchmark_code": benchmark.strip() or None,
                        "holding_start_offset": int(offset),
                        "table": table,
                    })
                    st.session_state["es_step"] = 3
                    st.session_state["es_result"] = None  # 触发重新计算
                    st.rerun()

        # ── 步骤 3：查看结果 ─────────────────────────────────────────────────
        elif step == 3:
            st.subheader(STEPS[3])
            p = st.session_state["es_params"]
            events = st.session_state["es_events"]

            if st.session_state["es_result"] is None:
                with st.spinner("正在分析..."):
                    try:
                        datafeed = Datafeed(p["table"])
                        es = EventStudy(datafeed)
                        result = es.analyze(
                            events=events,
                            code=p["codes"],
                            window_before=p["window_before"],
                            window_after=p["window_after"],
                            metric=p["metric"],
                            benchmark_code=p.get("benchmark_code"),
                            holding_start_offset=p.get("holding_start_offset", 0),
                        )
                        datafeed.close()
                        st.session_state["es_result"] = (es, result)
                    except Exception as e:
                        st.error(f"分析失败: {e}")
                        if st.button("← 返回修改参数"):
                            st.session_state["es_step"] = 2
                            st.rerun()
                        return

            if st.session_state["es_result"]:
                es, result = st.session_state["es_result"]

                if "error" in result:
                    st.error(f"分析错误: {result['error']}")
                else:
                    st.success(f"成功分析 {result['event_count']} 个事件")

                    daily_stats = result["daily_stats"]
                    cumulative_stats = result["cumulative_stats"]

                    # ── 图表 ────────────────────────────────────────────────
                    st.subheader("可视化结果")
                    tab_bar, tab_line, tab_events = st.tabs(["柱状图（日均收益）", "折线图（累积收益）", "多事件曲线"])

                    with tab_bar:
                        img = _capture_plot(es.plot_bar, daily_stats=daily_stats,
                                            title=f"{p['codes']} 事件前后平均收益率")
                        st.image(img)

                    with tab_line:
                        img = _capture_plot(es.plot_lines, cumulative_stats=cumulative_stats,
                                            title=f"{p['codes']} 事件前后平均累积收益率")
                        st.image(img)

                    with tab_events:
                        # 多事件曲线需要重新连接数据库
                        if st.button("生成多事件曲线"):
                            with st.spinner("绘图中..."):
                                try:
                                    df2 = Datafeed(p["table"])
                                    es2 = EventStudy(df2)
                                    img = _capture_plot(
                                        es2.plot_events_lines,
                                        events=events,
                                        code=p["codes"] if isinstance(p["codes"], str) else p["codes"][0],
                                        window_before=p["window_before"],
                                        window_after=p["window_after"],
                                        metric=p["metric"],
                                        max_events=10,
                                        title=f"{p['codes']} 多事件累积收益对比",
                                    )
                                    df2.close()
                                    st.image(img)
                                except Exception as e:
                                    st.error(f"绘图失败: {e}")

                    # ── 数据表 ───────────────────────────────────────────────
                    st.subheader("统计数据")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("每日收益率统计")
                        st.dataframe(daily_stats, use_container_width=True)
                    with c2:
                        st.write("累积收益率统计")
                        st.dataframe(cumulative_stats, use_container_width=True)

            if st.button("← 修改参数"):
                st.session_state["es_step"] = 2
                st.session_state["es_result"] = None
                st.rerun()
