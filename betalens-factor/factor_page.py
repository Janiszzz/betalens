"""因子类通用 streamlit 页面渲染器。

各类因子目录下的 page_<class>.py 只需：
    from factor_page import render_factor_class
    def render():
        render_factor_class(Path(__file__).parent, "tdx")

本模块据 spec_<class>.json 动态渲染：因子选择、参数面板（含中性化/
profiling 开关）、运行 FactorPipeline、5 tab 结果展示（策略报告 / 月度收益 /
因子Profiling / 中性化诊断 / 交互图表）。
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def _ensure_path(class_dir: Path):
    """把因子类目录与其父目录（含 factor_template.py）加入 sys.path。"""
    for p in (class_dir.parent, class_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load_spec_json(class_dir: Path, cls: str) -> dict:
    return json.loads((class_dir / f"spec_{cls}.json").read_text(encoding="utf-8"))


@st.cache_resource
def _load_factor_module(path_str: str):
    path = Path(path_str)
    ms = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(mod)
    return mod.spec, mod.__doc__


def render_factor_class(class_dir: Path, cls: str):
    """渲染某因子类的完整交互页面。"""
    _ensure_path(class_dir)
    from factor_template import FactorPipeline  # noqa: E402

    spec_data = _load_spec_json(class_dir, cls)
    cfg = spec_data["defaults"]
    factors = {f["name"]: class_dir / f["name"] / f"factor_{f['name']}.py"
               for f in spec_data["factors"]}

    st.header(f"{cls} 因子")
    st.caption(spec_data.get("source", ""))

    key = f"{cls}_"
    factor_name = st.selectbox("选择因子", list(factors.keys()), key=key + "factor")
    fpath = factors[factor_name]
    if not fpath.exists():
        st.error(f"未找到因子脚本：{fpath}")
        return

    factor_spec, doc = _load_factor_module(str(fpath))
    with st.expander("因子说明", expanded=False):
        st.text(doc or "（无说明）")

    # ---- 参数面板 ----
    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input("起始日期", value=pd.Timestamp(cfg["start_date"]), key=key + "start")
        end = st.date_input("结束日期", value=pd.Timestamp(cfg["end_date"]), key=key + "end")
    with c2:
        freq_opts = ["W", "ME", "QE", "D"]
        freq = st.selectbox(
            "再平衡频率", freq_opts,
            index=freq_opts.index(cfg["rebal_freq"]) if cfg["rebal_freq"] in freq_opts else 0,
            format_func=lambda f: {"W": "每周", "ME": "月末", "QE": "季末", "D": "每日"}[f],
            key=key + "freq")
        n_quantiles = st.slider("分组数", 5, 20, cfg["n_quantiles"], key=key + "nq")
    with c3:
        index_code = st.text_input("指数代码", value=cfg["index_code"], key=key + "idx")

    c4, c5, c6 = st.columns(3)
    with c4:
        use_industry = st.checkbox("行业中性化", value=cfg.get("use_industry", False),
                                   key=key + "ind")
        scheme = st.text_input("行业方案", value=cfg.get("industry_scheme", "申万一级行业"),
                               key=key + "scheme", disabled=not use_industry)
    with c5:
        use_mktcap = st.checkbox("市值中性化", value=cfg.get("use_mktcap", False),
                                 key=key + "mktcap")
    with c6:
        include_profiling = st.checkbox("因子 Profiling", value=cfg.get("include_profiling", True),
                                        key=key + "prof")

    if st.button("▶ 运行回测", key=key + "run", type="primary"):
        factor_dir = class_dir / factor_name
        _run(FactorPipeline, factor_spec, factor_dir, key,
             start=start, end=end, freq=freq, n_quantiles=n_quantiles,
             index_code=index_code, use_industry=use_industry, scheme=scheme,
             use_mktcap=use_mktcap, include_profiling=include_profiling)

    if key + "results" not in st.session_state:
        st.info("配置参数后点击「运行回测」。该过程会连接数据库并计算因子，耗时取决于股票池规模。")
        return

    _show_results(class_dir, key)


def _run(FactorPipeline, factor_spec, factor_dir, key, *,
         start, end, freq, n_quantiles, index_code,
         use_industry, scheme, use_mktcap, include_profiling):
    spec = dataclasses.replace(
        factor_spec, index_code=index_code,
        use_industry=use_industry, use_mktcap=use_mktcap,
        industry_scheme=scheme)
    with st.spinner(f"正在运行 {spec.name} 因子回测..."):
        try:
            result = FactorPipeline(spec).run(
                str(start), str(end), rebal_freq=freq,
                n_quantiles=n_quantiles, output_dir=str(factor_dir),
                include_profiling=include_profiling)
        except Exception as e:
            st.error(f"回测失败: {e}")
            return

    analyst = result.analyst
    html_path = factor_dir / f"{spec.name}_report.html"
    png_path = factor_dir / f"{spec.name}_profiling.png"
    neu = result.neutralize_stats

    st.session_state[key + "results"] = {
        "name": spec.name,
        "html": html_path.read_bytes() if html_path.exists() else None,
        "summary_grouped": analyst.an.summary_grouped(),
        "monthly": analyst.monthly_table() if hasattr(analyst, "monthly_table") else None,
        "profiling": result.profiling,
        "profiling_png": png_path.read_bytes() if png_path.exists() else None,
        "neu_stats": neu.reset_index() if neu is not None else None,
    }


def _show_results(class_dir, key):
    r = st.session_state[key + "results"]
    tabs = st.tabs(["策略报告", "月度收益", "因子Profiling", "中性化诊断", "交互图表"])

    with tabs[0]:
        sg = r.get("summary_grouped")
        if sg:
            for group, items in sg.items():
                if not items:
                    continue
                st.caption(group)
                rows = [{"指标": k, "数值": f"{v:.4f}" if isinstance(v, float) else v}
                        for k, v in items.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("无策略指标")

    with tabs[1]:
        monthly = r.get("monthly")
        if monthly is not None and not monthly.empty:
            st.dataframe(monthly.style.format("{:.2%}", na_rep="-"), use_container_width=True)
        else:
            st.info("数据不足以生成月度收益表")

    with tabs[2]:
        prof = r.get("profiling")
        if not prof:
            st.info("本次运行未开启因子 Profiling")
        else:
            ac = prof.get("autocorrelation")
            to = prof.get("turnover")
            out = prof.get("outliers")
            m1, m2, m3 = st.columns(3)
            if ac is not None and 1 in ac.index:
                m1.metric("lag-1 自相关", f"{ac.loc[1, '自相关均值']:.3f}")
            if to is not None:
                m2.metric("平均换手率", f"{to.mean():.2%}")
            if out is not None and "Total" in out.index:
                m3.metric("极值占比", f"{out.loc['Total', '极值占比']:.2%}")
            png = r.get("profiling_png")
            if png:
                st.image(png, use_container_width=True)
            with st.expander("分布统计明细"):
                d = prof.get("distribution")
                if d is not None:
                    st.dataframe(d, use_container_width=True)

    with tabs[3]:
        neu = r.get("neu_stats")
        if neu is None or neu.empty:
            st.info("本次运行未做中性化（行业/市值开关均关闭）")
        else:
            done = neu[~neu["skipped"]] if "skipped" in neu.columns else neu
            m1, m2, m3 = st.columns(3)
            m1.metric("总期数", len(neu))
            m2.metric("成功/跳过", f"{len(done)} / {len(neu) - len(done)}")
            if "r2" in done.columns and len(done):
                m3.metric("平均 R²", f"{done['r2'].mean():.4f}")
            st.dataframe(neu, use_container_width=True, hide_index=True)

    with tabs[4]:
        html = r.get("html")
        if html:
            components.html(html.decode("utf-8"), height=1200, scrolling=True)
            st.download_button("📥 下载 HTML 报告", data=html,
                               file_name=f"{r['name']}_report.html", mime="text/html")
        else:
            st.warning("未找到 HTML 报告文件")
