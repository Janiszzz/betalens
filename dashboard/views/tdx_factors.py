"""通达信因子页 — 选择因子、配置参数、运行 FactorPipeline 回测、展示结果"""
import dataclasses
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

_TDX_DIR = Path(__file__).resolve().parent.parent.parent.parent / "betalens-factor" / "tdx"
if str(_TDX_DIR) not in sys.path:
    sys.path.insert(0, str(_TDX_DIR))

from factor_template_tdx import FactorPipeline  # noqa: E402

_FACTORS = {
    "吸筹能量 (XICHOU)": {"file": _TDX_DIR / "factor_XICHOU.py"},
}


@st.cache_resource
def _load_spec(path: Path):
    ms = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(mod)
    return mod.spec, mod.__doc__


def render():
    st.header("通达信因子")
    st.caption("TDX 技术指标因子 —— 取数 → 算子 → 分组多空回测 → 绩效评价")

    factor_name = st.selectbox("选择因子", list(_FACTORS.keys()), key="tdx_factor")
    meta = _FACTORS[factor_name]

    if not meta["file"].exists():
        st.error(f"未找到因子脚本：{meta['file']}")
        return

    factor_spec, doc = _load_spec(meta["file"])

    with st.expander("因子说明", expanded=False):
        st.text(doc or "（无说明）")

    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input("起始日期", value=pd.Timestamp("2024-01-01"), key="tdx_start")
        end = st.date_input("结束日期", value=pd.Timestamp("2025-12-31"), key="tdx_end")
    with c2:
        freq = st.selectbox("再平衡频率", ["W", "ME", "QE", "D"],
                            format_func=lambda f: {"W": "每周", "ME": "月末", "QE": "季末", "D": "每日"}[f],
                            key="tdx_freq")
        n_quantiles = st.slider("分组数", 5, 20, 10, key="tdx_nq")
    with c3:
        index_code = st.text_input("指数代码", value=factor_spec.index_code or "000906.SH", key="tdx_idx")

    if st.button("▶ 运行回测", key="tdx_run", type="primary"):
        _run(factor_spec=factor_spec, start=start, end=end,
             freq=freq, n_quantiles=n_quantiles, index_code=index_code)

    if "tdx_results" not in st.session_state:
        st.info("配置参数后点击「运行回测」。该过程会连接数据库并计算 TDX 指标，耗时取决于股票池规模。")
        return

    _show_results()


def _run(*, factor_spec, start, end, freq, n_quantiles, index_code):
    spec = dataclasses.replace(factor_spec, index_code=index_code)
    with st.spinner(f"正在运行 {spec.name} 因子回测..."):
        try:
            bt, analyst = FactorPipeline(spec).run(
                start_date=str(start),
                end_date=str(end),
                rebal_freq=freq,
                n_quantiles=n_quantiles,
                output_dir=str(_TDX_DIR),
            )
        except Exception as e:
            st.error(f"回测失败: {e}")
            return

        html_path = _TDX_DIR / f"{spec.name}_report.html"
        results = {
            "name": spec.name,
            "html": html_path.read_bytes() if html_path.exists() else None,
            "summary_grouped": analyst.an.summary_grouped(),
            "monthly": analyst.monthly_table() if hasattr(analyst, "monthly_table") else None,
        }
        results.pop("bt", None)
        st.session_state["tdx_results"] = results


def _show_results():
    r = st.session_state["tdx_results"]

    tabs = st.tabs(["策略指标", "月度收益", "交互图表"])

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

    with tabs[1]:
        monthly = r.get("monthly")
        if monthly is not None and not monthly.empty:
            st.dataframe(monthly.style.format("{:.2%}", na_rep="-"), use_container_width=True)
        else:
            st.info("数据不足以生成月度收益表")

    with tabs[2]:
        html = r.get("html")
        if html:
            components.html(html.decode("utf-8"), height=1200, scrolling=True)
            st.download_button("📥 下载 HTML 报告", data=html,
                               file_name=f"{r['name']}_report.html", mime="text/html")
        else:
            st.warning("未找到 HTML 报告文件")
