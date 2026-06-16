import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(page_title="betalens 看板", layout="wide", initial_sidebar_state="expanded")


def _inject_css():
    st.markdown("""
<style>
/* ── 隐藏默认 chrome ── */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
footer { display: none !important; }

/* ── 侧边栏深色主题 ── */
[data-testid="stSidebar"] {
    background: #0d1b2a !important;
    min-width: 200px !important;
    max-width: 200px !important;
}
[data-testid="stSidebarResizeHandle"] { display: none !important; }

/* 侧边栏所有文字白色 */
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #e2e8f0 !important;
}

/* 侧边栏 title */
[data-testid="stSidebar"] h1 {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0 0 4px 0 !important;
}

/* ── Radio 导航重写 ── */
[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    padding: 7px 14px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    color: #94a3b8 !important;
    font-size: 13px !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.07) !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: #1a56db !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
/* 隐藏原始 radio 圆点 */
[data-testid="stSidebar"] .stRadio input[type="radio"] {
    opacity: 0 !important;
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
}

/* ── 侧边栏分隔线 ── */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
    margin: 8px 0 !important;
}

/* ── 侧边栏按钮 ── */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #94a3b8 !important;
    font-size: 12px !important;
    width: 100% !important;
    padding: 5px 10px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: rgba(255,255,255,0.4) !important;
    color: #e2e8f0 !important;
}

/* ── 主内容区 ── */
.main .block-container {
    padding-top: 1.2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

/* ── Topbar 标题 ── */
.bl-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 8px;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}
.bl-topbar-title {
    font-size: 18px;
    font-weight: 600;
    color: #1e293b;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)


def _render_topbar(title: str):
    st.markdown(f'<div class="bl-topbar"><span class="bl-topbar-title">{title}</span></div>',
                unsafe_allow_html=True)


# ── 页面定义 ─────────────────────────────────────────────────────────────────
from views import portfolio, eventstudy, factor_eval, factor_classes, database

PAGES = {
    "💼 策略评价": ("策略评价", portfolio),
    "📈 事件研究": ("事件研究", eventstudy),
    "📊 因子评价": ("因子评价", factor_eval),
    "📡 因子回测": ("因子回测", factor_classes),
    "🗄️ 数据库管理": ("数据库管理", database),
}

_inject_css()

with st.sidebar:
    st.title("betalens")
    st.divider()
    page_key = st.radio("导航", list(PAGES.keys()), label_visibility="collapsed", key="main_nav")
    st.divider()
    if st.button("🔄 刷新", key="main_refresh"):
        for k in [k for k in st.session_state if k != "main_nav"]:
            del st.session_state[k]
        st.rerun()

title, module = PAGES[page_key]
_render_topbar(title)
module.render()
