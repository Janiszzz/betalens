import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from views import eventstudy, database

st.set_page_config(page_title="betalens 看板", layout="wide")

PAGES = {"📈 事件研究": eventstudy, "🗄️ 数据库管理": database}

with st.sidebar:
    st.title("betalens 看板")
    st.divider()
    page = st.radio("导航", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    if st.button("🔄 刷新当前页", width="stretch"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

PAGES[page].render()
