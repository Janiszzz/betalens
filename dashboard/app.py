import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from views import query, eventstudy

st.set_page_config(page_title="betalens 看板", layout="wide")

PAGES = {"📊 数据查询": query, "📈 事件研究": eventstudy}

with st.sidebar:
    st.title("betalens 看板")
    st.divider()
    page = st.radio("导航", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    if st.button("🔄 刷新当前页", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

PAGES[page].render()
