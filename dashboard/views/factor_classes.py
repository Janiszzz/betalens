"""因子回测页 — 动态发现 betalens-factor 下各因子类，二级导航选类后渲染对应页面。

各因子类目录（betalens-factor/<class>/）只要有 page_<class>.py（暴露 render()），
即被自动发现，无需改动本文件。新增因子类零配置接入。
"""
import importlib.util
import sys
from pathlib import Path

import streamlit as st

_FACTOR_ROOT = Path(__file__).resolve().parent.parent.parent / "betalens-factor"


def _discover_classes() -> dict[str, Path]:
    """扫描 betalens-factor/*/page_*.py，返回 {类名: page文件路径}。"""
    found = {}
    if not _FACTOR_ROOT.exists():
        return found
    for sub in sorted(_FACTOR_ROOT.iterdir()):
        if not sub.is_dir() or sub.name.startswith((".", "__")):
            continue
        for page in sub.glob("page_*.py"):
            cls = page.stem[len("page_"):]
            found[cls] = page
            break
    return found


def _load_page(path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    ms = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(mod)
    return mod


def render():
    classes = _discover_classes()
    if not classes:
        st.warning(f"未在 {_FACTOR_ROOT} 下发现任何 page_*.py 因子类页面。")
        return

    with st.sidebar:
        st.divider()
        cls = st.radio("因子类", list(classes.keys()), label_visibility="collapsed", key="fc_class")
    try:
        page = _load_page(classes[cls])
        page.render()
    except Exception as e:
        st.error(f"加载因子类「{cls}」失败: {e}")
        st.exception(e)
