"""Bilingual UI helpers for the Streamlit workbench."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import streamlit as st

from fraclab_sdk.config import SDKConfig
from fraclab_sdk.workbench.page_state import read_global_setting, write_global_setting

LanguageCode = str

_SUPPORTED_LANGUAGES: tuple[LanguageCode, ...] = ("en", "zh-CN")
_LANGUAGE_SESSION_KEY = "_workbench_language"
_LANGUAGE_SELECT_KEY = "_workbench_language_select"
_LANGUAGE_PERSISTED_CACHE_KEY = "_workbench_language_persisted"
_LANGUAGE_SETTING_KEY = "language"
_CURRENT_PAGE_KEY = "_workbench_current_page"
_LAST_TOOLBAR_PAGE_KEY = "_workbench_language_toolbar_page"
_WORKBENCH_CONFIG = SDKConfig()


class _FormatDict(dict[str, object]):
    """Leave unknown format placeholders untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class WorkbenchPage:
    """Static metadata for a Workbench page."""

    key: str
    path: str
    icon: str
    title_en: str
    title_zh: str


_PAGES: tuple[WorkbenchPage, ...] = (
    WorkbenchPage("home", "Home.py", "🔬", "Workbench", "工作台总览"),
    WorkbenchPage("snapshots", "pages/1_Snapshots.py", "📦", "Snapshots", "快照管理"),
    WorkbenchPage("browse", "pages/2_Browse.py", "🔍", "Browse", "数据浏览"),
    WorkbenchPage("selection", "pages/3_Selection.py", "✅", "Selection", "运行配置"),
    WorkbenchPage("run", "pages/4_Run.py", "▶️", "Run", "运行管理"),
    WorkbenchPage("results", "pages/5_Results.py", "📊", "Results", "运行结果"),
    WorkbenchPage("algorithm_edit", "pages/6_Algorithm_Edit.py", "✏️", "Algorithm Editor", "算法代码编辑"),
    WorkbenchPage("schema_edit", "pages/7_Schema_Edit.py", "🧩", "Schema Editor", "输入参数编辑"),
    WorkbenchPage("output_edit", "pages/8_Output_Edit.py", "📤", "Output Editor", "输出结果定义"),
    WorkbenchPage("export_algorithm", "pages/9_Export_Algorithm.py", "📦", "Export Algorithm", "算法打包导出"),
)
_PAGE_BY_KEY = {page.key: page for page in _PAGES}


def _default_language() -> LanguageCode:
    """Resolve the initial language, defaulting to Chinese."""
    env_lang = os.environ.get("FRACLAB_WORKBENCH_LANG", "").strip()
    if env_lang in _SUPPORTED_LANGUAGES:
        return env_lang
    return "zh-CN"


def _read_persisted_language() -> LanguageCode | None:
    """Load the persisted language from disk once per session."""
    cached = st.session_state.get(_LANGUAGE_PERSISTED_CACHE_KEY)
    if cached in _SUPPORTED_LANGUAGES:
        return cached

    persisted = read_global_setting(_LANGUAGE_SETTING_KEY, _WORKBENCH_CONFIG)
    if persisted in _SUPPORTED_LANGUAGES:
        st.session_state[_LANGUAGE_PERSISTED_CACHE_KEY] = persisted
        return persisted

    return None


def _persist_language(lang: LanguageCode) -> None:
    """Persist the chosen language globally for future page loads."""
    if lang not in _SUPPORTED_LANGUAGES:
        return
    if st.session_state.get(_LANGUAGE_PERSISTED_CACHE_KEY) == lang:
        return
    write_global_setting(_LANGUAGE_SETTING_KEY, lang, _WORKBENCH_CONFIG)
    st.session_state[_LANGUAGE_PERSISTED_CACHE_KEY] = lang


def _apply_selected_language() -> None:
    """Sync the language widget selection into global workbench state."""
    selected_lang = st.session_state.get(_LANGUAGE_SELECT_KEY)
    if selected_lang not in _SUPPORTED_LANGUAGES:
        return
    st.session_state[_LANGUAGE_SESSION_KEY] = selected_lang
    _persist_language(selected_lang)


def get_language() -> LanguageCode:
    """Return the current UI language."""
    lang = st.session_state.get(_LANGUAGE_SESSION_KEY)
    if lang in _SUPPORTED_LANGUAGES:
        return lang

    resolved_lang = _read_persisted_language() or _default_language()
    st.session_state[_LANGUAGE_SESSION_KEY] = resolved_lang
    _persist_language(resolved_lang)
    return resolved_lang


def tx(en: str, zh: str, **kwargs: object) -> str:
    """Translate a short UI string between English and Chinese."""
    template = zh if get_language() == "zh-CN" else en
    if not kwargs:
        return template
    return template.format_map(_FormatDict(kwargs))


def page_title(page_key: str) -> str:
    """Return the localized title for a page key."""
    page = _PAGE_BY_KEY[page_key]
    return tx(page.title_en, page.title_zh)


def run_status_label(status: Any) -> str:
    """Return a localized label for a run status enum/string."""
    raw = getattr(status, "value", status)
    key = str(raw).strip().lower()
    labels = {
        "pending": tx("Pending", "待处理"),
        "running": tx("Running", "运行中"),
        "succeeded": tx("Succeeded", "成功"),
        "failed": tx("Failed", "失败"),
        "timeout": tx("Timeout", "超时"),
    }
    return labels.get(key, tx("Unknown", "未知"))


def _language_option_label(code: LanguageCode) -> str:
    """Return a fixed display label for a language code."""
    return "English" if code == "en" else "中文"


def set_current_page(page_key: str | None) -> None:
    """Record the current workbench page for page-local widget keys."""
    st.session_state[_CURRENT_PAGE_KEY] = str(page_key or "global")


def _language_widget_key() -> str:
    """Build a page-local widget key for the language switcher."""
    current_page = str(st.session_state.get(_CURRENT_PAGE_KEY) or "global")
    return f"{_LANGUAGE_SELECT_KEY}::{current_page}"


def render_language_toolbar() -> None:
    """Render the page-level language switcher."""
    current_lang = get_language()
    current_page = str(st.session_state.get(_CURRENT_PAGE_KEY) or "global")
    widget_key = _language_widget_key()
    last_toolbar_page = str(st.session_state.get(_LAST_TOOLBAR_PAGE_KEY) or "")
    widget_lang = st.session_state.get(widget_key)
    if widget_lang not in _SUPPORTED_LANGUAGES or last_toolbar_page != current_page:
        st.session_state[widget_key] = current_lang
    selected_lang = st.selectbox(
        tx("Language", "语言"),
        options=list(_SUPPORTED_LANGUAGES),
        format_func=_language_option_label,
        key=widget_key,
        label_visibility="collapsed",
        help=tx("Switch UI language", "切换界面语言"),
    )
    st.session_state[_LAST_TOOLBAR_PAGE_KEY] = current_page
    if selected_lang in _SUPPORTED_LANGUAGES and selected_lang != current_lang:
        st.session_state[_LANGUAGE_SESSION_KEY] = selected_lang
        _persist_language(selected_lang)
        st.rerun()


def render_sidebar_navigation(current_page: str | None = None) -> None:
    """Render the custom translated sidebar navigation."""
    with st.sidebar:
        st.markdown(f"### {tx('Navigation', '导航')}")
        for page in _PAGES:
            st.page_link(
                page.path,
                label=page_title(page.key),
                icon=page.icon,
                disabled=page.key == current_page,
            )


__all__ = [
    "get_language",
    "page_title",
    "render_language_toolbar",
    "render_sidebar_navigation",
    "run_status_label",
    "set_current_page",
    "tx",
]
