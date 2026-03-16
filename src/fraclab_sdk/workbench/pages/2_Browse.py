"""Browse snapshot data page."""

import html
import json
import threading
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from fraclab_sdk.config import SDKConfig
from fraclab_sdk.snapshot import SnapshotLibrary
from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.workbench.i18n import page_title, tx
from fraclab_sdk.workbench.page_state import choose_valid_option, read_page_state, write_page_state
from fraclab_sdk.workbench.parquet_preview import (
    ParquetPreviewCancelled,
    build_parquet_preview_figure,
    build_parquet_preview_from_files,
)
from fraclab_sdk.workbench.utils import format_snapshot_option

st.set_page_config(
    page_title=page_title("browse"),
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui_styles.apply_global_styles("browse")
ui_styles.render_page_header(tx("Browse", "数据浏览"))

_WORKBENCH_CONFIG = SDKConfig()
_PARQUET_PREVIEW_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="browse-parquet-preview")
_BROWSE_PAGE_STATE_CACHE_KEY = "_browse_page_state_cache"

# --- Page-Specific CSS ---
st.markdown("""
<style>
    /* =========================================
       1. DATASET TABS (一级导航 - 专业卡片风格)
       ========================================= */
    
    /* Tabs 容器布局 */
    [data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 8px;
        margin-bottom: 1.5rem;
        padding: 4px 2px; /* 给阴影和动画留出空间，防被切 */
        border-bottom: none !important; /* 去掉一级 tabs 下方多余分割线 */
        overflow-x: auto !important;
        overflow-y: hidden !important;
        scrollbar-width: thin;
    }

    /* BaseWeb 默认 tab 底部分割线（会出现在一级 tabs 下方） */
    [data-baseweb="tab-border"] {
        display: none !important;
        height: 0 !important;
        border: 0 !important;
    }

    /* 默认卡片样式 (未选中) */
    [data-baseweb="tab"] {
        flex: 0 0 auto !important;
        white-space: nowrap !important;
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important; /*稍微圆润一点*/
        color: #64748b !important;
        padding: 0.6rem 1.2rem !important; /* 增加点击区域 */
        font-weight: 500 !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-right: 4px !important;
        position: relative !important;
    }

    /* 悬停交互：微上浮 + 阴影加深 (增加可点击感) */
    [data-baseweb="tab"]:hover {
        border-color: #cbd5e1 !important;
        color: #334155 !important;
        background-color: #ffffff !important;
        transform: translateY(-2px); /* 关键：微动效 */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08) !important;
        z-index: 10;
    }

    /* 选中状态：顶部蓝条强调 + 文字加黑 */
    [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff !important;
        border-color: #e2e8f0 !important;
        /* 核心设计：顶部 3px 蓝条 */
        border-top: 3px solid #2563eb !important; 
        /* 稍微调整 padding 以补偿边框高度变化 */
        padding-top: calc(0.6rem - 3px) !important; 
        
        color: #1e293b !important; /* 深黑字 */
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    }

    /* =========================================
       2. ITEM EXPLORER TABS (二级导航 - 保持简约下划线)
       ========================================= */
    
    /* 覆盖二级 Tabs 容器 */
    [data-baseweb="tab-panel"] [data-baseweb="tab-list"] {
        gap: 24px !important;
        border-bottom: 2px solid #f1f5f9;
        margin-bottom: 1rem;
        background-color: transparent !important;
        padding: 0 !important;
        overflow-x: visible !important;
    }

    /* 覆盖二级 Tab 样式：去卡片化 */
    [data-baseweb="tab-panel"] [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0.5rem 0.25rem !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        margin: 0 !important;
        transform: none !important; /* 禁用上浮 */
    }

    /* 二级 Tab 悬停 */
    [data-baseweb="tab-panel"] [data-baseweb="tab"]:hover {
        color: #2563eb !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }

    /* 二级 Tab 选中：下划线 */
    [data-baseweb="tab-panel"] [data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb !important;
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid #2563eb !important;
        padding-top: 0.5rem !important; /* 恢复 padding */
        box-shadow: none !important;
    }

    /* =========================================
       3. OTHER UTILS
       ========================================= */

    /* Hide download button */
    [data-testid="stDownloadButton"] {
        display: none !important;
    }

    /* Pagination button styling */
    div[data-testid="stButton"] button {
        padding: 0.25rem 0.75rem !important;
        min-width: 40px !important;
    }

    /* Pagination ellipsis */
    .pagination-ellipsis {
        text-align: center;
        line-height: 2.3rem;
        color: #888;
        font-weight: bold;
    }

    /* Custom Static Table */
    .table-wrapper {
        max-height: 500px;
        overflow: auto;
        border: 1px solid #e6e9ef;
        border-radius: 6px;
        margin-bottom: 1rem;
        background-color: white;
    }

    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-family: "Source Sans Pro", sans-serif;
        font-size: 14px;
        color: #31333F;
        user-select: none !important;
    }

    .custom-table th {
        position: sticky;
        top: 0;
        background-color: #f8fafc;
        color: #475569;
        z-index: 2;
        padding: 10px 12px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #e2e8f0;
        white-space: nowrap;
    }

    .custom-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #f1f5f9;
        white-space: nowrap;
        vertical-align: middle;
        color: #334155;
    }

    .custom-table tr:nth-child(even) {
        background-color: #fcfcfc;
    }

    .custom-table tr:hover {
        background-color: #f1f5f9;
    }

    .file-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 8px;
        max-height: 220px;
        overflow-y: auto;
    }

    .file-chip {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 10px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 12px;
        color: #334155;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)


# --- Utils & Components ---

def _render_static_table(df: pd.DataFrame):
    """Renders a static HTML table to replace st.dataframe."""
    df_display = df.fillna("")
    html_table = df_display.to_html(index=False, classes="custom-table", border=0, escape=True)
    st.markdown(f'<div class="table-wrapper">{html_table}</div>', unsafe_allow_html=True)


def _read_ndjson_slice(path, start: int, limit: int) -> list[tuple[int, dict]]:
    """Read a slice of ndjson lines [start, start+limit)."""
    results: list[tuple[int, dict]] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if len(results) >= limit:
                break
            try:
                results.append((i, json.loads(line)))
            except Exception:
                results.append((i, {"_error": "Failed to parse line", "raw": line.strip()}))
    return results


def _render_pagination(current: int, total: int, key_prefix: str) -> int:
    """Render a compact, centered pagination bar."""
    if f"{key_prefix}_current" not in st.session_state:
        st.session_state[f"{key_prefix}_current"] = current
    
    display_current = int(st.session_state.get(f"{key_prefix}_current", current))
    clicked = False

    def _page_buttons(cur: int) -> Iterable[int | str]:
        if total <= 9:
            return list(range(1, total + 1))
        window = [cur - 1, cur, cur + 1]
        window = [p for p in window if 1 <= p <= total]
        pages = [1, 2] + window + [total - 1, total]
        pages = sorted(set(pages))
        display = []
        last = None
        for p in pages:
            if last and p - last > 1:
                display.append("…")
            display.append(p)
            last = p
        return display

    buttons = list(_page_buttons(display_current))
    
    st.markdown("---") 
    
    num_slots = len(buttons) + 2
    spacer_ratio = 6 if num_slots < 6 else 1.5
    col_ratios = [spacer_ratio] + [1] * num_slots + [spacer_ratio]
    
    cols = st.columns(col_ratios, gap="small")
    action_cols = cols[1:-1]
    
    chosen = display_current

    if action_cols[0].button("‹", key=f"{key_prefix}_prev", disabled=display_current <= 1):
        chosen = max(1, display_current - 1)
        clicked = True

    for idx, p in enumerate(buttons, start=1):
        if p == "…":
            action_cols[idx].markdown("<div class='pagination-ellipsis'>…</div>", unsafe_allow_html=True)
            continue
        
        if action_cols[idx].button(
            f"{p}",
            key=f"{key_prefix}_page_{p}",
            type="primary" if p == display_current else "secondary",
        ):
            chosen = p
            clicked = True

    if action_cols[-1].button("›", key=f"{key_prefix}_next", disabled=display_current >= total):
        chosen = min(total, display_current + 1)
        clicked = True

    st.session_state[f"{key_prefix}_current"] = chosen
    
    if clicked:
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
    return chosen


def _detect_layout(dir_path) -> str | None:
    if (dir_path / "object.ndjson").exists():
        return "object_ndjson_lines"
    if (dir_path / "parquet").exists():
        return "frame_parquet_item_dirs"
    return None


def get_library():
    return SnapshotLibrary(_WORKBENCH_CONFIG)


def _get_browse_page_state() -> dict[str, Any]:
    """Return the cached persisted state for the Browse page."""
    page_state = st.session_state.get(_BROWSE_PAGE_STATE_CACHE_KEY)
    if isinstance(page_state, dict):
        return page_state
    page_state = read_page_state("browse", _WORKBENCH_CONFIG)
    st.session_state[_BROWSE_PAGE_STATE_CACHE_KEY] = page_state
    return page_state


def _write_browse_page_state(page_state: dict[str, Any]) -> None:
    """Persist and cache Browse page state."""
    st.session_state[_BROWSE_PAGE_STATE_CACHE_KEY] = page_state
    write_page_state("browse", page_state, _WORKBENCH_CONFIG)


def _get_browse_snapshot_state(snapshot_id: str) -> dict[str, Any]:
    """Return persisted Browse state for one snapshot."""
    page_state = _get_browse_page_state()
    snapshots = page_state.get("snapshots")
    if not isinstance(snapshots, dict):
        return {}
    snapshot_state = snapshots.get(snapshot_id)
    return snapshot_state if isinstance(snapshot_state, dict) else {}


def _update_browse_snapshot_state(snapshot_id: str, **updates: Any) -> None:
    """Persist Browse state for one snapshot."""
    page_state = dict(_get_browse_page_state())
    snapshots = dict(page_state.get("snapshots") or {})
    snapshot_state = dict(snapshots.get(snapshot_id) or {})
    snapshot_state.update(updates)
    snapshots[snapshot_id] = snapshot_state
    page_state["selected_snapshot_id"] = snapshot_id
    page_state["snapshots"] = snapshots
    _write_browse_page_state(page_state)


def _get_preview_jobs() -> dict[str, dict[str, Any]]:
    jobs = st.session_state.get("_browse_parquet_preview_jobs")
    if isinstance(jobs, dict):
        return jobs
    jobs = {}
    st.session_state["_browse_parquet_preview_jobs"] = jobs
    return jobs


def _preview_request_key(parquet_files: list[Path]) -> tuple[str, ...]:
    """Build a stable signature for the current parquet preview request."""
    return tuple(str(path) for path in parquet_files)


def _build_preview_job_payload(parquet_files: tuple[Path, ...], cancel_event: threading.Event) -> dict[str, Any]:
    """Run parquet preview generation in a background worker."""
    try:
        traces, x_range, _, x_is_time = build_parquet_preview_from_files(
            list(parquet_files),
            max_points=800,
            should_cancel=cancel_event.is_set,
        )
    except ParquetPreviewCancelled:
        return {"kind": "cancelled"}
    return {"kind": "ready", "traces": traces, "x_range": x_range, "x_is_time": x_is_time}


def _cancel_preview_job(job_key: str) -> None:
    """Mark a background preview job as cancelled and drop finished jobs."""
    jobs = _get_preview_jobs()
    job = jobs.get(job_key)
    if not isinstance(job, dict):
        return
    cancel_event = job.get("cancel_event")
    if cancel_event is not None:
        cancel_event.set()
    future = job.get("future")
    if isinstance(future, Future) and future.done():
        jobs.pop(job_key, None)


def _prune_preview_jobs(active_keys: set[str]) -> None:
    """Cancel any parquet preview jobs that are no longer visible."""
    jobs = _get_preview_jobs()
    for job_key in list(jobs.keys()):
        if job_key in active_keys:
            continue
        _cancel_preview_job(job_key)
        future = jobs.get(job_key, {}).get("future")
        if isinstance(future, Future) and future.done():
            jobs.pop(job_key, None)


def _ensure_preview_job(parquet_files: list[Path], key_prefix: str) -> None:
    """Submit a background preview job when the current request has no live worker."""
    jobs = _get_preview_jobs()
    request_key = _preview_request_key(parquet_files)
    existing = jobs.get(key_prefix)
    if isinstance(existing, dict) and existing.get("request_key") == request_key:
        future = existing.get("future")
        cancel_event = existing.get("cancel_event")
        if isinstance(future, Future) and future.done():
            return
        if isinstance(future, Future) and cancel_event is not None and not cancel_event.is_set():
            return

    if existing is not None:
        _cancel_preview_job(key_prefix)

    cancel_event = threading.Event()
    jobs[key_prefix] = {
        "request_key": request_key,
        "cancel_event": cancel_event,
        "future": _PARQUET_PREVIEW_EXECUTOR.submit(
            _build_preview_job_payload,
            tuple(parquet_files),
            cancel_event,
        ),
    }


def _resolve_preview_job_state(parquet_files: list[Path], key_prefix: str) -> tuple[str, dict[str, Any] | None]:
    """Resolve the current state of a parquet preview background job."""
    _ensure_preview_job(parquet_files, key_prefix)
    jobs = _get_preview_jobs()
    job = jobs.get(key_prefix)
    if not isinstance(job, dict):
        return "pending", None

    future = job.get("future")
    if not isinstance(future, Future) or not future.done():
        return "pending", None

    try:
        payload = future.result()
    except Exception as error:
        return "error", {"error": error}

    if payload.get("kind") == "cancelled":
        jobs.pop(key_prefix, None)
        return "cancelled", None

    return "ready", payload


def _list_item_parquet_files(snapshot, dataset_key: str, item_index: int) -> list[Path]:
    """List parquet files for an item by directory, without trusting ds layout metadata."""
    item_dir = snapshot.get_item_dir(dataset_key, item_index)
    if not item_dir.exists():
        return []
    return sorted(item_dir.rglob("*.parquet"))


def _item_label(item_index: int, item_obj: Any) -> str:
    """Build a compact label for an item selector."""
    owner = getattr(item_obj, "owner", None)
    if isinstance(owner, dict):
        for key in ("stageId", "wellId", "platformId"):
            value = owner.get(key)
            if value:
                return tx("Item {index} · {owner}", "条目 {index} · {owner}", index=item_index, owner=value)
    if hasattr(item_obj, "owner") and hasattr(item_obj.owner, "get"):
        for key in ("stageId", "wellId", "platformId"):
            value = item_obj.owner.get(key)
            if value:
                return tx("Item {index} · {owner}", "条目 {index} · {owner}", index=item_index, owner=value)
    return tx("Item {index}", "条目 {index}", index=item_index)


def _render_item_detail(item_index: int, item_obj: Any) -> None:
    """Render JSON detail for a selected item."""
    with st.expander(tx("Selected Item Metadata", "已选条目元数据"), expanded=False):
        try:
            st.code(json.dumps(item_obj.model_dump(exclude_none=True), indent=2, ensure_ascii=False), language="json")
        except AttributeError:
            st.code(json.dumps({"raw": str(item_obj)}, indent=2, ensure_ascii=False), language="json")


def _render_parquet_file_grid(parquet_files: list[Path], item_dir: Path) -> None:
    """Render a compact multi-column parquet file list."""
    chips = "".join(
        f"<div class='file-chip' title='{html.escape(str(path.relative_to(item_dir)))}'>"
        f"{html.escape(str(path.relative_to(item_dir)))}</div>"
        for path in parquet_files
    )
    st.markdown(f"<div class='file-grid'>{chips}</div>", unsafe_allow_html=True)


def _emit_hidden_feedback(message: str, key_prefix: str) -> None:
    """Persist a hidden feedback message without showing a visible error."""
    st.session_state[f"{key_prefix}_hidden_feedback"] = message
    st.markdown(
        f"<div style='display:none' data-feedback-key='{html.escape(key_prefix)}'>{html.escape(message)}</div>",
        unsafe_allow_html=True,
    )


def _render_preview_loading_state() -> None:
    """Render a persistent loading state for the asynchronous chart preview."""
    st.status(tx("Loading visualization...", "正在加载可视化..."), state="running", expanded=False)


def _render_preview_polling_fragment(parquet_files: list[Path], key_prefix: str) -> None:
    """Poll a background parquet preview job until it settles."""

    @st.fragment(run_every="750ms")
    def _poll_preview() -> None:
        state, _ = _resolve_preview_job_state(parquet_files, key_prefix)
        if state == "pending":
            _render_preview_loading_state()
            return
        st.rerun()

    _poll_preview()


def _render_parquet_table_preview(sample_path: Path, key_prefix: str) -> None:
    """Render a paginated table preview for a parquet file."""
    import pyarrow.parquet as pq

    table = pq.read_table(sample_path)
    total_rows = table.num_rows
    if total_rows == 0:
        st.warning(tx("Empty file.", "空文件。"))
        return

    row_page_size = 20
    row_total_pages = (total_rows + row_page_size - 1) // row_page_size or 1
    current_page = st.session_state.get(f"{key_prefix}_current", 1)
    if current_page > row_total_pages:
        current_page = 1

    start_row = (current_page - 1) * row_page_size
    table_slice = table.slice(start_row, row_page_size)
    data_dict = table_slice.to_pydict()
    rows = [{column: data_dict[column][idx] for column in table_slice.column_names} for idx in range(table_slice.num_rows)]
    df_rows = pd.DataFrame(rows)

    for column in df_rows.columns:
        if pd.api.types.is_datetime64_any_dtype(df_rows[column]):
            try:
                df_rows[column] = df_rows[column].dt.round("1s")
                if df_rows[column].dt.tz is not None:
                    df_rows[column] = df_rows[column].dt.tz_localize(None)
            except Exception:
                continue

    _render_static_table(df_rows)
    if row_total_pages > 1:
        st.caption(
            tx(
                "Page {page} of {total} ({rows} rows)",
                "第 {page}/{total} 页（{rows} 行）",
                page=current_page,
                total=row_total_pages,
                rows=total_rows,
            )
        )
        center_cols = st.columns([1, 8, 1])
        with center_cols[1]:
            _render_pagination(current_page, row_total_pages, key_prefix)


def _render_parquet_visualization(parquet_files: list[Path], key_prefix: str) -> None:
    """Render a parquet preview chart with cancellable background loading."""
    _prune_preview_jobs({key_prefix})
    state, payload = _resolve_preview_job_state(parquet_files, key_prefix)
    if state == "cancelled":
        state, payload = _resolve_preview_job_state(parquet_files, key_prefix)

    if state == "pending":
        st.session_state.pop(f"{key_prefix}_hidden_feedback", None)
        _render_preview_polling_fragment(parquet_files, key_prefix)
        return

    if state == "error":
        error = payload.get("error") if isinstance(payload, dict) else "unknown error"
        _emit_hidden_feedback(f"Visualization failed: {error}", key_prefix)
        return

    traces = payload.get("traces") if isinstance(payload, dict) else None
    x_range = payload.get("x_range") if isinstance(payload, dict) else None
    x_is_time = bool(payload.get("x_is_time")) if isinstance(payload, dict) else False
    if not traces:
        st.session_state.pop(f"{key_prefix}_hidden_feedback", None)
        st.caption(
            tx(
                "No plottable numeric/time series found in the selected item's parquet files.",
                "所选条目下的 parquet 文件里没有可绘制的时间序列或数值列。",
            )
        )
        return

    figure = build_parquet_preview_figure(
        traces,
        x_range=x_range,
        x_is_time=x_is_time,
        height=360,
        legend_only_interaction=True,
    )
    st.session_state.pop(f"{key_prefix}_hidden_feedback", None)
    st.plotly_chart(
        figure,
        key=f"{key_prefix}_chart",
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
            "scrollZoom": False,
            "doubleClick": False,
            "showTips": False,
        },
    )


def _render_parquet_item_files(snapshot_id: str, snapshot, dataset_key: str, items: list[tuple[int, Any]]) -> None:
    """Render item-focused parquet file inspection."""
    if not items:
        _prune_preview_jobs(set())
        st.info(tx("No parquet items available.", "没有可用的 parquet 条目。"))
        return

    item_lookup = {item_index: item_obj for item_index, item_obj in items}
    default_item_index = next(iter(item_lookup))
    item_widget_key = f"browse_item_select_{dataset_key}"
    saved_snapshot_state = _get_browse_snapshot_state(snapshot_id)
    saved_item_index = (saved_snapshot_state.get("item_indices") or {}).get(dataset_key)
    restored_item_index = choose_valid_option(list(item_lookup.keys()), saved_item_index, default_item_index)
    if st.session_state.get(item_widget_key) not in item_lookup and restored_item_index is not None:
        st.session_state[item_widget_key] = restored_item_index

    selected_item_index = st.selectbox(
        tx("Select item", "选择条目"),
        options=list(item_lookup.keys()),
        format_func=lambda idx: _item_label(idx, item_lookup[idx]),
        key=item_widget_key,
    )
    item_indices = dict(saved_snapshot_state.get("item_indices") or {})
    item_indices[dataset_key] = selected_item_index
    _update_browse_snapshot_state(snapshot_id, item_indices=item_indices)
    item_obj = item_lookup.get(selected_item_index, item_lookup[default_item_index])
    _render_item_detail(selected_item_index, item_obj)
    item_dir = snapshot.get_item_dir(dataset_key, selected_item_index)
    preview_key = f"browse_parquet_plot_{dataset_key}_{selected_item_index}"
    _prune_preview_jobs({preview_key})

    try:
        parquet_files = _list_item_parquet_files(snapshot, dataset_key, selected_item_index)
    except Exception as error:
        st.error(tx("Failed to list parquet files: {error}", "列出 parquet 文件失败：{error}", error=error))
        return

    if not parquet_files:
        st.info(tx("No parquet files found for this item.", "这个条目下没有 parquet 文件。"))
        return

    st.caption(
        tx(
            "Showing {count} file(s) for item {item}.",
            "展示条目 {item} 下的 {count} 个文件。",
            count=len(parquet_files),
            item=selected_item_index,
        )
    )
    with st.container(border=True):
        _render_parquet_file_grid(parquet_files, item_dir)

    st.markdown(tx("#### Visualization", "#### 可视化"))
    st.caption(
        tx(
            "Preview merged from all parquet files under the selected item.",
            "预览基于当前所选条目下的全部 parquet 文件合并生成。",
        )
    )
    _render_parquet_visualization(
        parquet_files,
        key_prefix=preview_key,
    )

    file_widget_key = f"browse_parquet_file_select_{dataset_key}_{selected_item_index}"
    saved_file_map = dict(saved_snapshot_state.get("selected_files") or {})
    file_state_key = f"{dataset_key}::{selected_item_index}"
    saved_relative_file = saved_file_map.get(file_state_key)
    restored_file = None
    if isinstance(saved_relative_file, str):
        restored_file = next((path for path in parquet_files if str(path.relative_to(item_dir)) == saved_relative_file), None)
    if st.session_state.get(file_widget_key) not in parquet_files:
        st.session_state[file_widget_key] = restored_file or parquet_files[0]

    selected_file = st.selectbox(
        tx("Select parquet file", "选择 parquet 文件"),
        options=parquet_files,
        format_func=lambda path: str(path.relative_to(item_dir)),
        key=file_widget_key,
    )
    saved_file_map[file_state_key] = str(selected_file.relative_to(item_dir))
    _update_browse_snapshot_state(snapshot_id, selected_files=saved_file_map)

    st.markdown(tx("#### Table Preview: `{name}`", "#### 表格预览：`{name}`", name=selected_file.name))
    st.caption(tx("Path: `{path}`", "路径：`{path}`", path=selected_file))
    _render_parquet_table_preview(
        selected_file,
        key_prefix=f"browse_parquet_rows_{dataset_key}_{selected_item_index}_{selected_file.relative_to(item_dir)}",
    )


def render_dataset_panel(snapshot_id: str, snapshot, dataset_info: dict[str, Any]) -> None:
    """Render explorer and file preview for a single dataset."""
    dataset_key = dataset_info["key"]
    layout = dataset_info["layout"]
    manifest = snapshot.manifest
    manifest_ds = manifest.datasets.get(dataset_key)
    data_root = manifest.dataRoot or "data"
    dataset_dir = snapshot.directory / data_root / dataset_key
    resolved_layout = layout or (manifest_ds.layout if manifest_ds else None) or _detect_layout(dataset_dir)

    # Items Explorer
    items = snapshot.get_items(dataset_key)
    if not items:
        st.info(tx("No items in this dataset", "该数据集没有条目。"))
    else:
        items_per_page = 20
        total_items = len(items)
        total_pages = (total_items + items_per_page - 1) // items_per_page

        page_key = f"items_page_{dataset_key}"
        page = st.session_state.get(f"{page_key}_current", st.session_state.get(page_key, 1))

        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        current_items_slice = items[start_idx:end_idx]

        item_dicts = []
        for real_idx, item_obj in current_items_slice:
            try:
                d = item_obj.model_dump(exclude_none=True)
                d["_index"] = real_idx
                item_dicts.append(d)
            except AttributeError:
                item_dicts.append({"_index": real_idx, "raw": str(item_obj)})

        # Secondary Tabs (Table vs Cards) - Styles handled by CSS (Underline Style)
        tab_table, tab_cards = st.tabs([tx("📊 Table View", "📊 表格视图"), tx("📝 Detail Cards", "📝 详情卡片")])

        with tab_table:
            st.markdown(
                f"<small style='color:#666'>{tx('Showing items {start}-{end} of {total}', '显示条目 {start}-{end} / 共 {total}', start=start_idx + 1, end=end_idx, total=total_items)}</small>",
                unsafe_allow_html=True,
            )
            if item_dicts:
                df = pd.DataFrame(item_dicts)
                cols = df.columns.tolist()
                if "_index" in cols:
                    cols.insert(0, cols.pop(cols.index("_index")))
                    df = df[cols]
                _render_static_table(df)
            else:
                st.warning(tx("No data to display.", "没有可显示的数据。"))

        with tab_cards:
            st.markdown(
                f"<small style='color:#666'>{tx('Showing items {start}-{end} of {total}', '显示条目 {start}-{end} / 共 {total}', start=start_idx + 1, end=end_idx, total=total_items)}</small>",
                unsafe_allow_html=True,
            )
            for real_idx, item_obj in current_items_slice:
                with st.expander(tx("Item {index}", "条目 {index}", index=real_idx), expanded=False):
                    try:
                        json_str = json.dumps(item_obj.model_dump(exclude_none=True), indent=2, ensure_ascii=False)
                        st.code(json_str, language="json")
                    except AttributeError:
                        st.text(str(item_obj))

                    if layout == "object_ndjson_lines":
                        if st.button(tx("Load Data #{index}", "加载数据 #{index}", index=real_idx), key=f"btn_load_ndjson_{real_idx}_{dataset_key}"):
                            try:
                                data = snapshot.read_object_line(dataset_key, real_idx)
                                st.info(tx("Data Content:", "数据内容："))
                                st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")
                            except Exception as e:
                                st.error(tx("Error: {error}", "错误：{error}", error=e))
                    elif resolved_layout == "frame_parquet_item_dirs":
                        try:
                            files = _list_item_parquet_files(snapshot, dataset_key, real_idx)
                            if files:
                                st.markdown(tx("**Parquet Files:**", "**Parquet 文件：**"))
                                for f in files:
                                    st.code(f.name, language="text")
                            else:
                                st.caption(tx("No files found.", "未找到文件。"))
                        except Exception as e:
                            st.error(tx("Error: {error}", "错误：{error}", error=e))

        if total_pages > 1:
            center_cols = st.columns([1, 8, 1])
            with center_cols[1]:
                _render_pagination(page, total_pages, page_key)

    st.divider()
    
    # Files Preview
    st.subheader(tx("Raw Data Files", "原始数据文件"))

    if resolved_layout == "object_ndjson_lines":
        _prune_preview_jobs(set())
        ndjson_path = dataset_dir / "object.ndjson"
        if ndjson_path.exists():
            total_count = manifest_ds.count if manifest_ds else dataset_info["item_count"]
            st.caption(tx("File: {path} (count: {count})", "文件：{path}（数量：{count}）", path=ndjson_path, count=total_count))

            page_size = 10
            total_pages = (total_count + page_size - 1) // page_size or 1
            ndjson_page_key = f"ndjson_preview_page_{dataset_key}"
            cp = st.session_state.get(f"{ndjson_page_key}_current", st.session_state.get(ndjson_page_key, 1))
            if cp > total_pages:
                cp = 1

            start = (cp - 1) * page_size
            limit = page_size
            st.text(tx("Lines {start}-{end}", "行 {start}-{end}", start=start + 1, end=min(start + limit, total_count)))

            lines_data = _read_ndjson_slice(ndjson_path, start, limit)
            for line_idx, obj in lines_data:
                with st.expander(tx("Line {index}", "第 {index} 行", index=line_idx), expanded=True):
                    st.code(json.dumps(obj, indent=2, ensure_ascii=False), language="json")

            if total_pages > 1:
                center_cols = st.columns([1, 8, 1])
                with center_cols[1]:
                    _render_pagination(cp, total_pages, ndjson_page_key)
        else:
            st.warning(tx("File not found: {path}", "文件不存在：{path}", path=ndjson_path))

    elif resolved_layout == "frame_parquet_item_dirs":
        _render_parquet_item_files(snapshot_id, snapshot, dataset_key, items)

    st.divider()
    with st.expander(tx("Show DRS (Data Requirement Specification)", "显示 DRS（数据需求规范）")):
        try:
            drs = snapshot.drs
            st.code(json.dumps(drs.model_dump(exclude_none=True), indent=2, ensure_ascii=False), language="json")
        except Exception as e:
            st.error(tx("Failed to load DRS: {error}", "加载 DRS 失败：{error}", error=e))


# --- Main Logic ---

snapshot_lib = get_library()
snapshots = snapshot_lib.list_snapshots()

if not snapshots:
    st.info(tx("No snapshots available. Import a snapshot first.", "没有可用快照。请先导入快照。"))
    st.stop()

# 1. Select Snapshot
snapshot_options = {s.snapshot_id: s for s in snapshots}
browse_page_state = _get_browse_page_state()
snapshot_widget_key = "browse_snapshot_select"
restored_snapshot_id = choose_valid_option(
    list(snapshot_options.keys()),
    browse_page_state.get("selected_snapshot_id"),
)
if st.session_state.get(snapshot_widget_key) not in snapshot_options and restored_snapshot_id is not None:
    st.session_state[snapshot_widget_key] = restored_snapshot_id

selected_id = st.selectbox(
    tx("Select Snapshot", "选择快照"),
    options=list(snapshot_options.keys()),
    format_func=lambda x: format_snapshot_option(snapshot_options[x]),
    key=snapshot_widget_key,
)

if not selected_id:
    st.stop()

_update_browse_snapshot_state(selected_id)

snapshot = snapshot_lib.get_snapshot(selected_id)

st.divider()

# 2. Dataset Tabs
st.subheader(tx("Datasets", "数据集"))
datasets = snapshot.get_datasets()

if not datasets:
    st.info(tx("No datasets in this snapshot", "该快照没有数据集。"))
    st.stop()

dataset_options = {d["key"]: d for d in datasets}
dataset_keys = list(dataset_options.keys())
saved_snapshot_state = _get_browse_snapshot_state(selected_id)
default_dataset_key = choose_valid_option(
    dataset_keys,
    saved_snapshot_state.get("dataset_key"),
    st.session_state.get("browse_dataset_key"),
)
if default_dataset_key is None:
    st.stop()
if st.session_state.get("browse_dataset_key") not in dataset_options:
    st.session_state["browse_dataset_key"] = default_dataset_key

selected_dataset_key = st.pills(
    tx("Select dataset", "选择数据集"),
    options=dataset_keys,
    format_func=lambda key: f"{key} ({dataset_options[key]['itemCount']})",
    key="browse_dataset_key",
    width="stretch",
)
if selected_dataset_key not in dataset_options:
    selected_dataset_key = default_dataset_key
_update_browse_snapshot_state(selected_id, dataset_key=selected_dataset_key)
render_dataset_panel(selected_id, snapshot, dataset_options[selected_dataset_key])
