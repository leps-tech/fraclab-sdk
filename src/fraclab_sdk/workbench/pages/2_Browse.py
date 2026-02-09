"""Browse snapshot data page."""

import json
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from fraclab_sdk.config import SDKConfig
from fraclab_sdk.snapshot import SnapshotLibrary
from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.workbench.utils import format_snapshot_option

st.set_page_config(page_title="Browse", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
st.title("Browse")

ui_styles.apply_global_styles()

# --- Page-Specific CSS ---
st.markdown("""
<style>
    /* =========================================
       1. DATASET TABS (一级导航 - 专业卡片风格)
       ========================================= */
    
    /* Tabs 容器布局 */
    [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 1.5rem;
        padding: 4px 2px; /* 给阴影和动画留出空间，防被切 */
        border-bottom: none !important; /* 去掉一级 tabs 下方多余分割线 */
    }

    /* BaseWeb 默认 tab 底部分割线（会出现在一级 tabs 下方） */
    [data-baseweb="tab-border"] {
        display: none !important;
        height: 0 !important;
        border: 0 !important;
    }

    /* 默认卡片样式 (未选中) */
    [data-baseweb="tab"] {
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
    config = SDKConfig()
    return SnapshotLibrary(config)


def render_dataset_panel(snapshot, dataset_info: dict[str, Any]) -> None:
    """Render explorer and file preview for a single dataset."""
    dataset_key = dataset_info["dataset_key"]
    layout = dataset_info["layout"]

    # Items Explorer
    items = snapshot.get_items(dataset_key)
    if not items:
        st.info("No items in this dataset")
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
        tab_table, tab_cards = st.tabs(["📊 Table View", "📝 Detail Cards"])
        
        with tab_table:
            st.markdown(f"<small style='color:#666'>Showing items {start_idx + 1}-{end_idx} of {total_items}</small>", unsafe_allow_html=True)
            if item_dicts:
                df = pd.DataFrame(item_dicts)
                cols = df.columns.tolist()
                if "_index" in cols:
                    cols.insert(0, cols.pop(cols.index("_index")))
                    df = df[cols]
                _render_static_table(df)
            else:
                st.warning("No data to display.")

        with tab_cards:
            st.markdown(f"<small style='color:#666'>Showing items {start_idx + 1}-{end_idx} of {total_items}</small>", unsafe_allow_html=True)
            for real_idx, item_obj in current_items_slice:
                with st.expander(f"Item {real_idx}", expanded=False):
                    try:
                        json_str = json.dumps(item_obj.model_dump(exclude_none=True), indent=2, ensure_ascii=False)
                        st.code(json_str, language="json")
                    except AttributeError:
                        st.text(str(item_obj))

                    if layout == "object_ndjson_lines":
                        if st.button(f"Load Data #{real_idx}", key=f"btn_load_ndjson_{real_idx}_{dataset_key}"):
                            try:
                                data = snapshot.read_object_line(dataset_key, real_idx)
                                st.info("Data Content:")
                                st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")
                            except Exception as e:
                                st.error(f"Error: {e}")
                    elif layout == "frame_parquet_item_dirs":
                        try:
                            files = snapshot.read_frame_parts(dataset_key, real_idx)
                            if files:
                                st.markdown("**Parquet Files:**")
                                for f in files:
                                    st.code(f.name, language="text")
                            else:
                                st.caption("No files found.")
                        except Exception as e:
                            st.error(f"Error: {e}")

        if total_pages > 1:
            center_cols = st.columns([1, 8, 1])
            with center_cols[1]:
                _render_pagination(page, total_pages, page_key)

    st.divider()
    
    # Files Preview
    st.subheader("Raw Data Files")
    manifest = snapshot.manifest
    manifest_ds = manifest.datasets.get(dataset_key)
    data_root = manifest.dataRoot or "data"
    dataset_dir = snapshot.directory / data_root / dataset_key
    resolved_layout = layout or (manifest_ds.layout if manifest_ds else None) or _detect_layout(dataset_dir)

    if resolved_layout == "object_ndjson_lines":
        ndjson_path = dataset_dir / "object.ndjson"
        if ndjson_path.exists():
            total_count = manifest_ds.count if manifest_ds else dataset_info["item_count"]
            st.caption(f"File: {ndjson_path} (count: {total_count})")

            page_size = 10
            total_pages = (total_count + page_size - 1) // page_size or 1
            ndjson_page_key = f"ndjson_preview_page_{dataset_key}"
            cp = st.session_state.get(f"{ndjson_page_key}_current", st.session_state.get(ndjson_page_key, 1))
            if cp > total_pages:
                cp = 1

            start = (cp - 1) * page_size
            limit = page_size
            st.text(f"Lines {start + 1}-{min(start + limit, total_count)}")

            lines_data = _read_ndjson_slice(ndjson_path, start, limit)
            for line_idx, obj in lines_data:
                with st.expander(f"Line {line_idx}", expanded=True):
                    st.code(json.dumps(obj, indent=2, ensure_ascii=False), language="json")

            if total_pages > 1:
                center_cols = st.columns([1, 8, 1])
                with center_cols[1]:
                    _render_pagination(cp, total_pages, ndjson_page_key)
        else:
            st.warning(f"File not found: {ndjson_path}")

    elif resolved_layout == "frame_parquet_item_dirs":
        parquet_dir = dataset_dir / "parquet"
        search_dir = parquet_dir if parquet_dir.exists() else dataset_dir

        if search_dir.exists():
            st.caption(f"Searching Parquet in: {search_dir}")
            files = sorted(search_dir.rglob("*.parquet"))
            if not files:
                st.info("No parquet files found.")
            else:
                page_size = 10
                total_pages = (len(files) + page_size - 1) // page_size or 1
                file_page_key = f"parquet_file_page_{dataset_key}"
                cp_files = st.session_state.get(f"{file_page_key}_current", 1)
                if cp_files > total_pages:
                    cp_files = 1

                start = (cp_files - 1) * page_size
                page_files = files[start : start + page_size]

                with st.container(border=True):
                    st.markdown(f"**Parquet Files (Page {cp_files}/{total_pages})**")
                    for f in page_files:
                        st.text(f" {f.relative_to(dataset_dir)}")

                if total_pages > 1:
                    center_cols = st.columns([1, 8, 1])
                    with center_cols[1]:
                        _render_pagination(cp_files, total_pages, file_page_key)

                st.divider()
                options = [f.relative_to(dataset_dir) for f in files]
                select_key = f"parquet_file_select_{dataset_key}"
                selected_rel = st.selectbox("Select file to preview content", options=options, key=select_key)
                sample_path = dataset_dir / selected_rel
                st.markdown(f"#### Preview: `{sample_path.name}`")

                try:
                    import pyarrow.parquet as pq

                    table = pq.read_table(sample_path)
                    total_rows = table.num_rows
                    if total_rows == 0:
                        st.warning("Empty file.")
                    else:
                        row_page_size = 20
                        row_total_pages = (total_rows + row_page_size - 1) // row_page_size or 1
                        preview_page_key = f"pq_view_{dataset_key}_{str(selected_rel)}"
                        cp_row = st.session_state.get(f"{preview_page_key}_current", 1)
                        if cp_row > row_total_pages:
                            cp_row = 1

                        start_r = (cp_row - 1) * row_page_size
                        table_slice = table.slice(start_r, row_page_size)
                        cols = table_slice.column_names
                        data_dict = table_slice.to_pydict()
                        rows = [{col: data_dict[col][i] for col in cols} for i in range(table_slice.num_rows)]
                        df_rows = pd.DataFrame(rows)

                        for col in df_rows.columns:
                            if pd.api.types.is_datetime64_any_dtype(df_rows[col]):
                                try:
                                    df_rows[col] = df_rows[col].dt.round("1s")
                                    if df_rows[col].dt.tz is not None:
                                        df_rows[col] = df_rows[col].dt.tz_localize(None)
                                except Exception:
                                    pass

                        _render_static_table(df_rows)
                        if row_total_pages > 1:
                            st.caption(f"Page {cp_row} of {row_total_pages} ({total_rows} rows)")
                            center_cols = st.columns([1, 8, 1])
                            with center_cols[1]:
                                _render_pagination(cp_row, row_total_pages, preview_page_key)

                except ImportError:
                    st.error("pyarrow not installed.")
                except Exception as e:
                    st.error(f"Failed to read parquet: {e}")
        else:
            st.warning(f"Parquet directory not found: {search_dir}")

    st.divider()
    with st.expander("Show DRS (Data Requirement Specification)"):
        try:
            drs = snapshot.drs
            st.code(json.dumps(drs.model_dump(exclude_none=True), indent=2, ensure_ascii=False), language="json")
        except Exception as e:
            st.error(f"Failed to load DRS: {e}")


# --- Main Logic ---

snapshot_lib = get_library()
snapshots = snapshot_lib.list_snapshots()

if not snapshots:
    st.info("No snapshots available. Import a snapshot first.")
    st.stop()

# 1. Select Snapshot
snapshot_options = {s.snapshot_id: s for s in snapshots}
selected_id = st.selectbox(
    "Select Snapshot",
    options=list(snapshot_options.keys()),
    format_func=lambda x: format_snapshot_option(snapshot_options[x]),
)

if not selected_id:
    st.stop()

snapshot = snapshot_lib.get_snapshot(selected_id)

st.divider()

# 2. Dataset Tabs
st.subheader("Datasets")
datasets = snapshot.get_datasets()

if not datasets:
    st.info("No datasets in this snapshot")
    st.stop()

dataset_options = {d["dataset_key"]: d for d in datasets}
dataset_keys = list(dataset_options.keys())

# Create the Primary Tabs (Styles applied via CSS)
dataset_tabs = st.tabs([f"{k} ({dataset_options[k]['item_count']})" for k in dataset_keys])

for tab, key in zip(dataset_tabs, dataset_keys):
    with tab:
        render_dataset_panel(snapshot, dataset_options[key])
