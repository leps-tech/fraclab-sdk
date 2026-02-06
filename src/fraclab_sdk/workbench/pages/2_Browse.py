"""Browse snapshot data page."""

import json
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from fraclab_sdk.config import SDKConfig
from fraclab_sdk.snapshot import SnapshotLibrary
from fraclab_sdk.workbench import ui_styles

st.set_page_config(page_title="Browse", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
st.title("Browse")

ui_styles.apply_global_styles()

# --- Page-Specific CSS ---
st.markdown("""
<style>
    /* Hide download button */
    [data-testid="stDownloadButton"] {
        display: none !important;
    }

    /* Pagination button styling (override global) */
    div[data-testid="stButton"] button {
        padding: 0.25rem 0.75rem !important;
        min-width: 40px !important;
    }

    /* Pagination ellipsis styling */
    .pagination-ellipsis {
        text-align: center;
        line-height: 2.3rem;
        color: #888;
        font-weight: bold;
    }

    /* Custom static table styling (replacement for st.dataframe) */
    .table-wrapper {
        max-height: 500px;
        overflow: auto;
        border: 1px solid #e6e9ef;
        border-radius: 0.25rem;
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

    /* Sticky table header */
    .custom-table th {
        position: sticky;
        top: 0;
        background-color: #f0f2f6;
        color: #31333F;
        z-index: 2;
        padding: 8px 12px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #e6e9ef;
        white-space: nowrap;
    }

    /* Table cell styling */
    .custom-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #f0f2f6;
        white-space: nowrap;
        vertical-align: middle;
    }

    /* Zebra striping */
    .custom-table tr:nth-child(even) {
        background-color: #f9f9fb;
    }

    /* Hover highlight */
    .custom-table tr:hover {
        background-color: #f1f3f8;
    }
</style>
""", unsafe_allow_html=True)


# --- Utils & Components ---

def _render_static_table(df: pd.DataFrame):
    """
    Renders a static HTML table to replace st.dataframe.
    Features: No menus, full content display (no truncation), sticky headers, custom styling.
    """
    # 预处理：填充 NaN 为空字符串，防止 HTML 显示 'nan'
    df_display = df.fillna("")
    
    # 转换为 HTML，不包含索引列（我们会在外面手动处理索引或不需要索引）
    # escape=True 防止 XSS，但会转义 HTML 标签
    html_table = df_display.to_html(index=False, classes="custom-table", border=0, escape=True)
    
    # 渲染容器
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
    """
    Render a compact, centered pagination bar.
    Updates session state and reruns if clicked.
    """
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
    format_func=lambda x: f"{x} ({snapshot_options[x].bundle_id})",
)

if not selected_id:
    st.stop()

snapshot = snapshot_lib.get_snapshot(selected_id)

st.divider()

# 2. Select Dataset
st.subheader("Datasets")
datasets = snapshot.get_datasets()

if not datasets:
    st.info("No datasets in this snapshot")
    st.stop()

dataset_options = {d["dataset_key"]: d for d in datasets}
selected_dataset_key = st.selectbox(
    "Select Dataset",
    options=list(dataset_options.keys()),
    format_func=lambda k: f"{k} ({dataset_options[k]['item_count']} items)",
)

if not selected_dataset_key:
    st.stop()

dataset_info = dataset_options[selected_dataset_key]

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Items", dataset_info["item_count"])
    c2.caption("Layout")
    c2.markdown(f"**{dataset_info['layout'] or 'N/A'}**")
    c3.caption("Resource Type")
    c3.markdown(f"**{dataset_info['resource_type'] or 'N/A'}**")

st.divider()

# 3. Items Explorer
st.subheader("Items Explorer")

items = snapshot.get_items(selected_dataset_key)
layout = dataset_info["layout"]

if not items:
    st.info("No items in this dataset")
else:
    # --- Pagination for Items ---
    items_per_page = 20
    total_items = len(items)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    page_key = f"items_page_{selected_dataset_key}"
    page = st.session_state.get(f"{page_key}_current", st.session_state.get(page_key, 1))
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    current_items_slice = items[start_idx:end_idx]
    
    # --- Prepare Data ---
    item_dicts = []
    if current_items_slice:
        for real_idx, item_obj in current_items_slice:
            try:
                d = item_obj.model_dump(exclude_none=True)
                d["_index"] = real_idx
                item_dicts.append(d)
            except AttributeError:
                item_dicts.append({"_index": real_idx, "raw": str(item_obj)})

    # --- View Tabs ---
    tab_table, tab_cards = st.tabs(["📊 Table View", "📝 Detail Cards"])
    
    with tab_table:
        st.markdown(f"<small>Showing items {start_idx + 1}-{end_idx} of {total_items}</small>", unsafe_allow_html=True)
        
        if item_dicts:
            df = pd.DataFrame(item_dicts)
            
            # Reorder columns
            cols = df.columns.tolist()
            if "_index" in cols:
                cols.insert(0, cols.pop(cols.index("_index")))
                df = df[cols]
            
            # 使用自定义静态表格替代 st.dataframe
            _render_static_table(df)
            
        else:
            st.warning("No data to display.")

    with tab_cards:
        st.markdown(f"<small>Showing items {start_idx + 1}-{end_idx} of {total_items}</small>", unsafe_allow_html=True)
        for real_idx, item_obj in current_items_slice:
            with st.expander(f"**Item {real_idx}**", expanded=False):
                try:
                    json_str = json.dumps(item_obj.model_dump(exclude_none=True), indent=2, ensure_ascii=False)
                    st.code(json_str, language="json")
                except AttributeError:
                    st.text(str(item_obj))

                if layout == "object_ndjson_lines":
                    if st.button(f"Load Data #{real_idx}", key=f"btn_load_ndjson_{real_idx}_{selected_dataset_key}"):
                        try:
                            data = snapshot.read_object_line(selected_dataset_key, real_idx)
                            st.info("Data Content:")
                            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")
                        except Exception as e:
                            st.error(f"Error: {e}")

                elif layout == "frame_parquet_item_dirs":
                    try:
                        files = snapshot.read_frame_parts(selected_dataset_key, real_idx)
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
            page = _render_pagination(page, total_pages, page_key)

st.divider()

# 4. Data Files Preview
st.subheader("Data Files (from data/)")

manifest = snapshot.manifest
manifest_ds = manifest.datasets.get(selected_dataset_key)
data_root = manifest.dataRoot or "data"
dataset_dir = snapshot.directory / data_root / selected_dataset_key

resolved_layout = layout or (manifest_ds.layout if manifest_ds else None) or _detect_layout(dataset_dir)

if resolved_layout == "object_ndjson_lines":
    # --- NDJSON View ---
    ndjson_path = dataset_dir / "object.ndjson"
    if ndjson_path.exists():
        total_count = manifest_ds.count if manifest_ds else dataset_info["item_count"]
        st.caption(f"File: {ndjson_path} (count: {total_count})")
        
        page_size = 10
        total_pages = (total_count + page_size - 1) // page_size or 1

        ndjson_page_key = f"ndjson_preview_page_{selected_dataset_key}"
        cp = st.session_state.get(f"{ndjson_page_key}_current", st.session_state.get(ndjson_page_key, 1))
        
        if cp > total_pages: cp = 1 

        start = (cp - 1) * page_size
        limit = page_size
        
        st.text(f"Lines {start + 1}-{min(start + limit, total_count)}")
        
        lines_data = _read_ndjson_slice(ndjson_path, start, limit)
        for line_idx, obj in lines_data:
            with st.expander(f"Line {line_idx}", expanded=False):
                st.code(json.dumps(obj, indent=2, ensure_ascii=False), language="json")

        if total_pages > 1:
            center_cols = st.columns([1, 8, 1])
            with center_cols[1]:
                _render_pagination(cp, total_pages, ndjson_page_key)
    else:
        st.warning(f"File not found: {ndjson_path}")

elif resolved_layout == "frame_parquet_item_dirs":
    # --- Parquet View ---
    parquet_dir = dataset_dir / "parquet"
    search_dir = parquet_dir if parquet_dir.exists() else dataset_dir
    
    if search_dir.exists():
        st.caption(f"Searching Parquet in: {search_dir}")
        files = sorted(search_dir.rglob("*.parquet"))
        
        if not files:
            st.info("No parquet files found.")
        else:
            # --- File List Pagination ---
            page_size = 10
            total_pages = (len(files) + page_size - 1) // page_size or 1
            
            file_page_key = f"parquet_file_page_{selected_dataset_key}"
            cp_files = st.session_state.get(f"{file_page_key}_current", 1)
            
            if cp_files > total_pages: cp_files = 1

            start = (cp_files - 1) * page_size
            page_files = files[start : start + page_size]

            # Display File List
            with st.container(border=True):
                st.markdown(f"**Parquet Files (Page {cp_files}/{total_pages})**")
                for f in page_files:
                    st.text(f"📄 {f.relative_to(dataset_dir)}")
            
            if total_pages > 1:
                center_cols = st.columns([1, 8, 1])
                with center_cols[1]:
                    _render_pagination(cp_files, total_pages, file_page_key)

            st.divider()

            # --- File Selection Logic ---
            options = [f.relative_to(dataset_dir) for f in files]
            select_key = f"parquet_file_select_{selected_dataset_key}"
            
            selected_rel = st.selectbox(
                "Select file to preview content",
                options=options,
                key=select_key,
            )
            
            # --- Parquet Content Preview ---
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
                    
                    preview_page_key = f"pq_view_{selected_dataset_key}_{str(selected_rel)}"
                    
                    cp_row = st.session_state.get(f"{preview_page_key}_current", 1)
                    if cp_row > row_total_pages: cp_row = 1

                    start_r = (cp_row - 1) * row_page_size
                    table_slice = table.slice(start_r, row_page_size)
                    
                    cols = table_slice.column_names
                    data_dict = table_slice.to_pydict()
                    rows = [{col: data_dict[col][i] for col in cols} for i in range(table_slice.num_rows)]
                    
                    df_rows = pd.DataFrame(rows)

                    # --- 时间戳优化处理 ---
                    for col in df_rows.columns:
                        if pd.api.types.is_datetime64_any_dtype(df_rows[col]):
                            try:
                                df_rows[col] = df_rows[col].dt.round('1s')
                                if df_rows[col].dt.tz is not None:
                                    df_rows[col] = df_rows[col].dt.tz_localize(None)
                            except Exception:
                                pass
                    
                    # 使用自定义静态表格替代 st.dataframe
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

# DRS info
with st.expander("Show DRS (Data Requirement Specification)"):
    try:
        drs = snapshot.drs
        st.code(json.dumps(drs.model_dump(exclude_none=True), indent=2, ensure_ascii=False), language="json")
    except Exception as e:
        st.error(f"Failed to load DRS: {e}")
