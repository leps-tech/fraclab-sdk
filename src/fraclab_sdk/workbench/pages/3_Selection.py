"""Selection and configuration page."""

import json
from contextlib import suppress
from pathlib import Path

import pandas as pd
import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.run import RunManager
from fraclab_sdk.selection.model import SelectionModel
from fraclab_sdk.snapshot import SnapshotLibrary
from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.workbench.i18n import page_title, tx
from fraclab_sdk.workbench.page_state import choose_valid_option, read_page_state, write_page_state
from fraclab_sdk.workbench.utils import format_snapshot_option, format_timestamp

st.set_page_config(
    page_title=page_title("selection"),
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui_styles.apply_global_styles("selection")
ui_styles.render_page_header(tx("Selection", "运行配置"))

# --- Page-Specific CSS ---
st.markdown("""
<style>
    /* Hide Data Editor header buttons (sort arrows, menu) */
    [data-testid="stDataEditor"] th button {
        display: none !important;
    }
    /* Hide row number column if present */
    [data-testid="stDataEditor"] td[aria-selected="false"] {
        color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

_WORKBENCH_CONFIG = SDKConfig()
_SELECTION_PAGE_STATE_CACHE_KEY = "_selection_page_state_cache"


def get_libraries():
    """Get SDK libraries."""
    return (
        SnapshotLibrary(_WORKBENCH_CONFIG),
        AlgorithmLibrary(_WORKBENCH_CONFIG),
        RunManager(_WORKBENCH_CONFIG),
    )


snapshot_lib, algorithm_lib, run_manager = get_libraries()

# Initialize session state
if "selection_model" not in st.session_state:
    st.session_state.selection_model = None
if "selection_triggers" not in st.session_state:
    st.session_state.selection_triggers = {}

snapshots = snapshot_lib.list_snapshots()
algorithms = algorithm_lib.list_algorithms()

if not snapshots:
    st.info(tx("No snapshots available. Import a snapshot first.", "没有可用快照。请先导入快照。"))
    st.stop()

if not algorithms:
    st.info(tx("No algorithms available. Import an algorithm first.", "没有可用算法。请先导入算法。"))
    st.stop()


# --- Helper Functions ---

def _detect_layout(dir_path: Path) -> str | None:
    """Best-effort layout detection from on-disk files (Copied from Browse)."""
    if not dir_path.exists():
        return None
    if (dir_path / "object.ndjson").exists():
        return "object_ndjson_lines"
    if (dir_path / "parquet").exists():
        return "frame_parquet_item_dirs"
    # Fallback: check if any parquet files exist in subdirs
    if list(dir_path.rglob("*.parquet")):
        return "frame_parquet_item_dirs"
    return None


def _apply_default_select_all(selection_model: SelectionModel, snapshot) -> None:
    """Apply one-time default: select all items for every selectable dataset."""
    for ds in selection_model.get_selectable_datasets():
        dataset_key = ds.dataset_key
        current = selection_model.get_selected(dataset_key)
        if current:
            continue
        item_indices = [idx for idx, _ in snapshot.get_items(dataset_key)]
        selection_model.set_selected(dataset_key, item_indices)


def _get_selection_page_state() -> dict:
    """Return the cached persisted state for the Selection page."""
    page_state = st.session_state.get(_SELECTION_PAGE_STATE_CACHE_KEY)
    if isinstance(page_state, dict):
        return page_state
    page_state = read_page_state("selection", _WORKBENCH_CONFIG)
    st.session_state[_SELECTION_PAGE_STATE_CACHE_KEY] = page_state
    return page_state


def _write_selection_page_state(page_state: dict) -> None:
    """Persist and cache Selection page state."""
    st.session_state[_SELECTION_PAGE_STATE_CACHE_KEY] = page_state
    write_page_state("selection", page_state, _WORKBENCH_CONFIG)


def _selection_context_key(snapshot_id: str, algo_key: str) -> str:
    """Build a stable persistence key for snapshot+algorithm context."""
    return f"{snapshot_id}::{algo_key}"


def _get_saved_selection_context(snapshot_id: str, algo_key: str) -> dict:
    """Return saved selection context for a snapshot+algorithm pair."""
    page_state = _get_selection_page_state()
    contexts = page_state.get("contexts")
    if not isinstance(contexts, dict):
        return {}
    context_state = contexts.get(_selection_context_key(snapshot_id, algo_key))
    return context_state if isinstance(context_state, dict) else {}


def _update_selection_page_state(
    *,
    snapshot_id: str | None = None,
    algo_key: str | None = None,
    context_state: dict | None = None,
) -> None:
    """Persist Selection page state and optional context state."""
    page_state = dict(_get_selection_page_state())
    if snapshot_id is not None:
        page_state["selected_snapshot_id"] = snapshot_id
    if algo_key is not None:
        page_state["selected_algo_key"] = algo_key
    if snapshot_id is not None and algo_key is not None and context_state is not None:
        contexts = dict(page_state.get("contexts") or {})
        contexts[_selection_context_key(snapshot_id, algo_key)] = context_state
        page_state["contexts"] = contexts
    _write_selection_page_state(page_state)


def _serialize_selection_model(selection_model: SelectionModel) -> dict[str, list[int]]:
    """Serialize selected item indices for all selectable datasets."""
    return {
        dataset.dataset_key: selection_model.get_selected(dataset.dataset_key)
        for dataset in selection_model.get_selectable_datasets()
    }


def _restore_selection_model(selection_model: SelectionModel, saved_selections: dict[str, object]) -> bool:
    """Restore saved dataset selections onto a freshly created model."""
    restored_any = False
    selectable_keys = {dataset.dataset_key for dataset in selection_model.get_selectable_datasets()}
    for dataset_key in selectable_keys:
        raw_value = saved_selections.get(dataset_key)
        if not isinstance(raw_value, list):
            continue
        valid_indices = [int(idx) for idx in raw_value if isinstance(idx, int)]
        selection_model.set_selected(dataset_key, valid_indices)
        restored_any = True
    return restored_any


def _persist_selection_context(snapshot_id: str, algo_key: str, selection_model: SelectionModel, params_json: str) -> None:
    """Persist the current selection model and parameter text for the active context."""
    _update_selection_page_state(
        snapshot_id=snapshot_id,
        algo_key=algo_key,
        context_state={
            "dataset_selections": _serialize_selection_model(selection_model),
            "params_json": params_json,
        },
    )


# ==========================================
# Dialogs
# ==========================================
@st.dialog(tx("Data Requirement Specification (DRS)", "数据需求规范（DRS）"))
def show_drs_dialog(drs_data: dict):
    st.caption(tx("This defines the data structure required by the snapshot.", "这里定义了快照所需的数据结构。"))
    st.code(json.dumps(drs_data, indent=2, ensure_ascii=False), language="json")


# ==========================================
# 1 & 2. Context Selection (Snapshot & Algo)
# ==========================================
st.subheader(tx("1. Configuration Context", "1. 配置上下文"))

col_snap, col_algo = st.columns(2)

# --- Left: Snapshot ---
with col_snap, st.container(border=True):
    st.markdown(tx("#### 📦 Snapshot", "#### 📦 快照"))
    snapshot_options = {s.snapshot_id: s for s in snapshots}
    selection_page_state = _get_selection_page_state()
    snapshot_widget_key = "selection_snapshot_picker"
    restored_snapshot_id = choose_valid_option(
        list(snapshot_options.keys()),
        selection_page_state.get("selected_snapshot_id"),
    )
    if st.session_state.get(snapshot_widget_key) not in snapshot_options and restored_snapshot_id is not None:
        st.session_state[snapshot_widget_key] = restored_snapshot_id
    
    selected_snapshot_id = st.selectbox(
        tx("Select Snapshot", "选择快照"),
        options=list(snapshot_options.keys()),
        format_func=lambda x: format_snapshot_option(snapshot_options[x]),
        label_visibility="collapsed",
        key=snapshot_widget_key,
    )
    _update_selection_page_state(snapshot_id=selected_snapshot_id)
    
    if selected_snapshot_id:
        snap_obj = snapshot_options[selected_snapshot_id]
        
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            st.caption(tx("**Bundle ID:** `{bundle_id}`", "**包 ID：** `{bundle_id}`", bundle_id=snap_obj.bundle_id))
            st.caption(tx("**Imported:** {timestamp}", "**导入时间：** {timestamp}", timestamp=format_timestamp(snap_obj.imported_at)))
        with sc2:
            # Updated API: width="stretch"
                if st.button(tx("📜 DRS", "📜 DRS"), key=f"view_drs_{selected_snapshot_id}", help=tx("View Data Requirements", "查看数据需求"), width="stretch"):
                    try:
                        full_snap = snapshot_lib.get_snapshot(snap_obj.snapshot_id)
                        drs_data = full_snap.drs.model_dump(exclude_none=True)
                        show_drs_dialog(drs_data)
                    except Exception as e:
                        st.error(tx("Cannot load DRS: {error}", "无法加载 DRS：{error}", error=e))

        if snap_obj.description:
            st.info(snap_obj.description)

# --- Right: Algorithm ---
with col_algo, st.container(border=True):
    st.markdown(tx("#### 🧩 Algorithm", "#### 🧩 算法"))
    algo_options = {f"{a.algorithm_id}:{a.version}": a for a in algorithms}
    algo_widget_key = "selection_algorithm_picker"
    restored_algo_key = choose_valid_option(
        list(algo_options.keys()),
        _get_selection_page_state().get("selected_algo_key"),
    )
    if st.session_state.get(algo_widget_key) not in algo_options and restored_algo_key is not None:
        st.session_state[algo_widget_key] = restored_algo_key
    
    selected_algo_key = st.selectbox(
        tx("Select Algorithm", "选择算法"),
        options=list(algo_options.keys()),
        format_func=lambda k: f"{algo_options[k].name or algo_options[k].algorithm_id} (v{algo_options[k].version})",
        label_visibility="collapsed",
        key=algo_widget_key,
    )
    _update_selection_page_state(snapshot_id=selected_snapshot_id, algo_key=selected_algo_key)

    if selected_algo_key:
        algo_obj = algo_options[selected_algo_key]
        st.caption(tx("**Contract:** `{contract}`", "**契约：** `{contract}`", contract=algo_obj.contract_version))
        authors = getattr(algo_obj, "authors", [])
        if authors:
            author_names = ", ".join([a.get("name", tx("Unknown", "未知")) for a in authors])
            st.caption(tx("**Authors:** {authors}", "**作者：** {authors}", authors=author_names))
        
        if getattr(algo_obj, "summary", ""):
            st.info(algo_obj.summary)


# Initialize Logic
if selected_snapshot_id and selected_algo_key:
    snapshot = snapshot_lib.get_snapshot(selected_snapshot_id)
    algo = algo_options[selected_algo_key]
    algorithm = algorithm_lib.get_algorithm(algo.algorithm_id, algo.version)
    saved_context = _get_saved_selection_context(selected_snapshot_id, selected_algo_key)

    current_snap_id = st.session_state.get("selection_snapshot_id")
    current_algo_id = st.session_state.get("selection_algorithm_id")
    current_algo_ver = st.session_state.get("selection_algorithm_version")

    if (current_snap_id != selected_snapshot_id or 
        current_algo_id != algo.algorithm_id or 
        current_algo_ver != algo.version):
        
        try:
            selection_model = SelectionModel.from_snapshot_and_drs(snapshot, algorithm.drs)
            restored_any = False
            saved_selections = saved_context.get("dataset_selections")
            if isinstance(saved_selections, dict):
                restored_any = _restore_selection_model(selection_model, saved_selections)
            if not restored_any:
                _apply_default_select_all(selection_model, snapshot)
            st.session_state.selection_model = selection_model
            st.session_state.selection_snapshot_id = selected_snapshot_id
            st.session_state.selection_algorithm_id = algo.algorithm_id
            st.session_state.selection_algorithm_version = algo.version
            st.session_state.selection_triggers = {}
        except Exception as e:
            st.error(tx("Failed to create selection model: {error}", "创建选择模型失败：{error}", error=e))
            st.stop()
    else:
        selection_model = st.session_state.selection_model

    st.divider()

    # ==========================================
    # 3. Data Selection
    # ==========================================
    st.subheader(tx("2. Data Selection", "2. 数据选择"))
    
    selectable = selection_model.get_selectable_datasets()
    
    if not selectable:
        st.warning(tx("This algorithm does not require any specific dataset selection (DRS is empty).", "这个算法不需要指定数据集选择（DRS 为空）。"))

    for ds in selectable:
        dataset_key = ds.dataset_key
        
        with st.container(border=True):
            head_c1, head_c2 = st.columns([4, 1])
            with head_c1:
                st.markdown(f"##### 🗃️ {dataset_key}")
                if ds.description:
                    st.caption(ds.description)
            with head_c2:
                st.caption(tx("Req: **{value}**", "要求：**{value}**", value=ds.cardinality))
                st.caption(tx("Total: **{value}**", "总数：**{value}**", value=ds.total_items))

            # --- Layout Detection Logic (multi-level fallback) ---
            resolved_layout = None

            # 1. Try dataspec (ds.json)
            with suppress(Exception):
                resolved_layout = snapshot.get_layout(dataset_key)

            # 2. Try bundle manifest (manifest.json) - always has layout
            if not resolved_layout:
                try:
                    manifest_ds = snapshot.manifest.datasets.get(dataset_key)
                    if manifest_ds:
                        resolved_layout = manifest_ds.layout
                except Exception:
                    pass

            # 3. Fallback to filesystem auto-detection
            if not resolved_layout:
                data_root = snapshot.manifest.dataRoot or "data"
                dataset_dir = snapshot.directory / data_root / dataset_key
                resolved_layout = _detect_layout(dataset_dir)

            items = snapshot.get_items(dataset_key)

            # Pre-compute data paths for this dataset
            data_root = snapshot.manifest.dataRoot or "data"
            dataset_data_dir = snapshot.directory / data_root / dataset_key

            # --- Helper to check status (prioritize warnings) ---
            def _get_item_status(
                idx: int,
                layout_type: str | None,
                dataset_data_dir: Path = dataset_data_dir,
            ) -> tuple[str, str]:
                """Check item file status. Prioritize Empty/Missing warnings over format."""

                def _check_parquet_item(item_dir: Path) -> tuple[str, str]:
                    """Check parquet item directory for issues."""
                    import pyarrow.parquet as pq

                    if not item_dir.exists():
                        return (
                            tx("⚠️ Missing", "⚠️ 缺失"),
                            tx("Directory: {name}", "目录：{name}", name=item_dir.name),
                        )
                    parquet_files = list(item_dir.rglob("*.parquet"))
                    if not parquet_files:
                        return tx("⚠️ Empty", "⚠️ 空"), tx("No .parquet files", "没有 .parquet 文件")

                    # Check for zero-byte files
                    zero_byte_files = [f for f in parquet_files if f.stat().st_size == 0]
                    if zero_byte_files:
                        return (
                            tx("⚠️ Empty", "⚠️ 空"),
                            tx(
                                "{empty}/{total} files are 0 bytes",
                                "{empty}/{total} 个文件大小为 0 字节",
                                empty=len(zero_byte_files),
                                total=len(parquet_files),
                            ),
                        )

                    # Check for parquet files with metadata but 0 rows
                    total_rows = 0
                    empty_row_files = []
                    for pf in parquet_files:
                        try:
                            meta = pq.read_metadata(pf)
                            if meta.num_rows == 0:
                                empty_row_files.append(pf)
                            total_rows += meta.num_rows
                        except Exception:
                            pass  # If can't read metadata, skip this check

                    if empty_row_files and len(empty_row_files) == len(parquet_files):
                        return tx("⚠️ Empty", "⚠️ 空"), tx("All files have 0 rows", "所有文件均为 0 行")
                    if empty_row_files:
                        return (
                            tx("⚠️ Partial", "⚠️ 部分为空"),
                            tx(
                                "{empty}/{total} files have 0 rows",
                                "{empty}/{total} 个文件为 0 行",
                                empty=len(empty_row_files),
                                total=len(parquet_files),
                            ),
                        )

                    return (
                        "✓ Parquet",
                        tx(
                            "{count} file(s), {rows} rows",
                            "{count} 个文件，{rows} 行",
                            count=len(parquet_files),
                            rows=f"{total_rows:,}",
                        ),
                    )

                if layout_type == "frame_parquet_item_dirs":
                    item_dir = dataset_data_dir / "parquet" / f"item-{idx:05d}"
                    return _check_parquet_item(item_dir)

                elif layout_type == "object_ndjson_lines":
                    ndjson_path = dataset_data_dir / "object.ndjson"
                    if not ndjson_path.exists():
                        return tx("⚠️ Missing", "⚠️ 缺失"), tx("object.ndjson not found", "未找到 object.ndjson")
                    if ndjson_path.stat().st_size == 0:
                        return tx("⚠️ Empty", "⚠️ 空"), tx("object.ndjson is 0 bytes", "object.ndjson 大小为 0 字节")
                    return "✓ NDJSON", tx("OK", "正常")

                # Layout not detected - try to infer from files
                ndjson_path = dataset_data_dir / "object.ndjson"
                if ndjson_path.exists():
                    if ndjson_path.stat().st_size == 0:
                        return tx("⚠️ Empty", "⚠️ 空"), tx("object.ndjson is 0 bytes", "object.ndjson 大小为 0 字节")
                    return "✓ NDJSON", tx("Auto-detected", "自动识别")

                parquet_dir = dataset_data_dir / "parquet"
                if parquet_dir.exists():
                    item_dir = parquet_dir / f"item-{idx:05d}"
                    return _check_parquet_item(item_dir)

                return tx("❓ Unknown", "❓ 未知"), tx("No data files found", "未找到数据文件")

            # --- CASE A: Single Selection ---
            if ds.cardinality == "one":
                options = list(range(len(items)))
                
                def _fmt_single(idx: int, layout_type: str | None = resolved_layout):
                    status, _ = _get_item_status(idx, layout_type)
                    if status == tx("⚠️ Empty", "⚠️ 空"):
                        return tx("Item {index} (⚠️ Empty)", "条目 {index}（⚠️ 空）", index=idx)
                    return tx("Item {index} ({status})", "条目 {index}（{status}）", index=idx, status=status)

                selected_idx = st.selectbox(
                    tx("Select item for {dataset_key}", "为 {dataset_key} 选择条目", dataset_key=dataset_key),
                    options=options,
                    format_func=_fmt_single,
                    key=f"select_{dataset_key}"
                )
                
                if selected_idx is not None:
                    selection_model.set_selected(dataset_key, [selected_idx])
                    _persist_selection_context(
                        selected_snapshot_id,
                        selected_algo_key,
                        selection_model,
                        st.session_state.get("selection_params_json", ""),
                    )

            # --- CASE B: Multi Selection (Data Editor) ---
            else:
                current_selected_order = selection_model.get_selected(dataset_key)
                current_selected_set = set(current_selected_order)
                order_map = {idx: rank + 1 for rank, idx in enumerate(current_selected_order)}
                
                rows = []
                for idx, _ in items:
                    status_label, detail_help = _get_item_status(idx, resolved_layout)
                    
                    rows.append({
                        "Selected": idx in current_selected_set,
                        "Order": order_map.get(idx, ""),
                        "Index": idx,
                        "Type": status_label,
                        "_help": detail_help
                    })
                
                df_items = pd.DataFrame(rows)

                # Action Buttons
                editor_key = f"editor_{dataset_key}"
                
                col_btns, col_status = st.columns([2, 3])
                with col_btns:
                    b_c1, b_c2, _ = st.columns([1, 1, 2], gap="small")
                    with b_c1:
                        # Updated API: width="stretch"
                        if st.button(tx("All", "全选"), key=f"all_{dataset_key}", width="stretch"):
                            all_indices = [r["Index"] for r in rows]
                            selection_model.set_selected(dataset_key, all_indices)
                            _persist_selection_context(
                                selected_snapshot_id,
                                selected_algo_key,
                                selection_model,
                                st.session_state.get("selection_params_json", ""),
                            )
                            st.rerun()
                    with b_c2:
                        # Updated API: width="stretch"
                        if st.button(tx("None", "清空"), key=f"none_{dataset_key}", width="stretch"):
                            selection_model.set_selected(dataset_key, [])
                            _persist_selection_context(
                                selected_snapshot_id,
                                selected_algo_key,
                                selection_model,
                                st.session_state.get("selection_params_json", ""),
                            )
                            st.rerun()
                
                with col_status:
                    st.markdown(
                        tx(
                            "<div style='text-align:right; color:#666; padding-top:5px;'>Selected: <b>{selected}</b> / {total}</div>",
                            "<div style='text-align:right; color:#666; padding-top:5px;'>已选：<b>{selected}</b> / {total}</div>",
                            selected=len(current_selected_set),
                            total=len(items),
                        ),
                        unsafe_allow_html=True,
                    )

                # Render Data Editor
                # Updated API: use width="stretch" instead of use_container_width
                edited_df = st.data_editor(
                    df_items,
                    key=editor_key,
                    height=300, 
                    width="stretch", 
                    hide_index=True,
                    num_rows="fixed",
                    column_config={
                        "Selected": st.column_config.CheckboxColumn(
                            tx("Select", "选择"),
                            width="small",
                            default=False
                        ),
                        "Order": st.column_config.TextColumn(
                            tx("Selection Order", "选择顺序"),
                            width="small",
                            disabled=True,
                            help=tx("Selection order used by run stage mapping", "运行阶段映射使用的选择顺序")
                        ),
                        "Index": st.column_config.NumberColumn(
                            tx("Item ID", "条目 ID"),
                            format="%d",
                            width="small",
                            disabled=True
                        ),
                        "Type": st.column_config.TextColumn(
                            tx("File Type / Status", "文件类型 / 状态"),
                            width="medium",
                            disabled=True,
                            help=tx("Shows file format or warns if file is empty", "显示文件格式，或提示文件为空")
                        ),
                        "_help": None # Hide internal column
                    }
                )

                new_selected_indices = edited_df[edited_df["Selected"]]["Index"].tolist()

                merged_order = [idx for idx in current_selected_order if idx in new_selected_indices]
                merged_order.extend(idx for idx in new_selected_indices if idx not in current_selected_set)

                if merged_order != current_selected_order:
                    selection_model.set_selected(dataset_key, merged_order)
                    _persist_selection_context(
                        selected_snapshot_id,
                        selected_algo_key,
                        selection_model,
                        st.session_state.get("selection_params_json", ""),
                    )
                    st.rerun()

    st.divider()

    # ==========================================
    # 4. Validation & Parameters
    # ==========================================
    
    col_valid, col_params = st.columns([1, 1], gap="large")

    with col_valid:
        st.subheader(tx("3. Validation", "3. 校验"))
        errors = selection_model.validate()
        
        with st.container(border=True):
            if errors:
                for err in errors:
                    st.error(f"**{err.dataset_key}**: {err.message}", icon="🚫")
            else:
                st.success(tx("All selection requirements met.", "已满足所有选择要求。"), icon="✅")

    with col_params:
        st.subheader(tx("4. Parameters", "4. 参数"))
        params_schema = algorithm.params_schema
        params_widget_key = "selection_params_json"
        params_context_key = _selection_context_key(selected_snapshot_id, selected_algo_key)
        
        defaults = {}
        if "properties" in params_schema:
            for key, prop in params_schema["properties"].items():
                if "default" in prop:
                    defaults[key] = prop["default"]
        default_params_json = json.dumps(defaults, indent=2)
        saved_params_json = saved_context.get("params_json")
        restored_params_json = saved_params_json if isinstance(saved_params_json, str) else default_params_json
        if st.session_state.get("selection_params_context") != params_context_key:
            st.session_state["selection_params_context"] = params_context_key
            st.session_state[params_widget_key] = restored_params_json
        elif params_widget_key not in st.session_state:
            st.session_state[params_widget_key] = restored_params_json

        with st.expander(tx("Parameters Configuration", "参数配置"), expanded=True):
            params_json = st.text_area(
                tx("JSON Input", "JSON 输入"),
                height=200,
                help=tx("Enter algorithm parameters as JSON", "以 JSON 输入算法参数"),
                label_visibility="collapsed",
                key=params_widget_key,
            )
            
            try:
                params = json.loads(params_json) if params_json.strip() else {}
            except json.JSONDecodeError as e:
                st.error(tx("Invalid JSON: {error}", "无效 JSON：{error}", error=e))
                params = None
        _persist_selection_context(selected_snapshot_id, selected_algo_key, selection_model, params_json)

    st.divider()

    # ==========================================
    # 5. Execution
    # ==========================================
    
    col_spacer, col_action = st.columns([3, 1])
    
    with col_action:
        create_disabled = bool(errors) or (params is None)
        # Updated API: width="stretch"
        if st.button(tx("🚀 Create & Start Run", "🚀 创建并开始运行"), type="primary", disabled=create_disabled, width="stretch"):
            try:
                run_id = run_manager.create_run(
                    snapshot_id=selected_snapshot_id,
                    algorithm_id=algo.algorithm_id,
                    algorithm_version=algo.version,
                    selection=selection_model,
                    params=params,
                )
                st.session_state.created_run_id = run_id
                st.switch_page("pages/4_Run.py")
            except Exception as e:
                st.error(tx("Failed to create run: {error}", "创建运行失败：{error}", error=e))
