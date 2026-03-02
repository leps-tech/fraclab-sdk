"""Selection and configuration page."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.run import RunManager
from fraclab_sdk.selection.model import SelectionModel
from fraclab_sdk.snapshot import SnapshotLibrary
from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.workbench.utils import format_snapshot_option, format_timestamp

st.set_page_config(page_title="Selection", page_icon="✅", layout="wide", initial_sidebar_state="expanded")
st.title("Selection")

ui_styles.apply_global_styles()

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


def get_libraries():
    """Get SDK libraries."""
    config = SDKConfig()
    return (
        SnapshotLibrary(config),
        AlgorithmLibrary(config),
        RunManager(config),
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
    st.info("No snapshots available. Import a snapshot first.")
    st.stop()

if not algorithms:
    st.info("No algorithms available. Import an algorithm first.")
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


# ==========================================
# Dialogs
# ==========================================
@st.dialog("Data Requirement Specification (DRS)")
def show_drs_dialog(drs_data: dict):
    st.caption("This defines the data structure required by the snapshot.")
    st.code(json.dumps(drs_data, indent=2, ensure_ascii=False), language="json")


# ==========================================
# 1 & 2. Context Selection (Snapshot & Algo)
# ==========================================
st.subheader("1. Configuration Context")

col_snap, col_algo = st.columns(2)

# --- Left: Snapshot ---
with col_snap:
    with st.container(border=True):
        st.markdown("#### 📦 Snapshot")
        snapshot_options = {s.snapshot_id: s for s in snapshots}
        
        selected_snapshot_id = st.selectbox(
            "Select Snapshot",
            options=list(snapshot_options.keys()),
            format_func=lambda x: format_snapshot_option(snapshot_options[x]),
            label_visibility="collapsed"
        )
        
        if selected_snapshot_id:
            snap_obj = snapshot_options[selected_snapshot_id]
            
            sc1, sc2 = st.columns([3, 1])
            with sc1:
                st.caption(f"**Bundle ID:** `{snap_obj.bundle_id}`")
                st.caption(f"**Imported:** {format_timestamp(snap_obj.imported_at)}")
            with sc2:
                # Updated API: width="stretch"
                if st.button("📜 DRS", key=f"view_drs_{selected_snapshot_id}", help="View Data Requirements", width="stretch"):
                    try:
                        full_snap = snapshot_lib.get_snapshot(snap_obj.snapshot_id)
                        drs_data = full_snap.drs.model_dump(exclude_none=True)
                        show_drs_dialog(drs_data)
                    except Exception as e:
                        st.error(f"Cannot load DRS: {e}")

            if snap_obj.description:
                st.info(snap_obj.description)

# --- Right: Algorithm ---
with col_algo:
    with st.container(border=True):
        st.markdown("#### 🧩 Algorithm")
        algo_options = {f"{a.algorithm_id}:{a.version}": a for a in algorithms}
        
        selected_algo_key = st.selectbox(
            "Select Algorithm",
            options=list(algo_options.keys()),
            format_func=lambda k: f"{algo_options[k].name or algo_options[k].algorithm_id} (v{algo_options[k].version})",
            label_visibility="collapsed"
        )

        if selected_algo_key:
            algo_obj = algo_options[selected_algo_key]
            st.caption(f"**Contract:** `{algo_obj.contract_version}`")
            authors = getattr(algo_obj, "authors", [])
            if authors:
                author_names = ", ".join([a.get("name", "Unknown") for a in authors])
                st.caption(f"**Authors:** {author_names}")
            
            if getattr(algo_obj, "summary", ""):
                st.info(algo_obj.summary)


# Initialize Logic
if selected_snapshot_id and selected_algo_key:
    snapshot = snapshot_lib.get_snapshot(selected_snapshot_id)
    algo = algo_options[selected_algo_key]
    algorithm = algorithm_lib.get_algorithm(algo.algorithm_id, algo.version)

    current_snap_id = st.session_state.get("selection_snapshot_id")
    current_algo_id = st.session_state.get("selection_algorithm_id")
    current_algo_ver = st.session_state.get("selection_algorithm_version")

    if (current_snap_id != selected_snapshot_id or 
        current_algo_id != algo.algorithm_id or 
        current_algo_ver != algo.version):
        
        try:
            selection_model = SelectionModel.from_snapshot_and_drs(snapshot, algorithm.drs)
            _apply_default_select_all(selection_model, snapshot)
            st.session_state.selection_model = selection_model
            st.session_state.selection_snapshot_id = selected_snapshot_id
            st.session_state.selection_algorithm_id = algo.algorithm_id
            st.session_state.selection_algorithm_version = algo.version
            st.session_state.selection_triggers = {}
        except Exception as e:
            st.error(f"Failed to create selection model: {e}")
            st.stop()
    else:
        selection_model = st.session_state.selection_model

    st.divider()

    # ==========================================
    # 3. Data Selection
    # ==========================================
    st.subheader("2. Data Selection")
    
    selectable = selection_model.get_selectable_datasets()
    
    if not selectable:
        st.warning("This algorithm does not require any specific dataset selection (DRS is empty).")

    for ds in selectable:
        dataset_key = ds.dataset_key
        
        with st.container(border=True):
            head_c1, head_c2 = st.columns([4, 1])
            with head_c1:
                st.markdown(f"##### 🗃️ {dataset_key}")
                if ds.description:
                    st.caption(ds.description)
            with head_c2:
                st.caption(f"Req: **{ds.cardinality}**")
                st.caption(f"Total: **{ds.total_items}**")

            # --- Layout Detection Logic (multi-level fallback) ---
            resolved_layout = None

            # 1. Try dataspec (ds.json)
            try:
                resolved_layout = snapshot.get_layout(dataset_key)
            except Exception:
                pass

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
            def _get_item_status(idx: int, layout_type: str | None) -> tuple[str, str]:
                """Check item file status. Prioritize Empty/Missing warnings over format."""

                def _check_parquet_item(item_dir: Path) -> tuple[str, str]:
                    """Check parquet item directory for issues."""
                    import pyarrow.parquet as pq

                    if not item_dir.exists():
                        return "⚠️ Missing", f"Directory: {item_dir.name}"
                    parquet_files = list(item_dir.rglob("*.parquet"))
                    if not parquet_files:
                        return "⚠️ Empty", "No .parquet files"

                    # Check for zero-byte files
                    zero_byte_files = [f for f in parquet_files if f.stat().st_size == 0]
                    if zero_byte_files:
                        return "⚠️ Empty", f"{len(zero_byte_files)}/{len(parquet_files)} files are 0 bytes"

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
                        return "⚠️ Empty", "All files have 0 rows"
                    if empty_row_files:
                        return "⚠️ Partial", f"{len(empty_row_files)}/{len(parquet_files)} files have 0 rows"

                    return "✓ Parquet", f"{len(parquet_files)} file(s), {total_rows:,} rows"

                if layout_type == "frame_parquet_item_dirs":
                    item_dir = dataset_data_dir / "parquet" / f"item-{idx:05d}"
                    return _check_parquet_item(item_dir)

                elif layout_type == "object_ndjson_lines":
                    ndjson_path = dataset_data_dir / "object.ndjson"
                    if not ndjson_path.exists():
                        return "⚠️ Missing", "object.ndjson not found"
                    if ndjson_path.stat().st_size == 0:
                        return "⚠️ Empty", "object.ndjson is 0 bytes"
                    return "✓ NDJSON", "OK"

                # Layout not detected - try to infer from files
                ndjson_path = dataset_data_dir / "object.ndjson"
                if ndjson_path.exists():
                    if ndjson_path.stat().st_size == 0:
                        return "⚠️ Empty", "object.ndjson is 0 bytes"
                    return "✓ NDJSON", "Auto-detected"

                parquet_dir = dataset_data_dir / "parquet"
                if parquet_dir.exists():
                    item_dir = parquet_dir / f"item-{idx:05d}"
                    return _check_parquet_item(item_dir)

                return "❓ Unknown", "No data files found"

            # --- CASE A: Single Selection ---
            if ds.cardinality == "one":
                options = list(range(len(items)))
                
                def _fmt_single(idx):
                    status, _ = _get_item_status(idx, resolved_layout)
                    if "Empty" in status:
                        return f"Item {idx} (⚠️ Empty)"
                    return f"Item {idx} ({status})"

                selected_idx = st.selectbox(
                    f"Select item for {dataset_key}",
                    options=options,
                    format_func=_fmt_single,
                    key=f"select_{dataset_key}"
                )
                
                if selected_idx is not None:
                    selection_model.set_selected(dataset_key, [selected_idx])

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
                        if st.button("All", key=f"all_{dataset_key}", width="stretch"):
                            all_indices = [r["Index"] for r in rows]
                            selection_model.set_selected(dataset_key, all_indices)
                            st.rerun()
                    with b_c2:
                        # Updated API: width="stretch"
                        if st.button("None", key=f"none_{dataset_key}", width="stretch"):
                            selection_model.set_selected(dataset_key, [])
                            st.rerun()
                
                with col_status:
                    st.markdown(f"<div style='text-align:right; color:#666; padding-top:5px;'>Selected: <b>{len(current_selected_set)}</b> / {len(items)}</div>", unsafe_allow_html=True)

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
                            "Select",
                            width="small",
                            default=False
                        ),
                        "Order": st.column_config.TextColumn(
                            "Sel Order",
                            width="small",
                            disabled=True,
                            help="Selection order used by run stage mapping"
                        ),
                        "Index": st.column_config.NumberColumn(
                            "Item ID",
                            format="%d",
                            width="small",
                            disabled=True
                        ),
                        "Type": st.column_config.TextColumn(
                            "File Type / Status",
                            width="medium",
                            disabled=True,
                            help="Shows file format or warns if file is empty"
                        ),
                        "_help": None # Hide internal column
                    }
                )

                new_selected_indices = edited_df[edited_df["Selected"]]["Index"].tolist()

                merged_order = [idx for idx in current_selected_order if idx in new_selected_indices]
                merged_order.extend(idx for idx in new_selected_indices if idx not in current_selected_set)

                if merged_order != current_selected_order:
                    selection_model.set_selected(dataset_key, merged_order)
                    st.rerun()

    st.divider()

    # ==========================================
    # 4. Validation & Parameters
    # ==========================================
    
    col_valid, col_params = st.columns([1, 1], gap="large")

    with col_valid:
        st.subheader("3. Validation")
        errors = selection_model.validate()
        
        with st.container(border=True):
            if errors:
                for err in errors:
                    st.error(f"**{err.dataset_key}**: {err.message}", icon="🚫")
            else:
                st.success("All selection requirements met.", icon="✅")

    with col_params:
        st.subheader("4. Parameters")
        params_schema = algorithm.params_schema
        
        defaults = {}
        if "properties" in params_schema:
            for key, prop in params_schema["properties"].items():
                if "default" in prop:
                    defaults[key] = prop["default"]

        with st.expander("Parameters Configuration", expanded=True):
            params_json = st.text_area(
                "JSON Input",
                value=json.dumps(defaults, indent=2),
                height=200,
                help="Enter algorithm parameters as JSON",
                label_visibility="collapsed"
            )
            
            try:
                params = json.loads(params_json) if params_json.strip() else {}
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                params = None

    st.divider()

    # ==========================================
    # 5. Execution
    # ==========================================
    
    col_spacer, col_action = st.columns([3, 1])
    
    with col_action:
        create_disabled = bool(errors) or (params is None)
        # Updated API: width="stretch"
        if st.button("🚀 Create & Start Run", type="primary", disabled=create_disabled, width="stretch"):
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
                st.error(f"Failed to create run: {e}")
