"""Results viewing page."""

import json

import streamlit as st

from fraclab_sdk.config import SDKConfig
from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.results import (
    ResultReader,
    get_artifact_preview_type,
    preview_image,
    preview_json_raw,
    preview_json_table,
    preview_scalar,
)
from fraclab_sdk.run import RunManager, RunStatus
from fraclab_sdk.workbench import ui_styles

st.set_page_config(page_title="Results", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.title("Results")

ui_styles.apply_global_styles()

# Guidance: if artifacts preview looks empty, validate OutputContract via the editor page.
st.info(
    "看不到期望的输出？请在 Output Editor 页面点击 Validate 检查 "
    "`schema/output_contract.py` 的 datasets/items/artifacts 是否正确。",
    icon="ℹ️",
)

def get_manager():
    """Get run manager."""
    config = SDKConfig()
    return RunManager(config)


run_manager = get_manager()
algo_lib = AlgorithmLibrary(run_manager._config)
runs = run_manager.list_runs()

if not runs:
    st.info("No runs available.")
    st.stop()

# ==========================================
# 1. Run Selection & Status
# ==========================================

# Prepare options
run_options = {r.run_id: r for r in runs}
# Sort by latest first usually makes sense
run_ids = list(reversed(list(run_options.keys()))) 

# Check for navigation context
default_run_id = st.session_state.pop("executed_run_id", None) or st.session_state.pop("created_run_id", None)
default_index = run_ids.index(default_run_id) if default_run_id in run_ids else 0

with st.container(border=True):
    col_sel, col_stat = st.columns([4, 1])
    
    with col_sel:
        selected_run_id = st.selectbox(
            "Select Run",
            options=run_ids,
            index=default_index,
            format_func=lambda x: f"{x} — {run_options[x].algorithm_id} (v{run_options[x].algorithm_version})",
            label_visibility="collapsed"
        )
    
    with col_stat:
        if selected_run_id:
            status = run_options[selected_run_id].status
            status_color = {
                RunStatus.SUCCEEDED: "green",
                RunStatus.FAILED: "red",
                RunStatus.PENDING: "gray",
                RunStatus.RUNNING: "blue"
            }.get(status, "gray")
            st.markdown(f"<div style='text-align:center; padding: 8px; border: 1px solid #ddd; border-radius: 6px;'>Status: <b style='color:{status_color}'>{status.value}</b></div>", unsafe_allow_html=True)

if not selected_run_id:
    st.stop()

run = run_options[selected_run_id]
run_dir = run_manager.get_run_dir(selected_run_id)
reader = ResultReader(run_dir)

# ==========================================
# 2. Run Context
# ==========================================

def _load_output_contract(algo_id: str, algo_version: str):
    """Load output_contract.json from algorithm directory."""
    try:
        handle = algo_lib.get_algorithm(algo_id, algo_version)
        manifest_path = handle.directory / "manifest.json"
        if not manifest_path.exists():
            return None
        manifest_data = json.loads(manifest_path.read_text())
        files = manifest_data.get("files")
        if not isinstance(files, dict):
            return None
        rel = files.get("outputContractPath")
        if not isinstance(rel, str) or not rel:
            return None
        contract_path = (handle.directory / rel).resolve()
        if not contract_path.exists():
            return None
        return json.loads(contract_path.read_text())
    except Exception:
        return None

output_contract = _load_output_contract(run.algorithm_id, run.algorithm_version)

with st.expander("ℹ️ Run Metadata", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Snapshot ID")
        st.code(run.snapshot_id, language="text")
    with c2:
        st.caption("Algorithm ID")
        st.code(run.algorithm_id, language="text")
    with c3:
        st.caption("Timestamps")
        st.text(f"Start: {run.started_at}\nEnd:   {run.completed_at}")

if run.error:
    st.error(f"Run Error: {run.error}")
elif reader.has_manifest() and reader.get_error():
    st.error(f"Manifest Error: {reader.get_error()}")

st.divider()

# ==========================================
# 3. Artifacts (Default Expanded)
# ==========================================
st.subheader("Artifacts")

if not reader.has_manifest():
    st.warning("⚠️ Output manifest not found (Run may have failed or produced no output)")
else:
    manifest = reader.read_manifest()

    if not manifest.datasets:
        st.info("No artifacts produced.")
    else:
        for ds in manifest.datasets:
            # Match with contract
            contract_ds = None
            if output_contract:
                contract_ds = next((d for d in output_contract.get("datasets", []) if d.get("key") == ds.datasetKey), None)

            header = f"📂 {ds.datasetKey}"
            if contract_ds and contract_ds.get("role"):
                header += f" ({contract_ds.get('role')})"
            
            # --- DATASET LEVEL: EXPANDED BY DEFAULT ---
            with st.expander(header, expanded=True):
                # Contract Info Bar
                if contract_ds:
                    st.caption(
                        f"**Schema Definition:** Kind=`{contract_ds.get('kind')}` • "
                        f"Owner=`{contract_ds.get('owner')}` • "
                        f"Card=`{contract_ds.get('cardinality')}`"
                    )
                
                # --- ITEMS LEVEL: CARDS (Always Visible) ---
                for item in ds.items:
                    art = item.artifact
                    preview_type = get_artifact_preview_type(art)
                    
                    with st.container(border=True):
                        # Item Header & Metadata
                        m1, m2, m3 = st.columns([2, 2, 3])
                        with m1:
                            st.markdown(f"**Item:** `{item.itemKey or art.artifactKey}`")
                        with m2:
                            st.caption(f"Type: `{art.type}`")
                        with m3:
                            if art.mimeType: st.caption(f"MIME: `{art.mimeType}`")
                        
                        # [Modified] 删除了 Owner 和 Dims 的显示
                        st.markdown("---")
                        
                        # Content Preview
                        if preview_type == "scalar":
                            value = preview_scalar(art)
                            st.metric(label="Value", value=value)

                        elif preview_type == "image":
                            image_path = preview_image(art)
                            if image_path and image_path.exists():
                                # [Modified] use_column_width=True -> width="stretch"
                                st.image(str(image_path), caption=art.artifactKey, width="stretch")
                            else:
                                st.warning("Image file missing")

                        elif preview_type == "json_table":
                            table_data = preview_json_table(art)
                            if table_data:
                                # Use static table for cleaner look
                                st.table([dict(zip(table_data["columns"], row)) for row in table_data["rows"]])
                            else:
                                st.warning("Invalid table data")

                        elif preview_type == "json_raw":
                            json_content = preview_json_raw(art)
                            if json_content:
                                st.code(json_content, language="json")
                            else:
                                st.warning("Empty JSON")

                        elif preview_type == "file":
                            path = reader.get_artifact_path(art.artifactKey)
                            if path:
                                f_col1, f_col2 = st.columns([4, 1])
                                with f_col1:
                                    st.code(str(path), language="text")
                                with f_col2:
                                    if path.exists():
                                        st.download_button(
                                            "⬇️ Download",
                                            data=path.read_bytes(),
                                            file_name=path.name,
                                            mime=art.mimeType or "application/octet-stream",
                                            use_container_width=True # Button still uses old kwarg? No, replaced below if needed in logic, but standard download_button uses use_container_width in modern versions. If your version deprecated it for buttons too, this should be width="stretch". Let's stick to consistent modern API.
                                        )
                            else:
                                st.warning("File path resolution failed")

                        else:
                            st.info("No preview available for this artifact type.")

st.divider()

# ==========================================
# 4. Logs & Debug
# ==========================================
st.subheader("Logs & System Info")

tab1, tab2, tab3, tab4 = st.tabs(["📜 Algorithm Log", "📤 Stdout", "⚠️ Stderr", "🔍 Manifest JSON"])

with tab1:
    log = reader.read_algorithm_log()
    if log:
        st.code(log, language="text")
    else:
        st.caption("No algorithm log available.")

with tab2:
    stdout = reader.read_stdout()
    if stdout:
        st.code(stdout, language="text")
    else:
        st.caption("No stdout recorded.")

with tab3:
    stderr = reader.read_stderr()
    if stderr:
        st.code(stderr, language="text")
    else:
        st.caption("No stderr recorded.")

with tab4:
    if reader.has_manifest():
        manifest = reader.read_manifest()
        st.code(json.dumps(manifest.model_dump(exclude_none=True), indent=2), language="json")
    else:
        st.caption("No manifest file.")

st.divider()

# ==========================================
# 5. Danger Zone
# ==========================================
with st.expander("🗑️ Danger Zone", expanded=False):
    st.markdown("Deleting a run is irreversible. It will remove all artifacts and logs.")
    
    confirm_key = f"confirm_del_run_{run.run_id}"
    
    if st.button("Delete This Run", key=f"del_btn_{run.run_id}", type="secondary"):
        st.session_state[confirm_key] = True

    if st.session_state.get(confirm_key):
        st.warning(f"Are you sure you want to delete Run `{run.run_id}`?")
        d_c1, d_c2 = st.columns([1, 1])
        with d_c1:
            # [Modified] use_container_width -> width="stretch"
            if st.button("Yes, Delete", key=f"yes_del_{run.run_id}", type="primary", width="stretch"):
                try:
                    run_manager.delete_run(run.run_id)
                    st.success(f"Deleted run {run.run_id}")
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
        with d_c2:
            # [Modified] use_container_width -> width="stretch"
            if st.button("Cancel", key=f"no_del_{run.run_id}", width="stretch"):
                st.session_state.pop(confirm_key, None)
                st.rerun()
