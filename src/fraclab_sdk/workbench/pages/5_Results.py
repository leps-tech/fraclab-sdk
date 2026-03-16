"""Results viewing page."""

import json

import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
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
from fraclab_sdk.workbench.i18n import page_title, run_status_label, tx

st.set_page_config(
    page_title=page_title("results"),
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui_styles.apply_global_styles("results")
ui_styles.render_page_header(tx("Results", "运行结果"))

# Guidance: if artifacts preview looks empty, validate OutputContract via the editor page.
st.info(
    tx(
        "Missing expected outputs? Use Validate on the Output Editor page to check "
        "`schema/output_contract.py` datasets/items/artifacts.",
        "看不到期望的输出？请在输出结果定义页面点击 Validate 检查 "
        "`schema/output_contract.py` 的 datasets/items/artifacts 是否正确。",
    ),
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
    st.info(tx("No runs available.", "没有可用的运行记录。"))
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
            tx("Select Run", "选择运行"),
            options=run_ids,
            index=default_index,
            format_func=lambda x: f"{x} — {run_options[x].algorithm_id} (v{run_options[x].algorithm_version})",
            label_visibility="collapsed",
        )
    
    with col_stat:
        if selected_run_id:
            status = run_options[selected_run_id].status
            status_color = {
                RunStatus.SUCCEEDED: "green",
                RunStatus.FAILED: "red",
                RunStatus.PENDING: "gray",
                RunStatus.RUNNING: "blue",
                RunStatus.TIMEOUT: "orange",
            }.get(status, "gray")
            st.markdown(
                f"<div style='text-align:center; padding: 8px; border: 1px solid #ddd; border-radius: 6px;'>{tx('Status', '状态')}: <b style='color:{status_color}'>{run_status_label(status)}</b></div>",
                unsafe_allow_html=True,
            )

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

with st.expander(tx("ℹ️ Run Metadata", "ℹ️ 运行元数据"), expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption(tx("Snapshot ID", "快照 ID"))
        st.code(run.snapshot_id, language="text")
    with c2:
        st.caption(tx("Algorithm ID", "算法 ID"))
        st.code(run.algorithm_id, language="text")
    with c3:
        st.caption(tx("Timestamps", "时间戳"))
        st.text(
            tx(
                "Start: {start}\nEnd:   {end}",
                "开始：{start}\n结束：{end}",
                start=run.started_at,
                end=run.completed_at,
            )
        )

if run.error:
    st.error(tx("Run Error: {error}", "运行错误：{error}", error=run.error))
elif reader.has_manifest() and reader.get_error():
    st.error(tx("Manifest Error: {error}", "Manifest 错误：{error}", error=reader.get_error()))

st.divider()

# ==========================================
# 3. Artifacts (Default Expanded)
# ==========================================
st.subheader(tx("Artifacts", "产物"))

if not reader.has_manifest():
    st.warning(tx("⚠️ Output manifest not found (Run may have failed or produced no output)", "⚠️ 未找到输出 manifest（运行可能失败，或没有产出结果）"))
else:
    manifest = reader.read_manifest()

    if not manifest.datasets:
        st.info(tx("No artifacts produced.", "没有产生产物。"))
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
                        tx(
                            "**Schema Definition:** Kind=`{kind}` • Owner=`{owner}` • Card=`{card}`",
                            "**Schema 定义：** Kind=`{kind}` • Owner=`{owner}` • Card=`{card}`",
                            kind=contract_ds.get("kind"),
                            owner=contract_ds.get("owner"),
                            card=contract_ds.get("cardinality"),
                        )
                    )
                
                # --- ITEMS LEVEL: CARDS (Always Visible) ---
                for item in ds.items:
                    art = item.artifact
                    preview_type = get_artifact_preview_type(art)
                    
                    with st.container(border=True):
                        # Item Header & Metadata
                        m1, m2, m3 = st.columns([2, 2, 3])
                        with m1:
                            st.markdown(
                                tx("**Item:** `{item_key}`", "**条目：** `{item_key}`", item_key=item.itemKey or art.artifactKey)
                            )
                        with m2:
                            st.caption(tx("Type: `{artifact_type}`", "类型：`{artifact_type}`", artifact_type=art.type))
                        with m3:
                            if art.mimeType:
                                st.caption(tx("MIME: `{mime}`", "MIME：`{mime}`", mime=art.mimeType))
                        
                        # [Modified] 删除了 Owner 和 Dims 的显示
                        st.markdown("---")
                        
                        # Content Preview
                        if preview_type == "scalar":
                            value = preview_scalar(art)
                            st.metric(label=tx("Value", "数值"), value=value)

                        elif preview_type == "image":
                            image_path = preview_image(art)
                            if image_path and image_path.exists():
                                # [Modified] use_column_width=True -> width="stretch"
                                st.image(str(image_path), caption=art.artifactKey, width="stretch")
                            else:
                                st.warning(tx("Image file missing", "图片文件缺失"))

                        elif preview_type == "json_table":
                            table_data = preview_json_table(art)
                            if table_data:
                                # Use static table for cleaner look
                                st.table(
                                    [
                                        dict(zip(table_data["columns"], row, strict=False))
                                        for row in table_data["rows"]
                                    ]
                                )
                            else:
                                st.warning(tx("Invalid table data", "表格数据无效"))

                        elif preview_type == "json_raw":
                            json_content = preview_json_raw(art)
                            if json_content:
                                st.code(json_content, language="json")
                            else:
                                st.warning(tx("Empty JSON", "JSON 为空"))

                        elif preview_type == "file":
                            path = reader.get_artifact_path(art.artifactKey)
                            if path:
                                f_col1, f_col2 = st.columns([4, 1])
                                with f_col1:
                                    st.code(str(path), language="text")
                                with f_col2:
                                    if path.exists():
                                        st.download_button(
                                            tx("⬇️ Download", "⬇️ 下载"),
                                            data=path.read_bytes(),
                                            file_name=path.name,
                                            mime=art.mimeType or "application/octet-stream",
                                            use_container_width=True # Button still uses old kwarg? No, replaced below if needed in logic, but standard download_button uses use_container_width in modern versions. If your version deprecated it for buttons too, this should be width="stretch". Let's stick to consistent modern API.
                                        )
                            else:
                                st.warning(tx("File path resolution failed", "文件路径解析失败"))

                        else:
                            st.info(tx("No preview available for this artifact type.", "这种产物类型暂无预览。"))

st.divider()

# ==========================================
# 4. Logs & Debug
# ==========================================
st.subheader(tx("Logs & System Info", "日志与系统信息"))

tab1, tab2, tab3, tab4 = st.tabs(
    [
        tx("📜 Algorithm Log", "📜 算法日志"),
        tx("📤 Stdout", "📤 标准输出"),
        tx("⚠️ Stderr", "⚠️ 标准错误"),
        tx("🔍 Manifest JSON", "🔍 Manifest JSON"),
    ]
)

with tab1:
    log = reader.read_algorithm_log()
    if log:
        st.code(log, language="text")
    else:
        st.caption(tx("No algorithm log available.", "没有可用的算法日志。"))

with tab2:
    stdout = reader.read_stdout()
    if stdout:
        st.code(stdout, language="text")
    else:
        st.caption(tx("No stdout recorded.", "没有记录标准输出。"))

with tab3:
    stderr = reader.read_stderr()
    if stderr:
        st.code(stderr, language="text")
    else:
        st.caption(tx("No stderr recorded.", "没有记录标准错误。"))

with tab4:
    if reader.has_manifest():
        manifest = reader.read_manifest()
        st.code(json.dumps(manifest.model_dump(exclude_none=True), indent=2), language="json")
    else:
        st.caption(tx("No manifest file.", "没有 manifest 文件。"))

st.divider()

# ==========================================
# 5. Danger Zone
# ==========================================
with st.expander(tx("🗑️ Danger Zone", "🗑️ 危险操作"), expanded=False):
    st.markdown(tx("Deleting a run is irreversible. It will remove all artifacts and logs.", "删除运行不可撤销，它会移除所有产物和日志。"))
    
    confirm_key = f"confirm_del_run_{run.run_id}"
    
    if st.button(tx("Delete This Run", "删除此运行"), key=f"del_btn_{run.run_id}", type="secondary"):
        st.session_state[confirm_key] = True

    if st.session_state.get(confirm_key):
        st.warning(tx("Are you sure you want to delete Run `{run_id}`?", "确定要删除运行 `{run_id}` 吗？", run_id=run.run_id))
        d_c1, d_c2 = st.columns([1, 1])
        with d_c1:
            # [Modified] use_container_width -> width="stretch"
            if st.button(tx("Yes, Delete", "确认删除"), key=f"yes_del_{run.run_id}", type="primary", width="stretch"):
                try:
                    run_manager.delete_run(run.run_id)
                    st.success(tx("Deleted run {run_id}", "已删除运行 {run_id}", run_id=run.run_id))
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
                except Exception as e:
                    st.error(tx("Delete failed: {error}", "删除失败：{error}", error=e))
        with d_c2:
            # [Modified] use_container_width -> width="stretch"
            if st.button(tx("Cancel", "取消"), key=f"no_del_{run.run_id}", width="stretch"):
                st.session_state.pop(confirm_key, None)
                st.rerun()
