"""Algorithm editor page."""

import json
import shutil

import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.workbench.i18n import page_title, tx

st.set_page_config(
    page_title=page_title("algorithm_edit"),
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui_styles.apply_global_styles("algorithm_edit")
ui_styles.render_page_header(tx("Algorithm Editor", "算法代码编辑"))

# --- Page-Specific CSS: Editor Styling ---
st.markdown("""
<style>
    /* Make Text Area look like a code editor */
    textarea {
        font-family: "Source Code Pro", "Consolas", "Courier New", monospace !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
        color: #333 !important;
        background-color: #fcfcfc !important;
    }
</style>
""", unsafe_allow_html=True)


config = SDKConfig()
algo_lib = AlgorithmLibrary(config)
algos = algo_lib.list_algorithms()

if not algos:
    st.info(tx("No algorithms imported. Use the Snapshots page to import one.", "还没有已导入的算法。请先在“快照管理”页面导入。"))
    st.stop()

# --- 1. Selection Bar ---
with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    with c1:
        algo_options = {f"{a.algorithm_id}:{a.version}": a for a in algos}
        selected_key = st.selectbox(
            tx("Select Algorithm", "选择算法"),
            options=list(algo_options.keys()),
            format_func=lambda k: f"{algo_options[k].algorithm_id} (v{algo_options[k].version})",
            label_visibility="collapsed",
        )
    with c2:
        if selected_key:
            selected = algo_options[selected_key]
            st.caption(tx("ID: `{algorithm_id}`", "ID：`{algorithm_id}`", algorithm_id=selected.algorithm_id))

if not selected_key:
    st.stop()

# Load Data
handle = algo_lib.get_algorithm(selected.algorithm_id, selected.version)
algo_dir = handle.directory
algo_file = algo_dir / "main.py"
manifest_file = algo_dir / "manifest.json"

algo_text = algo_file.read_text(encoding="utf-8") if algo_file.exists() else ""
manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
current_version = manifest.get("codeVersion", selected.version)

# --- 2. Action Bar (Specific to Algorithm Edit: Versioning) ---
col_ver, col_spacer, col_save = st.columns([2, 4, 1])

with col_ver:
    new_version = st.text_input(
        tx("Target Version", "目标版本"),
        value=current_version,
        help=tx("Change this to save as a new version", "修改后将另存为新版本"),
    )

# --- 3. Editor Area ---
st.caption(tx("Editing: `{path}`", "正在编辑：`{path}`", path=algo_file))
edited_text = st.text_area(
    tx("Code Editor", "代码编辑器"),
    value=algo_text,
    height=600,
    label_visibility="collapsed",
)

# --- Save Logic ---
with col_save:
    # Button aligned with the input box visually
    st.write("")
    st.write("")
    if st.button(tx("💾 Save Changes", "💾 保存修改"), type="primary", width="stretch"):
        try:
            target_version = (new_version or "").strip()
            if not target_version:
                st.error(tx("Target Version is required.", "目标版本不能为空。"))
                raise ValueError(tx("Target Version is required.", "目标版本不能为空。"))

            if target_version == selected.version:
                # Save in-place (same version).
                algo_dir.mkdir(parents=True, exist_ok=True)
                algo_file.write_text(edited_text, encoding="utf-8")
                if manifest:
                    manifest["codeVersion"] = target_version
                    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                st.toast(tx("File saved successfully", "文件已保存"), icon="✅")
            else:
                # Save as new version: keep source version untouched.
                new_dir = config.algorithms_dir / selected.algorithm_id / target_version
                if new_dir.exists():
                    st.error(
                        tx(
                            "Target version already exists: {version}",
                            "目标版本已存在：{version}",
                            version=target_version,
                        )
                    )
                    raise ValueError(
                        tx(
                            "Target version already exists: {version}",
                            "目标版本已存在：{version}",
                            version=target_version,
                        )
                    )

                shutil.copytree(algo_dir, new_dir)
                (new_dir / "main.py").write_text(edited_text, encoding="utf-8")

                new_manifest_path = new_dir / "manifest.json"
                if new_manifest_path.exists():
                    new_manifest = json.loads(new_manifest_path.read_text(encoding="utf-8"))
                else:
                    new_manifest = {}
                new_manifest["codeVersion"] = target_version
                new_manifest_path.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
                algo_lib.import_algorithm(new_dir)

                st.toast(
                    tx(
                        "Saved as new version: {version}",
                        "已保存为新版本：{version}",
                        version=target_version,
                    ),
                    icon="✅",
                )
                st.success(
                    tx(
                        "New workspace created at: `{path}`",
                        "已创建新工作区：`{path}`",
                        path=new_dir,
                    )
                )
        except Exception as e:
            st.error(tx("Save failed: {error}", "保存失败：{error}", error=e))
