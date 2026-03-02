"""Algorithm editor page."""

import json
import shutil

import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.workbench import ui_styles

st.set_page_config(page_title="Algorithm Editor", page_icon="✏️", layout="wide", initial_sidebar_state="expanded")
st.title("Algorithm Editor")

ui_styles.apply_global_styles()

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
    st.info("No algorithms imported. Use the Snapshots page to import one.")
    st.stop()

# --- 1. Selection Bar ---
with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    with c1:
        algo_options = {f"{a.algorithm_id}:{a.version}": a for a in algos}
        selected_key = st.selectbox(
            "Select Algorithm",
            options=list(algo_options.keys()),
            format_func=lambda k: f"{algo_options[k].algorithm_id} (v{algo_options[k].version})",
            label_visibility="collapsed"
        )
    with c2:
        if selected_key:
            selected = algo_options[selected_key]
            st.caption(f"ID: `{selected.algorithm_id}`")

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
    new_version = st.text_input("Target Version", value=current_version, help="Change this to save as a new version")

# --- 3. Editor Area ---
st.caption(f"Editing: `{algo_file}`")
edited_text = st.text_area(
    "Code Editor",
    value=algo_text,
    height=600,
    label_visibility="collapsed"
)

# --- Save Logic ---
with col_save:
    # Button aligned with the input box visually
    st.write("") 
    st.write("") 
    if st.button("💾 Save Changes", type="primary", width="stretch"):
        try:
            target_version = (new_version or "").strip()
            if not target_version:
                st.error("Target Version is required.")
                raise ValueError("Target Version is required.")

            if target_version == selected.version:
                # Save in-place (same version).
                algo_dir.mkdir(parents=True, exist_ok=True)
                algo_file.write_text(edited_text, encoding="utf-8")
                if manifest:
                    manifest["codeVersion"] = target_version
                    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                st.toast("File saved successfully", icon="✅")
            else:
                # Save as new version: keep source version untouched.
                new_dir = config.algorithms_dir / selected.algorithm_id / target_version
                if new_dir.exists():
                    st.error(f"Target version already exists: {target_version}")
                    raise ValueError(f"Target version already exists: {target_version}")

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

                st.toast(f"Saved as new version: {target_version}", icon="✅")
                st.success(f"New workspace created at: `{new_dir}`")
        except Exception as e:
            st.error(f"Save failed: {e}")
