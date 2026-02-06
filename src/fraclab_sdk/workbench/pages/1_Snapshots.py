"""Snapshots management page."""

import json
import re
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.errors import SnapshotError
from fraclab_sdk.snapshot import SnapshotLibrary
from fraclab_sdk.version import __version__ as SDK_VERSION
from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.workbench.utils import get_workspace_dir

st.set_page_config(page_title="Snapshots", page_icon="📦", layout="wide", initial_sidebar_state="expanded")
st.title("Snapshots")

ui_styles.apply_global_styles()


config = SDKConfig()
config.ensure_dirs()
WORKSPACE_ROOT = get_workspace_dir(config)
snapshot_lib = SnapshotLibrary(config)
algorithm_lib = AlgorithmLibrary(config)


BASE_SCHEMA_UTILS = '''"""Schema base utilities for json_schema_extra helpers."""

from typing import Any


def show_when_condition(field: str, op: str = "equals", value: Any = True) -> dict[str, Any]:
    return {"field": field, "op": op, "value": value}


def show_when_and(*conditions: dict[str, Any]) -> dict[str, Any]:
    return {"and": list(conditions)}


def show_when_or(*conditions: dict[str, Any]) -> dict[str, Any]:
    return {"or": list(conditions)}


def schema_extra(
    *,
    group: str | None = None,
    order: int | None = None,
    unit: str | None = None,
    step: float | None = None,
    ui_type: str | None = None,
    collapsible: bool | None = None,
    show_when: dict[str, Any] | None = None,
    enum_labels: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if group is not None:
        result["group"] = group
    if order is not None:
        result["order"] = order
    if unit is not None:
        result["unit"] = unit
    if step is not None:
        result["step"] = step
    if ui_type is not None:
        result["ui_type"] = ui_type
    if collapsible is not None:
        result["collapsible"] = collapsible
    if show_when is not None:
        result["show_when"] = show_when
    if enum_labels is not None:
        result["enum_labels"] = enum_labels
    result.update(kwargs)
    return result
'''


def create_algorithm_scaffold(
    algo_id: str,
    code_version: str,
    contract_version: str,
    name: str,
    summary: str,
    authors: list[dict[str, str]],
    notes: str | None = None,
    tags: list[str] | None = None,
    *,
    workspace_root: Path,
) -> Path:
    """Create a new algorithm workspace with minimal files."""
    ws_dir = workspace_root / algo_id / code_version
    if ws_dir.exists():
        raise FileExistsError(f"Algorithm workspace already exists: {ws_dir}")
    ws_dir.mkdir(parents=True, exist_ok=True)

    authors_list = [
        {
            "name": (a.get("name") or "").strip(),
            "email": (a.get("email") or "").strip(),
            "organization": (a.get("organization") or "").strip(),
        }
        for a in authors
    ]
    authors_list = [a for a in authors_list if any(v for v in a.values())] or [{"name": "unknown"}]

    summary_val = summary.strip() or f"Algorithm {algo_id}"
    manifest = {
        "manifestVersion": "1",
        "algorithmId": algo_id,
        "name": name or algo_id,
        "summary": summary_val,
        "authors": authors_list,
        "contractVersion": contract_version,
        "codeVersion": code_version,
        "notes": notes or None,
        "tags": tags or None,
        "files": {
            "paramsSchemaPath": "dist/params.schema.json",
            "drsPath": "dist/drs.json",
            "outputContractPath": "dist/output_contract.json",
        },
        "requires": {"sdk": SDK_VERSION},
        "repository": None,
        "homepage": None,
        "license": None,
    }

    (ws_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dist_dir = ws_dir / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "drs.json").write_text(json.dumps({"datasets": []}, indent=2), encoding="utf-8")
    (dist_dir / "params.schema.json").write_text(
        json.dumps({"type": "object", "title": "Parameters", "properties": {}}, indent=2),
        encoding="utf-8",
    )
    (dist_dir / "output_contract.json").write_text(
        json.dumps({"datasets": [], "invariants": [], "relations": []}, indent=2),
        encoding="utf-8",
    )

    main_stub = '''"""Algorithm entrypoint."""

from __future__ import annotations

from fraclab_sdk.runtime.data_client import DataClient


def run(client: DataClient, params: dict) -> dict:
    """Implement algorithm logic here."""
    # TODO: replace with real logic
    return {"plots": [], "metrics": [], "diagnostics_zip": []}
'''
    (ws_dir / "main.py").write_text(main_stub, encoding="utf-8")

    schema_dir = ws_dir / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / "__init__.py").write_text("", encoding="utf-8")
    (schema_dir / "base.py").write_text(BASE_SCHEMA_UTILS, encoding="utf-8")

    return ws_dir


def render_manifest_fields(
    *,
    name: str,
    summary: str,
    contract_version: str,
    code_version: str,
    authors: list[dict],
    tags: list[str] | None,
    notes: str | None,
    key_prefix: str,
) -> dict:
    """Render common manifest fields and return updated values."""
    name_val = st.text_input("Name", value=name, key=f"{key_prefix}_name")
    summary_val = st.text_area("Summary", value=summary, key=f"{key_prefix}_summary")
    
    c1, c2 = st.columns(2)
    with c1:
        contract_val = st.text_input("Contract Version", value=contract_version, key=f"{key_prefix}_contract")
    with c2:
        code_val = st.text_input("Code Version", value=code_version, key=f"{key_prefix}_code")
        
    st.markdown("---")
    st.caption("Authors Info")
    
    authors_entries = authors or [{"name": "", "email": "", "organization": ""}]
    author_count = st.number_input(
        "Authors count",
        min_value=1,
        max_value=max(len(authors_entries), 10),
        value=len(authors_entries),
        step=1,
        key=f"{key_prefix}_author_count",
    )
    # ensure list length matches count
    if author_count > len(authors_entries):
        authors_entries.extend([{"name": "", "email": "", "organization": ""}] * (author_count - len(authors_entries)))
    elif author_count < len(authors_entries):
        authors_entries = authors_entries[:author_count]

    authors_val: list[dict] = []
    for idx in range(int(author_count)):
        author = authors_entries[idx]
        cols = st.columns(3)
        with cols[0]:
            name_a = st.text_input(
                f"Author {idx+1} Name",
                value=author.get("name", ""),
                key=f"{key_prefix}_author_name_{idx}",
            )
        with cols[1]:
            email_a = st.text_input(
                f"Author {idx+1} Email",
                value=author.get("email", ""),
                key=f"{key_prefix}_author_email_{idx}",
            )
        with cols[2]:
            org_a = st.text_input(
                f"Author {idx+1} Organization",
                value=author.get("organization", ""),
                key=f"{key_prefix}_author_org_{idx}",
            )
        authors_val.append({"name": name_a, "email": email_a, "organization": org_a})
        
    st.markdown("---")
    tags_val = st.text_input(
        "Tags (comma-separated)",
        value=",".join(tags or []),
        key=f"{key_prefix}_tags",
    )
    notes_val = st.text_area("Notes", value=notes or "", key=f"{key_prefix}_notes")

    return {
        "name": name_val,
        "summary": summary_val,
        "contract_version": contract_val,
        "code_version": code_val,
        "authors": [a for a in authors_val if any(v.strip() for v in a.values())] or [{"name": "unknown"}],
        "tags": [t.strip() for t in tags_val.split(",") if t.strip()] or None,
        "notes": notes_val.strip() or None,
    }

# ==========================================
# Dialogs (Modals)
# ==========================================

@st.dialog("Create New Algorithm")
def show_create_algo_dialog():
    with st.form("create_algo_form"):
        algo_id = st.text_input("Algorithm ID (e.g. my-algo)", key="create_algo_id")
        manifest_vals = render_manifest_fields(
            name="",
            summary="",
            contract_version="1.0.0",
            code_version="0.1.0",
            authors=[{"name": "Your Name", "email": "", "organization": ""}],
            tags=None,
            notes=None,
            key_prefix="create_algo",
        )
        
        f_c1, f_c2 = st.columns([1, 4])
        with f_c1:
            # Updated API
            create_submit = st.form_submit_button("Create", type="primary", width="stretch")
        with f_c2:
            pass # form layout spacer

    if create_submit:
        if not algo_id or not manifest_vals["code_version"]:
            st.error("Algorithm ID and Code Version are required.")
        elif not re.match(r"^[A-Za-z0-9_-]+$", algo_id):
            st.error("Algorithm ID may only contain letters, numbers, _ or -.")
        else:
            try:
                ws_dir = create_algorithm_scaffold(
                    algo_id=algo_id,
                    code_version=manifest_vals["code_version"],
                    contract_version=manifest_vals["contract_version"],
                    name=manifest_vals["name"] or algo_id,
                    summary=manifest_vals["summary"],
                    authors=manifest_vals["authors"],
                    notes=manifest_vals["notes"],
                    tags=manifest_vals["tags"],
                    workspace_root=WORKSPACE_ROOT,
                )
                algo_id, version = algorithm_lib.import_algorithm(ws_dir)
                st.success(f"Created and imported: {algo_id} v{version}")
                st.rerun()
            except FileExistsError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Create failed: {e}")


@st.dialog("Edit Manifest")
def show_edit_manifest_dialog(algo_id, version, manifest_path):
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to load manifest: {e}")
        return

    files_section = manifest_data.get("files") or {}
    default_files = {
        "paramsSchemaPath": files_section.get("paramsSchemaPath", "dist/params.schema.json"),
        "drsPath": files_section.get("drsPath", "dist/drs.json"),
        "outputContractPath": files_section.get("outputContractPath", "dist/output_contract.json"),
    }

    with st.form(f"manifest_form_{algo_id}_{version}"):
        manifest_vals = render_manifest_fields(
            name=manifest_data.get("name", ""),
            summary=manifest_data.get("summary", ""),
            contract_version=manifest_data.get("contractVersion", ""),
            code_version=manifest_data.get("codeVersion", ""),
            authors=manifest_data.get("authors") or [{"name": ""}],
            tags=manifest_data.get("tags"),
            notes=manifest_data.get("notes"),
            key_prefix=f"manifest_{algo_id}_{version}",
        )
        save_submit = st.form_submit_button("Save Changes", type="primary")

    if save_submit:
        try:
            manifest_data["name"] = manifest_vals["name"]
            manifest_data["summary"] = manifest_vals["summary"]
            manifest_data["contractVersion"] = manifest_vals["contract_version"]
            manifest_data["codeVersion"] = manifest_vals["code_version"]
            manifest_data["authors"] = [
                a for a in manifest_vals["authors"] if any(v.strip() for v in a.values())
            ] or [{"name": "unknown"}]
            manifest_data["notes"] = manifest_vals["notes"]
            manifest_data["tags"] = manifest_vals["tags"]
            manifest_data["files"] = default_files
            manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
            st.success("Manifest saved successfully")
            st.rerun()
        except Exception as e:
            st.error(f"Save failed: {e}")


# ==========================================
# 1. Snapshot Management
# ==========================================
st.subheader("Import Snapshot")

with st.container(border=True):
    # 1. File Uploader
    uploaded_snapshot = st.file_uploader(
        "Upload Snapshot (zip file)",
        type=["zip"],
        label_visibility="collapsed",
        key="snapshot_uploader",
    )

    # 2. Conditional Layout: Filename + Import Button
    if uploaded_snapshot is not None:
        # 使用列布局：左侧文件名，右侧按钮紧凑排列
        c_name, c_btn = st.columns([5, 1], gap="small")
        with c_name:
            # 垂直居中文件名文本
            st.markdown(f"<div style='padding-top: 5px; color: #444;'>📄 <b>{uploaded_snapshot.name}</b> <small>({uploaded_snapshot.size / 1024:.1f} KB)</small></div>", unsafe_allow_html=True)
        with c_btn:
            # Updated API
            if st.button("Import Snapshot", type="primary", key="import_snapshot_btn", width="stretch"):
                with st.spinner("Importing snapshot..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                            tmp_file.write(uploaded_snapshot.getvalue())
                            tmp_path = Path(tmp_file.name)

                        snapshot_id = snapshot_lib.import_snapshot(tmp_path)
                        st.success(f"Imported: {snapshot_id}")
                        tmp_path.unlink(missing_ok=True)
                        st.rerun()
                    except SnapshotError as e:
                        st.error(f"Import failed: {e}")
                    except Exception as e:
                        st.error(f"Error: {e}")

st.divider()

st.subheader("Imported Snapshots")

snapshots = snapshot_lib.list_snapshots()

if not snapshots:
    st.info("No snapshots imported yet")
else:
    for snap in snapshots:
        with st.expander(f"📦 {snap.snapshot_id}", expanded=False):
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 5, 1])
                
                with c1:
                    st.caption("Bundle ID")
                    st.code(snap.bundle_id, language="text")
                    st.caption("Imported At")
                    st.markdown(f"**{snap.imported_at}**")

                with c2:
                    st.caption("DRS (Data Requirement Specification)")
                    # 获取完整的 Snapshot 对象以读取 DRS
                    try:
                        full_snap = snapshot_lib.get_snapshot(snap.snapshot_id)
                        drs_data = full_snap.drs.model_dump(exclude_none=True)
                        st.code(json.dumps(drs_data, indent=2, ensure_ascii=False), language="json")
                    except Exception as e:
                        st.error(f"Failed to load DRS: {e}")

                with c3:
                    st.write("") # Spacer
                    if st.button("Delete", key=f"del_{snap.snapshot_id}", type="secondary"):
                        try:
                            snapshot_lib.delete_snapshot(snap.snapshot_id)
                            st.success(f"Deleted")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

st.divider()

# ==========================================
# 2. Algorithm Management
# ==========================================
st.subheader("Algorithm Workspace")

# Action Bar
col_create, col_spacer = st.columns([1, 4])
with col_create:
    # Updated API
    if st.button("✨ Create New Algorithm", key="create_algo_btn", width="stretch"):
        show_create_algo_dialog()

# Upload Section (Expanded by Default)
with st.expander("📤 Import Existing Algorithm", expanded=True):
    uploaded_algorithm = st.file_uploader(
        "Upload Algorithm (zip or .py)",
        type=["zip", "py"],
        key="algorithm_uploader",
    )

    if uploaded_algorithm is not None:
        if uploaded_algorithm.name.endswith(".zip"):
             if st.button("Import Algorithm Zip", type="primary", key="import_algo_btn_zip"):
                with st.spinner("Importing algorithm from zip..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                            tmp_file.write(uploaded_algorithm.getvalue())
                            tmp_path = Path(tmp_file.name)
                        algo_id, version = algorithm_lib.import_algorithm(tmp_path)
                        st.success(f"Imported algorithm: {algo_id} v{version}")
                        tmp_path.unlink(missing_ok=True)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Import failed: {e}")
        elif uploaded_algorithm.name.endswith(".py"):
             if st.button("Import Single Python File", type="primary", key="import_algo_btn_py"):
                with st.spinner("Importing algorithm from .py..."):
                    try:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            tmp_dir_path = Path(tmp_dir)

                            # Copy uploaded main.py
                            algo_path = tmp_dir_path / "main.py"
                            algo_path.write_bytes(uploaded_algorithm.getvalue())

                            # Create dist/ with template files
                            dist_dir = tmp_dir_path / "dist"
                            dist_dir.mkdir(parents=True, exist_ok=True)
                            (dist_dir / "drs.json").write_text(json.dumps({"datasets": []}, indent=2))
                            (dist_dir / "params.schema.json").write_text(
                                json.dumps({"type": "object", "title": "Parameters", "properties": {}}, indent=2)
                            )
                            (dist_dir / "output_contract.json").write_text(
                                json.dumps({"datasets": [], "invariants": [], "relations": []}, indent=2)
                            )

                            # Create schema/ with base utilities
                            schema_dir = tmp_dir_path / "schema"
                            schema_dir.mkdir(parents=True, exist_ok=True)
                            (schema_dir / "__init__.py").write_text("", encoding="utf-8")
                            (schema_dir / "base.py").write_text(BASE_SCHEMA_UTILS, encoding="utf-8")

                            # Create manifest
                            algo_id = uploaded_algorithm.name.removesuffix(".py")
                            manifest = {
                                "manifestVersion": "1",
                                "algorithmId": algo_id,
                                "codeVersion": "local",
                                "contractVersion": "1.0.0",
                                "name": algo_id,
                                "summary": "Imported from single python file",
                                "authors": [{"name": "unknown"}],
                                "files": {
                                    "paramsSchemaPath": "dist/params.schema.json",
                                    "drsPath": "dist/drs.json",
                                    "outputContractPath": "dist/output_contract.json",
                                },
                                "requires": {"sdk": SDK_VERSION},
                            }
                            (tmp_dir_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

                            algo_id, version = algorithm_lib.import_algorithm(tmp_dir_path)
                            st.success(f"Imported algorithm: {algo_id} v{version}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Import failed: {e}")

st.divider()

st.subheader("Imported Algorithms")

algorithms = algorithm_lib.list_algorithms()

if not algorithms:
    st.info("No algorithms imported yet")
else:
    for algo in algorithms:
        with st.expander(f"🧩 {algo.algorithm_id} (v{algo.version})", expanded=False):
            with st.container(border=True):
                # Header info
                head_c1, head_c2, head_c3 = st.columns([3, 2, 2])
                with head_c1:
                    st.caption("Name")
                    st.markdown(f"**{algo.name or 'N/A'}**")
                with head_c2:
                    st.caption("Contract")
                    st.code(algo.contract_version or 'N/A', language="text")
                with head_c3:
                    st.caption("Imported")
                    st.text(algo.imported_at)

                st.markdown("---")
                
                # Content info
                st.caption("Summary")
                if getattr(algo, 'summary', ''):
                    st.info(algo.summary)
                else:
                    st.text("No summary provided.")
                
                notes = getattr(algo, "notes", None)
                if notes:
                    st.caption("Notes")
                    st.write(notes)

                st.markdown("---")

                # Actions
                act_c1, act_c2 = st.columns([1, 5])
                
                with act_c1:
                    manifest_path = algorithm_lib.get_algorithm(algo.algorithm_id, algo.version).directory / "manifest.json"
                    if manifest_path.exists():
                        if st.button("📝 Edit Manifest", key=f"edit_manifest_btn_{algo.algorithm_id}_{algo.version}"):
                            show_edit_manifest_dialog(algo.algorithm_id, algo.version, manifest_path)
                
                with act_c2:
                    if st.button("🗑️ Delete", key=f"del_algo_{algo.algorithm_id}_{algo.version}"):
                        try:
                            algorithm_lib.delete_algorithm(algo.algorithm_id, algo.version)
                            ws_dir = WORKSPACE_ROOT / algo.algorithm_id / algo.version
                            if ws_dir.exists():
                                shutil.rmtree(ws_dir, ignore_errors=True)
                            st.success(f"Deleted")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
