"""Algorithm export page."""

from __future__ import annotations

import io
import json
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.devkit import (
    validate_algorithm_signature,
    validate_inputspec,
    validate_output_contract,
)
from fraclab_sdk.snapshot import SnapshotLibrary
from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.workbench.utils import format_snapshot_option

st.set_page_config(page_title="Export Algorithm", page_icon="📦", layout="wide", initial_sidebar_state="expanded")
st.title("Export Algorithm")

ui_styles.apply_global_styles()

# --- Page-Specific CSS ---
st.markdown("""
<style>
    /* Status badge styling */
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-ok { background-color: #d1fae5; color: #065f46; }
    .status-missing { background-color: #fee2e2; color: #991b1b; }
    .status-warning { background-color: #fef3c7; color: #92400e; }
</style>
""", unsafe_allow_html=True)


config = SDKConfig()
algo_lib = AlgorithmLibrary(config)
snap_lib = SnapshotLibrary(config)

algos = algo_lib.list_algorithms()
if not algos:
    st.info("No algorithms imported. Use Snapshots page to import or create one.")
    st.stop()

# ==========================================
# 1. Source Selection
# ==========================================
st.subheader("1. Select Algorithm Source")

with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    with c1:
        algo_options = {f"{a.algorithm_id}:{a.version}": a for a in algos}
        selected_key = st.selectbox(
            "Target Algorithm",
            options=list(algo_options.keys()),
            format_func=lambda k: f"{algo_options[k].algorithm_id} (v{algo_options[k].version})",
            label_visibility="collapsed"
        )
    with c2:
        if selected_key:
            selected_algo = algo_options[selected_key]
            st.caption(f"ID: `{selected_algo.algorithm_id}`")

if not selected_key:
    st.stop()

selected_algo = algo_options[selected_key]
handle = algo_lib.get_algorithm(selected_algo.algorithm_id, selected_algo.version)
algo_dir = handle.directory

# File paths
manifest_path = algo_dir / "manifest.json"
params_schema_path = algo_dir / "dist" / "params.schema.json"
output_contract_path = algo_dir / "dist" / "output_contract.json"
ds_path = algo_dir / "dist" / "ds.json"
drs_path = algo_dir / "dist" / "drs.json"

# ==========================================
# 2. Validation Status
# ==========================================
st.subheader("2. Validation Status")


def _get_algo_mtime(algo_dir: Path) -> float:
    """Get max mtime of source files for cache invalidation."""
    files = [
        algo_dir / "main.py",
        algo_dir / "manifest.json",
    ]
    # schema/*.py files
    schema_dir = algo_dir / "schema"
    if schema_dir.exists():
        files.extend(schema_dir.glob("*.py"))

    # dist/*.json (if exists)
    dist_dir = algo_dir / "dist"
    if dist_dir.exists():
        files.extend(dist_dir.glob("*.json"))

    mtimes = [f.stat().st_mtime for f in files if f.exists()]
    return max(mtimes) if mtimes else 0.0


@st.cache_data(ttl=60)
def _run_all_validations(algo_dir_str: str, _mtime: float) -> dict[str, dict]:
    """Run all validations. Cached by (path, mtime)."""
    workspace = Path(algo_dir_str)
    results: dict[str, dict] = {}

    # InputSpec validation
    try:
        inputspec_result = validate_inputspec(workspace)
        results["inputspec"] = {
            "valid": inputspec_result.valid,
            "errors": len(inputspec_result.errors),
            "warnings": len(inputspec_result.warnings),
            "issues": [
                {
                    "severity": i.severity.value,
                    "code": i.code,
                    "message": i.message,
                    "path": i.path,
                    "details": i.details,
                }
                for i in inputspec_result.issues
            ],
        }
    except Exception as e:
        results["inputspec"] = {
            "valid": False,
            "errors": 1,
            "warnings": 0,
            "issues": [{"severity": "error", "code": "VALIDATION_FAILED", "message": str(e), "path": None, "details": {}}],
        }

    # OutputContract validation
    try:
        output_result = validate_output_contract(workspace)
        results["output_contract"] = {
            "valid": output_result.valid,
            "errors": len(output_result.errors),
            "warnings": len(output_result.warnings),
            "issues": [
                {
                    "severity": i.severity.value,
                    "code": i.code,
                    "message": i.message,
                    "path": i.path,
                    "details": i.details,
                }
                for i in output_result.issues
            ],
        }
    except Exception as e:
        results["output_contract"] = {
            "valid": False,
            "errors": 1,
            "warnings": 0,
            "issues": [{"severity": "error", "code": "VALIDATION_FAILED", "message": str(e), "path": None, "details": {}}],
        }

    # Algorithm signature validation
    try:
        algo_result = validate_algorithm_signature(workspace)
        results["algorithm"] = {
            "valid": algo_result.valid,
            "errors": len(algo_result.errors),
            "warnings": len(algo_result.warnings),
            "issues": [
                {
                    "severity": i.severity.value,
                    "code": i.code,
                    "message": i.message,
                    "path": i.path,
                    "details": i.details,
                }
                for i in algo_result.issues
            ],
        }
    except Exception as e:
        results["algorithm"] = {
            "valid": False,
            "errors": 1,
            "warnings": 0,
            "issues": [{"severity": "error", "code": "VALIDATION_FAILED", "message": str(e), "path": None, "details": {}}],
        }

    return results


# Run validations with caching
mtime = _get_algo_mtime(algo_dir)
validation_results = _run_all_validations(str(algo_dir), mtime)

# Display validation status badges with revalidate button
with st.container(border=True):
    cols = st.columns([1, 1, 1, 0.5])
    names = ["InputSpec", "OutputContract", "Algorithm"]
    keys = ["inputspec", "output_contract", "algorithm"]

    for i, (key, name) in enumerate(zip(keys, names)):
        result = validation_results.get(key, {})
        error_count = result.get("errors", 0)
        warning_count = result.get("warnings", 0)

        with cols[i]:
            if error_count > 0:
                st.markdown(
                    f'<span class="status-badge status-missing">❌ {name}: {error_count} error{"s" if error_count > 1 else ""}</span>',
                    unsafe_allow_html=True,
                )
            elif warning_count > 0:
                st.markdown(
                    f'<span class="status-badge status-warning">⚠️ {name}: {warning_count} warning{"s" if warning_count > 1 else ""}</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="status-badge status-ok">✅ {name}</span>',
                    unsafe_allow_html=True,
                )

    with cols[3]:
        if st.button("🔄 Revalidate", key="rerun_validation"):
            _run_all_validations.clear()
            st.rerun()

# Collect all issues
all_issues = []
for key in keys:
    result = validation_results.get(key, {})
    for issue in result.get("issues", []):
        all_issues.append((key, issue))

# Show validation details if there are issues
if all_issues:
    with st.expander(f"📋 Validation Details ({len(all_issues)} issue{'s' if len(all_issues) > 1 else ''})", expanded=False):
        for source, issue in all_issues:
            icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(issue["severity"], "⚪")
            path_str = f" at `{issue['path']}`" if issue.get("path") else ""
            details_str = ""
            if issue.get("details"):
                # Show suggested fix for snake_case issues
                if "suggested" in issue["details"]:
                    details_str = f" → Suggested: `{issue['details']['suggested']}`"
                elif "missing" in issue["details"]:
                    details_str = f" (missing: {issue['details']['missing']})"
                elif "extra" in issue["details"]:
                    details_str = f" (extra: {issue['details']['extra']})"
            st.markdown(f"{icon} **[{source}]** `{issue['code']}`{path_str}: {issue['message']}{details_str}")

# File Inspector
with st.expander("📂 File Inspector", expanded=True):
    tab_man, tab_in, tab_out = st.tabs(["Manifest", "Input Spec", "Output Spec"])
    inspector_height = 320

    def _show_json_preview(path: Path):
        if path.exists():
            try:
                data = json.loads(path.read_text())
                st.code(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    language="json",
                    line_numbers=True,
                    height=inspector_height,
                )
            except Exception:
                st.error("Failed to parse JSON")
        else:
            st.info("File not generated yet.")

    with tab_man: _show_json_preview(manifest_path)
    with tab_in: _show_json_preview(params_schema_path)
    with tab_out: _show_json_preview(output_contract_path)


# ==========================================
# 3. DS/DRS Source Selection
# ==========================================
st.subheader("3. Select DS/DRS Source")

snapshots = snap_lib.list_snapshots()
snapshot_map = {s.snapshot_id: s for s in snapshots}

if not snapshots:
    st.warning("No snapshots available. Import a snapshot first to provide DS/DRS for export.")
    st.stop()

with st.container(border=True):
    st.caption(
        "Select a snapshot to provide DS/DRS (dist/ds.json and dist/drs.json) "
        "for the export package."
    )

    selected_snapshot_id = st.selectbox(
        "Snapshot (DS/DRS Source)",
        options=list(snapshot_map.keys()),
        format_func=lambda x: format_snapshot_option(snapshot_map[x]),
        label_visibility="collapsed"
    )

if not selected_snapshot_id:
    st.stop()

snapshot_handle = snap_lib.get_snapshot(selected_snapshot_id)
bundle_ds_path = snapshot_handle.directory / snapshot_handle.manifest.specFiles.dsPath
bundle_drs_path = snapshot_handle.directory / snapshot_handle.manifest.specFiles.drsPath

with st.container(border=True):
    st.caption("Selected Bundle DS/DRS Preview")
    tab_bundle_ds, tab_bundle_drs = st.tabs(["Bundle ds.json", "Bundle drs.json"])
    preview_height = 320

    def _show_bundle_spec(path: Path) -> None:
        if not path.exists():
            st.error(f"File not found: {path.name}")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")
            return
        st.code(
            json.dumps(data, indent=2, ensure_ascii=False),
            language="json",
            line_numbers=True,
            height=preview_height,
        )

    with tab_bundle_ds:
        _show_bundle_spec(bundle_ds_path)
    with tab_bundle_drs:
        _show_bundle_spec(bundle_drs_path)

# ==========================================
# 4. Export
# ==========================================
st.divider()
st.subheader("4. Export")

def build_zip() -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # copy installed algorithm content
        shutil.copytree(algo_dir, tmpdir_path / algo_dir.name, dirs_exist_ok=True)
        target_root = tmpdir_path / algo_dir.name

        # ensure manifest files paths cover dist outputs if present
        manifest_data = json.loads(manifest_path.read_text())
        files = manifest_data.get("files") or {}

        if output_contract_path.exists():
            files["outputContractPath"] = "dist/output_contract.json"
        if params_schema_path.exists():
            files["paramsSchemaPath"] = "dist/params.schema.json"
        if ds_path.exists():
            files["dsPath"] = "dist/ds.json"
        if drs_path.exists():
            files["drsPath"] = "dist/drs.json"

        if files:
            manifest_data["files"] = files

        # DRS Override Logic
        files_section = manifest_data.get("files")
        if not isinstance(files_section, dict):
            raise ValueError("manifest.json missing required files section")
        if not isinstance(files_section.get("dsPath"), str) or not files_section.get("dsPath"):
            files_section["dsPath"] = "dist/ds.json"
        if not isinstance(files_section.get("drsPath"), str) or not files_section.get("drsPath"):
            files_section["drsPath"] = "dist/drs.json"
        drs_rel_path = files_section.get("drsPath")
        if not isinstance(drs_rel_path, str) or not drs_rel_path:
            raise ValueError("manifest.json missing required files.drsPath")
        ds_rel_path = files_section.get("dsPath")
        if not isinstance(ds_rel_path, str) or not ds_rel_path:
            raise ValueError("manifest.json missing required files.dsPath")
        target_ds_path = target_root / ds_rel_path
        target_ds_path.parent.mkdir(parents=True, exist_ok=True)
        target_drs_path = target_root / drs_rel_path
        target_drs_path.parent.mkdir(parents=True, exist_ok=True)

        # Read DS/DRS from Snapshot
        snap_ds_path = bundle_ds_path
        snap_drs_path = bundle_drs_path

        if snap_ds_path.exists():
            target_ds_path.write_bytes(snap_ds_path.read_bytes())
        else:
            raise ValueError("Selected snapshot is missing ds.json")

        if snap_drs_path.exists():
            target_drs_path.write_bytes(snap_drs_path.read_bytes())
        else:
            raise ValueError("Selected snapshot is missing drs.json")

        # Persist final manifest after files.* normalization/overrides.
        (target_root / "manifest.json").write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        # Zip it up (flattened: no top-level version folder)
        zip_buf = io.BytesIO()
        shutil.make_archive(
            base_name=tmpdir_path / "algorithm_export",
            format="zip",
            root_dir=target_root,
            base_dir=".",
        )
        zip_path = tmpdir_path / "algorithm_export.zip"
        zip_buf.write(zip_path.read_bytes())
        zip_buf.seek(0)
        return zip_buf.read()


# Check if there are validation errors
has_validation_errors = any(not r.get("valid", True) for r in validation_results.values())

_, col_export_btn = st.columns([3, 1])
with col_export_btn:
    if has_validation_errors:
        st.error("Fix validation errors to export")
        st.button("📦 Build & Export", type="primary", disabled=True, key="export_disabled")
    else:
        if st.button("📦 Build & Export", type="primary", key="export_enabled"):
            try:
                with st.spinner("Packaging..."):
                    zip_bytes = build_zip()

                st.download_button(
                    label="⬇️ Download Zip",
                    data=zip_bytes,
                    file_name=f"{selected_algo.algorithm_id}-{selected_algo.version}.zip",
                    mime="application/zip",
                )
            except Exception as e:
                st.error(f"Export failed: {e}")
