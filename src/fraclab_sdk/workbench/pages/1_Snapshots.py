"""Snapshots management page."""

import json
import re
import shutil
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from fraclab_sdk.algorithm.library import AlgorithmLibrary
from fraclab_sdk.algorithm.scaffold import create_algorithm_scaffold, ensure_schema_base
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.errors import SnapshotError
from fraclab_sdk.snapshot import SnapshotLibrary
from fraclab_sdk.version import __version__ as SDK_VERSION
from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.workbench.i18n import page_title, tx
from fraclab_sdk.workbench.utils import get_workspace_dir

st.set_page_config(
    page_title=page_title("snapshots"),
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui_styles.apply_global_styles("snapshots")
ui_styles.render_page_header(tx("Snapshots", "快照管理"))


config = SDKConfig()
config.ensure_dirs()
WORKSPACE_ROOT = get_workspace_dir(config)
snapshot_lib = SnapshotLibrary(config)
algorithm_lib = AlgorithmLibrary(config)

FOCUS_SNAPSHOT_KEY = "snapshots_page_focus_snapshot_id"
FOCUS_ALGO_KEY = "snapshots_page_focus_algorithm_key"
SCROLL_TARGET_KEY = "snapshots_page_scroll_target"
FLASH_MESSAGE_KEY = "snapshots_page_flash_message"


def _set_snapshot_focus(snapshot_id: str) -> None:
    st.session_state[FOCUS_SNAPSHOT_KEY] = snapshot_id


def _set_algorithm_focus(algorithm_id: str, version: str) -> None:
    st.session_state[FOCUS_ALGO_KEY] = f"{algorithm_id}:{version}"


def _set_ui_feedback(message: str, scroll_target: str | None = None) -> None:
    """Store one-shot message/scroll target for the next rerun."""
    st.session_state[FLASH_MESSAGE_KEY] = message
    if scroll_target:
        st.session_state[SCROLL_TARGET_KEY] = scroll_target


def _emit_scroll_to_element(element_id: str) -> None:
    """Best-effort browser scroll to an element id."""
    components.html(
        f"""
<script>
const targetId = {element_id!r};
let tries = 0;
const timer = setInterval(() => {{
  const el = parent.document.getElementById(targetId);
  if (el) {{
    el.scrollIntoView({{ behavior: "smooth", block: "start" }});
    clearInterval(timer);
  }}
  tries += 1;
  if (tries > 20) clearInterval(timer);
}}, 80);
</script>
""",
        height=0,
    )


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
    name_val = st.text_input(tx("Name", "名称"), value=name, key=f"{key_prefix}_name")
    summary_val = st.text_area(tx("Summary", "摘要"), value=summary, key=f"{key_prefix}_summary")
    
    c1, c2 = st.columns(2)
    with c1:
        contract_val = st.text_input(tx("Contract Version", "契约版本"), value=contract_version, key=f"{key_prefix}_contract")
    with c2:
        code_val = st.text_input(tx("Code Version", "代码版本"), value=code_version, key=f"{key_prefix}_code")
        
    st.markdown("---")
    st.caption(tx("Authors Info", "作者信息"))
    
    authors_entries = authors or [{"name": "", "email": "", "organization": ""}]
    author_count = st.number_input(
        tx("Authors count", "作者数量"),
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
                tx("Author {index} Name", "作者 {index} 姓名", index=idx + 1),
                value=author.get("name", ""),
                key=f"{key_prefix}_author_name_{idx}",
            )
        with cols[1]:
            email_a = st.text_input(
                tx("Author {index} Email", "作者 {index} 邮箱", index=idx + 1),
                value=author.get("email", ""),
                key=f"{key_prefix}_author_email_{idx}",
            )
        with cols[2]:
            org_a = st.text_input(
                tx("Author {index} Organization", "作者 {index} 机构", index=idx + 1),
                value=author.get("organization", ""),
                key=f"{key_prefix}_author_org_{idx}",
            )
        authors_val.append({"name": name_a, "email": email_a, "organization": org_a})
        
    st.markdown("---")
    tags_val = st.text_input(
        tx("Tags (comma-separated)", "标签（逗号分隔）"),
        value=",".join(tags or []),
        key=f"{key_prefix}_tags",
    )
    notes_val = st.text_area(tx("Notes", "备注"), value=notes or "", key=f"{key_prefix}_notes")

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

@st.dialog(tx("Create New Algorithm", "创建新算法"))
def show_create_algo_dialog():
    with st.form("create_algo_form"):
        algo_id = st.text_input(tx("Algorithm ID (e.g. my-algo)", "算法 ID（例如 my-algo）"), key="create_algo_id")
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
            create_submit = st.form_submit_button(tx("Create", "创建"), type="primary", width="stretch")
        with f_c2:
            pass # form layout spacer

    if create_submit:
        if not algo_id or not manifest_vals["code_version"]:
            st.error(tx("Algorithm ID and Code Version are required.", "算法 ID 和代码版本不能为空。"))
        elif not re.match(r"^[A-Za-z0-9_-]+$", algo_id):
            st.error(tx("Algorithm ID may only contain letters, numbers, _ or -.", "算法 ID 只能包含字母、数字、_ 或 -。"))
        else:
            ws_dir: Path | None = None
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
                _set_algorithm_focus(algo_id, version)
                _set_ui_feedback(
                    tx("Created and imported: {algorithm_id} v{version}", "已创建并导入：{algorithm_id} v{version}", algorithm_id=algo_id, version=version),
                    scroll_target="imported-algorithms-anchor",
                )
                st.rerun()
            except FileExistsError as e:
                st.error(str(e))
            except Exception as e:
                if ws_dir is not None and ws_dir.exists():
                    shutil.rmtree(ws_dir, ignore_errors=True)
                st.error(tx("Create failed: {error}", "创建失败：{error}", error=e))


@st.dialog(tx("Edit Manifest", "编辑 Manifest"))
def show_edit_manifest_dialog(algo_id, version, manifest_path):
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(tx("Failed to load manifest: {error}", "加载 manifest 失败：{error}", error=e))
        return

    files_section = manifest_data.get("files")
    if not isinstance(files_section, dict):
        st.error(tx("Invalid manifest: missing required files section", "无效 manifest：缺少 files 段"))
        return

    required_file_keys = ("paramsSchemaPath", "outputContractPath")
    if any(not isinstance(files_section.get(k), str) or not files_section.get(k) for k in required_file_keys):
        st.error(
            tx(
                "Invalid manifest: files.paramsSchemaPath/files.outputContractPath are required",
                "无效 manifest：files.paramsSchemaPath/files.outputContractPath 为必填项",
            )
        )
        return

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
        save_submit = st.form_submit_button(tx("Save Changes", "保存修改"), type="primary")

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
            manifest_data["files"] = {
                "paramsSchemaPath": files_section["paramsSchemaPath"],
                "outputContractPath": files_section["outputContractPath"],
            }
            if isinstance(files_section.get("drsPath"), str) and files_section.get("drsPath"):
                manifest_data["files"]["drsPath"] = files_section["drsPath"]
            if isinstance(files_section.get("dsPath"), str) and files_section.get("dsPath"):
                manifest_data["files"]["dsPath"] = files_section["dsPath"]
            manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
            st.success(tx("Manifest saved successfully", "Manifest 保存成功"))
            st.rerun()
        except Exception as e:
            st.error(tx("Save failed: {error}", "保存失败：{error}", error=e))


# ==========================================
# 1. Snapshot Management
# ==========================================
st.subheader(tx("Import Snapshot", "导入快照"))

with st.container(border=True):
    # 1. File Uploader
    uploaded_snapshot = st.file_uploader(
        tx("Upload Snapshot (zip file)", "上传快照（zip 文件）"),
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
            if st.button(tx("Import Snapshot", "导入快照"), type="primary", key="import_snapshot_btn", width="stretch"):
                with st.spinner(tx("Importing snapshot...", "正在导入快照...")):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                            tmp_file.write(uploaded_snapshot.getvalue())
                            tmp_path = Path(tmp_file.name)

                        snapshot_id = snapshot_lib.import_snapshot(tmp_path)
                        _set_snapshot_focus(snapshot_id)
                        _set_ui_feedback(
                            tx("Imported snapshot: {snapshot_id}", "已导入快照：{snapshot_id}", snapshot_id=snapshot_id),
                            scroll_target="imported-snapshots-anchor",
                        )
                        tmp_path.unlink(missing_ok=True)
                        st.rerun()
                    except SnapshotError as e:
                        st.error(tx("Import failed: {error}", "导入失败：{error}", error=e))
                    except Exception as e:
                        st.error(tx("Error: {error}", "错误：{error}", error=e))

st.divider()

pending_flash = st.session_state.pop(FLASH_MESSAGE_KEY, None)
if pending_flash:
    st.success(pending_flash)
    st.toast(pending_flash, icon="✅")

st.markdown('<div id="imported-snapshots-anchor"></div>', unsafe_allow_html=True)
st.subheader(tx("Imported Snapshots", "已导入快照"))

snapshots = snapshot_lib.list_snapshots()
focus_snapshot_id = st.session_state.pop(FOCUS_SNAPSHOT_KEY, None)

if not snapshots:
    st.info(tx("No snapshots imported yet", "还没有已导入的快照。"))
else:
    if focus_snapshot_id:
        snapshots = sorted(
            snapshots,
            key=lambda s: (s.snapshot_id != focus_snapshot_id, s.snapshot_id),
        )

    for snap in snapshots:
        is_focus_snapshot = snap.snapshot_id == focus_snapshot_id
        with st.expander(f"📦 {snap.snapshot_id}", expanded=is_focus_snapshot):
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 5, 1])
                
                with c1:
                    st.caption(tx("Bundle ID", "包 ID"))
                    st.code(snap.bundle_id, language="text")
                    st.caption(tx("Imported At", "导入时间"))
                    st.markdown(f"**{snap.imported_at}**")

                with c2:
                    st.caption(tx("DRS (Data Requirement Specification)", "DRS（数据需求规范）"))
                    # 获取完整的 Snapshot 对象以读取 DRS
                    try:
                        full_snap = snapshot_lib.get_snapshot(snap.snapshot_id)
                        drs_data = full_snap.drs.model_dump(exclude_none=True)
                        st.code(json.dumps(drs_data, indent=2, ensure_ascii=False), language="json")
                    except Exception as e:
                        st.error(tx("Failed to load DRS: {error}", "加载 DRS 失败：{error}", error=e))

                with c3:
                    st.write("") # Spacer
                    if st.button(tx("Delete", "删除"), key=f"del_{snap.snapshot_id}", type="secondary"):
                        try:
                            snapshot_lib.delete_snapshot(snap.snapshot_id)
                            st.success(tx("Deleted", "已删除"))
                            st.rerun()
                        except Exception as e:
                            st.error(tx("Delete failed: {error}", "删除失败：{error}", error=e))

st.divider()

# ==========================================
# 2. Algorithm Management
# ==========================================
st.subheader(tx("Algorithm Workspace", "算法工作区"))

# Action Bar
col_create, col_spacer = st.columns([1, 4])
with col_create:
    # Updated API
    if st.button(tx("✨ Create New Algorithm", "✨ 创建新算法"), key="create_algo_btn", width="stretch"):
        show_create_algo_dialog()

# Upload Section (Expanded by Default)
with st.expander(tx("📤 Import Existing Algorithm", "📤 导入现有算法"), expanded=True):
    uploaded_algorithm = st.file_uploader(
        tx("Upload Algorithm (zip or .py)", "上传算法（zip 或 .py）"),
        type=["zip", "py"],
        key="algorithm_uploader",
    )

    if uploaded_algorithm is not None:
        if uploaded_algorithm.name.endswith(".zip"):
             if st.button(tx("Import Algorithm Zip", "导入算法 Zip"), type="primary", key="import_algo_btn_zip"):
                with st.spinner(tx("Importing algorithm from zip...", "正在从 zip 导入算法...")):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                            tmp_file.write(uploaded_algorithm.getvalue())
                            tmp_path = Path(tmp_file.name)
                        algo_id, version = algorithm_lib.import_algorithm(tmp_path)
                        _set_algorithm_focus(algo_id, version)
                        _set_ui_feedback(
                            tx("Imported algorithm: {algorithm_id} v{version}", "已导入算法：{algorithm_id} v{version}", algorithm_id=algo_id, version=version),
                            scroll_target="imported-algorithms-anchor",
                        )
                        tmp_path.unlink(missing_ok=True)
                        st.rerun()
                    except Exception as e:
                        st.error(tx("Import failed: {error}", "导入失败：{error}", error=e))
        elif uploaded_algorithm.name.endswith(".py"):
             if st.button(tx("Import Single Python File", "导入单个 Python 文件"), type="primary", key="import_algo_btn_py"):
                with st.spinner(tx("Importing algorithm from .py...", "正在从 .py 导入算法...")):
                    try:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            tmp_dir_path = Path(tmp_dir)

                            # Copy uploaded main.py
                            algo_path = tmp_dir_path / "main.py"
                            algo_path.write_bytes(uploaded_algorithm.getvalue())

                            # Create dist/ with template files
                            dist_dir = tmp_dir_path / "dist"
                            dist_dir.mkdir(parents=True, exist_ok=True)
                            (dist_dir / "ds.json").write_text(json.dumps({"datasets": []}, indent=2))
                            (dist_dir / "params.schema.json").write_text(
                                json.dumps({"type": "object", "title": "Parameters", "properties": {}}, indent=2)
                            )
                            (dist_dir / "output_contract.json").write_text(
                                json.dumps({"datasets": [], "invariants": [], "relations": []}, indent=2)
                            )

                            # Create schema/ with base utilities
                            ensure_schema_base(tmp_dir_path / "schema")

                            # Create manifest
                            algo_id = uploaded_algorithm.name.removesuffix(".py")
                            manifest = {
                                "manifestVersion": "1",
                                "algorithmId": algo_id,
                                "codeVersion": "local",
                                "contractVersion": "1.0.0",
                                "name": algo_id,
                                "summary": tx("Imported from single python file", "从单个 Python 文件导入"),
                                "authors": [{"name": "unknown"}],
                                "files": {
                                    "paramsSchemaPath": "dist/params.schema.json",
                                    "dsPath": "dist/ds.json",
                                    "outputContractPath": "dist/output_contract.json",
                                },
                                "requires": {"sdk": SDK_VERSION},
                            }
                            (tmp_dir_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

                            algo_id, version = algorithm_lib.import_algorithm(tmp_dir_path)
                            _set_algorithm_focus(algo_id, version)
                            _set_ui_feedback(
                                tx("Imported algorithm: {algorithm_id} v{version}", "已导入算法：{algorithm_id} v{version}", algorithm_id=algo_id, version=version),
                                scroll_target="imported-algorithms-anchor",
                            )
                            st.rerun()
                    except Exception as e:
                        st.error(tx("Import failed: {error}", "导入失败：{error}", error=e))

st.divider()

st.markdown('<div id="imported-algorithms-anchor"></div>', unsafe_allow_html=True)
st.subheader(tx("Imported Algorithms", "已导入算法"))

algorithms = algorithm_lib.list_algorithms()
focus_algo_key = st.session_state.pop(FOCUS_ALGO_KEY, None)

if not algorithms:
    st.info(tx("No algorithms imported yet", "还没有已导入的算法。"))
else:
    if focus_algo_key:
        algorithms = sorted(
            algorithms,
            key=lambda a: (f"{a.algorithm_id}:{a.version}" != focus_algo_key, a.algorithm_id, a.version),
        )

    for algo in algorithms:
        algo_key = f"{algo.algorithm_id}:{algo.version}"
        is_focus_algo = algo_key == focus_algo_key
        with st.expander(f"🧩 {algo.algorithm_id} (v{algo.version})", expanded=is_focus_algo):
            with st.container(border=True):
                # Header info
                head_c1, head_c2, head_c3 = st.columns([3, 2, 2])
                with head_c1:
                    st.caption(tx("Name", "名称"))
                    st.markdown(f"**{algo.name or tx('N/A', '无')}**")
                with head_c2:
                    st.caption(tx("Contract", "契约"))
                    st.code(algo.contract_version or tx("N/A", "无"), language="text")
                with head_c3:
                    st.caption(tx("Imported", "导入时间"))
                    st.text(algo.imported_at)

                st.markdown("---")
                
                # Content info
                st.caption(tx("Summary", "摘要"))
                if getattr(algo, 'summary', ''):
                    st.info(algo.summary)
                else:
                    st.text(tx("No summary provided.", "没有摘要。"))
                
                notes = getattr(algo, "notes", None)
                if notes:
                    st.caption(tx("Notes", "备注"))
                    st.write(notes)

                st.markdown("---")

                # Actions
                act_c1, act_c2 = st.columns([1, 5])
                
                with act_c1:
                    manifest_path = algorithm_lib.get_algorithm(algo.algorithm_id, algo.version).directory / "manifest.json"
                    if manifest_path.exists():
                        if st.button(tx("📝 Edit Manifest", "📝 编辑 Manifest"), key=f"edit_manifest_btn_{algo.algorithm_id}_{algo.version}"):
                            show_edit_manifest_dialog(algo.algorithm_id, algo.version, manifest_path)

                with act_c2:
                    if st.button(tx("🗑️ Delete", "🗑️ 删除"), key=f"del_algo_{algo.algorithm_id}_{algo.version}"):
                        try:
                            algorithm_lib.delete_algorithm(algo.algorithm_id, algo.version)
                            ws_dir = WORKSPACE_ROOT / algo.algorithm_id / algo.version
                            if ws_dir.exists():
                                shutil.rmtree(ws_dir, ignore_errors=True)
                            st.success(tx("Deleted", "已删除"))
                            st.rerun()
                        except Exception as e:
                            st.error(tx("Delete failed: {error}", "删除失败：{error}", error=e))

pending_scroll_target = st.session_state.pop(SCROLL_TARGET_KEY, None)
if pending_scroll_target:
    _emit_scroll_to_element(pending_scroll_target)
