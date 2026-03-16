"""OutputSpec editor page for editing algorithm output_contract."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.devkit.validate import validate_output_contract
from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.workbench.i18n import page_title, tx
from fraclab_sdk.workbench.utils import run_workspace_script

st.set_page_config(
    page_title=page_title("output_edit"),
    page_icon="📤",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui_styles.apply_global_styles("output_edit")
ui_styles.render_page_header(tx("OutputSpec Editor", "输出结果定义"))

# --- Page-Specific CSS: Editor Styling ---
st.markdown("""
<style>
    textarea {
        font-family: "Source Code Pro", monospace !important;
        font-size: 14px !important;
        background-color: #fcfcfc !important;
    }
</style>
""", unsafe_allow_html=True)
def write_dist_from_contract(ws_dir: Path, algo_id: str, version: str) -> None:
    """Import OUTPUT_CONTRACT and dump to dist/output_contract.json."""
    script = '''
import json
from schema.output_contract import OUTPUT_CONTRACT
if hasattr(OUTPUT_CONTRACT, "model_dump"):
    print(json.dumps(OUTPUT_CONTRACT.model_dump(mode="json", by_alias=True)))
else:
    print(json.dumps(OUTPUT_CONTRACT.dict()))
'''
    result = run_workspace_script(ws_dir, script)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "Failed to load OUTPUT_CONTRACT")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to parse OUTPUT_CONTRACT output") from exc

    dist_dir = ws_dir / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "output_contract.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


DOC_SUMMARY = """
**OutputSpec Cheatsheet:**
- **Datasets**: List of `OutputDatasetContract` inside `OutputContract`.
- **Props**: `key` (unique), `kind` (frame/object/blob/scalar), `owner`, `cardinality` (one/many), `required`.
- **Schema**: Must match `kind` (e.g., `ScalarSchema` for kind='scalar').
- **Dimensions**: List of string keys used in artifact dims.
"""
DOC_SUMMARY = tx(
    DOC_SUMMARY,
    """
**OutputSpec 速查：**
- **Datasets**：`OutputContract` 内部的 `OutputDatasetContract` 列表。
- **属性**：`key`（唯一）、`kind`（frame/object/blob/scalar）、`owner`、`cardinality`（one/many）、`required`。
- **Schema**：必须和 `kind` 匹配，例如 kind=`scalar` 时使用 `ScalarSchema`。
- **Dimensions**：artifact dims 使用的字符串维度键列表。
""",
)

config = SDKConfig()
algo_lib = AlgorithmLibrary(config)
algos = algo_lib.list_algorithms()

if not algos:
    st.info(tx("No algorithms imported.", "还没有已导入的算法。"))
    st.stop()

# --- 1. Selection ---
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

if not selected_key:
    st.stop()

selected = algo_options[selected_key]
handle = algo_lib.get_algorithm(selected.algorithm_id, selected.version)
workspace_dir = handle.directory

st.caption(tx("Algorithm dir: `{path}`", "算法目录：`{path}`", path=workspace_dir))
schema_dir = workspace_dir / "schema"
schema_dir.mkdir(parents=True, exist_ok=True)
output_spec_path = schema_dir / "output_contract.py"

DEFAULT_OUTPUTSPEC = '''from __future__ import annotations
from fraclab_sdk.models.output_spec import BlobSchema, OutputContract, OutputDatasetContract, ScalarSchema

OUTPUT_CONTRACT = OutputContract(
    datasets=[
        # OutputDatasetContract(key="metrics", kind="scalar", cardinality="many", ...)
    ]
)
'''

if not output_spec_path.exists():
    output_spec_path.write_text(DEFAULT_OUTPUTSPEC, encoding="utf-8")

# --- 2. Documentation ---
with st.expander(tx("📚 Documentation & Tips", "📚 文档与提示"), expanded=True):
    st.markdown(DOC_SUMMARY)

# --- 3. Editor ---
content = output_spec_path.read_text(encoding="utf-8")
edited = st.text_area("output_contract.py", value=content, height=600, label_visibility="collapsed")

# --- 4. Actions ---
col_save, col_valid, col_spacer = st.columns([1, 1, 4])

with col_save:
    if st.button(tx("💾 Save & Generate", "💾 保存并生成"), type="primary", width="stretch"):
        try:
            output_spec_path.write_text(edited, encoding="utf-8")
            write_dist_from_contract(workspace_dir, selected.algorithm_id, selected.version)
            st.toast(tx("Output spec saved and JSON generated!", "输出规范已保存并生成 JSON。"), icon="✅")
        except Exception as e:
            st.error(tx("Save failed: {error}", "保存失败：{error}", error=e))

with col_valid:
    if st.button(tx("🔍 Validate", "🔍 校验"), type="secondary", width="stretch"):
        try:
            # Auto-save before validate
            output_spec_path.write_text(edited, encoding="utf-8")
            write_dist_from_contract(workspace_dir, selected.algorithm_id, selected.version)

            result = validate_output_contract(workspace_dir)
            if result.valid:
                if result.warnings:
                    st.warning(
                        tx(
                            "Validation Passed with {count} warning(s)",
                            "校验通过，但有 {count} 条警告",
                            count=len(result.warnings),
                        ),
                        icon="⚠️",
                    )
                else:
                    st.success(tx("Validation Passed!", "校验通过！"), icon="✅")
            else:
                st.error(
                    tx(
                        "Validation Failed ({count} error(s))",
                        "校验失败（{count} 个错误）",
                        count=len(result.errors),
                    ),
                    icon="🚫",
                )

            # Show all issues (errors and warnings)
            for issue in result.issues:
                icon = "🔴" if issue.severity.value == "error" else "🟡"
                path_str = tx(" at `{path}`", " 于 `{path}`", path=issue.path) if issue.path else ""
                details_str = ""
                if issue.details and "suggested" in issue.details:
                    details_str = tx(
                        " → Suggested: `{value}`",
                        " → 建议：`{value}`",
                        value=issue.details["suggested"],
                    )
                st.markdown(f"{icon} **{issue.code}**{path_str}: {issue.message}{details_str}")
        except Exception as e:
            st.error(tx("Validation error: {error}", "校验异常：{error}", error=e))
