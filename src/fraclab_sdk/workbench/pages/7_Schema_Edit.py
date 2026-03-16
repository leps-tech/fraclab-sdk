"""Schema editor page for editing algorithm InputSpec."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.devkit.validate import validate_inputspec
from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.workbench.i18n import page_title, tx
from fraclab_sdk.workbench.utils import run_workspace_script

st.set_page_config(
    page_title=page_title("schema_edit"),
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui_styles.apply_global_styles("schema_edit")
ui_styles.render_page_header(tx("Schema Editor (InputSpec)", "输入参数编辑"))

# --- Page-Specific CSS: Editor Styling ---
st.markdown("""
<style>
    textarea {
        font-family: "Source Code Pro", monospace !important;
        font-size: 14px !important;
        background-color: #fcfcfc !important;
    }

    /* Validation result: full-width and copy-enabled on this page */
    [data-testid="stCode"] {
        user-select: text !important;
        -webkit-user-select: text !important;
    }
    [data-testid="stCode"] button {
        display: inline-flex !important;
    }
</style>
""", unsafe_allow_html=True)


def write_params_schema(ws_dir: Path) -> None:
    """Generate dist/params.schema.json via subprocess."""
    script = '''
import json
from schema.inputspec import INPUT_SPEC

if hasattr(INPUT_SPEC, "model_json_schema"):
    schema = INPUT_SPEC.model_json_schema(by_alias=True)
elif hasattr(INPUT_SPEC, "schema"):
    schema = INPUT_SPEC.schema()
else:
    raise SystemExit("INPUT_SPEC missing schema generator")
print(json.dumps(schema))
'''
    result = run_workspace_script(ws_dir, script)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "Failed to generate params.schema.json")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to parse generated params schema") from exc

    dist_dir = ws_dir / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "params.schema.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


DOC_SUMMARY = tx(
    """
**InputSpec Cheatsheet:**
- **Types**: `str`, `int`, `float`, `bool`, `datetime`, `Optional[T]`, `Literal["A", "B"]`.
- **Field**: `Field(..., title="Title", description="Desc")`.
- **Base Model**: `class INPUT_SPEC(CamelModel): ...` to emit camelCase JSON schema.
- **UI Metadata**: `json_schema_extra=schemaExtra(group="Basic", order=1, uiType="range")`.
- **Visibility**: `showWhen=showWhenCondition("mode", "equals", "advanced")`.
- **Validation**: `@field_validator("field")` or `@model_validator(mode="after")`.
""",
    """
**InputSpec 速查：**
- **类型**：`str`、`int`、`float`、`bool`、`datetime`、`Optional[T]`、`Literal["A", "B"]`。
- **字段**：`Field(..., title="标题", description="说明")`。
- **基础模型**：`class INPUT_SPEC(CamelModel): ...`，用于输出 camelCase JSON Schema。
- **UI 元数据**：`json_schema_extra=schemaExtra(group="Basic", order=1, uiType="range")`。
- **显示控制**：`showWhen=showWhenCondition("mode", "equals", "advanced")`。
- **校验**：`@field_validator("field")` 或 `@model_validator(mode="after")`。
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
algo_dir = handle.directory

st.caption(tx("Algorithm dir: `{path}`", "算法目录：`{path}`", path=algo_dir))
schema_dir = algo_dir / "schema"
schema_dir.mkdir(parents=True, exist_ok=True)
inputspec_path = schema_dir / "inputspec.py"

DEFAULT_INPUTSPEC = '''from __future__ import annotations
from pydantic import Field
from .base import CamelModel, schemaExtra, showWhenCondition, showWhenAnd, showWhenOr

class INPUT_SPEC(CamelModel):
    """Algorithm parameters."""
    # datasetKey: str = Field(..., title="Dataset Key")
'''

if not inputspec_path.exists():
    inputspec_path.write_text(DEFAULT_INPUTSPEC, encoding="utf-8")

# --- 2. Documentation ---
with st.expander(tx("📚 Documentation & Tips", "📚 文档与提示"), expanded=True):
    st.markdown(DOC_SUMMARY)

# --- 3. Editor ---
content = inputspec_path.read_text(encoding="utf-8")
edited = st.text_area("inputspec.py", value=content, height=600, label_visibility="collapsed")

# --- 4. Actions ---
col_save, col_valid, col_spacer = st.columns([1, 1, 4])

with col_save:
    if st.button(tx("💾 Save & Generate", "💾 保存并生成"), type="primary", width="stretch"):
        try:
            inputspec_path.write_text(edited, encoding="utf-8")
            write_params_schema(algo_dir)
            st.toast(tx("Schema saved and JSON generated!", "Schema 已保存并生成 JSON。"), icon="✅")
        except Exception as e:
            st.error(tx("Save failed: {error}", "保存失败：{error}", error=e))

with col_valid:
    if st.button(tx("🔍 Validate", "🔍 校验"), type="secondary", width="stretch"):
        try:
            # Auto-save before validate
            inputspec_path.write_text(edited, encoding="utf-8")
            write_params_schema(algo_dir)

            result = validate_inputspec(algo_dir)
            status = "pass"
            if not result.valid:
                status = "fail"
            elif result.warnings:
                status = "warn"

            lines: list[str] = []
            for issue in result.issues:
                icon = "ERROR" if issue.severity.value == "error" else "WARN"
                path_str = tx(" at {path}", " 于 {path}", path=issue.path) if issue.path else ""
                details_str = ""
                if issue.details and "suggested" in issue.details:
                    details_str = tx(
                        " | suggested={value}",
                        " | 建议值={value}",
                        value=issue.details["suggested"],
                    )
                lines.append(f"[{icon}] {issue.code}{path_str}: {issue.message}{details_str}")

            st.session_state["schema_edit_validate_status"] = status
            st.session_state["schema_edit_validate_result_text"] = "\n".join(lines) if lines else tx("No issues.", "没有问题。")
            st.session_state["schema_edit_validate_counts"] = {
                "errors": len(result.errors),
                "warnings": len(result.warnings),
            }
        except Exception as e:
            st.session_state["schema_edit_validate_status"] = "error"
            st.session_state["schema_edit_validate_result_text"] = tx(
                "Validation error: {error}",
                "校验异常：{error}",
                error=e,
            )
            st.session_state["schema_edit_validate_counts"] = {"errors": 1, "warnings": 0}

# --- 5. Validation Result (full width) ---
if "schema_edit_validate_status" in st.session_state:
    status = st.session_state["schema_edit_validate_status"]
    counts = st.session_state.get("schema_edit_validate_counts", {"errors": 0, "warnings": 0})
    text = st.session_state.get("schema_edit_validate_result_text", tx("No issues.", "没有问题。"))

    if status == "pass":
        st.success(tx("Validation Passed!", "校验通过！"), icon="✅")
    elif status == "warn":
        st.warning(
            tx(
                "Validation Passed with {count} warning(s)",
                "校验通过，但有 {count} 条警告",
                count=counts["warnings"],
            ),
            icon="⚠️",
        )
    elif status == "fail":
        st.error(
            tx(
                "Validation Failed ({count} error(s))",
                "校验失败（{count} 个错误）",
                count=counts["errors"],
            ),
            icon="🚫",
        )
    else:
        st.error(tx("Validation encountered an exception", "校验过程中出现异常"), icon="🚫")

    st.code(text, language="text")
