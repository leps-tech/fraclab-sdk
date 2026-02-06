"""Schema editor page for editing algorithm InputSpec."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.devkit.validate import validate_inputspec
from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.workbench.utils import run_workspace_script

st.set_page_config(page_title="Schema Editor", page_icon="🧩", layout="wide", initial_sidebar_state="expanded")
st.title("Schema Editor (InputSpec)")

ui_styles.apply_global_styles()

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


def write_params_schema(ws_dir: Path) -> None:
    """Generate dist/params.schema.json via subprocess."""
    script = '''
import json
from schema.inputspec import INPUT_SPEC

if hasattr(INPUT_SPEC, "model_json_schema"):
    schema = INPUT_SPEC.model_json_schema()
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


DOC_SUMMARY = """
**InputSpec Cheatsheet:**
- **Types**: `str`, `int`, `float`, `bool`, `datetime`, `Optional[T]`, `Literal["A", "B"]`.
- **Field**: `Field(..., title="Title", description="Desc")`.
- **UI Metadata**: `json_schema_extra=schema_extra(group="Basic", order=1, ui_type="range")`.
- **Visibility**: `show_when=show_when_condition("mode", "equals", "advanced")`.
- **Validation**: `@field_validator("field")` or `@model_validator(mode="after")`.
"""

config = SDKConfig()
algo_lib = AlgorithmLibrary(config)
algos = algo_lib.list_algorithms()

if not algos:
    st.info("No algorithms imported.")
    st.stop()

# --- 1. Selection ---
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

if not selected_key:
    st.stop()

selected = algo_options[selected_key]
handle = algo_lib.get_algorithm(selected.algorithm_id, selected.version)
algo_dir = handle.directory

st.caption(f"Algorithm dir: `{algo_dir}`")
schema_dir = algo_dir / "schema"
schema_dir.mkdir(parents=True, exist_ok=True)
inputspec_path = schema_dir / "inputspec.py"

DEFAULT_INPUTSPEC = '''from __future__ import annotations
from pydantic import BaseModel, Field
from .base import schema_extra, show_when_condition, show_when_and, show_when_or

class INPUT_SPEC(BaseModel):
    """Algorithm parameters."""
    # datasetKey: str = Field(..., title="Dataset Key")
'''

if not inputspec_path.exists():
    inputspec_path.write_text(DEFAULT_INPUTSPEC, encoding="utf-8")

# --- 2. Documentation ---
with st.expander("📚 Documentation & Tips", expanded=True):
    st.markdown(DOC_SUMMARY)

# --- 3. Editor ---
content = inputspec_path.read_text(encoding="utf-8")
edited = st.text_area("inputspec.py", value=content, height=600, label_visibility="collapsed")

# --- 4. Actions ---
col_save, col_valid, col_spacer = st.columns([1, 1, 4])

with col_save:
    if st.button("💾 Save & Generate", type="primary", width="stretch"):
        try:
            inputspec_path.write_text(edited, encoding="utf-8")
            write_params_schema(algo_dir)
            st.toast("Schema saved and JSON generated!", icon="✅")
        except Exception as e:
            st.error(f"Save failed: {e}")

with col_valid:
    if st.button("🔍 Validate", type="secondary", width="stretch"):
        try:
            # Auto-save before validate
            inputspec_path.write_text(edited, encoding="utf-8")
            write_params_schema(algo_dir)

            result = validate_inputspec(algo_dir)
            if result.valid:
                if result.warnings:
                    st.warning(f"Validation Passed with {len(result.warnings)} warning(s)", icon="⚠️")
                else:
                    st.success("Validation Passed!", icon="✅")
            else:
                st.error(f"Validation Failed ({len(result.errors)} error(s))", icon="🚫")

            # Show all issues (errors and warnings)
            for issue in result.issues:
                icon = "🔴" if issue.severity.value == "error" else "🟡"
                path_str = f" at `{issue.path}`" if issue.path else ""
                details_str = ""
                if issue.details:
                    if "suggested" in issue.details:
                        details_str = f" → Suggested: `{issue.details['suggested']}`"
                st.markdown(f"{icon} **{issue.code}**{path_str}: {issue.message}{details_str}")
        except Exception as e:
            st.error(f"Validation error: {e}")
