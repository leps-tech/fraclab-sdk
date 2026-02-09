"""Run page: edit params for existing runs and execute."""

import json
from pathlib import Path
from typing import Any, Dict
from decimal import Decimal, InvalidOperation

import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.run import RunManager, RunStatus
from fraclab_sdk.workbench import ui_styles

st.set_page_config(page_title="Run", page_icon="▶️", layout="wide", initial_sidebar_state="expanded")
st.title("Run")

ui_styles.apply_global_styles()

# Guidance: if no params UI shows up, validate InputSpec via the editor page.
st.info(
    "看不到参数输入框？请在 Schema Editor 页面点击 Validate 检查 "
    "`schema/inputspec.py`，确保生成的 schema 可用。",
    icon="ℹ️",
)
# --- Page-Specific CSS ---
st.markdown("""
<style>
    /* Compact form labels */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stCheckbox"] label {
        margin-bottom: 0px !important;
        font-size: 0.85rem !important;
        color: #666 !important;
    }

    /* Divider spacing */
    hr { margin-top: 1rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

config = SDKConfig()
run_mgr = RunManager(config)
algo_lib = AlgorithmLibrary(config)


# --- Intelligent Layout Engine ---

def _is_compact_field(schema: dict) -> bool:
    """Determine if a field is small enough to fit in a grid column."""
    ftype = schema.get("type")
    # Numbers, Booleans, and short Strings (without enums/long defaults) are compact
    if ftype in ["integer", "number", "boolean"]:
        return True
    if ftype == "string" and len(str(schema.get("default", ""))) < 50:
        return True
    return False


def _resolve_number_step_and_format(schema: dict) -> tuple[float, str]:
    """Resolve number widget step/format from schema.step.

    Rules:
    - If schema.step exists and is valid, use it and derive decimal places from it.
    - If schema.step is missing/invalid, default to integer-style display.
    """
    raw_step = schema.get("step")
    if raw_step is None:
        return 1.0, "%.0f"

    try:
        step_decimal = Decimal(str(raw_step))
    except (InvalidOperation, ValueError, TypeError):
        return 1.0, "%.0f"

    if step_decimal <= 0:
        return 1.0, "%.0f"

    # Decimal exponent: -3 means 3 decimal places.
    decimals = max(0, -step_decimal.normalize().as_tuple().exponent)
    return float(step_decimal), f"%.{decimals}f"

def render_field_widget(key: str, schema: dict, value: Any, path: str) -> Any:
    """Render a single widget based on schema type."""
    ftype = schema.get("type")
    title = schema.get("title") or key
    # Simplify label: if title is camelCase, maybe split it? For now use title directly.
    # description = schema.get("description") # Tooltip is enough, don't clutter UI text
    
    default_val = schema.get("default")
    help_text = schema.get("description")

    if ftype == "string":
        val = value if value is not None else (default_val or "")
        return st.text_input(title, value=val, help=help_text, key=path)
    
    if ftype == "number":
        val = value if value is not None else default_val
        step, number_format = _resolve_number_step_and_format(schema)
        return st.number_input(
            title,
            value=float(val or 0.0),
            step=step,
            format=number_format,
            help=help_text,
            key=path,
        )
    
    if ftype == "integer":
        val = value if value is not None else default_val
        return int(st.number_input(title, value=int(val or 0), step=1, help=help_text, key=path))
    
    if ftype == "boolean":
        val = value if value is not None else default_val
        # Toggle looks better than checkbox in grid
        return st.toggle(title, value=bool(val), help=help_text, key=path)
    
    if ftype == "array":
        # Arrays are complex, stick to full width expansion
        return _render_json_editor(title, value, default_val, help_text, path)
            
    if ftype == "object":
        # Nested object -> Recursive layout
        with st.container(border=True):
            st.markdown(f"**{title}**")
            props = schema.get("properties", {})
            obj = value if isinstance(value, dict) else (default_val if isinstance(default_val, dict) else {})
            
            # Recursive call to grid layout
            return render_schema_grid(props, obj, path)

    # Fallback
    return _render_json_editor(title, value, default_val, help_text, path)

def _render_json_editor(title, value, default, help_text, path):
    """Helper for raw JSON fields."""
    st.markdown(f"<small>{title}</small>", unsafe_allow_html=True)
    current = value if value is not None else (default if default is not None else [])
    text = st.text_area(
        title,
        value=json.dumps(current, indent=2, ensure_ascii=False),
        help=f"{help_text} (Edit as JSON)",
        key=path,
        label_visibility="collapsed",
        height=100
    )
    try:
        return json.loads(text) if text.strip() else current
    except Exception:
        return current

def render_schema_grid(properties: Dict[str, dict], current_values: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Renders fields in a smart grid layout:
    - Compact fields (numbers, bools) get packed into columns (up to 4).
    - Wide fields (objects, arrays) break the line and take full width.
    """
    result = {}
    
    # 1. Separate fields into groups to maintain partial order while grid-packing
    # Strategy: Iterate and buffer compact fields. Flush buffer when a wide field hits.
    
    compact_buffer = [] # List of (key, schema)
    
    def flush_buffer():
        if not compact_buffer:
            return
        
        # Calculate optimal columns (max 4, min 2)
        n_items = len(compact_buffer)
        n_cols = 4 if n_items >= 4 else (n_items if n_items > 0 else 1)
        
        # Split into rows if > 4 items? Simple logic: Just wrap
        # Actually st.columns handles wrapping poorly, better to batch by 4
        
        for i in range(0, n_items, 4):
            batch = compact_buffer[i : i+4]
            cols = st.columns(len(batch))
            for col, (b_key, b_schema) in zip(cols, batch):
                with col:
                    val = current_values.get(b_key)
                    result[b_key] = render_field_widget(b_key, b_schema, val, f"{prefix}.{b_key}")
        
        compact_buffer.clear()

    for key, prop_schema in properties.items():
        if _is_compact_field(prop_schema):
            compact_buffer.append((key, prop_schema))
        else:
            # Wide field encountered: flush buffer first
            flush_buffer()
            # Render wide field
            val = current_values.get(key)
            result[key] = render_field_widget(key, prop_schema, val, f"{prefix}.{key}")
    
    # Final flush
    flush_buffer()
    
    # Preserve extra keys in current_values that aren't in schema
    for k, v in current_values.items():
        if k not in result:
            result[k] = v
            
    return result


def load_params(run_dir: Path) -> dict:
    path = run_dir / "input" / "params.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ==========================================
# Main Logic
# ==========================================

runs = run_mgr.list_runs()

if not runs:
    st.info("No runs available. Create a run from the Selection page.")
    st.stop()

pending_runs = [r for r in runs if r.status == RunStatus.PENDING]
other_runs = [r for r in runs if r.status != RunStatus.PENDING]

# ------------------------------------------
# 1. Pending Runs (Editor Workspace)
# ------------------------------------------
st.subheader("Pending Runs")

if not pending_runs:
    st.info("No pending runs waiting for execution.")
else:
    # Use tabs for context switching
    tabs = st.tabs([f"⚙️ {run.run_id}" for run in pending_runs])
    
    for tab, run in zip(tabs, pending_runs):
        with tab:
            run_dir = run_mgr.get_run_dir(run.run_id)
            algo_handle = algo_lib.get_algorithm(run.algorithm_id, run.algorithm_version)
            schema = algo_handle.params_schema
            current_params = load_params(run_dir)
            
            # --- Layout: Top Info Bar ---
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
                with c1: st.caption(f"**Snapshot:** `{run.snapshot_id}`")
                with c2: st.caption(f"**Algo:** `{run.algorithm_id}` v{run.algorithm_version}")
                with c3: st.caption(f"**Created:** {run.created_at}")
                with c4: 
                    # Timeout setting tucked away here
                    timeout = st.number_input("Timeout (s)", value=300, step=10, key=f"to_{run.run_id}", label_visibility="collapsed")

            # --- Layout: Parameters Grid ---
            st.markdown("##### Parameters")
            with st.container(border=True):
                if schema.get("type") == "object":
                    props = schema.get("properties", {})
                    # CALL THE GRID ENGINE
                    new_params = render_schema_grid(props, current_params, prefix=f"run_{run.run_id}")
                else:
                    st.info("Schema is not an object, editing raw JSON.")
                    new_params = _render_json_editor("Raw Params", current_params, {}, "", f"run_raw_{run.run_id}")

            st.divider()
            
            # --- Layout: Action Footer ---
            # Right-aligned actions
            _, col_btns = st.columns([3, 4])
            with col_btns:
                b1, b2, b3 = st.columns([1, 1, 1.5], gap="small")
                
                with b1:
                    if st.button("🚫 Cancel", key=f"cancel_{run.run_id}", width="stretch"):
                        try:
                            run_mgr.delete_run(run.run_id)
                            st.success("Cancelled")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                with b2:
                    if st.button("💾 Save", key=f"save_{run.run_id}", width="stretch"):
                        try:
                            (run_dir / "input").mkdir(parents=True, exist_ok=True)
                            (run_dir / "input" / "params.json").write_text(
                                json.dumps(new_params, indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                            st.toast("Parameters saved successfully!", icon="💾")
                        except Exception as e:
                            st.error(f"Save failed: {e}")

                with b3:
                    if st.button("▶️ Run Algorithm", key=f"exec_{run.run_id}", type="primary", width="stretch"):
                        try:
                            # Auto-save before run
                            (run_dir / "input").mkdir(parents=True, exist_ok=True)
                            (run_dir / "input" / "params.json").write_text(
                                json.dumps(new_params, indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )

                            with st.spinner("Initializing execution..."):
                                result = run_mgr.execute(run.run_id, timeout_s=int(timeout))

                            if result.error:
                                st.error(f"Run Finished: {result.status.value}\n{result.error}")
                            else:
                                # Navigate to Results page with executed run
                                st.session_state.executed_run_id = run.run_id
                                st.switch_page("pages/5_Results.py")
                        except Exception as e:
                            st.error(f"Execution Exception: {e}")


# ------------------------------------------
# 2. History (Other Runs)
# ------------------------------------------
st.subheader("Run History")

if not other_runs:
    st.caption("No historical runs.")

other_runs_reversed = other_runs[::-1]

for run in other_runs_reversed:
    status_config = {
        RunStatus.PENDING:   ("⏳", "Pending", "gray"),
        RunStatus.RUNNING:   ("🔄", "Running", "blue"),
        RunStatus.SUCCEEDED: ("✅", "Succeeded", "green"),
        RunStatus.FAILED:    ("❌", "Failed", "red"),
        RunStatus.TIMEOUT:   ("⏱️", "Timeout", "orange"),
    }
    icon, label, color = status_config.get(run.status, ("❓", "Unknown", "gray"))
    
    with st.expander(f"{icon} {run.run_id}", expanded=False):
        with st.container(border=True):
            # Info
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1: 
                st.caption("Context")
                st.markdown(f"**{run.algorithm_id}** v{run.algorithm_version}")
                st.text(f"Snap: {run.snapshot_id}")
            with c2:
                st.caption("Timing")
                st.text(f"Start: {run.started_at or '--'}")
                st.text(f"End:   {run.completed_at or '--'}")
            with c3:
                st.caption("Status")
                st.markdown(f":{color}[**{label}**]")
                if run.error:
                    st.error(run.error)
            
            # Params Read-only
            st.divider()
            st.caption("Run Parameters")
            run_dir = run_mgr.get_run_dir(run.run_id)
            params_view = load_params(run_dir)
            if params_view:
                st.code(json.dumps(params_view, indent=2, ensure_ascii=False), language="json")
