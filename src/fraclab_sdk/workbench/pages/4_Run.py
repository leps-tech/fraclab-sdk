"""Run page: edit params for existing runs and execute."""

import json
from contextlib import suppress
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.models import DataSpec
from fraclab_sdk.run import RunManager, RunStatus
from fraclab_sdk.workbench import ui_styles
from fraclab_sdk.workbench.components.time_window_picker import time_window_picker
from fraclab_sdk.workbench.i18n import page_title, run_status_label, tx
from fraclab_sdk.workbench.parquet_preview import build_parquet_preview_from_files
from fraclab_sdk.workbench.schema_form import (
    constant_field_value,
    extract_array_enum_choices,
    extract_enum_choices,
    extract_object_union,
    extract_ui_type,
    field_is_visible,
    is_nullable_schema,
    match_object_union_branch,
    normalize_schema,
    schema_meta,
    sort_schema_properties,
)

st.set_page_config(
    page_title=page_title("run"),
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui_styles.apply_global_styles("run")
ui_styles.render_page_header(tx("Run", "运行管理"))

# Guidance: if no params UI shows up, validate InputSpec via the editor page.
st.info(
    tx(
        "Missing parameter inputs? Use Validate on the Schema Editor page to check "
        "`schema/inputspec.py` and ensure the generated schema is usable.",
        "看不到参数输入框？请在输入参数编辑页面点击 Validate 检查 "
        "`schema/inputspec.py`，确保生成的 schema 可用。",
    ),
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

    /* Re-enable interaction on Run page (global anti-copy CSS disables user-select). */
    .element-container,
    [data-testid="stPlotlyChart"],
    [data-testid="stCustomComponentV1"],
    .js-plotly-plot,
    .plot-container {
        user-select: auto !important;
        -webkit-user-select: auto !important;
        -moz-user-select: auto !important;
        -ms-user-select: auto !important;
        pointer-events: auto !important;
    }
    /* Ensure streamlit-plotly-events iframes receive pointer events */
    [data-testid="stCustomComponentV1"] iframe {
        pointer-events: auto !important;
    }
</style>
""", unsafe_allow_html=True)

config = SDKConfig()
run_mgr = RunManager(config)
algo_lib = AlgorithmLibrary(config)


# --- Intelligent Layout Engine ---

def _is_compact_field(schema: dict[str, Any], root_schema: dict[str, Any]) -> bool:
    """Determine if a field is small enough to fit in a grid column."""
    if constant_field_value(schema, root_schema) is not None:
        return True
    if extract_enum_choices(schema, root_schema) is not None:
        return True
    ftype = normalize_schema(schema, root_schema).get("type") or schema.get("type")
    # Numbers, Booleans, and short Strings (without enums/long defaults) are compact
    if ftype in ["integer", "number", "boolean"]:
        return True
    return ftype == "string" and len(str(schema.get("default", ""))) < 50


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

def _load_run_ds(run_dir: Path) -> DataSpec | None:
    """Load run input ds.json."""
    try:
        ds_path = run_dir / "input" / "ds.json"
        if not ds_path.exists():
            return None
        return DataSpec.model_validate_json(ds_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _owner_stage_id(item_owner: Any, fallback: str) -> str:
    """Extract stage-like id from owner for binding."""
    if isinstance(item_owner, dict):
        for key in ("stageId", "wellId", "platformId"):
            if item_owner.get(key):
                return str(item_owner[key])
    return fallback


def _coerce_ranges(value: Any) -> list[dict[str, Any]]:
    """Coerce list of windows: {min,max} with optional itemKey."""
    out: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return out
    for v in value:
        if not isinstance(v, dict):
            continue
        if "min" not in v or "max" not in v:
            continue
        try:
            w: dict[str, Any] = {"min": float(v["min"]), "max": float(v["max"])}
            item_key = str(v.get("itemKey") or "").strip()
            if item_key:
                w["itemKey"] = item_key
            out.append(w)
        except Exception:
            continue
    return out


def _to_item_window_map(
    windows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, float]]], list[dict[str, float]]]:
    """Split flattened windows into per-item windows and shared windows."""
    by_item: dict[str, list[dict[str, float]]] = {}
    shared: list[dict[str, float]] = []
    for w in windows:
        item_key = str(w.get("itemKey") or "").strip()
        range_only = {"min": float(w["min"]), "max": float(w["max"])}
        if item_key:
            by_item.setdefault(item_key, []).append(range_only)
        else:
            shared.append(range_only)
    return by_item, shared


def _flatten_item_window_map(
    by_item: dict[str, list[dict[str, float]]],
    preferred_item_keys: list[str],
) -> list[dict[str, Any]]:
    """Flatten per-item windows to list[{itemKey,min,max}] in stable item order."""
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item_key in preferred_item_keys:
        seen.add(item_key)
        for w in by_item.get(item_key, []):
            output.append({"itemKey": item_key, "min": w["min"], "max": w["max"]})
    for item_key, windows in by_item.items():
        if item_key in seen:
            continue
        for w in windows:
            output.append({"itemKey": item_key, "min": w["min"], "max": w["max"]})
    return output


def _build_item_label(item: dict[str, Any]) -> str:
    """Build compact item label for selectors."""
    item_key = str(item.get("itemKey", ""))
    stage_id = str(item.get("stageId", ""))
    if stage_id and stage_id != item_key:
        return f"{item_key} · {stage_id}"
    return item_key


def _time_window_mode(schema: dict[str, Any], root_schema: dict[str, Any]) -> str | None:
    """Resolve schema mode for ui_type=time_window."""
    normalized = normalize_schema(schema, root_schema)
    if normalized.get("type") != "array":
        return None
    item_schema = normalize_schema(normalized.get("items", {}), root_schema)
    if item_schema.get("type") != "object":
        return None
    props = item_schema.get("properties")
    if isinstance(props, dict) and "min" in props and "max" in props:
        return "window_list"
    return None


def _extract_window_bounds(schema: dict[str, Any], root_schema: dict[str, Any], mode: str) -> tuple[int | None, int | None]:
    """Extract min/max windows per item from schema."""
    if mode != "window_list":
        return None, None
    normalized = normalize_schema(schema, root_schema)
    min_items = normalized.get("minItems")
    max_items = normalized.get("maxItems")
    return (min_items if isinstance(min_items, int) else None, max_items if isinstance(max_items, int) else None)



def _as_dataset_value_for_component(
    mode: str,
    value: Any,
    bind_dataset_key: str | None,
    default_dataset_key: str | None,
    datasets_config: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Convert field value to dataset-list shape expected by the component."""
    if mode != "window_list":
        return None
    windows = _coerce_ranges(value)
    if not windows:
        return None
    dataset_key = bind_dataset_key or default_dataset_key
    if not dataset_key:
        return None
    ds_cfg = datasets_config.get(dataset_key, {}) if isinstance(datasets_config, dict) else {}
    items_cfg = ds_cfg.get("items", {}) if isinstance(ds_cfg, dict) else {}
    item_keys = list(items_cfg.keys()) if isinstance(items_cfg, dict) else []
    by_item, shared = _to_item_window_map(windows)
    return [
        {
            "datasetKey": dataset_key,
            "itemsWithWindows": "all",
            "items": [
                {"itemKey": item_key, "windows": by_item.get(item_key, shared)}
                for item_key in item_keys
            ],
        }
    ]


def _from_dataset_value_for_component(
    mode: str,
    component_value: Any,
    bind_dataset_key: str | None,
    default_dataset_key: str | None,
    datasets_config: dict[str, Any],
) -> Any:
    """Convert component dataset-list value back to schema mode."""
    if mode != "window_list":
        return None
    if not isinstance(component_value, list):
        return None
    dataset_key = bind_dataset_key or default_dataset_key
    if not dataset_key:
        return None
    entry = next((e for e in component_value if isinstance(e, dict) and str(e.get("datasetKey")) == dataset_key), None)
    if not isinstance(entry, dict):
        return None
    items = entry.get("items")
    if not isinstance(items, list):
        return None
    by_key: dict[str, list[dict[str, float]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_key = str(item.get("itemKey") or "").strip()
        if not item_key:
            continue
        raw_windows = _coerce_ranges(item.get("windows"))
        windows = [{"min": float(w["min"]), "max": float(w["max"])} for w in raw_windows]
        by_key[item_key] = windows

    ds_cfg = datasets_config.get(dataset_key, {}) if isinstance(datasets_config, dict) else {}
    items_cfg = ds_cfg.get("items", {}) if isinstance(ds_cfg, dict) else {}
    preferred_keys = list(items_cfg.keys()) if isinstance(items_cfg, dict) else []
    flattened = _flatten_item_window_map(by_key, preferred_keys)
    return flattened or None


def _coerce_time_window_group_value_for_component(
    datasets_config: dict[str, Any],
    dataset_to_field: dict[str, str],
    current_values: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Build component value from grouped schema fields (one field per dataset)."""
    output: list[dict[str, Any]] = []
    for dataset_key, field_key in dataset_to_field.items():
        ds_cfg = datasets_config.get(dataset_key, {}) if isinstance(datasets_config, dict) else {}
        items_cfg = ds_cfg.get("items", {}) if isinstance(ds_cfg, dict) else {}
        item_keys = list(items_cfg.keys()) if isinstance(items_cfg, dict) else []
        windows = _coerce_ranges(current_values.get(field_key))
        by_item, shared = _to_item_window_map(windows)
        items = [{"itemKey": item_key, "windows": by_item.get(item_key, shared)} for item_key in item_keys]
        output.append(
            {
                "datasetKey": dataset_key,
                "itemsWithWindows": "all",
                "items": items,
            }
        )
    return output or None


def _extract_time_window_group_values_from_component(
    component_value: Any,
    datasets_config: dict[str, Any],
    dataset_to_field: dict[str, str],
    current_values: dict[str, Any],
) -> dict[str, Any]:
    """Map component dataset-list value back to grouped field values."""
    updated: dict[str, Any] = {}
    for field_key in dataset_to_field.values():
        updated[field_key] = current_values.get(field_key)

    if not isinstance(component_value, list):
        return updated

    by_dataset: dict[str, dict[str, list[dict[str, float]]]] = {}
    for entry in component_value:
        if not isinstance(entry, dict):
            continue
        dataset_key = str(entry.get("datasetKey") or "").strip()
        if not dataset_key:
            continue
        items = entry.get("items")
        if not isinstance(items, list):
            continue
        by_item: dict[str, list[dict[str, float]]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            item_key = str(item.get("itemKey") or "").strip()
            if not item_key:
                continue
            raw_windows = _coerce_ranges(item.get("windows"))
            windows = [{"min": float(w["min"]), "max": float(w["max"])} for w in raw_windows]
            by_item[item_key] = windows
        by_dataset[dataset_key] = by_item

    for dataset_key, field_key in dataset_to_field.items():
        item_windows = by_dataset.get(dataset_key, {})
        ds_cfg = datasets_config.get(dataset_key, {}) if isinstance(datasets_config, dict) else {}
        items_cfg = ds_cfg.get("items", {}) if isinstance(ds_cfg, dict) else {}
        item_keys = list(items_cfg.keys()) if isinstance(items_cfg, dict) else []
        flattened = _flatten_item_window_map(item_windows, item_keys)
        updated[field_key] = flattened or None
    return updated


@st.cache_data(show_spinner=False)
def _build_datasets_config(
    run_dir: Path,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Build datasets_config for the time_window custom component."""
    ds = _load_run_ds(run_dir)
    if ds is None or not ds.datasets:
        return {}

    min_windows, max_windows = _extract_window_bounds(schema, root_schema, mode)
    if max_windows is None:
        max_windows = 64
    item_scope = {"policy": "all", "exact": None, "min": None, "max": None}

    config: dict[str, Any] = {}
    for dataset in ds.datasets:
        items_cfg: dict[str, Any] = {}
        data_base = run_dir / "input" / "data" / dataset.key / "parquet"
        total = len(dataset.items)
        width = max(2, len(str(max(total - 1, 0))))

        for idx, item in enumerate(dataset.items):
            item_key = f"{idx:0{width}d}"
            stage_id = _owner_stage_id(item.owner, f"item-{idx}")
            item_dir = data_base / f"item-{idx:05d}"
            traces: list[dict[str, Any]] = []
            x_range = [0.0, 1000.0]
            x_step = 1.0
            x_is_time = False

            if item_dir.exists():
                parquet_files = sorted(item_dir.rglob("*.parquet"))
                if parquet_files:
                    with suppress(Exception):
                        traces, x_range, x_step, x_is_time = build_parquet_preview_from_files(
                            parquet_files,
                            max_points=600,
                        )

            items_cfg[item_key] = {
                "stage_id": stage_id,
                "traces": traces,
                "x_is_time": x_is_time,
                "x_range": x_range,
                "x_step": x_step,
            }

        config[dataset.key] = {
            "items": items_cfg,
            "items_with_windows": [],
            "items_scope": item_scope,
            "window_count": {"min": min_windows, "max": max_windows},
        }
    return config


def _render_time_window_v2(
    key: str,
    schema: dict[str, Any],
    value: Any,
    path: str,
    root_schema: dict[str, Any],
) -> Any:
    """Render time_window field using custom component."""
    run_dir_str = st.session_state.get("_active_run_dir")
    run_id = st.session_state.get("_active_run_id", "unknown")
    if not run_dir_str:
        st.warning(tx("Run context unavailable.", "运行上下文不可用。"))
        return value

    mode = _time_window_mode(schema, root_schema)
    if mode != "window_list":
        st.error(tx("time_window schema invalid. Expected Optional[list[TimeWindow]].", "time_window schema 无效，应为 Optional[list[TimeWindow]]。"))
        return value

    loading_slot = st.empty()
    with loading_slot.container(), st.spinner(tx("Loading time-window data preview...", "正在加载时间窗数据预览...")):
        datasets_config = _build_datasets_config(Path(run_dir_str), schema, root_schema, mode)
    loading_slot.empty()
    if not datasets_config:
        st.warning(tx("No datasets found in run input.", "在运行输入中未找到数据集。"))
        return value

    bind_dataset_key = str(schema_meta(schema, "bindDatasetKey", "") or "").strip() or None
    if bind_dataset_key:
        if bind_dataset_key not in datasets_config:
            st.warning(
                tx(
                    "Bound dataset `{dataset_key}` not found in current run DS.",
                    "当前运行 DS 中未找到绑定数据集 `{dataset_key}`。",
                    dataset_key=bind_dataset_key,
                )
            )
            return value
        datasets_config = {bind_dataset_key: datasets_config[bind_dataset_key]}
    default_dataset_key = bind_dataset_key or (next(iter(datasets_config.keys()), None))
    current_value = _as_dataset_value_for_component(
        mode,
        value,
        bind_dataset_key,
        default_dataset_key,
        datasets_config,
    )

    result = time_window_picker(
        datasets_config=datasets_config,
        current_value=current_value,
        max_points_per_trace=600,
        height=500,
        run_id=str(run_id),
        key=f"tw_{path}_{run_id}",
    )
    return _from_dataset_value_for_component(
        mode,
        result,
        bind_dataset_key,
        default_dataset_key,
        datasets_config,
    )


def _enum_format(choice_map: dict[str, str], value: Any) -> str:
    """Format enum value using enumLabels when available."""
    return choice_map.get(str(value), str(value))


def _render_scalar_enum(
    title: str,
    help_text: str | None,
    path: str,
    value: Any,
    default_val: Any,
    choices: tuple[Any, ...],
    choice_map: dict[str, str],
    *,
    nullable: bool,
) -> Any:
    """Render scalar enum as a selectbox."""
    options: list[Any] = [None] if nullable else []
    options.extend(choices)
    current = value if value in options else default_val if default_val in options else (None if nullable else choices[0])
    return st.selectbox(
        title,
        options=options,
        index=options.index(current),
        format_func=lambda option: tx("None", "空值") if option is None else _enum_format(choice_map, option),
        help=help_text,
        key=path,
    )


def _render_array_enum(
    title: str,
    help_text: str | None,
    path: str,
    value: Any,
    default_val: Any,
    choices: tuple[Any, ...],
    choice_map: dict[str, str],
) -> Any:
    """Render enum array as a multiselect."""
    current = value if isinstance(value, list) else default_val if isinstance(default_val, list) else []
    selected = [item for item in current if item in choices]
    return st.multiselect(
        title,
        options=list(choices),
        default=selected,
        format_func=lambda option: _enum_format(choice_map, option),
        help=help_text,
        key=path,
    )


def _render_object_union(
    key: str,
    title: str,
    help_text: str | None,
    schema: dict[str, Any],
    value: Any,
    default_val: Any,
    path: str,
    root_schema: dict[str, Any],
) -> Any:
    """Render discriminator-based object unions."""
    union_schema = extract_object_union(schema, root_schema)
    if union_schema is None:
        return _render_json_editor(title, value, default_val, help_text, path)

    branch_by_key = {branch.key: branch for branch in union_schema.branches}
    branch_options: list[str | None] = [None] if union_schema.nullable else []
    branch_options.extend(branch_by_key.keys())

    current_value = value if isinstance(value, dict) else (default_val if isinstance(default_val, dict) else None)
    active_key = match_object_union_branch(union_schema, current_value)
    current_option: str | None
    if current_value is None and union_schema.nullable:
        current_option = None
    elif active_key not in branch_by_key:
        active_key = union_schema.branches[0].key if union_schema.branches else None
        current_option = active_key if active_key is not None else None
    else:
        current_option = active_key

    selected_option = st.selectbox(
        title,
        options=branch_options,
        index=branch_options.index(current_option if current_option in branch_options else None if union_schema.nullable else branch_options[0]),
        format_func=lambda option: tx("None", "空值")
        if option is None
        else branch_by_key[str(option)].label,
        help=help_text,
        key=f"{path}.__branch__",
    )
    if selected_option is None:
        return None

    branch = branch_by_key[str(selected_option)]
    props = branch.schema.get("properties", {})
    current_obj = current_value if isinstance(current_value, dict) else {}
    default_obj = default_val if isinstance(default_val, dict) else {}
    branch_values = {
        **default_obj,
        **current_obj,
    }
    if union_schema.discriminator:
        branch_values[union_schema.discriminator] = branch.key

    with st.container(border=True):
        st.markdown(f"**{branch.label}**")
        return render_schema_grid(props, branch_values, path, root_schema)


def _coerce_date_string(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if hasattr(raw_value, "isoformat"):
        with suppress(Exception):
            return raw_value.isoformat()
    text = str(raw_value).strip()
    return text or None


def _render_datetime_string(
    title: str,
    help_text: str | None,
    path: str,
    value: Any,
    default_val: Any,
    *,
    date_only: bool,
) -> str | None:
    """Render date/date-time values as explicit ISO text."""
    current = _coerce_date_string(value)
    default_text = _coerce_date_string(default_val)
    val = current if current is not None else (default_text or "")
    suffix = tx("Use ISO format.", "请使用 ISO 格式。")
    merged_help = f"{help_text} {suffix}".strip() if help_text else suffix
    return st.text_input(title, value=val, help=merged_help, key=path, placeholder="YYYY-MM-DD" if date_only else "YYYY-MM-DDTHH:MM:SSZ") or None


def _render_nullable_numeric(
    title: str,
    help_text: str | None,
    path: str,
    value: Any,
    default_val: Any,
    *,
    integer: bool,
) -> int | float | None:
    """Render nullable numeric fields as text so blank can map to None."""
    current = value if value is not None else default_val
    raw = "" if current is None else str(int(current) if integer else current)
    text = st.text_input(title, value=raw, help=help_text, key=path)
    if not text.strip():
        return None
    try:
        return int(text.strip()) if integer else float(text.strip())
    except ValueError:
        return current


def _render_nullable_boolean(
    title: str,
    help_text: str | None,
    path: str,
    value: Any,
    default_val: Any,
) -> bool | None:
    """Render nullable booleans with an explicit null option."""
    options = [None, True, False]
    current = value if value in options else default_val if default_val in options else None
    return st.selectbox(
        title,
        options=options,
        index=options.index(current),
        format_func=lambda option: tx("None", "空值") if option is None else tx("True", "是") if option else tx("False", "否"),
        help=help_text,
        key=path,
    )

def render_field_widget(
    key: str,
    schema: dict[str, Any],
    value: Any,
    path: str,
    root_schema: dict[str, Any],
) -> Any:
    """Render a single widget based on schema type."""
    ui_type = extract_ui_type(schema)
    if ui_type == "time_window":
        return _render_time_window_v2(key, schema, value, path, root_schema)

    const_value = constant_field_value(schema, root_schema)
    if const_value is not None:
        return const_value

    effective_schema = normalize_schema(schema, root_schema)
    ftype = effective_schema.get("type") or schema.get("type")
    title = schema.get("title") or key
    default_val = schema.get("default")
    help_text = schema.get("description")
    nullable = is_nullable_schema(schema, root_schema)

    enum_config = extract_enum_choices(schema, root_schema)
    if enum_config is not None:
        enum_choices, nullable = enum_config
        choice_values = tuple(choice.value for choice in enum_choices)
        choice_map = {str(choice.value): choice.label for choice in enum_choices}
        return _render_scalar_enum(
            title,
            help_text,
            path,
            value,
            default_val,
            choice_values,
            choice_map,
            nullable=nullable,
        )

    array_enum_config = extract_array_enum_choices(schema, root_schema)
    if array_enum_config is not None:
        array_choices, _nullable = array_enum_config
        choice_values = tuple(choice.value for choice in array_choices)
        choice_map = {str(choice.value): choice.label for choice in array_choices}
        return _render_array_enum(title, help_text, path, value, default_val, choice_values, choice_map)

    object_union = extract_object_union(schema, root_schema)
    if object_union is not None:
        return _render_object_union(key, title, help_text, schema, value, default_val, path, root_schema)

    if ftype == "string":
        string_format = effective_schema.get("format")
        if string_format == "date-time":
            return _render_datetime_string(title, help_text, path, value, default_val, date_only=False)
        if string_format == "date":
            return _render_datetime_string(title, help_text, path, value, default_val, date_only=True)
        val = value if value is not None else (default_val or "")
        text = st.text_input(title, value=val, help=help_text, key=path)
        return None if nullable and not text.strip() else text
    
    if ftype == "number":
        if nullable:
            return _render_nullable_numeric(title, help_text, path, value, default_val, integer=False)
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
        if nullable:
            return _render_nullable_numeric(title, help_text, path, value, default_val, integer=True)
        val = value if value is not None else default_val
        return int(st.number_input(title, value=int(val or 0), step=1, help=help_text, key=path))
    
    if ftype == "boolean":
        if nullable:
            return _render_nullable_boolean(title, help_text, path, value, default_val)
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
            props = effective_schema.get("properties", {})
            obj = value if isinstance(value, dict) else (default_val if isinstance(default_val, dict) else {})
            
            # Recursive call to grid layout
            return render_schema_grid(props, obj, path, root_schema)

    # Fallback
    return _render_json_editor(title, value, default_val, help_text, path)

def _render_json_editor(title, value, default, help_text, path):
    """Helper for raw JSON fields."""
    st.markdown(f"<small>{title}</small>", unsafe_allow_html=True)
    current = value if value is not None else (default if default is not None else [])
    json_help = (
        tx("{help_text} (Edit as JSON)", "{help_text}（以 JSON 编辑）", help_text=help_text)
        if help_text
        else tx("Edit as JSON", "以 JSON 编辑")
    )
    text = st.text_area(
        title,
        value=json.dumps(current, indent=2, ensure_ascii=False),
        help=json_help,
        key=path,
        label_visibility="collapsed",
        height=100,
    )
    try:
        return json.loads(text) if text.strip() else current
    except Exception:
        return current

def render_schema_grid(
    properties: dict[str, dict],
    current_values: dict[str, Any],
    prefix: str,
    root_schema: dict[str, Any],
) -> dict[str, Any]:
    """
    Renders fields in a smart grid layout:
    - Compact fields (numbers, bools) get packed into columns (up to 4).
    - Wide fields (objects, arrays) break the line and take full width.
    """
    result: dict[str, Any] = {}
    ordered_properties = sort_schema_properties(properties)
    visible_time_window_fields: list[tuple[str, dict[str, Any], str]] = []
    grouped_time_window_values: dict[str, Any] = {}
    grouped_time_window_context: tuple[dict[str, Any], dict[str, str], str] | None = None
    compact_buffer: list[tuple[str, dict[str, Any]]] = []

    def live_values() -> dict[str, Any]:
        return {**current_values, **result}

    def flush_buffer() -> None:
        if not compact_buffer:
            return
        for i in range(0, len(compact_buffer), 4):
            batch = compact_buffer[i : i + 4]
            cols = st.columns(len(batch))
            for col, (field_key, field_schema) in zip(cols, batch, strict=True):
                with col:
                    result[field_key] = render_field_widget(
                        field_key,
                        field_schema,
                        current_values.get(field_key),
                        f"{prefix}.{field_key}",
                        root_schema,
                    )
        compact_buffer.clear()

    for key, prop_schema in ordered_properties:
        if not field_is_visible(prop_schema, live_values()):
            continue
        if extract_ui_type(prop_schema) == "time_window":
            bind_key = str(schema_meta(prop_schema, "bindDatasetKey", "") or "").strip()
            visible_time_window_fields.append((key, prop_schema, bind_key))
            continue
        const_value = constant_field_value(prop_schema, root_schema)
        if const_value is not None:
            result[key] = const_value
            continue
        if _is_compact_field(prop_schema, root_schema):
            compact_buffer.append((key, prop_schema))
            continue
        flush_buffer()
        result[key] = render_field_widget(key, prop_schema, current_values.get(key), f"{prefix}.{key}", root_schema)

    flush_buffer()

    if visible_time_window_fields:
        missing_bind = [key for key, _schema, bind_key in visible_time_window_fields if not bind_key]
        if missing_bind:
            st.error(
                tx(
                    "time_window schema invalid: every time_window field must define `bindDatasetKey` (top-level or in json_schema_extra). Missing: {missing}",
                    "time_window schema 无效：每个 time_window 字段都必须定义 `bindDatasetKey`（顶层或 json_schema_extra 中）。缺少：{missing}",
                    missing=", ".join(missing_bind),
                )
            )
        else:
            bind_to_field: dict[str, tuple[str, dict[str, Any]]] = {}
            duplicate_binds: list[str] = []
            for field_key, field_schema, bind_key in visible_time_window_fields:
                if bind_key in bind_to_field:
                    duplicate_binds.append(bind_key)
                else:
                    bind_to_field[bind_key] = (field_key, field_schema)
            if duplicate_binds:
                st.error(
                    tx(
                        "time_window schema invalid: duplicate bindDatasetKey detected: {keys}",
                        "time_window schema 无效：检测到重复的 bindDatasetKey：{keys}",
                        keys=", ".join(sorted(set(duplicate_binds))),
                    )
                )
            else:
                run_dataset_keys: list[str] = []
                run_dir_str = st.session_state.get("_active_run_dir")
                if isinstance(run_dir_str, str) and run_dir_str.strip():
                    run_ds = _load_run_ds(Path(run_dir_str))
                    if run_ds is not None:
                        run_dataset_keys = [str(ds.key) for ds in run_ds.datasets]

                if not run_dataset_keys:
                    st.error(
                        tx(
                            "Cannot resolve selected datasets from current run (input/ds.json missing or invalid).",
                            "无法从当前运行中解析已选数据集（input/ds.json 缺失或无效）。",
                        )
                    )
                else:
                    active_datasets = [dataset_key for dataset_key in run_dataset_keys if dataset_key in bind_to_field]
                    if not active_datasets:
                        st.error(
                            tx(
                                "No time_window field matches selected run datasets `{dataset_keys}`.",
                                "没有 time_window 字段匹配当前选中的运行数据集 `{dataset_keys}`。",
                                dataset_keys=", ".join(run_dataset_keys),
                            )
                        )
                    else:
                        group_schema = bind_to_field[active_datasets[0]][1]
                        mode = _time_window_mode(group_schema, root_schema)
                        if mode != "window_list":
                            st.error(
                                tx(
                                    "time_window schema invalid. Expected Optional[list[TimeWindow]].",
                                    "time_window schema 无效，应为 Optional[list[TimeWindow]]。",
                                )
                            )
                        elif not isinstance(run_dir_str, str) or not run_dir_str.strip():
                            st.error(tx("Run context unavailable for time_window widget.", "time_window 组件的运行上下文不可用。"))
                        else:
                            with st.spinner(tx("Loading time-window data preview...", "正在加载时间窗数据预览...")):
                                datasets_config = _build_datasets_config(Path(run_dir_str), group_schema, root_schema, mode)
                            if not datasets_config:
                                st.error(tx("No datasets found in run input.", "在运行输入中未找到数据集。"))
                            else:
                                datasets_config = {
                                    key: value for key, value in datasets_config.items() if key in set(active_datasets)
                                }
                                for dataset_key in active_datasets:
                                    field_key, field_schema = bind_to_field[dataset_key]
                                    min_windows, max_windows = _extract_window_bounds(field_schema, root_schema, "window_list")
                                    if max_windows is None:
                                        max_windows = 64
                                    if dataset_key in datasets_config:
                                        datasets_config[dataset_key]["window_count"] = {"min": min_windows, "max": max_windows}
                                        datasets_config[dataset_key]["window_slots"] = schema_meta(field_schema, "windowSlots", [])
                                        datasets_config[dataset_key]["window_slot_fallback_note"] = str(
                                            schema_meta(field_schema, "windowSlotFallbackNote", "") or ""
                                        ).strip()

                                dataset_to_field = {
                                    dataset_key: bind_to_field[dataset_key][0] for dataset_key in active_datasets
                                }
                                run_id = str(st.session_state.get("_active_run_id", "unknown"))
                                grouped_time_window_context = (datasets_config, dataset_to_field, run_id)

    if grouped_time_window_context is not None:
        datasets_config, dataset_to_field, run_id = grouped_time_window_context
        component_value = _coerce_time_window_group_value_for_component(
            datasets_config,
            dataset_to_field,
            current_values,
        )
        result_value = time_window_picker(
            datasets_config=datasets_config,
            current_value=component_value,
            max_points_per_trace=600,
            height=500,
            run_id=run_id,
            key=f"tw_group_{prefix}_{run_id}",
        )
        grouped_time_window_values = _extract_time_window_group_values_from_component(
            result_value,
            datasets_config,
            dataset_to_field,
            current_values,
        )

    for field_key, _field_schema, _bind_key in visible_time_window_fields:
        result[field_key] = grouped_time_window_values.get(field_key, current_values.get(field_key))

    for key, value in current_values.items():
        if key not in result:
            result[key] = value

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
    st.info(tx("No runs available. Create a run from the Selection page.", "没有可用运行。请先在“运行配置”页面创建运行。"))
    st.stop()

pending_runs = [r for r in runs if r.status == RunStatus.PENDING]
other_runs = [r for r in runs if r.status != RunStatus.PENDING]

# ------------------------------------------
# 1. Pending Runs (Editor Workspace)
# ------------------------------------------
st.subheader(tx("Pending Runs", "待运行任务"))

if not pending_runs:
    st.info(tx("No pending runs waiting for execution.", "没有等待执行的待运行任务。"))
else:
    # Use tabs for context switching
    tabs = st.tabs([f"⚙️ {run.run_id}" for run in pending_runs])
    
    for tab, run in zip(tabs, pending_runs, strict=True):
        with tab:
            run_dir = run_mgr.get_run_dir(run.run_id)
            st.session_state["_active_run_id"] = run.run_id
            st.session_state["_active_run_dir"] = str(run_dir)
            algo_handle = algo_lib.get_algorithm(run.algorithm_id, run.algorithm_version)
            schema = algo_handle.params_schema
            current_params = load_params(run_dir)
            
            # --- Layout: Top Info Bar ---
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
                with c1:
                    st.caption(tx("**Snapshot:** `{snapshot_id}`", "**快照：** `{snapshot_id}`", snapshot_id=run.snapshot_id))
                with c2:
                    st.caption(tx("**Algo:** `{algorithm_id}` v{version}", "**算法：** `{algorithm_id}` v{version}", algorithm_id=run.algorithm_id, version=run.algorithm_version))
                with c3:
                    st.caption(tx("**Created:** {created_at}", "**创建时间：** {created_at}", created_at=run.created_at))
                with c4:
                    # Timeout setting tucked away here
                    timeout = st.number_input(tx("Timeout (s)", "超时（秒）"), value=300, step=10, key=f"to_{run.run_id}", label_visibility="collapsed")

            # --- Layout: Parameters Grid ---
            st.markdown(tx("##### Parameters", "##### 参数"))
            with st.container(border=True):
                if schema.get("type") == "object":
                    props = schema.get("properties", {})
                    # CALL THE GRID ENGINE
                    new_params = render_schema_grid(props, current_params, prefix=f"run_{run.run_id}", root_schema=schema)
                else:
                    st.info(tx("Schema is not an object, editing raw JSON.", "Schema 不是 object，将直接编辑原始 JSON。"))
                    new_params = _render_json_editor(tx("Raw Params", "原始参数"), current_params, {}, "", f"run_raw_{run.run_id}")

            st.divider()
            
            # --- Layout: Action Footer ---
            # Right-aligned actions
            _, col_btns = st.columns([3, 4])
            with col_btns:
                b1, b2, b3 = st.columns([1, 1, 1.5], gap="small")

                with b1:
                    if st.button(tx("🚫 Cancel", "🚫 取消"), key=f"cancel_{run.run_id}", width="stretch"):
                        try:
                            run_mgr.delete_run(run.run_id)
                            st.success(tx("Cancelled", "已取消"))
                            st.rerun()
                        except Exception as e:
                            st.error(tx("Error: {error}", "错误：{error}", error=e))

                with b2:
                    if st.button(tx("💾 Save", "💾 保存"), key=f"save_{run.run_id}", width="stretch"):
                        try:
                            (run_dir / "input").mkdir(parents=True, exist_ok=True)
                            (run_dir / "input" / "params.json").write_text(
                                json.dumps(new_params, indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                            st.toast(tx("Parameters saved successfully!", "参数保存成功！"), icon="💾")
                        except Exception as e:
                            st.error(tx("Save failed: {error}", "保存失败：{error}", error=e))

                with b3:
                    if st.button(tx("▶️ Run Algorithm", "▶️ 运行算法"), key=f"exec_{run.run_id}", type="primary", width="stretch"):
                        try:
                            # Auto-save before run
                            (run_dir / "input").mkdir(parents=True, exist_ok=True)
                            (run_dir / "input" / "params.json").write_text(
                                json.dumps(new_params, indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )

                            with st.spinner(tx("Initializing execution...", "正在初始化执行...")):
                                result = run_mgr.execute(run.run_id, timeout_s=int(timeout))

                            if result.error:
                                st.error(
                                    tx(
                                        "Run Finished: {status}\n{error}",
                                        "运行结束：{status}\n{error}",
                                        status=run_status_label(result.status),
                                        error=result.error,
                                    )
                                )
                            else:
                                # Navigate to Results page with executed run
                                st.session_state.executed_run_id = run.run_id
                                st.switch_page("pages/5_Results.py")
                        except Exception as e:
                            st.error(tx("Execution Exception: {error}", "执行异常：{error}", error=e))


# ------------------------------------------
# 2. History (Other Runs)
# ------------------------------------------
st.subheader(tx("Run History", "运行历史"))

if not other_runs:
    st.caption(tx("No historical runs.", "没有历史运行记录。"))

other_runs_reversed = other_runs[::-1]

for run in other_runs_reversed:
    status_config = {
        RunStatus.PENDING: ("⏳", run_status_label(RunStatus.PENDING), "gray"),
        RunStatus.RUNNING: ("🔄", run_status_label(RunStatus.RUNNING), "blue"),
        RunStatus.SUCCEEDED: ("✅", run_status_label(RunStatus.SUCCEEDED), "green"),
        RunStatus.FAILED: ("❌", run_status_label(RunStatus.FAILED), "red"),
        RunStatus.TIMEOUT: ("⏱️", run_status_label(RunStatus.TIMEOUT), "orange"),
    }
    icon, label, color = status_config.get(run.status, ("❓", run_status_label("unknown"), "gray"))
    
    with st.expander(f"{icon} {run.run_id}", expanded=False), st.container(border=True):
        # Info
        c1, c2, c3 = st.columns([3, 2, 2])
        with c1:
            st.caption(tx("Context", "上下文"))
            st.markdown(f"**{run.algorithm_id}** v{run.algorithm_version}")
            st.text(tx("Snap: {snapshot_id}", "快照：{snapshot_id}", snapshot_id=run.snapshot_id))
        with c2:
            st.caption(tx("Timing", "时间"))
            st.text(tx("Start: {value}", "开始：{value}", value=run.started_at or "--"))
            st.text(tx("End:   {value}", "结束：{value}", value=run.completed_at or "--"))
        with c3:
            st.caption(tx("Status", "状态"))
            st.markdown(f":{color}[**{label}**]")
            if run.error:
                st.error(run.error)

        # Params Read-only
        st.divider()
        st.caption(tx("Run Parameters", "运行参数"))
        run_dir = run_mgr.get_run_dir(run.run_id)
        params_view = load_params(run_dir)
        if params_view:
            st.code(json.dumps(params_view, indent=2, ensure_ascii=False), language="json")
