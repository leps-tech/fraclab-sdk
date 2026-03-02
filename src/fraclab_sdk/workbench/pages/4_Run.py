"""Run page: edit params for existing runs and execute."""

import json
from pathlib import Path
from typing import Any, Dict
from decimal import Decimal, InvalidOperation

import pandas as pd
import streamlit as st

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.models import DataSpec
from fraclab_sdk.run import RunManager, RunStatus
from fraclab_sdk.workbench.components.time_window_picker import time_window_picker
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


def _extract_ui_type(schema: dict[str, Any]) -> str | None:
    """Extract ui_type from schema field metadata."""
    if isinstance(schema.get("ui_type"), str):
        return schema.get("ui_type")
    extra = schema.get("json_schema_extra")
    if isinstance(extra, dict) and isinstance(extra.get("ui_type"), str):
        return extra.get("ui_type")
    return None


def _schema_meta(schema: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read metadata from top-level first, then json_schema_extra."""
    if key in schema:
        return schema.get(key)
    extra = schema.get("json_schema_extra")
    if isinstance(extra, dict) and key in extra:
        return extra.get(key)
    return default


def _resolve_ref_schema(schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local JSON-schema $ref like '#/$defs/Name'."""
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    current: Any = root_schema
    for part in ref[2:].split("/"):
        if not isinstance(current, dict):
            return schema
        current = current.get(part)
        if current is None:
            return schema
    return current if isinstance(current, dict) else schema


def _unwrap_nullable_anyof(schema: dict[str, Any]) -> dict[str, Any]:
    """For Optional[T], prefer the non-null branch."""
    any_of = schema.get("anyOf")
    if not isinstance(any_of, list):
        return schema
    non_null = [branch for branch in any_of if isinstance(branch, dict) and branch.get("type") != "null"]
    if len(non_null) == 1:
        return non_null[0]
    return schema


def _normalize_schema(schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve common wrappers (anyOf/null + $ref) for type-shape checks."""
    current = _unwrap_nullable_anyof(schema)
    if "$ref" in current:
        current = _resolve_ref_schema(current, root_schema)
    return current


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


def _pick_xy_columns(df: pd.DataFrame) -> tuple[str | None, list[str], str]:
    """Pick x column and all numeric y columns, preferring datetime x-axis."""
    if df.empty:
        return None, [], "numeric"
    x_time_candidates = ["timestamp", "bucket", "ts", "time", "datetime", "dateTime", "date", "t"]
    x_numeric_candidates = ["sec", "seconds", "x"]
    y_candidates = ["treatingPressure", "pressure", "value", "y", "slurryRate", "proppantConc"]
    col_map = {str(c).lower(): c for c in df.columns}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for c in x_time_candidates:
        actual = col_map.get(c.lower())
        if actual is not None:
            parsed = pd.to_datetime(df[actual], errors="coerce", utc=True)
            if parsed.notna().any():
                x_col = actual
                break
    else:
        x_col = None

    x_mode = "datetime"
    if x_col is None:
        x_mode = "numeric"
        x_col = next((col_map.get(c.lower()) for c in x_numeric_candidates if col_map.get(c.lower()) in numeric_cols), None)
        if x_col is None and numeric_cols:
            x_col = numeric_cols[0]

    y_priority = [col_map[c.lower()] for c in y_candidates if col_map.get(c.lower()) in numeric_cols and col_map[c.lower()] != x_col]
    y_remaining = [c for c in numeric_cols if c != x_col and c not in y_priority]
    y_cols = y_priority + y_remaining
    return x_col, y_cols, x_mode


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    """Parse timestamp column with numeric-unit inference."""
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric.dropna()
        if finite.empty:
            return pd.to_datetime(series, errors="coerce", utc=True)
        # Try multiple units and choose the most plausible calendar range.
        candidates: list[tuple[int, float, pd.Series]] = []
        now_year = pd.Timestamp.utcnow().year
        for unit in ("ns", "us", "ms", "s"):
            dt = pd.to_datetime(numeric, errors="coerce", utc=True, unit=unit)
            valid = dt.dropna()
            if valid.empty:
                continue
            years = valid.dt.year
            in_range = int(((years >= 2000) & (years <= 2100)).sum())
            median_year = float(years.median())
            year_distance = abs(median_year - now_year)
            # prioritize in-range count, then closeness to current year
            score = in_range * 1000 - year_distance
            candidates.append((len(valid), score, dt))
        if candidates:
            candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
            return candidates[0][2]
        return pd.to_datetime(numeric, errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)


def _datetime_series_to_epoch_seconds(series: pd.Series) -> list[float]:
    """Convert datetime-like series to epoch seconds with fixed ns precision."""
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    dt = dt.dropna()
    if dt.empty:
        return []
    # Force ns precision before integer conversion to avoid ms/us unit drift.
    dt_ns = dt.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64")
    return (dt_ns / 1e9).astype(float).tolist()


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


def _downsample_trace(
    x_vals: list[Any],
    y_vals: list[float],
    max_points: int = 1500,
) -> tuple[list[Any], list[float]]:
    """Downsample trace for display performance while keeping endpoints."""
    n = min(len(x_vals), len(y_vals))
    if n <= max_points:
        return x_vals[:n], y_vals[:n]
    step = max(1, n // max_points)
    xs = x_vals[::step]
    ys = y_vals[::step]
    if xs and x_vals[n - 1] != xs[-1]:
        xs.append(x_vals[n - 1])
        ys.append(y_vals[n - 1])
    return xs, ys


def _infer_min_positive_step(values: list[float]) -> float | None:
    """Infer minimum positive step from x-values."""
    if len(values) < 2:
        return None
    uniq = sorted(set(float(v) for v in values))
    if len(uniq) < 2:
        return None
    min_diff: float | None = None
    prev = uniq[0]
    for cur in uniq[1:]:
        diff = cur - prev
        prev = cur
        if diff <= 0:
            continue
        if min_diff is None or diff < min_diff:
            min_diff = diff
    return min_diff


def _merge_parquet_parts_for_preview(
    parquet_files: list[Path],
    *,
    max_points: int = 600,
) -> tuple[list[dict[str, Any]], list[float], float, bool]:
    """Merge all parquet parts for one item and build downsampled preview traces.

    Returns:
        traces, x_range, x_step, x_is_datetime
    """
    if not parquet_files:
        return [], [0.0, 1000.0], 1.0, False

    # Probe schema from first readable part.
    probe_df: pd.DataFrame | None = None
    for pf in parquet_files:
        try:
            probe_df = pd.read_parquet(pf)
            break
        except Exception:
            continue
    if probe_df is None or probe_df.empty:
        return [], [0.0, 1000.0], 1.0, False

    x_col, y_cols, x_mode = _pick_xy_columns(probe_df)
    if not x_col or not y_cols:
        return [], [0.0, 1000.0], 1.0, False

    needed_cols = [x_col, *y_cols]
    per_file_budget = max(4, min(96, (max_points * 4) // max(1, len(parquet_files))))
    x_is_datetime = x_mode == "datetime"

    buckets: dict[str, dict[str, list[float]]] = {name: {"x": [], "y": []} for name in y_cols}
    global_x_min: float | None = None
    global_x_max: float | None = None
    global_x_step: float | None = None

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf, columns=needed_cols)
        except Exception:
            continue
        if df.empty or x_col not in df.columns:
            continue

        if x_is_datetime:
            x_dt = _parse_timestamp_series(df[x_col])
            base_mask = x_dt.notna()
            if not base_mask.any():
                continue
            x_valid = _datetime_series_to_epoch_seconds(x_dt[base_mask])
        else:
            x_num = pd.to_numeric(df[x_col], errors="coerce")
            base_mask = x_num.notna()
            if not base_mask.any():
                continue
            x_valid = x_num[base_mask].astype(float).tolist()

        if x_valid:
            local_min = float(min(x_valid))
            local_max = float(max(x_valid))
            global_x_min = local_min if global_x_min is None else min(global_x_min, local_min)
            global_x_max = local_max if global_x_max is None else max(global_x_max, local_max)
            local_step = _infer_min_positive_step(x_valid[:2000])
            if local_step is not None and local_step > 0:
                global_x_step = local_step if global_x_step is None else min(global_x_step, local_step)

        for y_col in y_cols:
            if y_col not in df.columns:
                continue
            y_series = pd.to_numeric(df[y_col], errors="coerce")
            mask = base_mask & y_series.notna()
            if not mask.any():
                continue

            if x_is_datetime:
                x_vals = _datetime_series_to_epoch_seconds(x_dt[mask])
            else:
                x_vals = pd.to_numeric(df.loc[mask, x_col], errors="coerce").astype(float).tolist()
            y_vals = y_series[mask].astype(float).tolist()
            if not x_vals or len(x_vals) != len(y_vals):
                continue

            x_ds, y_ds = _downsample_trace(x_vals, y_vals, max_points=per_file_budget)
            buckets[y_col]["x"].extend(float(v) for v in x_ds)
            buckets[y_col]["y"].extend(float(v) for v in y_ds)

    traces: list[dict[str, Any]] = []
    for y_col in y_cols:
        xs = buckets[y_col]["x"]
        ys = buckets[y_col]["y"]
        if not xs or len(xs) != len(ys):
            continue

        # Keep monotonic x for plotting, then globally downsample.
        pairs = sorted(zip(xs, ys), key=lambda it: it[0])
        xs_sorted = [float(p[0]) for p in pairs]
        ys_sorted = [float(p[1]) for p in pairs]
        xs_final, ys_final = _downsample_trace(xs_sorted, ys_sorted, max_points=max_points)
        traces.append({"name": y_col, "x": xs_final, "y": ys_final})

    if not traces:
        return [], [0.0, 1000.0], (1.0 if x_is_datetime else 1e-6), x_is_datetime

    if global_x_min is None or global_x_max is None:
        x_all = traces[0]["x"]
        global_x_min = float(min(x_all)) if x_all else 0.0
        global_x_max = float(max(x_all)) if x_all else 1000.0
    x_step = global_x_step if global_x_step is not None else (1.0 if x_is_datetime else 1e-6)
    return traces, [global_x_min, global_x_max], float(x_step), x_is_datetime


def _build_item_label(item: dict[str, Any]) -> str:
    """Build compact item label for selectors."""
    item_key = str(item.get("itemKey", ""))
    stage_id = str(item.get("stageId", ""))
    if stage_id and stage_id != item_key:
        return f"{item_key} · {stage_id}"
    return item_key


def _time_window_mode(schema: dict[str, Any], root_schema: dict[str, Any]) -> str | None:
    """Resolve schema mode for ui_type=time_window."""
    normalized = _normalize_schema(schema, root_schema)
    if normalized.get("type") != "array":
        return None
    item_schema = _normalize_schema(normalized.get("items", {}), root_schema)
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
    normalized = _normalize_schema(schema, root_schema)
    min_items = normalized.get("minItems")
    max_items = normalized.get("maxItems")
    return (min_items if isinstance(min_items, int) else None, max_items if isinstance(max_items, int) else None)


def _extract_items_scope_bounds_from_schema(
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Extract item scope policy/count constraints from schema."""
    _ = (schema, root_schema, mode)
    # New simplified schema: apply template windows to all selected items.
    return {"policy": "all", "exact": None, "min": None, "max": None}


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
    item_scope = _extract_items_scope_bounds_from_schema(schema, root_schema, mode)

    config: dict[str, Any] = {}
    for dataset in ds.datasets:
        items_cfg: dict[str, Any] = {}
        data_base = run_dir / "input" / "data" / dataset.datasetKey / "parquet"
        total = len(dataset.items)
        width = max(2, len(str(max(total - 1, 0))))

        for idx, item in enumerate(dataset.items):
            item_key = f"{idx:0{width}d}"
            stage_id = _owner_stage_id(item.owner, f"item-{idx}")
            item_dir = data_base / f"item-{idx:05d}"
            traces: list[dict[str, Any]] = []
            x_range = [0.0, 1000.0]
            x_step = 1.0
            x_is_datetime = False

            if item_dir.exists():
                parquet_files = sorted(item_dir.rglob("*.parquet"))
                if parquet_files:
                    try:
                        traces, x_range, x_step, x_is_datetime = _merge_parquet_parts_for_preview(
                            parquet_files,
                            max_points=600,
                        )
                    except Exception:
                        pass

            items_cfg[item_key] = {
                "stage_id": stage_id,
                "traces": traces,
                "x_is_datetime": x_is_datetime,
                "x_range": x_range,
                "x_step": x_step,
            }

        config[dataset.datasetKey] = {
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
        st.warning("Run context unavailable.")
        return value

    mode = _time_window_mode(schema, root_schema)
    if mode != "window_list":
        st.error("time_window schema invalid. Expected Optional[list[TimeWindow]].")
        return value

    loading_slot = st.empty()
    with loading_slot.container():
        with st.spinner("Loading time-window data preview..."):
            datasets_config = _build_datasets_config(Path(run_dir_str), schema, root_schema, mode)
    loading_slot.empty()
    if not datasets_config:
        st.warning("No datasets found in run input.")
        return value

    bind_dataset_key = str(_schema_meta(schema, "bind_dataset_key", "") or "").strip() or None
    if bind_dataset_key:
        if bind_dataset_key not in datasets_config:
            st.warning(f"Bound dataset `{bind_dataset_key}` not found in current run DS.")
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

def render_field_widget(
    key: str,
    schema: dict[str, Any],
    value: Any,
    path: str,
    root_schema: dict[str, Any],
) -> Any:
    """Render a single widget based on schema type."""
    ui_type = _extract_ui_type(schema)
    if ui_type == "time_window":
        return _render_time_window_v2(key, schema, value, path, root_schema)

    effective_schema = _normalize_schema(schema, root_schema)
    ftype = effective_schema.get("type") or schema.get("type")
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

def render_schema_grid(
    properties: Dict[str, dict],
    current_values: Dict[str, Any],
    prefix: str,
    root_schema: dict[str, Any],
) -> Dict[str, Any]:
    """
    Renders fields in a smart grid layout:
    - Compact fields (numbers, bools) get packed into columns (up to 4).
    - Wide fields (objects, arrays) break the line and take full width.
    """
    result = {}
    time_window_fields: list[tuple[str, dict[str, Any], str]] = []
    for p_key, p_schema in properties.items():
        if _extract_ui_type(p_schema) == "time_window":
            bind_key = str(_schema_meta(p_schema, "bind_dataset_key", "") or "").strip()
            time_window_fields.append((p_key, p_schema, bind_key))

    grouped_time_window_values: dict[str, Any] = {}
    grouped_time_window_context: tuple[dict[str, Any], dict[str, str], str] | None = None
    if time_window_fields:
        missing_bind = [k for (k, _, b) in time_window_fields if not b]
        if missing_bind:
            st.error(
                "time_window schema invalid: every time_window field must define "
                f"`bind_dataset_key` (top-level or in json_schema_extra). Missing: {', '.join(missing_bind)}"
            )
        else:
            bind_to_field: dict[str, tuple[str, dict[str, Any]]] = {}
            duplicate_binds: list[str] = []
            for field_key, field_schema, bind_key in time_window_fields:
                if bind_key in bind_to_field:
                    duplicate_binds.append(bind_key)
                else:
                    bind_to_field[bind_key] = (field_key, field_schema)
            if duplicate_binds:
                st.error(
                    "time_window schema invalid: duplicate bind_dataset_key detected: "
                    f"{', '.join(sorted(set(duplicate_binds)))}"
                )
            else:
                run_dataset_keys: list[str] = []
                run_dir_str = st.session_state.get("_active_run_dir")
                if isinstance(run_dir_str, str) and run_dir_str.strip():
                    run_ds = _load_run_ds(Path(run_dir_str))
                    if run_ds is not None:
                        run_dataset_keys = [str(ds.datasetKey) for ds in run_ds.datasets]

                if not run_dataset_keys:
                    st.error("Cannot resolve selected datasets from current run (input/ds.json missing or invalid).")
                else:
                    active_datasets = [ds_key for ds_key in run_dataset_keys if ds_key in bind_to_field]
                    if not active_datasets:
                        st.error(
                            "No time_window field matches selected run datasets "
                            f"`{', '.join(run_dataset_keys)}`."
                        )
                    else:
                        group_schema = bind_to_field[active_datasets[0]][1]
                        mode = _time_window_mode(group_schema, root_schema)
                        if mode != "window_list":
                            st.error("time_window schema invalid. Expected Optional[list[TimeWindow]].")
                        elif not isinstance(run_dir_str, str) or not run_dir_str.strip():
                            st.error("Run context unavailable for time_window widget.")
                        else:
                            with st.spinner("Loading time-window data preview..."):
                                datasets_config = _build_datasets_config(Path(run_dir_str), group_schema, root_schema, mode)
                            if not datasets_config:
                                st.error("No datasets found in run input.")
                            else:
                                datasets_config = {k: v for k, v in datasets_config.items() if k in set(active_datasets)}
                                for dataset_key in active_datasets:
                                    field_key, field_schema = bind_to_field[dataset_key]
                                    min_windows, max_windows = _extract_window_bounds(field_schema, root_schema, "window_list")
                                    if max_windows is None:
                                        max_windows = 64
                                    if dataset_key in datasets_config:
                                        datasets_config[dataset_key]["window_count"] = {"min": min_windows, "max": max_windows}
                                        datasets_config[dataset_key]["window_slots"] = _schema_meta(field_schema, "window_slots", [])
                                        datasets_config[dataset_key]["window_slot_fallback_note"] = (
                                            str(_schema_meta(field_schema, "window_slot_fallback_note", "") or "").strip()
                                        )

                                dataset_to_field = {dataset_key: bind_to_field[dataset_key][0] for dataset_key in active_datasets}
                                run_id = str(st.session_state.get("_active_run_id", "unknown"))
                                grouped_time_window_context = (datasets_config, dataset_to_field, run_id)
    
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
                    result[b_key] = render_field_widget(b_key, b_schema, val, f"{prefix}.{b_key}", root_schema)
        
        compact_buffer.clear()

    for key, prop_schema in properties.items():
        if _extract_ui_type(prop_schema) == "time_window":
            continue
        if _is_compact_field(prop_schema):
            compact_buffer.append((key, prop_schema))
        else:
            # Wide field encountered: flush buffer first
            flush_buffer()
            # Render wide field
            val = current_values.get(key)
            result[key] = render_field_widget(key, prop_schema, val, f"{prefix}.{key}", root_schema)
    
    # Final flush
    flush_buffer()

    # Render one unified time-window selector at the bottom.
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

    for field_key, _, _ in time_window_fields:
        result[field_key] = grouped_time_window_values.get(field_key, current_values.get(field_key))
    
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
            st.session_state["_active_run_id"] = run.run_id
            st.session_state["_active_run_dir"] = str(run_dir)
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
                    new_params = render_schema_grid(props, current_params, prefix=f"run_{run.run_id}", root_schema=schema)
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
