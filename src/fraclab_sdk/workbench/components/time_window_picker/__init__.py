"""Streamlit wrapper for time window picker component."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

from fraclab_sdk.workbench.i18n import get_language, tx

_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_component = components.declare_component("time_window_picker", path=str(_FRONTEND_DIR))


def _component_i18n() -> dict[str, str]:
    """Return localized strings consumed by the frontend component."""
    return {
        "dataset_label": tx("Dataset", "数据集"),
        "items_scope_label": tx("Items Scope", "条目范围"),
        "clear_pending_button": tx("Clear Pending", "清除暂存"),
        "selected_windows_title": tx("Selected Windows · Run", "已选时间窗 · 运行"),
        "run_id_unknown": tx("unknown", "未知"),
        "no_datasets_available": tx("No datasets available.", "没有可用数据集。"),
        "scope_required_all": tx(
            "Required by schema: all {count} item(s) must be selected.",
            "Schema 要求：必须选择全部 {count} 个条目。",
        ),
        "scope_required_exact": tx(
            "Required by schema: exactly {exact} item(s). Current: {count}.",
            "Schema 要求：必须精确选择 {exact} 个条目。当前：{count}。",
        ),
        "scope_required_min_max": tx(
            "Required by schema: min {min}, max {max}. Current: {count}.",
            "Schema 要求：最少 {min} 个、最多 {max} 个。当前：{count}。",
        ),
        "time_axis_label": tx("Time (UTC)", "时间（UTC）"),
        "x_axis_label": tx("X", "X"),
        "value_axis_label": tx("Value", "数值"),
        "pending_start_hint": tx(
            "Pending start: {start}. Click again to set end.",
            "起点暂存：{start}。再次点击以设置终点。",
        ),
        "click_to_set_hint": tx(
            "Click chart to set start, then click again to set end.",
            "点击图表设置起点，再次点击设置终点。",
        ),
        "window_default_title": tx("Window {index}", "时间窗 {index}"),
        "constraint_current_exact": tx(
            "Current: {count}, requires exactly {exact}.",
            "当前：{count}，要求精确为 {exact}。",
        ),
        "constraint_current_min": tx(
            "Current: {count}, requires at least {min}.",
            "当前：{count}，至少需要 {min}。",
        ),
        "constraint_current_max": tx(
            "Current: {count}, maximum is {max}.",
            "当前：{count}，最大允许 {max}。",
        ),
        "constraint_exact": tx(
            "Constraint: exactly {exact} window(s){invalid_suffix}",
            "约束：精确 {exact} 个时间窗{invalid_suffix}",
        ),
        "constraint_min_max": tx(
            "Constraint: {bounds} window(s){invalid_suffix}",
            "约束：{bounds} 个时间窗{invalid_suffix}",
        ),
        "constraint_bounds_min_max": tx(
            "min {min}, max {max}",
            "最少 {min}、最多 {max}",
        ),
        "constraint_bounds_max_only": tx(
            "max {max}",
            "最多 {max}",
        ),
        "no_windows_selected": tx("No windows selected yet.", "还没有选择时间窗。"),
        "summary_window_line": tx(
            "Window {index}: {start} -> {end}",
            "时间窗 {index}：{start} -> {end}",
        ),
        "delete_button": tx("Delete", "删除"),
    }


def time_window_picker(
    *,
    datasets_config: dict[str, Any],
    current_value: list[dict[str, Any]] | None,
    max_points_per_trace: int = 600,
    height: int = 520,
    run_id: str = "unknown",
    key: str | None = None,
) -> list[dict[str, Any]] | None:
    """Render time-window picker and return edited value."""
    language = get_language()
    return _component(
        datasets_config=datasets_config,
        current_value=current_value,
        max_points_per_trace=max_points_per_trace,
        height=height,
        run_id=run_id,
        language=language,
        i18n=_component_i18n(),
        key=key,
        default=current_value,
    )
