"""Streamlit wrapper for time window picker component."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_component = components.declare_component("time_window_picker", path=str(_FRONTEND_DIR))


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
    return _component(
        datasets_config=datasets_config,
        current_value=current_value,
        max_points_per_trace=max_points_per_trace,
        height=height,
        run_id=run_id,
        key=key,
        default=current_value,
    )
