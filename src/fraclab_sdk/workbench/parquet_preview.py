"""Shared parquet preview helpers for workbench pages."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


class ParquetPreviewCancelled(RuntimeError):
    """Signal cooperative cancellation for parquet preview work."""


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    """Abort preview generation when a caller marks the job as cancelled."""
    if should_cancel is not None and should_cancel():
        raise ParquetPreviewCancelled


def _pick_xy_columns(df: pd.DataFrame) -> tuple[str | None, list[str], str]:
    """Pick x column and all numeric y columns, preferring datetime x-axis."""
    if df.empty:
        return None, [], "numeric"

    x_time_candidates = ["timestamp", "bucket", "ts", "ts_us", "time", "datetime", "dateTime", "date", "t"]
    x_numeric_candidates = ["sec", "seconds", "x"]
    y_candidates = [
        "treatingPressure",
        "pressure",
        "casingPressure",
        "flowbackRate",
        "gasRate",
        "value",
        "y",
        "slurryRate",
        "proppantConc",
        "avg",
        "last",
        "max",
        "min",
    ]
    col_map = {str(c).lower(): c for c in df.columns}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for candidate in x_time_candidates:
        actual = col_map.get(candidate.lower())
        if actual is None:
            continue
        parsed = _parse_timestamp_series(df[actual])
        if parsed.notna().any():
            return actual, _ordered_y_columns(col_map, numeric_cols, actual, y_candidates), "datetime"

    x_col = next(
        (col_map.get(candidate.lower()) for candidate in x_numeric_candidates if col_map.get(candidate.lower()) in numeric_cols),
        None,
    )
    if x_col is None and numeric_cols:
        x_col = numeric_cols[0]
    y_cols = _ordered_y_columns(col_map, numeric_cols, x_col, y_candidates)
    return x_col, y_cols, "numeric"


def _ordered_y_columns(
    col_map: dict[str, Any],
    numeric_cols: list[str],
    x_col: str | None,
    priority_candidates: list[str],
) -> list[str]:
    """Return numeric y columns with preferred fields first."""
    y_priority = [
        col_map[candidate.lower()]
        for candidate in priority_candidates
        if col_map.get(candidate.lower()) in numeric_cols and col_map[candidate.lower()] != x_col
    ]
    y_remaining = [col for col in numeric_cols if col != x_col and col not in y_priority]
    return y_priority + y_remaining


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    """Parse a timestamp-like column, inferring likely numeric epoch units."""
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric.dropna()
        if finite.empty:
            return pd.to_datetime(series, errors="coerce", utc=True)

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
            score = in_range * 1000 - year_distance
            candidates.append((len(valid), score, dt))
        if candidates:
            candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
            return candidates[0][2]
        return pd.to_datetime(numeric, errors="coerce", utc=True)

    return pd.to_datetime(series, errors="coerce", utc=True)


def _datetime_series_to_epoch_microseconds(series: pd.Series) -> list[float]:
    """Convert datetime-like values to epoch microseconds."""
    dt = pd.to_datetime(series, errors="coerce", utc=True).dropna()
    if dt.empty:
        return []
    dt_ns = dt.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64")
    return (dt_ns // 1_000).astype(float).tolist()


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
    uniq = sorted(set(float(value) for value in values))
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


def build_parquet_preview_from_files(
    parquet_files: list[Path],
    *,
    max_points: int = 600,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[list[dict[str, Any]], list[float], float, bool]:
    """Merge parquet parts and build downsampled preview traces."""
    _raise_if_cancelled(should_cancel)
    if not parquet_files:
        return [], [0.0, 1000.0], 1.0, False

    probe_df: pd.DataFrame | None = None
    for parquet_file in parquet_files:
        _raise_if_cancelled(should_cancel)
        try:
            probe_df = pd.read_parquet(parquet_file)
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

    for parquet_file in parquet_files:
        _raise_if_cancelled(should_cancel)
        try:
            df = pd.read_parquet(parquet_file, columns=needed_cols)
        except Exception:
            continue
        if df.empty or x_col not in df.columns:
            continue

        if x_is_datetime:
            x_dt = _parse_timestamp_series(df[x_col])
            base_mask = x_dt.notna()
            if not base_mask.any():
                continue
            x_valid = _datetime_series_to_epoch_microseconds(x_dt[base_mask])
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
            _raise_if_cancelled(should_cancel)
            if y_col not in df.columns:
                continue
            y_series = pd.to_numeric(df[y_col], errors="coerce")
            mask = base_mask & y_series.notna()
            if not mask.any():
                continue

            if x_is_datetime:
                x_vals = _datetime_series_to_epoch_microseconds(x_dt[mask])
            else:
                x_vals = pd.to_numeric(df.loc[mask, x_col], errors="coerce").astype(float).tolist()
            y_vals = y_series[mask].astype(float).tolist()
            if not x_vals or len(x_vals) != len(y_vals):
                continue

            x_ds, y_ds = _downsample_trace(x_vals, y_vals, max_points=per_file_budget)
            buckets[y_col]["x"].extend(float(value) for value in x_ds)
            buckets[y_col]["y"].extend(float(value) for value in y_ds)

    traces: list[dict[str, Any]] = []
    for y_col in y_cols:
        xs = buckets[y_col]["x"]
        ys = buckets[y_col]["y"]
        if not xs or len(xs) != len(ys):
            continue
        pairs = sorted(zip(xs, ys, strict=True), key=lambda item: item[0])
        xs_sorted = [float(pair[0]) for pair in pairs]
        ys_sorted = [float(pair[1]) for pair in pairs]
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


def build_parquet_preview_figure(
    traces: list[dict[str, Any]],
    *,
    x_range: list[float] | None = None,
    x_is_datetime: bool,
    height: int = 360,
    legend_only_interaction: bool = False,
):
    """Build a read-only Plotly figure from preview traces."""
    import plotly.graph_objects as go

    figure = go.Figure()

    for index, trace in enumerate(traces):
        x_values = trace.get("x") or []
        if x_is_datetime:
            x_values = [_epoch_microseconds_to_utc_naive(value) for value in x_values]
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=trace.get("y") or [],
                mode="lines",
                name=str(trace.get("name") or f"curve-{index + 1}"),
                line={"width": 1.5 if index == 0 else 1.2},
                hoverinfo="skip" if legend_only_interaction else None,
                hovertemplate=None if legend_only_interaction else None,
            )
        )

    figure.update_layout(
        autosize=True,
        height=height,
        margin={"l": 20, "r": 20, "t": 24, "b": 20},
        showlegend=True,
        dragmode=False,
        hovermode=False if legend_only_interaction else "x unified",
        clickmode="none" if legend_only_interaction else "event",
        xaxis={"title": "Time (UTC)" if x_is_datetime else "X", "fixedrange": legend_only_interaction},
        yaxis={"title": "Value", "autorange": True, "fixedrange": legend_only_interaction},
    )
    if x_range and len(x_range) == 2:
        if x_is_datetime:
            figure.update_xaxes(range=[_epoch_microseconds_to_utc_naive(x_range[0]), _epoch_microseconds_to_utc_naive(x_range[1])])
        else:
            figure.update_xaxes(range=x_range)
    return figure


def _epoch_microseconds_to_utc_naive(epoch_microseconds: float) -> datetime:
    """Convert epoch microseconds to a naive UTC datetime for Plotly display."""
    return datetime.fromtimestamp(float(epoch_microseconds) / 1_000_000.0, tz=UTC).replace(tzinfo=None)


__all__ = ["build_parquet_preview_figure", "build_parquet_preview_from_files"]
