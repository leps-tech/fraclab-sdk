"""Streaming downsample helpers for high-frequency core datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite
from typing import Any

import numpy as np
import pandas as pd

_REQUIRED_BASE_COLUMNS = {"ts_us", "value"}
_SYSTEM_COLUMNS = {"ts_us", "value", "period_us"}


@dataclass
class DownsamplePlan:
    """Resolved delivery plan for a dataset."""

    delivery_hz: int
    raw_period_us: int | None
    output_columns: list[str]
    passthrough_columns: list[str]
    raw_read_columns: list[str]
    synthetic_columns: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_hz(self) -> float | None:
        if self.raw_period_us is None:
            return None
        return 1_000_000.0 / self.raw_period_us

    @property
    def bucket_width_us(self) -> int:
        return int(round(1_000_000 / self.delivery_hz))

    def raw_batch_rows_for_output_rows(self, output_rows: int) -> int:
        if self.raw_period_us is None:
            return max(1, int(output_rows))
        estimated_rows = ceil(int(output_rows) * self.bucket_width_us / self.raw_period_us) + 1
        return max(1, estimated_rows)


class CoreFrameDownsampler:
    """Stateful streaming downsampler for detected-rate core frame rows."""

    def __init__(self, plan: DownsamplePlan) -> None:
        self._plan = plan
        self._anchor_ts_us: int | None = None
        self._current_bucket_id: int | None = None
        self._current_value_sum = 0.0
        self._current_value_count = 0
        self._current_passthrough: dict[str, Any] = {}
        self._last_ts_us: int | None = None

    def consume(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Consume raw rows and emit completed downsampled rows."""
        if frame.empty:
            return self._empty_output_frame()

        self._validate_raw_frame(frame)
        rows: list[dict[str, Any]] = []

        passthrough_values = {
            column: frame[column].tolist() if column in frame.columns else [self._plan.synthetic_columns[column]] * len(frame)
            for column in self._plan.passthrough_columns
        }
        ts_values = pd.to_numeric(frame["ts_us"], errors="raise").to_numpy(dtype=np.int64)
        value_values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float)

        for row_index, ts_us in enumerate(ts_values):
            if self._anchor_ts_us is None:
                self._anchor_ts_us = int(ts_us)
            bucket_id = int((int(ts_us) - self._anchor_ts_us) // self._plan.bucket_width_us)

            if self._current_bucket_id is None:
                self._start_bucket(bucket_id, row_index, passthrough_values)
            elif bucket_id != self._current_bucket_id:
                rows.append(self._flush_bucket())
                self._start_bucket(bucket_id, row_index, passthrough_values)

            self._merge_passthrough(row_index, passthrough_values)
            value = float(value_values[row_index])
            if isfinite(value):
                self._current_value_sum += value
                self._current_value_count += 1
            self._last_ts_us = int(ts_us)

        return self._build_output_frame(rows)

    def finalize(self) -> pd.DataFrame:
        """Flush the final incomplete bucket, if any."""
        if self._current_bucket_id is None:
            return self._empty_output_frame()
        return self._build_output_frame([self._flush_bucket()])

    def _validate_raw_frame(self, frame: pd.DataFrame) -> None:
        missing = sorted(column for column in _REQUIRED_BASE_COLUMNS if column not in frame.columns)
        if missing:
            raise ValueError(f"core delivery downsample requires columns {missing}")

        detected_period_us = infer_raw_period_us(frame)
        if detected_period_us is not None and self._plan.bucket_width_us < detected_period_us:
            detected_hz = 1_000_000.0 / detected_period_us
            raise ValueError(
                "deliveryHz must be <= detected raw frequency "
                f"{detected_hz:.6f}Hz (period_us={detected_period_us}), "
                f"got {self._plan.delivery_hz}"
            )

        ts_values = pd.to_numeric(frame["ts_us"], errors="raise").to_numpy(dtype=np.int64)
        if ts_values.size == 0:
            return
        if self._last_ts_us is not None and int(ts_values[0]) < self._last_ts_us:
            raise ValueError("core delivery downsample requires non-decreasing ts_us order")
        if ts_values.size > 1 and np.any(np.diff(ts_values) < 0):
            raise ValueError("core delivery downsample requires non-decreasing ts_us order")

    def _start_bucket(
        self,
        bucket_id: int,
        row_index: int,
        passthrough_values: dict[str, list[Any]],
    ) -> None:
        self._current_bucket_id = bucket_id
        self._current_value_sum = 0.0
        self._current_value_count = 0
        self._current_passthrough = {
            column: values[row_index]
            for column, values in passthrough_values.items()
        }

    def _merge_passthrough(
        self,
        row_index: int,
        passthrough_values: dict[str, list[Any]],
    ) -> None:
        for column, values in passthrough_values.items():
            current = self._current_passthrough.get(column)
            next_value = values[row_index]
            if pd.isna(current) and not pd.isna(next_value):
                self._current_passthrough[column] = next_value
                continue
            if pd.isna(next_value) or current == next_value:
                continue
            raise ValueError(
                f"core delivery downsample cannot aggregate varying column '{column}'"
            )

    def _flush_bucket(self) -> dict[str, Any]:
        assert self._anchor_ts_us is not None
        assert self._current_bucket_id is not None
        row: dict[str, Any] = {
            "ts_us": int(self._anchor_ts_us + self._current_bucket_id * self._plan.bucket_width_us),
            "value": (
                self._current_value_sum / self._current_value_count
                if self._current_value_count > 0
                else np.nan
            ),
            "period_us": self._plan.bucket_width_us,
        }
        row.update(self._current_passthrough)
        self._current_bucket_id = None
        self._current_passthrough = {}
        return row

    def _build_output_frame(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return self._empty_output_frame()
        frame = pd.DataFrame(rows)
        return frame.loc[:, self._plan.output_columns]

    def _empty_output_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self._plan.output_columns)


def build_downsample_plan(
    available_columns: list[str],
    requested_columns: list[str] | None,
    resolution_params: dict[str, Any] | None,
    delivery_hz: int,
    raw_period_us: int | None,
) -> DownsamplePlan:
    """Build a downsample plan for a single frame item."""
    if delivery_hz <= 0:
        raise ValueError("deliveryHz must be positive")
    bucket_width_us = int(round(1_000_000 / delivery_hz))
    if bucket_width_us <= 0:
        raise ValueError("deliveryHz produced an invalid output period")
    if raw_period_us is not None and raw_period_us <= 0:
        raise ValueError("raw_period_us must be positive when provided")
    if raw_period_us is not None and bucket_width_us < raw_period_us:
        raw_hz = 1_000_000.0 / raw_period_us
        raise ValueError(
            "deliveryHz must be <= detected raw frequency "
            f"{raw_hz:.6f}Hz (period_us={raw_period_us}), got {delivery_hz}"
        )

    available = list(dict.fromkeys(str(column) for column in available_columns))
    requested = (
        list(dict.fromkeys(str(column) for column in requested_columns))
        if requested_columns is not None
        else list(available)
    )
    resolution_values = {
        str(key): value
        for key, value in (resolution_params or {}).items()
        if value is not None
    }

    missing = sorted(
        column
        for column in _REQUIRED_BASE_COLUMNS
        if column not in available and column not in resolution_values
    )
    if missing:
        raise ValueError(
            f"core delivery downsample requires source columns {missing}"
        )

    output_columns = list(dict.fromkeys(requested))
    for column in ("ts_us", "value"):
        if column not in output_columns:
            output_columns.append(column)

    if ("period_us" in available or "period_us" in output_columns) and "period_us" not in output_columns:
        output_columns.append("period_us")

    synthetic_columns = {
        column: resolution_values[column]
        for column in output_columns
        if column not in available and column in resolution_values
    }
    unresolved = sorted(
        column
        for column in output_columns
        if column not in available and column not in synthetic_columns and column not in _SYSTEM_COLUMNS
    )
    if unresolved:
        raise ValueError(
            f"core delivery downsample cannot provide columns {unresolved}"
        )

    raw_read_columns = list(
        dict.fromkeys(
            [
                *[column for column in output_columns if column in available],
                *[column for column in ("ts_us", "value", "period_us") if column in available],
            ]
        )
    )
    passthrough_columns = [
        column
        for column in output_columns
        if column not in _SYSTEM_COLUMNS
    ]
    if "period_us" not in available and "period_us" in raw_read_columns:
        raw_read_columns.remove("period_us")

    return DownsamplePlan(
        delivery_hz=delivery_hz,
        raw_period_us=raw_period_us,
        output_columns=output_columns,
        passthrough_columns=passthrough_columns,
        raw_read_columns=raw_read_columns,
        synthetic_columns=synthetic_columns,
    )


def infer_raw_period_us(frame: pd.DataFrame) -> int | None:
    """Infer raw sample period from period_us or ts_us deltas."""
    if "period_us" in frame.columns:
        period_values = pd.to_numeric(frame["period_us"], errors="coerce").to_numpy(dtype=float)
        valid_periods = period_values[np.isfinite(period_values) & (period_values > 0)]
        if valid_periods.size:
            return int(np.median(valid_periods))

    if "ts_us" not in frame.columns:
        return None

    ts_values = pd.to_numeric(frame["ts_us"], errors="coerce").dropna().to_numpy(dtype=np.int64)
    if ts_values.size < 2:
        return None

    diffs = np.diff(ts_values)
    valid_diffs = diffs[diffs > 0]
    if not valid_diffs.size:
        return None
    return int(np.median(valid_diffs))
