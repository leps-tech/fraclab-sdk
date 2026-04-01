"""Data client for algorithm runtime."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd

from fraclab_sdk.models import DataSpec
from fraclab_sdk.runtime.core_downsample import (
    CoreFrameDownsampler,
    DownsamplePlan,
    build_downsample_plan,
    infer_raw_period_us,
)
from fraclab_sdk.runtime.data_requirements import (
    load_raw_drs,
    resolve_dataset_delivery_hz,
)

_DEFAULT_FRAME_CHUNK_ROWS = 50_000
_FRAME_CHUNK_PREFETCH_DEPTH = 1
_RAW_PERIOD_DETECTION_ROWS = 4_096
_STREAM_END = object()


def _load_pyarrow() -> Any:
    """Import pyarrow on demand for frame reads."""
    try:
        return import_module("pyarrow")
    except ImportError as exc:  # pragma: no cover - depends on optional install state
        raise RuntimeError(
            "pyarrow is required for DataClient frame reads; "
            "install fraclab-sdk with pyarrow available."
        ) from exc


def _load_pyarrow_parquet() -> Any:
    """Import pyarrow.parquet on demand for frame reads."""
    try:
        return import_module("pyarrow.parquet")
    except ImportError as exc:  # pragma: no cover - depends on optional install state
        raise RuntimeError(
            "pyarrow is required for DataClient frame reads; "
            "install fraclab-sdk with pyarrow available."
        ) from exc


def _next_or_stream_end(iterator: Iterator[Any]) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_END


def _iter_with_single_item_prefetch(iterator: Iterator[Any]) -> Iterator[Any]:
    """Yield iterator items while prefetching one item ahead in a worker thread."""
    with ThreadPoolExecutor(max_workers=_FRAME_CHUNK_PREFETCH_DEPTH) as executor:
        future = executor.submit(_next_or_stream_end, iterator)
        while True:
            item = future.result()
            if item is _STREAM_END:
                return
            future = executor.submit(_next_or_stream_end, iterator)
            yield item


@dataclass(frozen=True)
class _FrameReadRequest:
    dataset_key: str
    item_index: int
    output_rows: int
    output_columns: list[str]
    raw_read_columns: list[str]
    synthetic_columns: dict[str, Any]
    downsample_plan: DownsamplePlan | None


class DataClient:
    """Client for reading input data during algorithm execution."""

    def __init__(self, input_dir: Path) -> None:
        """Initialize data client.

        Args:
            input_dir: The run input directory containing ds.json and data/.
        """
        self._input_dir = input_dir
        self._dataspec: DataSpec | None = None
        self._raw_drs: dict[str, Any] | None = None

    @property
    def dataspec(self) -> DataSpec:
        """Get the data specification."""
        if self._dataspec is None:
            ds_path = self._input_dir / "ds.json"
            self._dataspec = DataSpec.model_validate_json(ds_path.read_text(encoding="utf-8"))
        return self._dataspec

    @property
    def raw_drs(self) -> dict[str, Any]:
        """Get the raw input drs.json payload."""
        if self._raw_drs is None:
            self._raw_drs = load_raw_drs(self._input_dir)
        return self._raw_drs

    def get_dataset_keys(self) -> list[str]:
        """Get list of available dataset keys."""
        return self.dataspec.get_dataset_keys()

    def get_item_count(self, dataset_key: str) -> int:
        """Get number of items in a dataset."""
        return len(self._get_dataset(dataset_key).items)

    def get_layout(self, dataset_key: str) -> str | None:
        """Get the layout type for a dataset."""
        return self._get_dataset(dataset_key).layout

    def get_frame_columns(self, dataset_key: str, item_index: int) -> list[str]:
        """Get logical frame columns exposed by DataClient for an item."""
        self._require_frame_layout(dataset_key)
        schema_columns = self._get_raw_frame_columns(dataset_key, item_index)
        seen = set(schema_columns)
        item = self._get_dataset_item(dataset_key, item_index)
        for key in (item.resolutionParams or {}):
            column = str(key)
            if column not in seen:
                seen.add(column)
                schema_columns.append(column)
        return schema_columns

    def read_object(self, dataset_key: str, item_index: int) -> dict:
        """Read an object from ndjson dataset."""
        layout = self.get_layout(dataset_key)
        if layout != "object_ndjson_lines":
            raise ValueError(
                f"Cannot read object from layout '{layout}', "
                f"expected 'object_ndjson_lines'"
            )

        ndjson_path = self._input_dir / "data" / dataset_key / "object.ndjson"
        with ndjson_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == item_index:
                    return json.loads(line)

        raise IndexError(f"Item index {item_index} not found")

    def read_frame(
        self,
        dataset_key: str,
        item_index: int,
        *,
        columns: Sequence[str] | None = None,
        chunk_rows: int | None = None,
    ) -> pd.DataFrame:
        """Read a frame item using the resolved DataClient delivery semantics."""
        request = self._build_frame_request(
            dataset_key,
            item_index,
            columns=columns,
            output_rows=chunk_rows,
        )
        frames = list(self._iter_frame_frames(request))
        if not frames:
            return pd.DataFrame(columns=request.output_columns)
        return pd.concat(frames, ignore_index=True)

    def iter_frame_chunks(
        self,
        dataset_key: str,
        item_index: int,
        *,
        columns: Sequence[str] | None = None,
        chunk_rows: int | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Iterate a frame item as DataFrame chunks after DataClient processing."""
        request = self._build_frame_request(
            dataset_key,
            item_index,
            columns=columns,
            output_rows=chunk_rows,
        )
        chunk_iterator = self._iter_frame_frames(request)
        yield from _iter_with_single_item_prefetch(chunk_iterator)

    def iter_dataset_frame_chunks(
        self,
        dataset_key: str,
        *,
        columns: Sequence[str] | None = None,
        chunk_rows: int | None = None,
    ) -> Iterator[tuple[int, pd.DataFrame]]:
        """Iterate all frame items in a dataset as processed DataFrame chunks."""
        output_rows = self._resolve_chunk_rows(chunk_rows)
        for item_index in range(self.get_item_count(dataset_key)):
            request = self._build_frame_request(
                dataset_key,
                item_index,
                columns=columns,
                output_rows=output_rows,
            )
            for chunk in self._iter_frame_frames(request):
                yield item_index, chunk

    def iter_frame_batches(
        self,
        dataset_key: str,
        item_index: int,
        *,
        columns: Sequence[str] | None = None,
        batch_rows: int | None = None,
    ) -> Iterator[Any]:
        """Iterate a frame item as Arrow record batches after DataClient processing."""
        request = self._build_frame_request(
            dataset_key,
            item_index,
            columns=columns,
            output_rows=batch_rows,
        )
        pyarrow = _load_pyarrow()
        for frame in self._iter_frame_frames(request):
            if frame.empty:
                continue
            table = pyarrow.Table.from_pandas(frame, preserve_index=False)
            yield from table.to_batches(max_chunksize=request.output_rows)

    def get_parquet_dir(self, dataset_key: str, item_index: int) -> Path:
        """Raw parquet path access is no longer supported."""
        self._require_frame_layout(dataset_key)
        raise RuntimeError(
            "Raw parquet directory access is no longer supported. "
            "Use DataClient.read_frame(), iter_frame_chunks(), or iter_frame_batches()."
        )

    def get_parquet_files(self, dataset_key: str, item_index: int) -> list[Path]:
        """Raw parquet path access is no longer supported."""
        self._require_frame_layout(dataset_key)
        raise RuntimeError(
            "Raw parquet file access is no longer supported. "
            "Use DataClient.read_frame(), iter_frame_chunks(), or iter_frame_batches()."
        )

    def iterate_objects(self, dataset_key: str):
        """Iterate over all objects in an ndjson dataset."""
        layout = self.get_layout(dataset_key)
        if layout != "object_ndjson_lines":
            raise ValueError(
                f"Cannot iterate objects from layout '{layout}', "
                f"expected 'object_ndjson_lines'"
            )

        ndjson_path = self._input_dir / "data" / dataset_key / "object.ndjson"
        with ndjson_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                yield i, json.loads(line)

    def _get_dataset(self, dataset_key: str):
        dataset = self.dataspec.get_dataset(dataset_key)
        if dataset is None:
            raise KeyError(f"Dataset not found: {dataset_key}")
        return dataset

    def _get_dataset_item(self, dataset_key: str, item_index: int):
        dataset = self._get_dataset(dataset_key)
        if item_index < 0 or item_index >= len(dataset.items):
            raise IndexError(f"Item index {item_index} out of range for dataset {dataset_key}")
        return dataset.items[item_index]

    def _resolve_chunk_rows(self, chunk_rows: int | None) -> int:
        if chunk_rows is None:
            return _DEFAULT_FRAME_CHUNK_ROWS
        if int(chunk_rows) <= 0:
            raise ValueError("chunk_rows must be a positive integer")
        return int(chunk_rows)

    def _require_frame_layout(self, dataset_key: str) -> None:
        layout = self.get_layout(dataset_key)
        if layout != "frame_parquet_item_dirs":
            raise ValueError(
                f"Cannot read frame from layout '{layout}', "
                f"expected 'frame_parquet_item_dirs'"
            )

    def _iter_parquet_files(self, dataset_key: str, item_index: int) -> Iterator[Path]:
        item_dir = self._input_dir / "data" / dataset_key / "parquet" / f"item-{item_index:05d}"
        if not item_dir.exists():
            return iter(())
        return iter(sorted(item_dir.rglob("*.parquet")))

    def _get_raw_frame_columns(self, dataset_key: str, item_index: int) -> list[str]:
        self._require_frame_layout(dataset_key)
        parquet = _load_pyarrow_parquet()
        schema_columns: list[str] = []
        seen: set[str] = set()
        for parquet_file in self._iter_parquet_files(dataset_key, item_index):
            for name in parquet.ParquetFile(parquet_file).schema.names:
                column = str(name)
                if column in seen:
                    continue
                seen.add(column)
                schema_columns.append(column)
        return schema_columns

    def _build_frame_request(
        self,
        dataset_key: str,
        item_index: int,
        *,
        columns: Sequence[str] | None,
        output_rows: int | None,
    ) -> _FrameReadRequest:
        self._require_frame_layout(dataset_key)
        resolved_output_rows = self._resolve_chunk_rows(output_rows)
        raw_available_columns = self._get_raw_frame_columns(dataset_key, item_index)
        item = self._get_dataset_item(dataset_key, item_index)
        resolution_params = {
            str(key): value
            for key, value in (item.resolutionParams or {}).items()
            if value is not None
        }
        default_output_columns = list(raw_available_columns)
        for column in resolution_params:
            if column not in default_output_columns:
                default_output_columns.append(column)
        delivery_hz = resolve_dataset_delivery_hz(self.raw_drs, dataset_key)

        if delivery_hz is None:
            return self._build_raw_frame_request(
                dataset_key,
                item_index,
                columns=default_output_columns if columns is None else columns,
                output_rows=resolved_output_rows,
                raw_available_columns=raw_available_columns,
                resolution_params=resolution_params,
            )

        plan = build_downsample_plan(
            available_columns=raw_available_columns,
            requested_columns=(
                [str(column) for column in columns]
                if columns is not None
                else default_output_columns
            ),
            resolution_params=resolution_params,
            delivery_hz=delivery_hz,
            raw_period_us=self._infer_raw_frame_period_us(
                dataset_key,
                item_index,
                raw_available_columns,
            ),
        )
        return _FrameReadRequest(
            dataset_key=dataset_key,
            item_index=item_index,
            output_rows=resolved_output_rows,
            output_columns=plan.output_columns,
            raw_read_columns=plan.raw_read_columns,
            synthetic_columns=plan.synthetic_columns,
            downsample_plan=plan,
        )

    def _build_raw_frame_request(
        self,
        dataset_key: str,
        item_index: int,
        *,
        columns: Sequence[str] | None,
        output_rows: int,
        raw_available_columns: list[str],
        resolution_params: dict[str, Any],
    ) -> _FrameReadRequest:
        requested = (
            list(dict.fromkeys(str(column) for column in columns))
            if columns is not None
            else list(raw_available_columns)
        )
        synthetic_columns = {
            column: resolution_params[column]
            for column in requested
            if column not in raw_available_columns and column in resolution_params
        }
        unresolved = sorted(
            column
            for column in requested
            if column not in raw_available_columns and column not in synthetic_columns
        )
        if unresolved:
            raise ValueError(f"frame dataset cannot provide columns {unresolved}")
        raw_read_columns = [column for column in requested if column in raw_available_columns]
        return _FrameReadRequest(
            dataset_key=dataset_key,
            item_index=item_index,
            output_rows=output_rows,
            output_columns=requested,
            raw_read_columns=raw_read_columns,
            synthetic_columns=synthetic_columns,
            downsample_plan=None,
        )

    def _infer_raw_frame_period_us(
        self,
        dataset_key: str,
        item_index: int,
        raw_available_columns: list[str],
    ) -> int | None:
        detection_columns = [
            column for column in ("period_us", "ts_us") if column in raw_available_columns
        ]
        if not detection_columns:
            return None

        for batch in self._iter_raw_frame_batches(
            dataset_key,
            item_index,
            columns=detection_columns,
            batch_rows=_RAW_PERIOD_DETECTION_ROWS,
        ):
            detected_period_us = infer_raw_period_us(batch.to_pandas())
            if detected_period_us is not None:
                return detected_period_us
        return None

    def _iter_frame_frames(self, request: _FrameReadRequest) -> Iterator[pd.DataFrame]:
        if request.downsample_plan is None:
            yield from self._iter_raw_projected_frames(request)
            return
        yield from self._iter_downsampled_frames(request)

    def _iter_raw_projected_frames(self, request: _FrameReadRequest) -> Iterator[pd.DataFrame]:
        for batch in self._iter_raw_frame_batches(
            request.dataset_key,
            request.item_index,
            columns=request.raw_read_columns,
            batch_rows=request.output_rows,
        ):
            frame = batch.to_pandas()
            yield self._project_frame(frame, request.output_columns, request.synthetic_columns)

    def _iter_downsampled_frames(self, request: _FrameReadRequest) -> Iterator[pd.DataFrame]:
        assert request.downsample_plan is not None
        downsampler = CoreFrameDownsampler(request.downsample_plan)
        raw_batch_rows = request.downsample_plan.raw_batch_rows_for_output_rows(request.output_rows)
        for batch in self._iter_raw_frame_batches(
            request.dataset_key,
            request.item_index,
            columns=request.raw_read_columns,
            batch_rows=raw_batch_rows,
        ):
            frame = batch.to_pandas()
            output = downsampler.consume(frame)
            yield from self._yield_rechunked_frames(output, request.output_rows)
        yield from self._yield_rechunked_frames(downsampler.finalize(), request.output_rows)

    def _iter_raw_frame_batches(
        self,
        dataset_key: str,
        item_index: int,
        *,
        columns: list[str],
        batch_rows: int,
    ) -> Iterator[Any]:
        parquet = _load_pyarrow_parquet()
        selected_columns = columns or None
        for parquet_file in self._iter_parquet_files(dataset_key, item_index):
            parquet_file_reader = parquet.ParquetFile(parquet_file)
            yield from parquet_file_reader.iter_batches(
                columns=selected_columns,
                batch_size=batch_rows,
            )

    def _project_frame(
        self,
        frame: pd.DataFrame,
        output_columns: list[str],
        synthetic_columns: dict[str, Any],
    ) -> pd.DataFrame:
        for column, value in synthetic_columns.items():
            frame[column] = value
        return frame.loc[:, output_columns]

    def _yield_rechunked_frames(
        self,
        frame: pd.DataFrame,
        output_rows: int,
    ) -> Iterator[pd.DataFrame]:
        if frame.empty:
            return
        for start in range(0, len(frame), output_rows):
            yield frame.iloc[start : start + output_rows].reset_index(drop=True)
