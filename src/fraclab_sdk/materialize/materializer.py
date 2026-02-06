"""Materializer implementation."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fraclab_sdk.errors import MaterializeError
from fraclab_sdk.materialize.fsops import copy_directory_smart, extract_ndjson_lines
from fraclab_sdk.materialize.hash import compute_sha256
from fraclab_sdk.models import DRS, DataSpec
from fraclab_sdk.snapshot.loader import SnapshotHandle


@dataclass
class MaterializeResult:
    """Result of materialization."""

    input_dir: Path
    ds_sha256: str
    drs_sha256: str
    copy_stats: dict[str, dict[str, int]]  # dataset_key -> {hardlink, symlink, copy}


class Materializer:
    """Materializes run input from snapshot and selection."""

    def materialize(
        self,
        run_dir: Path,
        snapshot: SnapshotHandle,
        run_ds: DataSpec,
        drs: DRS,
        params: dict[str, Any],
        run_context: dict[str, Any],
    ) -> MaterializeResult:
        """Materialize run input directory.

        Creates runs/<run_id>/input/ with:
        - manifest.json (with sha256 hashes)
        - ds.json (run subset, re-indexed)
        - drs.json (from algorithm)
        - params.json
        - run_context.json
        - data/ (layout-aware materialization)

        Args:
            run_dir: The run directory (will create input/ subdirectory).
            snapshot: Source snapshot handle.
            run_ds: Run DataSpec (re-indexed from selection).
            drs: Algorithm's DRS.
            params: Algorithm parameters.
            run_context: Run context metadata.

        Returns:
            MaterializeResult with paths and hashes.

        Raises:
            MaterializeError: If materialization fails.
        """
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Write ds.json and compute hash
        ds_bytes = self._write_json(input_dir / "ds.json", run_ds.model_dump())
        ds_sha256 = compute_sha256(ds_bytes)

        # Write drs.json and compute hash
        drs_bytes = self._write_json(input_dir / "drs.json", drs.model_dump())
        drs_sha256 = compute_sha256(drs_bytes)

        # Write params.json
        self._write_json(input_dir / "params.json", params)

        # Write run_context.json
        self._write_json(input_dir / "run_context.json", run_context)

        # Materialize data
        copy_stats = self._materialize_data(input_dir, snapshot, run_ds)

        # Write manifest.json (last, with computed hashes)
        # Build datasets entry for manifest
        datasets_manifest: dict[str, dict] = {}
        for dataset in run_ds.datasets:
            datasets_manifest[dataset.datasetKey] = {
                "layout": dataset.layout,
                "count": len(dataset.items),
            }

        manifest = {
            "bundleVersion": "1.0.0",
            "createdAtUs": int(time.time() * 1_000_000),
            "specFiles": {
                "dsPath": "ds.json",
                "drsPath": "drs.json",
                "dsSha256": ds_sha256,
                "drsSha256": drs_sha256,
            },
            "dataRoot": "data",
            "datasets": datasets_manifest,
        }
        self._write_json(input_dir / "manifest.json", manifest)

        return MaterializeResult(
            input_dir=input_dir,
            ds_sha256=ds_sha256,
            drs_sha256=drs_sha256,
            copy_stats=copy_stats,
        )

    def _write_json(self, path: Path, data: Any) -> bytes:
        """Write JSON file and return bytes written.

        Args:
            path: File path to write.
            data: Data to serialize as JSON.

        Returns:
            Bytes that were written.
        """
        content = json.dumps(data, indent=2, ensure_ascii=False)
        content_bytes = content.encode("utf-8")
        path.write_bytes(content_bytes)
        return content_bytes

    def _materialize_data(
        self,
        input_dir: Path,
        snapshot: SnapshotHandle,
        run_ds: DataSpec,
    ) -> dict[str, dict[str, int]]:
        """Materialize data directory with layout-aware copying.

        Args:
            input_dir: The input directory.
            snapshot: Source snapshot.
            run_ds: Run DataSpec with re-indexed items.

        Returns:
            Copy stats per dataset.
        """
        data_dir = input_dir / "data"
        copy_stats: dict[str, dict[str, int]] = {}

        for dataset in run_ds.datasets:
            dataset_key = dataset.datasetKey
            layout = dataset.layout

            if layout == "frame_parquet_item_dirs":
                stats = self._materialize_parquet(
                    data_dir, snapshot, dataset_key, dataset.items
                )
                copy_stats[dataset_key] = stats
            elif layout == "object_ndjson_lines":
                self._materialize_ndjson(
                    data_dir, snapshot, dataset_key, dataset.items
                )
                copy_stats[dataset_key] = {"ndjson_lines": len(dataset.items)}
            else:
                raise MaterializeError(f"Unknown layout: {layout}")

        return copy_stats

    def _materialize_parquet(
        self,
        data_dir: Path,
        snapshot: SnapshotHandle,
        dataset_key: str,
        items: list,
    ) -> dict[str, int]:
        """Materialize parquet item directories with re-indexing.

        Source: snapshot/data/<datasetKey>/parquet/item-<snapshot_index:05d>/
        Target: run/input/data/<datasetKey>/parquet/item-<run_index:05d>/

        Args:
            data_dir: Target data directory.
            snapshot: Source snapshot.
            dataset_key: Dataset key.
            items: List of DataSpecItem (with sourceItemIndex).

        Returns:
            Copy stats {hardlink, symlink, copy}.
        """
        total_stats = {"hardlink": 0, "symlink": 0, "copy": 0}

        for run_index, item in enumerate(items):
            snapshot_index = item.sourceItemIndex
            if snapshot_index is None:
                raise MaterializeError(
                    f"Item at run index {run_index} missing sourceItemIndex"
                )

            src_dir = snapshot.get_item_dir(dataset_key, snapshot_index)
            dst_dir = data_dir / dataset_key / "parquet" / f"item-{run_index:05d}"

            if not src_dir.exists():
                raise MaterializeError(f"Source item directory not found: {src_dir}")

            stats = copy_directory_smart(src_dir, dst_dir)
            for key in total_stats:
                total_stats[key] += stats[key]

        return total_stats

    def _materialize_ndjson(
        self,
        data_dir: Path,
        snapshot: SnapshotHandle,
        dataset_key: str,
        items: list,
    ) -> None:
        """Materialize ndjson by extracting selected lines.

        Extracts lines by snapshot index and writes contiguously.
        Run item 0 = line 0, run item 1 = line 1, etc.

        Args:
            data_dir: Target data directory.
            snapshot: Source snapshot.
            dataset_key: Dataset key.
            items: List of DataSpecItem (with sourceItemIndex).
        """
        # Get snapshot indices in order
        snapshot_indices = []
        for run_index, item in enumerate(items):
            snapshot_index = item.sourceItemIndex
            if snapshot_index is None:
                raise MaterializeError(
                    f"Item at run index {run_index} missing sourceItemIndex"
                )
            snapshot_indices.append(snapshot_index)

        src_path = snapshot.get_ndjson_path(dataset_key)
        dst_path = data_dir / dataset_key / "object.ndjson"

        if not src_path.exists():
            raise MaterializeError(f"Source ndjson not found: {src_path}")

        extract_ndjson_lines(src_path, dst_path, snapshot_indices)
