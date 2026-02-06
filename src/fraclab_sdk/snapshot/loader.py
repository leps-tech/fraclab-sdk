"""Snapshot loader implementation."""

import json
from pathlib import Path

from fraclab_sdk.errors import SnapshotError
from fraclab_sdk.models import DRS, BundleManifest, DataSpec, DataSpecItem


class SnapshotHandle:
    """Handle for accessing snapshot contents."""

    def __init__(self, snapshot_dir: Path) -> None:
        """Initialize snapshot handle.

        Args:
            snapshot_dir: Path to the snapshot directory.
        """
        self._dir = snapshot_dir
        self._manifest: BundleManifest | None = None
        self._dataspec: DataSpec | None = None
        self._drs: DRS | None = None

    @property
    def directory(self) -> Path:
        """Get snapshot directory path."""
        return self._dir

    @property
    def manifest(self) -> BundleManifest:
        """Get bundle manifest."""
        if self._manifest is None:
            manifest_path = self._dir / "manifest.json"
            if not manifest_path.exists():
                raise SnapshotError(f"Manifest not found: {manifest_path}")
            self._manifest = BundleManifest.model_validate_json(manifest_path.read_text())
        return self._manifest

    @property
    def dataspec(self) -> DataSpec:
        """Get data specification."""
        if self._dataspec is None:
            ds_path = self._dir / self.manifest.specFiles.dsPath
            if not ds_path.exists():
                raise SnapshotError(f"DataSpec not found: {ds_path}")
            self._dataspec = DataSpec.model_validate_json(ds_path.read_text())
        return self._dataspec

    @property
    def drs(self) -> DRS:
        """Get data requirement specification."""
        if self._drs is None:
            drs_path = self._dir / self.manifest.specFiles.drsPath
            if not drs_path.exists():
                raise SnapshotError(f"DRS not found: {drs_path}")
            self._drs = DRS.model_validate_json(drs_path.read_text())
        return self._drs

    @property
    def data_root(self) -> Path:
        """Get data root directory path."""
        return self._dir / self.manifest.dataRoot

    def get_raw_ds_bytes(self) -> bytes:
        """Get raw bytes of ds.json for hash verification."""
        return (self._dir / self.manifest.specFiles.dsPath).read_bytes()

    def get_raw_drs_bytes(self) -> bytes:
        """Get raw bytes of drs.json for hash verification."""
        return (self._dir / self.manifest.specFiles.drsPath).read_bytes()

    def get_datasets(self) -> list[dict]:
        """Get list of datasets with metadata.

        Returns:
            List of dicts with dataset_key, resource_type, layout, item_count.
        """
        return [
            {
                "dataset_key": ds.datasetKey,
                "resource_type": ds.resourceType,
                "layout": ds.layout,
                "item_count": len(ds.items),
            }
            for ds in self.dataspec.datasets
        ]

    def get_items(self, dataset_key: str) -> list[tuple[int, DataSpecItem]]:
        """Get items for a dataset as (index, item) tuples.

        Args:
            dataset_key: The dataset key.

        Returns:
            List of (index, DataSpecItem) tuples.

        Raises:
            SnapshotError: If dataset not found.
        """
        dataset = self.dataspec.get_dataset(dataset_key)
        if dataset is None:
            raise SnapshotError(f"Dataset not found: {dataset_key}")
        return list(enumerate(dataset.items))

    def get_layout(self, dataset_key: str) -> str | None:
        """Get the layout type for a dataset."""
        dataset = self.dataspec.get_dataset(dataset_key)
        if dataset is None:
            raise SnapshotError(f"Dataset not found: {dataset_key}")
        return dataset.layout

    def read_object_line(self, dataset_key: str, item_index: int) -> dict:
        """Read a single line from object.ndjson by index.

        Args:
            dataset_key: The dataset key.
            item_index: The item index (0-based).

        Returns:
            Parsed JSON object from the line.

        Raises:
            SnapshotError: If dataset not found or invalid layout.
        """
        layout = self.get_layout(dataset_key)
        if layout != "object_ndjson_lines":
            raise SnapshotError(
                f"Cannot read object line from layout '{layout}', expected 'object_ndjson_lines'"
            )

        ndjson_path = self.data_root / dataset_key / "object.ndjson"
        if not ndjson_path.exists():
            raise SnapshotError(f"object.ndjson not found: {ndjson_path}")

        # Check for index file for faster random access
        idx_path = self.data_root / dataset_key / "object.idx.u64"
        if idx_path.exists():
            return self._read_object_line_indexed(ndjson_path, idx_path, item_index)

        # Fallback to linear scan
        return self._read_object_line_linear(ndjson_path, item_index)

    def _read_object_line_indexed(
        self, ndjson_path: Path, idx_path: Path, item_index: int
    ) -> dict:
        """Read object line using index file for random access."""
        import struct

        idx_data = idx_path.read_bytes()
        num_entries = len(idx_data) // 8

        if item_index < 0 or item_index >= num_entries:
            raise SnapshotError(f"Item index {item_index} out of range [0, {num_entries})")

        offset = struct.unpack("<Q", idx_data[item_index * 8 : (item_index + 1) * 8])[0]

        with ndjson_path.open("rb") as f:
            f.seek(offset)
            line = f.readline()
            return json.loads(line)

    def _read_object_line_linear(self, ndjson_path: Path, item_index: int) -> dict:
        """Read object line via linear scan."""
        with ndjson_path.open() as f:
            for i, line in enumerate(f):
                if i == item_index:
                    return json.loads(line)
        raise SnapshotError(f"Item index {item_index} not found in {ndjson_path}")

    def read_frame_parts(self, dataset_key: str, item_index: int) -> list[Path]:
        """Get paths to parquet files for an item.

        Args:
            dataset_key: The dataset key.
            item_index: The item index (0-based).

        Returns:
            List of paths to parquet files in the item directory.

        Raises:
            SnapshotError: If dataset not found or invalid layout.
        """
        layout = self.get_layout(dataset_key)
        if layout != "frame_parquet_item_dirs":
            raise SnapshotError(
                f"Cannot read frame parts from layout '{layout}', "
                f"expected 'frame_parquet_item_dirs'"
            )

        item_dir = self.data_root / dataset_key / "parquet" / f"item-{item_index:05d}"
        if not item_dir.exists():
            raise SnapshotError(f"Item directory not found: {item_dir}")

        return list(item_dir.rglob("*.parquet"))

    def get_item_dir(self, dataset_key: str, item_index: int) -> Path:
        """Get the directory path for a parquet item.

        Args:
            dataset_key: The dataset key.
            item_index: The item index (0-based).

        Returns:
            Path to the item directory.
        """
        return self.data_root / dataset_key / "parquet" / f"item-{item_index:05d}"

    def get_ndjson_path(self, dataset_key: str) -> Path:
        """Get the path to object.ndjson for a dataset.

        Args:
            dataset_key: The dataset key.

        Returns:
            Path to object.ndjson file.
        """
        return self.data_root / dataset_key / "object.ndjson"
