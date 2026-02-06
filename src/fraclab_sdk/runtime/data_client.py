"""Data client for algorithm runtime."""

import json
from pathlib import Path

from fraclab_sdk.models import DataSpec


class DataClient:
    """Client for reading input data during algorithm execution."""

    def __init__(self, input_dir: Path) -> None:
        """Initialize data client.

        Args:
            input_dir: The run input directory containing ds.json and data/.
        """
        self._input_dir = input_dir
        self._dataspec: DataSpec | None = None

    @property
    def dataspec(self) -> DataSpec:
        """Get the data specification."""
        if self._dataspec is None:
            ds_path = self._input_dir / "ds.json"
            self._dataspec = DataSpec.model_validate_json(ds_path.read_text())
        return self._dataspec

    def get_dataset_keys(self) -> list[str]:
        """Get list of available dataset keys."""
        return self.dataspec.get_dataset_keys()

    def get_item_count(self, dataset_key: str) -> int:
        """Get number of items in a dataset."""
        dataset = self.dataspec.get_dataset(dataset_key)
        if dataset is None:
            raise KeyError(f"Dataset not found: {dataset_key}")
        return len(dataset.items)

    def get_layout(self, dataset_key: str) -> str | None:
        """Get the layout type for a dataset."""
        dataset = self.dataspec.get_dataset(dataset_key)
        if dataset is None:
            raise KeyError(f"Dataset not found: {dataset_key}")
        return dataset.layout

    def read_object(self, dataset_key: str, item_index: int) -> dict:
        """Read an object from ndjson dataset.

        Args:
            dataset_key: The dataset key.
            item_index: The item index (0-based, run-indexed).

        Returns:
            Parsed JSON object.
        """
        layout = self.get_layout(dataset_key)
        if layout != "object_ndjson_lines":
            raise ValueError(
                f"Cannot read object from layout '{layout}', "
                f"expected 'object_ndjson_lines'"
            )

        ndjson_path = self._input_dir / "data" / dataset_key / "object.ndjson"
        with ndjson_path.open() as f:
            for i, line in enumerate(f):
                if i == item_index:
                    return json.loads(line)

        raise IndexError(f"Item index {item_index} not found")

    def get_parquet_dir(self, dataset_key: str, item_index: int) -> Path:
        """Get path to parquet item directory.

        Args:
            dataset_key: The dataset key.
            item_index: The item index (0-based, run-indexed).

        Returns:
            Path to the item directory.
        """
        layout = self.get_layout(dataset_key)
        if layout != "frame_parquet_item_dirs":
            raise ValueError(
                f"Cannot get parquet dir from layout '{layout}', "
                f"expected 'frame_parquet_item_dirs'"
            )

        return self._input_dir / "data" / dataset_key / "parquet" / f"item-{item_index:05d}"

    def get_parquet_files(self, dataset_key: str, item_index: int) -> list[Path]:
        """Get list of parquet files for an item.

        Args:
            dataset_key: The dataset key.
            item_index: The item index (0-based, run-indexed).

        Returns:
            List of parquet file paths.
        """
        item_dir = self.get_parquet_dir(dataset_key, item_index)
        return list(item_dir.rglob("*.parquet"))

    def iterate_objects(self, dataset_key: str):
        """Iterate over all objects in an ndjson dataset.

        Args:
            dataset_key: The dataset key.

        Yields:
            Tuple of (index, object dict).
        """
        layout = self.get_layout(dataset_key)
        if layout != "object_ndjson_lines":
            raise ValueError(
                f"Cannot iterate objects from layout '{layout}', "
                f"expected 'object_ndjson_lines'"
            )

        ndjson_path = self._input_dir / "data" / dataset_key / "object.ndjson"
        with ndjson_path.open() as f:
            for i, line in enumerate(f):
                yield i, json.loads(line)
