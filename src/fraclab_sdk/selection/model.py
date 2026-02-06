"""Selection model implementation."""

from dataclasses import dataclass

from fraclab_sdk.errors import DatasetKeyError
from fraclab_sdk.models import DRS, DataSpec, DataSpecDataset, DataSpecItem, DRSDataset
from fraclab_sdk.selection.validate import ValidationError, validate_cardinality
from fraclab_sdk.snapshot.loader import SnapshotHandle


@dataclass
class SelectableDataset:
    """Information about a selectable dataset."""

    dataset_key: str
    cardinality: str
    total_items: int
    resource_type: str | None = None
    description: str | None = None


class SelectionModel:
    """Manages selection state for creating a run.

    Selection uses snapshot item indices (integers).
    Items are automatically sorted and deduplicated.
    build_run_ds() produces a re-indexed DataSpec (0..N-1).
    """

    def __init__(
        self,
        snapshot: SnapshotHandle,
        drs: DRS,
    ) -> None:
        """Initialize selection model.

        Args:
            snapshot: The snapshot handle.
            drs: The data requirement specification (from algorithm).

        Raises:
            DatasetKeyError: If DRS requires a dataset not in snapshot.
        """
        self._snapshot = snapshot

        # If DRS is empty, infer from snapshot dataspec so users can still select data.
        if not drs.datasets:
            inferred = [
                DRSDataset(
                    datasetKey=ds.datasetKey,
                    resourceType=ds.resourceType,
                    cardinality="many",
                    description=ds.layout,
                )
                for ds in snapshot.dataspec.datasets
            ]
            self._drs = DRS(schemaVersion=drs.schemaVersion, datasets=inferred)
        else:
            self._drs = drs
        self._selections: dict[str, list[int]] = {}

        # Validate that all DRS dataset keys exist in snapshot
        snapshot_keys = set(snapshot.dataspec.get_dataset_keys())
        for drs_dataset in self._drs.datasets:
            if drs_dataset.datasetKey not in snapshot_keys:
                raise DatasetKeyError(
                    dataset_key=drs_dataset.datasetKey,
                    available_keys=list(snapshot_keys),
                )
            # Initialize empty selection
            self._selections[drs_dataset.datasetKey] = []

    @classmethod
    def from_snapshot_and_drs(
        cls,
        snapshot: SnapshotHandle,
        drs: DRS,
    ) -> "SelectionModel":
        """Create a SelectionModel from snapshot and DRS.

        Args:
            snapshot: The snapshot handle.
            drs: The data requirement specification.

        Returns:
            Initialized SelectionModel.
        """
        return cls(snapshot=snapshot, drs=drs)

    def get_selectable_datasets(self) -> list[SelectableDataset]:
        """Get list of datasets that can be selected.

        Returns:
            List of SelectableDataset with metadata.
        """
        result = []
        for drs_dataset in self._drs.datasets:
            snapshot_dataset = self._snapshot.dataspec.get_dataset(drs_dataset.datasetKey)
            if snapshot_dataset:
                result.append(
                    SelectableDataset(
                        dataset_key=drs_dataset.datasetKey,
                        cardinality=drs_dataset.cardinality,
                        total_items=len(snapshot_dataset.items),
                        resource_type=drs_dataset.resourceType,
                        description=drs_dataset.description,
                    )
                )
        return result

    def set_selected(self, dataset_key: str, item_indices: list[int]) -> None:
        """Set selected items for a dataset.

        Items are automatically sorted (ascending) and deduplicated.

        Args:
            dataset_key: The dataset key.
            item_indices: List of item indices to select.

        Raises:
            DatasetKeyError: If dataset_key is not in the selection.
        """
        if dataset_key not in self._selections:
            raise DatasetKeyError(
                dataset_key=dataset_key,
                available_keys=list(self._selections.keys()),
            )
        # Sort and deduplicate
        self._selections[dataset_key] = sorted(set(item_indices))

    def get_selected(self, dataset_key: str) -> list[int]:
        """Get selected item indices for a dataset.

        Returns:
            Sorted list of selected item indices.

        Raises:
            DatasetKeyError: If dataset_key is not in the selection.
        """
        if dataset_key not in self._selections:
            raise DatasetKeyError(
                dataset_key=dataset_key,
                available_keys=list(self._selections.keys()),
            )
        return self._selections[dataset_key]

    def validate(self) -> list[ValidationError]:
        """Validate all selections against cardinality constraints.

        Returns:
            List of validation errors (empty if all valid).
        """
        errors = []
        for drs_dataset in self._drs.datasets:
            selected = self._selections.get(drs_dataset.datasetKey, [])
            error = validate_cardinality(
                dataset_key=drs_dataset.datasetKey,
                cardinality=drs_dataset.cardinality,
                selected_count=len(selected),
            )
            if error:
                errors.append(error)
        return errors

    def is_valid(self) -> bool:
        """Check if current selection is valid.

        Returns:
            True if all selections satisfy cardinality constraints.
        """
        return len(self.validate()) == 0

    def _infer_layout(self, dataset_key: str) -> str | None:
        """Infer layout from dataspec, manifest, or on-disk data."""
        ds = self._snapshot.dataspec.get_dataset(dataset_key)
        if ds and ds.layout:
            return ds.layout

        manifest_ds = self._snapshot.manifest.datasets.get(dataset_key)
        if manifest_ds and getattr(manifest_ds, "layout", None):
            return manifest_ds.layout

        data_root = self._snapshot.manifest.dataRoot or "data"
        base = self._snapshot.directory / data_root / dataset_key
        if (base / "object.ndjson").exists():
            return "object_ndjson_lines"
        if (base / "parquet").exists():
            return "frame_parquet_item_dirs"
        return None

    def build_run_ds(self) -> DataSpec:
        """Build a run DataSpec from current selection.

        Selected items are re-indexed to 0..N-1 (compact indices).
        Each item includes sourceItemIndex for traceability.

        Returns:
            DataSpec with selected items, re-indexed.
        """
        datasets = []

        for drs_dataset in self._drs.datasets:
            snapshot_dataset = self._snapshot.dataspec.get_dataset(drs_dataset.datasetKey)
            if not snapshot_dataset:
                continue

            selected_indices = self._selections.get(drs_dataset.datasetKey, [])

            # Build re-indexed items
            items = []
            for snapshot_index in selected_indices:
                if 0 <= snapshot_index < len(snapshot_dataset.items):
                    original_item = snapshot_dataset.items[snapshot_index]
                    # Create new item with sourceItemIndex for traceability
                    new_item = DataSpecItem(
                        owner=original_item.owner,
                        resolutionParams=original_item.resolutionParams,
                        range=original_item.range,
                        sourceItemIndex=snapshot_index,
                    )
                    items.append(new_item)

            datasets.append(
                DataSpecDataset(
                    datasetKey=snapshot_dataset.datasetKey,
                    resourceType=snapshot_dataset.resourceType,
                    layout=self._infer_layout(snapshot_dataset.datasetKey),
                    items=items,
                )
            )

        return DataSpec(
            schemaVersion=self._snapshot.dataspec.schemaVersion,
            datasets=datasets,
        )

    def get_selection_mapping(self, dataset_key: str) -> list[tuple[int, int]]:
        """Get mapping from run index to snapshot index.

        Args:
            dataset_key: The dataset key.

        Returns:
            List of (run_index, snapshot_index) tuples.
        """
        selected = self.get_selected(dataset_key)
        return list(enumerate(selected))
