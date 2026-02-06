"""DataSpec model."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OwnerIds(BaseModel):
    """Structured owner identifiers."""

    model_config = ConfigDict(extra="ignore")

    platformId: str | None = None
    wellId: str | None = None
    stageId: str | None = None


class DataSpecItem(BaseModel):
    """An item within a dataset."""

    model_config = ConfigDict(extra="ignore")

    owner: OwnerIds | str | None = None
    resolutionParams: dict[str, Any] | None = None
    range: dict[str, Any] | None = None
    sourceItemIndex: int | None = None  # For traceability in run ds


class DataSpecDataset(BaseModel):
    """A dataset within a DataSpec."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    datasetKey: str = Field(alias="key")
    resourceType: str | None = None
    layout: str | None = None  # e.g., "frame_parquet_item_dirs" or "object_ndjson_lines"
    items: list[DataSpecItem] = Field(default_factory=list)


class DataSpec(BaseModel):
    """Data specification describing datasets and their items."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    schemaVersion: str | None = None
    datasets: list[DataSpecDataset] = Field(default_factory=list)

    @field_validator("datasets", mode="before")
    @classmethod
    def _coerce_datasets(cls, v):
        """Accept mapping form {'key': {...}} by converting to list."""
        if isinstance(v, dict):
            return [{"key": k, **(val or {})} for k, val in v.items()]
        return v

    def get_dataset(self, dataset_key: str) -> DataSpecDataset | None:
        """Get a dataset by key."""
        for ds in self.datasets:
            if ds.datasetKey == dataset_key:
                return ds
        return None

    def get_dataset_keys(self) -> list[str]:
        """Get all dataset keys."""
        return [ds.datasetKey for ds in self.datasets]
