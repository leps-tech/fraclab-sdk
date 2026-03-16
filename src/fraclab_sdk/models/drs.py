"""DRS (Data Requirement Specification) model."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DRSDataset(BaseModel):
    """A dataset requirement in a DRS."""

    model_config = ConfigDict(extra="ignore")

    key: str
    resource: str | None = None
    cardinality: Literal["one", "many", "zeroOrMany"] = "many"
    description: str | None = None


class DRS(BaseModel):
    """Data Requirement Specification.

    Defines what data an algorithm requires as input.
    """

    model_config = ConfigDict(extra="ignore")

    schemaVersion: str | None = None
    datasets: list[DRSDataset] = Field(default_factory=list)

    @field_validator("datasets", mode="before")
    @classmethod
    def _coerce_datasets(cls, v):
        """Accept mapping form {'key': {...}} by converting to list."""
        if isinstance(v, dict):
            return [{"key": k, **(val or {})} for k, val in v.items()]
        return v

    def get_dataset(self, dataset_key: str) -> DRSDataset | None:
        """Get a dataset requirement by key."""
        for ds in self.datasets:
            if ds.key == dataset_key:
                return ds
        return None

    def get_dataset_keys(self) -> list[str]:
        """Get all required dataset keys."""
        return [ds.key for ds in self.datasets]
