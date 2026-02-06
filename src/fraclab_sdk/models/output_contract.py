"""OutputContract model aligned with OutputSpec documentation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ScalarOutputSchema(BaseModel):
    """Schema for scalar outputs."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["scalar"]
    dtype: str | None = None
    precision: int | None = None


class FrameOutputSchema(BaseModel):
    """Schema for frame (tabular) outputs."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["frame"]
    index: Literal["time", "depth", "none"] | None = None
    # allow extra fields for forward-compatibility


class ObjectOutputSchema(BaseModel):
    """Schema for structured object outputs."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["object"]
    # keep open for schema details


class BlobOutputSchema(BaseModel):
    """Schema for blob outputs."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["blob"]
    mime: str | None = None
    ext: str | None = None


OutputSchema = ScalarOutputSchema | FrameOutputSchema | ObjectOutputSchema | BlobOutputSchema

# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

OwnerType = Literal["stage", "well", "platform"]
Cardinality = Literal["one", "many"]
DatasetKind = Literal["frame", "object", "blob", "scalar"]
DatasetRole = Literal["primary", "supporting", "debug"]


class OutputDatasetContract(BaseModel):
    """Dataset-level contract (declares a named output channel)."""

    model_config = ConfigDict(extra="ignore")

    key: str
    kind: DatasetKind
    owner: OwnerType
    cardinality: Cardinality = "many"
    required: bool = True
    schema: OutputSchema | dict[str, Any]
    role: DatasetRole | None = None
    groupPath: list[str] | None = None
    dimensions: list[str] = Field(default_factory=list)
    description: str | None = None


class OutputContract(BaseModel):
    """Full output contract for an algorithm."""

    model_config = ConfigDict(extra="ignore")

    datasets: list[OutputDatasetContract] = Field(default_factory=list)
    invariants: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)

    def get_dataset(self, key: str) -> OutputDatasetContract | None:
        """Get dataset by key."""
        for ds in self.datasets:
            if ds.key == key:
                return ds
        return None


__all__ = [
    "OutputSchema",
    "ScalarOutputSchema",
    "FrameOutputSchema",
    "ObjectOutputSchema",
    "BlobOutputSchema",
    "OutputDatasetContract",
    "OutputContract",
    "OwnerType",
    "DatasetKind",
    "Cardinality",
    "DatasetRole",
]
