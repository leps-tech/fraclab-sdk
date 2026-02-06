"""Bundle manifest model (Data Bundle Spec v1.0.0)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class SpecFiles(BaseModel):
    """Specification files metadata."""

    model_config = ConfigDict(extra="ignore")

    dsPath: str = "ds.json"
    drsPath: str = "drs.json"
    dsSha256: str
    drsSha256: str


class DatasetEntryFile(BaseModel):
    """Individual file entry in dataset files list."""

    model_config = ConfigDict(extra="ignore")

    path: str
    sha256: str
    bytes: int | None = None


class DatasetEntry(BaseModel):
    """Dataset entry in manifest."""

    model_config = ConfigDict(extra="ignore")

    layout: Literal["object_ndjson_lines", "frame_parquet_item_dirs"]
    count: int
    files: list[DatasetEntryFile] | None = None


class BundleManifest(BaseModel):
    """Manifest for a data bundle (Data Bundle Spec v1.0.0).

    Contains metadata and integrity hashes for the bundle contents.
    """

    model_config = ConfigDict(extra="ignore")

    bundleVersion: str
    createdAtUs: int
    specFiles: SpecFiles
    dataRoot: str = "data"
    datasets: dict[str, DatasetEntry]
