"""Run output manifest model aligned with OutputSpec."""

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class RunInfo(BaseModel):
    """Information about a run."""

    model_config = ConfigDict(extra="ignore")

    runId: str
    algorithmId: str
    contractVersion: str | None = None
    codeVersion: str | None = None


class OwnerRef(BaseModel):
    """Owner reference for an item."""

    model_config = ConfigDict(extra="ignore")

    platformId: str | None = None
    wellId: str | None = None
    stageId: str | None = None


class ArtifactInfo(BaseModel):
    """Information about an output artifact."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    artifactKey: str
    type: str = Field(
        validation_alias=AliasChoices("type", "artifactType"),
        serialization_alias="type",
    )  # e.g., "scalar", "blob", "json", "frame", "parquet"
    uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("uri", "fileUri"),
        serialization_alias="uri",
    )
    mimeType: str | None = None
    description: str | None = None
    value: Any | None = None  # For scalar artifacts
    inline: dict[str, Any] | None = None  # Optional embedded payload

    @property
    def artifactType(self) -> str:
        """Backward-compatible accessor for artifact type."""
        return self.type

    @property
    def fileUri(self) -> str | None:
        """Backward-compatible accessor for file URI."""
        return self.uri


class RunOutputItem(BaseModel):
    """Single item within a dataset."""

    model_config = ConfigDict(extra="ignore")

    itemKey: str | None = Field(default=None, alias="key")
    owner: OwnerRef | None = None
    dims: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
    inline: dict[str, Any] | None = None
    artifact: ArtifactInfo


class RunOutputDataset(BaseModel):
    """Output dataset in run output manifest."""

    model_config = ConfigDict(extra="ignore")

    datasetKey: str
    items: list[RunOutputItem] = Field(default_factory=list)


class RunOutputManifest(BaseModel):
    """Manifest for run output."""

    model_config = ConfigDict(extra="ignore")

    schemaVersion: str | None = None
    run: RunInfo
    status: str | None = None
    error: str | None = None
    startedAt: str | None = None
    completedAt: str | None = None
    datasets: list[RunOutputDataset] = Field(default_factory=list)

    def get_artifact(self, artifact_key: str) -> ArtifactInfo | None:
        """Get an artifact by key (searches all datasets)."""
        for dataset in self.datasets:
            for item in dataset.items:
                if item.artifact.artifactKey == artifact_key:
                    return item.artifact
        return None

    def list_all_artifacts(self) -> list[ArtifactInfo]:
        """List all artifacts from all datasets."""
        all_artifacts: list[ArtifactInfo] = []
        for dataset in self.datasets:
            for item in dataset.items:
                all_artifacts.append(item.artifact)
        return all_artifacts


__all__ = [
    "RunInfo",
    "OwnerRef",
    "ArtifactInfo",
    "RunOutputItem",
    "RunOutputDataset",
    "RunOutputManifest",
]
