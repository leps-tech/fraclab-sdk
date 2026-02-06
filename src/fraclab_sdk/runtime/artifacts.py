"""Artifact writer for algorithm runtime."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fraclab_sdk.errors import OutputContainmentError


@dataclass
class ArtifactRecord:
    """Record of a written artifact."""

    dataset_key: str
    owner: dict[str, Any] | None
    dims: dict[str, Any] | None
    meta: dict[str, Any] | None
    inline: dict[str, Any] | None
    item_key: str | None
    artifact_key: str
    artifact_type: str  # "scalar", "blob", "json"
    mime_type: str | None = None
    file_uri: str | None = None
    value: Any = None


class ArtifactWriter:
    """Writer for algorithm output artifacts with containment enforcement."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize artifact writer.

        Args:
            output_dir: The run output directory. All writes must be under this.
        """
        self._output_dir = output_dir.resolve()
        self._artifacts_dir = self._output_dir / "artifacts"
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[ArtifactRecord] = []

    def _validate_path(self, path: Path) -> Path:
        """Validate that path is within output directory.

        Args:
            path: Path to validate.

        Returns:
            Resolved path.

        Raises:
            OutputContainmentError: If path is outside output directory.
        """
        resolved = path.resolve()
        try:
            resolved.relative_to(self._output_dir)
        except ValueError:
            raise OutputContainmentError(str(resolved), str(self._output_dir)) from None
        return resolved

    def write_scalar(
        self,
        artifact_key: str,
        value: Any,
        *,
        dataset_key: str = "artifacts",
        owner: dict[str, Any] | None = None,
        dims: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        inline: dict[str, Any] | None = None,
        item_key: str | None = None,
    ) -> None:
        """Write a scalar artifact (number, string, bool).

        Args:
            artifact_key: Unique key for the artifact.
            value: Scalar value to store.
        """
        self._records.append(
            ArtifactRecord(
                dataset_key=dataset_key,
                owner=owner,
                dims=dims,
                meta=meta,
                inline=inline,
                item_key=item_key,
                artifact_key=artifact_key,
                artifact_type="scalar",
                value=value,
            )
        )

    def write_json(
        self,
        artifact_key: str,
        data: Any,
        filename: str | None = None,
        *,
        dataset_key: str = "artifacts",
        owner: dict[str, Any] | None = None,
        dims: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        inline: dict[str, Any] | None = None,
        item_key: str | None = None,
    ) -> Path:
        """Write a JSON artifact.

        Args:
            artifact_key: Unique key for the artifact.
            data: Data to serialize as JSON.
            filename: Optional filename. Defaults to {artifact_key}.json.
            dataset_key: Dataset key this artifact belongs to.
            owner: Optional owner map (stageId/wellId/platformId).
            dims: Optional dimensions values.
            meta: Optional meta info.
            inline: Optional inline payload.
            item_key: Optional item key override.

        Returns:
            Path to the written file.
        """
        if filename is None:
            filename = f"{artifact_key}.json"

        file_path = self._artifacts_dir / filename
        file_path = self._validate_path(file_path)

        content = json.dumps(data, indent=2, ensure_ascii=False)
        file_path.write_text(content, encoding="utf-8")

        file_uri = f"file://{file_path}"
        self._records.append(
            ArtifactRecord(
                dataset_key=dataset_key,
                owner=owner,
                dims=dims,
                meta=meta,
                inline=inline,
                item_key=item_key,
                artifact_key=artifact_key,
                artifact_type="json",
                mime_type="application/json",
                file_uri=file_uri,
            )
        )
        return file_path

    def write_blob(
        self,
        artifact_key: str,
        data: bytes,
        filename: str,
        mime_type: str | None = None,
        *,
        dataset_key: str = "artifacts",
        owner: dict[str, Any] | None = None,
        dims: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        inline: dict[str, Any] | None = None,
        item_key: str | None = None,
    ) -> Path:
        """Write a binary blob artifact.

        Args:
            artifact_key: Unique key for the artifact.
            data: Binary data to write.
            filename: Filename for the blob.
            mime_type: MIME type of the data.
            dataset_key: Dataset key this artifact belongs to.
            owner: Optional owner map (stageId/wellId/platformId).
            dims: Optional dimensions values.
            meta: Optional meta info.
            inline: Optional inline payload.
            item_key: Optional item key override.

        Returns:
            Path to the written file.
        """
        file_path = self._artifacts_dir / filename
        file_path = self._validate_path(file_path)

        file_path.write_bytes(data)

        file_uri = f"file://{file_path}"
        self._records.append(
            ArtifactRecord(
                dataset_key=dataset_key,
                owner=owner,
                dims=dims,
                meta=meta,
                inline=inline,
                item_key=item_key,
                artifact_key=artifact_key,
                artifact_type="blob",
                mime_type=mime_type,
                file_uri=file_uri,
            )
        )
        return file_path

    def write_file(
        self,
        artifact_key: str,
        source_path: Path,
        filename: str | None = None,
        mime_type: str | None = None,
        *,
        dataset_key: str = "artifacts",
        owner: dict[str, Any] | None = None,
        dims: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        inline: dict[str, Any] | None = None,
        item_key: str | None = None,
    ) -> Path:
        """Copy a file as an artifact.

        Args:
            artifact_key: Unique key for the artifact.
            source_path: Path to source file.
            filename: Optional destination filename. Defaults to source name.
            mime_type: MIME type of the file.
            dataset_key: Dataset key this artifact belongs to.
            owner: Optional owner map (stageId/wellId/platformId).
            dims: Optional dimensions values.
            meta: Optional meta info.
            inline: Optional inline payload.
            item_key: Optional item key override.

        Returns:
            Path to the copied file.
        """
        import shutil

        if filename is None:
            filename = source_path.name

        dest_path = self._artifacts_dir / filename
        dest_path = self._validate_path(dest_path)

        shutil.copy2(source_path, dest_path)

        file_uri = f"file://{dest_path}"
        self._records.append(
            ArtifactRecord(
                dataset_key=dataset_key,
                owner=owner,
                dims=dims,
                meta=meta,
                inline=inline,
                item_key=item_key,
                artifact_key=artifact_key,
                artifact_type="blob",
                mime_type=mime_type,
                file_uri=file_uri,
            )
        )
        return dest_path

    def get_records(self) -> list[ArtifactRecord]:
        """Get all artifact records."""
        return self._records.copy()

    def build_manifest_datasets(self) -> list[dict]:
        """Build dataset -> items structure for output manifest."""
        by_ds: dict[str, list[ArtifactRecord]] = {}
        for rec in self._records:
            by_ds.setdefault(rec.dataset_key, []).append(rec)

        datasets: list[dict[str, Any]] = []
        for ds_key, records in by_ds.items():
            items: list[dict[str, Any]] = []
            for rec in records:
                artifact = {
                    "artifactKey": rec.artifact_key,
                    "type": rec.artifact_type,
                }
                if rec.mime_type:
                    artifact["mimeType"] = rec.mime_type
                if rec.file_uri is not None:
                    artifact["uri"] = rec.file_uri
                if rec.value is not None:
                    artifact["value"] = rec.value
                if rec.inline is not None:
                    artifact["inline"] = rec.inline

                item: dict[str, Any] = {
                    "itemKey": rec.item_key or rec.artifact_key,
                    "artifact": artifact,
                }
                if rec.owner:
                    item["owner"] = rec.owner
                if rec.dims:
                    item["dims"] = rec.dims
                if rec.meta:
                    item["meta"] = rec.meta
                if rec.inline is not None:
                    item["inline"] = rec.inline

                items.append(item)

            datasets.append({"datasetKey": ds_key, "items": items})

        return datasets
