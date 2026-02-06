"""Result reader implementation."""

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from fraclab_sdk.errors import OutputContainmentError, ResultError
from fraclab_sdk.models import ArtifactInfo, RunOutputManifest


def file_uri_to_path(uri: str) -> Path:
    """Convert file:// URI to Path.

    Args:
        uri: A file:// URI string.

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If URI scheme is not file://.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Expected file:// URI, got: {uri}")
    decoded = unquote(parsed.path)
    return Path(decoded).expanduser().resolve()


@dataclass
class ArtifactWithPath:
    """Artifact info with resolved file path."""

    artifact: ArtifactInfo
    path: Path | None


class ResultReader:
    """Reader for run output results."""

    def __init__(self, run_dir: Path) -> None:
        """Initialize result reader.

        Args:
            run_dir: The run directory.
        """
        self._run_dir = run_dir
        self._output_dir = run_dir / "output"
        self._manifest: RunOutputManifest | None = None

    @property
    def output_dir(self) -> Path:
        """Get the output directory path."""
        return self._output_dir

    def has_manifest(self) -> bool:
        """Check if output manifest exists."""
        return (self._output_dir / "manifest.json").exists()

    def read_manifest(self) -> RunOutputManifest:
        """Read the output manifest.

        Returns:
            Parsed RunOutputManifest.

        Raises:
            ResultError: If manifest not found or invalid.
        """
        if self._manifest is not None:
            return self._manifest

        manifest_path = self._output_dir / "manifest.json"
        if not manifest_path.exists():
            raise ResultError(f"Output manifest not found: {manifest_path}")

        content = manifest_path.read_text()
        try:
            self._manifest = RunOutputManifest.model_validate_json(content)
        except Exception:
            # Legacy shape fallback: convert if possible
            try:
                data = json.loads(content)
                data = self._coerce_legacy_manifest(data)
                self._manifest = RunOutputManifest.model_validate(data)
            except Exception as e:  # pragma: no cover - best effort
                raise ResultError(f"Failed to parse output manifest: {e}") from e

        return self._manifest

    def get_status(self) -> str:
        """Get run status from manifest.

        Returns:
            Status string (e.g., "completed", "failed").
        """
        manifest = self.read_manifest()
        return manifest.status or "unknown"

    def get_error(self) -> str | None:
        """Get error message if run failed.

        Returns:
            Error message or None.
        """
        manifest = self.read_manifest()
        return manifest.error

    def list_artifacts(self) -> list[ArtifactInfo]:
        """List all artifacts.

        Returns:
            List of ArtifactInfo objects.
        """
        manifest = self.read_manifest()
        return manifest.list_all_artifacts()

    def get_artifact(self, artifact_key: str) -> ArtifactInfo | None:
        """Get artifact by key.

        Args:
            artifact_key: The artifact key.

        Returns:
            ArtifactInfo or None if not found.
        """
        manifest = self.read_manifest()
        return manifest.get_artifact(artifact_key)

    def get_artifact_path(self, artifact_key: str) -> Path | None:
        """Get file path for an artifact.

        Args:
            artifact_key: The artifact key.

        Returns:
            Path to artifact file or None if no file URI.
        """
        artifact = self.get_artifact(artifact_key)
        if artifact is None or artifact.fileUri is None:
            return None
        return file_uri_to_path(artifact.fileUri)

    def get_artifact_with_path(self, artifact_key: str) -> ArtifactWithPath | None:
        """Get artifact with resolved path.

        Args:
            artifact_key: The artifact key.

        Returns:
            ArtifactWithPath or None if artifact not found.
        """
        artifact = self.get_artifact(artifact_key)
        if artifact is None:
            return None

        path = None
        if artifact.fileUri:
            path = file_uri_to_path(artifact.fileUri)

        return ArtifactWithPath(artifact=artifact, path=path)

    def read_artifact_json(self, artifact_key: str) -> dict | list | None:
        """Read JSON artifact content.

        Args:
            artifact_key: The artifact key.

        Returns:
            Parsed JSON content or None if not a JSON artifact.
        """
        artifact = self.get_artifact(artifact_key)
        if artifact is None:
            return None

        if artifact.artifactType not in {"json", "object"}:
            return None

        if artifact.fileUri:
            path = self._safe_artifact_path(file_uri_to_path(artifact.fileUri))
            return json.loads(path.read_text())

        if artifact.inline and "data" in artifact.inline:
            return artifact.inline.get("data")

        return None

    def _safe_artifact_path(self, path: Path) -> Path:
        """Ensure artifact path is within output dir."""
        try:
            path.resolve().relative_to(self._output_dir)
        except Exception:
            raise ResultError(f"Artifact path escapes output dir: {path}")
        return path

    def _coerce_legacy_manifest(self, data: dict) -> dict:
        """
        Convert legacy manifest shapes:
        - top-level artifacts[] -> dataset 'artifacts' with items
        - datasets[].artifacts[] -> datasets[].items with single artifact
        """
        datasets = data.get("datasets", [])
        new_datasets = []
        for ds in datasets:
            if "items" in ds:
                new_datasets.append(ds)
                continue
            artifacts = ds.get("artifacts", [])
            items = []
            for art in artifacts:
                items.append(
                    {
                        "itemKey": art.get("artifactKey") or art.get("key"),
                        "artifact": art,
                    }
                )
            new_datasets.append({"datasetKey": ds.get("datasetKey") or ds.get("key"), "items": items})

        # If legacy top-level artifacts
        top_artifacts = data.get("artifacts", [])
        if top_artifacts:
            items = [
                {"itemKey": art.get("artifactKey") or art.get("key"), "artifact": art}
                for art in top_artifacts
            ]
            new_datasets.append({"datasetKey": "artifacts", "items": items})

        data["datasets"] = new_datasets
        data.pop("artifacts", None)
        return data

    def read_artifact_scalar(self, artifact_key: str):
        """Read scalar artifact value.

        Args:
            artifact_key: The artifact key.

        Returns:
            Scalar value or None if not a scalar artifact.
        """
        artifact = self.get_artifact(artifact_key)
        if artifact is None:
            return None

        if artifact.artifactType != "scalar":
            return None

        return artifact.value

    def get_logs_dir(self) -> Path:
        """Get the logs directory path."""
        return self._output_dir / "_logs"

    def read_stdout(self) -> str | None:
        """Read stdout log if exists."""
        stdout_path = self.get_logs_dir() / "stdout.log"
        if stdout_path.exists():
            return stdout_path.read_text()
        return None

    def read_stderr(self) -> str | None:
        """Read stderr log if exists."""
        stderr_path = self.get_logs_dir() / "stderr.log"
        if stderr_path.exists():
            return stderr_path.read_text()
        return None

    def read_algorithm_log(self) -> str | None:
        """Read algorithm log if exists."""
        log_path = self.get_logs_dir() / "algorithm.log"
        if log_path.exists():
            return log_path.read_text()
        return None

    def open_artifact(self, artifact_key: str) -> Path:
        """Get validated file path for an artifact.

        This is the primary interface for UI/CLI to access artifact files.
        It validates that the artifact path is contained within the output
        directory to prevent path traversal attacks.

        Args:
            artifact_key: The artifact key.

        Returns:
            Validated Path to artifact file.

        Raises:
            ResultError: If artifact not found or has no file URI.
            OutputContainmentError: If artifact path is outside output directory.
        """
        artifact = self.get_artifact(artifact_key)
        if artifact is None:
            raise ResultError(f"Artifact not found: {artifact_key}")

        if artifact.fileUri is None:
            raise ResultError(f"Artifact '{artifact_key}' has no file URI (may be a scalar)")

        path = file_uri_to_path(artifact.fileUri)

        # Validate path is within output directory
        self._validate_path_containment(path)

        if not path.exists():
            raise ResultError(f"Artifact file not found: {path}")

        return path

    def _validate_path_containment(self, path: Path) -> Path:
        """Validate that path is contained within output directory.

        Args:
            path: Path to validate.

        Returns:
            Resolved path.

        Raises:
            OutputContainmentError: If path is outside output directory.
        """
        resolved = path.resolve()
        output_resolved = self._output_dir.resolve()

        try:
            resolved.relative_to(output_resolved)
        except ValueError:
            raise OutputContainmentError(str(resolved), str(output_resolved)) from None

        return resolved
