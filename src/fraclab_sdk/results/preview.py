"""Preview helpers for result artifacts."""

import json
from pathlib import Path
from typing import Any

from fraclab_sdk.models import ArtifactInfo
from fraclab_sdk.results.reader import file_uri_to_path


def preview_scalar(artifact: ArtifactInfo) -> Any:
    """Preview a scalar artifact.

    Args:
        artifact: The artifact info.

    Returns:
        The scalar value or None.
    """
    if artifact.artifactType != "scalar":
        return None
    return artifact.value


def preview_image(artifact: ArtifactInfo) -> Path | None:
    """Get image path for preview.

    Args:
        artifact: The artifact info.

    Returns:
        Path to image file or None if not an image.
    """
    if artifact.artifactType != "blob":
        return None

    if artifact.mimeType and not artifact.mimeType.startswith("image/"):
        return None

    if artifact.fileUri:
        return file_uri_to_path(artifact.fileUri)

    return None


def preview_json_table(artifact: ArtifactInfo) -> dict | None:
    """Preview JSON artifact as table data.

    For array of objects, extracts columns and rows for table display.
    Format: {"columns": [...], "rows": [[...], ...]}

    Args:
        artifact: The artifact info.

    Returns:
        Table data dict or None if not suitable for table display.
    """
    if artifact.artifactType not in {"json", "object"}:
        return None

    if not artifact.fileUri:
        return None

    path = file_uri_to_path(artifact.fileUri)
    data = json.loads(path.read_text())

    # Handle array of objects
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        # Extract columns from first object
        columns = list(data[0].keys())

        # Extract rows
        rows = []
        for item in data:
            if isinstance(item, dict):
                row = [item.get(col) for col in columns]
                rows.append(row)

        return {"columns": columns, "rows": rows}

    # Handle single object
    if isinstance(data, dict):
        columns = ["key", "value"]
        rows = [[k, v] for k, v in data.items()]
        return {"columns": columns, "rows": rows}

    return None


def preview_json_raw(artifact: ArtifactInfo, max_lines: int = 50) -> str | None:
    """Preview raw JSON content.

    Args:
        artifact: The artifact info.
        max_lines: Maximum lines to return.

    Returns:
        Pretty-printed JSON string or None.
    """
    if artifact.artifactType != "json":
        return None

    if not artifact.fileUri:
        return None

    path = file_uri_to_path(artifact.fileUri)
    data = json.loads(path.read_text())
    formatted = json.dumps(data, indent=2)

    lines = formatted.split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append(f"... ({len(formatted.split(chr(10))) - max_lines} more lines)")

    return "\n".join(lines)


def get_artifact_preview_type(artifact: ArtifactInfo) -> str:
    """Determine the best preview type for an artifact.

    Args:
        artifact: The artifact info.

    Returns:
        Preview type: "scalar", "image", "json_table", "json_raw", "file", or "none".
    """
    if artifact.artifactType == "scalar":
        return "scalar"

    if artifact.artifactType == "blob":
        if artifact.mimeType and artifact.mimeType.startswith("image/"):
            return "image"
        return "file"

    if artifact.artifactType in {"json", "object"}:
        # Check if suitable for table display
        if artifact.fileUri:
            try:
                path = file_uri_to_path(artifact.fileUri)
                data = json.loads(path.read_text())
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    return "json_table"
            except Exception:
                pass
        return "json_raw"

    if artifact.artifactType in {"frame", "parquet"}:
        return "file"

    return "none"
