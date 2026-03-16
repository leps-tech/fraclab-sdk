"""Zip archive import utilities."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from fraclab_sdk.errors import PathTraversalError
from fraclab_sdk.utils.path_safety import is_safe_relative_path


def extract_zip_and_find_root(
    zip_path: Path,
    marker: str = "manifest.json",
) -> tuple[Path, tempfile.TemporaryDirectory]:
    """Extract a zip archive safely and locate the root containing *marker*.

    The caller **must** keep the returned ``TemporaryDirectory`` alive for as
    long as the extracted files are needed; deleting or garbage-collecting it
    removes the temporary tree.

    Args:
        zip_path: Path to the ``.zip`` file.
        marker: Filename whose presence indicates the logical root directory.

    Returns:
        ``(root_path, tmp_dir)`` – *root_path* is the directory that contains
        *marker*; *tmp_dir* is the ``TemporaryDirectory`` handle.

    Raises:
        PathTraversalError: If any entry in the zip has an unsafe path.
        FileNotFoundError: If *marker* is not found at the top level or one
            directory deep.
    """
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not is_safe_relative_path(name):
                tmp_dir.cleanup()
                raise PathTraversalError(name)
        zf.extractall(tmp_path)

    root = _find_root(tmp_path, marker)
    return root, tmp_dir


def _find_root(path: Path, marker: str) -> Path:
    """Find the directory containing *marker*, checking at most one level deep."""
    if (path / marker).exists():
        return path

    for subdir in path.iterdir():
        if subdir.is_dir() and (subdir / marker).exists():
            return subdir

    raise FileNotFoundError(f"No {marker} found in {path}")
