"""Snapshot library implementation."""

import shutil
from pathlib import Path

from fraclab_sdk.config import SDKConfig
from fraclab_sdk.errors import HashMismatchError, SnapshotError
from fraclab_sdk.materialize.hash import compute_sha256
from fraclab_sdk.models import BundleManifest
from fraclab_sdk.snapshot.index import SnapshotIndex, SnapshotMeta
from fraclab_sdk.snapshot.loader import SnapshotHandle
from fraclab_sdk.utils.path_safety import is_safe_relative_path
from fraclab_sdk.utils.zip_import import extract_zip_and_find_root


def _generate_snapshot_id(manifest_bytes: bytes) -> str:
    """Generate snapshot ID from manifest content hash.

    Uses SHA256 of manifest bytes, truncated to 16 chars for readability.
    """
    full_hash = compute_sha256(manifest_bytes)
    return full_hash[:16]


class SnapshotLibrary:
    """Library for managing snapshots."""

    def __init__(self, config: SDKConfig | None = None) -> None:
        """Initialize snapshot library.

        Args:
            config: SDK configuration. If None, uses default.
        """
        self._config = config or SDKConfig()
        self._index = SnapshotIndex(self._config.snapshots_dir)

    def import_snapshot(self, path: Path) -> str:
        """Import a snapshot from a directory or zip file.

        Args:
            path: Path to snapshot directory or zip file.

        Returns:
            The snapshot_id of the imported snapshot.

        Raises:
            SnapshotError: If import fails.
            HashMismatchError: If hash verification fails.
            PathTraversalError: If zip contains unsafe paths.
        """
        path = path.resolve()
        if not path.exists():
            raise SnapshotError(f"Path does not exist: {path}")

        if path.is_file() and path.suffix == ".zip":
            return self._import_from_zip(path)
        elif path.is_dir():
            return self._import_from_dir(path)
        else:
            raise SnapshotError(f"Path must be a directory or .zip file: {path}")

    def _import_from_zip(self, zip_path: Path) -> str:
        """Import snapshot from zip file."""
        try:
            root, tmp_dir = extract_zip_and_find_root(zip_path)
        except FileNotFoundError as exc:
            raise SnapshotError(str(exc)) from exc
        with tmp_dir:
            return self._import_from_dir(root)

    def _import_from_dir(self, source_dir: Path) -> str:
        """Import snapshot from directory."""
        # Validate manifest exists
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.exists():
            raise SnapshotError(f"manifest.json not found in {source_dir}")

        # Parse manifest and get file paths
        manifest_bytes = manifest_path.read_bytes()
        manifest = BundleManifest.model_validate_json(manifest_bytes.decode())

        manifest_paths = [
            ("specFiles.dsPath", manifest.specFiles.dsPath),
            ("specFiles.drsPath", manifest.specFiles.drsPath),
            ("dataRoot", manifest.dataRoot),
        ]
        for field_name, rel_path in manifest_paths:
            if not is_safe_relative_path(rel_path):
                raise SnapshotError(f"Unsafe manifest path {field_name}: {rel_path}")

        ds_path = source_dir / manifest.specFiles.dsPath
        drs_path = source_dir / manifest.specFiles.drsPath
        data_dir = source_dir / manifest.dataRoot

        # Validate required files exist
        if not ds_path.exists():
            raise SnapshotError(f"{manifest.specFiles.dsPath} not found in {source_dir}")
        if not drs_path.exists():
            raise SnapshotError(
                f"{manifest.specFiles.drsPath} not found (REQUIRED): {drs_path}"
            )
        if not data_dir.exists():
            raise SnapshotError(f"{manifest.dataRoot}/ directory not found in {source_dir}")

        # Verify hashes on raw bytes
        ds_bytes = ds_path.read_bytes()
        ds_hash = compute_sha256(ds_bytes)
        if ds_hash != manifest.specFiles.dsSha256:
            raise HashMismatchError(
                manifest.specFiles.dsPath, manifest.specFiles.dsSha256, ds_hash
            )

        drs_bytes = drs_path.read_bytes()
        drs_hash = compute_sha256(drs_bytes)
        if drs_hash != manifest.specFiles.drsSha256:
            raise HashMismatchError(
                manifest.specFiles.drsPath, manifest.specFiles.drsSha256, drs_hash
            )

        # Generate snapshot_id from manifest hash
        snapshot_id = _generate_snapshot_id(manifest_bytes)

        # Create target directory
        self._config.ensure_dirs()
        target_dir = self._config.snapshots_dir / snapshot_id

        if target_dir.exists():
            # Already imported
            return snapshot_id

        # Copy to library
        shutil.copytree(source_dir, target_dir)

        # Add to index
        self._index.add(
            SnapshotMeta(
                snapshot_id=snapshot_id,
                bundle_id=snapshot_id,
                created_at=str(manifest.createdAtUs),
                description=None,
            )
        )

        return snapshot_id

    def list_snapshots(self) -> list[SnapshotMeta]:
        """List all imported snapshots.

        Returns:
            List of snapshot metadata.
        """
        return self._index.list_all()

    def get_snapshot(self, snapshot_id: str) -> SnapshotHandle:
        """Get a handle to a snapshot.

        Args:
            snapshot_id: The snapshot ID.

        Returns:
            SnapshotHandle for accessing snapshot contents.

        Raises:
            SnapshotError: If snapshot not found.
        """
        snapshot_dir = self._config.snapshots_dir / snapshot_id
        if not snapshot_dir.exists():
            raise SnapshotError(f"Snapshot not found: {snapshot_id}")
        return SnapshotHandle(snapshot_dir)

    def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete a snapshot from the library.

        Args:
            snapshot_id: The snapshot ID to delete.

        Raises:
            SnapshotError: If snapshot not found.
        """
        snapshot_dir = self._config.snapshots_dir / snapshot_id
        if not snapshot_dir.exists():
            raise SnapshotError(f"Snapshot not found: {snapshot_id}")

        shutil.rmtree(snapshot_dir)
        self._index.remove(snapshot_id)
