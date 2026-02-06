"""Snapshot library implementation."""

import shutil
import zipfile
from pathlib import Path

from fraclab_sdk.config import SDKConfig
from fraclab_sdk.errors import HashMismatchError, PathTraversalError, SnapshotError
from fraclab_sdk.materialize.hash import compute_sha256
from fraclab_sdk.models import BundleManifest
from fraclab_sdk.snapshot.index import SnapshotIndex, SnapshotMeta
from fraclab_sdk.snapshot.loader import SnapshotHandle


def _is_safe_path(path: str) -> bool:
    """Check if a path is safe (no traversal attacks)."""
    if path.startswith("/") or path.startswith("\\"):
        return False
    if ".." in path:
        return False
    return not any(c in path for c in [":", "*", "?", '"', "<", ">", "|"])


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
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with zipfile.ZipFile(zip_path) as zf:
                # Security check: verify all paths are safe
                for name in zf.namelist():
                    if not _is_safe_path(name):
                        raise PathTraversalError(name)
                zf.extractall(tmp_path)

            # Find the actual snapshot root (may be in a subdirectory)
            snapshot_root = self._find_snapshot_root(tmp_path)
            return self._import_from_dir(snapshot_root)

    def _find_snapshot_root(self, path: Path) -> Path:
        """Find the snapshot root directory (contains manifest.json)."""
        if (path / "manifest.json").exists():
            return path

        # Check one level of subdirectories
        for subdir in path.iterdir():
            if subdir.is_dir() and (subdir / "manifest.json").exists():
                return subdir

        raise SnapshotError(f"No manifest.json found in {path}")

    def _import_from_dir(self, source_dir: Path) -> str:
        """Import snapshot from directory."""
        # Validate manifest exists
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.exists():
            raise SnapshotError(f"manifest.json not found in {source_dir}")

        # Parse manifest and get file paths
        manifest_bytes = manifest_path.read_bytes()
        manifest = BundleManifest.model_validate_json(manifest_bytes.decode())

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
