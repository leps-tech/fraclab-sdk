"""Algorithm library implementation."""

import json
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fraclab_sdk.config import SDKConfig
from fraclab_sdk.errors import AlgorithmError, PathTraversalError
from fraclab_sdk.models import DRS
from fraclab_sdk.specs.manifest import FracLabAlgorithmManifestV1
from fraclab_sdk.utils.io import atomic_write_json


def _is_safe_path(path: str) -> bool:
    """Check if a path is safe (no traversal attacks)."""
    if path.startswith("/") or path.startswith("\\"):
        return False
    if ".." in path:
        return False
    return not any(c in path for c in [":", "*", "?", '"', "<", ">", "|"])


@dataclass
class AlgorithmMeta:
    """Metadata for an indexed algorithm."""

    algorithm_id: str
    version: str  # = codeVersion
    contract_version: str
    name: str
    summary: str
    notes: str | None = None
    imported_at: str = field(default_factory=lambda: datetime.now().isoformat())


class AlgorithmHandle:
    """Handle for accessing algorithm contents."""

    def __init__(self, algorithm_dir: Path) -> None:
        """Initialize algorithm handle.

        Args:
            algorithm_dir: Path to the algorithm version directory.
        """
        self._dir = algorithm_dir
        self._manifest: FracLabAlgorithmManifestV1 | None = None
        self._drs: DRS | None = None
        self._params_schema: dict | None = None
        self._manifest_data: dict[str, Any] | None = None

    @property
    def directory(self) -> Path:
        """Get algorithm directory path."""
        return self._dir

    @property
    def manifest(self) -> FracLabAlgorithmManifestV1:
        """Get algorithm manifest."""
        if self._manifest is None:
            manifest_path = self._dir / "manifest.json"
            if not manifest_path.exists():
                raise AlgorithmError(f"manifest.json not found: {manifest_path}")
            data = json.loads(manifest_path.read_text())
            self._manifest = FracLabAlgorithmManifestV1.model_validate(data)
            self._manifest_data = data
        return self._manifest

    def _resolve_manifest_file(self, files_key: str, default_rel: str) -> Path:
        """
        Resolve a file path declared in manifest.json under `files.*`.
        Fallback to `default_rel` for backward compatibility.
        """
        if self._manifest_data is None:
            _ = self.manifest  # loads manifest and manifest_data
        files = self._manifest_data.get("files") or {}
        rel = files.get(files_key, default_rel)
        if not isinstance(rel, str) or not rel:
            raise AlgorithmError(f"Invalid manifest.files.{files_key}: {rel!r}")
        if not _is_safe_path(rel):
            raise AlgorithmError(f"Unsafe manifest path files.{files_key}: {rel}")
        p = (self._dir / rel).resolve()
        if not p.exists():
            raise AlgorithmError(f"{rel} not found: {p}")
        return p

    @property
    def drs(self) -> DRS:
        """Get data requirement specification."""
        if self._drs is None:
            drs_path = self._resolve_manifest_file("drsPath", "drs.json")
            self._drs = DRS.model_validate_json(drs_path.read_text(encoding="utf-8"))
        return self._drs

    @property
    def params_schema(self) -> dict[str, Any]:
        """Get parameters JSON schema."""
        if self._params_schema is None:
            schema_path = self._resolve_manifest_file("paramsSchemaPath", "params.schema.json")
            self._params_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        return self._params_schema

    @property
    def algorithm_path(self) -> Path:
        """Get path to algorithm entrypoint."""
        main_path = self._dir / "main.py"
        if not main_path.exists():
            raise AlgorithmError(f"Entrypoint not found: {main_path}")
        return main_path


class AlgorithmIndex:
    """Manages the algorithm index file."""

    def __init__(self, algorithms_dir: Path) -> None:
        """Initialize algorithm index."""
        self._algorithms_dir = algorithms_dir
        self._index_path = algorithms_dir / "index.json"

    def _load(self) -> dict[str, dict]:
        """Load index from disk."""
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text())

    def _save(self, data: dict[str, dict]) -> None:
        """Save index to disk."""
        self._algorithms_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self._index_path, data)

    def _make_key(self, algorithm_id: str, version: str) -> str:
        """Create index key from algorithm_id and version."""
        return f"{algorithm_id}:{version}"

    def add(self, meta: AlgorithmMeta) -> None:
        """Add an algorithm to the index."""
        data = self._load()
        key = self._make_key(meta.algorithm_id, meta.version)
        data[key] = {
            "algorithm_id": meta.algorithm_id,
            "version": meta.version,
            "contract_version": meta.contract_version,
            "name": meta.name,
            "summary": meta.summary,
            "notes": meta.notes,
            "imported_at": meta.imported_at,
        }
        self._save(data)

    def remove(self, algorithm_id: str, version: str) -> None:
        """Remove an algorithm from the index."""
        data = self._load()
        key = self._make_key(algorithm_id, version)
        if key in data:
            del data[key]
            self._save(data)

    def get(self, algorithm_id: str, version: str) -> AlgorithmMeta | None:
        """Get algorithm metadata."""
        data = self._load()
        key = self._make_key(algorithm_id, version)
        if key not in data:
            return None
        entry = data[key]
        return AlgorithmMeta(
            algorithm_id=entry["algorithm_id"],
            version=entry["version"],
            contract_version=entry.get("contract_version", ""),
            name=entry.get("name", ""),
            summary=entry.get("summary", ""),
            notes=entry.get("notes"),
            imported_at=entry.get("imported_at", ""),
        )

    def list_all(self) -> list[AlgorithmMeta]:
        """List all indexed algorithms."""
        data = self._load()
        return [
            AlgorithmMeta(
                algorithm_id=entry["algorithm_id"],
                version=entry["version"],
                contract_version=entry.get("contract_version", ""),
                name=entry.get("name", ""),
                summary=entry.get("summary", ""),
                notes=entry.get("notes"),
                imported_at=entry.get("imported_at", ""),
            )
            for entry in data.values()
        ]


class AlgorithmLibrary:
    """Library for managing algorithms."""

    # Core required files that must exist at root
    REQUIRED_ROOT_FILES = ["main.py", "manifest.json"]

    def __init__(self, config: SDKConfig | None = None) -> None:
        """Initialize algorithm library.

        Args:
            config: SDK configuration. If None, uses default.
        """
        self._config = config or SDKConfig()
        self._index = AlgorithmIndex(self._config.algorithms_dir)

    def import_algorithm(self, path: Path) -> tuple[str, str]:
        """Import an algorithm from a directory or zip file.

        Args:
            path: Path to algorithm directory or zip file.

        Returns:
            Tuple of (algorithm_id, version).

        Raises:
            AlgorithmError: If import fails.
            PathTraversalError: If zip contains unsafe paths.
        """
        path = path.resolve()
        if not path.exists():
            raise AlgorithmError(f"Path does not exist: {path}")

        if path.is_file() and path.suffix == ".zip":
            return self._import_from_zip(path)
        elif path.is_dir():
            return self._import_from_dir(path)
        else:
            raise AlgorithmError(f"Path must be a directory or .zip file: {path}")

    def _import_from_zip(self, zip_path: Path) -> tuple[str, str]:
        """Import algorithm from zip file."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if not _is_safe_path(name):
                        raise PathTraversalError(name)
                zf.extractall(tmp_path)

            algorithm_root = self._find_algorithm_root(tmp_path)
            return self._import_from_dir(algorithm_root)

    def _find_algorithm_root(self, path: Path) -> Path:
        """Find the algorithm root directory (contains manifest.json)."""
        if (path / "manifest.json").exists():
            return path

        for subdir in path.iterdir():
            if subdir.is_dir() and (subdir / "manifest.json").exists():
                return subdir

        raise AlgorithmError(f"No manifest.json found in {path}")

    def _import_from_dir(self, source_dir: Path) -> tuple[str, str]:
        """Import algorithm from directory."""
        # Validate core root files exist
        for filename in self.REQUIRED_ROOT_FILES:
            file_path = source_dir / filename
            if not file_path.exists():
                raise AlgorithmError(f"{filename} not found in {source_dir}")

        # Parse algorithm manifest
        manifest_path = source_dir / "manifest.json"
        manifest = FracLabAlgorithmManifestV1.model_validate_json(manifest_path.read_text())

        # Validate files referenced in manifest.json exist
        manifest_data = json.loads(manifest_path.read_text())
        files_section = manifest_data.get("files", {})

        # Check required file references
        for key, default_path in [
            ("paramsSchemaPath", "params.schema.json"),
            ("drsPath", "drs.json"),
        ]:
            file_path_str = files_section.get(key, default_path)
            file_path = source_dir / file_path_str
            if not file_path.exists():
                raise AlgorithmError(f"{file_path_str} not found in {source_dir}")

        algorithm_id = manifest.algorithmId
        version = manifest.codeVersion  # version = codeVersion (pinned)

        # Create target directory
        self._config.ensure_dirs()
        target_dir = self._config.algorithms_dir / algorithm_id / version

        if target_dir.exists():
            # Already imported
            return algorithm_id, version

        # Copy to library
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, target_dir)

        # Add to index
        self._index.add(
            AlgorithmMeta(
                algorithm_id=algorithm_id,
                version=version,
                contract_version=manifest.contractVersion,
                name=manifest.name,
                summary=manifest.summary,
                notes=manifest.notes,
            )
        )

        return algorithm_id, version

    def list_algorithms(self) -> list[AlgorithmMeta]:
        """List all imported algorithms.

        Returns:
            List of algorithm metadata.
        """
        return self._index.list_all()

    def get_algorithm(self, algorithm_id: str, version: str) -> AlgorithmHandle:
        """Get a handle to an algorithm.

        Args:
            algorithm_id: The algorithm ID.
            version: The algorithm version (codeVersion).

        Returns:
            AlgorithmHandle for accessing algorithm contents.

        Raises:
            AlgorithmError: If algorithm not found.
        """
        algorithm_dir = self._config.algorithms_dir / algorithm_id / version
        if not algorithm_dir.exists():
            raise AlgorithmError(f"Algorithm not found: {algorithm_id}:{version}")
        return AlgorithmHandle(algorithm_dir)

    def delete_algorithm(self, algorithm_id: str, version: str) -> None:
        """Delete an algorithm from the library.

        Args:
            algorithm_id: The algorithm ID.
            version: The algorithm version.

        Raises:
            AlgorithmError: If algorithm not found.
        """
        algorithm_dir = self._config.algorithms_dir / algorithm_id / version
        if not algorithm_dir.exists():
            raise AlgorithmError(f"Algorithm not found: {algorithm_id}:{version}")

        shutil.rmtree(algorithm_dir)
        self._index.remove(algorithm_id, version)

        # Clean up empty parent directory
        parent_dir = self._config.algorithms_dir / algorithm_id
        if parent_dir.exists() and not any(parent_dir.iterdir()):
            parent_dir.rmdir()

    def export_algorithm(
        self, algorithm_id: str, version: str, out_path: Path
    ) -> None:
        """Export an algorithm to a directory.

        Args:
            algorithm_id: The algorithm ID.
            version: The algorithm version.
            out_path: Output directory path.

        Raises:
            AlgorithmError: If algorithm not found.
        """
        handle = self.get_algorithm(algorithm_id, version)
        shutil.copytree(handle.directory, out_path)
