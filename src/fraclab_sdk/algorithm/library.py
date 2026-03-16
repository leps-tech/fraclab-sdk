"""Algorithm library implementation."""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fraclab_sdk.config import SDKConfig
from fraclab_sdk.errors import AlgorithmError
from fraclab_sdk.models import DRS
from fraclab_sdk.models.algorithm_manifest import FracLabAlgorithmManifestV1
from fraclab_sdk.utils.json_index_store import JsonIndexStore
from fraclab_sdk.utils.path_safety import is_safe_relative_path
from fraclab_sdk.utils.zip_import import extract_zip_and_find_root


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
            self._manifest = FracLabAlgorithmManifestV1.model_validate_json(
                manifest_path.read_text()
            )
        return self._manifest

    def _resolve_manifest_file(self, rel: str, field_name: str) -> Path:
        """Resolve and validate a file path declared in manifest.json under files.*."""
        if not isinstance(rel, str) or not rel:
            raise AlgorithmError(f"Invalid manifest.files.{field_name}: {rel!r}")
        if not is_safe_relative_path(rel):
            raise AlgorithmError(f"Unsafe manifest path files.{field_name}: {rel}")
        p = (self._dir / rel).resolve()
        if not p.exists():
            raise AlgorithmError(f"{rel} not found: {p}")
        return p

    @property
    def drs(self) -> DRS:
        """Get data requirement specification."""
        if self._drs is None:
            drs_rel = self.manifest.files.drsPath
            if not drs_rel:
                # Missing drsPath is allowed for minimal packages; selection will infer from snapshot.
                self._drs = DRS(schemaVersion="1.0", datasets=[])
            else:
                drs_path = self._resolve_manifest_file(drs_rel, "drsPath")
                self._drs = DRS.model_validate_json(drs_path.read_text(encoding="utf-8"))
        return self._drs

    @property
    def params_schema(self) -> dict[str, Any]:
        """Get parameters JSON schema."""
        if self._params_schema is None:
            schema_path = self._resolve_manifest_file(
                self.manifest.files.paramsSchemaPath, "paramsSchemaPath"
            )
            self._params_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        return self._params_schema

    @property
    def algorithm_path(self) -> Path:
        """Get path to algorithm entrypoint."""
        main_path = self._dir / "main.py"
        if not main_path.exists():
            raise AlgorithmError(f"Entrypoint not found: {main_path}")
        return main_path


def _algo_to_entry(meta: AlgorithmMeta) -> dict[str, Any]:
    return {
        "algorithm_id": meta.algorithm_id,
        "version": meta.version,
        "contract_version": meta.contract_version,
        "name": meta.name,
        "summary": meta.summary,
        "notes": meta.notes,
        "imported_at": meta.imported_at,
    }


def _algo_from_entry(entry: dict[str, Any]) -> AlgorithmMeta:
    return AlgorithmMeta(
        algorithm_id=entry["algorithm_id"],
        version=entry["version"],
        contract_version=entry.get("contract_version", ""),
        name=entry.get("name", ""),
        summary=entry.get("summary", ""),
        notes=entry.get("notes"),
        imported_at=entry.get("imported_at", ""),
    )


class AlgorithmIndex:
    """Manages the algorithm index file."""

    def __init__(self, algorithms_dir: Path) -> None:
        self._store: JsonIndexStore[AlgorithmMeta] = JsonIndexStore(
            algorithms_dir,
            make_key=lambda m: f"{m.algorithm_id}:{m.version}",
            to_entry=_algo_to_entry,
            from_entry=_algo_from_entry,
        )

    def add(self, meta: AlgorithmMeta) -> None:
        self._store.add(meta)

    def remove(self, algorithm_id: str, version: str) -> None:
        self._store.remove(f"{algorithm_id}:{version}")

    def get(self, algorithm_id: str, version: str) -> AlgorithmMeta | None:
        return self._store.get(f"{algorithm_id}:{version}")

    def list_all(self) -> list[AlgorithmMeta]:
        return self._store.list_all()


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
        try:
            root, tmp_dir = extract_zip_and_find_root(zip_path)
        except FileNotFoundError as exc:
            raise AlgorithmError(str(exc)) from exc
        with tmp_dir:
            return self._import_from_dir(root)

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
        required_files = [
            ("paramsSchemaPath", manifest.files.paramsSchemaPath),
        ]
        optional_files = [
            ("outputContractPath", manifest.files.outputContractPath),
            ("dsPath", manifest.files.dsPath),
            ("drsPath", manifest.files.drsPath),
        ]

        for field_name, file_path_str in required_files:
            if not is_safe_relative_path(file_path_str):
                raise AlgorithmError(
                    f"Unsafe manifest path files.{field_name}: {file_path_str}"
                )
            file_path = source_dir / file_path_str
            if not file_path.exists():
                raise AlgorithmError(f"{file_path_str} not found in {source_dir}")

        for field_name, file_path_str in optional_files:
            if not file_path_str:
                continue
            if not is_safe_relative_path(file_path_str):
                raise AlgorithmError(
                    f"Unsafe manifest path files.{field_name}: {file_path_str}"
                )
            file_path = source_dir / file_path_str
            if not file_path.exists():
                raise AlgorithmError(f"{file_path_str} not found in {source_dir}")

        algorithm_id = manifest.algorithmId
        version = manifest.codeVersion  # version = codeVersion (pinned)

        # Create target directory
        self._config.ensure_dirs()
        target_dir = self._config.algorithms_dir / algorithm_id / version

        if target_dir.exists():
            # Already exists on disk: ensure it's indexed.
            if self._index.get(algorithm_id, version) is None:
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
