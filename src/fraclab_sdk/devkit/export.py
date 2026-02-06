"""Algorithm export: package workspace for distribution.

Export creates a distributable algorithm package containing:
- main.py (required)
- manifest.json (required)
- dist/params.schema.json (required)
- dist/output_contract.json (required)
- dist/drs.json (required)
- README.md (optional)
- schema/** (optional, source code)
- examples/** (optional)
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from fraclab_sdk.errors import AlgorithmError

# Files that must exist in dist/
REQUIRED_DIST_FILES = [
    "params.schema.json",
    "output_contract.json",
    "drs.json",
]

# Required workspace files
REQUIRED_WORKSPACE_FILES = [
    "main.py",
    "manifest.json",
]

# Patterns to include (relative to workspace)
INCLUDE_PATTERNS = [
    "main.py",
    "manifest.json",
    "dist/params.schema.json",
    "dist/output_contract.json",
    "dist/drs.json",
    "README.md",
    "schema/**",
    "examples/**",
]

# Patterns to always reject
REJECT_PATTERNS = [
    "__pycache__",
    ".DS_Store",
    ".idea",
    ".git",
    ".gitignore",
    ".vscode",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
    ".eggs",
    "*.tmp",
    "*.temp",
    "*.bak",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
]


@dataclass
class ExportResult:
    """Result of algorithm export."""

    output_path: Path
    files_included: list[str]
    files_rejected: list[str]


def _should_reject(path: Path, workspace: Path) -> bool:
    """Check if a path should be rejected.

    Args:
        path: Path to check.
        workspace: Workspace root.

    Returns:
        True if path should be rejected.
    """
    rel_path = path.relative_to(workspace)
    name = path.name

    for pattern in REJECT_PATTERNS:
        if pattern.startswith("*."):
            if name.endswith(pattern[1:]):
                return True
        elif name == pattern or pattern in str(rel_path):
            return True

    # Reject hidden files/directories
    return name.startswith(".")


def _is_path_contained(path: Path, root: Path) -> bool:
    """Check if a path is contained within root (no directory traversal).

    Args:
        path: Path to check (will be resolved).
        root: Root directory (will be resolved).

    Returns:
        True if path is within root.
    """
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        resolved.relative_to(root_resolved)
        return True
    except ValueError:
        return False


def _check_symlink_safety(path: Path, workspace: Path) -> tuple[bool, str | None]:
    """Check if a symlink is safe to include.

    Args:
        path: Path to check.
        workspace: Workspace root.

    Returns:
        Tuple of (is_safe, reason_if_unsafe).
    """
    if not path.is_symlink():
        return True, None

    # Resolve the symlink target
    try:
        target = path.resolve()
    except (OSError, RuntimeError) as e:
        return False, f"Failed to resolve symlink: {e}"

    # Check if target is within workspace
    if not _is_path_contained(target, workspace):
        return False, f"Symlink points outside workspace: {target}"

    return True, None


def _collect_files(workspace: Path) -> tuple[list[Path], list[Path]]:
    """Collect files to include in export.

    Args:
        workspace: Workspace root.

    Returns:
        Tuple of (included_files, rejected_files).

    Raises:
        AlgorithmError: If required files are missing or symlinks escape workspace.
    """
    included: list[Path] = []
    rejected: list[Path] = []

    def check_and_add(path: Path, required: bool = False) -> None:
        """Check a file and add to included/rejected lists.

        Args:
            path: Path to check.
            required: If True, raise error if file is unsafe.
        """
        if not path.exists():
            if required:
                raise AlgorithmError(f"Required file not found: {path.relative_to(workspace)}")
            return

        # Check symlink safety
        is_safe, reason = _check_symlink_safety(path, workspace)
        if not is_safe:
            if required:
                raise AlgorithmError(f"Required file is unsafe: {path.relative_to(workspace)} - {reason}")
            rejected.append(path)
            return

        # Check path containment (resolved path must be in workspace)
        if not _is_path_contained(path, workspace):
            if required:
                raise AlgorithmError(f"Required file escapes workspace: {path.relative_to(workspace)}")
            rejected.append(path)
            return

        included.append(path)

    # Required files
    for filename in REQUIRED_WORKSPACE_FILES:
        check_and_add(workspace / filename, required=True)

    # Required dist files
    dist_dir = workspace / "dist"
    for filename in REQUIRED_DIST_FILES:
        path = dist_dir / filename
        if not path.exists():
            raise AlgorithmError(
                f"Required dist file not found: dist/{filename}. "
                f"Run 'fraclab-sdk algo compile' first."
            )
        check_and_add(path, required=True)

    # Optional files
    readme = workspace / "README.md"
    if readme.exists():
        check_and_add(readme, required=False)

    # Optional directories (schema, examples)
    for dir_name in ["schema", "examples"]:
        dir_path = workspace / dir_name
        if dir_path.is_dir():
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    # Check symlink and containment first
                    is_safe, _ = _check_symlink_safety(file_path, workspace)
                    if not is_safe:
                        rejected.append(file_path)
                        continue

                    if not _is_path_contained(file_path, workspace):
                        rejected.append(file_path)
                        continue

                    # Then check standard rejection patterns
                    if _should_reject(file_path, workspace):
                        rejected.append(file_path)
                    else:
                        included.append(file_path)

    return included, rejected


def _validate_workspace(workspace: Path) -> None:
    """Validate workspace has required structure.

    Args:
        workspace: Workspace path.

    Raises:
        AlgorithmError: If validation fails.
    """
    if not workspace.is_dir():
        raise AlgorithmError(f"Workspace not found: {workspace}")

    for filename in REQUIRED_WORKSPACE_FILES:
        if not (workspace / filename).exists():
            raise AlgorithmError(f"Required file not found: {filename}")

    dist_dir = workspace / "dist"
    if not dist_dir.is_dir():
        raise AlgorithmError(
            "dist/ directory not found. Run 'fraclab-sdk algo compile' first."
        )

    for filename in REQUIRED_DIST_FILES:
        if not (dist_dir / filename).exists():
            raise AlgorithmError(
                f"dist/{filename} not found. Run 'fraclab-sdk algo compile' first."
            )


def export_algorithm_package(
    workspace: Path,
    output: Path,
    auto_compile: bool = False,
    bundle_path: Path | None = None,
) -> ExportResult:
    """Export algorithm workspace as a distributable package.

    Args:
        workspace: Path to algorithm workspace.
        output: Output path (.zip file or directory).
        auto_compile: If True, run compile before export if dist/ is missing.
        bundle_path: Bundle path for auto-compile.

    Returns:
        ExportResult with export details.

    Raises:
        AlgorithmError: If export fails.
    """
    workspace = Path(workspace).resolve()
    output = Path(output).resolve()

    # Auto-compile if needed
    if auto_compile:
        dist_dir = workspace / "dist"
        needs_compile = not dist_dir.exists() or not all(
            (dist_dir / f).exists() for f in REQUIRED_DIST_FILES
        )
        if needs_compile:
            from fraclab_sdk.devkit.compile import compile_algorithm

            compile_algorithm(workspace, bundle_path=bundle_path)

    # Validate workspace
    _validate_workspace(workspace)

    # Collect files
    included, rejected = _collect_files(workspace)

    # Export
    if output.suffix == ".zip":
        _export_to_zip(workspace, output, included)
    else:
        _export_to_dir(workspace, output, included)

    return ExportResult(
        output_path=output,
        files_included=[str(f.relative_to(workspace)) for f in included],
        files_rejected=[str(f.relative_to(workspace)) for f in rejected],
    )


def _export_to_zip(workspace: Path, output: Path, files: list[Path]) -> None:
    """Export files to a zip archive.

    Args:
        workspace: Workspace root.
        output: Output zip path.
        files: Files to include.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            rel_path = file_path.relative_to(workspace)
            zf.write(file_path, rel_path)


def _export_to_dir(workspace: Path, output: Path, files: list[Path]) -> None:
    """Export files to a directory.

    Args:
        workspace: Workspace root.
        output: Output directory path.
        files: Files to include.
    """
    if output.exists():
        shutil.rmtree(output)

    output.mkdir(parents=True)

    for file_path in files:
        rel_path = file_path.relative_to(workspace)
        dest = output / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)


__all__ = ["export_algorithm_package", "ExportResult"]
