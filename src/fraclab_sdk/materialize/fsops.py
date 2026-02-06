"""File system operations for materialization."""

import os
import shutil
from pathlib import Path


def copy_file_smart(src: Path, dst: Path) -> str:
    """Copy a file using hardlink > symlink > copy fallback strategy.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        Strategy used: "hardlink", "symlink", or "copy".

    Raises:
        FileNotFoundError: If source file doesn't exist.
        OSError: If all copy strategies fail.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    # Ensure parent directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Try hardlink first
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        pass

    # Try symlink
    try:
        os.symlink(src.resolve(), dst)
        return "symlink"
    except OSError:
        pass

    # Fall back to copy
    shutil.copy2(src, dst)
    return "copy"


def copy_directory_smart(src_dir: Path, dst_dir: Path) -> dict[str, int]:
    """Copy directory contents using smart file copy strategy.

    Each file in the source directory (including nested) is copied
    using the hardlink > symlink > copy fallback strategy.

    Args:
        src_dir: Source directory path.
        dst_dir: Destination directory path.

    Returns:
        Dict with counts: {"hardlink": N, "symlink": N, "copy": N}

    Raises:
        FileNotFoundError: If source directory doesn't exist.
    """
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    counts = {"hardlink": 0, "symlink": 0, "copy": 0}

    for src_file in src_dir.rglob("*"):
        if src_file.is_file():
            rel_path = src_file.relative_to(src_dir)
            dst_file = dst_dir / rel_path
            strategy = copy_file_smart(src_file, dst_file)
            counts[strategy] += 1

    return counts


def extract_ndjson_lines(
    src_path: Path,
    dst_path: Path,
    line_indices: list[int],
) -> int:
    """Extract specific lines from ndjson file and write to new file.

    Lines are written in the order of line_indices (which should be sorted).
    Output file has contiguous line numbers (0..N-1).

    Args:
        src_path: Source ndjson file path.
        dst_path: Destination ndjson file path.
        line_indices: List of 0-based line indices to extract (must be sorted).

    Returns:
        Number of lines written.

    Raises:
        FileNotFoundError: If source file doesn't exist.
        IndexError: If line index is out of range.
    """
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    # Ensure parent directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Read source and extract lines
    with src_path.open("r", encoding="utf-8") as f:
        all_lines = f.readlines()

    # Validate indices and extract
    extracted = []
    for idx in line_indices:
        if idx < 0 or idx >= len(all_lines):
            raise IndexError(f"Line index {idx} out of range (0-{len(all_lines)-1})")
        extracted.append(all_lines[idx])

    # Write extracted lines
    with dst_path.open("w", encoding="utf-8") as f:
        for line in extracted:
            # Ensure line ends with newline
            if not line.endswith("\n"):
                line += "\n"
            f.write(line)

    return len(extracted)
