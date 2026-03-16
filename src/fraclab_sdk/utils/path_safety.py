"""Path safety utilities."""

from __future__ import annotations


def is_safe_relative_path(path: str) -> bool:
    """Check if a path component is safe (no traversal or shell-special chars).

    Returns False for absolute paths, parent-directory references,
    and characters that are invalid or dangerous on common file systems.
    """
    if path.startswith("/") or path.startswith("\\"):
        return False
    if ".." in path:
        return False
    return not any(c in path for c in [":", "*", "?", '"', "<", ">", "|"])
