"""Hash computation utilities."""

import hashlib
from pathlib import Path


def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of bytes data.

    Args:
        data: Bytes to hash.

    Returns:
        Hex-encoded SHA256 hash string.
    """
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        path: Path to file.

    Returns:
        Hex-encoded SHA256 hash string.
    """
    return compute_sha256(path.read_bytes())
