"""Snapshot index management."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fraclab_sdk.utils.io import atomic_write_json


@dataclass
class SnapshotMeta:
    """Metadata for an indexed snapshot."""

    snapshot_id: str
    bundle_id: str
    created_at: str
    description: str | None = None
    imported_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SnapshotIndex:
    """Manages the snapshot index file."""

    def __init__(self, snapshots_dir: Path) -> None:
        """Initialize snapshot index.

        Args:
            snapshots_dir: Directory containing snapshots.
        """
        self._snapshots_dir = snapshots_dir
        self._index_path = snapshots_dir / "index.json"

    def _load(self) -> dict[str, dict]:
        """Load index from disk."""
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text())

    def _save(self, data: dict[str, dict]) -> None:
        """Save index to disk."""
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self._index_path, data)

    def add(self, meta: SnapshotMeta) -> None:
        """Add a snapshot to the index."""
        data = self._load()
        data[meta.snapshot_id] = {
            "snapshot_id": meta.snapshot_id,
            "bundle_id": meta.bundle_id,
            "created_at": meta.created_at,
            "description": meta.description,
            "imported_at": meta.imported_at,
        }
        self._save(data)

    def remove(self, snapshot_id: str) -> None:
        """Remove a snapshot from the index."""
        data = self._load()
        if snapshot_id in data:
            del data[snapshot_id]
            self._save(data)

    def get(self, snapshot_id: str) -> SnapshotMeta | None:
        """Get snapshot metadata by ID."""
        data = self._load()
        if snapshot_id not in data:
            return None
        entry = data[snapshot_id]
        return SnapshotMeta(
            snapshot_id=entry["snapshot_id"],
            bundle_id=entry["bundle_id"],
            created_at=entry["created_at"],
            description=entry.get("description"),
            imported_at=entry.get("imported_at", ""),
        )

    def list_all(self) -> list[SnapshotMeta]:
        """List all indexed snapshots."""
        data = self._load()
        return [
            SnapshotMeta(
                snapshot_id=entry["snapshot_id"],
                bundle_id=entry["bundle_id"],
                created_at=entry["created_at"],
                description=entry.get("description"),
                imported_at=entry.get("imported_at", ""),
            )
            for entry in data.values()
        ]

    def contains(self, snapshot_id: str) -> bool:
        """Check if a snapshot is in the index."""
        return snapshot_id in self._load()
