"""Snapshot index management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fraclab_sdk.utils.json_index_store import JsonIndexStore


@dataclass
class SnapshotMeta:
    """Metadata for an indexed snapshot."""

    snapshot_id: str
    bundle_id: str
    created_at: str
    description: str | None = None
    imported_at: str = field(default_factory=lambda: datetime.now().isoformat())


def _to_entry(meta: SnapshotMeta) -> dict:
    return {
        "snapshot_id": meta.snapshot_id,
        "bundle_id": meta.bundle_id,
        "created_at": meta.created_at,
        "description": meta.description,
        "imported_at": meta.imported_at,
    }


def _from_entry(entry: dict) -> SnapshotMeta:
    return SnapshotMeta(
        snapshot_id=entry["snapshot_id"],
        bundle_id=entry["bundle_id"],
        created_at=entry["created_at"],
        description=entry.get("description"),
        imported_at=entry.get("imported_at", ""),
    )


class SnapshotIndex:
    """Manages the snapshot index file."""

    def __init__(self, snapshots_dir: Path) -> None:
        self._store: JsonIndexStore[SnapshotMeta] = JsonIndexStore(
            snapshots_dir,
            make_key=lambda m: m.snapshot_id,
            to_entry=_to_entry,
            from_entry=_from_entry,
        )

    def add(self, meta: SnapshotMeta) -> None:
        self._store.add(meta)

    def remove(self, snapshot_id: str) -> None:
        self._store.remove(snapshot_id)

    def get(self, snapshot_id: str) -> SnapshotMeta | None:
        return self._store.get(snapshot_id)

    def list_all(self) -> list[SnapshotMeta]:
        return self._store.list_all()

    def contains(self, snapshot_id: str) -> bool:
        return self._store.contains(snapshot_id)
