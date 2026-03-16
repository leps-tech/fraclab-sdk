"""Generic JSON-backed key→entry index store."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar

from fraclab_sdk.utils.io import atomic_write_json

MetaT = TypeVar("MetaT")


class JsonIndexStore(Generic[MetaT]):
    """File-backed JSON dictionary with typed meta conversion.

    Handles _load / _save / add / remove / get / list_all / contains
    so that domain-specific index classes only need to supply three callables:

    - ``make_key(meta) -> str``
    - ``to_entry(meta) -> dict``
    - ``from_entry(entry) -> MetaT``
    """

    def __init__(
        self,
        directory: Path,
        *,
        make_key: Callable[[MetaT], str],
        to_entry: Callable[[MetaT], dict[str, Any]],
        from_entry: Callable[[dict[str, Any]], MetaT],
    ) -> None:
        self._dir = directory
        self._index_path = directory / "index.json"
        self._make_key = make_key
        self._to_entry = to_entry
        self._from_entry = from_entry

    def _load(self) -> dict[str, dict]:
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text())

    def _save(self, data: dict[str, dict]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self._index_path, data)

    def add(self, meta: MetaT) -> None:
        data = self._load()
        data[self._make_key(meta)] = self._to_entry(meta)
        self._save(data)

    def remove(self, key: str) -> None:
        data = self._load()
        if key in data:
            del data[key]
            self._save(data)

    def get(self, key: str) -> MetaT | None:
        data = self._load()
        entry = data.get(key)
        if entry is None:
            return None
        return self._from_entry(entry)

    def list_all(self) -> list[MetaT]:
        return [self._from_entry(e) for e in self._load().values()]

    def contains(self, key: str) -> bool:
        return key in self._load()
