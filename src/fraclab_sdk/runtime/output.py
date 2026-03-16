"""Output client for algorithm runtime."""

from __future__ import annotations

import json
import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fraclab_sdk.errors import OutputContainmentError


@dataclass
class OutputRecord:
    """Record of a written or registered output."""

    dataset_key: str
    owner: dict[str, Any] | None
    dims: dict[str, Any] | None
    meta: dict[str, Any] | None
    inline: dict[str, Any] | None
    item_key: str | None
    artifact_key: str
    artifact_type: str  # "scalar", "blob", "json"
    file_path: Path | None = None
    mime_type: str | None = None
    value: Any = None


class OutputClient:
    """Client for algorithm outputs under ``run/output/artifacts``."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir.resolve()
        self._dir = self._output_dir / "artifacts"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._records: list[OutputRecord] = []

    @property
    def dir(self) -> Path:
        """Return the root directory for algorithm-visible output files."""
        return self._dir

    @property
    def output_dir(self) -> Path:
        """Return the run output directory containing manifest and logs."""
        return self._output_dir

    def _validate_path(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self._dir)
        except ValueError:
            raise OutputContainmentError(str(resolved), str(self._dir)) from None
        return resolved

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return self._validate_path(candidate)
        return self._validate_path(self._dir / candidate)

    def path(self, relative_path: str | Path) -> Path:
        """Return a validated file path under ``output/artifacts``."""
        return self._resolve_path(relative_path)

    def _append_record(
        self,
        *,
        artifact_key: str,
        artifact_type: str,
        dataset_key: str,
        owner: dict[str, Any] | None,
        dims: dict[str, Any] | None,
        meta: dict[str, Any] | None,
        inline: dict[str, Any] | None,
        item_key: str | None,
        file_path: Path | None = None,
        mime_type: str | None = None,
        value: Any = None,
    ) -> None:
        self._records.append(
            OutputRecord(
                dataset_key=dataset_key,
                owner=owner,
                dims=dims,
                meta=meta,
                inline=inline,
                item_key=item_key,
                artifact_key=artifact_key,
                artifact_type=artifact_type,
                file_path=file_path.resolve() if file_path is not None else None,
                mime_type=mime_type,
                value=value,
            )
        )

    def write_scalar(
        self,
        artifact_key: str,
        value: Any,
        *,
        dataset_key: str = "artifacts",
        owner: dict[str, Any] | None = None,
        dims: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        inline: dict[str, Any] | None = None,
        item_key: str | None = None,
    ) -> None:
        """Write a scalar output entry without a backing file."""
        self._append_record(
            artifact_key=artifact_key,
            artifact_type="scalar",
            dataset_key=dataset_key,
            owner=owner,
            dims=dims,
            meta=meta,
            inline=inline,
            item_key=item_key,
            value=value,
        )

    def write_text(
        self,
        artifact_key: str,
        text: str,
        filename: str | None = None,
        *,
        encoding: str = "utf-8",
        mime_type: str = "text/plain",
        dataset_key: str = "artifacts",
        owner: dict[str, Any] | None = None,
        dims: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        inline: dict[str, Any] | None = None,
        item_key: str | None = None,
    ) -> Path:
        """Write a text file under ``output/artifacts`` and register it."""
        file_path = self.path(filename or f"{artifact_key}.txt")
        file_path.write_text(text, encoding=encoding)
        self._append_record(
            artifact_key=artifact_key,
            artifact_type="blob",
            dataset_key=dataset_key,
            owner=owner,
            dims=dims,
            meta=meta,
            inline=inline,
            item_key=item_key,
            file_path=file_path,
            mime_type=mime_type,
        )
        return file_path

    def write_json(
        self,
        artifact_key: str,
        data: Any,
        filename: str | None = None,
        *,
        dataset_key: str = "artifacts",
        owner: dict[str, Any] | None = None,
        dims: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        inline: dict[str, Any] | None = None,
        item_key: str | None = None,
    ) -> Path:
        """Write a JSON output file and register it."""
        file_path = self.path(filename or f"{artifact_key}.json")
        content = json.dumps(data, indent=2, ensure_ascii=False)
        file_path.write_text(content, encoding="utf-8")
        self._append_record(
            artifact_key=artifact_key,
            artifact_type="json",
            dataset_key=dataset_key,
            owner=owner,
            dims=dims,
            meta=meta,
            inline=inline,
            item_key=item_key,
            file_path=file_path,
            mime_type="application/json",
        )
        return file_path

    def write_bytes(
        self,
        artifact_key: str,
        data: bytes,
        filename: str,
        mime_type: str | None = None,
        *,
        dataset_key: str = "artifacts",
        owner: dict[str, Any] | None = None,
        dims: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        inline: dict[str, Any] | None = None,
        item_key: str | None = None,
    ) -> Path:
        """Write a binary output file and register it."""
        file_path = self.path(filename)
        file_path.write_bytes(data)
        self._append_record(
            artifact_key=artifact_key,
            artifact_type="blob",
            dataset_key=dataset_key,
            owner=owner,
            dims=dims,
            meta=meta,
            inline=inline,
            item_key=item_key,
            file_path=file_path,
            mime_type=mime_type,
        )
        return file_path

    def write_blob(
        self,
        artifact_key: str,
        data: bytes,
        filename: str,
        mime_type: str | None = None,
        *,
        dataset_key: str = "artifacts",
        owner: dict[str, Any] | None = None,
        dims: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        inline: dict[str, Any] | None = None,
        item_key: str | None = None,
    ) -> Path:
        """Write a binary output file and register it."""
        return self.write_bytes(
            artifact_key,
            data,
            filename,
            mime_type,
            dataset_key=dataset_key,
            owner=owner,
            dims=dims,
            meta=meta,
            inline=inline,
            item_key=item_key,
        )

    def register_file(
        self,
        artifact_key: str,
        file_path: Path,
        mime_type: str | None = None,
        *,
        dataset_key: str = "artifacts",
        owner: dict[str, Any] | None = None,
        dims: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        inline: dict[str, Any] | None = None,
        item_key: str | None = None,
    ) -> Path:
        """Register an existing file already written under ``output/artifacts``."""
        resolved = self._resolve_path(file_path)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"Output file not found: {resolved}")
        self._append_record(
            artifact_key=artifact_key,
            artifact_type="json" if resolved.suffix.lower() == ".json" else "blob",
            dataset_key=dataset_key,
            owner=owner,
            dims=dims,
            meta=meta,
            inline=inline,
            item_key=item_key,
            file_path=resolved,
            mime_type=mime_type or mimetypes.guess_type(resolved.name)[0] or "application/octet-stream",
        )
        return resolved

    def write_file(
        self,
        artifact_key: str,
        source_path: Path,
        filename: str | None = None,
        mime_type: str | None = None,
        *,
        dataset_key: str = "artifacts",
        owner: dict[str, Any] | None = None,
        dims: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        inline: dict[str, Any] | None = None,
        item_key: str | None = None,
    ) -> Path:
        """Copy or register a file as an output artifact."""
        resolved_source = source_path.resolve()
        try:
            relative_source = resolved_source.relative_to(self._dir)
        except ValueError:
            relative_source = None

        if relative_source is not None and filename is None:
            dest_path = resolved_source
        else:
            dest_path = self.path(filename or source_path.name)
        if resolved_source != dest_path:
            shutil.copy2(resolved_source, dest_path)
        return self.register_file(
            artifact_key,
            dest_path,
            mime_type=mime_type,
            dataset_key=dataset_key,
            owner=owner,
            dims=dims,
            meta=meta,
            inline=inline,
            item_key=item_key,
        )

    def write_dataframe_csv(
        self,
        artifact_key: str,
        dataframe: Any,
        filename: str | None = None,
        *,
        index: bool = False,
        encoding: str = "utf-8-sig",
        dataset_key: str = "artifacts",
        owner: dict[str, Any] | None = None,
        dims: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        inline: dict[str, Any] | None = None,
        item_key: str | None = None,
    ) -> Path:
        """Write a dataframe as CSV and register it."""
        file_path = self.path(filename or f"{artifact_key}.csv")
        dataframe.to_csv(file_path, index=index, encoding=encoding)
        self._append_record(
            artifact_key=artifact_key,
            artifact_type="blob",
            dataset_key=dataset_key,
            owner=owner,
            dims=dims,
            meta=meta,
            inline=inline,
            item_key=item_key,
            file_path=file_path,
            mime_type="text/csv",
        )
        return file_path

    def write_figure_png(
        self,
        artifact_key: str,
        figure: Any,
        filename: str | None = None,
        *,
        dpi: int = 300,
        bbox_inches: str | None = "tight",
        close: bool = False,
        dataset_key: str = "artifacts",
        owner: dict[str, Any] | None = None,
        dims: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        inline: dict[str, Any] | None = None,
        item_key: str | None = None,
    ) -> Path:
        """Write a matplotlib figure as PNG and register it."""
        file_path = self.path(filename or f"{artifact_key}.png")
        figure.savefig(file_path, format="png", dpi=dpi, bbox_inches=bbox_inches)
        if close:
            import matplotlib.pyplot as plt

            plt.close(figure)
        self._append_record(
            artifact_key=artifact_key,
            artifact_type="blob",
            dataset_key=dataset_key,
            owner=owner,
            dims=dims,
            meta=meta,
            inline=inline,
            item_key=item_key,
            file_path=file_path,
            mime_type="image/png",
        )
        return file_path

    def get_records(self) -> list[OutputRecord]:
        """Get all explicit output records."""
        return self._records.copy()

    def _auto_record_for_path(self, file_path: Path) -> OutputRecord:
        relative_key = file_path.relative_to(self._dir).as_posix()
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        artifact_type = "json" if file_path.suffix.lower() == ".json" else "blob"
        return OutputRecord(
            dataset_key="artifacts",
            owner=None,
            dims=None,
            meta=None,
            inline=None,
            item_key=relative_key,
            artifact_key=relative_key,
            artifact_type=artifact_type,
            file_path=file_path.resolve(),
            mime_type=mime_type,
        )

    def _discover_untracked_files(self) -> list[OutputRecord]:
        tracked_paths = {
            rec.file_path.resolve()
            for rec in self._records
            if rec.file_path is not None
        }
        return [
            self._auto_record_for_path(path)
            for path in sorted(self._dir.rglob("*"))
            if path.is_file() and path.resolve() not in tracked_paths
        ]

    @staticmethod
    def _record_to_item(record: OutputRecord) -> dict[str, Any]:
        artifact: dict[str, Any] = {
            "artifactKey": record.artifact_key,
            "type": record.artifact_type,
        }
        if record.mime_type:
            artifact["mimeType"] = record.mime_type
        if record.file_path is not None:
            artifact["uri"] = record.file_path.resolve().as_uri()
        if record.value is not None:
            artifact["value"] = record.value
        if record.inline is not None:
            artifact["inline"] = record.inline

        item: dict[str, Any] = {
            "itemKey": record.item_key or record.artifact_key,
            "artifact": artifact,
        }
        if record.owner:
            item["owner"] = record.owner
        if record.dims:
            item["dims"] = record.dims
        if record.meta:
            item["meta"] = record.meta
        if record.inline is not None:
            item["inline"] = record.inline
        return item

    def build_manifest_datasets(self) -> list[dict[str, Any]]:
        """Build the run output manifest datasets structure."""
        records = [*self._records, *self._discover_untracked_files()]
        by_dataset: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_dataset.setdefault(record.dataset_key, []).append(self._record_to_item(record))
        return [
            {"datasetKey": dataset_key, "items": items}
            for dataset_key, items in by_dataset.items()
        ]


__all__ = ["OutputClient", "OutputRecord"]
