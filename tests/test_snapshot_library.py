import json
from pathlib import Path

import pytest

from fraclab_sdk.config import SDKConfig
from fraclab_sdk.errors import SnapshotError
from fraclab_sdk.materialize.hash import compute_sha256
from fraclab_sdk.snapshot import SnapshotHandle, SnapshotLibrary


def _write_snapshot_source(
    tmp_path: Path,
    *,
    ds_path: str = "ds.json",
    drs_path: str = "drs.json",
    data_root: str = "data",
) -> Path:
    source_dir = tmp_path / "bundle"
    source_dir.mkdir()

    ds_bytes = json.dumps({"schemaVersion": "1.0", "datasets": []}).encode("utf-8")
    drs_bytes = json.dumps({"schemaVersion": "1.0", "datasets": []}).encode("utf-8")

    (source_dir / "ds.json").write_bytes(ds_bytes)
    (source_dir / "drs.json").write_bytes(drs_bytes)
    (source_dir / "data").mkdir()

    manifest = {
        "bundleVersion": "1.0.0",
        "createdAtUs": 0,
        "specFiles": {
            "dsPath": ds_path,
            "drsPath": drs_path,
            "dsSha256": compute_sha256(ds_bytes),
            "drsSha256": compute_sha256(drs_bytes),
        },
        "dataRoot": data_root,
        "datasets": {},
    }
    (source_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return source_dir


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("ds_path", "../ds.json"),
        ("drs_path", "../drs.json"),
        ("data_root", "../data"),
    ],
)
def test_import_snapshot_rejects_unsafe_manifest_paths(
    tmp_path: Path,
    field_name: str,
    value: str,
) -> None:
    source_dir = _write_snapshot_source(tmp_path, **{field_name: value})
    snapshot_lib = SnapshotLibrary(SDKConfig(tmp_path / "sdk-home"))

    with pytest.raises(SnapshotError, match="Unsafe manifest path"):
        snapshot_lib.import_snapshot(source_dir)


def test_snapshot_handle_rejects_unsafe_manifest_paths_on_access(tmp_path: Path) -> None:
    source_dir = _write_snapshot_source(tmp_path, ds_path="../ds.json")
    snapshot = SnapshotHandle(source_dir)

    with pytest.raises(SnapshotError, match="Unsafe manifest path specFiles.dsPath"):
        _ = snapshot.dataspec
