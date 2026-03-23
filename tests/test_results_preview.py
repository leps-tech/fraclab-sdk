import json

from fraclab_sdk.models import ArtifactInfo
from fraclab_sdk.results.preview import preview_json_raw, preview_json_table
from fraclab_sdk.results.reader import ResultReader


def test_preview_json_raw_reads_gb18030_encoded_artifact(tmp_path) -> None:
    artifact_path = tmp_path / "summary.json"
    artifact_path.write_bytes(json.dumps({"text": "测试"}, ensure_ascii=False).encode("gb18030"))
    artifact = ArtifactInfo(
        artifactKey="summary",
        type="json",
        uri=artifact_path.resolve().as_uri(),
    )

    preview = preview_json_raw(artifact)

    assert preview is not None
    assert '"text": "测试"' in preview


def test_preview_json_table_reads_legacy_encoded_artifact(tmp_path) -> None:
    artifact_path = tmp_path / "table.json"
    artifact_path.write_bytes(
        json.dumps([{"name": "测试", "value": 1}], ensure_ascii=False).encode("gb18030")
    )
    artifact = ArtifactInfo(
        artifactKey="table",
        type="json",
        uri=artifact_path.resolve().as_uri(),
    )

    table = preview_json_table(artifact)

    assert table == {
        "columns": ["name", "value"],
        "rows": [["测试", 1]],
    }


def test_result_reader_reads_gb18030_encoded_json_artifact(tmp_path) -> None:
    run_dir = tmp_path / "run"
    output_dir = run_dir / "output"
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)

    artifact_path = artifacts_dir / "summary.json"
    artifact_path.write_bytes(json.dumps({"text": "测试"}, ensure_ascii=False).encode("gb18030"))

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run": {
                    "runId": "run-1",
                    "algorithmId": "algo-1",
                },
                "datasets": [
                    {
                        "datasetKey": "summary",
                        "items": [
                            {
                                "artifact": {
                                    "artifactKey": "summary",
                                    "type": "json",
                                    "uri": artifact_path.resolve().as_uri(),
                                }
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    reader = ResultReader(run_dir)

    assert reader.read_artifact_json("summary") == {"text": "测试"}
