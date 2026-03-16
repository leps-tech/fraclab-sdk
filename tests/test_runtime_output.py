from fraclab_sdk.runtime.output import OutputClient


def test_output_client_writes_json_with_standard_file_uri(tmp_path) -> None:
    client = OutputClient(tmp_path / "output")

    output_path = client.write_json("summary", {"ok": True}, dataset_key="summary")
    datasets = client.build_manifest_datasets()

    assert output_path == (tmp_path / "output" / "artifacts" / "summary.json")
    assert datasets == [
        {
            "datasetKey": "summary",
            "items": [
                {
                    "itemKey": "summary",
                    "artifact": {
                        "artifactKey": "summary",
                        "type": "json",
                        "mimeType": "application/json",
                        "uri": output_path.resolve().as_uri(),
                    },
                }
            ],
        }
    ]


def test_output_client_registers_existing_files_and_discovers_untracked_files(tmp_path) -> None:
    client = OutputClient(tmp_path / "output")

    plot_path = client.path("plot.png")
    plot_path.write_bytes(b"png")
    registered_path = client.write_file("plot", plot_path, mime_type="image/png")

    debug_path = client.path("debug.txt")
    debug_path.write_text("debug", encoding="utf-8")

    datasets = client.build_manifest_datasets()
    assert len(datasets) == 1
    assert datasets[0]["datasetKey"] == "artifacts"
    assert registered_path == plot_path

    artifacts = {
        item["artifact"]["artifactKey"]: item["artifact"]
        for item in datasets[0]["items"]
    }
    assert artifacts["plot"]["uri"] == plot_path.resolve().as_uri()
    assert artifacts["plot"]["mimeType"] == "image/png"
    assert artifacts["debug.txt"]["uri"] == debug_path.resolve().as_uri()
    assert artifacts["debug.txt"]["mimeType"] == "text/plain"
