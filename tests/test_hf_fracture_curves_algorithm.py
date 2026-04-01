from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from fraclab_sdk.devkit.compile import compile_algorithm
from fraclab_sdk.runtime.runner_main import run_algorithm


def _write_test_bundle(bundle_dir: Path) -> None:
    data_dir = bundle_dir / "data" / "samples_core_stage_test" / "parquet" / "item-00000"
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_hz = 2000.0
    seconds = np.arange(0.0, 180.0, 1.0 / sample_hz)
    ts_us = (seconds * 1_000_000.0).astype(np.int64)
    baseline = 30.0 + 3.0 * np.sin(2 * np.pi * seconds / 60.0)
    burst_gate = ((seconds > 40.0) & (seconds < 70.0)) | ((seconds > 110.0) & (seconds < 145.0))
    fracture_component = burst_gate.astype(float) * (
        0.8 * np.sin(2 * np.pi * 2.5 * seconds) + 0.4 * np.sin(2 * np.pi * 5.0 * seconds)
    )
    noise = 0.08 * np.sin(2 * np.pi * 13.0 * seconds)
    pressure = baseline + fracture_component + noise

    frame = pd.DataFrame(
        {
            "ts_us": ts_us,
            "signal": "pressure_gauge.pressure",
            "value": pressure,
            "period_us": int(round(1_000_000.0 / sample_hz)),
        }
    )
    frame.to_parquet(data_dir / "part-00000.parquet", index=False)

    ds = {
        "datasets": [
            {
                "key": "samples_core_stage_test",
                "resource": "ts.samples_core.stage",
                "layout": "frame_parquet_item_dirs",
                "items": [
                    {
                        "owner": {
                            "stageId": "stage-test",
                            "wellId": "well-test",
                            "platformId": "platform-test",
                        },
                        "resolutionParams": {
                            "deviceId": "device-test",
                            "signal": "pressure_gauge.pressure",
                        },
                        "range": {"kind": "all"},
                    }
                ],
            }
        ]
    }
    drs = {
        "datasets": [
            {
                "key": "samples_core_stage_test",
                "resource": "ts.samples_core.stage",
                "cardinality": "one",
                "sampling": {"deliveryHz": 200},
                "select": ["ts_us", "signal", "value", "period_us"],
            }
        ]
    }
    manifest = {
        "bundleVersion": "1.0.0",
        "createdAtUs": 1_700_000_000_000_000,
        "specFiles": {
            "dsPath": "ds.json",
            "drsPath": "drs.json",
            "dsSha256": "test",
            "drsSha256": "test",
            "dsBytes": 0,
            "drsBytes": 0,
        },
        "dataRoot": "data",
        "datasets": {
            "samples_core_stage_test": {
                "layout": "frame_parquet_item_dirs",
                "count": 1,
            }
        },
    }

    (bundle_dir / "ds.json").write_text(json.dumps(ds, indent=2), encoding="utf-8")
    (bundle_dir / "drs.json").write_text(json.dumps(drs, indent=2), encoding="utf-8")
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_hf_fracture_curves_algorithm_runs_end_to_end(tmp_path) -> None:
    workspace_src = Path("algorithms/hf-fracture-curves/0.1.0")
    workspace = tmp_path / "algo"
    shutil.copytree(workspace_src, workspace)

    bundle_dir = tmp_path / "bundle"
    _write_test_bundle(bundle_dir)
    compile_algorithm(workspace, bundle_path=bundle_dir)

    run_dir = tmp_path / "run"
    input_dir = run_dir / "input"
    shutil.copytree(bundle_dir, input_dir)
    (input_dir / "params.json").write_text(json.dumps({}), encoding="utf-8")
    (input_dir / "run_context.json").write_text(
        json.dumps(
            {
                "runId": "run-test",
                "snapshotId": "snapshot-test",
                "algorithmId": "hf-fracture-curves",
                "algorithmVersion": "0.1.0",
                "contractVersion": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    exit_code = run_algorithm(run_dir, workspace / "main.py")
    assert exit_code == 0

    summary_json = run_dir / "output" / "artifacts" / "summary.json"
    overview_png = run_dir / "output" / "artifacts" / "overview_plot.png"
    assert summary_json.exists()
    assert overview_png.exists()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["curveFamily"] == "hf_pressure_fracture_proxies"
    assert summary["analysisSampleRateHz"] > 0.0
    assert summary["metrics"]["fractureFrequencyP90"] >= 0.0
