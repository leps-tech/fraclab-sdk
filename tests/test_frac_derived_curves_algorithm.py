from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from fraclab_sdk.devkit.compile import compile_algorithm
from fraclab_sdk.runtime.runner_main import run_algorithm


def _write_test_bundle(bundle_dir: Path) -> None:
    data_dir = bundle_dir / "data" / "fracRecord_stage_test" / "parquet" / "item-00000"
    data_dir.mkdir(parents=True, exist_ok=True)

    seconds = np.arange(0, 480, dtype=float)
    ts_us = ((seconds + 1_700_000_000.0) * 1_000_000.0).astype(np.int64)

    slurry_rate = np.piecewise(
        seconds,
        [seconds < 40, (seconds >= 40) & (seconds < 320), seconds >= 320],
        [0.0, lambda x: 8.0 + 0.03 * (x - 40.0), 0.0],
    )
    proppant_ratio = np.piecewise(
        seconds,
        [seconds < 100, (seconds >= 100) & (seconds < 300), seconds >= 300],
        [0.0, lambda x: np.minimum(18.0, (x - 100.0) * 0.09), 0.0],
    )
    section_fluid_volume = np.clip(np.cumsum(slurry_rate) / 60.0, 0.0, None)
    base_pressure = 28.0 + 0.18 * slurry_rate ** 2 + 0.06 * proppant_ratio + 0.012 * seconds
    anomaly = np.where((seconds >= 180) & (seconds <= 240), 4.0 * np.sin((seconds - 180.0) / 60.0 * np.pi), 0.0)
    treating_pressure = base_pressure + anomaly

    frame = pd.DataFrame(
        {
            "ts_us": ts_us,
            "treatingPressure": treating_pressure,
            "slurryRate": slurry_rate,
            "proppantRatio": proppant_ratio,
            "sectionFluidVolume": section_fluid_volume,
        }
    )
    frame.to_parquet(data_dir / "part-00000.parquet", index=False)

    ds = {
        "datasets": [
            {
                "key": "fracRecord_stage_test",
                "resource": "record.fracRecord.stage",
                "layout": "frame_parquet_item_dirs",
                "items": [
                    {
                        "owner": {
                            "stageId": "stage-test",
                            "wellId": "well-test",
                            "platformId": "platform-test",
                        },
                        "resolutionParams": {"metadata.metadataId": "meta-test"},
                        "range": {"kind": "all"},
                    }
                ],
            }
        ]
    }
    drs = {
        "datasets": [
            {
                "key": "fracRecord_stage_test",
                "resource": "record.fracRecord.stage",
                "cardinality": "one",
                "select": [
                    "ts_us",
                    "treatingPressure",
                    "slurryRate",
                    "proppantRatio",
                    "sectionFluidVolume",
                ],
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
            "fracRecord_stage_test": {
                "layout": "frame_parquet_item_dirs",
                "count": 1,
            }
        },
    }

    (bundle_dir / "ds.json").write_text(json.dumps(ds, indent=2), encoding="utf-8")
    (bundle_dir / "drs.json").write_text(json.dumps(drs, indent=2), encoding="utf-8")
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_frac_derived_curves_algorithm_runs_end_to_end(tmp_path) -> None:
    workspace_src = Path("algorithms/frac-derived-curves/0.1.0")
    workspace = tmp_path / "algo"
    shutil.copytree(workspace_src, workspace)

    bundle_dir = tmp_path / "bundle"
    _write_test_bundle(bundle_dir)
    compile_algorithm(workspace, bundle_path=bundle_dir)

    run_dir = tmp_path / "run"
    input_dir = run_dir / "input"
    shutil.copytree(bundle_dir, input_dir)
    (input_dir / "params.json").write_text(
        json.dumps({"emitPlot": True}),
        encoding="utf-8",
    )
    (input_dir / "run_context.json").write_text(
        json.dumps(
            {
                "runId": "run-test",
                "snapshotId": "snapshot-test",
                "algorithmId": "frac-derived-curves",
                "algorithmVersion": "0.1.0",
                "contractVersion": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    exit_code = run_algorithm(run_dir, workspace / "main.py")
    assert exit_code == 0

    manifest = json.loads((run_dir / "output" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"

    summary_json = run_dir / "output" / "artifacts" / "summary.json"
    diagnostics_json = run_dir / "output" / "artifacts" / "diagnostics.json"
    overview_png = run_dir / "output" / "artifacts" / "overview_plot.png"

    assert summary_json.exists()
    assert diagnostics_json.exists()
    assert overview_png.exists()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["curveFamily"] == "offline_frac_derived_curves"
    assert summary["datasetKey"] == "fracRecord_stage_test"
    assert summary["alignedMetrics"]["perforationEfficiency"]["Model_Friction_MPa"] >= 0.0
