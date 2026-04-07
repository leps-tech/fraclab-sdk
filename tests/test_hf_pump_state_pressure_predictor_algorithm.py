from __future__ import annotations

import json
import shutil
from pathlib import Path

from fraclab_sdk.devkit.compile import compile_algorithm
from fraclab_sdk.runtime.runner_main import run_algorithm
from tests._hf_pressure_bundle_fixture import write_hf_pressure_test_bundle


def test_hf_pump_state_pressure_predictor_runs_end_to_end(tmp_path) -> None:
    workspace_src = Path("algorithms/hf-pump-state-pressure-predictor/0.1.0")
    workspace = tmp_path / "algo"
    shutil.copytree(workspace_src, workspace)

    bundle_dir = tmp_path / "bundle"
    write_hf_pressure_test_bundle(bundle_dir)
    compile_algorithm(workspace, bundle_path=bundle_dir)

    params_schema = json.loads((workspace / "dist" / "params.schema.json").read_text(encoding="utf-8"))
    assert "timeWindows_samples_core_stage_5826" in params_schema["properties"]
    assert params_schema["properties"]["pumpAnalysisHz"]["default"] == 200.0
    assert params_schema["properties"]["predictorAnalysisHz"]["default"] == 20.0
    assert params_schema["properties"]["slow2LevelSec"]["default"] == 2.0
    assert params_schema["properties"]["slow2TrendSec"]["default"] == 4.0

    run_dir = tmp_path / "run"
    input_dir = run_dir / "input"
    shutil.copytree(bundle_dir, input_dir)
    (input_dir / "params.json").write_text(json.dumps({}), encoding="utf-8")
    (input_dir / "run_context.json").write_text(
        json.dumps(
            {
                "runId": "run-test",
                "snapshotId": "snapshot-test",
                "algorithmId": "hf-pump-state-pressure-predictor",
                "algorithmVersion": "0.1.0",
                "contractVersion": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    exit_code = run_algorithm(run_dir, workspace / "main.py")
    assert exit_code == 0

    summary_path = run_dir / "output" / "artifacts" / "summary.json"
    diagnostics_path = run_dir / "output" / "artifacts" / "diagnostics.json"
    overview_plot_path = run_dir / "output" / "artifacts" / "overview_plot.png"
    assert summary_path.exists()
    assert diagnostics_path.exists()
    assert overview_plot_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["curveFamily"] == "hf_pump_state_pressure_predictor_v1"
    assert summary["rows"]["emitted"] > 0
    assert summary["benchmark"]["name"] == "slow2_only_linear_extrapolation"
    assert summary["predictorArchitecture"]["forecastModel"] == "slow2_only_linear_extrapolation"
    assert summary["predictorArchitecture"]["usesPumpFeaturesForForecast"] is False
    assert summary["latentTarget"]["tauSec"] > 0.0
    assert summary["latentTarget"]["measurementSigma"] >= 0.0
    assert summary["decisionWindows"]["gateOpenFraction"] >= 0.0
    assert summary["decisionWindows"]["activeStableFraction"] >= 0.0
    assert summary["metrics"]["forecastAccuracy"]["1s"]["mae"] >= 0.0
    assert summary["metrics"]["forecastAccuracy"]["5s"]["rmse"] >= 0.0
    assert summary["metrics"]["forecastAccuracy"]["10s"]["huber"] >= 0.0
    assert summary["metrics"]["predictedPressureResidualMae"] >= 0.0
    assert "active_stable" in summary["metrics"]["byRegime"]
    assert summary["metrics"]["stateHealth"]["familyEnergyShareMean"] >= 0.0

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert len(diagnostics["timeSeries"]["slow2"]) == summary["rows"]["emitted"]
    assert len(diagnostics["timeSeries"]["latentTarget"]) == summary["rows"]["emitted"]
    assert len(diagnostics["timeSeries"]["forecast5s"]) == summary["rows"]["emitted"]
    assert len(diagnostics["timeSeries"]["pumpActivityState"]) == summary["rows"]["emitted"]


def test_hf_pump_state_pressure_predictor_window_reanchors_time(tmp_path) -> None:
    workspace_src = Path("algorithms/hf-pump-state-pressure-predictor/0.1.0")
    workspace = tmp_path / "algo"
    shutil.copytree(workspace_src, workspace)

    bundle_dir = tmp_path / "bundle"
    write_hf_pressure_test_bundle(bundle_dir)
    compile_algorithm(workspace, bundle_path=bundle_dir)

    run_dir = tmp_path / "run"
    input_dir = run_dir / "input"
    shutil.copytree(bundle_dir, input_dir)
    params = {
        "timeWindows_samples_core_stage_5826": [
            {
                "itemKey": "item00",
                "min": 40_000_000.0,
                "max": 120_000_000.0,
            }
        ]
    }
    (input_dir / "params.json").write_text(json.dumps(params), encoding="utf-8")
    (input_dir / "run_context.json").write_text(
        json.dumps(
            {
                "runId": "run-test-window",
                "snapshotId": "snapshot-test",
                "algorithmId": "hf-pump-state-pressure-predictor",
                "algorithmVersion": "0.1.0",
                "contractVersion": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    exit_code = run_algorithm(run_dir, workspace / "main.py")
    assert exit_code == 0

    summary = json.loads((run_dir / "output" / "artifacts" / "summary.json").read_text(encoding="utf-8"))
    assert summary["selectedWindow"]["requestedItemKey"] == "item00"
    assert 0.0 <= summary["selectedWindow"]["actualStartSec"] <= 0.6
    assert 79.0 <= summary["selectedWindow"]["actualEndSec"] <= 80.5
    assert summary["rows"]["emitted"] > 100
