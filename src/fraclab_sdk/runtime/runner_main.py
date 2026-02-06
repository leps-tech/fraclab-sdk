"""Main entry point for algorithm runner subprocess."""

import importlib.util
import json
import logging
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fraclab_sdk.runtime.artifacts import ArtifactWriter
from fraclab_sdk.runtime.data_client import DataClient


def validate_manifest_against_contract(
    manifest: dict[str, Any],
    contract_path: Path,
    logger: logging.Logger,
) -> tuple[bool, list[str]]:
    """Validate run output manifest against OutputContract.

    The manifest has a flat artifacts list, while the contract has hierarchical
    datasets -> items -> artifacts structure. We validate that all contract
    artifacts are present in the manifest.

    Args:
        manifest: The run output manifest dict.
        contract_path: Path to output_contract.json.
        logger: Logger for diagnostics.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    if not contract_path.exists():
        logger.debug(f"No contract found at {contract_path}, skipping validation")
        return True, []

    try:
        contract = json.loads(contract_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load contract: {e}")
        return True, []  # Don't fail on contract load errors

    errors: list[str] = []

    # Get all artifact keys from manifest (flat list)
    manifest_artifact_keys = {
        a.get("artifactKey") or a.get("key")
        for a in manifest.get("artifacts", [])
    }
    manifest_artifact_keys.discard(None)

    # Also check datasets structure if present (for hierarchical manifests)
    for ds in manifest.get("datasets", []):
        for item in ds.get("items", []):
            for art in item.get("artifacts", []):
                key = art.get("artifactKey") or art.get("key")
                if key:
                    manifest_artifact_keys.add(key)

    # Extract all required artifact keys from contract
    required_artifacts: list[tuple[str, str, str]] = []  # (ds_key, item_key, art_key)
    for ds in contract.get("datasets", []):
        ds_key = ds.get("key", "")
        for item in ds.get("items", []):
            item_key = item.get("key", "")
            for art in item.get("artifacts", []):
                art_key = art.get("key", "")
                if art_key:
                    required_artifacts.append((ds_key, item_key, art_key))

    # Check all required artifacts are present
    for ds_key, item_key, art_key in required_artifacts:
        if art_key not in manifest_artifact_keys:
            errors.append(f"Missing artifact: {ds_key}/{item_key}/{art_key}")

    if errors:
        logger.warning(
            f"Contract validation found {len(errors)} missing artifacts. "
            f"Required: {[a[2] for a in required_artifacts]}, "
            f"Found: {manifest_artifact_keys}"
        )

    return len(errors) == 0, errors


@dataclass
class RunContext:
    """Context provided to algorithm's run() function."""

    data_client: DataClient
    params: dict[str, Any]
    artifacts: ArtifactWriter
    logger: logging.Logger
    run_context: dict[str, Any]


def load_algorithm_module(algorithm_path: Path):
    """Load algorithm.py as a module.

    Args:
        algorithm_path: Path to algorithm.py file.

    Returns:
        Loaded module.
    """
    spec = importlib.util.spec_from_file_location("algorithm", algorithm_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load algorithm from {algorithm_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["algorithm"] = module
    spec.loader.exec_module(module)
    return module


def write_manifest_atomic(output_dir: Path, manifest: dict) -> None:
    """Write manifest.json atomically.

    Writes to temp file, then renames to ensure atomic operation.

    Args:
        output_dir: Output directory.
        manifest: Manifest dict to write.
    """
    manifest_path = output_dir / "manifest.json"
    content = json.dumps(manifest, indent=2, ensure_ascii=False)

    # Write to temp file in same directory (ensures same filesystem)
    fd, tmp_path = tempfile.mkstemp(
        dir=output_dir, prefix="manifest_", suffix=".json.tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        os.rename(tmp_path, manifest_path)
    except Exception:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def run_algorithm(run_dir: Path, algorithm_path: Path) -> int:
    """Run the algorithm.

    Args:
        run_dir: The run directory.
        algorithm_path: Path to algorithm.py.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logs_dir = output_dir / "_logs"
    logs_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("algorithm")
    logger.setLevel(logging.DEBUG)

    # File handler
    log_file = logs_dir / "algorithm.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

    start_time = datetime.now()
    exit_code = 0
    error_message = None

    try:
        # Load input data
        params = json.loads((input_dir / "params.json").read_text())
        run_context_data = json.loads((input_dir / "run_context.json").read_text())

        # Create context components
        data_client = DataClient(input_dir)
        artifacts = ArtifactWriter(output_dir)

        ctx = RunContext(
            data_client=data_client,
            params=params,
            artifacts=artifacts,
            logger=logger,
            run_context=run_context_data,
        )

        # Load and run algorithm
        logger.info(f"Loading algorithm from {algorithm_path}")
        module = load_algorithm_module(algorithm_path)

        if not hasattr(module, "run"):
            raise RuntimeError("Algorithm module must define a 'run' function")
        logger.info("Starting algorithm execution")
        module.run(ctx)
        logger.info("Algorithm execution completed successfully")

    except Exception as e:
        exit_code = 1
        error_message = f"{type(e).__name__}: {e}"
        logger.error(f"Algorithm failed: {error_message}")
        logger.debug(traceback.format_exc())

    end_time = datetime.now()

    # Build output manifest
    manifest = {
        "schemaVersion": "1.0",
        "run": run_context_data if "run_context_data" in dir() else {},
        "status": "succeeded" if exit_code == 0 else "failed",
        "startedAt": start_time.isoformat(),
        "completedAt": end_time.isoformat(),
        "datasets": artifacts.build_manifest_datasets() if exit_code == 0 else [],
    }

    if error_message:
        manifest["error"] = error_message

    # Validate manifest against OutputContract if algorithm succeeded
    if exit_code == 0:
        # Find output_contract.json in algorithm's dist/ directory
        algorithm_dir = algorithm_path.parent
        contract_path = algorithm_dir / "dist" / "output_contract.json"

        is_valid, validation_errors = validate_manifest_against_contract(
            manifest, contract_path, logger
        )

        if not is_valid:
            exit_code = 1
            error_message = f"Output validation failed: {'; '.join(validation_errors)}"
            manifest["status"] = "failed"
            manifest["error"] = error_message
            manifest["validationErrors"] = validation_errors
            logger.error(f"Output validation failed: {validation_errors}")

    # Write manifest atomically
    write_manifest_atomic(output_dir, manifest)

    return exit_code


def main() -> None:
    """Entry point for fraclab-runner command."""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <run_dir> <algorithm_path>", file=sys.stderr)
        sys.exit(2)

    run_dir = Path(sys.argv[1])
    algorithm_path = Path(sys.argv[2])

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(2)

    if not algorithm_path.exists():
        print(f"Algorithm not found: {algorithm_path}", file=sys.stderr)
        sys.exit(2)

    exit_code = run_algorithm(run_dir, algorithm_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
