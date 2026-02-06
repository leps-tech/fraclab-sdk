"""Algorithm compilation: generate static artifacts from workspace.

Compile workflow:
1. Import schema.inputspec:INPUT_SPEC → model_json_schema() → dist/params.schema.json
2. Import schema.output_contract:OUTPUT_CONTRACT → model_dump() → dist/output_contract.json
3. Copy drs.json from bundle → dist/drs.json
4. Update manifest.json with files pointers
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fraclab_sdk.errors import AlgorithmError


@dataclass
class CompileLog:
    """Log entry for a compile subprocess."""

    step: str
    stdout: str
    stderr: str
    returncode: int
    timestamp: str


@dataclass
class CompileResult:
    """Result of algorithm compilation."""

    params_schema_path: Path
    output_contract_path: Path
    drs_path: Path
    manifest_updated: bool
    bound_bundle: dict[str, str] | None = None


def _run_in_subprocess(
    workspace: Path,
    script: str,
    step_name: str,
    log_dir: Path | None = None,
) -> tuple[dict[str, Any], CompileLog]:
    """Run Python script in isolated subprocess with workspace on PYTHONPATH.

    Args:
        workspace: Algorithm workspace directory.
        script: Python script to execute.
        step_name: Name of the compilation step (for logging).
        log_dir: Directory to save logs (optional).

    Returns:
        Tuple of (parsed JSON output, compile log).

    Raises:
        AlgorithmError: If subprocess fails.
    """
    env = {
        "PYTHONPATH": str(workspace),
        "PYTHONUNBUFFERED": "1",
    }

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=workspace,
        env={**dict(__import__("os").environ), **env},
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Create log entry
    log = CompileLog(
        step=step_name,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        timestamp=datetime.now().isoformat(),
    )

    # Save log to file if log_dir provided
    if log_dir is not None:
        log_file = log_dir / f"{step_name}.log"
        log_content = (
            f"Step: {step_name}\n"
            f"Timestamp: {log.timestamp}\n"
            f"Return code: {result.returncode}\n"
            f"\n=== STDOUT ===\n{result.stdout}\n"
            f"\n=== STDERR ===\n{result.stderr}\n"
        )
        log_file.write_text(log_content, encoding="utf-8")

    if result.returncode != 0:
        error_summary = result.stderr.strip() or result.stdout.strip() or "Unknown error"
        # Truncate for CLI display, full log is in file
        if len(error_summary) > 500:
            error_summary = error_summary[:500] + "..."
        log_path_hint = f" (full log: {log_dir / f'{step_name}.log'})" if log_dir else ""
        raise AlgorithmError(
            f"Compilation step '{step_name}' failed: {error_summary}{log_path_hint}"
        )

    try:
        return json.loads(result.stdout), log
    except json.JSONDecodeError as e:
        log_path_hint = f" (full log: {log_dir / f'{step_name}.log'})" if log_dir else ""
        raise AlgorithmError(
            f"Invalid JSON output from '{step_name}': {e}{log_path_hint}"
        ) from e


def _extract_params_schema(workspace: Path, log_dir: Path | None = None) -> dict[str, Any]:
    """Extract params JSON Schema from INPUT_SPEC.

    Args:
        workspace: Algorithm workspace directory.
        log_dir: Directory to save logs (optional).

    Returns:
        JSON Schema dict.
    """
    script = '''
import json
import sys

try:
    from schema.inputspec import INPUT_SPEC
    model = INPUT_SPEC
    schema = model.model_json_schema()
    print(json.dumps(schema))
except ImportError as e:
    print(json.dumps({"error": f"Failed to import INPUT_SPEC: {e}"}))
    sys.exit(0)
except Exception as e:
    print(json.dumps({"error": f"Failed to generate schema: {e}"}))
    sys.exit(0)
'''

    result, _ = _run_in_subprocess(workspace, script, "params_schema", log_dir)
    if "error" in result:
        raise AlgorithmError(result["error"])
    return result


def _extract_output_contract(workspace: Path, log_dir: Path | None = None) -> dict[str, Any]:
    """Extract OutputContract from OUTPUT_CONTRACT.

    Args:
        workspace: Algorithm workspace directory.
        log_dir: Directory to save logs (optional).

    Returns:
        OutputContract dict.
    """
    script = '''
import json
import sys

try:
    from schema.output_contract import OUTPUT_CONTRACT
    # Use model_dump with mode="json" for JSON-serializable output
    if hasattr(OUTPUT_CONTRACT, 'model_dump'):
        data = OUTPUT_CONTRACT.model_dump(mode="json")
    else:
        data = OUTPUT_CONTRACT.dict()
    print(json.dumps(data))
except ImportError as e:
    print(json.dumps({"error": f"Failed to import OUTPUT_CONTRACT: {e}"}))
    sys.exit(0)
except Exception as e:
    print(json.dumps({"error": f"Failed to dump contract: {e}"}))
    sys.exit(0)
'''

    result, _ = _run_in_subprocess(workspace, script, "output_contract", log_dir)
    if "error" in result:
        raise AlgorithmError(result["error"])
    return result


def _compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file contents (raw bytes).

    Args:
        path: File path.

    Returns:
        Hex-encoded SHA256 hash.
    """
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def compile_algorithm(
    workspace: Path,
    bundle_path: Path | None = None,
    skip_inputspec: bool = False,
    skip_output_contract: bool = False,
) -> CompileResult:
    """Compile algorithm workspace to generate static artifacts.

    This generates:
    - dist/params.schema.json (from schema.inputspec:CONFIG_MODEL)
    - dist/output_contract.json (from schema.output_contract:OUTPUT_CONTRACT)
    - dist/drs.json (copied from bundle)

    And updates manifest.json with files pointers.

    Args:
        workspace: Path to algorithm workspace directory.
        bundle_path: Path to data bundle (for drs.json). If None, drs.json must exist.
        skip_inputspec: Skip InputSpec compilation (use existing params.schema.json).
        skip_output_contract: Skip OutputContract compilation.

    Returns:
        CompileResult with paths to generated artifacts.

    Raises:
        AlgorithmError: If compilation fails.
    """
    workspace = Path(workspace).resolve()

    if not workspace.is_dir():
        raise AlgorithmError(f"Workspace not found: {workspace}")

    # Validate workspace structure
    manifest_path = workspace / "manifest.json"
    algorithm_py_path = workspace / "main.py"

    if not manifest_path.exists():
        raise AlgorithmError(f"manifest.json not found in workspace: {workspace}")
    if not algorithm_py_path.exists():
        raise AlgorithmError(f"main.py not found in workspace: {workspace}")

    # Create dist directory and compile logs directory
    dist_dir = workspace / "dist"
    dist_dir.mkdir(exist_ok=True)

    log_dir = dist_dir / "_compile_logs"
    log_dir.mkdir(exist_ok=True)

    # 1. Generate params.schema.json from InputSpec
    params_schema_path = dist_dir / "params.schema.json"
    if not skip_inputspec:
        schema_dir = workspace / "schema"
        inputspec_path = schema_dir / "inputspec.py"
        if not inputspec_path.exists():
            raise AlgorithmError(
                "schema/inputspec.py not found. Required for params schema generation."
            )

        params_schema = _extract_params_schema(workspace, log_dir)
        params_schema_path.write_text(json.dumps(params_schema, indent=2), encoding="utf-8")

    # 2. Generate output_contract.json from OutputContract
    output_contract_path = dist_dir / "output_contract.json"
    if not skip_output_contract:
        schema_dir = workspace / "schema"
        output_contract_file = schema_dir / "output_contract.py"
        if not output_contract_file.exists():
            raise AlgorithmError(
                "schema/output_contract.py not found. Required for output contract generation."
            )

        output_contract = _extract_output_contract(workspace, log_dir)
        output_contract_path.write_text(
            json.dumps(output_contract, indent=2), encoding="utf-8"
        )

    # 3. Copy drs.json from bundle (or use existing)
    drs_path = dist_dir / "drs.json"
    bound_bundle: dict[str, str] | None = None

    if bundle_path is not None:
        bundle_path = Path(bundle_path).resolve()
        if not bundle_path.is_dir():
            raise AlgorithmError(f"Bundle path not found: {bundle_path}")

        bundle_drs = bundle_path / "drs.json"
        bundle_ds = bundle_path / "ds.json"
        bundle_manifest = bundle_path / "manifest.json"

        if not bundle_drs.exists():
            raise AlgorithmError(f"drs.json not found in bundle: {bundle_path}")

        # Copy drs.json (raw bytes to preserve hash)
        shutil.copy2(bundle_drs, drs_path)

        # Extract hash info from bundle manifest if available
        if bundle_manifest.exists():
            try:
                manifest = json.loads(bundle_manifest.read_text())
                spec_files = manifest.get("specFiles", {})
                bound_bundle = {
                    "drsSha256": spec_files.get("drsSha256") or _compute_file_hash(bundle_drs),
                }
                if bundle_ds.exists():
                    bound_bundle["dsSha256"] = spec_files.get("dsSha256") or _compute_file_hash(
                        bundle_ds
                    )
            except (json.JSONDecodeError, KeyError):
                pass
    elif not drs_path.exists():
        raise AlgorithmError(
            "drs.json not found in dist/. Provide --bundle to copy from bundle."
        )

    # 4. Update manifest.json with files pointers
    manifest = json.loads(manifest_path.read_text())

    files = manifest.get("files", {})
    files["paramsSchemaPath"] = "dist/params.schema.json"
    files["outputContractPath"] = "dist/output_contract.json"
    files["drsPath"] = "dist/drs.json"
    manifest["files"] = files

    # Add bound bundle info if available
    if bound_bundle:
        manifest["boundBundle"] = bound_bundle

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return CompileResult(
        params_schema_path=params_schema_path,
        output_contract_path=output_contract_path,
        drs_path=drs_path,
        manifest_updated=True,
        bound_bundle=bound_bundle,
    )


__all__ = ["compile_algorithm", "CompileResult"]
