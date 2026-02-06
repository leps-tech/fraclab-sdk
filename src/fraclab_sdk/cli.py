"""Fraclab SDK CLI."""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TypeVar

import typer

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.errors import ExitCode, FraclabError
from fraclab_sdk.results import ResultReader
from fraclab_sdk.run import RunManager
from fraclab_sdk.run.logs import tail_stderr, tail_stdout
from fraclab_sdk.run.manager import RunStatus
from fraclab_sdk.selection.model import SelectionModel
from fraclab_sdk.snapshot import SnapshotLibrary

app = typer.Typer(help="Fraclab SDK CLI")
snapshot_app = typer.Typer()
algo_app = typer.Typer()
run_app = typer.Typer()
results_app = typer.Typer()
validate_app = typer.Typer()

app.add_typer(snapshot_app, name="snapshot")
app.add_typer(algo_app, name="algo")
app.add_typer(run_app, name="run")
app.add_typer(results_app, name="results")
app.add_typer(validate_app, name="validate")


def _error(msg: str) -> None:
    """Print error message to stderr."""
    typer.echo(f"Error: {msg}", err=True)


def _get_libs() -> tuple[SDKConfig, SnapshotLibrary, AlgorithmLibrary, RunManager]:
    cfg = SDKConfig()
    return cfg, SnapshotLibrary(cfg), AlgorithmLibrary(cfg), RunManager(cfg)


F = TypeVar("F", bound=Callable)


def handle_errors(func: F) -> F:
    """Decorator to handle FraclabError and other exceptions uniformly."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            # Re-raise typer.Exit unchanged (it's used for normal exits)
            raise
        except FraclabError as e:
            _error(str(e))
            raise typer.Exit(e.exit_code) from None
        except FileNotFoundError as e:
            _error(f"File not found: {e.filename or e}")
            raise typer.Exit(ExitCode.INPUT_ERROR) from None
        except json.JSONDecodeError as e:
            _error(f"Invalid JSON: {e.msg} at line {e.lineno}")
            raise typer.Exit(ExitCode.INPUT_ERROR) from None
        except Exception as e:
            _error(f"Internal error: {e}")
            if "--debug" in sys.argv:
                traceback.print_exc(file=sys.stderr)
            raise typer.Exit(ExitCode.INTERNAL_ERROR) from None

    return wrapper  # type: ignore


@snapshot_app.command("import")
@handle_errors
def snapshot_import(path: Path):
    """Import a snapshot (dir or zip)."""
    _, snap_lib, _, _ = _get_libs()
    snap_id = snap_lib.import_snapshot(path)
    typer.echo(f"Imported snapshot: {snap_id}")


@snapshot_app.command("list")
@handle_errors
def snapshot_list():
    """List snapshots."""
    _, snap_lib, _, _ = _get_libs()
    snaps = snap_lib.list_snapshots()
    if not snaps:
        typer.echo("No snapshots")
        raise typer.Exit(0)
    for s in snaps:
        typer.echo(f"{s.snapshot_id}\t{ s.bundle_id }\t{ s.created_at }")


@algo_app.command("import")
@handle_errors
def algo_import(path: Path):
    """Import an algorithm (dir or zip)."""
    _, _, algo_lib, _ = _get_libs()
    algo_id, ver = algo_lib.import_algorithm(path)
    typer.echo(f"Imported algorithm: {algo_id}:{ver}")


@algo_app.command("list")
@handle_errors
def algo_list():
    """List algorithms."""
    _, _, algo_lib, _ = _get_libs()
    algos = algo_lib.list_algorithms()
    if not algos:
        typer.echo("No algorithms")
        raise typer.Exit(0)
    for a in algos:
        typer.echo(f"{a.algorithm_id}\t{a.version}\t{a.imported_at}")


@algo_app.command("compile")
@handle_errors
def algo_compile(
    workspace: Path = typer.Argument(..., help="Path to algorithm workspace"),
    bundle: Path | None = typer.Option(None, "--bundle", "-b", help="Bundle path for drs.json"),
    skip_inputspec: bool = typer.Option(False, "--skip-inputspec", help="Skip InputSpec compilation (schema.inputspec:INPUT_SPEC)"),
    skip_output_contract: bool = typer.Option(
        False, "--skip-output-contract", help="Skip OutputContract compilation"
    ),
):
    """Compile algorithm workspace to generate dist/ artifacts.

    Generates:
    - dist/params.schema.json (from schema.inputspec:INPUT_SPEC)
    - dist/output_contract.json (from schema.output_contract:OUTPUT_CONTRACT)
    - dist/drs.json (from bundle)
    """
    from fraclab_sdk.devkit.compile import compile_algorithm

    result = compile_algorithm(
        workspace=workspace,
        bundle_path=bundle,
        skip_inputspec=skip_inputspec,
        skip_output_contract=skip_output_contract,
    )

    typer.echo(f"Compiled algorithm workspace: {workspace}")
    typer.echo(f"  params.schema.json: {result.params_schema_path}")
    typer.echo(f"  output_contract.json: {result.output_contract_path}")
    typer.echo(f"  drs.json: {result.drs_path}")
    if result.bound_bundle:
        typer.echo(f"  Bound bundle hashes: {result.bound_bundle}")


@algo_app.command("export")
@handle_errors
def algo_export(
    workspace: Path = typer.Argument(..., help="Path to algorithm workspace"),
    output: Path = typer.Argument(..., help="Output path (.zip or directory)"),
    auto_compile: bool = typer.Option(False, "--auto-compile", "-c", help="Auto-compile if needed"),
    bundle: Path | None = typer.Option(None, "--bundle", "-b", help="Bundle path for auto-compile"),
):
    """Export algorithm workspace as distributable package.

    Creates a package containing main.py, manifest.json, and dist/ artifacts.
    """
    from fraclab_sdk.devkit.export import export_algorithm_package

    result = export_algorithm_package(
        workspace=workspace,
        output=output,
        auto_compile=auto_compile,
        bundle_path=bundle,
    )

    typer.echo(f"Exported algorithm to: {result.output_path}")
    typer.echo(f"  Files included: {len(result.files_included)}")
    if result.files_rejected:
        typer.echo(f"  Files rejected: {len(result.files_rejected)}", err=True)


@run_app.command("create")
@handle_errors
def run_create(
    snapshot_id: str = typer.Argument(...),
    algorithm_id: str = typer.Argument(...),
    algorithm_version: str = typer.Argument(...),
    params_path: Path | None = typer.Option(None, "--params", "-p", help="JSON file with params"),
):
    """Create a run selecting all items."""
    _, snap_lib, algo_lib, run_mgr = _get_libs()
    snapshot = snap_lib.get_snapshot(snapshot_id)
    algorithm = algo_lib.get_algorithm(algorithm_id, algorithm_version)

    selection = SelectionModel.from_snapshot_and_drs(snapshot, algorithm.drs)
    # select all items for each dataset
    for ds in selection.get_selectable_datasets():
        selection.set_selected(ds.dataset_key, list(range(ds.total_items)))

    params: dict = {}
    if params_path:
        params = json.loads(params_path.read_text())

    run_id = run_mgr.create_run(
        snapshot_id=snapshot_id,
        algorithm_id=algorithm_id,
        algorithm_version=algorithm_version,
        selection=selection,
        params=params,
    )
    typer.echo(run_id)


@run_app.command("exec")
@handle_errors
def run_exec(run_id: str, timeout: int | None = typer.Option(None, "--timeout", "-t")):
    """Execute a run."""
    _, _, _, run_mgr = _get_libs()
    result = run_mgr.execute(run_id, timeout_s=timeout)
    typer.echo(f"{result.status.value} (exit_code={result.exit_code})")
    if result.status == RunStatus.TIMEOUT:
        _error(result.error or f"Timeout after {timeout}s")
        raise typer.Exit(ExitCode.TIMEOUT)
    if result.status == RunStatus.FAILED:
        _error(result.error or "Run failed")
        raise typer.Exit(ExitCode.RUN_FAILED)


@run_app.command("tail")
@handle_errors
def run_tail(run_id: str, stderr: bool = typer.Option(False, "--stderr")):
    """Tail stdout/stderr."""
    _, _, _, run_mgr = _get_libs()
    run_dir = run_mgr.get_run_dir(run_id)
    if not run_dir.exists():
        _error(f"Run directory not found: {run_id}")
        raise typer.Exit(ExitCode.INPUT_ERROR)
    if stderr:
        typer.echo(tail_stderr(run_dir))
    else:
        typer.echo(tail_stdout(run_dir))


@results_app.command("list")
@handle_errors
def results_list(run_id: str):
    """List artifacts for a run."""
    _, _, _, run_mgr = _get_libs()
    run_dir = run_mgr.get_run_dir(run_id)
    if not run_dir.exists():
        _error(f"Run directory not found: {run_id}")
        raise typer.Exit(ExitCode.INPUT_ERROR)
    reader = ResultReader(run_dir)
    if not reader.has_manifest():
        _error("Manifest not found (run may not have completed)")
        raise typer.Exit(ExitCode.INPUT_ERROR)
    manifest = reader.read_manifest()
    typer.echo(f"Status: {manifest.status}")
    for art in manifest.list_all_artifacts():
        typer.echo(f"{art.artifactKey}\t{art.type}\t{art.uri or ''}")


@validate_app.command("inputspec")
@handle_errors
def validate_inputspec_cmd(
    workspace: Path = typer.Argument(..., help="Path to algorithm workspace"),
):
    """Validate InputSpec (schema.inputspec:INPUT_SPEC).

    Checks json_schema_extra fields, show_when conditions, and enum_labels.
    """
    from fraclab_sdk.devkit.validate import ValidationSeverity, validate_inputspec

    result = validate_inputspec(workspace)

    if result.valid:
        typer.echo("InputSpec validation passed")
    else:
        typer.echo("InputSpec validation failed", err=True)

    for issue in result.issues:
        prefix = "ERROR" if issue.severity == ValidationSeverity.ERROR else "WARN"
        path_str = f" at {issue.path}" if issue.path else ""
        typer.echo(f"  [{prefix}] {issue.code}{path_str}: {issue.message}", err=True)

    if not result.valid:
        raise typer.Exit(ExitCode.INPUT_ERROR)


@validate_app.command("output-contract")
@handle_errors
def validate_output_contract_cmd(
    workspace_or_path: Path = typer.Argument(..., help="Workspace or output_contract.json path"),
):
    """Validate OutputContract structure.

    Checks key uniqueness, kind/schema consistency, and dimensions.
    """
    from fraclab_sdk.devkit.validate import ValidationSeverity, validate_output_contract

    result = validate_output_contract(workspace_or_path)

    if result.valid:
        typer.echo("OutputContract validation passed")
    else:
        typer.echo("OutputContract validation failed", err=True)

    for issue in result.issues:
        prefix = "ERROR" if issue.severity == ValidationSeverity.ERROR else "WARN"
        path_str = f" at {issue.path}" if issue.path else ""
        typer.echo(f"  [{prefix}] {issue.code}{path_str}: {issue.message}", err=True)

    if not result.valid:
        raise typer.Exit(ExitCode.INPUT_ERROR)


@validate_app.command("bundle")
@handle_errors
def validate_bundle_cmd(
    bundle_path: Path = typer.Argument(..., help="Path to bundle directory"),
):
    """Validate bundle hash integrity.

    Checks ds.json and drs.json hashes against manifest.
    """
    from fraclab_sdk.devkit.validate import ValidationSeverity, validate_bundle

    result = validate_bundle(bundle_path)

    if result.valid:
        typer.echo("Bundle validation passed")
    else:
        typer.echo("Bundle validation failed", err=True)

    for issue in result.issues:
        prefix = "ERROR" if issue.severity == ValidationSeverity.ERROR else "WARN"
        typer.echo(f"  [{prefix}] {issue.code}: {issue.message}", err=True)

    if not result.valid:
        raise typer.Exit(ExitCode.INPUT_ERROR)


@validate_app.command("run-manifest")
@handle_errors
def validate_run_manifest_cmd(
    manifest_path: Path = typer.Argument(..., help="Path to run output manifest.json"),
    contract_path: Path | None = typer.Option(
        None, "--contract", "-c", help="Path to output_contract.json for alignment check"
    ),
):
    """Validate run output manifest against OutputContract.

    If contract is provided, checks that all required datasets/items/artifacts are present.
    """
    from fraclab_sdk.devkit.validate import ValidationSeverity, validate_run_manifest

    result = validate_run_manifest(manifest_path, contract_path)

    if result.valid:
        typer.echo("Run manifest validation passed")
    else:
        typer.echo("Run manifest validation failed", err=True)

    for issue in result.issues:
        prefix = "ERROR" if issue.severity == ValidationSeverity.ERROR else "WARN"
        path_str = f" at {issue.path}" if issue.path else ""
        typer.echo(f"  [{prefix}] {issue.code}{path_str}: {issue.message}", err=True)

    if not result.valid:
        raise typer.Exit(ExitCode.INPUT_ERROR)


def main():
    app()


if __name__ == "__main__":
    main()
