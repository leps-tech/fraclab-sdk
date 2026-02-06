"""Run manager implementation."""

import json
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from fraclab_sdk.algorithm import AlgorithmLibrary
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.errors import RunError
from fraclab_sdk.materialize import Materializer
from fraclab_sdk.run.logs import tail_stderr, tail_stdout
from fraclab_sdk.run.subprocess_runner import SubprocessRunner
from fraclab_sdk.selection.model import SelectionModel
from fraclab_sdk.snapshot import SnapshotLibrary
from fraclab_sdk.utils.io import atomic_write_json
from fraclab_sdk.utils.io import atomic_write_json


class RunStatus(Enum):
    """Status of a run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class RunMeta:
    """Metadata for a run."""

    run_id: str
    snapshot_id: str
    algorithm_id: str
    algorithm_version: str
    status: RunStatus
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


@dataclass
class RunResult:
    """Result of run execution."""

    run_id: str
    status: RunStatus
    exit_code: int | None = None
    error: str | None = None
    stdout: str | None = None
    stderr: str | None = None


class RunIndex:
    """Manages the run index file."""

    def __init__(self, runs_dir: Path) -> None:
        """Initialize run index."""
        self._runs_dir = runs_dir
        self._index_path = runs_dir / "index.json"

    def _load(self) -> dict[str, dict]:
        """Load index from disk."""
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text())

    def _save(self, data: dict[str, dict]) -> None:
        """Save index to disk."""
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self._index_path, data)

    def add(self, meta: RunMeta) -> None:
        """Add a run to the index."""
        data = self._load()
        data[meta.run_id] = {
            "run_id": meta.run_id,
            "snapshot_id": meta.snapshot_id,
            "algorithm_id": meta.algorithm_id,
            "algorithm_version": meta.algorithm_version,
            "status": meta.status.value,
            "created_at": meta.created_at,
            "started_at": meta.started_at,
            "completed_at": meta.completed_at,
            "error": meta.error,
        }
        self._save(data)

    def update(self, meta: RunMeta) -> None:
        """Update a run in the index."""
        self.add(meta)

    def remove(self, run_id: str) -> None:
        """Remove a run from the index."""
        data = self._load()
        if run_id in data:
            del data[run_id]
            self._save(data)

    def get(self, run_id: str) -> RunMeta | None:
        """Get run metadata."""
        data = self._load()
        if run_id not in data:
            return None
        entry = data[run_id]
        return RunMeta(
            run_id=entry["run_id"],
            snapshot_id=entry["snapshot_id"],
            algorithm_id=entry["algorithm_id"],
            algorithm_version=entry["algorithm_version"],
            status=self._coerce_status(entry.get("status", "")),
            created_at=entry["created_at"],
            started_at=entry.get("started_at"),
            completed_at=entry.get("completed_at"),
            error=entry.get("error"),
        )

    def list_all(self) -> list[RunMeta]:
        """List all runs."""
        data = self._load()
        return [
            RunMeta(
                run_id=entry["run_id"],
                snapshot_id=entry["snapshot_id"],
                algorithm_id=entry["algorithm_id"],
                algorithm_version=entry["algorithm_version"],
                status=self._coerce_status(entry.get("status", "")),
                created_at=entry["created_at"],
                started_at=entry.get("started_at"),
                completed_at=entry.get("completed_at"),
                error=entry.get("error"),
            )
            for entry in data.values()
        ]

    @staticmethod
    def _coerce_status(value: str) -> RunStatus:
        """Map legacy statuses to new enum."""
        mapping = {
            "completed": RunStatus.SUCCEEDED,
            "failed": RunStatus.FAILED,
            "pending": RunStatus.PENDING,
            "running": RunStatus.RUNNING,
            "timeout": RunStatus.TIMEOUT,
            "succeeded": RunStatus.SUCCEEDED,
        }
        if value in mapping:
            return mapping[value]
        try:
            return RunStatus(value)
        except Exception:
            return RunStatus.FAILED


class RunManager:
    """Manages algorithm runs."""

    def __init__(self, config: SDKConfig | None = None) -> None:
        """Initialize run manager.

        Args:
            config: SDK configuration. If None, uses default.
        """
        self._config = config or SDKConfig()
        self._index = RunIndex(self._config.runs_dir)
        self._snapshot_lib = SnapshotLibrary(self._config)
        self._algorithm_lib = AlgorithmLibrary(self._config)
        self._materializer = Materializer()

    def create_run(
        self,
        snapshot_id: str,
        algorithm_id: str,
        algorithm_version: str,
        selection: SelectionModel,
        params: dict[str, Any],
    ) -> str:
        """Create a new run.

        Args:
            snapshot_id: The snapshot ID.
            algorithm_id: The algorithm ID.
            algorithm_version: The algorithm version.
            selection: The selection model with selected items.
            params: Algorithm parameters.

        Returns:
            The run ID.

        Raises:
            RunError: If run creation fails.
        """
        # Validate selection
        errors = selection.validate()
        if errors:
            error_msgs = [f"{e.dataset_key}: {e.message}" for e in errors]
            raise RunError(f"Selection validation failed: {'; '.join(error_msgs)}")

        # Get handles
        snapshot = self._snapshot_lib.get_snapshot(snapshot_id)
        algorithm = self._algorithm_lib.get_algorithm(algorithm_id, algorithm_version)

        # Generate run ID
        run_id = str(uuid.uuid4())[:8]

        # Create run directory
        self._config.ensure_dirs()
        run_dir = self._config.runs_dir / run_id
        run_dir.mkdir(parents=True)

        # Build run DataSpec
        run_ds = selection.build_run_ds()

        # Build run context
        run_context = {
            "runId": run_id,
            "snapshotId": snapshot_id,
            "algorithmId": algorithm_id,
            "algorithmVersion": algorithm_version,
            "contractVersion": algorithm.manifest.contractVersion,
        }

        # Materialize input
        self._materializer.materialize(
            run_dir=run_dir,
            snapshot=snapshot,
            run_ds=run_ds,
            drs=algorithm.drs,
            params=params,
            run_context=run_context,
        )

        # Create run metadata
        meta = RunMeta(
            run_id=run_id,
            snapshot_id=snapshot_id,
            algorithm_id=algorithm_id,
            algorithm_version=algorithm_version,
            status=RunStatus.PENDING,
            created_at=datetime.now().isoformat(),
        )
        self._index.add(meta)

        # Write run_meta.json
        run_meta_path = run_dir / "run_meta.json"
        run_meta_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "snapshot_id": snapshot_id,
                    "algorithm_id": algorithm_id,
                    "algorithm_version": algorithm_version,
                    "created_at": meta.created_at,
                },
                indent=2,
            )
        )

        return run_id

    def delete_run(self, run_id: str) -> None:
        """Delete a run and its outputs."""
        run_dir = self._config.runs_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        self._index.remove(run_id)

    def execute(
        self,
        run_id: str,
        timeout_s: int | None = None,
    ) -> RunResult:
        """Execute a run.

        Args:
            run_id: The run ID.
            timeout_s: Optional timeout in seconds.

        Returns:
            RunResult with execution outcome.

        Raises:
            RunError: If run not found or already executed.
        """
        meta = self._index.get(run_id)
        if meta is None:
            raise RunError(f"Run not found: {run_id}")

        if meta.status not in (RunStatus.PENDING, RunStatus.FAILED, RunStatus.TIMEOUT):
            raise RunError(f"Run {run_id} already executed with status {meta.status}")

        run_dir = self._config.runs_dir / run_id
        algorithm = self._algorithm_lib.get_algorithm(
            meta.algorithm_id, meta.algorithm_version
        )

        # Update status to running
        meta.status = RunStatus.RUNNING
        meta.started_at = datetime.now().isoformat()
        self._index.update(meta)

        # Execute via subprocess runner (streaming logs)
        cmd = [
            sys.executable,
            "-m",
            "fraclab_sdk.runtime.runner_main",
            str(run_dir),
            str(algorithm.algorithm_path),
        ]

        stdout_log = run_dir / "output" / "_logs" / "stdout.log"
        stderr_log = run_dir / "output" / "_logs" / "stderr.log"
        execute_meta = run_dir / "output" / "_logs" / "execute.json"

        runner = SubprocessRunner(cmd=cmd, cwd=run_dir, timeout_s=timeout_s)
        exit_code, timed_out = runner.run(stdout_log, stderr_log, execute_meta)

        error = None
        if timed_out:
            error = f"Timeout after {timeout_s}s"
        elif exit_code != 0:
            error = f"Exit code: {exit_code}"

        # Update final status
        if timed_out:
            meta.status = RunStatus.TIMEOUT
        elif exit_code == 0:
            meta.status = RunStatus.SUCCEEDED
        else:
            meta.status = RunStatus.FAILED
        meta.completed_at = datetime.now().isoformat()
        meta.error = error
        self._index.update(meta)

        return RunResult(
            run_id=run_id,
            status=meta.status,
            exit_code=exit_code,
            error=error,
            stdout=tail_stdout(run_dir),
            stderr=tail_stderr(run_dir),
        )

    def get_run_status(self, run_id: str) -> RunStatus:
        """Get the status of a run.

        Args:
            run_id: The run ID.

        Returns:
            Run status.

        Raises:
            RunError: If run not found.
        """
        meta = self._index.get(run_id)
        if meta is None:
            raise RunError(f"Run not found: {run_id}")
        return meta.status

    def get_run(self, run_id: str) -> RunMeta:
        """Get run metadata.

        Args:
            run_id: The run ID.

        Returns:
            Run metadata.

        Raises:
            RunError: If run not found.
        """
        meta = self._index.get(run_id)
        if meta is None:
            raise RunError(f"Run not found: {run_id}")
        return meta

    def get_run_dir(self, run_id: str) -> Path:
        """Get the run directory path.

        Args:
            run_id: The run ID.

        Returns:
            Path to run directory.
        """
        return self._config.runs_dir / run_id

    def list_runs(self) -> list[RunMeta]:
        """List all runs.

        Returns:
            List of run metadata.
        """
        return self._index.list_all()
