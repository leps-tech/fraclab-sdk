"""Shared helpers for the Streamlit workbench."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fraclab_sdk.config import SDKConfig

# Keep a dedicated workspace separate from the installed algorithm library.
WORKSPACE_ALGOS_SUBDIR = "workspace_algorithms"


def get_workspace_dir(config: SDKConfig) -> Path:
    """Return the workspace directory for editable algorithms."""
    workspace = config.sdk_home / WORKSPACE_ALGOS_SUBDIR
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def run_workspace_script(workspace: Path, script: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a Python snippet with the workspace on PYTHONPATH."""
    pythonpath = [str(workspace)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath.append(existing)

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(pythonpath),
        "PYTHONUNBUFFERED": "1",
    }

    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
