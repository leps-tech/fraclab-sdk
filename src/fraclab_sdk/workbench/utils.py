"""Shared helpers for the Streamlit workbench."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
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


def format_snapshot_option(snapshot: object) -> str:
    """Return a consistent snapshot option label for workbench selects."""
    snapshot_id = getattr(snapshot, "snapshot_id", "")
    imported_at = getattr(snapshot, "imported_at", "")
    imported_text = format_timestamp(imported_at)
    return f"{snapshot_id} - {imported_text}"


def format_timestamp(value: str | None) -> str:
    """Format an ISO-like timestamp into second precision: YYYY-MM-DD HH:MM:SS."""
    if not value:
        return "unknown import time"

    raw = value.strip()
    try:
        normalized = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Fallback for non-standard strings: replace separator and drop fractions.
        fallback = raw.replace("T", " ")
        if "." in fallback:
            fallback = fallback.split(".", 1)[0]
        return fallback
