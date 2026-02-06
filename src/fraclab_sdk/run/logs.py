"""Log management for runs."""

from __future__ import annotations

from pathlib import Path


def _tail(path: Path, max_lines: int, max_bytes: int) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    text = data.decode(errors="replace")
    lines = text.splitlines()[-max_lines:]
    return "\n".join(lines)


def tail_stdout(run_dir: Path, max_lines: int = 200, max_bytes: int = 65_536) -> str:
    """Tail stdout log."""
    return _tail(run_dir / "output" / "_logs" / "stdout.log", max_lines, max_bytes)


def tail_stderr(run_dir: Path, max_lines: int = 200, max_bytes: int = 65_536) -> str:
    """Tail stderr log."""
    return _tail(run_dir / "output" / "_logs" / "stderr.log", max_lines, max_bytes)


def read_execute(run_dir: Path) -> dict | None:
    """Read execute metadata."""
    path = run_dir / "output" / "_logs" / "execute.json"
    if not path.exists():
        return None
    import json

    try:
        return json.loads(path.read_text())
    except Exception:
        return None


__all__ = ["tail_stdout", "tail_stderr", "read_execute"]
